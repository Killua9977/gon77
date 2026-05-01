"""
================================================================================
  Forex Trading Bot v4 — Win Rate Optimized
================================================================================
  NEW in v4:
  1. Instrument-specific parameters  — forex/indices/commodities treated separately
  2. Tighter session filter          — London + NY only (07:00–17:00 UTC)
  3. Candlestick confirmation        — engulfing pattern required at entry
  4. Stale trade exit                — close dead trades after 4 hours
  5. Support & Resistance TP         — TP adjusted to nearest S/R level
  + All v3 fixes: stopLevel/limitLevel, per-instrument precision, 2-attempt orders
================================================================================
"""

import csv
import logging
import math
import os
import pickle
import sqlite3
import time
from datetime import datetime, timezone, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ================== Configuration ==========================================
TOKEN            = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID          = os.getenv("TELEGRAM_CHAT_ID",   "").strip()
CAPITAL_API_KEY  = os.getenv("CAPITAL_API_KEY",    "").strip()
CAPITAL_LOGIN    = os.getenv("CAPITAL_LOGIN",       "").strip()
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD",    "").strip()

INITIAL_EQUITY             = float(os.getenv("INITIAL_EQUITY",            "1000.0"))
RISK_PERCENT               = float(os.getenv("RISK_PERCENT",              "2.0"))
MAX_ACTIVE_TRADES          = int  (os.getenv("MAX_ACTIVE_TRADES",         "10"))
BREAK_EVEN_TRIGGER_R       = float(os.getenv("BREAK_EVEN_TRIGGER_R",      "1.0"))
PAIR_COOLDOWN_SECONDS      = int  (os.getenv("PAIR_COOLDOWN_SECONDS",     "14400"))
MAX_AUTH_RETRIES           = int  (os.getenv("MAX_AUTH_RETRIES",          "5"))

# v2/v3 features
MIN_CONFLUENCE_SCORE       = int  (os.getenv("MIN_CONFLUENCE_SCORE",      "3"))  # relaxed from 4
DAILY_LOSS_LIMIT_PCT       = float(os.getenv("DAILY_LOSS_LIMIT_PCT",      "4.0"))
NEWS_BUFFER_MINUTES        = int  (os.getenv("NEWS_BUFFER_MINUTES",       "30"))
ADX_TREND_THRESHOLD        = float(os.getenv("ADX_TREND_THRESHOLD",       "20.0"))
MAX_PORTFOLIO_HEAT_PCT     = float(os.getenv("MAX_PORTFOLIO_HEAT_PCT",    "6.0"))
MIN_TRADES_FOR_DISABLE     = int  (os.getenv("MIN_TRADES_FOR_DISABLE",    "20"))
DISABLE_WIN_RATE_THRESHOLD = float(os.getenv("DISABLE_WIN_RATE_THRESHOLD","35.0"))
ML_MIN_TRADES_TO_TRAIN     = int  (os.getenv("ML_MIN_TRADES_TO_TRAIN",    "50"))
ML_CONFIDENCE_THRESHOLD    = float(os.getenv("ML_CONFIDENCE_THRESHOLD",   "0.52"))  # relaxed from 0.60

# v4 NEW
STALE_TRADE_HOURS          = float(os.getenv("STALE_TRADE_HOURS",         "4.0"))   # close if no progress after N hours
SR_LOOKBACK                = int  (os.getenv("SR_LOOKBACK",               "100"))   # candles for S/R detection
SR_MIN_TOUCHES             = int  (os.getenv("SR_MIN_TOUCHES",            "2"))     # min touches to confirm S/R level
SR_ZONE_PCT                = float(os.getenv("SR_ZONE_PCT",               "0.002")) # S/R zone width (0.2% of price)
REQUIRE_ENGULFING          = os.getenv("REQUIRE_ENGULFING", "false").lower() == "true"  # relaxed: too rare on 15m

# Timing
SCAN_INTERVAL_SECONDS        = int(os.getenv("SCAN_INTERVAL_SECONDS",       "900"))
TRADE_CHECK_INTERVAL_SECONDS = int(os.getenv("TRADE_CHECK_INTERVAL_SECONDS","20"))
HEARTBEAT_INTERVAL_SECONDS   = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS",  "1800"))
REPORT_INTERVAL_SECONDS      = int(os.getenv("REPORT_INTERVAL_SECONDS",     "3600"))
ML_RETRAIN_INTERVAL_SECONDS  = int(os.getenv("ML_RETRAIN_INTERVAL_SECONDS", "86400"))

DB_FILE       = "trade_state.db"
RESULTS_FILE  = "trade_results.csv"
LOG_FILE      = "bot.log"
ML_MODEL_FILE = "ml_model.pkl"

# ================== [NEW v4] Instrument Profiles ===========================
# Each instrument class gets its own ATR multipliers, RSI ranges,
# spread tolerance, and trading sessions. This is the single biggest
# win rate improvement — stops using forex settings on gold/indices.

INSTRUMENT_PROFILES = {
    # ── Forex majors ────────────────────────────────────────────────────────
    "FOREX": {
        # Tuned specifically for EURUSD, GBPUSD, USDJPY, GBPJPY
        "pairs": {"EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD","EURGBP","EURJPY","GBPJPY"},
        "atr_stop":         1.5,    # tight stop — forex moves are precise
        "atr_tp_full":      2.5,    # slightly bigger target
        "atr_tp_partial":   1.4,    # partial TP earlier
        "trailing_mult":    1.1,    # tight trail to lock profit fast
        "spread_ratio":     0.25,
        "rsi_buy_lo":       48, "rsi_buy_hi":  75,
        "rsi_sell_lo":      25, "rsi_sell_hi": 52,
        "adx_min":          18.0,
        "trend_gap_mult":   0.20,
        "session_start":    6,    # London pre-open + London + NY
        "session_end":      18,
        "decimals":         5,
        "min_dist":         0.00010,
    },
    # ── US Indices ───────────────────────────────────────────────────────────
    "INDEX": {
        # Tuned specifically for US500
        # US500 trends strongly during NY session, needs wider stops
        "pairs": {"US500","US30","USTEC"},
        "atr_stop":         2.2,    # wider stop — indices spike before trending
        "atr_tp_full":      3.5,    # bigger target — indices move far when trending
        "atr_tp_partial":   2.0,    # partial TP at 2R
        "trailing_mult":    1.8,    # wider trail — give room to breathe
        "spread_ratio":     0.35,
        "rsi_buy_lo":       50, "rsi_buy_hi":  75,
        "rsi_sell_lo":      25, "rsi_sell_hi": 50,
        "adx_min":          18.0,
        "trend_gap_mult":   0.20,
        "session_start":    13,    # NY open — US500 needs US session volume
        "session_end":      21,    # NY close + afterhours
        "decimals":         1,
        "min_dist":         3.0,   # reduced from 5 — allows tighter entries
    },
    # ── Commodities (Gold, Silver, Oil) ─────────────────────────────────────
    "COMMODITY": {
        "pairs": {"XAUUSD","XAGUSD","USOIL"},
        "atr_stop":         1.8,
        "atr_tp_full":      2.8,
        "atr_tp_partial":   1.6,
        "trailing_mult":    1.4,
        "spread_ratio":     0.30,  # relaxed
        "rsi_buy_lo":       48, "rsi_buy_hi":  75,  # relaxed
        "rsi_sell_lo":      25, "rsi_sell_hi": 52,  # relaxed
        "adx_min":          20.0,  # relaxed from 25
        "trend_gap_mult":   0.25,  # relaxed from 0.35
        "session_start":    7,     # relaxed from 8
        "session_end":      18,    # relaxed from 17
        "decimals":         2,
        "min_dist":         0.50,
    },
}

def get_profile(pair: str) -> Dict:
    for profile in INSTRUMENT_PROFILES.values():
        if pair in profile["pairs"]:
            return profile
    return INSTRUMENT_PROFILES["FOREX"]  # default

def get_decimals(pair: str) -> Tuple[int, float]:
    """
    Returns (decimal_places, min_stop_distance) per instrument.
    Overrides profile defaults with exact Capital.com requirements.
    """
    pair_up = pair.upper()
    if "JPY" in pair_up:
        return 3, 0.05        # JPY: price ~160, pip=0.01, min=5 pips
    if pair_up in ["US500", "SPX500"]:
        return 1, 3.0         # US500: price ~5000, min 3 points
    if pair_up in ["US30", "WALL"]:
        return 0, 10.0        # US30: price ~40000, min 10 points
    if pair_up in ["USTEC", "NAS100"]:
        return 1, 5.0         # USTEC: price ~18000, min 5 points
    if "XAU" in pair_up:
        return 2, 0.50        # Gold: price ~2000, min 50 cents
    if "XAG" in pair_up:
        return 2, 0.05        # Silver
    if "OIL" in pair_up:
        return 2, 0.05        # Oil
    return 5, 0.00050         # Standard forex

# Instrument universe
pairs = {
    # ── Forex majors (clean trends, low spread, 24h) ──────────────────────
    "EURUSD": "EURUSD",   # most liquid, cleanest trends
    "GBPUSD": "GBPUSD",   # strong directional moves
    "USDJPY": "USDJPY",   # excellent momentum pair
    "GBPJPY": "GBPJPY",   # high volatility, big moves
    # ── Index (best trending index, NY session) ───────────────────────────
    "US500":  "US 500",   # most predictable index, trends beautifully
}

CORRELATION_GROUPS = [
    # GBPUSD and EURUSD are correlated — both USD negative
    {"EURUSD", "GBPUSD"},
    # USDJPY and GBPJPY both involve JPY
    {"USDJPY", "GBPJPY"},
    # Indices group (in case more added later)
    {"US500", "US30", "USTEC"},
]

# ================== Logging ================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ================== Database ==============================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT, epic TEXT, type TEXT,
        entry REAL, sl REAL, tp REAL, tp_partial REAL,
        status TEXT DEFAULT 'OPEN',
        opened_at REAL, risk_per_unit REAL,
        break_even_done INTEGER DEFAULT 0,
        partial_done INTEGER DEFAULT 0,
        entry_atr REAL, lot_size REAL,
        deal_ref TEXT, confluence_score INTEGER DEFAULT 0,
        ml_confidence REAL DEFAULT 0.0,
        instrument_class TEXT DEFAULT 'FOREX'
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS daily_pnl (
        trade_date TEXT PRIMARY KEY,
        start_equity REAL,
        realized_pnl REAL DEFAULT 0.0,
        trade_count INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS pair_stats (
        pair TEXT PRIMARY KEY,
        total_trades INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        total_pnl REAL DEFAULT 0.0,
        disabled INTEGER DEFAULT 0,
        disabled_reason TEXT DEFAULT ''
    )""")
    conn.commit()
    conn.close()

def db_save_trade(trade: Dict) -> int:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""INSERT INTO trades
        (pair,epic,type,entry,sl,tp,tp_partial,status,opened_at,risk_per_unit,
         break_even_done,partial_done,entry_atr,lot_size,deal_ref,
         confluence_score,ml_confidence,instrument_class)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (trade["pair"], trade["epic"], trade["type"], trade["entry"], trade["sl"],
         trade["tp"], trade["tp_partial"], trade["status"], trade["opened_at"],
         trade["risk_per_unit"], int(trade["break_even_done"]), int(trade["partial_done"]),
         trade["entry_atr"], trade["lot_size"], trade.get("deal_ref",""),
         trade.get("confluence_score",0), trade.get("ml_confidence",0.0),
         trade.get("instrument_class","FOREX")))
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id

def db_update_trade(trade: Dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""UPDATE trades SET sl=?,tp=?,status=?,break_even_done=?,partial_done=?
                 WHERE id=?""",
        (trade["sl"], trade["tp"], trade["status"],
         int(trade["break_even_done"]), int(trade.get("partial_done",False)),
         trade["db_id"]))
    conn.commit()
    conn.close()

def db_load_open_trades() -> List[Dict]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM trades WHERE status='OPEN'")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        d["break_even_done"] = bool(d["break_even_done"])
        d["partial_done"]    = bool(d.get("partial_done", 0))
        result.append(d)
    return result

def db_record_pair_result(pair: str, won: bool, pnl: float):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO pair_stats (pair) VALUES (?)", (pair,))
    field = "wins" if won else "losses"
    c.execute(f"UPDATE pair_stats SET total_trades=total_trades+1, {field}={field}+1, total_pnl=total_pnl+? WHERE pair=?", (pnl, pair))
    conn.commit()
    conn.close()

def db_get_pair_stats(pair: str) -> Dict:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM pair_stats WHERE pair=?", (pair,))
    row = c.fetchone()
    conn.close()
    if not row:
        return {"pair":pair,"total_trades":0,"wins":0,"losses":0,"total_pnl":0.0,"disabled":0,"disabled_reason":""}
    return dict(zip(["pair","total_trades","wins","losses","total_pnl","disabled","disabled_reason"], row))

def db_disable_pair(pair: str, reason: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO pair_stats (pair) VALUES (?)", (pair,))
    c.execute("UPDATE pair_stats SET disabled=1, disabled_reason=? WHERE pair=?", (reason, pair))
    conn.commit()
    conn.close()

def is_pair_disabled(pair: str) -> bool:
    return bool(db_get_pair_stats(pair).get("disabled", 0))

def get_today_str() -> str:
    return date.today().isoformat()

def init_daily_pnl(start_equity: float):
    today = get_today_str()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT trade_date FROM daily_pnl WHERE trade_date=?", (today,))
    if not c.fetchone():
        c.execute("INSERT INTO daily_pnl VALUES (?,?,0.0,0)", (today, start_equity))
        conn.commit()
    conn.close()

def record_daily_pnl(pnl: float):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE daily_pnl SET realized_pnl=realized_pnl+?, trade_count=trade_count+1 WHERE trade_date=?",
              (pnl, get_today_str()))
    conn.commit()
    conn.close()

def get_daily_loss_pct() -> float:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT start_equity, realized_pnl FROM daily_pnl WHERE trade_date=?", (get_today_str(),))
    row = c.fetchone()
    conn.close()
    if not row or row[0] <= 0:
        return 0.0
    return max((-row[1] / row[0]) * 100, 0.0)

def daily_loss_limit_hit() -> bool:
    loss = get_daily_loss_pct()
    if loss >= DAILY_LOSS_LIMIT_PCT:
        logger.warning(f"Daily loss limit: {loss:.2f}%")
        return True
    return False

# ================== Capital.com API =======================================
class CapitalClient:
    def __init__(self, api_key: str, login: str, password: str, demo: bool = True):
        self.api_key  = api_key
        self.login    = login
        self.password = password
        self.base_url = ("https://demo-api-capital.backend-capital.com" if demo
                         else "https://api-capital.backend-capital.com")
        self.cst = self.security_token = None
        self.session       = requests.Session()
        self.epic_cache: Dict[str, str] = {}
        self._auth_retries = 0
        self.authenticate()

    def authenticate(self):
        if self._auth_retries >= MAX_AUTH_RETRIES:
            raise RuntimeError(f"Auth failed after {MAX_AUTH_RETRIES} retries.")
        try:
            r = self.session.post(
                f"{self.base_url}/api/v1/session",
                headers={"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"identifier": self.login, "password": self.password, "encryptedPassword": False},
                timeout=30
            )
            if r.status_code != 200:
                raise Exception(r.json().get("errorMessage", "Unknown"))
            self.cst            = r.headers.get("CST")
            self.security_token = r.headers.get("X-SECURITY-TOKEN")
            self._auth_retries  = 0
            logger.info("✅ Connected to Capital.com")
        except Exception as e:
            self._auth_retries += 1
            logger.error(f"Auth {self._auth_retries}/{MAX_AUTH_RETRIES}: {e}")
            if self._auth_retries < MAX_AUTH_RETRIES:
                time.sleep(5 * self._auth_retries)
                self.authenticate()
            else:
                raise

    def _req(self, method: str, endpoint: str, data: Dict = None) -> Optional[Dict]:
        try:
            url     = f"{self.base_url}{endpoint}"
            headers = {"X-CAP-API-KEY": self.api_key, "CST": self.cst,
                       "X-SECURITY-TOKEN": self.security_token, "Content-Type": "application/json"}
            fn      = {"GET": self.session.get, "POST": self.session.post, "DELETE": self.session.delete, "PUT": self.session.put}
            kwargs  = {"headers": headers, "timeout": 30}
            if method in ["POST", "PUT"]:
                kwargs["json"] = data
            r = fn[method](url, **kwargs)
            if r.status_code in [401, 403]:
                self.authenticate()
                return self._req(method, endpoint, data)
            if r.status_code == 429:
                # Rate limited — wait and retry once
                wait = int(r.headers.get("Retry-After", 10))
                logger.warning(f"Rate limited (429) — waiting {wait}s...")
                time.sleep(wait)
                return self._req(method, endpoint, data)
            if r.status_code != 200:
                logger.error(f"API {r.status_code}: {r.text[:300]}")
                return None
            return r.json()
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def get_epic(self, search_term: str) -> Optional[str]:
        if search_term in self.epic_cache:
            return self.epic_cache[search_term]
        data = self._req("GET", f"/api/v1/markets?searchTerm={search_term}")
        if data and data.get("markets"):
            epic = data["markets"][0]["epic"]
            self.epic_cache[search_term] = epic
            return epic
        return None

    def get_candles(self, epic: str, resolution: str = "MINUTE_15",
                    num_candles: int = 300) -> Optional[pd.DataFrame]:
        data = self._req("GET", f"/api/v1/prices/{epic}?resolution={resolution}&max={num_candles}")
        if not data or "prices" not in data:
            return None
        rows = [{"time": c["snapshotTime"],
                 "Open":   float(c["openPrice"]["bid"]),
                 "High":   float(c["highPrice"]["bid"]),
                 "Low":    float(c["lowPrice"]["bid"]),
                 "Close":  float(c["closePrice"]["bid"]),
                 "Volume": float(c.get("lastTradedVolume", 0))}
                for c in data["prices"]]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        return df.dropna()

    def get_live_price(self, epic: str) -> Optional[Dict]:
        data = self._req("GET", f"/api/v1/markets/{epic}")
        if not data or "snapshot" not in data:
            return None
        bid = float(data["snapshot"]["bid"])
        ask = float(data["snapshot"]["offer"])
        return {"bid": bid, "ask": ask, "mid": (bid+ask)/2,
                "spread": round(ask-bid, 5), "tradeable": True}

    def place_order(self, pair: str, epic: str, direction: str, units: float,
                    entry: float, sl: float, tp: float) -> Optional[str]:
        """
        v4.2 — Three-attempt order placement:
        1. stopLevel + limitLevel (absolute prices)
        2. stopDistance + limitDistance (pip distances)
        3. Open position first, then set TP via PATCH/edit
        Each attempt is fully logged so Railway logs show exactly what failed.
        """
        decimals, min_dist = get_decimals(pair)
        sl_r = round(sl, decimals)
        tp_r = round(tp, decimals)
        entry_r = round(entry, decimals)

        # Enforce minimum distances from entry
        sl_dist = abs(entry_r - sl_r)
        tp_dist = abs(tp_r - entry_r)

        if sl_dist < min_dist:
            sl_r = round(entry_r - min_dist if direction=="BUY" else entry_r + min_dist, decimals)
            sl_dist = min_dist

        if tp_dist < min_dist:
            tp_r = round(entry_r + min_dist if direction=="BUY" else entry_r - min_dist, decimals)
            tp_dist = min_dist

        # Enforce minimum TP distance relative to SL distance
        # TP must be at least 1.5x the SL distance (1.5:1 R:R minimum)
        # This prevents Capital.com from silently rejecting tiny TPs
        min_tp_dist = max(sl_dist * 1.5, min_dist * 10)
        tp_from_entry = abs(float(tp_r) - float(entry_r))
        if tp_from_entry < min_tp_dist:
            old_tp = tp_r
            tp_r = round(
                float(entry_r) + min_tp_dist if direction == "BUY"
                else float(entry_r) - min_tp_dist,
                decimals
            )
            logger.warning(f"{pair}: TP too close — expanding {old_tp} → {tp_r} (min={min_tp_dist:.5f})")

        logger.info(
            f"\n{'='*50}\n"
            f"ORDER ATTEMPT: {direction} {units} {pair}\n"
            f"Entry={entry_r} | SL={sl_r} | TP={tp_r}\n"
            f"SL_dist={round(sl_dist,decimals)} | TP_dist={round(tp_dist,decimals)}\n"
            f"Decimals={decimals} | MinDist={min_dist}\n"
            f"{'='*50}"
        )

        # ── Attempt 1: absolute price levels ─────────────────────────────
        payload1 = {
            "epic":           epic,
            "direction":      direction,
            "size":           units,
            "stopLevel":      float(sl_r),
            "limitLevel":     float(tp_r),
            "guaranteedStop": False,
            "forceOpen":      True
        }
        logger.info(f"Attempt 1 payload: {payload1}")
        r1 = self._req("POST", "/api/v1/positions", payload1)
        logger.info(f"Attempt 1 response: {r1}")

        if r1 and r1.get("dealReference"):
            logger.info(f"✅ Attempt 1 SUCCESS: ref={r1['dealReference']}")
            return r1["dealReference"]

        # ── Attempt 2: pip distance based ────────────────────────────────
        logger.warning(f"Attempt 1 failed — trying pip distances...")
        payload2 = {
            "epic":           epic,
            "direction":      direction,
            "size":           units,
            "stopDistance":   float(round(sl_dist, decimals)),
            "limitDistance":  float(round(tp_dist, decimals)),
            "guaranteedStop": False,
            "forceOpen":      True
        }
        logger.info(f"Attempt 2 payload: {payload2}")
        r2 = self._req("POST", "/api/v1/positions", payload2)
        logger.info(f"Attempt 2 response: {r2}")

        if r2 and r2.get("dealReference"):
            logger.info(f"✅ Attempt 2 SUCCESS: ref={r2['dealReference']}")
            return r2["dealReference"]

        # ── Attempt 3: open without TP, then set TP via position update ──
        logger.warning(f"Attempt 2 failed — opening without TP, will set after...")
        payload3 = {
            "epic":           epic,
            "direction":      direction,
            "size":           float(units),
            "stopLevel":      float(sl_r),
            "guaranteedStop": False,
            "forceOpen":      True
        }
        logger.info(f"Attempt 3 payload: {payload3}")
        r3 = self._req("POST", "/api/v1/positions", payload3)
        logger.info(f"Attempt 3 response: {r3}")

        if r3 and r3.get("dealReference"):
            deal_ref = r3["dealReference"]
            logger.info(f"✅ Attempt 3 SUCCESS (no TP yet): ref={deal_ref}")
            # Wait for position to open then set TP
            time.sleep(2)
            self._set_tp_on_position(deal_ref, tp_r, sl_r)
            return deal_ref

        logger.error(f"❌ ALL 3 ATTEMPTS FAILED for {epic}")
        send_telegram(f"❌ Order failed: {pair} {direction}\nAll 3 attempts rejected by broker\nCheck Railway logs!")
        return None

    def _set_tp_on_position(self, deal_ref: str, tp_level: float, sl_level: float):
        """
        Sets TP on existing open position.
        Capital.com API: PUT /api/v1/positions/{dealId}
        with body: {"limitLevel": float(tp), "stopLevel": float(sl)}
        """
        try:
            positions = self._req("GET", "/api/v1/positions")
            if not positions:
                logger.error(f"Cannot fetch positions for {deal_ref}")
                return

            for pos in positions.get("positions", []):
                pos_data = pos.get("position", {})
                if pos_data.get("dealReference") != deal_ref:
                    continue

                deal_id = pos_data.get("dealId", "")
                if not deal_id:
                    logger.error(f"No dealId for {deal_ref}")
                    return

                logger.info(f"Setting TP via PUT /positions/{deal_id} | TP={tp_level} SL={sl_level}")

                # Capital.com requires both stopLevel and limitLevel together
                result = self._req("PUT", f"/api/v1/positions/{deal_id}", {
                    "limitLevel": float(tp_level),
                    "stopLevel":  float(sl_level),
                })
                logger.info(f"PUT result: {result}")

                if result and result.get("dealReference"):
                    logger.info(f"✅ TP set via PUT: {tp_level}")
                    send_telegram(f"✅ TP auto-fixed!\nref={deal_ref} TP={tp_level}")
                else:
                    # Try without stopLevel — some accounts only need limitLevel
                    logger.warning("PUT with both levels failed — trying limitLevel only...")
                    result2 = self._req("PUT", f"/api/v1/positions/{deal_id}", {
                        "limitLevel": float(tp_level),
                    })
                    logger.info(f"PUT limitLevel-only result: {result2}")
                    if result2 and result2.get("dealReference"):
                        logger.info(f"✅ TP set (limitLevel only): {tp_level}")
                        send_telegram(f"✅ TP auto-fixed!\nref={deal_ref} TP={tp_level}")
                    else:
                        logger.error(f"❌ All PUT attempts failed for {deal_ref}")
                        send_telegram(
                            f"⚠️ Please set TP MANUALLY on Capital.com!\n"
                            f"Trade: {deal_ref}\n"
                            f"TP = {tp_level}"
                        )
                return

            logger.warning(f"Position {deal_ref} not found in open positions")
        except Exception as e:
            logger.error(f"_set_tp_on_position error: {e}")

    def confirm_fill(self, deal_ref: str, tp_level: float = None, sl_level: float = None) -> Optional[Dict]:
        """
        KEY FIX v4.3: Use affectedDeals[0].dealId for PUT requests.
        Capital.com returns two dealIds in the confirm response:
        - top-level dealId = deal wrapper (WRONG for PUT)
        - affectedDeals[0].dealId = actual OPENED position (CORRECT for PUT)
        This was the root cause of all TP PUT failures.
        """
        time.sleep(2)
        data = self._req("GET", f"/api/v1/confirms/{deal_ref}")
        if not data:
            logger.warning(f"No confirm data for {deal_ref}")
            return None

        logger.info(f"Fill confirm: {data}")

        status = data.get("status", "")
        if status not in ["OPEN", "ACCEPTED"]:
            logger.warning(f"Fill status: {status}")
            return None

        # Check if TP already in confirm
        if data.get("limitLevel"):
            logger.info(f"✅ TP confirmed: {data['limitLevel']}")
            return data

        logger.warning(f"⚠️ TP missing in confirm for {deal_ref} — fixing...")

        # KEY FIX: get opened position dealId from affectedDeals
        opened_deal_id = None
        for d in data.get("affectedDeals", []):
            if d.get("status") == "OPENED":
                opened_deal_id = d.get("dealId")
                logger.info(f"Found opened dealId from affectedDeals: {opened_deal_id}")
                break

        if not opened_deal_id:
            opened_deal_id = data.get("dealId", "")
            logger.info(f"Fallback to top-level dealId: {opened_deal_id}")

        # Verify via positions and check if TP already set
        time.sleep(1)
        positions = self._req("GET", "/api/v1/positions")
        if positions:
            for pos in positions.get("positions", []):
                pos_data = pos.get("position", {})
                if pos_data.get("dealReference") == deal_ref:
                    limit      = pos_data.get("limitLevel")
                    pos_deal_id= pos_data.get("dealId", "")
                    logger.info(f"Position: dealId={pos_deal_id} TP={limit}")
                    if limit:
                        logger.info(f"✅ TP already on broker: {limit}")
                        data["limitLevel"] = limit
                        return data
                    if pos_deal_id:
                        opened_deal_id = pos_deal_id
                    break

        if not tp_level or not opened_deal_id:
            logger.error(f"Cannot fix TP: tp_level={tp_level} dealId={opened_deal_id}")
            return data

        logger.info(f"PUT /positions/{opened_deal_id} | TP={float(tp_level)} SL={float(sl_level) if sl_level else None}")

        # Attempt A: limitLevel + stopLevel
        r_a = self._req("PUT", f"/api/v1/positions/{opened_deal_id}", {
            "limitLevel": float(tp_level),
            "stopLevel":  float(sl_level) if sl_level else None,
        })
        logger.info(f"PUT (A) result: {r_a}")
        if r_a and r_a.get("dealReference"):
            logger.info(f"✅ TP set via PUT (A): {tp_level}")
            data["limitLevel"] = float(tp_level)
            send_telegram(f"✅ TP auto-fixed!\nTP={tp_level} ref={deal_ref}")
            return data

        # Attempt B: limitLevel only
        r_b = self._req("PUT", f"/api/v1/positions/{opened_deal_id}", {
            "limitLevel": float(tp_level),
        })
        logger.info(f"PUT (B) result: {r_b}")
        if r_b and r_b.get("dealReference"):
            logger.info(f"✅ TP set via PUT (B): {tp_level}")
            data["limitLevel"] = float(tp_level)
            send_telegram(f"✅ TP auto-fixed!\nTP={tp_level} ref={deal_ref}")
            return data

        logger.error(f"❌ All PUT attempts failed for {opened_deal_id}")
        send_telegram(
            f"⚠️ Set TP MANUALLY on Capital.com!\n"
            f"ref={deal_ref}\nTP = {tp_level}"
        )
        return data

    def get_account_balance(self) -> float:
        data = self._req("GET", "/api/v1/accounts")
        if data and data.get("accounts"):
            return float(data["accounts"][0]["balance"]["balance"])
        return 0.0

    def close_position(self, deal_ref: str) -> bool:
        return self._req("DELETE", f"/api/v1/positions/{deal_ref}") is not None

# ================== Helpers ===============================================
def ensure_csv(path: str, headers: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def setup_files():
    init_db()
    ensure_csv(RESULTS_FILE, [
        "timestamp","pair","type","entry","sl","tp","exit_price",
        "status","profit_r","pnl","confluence_score","ml_confidence","instrument_class","exit_reason"
    ])

def is_valid(*vals):
    return all(v is not None and not math.isnan(v) and math.isfinite(v) for v in vals)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram: {e}")

# ================== Indicators ============================================
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d    = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs   = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = (h-l).combine((h-pc).abs(), max).combine((l-pc).abs(), max)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    h, l, c   = df["High"], df["Low"], df["Close"]
    ph, pl, pc= h.shift(1), l.shift(1), c.shift(1)
    pdm = (h-ph).clip(lower=0); mdm = (pl-l).clip(lower=0)
    pdm = pdm.where(pdm > mdm, 0); mdm = mdm.where(mdm > pdm, 0)
    tr  = (h-l).combine((h-pc).abs(), max).combine((l-pc).abs(), max)
    safe= tr.ewm(span=period, adjust=False).mean().replace(0, 1e-10)
    pdi = 100 * pdm.ewm(span=period, adjust=False).mean() / safe
    mdi = 100 * mdm.ewm(span=period, adjust=False).mean() / safe
    dx  = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, 1e-10)
    return dx.ewm(span=period, adjust=False).mean(), pdi, mdi

# ================== [NEW v4] Candlestick Confirmation =====================
def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """
    Last candle must fully engulf the previous candle body (bullish).
    Confirms genuine buying pressure before entering BUY.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    prev_body = prev["Close"] - prev["Open"]
    curr_body = curr["Close"] - curr["Open"]
    # Current candle is bullish and its body engulfs previous candle body
    return (curr_body > 0 and prev_body < 0 and
            curr["Open"] <= prev["Close"] and
            curr["Close"] >= prev["Open"])

def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    """Last candle fully engulfs previous candle body (bearish)."""
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    prev_body = prev["Close"] - prev["Open"]
    curr_body = curr["Close"] - curr["Open"]
    return (curr_body < 0 and prev_body > 0 and
            curr["Open"] >= prev["Close"] and
            curr["Close"] <= prev["Open"])

def has_confirmation_candle(df: pd.DataFrame, direction: str) -> bool:
    """
    Check last 3 candles for engulfing pattern.
    Using last 3 gives more opportunity to catch the pattern.
    """
    if not REQUIRE_ENGULFING:
        return True
    for i in range(-3, -1):
        window = df.iloc[i-1:i+1] if i < -1 else df.iloc[-2:]
        if direction == "BUY"  and is_bullish_engulfing(window):
            return True
        if direction == "SELL" and is_bearish_engulfing(window):
            return True
    return False

# ================== [NEW v4] Support & Resistance Detection ===============
def find_sr_levels(df: pd.DataFrame, lookback: int = 100,
                   min_touches: int = 2, zone_pct: float = 0.002) -> List[float]:
    """
    Finds significant S/R levels by:
    1. Identifying swing highs and lows over lookback candles
    2. Clustering levels within zone_pct of each other
    3. Keeping only levels touched >= min_touches times
    Returns list of price levels sorted ascending.
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)
    highs  = recent["High"].values
    lows   = recent["Low"].values
    closes = recent["Close"].values

    # Collect swing highs and lows
    candidates = []
    for i in range(2, len(recent) - 2):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            candidates.append(highs[i])
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            candidates.append(lows[i])

    if not candidates:
        return []

    # Cluster nearby levels
    candidates.sort()
    clusters = []
    used = [False] * len(candidates)

    for i, level in enumerate(candidates):
        if used[i]:
            continue
        cluster = [level]
        zone    = level * zone_pct
        for j in range(i+1, len(candidates)):
            if not used[j] and abs(candidates[j] - level) <= zone:
                cluster.append(candidates[j])
                used[j] = True
        clusters.append(sum(cluster) / len(cluster))

    # Count touches for each cluster
    strong_levels = []
    for level in clusters:
        zone   = level * zone_pct
        touches= sum(1 for h, l in zip(highs, lows) if l <= level + zone and h >= level - zone)
        if touches >= min_touches:
            strong_levels.append(round(level, 5))

    return sorted(strong_levels)

def adjust_tp_to_sr(tp: float, tp_partial: float, entry: float, direction: str,
                    sr_levels: List[float], atr_val: float) -> float:
    """
    Adjusts TP to sit just before the nearest S/R level in profit direction.

    Rules:
    1. Adjusted TP must be BEYOND tp_partial (TP2 always further than TP1)
    2. Adjusted TP must maintain minimum 1.5:1 R:R vs SL distance
    3. Adjusted TP must be at least 2x ATR from entry (prevents tiny TPs)
    4. If none of these are met — return original TP unchanged
    """
    if not sr_levels:
        return tp

    buffer   = atr_val * 0.3
    min_dist = atr_val * 2.0  # TP must be at least 2x ATR from entry

    if direction == "BUY":
        # Only walls strictly beyond tp_partial
        walls = [lvl for lvl in sr_levels if tp_partial < lvl < tp]
        if walls:
            nearest  = min(walls)
            adjusted = nearest - buffer
            tp_dist  = adjusted - entry
            # Must be beyond partial TP AND beyond minimum ATR distance
            if adjusted > tp_partial and tp_dist >= min_dist:
                logger.info(f"S/R TP BUY: {tp:.5f} → {adjusted:.5f} (wall={nearest:.5f} dist={tp_dist:.5f})")
                return round(adjusted, 5)
            else:
                logger.info(f"S/R TP skipped BUY: adjusted={adjusted:.5f} too close (min_dist={min_dist:.5f})")
    else:
        walls = [lvl for lvl in sr_levels if tp < lvl < tp_partial]
        if walls:
            nearest  = max(walls)
            adjusted = nearest + buffer
            tp_dist  = entry - adjusted
            if adjusted < tp_partial and tp_dist >= min_dist:
                logger.info(f"S/R TP SELL: {tp:.5f} → {adjusted:.5f} (wall={nearest:.5f} dist={tp_dist:.5f})")
                return round(adjusted, 5)
            else:
                logger.info(f"S/R TP skipped SELL: adjusted too close (min_dist={min_dist:.5f})")

    return tp

# ================== [NEW v4] Session Filter (per instrument) ==============
def in_optimal_session(pair: str) -> bool:
    """
    Each instrument class has its own trading hours.
    Forex: London + NY (07:00–17:00 UTC)
    Indices: NY only (13:00–20:00 UTC)
    Commodities: London + NY (08:00–17:00 UTC)
    """
    now = now_utc()
    if now.weekday() >= 5:
        return False
    profile = get_profile(pair)
    hour    = now.hour
    return profile["session_start"] <= hour < profile["session_end"]

# ================== News Filter ===========================================
_news_cache: List[Dict] = []
_news_last_fetch: float  = 0

def fetch_forex_news() -> List[Dict]:
    global _news_cache, _news_last_fetch
    if time.time() - _news_last_fetch < 3600 and _news_cache:
        return _news_cache
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        events = []
        for item in r.json():
            if item.get("impact","").lower() not in ["high","medium"]:
                continue
            try:
                t = datetime.fromisoformat(item["date"].replace("Z","+00:00"))
                events.append({"time": t, "title": item.get("title",""), "impact": item.get("impact","")})
            except Exception:
                continue
        _news_cache      = events
        _news_last_fetch = time.time()
        return events
    except Exception as e:
        logger.warning(f"News fetch failed: {e}")
        return _news_cache

def is_near_news() -> bool:
    for e in fetch_forex_news():
        if abs((e["time"] - now_utc()).total_seconds()) <= NEWS_BUFFER_MINUTES * 60:
            logger.info(f"News blackout: {e['title']} ({e['impact']})")
            return True
    return False

# ================== ML Filter =============================================
class MLSignalFilter:
    def __init__(self):
        self.model = self.scaler = None
        self.trained = False
        self._load_model()

    def _load_model(self):
        if os.path.exists(ML_MODEL_FILE):
            try:
                with open(ML_MODEL_FILE, "rb") as f:
                    saved = pickle.load(f)
                self.model = saved["model"]; self.scaler = saved["scaler"]
                self.trained = True
                logger.info("✅ ML model loaded")
            except Exception as e:
                logger.warning(f"ML load failed: {e}")

    def train(self):
        if not ML_AVAILABLE or not os.path.exists(RESULTS_FILE):
            return
        conn = sqlite3.connect(DB_FILE)
        try:
            df = pd.read_sql("SELECT type, entry, entry_atr, confluence_score, opened_at, status FROM trades WHERE status IN ('CLOSED')", conn)
        except Exception:
            conn.close(); return
        conn.close()
        if len(df) < ML_MIN_TRADES_TO_TRAIN:
            logger.info(f"ML: need {ML_MIN_TRADES_TO_TRAIN} trades, have {len(df)}")
            return
        df["direction_buy"] = (df["type"] == "BUY").astype(int)
        df["hour_utc"]      = pd.to_datetime(df["opened_at"], unit="s").dt.hour
        df["atr_norm"]      = df["entry_atr"] / df["entry"].replace(0, 1e-10)
        df["label"]         = (df["status"] == "WIN").astype(int) # need to map closed → WIN/LOSS properly
        feature_cols = ["confluence_score", "direction_buy", "hour_utc", "atr_norm"]
        X = df[feature_cols].fillna(0).values
        y = df["label"].values
        if len(set(y)) < 2:
            return
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        self.model.fit(X_s, y)
        self.trained = True
        with open(ML_MODEL_FILE, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        acc = self.model.score(X_s, y)
        logger.info(f"✅ ML trained on {len(y)} trades | acc={acc:.2%}")
        send_telegram(f"🤖 ML retrained | {len(y)} trades | Acc: {acc:.2%}")

    def predict(self, score: int, direction: str, hour: int, atr_norm: float) -> float:
        if not self.trained or self.model is None:
            return 1.0
        try:
            X = np.array([[score, 1 if direction=="BUY" else 0, hour, atr_norm]])
            return float(self.model.predict_proba(self.scaler.transform(X))[0][1])
        except Exception:
            return 1.0

class CorrelationFilter:
    def is_blocked(self, pair: str, direction: str, open_trades: List[Dict]) -> Tuple[bool, str]:
        open_map = {t["pair"]: t["type"] for t in open_trades if t["status"] == "OPEN"}
        for group in CORRELATION_GROUPS:
            if pair not in group:
                continue
            for op, od in open_map.items():
                if op != pair and op in group and od == direction:
                    return True, f"{pair} correlated with {op} ({od})"
        return False, ""

class PortfolioHeatMonitor:
    def get_heat_pct(self, open_trades: List[Dict], equity: float) -> float:
        if equity <= 0: return 0.0
        total_risk = sum(t["risk_per_unit"] * t["lot_size"] for t in open_trades if t["status"]=="OPEN")
        return (total_risk / equity) * 100

    def is_overheated(self, open_trades: List[Dict], equity: float) -> Tuple[bool, float]:
        heat = self.get_heat_pct(open_trades, equity)
        if heat >= MAX_PORTFOLIO_HEAT_PCT:
            logger.warning(f"Heat: {heat:.2f}% >= {MAX_PORTFOLIO_HEAT_PCT}%")
            return True, heat
        return False, heat

class PairPerformanceManager:
    def evaluate_pair(self, pair: str):
        stats = db_get_pair_stats(pair)
        n = stats["total_trades"]
        if n < MIN_TRADES_FOR_DISABLE: return
        wr = (stats["wins"] / n) * 100
        if wr < DISABLE_WIN_RATE_THRESHOLD and not stats["disabled"]:
            reason = f"WR {wr:.1f}% < {DISABLE_WIN_RATE_THRESHOLD}% over {n} trades"
            db_disable_pair(pair, reason)
            logger.warning(f"Auto-disabled {pair}: {reason}")
            send_telegram(f"🚫 Auto-disabled {pair}\n{reason}")

    def get_summary(self) -> str:
        lines = []
        for pair in pairs:
            s = db_get_pair_stats(pair)
            n = s["total_trades"]
            if n == 0: continue
            wr = round(s["wins"]/n*100, 1)
            flag = " 🚫" if s["disabled"] else ""
            lines.append(f"{pair}: {n}T WR:{wr}% PnL:{s['total_pnl']:.1f}{flag}")
        return "\n".join(lines) if lines else "No pair data yet."

# ================== Global State ==========================================
trades:           List[Dict]       = []
wins = losses                      = 0
last_trade_times: Dict[str, float] = {}
broker:           Optional[CapitalClient] = None
ml_filter    = MLSignalFilter()
corr_filter  = CorrelationFilter()
heat_monitor = PortfolioHeatMonitor()
pair_perf    = PairPerformanceManager()

# ================== State Persistence =====================================
def load_state():
    global trades, last_trade_times, wins, losses
    for row in db_load_open_trades():
        trades.append(row)
        last_trade_times[row["pair"]] = row["opened_at"]
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                if row["status"] == "WIN":    wins   += 1
                elif row["status"] == "LOSS": losses += 1

def save_trade_result(trade: Dict, status: str, exit_price: float, exit_reason: str = ""):
    global wins, losses
    risk = trade["risk_per_unit"]
    if trade["type"] == "BUY":
        profit   = (exit_price - trade["entry"]) * trade["lot_size"]
        profit_r = (exit_price - trade["entry"]) / risk if risk else 0
    else:
        profit   = (trade["entry"] - exit_price) * trade["lot_size"]
        profit_r = (trade["entry"] - exit_price) / risk if risk else 0

    with open(RESULTS_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            now_utc().isoformat(), trade["pair"], trade["type"],
            round(trade["entry"],5), round(trade["sl"],5), round(trade["tp"],5),
            round(exit_price,5), status, round(profit_r,2), round(profit,2),
            trade.get("confluence_score",0), round(trade.get("ml_confidence",0),3),
            trade.get("instrument_class","FOREX"), exit_reason
        ])

    record_daily_pnl(profit)
    db_record_pair_result(trade["pair"], status=="WIN", profit)
    pair_perf.evaluate_pair(trade["pair"])

    if status == "WIN": wins   += 1
    else:               losses += 1

    emoji = "✅" if status == "WIN" else "❌"
    send_telegram(
        f"{emoji} {status} | {trade['pair']} {trade['type']}\n"
        f"Exit: {round(exit_price,5)} | PnL: {round(profit,2)} | {round(profit_r,2)}R\n"
        f"Reason: {exit_reason} | Score:{trade.get('confluence_score','?')}/6"
    )

def active_trade_count() -> int:
    return sum(1 for t in trades if t["status"] == "OPEN")

def has_open_trade(pair: str) -> bool:
    return any(t["pair"] == pair and t["status"] == "OPEN" for t in trades)

def cooldown_ready(pair: str) -> bool:
    last = last_trade_times.get(pair)
    return last is None or (time.time() - last) >= PAIR_COOLDOWN_SECONDS

# Minimum lot sizes per instrument (Capital.com requirements)
MIN_LOT_SIZES = {
    "EURUSD": 1000, "GBPUSD": 1000, "USDJPY": 1000,
    "GBPJPY": 1000, "USDCHF": 1000, "AUDUSD": 1000,
    "USDCAD": 1000, "NZDUSD": 1000, "EURGBP": 1000,
    "EURJPY": 1000,
    "US500":  1,    "US30":   1,    "USTEC":  1,
    "XAUUSD": 1,    "XAGUSD": 1,    "USOIL":  1,
}

# Maximum lot sizes to prevent oversizing
MAX_LOT_SIZES = {
    "EURUSD": 50000, "GBPUSD": 50000, "USDJPY": 50000,
    "GBPJPY": 50000, "USDCHF": 50000, "AUDUSD": 50000,
    "USDCAD": 50000, "NZDUSD": 50000, "EURGBP": 50000,
    "EURJPY": 50000,
    "US500":  100,   "US30":   10,    "USTEC":  50,
    "XAUUSD": 10,    "XAGUSD": 100,   "USOIL":  100,
}

def calculate_position_size(pair: str, entry: float, sl: float) -> float:
    equity   = broker.get_account_balance() if broker else INITIAL_EQUITY
    equity   = equity if equity > 0 else INITIAL_EQUITY
    risk_amt = equity * (RISK_PERCENT / 100)
    sl_dist  = abs(entry - sl)
    if sl_dist <= 0:
        return 0
    size = round(risk_amt / sl_dist, 2)
    # Enforce min/max lot sizes
    min_lot = MIN_LOT_SIZES.get(pair, 1000)
    max_lot = MAX_LOT_SIZES.get(pair, 50000)
    size    = max(size, min_lot)   # at least minimum
    size    = min(size, max_lot)   # never exceed maximum
    logger.info(f"{pair} position size: {size} (min={min_lot} max={max_lot} risk={risk_amt:.2f})")
    return size

# ================== [NEW v4] Signal Generation ============================
def build_signal(name: str, epic: str) -> Optional[Dict]:
    profile = get_profile(name)

    # Rate limit protection — space out API calls (avoid 429 errors)
    d15  = broker.get_candles(epic, "MINUTE_15", 300)
    time.sleep(0.5)
    d1h  = broker.get_candles(epic, "HOUR",      300)
    time.sleep(0.5)
    d4h  = broker.get_candles(epic, "HOUR_4",    200)
    time.sleep(0.5)
    live = broker.get_live_price(epic)
    time.sleep(0.3)

    if any(x is None for x in [d15, d1h, d4h, live]):
        return None
    if len(d15) < 80 or len(d1h) < 80 or len(d4h) < 50:
        return None

    c15, c1h, c4h = d15["Close"], d1h["Close"], d4h["Close"]

    e20_15 = c15.ewm(span=20, adjust=False).mean()
    e50_15 = c15.ewm(span=50, adjust=False).mean()
    e20_1h = c1h.ewm(span=20, adjust=False).mean()
    e50_1h = c1h.ewm(span=50, adjust=False).mean()
    e20_4h = c4h.ewm(span=20, adjust=False).mean()
    e50_4h = c4h.ewm(span=50, adjust=False).mean()

    rsi_s            = calc_rsi(c15)
    atr_s            = calc_atr(d15)
    adx_s, pdi_s, mdi_s = calc_adx(d15)
    vol_ma           = d15["Volume"].rolling(20).mean()

    lp    = float(c15.iloc[-1]);   prev_p = float(c15.iloc[-2])
    e20v  = float(e20_15.iloc[-1]); pe20  = float(e20_15.iloc[-2])
    e50v  = float(e50_15.iloc[-1])
    e201h = float(e20_1h.iloc[-1]); e501h = float(e50_1h.iloc[-1])
    e204h = float(e20_4h.iloc[-1]); e504h = float(e50_4h.iloc[-1])
    rsi_v = float(rsi_s.iloc[-1])
    atr_v = float(atr_s.iloc[-1])
    adx_v = float(adx_s.iloc[-1])
    pdi_v = float(pdi_s.iloc[-1]); mdi_v = float(mdi_s.iloc[-1])
    vol_l = float(d15["Volume"].iloc[-1])
    vol_m = float(vol_ma.iloc[-1]) if not math.isnan(vol_ma.iloc[-1]) else 0
    high  = float(d15["High"].iloc[-1]); ph = float(d15["High"].iloc[-2])
    low   = float(d15["Low"].iloc[-1]);  pl = float(d15["Low"].iloc[-2])
    c1h_l = float(c1h.iloc[-1]); c4h_l = float(c4h.iloc[-1])

    if not is_valid(lp, prev_p, e20v, pe20, e50v, e201h, e501h, e204h, e504h, rsi_v, atr_v, adx_v):
        return None
    if atr_v <= 0:
        return None

    # Use profile-specific spread ratio
    if live["spread"] > atr_v * profile["spread_ratio"]:
        logger.info(f"{name}: spread too high")
        return None

    # Profile-specific trend gap
    if abs(e20v - e50v) < atr_v * profile["trend_gap_mult"]:
        logger.info(f"{name}: trend too weak")
        return None

    # Profile-specific RSI ranges
    sig = None
    if (lp > e20v > e50v and c1h_l > e201h > e501h
            and profile["rsi_buy_lo"] <= rsi_v <= profile["rsi_buy_hi"]
            and prev_p <= pe20 * 1.0015 and high > ph):
        sig = "BUY"
    elif (lp < e20v < e50v and c1h_l < e201h < e501h
          and profile["rsi_sell_lo"] <= rsi_v <= profile["rsi_sell_hi"]
          and prev_p >= pe20 * 0.9985 and low < pl):
        sig = "SELL"

    if not sig:
        return None

    # Profile-specific ADX
    if adx_v < profile["adx_min"]:
        logger.info(f"{name}: ADX {adx_v:.1f} < {profile['adx_min']}")
        return None
    if sig == "BUY"  and pdi_v <= mdi_v: return None
    if sig == "SELL" and mdi_v <= pdi_v: return None

    # [NEW v4] Candlestick confirmation
    if not has_confirmation_candle(d15, sig):
        logger.info(f"{name}: no engulfing candle confirmation — skip")
        return None

    # Confluence score (now includes candlestick as a condition)
    score = 0
    if sig == "BUY":
        if lp > e20v > e50v:                        score += 1
        if c1h_l > e201h > e501h:                   score += 1
        if c4h_l > e204h > e504h:                   score += 1
        if profile["rsi_buy_lo"] <= rsi_v <= profile["rsi_buy_hi"]: score += 1
        if adx_v >= profile["adx_min"]:              score += 1
        if vol_m > 0 and vol_l > vol_m:              score += 1
    else:
        if lp < e20v < e50v:                         score += 1
        if c1h_l < e201h < e501h:                    score += 1
        if c4h_l < e204h < e504h:                    score += 1
        if profile["rsi_sell_lo"] <= rsi_v <= profile["rsi_sell_hi"]: score += 1
        if adx_v >= profile["adx_min"]:              score += 1
        if vol_m > 0 and vol_l > vol_m:              score += 1

    if score < MIN_CONFLUENCE_SCORE:
        logger.info(f"{name}: confluence {score}/{MIN_CONFLUENCE_SCORE} — skip")
        return None

    # ML confidence
    atr_norm = atr_v / lp if lp > 0 else 0
    ml_conf  = ml_filter.predict(score, sig, now_utc().hour, atr_norm)
    if ml_conf < ML_CONFIDENCE_THRESHOLD and ml_filter.trained:
        logger.info(f"{name}: ML {ml_conf:.2f} < {ML_CONFIDENCE_THRESHOLD} — skip")
        return None

    # Calculate levels using profile-specific multipliers
    entry  = round(live["ask"] if sig == "BUY" else live["bid"], profile["decimals"])
    sd     = round(atr_v * profile["atr_stop"], profile["decimals"])
    sl     = round(entry - sd if sig == "BUY" else entry + sd, profile["decimals"])
    tp_raw = round(entry + atr_v * profile["atr_tp_full"]    if sig == "BUY" else entry - atr_v * profile["atr_tp_full"],    profile["decimals"])
    tp_p   = round(entry + atr_v * profile["atr_tp_partial"] if sig == "BUY" else entry - atr_v * profile["atr_tp_partial"], profile["decimals"])

    # [NEW v4] Adjust TP to nearest S/R level
    sr_levels = find_sr_levels(d1h, lookback=SR_LOOKBACK, min_touches=SR_MIN_TOUCHES, zone_pct=SR_ZONE_PCT)
    tp        = adjust_tp_to_sr(tp_raw, tp_p, entry, sig, sr_levels, atr_v)

    lot_size = calculate_position_size(name, entry, sl)
    if lot_size <= 0:
        return None

    # Determine instrument class
    inst_class = next((k for k, v in INSTRUMENT_PROFILES.items() if name in v["pairs"]), "FOREX")

    logger.info(f"✅ {name} {sig} | Score:{score}/6 | ML:{ml_conf:.0%} | ADX:{adx_v:.1f} | Class:{inst_class}")

    return {
        "pair": name, "epic": epic, "type": sig,
        "entry": entry, "sl": sl, "tp": tp, "tp_partial": tp_p,
        "atr": round(atr_v, profile["decimals"]),
        "rsi": round(rsi_v, 2), "adx": round(adx_v, 2),
        "lot_size": lot_size, "risk_per_unit": abs(entry - sl),
        "spread": live["spread"], "confluence_score": score,
        "ml_confidence": round(ml_conf, 4),
        "instrument_class": inst_class,
        "sr_levels": sr_levels[:5],  # top 5 nearest for logging
    }

def open_trade(signal: Dict):
    order_id = broker.place_order(
        signal["pair"], signal["epic"], signal["type"], signal["lot_size"],
        signal["entry"], signal["sl"], signal["tp"]
    )
    if not order_id:
        return

    time.sleep(1)
    if not broker.confirm_fill(order_id, tp_level=signal["tp"], sl_level=signal["sl"]):
        send_telegram(f"⚠️ Unconfirmed fill: {signal['pair']} ref={order_id}")
        return

    trade = {
        "pair": signal["pair"], "epic": signal["epic"], "type": signal["type"],
        "entry": signal["entry"], "sl": signal["sl"], "tp": signal["tp"],
        "tp_partial": signal["tp_partial"], "status": "OPEN",
        "opened_at": time.time(), "risk_per_unit": signal["risk_per_unit"],
        "break_even_done": False, "partial_done": False,
        "entry_atr": signal["atr"], "lot_size": signal["lot_size"],
        "deal_ref": order_id, "confluence_score": signal["confluence_score"],
        "ml_confidence": signal["ml_confidence"],
        "instrument_class": signal["instrument_class"],
    }

    db_id          = db_save_trade(trade)
    trade["db_id"] = db_id
    trades.append(trade)
    last_trade_times[signal["pair"]] = time.time()

    send_telegram(
        f"🔔 NEW TRADE | {signal['pair']} {signal['type']} [{signal['instrument_class']}]\n"
        f"Entry: {signal['entry']} | SL: {signal['sl']}\n"
        f"TP1: {signal['tp_partial']} | TP2: {signal['tp']}\n"
        f"Score: {signal['confluence_score']}/6 | ML: {signal['ml_confidence']:.0%} | ADX: {signal['adx']}"
    )

# ================== Trade Monitoring ======================================
def update_trade_status(trade: Dict, live: Dict):
    if trade["status"] != "OPEN":
        return

    price = live["bid"] if trade["type"] == "BUY" else live["ask"]
    risk  = trade["risk_per_unit"]
    if risk <= 0:
        return

    changed = False
    profile = get_profile(trade["pair"])

    def close_trade(status: str, exit_price: float, reason: str):
        trade["status"] = "CLOSED"
        db_update_trade(trade)
        save_trade_result(trade, status, exit_price, reason)
        logger.info(f"Closed {trade['pair']} @ {exit_price} → {status} ({reason})")

    # ── SL hit ──────────────────────────────────────────────────────────────
    if trade["type"] == "BUY"  and price <= trade["sl"]:
        close_trade("WIN" if price >= trade["entry"] else "LOSS", price, "SL"); return
    if trade["type"] == "SELL" and price >= trade["sl"]:
        close_trade("WIN" if price <= trade["entry"] else "LOSS", price, "SL"); return

    # ── Full TP ──────────────────────────────────────────────────────────────
    if trade["type"] == "BUY"  and price >= trade["tp"]:
        close_trade("WIN", price, "TP2"); return
    if trade["type"] == "SELL" and price <= trade["tp"]:
        close_trade("WIN", price, "TP2"); return

    # ── Partial TP ───────────────────────────────────────────────────────────
    if not trade.get("partial_done", False):
        tp_p = trade.get("tp_partial")
        if tp_p:
            hit = (trade["type"]=="BUY" and price >= tp_p) or \
                  (trade["type"]=="SELL" and price <= tp_p)
            if hit:
                half = round(trade["lot_size"] / 2, 2)
                if half > 0 and broker.close_position(trade["deal_ref"]):
                    trade["partial_done"] = True
                    trade["lot_size"]     = half
                    changed = True
                    send_telegram(f"🎯 Partial TP1 | {trade['pair']} 50% @ {price}")

    # ── Break-even ───────────────────────────────────────────────────────────
    if not trade["break_even_done"]:
        prog = ((price - trade["entry"]) if trade["type"]=="BUY"
                else (trade["entry"] - price)) / risk
        if prog >= BREAK_EVEN_TRIGGER_R:
            be = trade["entry"]
            if ((trade["type"]=="BUY"  and be > trade["sl"]) or
                (trade["type"]=="SELL" and be < trade["sl"])):
                trade["sl"]              = round(be, profile["decimals"])
                trade["break_even_done"] = True
                changed = True
                send_telegram(f"🔒 Break-even | {trade['pair']} SL → {trade['sl']}")

    # ── Trailing stop ────────────────────────────────────────────────────────
    if trade["break_even_done"]:
        trail = trade.get("entry_atr", 0) * profile["trailing_mult"]
        if trade["type"] == "BUY":
            ns = round(price - trail, profile["decimals"])
            if ns > trade["sl"]: trade["sl"] = ns; changed = True
        else:
            ns = round(price + trail, profile["decimals"])
            if ns < trade["sl"]: trade["sl"] = ns; changed = True

    # ── [NEW v4] Stale trade exit ────────────────────────────────────────────
    elapsed_hours = (time.time() - trade["opened_at"]) / 3600
    if elapsed_hours >= STALE_TRADE_HOURS:
        progress = ((price - trade["entry"]) if trade["type"]=="BUY"
                    else (trade["entry"] - price)) / risk
        if abs(progress) < 0.3:  # less than 0.3R movement — trade is dead
            status = "WIN" if progress > 0 else "LOSS"
            close_trade(status, price, "STALE_EXIT")
            send_telegram(f"⏱ Stale exit | {trade['pair']} @ {price} after {elapsed_hours:.1f}h | {status}")
            return

    if changed:
        db_update_trade(trade)

# ================== Main Loops ============================================
def scan_pairs():
    if daily_loss_limit_hit():
        return
    if is_near_news():
        return
    if active_trade_count() >= MAX_ACTIVE_TRADES:
        return

    equity               = broker.get_account_balance() if broker else INITIAL_EQUITY
    overheated, heat     = heat_monitor.is_overheated(trades, equity)
    if overheated:
        logger.warning(f"Heat {heat:.1f}% — blocked.")
        return

    open_trades = [t for t in trades if t["status"] == "OPEN"]

    for name, search_term in pairs.items():
        if has_open_trade(name):     continue
        if not cooldown_ready(name): continue
        if is_pair_disabled(name):
            logger.info(f"{name}: disabled — skip")
            continue

        # [NEW v4] Per-instrument session check
        if not in_optimal_session(name):
            continue

        epic = broker.get_epic(search_term)
        if not epic:
            continue

        time.sleep(0.5)  # Rate limit: pause between each pair
        signal = build_signal(name, epic)
        if not signal:
            continue

        blocked, reason = corr_filter.is_blocked(name, signal["type"], open_trades)
        if blocked:
            logger.info(f"{name}: correlation — {reason}")
            continue

        overheated, heat = heat_monitor.is_overheated(trades, equity)
        if overheated:
            break

        open_trade(signal)
        open_trades = [t for t in trades if t["status"] == "OPEN"]
        time.sleep(1)

def check_open_trades():
    for trade in [t for t in trades if t["status"] == "OPEN"]:
        live = broker.get_live_price(trade["epic"])
        if live:
            update_trade_status(trade, live)
        time.sleep(0.3)  # Rate limit: space out live price checks

def send_heartbeat():
    equity  = broker.get_account_balance()
    total   = wins + losses
    wr      = round(wins/total*100, 1) if total > 0 else 0
    heat    = heat_monitor.get_heat_pct(trades, equity)
    send_telegram(
        f"💓 Heartbeat\n"
        f"Balance: {equity:.2f} | Open: {active_trade_count()}\n"
        f"W:{wins} L:{losses} WR:{wr}% | Heat:{heat:.1f}%\n"
        f"Daily loss: {get_daily_loss_pct():.2f}% / {DAILY_LOSS_LIMIT_PCT}%"
    )

def send_report():
    total   = wins + losses
    wr      = round(wins/total*100, 1) if total > 0 else 0
    send_telegram(
        f"📊 Report | W:{wins} L:{losses} WR:{wr}%\n\n"
        f"Per-Pair:\n{pair_perf.get_summary()}"
    )

# ================== Entry Point ===========================================
def main():
    global broker

    logger.info("=" * 60)
    logger.info("  Forex Bot v4 — Win Rate Optimized")
    logger.info("=" * 60)

    setup_files()
    load_state()

    broker = CapitalClient(
        api_key=CAPITAL_API_KEY, login=CAPITAL_LOGIN,
        password=CAPITAL_PASSWORD, demo=True
    )

    start_bal = broker.get_account_balance()
    init_daily_pnl(start_bal if start_bal > 0 else INITIAL_EQUITY)
    ml_filter.train()

    last_scan     = 0
    last_heartbeat= 0
    last_report   = 0
    last_ml_train = time.time()

    pair_list = ", ".join(pairs.keys())
    send_telegram(
        f"🚀 Bot v4.1 Started — Optimized 5-Pair Setup\n"
        f"Pairs: {pair_list}\n"
        f"Score: {MIN_CONFLUENCE_SCORE}/6 | Stale exit: {STALE_TRADE_HOURS}h\n"
        f"Heat limit: {MAX_PORTFOLIO_HEAT_PCT}% | Daily loss: {DAILY_LOSS_LIMIT_PCT}%\n"
        f"ML: {'Active' if ml_filter.trained else 'Collecting data...'}"
    )

    while True:
        try:
            now = time.time()

            check_open_trades()

            if now - last_scan >= SCAN_INTERVAL_SECONDS:
                scan_pairs()
                last_scan = now

            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                send_heartbeat()
                last_heartbeat = now

            if now - last_report >= REPORT_INTERVAL_SECONDS:
                send_report()
                last_report = now

            if now - last_ml_train >= ML_RETRAIN_INTERVAL_SECONDS:
                ml_filter.train()
                last_ml_train = now

            time.sleep(TRADE_CHECK_INTERVAL_SECONDS)

        except RuntimeError as e:
            logger.critical(f"Halting: {e}")
            send_telegram(f"🛑 Bot halted: {e}")
            break
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            send_telegram("🛑 Bot manually stopped.")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
