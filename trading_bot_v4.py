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
MIN_CONFLUENCE_SCORE       = int  (os.getenv("MIN_CONFLUENCE_SCORE",      "4"))
DAILY_LOSS_LIMIT_PCT       = float(os.getenv("DAILY_LOSS_LIMIT_PCT",      "4.0"))
NEWS_BUFFER_MINUTES        = int  (os.getenv("NEWS_BUFFER_MINUTES",       "30"))
ADX_TREND_THRESHOLD        = float(os.getenv("ADX_TREND_THRESHOLD",       "20.0"))
MAX_PORTFOLIO_HEAT_PCT     = float(os.getenv("MAX_PORTFOLIO_HEAT_PCT",    "6.0"))
MIN_TRADES_FOR_DISABLE     = int  (os.getenv("MIN_TRADES_FOR_DISABLE",    "20"))
DISABLE_WIN_RATE_THRESHOLD = float(os.getenv("DISABLE_WIN_RATE_THRESHOLD","35.0"))
ML_MIN_TRADES_TO_TRAIN     = int  (os.getenv("ML_MIN_TRADES_TO_TRAIN",    "50"))
ML_CONFIDENCE_THRESHOLD    = float(os.getenv("ML_CONFIDENCE_THRESHOLD",   "0.60"))

# v4 NEW
STALE_TRADE_HOURS          = float(os.getenv("STALE_TRADE_HOURS",         "4.0"))   # close if no progress after N hours
SR_LOOKBACK                = int  (os.getenv("SR_LOOKBACK",               "100"))   # candles for S/R detection
SR_MIN_TOUCHES             = int  (os.getenv("SR_MIN_TOUCHES",            "2"))     # min touches to confirm S/R level
SR_ZONE_PCT                = float(os.getenv("SR_ZONE_PCT",               "0.002")) # S/R zone width (0.2% of price)
REQUIRE_ENGULFING          = os.getenv("REQUIRE_ENGULFING", "true").lower() == "true"

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
        "pairs": {"EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD","EURGBP","EURJPY","GBPJPY"},
        "atr_stop":         1.5,
        "atr_tp_full":      2.4,
        "atr_tp_partial":   1.5,
        "trailing_mult":    1.2,
        "spread_ratio":     0.20,
        "rsi_buy_lo":       50, "rsi_buy_hi":  72,
        "rsi_sell_lo":      28, "rsi_sell_hi": 50,
        "adx_min":          20.0,
        "trend_gap_mult":   0.25,
        "session_start":    7,   # London open
        "session_end":      17,  # NY close
        "decimals":         5,
        "min_dist":         0.00010,
    },
    # ── US Indices ───────────────────────────────────────────────────────────
    "INDEX": {
        "pairs": {"US500","US30","USTEC"},
        "atr_stop":         2.0,   # wider stop — indices are more volatile
        "atr_tp_full":      3.0,   # bigger target
        "atr_tp_partial":   1.8,
        "trailing_mult":    1.5,
        "spread_ratio":     0.30,  # allow wider spread
        "rsi_buy_lo":       52, "rsi_buy_hi":  70,
        "rsi_sell_lo":      30, "rsi_sell_hi": 48,
        "adx_min":          22.0,
        "trend_gap_mult":   0.30,
        "session_start":    13,  # NY open only — indices need NY session
        "session_end":      20,
        "decimals":         1,
        "min_dist":         5.0,
    },
    # ── Commodities (Gold, Silver, Oil) ─────────────────────────────────────
    "COMMODITY": {
        "pairs": {"XAUUSD","XAGUSD","USOIL"},
        "atr_stop":         1.8,
        "atr_tp_full":      2.8,
        "atr_tp_partial":   1.6,
        "trailing_mult":    1.4,
        "spread_ratio":     0.25,
        "rsi_buy_lo":       50, "rsi_buy_hi":  70,
        "rsi_sell_lo":      30, "rsi_sell_hi": 50,
        "adx_min":          25.0,  # higher ADX needed — commodities trend strongly or not at all
        "trend_gap_mult":   0.35,
        "session_start":    8,
        "session_end":      17,
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
    p = get_profile(pair)
    return p["decimals"], p["min_dist"]

# Instrument universe
pairs = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "USDCHF": "USDCHF", "AUDUSD": "AUDUSD", "USDCAD": "USDCAD",
    "NZDUSD": "NZDUSD", "EURGBP": "EURGBP", "EURJPY": "EURJPY",
    "GBPJPY": "GBPJPY",
    "XAUUSD": "Gold",   "XAGUSD": "Silver",  "USOIL": "Oil - US Crude",
    "US500":  "US 500", "US30":   "Wall Street", "USTEC": "US Tech 100",
}

CORRELATION_GROUPS = [
    {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},
    {"USDJPY", "USDCHF", "USDCAD"},
    {"EURJPY", "GBPJPY"},
    {"US500", "US30", "USTEC"},
    {"XAUUSD", "XAGUSD"},
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
            fn      = {"GET": self.session.get, "POST": self.session.post, "DELETE": self.session.delete}
            kwargs  = {"headers": headers, "timeout": 30}
            if method == "POST":
                kwargs["json"] = data
            r = fn[method](url, **kwargs)
            if r.status_code in [401, 403]:
                self.authenticate()
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
        v4: Uses per-instrument decimal precision.
        Tries stopLevel/limitLevel first, falls back to stopDistance/limitDistance.
        """
        decimals, min_dist = get_decimals(pair)
        sl_r = round(sl, decimals)
        tp_r = round(tp, decimals)

        # Enforce minimum distances
        if abs(entry - sl_r) < min_dist:
            sl_r = round(entry - min_dist if direction=="BUY" else entry + min_dist, decimals)
        if abs(tp_r - entry) < min_dist:
            tp_r = round(entry + min_dist if direction=="BUY" else entry - min_dist, decimals)

        logger.info(f"Order: {direction} {units} {epic} | SL={sl_r} TP={tp_r}")

        # Attempt 1 — absolute levels
        result = self._req("POST", "/api/v1/positions", {
            "epic": epic, "direction": direction, "size": units,
            "stopLevel": sl_r, "limitLevel": tp_r,
            "guaranteedStop": False, "forceOpen": True
        })
        logger.info(f"Broker response (attempt 1): {result}")
        if result and result.get("dealReference"):
            logger.info(f"✅ Order placed (levels): {epic} ref={result['dealReference']}")
            return result["dealReference"]

        # Attempt 2 — distances
        logger.warning(f"{epic}: levels rejected, trying distances...")
        sl_dist = max(round(abs(entry - sl_r), decimals), min_dist)
        tp_dist = max(round(abs(tp_r - entry), decimals), min_dist)
        result2 = self._req("POST", "/api/v1/positions", {
            "epic": epic, "direction": direction, "size": units,
            "stopDistance": sl_dist, "limitDistance": tp_dist,
            "guaranteedStop": False, "forceOpen": True
        })
        logger.info(f"Broker response (attempt 2): {result2}")
        if result2 and result2.get("dealReference"):
            logger.info(f"✅ Order placed (distances): {epic} ref={result2['dealReference']}")
            return result2["dealReference"]

        logger.error(f"Both order attempts failed for {epic}")
        return None

    def confirm_fill(self, deal_ref: str) -> Optional[Dict]:
        data = self._req("GET", f"/api/v1/confirms/{deal_ref}")
        if not da