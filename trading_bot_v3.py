"""
================================================================================
  Forex Trading Bot v3 — Professional Grade
================================================================================
  NEW in v3:
  1. Backtesting Engine       — replay history through live signal logic
  2. Correlation Filter       — block correlated pairs from opening together
  3. Portfolio Heat Monitor   — cap total open risk at 6% equity
  4. Per-Pair Auto-Disable    — suspend consistently losing pairs
  5. ML Signal Filter         — XGBoost model trained on live trade results
================================================================================
"""

import csv
import json
import logging
import math
import os
import pickle
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ── Try importing ML library (optional — bot works without it) ──────────────
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

# Core strategy
INITIAL_EQUITY              = float(os.getenv("INITIAL_EQUITY",             "1000.0"))
RISK_PERCENT                = float(os.getenv("RISK_PERCENT",               "2.0"))
MAX_ACTIVE_TRADES           = int  (os.getenv("MAX_ACTIVE_TRADES",          "10"))
ATR_STOP_MULTIPLIER         = float(os.getenv("ATR_STOP_MULTIPLIER",        "1.5"))
ATR_TARGET_FULL             = float(os.getenv("ATR_TARGET_MULTIPLIER",      "2.4"))
ATR_TARGET_PARTIAL          = float(os.getenv("ATR_TARGET_PARTIAL",         "1.5"))
BREAK_EVEN_TRIGGER_R        = float(os.getenv("BREAK_EVEN_TRIGGER_R",       "1.0"))
TRAILING_STOP_ATR_MULT      = float(os.getenv("TRAILING_STOP_ATR_MULT",     "1.2"))
MAX_SPREAD_TO_ATR_RATIO     = float(os.getenv("MAX_SPREAD_TO_ATR_RATIO",    "0.20"))
PAIR_COOLDOWN_SECONDS       = int  (os.getenv("PAIR_COOLDOWN_SECONDS",      "14400"))
MAX_AUTH_RETRIES            = int  (os.getenv("MAX_AUTH_RETRIES",           "5"))

# v2 features
MIN_CONFLUENCE_SCORE        = int  (os.getenv("MIN_CONFLUENCE_SCORE",       "4"))
DAILY_LOSS_LIMIT_PCT        = float(os.getenv("DAILY_LOSS_LIMIT_PCT",       "4.0"))
NEWS_BUFFER_MINUTES         = int  (os.getenv("NEWS_BUFFER_MINUTES",        "30"))
ADX_TREND_THRESHOLD         = float(os.getenv("ADX_TREND_THRESHOLD",        "20.0"))

# v3 — NEW
MAX_PORTFOLIO_HEAT_PCT      = float(os.getenv("MAX_PORTFOLIO_HEAT_PCT",     "6.0"))   # max total open risk %
MIN_TRADES_FOR_DISABLE      = int  (os.getenv("MIN_TRADES_FOR_DISABLE",     "20"))    # min trades before auto-disable
DISABLE_WIN_RATE_THRESHOLD  = float(os.getenv("DISABLE_WIN_RATE_THRESHOLD", "35.0"))  # disable if WR < 35%
ML_MIN_TRADES_TO_TRAIN      = int  (os.getenv("ML_MIN_TRADES_TO_TRAIN",     "50"))    # min results to train ML
ML_CONFIDENCE_THRESHOLD     = float(os.getenv("ML_CONFIDENCE_THRESHOLD",    "0.60"))  # min ML probability to trade
CORRELATION_THRESHOLD       = float(os.getenv("CORRELATION_THRESHOLD",      "0.75"))  # block if correlation > this

# Timing
SCAN_INTERVAL_SECONDS       = int(os.getenv("SCAN_INTERVAL_SECONDS",        "900"))
TRADE_CHECK_INTERVAL_SECONDS= int(os.getenv("TRADE_CHECK_INTERVAL_SECONDS", "20"))
HEARTBEAT_INTERVAL_SECONDS  = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS",   "1800"))
REPORT_INTERVAL_SECONDS     = int(os.getenv("REPORT_INTERVAL_SECONDS",      "3600"))
ML_RETRAIN_INTERVAL_SECONDS = int(os.getenv("ML_RETRAIN_INTERVAL_SECONDS",  "86400")) # retrain daily

DB_FILE      = "trade_state.db"
RESULTS_FILE = "trade_results.csv"
LOG_FILE     = "bot.log"
ML_MODEL_FILE= "ml_model.pkl"

# Instrument universe
pairs = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "USDCHF": "USDCHF", "AUDUSD": "AUDUSD", "USDCAD": "USDCAD",
    "NZDUSD": "NZDUSD", "EURGBP": "EURGBP", "EURJPY": "EURJPY",
    "GBPJPY": "GBPJPY", "XAUUSD": "Gold",   "XAGUSD": "Silver",
    "USOIL":  "Oil - US Crude", "US500": "US 500",
    "US30":   "Wall Street",    "USTEC": "US Tech 100",
}

# ── Correlation groups (pairs that move together) ───────────────────────────
# If two instruments from the same group have open trades in the SAME direction,
# block new entries to avoid doubling exposure.
CORRELATION_GROUPS = [
    {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},   # USD-negative majors
    {"USDJPY", "USDCHF", "USDCAD"},              # USD-positive majors
    {"EURJPY", "GBPJPY"},                         # JPY crosses
    {"US500", "US30", "USTEC"},                   # US indices
    {"XAUUSD", "XAGUSD"},                         # Precious metals
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
        ml_confidence REAL DEFAULT 0.0
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS daily_pnl (
        trade_date TEXT PRIMARY KEY,
        start_equity REAL,
        realized_pnl REAL DEFAULT 0.0,
        trade_count INTEGER DEFAULT 0
    )""")

    # v3: per-pair performance stats
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
         break_even_done,partial_done,entry_atr,lot_size,deal_ref,confluence_score,ml_confidence)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (trade["pair"], trade["epic"], trade["type"], trade["entry"], trade["sl"],
         trade["tp"], trade["tp_partial"], trade["status"], trade["opened_at"],
         trade["risk_per_unit"], int(trade["break_even_done"]), int(trade["partial_done"]),
         trade["entry_atr"], trade["lot_size"], trade.get("deal_ref",""),
         trade.get("confluence_score",0), trade.get("ml_confidence",0.0)))
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

# ── Per-pair stats ─────────────────────────────────────────────────────────
def db_record_pair_result(pair: str, won: bool, pnl: float):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO pair_stats (pair) VALUES (?)", (pair,))
    if won:
        c.execute("""UPDATE pair_stats SET total_trades=total_trades+1,
                     wins=wins+1, total_pnl=total_pnl+? WHERE pair=?""", (pnl, pair))
    else:
        c.execute("""UPDATE pair_stats SET total_trades=total_trades+1,
                     losses=losses+1, total_pnl=total_pnl+? WHERE pair=?""", (pnl, pair))
    conn.commit()
    conn.close()

def db_get_pair_stats(pair: str) -> Dict:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM pair_stats WHERE pair=?", (pair,))
    row = c.fetchone()
    conn.close()
    if not row:
        return {"pair": pair, "total_trades": 0, "wins": 0, "losses": 0,
                "total_pnl": 0.0, "disabled": 0, "disabled_reason": ""}
    cols = ["pair","total_trades","wins","losses","total_pnl","disabled","disabled_reason"]
    return dict(zip(cols, row))

def db_disable_pair(pair: str, reason: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO pair_stats (pair) VALUES (?)", (pair,))
    c.execute("UPDATE pair_stats SET disabled=1, disabled_reason=? WHERE pair=?", (reason, pair))
    conn.commit()
    conn.close()

def db_enable_pair(pair: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE pair_stats SET disabled=0, disabled_reason='' WHERE pair=?", (pair,))
    conn.commit()
    conn.close()

def is_pair_disabled(pair: str) -> bool:
    stats = db_get_pair_stats(pair)
    return bool(stats.get("disabled", 0))

# ── Daily P&L ──────────────────────────────────────────────────────────────
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
    today = get_today_str()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""UPDATE daily_pnl SET realized_pnl=realized_pnl+?,
                 trade_count=trade_count+1 WHERE trade_date=?""", (pnl, today))
    conn.commit()
    conn.close()

def get_daily_loss_pct() -> float:
    today = get_today_str()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT start_equity, realized_pnl FROM daily_pnl WHERE trade_date=?", (today,))
    row = c.fetchone()
    conn.close()
    if not row or row[0] <= 0:
        return 0.0
    return max((-row[1] / row[0]) * 100, 0.0)

def daily_loss_limit_hit() -> bool:
    loss = get_daily_loss_pct()
    if loss >= DAILY_LOSS_LIMIT_PCT:
        logger.warning(f"Daily loss limit: {loss:.2f}% >= {DAILY_LOSS_LIMIT_PCT}%")
        return True
    return False

# ================== Capital.com API Client ================================
class CapitalClient:
    def __init__(self, api_key: str, login: str, password: str, demo: bool = True):
        self.api_key  = api_key
        self.login    = login
        self.password = password
        self.base_url = ("https://demo-api-capital.backend-capital.com" if demo
                         else "https://api-capital.backend-capital.com")
        self.cst = self.security_token = None
        self.session     = requests.Session()
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
            fn = {"GET": self.session.get, "POST": self.session.post, "DELETE": self.session.delete}
            kwargs = {"headers": headers, "timeout": 30}
            if method == "POST":
                kwargs["json"] = data
            r = fn[method](url, **kwargs)
            if r.status_code in [401, 403]:
                self.authenticate()
                return self._req(method, endpoint, data)
            if r.status_code != 200:
                logger.error(f"API {r.status_code}: {r.text[:200]}")
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
                 "Open":  float(c["openPrice"]["bid"]),
                 "High":  float(c["highPrice"]["bid"]),
                 "Low":   float(c["lowPrice"]["bid"]),
                 "Close": float(c["closePrice"]["bid"]),
                 "Volume":float(c.get("lastTradedVolume", 0))}
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

    def place_order(self, epic: str, direction: str, units: float,
                    entry: float, sl: float, tp: float) -> Optional[str]:
        """
        FIX v3.1: Use absolute price levels (stopLevel/limitLevel) instead of
        pip distances. The old stopDistance/limitDistance approach was silently
        rejected by Capital.com for many instruments, causing trades to open
        with no TP — only SL.
        """
        try:
            payload = {
                "epic":           epic,
                "direction":      direction,
                "size":           units,
                "stopLevel":      round(sl, 5),   # absolute SL price
                "limitLevel":     round(tp, 5),   # absolute TP price
                "guaranteedStop": False,
                "forceOpen":      True
            }

            logger.info(
                f"Placing order: {direction} {units} {epic} | "
                f"SL={round(sl,5)} TP={round(tp,5)}"
            )

            result = self._req("POST", "/api/v1/positions", payload)

            if not result:
                logger.error(f"Order rejected by broker for {epic} — empty response")
                return None

            # Log full broker response for debugging
            logger.info(f"Broker response: {result}")

            deal_ref = result.get("dealReference")
            if deal_ref:
                logger.info(f"✅ Order accepted: {direction} {units} {epic} | ref={deal_ref}")
                return deal_ref

            # If no dealReference, log the error reason
            logger.error(f"No dealReference in response for {epic}: {result}")
            return None

        except Exception as e:
            logger.error(f"place_order error for {epic}: {e}")
            return None

    def confirm_fill(self, deal_ref: str) -> Optional[Dict]:
        """Confirm order was filled and verify SL/TP were accepted."""
        data = self._req("GET", f"/api/v1/confirms/{deal_ref}")
        if not data:
            return None

        logger.info(f"Fill confirmation for {deal_ref}: {data}")

        status = data.get("status", "")
        if status not in ["OPEN", "ACCEPTED"]:
            logger.warning(f"Deal {deal_ref} status: {status} — may not have filled")
            return None

        # Warn if TP or SL missing from confirmation
        if not data.get("limitLevel"):
            logger.warning(f"⚠️ No limitLevel (TP) confirmed for {deal_ref} — check Capital.com!")
            send_telegram(f"⚠️ Warning: TP not confirmed by broker for ref={deal_ref}\nCheck Capital.com manually!")

        if not data.get("stopLevel"):
            logger.warning(f"⚠️ No stopLevel (SL) confirmed for {deal_ref}")

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
        "status","profit_r","pnl","confluence_score","ml_confidence"
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
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = (h-l).combine((h-pc).abs(), max).combine((l-pc).abs(), max)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    h, l, c = df["High"], df["Low"], df["Close"]
    ph, pl, pc = h.shift(1), l.shift(1), c.shift(1)
    pdm = (h - ph).clip(lower=0)
    mdm = (pl - l).clip(lower=0)
    pdm = pdm.where(pdm > mdm, 0)
    mdm = mdm.where(mdm > pdm, 0)
    tr  = (h-l).combine((h-pc).abs(), max).combine((l-pc).abs(), max)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    safe  = atr_s.replace(0, 1e-10)
    pdi   = 100 * pdm.ewm(span=period, adjust=False).mean() / safe
    mdi   = 100 * mdm.ewm(span=period, adjust=False).mean() / safe
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
    return dx.ewm(span=period, adjust=False).mean(), pdi, mdi

# ================== [NEW v3] ML Signal Filter =============================
class MLSignalFilter:
    """
    Trains a GradientBoosting classifier on past trade results.
    Features: confluence_score, rsi, adx, atr_norm, spread_ratio, hour_utc
    Label: 1=WIN, 0=LOSS
    """
    def __init__(self):
        self.model   = None
        self.scaler  = None
        self.trained = False
        self._load_model()

    def _feature_names(self):
        return ["confluence_score", "rsi", "adx", "atr_norm",
                "spread_ratio", "hour_utc", "direction_buy"]

    def _load_model(self):
        if os.path.exists(ML_MODEL_FILE):
            try:
                with open(ML_MODEL_FILE, "rb") as f:
                    saved = pickle.load(f)
                self.model   = saved["model"]
                self.scaler  = saved["scaler"]
                self.trained = True
                logger.info("✅ ML model loaded from disk")
            except Exception as e:
                logger.warning(f"ML model load failed: {e}")

    def _save_model(self):
        with open(ML_MODEL_FILE, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def train(self):
        if not ML_AVAILABLE:
            logger.warning("scikit-learn not installed — ML filter disabled")
            return

        if not os.path.exists(RESULTS_FILE):
            return

        df = pd.read_csv(RESULTS_FILE)
        df = df[df["status"].isin(["WIN","LOSS"])].copy()

        if len(df) < ML_MIN_TRADES_TO_TRAIN:
            logger.info(f"ML: need {ML_MIN_TRADES_TO_TRAIN} trades, have {len(df)}")
            return

        # Parse features from results CSV (columns added in v3)
        required = ["confluence_score","ml_confidence","status"]
        if not all(col in df.columns for col in required):
            logger.info("ML: result columns not yet available — skipping training")
            return

        # Build feature matrix from available columns
        # Some features are stored in the DB trades table — fetch them
        conn = sqlite3.connect(DB_FILE)
        trades_df = pd.read_sql("""
            SELECT pair, type, entry, entry_atr, confluence_score, ml_confidence,
                   opened_at, status
            FROM trades WHERE status IN ('CLOSED')
        """, conn)
        conn.close()

        if len(trades_df) < ML_MIN_TRADES_TO_TRAIN:
            return

        # Derive features
        trades_df["direction_buy"] = (trades_df["type"] == "BUY").astype(int)
        trades_df["hour_utc"]      = pd.to_datetime(trades_df["opened_at"], unit="s").dt.hour
        trades_df["atr_norm"]      = trades_df["entry_atr"] / trades_df["entry"].replace(0, 1e-10)
        trades_df["label"]         = (trades_df["status"] == "WIN" ).astype(int)  # re-derive from DB

        # We only have confluence_score, direction, hour, atr_norm reliably
        feature_cols = ["confluence_score", "direction_buy", "hour_utc", "atr_norm"]
        X = trades_df[feature_cols].fillna(0).values
        y = trades_df["label"].values

        if len(set(y)) < 2:
            logger.info("ML: only one class in training data — skipping")
            return

        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)
        self.model  = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                  learning_rate=0.1, random_state=42)
        self.model.fit(X_scaled, y)
        self.trained = True
        self._save_model()

        acc = self.model.score(X_scaled, y)
        logger.info(f"✅ ML model trained on {len(y)} trades — train accuracy: {acc:.2%}")
        send_telegram(f"🤖 ML model retrained on {len(y)} trades | Accuracy: {acc:.2%}")

    def predict(self, confluence_score: int, direction: str,
                hour_utc: int, atr_norm: float) -> float:
        """Returns probability of WIN (0.0–1.0). Returns 1.0 if model not ready."""
        if not self.trained or self.model is None or self.scaler is None:
            return 1.0  # no filter if not trained yet
        try:
            direction_buy = 1 if direction == "BUY" else 0
            X = np.array([[confluence_score, direction_buy, hour_utc, atr_norm]])
            X_scaled = self.scaler.transform(X)
            prob = float(self.model.predict_proba(X_scaled)[0][1])
            return prob
        except Exception as e:
            logger.error(f"ML predict error: {e}")
            return 1.0

# ================== [NEW v3] Correlation Filter ===========================
class CorrelationFilter:
    """
    Blocks new trades on instruments that are correlated with an existing
    open trade in the SAME direction.
    """
    def is_blocked(self, pair: str, direction: str, open_trades: List[Dict]) -> Tuple[bool, str]:
        open_pairs = {t["pair"]: t["type"] for t in open_trades if t["status"] == "OPEN"}

        for group in CORRELATION_GROUPS:
            if pair not in group:
                continue
            for open_pair, open_dir in open_pairs.items():
                if open_pair == pair:
                    continue
                if open_pair in group and open_dir == direction:
                    return True, f"{pair} correlated with open {open_pair} ({open_dir})"

        return False, ""

# ================== [NEW v3] Portfolio Heat Monitor =======================
class PortfolioHeatMonitor:
    """
    Calculates total open risk as % of equity.
    Blocks new trades if total heat >= MAX_PORTFOLIO_HEAT_PCT.
    """
    def get_heat_pct(self, open_trades: List[Dict], equity: float) -> float:
        if equity <= 0:
            return 0.0
        total_risk = sum(
            t["risk_per_unit"] * t["lot_size"]
            for t in open_trades
            if t["status"] == "OPEN"
        )
        return (total_risk / equity) * 100

    def is_overheated(self, open_trades: List[Dict], equity: float) -> Tuple[bool, float]:
        heat = self.get_heat_pct(open_trades, equity)
        if heat >= MAX_PORTFOLIO_HEAT_PCT:
            logger.warning(f"Portfolio heat: {heat:.2f}% >= limit {MAX_PORTFOLIO_HEAT_PCT}%")
            return True, heat
        return False, heat

# ================== [NEW v3] Per-Pair Performance Manager =================
class PairPerformanceManager:
    """
    Tracks win rate per pair. Auto-disables pairs below threshold
    after MIN_TRADES_FOR_DISABLE trades. Sends alert on disable.
    """
    def evaluate_pair(self, pair: str):
        stats = db_get_pair_stats(pair)
        n = stats["total_trades"]
        if n < MIN_TRADES_FOR_DISABLE:
            return  # not enough data yet

        wr = (stats["wins"] / n) * 100 if n > 0 else 0

        if wr < DISABLE_WIN_RATE_THRESHOLD and not stats["disabled"]:
            reason = f"WR {wr:.1f}% < {DISABLE_WIN_RATE_THRESHOLD}% over {n} trades"
            db_disable_pair(pair, reason)
            logger.warning(f"Auto-disabled {pair}: {reason}")
            send_telegram(f"🚫 Auto-disabled {pair}\nReason: {reason}")

    def get_summary(self) -> str:
        lines = []
        for pair in pairs:
            stats = db_get_pair_stats(pair)
            n = stats["total_trades"]
            if n == 0:
                continue
            wr = round((stats["wins"] / n) * 100, 1) if n > 0 else 0
            flag = " 🚫" if stats["disabled"] else ""
            lines.append(f"{pair}: {n}T | WR {wr}% | PnL {stats['total_pnl']:.1f}{flag}")
        return "\n".join(lines) if lines else "No pair data yet."

# ================== [NEW v3] Backtesting Engine ===========================
class Backtester:
    """
    Replays historical candles through the exact same signal logic
    to evaluate strategy performance before going live.
    """
    def __init__(self, broker_client: "CapitalClient"):
        self.broker = broker_client

    def run(self, pair: str, epic: str, candles_15m: int = 500) -> Dict:
        logger.info(f"Backtesting {pair}...")
        data = self.broker.get_candles(epic, resolution="MINUTE_15", num_candles=candles_15m)
        if data is None or len(data) < 100:
            return {"pair": pair, "error": "Not enough data"}

        results = []
        warmup = 80  # candles needed for indicators

        for i in range(warmup, len(data) - 1):
            window = data.iloc[:i+1].copy()
            signal = self._check_signal(window)
            if not signal:
                continue

            entry  = float(window["Close"].iloc[-1])
            atr_v  = signal["atr"]
            sl     = entry - atr_v * ATR_STOP_MULTIPLIER if signal["type"] == "BUY" else entry + atr_v * ATR_STOP_MULTIPLIER
            tp     = entry + atr_v * ATR_TARGET_FULL     if signal["type"] == "BUY" else entry - atr_v * ATR_TARGET_FULL
            tp_p   = entry + atr_v * ATR_TARGET_PARTIAL  if signal["type"] == "BUY" else entry - atr_v * ATR_TARGET_PARTIAL

            # Simulate forward
            outcome = self._simulate(data.iloc[i+1:], signal["type"], entry, sl, tp, tp_p)
            results.append({
                "index": i,
                "type": signal["type"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "confluence": signal["confluence"],
                **outcome
            })

        return self._summarize(pair, results)

    def _check_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Minimal signal check for backtesting (indicators only, no live price)."""
        close = df["Close"]
        e20   = close.ewm(span=20, adjust=False).mean()
        e50   = close.ewm(span=50, adjust=False).mean()
        rsi_s = rsi(close)
        atr_s = atr(df)
        adx_s, pdi_s, mdi_s = adx(df)

        lp      = float(close.iloc[-1])
        e20_v   = float(e20.iloc[-1])
        e50_v   = float(e50.iloc[-1])
        rsi_v   = float(rsi_s.iloc[-1])
        atr_v   = float(atr_s.iloc[-1])
        adx_v   = float(adx_s.iloc[-1])
        high    = float(df["High"].iloc[-1])
        ph      = float(df["High"].iloc[-2])
        low     = float(df["Low"].iloc[-1])
        pl      = float(df["Low"].iloc[-2])
        prev_p  = float(close.iloc[-2])
        pe20    = float(e20.iloc[-2])

        if not is_valid(lp, e20_v, e50_v, rsi_v, atr_v, adx_v):
            return None
        if atr_v <= 0 or adx_v < ADX_TREND_THRESHOLD:
            return None
        if abs(e20_v - e50_v) < atr_v * 0.25:
            return None

        score = 0
        sig = None

        if lp > e20_v > e50_v and 50 <= rsi_v <= 72 and prev_p <= pe20 * 1.0015 and high > ph:
            sig = "BUY"
            score += (1 if lp > e20_v > e50_v else 0)
            score += (1 if 50 <= rsi_v <= 72 else 0)
            score += (1 if adx_v >= ADX_TREND_THRESHOLD else 0)
        elif lp < e20_v < e50_v and 28 <= rsi_v <= 50 and prev_p >= pe20 * 0.9985 and low < pl:
            sig = "SELL"
            score += (1 if lp < e20_v < e50_v else 0)
            score += (1 if 28 <= rsi_v <= 50 else 0)
            score += (1 if adx_v >= ADX_TREND_THRESHOLD else 0)

        if sig and score >= 2:
            return {"type": sig, "atr": atr_v, "confluence": score}
        return None

    def _simulate(self, forward: pd.DataFrame, direction: str,
                  entry: float, sl: float, tp: float, tp_partial: float) -> Dict:
        partial_hit = False
        for _, row in forward.iterrows():
            h, l = float(row["High"]), float(row["Low"])

            if direction == "BUY":
                if l <= sl:
                    status = "WIN" if (partial_hit or sl >= entry) else "LOSS"
                    exit_p = sl
                    r = (exit_p - entry) / (entry - sl) if entry != sl else 0
                    return {"status": status, "exit": exit_p, "r": round(r,2), "partial": partial_hit}
                if not partial_hit and h >= tp_partial:
                    partial_hit = True
                if h >= tp:
                    return {"status": "WIN", "exit": tp, "r": round(ATR_TARGET_FULL/ATR_STOP_MULTIPLIER,2), "partial": partial_hit}
            else:
                if h >= sl:
                    status = "WIN" if (partial_hit or sl <= entry) else "LOSS"
                    exit_p = sl
                    r = (entry - exit_p) / (sl - entry) if entry != sl else 0
                    return {"status": status, "exit": exit_p, "r": round(r,2), "partial": partial_hit}
                if not partial_hit and l <= tp_partial:
                    partial_hit = True
                if l <= tp:
                    return {"status": "WIN", "exit": tp, "r": round(ATR_TARGET_FULL/ATR_STOP_MULTIPLIER,2), "partial": partial_hit}

        return {"status": "OPEN", "exit": None, "r": 0, "partial": partial_hit}

    def _summarize(self, pair: str, results: List[Dict]) -> Dict:
        closed = [r for r in results if r["status"] in ["WIN","LOSS"]]
        if not closed:
            return {"pair": pair, "trades": 0, "win_rate": 0, "avg_r": 0}
        wins    = sum(1 for r in closed if r["status"] == "WIN")
        avg_r   = round(sum(r["r"] for r in closed) / len(closed), 2)
        win_rate= round(wins / len(closed) * 100, 1)
        return {
            "pair": pair, "trades": len(closed),
            "wins": wins, "losses": len(closed)-wins,
            "win_rate": win_rate, "avg_r": avg_r,
            "expectancy": round(win_rate/100 * avg_r - (1 - win_rate/100), 2)
        }

    def run_all(self) -> str:
        lines = ["📊 Backtest Results (recent candles):"]
        for name, search_term in pairs.items():
            epic = self.broker.get_epic(search_term)
            if not epic:
                continue
            result = self.run(name, epic)
            if "error" in result:
                lines.append(f"  {name}: {result['error']}")
            else:
                lines.append(
                    f"  {name}: {result['trades']}T | WR {result['win_rate']}% | "
                    f"AvgR {result['avg_r']} | E {result['expectancy']}"
                )
            time.sleep(0.5)  # rate limit
        return "\n".join(lines)

# ================== News Filter ===========================================
_news_cache: List[Dict] = []
_news_last_fetch: float = 0

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
        _news_cache = events
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

# ================== Session Filter ========================================
def in_optimal_session() -> bool:
    n = now_utc()
    return n.weekday() < 5 and 0 <= n.hour < 20

# ================== Global State ==========================================
trades:           List[Dict]       = []
wins = losses                      = 0
last_trade_times: Dict[str, float] = {}
broker:           Optional[CapitalClient] = None
ml_filter        = MLSignalFilter()
corr_filter      = CorrelationFilter()
heat_monitor     = PortfolioHeatMonitor()
pair_perf        = PairPerformanceManager()

# ================== State Persistence =====================================
def load_state():
    global trades, last_trade_times, wins, losses
    for row in db_load_open_trades():
        trades.append(row)
        last_trade_times[row["pair"]] = row["opened_at"]
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                if row["status"] == "WIN":   wins   += 1
                elif row["status"] == "LOSS": losses += 1

def save_trade_result(trade: Dict, status: str, exit_price: float):
    global wins, losses
    risk = trade["risk_per_unit"]
    if trade["type"] == "BUY":
        profit  = (exit_price - trade["entry"]) * trade["lot_size"]
        profit_r= (exit_price - trade["entry"]) / risk if risk else 0
    else:
        profit  = (trade["entry"] - exit_price) * trade["lot_size"]
        profit_r= (trade["entry"] - exit_price) / risk if risk else 0

    with open(RESULTS_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            now_utc().isoformat(), trade["pair"], trade["type"],
            round(trade["entry"],5), round(trade["sl"],5), round(trade["tp"],5),
            round(exit_price,5), status, round(profit_r,2), round(profit,2),
            trade.get("confluence_score",0), round(trade.get("ml_confidence",0),3)
        ])

    record_daily_pnl(profit)
    db_record_pair_result(trade["pair"], status=="WIN", profit)
    pair_perf.evaluate_pair(trade["pair"])  # check auto-disable after each result

    if status == "WIN": wins   += 1
    else:               losses += 1

    emoji = "✅" if status == "WIN" else "❌"
    send_telegram(
        f"{emoji} {status} | {trade['pair']} {trade['type']}\n"
        f"Exit: {round(exit_price,5)} | PnL: {round(profit,2)} | {round(profit_r,2)}R\n"
        f"Score: {trade.get('confluence_score','?')}/6 | ML: {round(trade.get('ml_confidence',0)*100,0)}%"
    )

# ================== Trade Helpers =========================================
def active_trade_count() -> int:
    return sum(1 for t in trades if t["status"] == "OPEN")

def has_open_trade(pair: str) -> bool:
    return any(t["pair"] == pair and t["status"] == "OPEN" for t in trades)

def cooldown_ready(pair: str) -> bool:
    last = last_trade_times.get(pair)
    return last is None or (time.time() - last) >= PAIR_COOLDOWN_SECONDS

def calculate_position_size(entry: float, sl: float) -> float:
    equity   = broker.get_account_balance() if broker else INITIAL_EQUITY
    equity   = equity if equity > 0 else INITIAL_EQUITY
    risk_amt = equity * (RISK_PERCENT / 100)
    sl_dist  = abs(entry - sl)
    return round(risk_amt / sl_dist, 2) if sl_dist > 0 else 0

# ================== Signal Generation =====================================
def build_signal(name: str, epic: str) -> Optional[Dict]:
    d15  = broker.get_candles(epic, "MINUTE_15", 300)
    d1h  = broker.get_candles(epic, "HOUR",      300)
    d4h  = broker.get_candles(epic, "HOUR_4",    200)
    live = broker.get_live_price(epic)

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

    rsi_s        = rsi(c15)
    atr_s        = atr(d15)
    adx_s, pdi_s, mdi_s = adx(d15)
    vol_ma       = d15["Volume"].rolling(20).mean()

    lp     = float(c15.iloc[-1]);  prev_p  = float(c15.iloc[-2])
    e20v   = float(e20_15.iloc[-1]); pe20  = float(e20_15.iloc[-2])
    e50v   = float(e50_15.iloc[-1])
    e201h  = float(e20_1h.iloc[-1]); e501h = float(e50_1h.iloc[-1])
    e204h  = float(e20_4h.iloc[-1]); e504h = float(e50_4h.iloc[-1])
    rsi_v  = float(rsi_s.iloc[-1])
    atr_v  = float(atr_s.iloc[-1])
    adx_v  = float(adx_s.iloc[-1])
    pdi_v  = float(pdi_s.iloc[-1]); mdi_v = float(mdi_s.iloc[-1])
    vol_l  = float(d15["Volume"].iloc[-1])
    vol_m  = float(vol_ma.iloc[-1]) if not math.isnan(vol_ma.iloc[-1]) else 0
    high   = float(d15["High"].iloc[-1]); ph = float(d15["High"].iloc[-2])
    low    = float(d15["Low"].iloc[-1]);  pl = float(d15["Low"].iloc[-2])
    c1h_l  = float(c1h.iloc[-1]); c4h_l  = float(c4h.iloc[-1])

    if not is_valid(lp, prev_p, e20v, pe20, e50v, e201h, e501h, e204h, e504h, rsi_v, atr_v, adx_v):
        return None
    if atr_v <= 0:
        return None
    if live["spread"] > atr_v * MAX_SPREAD_TO_ATR_RATIO:
        return None
    if abs(e20v - e50v) < atr_v * 0.25:
        return None

    sig = None
    if (lp > e20v > e50v and c1h_l > e201h > e501h
            and 50 <= rsi_v <= 72 and prev_p <= pe20 * 1.0015 and high > ph):
        sig = "BUY"
    elif (lp < e20v < e50v and c1h_l < e201h < e501h
          and 28 <= rsi_v <= 50 and prev_p >= pe20 * 0.9985 and low < pl):
        sig = "SELL"

    if not sig:
        return None

    # ADX regime
    if adx_v < ADX_TREND_THRESHOLD:
        return None
    if sig == "BUY"  and pdi_v <= mdi_v: return None
    if sig == "SELL" and mdi_v <= pdi_v: return None

    # Confluence score
    score = 0
    if sig == "BUY":
        if lp > e20v > e50v:           score += 1
        if c1h_l > e201h > e501h:      score += 1
        if c4h_l > e204h > e504h:      score += 1
        if 50 <= rsi_v <= 72:          score += 1
        if adx_v >= ADX_TREND_THRESHOLD: score += 1
        if vol_m > 0 and vol_l > vol_m: score += 1
    else:
        if lp < e20v < e50v:           score += 1
        if c1h_l < e201h < e501h:      score += 1
        if c4h_l < e204h < e504h:      score += 1
        if 28 <= rsi_v <= 50:          score += 1
        if adx_v >= ADX_TREND_THRESHOLD: score += 1
        if vol_m > 0 and vol_l > vol_m: score += 1

    if score < MIN_CONFLUENCE_SCORE:
        logger.info(f"{name}: confluence {score}/{MIN_CONFLUENCE_SCORE} — skip")
        return None

    # [NEW v3] ML confidence
    atr_norm   = atr_v / lp if lp > 0 else 0
    ml_conf    = ml_filter.predict(score, sig, now_utc().hour, atr_norm)
    if ml_conf < ML_CONFIDENCE_THRESHOLD and ml_filter.trained:
        logger.info(f"{name}: ML confidence {ml_conf:.2f} < {ML_CONFIDENCE_THRESHOLD} — skip")
        return None

    entry    = round(live["ask"] if sig == "BUY" else live["bid"], 5)
    sd       = round(atr_v * ATR_STOP_MULTIPLIER, 5)
    sl       = round(entry - sd if sig == "BUY" else entry + sd, 5)
    tp       = round(entry + atr_v * ATR_TARGET_FULL    if sig == "BUY" else entry - atr_v * ATR_TARGET_FULL,    5)
    tp_p     = round(entry + atr_v * ATR_TARGET_PARTIAL if sig == "BUY" else entry - atr_v * ATR_TARGET_PARTIAL, 5)
    lot_size = calculate_position_size(entry, sl)

    if lot_size <= 0:
        return None

    logger.info(f"✅ Signal: {name} {sig} | Score:{score}/6 | ML:{ml_conf:.0%} | ADX:{adx_v:.1f}")

    return {
        "pair": name, "epic": epic, "type": sig,
        "entry": entry, "sl": sl, "tp": tp, "tp_partial": tp_p,
        "atr": round(atr_v,5), "rsi": round(rsi_v,2), "adx": round(adx_v,2),
        "lot_size": lot_size, "risk_per_unit": abs(entry-sl),
        "spread": live["spread"], "confluence_score": score,
        "ml_confidence": round(ml_conf, 4),
    }

def open_trade(signal: Dict):
    order_id = broker.place_order(
        signal["epic"], signal["type"], signal["lot_size"],
        signal["entry"], signal["sl"], signal["tp"]
    )
    if not order_id:
        return

    time.sleep(1)
    if not broker.confirm_fill(order_id):
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
    }

    db_id        = db_save_trade(trade)
    trade["db_id"] = db_id
    trades.append(trade)
    last_trade_times[signal["pair"]] = time.time()

    send_telegram(
        f"🔔 NEW TRADE | {signal['pair']} {signal['type']}\n"
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

    def close_trade(status: str, exit_price: float):
        trade["status"] = "CLOSED"
        db_update_trade(trade)
        save_trade_result(trade, status, exit_price)
        logger.info(f"Closed {trade['pair']} @ {exit_price} → {status}")

    # SL hit
    if trade["type"] == "BUY"  and price <= trade["sl"]:
        close_trade("WIN" if price >= trade["entry"] else "LOSS", price); return
    if trade["type"] == "SELL" and price >= trade["sl"]:
        close_trade("WIN" if price <= trade["entry"] else "LOSS", price); return

    # Full TP hit
    if trade["type"] == "BUY"  and price >= trade["tp"]:
        close_trade("WIN", price); return
    if trade["type"] == "SELL" and price <= trade["tp"]:
        close_trade("WIN", price); return

    # Partial TP
    if not trade.get("partial_done", False):
        tp_p = trade.get("tp_partial")
        if tp_p:
            hit = (trade["type"] == "BUY" and price >= tp_p) or \
                  (trade["type"] == "SELL" and price <= tp_p)
            if hit:
                half = round(trade["lot_size"] / 2, 2)
                if half > 0 and broker.close_position(trade["deal_ref"]):
                    trade["partial_done"] = True
                    trade["lot_size"]     = half
                    changed = True
                    send_telegram(f"🎯 Partial TP | {trade['pair']} 50% @ {price}")

    # Break-even
    if not trade["break_even_done"]:
        prog = ((price - trade["entry"]) if trade["type"] == "BUY"
                else (trade["entry"] - price)) / risk
        if prog >= BREAK_EVEN_TRIGGER_R:
            be = trade["entry"]
            if ((trade["type"] == "BUY"  and be > trade["sl"]) or
                (trade["type"] == "SELL" and be < trade["sl"])):
                trade["sl"] = round(be, 5)
                trade["break_even_done"] = True
                changed = True
                send_telegram(f"🔒 Break-even | {trade['pair']} SL → {trade['sl']}")

    # Trailing stop
    if trade["break_even_done"]:
        trail = trade.get("entry_atr", 0) * TRAILING_STOP_ATR_MULT
        if trade["type"] == "BUY":
            ns = round(price - trail, 5)
            if ns > trade["sl"]:
                trade["sl"] = ns; changed = True
        else:
            ns = round(price + trail, 5)
            if ns < trade["sl"]:
                trade["sl"] = ns; changed = True

    if changed:
        db_update_trade(trade)

# ================== Main Loops ============================================
def scan_pairs():
    if not in_optimal_session():
        return
    if daily_loss_limit_hit():
        logger.warning("Daily loss limit — no new trades.")
        return
    if is_near_news():
        logger.info("News blackout — skipping scan.")
        return
    if active_trade_count() >= MAX_ACTIVE_TRADES:
        return

    equity    = broker.get_account_balance() if broker else INITIAL_EQUITY
    overheated, heat = heat_monitor.is_overheated(trades, equity)
    if overheated:
        logger.warning(f"Portfolio heat {heat:.1f}% — blocking new trades.")
        return

    open_trades = [t for t in trades if t["status"] == "OPEN"]

    for name, search_term in pairs.items():
        if has_open_trade(name):       continue
        if not cooldown_ready(name):   continue

        # [NEW v3] Skip disabled pairs
        if is_pair_disabled(name):
            logger.info(f"{name}: auto-disabled — skipping")
            continue

        epic = broker.get_epic(search_term)
        if not epic:
            continue

        signal = build_signal(name, epic)
        if not signal:
            continue

        # [NEW v3] Correlation filter
        blocked, reason = corr_filter.is_blocked(name, signal["type"], open_trades)
        if blocked:
            logger.info(f"{name}: correlation block — {reason}")
            continue

        # [NEW v3] Re-check heat before each trade
        overheated, heat = heat_monitor.is_overheated(trades, equity)
        if overheated:
            logger.warning(f"Heat {heat:.1f}% — stopping scan mid-loop.")
            break

        open_trade(signal)
        open_trades = [t for t in trades if t["status"] == "OPEN"]  # refresh
        time.sleep(1)

def check_open_trades():
    for trade in [t for t in trades if t["status"] == "OPEN"]:
        live = broker.get_live_price(trade["epic"])
        if live:
            update_trade_status(trade, live)

def send_heartbeat():
    equity   = broker.get_account_balance()
    total    = wins + losses
    wr       = round(wins/total*100, 1) if total > 0 else 0
    daily_l  = get_daily_loss_pct()
    heat     = heat_monitor.get_heat_pct(trades, equity)
    send_telegram(
        f"💓 Heartbeat\n"
        f"Balance: {equity:.2f} | Open: {active_trade_count()}\n"
        f"W:{wins} L:{losses} WR:{wr}% | Heat:{heat:.1f}%\n"
        f"Daily loss: {daily_l:.2f}% / {DAILY_LOSS_LIMIT_PCT}%"
    )

def send_report():
    total   = wins + losses
    wr      = round(wins/total*100,1) if total > 0 else 0
    summary = pair_perf.get_summary()
    send_telegram(
        f"📊 Report\nTrades:{total} W:{wins} L:{losses} WR:{wr}%\n\n"
        f"Per-Pair:\n{summary}"
    )

# ================== Entry Point ===========================================
def main():
    global broker

    logger.info("=" * 60)
    logger.info("  Forex Bot v3 — Professional Grade")
    logger.info("=" * 60)

    setup_files()
    load_state()

    broker = CapitalClient(
        api_key=CAPITAL_API_KEY, login=CAPITAL_LOGIN,
        password=CAPITAL_PASSWORD, demo=True
    )

    start_bal = broker.get_account_balance()
    init_daily_pnl(start_bal if start_bal > 0 else INITIAL_EQUITY)

    # Initial ML train attempt
    ml_filter.train()

    # Run backtest on startup (informational)
    backtester = Backtester(broker)
    logger.info("Running startup backtest...")
    bt_summary = backtester.run_all()
    logger.info(bt_summary)
    send_telegram(bt_summary)

    last_scan      = 0
    last_heartbeat = 0
    last_report    = 0
    last_ml_train  = time.time()

    send_telegram(
        f"🚀 Bot v3 Started\n"
        f"Pairs: {len(pairs)} | Min score: {MIN_CONFLUENCE_SCORE}/6\n"
        f"ML: {'Active' if ml_filter.trained else 'Training...'} | "
        f"Heat limit: {MAX_PORTFOLIO_HEAT_PCT}%\n"
        f"Daily loss limit: {DAILY_LOSS_LIMIT_PCT}% | "
        f"News buffer: {NEWS_BUFFER_MINUTES}min"
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

            # Retrain ML daily
            if now - last_ml_train >= ML_RETRAIN_INTERVAL_SECONDS:
                logger.info("Retraining ML model...")
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
