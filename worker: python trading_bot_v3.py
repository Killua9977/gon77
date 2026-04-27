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
        pip = 0.01 if "JPY" in epic else (1.0 if any(x in epic for x in ["US500","US30","USTEC"])
              else (0.1 if "XAU" in epic else 0.0001))
        result = self._req("POST", "/api/v1/positions", {
            "epic": epic, "direction": direction, "size": units,
            "stopDistance":  max(int(abs(entry-sl) / pip), 1),
            "limitDistance": max(int(abs(tp-entry) / pip), 1),
            "guaranteedStop": False, "forceOpen": True
        })
        return result.get("dealReference") if result else None

    def confirm_fill(self, deal_ref: str) -> Optional[Dict]:
        data = self._req("GET", f"/api/v1/confirms/{deal_ref}")
        return data if data and data.get("status") in ["OPEN","ACCEPTED"] else None

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
        