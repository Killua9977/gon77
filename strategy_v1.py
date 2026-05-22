"""
================================================================================
  Multi-Strategy Professional Trading Bot
================================================================================
  US500  → Opening Range Breakout + Pre-market Bias Filter
           Range: 13:30-14:00 UTC | Trade: 14:00-20:00 UTC

  EURUSD → Asian Range Breakout at London Open
  GBPUSD → Asian Range Breakout at London Open
           Range: 23:00-07:00 UTC | Trade: 07:00-12:00 UTC

  USDJPY → Previous Day High/Low Breakout
           Range: previous day high/low | Trade: 07:00-20:00 UTC

  ALL PAIRS:
  - SL auto-moves to TP1 when hit (trade becomes risk-free)
  - SL auto-moves to TP2 when hit (trade closes on reversal)
  - No broker TP needed — all managed internally
  - Daily loss limit per pair and overall
  - One trade per pair per day
================================================================================
"""

import csv
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple

import requests

# ================== Configuration ==========================================
TOKEN            = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID          = os.getenv("TELEGRAM_CHAT_ID",   "").strip()
CAPITAL_API_KEY  = os.getenv("CAPITAL_API_KEY",    "").strip()
CAPITAL_LOGIN    = os.getenv("CAPITAL_LOGIN",       "").strip()
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD",    "").strip()
CAPITAL_DEMO     = os.getenv("CAPITAL_DEMO", "true").lower() == "true"

INITIAL_EQUITY       = float(os.getenv("INITIAL_EQUITY",       "1000.0"))
RISK_PERCENT         = float(os.getenv("RISK_PERCENT",         "1.0"))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0"))
MAX_AUTH_RETRIES     = int  (os.getenv("MAX_AUTH_RETRIES",     "5"))
NEWS_BUFFER_MINS     = int  (os.getenv("NEWS_BUFFER_MINS",     "30"))

# ORB Settings (US500)
ORB_RANGE_START  = (13, 30)   # NY open
ORB_RANGE_END    = (14,  0)   # range complete
ORB_TRADE_END    = (20,  0)   # force close
ORB_TP1_MULT     = float(os.getenv("ORB_TP1_MULT",  "1.0"))
ORB_TP2_MULT     = float(os.getenv("ORB_TP2_MULT",  "2.0"))
ORB_MIN_RANGE    = float(os.getenv("ORB_MIN_RANGE",  "5.0"))
ORB_MAX_RANGE    = float(os.getenv("ORB_MAX_RANGE",  "60.0"))
ORB_BUFFER       = float(os.getenv("ORB_BUFFER",     "0.5"))
PRE_MARKET_BIAS  = float(os.getenv("PRE_MARKET_BIAS","0.003")) # 0.3% drift needed

# Asian Range Breakout Settings (EURUSD, GBPUSD)
ARB_RANGE_START  = (23,  0)   # Asian session opens
ARB_RANGE_END    = ( 7,  0)   # London opens
ARB_TRADE_END    = (12,  0)   # stop trading
ARB_TP1_MULT     = float(os.getenv("ARB_TP1_MULT",  "1.0"))
ARB_TP2_MULT     = float(os.getenv("ARB_TP2_MULT",  "2.0"))
ARB_MIN_RANGE    = float(os.getenv("ARB_MIN_RANGE",  "0.0010"))  # 10 pips minimum
ARB_MAX_RANGE    = float(os.getenv("ARB_MAX_RANGE",  "0.0060"))  # 60 pips max
ARB_BUFFER       = float(os.getenv("ARB_BUFFER",     "0.0005"))  # 5 pip buffer
ARB_ADX_MIN      = float(os.getenv("ARB_ADX_MIN",    "18.0"))

# Previous Day Breakout Settings (USDJPY)
PDB_TRADE_START  = ( 7,  0)
PDB_TRADE_END    = (20,  0)
PDB_TP1_MULT     = float(os.getenv("PDB_TP1_MULT",  "1.0"))
PDB_TP2_MULT     = float(os.getenv("PDB_TP2_MULT",  "2.0"))
PDB_BUFFER       = float(os.getenv("PDB_BUFFER",     "0.05"))   # 5 pip buffer
PDB_MIN_RANGE    = float(os.getenv("PDB_MIN_RANGE",  "0.30"))   # 30 pip min range
PDB_MAX_RANGE    = float(os.getenv("PDB_MAX_RANGE",  "2.00"))   # 200 pip max

SCAN_INTERVAL    = int(os.getenv("SCAN_INTERVAL",    "10"))
HEARTBEAT_SECS   = int(os.getenv("HEARTBEAT_SECS",  "1800"))

DATA_DIR = os.getenv("DATA_DIR", ".").strip() or "."
os.makedirs(DATA_DIR, exist_ok=True)

DB_FILE      = os.path.join(DATA_DIR, "multi_state.db")
RESULTS_FILE = os.path.join(DATA_DIR, "multi_results.csv")
LOG_FILE     = os.path.join(DATA_DIR, "multi_bot.log")

# Instrument map
PAIRS = {
    "US500":  {"search": "US 500",  "strategy": "ORB",       "decimals": 1},
    "EURUSD": {"search": "EURUSD",  "strategy": "ARB",       "decimals": 5},
    "GBPUSD": {"search": "GBPUSD",  "strategy": "ARB",       "decimals": 5},
    "USDJPY": {"search": "USDJPY",  "strategy": "PDB",       "decimals": 3},
}

PAIR_NEWS_CURRENCIES = {
    "EURUSD": {"EUR", "USD"},
    "GBPUSD": {"GBP", "USD"},
    "USDJPY": {"USD", "JPY"},
}

US500_NEWS_KEYWORDS = (
    "fomc", "fed", "powell", "interest rate", "cpi", "inflation",
    "ppi", "pce", "payroll", "non-farm", "nfp", "unemployment",
    "jobless", "gdp", "retail sales", "ism", "pmi", "jolts",
    "consumer confidence", "minutes", "rate decision"
)

# ================== Logging ================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ================== Helpers ================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def today_str() -> str:
    return now_utc().date().isoformat()

def yesterday_str() -> str:
    return (now_utc() - timedelta(days=1)).date().isoformat()

def mins(h: int, m: int) -> int:
    return h * 60 + m

def cur_mins() -> int:
    n = now_utc()
    return mins(n.hour, n.minute)

def in_window(start: Tuple, end: Tuple) -> bool:
    """True if current UTC time is within window. Handles overnight windows."""
    now = now_utc()
    if now.weekday() >= 5:
        return False
    c = cur_mins()
    s = mins(*start)
    e = mins(*end)
    if s < e:
        return s <= c < e
    else:  # overnight window e.g. 23:00-07:00
        return c >= s or c < e

def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=10
        )
    except Exception as e:
        logger.error("Telegram: %s", e)

def ensure_csv(path: str, headers: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

# ================== News Filter ===========================================
_news_cache: List[Dict] = []
_news_last_fetch: float = 0.0

def fetch_news() -> List[Dict]:
    global _news_cache, _news_last_fetch
    if time.time() - _news_last_fetch < 3600 and _news_cache:
        return _news_cache
    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=10
        )
        events = []
        for item in r.json():
            if item.get("impact","").lower() not in {"high","medium"}:
                continue
            try:
                t = datetime.fromisoformat(item["date"].replace("Z","+00:00")).astimezone(timezone.utc)
                events.append({
                    "time": t,
                    "title": item.get("title",""),
                    "impact": item.get("impact",""),
                    "country": item.get("country","").upper(),
                })
            except Exception:
                continue
        _news_cache = events
        _news_last_fetch = time.time()
        return events
    except Exception as e:
        logger.warning("News fetch failed: %s", e)
        return _news_cache

def event_affects_pair(pair: str, event: Dict) -> bool:
    country = event.get("country", "").upper()
    impact = event.get("impact", "").lower()
    title = event.get("title", "").lower()

    if pair == "US500":
        if country == "ALL":
            return impact == "high"
        if country != "USD":
            return False
        return impact == "high" or any(keyword in title for keyword in US500_NEWS_KEYWORDS)

    currencies = PAIR_NEWS_CURRENCIES.get(pair, set())
    return country == "ALL" or country in currencies


def is_near_news(pair: Optional[str] = None) -> bool:
    now = now_utc()
    for event in fetch_news():
        if pair and not event_affects_pair(pair, event):
            continue
        if abs((event["time"] - now).total_seconds()) <= NEWS_BUFFER_MINS * 60:
            label = pair or "GLOBAL"
            logger.info(
                "%s news blackout: %s [%s/%s]",
                label, event["title"], event.get("country", "?"), event["impact"]
            )
            return True
    return False

# ================== Database ===============================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Daily ranges table
    c.execute("""CREATE TABLE IF NOT EXISTS daily_ranges (
        pair TEXT, trade_date TEXT,
        range_high REAL, range_low REAL, range_size REAL,
        bias TEXT DEFAULT '',
        PRIMARY KEY (pair, trade_date)
    )""")
    # Trades table
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT, strategy TEXT, trade_date TEXT,
        direction TEXT, entry REAL, sl REAL, tp1 REAL, tp2 REAL,
        range_high REAL, range_low REAL, range_size REAL,
        lot_size REAL, deal_ref TEXT, deal_id TEXT,
        initial_sl REAL DEFAULT 0, broker_sl REAL DEFAULT 0,
        risk_per_unit REAL DEFAULT 0,
        status TEXT DEFAULT 'OPEN', result TEXT DEFAULT '',
        exit_price REAL, exit_reason TEXT DEFAULT '',
        opened_at REAL, closed_at REAL,
        pnl REAL DEFAULT 0, profit_r REAL DEFAULT 0,
        tp1_locked INTEGER DEFAULT 0
    )""")
    # Daily PnL
    c.execute("""CREATE TABLE IF NOT EXISTS daily_pnl (
        trade_date TEXT PRIMARY KEY,
        start_equity REAL,
        realized_pnl REAL DEFAULT 0,
        trade_count INTEGER DEFAULT 0
    )""")
    existing_cols = {
        row[1] for row in c.execute("PRAGMA table_info(trades)").fetchall()
    }
    for col_name, col_type, default in [
        ("initial_sl", "REAL", "0"),
        ("broker_sl", "REAL", "0"),
        ("risk_per_unit", "REAL", "0"),
    ]:
        if col_name not in existing_cols:
            c.execute(
                f"ALTER TABLE trades ADD COLUMN {col_name} {col_type} DEFAULT {default}"
            )
    conn.commit()
    conn.close()

def db_save_range(pair: str, high: float, low: float, bias: str = ""):
    size = round(abs(high - low), 6)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""INSERT OR REPLACE INTO daily_ranges
        (pair, trade_date, range_high, range_low, range_size, bias)
        VALUES (?,?,?,?,?,?)""",
        (pair, today_str(), high, low, size, bias))
    conn.commit()
    conn.close()

def db_get_range(pair: str, date: str = None) -> Optional[Dict]:
    d    = date or today_str()
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""SELECT range_high, range_low, range_size, bias
                 FROM daily_ranges WHERE pair=? AND trade_date=?""", (pair, d))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"high": row[0], "low": row[1], "size": row[2], "bias": row[3]}

def db_save_trade(trade: Dict) -> int:
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""INSERT INTO trades
        (pair, strategy, trade_date, direction, entry, sl, tp1, tp2,
         range_high, range_low, range_size, lot_size, deal_ref, deal_id,
         initial_sl, broker_sl, risk_per_unit, status, opened_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (trade["pair"], trade["strategy"], today_str(),
         trade["direction"], trade["entry"], trade["sl"],
         trade["tp1"], trade["tp2"], trade["range_high"], trade["range_low"],
         trade["range_size"], trade["lot_size"], trade.get("deal_ref",""),
         trade.get("deal_id",""), trade.get("initial_sl", trade["sl"]),
         trade.get("broker_sl", trade["sl"]), trade.get("risk_per_unit", 0),
         "OPEN", trade["opened_at"]))
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id

def db_update_trade(trade: Dict):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""UPDATE trades SET sl=?, status=?, result=?,
        exit_price=?, exit_reason=?, closed_at=?, pnl=?,
        profit_r=?, tp1_locked=?, deal_id=?, deal_ref=?, broker_sl=?,
        initial_sl=?, risk_per_unit=? WHERE id=?""",
        (trade["sl"], trade["status"], trade.get("result",""),
         trade.get("exit_price"), trade.get("exit_reason",""),
         trade.get("closed_at"), trade.get("pnl",0),
         trade.get("profit_r",0), int(trade.get("tp1_locked",False)),
         trade.get("deal_id",""), trade.get("deal_ref",""),
         trade.get("broker_sl", trade["sl"]),
         trade.get("initial_sl", trade["sl"]), trade.get("risk_per_unit", 0),
         trade["db_id"]))
    conn.commit()
    conn.close()

def db_load_open_trades() -> List[Dict]:
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT * FROM trades WHERE status='OPEN'")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        d["tp1_locked"] = bool(d.get("tp1_locked",0))
        if not d.get("initial_sl"):
            d["initial_sl"] = d["sl"]
        if not d.get("broker_sl"):
            d["broker_sl"] = d["sl"]
        if not d.get("risk_per_unit"):
            d["risk_per_unit"] = abs(d["entry"] - d["initial_sl"])
        result.append(d)
    return result

def has_trade_today(pair: str) -> bool:
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT id FROM trades WHERE pair=? AND trade_date=?",
              (pair, today_str()))
    row = c.fetchone()
    conn.close()
    return row is not None

def init_daily_pnl(equity: float):
    today = today_str()
    conn  = sqlite3.connect(DB_FILE)
    c     = conn.cursor()
    c.execute("SELECT trade_date FROM daily_pnl WHERE trade_date=?", (today,))
    if not c.fetchone():
        c.execute("INSERT INTO daily_pnl VALUES (?,?,0,0)", (today, equity))
        conn.commit()
    conn.close()

def record_daily_pnl(pnl: float):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""UPDATE daily_pnl
        SET realized_pnl=realized_pnl+?, trade_count=trade_count+1
        WHERE trade_date=?""", (pnl, today_str()))
    conn.commit()
    conn.close()

def get_daily_loss_pct() -> float:
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("SELECT start_equity, realized_pnl FROM daily_pnl WHERE trade_date=?",
              (today_str(),))
    row = c.fetchone()
    conn.close()
    if not row or row[0] <= 0:
        return 0.0
    return max((-row[1] / row[0]) * 100, 0.0)

# ================== Capital.com API =======================================
class CapitalClient:
    def __init__(self, api_key: str, login: str, password: str, demo: bool = True):
        self.api_key  = api_key
        self.login    = login
        self.password = password
        self.demo     = demo
        self.base_url = ("https://demo-api-capital.backend-capital.com" if demo
                         else "https://api-capital.backend-capital.com")
        self.cst = self.security_token = None
        self.session       = requests.Session()
        self.epic_cache    = {}
        self._auth_retries = 0
        self.authenticate()

    def authenticate(self):
        if self._auth_retries >= MAX_AUTH_RETRIES:
            raise RuntimeError(f"Auth failed after {MAX_AUTH_RETRIES} retries")
        try:
            r = self.session.post(
                f"{self.base_url}/api/v1/session",
                headers={"X-CAP-API-KEY": self.api_key,
                         "Content-Type": "application/json"},
                json={"identifier": self.login, "password": self.password,
                      "encryptedPassword": False},
                timeout=30
            )
            if r.status_code != 200:
                raise Exception(r.json().get("errorMessage", r.text[:100]))
            self.cst            = r.headers.get("CST")
            self.security_token = r.headers.get("X-SECURITY-TOKEN")
            self._auth_retries  = 0
            logger.info("Connected to Capital.com (%s)",
                        "DEMO" if self.demo else "LIVE")
        except Exception as e:
            self._auth_retries += 1
            logger.error("Auth %s/%s: %s", self._auth_retries,
                         MAX_AUTH_RETRIES, e)
            if self._auth_retries < MAX_AUTH_RETRIES:
                time.sleep(5 * self._auth_retries)
                self.authenticate()
            else:
                raise

    def _req(self, method: str, endpoint: str,
             data: Dict = None) -> Optional[Dict]:
        try:
            url  = f"{self.base_url}{endpoint}"
            hdrs = {
                "X-CAP-API-KEY":    self.api_key,
                "CST":              self.cst,
                "X-SECURITY-TOKEN": self.security_token,
                "Content-Type":     "application/json"
            }
            fn = {"GET": self.session.get, "POST": self.session.post,
                  "PUT": self.session.put,  "DELETE": self.session.delete}[method]
            kw = {"headers": hdrs, "timeout": 30}
            if method in {"POST","PUT"}:
                kw["json"] = data
            r = fn(url, **kw)
            if r.status_code in {401, 403}:
                self.authenticate()
                return self._req(method, endpoint, data)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                logger.warning("Rate limited — waiting %ss", wait)
                time.sleep(wait)
                return self._req(method, endpoint, data)
            if r.status_code != 200:
                logger.error("API %s %s: %s",
                             method, endpoint, r.text[:200])
                return None
            return r.json()
        except RuntimeError:
            raise
        except Exception as e:
            logger.error("Request error: %s", e)
            return None

    def get_epic(self, search: str) -> Optional[str]:
        if search in self.epic_cache:
            return self.epic_cache[search]
        data = self._req("GET", f"/api/v1/markets?searchTerm={search}")
        if data and data.get("markets"):
            epic = data["markets"][0]["epic"]
            self.epic_cache[search] = epic
            return epic
        return None

    def get_candles(self, epic: str, resolution: str = "HOUR",
                    num: int = 24) -> Optional[List[Dict]]:
        data = self._req("GET",
            f"/api/v1/prices/{epic}?resolution={resolution}&max={num}")
        if not data or "prices" not in data:
            return None
        candles = []
        for c in data["prices"]:
            try:
                candles.append({
                    "time":   c["snapshotTime"],
                    "open":   float(c["openPrice"]["bid"]),
                    "high":   float(c["highPrice"]["bid"]),
                    "low":    float(c["lowPrice"]["bid"]),
                    "close":  float(c["closePrice"]["bid"]),
                    "volume": float(c.get("lastTradedVolume", 0)),
                })
            except Exception:
                continue
        return candles

    def get_live_price(self, epic: str) -> Optional[Dict]:
        data = self._req("GET", f"/api/v1/markets/{epic}")
        if not data or "snapshot" not in data:
            return None
        bid = float(data["snapshot"]["bid"])
        ask = float(data["snapshot"]["offer"])
        return {"bid": bid, "ask": ask,
                "mid": (bid+ask)/2, "spread": round(ask-bid, 6)}

    def get_open_positions(self) -> List[Dict]:
        data = self._req("GET", "/api/v1/positions")
        if not data:
            return []
        result = []
        for item in data.get("positions", []):
            pos    = item.get("position", {})
            market = item.get("market", {})
            result.append({
                "dealId":        pos.get("dealId",""),
                "dealReference": pos.get("dealReference",""),
                "epic":          market.get("epic",""),
                "direction":     pos.get("direction",""),
                "size":          float(pos.get("size",0) or 0),
                "level":         float(pos.get("level",0) or 0),
                "stopLevel":     pos.get("stopLevel"),
            })
        return result

    def _resolve(self, identifier: str,
                 epic_fallback: str = "") -> Optional[Dict]:
        """
        Find open position by dealId or dealReference.
        Falls back to epic name search if dealId is empty or mismatched.
        This fixes US500 close failures where dealId is not stored correctly.
        """
        positions = self.get_open_positions()

        # Primary: match by dealId or dealReference
        for pos in positions:
            if (pos["dealId"] == identifier or
                    pos["dealReference"] == identifier):
                return pos

        # Fallback: match by epic (when dealId is empty or wrong)
        if epic_fallback:
            for pos in positions:
                if pos.get("epic","").upper() == epic_fallback.upper():
                    logger.info(
                        "Position resolved by epic fallback: %s dealId=%s",
                        epic_fallback, pos["dealId"]
                    )
                    return pos

        logger.warning("Position not found: id=%s epic=%s",
                       identifier, epic_fallback)
        return None

    def place_order(self, epic: str, direction: str,
                    units: float, sl: float,
                    decimals: int = 5) -> Optional[str]:
        payload = {
            "epic":           epic,
            "direction":      direction,
            "size":           float(units),
            "stopLevel":      float(round(sl, decimals)),
            "guaranteedStop": False,
            "forceOpen":      True
        }
        logger.info("Order: %s %s %s sl=%.5f", direction, units, epic, sl)
        r = self._req("POST", "/api/v1/positions", payload)
        if r and r.get("dealReference"):
            logger.info("Order placed: %s", r["dealReference"])
            return r["dealReference"]
        logger.error("Order failed: %s", r)
        return None

    def confirm_fill(self, deal_ref: str) -> Optional[Dict]:
        time.sleep(2)
        data = self._req("GET", f"/api/v1/confirms/{deal_ref}")
        if not data:
            return None
        for item in data.get("affectedDeals", []):
            if item.get("status") == "OPENED":
                data["opened_deal_id"] = item.get("dealId")
                break
        return data

    def update_sl(self, identifier: str, new_sl: float,
                  decimals: int = 5) -> bool:
        pos = self._resolve(identifier)
        if not pos or not pos.get("dealId"):
            logger.warning("Cannot update SL — position not found: %s",
                           identifier)
            return False
        r = self._req("PUT", f"/api/v1/positions/{pos['dealId']}", {
            "stopLevel": float(round(new_sl, decimals))
        })
        if r:
            logger.info("SL updated: %s → %.5f", identifier, new_sl)
        return bool(r)

    def close_position(self, identifier: str,
                        epic_fallback: str = "") -> bool:
        pos = self._resolve(identifier, epic_fallback)
        if not pos:
            logger.warning(
                "Position not found: %s (epic=%s) — treating as closed",
                identifier, epic_fallback
            )
            return True
        result = self._req("DELETE", f"/api/v1/positions/{pos['dealId']}")
        if result is not None:
            logger.info("Position closed: %s dealId=%s",
                        identifier, pos["dealId"])
            return True
        logger.error("Close failed for dealId=%s", pos["dealId"])
        return False

    def get_account_balance(self) -> float:
        data = self._req("GET", "/api/v1/accounts")
        if data and data.get("accounts"):
            return float(data["accounts"][0]["balance"]["balance"])
        return 0.0

# ================== Shared Utilities ======================================
def get_point_value(pair: str) -> float:
    """Approximate quote-currency PnL per 1.0 price move for one unit."""
    if pair == "US500":
        return 1.0
    return 1.0

def calculate_lot_size(pair: str, equity: float,
                       entry: float, sl: float) -> float:
    """Risk RISK_PERCENT% of equity using per-instrument point value."""
    risk_amt = equity * (RISK_PERCENT / 100)
    sl_dist  = abs(entry - sl)
    point_value = get_point_value(pair)
    if sl_dist <= 0 or point_value <= 0:
        return 0.0
    size = round(risk_amt / (sl_dist * point_value), 2)
    # Min/max per instrument
    minimums = {"US500": 1, "EURUSD": 1000, "GBPUSD": 1000, "USDJPY": 1000}
    maximums = {"US500": 100, "EURUSD": 50000, "GBPUSD": 50000, "USDJPY": 50000}
    size = max(size, minimums.get(pair, 1000))
    size = min(size, maximums.get(pair, 50000))
    return size

def calculate_trade_pnl(pair: str, direction: str,
                        entry: float, exit_price: float,
                        lot_size: float) -> float:
    move = ((exit_price - entry) if direction == "BUY"
            else (entry - exit_price))
    return round(move * lot_size * get_point_value(pair), 2)


def make_disabled_range(high: float, low: float, size: float,
                        bias: str = "", reason: str = "") -> Dict:
    """Range exists, but the bot should not trade it again today."""
    return {
        "high": high,
        "low": low,
        "size": size,
        "bias": bias,
        "disabled": True,
        "skip_reason": reason,
    }

# ================== Strategy 1: ORB (US500) ================================
class ORBStrategy:
    """
    Opening Range Breakout for US500.
    Range: 13:30-14:00 UTC
    Entry: breakout after 14:00 UTC
    Filter: pre-market bias (08:00-13:30 drift direction)
    """

    @staticmethod
    def build_range(epic: str) -> Optional[Dict]:
        """
        Calculate opening range from 5-min candles.
        Also calculates pre-market bias from hourly candles.
        """
        # Get 5-min candles for range
        candles_5m = broker.get_candles(epic, "MINUTE_5", 20)
        if not candles_5m:
            return None

        # Filter to 13:30-14:00 UTC
        range_candles = []
        for c in candles_5m:
            try:
                t   = datetime.fromisoformat(c["time"].replace("Z","+00:00"))
                cur = mins(t.hour, t.minute)
                if mins(13,30) <= cur < mins(14,0):
                    range_candles.append(c)
            except Exception:
                continue

        if len(range_candles) < 4:
            logger.warning("US500: incomplete 13:30-14:00 candle set (%s candles)", len(range_candles))
            return None

        high = round(max(c["high"] for c in range_candles), 1)
        low  = round(min(c["low"]  for c in range_candles), 1)
        size = round(high - low, 2)

        if size < ORB_MIN_RANGE:
            logger.info("US500: range too small %.2f", size)
            send_telegram(f"?? US500 range too small: {size:.2f}pts ? no trade today")
            return make_disabled_range(high, low, size, reason="too_small")

        if size > ORB_MAX_RANGE:
            logger.info("US500: range too large %.2f (news spike?)", size)
            send_telegram(f"?? US500 range too large: {size:.2f}pts ? no trade today")
            return make_disabled_range(high, low, size, reason="too_large")

        # Pre-market bias: compare 08:00 price to 13:30 price
        candles_1h = broker.get_candles(epic, "HOUR", 8)
        bias = ""
        if candles_1h and len(candles_1h) >= 6:
            open_price = candles_1h[-6]["open"]   # ~08:00 UTC
            close_price= candles_1h[-1]["close"]  # ~13:00 UTC
            drift = (close_price - open_price) / open_price
            if drift > PRE_MARKET_BIAS:
                bias = "BULL"
            elif drift < -PRE_MARKET_BIAS:
                bias = "BEAR"
            logger.info("US500 pre-market bias: %.3f%% → %s",
                        drift*100, bias or "NEUTRAL")

        rng = {"high": high, "low": low, "size": size, "bias": bias, "disabled": False}
        db_save_range("US500", high, low, bias)

        send_telegram(
            f"📐 US500 Opening Range\n"
            f"High: {high:.2f} | Low: {low:.2f} | Size: {size:.2f}pts\n"
            f"Pre-market bias: {bias or 'NEUTRAL'}\n"
            f"BUY >  {high+ORB_BUFFER:.2f} | SELL < {low-ORB_BUFFER:.2f}\n"
            f"{'⚠️ Neutral bias — will trade both directions' if not bias else ''}"
        )
        return rng

    @staticmethod
    def check_entry(live: Dict, rng: Dict) -> Optional[str]:
        mid      = live["mid"]
        buy_lvl  = rng["high"] + ORB_BUFFER
        sell_lvl = rng["low"]  - ORB_BUFFER
        bias     = rng.get("bias","")

        if mid > buy_lvl:
            if bias == "BEAR":
                logger.info("US500: BUY signal but BEAR bias — skipping")
                return None
            return "BUY"
        if mid < sell_lvl:
            if bias == "BULL":
                logger.info("US500: SELL signal but BULL bias — skipping")
                return None
            return "SELL"
        return None

    @staticmethod
    def calculate_levels(direction: str, entry: float,
                         rng: Dict) -> Tuple[float, float, float]:
        size = rng["size"]
        if direction == "BUY":
            sl  = round(rng["low"]  - ORB_BUFFER,       1)
            tp1 = round(entry + size * ORB_TP1_MULT,     1)
            tp2 = round(entry + size * ORB_TP2_MULT,     1)
        else:
            sl  = round(rng["high"] + ORB_BUFFER,        1)
            tp1 = round(entry - size * ORB_TP1_MULT,     1)
            tp2 = round(entry - size * ORB_TP2_MULT,     1)
        return sl, tp1, tp2

    @staticmethod
    def is_trading_time() -> bool:
        now = now_utc()
        if now.weekday() >= 5:
            return False
        c = cur_mins()
        return mins(14,0) <= c < mins(*ORB_TRADE_END)

    @staticmethod
    def is_force_close_time() -> bool:
        now = now_utc()
        return now.weekday() < 5 and now.hour >= ORB_TRADE_END[0]

# ================== Strategy 2: Asian Range Breakout (EURUSD, GBPUSD) ===
class ARBStrategy:
    """
    Asian Range Breakout at London Open.
    Range: 23:00-07:00 UTC (Asian session)
    Entry: breakout at London open (07:00-12:00 UTC)
    Logic: London deliberately breaks Asian range to hunt stops,
           then trends in that direction for hours.
    """

    @staticmethod
    def build_range(pair: str, epic: str) -> Optional[Dict]:
        """
        Calculate Asian session range from hourly candles.
        Covers 23:00 yesterday to 07:00 today.
        """
        candles = broker.get_candles(epic, "HOUR", 12)
        if not candles:
            return None

        # Filter to Asian session: 23:00-07:00 UTC
        asian_candles = []
        for c in candles:
            try:
                t   = datetime.fromisoformat(c["time"].replace("Z","+00:00"))
                cur = mins(t.hour, t.minute)
                # 23:00 onwards or before 07:00
                if cur >= mins(23,0) or cur < mins(7,0):
                    asian_candles.append(c)
            except Exception:
                continue

        if not asian_candles:
            logger.warning("%s: no Asian session candles found", pair)
            return None

        high = round(max(c["high"] for c in asian_candles), 5)
        low  = round(min(c["low"]  for c in asian_candles), 5)
        size = round(high - low, 5)

        if size < ARB_MIN_RANGE:
            logger.info("%s: Asian range too small %.5f", pair, size)
            send_telegram(f"?? {pair} Asian range too small: "
                          f"{round(size*10000,1)}pips ? no trade today")
            return make_disabled_range(high, low, size, reason="too_small")

        if size > ARB_MAX_RANGE:
            logger.info("%s: Asian range too large %.5f", pair, size)
            send_telegram(f"?? {pair} Asian range too large: "
                          f"{round(size*10000,1)}pips ? likely news ? no trade")
            return make_disabled_range(high, low, size, reason="too_large")

        rng = {"high": high, "low": low, "size": size, "bias": "", "disabled": False}
        db_save_range(pair, high, low)

        send_telegram(
            f"📐 {pair} Asian Range\n"
            f"High: {high:.5f} | Low: {low:.5f}\n"
            f"Size: {round(size*10000,1)}pips\n"
            f"BUY  > {round(high+ARB_BUFFER,5):.5f}\n"
            f"SELL < {round(low -ARB_BUFFER,5):.5f}\n"
            f"London opens in a few minutes — watching..."
        )
        logger.info("%s Asian range: H=%.5f L=%.5f Size=%.5f",
                    pair, high, low, size)
        return rng

    @staticmethod
    def check_entry(live: Dict, rng: Dict) -> Optional[str]:
        mid      = live["mid"]
        buy_lvl  = rng["high"] + ARB_BUFFER
        sell_lvl = rng["low"]  - ARB_BUFFER

        if mid > buy_lvl:
            return "BUY"
        if mid < sell_lvl:
            return "SELL"
        return None

    @staticmethod
    def calculate_levels(direction: str, entry: float,
                         rng: Dict) -> Tuple[float, float, float]:
        size = rng["size"]
        if direction == "BUY":
            sl  = round(rng["low"]  - ARB_BUFFER,          5)
            tp1 = round(entry + size * ARB_TP1_MULT,        5)
            tp2 = round(entry + size * ARB_TP2_MULT,        5)
        else:
            sl  = round(rng["high"] + ARB_BUFFER,           5)
            tp1 = round(entry - size * ARB_TP1_MULT,        5)
            tp2 = round(entry - size * ARB_TP2_MULT,        5)
        return sl, tp1, tp2

    @staticmethod
    def is_trading_time() -> bool:
        now = now_utc()
        if now.weekday() >= 5:
            return False
        c = cur_mins()
        return mins(7,0) <= c < mins(*ARB_TRADE_END)

    @staticmethod
    def is_force_close_time() -> bool:
        now = now_utc()
        return now.weekday() < 5 and now.hour >= ARB_TRADE_END[0]

# ================== Strategy 3: Previous Day Breakout (USDJPY) ===========
class PDBStrategy:
    """
    Previous Day High/Low Breakout for USDJPY.
    USDJPY trends very strongly when it breaks key levels.
    Range: yesterday's high and low
    Entry: break of previous day high → BUY, low → SELL
    """

    @staticmethod
    def build_range(epic: str) -> Optional[Dict]:
        """Get yesterday's high and low from daily candles."""
        candles = broker.get_candles(epic, "DAY", 3)
        if not candles or len(candles) < 2:
            logger.warning("USDJPY: cannot get daily candles")
            return None

        # Previous day = second to last candle
        prev = candles[-2]
        high = round(prev["high"], 3)
        low  = round(prev["low"],  3)
        size = round(high - low,   3)

        if size < PDB_MIN_RANGE:
            logger.info("USDJPY: previous day range too small %.3f", size)
            return make_disabled_range(high, low, size, reason="too_small")

        if size > PDB_MAX_RANGE:
            logger.info("USDJPY: previous day range too large %.3f", size)
            return make_disabled_range(high, low, size, reason="too_large")

        rng = {"high": high, "low": low, "size": size, "bias": ""}
        db_save_range("USDJPY", high, low)

        send_telegram(
            f"📐 USDJPY Previous Day Range\n"
            f"High: {high:.3f} | Low: {low:.3f}\n"
            f"Range: {round(size*100,1)}pips\n"
            f"BUY  > {round(high+PDB_BUFFER,3):.3f}\n"
            f"SELL < {round(low -PDB_BUFFER,3):.3f}\n"
            f"Watching for breakout..."
        )
        logger.info("USDJPY range: H=%.3f L=%.3f Size=%.3f", high, low, size)
        return rng

    @staticmethod
    def check_entry(live: Dict, rng: Dict) -> Optional[str]:
        mid      = live["mid"]
        buy_lvl  = rng["high"] + PDB_BUFFER
        sell_lvl = rng["low"]  - PDB_BUFFER

        if mid > buy_lvl:
            return "BUY"
        if mid < sell_lvl:
            return "SELL"
        return None

    @staticmethod
    def calculate_levels(direction: str, entry: float,
                         rng: Dict) -> Tuple[float, float, float]:
        size = rng["size"]
        if direction == "BUY":
            sl  = round(rng["low"]  - PDB_BUFFER,          3)
            tp1 = round(entry + size * PDB_TP1_MULT,        3)
            tp2 = round(entry + size * PDB_TP2_MULT,        3)
        else:
            sl  = round(rng["high"] + PDB_BUFFER,           3)
            tp1 = round(entry - size * PDB_TP1_MULT,        3)
            tp2 = round(entry - size * PDB_TP2_MULT,        3)
        return sl, tp1, tp2

    @staticmethod
    def is_trading_time() -> bool:
        now = now_utc()
        if now.weekday() >= 5:
            return False
        c = cur_mins()
        return mins(*PDB_TRADE_START) <= c < mins(*PDB_TRADE_END)

    @staticmethod
    def is_force_close_time() -> bool:
        now = now_utc()
        return now.weekday() < 5 and now.hour >= PDB_TRADE_END[0]

# ================== Global State ==========================================
broker:       Optional[CapitalClient] = None
open_trades:  List[Dict]              = []
epics:        Dict[str, str]          = {}
ranges:       Dict[str, Dict]         = {}
ranges_built: Dict[str, bool]         = {p: False for p in PAIRS}
blocked_pairs: Set[str]               = set()
wins = losses = breakevens            = 0

orb = ORBStrategy()
arb = ARBStrategy()
pdb = PDBStrategy()

# ================== Trade Management =====================================
def load_state():
    global open_trades, wins, losses, breakevens, ranges

    open_trades = db_load_open_trades()
    for pair in PAIRS:
        rng = db_get_range(pair)
        if rng:
            ranges[pair]        = rng
            ranges_built[pair]  = True

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s = row.get("status","")
                if s == "WIN":    wins      += 1
                elif s == "LOSS": losses    += 1
                elif s == "BE":   breakevens+= 1

    logger.info("Loaded %s open trade(s)", len(open_trades))
    for pair, rng in ranges.items():
        logger.info("Loaded %s range: H=%s L=%s",
                    pair, rng["high"], rng["low"])

def save_result(trade: Dict, result: str,
                exit_price: float, reason: str):
    global wins, losses, breakevens

    decimals = PAIRS[trade["pair"]]["decimals"]
    risk_per_unit = trade.get("risk_per_unit") or abs(
        trade["entry"] - trade.get("initial_sl", trade["sl"])
    )
    pnl_dir  = ((exit_price - trade["entry"]) if trade["direction"]=="BUY"
                else (trade["entry"] - exit_price))
    pnl      = calculate_trade_pnl(
        trade["pair"], trade["direction"], trade["entry"], exit_price, trade["lot_size"]
    )
    profit_r = round(pnl_dir / risk_per_unit, 2) if risk_per_unit else 0

    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            now_utc().isoformat(), trade["pair"], trade["strategy"],
            trade["direction"], trade["entry"], trade["sl"],
            trade["tp1"], trade["tp2"], exit_price, result,
            profit_r, pnl, trade["range_size"], reason
        ])

    record_daily_pnl(pnl)
    if result=="WIN":    wins      += 1
    elif result=="LOSS": losses    += 1
    else:                breakevens+= 1

    emoji = "✅" if result=="WIN" else ("❌" if result=="LOSS" else "➖")
    send_telegram(
        f"{emoji} {result} | {trade['pair']} {trade['direction']}\n"
        f"Strategy: {trade['strategy']}\n"
        f"Exit: {round(exit_price,decimals)} | "
        f"PnL: {pnl} | {profit_r}R\n"
        f"Reason: {reason}"
    )

def finalize_trade(trade: Dict, exit_price: float,
                   reason: str, broker_close: bool = True):
    if trade["status"] != "OPEN":
        return

    if broker_close and broker:
        identifier = trade.get("deal_id") or trade.get("deal_ref","")
        epic_fb = epics.get(trade.get("pair",""), "")
        closed  = broker.close_position(identifier, epic_fb)
        if not closed:
            logger.error(
                "CLOSE FAILED for %s - position may still be open on broker!",
                trade.get("pair","")
            )
            send_telegram(
                f"Close failed for {trade.get('pair','')} {trade.get('direction','')}\n"
                f"Please close manually on Capital.com!"
            )
            return

    risk_per_unit = trade.get("risk_per_unit") or abs(
        trade["entry"] - trade.get("initial_sl", trade["sl"])
    )
    pnl_dir  = ((exit_price - trade["entry"]) if trade["direction"]=="BUY"
                else (trade["entry"] - exit_price))
    profit_r = pnl_dir / risk_per_unit if risk_per_unit else 0
    result   = ("WIN" if profit_r > 0.1
                else ("LOSS" if profit_r < -0.1 else "BE"))

    trade.update({
        "status":      "CLOSED",
        "result":      result,
        "exit_price":  exit_price,
        "exit_reason": reason,
        "closed_at":   time.time(),
        "pnl":         calculate_trade_pnl(
            trade["pair"], trade["direction"], trade["entry"], exit_price, trade["lot_size"]
        ),
        "profit_r":    round(profit_r, 2),
    })
    db_update_trade(trade)
    save_result(trade, result, exit_price, reason)
    logger.info("Closed %s %s @ %.5f → %s (%s)",
                trade["pair"], trade["direction"],
                exit_price, result, reason)

def update_sl_broker(trade: Dict, new_sl: float) -> bool:
    if not broker:
        return False
    decimals   = PAIRS[trade["pair"]]["decimals"]
    identifier = trade.get("deal_id") or trade.get("deal_ref","")
    epic_fb    = epics.get(trade.get("pair",""), "")

    if identifier:
        success = broker.update_sl(identifier, new_sl, decimals)
        if success:
            trade["broker_sl"] = new_sl
            db_update_trade(trade)
            return True

    if epic_fb:
        logger.warning(
            "%s: SL update by identifier failed - trying epic fallback",
            trade.get("pair","")
        )
        pos = broker._resolve("", epic_fb)
        if pos and pos.get("dealId"):
            trade["deal_id"] = pos["dealId"]
            result = broker._req(
                "PUT",
                f"/api/v1/positions/{pos['dealId']}",
                {"stopLevel": float(round(new_sl, decimals))}
            )
            if result:
                trade["broker_sl"] = new_sl
                db_update_trade(trade)
                logger.info("%s SL updated via epic fallback -> %.5f",
                            trade.get("pair",""), new_sl)
                return True
            logger.error("%s SL update failed completely",
                         trade.get("pair",""))
    return False

def pair_from_epic(epic: str) -> Optional[str]:
    epic_upper = epic.upper()
    for pair, mapped_epic in epics.items():
        if mapped_epic.upper() == epic_upper:
            return pair
    return None


def find_matching_broker_position(trade: Dict, broker_positions: List[Dict]) -> Optional[Dict]:
    identifiers = {trade.get("deal_id", ""), trade.get("deal_ref", "")}
    identifiers.discard("")
    for pos in broker_positions:
        if (pos.get("dealId", "") in identifiers or
                pos.get("dealReference", "") in identifiers):
            return pos

    pair_epic = epics.get(trade["pair"], "").upper()
    if not pair_epic:
        return None

    candidates = [
        pos for pos in broker_positions
        if pos.get("epic", "").upper() == pair_epic
        and pos.get("direction", "").upper() == trade["direction"].upper()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning("%s: multiple broker positions match the local trade", trade["pair"])
    return None


def close_trade_as_unknown(trade: Dict, reason: str, exit_price: Optional[float] = None):
    if trade["status"] != "OPEN":
        return

    exit_value = trade["entry"] if exit_price is None else exit_price
    trade.update({
        "status": "CLOSED",
        "result": "UNKNOWN",
        "exit_price": exit_value,
        "exit_reason": reason,
        "closed_at": time.time(),
        "pnl": 0,
        "profit_r": 0,
    })
    db_update_trade(trade)

    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            now_utc().isoformat(), trade["pair"], trade["strategy"],
            trade["direction"], trade["entry"], trade["sl"],
            trade["tp1"], trade["tp2"], exit_value, "UNKNOWN",
            0, 0, trade["range_size"], reason
        ])

    logger.warning("Closed %s locally as UNKNOWN (%s)", trade["pair"], reason)


def reconcile_state_with_broker():
    global blocked_pairs

    if not broker:
        return

    blocked_pairs.clear()
    broker_positions = broker.get_open_positions()
    matched_ids: Set[str] = set()
    matched_refs: Set[str] = set()

    for trade in [t for t in open_trades if t["status"] == "OPEN"]:
        pos = find_matching_broker_position(trade, broker_positions)
        if not pos:
            live = None
            epic = epics.get(trade["pair"], "")
            if epic:
                live = broker.get_live_price(epic)
            exit_price = None
            if live:
                exit_price = live["bid"] if trade["direction"] == "BUY" else live["ask"]
            close_trade_as_unknown(trade, "BROKER_POSITION_MISSING", exit_price)
            send_telegram(
                f"Startup reconciliation closed local {trade['pair']} {trade['direction']}\n"
                f"No matching broker position was found. Result marked UNKNOWN."
            )
            continue

        deal_id = pos.get("dealId", "")
        deal_ref = pos.get("dealReference", "")
        if deal_id:
            matched_ids.add(deal_id)
        if deal_ref:
            matched_refs.add(deal_ref)

        updated = False
        if deal_id and trade.get("deal_id") != deal_id:
            trade["deal_id"] = deal_id
            updated = True
        if deal_ref and trade.get("deal_ref") != deal_ref:
            trade["deal_ref"] = deal_ref
            updated = True

        stop_level = pos.get("stopLevel")
        if stop_level is not None:
            try:
                broker_sl = float(stop_level)
                if trade.get("broker_sl") != broker_sl:
                    trade["broker_sl"] = broker_sl
                    updated = True
            except (TypeError, ValueError):
                pass

        if updated:
            db_update_trade(trade)

    for pos in broker_positions:
        deal_id = pos.get("dealId", "")
        deal_ref = pos.get("dealReference", "")
        if (deal_id and deal_id in matched_ids) or (deal_ref and deal_ref in matched_refs):
            continue

        pair = pair_from_epic(pos.get("epic", ""))
        label = pair or pos.get("epic", "UNKNOWN")
        if pair:
            blocked_pairs.add(pair)

        logger.warning(
            "Untracked broker position detected: %s %s size=%s deal=%s",
            label, pos.get("direction", ""), pos.get("size", 0), deal_id or deal_ref
        )
        send_telegram(
            f"Untracked broker position detected\n"
            f"Instrument: {label}\n"
            f"Direction: {pos.get('direction', '')}\n"
            f"Size: {pos.get('size', 0)}\n"
            f"Deal ID: {deal_id or deal_ref}\n"
            f"This pair is blocked from new trades until you close it or restart after syncing."
        )

    logger.info(
        "Startup reconciliation complete: local_open=%s broker_open=%s blocked_pairs=%s",
        sum(1 for t in open_trades if t["status"] == "OPEN"),
        len(broker_positions),
        ",".join(sorted(blocked_pairs)) or "none"
    )


def check_trades(live_prices: Dict[str, Dict]):
    """
    Monitor all open trades.
    Internal logical SL can differ from the broker stop after TP1,
    so broker closes are always explicit when the bot decides the trade is done.
    """
    for trade in [t for t in open_trades if t["status"]=="OPEN"]:
        pair  = trade["pair"]
        live  = live_prices.get(pair)
        if not live:
            continue

        price    = live["bid"] if trade["direction"]=="BUY" else live["ask"]
        decimals = PAIRS[pair]["decimals"]

        # ── SL hit ──────────────────────────────────────────────────────
        if trade["direction"]=="BUY"  and price <= trade["sl"]:
            finalize_trade(trade, price, "SL", broker_close=True)
            continue
        if trade["direction"]=="SELL" and price >= trade["sl"]:
            finalize_trade(trade, price, "SL", broker_close=True)
            continue

        # ── TP2 hit → close ─────────────────────────────────────────────
        if trade["direction"]=="BUY"  and price >= trade["tp2"]:
            finalize_trade(trade, price, "TP2", broker_close=True)
            continue
        if trade["direction"]=="SELL" and price <= trade["tp2"]:
            finalize_trade(trade, price, "TP2", broker_close=True)
            continue

        # ── TP1 hit → move SL to TP1 (risk-free) ───────────────────────
        if not trade["tp1_locked"]:
            tp1_hit = ((trade["direction"]=="BUY"  and price >= trade["tp1"])
                    or (trade["direction"]=="SELL" and price <= trade["tp1"]))
            if tp1_hit:
                new_sl = round(trade["tp1"], decimals)
                trade["sl"]         = new_sl
                trade["tp1_locked"] = True

                pip = 0.0001 if pair != "USDJPY" else 0.01
                if pair == "US500":
                    pip = 1.0
                broker_sl = round(
                    new_sl - pip * 3 if trade["direction"]=="BUY"
                    else new_sl + pip * 3,
                    decimals
                )
                trade["broker_sl"] = broker_sl
                db_update_trade(trade)
                update_sl_broker(trade, broker_sl)
                logger.info("%s TP1 locked: SL → %.5f (broker=%.5f)",
                            pair, new_sl, broker_sl)
                send_telegram(
                    f"🔒 TP1 LOCKED | {pair} {trade['direction']}\n"
                    f"Price: {round(price,decimals)} hit TP1: "
                    f"{round(trade['tp1'],decimals)}\n"
                    f"SL moved to TP1 → RISK-FREE\n"
                    f"Targeting TP2: {round(trade['tp2'],decimals)}"
                )

        # ── Force close at strategy end time ────────────────────────────
        strategy = trade["strategy"]
        force_close = False
        if strategy == "ORB" and orb.is_force_close_time():
            force_close = True
        elif strategy == "ARB" and arb.is_force_close_time():
            force_close = True
        elif strategy == "PDB" and pdb.is_force_close_time():
            force_close = True

        if force_close:
            finalize_trade(trade, price, "EOD_CLOSE", broker_close=True)
            send_telegram(
                f"🕗 EOD Close | {pair} {trade['direction']} "
                f"@ {round(price,decimals)}"
            )

def open_trade(pair: str, direction: str, entry: float,
               sl: float, tp1: float, tp2: float,
               rng: Dict, strategy_name: str):
    """Open a trade on broker and record it."""
    if get_daily_loss_pct() >= DAILY_LOSS_LIMIT_PCT:
        logger.warning("Daily loss limit — no new trades")
        return
    if pair in blocked_pairs:
        logger.warning("%s: blocked from new trades due to untracked broker position", pair)
        return
    if is_near_news(pair):
        logger.info("News blackout — skipping %s entry", pair)
        return

    epic     = epics.get(pair)
    decimals = PAIRS[pair]["decimals"]
    if not epic:
        return

    equity   = broker.get_account_balance() if broker else INITIAL_EQUITY
    lot_size = calculate_lot_size(pair, equity, entry, sl)
    if lot_size <= 0:
        return

    order_ref = broker.place_order(epic, direction, lot_size, sl, decimals)
    if not order_ref:
        send_telegram(f"❌ Order failed: {pair} {direction}")
        return

    fill    = broker.confirm_fill(order_ref)
    deal_id = fill.get("opened_deal_id","") if fill else ""

    trade = {
        "pair":       pair,
        "strategy":   strategy_name,
        "direction":  direction,
        "entry":      entry,
        "sl":         sl,
        "initial_sl": sl,
        "broker_sl":  sl,
        "risk_per_unit": abs(entry - sl),
        "tp1":        tp1,
        "tp2":        tp2,
        "range_high": rng["high"],
        "range_low":  rng["low"],
        "range_size": rng["size"],
        "lot_size":   lot_size,
        "deal_ref":   order_ref,
        "deal_id":    deal_id,
        "status":     "OPEN",
        "result":     "",
        "exit_price": None,
        "exit_reason":"",
        "opened_at":  time.time(),
        "closed_at":  None,
        "pnl":        0,
        "profit_r":   0,
        "tp1_locked": False,
    }
    trade["db_id"] = db_save_trade(trade)
    open_trades.append(trade)

    size_pips = (rng["size"] * 10000 if pair != "USDJPY"
                 else rng["size"] * 100)

    send_telegram(
        f"🔔 NEW TRADE | {pair} {direction} [{strategy_name}]\n"
        f"Entry: {round(entry,decimals)} | SL: {round(sl,decimals)}\n"
        f"TP1 (1R): {round(tp1,decimals)} | TP2 (2R): {round(tp2,decimals)}\n"
        f"Range: {round(rng['low'],decimals)}→{round(rng['high'],decimals)} "
        f"({round(size_pips,1)}pips)\n"
        f"Lot: {lot_size} | Risk: {RISK_PERCENT}%"
    )
    logger.info("Trade opened: %s %s entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f",
                pair, direction, entry, sl, tp1, tp2)

# ================== Scan Logic ============================================
def scan_orb(live: Dict):
    """Check ORB entry for US500."""
    if has_trade_today("US500"):
        return
    if not orb.is_trading_time():
        return

    rng = ranges.get("US500")
    if not rng:
        return

    if rng.get("disabled"):
        return

    direction = orb.check_entry(live, rng)
    if not direction:
        return

    entry  = round(live["ask"] if direction=="BUY" else live["bid"], 1)
    sl, tp1, tp2 = orb.calculate_levels(direction, entry, rng)
    open_trade("US500", direction, entry, sl, tp1, tp2, rng, "ORB")

def scan_arb(pair: str, live: Dict):
    """Check Asian Range Breakout entry for EURUSD/GBPUSD."""
    if has_trade_today(pair):
        return
    if not arb.is_trading_time():
        return

    rng = ranges.get(pair)
    if not rng:
        return

    if rng.get("disabled"):
        return

    direction = arb.check_entry(live, rng)
    if not direction:
        return

    entry  = round(live["ask"] if direction=="BUY" else live["bid"], 5)
    sl, tp1, tp2 = arb.calculate_levels(direction, entry, rng)
    open_trade(pair, direction, entry, sl, tp1, tp2, rng, "ARB")

def scan_pdb(live: Dict):
    """Check Previous Day Breakout entry for USDJPY."""
    if has_trade_today("USDJPY"):
        return
    if not pdb.is_trading_time():
        return

    rng = ranges.get("USDJPY")
    if not rng:
        return

    if rng.get("disabled"):
        return

    direction = pdb.check_entry(live, rng)
    if not direction:
        return

    entry  = round(live["ask"] if direction=="BUY" else live["bid"], 3)
    sl, tp1, tp2 = pdb.calculate_levels(direction, entry, rng)
    open_trade("USDJPY", direction, entry, sl, tp1, tp2, rng, "PDB")

def build_daily_ranges():
    """Build ranges once the required source session has actually completed."""
    now = now_utc()
    if now.weekday() >= 5:
        return

    c = cur_mins()

    if not ranges_built["US500"] and c >= mins(*ORB_RANGE_END):
        epic = epics.get("US500")
        if epic:
            logger.info("Building US500 ORB range...")
            rng = orb.build_range(epic)
            if rng:
                ranges["US500"] = rng
                ranges_built["US500"] = True

    for pair in ["EURUSD", "GBPUSD"]:
        if not ranges_built[pair] and c >= mins(*ARB_RANGE_END):
            epic = epics.get(pair)
            if epic:
                logger.info("Building %s Asian range...", pair)
                rng = arb.build_range(pair, epic)
                if rng:
                    ranges[pair] = rng
                    ranges_built[pair] = True
                time.sleep(1)

    if not ranges_built["USDJPY"] and c >= mins(*PDB_TRADE_START):
        epic = epics.get("USDJPY")
        if epic:
            logger.info("Building USDJPY previous day range...")
            rng = pdb.build_range(epic)
            if rng:
                ranges["USDJPY"] = rng
                ranges_built["USDJPY"] = True

# ================== Heartbeat =============================================
def send_heartbeat():
    if not broker:
        return
    equity   = broker.get_account_balance()
    decisive = wins + losses
    wr       = round(wins/decisive*100,1) if decisive else 0
    open_c   = sum(1 for t in open_trades if t["status"]=="OPEN")

    range_lines = []
    for pair in PAIRS:
        rng = ranges.get(pair)
        if rng:
            range_lines.append(
                f"  {pair}: {round(rng['low'],PAIRS[pair]['decimals'])}→"
                f"{round(rng['high'],PAIRS[pair]['decimals'])}"
            )
        else:
            range_lines.append(f"  {pair}: range not set")

    send_telegram(
        f"💓 Multi-Strategy Bot\n"
        f"Balance: {equity:.2f} | Open: {open_c}\n"
        f"W:{wins} L:{losses} BE:{breakevens} WR:{wr}%\n"
        f"Daily loss: {get_daily_loss_pct():.2f}% / {DAILY_LOSS_LIMIT_PCT}%\n"
        f"Today's ranges:\n" + "\n".join(range_lines)
    )

# ================== Main ==================================================
def main():
    global broker

    logger.info("=" * 60)
    logger.info("  Multi-Strategy Trading Bot")
    logger.info("  US500=ORB | EURUSD/GBPUSD=ARB | USDJPY=PDB")
    logger.info("=" * 60)

    ensure_csv(RESULTS_FILE, [
        "timestamp","pair","strategy","direction","entry","sl","tp1","tp2",
        "exit_price","status","profit_r","pnl","range_size","exit_reason"
    ])
    init_db()
    load_state()

    broker = CapitalClient(
        api_key=CAPITAL_API_KEY, login=CAPITAL_LOGIN,
        password=CAPITAL_PASSWORD, demo=CAPITAL_DEMO
    )

    # Resolve all epics upfront
    for pair, cfg in PAIRS.items():
        epic = broker.get_epic(cfg["search"])
        if epic:
            epics[pair] = epic
            logger.info("Epic resolved: %s → %s", pair, epic)
        else:
            logger.error("Cannot resolve epic for %s", pair)

    reconcile_state_with_broker()

    start_bal = broker.get_account_balance()
    init_daily_pnl(start_bal if start_bal > 0 else INITIAL_EQUITY)

    send_telegram(
        f"🚀 Multi-Strategy Bot Started | "
        f"{'DEMO' if CAPITAL_DEMO else 'LIVE'}\n\n"
        f"US500  → ORB (13:30-14:00 UTC range)\n"
        f"         Pre-market bias filter: ON\n"
        f"         Trade: 14:00-20:00 UTC\n\n"
        f"EURUSD → Asian Range Breakout\n"
        f"GBPUSD → Asian Range Breakout\n"
        f"         Range: 23:00-07:00 UTC\n"
        f"         Trade: 07:00-12:00 UTC\n\n"
        f"USDJPY → Previous Day High/Low\n"
        f"         Trade: 07:00-20:00 UTC\n\n"
        f"SL System: Auto-moves TP1→TP2\n"
        f"Risk: {RISK_PERCENT}% | Daily limit: {DAILY_LOSS_LIMIT_PCT}%\n"
        f"Balance: {start_bal:.2f}"
    )

    last_heartbeat = 0.0
    last_day       = today_str()

    while True:
        try:
            now = now_utc()

            # ── Daily reset ───────────────────────────────────────────────
            if today_str() != last_day:
                for pair in PAIRS:
                    ranges_built[pair] = False
                ranges.clear()
                last_day = today_str()
                day_start_equity = broker.get_account_balance() if broker else INITIAL_EQUITY
                init_daily_pnl(day_start_equity if day_start_equity > 0 else INITIAL_EQUITY)
                logger.info("New trading day: %s", last_day)

            # ── Build ranges at correct times ─────────────────────────────
            build_daily_ranges()

            # ── Get all live prices ───────────────────────────────────────
            live_prices = {}
            for pair, epic in epics.items():
                live = broker.get_live_price(epic)
                if live:
                    live_prices[pair] = live
                time.sleep(0.2)

            # ── Monitor open trades ───────────────────────────────────────
            if any(t["status"]=="OPEN" for t in open_trades):
                check_trades(live_prices)

            # ── Scan for entries ──────────────────────────────────────────
            # US500 ORB
            if "US500" in live_prices:
                scan_orb(live_prices["US500"])

            # EURUSD / GBPUSD ARB
            for pair in ["EURUSD","GBPUSD"]:
                if pair in live_prices:
                    scan_arb(pair, live_prices[pair])

            # USDJPY PDB
            if "USDJPY" in live_prices:
                scan_pdb(live_prices["USDJPY"])

            # ── Heartbeat ─────────────────────────────────────────────────
            if time.time() - last_heartbeat >= HEARTBEAT_SECS:
                send_heartbeat()
                last_heartbeat = time.time()

            time.sleep(SCAN_INTERVAL)

        except RuntimeError as e:
            logger.critical("Halted: %s", e)
            send_telegram(f"🛑 Bot halted: {e}")
            break
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            send_telegram("🛑 Bot stopped manually")
            break
        except Exception as e:
            logger.error("Main loop error: %s", e)
            time.sleep(30)

if __name__ == "__main__":
    main()
