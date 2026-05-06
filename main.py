import csv
import logging
import math
import os
import pickle
import sqlite3
import time
from datetime import datetime, timedelta, timezone
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


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ================== Configuration ==========================================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY", "").strip()
CAPITAL_LOGIN = os.getenv("CAPITAL_LOGIN", "").strip()
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD", "").strip()
CAPITAL_DEMO = env_bool("CAPITAL_DEMO", True)

INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", "1000.0"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "1.0"))
MAX_ACTIVE_TRADES = int(os.getenv("MAX_ACTIVE_TRADES", "10"))
BREAK_EVEN_TRIGGER_R = float(os.getenv("BREAK_EVEN_TRIGGER_R", "1.0"))
PAIR_COOLDOWN_SECONDS = int(os.getenv("PAIR_COOLDOWN_SECONDS", "14400"))
MAX_AUTH_RETRIES = int(os.getenv("MAX_AUTH_RETRIES", "5"))

MIN_CONFLUENCE_SCORE = int(os.getenv("MIN_CONFLUENCE_SCORE", "3"))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "4.0"))
NEWS_BUFFER_MINUTES = int(os.getenv("NEWS_BUFFER_MINUTES", "30"))
ADX_TREND_THRESHOLD = float(os.getenv("ADX_TREND_THRESHOLD", "20.0"))
MAX_PORTFOLIO_HEAT_PCT = float(os.getenv("MAX_PORTFOLIO_HEAT_PCT", "6.0"))
MIN_TRADES_FOR_DISABLE = int(os.getenv("MIN_TRADES_FOR_DISABLE", "20"))
DISABLE_WIN_RATE_THRESHOLD = float(os.getenv("DISABLE_WIN_RATE_THRESHOLD", "35.0"))
ML_MIN_TRADES_TO_TRAIN = int(os.getenv("ML_MIN_TRADES_TO_TRAIN", "50"))
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.52"))

STALE_TRADE_HOURS = float(os.getenv("STALE_TRADE_HOURS", "4.0"))

MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "2"))
DYNAMIC_RISK_REDUCTION = float(os.getenv("DYNAMIC_RISK_REDUCTION", "0.5"))
TREND_MATURITY_BARS = int(os.getenv("TREND_MATURITY_BARS", "40"))
WEEKLY_BIAS_FILTER = env_bool("WEEKLY_BIAS_FILTER", False)
SR_LOOKBACK = int(os.getenv("SR_LOOKBACK", "100"))
SR_MIN_TOUCHES = int(os.getenv("SR_MIN_TOUCHES", "2"))
SR_ZONE_PCT = float(os.getenv("SR_ZONE_PCT", "0.002"))
REQUIRE_ENGULFING = env_bool("REQUIRE_ENGULFING", False)

SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "900"))
TRADE_CHECK_INTERVAL_SECONDS = int(os.getenv("TRADE_CHECK_INTERVAL_SECONDS", "20"))
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))
REPORT_INTERVAL_SECONDS = int(os.getenv("REPORT_INTERVAL_SECONDS", "3600"))
ML_RETRAIN_INTERVAL_SECONDS = int(os.getenv("ML_RETRAIN_INTERVAL_SECONDS", "86400"))

DB_FILE = "trade_state.db"
RESULTS_FILE = "trade_results.csv"
LOG_FILE = "bot.log"
ML_MODEL_FILE = "ml_model.pkl"


INSTRUMENT_PROFILES = {
    "FOREX": {
        "pairs": {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"},
        "atr_stop": 1.5,
        "atr_tp_full": 2.5,
        "atr_tp_partial": 1.4,
        "trailing_mult": 1.1,
        "spread_ratio": 0.25,
        "rsi_buy_lo": 48,
        "rsi_buy_hi": 75,
        "rsi_sell_lo": 25,
        "rsi_sell_hi": 52,
        "adx_min": 18.0,
        "trend_gap_mult": 0.20,
        "session_start": 6,
        "session_end": 18,
        "decimals": 5,
        "min_dist": 0.00010,
    },
    "INDEX": {
        "pairs": {"US500", "US30", "USTEC"},
        "atr_stop": 2.2,
        "atr_tp_full": 3.5,
        "atr_tp_partial": 2.0,
        "trailing_mult": 1.8,
        "spread_ratio": 0.35,
        "rsi_buy_lo": 50,
        "rsi_buy_hi": 75,
        "rsi_sell_lo": 25,
        "rsi_sell_hi": 50,
        "adx_min": 18.0,
        "trend_gap_mult": 0.20,
        "session_start": 13,
        "session_end": 21,
        "decimals": 1,
        "min_dist": 3.0,
    },
    "COMMODITY": {
        "pairs": {"XAUUSD", "XAGUSD", "USOIL"},
        "atr_stop": 1.8,
        "atr_tp_full": 2.8,
        "atr_tp_partial": 1.6,
        "trailing_mult": 1.4,
        "spread_ratio": 0.30,
        "rsi_buy_lo": 48,
        "rsi_buy_hi": 75,
        "rsi_sell_lo": 25,
        "rsi_sell_hi": 52,
        "adx_min": 20.0,
        "trend_gap_mult": 0.25,
        "session_start": 7,
        "session_end": 18,
        "decimals": 2,
        "min_dist": 0.50,
    },
}


def get_profile(pair: str) -> Dict:
    for profile in INSTRUMENT_PROFILES.values():
        if pair in profile["pairs"]:
            return profile
    return INSTRUMENT_PROFILES["FOREX"]


def get_decimals(pair: str) -> Tuple[int, float]:
    pair_up = pair.upper()
    if "JPY" in pair_up:
        return 3, 0.05
    if pair_up in {"US500", "SPX500"}:
        return 1, 3.0
    if pair_up in {"US30", "WALL"}:
        return 0, 10.0
    if pair_up in {"USTEC", "NAS100"}:
        return 1, 5.0
    if "XAU" in pair_up:
        return 2, 0.50
    if "XAG" in pair_up or "OIL" in pair_up:
        return 2, 0.05
    return 5, 0.00050


pairs = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "US500": "US 500",
}

CORRELATION_GROUPS = [
    {"EURUSD", "GBPUSD"},
    {"USDJPY", "GBPJPY"},
    {"US500", "US30", "USTEC"},
]

MIN_LOT_SIZES = {
    "EURUSD": 1000,
    "GBPUSD": 1000,
    "USDJPY": 1000,
    "GBPJPY": 1000,
    "USDCHF": 1000,
    "AUDUSD": 1000,
    "USDCAD": 1000,
    "NZDUSD": 1000,
    "EURGBP": 1000,
    "EURJPY": 1000,
    "US500": 1,
    "US30": 1,
    "USTEC": 1,
    "XAUUSD": 1,
    "XAGUSD": 1,
    "USOIL": 1,
}

MAX_LOT_SIZES = {
    "EURUSD": 50000,
    "GBPUSD": 50000,
    "USDJPY": 50000,
    "GBPJPY": 50000,
    "USDCHF": 50000,
    "AUDUSD": 50000,
    "USDCAD": 50000,
    "NZDUSD": 50000,
    "EURGBP": 50000,
    "EURJPY": 50000,
    "US500": 100,
    "US30": 10,
    "USTEC": 50,
    "XAUUSD": 10,
    "XAGUSD": 100,
    "USOIL": 100,
}

RESULT_HEADERS = [
    "timestamp",
    "pair",
    "type",
    "entry",
    "sl",
    "tp",
    "exit_price",
    "status",
    "profit_r",
    "pnl",
    "confluence_score",
    "ml_confidence",
    "instrument_class",
    "exit_reason",
    "entry_atr",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_today_str() -> str:
    return now_utc().date().isoformat()


def is_valid(*vals) -> bool:
    return all(v is not None and not math.isnan(v) and math.isfinite(v) for v in vals)


def ensure_csv(path: str, headers: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=10,
        )
    except Exception as exc:
        logger.error("Telegram error: %s", exc)


# ================== Database ===============================================
def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str):
    if column not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            epic TEXT,
            type TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            tp_partial REAL,
            status TEXT DEFAULT 'OPEN',
            result TEXT DEFAULT '',
            opened_at REAL,
            closed_at REAL,
            risk_per_unit REAL,
            break_even_done INTEGER DEFAULT 0,
            partial_done INTEGER DEFAULT 0,
            entry_atr REAL,
            lot_size REAL,
            deal_ref TEXT,
            deal_id TEXT,
            confluence_score INTEGER DEFAULT 0,
            ml_confidence REAL DEFAULT 0.0,
            instrument_class TEXT DEFAULT 'FOREX',
            exit_reason TEXT DEFAULT '',
            exit_price REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_pnl (
            trade_date TEXT PRIMARY KEY,
            start_equity REAL,
            realized_pnl REAL DEFAULT 0.0,
            trade_count INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pair_stats (
            pair TEXT PRIMARY KEY,
            total_trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            breakevens INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0.0,
            disabled INTEGER DEFAULT 0,
            disabled_reason TEXT DEFAULT ''
        )
        """
    )

    _ensure_column(conn, "trades", "result", "TEXT DEFAULT ''")
    _ensure_column(conn, "trades", "closed_at", "REAL")
    _ensure_column(conn, "trades", "deal_id", "TEXT")
    _ensure_column(conn, "trades", "exit_reason", "TEXT DEFAULT ''")
    _ensure_column(conn, "trades", "exit_price", "REAL")
    _ensure_column(conn, "pair_stats", "breakevens", "INTEGER DEFAULT 0")

    conn.commit()
    conn.close()


def db_save_trade(trade: Dict) -> int:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO trades (
            pair, epic, type, entry, sl, tp, tp_partial, status, result, opened_at,
            closed_at, risk_per_unit, break_even_done, partial_done, entry_atr,
            lot_size, deal_ref, deal_id, confluence_score, ml_confidence,
            instrument_class, exit_reason, exit_price
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            trade["pair"],
            trade["epic"],
            trade["type"],
            trade["entry"],
            trade["sl"],
            trade["tp"],
            trade["tp_partial"],
            trade["status"],
            trade.get("result", ""),
            trade["opened_at"],
            trade.get("closed_at"),
            trade["risk_per_unit"],
            int(trade["break_even_done"]),
            int(trade["partial_done"]),
            trade["entry_atr"],
            trade["lot_size"],
            trade.get("deal_ref", ""),
            trade.get("deal_id", ""),
            trade.get("confluence_score", 0),
            trade.get("ml_confidence", 0.0),
            trade.get("instrument_class", "FOREX"),
            trade.get("exit_reason", ""),
            trade.get("exit_price"),
        ),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def db_update_trade(trade: Dict):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """
        UPDATE trades
        SET sl=?, tp=?, tp_partial=?, status=?, result=?, break_even_done=?,
            partial_done=?, deal_ref=?, deal_id=?, exit_reason=?, exit_price=?, closed_at=?
        WHERE id=?
        """,
        (
            trade["sl"],
            trade["tp"],
            trade["tp_partial"],
            trade["status"],
            trade.get("result", ""),
            int(trade["break_even_done"]),
            int(trade["partial_done"]),
            trade.get("deal_ref", ""),
            trade.get("deal_id", ""),
            trade.get("exit_reason", ""),
            trade.get("exit_price"),
            trade.get("closed_at"),
            trade["db_id"],
        ),
    )
    conn.commit()
    conn.close()


def db_load_open_trades() -> List[Dict]:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM trades WHERE status='OPEN'")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    conn.close()
    loaded = []
    for row in rows:
        trade = dict(zip(cols, row))
        trade["break_even_done"] = bool(trade["break_even_done"])
        trade["partial_done"] = bool(trade["partial_done"])
        loaded.append(trade)
    return loaded


def db_record_pair_result(pair: str, result: str, pnl: float):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO pair_stats (pair) VALUES (?)", (pair,))
    if result == "WIN":
        cur.execute(
            """
            UPDATE pair_stats
            SET total_trades=total_trades+1, wins=wins+1, total_pnl=total_pnl+?
            WHERE pair=?
            """,
            (pnl, pair),
        )
    elif result == "LOSS":
        cur.execute(
            """
            UPDATE pair_stats
            SET total_trades=total_trades+1, losses=losses+1, total_pnl=total_pnl+?
            WHERE pair=?
            """,
            (pnl, pair),
        )
    else:
        cur.execute(
            """
            UPDATE pair_stats
            SET total_trades=total_trades+1, breakevens=breakevens+1, total_pnl=total_pnl+?
            WHERE pair=?
            """,
            (pnl, pair),
        )
    conn.commit()
    conn.close()


def db_get_pair_stats(pair: str) -> Dict:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM pair_stats WHERE pair=?", (pair,))
    row = cur.fetchone()
    cols = [d[0] for d in cur.description] if cur.description else []
    conn.close()
    if not row:
        return {
            "pair": pair,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "breakevens": 0,
            "total_pnl": 0.0,
            "disabled": 0,
            "disabled_reason": "",
        }
    return dict(zip(cols, row))


def db_disable_pair(pair: str, reason: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO pair_stats (pair) VALUES (?)", (pair,))
    cur.execute("UPDATE pair_stats SET disabled=1, disabled_reason=? WHERE pair=?", (reason, pair))
    conn.commit()
    conn.close()


def is_pair_disabled(pair: str) -> bool:
    return bool(db_get_pair_stats(pair).get("disabled", 0))


def init_daily_pnl(start_equity: float):
    today = get_today_str()
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT trade_date FROM daily_pnl WHERE trade_date=?", (today,))
    if not cur.fetchone():
        cur.execute("INSERT INTO daily_pnl VALUES (?,?,0.0,0)", (today, start_equity))
        conn.commit()
    conn.close()


def record_daily_pnl(pnl: float):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "UPDATE daily_pnl SET realized_pnl=realized_pnl+?, trade_count=trade_count+1 WHERE trade_date=?",
        (pnl, get_today_str()),
    )
    conn.commit()
    conn.close()


def get_daily_loss_pct() -> float:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT start_equity, realized_pnl FROM daily_pnl WHERE trade_date=?", (get_today_str(),))
    row = cur.fetchone()
    conn.close()
    if not row or row[0] <= 0:
        return 0.0
    return max((-row[1] / row[0]) * 100, 0.0)


def daily_loss_limit_hit() -> bool:
    loss = get_daily_loss_pct()
    if loss >= DAILY_LOSS_LIMIT_PCT:
        logger.warning("Daily loss limit hit: %.2f%%", loss)
        return True
    return False


# ================== Capital.com API =======================================
class CapitalClient:
    def __init__(self, api_key: str, login: str, password: str, demo: bool = True):
        self.api_key = api_key
        self.login = login
        self.password = password
        self.base_url = "https://demo-api-capital.backend-capital.com" if demo else "https://api-capital.backend-capital.com"
        self.cst = None
        self.security_token = None
        self.session = requests.Session()
        self.epic_cache: Dict[str, str] = {}
        self._auth_retries = 0
        self.authenticate()

    def authenticate(self):
        if self._auth_retries >= MAX_AUTH_RETRIES:
            raise RuntimeError(f"Auth failed after {MAX_AUTH_RETRIES} retries.")
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/session",
                headers={"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"identifier": self.login, "password": self.password, "encryptedPassword": False},
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(response.json().get("errorMessage", response.text[:200]))
            self.cst = response.headers.get("CST")
            self.security_token = response.headers.get("X-SECURITY-TOKEN")
            self._auth_retries = 0
            logger.info("Connected to Capital.com (%s mode)", "demo" if CAPITAL_DEMO else "live")
        except Exception as exc:
            self._auth_retries += 1
            logger.error("Auth attempt %s/%s failed: %s", self._auth_retries, MAX_AUTH_RETRIES, exc)
            if self._auth_retries < MAX_AUTH_RETRIES:
                time.sleep(5 * self._auth_retries)
                self.authenticate()
            else:
                raise

    def _req(self, method: str, endpoint: str, data: Dict = None) -> Optional[Dict]:
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {
                "X-CAP-API-KEY": self.api_key,
                "CST": self.cst,
                "X-SECURITY-TOKEN": self.security_token,
                "Content-Type": "application/json",
            }
            fn = {
                "GET": self.session.get,
                "POST": self.session.post,
                "PUT": self.session.put,
                "DELETE": self.session.delete,
            }[method]
            kwargs = {"headers": headers, "timeout": 30}
            if method in {"POST", "PUT"}:
                kwargs["json"] = data
            response = fn(url, **kwargs)
            if response.status_code in {401, 403}:
                self.authenticate()
                return self._req(method, endpoint, data)
            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 10))
                logger.warning("Rate limited, waiting %ss", wait)
                time.sleep(wait)
                return self._req(method, endpoint, data)
            if response.status_code != 200:
                logger.error("API %s %s failed: %s", method, endpoint, response.text[:300])
                return None
            return response.json()
        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("Request error on %s %s: %s", method, endpoint, exc)
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

    def get_candles(self, epic: str, resolution: str = "MINUTE_15", num_candles: int = 300) -> Optional[pd.DataFrame]:
        data = self._req("GET", f"/api/v1/prices/{epic}?resolution={resolution}&max={num_candles}")
        if not data or "prices" not in data:
            return None
        rows = []
        for candle in data["prices"]:
            try:
                rows.append(
                    {
                        "time": candle["snapshotTime"],
                        "Open": float(candle["openPrice"]["bid"]),
                        "High": float(candle["highPrice"]["bid"]),
                        "Low": float(candle["lowPrice"]["bid"]),
                        "Close": float(candle["closePrice"]["bid"]),
                        "Volume": float(candle.get("lastTradedVolume", 0)),
                    }
                )
            except Exception:
                continue
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
        return {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0,
            "spread": round(ask - bid, 6),
            "tradeable": True,
        }

    def get_open_positions(self) -> List[Dict]:
        data = self._req("GET", "/api/v1/positions")
        if not data:
            return []
        positions = []
        for item in data.get("positions", []):
            pos = item.get("position", {})
            market = item.get("market", {})
            positions.append(
                {
                    "dealId": pos.get("dealId", ""),
                    "dealReference": pos.get("dealReference", ""),
                    "epic": market.get("epic", pos.get("epic", "")),
                    "direction": pos.get("direction", ""),
                    "size": float(pos.get("size", 0) or 0),
                    "level": float(pos.get("level", 0) or 0),
                    "stopLevel": pos.get("stopLevel"),
                    "profitLevel": pos.get("limitLevel") or pos.get("profitLevel"),
                }
            )
        return positions

    def _resolve_open_position(self, deal_identifier: str) -> Optional[Dict]:
        for pos in self.get_open_positions():
            if pos["dealId"] == deal_identifier or pos["dealReference"] == deal_identifier:
                return pos
        return None

    def place_order(self, pair: str, epic: str, direction: str, units: float, entry: float, sl: float, tp: float) -> Optional[str]:
        decimals, min_dist = get_decimals(pair)
        sl_r = round(sl, decimals)
        tp_r = round(tp, decimals)
        entry_r = round(entry, decimals)

        sl_dist = abs(entry_r - sl_r)
        tp_dist = abs(tp_r - entry_r)

        if sl_dist < min_dist:
            sl_r = round(entry_r - min_dist if direction == "BUY" else entry_r + min_dist, decimals)
            sl_dist = min_dist

        if tp_dist < min_dist:
            tp_r = round(entry_r + min_dist if direction == "BUY" else entry_r - min_dist, decimals)
            tp_dist = min_dist

        min_tp_dist = max(sl_dist * 1.5, min_dist * 10)
        if abs(tp_r - entry_r) < min_tp_dist:
            tp_r = round(entry_r + min_tp_dist if direction == "BUY" else entry_r - min_tp_dist, decimals)

        payload = {
            "epic": epic,
            "direction": direction,
            "size": float(units),
            "stopLevel": float(sl_r),
            "guaranteedStop": False,
            "forceOpen": True,
        }
        logger.info(
            "Placing %s %s %s | entry=%s sl=%s tp=%s size=%s",
            direction,
            pair,
            epic,
            entry_r,
            sl_r,
            tp_r,
            units,
        )
        result = self._req("POST", "/api/v1/positions", payload)
        if result and result.get("dealReference"):
            return result["dealReference"]
        return None

    def confirm_fill(self, deal_ref: str) -> Optional[Dict]:
        time.sleep(2)
        data = self._req("GET", f"/api/v1/confirms/{deal_ref}")
        if not data:
            return None
        opened_deal_id = None
        for item in data.get("affectedDeals", []):
            if item.get("status") == "OPENED":
                opened_deal_id = item.get("dealId")
                break
        if opened_deal_id:
            data["opened_deal_id"] = opened_deal_id
        return data

    def get_account_balance(self) -> float:
        data = self._req("GET", "/api/v1/accounts")
        if data and data.get("accounts"):
            return float(data["accounts"][0]["balance"]["balance"])
        return 0.0

    def update_sl(self, deal_identifier: str, new_sl: float) -> bool:
        pos = self._resolve_open_position(deal_identifier)
        if not pos or not pos.get("dealId"):
            logger.warning("Could not resolve position for SL update: %s", deal_identifier)
            return False
        payload = {"stopLevel": float(new_sl)}
        result = self._req("PUT", f"/api/v1/positions/{pos['dealId']}", payload)
        return bool(result)

    def close_position(self, deal_identifier: str) -> bool:
        pos = self._resolve_open_position(deal_identifier)
        if not pos:
            logger.warning("Position %s not found on broker; treating as already closed", deal_identifier)
            return True
        result = self._req("DELETE", f"/api/v1/positions/{pos['dealId']}")
        return result is not None


# ================== Indicators ============================================
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = (high - low).combine((high - prev_close).abs(), max).combine((low - prev_close).abs(), max)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    tr = (high - low).combine((high - prev_close).abs(), max).combine((low - prev_close).abs(), max)
    safe_tr = tr.ewm(span=period, adjust=False).mean().replace(0, 1e-10)
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / safe_tr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / safe_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    return dx.ewm(span=period, adjust=False).mean(), plus_di, minus_di


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (
        curr["Close"] > curr["Open"]
        and prev["Close"] < prev["Open"]
        and curr["Open"] <= prev["Close"]
        and curr["Close"] >= prev["Open"]
    )


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (
        curr["Close"] < curr["Open"]
        and prev["Close"] > prev["Open"]
        and curr["Open"] >= prev["Close"]
        and curr["Close"] <= prev["Open"]
    )


def has_confirmation_candle(df: pd.DataFrame, direction: str) -> bool:
    if not REQUIRE_ENGULFING:
        return True
    if len(df) < 4:
        return False
    for start in range(len(df) - 4, len(df) - 1):
        window = df.iloc[start : start + 2]
        if direction == "BUY" and is_bullish_engulfing(window):
            return True
        if direction == "SELL" and is_bearish_engulfing(window):
            return True
    return False


def get_weekly_bias(epic: str) -> Optional[str]:
    if not WEEKLY_BIAS_FILTER:
        return None
    try:
        data = broker.get_candles(epic, "WEEK", 30)
        if data is None or len(data) < 10:
            return None
        close = data["Close"]
        ema20 = close.ewm(span=20, adjust=False).mean()
        return "BUY" if float(close.iloc[-1]) > float(ema20.iloc[-1]) else "SELL"
    except Exception as exc:
        logger.warning("Weekly bias check failed for %s: %s", epic, exc)
        return None


def is_trend_mature(ema20: pd.Series, ema50: pd.Series, lookback: int = 20) -> bool:
    if len(ema20) < lookback or len(ema50) < lookback:
        return False
    gaps = (ema20.iloc[-lookback:] - ema50.iloc[-lookback:]).abs()
    widening = sum(1 for idx in range(1, len(gaps)) if gaps.iloc[idx] > gaps.iloc[idx - 1])
    mature = widening > lookback * 0.85
    if mature:
        logger.info("Trend mature: EMA gap widened on %s/%s bars", widening, lookback)
    return mature


def find_sr_levels(df: pd.DataFrame, lookback: int = 100, min_touches: int = 2, zone_pct: float = 0.002) -> List[float]:
    if len(df) < lookback:
        lookback = len(df)
    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values

    candidates = []
    for idx in range(2, len(recent) - 2):
        if highs[idx] > highs[idx - 1] and highs[idx] > highs[idx - 2] and highs[idx] > highs[idx + 1] and highs[idx] > highs[idx + 2]:
            candidates.append(highs[idx])
        if lows[idx] < lows[idx - 1] and lows[idx] < lows[idx - 2] and lows[idx] < lows[idx + 1] and lows[idx] < lows[idx + 2]:
            candidates.append(lows[idx])

    if not candidates:
        return []

    candidates.sort()
    used = [False] * len(candidates)
    clusters = []
    for idx, level in enumerate(candidates):
        if used[idx]:
            continue
        cluster = [level]
        zone = level * zone_pct
        used[idx] = True
        for jdx in range(idx + 1, len(candidates)):
            if not used[jdx] and abs(candidates[jdx] - level) <= zone:
                cluster.append(candidates[jdx])
                used[jdx] = True
        clusters.append(sum(cluster) / len(cluster))

    strong_levels = []
    for level in clusters:
        zone = level * zone_pct
        touches = sum(1 for high, low in zip(highs, lows) if low <= level + zone and high >= level - zone)
        if touches >= min_touches:
            strong_levels.append(round(level, 5))
    return sorted(strong_levels)


def adjust_tp_to_sr(tp: float, tp_partial: float, entry: float, direction: str, sr_levels: List[float], atr_val: float) -> float:
    if not sr_levels:
        return tp

    buffer = atr_val * 0.3
    min_dist = atr_val * 2.0

    if direction == "BUY":
        walls = [lvl for lvl in sr_levels if tp_partial < lvl < tp]
        if walls:
            nearest = min(walls)
            adjusted = nearest - buffer
            if adjusted > tp_partial and (adjusted - entry) >= min_dist:
                return round(adjusted, 5)
    else:
        walls = [lvl for lvl in sr_levels if tp < lvl < tp_partial]
        if walls:
            nearest = max(walls)
            adjusted = nearest + buffer
            if adjusted < tp_partial and (entry - adjusted) >= min_dist:
                return round(adjusted, 5)
    return tp


def in_optimal_session(pair: str) -> bool:
    now = now_utc()
    if now.weekday() >= 5:
        return False
    profile = get_profile(pair)
    return profile["session_start"] <= now.hour < profile["session_end"]


# ================== News Filter ===========================================
_news_cache: List[Dict] = []
_news_last_fetch = 0.0


def fetch_forex_news() -> List[Dict]:
    global _news_cache, _news_last_fetch
    if time.time() - _news_last_fetch < 3600 and _news_cache:
        return _news_cache
    try:
        response = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        events = []
        for item in response.json():
            if item.get("impact", "").lower() not in {"high", "medium"}:
                continue
            try:
                event_time = datetime.fromisoformat(item["date"].replace("Z", "+00:00"))
                events.append({"time": event_time, "title": item.get("title", ""), "impact": item.get("impact", "")})
            except Exception:
                continue
        _news_cache = events
        _news_last_fetch = time.time()
        return events
    except Exception as exc:
        logger.warning("News fetch failed: %s", exc)
        return _news_cache


def is_near_news() -> bool:
    now = now_utc()
    for event in fetch_forex_news():
        if abs((event["time"] - now).total_seconds()) <= NEWS_BUFFER_MINUTES * 60:
            logger.info("News blackout: %s (%s)", event["title"], event["impact"])
            return True
    return False


# ================== ML Filter =============================================
class MLSignalFilter:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.trained = False
        self._load_model()

    def _load_model(self):
        if not os.path.exists(ML_MODEL_FILE):
            return
        try:
            with open(ML_MODEL_FILE, "rb") as handle:
                saved = pickle.load(handle)
            self.model = saved["model"]
            self.scaler = saved["scaler"]
            self.trained = True
            logger.info("ML model loaded")
        except Exception as exc:
            logger.warning("Could not load ML model: %s", exc)

    def train(self):
        if not ML_AVAILABLE or not os.path.exists(RESULTS_FILE):
            return
        try:
            df = pd.read_csv(RESULTS_FILE)
        except Exception as exc:
            logger.warning("Could not read results file for ML training: %s", exc)
            return
        if df.empty:
            return
        df = df[df["status"].isin(["WIN", "LOSS"])].copy()
        if len(df) < ML_MIN_TRADES_TO_TRAIN:
            logger.info("ML needs %s decisive trades, has %s", ML_MIN_TRADES_TO_TRAIN, len(df))
            return

        df["direction_buy"] = (df["type"] == "BUY").astype(int)
        df["hour_utc"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.hour.fillna(0).astype(int)
        df["entry"] = pd.to_numeric(df["entry"], errors="coerce").fillna(0.0)
        df["entry_atr"] = pd.to_numeric(df["entry_atr"], errors="coerce").fillna(0.0)
        df["confluence_score"] = pd.to_numeric(df["confluence_score"], errors="coerce").fillna(0.0)
        df["atr_norm"] = df["entry_atr"] / df["entry"].replace(0, 1e-10)
        df["label"] = (df["status"] == "WIN").astype(int)

        feature_cols = ["confluence_score", "direction_buy", "hour_utc", "atr_norm"]
        X = df[feature_cols].values
        y = df["label"].values
        if len(set(y)) < 2:
            logger.info("ML skipped: only one class present in decisive trade history")
            return

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        self.model.fit(X_scaled, y)
        self.trained = True
        with open(ML_MODEL_FILE, "wb") as handle:
            pickle.dump({"model": self.model, "scaler": self.scaler}, handle)
        score = self.model.score(X_scaled, y)
        logger.info("ML retrained on %s decisive trades | in-sample acc=%.2f%%", len(df), score * 100)
        send_telegram(f"ML retrained on {len(df)} decisive trades | in-sample acc {score:.1%}")

    def predict(self, score: int, direction: str, hour: int, atr_norm: float) -> float:
        if not self.trained or self.model is None or self.scaler is None:
            return 1.0
        try:
            X = np.array([[score, 1 if direction == "BUY" else 0, hour, atr_norm]])
            return float(self.model.predict_proba(self.scaler.transform(X))[0][1])
        except Exception:
            return 1.0


class CorrelationFilter:
    def is_blocked(self, pair: str, direction: str, open_trades: List[Dict]) -> Tuple[bool, str]:
        open_map = {trade["pair"]: trade["type"] for trade in open_trades if trade["status"] == "OPEN"}
        for group in CORRELATION_GROUPS:
            if pair not in group:
                continue
            for open_pair, open_dir in open_map.items():
                if open_pair != pair and open_pair in group and open_dir == direction:
                    return True, f"{pair} correlated with {open_pair} ({open_dir})"
        return False, ""


class PortfolioHeatMonitor:
    def get_heat_pct(self, open_trades: List[Dict], equity: float) -> float:
        if equity <= 0:
            return 0.0
        total_risk = sum(trade["risk_per_unit"] * trade["lot_size"] for trade in open_trades if trade["status"] == "OPEN")
        return (total_risk / equity) * 100

    def is_overheated(self, open_trades: List[Dict], equity: float) -> Tuple[bool, float]:
        heat = self.get_heat_pct(open_trades, equity)
        if heat >= MAX_PORTFOLIO_HEAT_PCT:
            logger.warning("Portfolio heat %.2f%% >= %.2f%%", heat, MAX_PORTFOLIO_HEAT_PCT)
            return True, heat
        return False, heat


class PairPerformanceManager:
    def evaluate_pair(self, pair: str):
        stats = db_get_pair_stats(pair)
        decisive_trades = int(stats["wins"]) + int(stats["losses"])
        if decisive_trades < MIN_TRADES_FOR_DISABLE:
            return
        win_rate = (stats["wins"] / decisive_trades) * 100 if decisive_trades else 0
        if win_rate < DISABLE_WIN_RATE_THRESHOLD and not stats["disabled"]:
            reason = f"WR {win_rate:.1f}% < {DISABLE_WIN_RATE_THRESHOLD}% over {decisive_trades} decisive trades"
            db_disable_pair(pair, reason)
            logger.warning("Auto-disabled %s: %s", pair, reason)
            send_telegram(f"Auto-disabled {pair}: {reason}")

    def get_summary(self) -> str:
        lines = []
        for pair in pairs:
            stats = db_get_pair_stats(pair)
            decisive_trades = int(stats["wins"]) + int(stats["losses"])
            if stats["total_trades"] == 0:
                continue
            win_rate = round(stats["wins"] / decisive_trades * 100, 1) if decisive_trades else 0.0
            flag = " DISABLED" if stats["disabled"] else ""
            lines.append(
                f"{pair}: total={stats['total_trades']} decisive={decisive_trades} WR={win_rate}% "
                f"BE={stats.get('breakevens', 0)} PnL={stats['total_pnl']:.2f}{flag}"
            )
        return "\n".join(lines) if lines else "No pair data yet."


# ================== Global State ==========================================
trades: List[Dict] = []
wins = 0
losses = 0
breakevens = 0
consecutive_losses = 0
last_trade_times: Dict[str, float] = {}
broker: Optional[CapitalClient] = None
ml_filter = MLSignalFilter()
corr_filter = CorrelationFilter()
heat_monitor = PortfolioHeatMonitor()
pair_perf = PairPerformanceManager()


def setup_files():
    init_db()
    ensure_csv(RESULTS_FILE, RESULT_HEADERS)


def validate_runtime_config():
    missing = [name for name, value in {
        "CAPITAL_API_KEY": CAPITAL_API_KEY,
        "CAPITAL_LOGIN": CAPITAL_LOGIN,
        "CAPITAL_PASSWORD": CAPITAL_PASSWORD,
    }.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if not TOKEN or not CHAT_ID:
        logger.warning("Telegram env vars are missing; live alerts are disabled")


def compute_trade_pnl(trade: Dict, exit_price: float) -> Tuple[str, float, float]:
    risk = float(trade["risk_per_unit"])
    if risk <= 0:
        return "BREAKEVEN", 0.0, 0.0
    if trade["type"] == "BUY":
        pnl = (exit_price - trade["entry"]) * trade["lot_size"]
        profit_r = (exit_price - trade["entry"]) / risk
    else:
        pnl = (trade["entry"] - exit_price) * trade["lot_size"]
        profit_r = (trade["entry"] - exit_price) / risk

    if profit_r > 0.10:
        result = "WIN"
    elif profit_r < -0.10:
        result = "LOSS"
    else:
        result = "BREAKEVEN"
    return result, pnl, profit_r


def load_state():
    global wins, losses, breakevens, consecutive_losses
    trades.clear()
    wins = losses = breakevens = consecutive_losses = 0
    last_trade_times.clear()

    for trade in db_load_open_trades():
        trades.append(trade)
        last_trade_times[trade["pair"]] = trade["opened_at"]

    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                status = row.get("status", "")
                if status == "WIN":
                    wins += 1
                    results.append(status)
                elif status == "LOSS":
                    losses += 1
                    results.append(status)
                elif status == "BREAKEVEN":
                    breakevens += 1
                    results.append(status)

    for status in reversed(results):
        if status == "LOSS":
            consecutive_losses += 1
        elif status == "WIN":
            break


def save_trade_result(trade: Dict, result: str, exit_price: float, exit_reason: str):
    global wins, losses, breakevens, consecutive_losses

    result, pnl, profit_r = compute_trade_pnl(trade, exit_price)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                now_utc().isoformat(),
                trade["pair"],
                trade["type"],
                round(trade["entry"], 5),
                round(trade["sl"], 5),
                round(trade["tp"], 5),
                round(exit_price, 5),
                result,
                round(profit_r, 2),
                round(pnl, 2),
                trade.get("confluence_score", 0),
                round(trade.get("ml_confidence", 0), 4),
                trade.get("instrument_class", "FOREX"),
                exit_reason,
                trade.get("entry_atr", 0),
            ]
        )

    record_daily_pnl(pnl)
    db_record_pair_result(trade["pair"], result, pnl)
    pair_perf.evaluate_pair(trade["pair"])

    if result == "WIN":
        wins += 1
        consecutive_losses = 0
    elif result == "LOSS":
        losses += 1
        consecutive_losses += 1
    else:
        breakevens += 1

    emoji = {"WIN": "WIN", "LOSS": "LOSS", "BREAKEVEN": "BE"}[result]
    send_telegram(
        f"{emoji} | {trade['pair']} {trade['type']}\n"
        f"Exit: {round(exit_price, 5)} | PnL: {round(pnl, 2)} | {round(profit_r, 2)}R\n"
        f"Reason: {exit_reason} | Score:{trade.get('confluence_score', '?')}/6"
    )


def active_trade_count() -> int:
    return sum(1 for trade in trades if trade["status"] == "OPEN")


def has_open_trade(pair: str) -> bool:
    return any(trade["pair"] == pair and trade["status"] == "OPEN" for trade in trades)


def cooldown_ready(pair: str) -> bool:
    last = last_trade_times.get(pair)
    return last is None or (time.time() - last) >= PAIR_COOLDOWN_SECONDS


def calculate_position_size(pair: str, entry: float, sl: float) -> float:
    equity = broker.get_account_balance() if broker else INITIAL_EQUITY
    if equity <= 0:
        equity = INITIAL_EQUITY

    effective_risk = RISK_PERCENT
    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        effective_risk *= DYNAMIC_RISK_REDUCTION
        logger.info("Dynamic risk activated: %.2f%% after %s consecutive losses", effective_risk, consecutive_losses)

    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0.0
    size = round((equity * (effective_risk / 100)) / sl_dist, 2)
    size = max(size, MIN_LOT_SIZES.get(pair, 1000))
    size = min(size, MAX_LOT_SIZES.get(pair, 50000))
    return size


def _update_sl_on_broker(trade: Dict, new_sl: float):
    try:
        deal_identifier = trade.get("deal_id") or trade.get("deal_ref", "")
        if deal_identifier and broker:
            if broker.update_sl(deal_identifier, new_sl):
                logger.info("Broker SL updated for %s -> %s", trade["pair"], new_sl)
            else:
                logger.warning("Broker SL update failed for %s", trade["pair"])
    except Exception as exc:
        logger.error("SL update error for %s: %s", trade["pair"], exc)


def finalize_trade(trade: Dict, exit_price: float, reason: str, broker_close: bool = False):
    if trade["status"] != "OPEN":
        return

    if broker_close and broker:
        deal_identifier = trade.get("deal_id") or trade.get("deal_ref", "")
        if deal_identifier:
            closed = broker.close_position(deal_identifier)
            if not closed:
                logger.warning("Broker close failed for %s, keeping local trade OPEN", trade["pair"])
                return

    result, _, _ = compute_trade_pnl(trade, exit_price)
    trade["status"] = "CLOSED"
    trade["result"] = result
    trade["exit_reason"] = reason
    trade["exit_price"] = exit_price
    trade["closed_at"] = time.time()
    db_update_trade(trade)
    save_trade_result(trade, result, exit_price, reason)
    logger.info("Closed %s @ %s -> %s (%s)", trade["pair"], exit_price, result, reason)


def reconcile_open_trades():
    if not broker:
        return
    broker_positions = broker.get_open_positions()
    by_ref = {pos["dealReference"]: pos for pos in broker_positions if pos.get("dealReference")}
    by_id = {pos["dealId"]: pos for pos in broker_positions if pos.get("dealId")}

    synced = []
    for trade in trades:
        pos = None
        if trade.get("deal_id"):
            pos = by_id.get(trade["deal_id"])
        if not pos and trade.get("deal_ref"):
            pos = by_ref.get(trade["deal_ref"])

        if not pos:
            trade["status"] = "SYNC_MISSING"
            trade["result"] = ""
            trade["exit_reason"] = "BROKER_POSITION_NOT_FOUND"
            trade["closed_at"] = time.time()
            db_update_trade(trade)
            logger.warning("Local open trade missing on broker: %s %s", trade["pair"], trade.get("deal_ref", ""))
            send_telegram(f"Sync alert: local trade missing on broker for {trade['pair']} ({trade.get('deal_ref', '')})")
            continue

        if not trade.get("deal_id") and pos.get("dealId"):
            trade["deal_id"] = pos["dealId"]
            db_update_trade(trade)
        synced.append(trade)

    trades[:] = synced

    local_keys = {
        trade.get("deal_id") or trade.get("deal_ref")
        for trade in trades
        if trade.get("deal_id") or trade.get("deal_ref")
    }
    orphaned = [pos for pos in broker_positions if pos.get("dealId") not in local_keys and pos.get("dealReference") not in local_keys]
    if orphaned:
        logger.warning("Broker has %s open positions not tracked locally", len(orphaned))
        send_telegram(f"Sync alert: {len(orphaned)} broker open position(s) are not tracked locally. Check Capital.com manually.")


def build_signal(name: str, epic: str) -> Optional[Dict]:
    profile = get_profile(name)

    d15 = broker.get_candles(epic, "MINUTE_15", 300)
    time.sleep(0.4)
    d1h = broker.get_candles(epic, "HOUR", 300)
    time.sleep(0.4)
    d4h = broker.get_candles(epic, "HOUR_4", 200)
    time.sleep(0.4)
    live = broker.get_live_price(epic)
    time.sleep(0.2)

    if any(item is None for item in [d15, d1h, d4h, live]):
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

    rsi_series = calc_rsi(c15)
    atr_series = calc_atr(d15)
    adx_series, plus_di_series, minus_di_series = calc_adx(d15)
    vol_ma = d15["Volume"].rolling(20).mean()

    lp = float(c15.iloc[-1])
    prev_p = float(c15.iloc[-2])
    e20v = float(e20_15.iloc[-1])
    pe20 = float(e20_15.iloc[-2])
    e50v = float(e50_15.iloc[-1])
    e201h = float(e20_1h.iloc[-1])
    e501h = float(e50_1h.iloc[-1])
    e204h = float(e20_4h.iloc[-1])
    e504h = float(e50_4h.iloc[-1])
    rsi_v = float(rsi_series.iloc[-1])
    atr_v = float(atr_series.iloc[-1])
    adx_v = float(adx_series.iloc[-1])
    plus_di = float(plus_di_series.iloc[-1])
    minus_di = float(minus_di_series.iloc[-1])
    vol_l = float(d15["Volume"].iloc[-1])
    vol_m = float(vol_ma.iloc[-1]) if not math.isnan(vol_ma.iloc[-1]) else 0.0
    high = float(d15["High"].iloc[-1])
    prev_high = float(d15["High"].iloc[-2])
    low = float(d15["Low"].iloc[-1])
    prev_low = float(d15["Low"].iloc[-2])
    c1h_last = float(c1h.iloc[-1])
    c4h_last = float(c4h.iloc[-1])

    if not is_valid(lp, prev_p, e20v, pe20, e50v, e201h, e501h, e204h, e504h, rsi_v, atr_v, adx_v):
        return None
    if atr_v <= 0:
        return None

    if live["spread"] > atr_v * profile["spread_ratio"]:
        return None
    if abs(e20v - e50v) < atr_v * profile["trend_gap_mult"]:
        return None

    signal_type = None
    if (
        lp > e20v > e50v
        and c1h_last > e201h > e501h
        and profile["rsi_buy_lo"] <= rsi_v <= profile["rsi_buy_hi"]
        and prev_p <= pe20 * 1.0015
        and high > prev_high
    ):
        signal_type = "BUY"
    elif (
        lp < e20v < e50v
        and c1h_last < e201h < e501h
        and profile["rsi_sell_lo"] <= rsi_v <= profile["rsi_sell_hi"]
        and prev_p >= pe20 * 0.9985
        and low < prev_low
    ):
        signal_type = "SELL"

    if not signal_type:
        return None
    if adx_v < max(ADX_TREND_THRESHOLD, profile["adx_min"]):
        return None
    if signal_type == "BUY" and plus_di <= minus_di:
        return None
    if signal_type == "SELL" and minus_di <= plus_di:
        return None

    weekly_bias = get_weekly_bias(epic)
    if weekly_bias and weekly_bias != signal_type:
        return None

    if is_trend_mature(e20_15, e50_15, lookback=TREND_MATURITY_BARS):
        return None

    if not has_confirmation_candle(d15, signal_type):
        return None

    score = 0
    if signal_type == "BUY":
        if lp > e20v > e50v:
            score += 1
        if c1h_last > e201h > e501h:
            score += 1
        if c4h_last > e204h > e504h:
            score += 1
        if profile["rsi_buy_lo"] <= rsi_v <= profile["rsi_buy_hi"]:
            score += 1
        if adx_v >= profile["adx_min"]:
            score += 1
        if vol_m > 0 and vol_l > vol_m:
            score += 1
    else:
        if lp < e20v < e50v:
            score += 1
        if c1h_last < e201h < e501h:
            score += 1
        if c4h_last < e204h < e504h:
            score += 1
        if profile["rsi_sell_lo"] <= rsi_v <= profile["rsi_sell_hi"]:
            score += 1
        if adx_v >= profile["adx_min"]:
            score += 1
        if vol_m > 0 and vol_l > vol_m:
            score += 1

    if score < MIN_CONFLUENCE_SCORE:
        return None

    atr_norm = atr_v / lp if lp > 0 else 0
    ml_confidence = ml_filter.predict(score, signal_type, now_utc().hour, atr_norm)
    if ml_filter.trained and ml_confidence < ML_CONFIDENCE_THRESHOLD:
        return None

    entry = round(live["ask"] if signal_type == "BUY" else live["bid"], profile["decimals"])
    stop_dist = round(atr_v * profile["atr_stop"], profile["decimals"])
    sl = round(entry - stop_dist if signal_type == "BUY" else entry + stop_dist, profile["decimals"])
    tp_raw = round(entry + atr_v * profile["atr_tp_full"] if signal_type == "BUY" else entry - atr_v * profile["atr_tp_full"], profile["decimals"])
    tp_partial = round(entry + atr_v * profile["atr_tp_partial"] if signal_type == "BUY" else entry - atr_v * profile["atr_tp_partial"], profile["decimals"])

    sr_levels = find_sr_levels(d1h, lookback=SR_LOOKBACK, min_touches=SR_MIN_TOUCHES, zone_pct=SR_ZONE_PCT)
    tp = adjust_tp_to_sr(tp_raw, tp_partial, entry, signal_type, sr_levels, atr_v)

    lot_size = calculate_position_size(name, entry, sl)
    if lot_size <= 0:
        return None

    inst_class = next((klass for klass, profile_data in INSTRUMENT_PROFILES.items() if name in profile_data["pairs"]), "FOREX")
    logger.info("%s %s signal | score=%s/6 ml=%.0f%% adx=%.1f", name, signal_type, score, ml_confidence * 100, adx_v)

    return {
        "pair": name,
        "epic": epic,
        "type": signal_type,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "tp_partial": tp_partial,
        "atr": round(atr_v, profile["decimals"]),
        "adx": round(adx_v, 2),
        "lot_size": lot_size,
        "risk_per_unit": abs(entry - sl),
        "spread": live["spread"],
        "confluence_score": score,
        "ml_confidence": round(ml_confidence, 4),
        "instrument_class": inst_class,
        "sr_levels": sr_levels[:5],
    }


def open_trade(signal: Dict):
    order_ref = broker.place_order(
        signal["pair"],
        signal["epic"],
        signal["type"],
        signal["lot_size"],
        signal["entry"],
        signal["sl"],
        signal["tp"],
    )
    if not order_ref:
        logger.error("Order placement failed for %s", signal["pair"])
        send_telegram(f"Order failed for {signal['pair']} {signal['type']}")
        return

    fill = broker.confirm_fill(order_ref)
    if not fill:
        logger.warning("Fill confirmation unavailable for %s (%s)", signal["pair"], order_ref)
        fill = {}

    effective_risk = RISK_PERCENT * (DYNAMIC_RISK_REDUCTION if consecutive_losses >= MAX_CONSECUTIVE_LOSSES else 1.0)
    trade = {
        "pair": signal["pair"],
        "epic": signal["epic"],
        "type": signal["type"],
        "entry": signal["entry"],
        "sl": signal["sl"],
        "tp": signal["tp"],
        "tp_partial": signal["tp_partial"],
        "status": "OPEN",
        "result": "",
        "opened_at": time.time(),
        "closed_at": None,
        "risk_per_unit": signal["risk_per_unit"],
        "break_even_done": False,
        "partial_done": False,
        "entry_atr": signal["atr"],
        "lot_size": signal["lot_size"],
        "deal_ref": order_ref,
        "deal_id": fill.get("opened_deal_id", ""),
        "confluence_score": signal["confluence_score"],
        "ml_confidence": signal["ml_confidence"],
        "instrument_class": signal["instrument_class"],
        "exit_reason": "",
        "exit_price": None,
    }
    trade["db_id"] = db_save_trade(trade)
    trades.append(trade)
    last_trade_times[signal["pair"]] = trade["opened_at"]

    send_telegram(
        f"NEW TRADE | {signal['pair']} {signal['type']} [{signal['instrument_class']}]\n"
        f"Entry: {signal['entry']} | SL: {signal['sl']}\n"
        f"TP1: {signal['tp_partial']} | TP2: {signal['tp']}\n"
        f"Score: {signal['confluence_score']}/6 | ML: {signal['ml_confidence']:.0%} | ADX: {signal['adx']}\n"
        f"Risk: {effective_risk:.2f}% | Loss streak: {consecutive_losses}"
    )


def update_trade_status(trade: Dict, live: Dict):
    if trade["status"] != "OPEN":
        return

    price = live["bid"] if trade["type"] == "BUY" else live["ask"]
    risk = float(trade["risk_per_unit"])
    if risk <= 0:
        return

    profile = get_profile(trade["pair"])
    changed = False

    if trade["type"] == "BUY" and price <= trade["sl"]:
        finalize_trade(trade, price, "SL", broker_close=True)
        return
    if trade["type"] == "SELL" and price >= trade["sl"]:
        finalize_trade(trade, price, "SL", broker_close=True)
        return

    decimals = get_decimals(trade["pair"])[0]
    tp1 = trade.get("tp_partial")
    tp2 = trade.get("tp")

    if trade["type"] == "BUY":
        if tp2 and price >= float(tp2):
            finalize_trade(trade, price, "TP2", broker_close=True)
            return
        if tp1 and price >= float(tp1) and not trade["partial_done"]:
            new_sl = round(float(tp1), decimals)
            if new_sl > trade["sl"]:
                trade["sl"] = new_sl
                trade["partial_done"] = True
                changed = True
                _update_sl_on_broker(trade, new_sl)
                send_telegram(f"TP1 locked | {trade['pair']} BUY | SL moved to {new_sl}")
    else:
        if tp2 and price <= float(tp2):
            finalize_trade(trade, price, "TP2", broker_close=True)
            return
        if tp1 and price <= float(tp1) and not trade["partial_done"]:
            new_sl = round(float(tp1), decimals)
            if new_sl < trade["sl"]:
                trade["sl"] = new_sl
                trade["partial_done"] = True
                changed = True
                _update_sl_on_broker(trade, new_sl)
                send_telegram(f"TP1 locked | {trade['pair']} SELL | SL moved to {new_sl}")

    if not trade["break_even_done"]:
        progress_r = ((price - trade["entry"]) if trade["type"] == "BUY" else (trade["entry"] - price)) / risk
        if progress_r >= BREAK_EVEN_TRIGGER_R:
            be = round(trade["entry"], profile["decimals"])
            if (trade["type"] == "BUY" and be > trade["sl"]) or (trade["type"] == "SELL" and be < trade["sl"]):
                trade["sl"] = be
                trade["break_even_done"] = True
                changed = True
                _update_sl_on_broker(trade, be)
                send_telegram(f"Break-even locked | {trade['pair']} SL -> {be}")

    if trade["break_even_done"]:
        trail = float(trade.get("entry_atr", 0)) * profile["trailing_mult"]
        if trade["type"] == "BUY":
            new_sl = round(price - trail, profile["decimals"])
            if new_sl > trade["sl"]:
                trade["sl"] = new_sl
                changed = True
                _update_sl_on_broker(trade, new_sl)
        else:
            new_sl = round(price + trail, profile["decimals"])
            if new_sl < trade["sl"]:
                trade["sl"] = new_sl
                changed = True
                _update_sl_on_broker(trade, new_sl)

    elapsed_hours = (time.time() - trade["opened_at"]) / 3600
    if elapsed_hours >= STALE_TRADE_HOURS:
        progress_r = ((price - trade["entry"]) if trade["type"] == "BUY" else (trade["entry"] - price)) / risk
        if abs(progress_r) < 0.3:
            finalize_trade(trade, price, "STALE_EXIT", broker_close=True)
            return

    if changed:
        db_update_trade(trade)


def scan_pairs():
    if daily_loss_limit_hit() or is_near_news() or active_trade_count() >= MAX_ACTIVE_TRADES:
        return

    equity = broker.get_account_balance() if broker else INITIAL_EQUITY
    overheated, heat = heat_monitor.is_overheated(trades, equity)
    if overheated:
        logger.warning("Portfolio heat %.2f%% blocked new entries", heat)
        return

    open_trades = [trade for trade in trades if trade["status"] == "OPEN"]
    for name, search_term in pairs.items():
        if has_open_trade(name) or not cooldown_ready(name) or is_pair_disabled(name):
            continue
        if not in_optimal_session(name):
            continue

        epic = broker.get_epic(search_term)
        if not epic:
            continue

        signal = build_signal(name, epic)
        if not signal:
            continue

        blocked, reason = corr_filter.is_blocked(name, signal["type"], open_trades)
        if blocked:
            logger.info("Correlation blocked %s: %s", name, reason)
            continue

        overheated, _ = heat_monitor.is_overheated(trades, equity)
        if overheated:
            break

        open_trade(signal)
        open_trades = [trade for trade in trades if trade["status"] == "OPEN"]
        time.sleep(1)


def check_open_trades():
    for trade in [trade for trade in trades if trade["status"] == "OPEN"]:
        live = broker.get_live_price(trade["epic"])
        if live:
            update_trade_status(trade, live)
        time.sleep(0.3)


def send_heartbeat():
    equity = broker.get_account_balance() if broker else INITIAL_EQUITY
    decisive = wins + losses
    win_rate = round(wins / decisive * 100, 1) if decisive else 0.0
    heat = heat_monitor.get_heat_pct(trades, equity)
    send_telegram(
        f"Heartbeat\n"
        f"Mode: {'DEMO' if CAPITAL_DEMO else 'LIVE'} | Balance: {equity:.2f} | Open: {active_trade_count()}\n"
        f"W:{wins} L:{losses} BE:{breakevens} WR:{win_rate}% | Heat:{heat:.1f}%\n"
        f"Daily loss: {get_daily_loss_pct():.2f}% / {DAILY_LOSS_LIMIT_PCT}% | Loss streak: {consecutive_losses}"
    )


def send_report():
    decisive = wins + losses
    win_rate = round(wins / decisive * 100, 1) if decisive else 0.0
    send_telegram(
        f"Report | W:{wins} L:{losses} BE:{breakevens} WR:{win_rate}%\n\n"
        f"{pair_perf.get_summary()}"
    )


def main():
    global broker

    logger.info("=" * 60)
    logger.info("Forex Bot Live - Hardened")
    logger.info("=" * 60)

    setup_files()
    validate_runtime_config()
    load_state()

    broker = CapitalClient(
        api_key=CAPITAL_API_KEY,
        login=CAPITAL_LOGIN,
        password=CAPITAL_PASSWORD,
        demo=CAPITAL_DEMO,
    )

    start_balance = broker.get_account_balance()
    init_daily_pnl(start_balance if start_balance > 0 else INITIAL_EQUITY)
    reconcile_open_trades()
    ml_filter.train()

    pair_list = ", ".join(pairs.keys())
    send_telegram(
        f"Bot started | Mode: {'DEMO' if CAPITAL_DEMO else 'LIVE'}\n"
        f"Pairs: {pair_list}\n"
        f"Risk: {RISK_PERCENT}% | Score threshold: {MIN_CONFLUENCE_SCORE}/6\n"
        f"Dynamic risk: {'ON' if DYNAMIC_RISK_REDUCTION < 1 else 'OFF'} after {MAX_CONSECUTIVE_LOSSES} losses\n"
        f"ML: {'Active' if ml_filter.trained else 'Collecting data'}"
    )

    last_scan = 0.0
    last_heartbeat = 0.0
    last_report = 0.0
    last_ml_train = time.time()

    while True:
        try:
            now_ts = time.time()
            check_open_trades()

            if now_ts - last_scan >= SCAN_INTERVAL_SECONDS:
                scan_pairs()
                last_scan = now_ts

            if now_ts - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                send_heartbeat()
                last_heartbeat = now_ts

            if now_ts - last_report >= REPORT_INTERVAL_SECONDS:
                send_report()
                last_report = now_ts

            if now_ts - last_ml_train >= ML_RETRAIN_INTERVAL_SECONDS:
                ml_filter.train()
                last_ml_train = now_ts

            time.sleep(TRADE_CHECK_INTERVAL_SECONDS)
        except RuntimeError as exc:
            logger.critical("Bot halted: %s", exc)
            send_telegram(f"Bot halted: {exc}")
            break
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            send_telegram("Bot manually stopped.")
            break
        except Exception as exc:
            logger.error("Main loop error: %s", exc)
            time.sleep(30)


if __name__ == "__main__":
    main()
