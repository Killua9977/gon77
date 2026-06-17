import csv
import json
import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests


UTC = timezone.utc


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def now_utc() -> datetime:
    return datetime.now(UTC)


def combine_utc(day: date, hm: Tuple[int, int]) -> datetime:
    return datetime.combine(day, dt_time(hm[0], hm[1]), tzinfo=UTC)


def iso_z(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def capital_api_time(value: datetime) -> str:
    """Capital.com price history expects naive UTC timestamps without a Z suffix."""
    return value.astimezone(UTC).replace(tzinfo=None, microsecond=0).isoformat(timespec="seconds")


def extract_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def parse_snapshot_time(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    cleaned = raw.strip()
    candidates = [
        cleaned.replace("Z", "+00:00"),
        cleaned.replace("/", "-"),
    ]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            pass
    formats = [
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def round_price(value: float, decimals: int) -> float:
    return round(float(value), decimals)


def round_size(value: float, step: float) -> float:
    if step <= 0:
        return round(value, 6)
    rounded = math.floor((value + 1e-12) / step) * step
    return round(rounded, 6)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class PairConfig:
    pair: str
    strategy: str
    search: str
    decimals: int
    pip_size: float
    buffer: float
    min_range: float
    max_range: float
    tp1_mult: float
    tp2_mult: float
    trade_start: Tuple[int, int]
    trade_end: Tuple[int, int]
    max_spread: float
    min_size: float
    max_size: float
    size_step: float
    default_point_value: float
    news_currencies: Tuple[str, ...]
    bias_threshold: float = 0.0
    breakout_resolution: str = "MINUTE_5"
    range_resolution: str = "MINUTE_5"
    adx_min: float = 0.0


@dataclass(frozen=True)
class SessionWindow:
    pair: str
    strategy: str
    trade_date: str
    session_date: str
    range_start: datetime
    range_end: datetime
    trade_start: datetime
    trade_end: datetime
    source_day: Optional[str] = None


@dataclass
class InstrumentMeta:
    epic: str
    point_value: float
    min_size: float
    max_size: float
    size_step: float
    pip_size: float
    decimals: int
    account_currency: str = ""
    raw: Optional[Dict[str, Any]] = None


DATA_DIR = os.getenv("DATA_DIR", ".").strip() or "."
os.makedirs(DATA_DIR, exist_ok=True)

DB_FILE = os.path.join(DATA_DIR, "multi_state_fixed.db")
RESULTS_FILE = os.path.join(DATA_DIR, "multi_results_fixed.csv")
LOG_FILE = os.path.join(DATA_DIR, "multi_bot_fixed.log")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY", "").strip()
CAPITAL_LOGIN = os.getenv("CAPITAL_LOGIN", "").strip()
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD", "").strip()
CAPITAL_DEMO = env_bool("CAPITAL_DEMO", True)

INITIAL_EQUITY = env_float("INITIAL_EQUITY", 1000.0)
RISK_PERCENT = env_float("RISK_PERCENT", 1.0)
DAILY_LOSS_LIMIT_PCT = env_float("DAILY_LOSS_LIMIT_PCT", 3.0)
PAIR_DAILY_LOSS_LIMIT_PCT = env_float("PAIR_DAILY_LOSS_LIMIT_PCT", 1.5)
MAX_AUTH_RETRIES = env_int("MAX_AUTH_RETRIES", 5)
NEWS_BUFFER_MINS = env_int("NEWS_BUFFER_MINS", 30)
SCAN_INTERVAL = env_int("SCAN_INTERVAL", 10)
HEARTBEAT_SECS = env_int("HEARTBEAT_SECS", 1800)
REPORT_SECS = env_int("REPORT_SECS", 3600)
MAX_ACTIVE_TRADES = env_int("MAX_ACTIVE_TRADES", 3)
MAX_PORTFOLIO_HEAT_PCT = env_float("MAX_PORTFOLIO_HEAT_PCT", 3.0)
COUNT_FLOATING_DRAWDOWN = env_bool("COUNT_FLOATING_DRAWDOWN", True)
ALLOW_CORRELATED_TRADES = env_bool("ALLOW_CORRELATED_TRADES", False)
MAX_SPREAD_TO_RANGE_PCT = env_float("MAX_SPREAD_TO_RANGE_PCT", 0.15)
MAX_ENTRY_DISTANCE_PCT = env_float("MAX_ENTRY_DISTANCE_PCT", 0.25)
REQUIRE_BREAKOUT_CLOSE = env_bool("REQUIRE_BREAKOUT_CLOSE", True)
ENABLE_EOD_CLOSE = env_bool("ENABLE_EOD_CLOSE", True)
TP1_PARTIAL_CLOSE_PCT = clamp(env_float("TP1_PARTIAL_CLOSE_PCT", 0.50), 0.0, 0.90)
TP1_SL_MODE = os.getenv("TP1_SL_MODE", "BREAKEVEN").strip().upper() or "BREAKEVEN"
TP2_MODE = os.getenv("TP2_MODE", "CLOSE").strip().upper() or "CLOSE"
ORB_PREMARKET_BIAS = env_float("PRE_MARKET_BIAS", 0.003)
ARB_ADX_MIN = env_float("ARB_ADX_MIN", 18.0)

US500_NEWS_KEYWORDS = (
    "fomc",
    "fed",
    "powell",
    "interest rate",
    "cpi",
    "inflation",
    "ppi",
    "pce",
    "payroll",
    "non-farm",
    "nfp",
    "unemployment",
    "jobless",
    "gdp",
    "retail sales",
    "ism",
    "pmi",
    "jolts",
    "consumer confidence",
    "minutes",
    "rate decision",
)

PAIR_CONFIGS: Dict[str, PairConfig] = {
    "US500": PairConfig(
        pair="US500",
        strategy="ORB",
        search="US 500",
        decimals=1,
        pip_size=1.0,
        buffer=env_float("ORB_BUFFER", 0.5),
        min_range=env_float("ORB_MIN_RANGE", 5.0),
        max_range=env_float("ORB_MAX_RANGE", 60.0),
        tp1_mult=env_float("ORB_TP1_MULT", 1.0),
        tp2_mult=env_float("ORB_TP2_MULT", 2.0),
        trade_start=(14, 0),
        trade_end=(20, 0),
        max_spread=env_float("US500_MAX_SPREAD", 2.5),
        min_size=1.0,
        max_size=100.0,
        size_step=1.0,
        default_point_value=1.0,
        news_currencies=("USD",),
        bias_threshold=ORB_PREMARKET_BIAS,
        breakout_resolution="MINUTE_5",
        range_resolution="MINUTE_5",
    ),
    "EURUSD": PairConfig(
        pair="EURUSD",
        strategy="ARB",
        search="EURUSD",
        decimals=5,
        pip_size=0.0001,
        buffer=env_float("ARB_BUFFER", 0.0005),
        min_range=env_float("ARB_MIN_RANGE", 0.0010),
        max_range=env_float("ARB_MAX_RANGE", 0.0060),
        tp1_mult=env_float("ARB_TP1_MULT", 1.0),
        tp2_mult=env_float("ARB_TP2_MULT", 2.0),
        trade_start=(7, 0),
        trade_end=(12, 0),
        max_spread=env_float("EURUSD_MAX_SPREAD", 0.00020),
        min_size=1000.0,
        max_size=50000.0,
        size_step=1000.0,
        default_point_value=1.0,
        news_currencies=("EUR", "USD"),
        breakout_resolution="MINUTE_5",
        range_resolution="MINUTE_15",
        adx_min=ARB_ADX_MIN,
    ),
    "GBPUSD": PairConfig(
        pair="GBPUSD",
        strategy="ARB",
        search="GBPUSD",
        decimals=5,
        pip_size=0.0001,
        buffer=env_float("ARB_BUFFER", 0.0005),
        min_range=env_float("ARB_MIN_RANGE", 0.0010),
        max_range=env_float("ARB_MAX_RANGE", 0.0060),
        tp1_mult=env_float("ARB_TP1_MULT", 1.0),
        tp2_mult=env_float("ARB_TP2_MULT", 2.0),
        trade_start=(7, 0),
        trade_end=(12, 0),
        max_spread=env_float("GBPUSD_MAX_SPREAD", 0.00025),
        min_size=1000.0,
        max_size=50000.0,
        size_step=1000.0,
        default_point_value=1.0,
        news_currencies=("GBP", "USD"),
        breakout_resolution="MINUTE_5",
        range_resolution="MINUTE_15",
        adx_min=ARB_ADX_MIN,
    ),
    "USDJPY": PairConfig(
        pair="USDJPY",
        strategy="PDB",
        search="USDJPY",
        decimals=3,
        pip_size=0.01,
        buffer=env_float("PDB_BUFFER", 0.05),
        min_range=env_float("PDB_MIN_RANGE", 0.30),
        max_range=env_float("PDB_MAX_RANGE", 2.00),
        tp1_mult=env_float("PDB_TP1_MULT", 1.0),
        tp2_mult=env_float("PDB_TP2_MULT", 2.0),
        trade_start=(7, 0),
        trade_end=(20, 0),
        max_spread=env_float("USDJPY_MAX_SPREAD", 0.030),
        min_size=1000.0,
        max_size=50000.0,
        size_step=1000.0,
        default_point_value=1.0,
        news_currencies=("USD", "JPY"),
        breakout_resolution="MINUTE_5",
        range_resolution="HOUR",
    ),
}

CORRELATION_GROUPS = [
    {"EURUSD", "GBPUSD"},
]

RESULT_HEADERS = [
    "timestamp",
    "pair",
    "strategy",
    "trade_date",
    "direction",
    "entry",
    "initial_sl",
    "final_sl",
    "tp1",
    "tp2",
    "exit_price",
    "status",
    "profit_r",
    "pnl",
    "partial_pnl",
    "range_size",
    "point_value",
    "lot_size",
    "partial_done",
    "tp1_locked",
    "tp2_locked",
    "exit_reason",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


broker: Optional["CapitalClient"] = None
open_trades: List[Dict[str, Any]] = []
ranges: Dict[str, Dict[str, Any]] = {}
epics: Dict[str, str] = {}
market_meta: Dict[str, InstrumentMeta] = {}
blocked_pairs: Set[str] = set()
account_currency = ""
last_trade_day = ""

_news_cache: List[Dict[str, Any]] = []
_news_last_fetch = 0.0


def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - best effort alerting
        logger.error("Telegram send failed: %s", exc)


def ensure_csv(path: str, headers: Sequence[str]):
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(headers)


def validate_runtime_config():
    missing = [
        name
        for name, value in {
            "CAPITAL_API_KEY": CAPITAL_API_KEY,
            "CAPITAL_LOGIN": CAPITAL_LOGIN,
            "CAPITAL_PASSWORD": CAPITAL_PASSWORD,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if TP1_SL_MODE not in {"BREAKEVEN", "TP1"}:
        raise RuntimeError("TP1_SL_MODE must be BREAKEVEN or TP1")
    if TP2_MODE not in {"CLOSE", "TRAIL"}:
        raise RuntimeError("TP2_MODE must be CLOSE or TRAIL")
    if not 0 < RISK_PERCENT <= 5:
        raise RuntimeError("RISK_PERCENT must be between 0 and 5")
    if DAILY_LOSS_LIMIT_PCT <= 0 or PAIR_DAILY_LOSS_LIMIT_PCT <= 0:
        raise RuntimeError("Loss limits must be positive")
    if MAX_ACTIVE_TRADES < 1:
        raise RuntimeError("MAX_ACTIVE_TRADES must be at least 1")
    if MAX_PORTFOLIO_HEAT_PCT <= 0:
        raise RuntimeError("MAX_PORTFOLIO_HEAT_PCT must be positive")
    if not TOKEN or not CHAT_ID:
        logger.warning("Telegram env vars are missing; alerts are disabled")


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_ranges (
            pair TEXT,
            trade_date TEXT,
            strategy TEXT,
            session_date TEXT,
            session_start TEXT,
            session_end TEXT,
            range_high REAL,
            range_low REAL,
            range_size REAL,
            bias TEXT DEFAULT '',
            disabled INTEGER DEFAULT 0,
            skip_reason TEXT DEFAULT '',
            built_at REAL,
            PRIMARY KEY (pair, trade_date)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            strategy TEXT,
            trade_date TEXT,
            session_date TEXT,
            direction TEXT,
            entry REAL,
            requested_entry REAL DEFAULT 0,
            sl REAL,
            initial_sl REAL,
            broker_sl REAL,
            tp1 REAL,
            tp2 REAL,
            range_high REAL,
            range_low REAL,
            range_size REAL,
            lot_size REAL,
            initial_size REAL,
            remaining_size REAL,
            closed_size REAL DEFAULT 0,
            partial_pnl REAL DEFAULT 0,
            risk_per_unit REAL DEFAULT 0,
            point_value REAL DEFAULT 1,
            initial_risk_cash REAL DEFAULT 0,
            deal_ref TEXT,
            deal_id TEXT,
            epic TEXT,
            status TEXT DEFAULT 'OPEN',
            result TEXT DEFAULT '',
            exit_price REAL,
            exit_reason TEXT DEFAULT '',
            opened_at REAL,
            closed_at REAL,
            pnl REAL DEFAULT 0,
            profit_r REAL DEFAULT 0,
            tp1_locked INTEGER DEFAULT 0,
            tp2_locked INTEGER DEFAULT 0,
            partial_done INTEGER DEFAULT 0,
            break_even_done INTEGER DEFAULT 0,
            review_required INTEGER DEFAULT 0,
            review_note TEXT DEFAULT '',
            broker_fill_price REAL DEFAULT 0,
            broker_partial_ref TEXT DEFAULT ''
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            event_time REAL,
            event_type TEXT,
            price REAL,
            note TEXT DEFAULT ''
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_pnl (
            trade_date TEXT PRIMARY KEY,
            start_equity REAL,
            realized_pnl REAL DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            win_count INTEGER DEFAULT 0,
            loss_count INTEGER DEFAULT 0,
            breakeven_count INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pair_daily_pnl (
            trade_date TEXT,
            pair TEXT,
            realized_pnl REAL DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            win_count INTEGER DEFAULT 0,
            loss_count INTEGER DEFAULT 0,
            breakeven_count INTEGER DEFAULT 0,
            PRIMARY KEY (trade_date, pair)
        )
        """
    )
    conn.commit()
    conn.close()


def db_log_event(trade_id: int, event_type: str, price: Optional[float], note: str = ""):
    conn = connect_db()
    conn.execute(
        """
        INSERT INTO trade_events (trade_id, event_time, event_type, price, note)
        VALUES (?,?,?,?,?)
        """,
        (trade_id, time.time(), event_type, price, note),
    )
    conn.commit()
    conn.close()


def db_save_range(state: Dict[str, Any]):
    conn = connect_db()
    conn.execute(
        """
        INSERT OR REPLACE INTO daily_ranges (
            pair, trade_date, strategy, session_date, session_start, session_end,
            range_high, range_low, range_size, bias, disabled, skip_reason, built_at
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            state["pair"],
            state["trade_date"],
            state["strategy"],
            state["session_date"],
            state["session_start"],
            state["session_end"],
            state["high"],
            state["low"],
            state["size"],
            state.get("bias", ""),
            int(state.get("disabled", False)),
            state.get("skip_reason", ""),
            state.get("built_at", time.time()),
        ),
    )
    conn.commit()
    conn.close()


def db_get_range(pair: str, trade_date: str) -> Optional[Dict[str, Any]]:
    conn = connect_db()
    row = conn.execute(
        """
        SELECT * FROM daily_ranges
        WHERE pair=? AND trade_date=?
        """,
        (pair, trade_date),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "pair": row["pair"],
        "trade_date": row["trade_date"],
        "strategy": row["strategy"],
        "session_date": row["session_date"],
        "session_start": row["session_start"],
        "session_end": row["session_end"],
        "high": row["range_high"],
        "low": row["range_low"],
        "size": row["range_size"],
        "bias": row["bias"],
        "disabled": bool(row["disabled"]),
        "skip_reason": row["skip_reason"],
        "built_at": row["built_at"],
    }


def db_save_trade(trade: Dict[str, Any]) -> int:
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO trades (
            pair, strategy, trade_date, session_date, direction, entry, requested_entry,
            sl, initial_sl, broker_sl, tp1, tp2, range_high, range_low, range_size,
            lot_size, initial_size, remaining_size, closed_size, partial_pnl, risk_per_unit,
            point_value, initial_risk_cash, deal_ref, deal_id, epic, status, result,
            exit_price, exit_reason, opened_at, closed_at, pnl, profit_r, tp1_locked,
            tp2_locked, partial_done, break_even_done, review_required, review_note,
            broker_fill_price, broker_partial_ref
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            trade["pair"],
            trade["strategy"],
            trade["trade_date"],
            trade["session_date"],
            trade["direction"],
            trade["entry"],
            trade["requested_entry"],
            trade["sl"],
            trade["initial_sl"],
            trade["broker_sl"],
            trade["tp1"],
            trade["tp2"],
            trade["range_high"],
            trade["range_low"],
            trade["range_size"],
            trade["lot_size"],
            trade["initial_size"],
            trade["remaining_size"],
            trade["closed_size"],
            trade["partial_pnl"],
            trade["risk_per_unit"],
            trade["point_value"],
            trade["initial_risk_cash"],
            trade.get("deal_ref", ""),
            trade.get("deal_id", ""),
            trade["epic"],
            trade["status"],
            trade.get("result", ""),
            trade.get("exit_price"),
            trade.get("exit_reason", ""),
            trade["opened_at"],
            trade.get("closed_at"),
            trade.get("pnl", 0),
            trade.get("profit_r", 0),
            int(trade.get("tp1_locked", False)),
            int(trade.get("tp2_locked", False)),
            int(trade.get("partial_done", False)),
            int(trade.get("break_even_done", False)),
            int(trade.get("review_required", False)),
            trade.get("review_note", ""),
            trade.get("broker_fill_price", 0),
            trade.get("broker_partial_ref", ""),
        ),
    )
    trade_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return trade_id


def db_update_trade(trade: Dict[str, Any]):
    conn = connect_db()
    conn.execute(
        """
        UPDATE trades
        SET sl=?, broker_sl=?, tp1=?, tp2=?, remaining_size=?, closed_size=?, partial_pnl=?,
            deal_ref=?, deal_id=?, status=?, result=?, exit_price=?, exit_reason=?, opened_at=?,
            closed_at=?, pnl=?, profit_r=?, tp1_locked=?, tp2_locked=?, partial_done=?,
            break_even_done=?, review_required=?, review_note=?, broker_fill_price=?,
            broker_partial_ref=?
        WHERE id=?
        """,
        (
            trade["sl"],
            trade["broker_sl"],
            trade["tp1"],
            trade["tp2"],
            trade["remaining_size"],
            trade["closed_size"],
            trade["partial_pnl"],
            trade.get("deal_ref", ""),
            trade.get("deal_id", ""),
            trade["status"],
            trade.get("result", ""),
            trade.get("exit_price"),
            trade.get("exit_reason", ""),
            trade["opened_at"],
            trade.get("closed_at"),
            trade.get("pnl", 0),
            trade.get("profit_r", 0),
            int(trade.get("tp1_locked", False)),
            int(trade.get("tp2_locked", False)),
            int(trade.get("partial_done", False)),
            int(trade.get("break_even_done", False)),
            int(trade.get("review_required", False)),
            trade.get("review_note", ""),
            trade.get("broker_fill_price", 0),
            trade.get("broker_partial_ref", ""),
            trade["id"],
        ),
    )
    conn.commit()
    conn.close()


def row_to_trade(row: sqlite3.Row) -> Dict[str, Any]:
    trade = dict(row)
    for key in ["tp1_locked", "tp2_locked", "partial_done", "break_even_done", "review_required"]:
        trade[key] = bool(trade.get(key, 0))
    return trade


def db_load_open_trades() -> List[Dict[str, Any]]:
    conn = connect_db()
    rows = conn.execute(
        """
        SELECT * FROM trades
        WHERE status='OPEN'
        ORDER BY opened_at
        """
    ).fetchall()
    conn.close()
    return [row_to_trade(row) for row in rows]


def db_load_review_pairs(trade_date: str) -> Set[str]:
    conn = connect_db()
    rows = conn.execute(
        """
        SELECT DISTINCT pair FROM trades
        WHERE trade_date=? AND review_required=1 AND status!='CLOSED'
        """,
        (trade_date,),
    ).fetchall()
    conn.close()
    return {row["pair"] for row in rows}


def db_has_trade_today(pair: str, trade_date: str) -> bool:
    conn = connect_db()
    row = conn.execute(
        """
        SELECT id FROM trades
        WHERE pair=? AND trade_date=?
        LIMIT 1
        """,
        (pair, trade_date),
    ).fetchone()
    conn.close()
    return row is not None


def init_daily_pnl(trade_date: str, start_equity: float):
    conn = connect_db()
    conn.execute(
        """
        INSERT OR IGNORE INTO daily_pnl (
            trade_date, start_equity, realized_pnl, trade_count, win_count, loss_count, breakeven_count
        )
        VALUES (?,?,0,0,0,0,0)
        """,
        (trade_date, start_equity),
    )
    for pair in PAIR_CONFIGS:
        conn.execute(
            """
            INSERT OR IGNORE INTO pair_daily_pnl (
                trade_date, pair, realized_pnl, trade_count, win_count, loss_count, breakeven_count
            )
            VALUES (?,?,0,0,0,0,0)
            """,
            (trade_date, pair),
        )
    conn.commit()
    conn.close()


def get_day_start_equity(trade_date: str) -> float:
    conn = connect_db()
    row = conn.execute(
        "SELECT start_equity FROM daily_pnl WHERE trade_date=?",
        (trade_date,),
    ).fetchone()
    conn.close()
    if not row:
        return INITIAL_EQUITY
    return float(row["start_equity"] or INITIAL_EQUITY)


def record_closed_trade_pnl(trade_date: str, pair: str, result: str, pnl: float):
    conn = connect_db()
    conn.execute(
        """
        UPDATE daily_pnl
        SET realized_pnl=realized_pnl+?,
            trade_count=trade_count+1,
            win_count=win_count+?,
            loss_count=loss_count+?,
            breakeven_count=breakeven_count+?
        WHERE trade_date=?
        """,
        (
            pnl,
            1 if result == "WIN" else 0,
            1 if result == "LOSS" else 0,
            1 if result == "BE" else 0,
            trade_date,
        ),
    )
    conn.execute(
        """
        UPDATE pair_daily_pnl
        SET realized_pnl=realized_pnl+?,
            trade_count=trade_count+1,
            win_count=win_count+?,
            loss_count=loss_count+?,
            breakeven_count=breakeven_count+?
        WHERE trade_date=? AND pair=?
        """,
        (
            pnl,
            1 if result == "WIN" else 0,
            1 if result == "LOSS" else 0,
            1 if result == "BE" else 0,
            trade_date,
            pair,
        ),
    )
    conn.commit()
    conn.close()


def get_daily_realized_pnl(trade_date: str) -> float:
    conn = connect_db()
    row = conn.execute(
        "SELECT realized_pnl FROM daily_pnl WHERE trade_date=?",
        (trade_date,),
    ).fetchone()
    conn.close()
    return float(row["realized_pnl"] or 0.0) if row else 0.0


def get_pair_realized_pnl(trade_date: str, pair: str) -> float:
    conn = connect_db()
    row = conn.execute(
        """
        SELECT realized_pnl FROM pair_daily_pnl
        WHERE trade_date=? AND pair=?
        """,
        (trade_date, pair),
    ).fetchone()
    conn.close()
    return float(row["realized_pnl"] or 0.0) if row else 0.0


def db_closed_trade_rows() -> List[sqlite3.Row]:
    conn = connect_db()
    rows = conn.execute(
        """
        SELECT pair, strategy, result, pnl, profit_r, exit_reason
        FROM trades
        WHERE status='CLOSED'
        ORDER BY opened_at
        """
    ).fetchall()
    conn.close()
    return rows


class CapitalClient:
    def __init__(self, api_key: str, login: str, password: str, demo: bool = True):
        self.api_key = api_key
        self.login = login
        self.password = password
        self.demo = demo
        self.base_url = (
            "https://demo-api-capital.backend-capital.com"
            if demo
            else "https://api-capital.backend-capital.com"
        )
        self.cst: Optional[str] = None
        self.security_token: Optional[str] = None
        self.session = requests.Session()
        self.epic_cache: Dict[str, str] = {}
        self.market_cache: Dict[str, Dict[str, Any]] = {}
        self.authenticate()

    def authenticate(self):
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_AUTH_RETRIES + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/v1/session",
                    headers={
                        "X-CAP-API-KEY": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "identifier": self.login,
                        "password": self.password,
                        "encryptedPassword": False,
                    },
                    timeout=30,
                )
                if response.status_code != 200:
                    raise RuntimeError(response.text[:200])
                self.cst = response.headers.get("CST")
                self.security_token = response.headers.get("X-SECURITY-TOKEN")
                logger.info("Connected to Capital.com (%s)", "DEMO" if self.demo else "LIVE")
                return
            except Exception as exc:
                last_error = exc
                logger.error("Auth attempt %s/%s failed: %s", attempt, MAX_AUTH_RETRIES, exc)
                if attempt < MAX_AUTH_RETRIES:
                    time.sleep(5 * attempt)
        raise RuntimeError(f"Auth failed after {MAX_AUTH_RETRIES} retries: {last_error}")

    def _req(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        for attempt in range(1, MAX_AUTH_RETRIES + 1):
            try:
                url = f"{self.base_url}{endpoint}"
                headers = {
                    "X-CAP-API-KEY": self.api_key,
                    "CST": self.cst or "",
                    "X-SECURITY-TOKEN": self.security_token or "",
                    "Content-Type": "application/json",
                }
                fn = {
                    "GET": self.session.get,
                    "POST": self.session.post,
                    "PUT": self.session.put,
                    "DELETE": self.session.delete,
                }[method]
                kwargs: Dict[str, Any] = {"headers": headers, "timeout": 30}
                if params:
                    kwargs["params"] = params
                if method in {"POST", "PUT", "DELETE"} and data is not None:
                    kwargs["json"] = data
                response = fn(url, **kwargs)
                if response.status_code in {401, 403}:
                    self.authenticate()
                    continue
                if response.status_code == 429:
                    wait = int(response.headers.get("Retry-After", "10"))
                    logger.warning("Rate limited, waiting %ss", wait)
                    time.sleep(wait)
                    continue
                if response.status_code != 200:
                    logger.error("API %s %s failed: %s", method, endpoint, response.text[:300])
                    return None
                if not response.text.strip():
                    return {}
                return response.json()
            except Exception as exc:
                logger.error("Request error on %s %s attempt %s: %s", method, endpoint, attempt, exc)
                if attempt < MAX_AUTH_RETRIES:
                    time.sleep(attempt)
                    continue
        return None

    def get_epic(self, search_term: str) -> Optional[str]:
        if search_term in self.epic_cache:
            return self.epic_cache[search_term]
        data = self._req("GET", "/api/v1/markets", params={"searchTerm": search_term})
        if data and data.get("markets"):
            epic = data["markets"][0]["epic"]
            self.epic_cache[search_term] = epic
            return epic
        return None

    def get_market_details(self, epic: str) -> Optional[Dict[str, Any]]:
        if epic in self.market_cache:
            return self.market_cache[epic]
        data = self._req("GET", f"/api/v1/markets/{epic}")
        if data:
            self.market_cache[epic] = data
        return data

    def get_account_info(self) -> Dict[str, Any]:
        data = self._req("GET", "/api/v1/accounts")
        if not data or not data.get("accounts"):
            return {}
        account = data["accounts"][0]
        balance = float(account.get("balance", {}).get("balance", 0) or 0)
        currency = account.get("currency", "") or account.get("preferred") or ""
        return {"balance": balance, "currency": currency}

    def get_account_balance(self) -> float:
        return float(self.get_account_info().get("balance", 0) or 0)

    def get_candles(
        self,
        epic: str,
        resolution: str,
        max_count: int = 200,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        params: Dict[str, Any] = {"resolution": resolution, "max": max_count}
        if start and end:
            params["from"] = capital_api_time(start)
            params["to"] = capital_api_time(end)
        data = self._req("GET", f"/api/v1/prices/{epic}", params=params)
        prices = data.get("prices") if data else None
        if not prices and start and end:
            fallback = self._req(
                "GET",
                f"/api/v1/prices/{epic}",
                params={"resolution": resolution, "max": max(max_count, 400)},
            )
            prices = fallback.get("prices") if fallback else None
        if not prices:
            return None
        candles: List[Dict[str, Any]] = []
        for item in prices:
            raw_time = item.get("snapshotTimeUTC") or item.get("snapshotTime")
            candle_time = parse_snapshot_time(str(raw_time))
            if not candle_time:
                continue
            if start and candle_time < start:
                continue
            if end and candle_time >= end:
                continue
            try:
                candles.append(
                    {
                        "time": candle_time,
                        "open": float(item["openPrice"]["bid"]),
                        "high": float(item["highPrice"]["bid"]),
                        "low": float(item["lowPrice"]["bid"]),
                        "close": float(item["closePrice"]["bid"]),
                        "volume": float(item.get("lastTradedVolume", 0) or 0),
                    }
                )
            except Exception:
                continue
        candles.sort(key=lambda candle: candle["time"])
        return candles

    def get_live_price(self, epic: str) -> Optional[Dict[str, float]]:
        data = self._req("GET", f"/api/v1/markets/{epic}")
        if not data or "snapshot" not in data:
            return None
        bid = float(data["snapshot"]["bid"])
        ask = float(data["snapshot"]["offer"])
        return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2.0, "spread": ask - bid}

    def get_open_positions(self) -> List[Dict[str, Any]]:
        data = self._req("GET", "/api/v1/positions")
        if not data:
            return []
        positions = []
        for item in data.get("positions", []):
            position = item.get("position", {})
            market = item.get("market", {})
            positions.append(
                {
                    "dealId": position.get("dealId", ""),
                    "dealReference": position.get("dealReference", ""),
                    "epic": market.get("epic", ""),
                    "direction": position.get("direction", ""),
                    "size": float(position.get("size", 0) or 0),
                    "level": float(position.get("level", 0) or 0),
                    "stopLevel": extract_float(position.get("stopLevel")),
                }
            )
        return positions

    def place_order(
        self,
        epic: str,
        direction: str,
        units: float,
        sl: float,
        decimals: int,
        force_open: bool = True,
    ) -> Optional[str]:
        payload = {
            "epic": epic,
            "direction": direction,
            "size": float(units),
            "stopLevel": float(round(sl, decimals)),
            "guaranteedStop": False,
            "forceOpen": force_open,
        }
        logger.info("Order: %s %s %s stop=%.5f", direction, units, epic, sl)
        data = self._req("POST", "/api/v1/positions", data=payload)
        if data and data.get("dealReference"):
            return data["dealReference"]
        return None

    def confirm_fill(self, deal_ref: str) -> Optional[Dict[str, Any]]:
        time.sleep(2)
        data = self._req("GET", f"/api/v1/confirms/{deal_ref}")
        if not data:
            return None
        for item in data.get("affectedDeals", []):
            if item.get("status") == "OPENED":
                data["opened_deal_id"] = item.get("dealId", "")
                break
        return data

    def _resolve_position(
        self,
        identifier: str,
        epic_fallback: str = "",
        direction: str = "",
    ) -> Optional[Dict[str, Any]]:
        positions = self.get_open_positions()
        for position in positions:
            if identifier and (
                position["dealId"] == identifier or position["dealReference"] == identifier
            ):
                return position
        if epic_fallback:
            matches = [
                position
                for position in positions
                if position.get("epic", "").upper() == epic_fallback.upper()
                and (not direction or position.get("direction", "").upper() == direction.upper())
            ]
            if len(matches) == 1:
                return matches[0]
        return None

    def update_sl(
        self,
        identifier: str,
        new_sl: float,
        decimals: int,
        epic_fallback: str = "",
        direction: str = "",
    ) -> bool:
        position = self._resolve_position(identifier, epic_fallback, direction)
        if not position or not position.get("dealId"):
            return False
        result = self._req(
            "PUT",
            f"/api/v1/positions/{position['dealId']}",
            data={"stopLevel": float(round(new_sl, decimals))},
        )
        return result is not None

    def close_position(
        self,
        identifier: str,
        epic_fallback: str = "",
        size: Optional[float] = None,
        direction: str = "",
    ) -> bool:
        position = self._resolve_position(identifier, epic_fallback, direction)
        if not position:
            logger.warning("Position not found for close: %s %s", identifier, epic_fallback)
            return True
        current_size = float(position.get("size", 0) or 0)
        if size is not None and 0 < size < current_size:
            close_direction = "SELL" if position["direction"].upper() == "BUY" else "BUY"
            result = self._req(
                "POST",
                "/api/v1/positions",
                data={
                    "epic": position["epic"],
                    "direction": close_direction,
                    "size": float(size),
                    "forceOpen": False,
                },
            )
            return bool(result and result.get("dealReference"))
        result = self._req("DELETE", f"/api/v1/positions/{position['dealId']}")
        return result is not None


def instrument_meta_for_pair(pair: str) -> InstrumentMeta:
    cfg = PAIR_CONFIGS[pair]
    epic = epics[pair]
    cached = market_meta.get(pair)
    if cached:
        return cached
    details = broker.get_market_details(epic) if broker else None
    point_value = cfg.default_point_value
    min_size = cfg.min_size
    max_size = cfg.max_size
    size_step = cfg.size_step
    account_ccy = ""
    if details:
        instrument = details.get("instrument", {})
        dealing_rules = details.get("dealingRules", {})
        currencies = instrument.get("currencies") or details.get("currencies") or []
        if currencies:
            first = currencies[0]
            account_ccy = first.get("code", "") or first.get("symbol", "")
            exchange_rate = extract_float(first.get("exchangeRate")) or 1.0
        else:
            exchange_rate = 1.0
        pip_value = extract_float(instrument.get("valueOfOnePip"))
        pip_size = extract_float(instrument.get("onePipMeans")) or cfg.pip_size
        contract_size = extract_float(instrument.get("contractSize")) or 1.0
        if pip_value and pip_size and pip_size > 0:
            point_value = (pip_value / pip_size) * exchange_rate
        else:
            point_value = contract_size * exchange_rate
        parsed_min = extract_float(
            (dealing_rules.get("minDealSize") or {}).get("value")
            if isinstance(dealing_rules.get("minDealSize"), dict)
            else dealing_rules.get("minDealSize")
        )
        parsed_step = extract_float(
            (dealing_rules.get("minStepDistance") or {}).get("value")
            if isinstance(dealing_rules.get("minStepDistance"), dict)
            else None
        )
        if parsed_min and parsed_min > 0:
            min_size = parsed_min
        if parsed_step and parsed_step > 0 and cfg.strategy != "ORB":
            size_step = cfg.size_step
        parsed_contract = extract_float(instrument.get("lotSize"))
        if parsed_contract and parsed_contract > 0 and cfg.strategy != "ORB":
            size_step = max(size_step, min_size)
    meta = InstrumentMeta(
        epic=epic,
        point_value=max(point_value, 1e-6),
        min_size=min_size,
        max_size=max_size,
        size_step=size_step,
        pip_size=cfg.pip_size,
        decimals=cfg.decimals,
        account_currency=account_ccy,
        raw=details,
    )
    market_meta[pair] = meta
    return meta


def previous_weekday(day: date) -> date:
    current = day - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def get_session_window(pair: str, ref: Optional[datetime] = None) -> Optional[SessionWindow]:
    ref = ref or now_utc()
    trade_day = ref.date()
    if trade_day.weekday() >= 5:
        return None
    cfg = PAIR_CONFIGS[pair]
    if cfg.strategy == "ORB":
        range_start = combine_utc(trade_day, (13, 30))
        range_end = combine_utc(trade_day, (14, 0))
        trade_start = combine_utc(trade_day, cfg.trade_start)
        trade_end = combine_utc(trade_day, cfg.trade_end)
        return SessionWindow(
            pair=pair,
            strategy=cfg.strategy,
            trade_date=trade_day.isoformat(),
            session_date=trade_day.isoformat(),
            range_start=range_start,
            range_end=range_end,
            trade_start=trade_start,
            trade_end=trade_end,
            source_day=trade_day.isoformat(),
        )
    if cfg.strategy == "ARB":
        range_start = combine_utc(trade_day - timedelta(days=1), (23, 0))
        range_end = combine_utc(trade_day, (7, 0))
        trade_start = combine_utc(trade_day, cfg.trade_start)
        trade_end = combine_utc(trade_day, cfg.trade_end)
        return SessionWindow(
            pair=pair,
            strategy=cfg.strategy,
            trade_date=trade_day.isoformat(),
            session_date=(trade_day - timedelta(days=1)).isoformat(),
            range_start=range_start,
            range_end=range_end,
            trade_start=trade_start,
            trade_end=trade_end,
            source_day=(trade_day - timedelta(days=1)).isoformat(),
        )
    prev_day = previous_weekday(trade_day)
    range_start = combine_utc(prev_day, (0, 0))
    range_end = combine_utc(prev_day + timedelta(days=1), (0, 0))
    trade_start = combine_utc(trade_day, cfg.trade_start)
    trade_end = combine_utc(trade_day, cfg.trade_end)
    return SessionWindow(
        pair=pair,
        strategy=cfg.strategy,
        trade_date=trade_day.isoformat(),
        session_date=prev_day.isoformat(),
        range_start=range_start,
        range_end=range_end,
        trade_start=trade_start,
        trade_end=trade_end,
        source_day=prev_day.isoformat(),
    )


def make_range_state(
    pair: str,
    window: SessionWindow,
    high: float,
    low: float,
    size: float,
    bias: str = "",
    disabled: bool = False,
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "pair": pair,
        "trade_date": window.trade_date,
        "strategy": window.strategy,
        "session_date": window.session_date,
        "session_start": iso_z(window.range_start),
        "session_end": iso_z(window.range_end),
        "high": high,
        "low": low,
        "size": size,
        "bias": bias,
        "disabled": disabled,
        "skip_reason": reason,
        "built_at": time.time(),
    }


def range_disabled(pair: str, window: SessionWindow, high: float, low: float, size: float, reason: str) -> Dict[str, Any]:
    state = make_range_state(pair, window, high, low, size, "", True, reason)
    db_save_range(state)
    return state


def fetch_news() -> List[Dict[str, Any]]:
    global _news_cache, _news_last_fetch
    if time.time() - _news_last_fetch < 3600 and _news_cache:
        return _news_cache
    sources = [
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json?refresh=1",
    ]
    for url in sources:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            events: List[Dict[str, Any]] = []
            for item in response.json():
                impact = str(item.get("impact", "")).lower()
                if impact not in {"high", "medium"}:
                    continue
                raw_date = item.get("date") or item.get("dateUtc") or item.get("timestamp")
                event_time = parse_snapshot_time(str(raw_date))
                if not event_time:
                    continue
                events.append(
                    {
                        "time": event_time,
                        "title": item.get("title", ""),
                        "impact": impact,
                        "country": str(item.get("country", "")).upper(),
                    }
                )
            _news_cache = events
            _news_last_fetch = time.time()
            return _news_cache
        except Exception as exc:
            logger.warning("News fetch failed from %s: %s", url, exc)
    return _news_cache


def event_affects_pair(pair: str, event: Dict[str, Any]) -> bool:
    country = event.get("country", "").upper()
    impact = event.get("impact", "").lower()
    title = str(event.get("title", "")).lower()
    cfg = PAIR_CONFIGS[pair]
    if pair == "US500":
        if country == "ALL":
            return impact == "high"
        if country != "USD":
            return False
        return impact == "high" or any(keyword in title for keyword in US500_NEWS_KEYWORDS)
    return country == "ALL" or country in cfg.news_currencies


def is_near_news(pair: str) -> bool:
    now = now_utc()
    for event in fetch_news():
        if not event_affects_pair(pair, event):
            continue
        if abs((event["time"] - now).total_seconds()) <= NEWS_BUFFER_MINS * 60:
            logger.info(
                "%s news blackout: %s [%s/%s]",
                pair,
                event["title"],
                event.get("country", "?"),
                event.get("impact", "?"),
            )
            return True
    return False


def build_orb_range(pair: str, epic: str, window: SessionWindow) -> Optional[Dict[str, Any]]:
    cfg = PAIR_CONFIGS[pair]
    candles = broker.get_candles(
        epic,
        cfg.range_resolution,
        max_count=24,
        start=window.range_start,
        end=window.range_end,
    )
    if not candles or len(candles) < 5:
        logger.warning("%s: incomplete ORB window", pair)
        return None
    high = round_price(max(candle["high"] for candle in candles), cfg.decimals)
    low = round_price(min(candle["low"] for candle in candles), cfg.decimals)
    size = round(abs(high - low), cfg.decimals)
    bias = ""
    bias_candles = broker.get_candles(
        epic,
        "HOUR",
        max_count=12,
        start=combine_utc(window.range_start.date(), (8, 0)),
        end=window.range_start,
    )
    if bias_candles and len(bias_candles) >= 4:
        drift = safe_div(bias_candles[-1]["close"] - bias_candles[0]["open"], bias_candles[0]["open"])
        if drift > cfg.bias_threshold:
            bias = "BULL"
        elif drift < -cfg.bias_threshold:
            bias = "BEAR"
        logger.info("%s pre-market bias drift %.3f%% -> %s", pair, drift * 100, bias or "NEUTRAL")
    if size < cfg.min_range:
        return range_disabled(pair, window, high, low, size, "too_small")
    if size > cfg.max_range:
        return range_disabled(pair, window, high, low, size, "too_large")
    state = make_range_state(pair, window, high, low, size, bias, False, "")
    db_save_range(state)
    return state


def build_arb_range(pair: str, epic: str, window: SessionWindow) -> Optional[Dict[str, Any]]:
    cfg = PAIR_CONFIGS[pair]
    candles = broker.get_candles(
        epic,
        cfg.range_resolution,
        max_count=48,
        start=window.range_start,
        end=window.range_end,
    )
    if not candles or len(candles) < 12:
        logger.warning("%s: incomplete Asian range window", pair)
        return None
    high = round_price(max(candle["high"] for candle in candles), cfg.decimals)
    low = round_price(min(candle["low"] for candle in candles), cfg.decimals)
    size = round(abs(high - low), cfg.decimals)
    if size < cfg.min_range:
        return range_disabled(pair, window, high, low, size, "too_small")
    if size > cfg.max_range:
        return range_disabled(pair, window, high, low, size, "too_large")
    state = make_range_state(pair, window, high, low, size)
    db_save_range(state)
    return state


def build_pdb_range(pair: str, epic: str, window: SessionWindow) -> Optional[Dict[str, Any]]:
    cfg = PAIR_CONFIGS[pair]
    candles = broker.get_candles(
        epic,
        "HOUR",
        max_count=36,
        start=window.range_start,
        end=window.range_end,
    )
    if not candles or len(candles) < 12:
        logger.warning("%s: incomplete previous-day window", pair)
        return None
    high = round_price(max(candle["high"] for candle in candles), cfg.decimals)
    low = round_price(min(candle["low"] for candle in candles), cfg.decimals)
    size = round(abs(high - low), cfg.decimals)
    if size < cfg.min_range:
        return range_disabled(pair, window, high, low, size, "too_small")
    if size > cfg.max_range:
        return range_disabled(pair, window, high, low, size, "too_large")
    state = make_range_state(pair, window, high, low, size)
    db_save_range(state)
    return state


def build_range_for_pair(pair: str, ref: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    window = get_session_window(pair, ref)
    if not window:
        return None
    if now_utc() < window.range_end:
        return None
    cached = db_get_range(pair, window.trade_date)
    if cached:
        return cached
    epic = epics.get(pair)
    if not epic:
        return None
    logger.info("Building range for %s (%s)", pair, window.strategy)
    if window.strategy == "ORB":
        state = build_orb_range(pair, epic, window)
    elif window.strategy == "ARB":
        state = build_arb_range(pair, epic, window)
    else:
        state = build_pdb_range(pair, epic, window)
    if state:
        ranges[pair] = state
    return state


def calculate_adx(candles: Sequence[Dict[str, Any]], period: int = 14) -> float:
    if len(candles) <= period + 1:
        return 0.0
    highs = [candle["high"] for candle in candles]
    lows = [candle["low"] for candle in candles]
    closes = [candle["close"] for candle in candles]
    trs: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    for index in range(1, len(candles)):
        up_move = highs[index] - highs[index - 1]
        down_move = lows[index - 1] - lows[index]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        tr = max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        )
        trs.append(tr)
    atr = sum(trs[:period])
    plus = sum(plus_dm[:period])
    minus = sum(minus_dm[:period])
    if atr <= 0:
        return 0.0
    dx_values: List[float] = []
    for index in range(period, len(trs)):
        atr = atr - (atr / period) + trs[index]
        plus = plus - (plus / period) + plus_dm[index]
        minus = minus - (minus / period) + minus_dm[index]
        if atr <= 0:
            dx_values.append(0.0)
            continue
        plus_di = 100.0 * (plus / atr)
        minus_di = 100.0 * (minus / atr)
        dx = 100.0 * safe_div(abs(plus_di - minus_di), plus_di + minus_di)
        dx_values.append(dx)
    if not dx_values:
        return 0.0
    tail = dx_values[-period:] if len(dx_values) >= period else dx_values
    return round(sum(tail) / len(tail), 2)


def confirm_breakout_with_close(pair: str, direction: str, level: float) -> bool:
    if not REQUIRE_BREAKOUT_CLOSE:
        return True
    cfg = PAIR_CONFIGS[pair]
    recent = broker.get_candles(epics[pair], cfg.breakout_resolution, max_count=4)
    if not recent or len(recent) < 2:
        return False
    last_closed = recent[-1]
    close_price = last_closed["close"]
    return close_price > level if direction == "BUY" else close_price < level


def check_spread_ok(pair: str, live: Dict[str, float], rng: Dict[str, Any]) -> bool:
    cfg = PAIR_CONFIGS[pair]
    spread = live["spread"]
    range_cap = max(rng["size"] * MAX_SPREAD_TO_RANGE_PCT, cfg.pip_size)
    if spread <= cfg.max_spread and spread <= range_cap:
        return True
    logger.info("%s skipped: spread %.6f too wide for range %.6f", pair, spread, rng["size"])
    return False


def breakout_levels(pair: str, rng: Dict[str, Any]) -> Tuple[float, float]:
    cfg = PAIR_CONFIGS[pair]
    return rng["high"] + cfg.buffer, rng["low"] - cfg.buffer


def entry_direction(pair: str, live: Dict[str, float], rng: Dict[str, Any]) -> Optional[str]:
    buy_level, sell_level = breakout_levels(pair, rng)
    mid = live["mid"]
    if mid > buy_level:
        if pair == "US500" and rng.get("bias") == "BEAR":
            return None
        return "BUY"
    if mid < sell_level:
        if pair == "US500" and rng.get("bias") == "BULL":
            return None
        return "SELL"
    return None


def calculate_levels(pair: str, direction: str, entry: float, rng: Dict[str, Any]) -> Tuple[float, float, float]:
    cfg = PAIR_CONFIGS[pair]
    size = rng["size"]
    if direction == "BUY":
        sl = round_price(rng["low"] - cfg.buffer, cfg.decimals)
        tp1 = round_price(entry + size * cfg.tp1_mult, cfg.decimals)
        tp2 = round_price(entry + size * cfg.tp2_mult, cfg.decimals)
    else:
        sl = round_price(rng["high"] + cfg.buffer, cfg.decimals)
        tp1 = round_price(entry - size * cfg.tp1_mult, cfg.decimals)
        tp2 = round_price(entry - size * cfg.tp2_mult, cfg.decimals)
    return sl, tp1, tp2


def current_trade_date() -> str:
    return now_utc().date().isoformat()


def trade_window_open(pair: str, ref: Optional[datetime] = None) -> bool:
    window = get_session_window(pair, ref)
    if not window:
        return False
    current = ref or now_utc()
    return window.trade_start <= current < window.trade_end


def force_close_due(pair: str, ref: Optional[datetime] = None) -> bool:
    if not ENABLE_EOD_CLOSE:
        return False
    window = get_session_window(pair, ref)
    if not window:
        return False
    current = ref or now_utc()
    return current >= window.trade_end


def calculate_position_size(pair: str, entry: float, sl: float, equity: float) -> float:
    meta = instrument_meta_for_pair(pair)
    risk_cash = equity * (RISK_PERCENT / 100.0)
    stop_distance = abs(entry - sl)
    if stop_distance <= 0 or meta.point_value <= 0:
        return 0.0
    raw_size = risk_cash / (stop_distance * meta.point_value)
    rounded = round_size(raw_size, meta.size_step)
    clamped = clamp(rounded, meta.min_size, meta.max_size)
    return round_size(clamped, meta.size_step)


def compute_position_pnl(trade: Dict[str, Any], exit_price: float, size: Optional[float] = None) -> float:
    quantity = trade["remaining_size"] if size is None else size
    move = exit_price - trade["entry"] if trade["direction"] == "BUY" else trade["entry"] - exit_price
    return round(move * quantity * trade["point_value"], 2)


def compute_unrealized_pnl(trade: Dict[str, Any], live: Dict[str, float]) -> float:
    price = live["bid"] if trade["direction"] == "BUY" else live["ask"]
    return compute_position_pnl(trade, price)


def current_heat_pct(live_prices: Dict[str, Dict[str, float]], equity: float) -> float:
    if equity <= 0:
        return 0.0
    risk_cash = 0.0
    for trade in open_trades:
        if trade["status"] != "OPEN":
            continue
        stop_distance = abs(trade["entry"] - trade["sl"])
        risk_cash += stop_distance * trade["remaining_size"] * trade["point_value"]
    return (risk_cash / equity) * 100.0


def floating_drawdown_pct(trade_date: str, live_prices: Dict[str, Dict[str, float]], pair: Optional[str] = None) -> float:
    start_equity = get_day_start_equity(trade_date)
    if start_equity <= 0:
        return 0.0
    realized = get_pair_realized_pnl(trade_date, pair) if pair else get_daily_realized_pnl(trade_date)
    floating = 0.0
    for trade in open_trades:
        if trade["status"] != "OPEN":
            continue
        if pair and trade["pair"] != pair:
            continue
        live = live_prices.get(trade["pair"])
        if not live:
            continue
        floating += compute_unrealized_pnl(trade, live)
    pnl = realized + (floating if COUNT_FLOATING_DRAWDOWN else 0.0)
    return max((-pnl / start_equity) * 100.0, 0.0)


def loss_limits_hit(pair: str, live_prices: Dict[str, Dict[str, float]], trade_date: str) -> bool:
    overall_loss = floating_drawdown_pct(trade_date, live_prices)
    pair_loss = floating_drawdown_pct(trade_date, live_prices, pair)
    if overall_loss >= DAILY_LOSS_LIMIT_PCT:
        logger.warning("Overall daily loss limit hit: %.2f%%", overall_loss)
        return True
    if pair_loss >= PAIR_DAILY_LOSS_LIMIT_PCT:
        logger.warning("%s pair daily loss limit hit: %.2f%%", pair, pair_loss)
        return True
    return False


def correlation_blocked(pair: str) -> bool:
    if ALLOW_CORRELATED_TRADES:
        return False
    for group in CORRELATION_GROUPS:
        if pair not in group:
            continue
        for trade in open_trades:
            if trade["status"] == "OPEN" and trade["pair"] in group and trade["pair"] != pair:
                return True
    return False


def arb_adx_ok(pair: str) -> bool:
    cfg = PAIR_CONFIGS[pair]
    if cfg.strategy != "ARB" or cfg.adx_min <= 0:
        return True
    candles = broker.get_candles(epics[pair], "MINUTE_15", max_count=40)
    if not candles:
        return False
    adx = calculate_adx(candles)
    logger.info("%s ADX %.2f", pair, adx)
    return adx >= cfg.adx_min


def entry_distance_ok(pair: str, entry: float, direction: str, rng: Dict[str, Any]) -> bool:
    buy_level, sell_level = breakout_levels(pair, rng)
    trigger = buy_level if direction == "BUY" else sell_level
    distance = abs(entry - trigger)
    allowed = max(rng["size"] * MAX_ENTRY_DISTANCE_PCT, PAIR_CONFIGS[pair].pip_size * 3)
    if distance <= allowed:
        return True
    logger.info("%s skipped: entry distance %.6f too far from breakout %.6f", pair, distance, allowed)
    return False


def signal_for_pair(pair: str, live: Dict[str, float], trade_date: str) -> Optional[Dict[str, Any]]:
    if pair in blocked_pairs:
        return None
    if not trade_window_open(pair):
        return None
    if db_has_trade_today(pair, trade_date):
        return None
    if any(trade["pair"] == pair and trade["status"] == "OPEN" for trade in open_trades):
        return None
    if correlation_blocked(pair):
        return None
    rng = ranges.get(pair) or db_get_range(pair, trade_date)
    if not rng:
        return None
    ranges[pair] = rng
    if rng.get("disabled"):
        return None
    if is_near_news(pair):
        return None
    if not check_spread_ok(pair, live, rng):
        return None
    direction = entry_direction(pair, live, rng)
    if not direction:
        return None
    if PAIR_CONFIGS[pair].strategy == "ARB" and not arb_adx_ok(pair):
        return None
    if not confirm_breakout_with_close(pair, direction, breakout_levels(pair, rng)[0 if direction == "BUY" else 1]):
        return None
    entry = round_price(live["ask"] if direction == "BUY" else live["bid"], PAIR_CONFIGS[pair].decimals)
    if not entry_distance_ok(pair, entry, direction, rng):
        return None
    sl, tp1, tp2 = calculate_levels(pair, direction, entry, rng)
    return {
        "pair": pair,
        "strategy": PAIR_CONFIGS[pair].strategy,
        "epic": epics[pair],
        "direction": direction,
        "entry": entry,
        "requested_entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "range": rng,
    }


def pair_from_epic(epic: str) -> Optional[str]:
    for pair, mapped in epics.items():
        if mapped.upper() == epic.upper():
            return pair
    return None


def match_broker_position(trade: Dict[str, Any], broker_positions: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    identifiers = {trade.get("deal_id", ""), trade.get("deal_ref", "")}
    identifiers.discard("")
    for position in broker_positions:
        if position.get("dealId", "") in identifiers or position.get("dealReference", "") in identifiers:
            return position
    pair_epic = trade.get("epic", "")
    candidates = [
        position
        for position in broker_positions
        if position.get("epic", "").upper() == pair_epic.upper()
        and position.get("direction", "").upper() == trade["direction"].upper()
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def reconcile_state_with_broker():
    blocked_pairs.clear()
    blocked_pairs.update(db_load_review_pairs(current_trade_date()))
    if not broker:
        return
    broker_positions = broker.get_open_positions()
    local_keys: Set[str] = set()
    for trade in open_trades:
        match = match_broker_position(trade, broker_positions)
        if not match:
            trade["review_required"] = True
            trade["review_note"] = "Broker position missing at reconciliation"
            trade["status"] = "REVIEW"
            db_update_trade(trade)
            blocked_pairs.add(trade["pair"])
            logger.warning("Trade moved to REVIEW: %s %s", trade["pair"], trade.get("deal_ref", ""))
            send_telegram(
                f"Review needed for {trade['pair']} {trade['direction']}\n"
                f"No matching broker position was found at startup."
            )
            continue
        if match.get("dealId"):
            trade["deal_id"] = match["dealId"]
            local_keys.add(match["dealId"])
        if match.get("dealReference"):
            trade["deal_ref"] = match["dealReference"]
            local_keys.add(match["dealReference"])
        if match.get("stopLevel") is not None:
            trade["broker_sl"] = float(match["stopLevel"])
        db_update_trade(trade)
    for position in broker_positions:
        if position.get("dealId") in local_keys or position.get("dealReference") in local_keys:
            continue
        pair = pair_from_epic(position.get("epic", ""))
        if pair:
            blocked_pairs.add(pair)
        logger.warning(
            "Untracked broker position detected: %s %s size=%s",
            pair or position.get("epic", "UNKNOWN"),
            position.get("direction", ""),
            position.get("size", 0),
        )
        send_telegram(
            f"Untracked broker position detected\n"
            f"Instrument: {pair or position.get('epic', 'UNKNOWN')}\n"
            f"Direction: {position.get('direction', '')}\n"
            f"Size: {position.get('size', 0)}\n"
            f"This pair is blocked until the state is reviewed."
        )
    open_trades[:] = [trade for trade in open_trades if trade["status"] == "OPEN"]


def classify_result(total_pnl: float, initial_risk_cash: float) -> Tuple[str, float]:
    profit_r = round(safe_div(total_pnl, initial_risk_cash), 2)
    if profit_r > 0.10:
        return "WIN", profit_r
    if profit_r < -0.10:
        return "LOSS", profit_r
    return "BE", profit_r


def export_result(trade: Dict[str, Any]):
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            [
                now_utc().isoformat(),
                trade["pair"],
                trade["strategy"],
                trade["trade_date"],
                trade["direction"],
                trade["entry"],
                trade["initial_sl"],
                trade["sl"],
                trade["tp1"],
                trade["tp2"],
                trade.get("exit_price"),
                trade.get("result", ""),
                trade.get("profit_r", 0),
                trade.get("pnl", 0),
                trade.get("partial_pnl", 0),
                trade["range_size"],
                trade["point_value"],
                trade["lot_size"],
                int(trade.get("partial_done", False)),
                int(trade.get("tp1_locked", False)),
                int(trade.get("tp2_locked", False)),
                trade.get("exit_reason", ""),
            ]
        )


def update_broker_sl(trade: Dict[str, Any], new_sl: float) -> bool:
    meta = instrument_meta_for_pair(trade["pair"])
    identifier = trade.get("deal_id") or trade.get("deal_ref", "")
    success = broker.update_sl(
        identifier,
        new_sl,
        meta.decimals,
        epic_fallback=trade["epic"],
        direction=trade["direction"],
    )
    if success:
        trade["broker_sl"] = round_price(new_sl, meta.decimals)
        db_update_trade(trade)
    else:
        logger.warning("Broker SL update failed for %s", trade["pair"])
    return success


def finalize_trade(
    trade: Dict[str, Any],
    exit_price: float,
    reason: str,
    broker_close: bool = True,
):
    if trade["status"] != "OPEN":
        return
    if broker_close and broker:
        identifier = trade.get("deal_id") or trade.get("deal_ref", "")
        closed = broker.close_position(
            identifier,
            epic_fallback=trade["epic"],
            size=trade["remaining_size"],
            direction=trade["direction"],
        )
        if not closed:
            logger.error("Close failed for %s", trade["pair"])
            send_telegram(
                f"Close failed for {trade['pair']} {trade['direction']}\n"
                f"Please check Capital.com manually."
            )
            return
    remaining_pnl = compute_position_pnl(trade, exit_price, size=trade["remaining_size"])
    total_pnl = round(trade["partial_pnl"] + remaining_pnl, 2)
    result, profit_r = classify_result(total_pnl, trade["initial_risk_cash"])
    trade.update(
        {
            "status": "CLOSED",
            "result": result,
            "exit_price": exit_price,
            "exit_reason": reason,
            "closed_at": time.time(),
            "pnl": total_pnl,
            "profit_r": profit_r,
            "closed_size": trade["initial_size"],
            "remaining_size": 0.0,
        }
    )
    db_update_trade(trade)
    db_log_event(trade["id"], "CLOSE", exit_price, reason)
    export_result(trade)
    record_closed_trade_pnl(trade["trade_date"], trade["pair"], result, total_pnl)
    logger.info(
        "Closed %s %s @ %.5f -> %s (%s %.2fR)",
        trade["pair"],
        trade["direction"],
        exit_price,
        result,
        reason,
        profit_r,
    )
    emoji = {"WIN": "WIN", "LOSS": "LOSS", "BE": "BE"}[result]
    send_telegram(
        f"{emoji} | {trade['pair']} {trade['direction']}\n"
        f"Strategy: {trade['strategy']}\n"
        f"Exit: {exit_price} | PnL: {total_pnl} | {profit_r}R\n"
        f"Reason: {reason}"
    )


def handle_tp1(trade: Dict[str, Any], live: Dict[str, float]):
    if trade.get("tp1_locked"):
        return
    meta = instrument_meta_for_pair(trade["pair"])
    partial_qty = 0.0
    if TP1_PARTIAL_CLOSE_PCT > 0:
        partial_qty = round_size(trade["initial_size"] * TP1_PARTIAL_CLOSE_PCT, meta.size_step)
        if partial_qty >= trade["remaining_size"]:
            partial_qty = round_size(trade["remaining_size"] - meta.min_size, meta.size_step)
    if partial_qty > 0 and partial_qty >= meta.min_size and broker:
        identifier = trade.get("deal_id") or trade.get("deal_ref", "")
        closed = broker.close_position(
            identifier,
            epic_fallback=trade["epic"],
            size=partial_qty,
            direction=trade["direction"],
        )
        if closed:
            trade["partial_pnl"] = round(
                trade["partial_pnl"] + compute_position_pnl(trade, trade["tp1"], size=partial_qty),
                2,
            )
            trade["closed_size"] = round(trade["closed_size"] + partial_qty, 6)
            trade["remaining_size"] = round(max(trade["remaining_size"] - partial_qty, 0.0), 6)
            trade["partial_done"] = True
            db_log_event(trade["id"], "TP1_PARTIAL", trade["tp1"], f"size={partial_qty}")
        else:
            logger.warning("%s TP1 partial close failed", trade["pair"])
            send_telegram(f"TP1 partial close failed for {trade['pair']} {trade['direction']}")
            return
    new_sl = trade["entry"] if TP1_SL_MODE == "BREAKEVEN" else trade["tp1"]
    trade["sl"] = round_price(new_sl, meta.decimals)
    trade["tp1_locked"] = True
    trade["break_even_done"] = TP1_SL_MODE == "BREAKEVEN"
    db_update_trade(trade)
    update_broker_sl(trade, trade["sl"])
    send_telegram(
        f"TP1 reached | {trade['pair']} {trade['direction']}\n"
        f"Partial closed: {trade['partial_done']}\n"
        f"New SL: {trade['sl']} | Remaining size: {trade['remaining_size']}\n"
        f"Targeting TP2: {trade['tp2']}"
    )


def handle_tp2(trade: Dict[str, Any], live: Dict[str, float]):
    if TP2_MODE == "TRAIL":
        meta = instrument_meta_for_pair(trade["pair"])
        if trade["direction"] == "BUY":
            trade["sl"] = max(trade["sl"], trade["tp1"])
        else:
            trade["sl"] = min(trade["sl"], trade["tp1"])
        trade["sl"] = round_price(trade["sl"], meta.decimals)
        trade["tp2_locked"] = True
        db_update_trade(trade)
        update_broker_sl(trade, trade["sl"])
        send_telegram(
            f"TP2 trail armed | {trade['pair']} {trade['direction']}\n"
            f"Trailing stop now: {trade['sl']}"
        )
        return
    price = max(live["bid"], trade["tp2"]) if trade["direction"] == "BUY" else min(live["ask"], trade["tp2"])
    finalize_trade(trade, round_price(price, PAIR_CONFIGS[trade["pair"]].decimals), "TP2", broker_close=True)


def check_open_trades(live_prices: Dict[str, Dict[str, float]]):
    for trade in list(open_trades):
        if trade["status"] != "OPEN":
            continue
        live = live_prices.get(trade["pair"])
        if not live:
            continue
        price = live["bid"] if trade["direction"] == "BUY" else live["ask"]
        stop_hit = (trade["direction"] == "BUY" and price <= trade["sl"]) or (
            trade["direction"] == "SELL" and price >= trade["sl"]
        )
        if stop_hit:
            exit_price = round_price(trade["sl"], PAIR_CONFIGS[trade["pair"]].decimals)
            finalize_trade(trade, exit_price, "SL", broker_close=True)
            continue
        tp1_hit = (trade["direction"] == "BUY" and price >= trade["tp1"]) or (
            trade["direction"] == "SELL" and price <= trade["tp1"]
        )
        if tp1_hit and not trade.get("tp1_locked"):
            handle_tp1(trade, live)
        tp2_hit = (trade["direction"] == "BUY" and price >= trade["tp2"]) or (
            trade["direction"] == "SELL" and price <= trade["tp2"]
        )
        if tp2_hit:
            handle_tp2(trade, live)
            continue
        if force_close_due(trade["pair"]):
            exit_now = round_price(price, PAIR_CONFIGS[trade["pair"]].decimals)
            finalize_trade(trade, exit_now, "EOD_CLOSE", broker_close=True)
    open_trades[:] = [trade for trade in open_trades if trade["status"] == "OPEN"]


def open_trade(signal: Dict[str, Any], live_prices: Dict[str, Dict[str, float]]):
    trade_date = signal["range"]["trade_date"]
    if loss_limits_hit(signal["pair"], live_prices, trade_date):
        return
    balance = broker.get_account_balance() if broker else INITIAL_EQUITY
    if balance <= 0:
        balance = INITIAL_EQUITY
    heat = current_heat_pct(live_prices, balance)
    if heat >= MAX_PORTFOLIO_HEAT_PCT:
        logger.warning("Portfolio heat limit hit: %.2f%%", heat)
        return
    if sum(1 for trade in open_trades if trade["status"] == "OPEN") >= MAX_ACTIVE_TRADES:
        logger.warning("Max active trades reached")
        return
    meta = instrument_meta_for_pair(signal["pair"])
    lot_size = calculate_position_size(signal["pair"], signal["entry"], signal["sl"], balance)
    if lot_size < meta.min_size:
        logger.warning("%s position size below min size", signal["pair"])
        return
    order_ref = broker.place_order(
        signal["epic"],
        signal["direction"],
        lot_size,
        signal["sl"],
        meta.decimals,
        force_open=True,
    )
    if not order_ref:
        send_telegram(f"Order failed for {signal['pair']} {signal['direction']}")
        return
    fill = broker.confirm_fill(order_ref) or {}
    trade = {
        "pair": signal["pair"],
        "strategy": signal["strategy"],
        "trade_date": trade_date,
        "session_date": signal["range"]["session_date"],
        "direction": signal["direction"],
        "entry": signal["entry"],
        "requested_entry": signal["requested_entry"],
        "sl": signal["sl"],
        "initial_sl": signal["sl"],
        "broker_sl": signal["sl"],
        "tp1": signal["tp1"],
        "tp2": signal["tp2"],
        "range_high": signal["range"]["high"],
        "range_low": signal["range"]["low"],
        "range_size": signal["range"]["size"],
        "lot_size": lot_size,
        "initial_size": lot_size,
        "remaining_size": lot_size,
        "closed_size": 0.0,
        "partial_pnl": 0.0,
        "risk_per_unit": abs(signal["entry"] - signal["sl"]),
        "point_value": meta.point_value,
        "initial_risk_cash": abs(signal["entry"] - signal["sl"]) * lot_size * meta.point_value,
        "deal_ref": order_ref,
        "deal_id": fill.get("opened_deal_id", ""),
        "epic": signal["epic"],
        "status": "OPEN",
        "result": "",
        "exit_price": None,
        "exit_reason": "",
        "opened_at": time.time(),
        "closed_at": None,
        "pnl": 0.0,
        "profit_r": 0.0,
        "tp1_locked": False,
        "tp2_locked": False,
        "partial_done": False,
        "break_even_done": False,
        "review_required": False,
        "review_note": "",
        "broker_fill_price": fill.get("level") or signal["entry"],
        "broker_partial_ref": "",
    }
    trade["id"] = db_save_trade(trade)
    db_log_event(trade["id"], "OPEN", trade["entry"], f"size={lot_size}")
    open_trades.append(trade)
    send_telegram(
        f"NEW TRADE | {trade['pair']} {trade['direction']} [{trade['strategy']}]\n"
        f"Entry: {trade['entry']} | SL: {trade['sl']}\n"
        f"TP1: {trade['tp1']} | TP2: {trade['tp2']}\n"
        f"Size: {lot_size} | Risk cash: {round(trade['initial_risk_cash'], 2)}"
    )


def refresh_range_cache():
    ranges.clear()
    trade_date = current_trade_date()
    for pair in PAIR_CONFIGS:
        stored = db_get_range(pair, trade_date)
        if stored:
            ranges[pair] = stored


def load_state():
    open_trades.clear()
    open_trades.extend(db_load_open_trades())
    refresh_range_cache()
    blocked_pairs.clear()
    blocked_pairs.update(db_load_review_pairs(current_trade_date()))
    logger.info("Loaded %s open trade(s)", len(open_trades))


def resolve_epics_and_metadata():
    global account_currency
    account_info = broker.get_account_info()
    account_currency = str(account_info.get("currency", "") or "")
    for pair, cfg in PAIR_CONFIGS.items():
        epic = broker.get_epic(cfg.search)
        if not epic:
            raise RuntimeError(f"Could not resolve epic for {pair}")
        epics[pair] = epic
        meta = instrument_meta_for_pair(pair)
        logger.info(
            "Resolved %s -> %s | point_value=%s min=%s step=%s",
            pair,
            epic,
            meta.point_value,
            meta.min_size,
            meta.size_step,
        )


def performance_summary() -> str:
    rows = db_closed_trade_rows()
    if not rows:
        return "No closed trades yet."

    def build_stats(filtered: Iterable[sqlite3.Row]) -> Tuple[int, float, float, float, float]:
        items = list(filtered)
        if not items:
            return 0, 0.0, 0.0, 0.0, 0.0
        wins = [float(item["pnl"]) for item in items if float(item["pnl"]) > 0]
        losses = [float(item["pnl"]) for item in items if float(item["pnl"]) < 0]
        r_wins = [float(item["profit_r"]) for item in items if float(item["profit_r"]) > 0]
        r_losses = [abs(float(item["profit_r"])) for item in items if float(item["profit_r"]) < 0]
        total = len(items)
        win_rate = (len(wins) / total) * 100.0
        profit_factor = safe_div(sum(wins), abs(sum(losses)))
        avg_win_r = sum(r_wins) / len(r_wins) if r_wins else 0.0
        avg_loss_r = sum(r_losses) / len(r_losses) if r_losses else 0.0
        expectancy = (win_rate / 100.0) * avg_win_r - ((100.0 - win_rate) / 100.0) * avg_loss_r
        return total, win_rate, profit_factor, avg_win_r, avg_loss_r + expectancy * 0

    overall_total = len(rows)
    gross_win = sum(float(row["pnl"]) for row in rows if float(row["pnl"]) > 0)
    gross_loss = sum(float(row["pnl"]) for row in rows if float(row["pnl"]) < 0)
    win_count = sum(1 for row in rows if row["result"] == "WIN")
    loss_count = sum(1 for row in rows if row["result"] == "LOSS")
    win_rate = (win_count / overall_total) * 100.0 if overall_total else 0.0
    profit_factor = safe_div(gross_win, abs(gross_loss))
    r_wins = [float(row["profit_r"]) for row in rows if float(row["profit_r"]) > 0]
    r_losses = [abs(float(row["profit_r"])) for row in rows if float(row["profit_r"]) < 0]
    avg_win_r = sum(r_wins) / len(r_wins) if r_wins else 0.0
    avg_loss_r = sum(r_losses) / len(r_losses) if r_losses else 0.0
    expectancy = (win_rate / 100.0) * avg_win_r - ((100.0 - win_rate) / 100.0) * avg_loss_r

    lines = [
        f"Overall | trades={overall_total} WR={win_rate:.1f}% PF={profit_factor:.2f} AvgWinR={avg_win_r:.2f} AvgLossR={avg_loss_r:.2f} Exp={expectancy:.2f}R",
    ]
    for strategy in sorted({row["strategy"] for row in rows}):
        items = [row for row in rows if row["strategy"] == strategy]
        total = len(items)
        s_wins = sum(1 for row in items if row["result"] == "WIN")
        s_rate = (s_wins / total) * 100.0 if total else 0.0
        s_gross_win = sum(float(row["pnl"]) for row in items if float(row["pnl"]) > 0)
        s_gross_loss = sum(float(row["pnl"]) for row in items if float(row["pnl"]) < 0)
        s_pf = safe_div(s_gross_win, abs(s_gross_loss))
        s_r_wins = [float(row["profit_r"]) for row in items if float(row["profit_r"]) > 0]
        s_r_losses = [abs(float(row["profit_r"])) for row in items if float(row["profit_r"]) < 0]
        s_avg_win = sum(s_r_wins) / len(s_r_wins) if s_r_wins else 0.0
        s_avg_loss = sum(s_r_losses) / len(s_r_losses) if s_r_losses else 0.0
        s_exp = (s_rate / 100.0) * s_avg_win - ((100.0 - s_rate) / 100.0) * s_avg_loss
        lines.append(
            f"{strategy} | trades={total} WR={s_rate:.1f}% PF={s_pf:.2f} AvgWinR={s_avg_win:.2f} AvgLossR={s_avg_loss:.2f} Exp={s_exp:.2f}R"
        )
    for pair in PAIR_CONFIGS:
        items = [row for row in rows if row["pair"] == pair]
        if not items:
            continue
        p_total = len(items)
        p_wins = sum(1 for row in items if row["result"] == "WIN")
        p_rate = (p_wins / p_total) * 100.0
        p_gross_win = sum(float(row["pnl"]) for row in items if float(row["pnl"]) > 0)
        p_gross_loss = sum(float(row["pnl"]) for row in items if float(row["pnl"]) < 0)
        p_pf = safe_div(p_gross_win, abs(p_gross_loss))
        p_r_wins = [float(row["profit_r"]) for row in items if float(row["profit_r"]) > 0]
        p_r_losses = [abs(float(row["profit_r"])) for row in items if float(row["profit_r"]) < 0]
        p_avg_win = sum(p_r_wins) / len(p_r_wins) if p_r_wins else 0.0
        p_avg_loss = sum(p_r_losses) / len(p_r_losses) if p_r_losses else 0.0
        p_exp = (p_rate / 100.0) * p_avg_win - ((100.0 - p_rate) / 100.0) * p_avg_loss
        lines.append(
            f"{pair} | trades={p_total} WR={p_rate:.1f}% PF={p_pf:.2f} AvgWinR={p_avg_win:.2f} AvgLossR={p_avg_loss:.2f} Exp={p_exp:.2f}R"
        )
    return "\n".join(lines)


def send_heartbeat(live_prices: Dict[str, Dict[str, float]]):
    balance = broker.get_account_balance() if broker else INITIAL_EQUITY
    trade_date = current_trade_date()
    overall_loss = floating_drawdown_pct(trade_date, live_prices)
    heat = current_heat_pct(live_prices, balance if balance > 0 else INITIAL_EQUITY)
    range_lines = []
    for pair in PAIR_CONFIGS:
        rng = ranges.get(pair)
        if rng:
            range_lines.append(
                f"{pair}: {rng['low']}->{rng['high']} size={rng['size']} disabled={int(rng.get('disabled', False))}"
            )
        else:
            range_lines.append(f"{pair}: range not built")
    send_telegram(
        f"Multi-Strategy Bot\n"
        f"Mode: {'DEMO' if CAPITAL_DEMO else 'LIVE'} | Balance: {balance:.2f} {account_currency}\n"
        f"Open trades: {len(open_trades)} | Heat: {heat:.2f}% | Daily DD: {overall_loss:.2f}%/{DAILY_LOSS_LIMIT_PCT}%\n"
        f"Blocked pairs: {', '.join(sorted(blocked_pairs)) or 'none'}\n"
        f"Ranges:\n" + "\n".join(range_lines)
    )


def send_report():
    send_telegram("Performance report\n" + performance_summary())


def startup_report(start_balance: float):
    pair_lines = []
    for pair in PAIR_CONFIGS:
        meta = market_meta[pair]
        pair_lines.append(
            f"{pair} -> {epics[pair]} | point={meta.point_value:.6f} | min={meta.min_size} | step={meta.size_step}"
        )
    send_telegram(
        f"Bot started | {'DEMO' if CAPITAL_DEMO else 'LIVE'}\n"
        f"Balance: {start_balance:.2f} {account_currency}\n"
        f"Risk: {RISK_PERCENT}% | Overall daily limit: {DAILY_LOSS_LIMIT_PCT}% | Pair daily limit: {PAIR_DAILY_LOSS_LIMIT_PCT}%\n"
        f"TP1 partial: {TP1_PARTIAL_CLOSE_PCT * 100:.0f}% | TP1 stop mode: {TP1_SL_MODE} | TP2 mode: {TP2_MODE}\n"
        f"Breakout close confirmation: {REQUIRE_BREAKOUT_CLOSE}\n"
        f"Pairs:\n" + "\n".join(pair_lines)
    )


def build_ready_ranges():
    for pair in PAIR_CONFIGS:
        window = get_session_window(pair)
        if not window:
            continue
        if now_utc() < window.range_end:
            continue
        if pair in ranges and ranges[pair]["trade_date"] == window.trade_date:
            continue
        built = build_range_for_pair(pair)
        if built:
            ranges[pair] = built
            if built.get("disabled"):
                logger.info("%s disabled for %s: %s", pair, built["trade_date"], built.get("skip_reason", ""))


def fetch_live_prices() -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for pair, epic in epics.items():
        live = broker.get_live_price(epic)
        if live:
            result[pair] = live
        time.sleep(0.2)
    return result


def maybe_reset_day():
    global last_trade_day
    today = current_trade_date()
    if today == last_trade_day:
        return
    last_trade_day = today
    balance = broker.get_account_balance() if broker else INITIAL_EQUITY
    if balance <= 0:
        balance = INITIAL_EQUITY
    init_daily_pnl(today, balance)
    refresh_range_cache()
    blocked_pairs.update(db_load_review_pairs(today))
    logger.info("New trading day initialised: %s", today)


def main():
    global broker, last_trade_day

    logger.info("=" * 72)
    logger.info("Multi-Strategy Professional Trading Bot - Fixed")
    logger.info("=" * 72)

    ensure_csv(RESULTS_FILE, RESULT_HEADERS)
    validate_runtime_config()
    init_db()
    load_state()

    broker = CapitalClient(
        api_key=CAPITAL_API_KEY,
        login=CAPITAL_LOGIN,
        password=CAPITAL_PASSWORD,
        demo=CAPITAL_DEMO,
    )
    resolve_epics_and_metadata()

    start_balance = broker.get_account_balance()
    if start_balance <= 0:
        start_balance = INITIAL_EQUITY
    last_trade_day = current_trade_date()
    init_daily_pnl(last_trade_day, start_balance)
    reconcile_state_with_broker()
    startup_report(start_balance)

    last_heartbeat = 0.0
    last_report = 0.0

    while True:
        try:
            maybe_reset_day()
            build_ready_ranges()
            live_prices = fetch_live_prices()
            if open_trades:
                check_open_trades(live_prices)
            trade_date = current_trade_date()
            for pair, live in live_prices.items():
                signal = signal_for_pair(pair, live, trade_date)
                if signal:
                    open_trade(signal, live_prices)
            now_ts = time.time()
            if now_ts - last_heartbeat >= HEARTBEAT_SECS:
                send_heartbeat(live_prices)
                last_heartbeat = now_ts
            if now_ts - last_report >= REPORT_SECS:
                send_report()
                last_report = now_ts
            time.sleep(SCAN_INTERVAL)
        except RuntimeError as exc:
            logger.critical("Bot halted: %s", exc)
            send_telegram(f"Bot halted: {exc}")
            break
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            send_telegram("Bot stopped manually")
            break
        except Exception as exc:
            logger.error("Main loop error: %s", exc)
            time.sleep(30)


if __name__ == "__main__":
    main()
