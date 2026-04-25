import csv
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

# -------------------- Configuration --------------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY", "").strip()
CAPITAL_LOGIN = os.getenv("CAPITAL_LOGIN", "").strip()
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD", "").strip()

# Strategy Parameters
INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", "1000.0"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "2.0"))
MAX_ACTIVE_TRADES = int(os.getenv("MAX_ACTIVE_TRADES", "10"))
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "1.5"))
ATR_TARGET_MULTIPLIER = float(os.getenv("ATR_TARGET_MULTIPLIER", "2.4"))
BREAK_EVEN_TRIGGER_R = float(os.getenv("BREAK_EVEN_TRIGGER_R", "1.0"))
TRAILING_STOP_ATR_MULTIPLIER = float(os.getenv("TRAILING_STOP_ATR_MULTIPLIER", "1.2"))
MAX_SPREAD_TO_ATR_RATIO = float(os.getenv("MAX_SPREAD_TO_ATR_RATIO", "0.20"))
PAIR_COOLDOWN_SECONDS = int(os.getenv("PAIR_COOLDOWN_SECONDS", "14400"))
MAX_AUTH_RETRIES = int(os.getenv("MAX_AUTH_RETRIES", "5"))  # TWEAK: limit auth retry loops

# Timing
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "900"))
TRADE_CHECK_INTERVAL_SECONDS = int(os.getenv("TRADE_CHECK_INTERVAL_SECONDS", "20"))
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))
REPORT_INTERVAL_SECONDS = int(os.getenv("REPORT_INTERVAL_SECONDS", "3600"))

DB_FILE = "trade_state.db"
RESULTS_FILE = "trade_results.csv"
LOG_FILE = "bot.log"

pairs = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD",
    "NZDUSD": "NZDUSD",
    "EURGBP": "EURGBP",
    "EURJPY": "EURJPY",
    "GBPJPY": "GBPJPY",
}

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- SQLite State Manager --------------------
# TWEAK: Replaced CSV state file with SQLite to prevent concurrent write corruption

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            epic TEXT,
            type TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            status TEXT DEFAULT 'OPEN',
            opened_at REAL,
            risk_per_unit REAL,
            break_even_done INTEGER DEFAULT 0,
            entry_atr REAL,
            lot_size REAL,
            deal_ref TEXT
        )
    """)
    conn.commit()
    conn.close()

def db_save_trade(trade: Dict) -> int:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO trades (pair, epic, type, entry, sl, tp, status, opened_at,
            risk_per_unit, break_even_done, entry_atr, lot_size, deal_ref)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade["pair"], trade["epic"], trade["type"], trade["entry"], trade["sl"],
        trade["tp"], trade["status"], trade["opened_at"], trade["risk_per_unit"],
        int(trade["break_even_done"]), trade["entry_atr"], trade["lot_size"],
        trade.get("deal_ref", "")
    ))
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id

def db_update_trade(trade: Dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        UPDATE trades SET sl=?, tp=?, status=?, break_even_done=?
        WHERE id=?
    """, (trade["sl"], trade["tp"], trade["status"], int(trade["break_even_done"]), trade["db_id"]))
    conn.commit()
    conn.close()

def db_load_open_trades() -> List[Dict]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM trades WHERE status='OPEN'")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

# -------------------- Capital.com API Client --------------------
class CapitalClient:
    def __init__(self, api_key: str, login: str, password: str, demo: bool = True):
        self.api_key = api_key
        self.login = login
        self.password = password
        self.base_url = (
            "https://demo-api-capital.backend-capital.com" if demo
            else "https://api-capital.backend-capital.com"
        )
        self.cst = None
        self.security_token = None
        self.session = requests.Session()
        self.epic_cache: Dict[str, str] = {}
        self._auth_retries = 0  # TWEAK: track retries to prevent infinite loop
        self.authenticate()

    def authenticate(self):
        # TWEAK: Cap authentication retries to prevent infinite looping
        if self._auth_retries >= MAX_AUTH_RETRIES:
            raise RuntimeError(f"Authentication failed after {MAX_AUTH_RETRIES} retries. Halting.")

        try:
            url = f"{self.base_url}/api/v1/session"
            headers = {
                "X-CAP-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            data = {
                "identifier": self.login,
                "password": self.password,
                "encryptedPassword": False
            }
            response = self.session.post(url, headers=headers, json=data, timeout=30)

            if response.status_code != 200:
                error_msg = response.json().get("errorMessage", "Unknown error")
                raise Exception(f"Authentication failed: {error_msg}")

            self.cst = response.headers.get("CST")
            self.security_token = response.headers.get("X-SECURITY-TOKEN")
            self._auth_retries = 0  # reset on success
            logger.info("✅ Connected to Capital.com")

        except Exception as e:
            self._auth_retries += 1
            logger.error(f"Auth attempt {self._auth_retries}/{MAX_AUTH_RETRIES} failed: {e}")
            if self._auth_retries < MAX_AUTH_RETRIES:
                time.sleep(5 * self._auth_retries)
                self.authenticate()
            else:
                raise

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Optional[Dict]:
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {
                "X-CAP-API-KEY": self.api_key,
                "CST": self.cst,
                "X-SECURITY-TOKEN": self.security_token,
                "Content-Type": "application/json"
            }

            if method == "GET":
                response = self.session.get(url, headers=headers, timeout=30)
            elif method == "POST":
                response = self.session.post(url, headers=headers, json=data, timeout=30)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers, timeout=30)
            else:
                return None

            # TWEAK: re-auth only once per call (not recursive without limit)
            if response.status_code in [401, 403]:
                logger.info("Session expired, re-authenticating...")
                self.authenticate()
                # Retry once after re-auth
                return self._make_request(method, endpoint, data)

            if response.status_code != 200:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None

            return response.json()

        except RuntimeError:
            raise  # propagate auth halt
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def get_epic(self, search_term: str) -> Optional[str]:
        if search_term in self.epic_cache:
            return self.epic_cache[search_term]
        try:
            data = self._make_request("GET", f"/api/v1/markets?searchTerm={search_term}")
            if data and data.get("markets"):
                epic = data["markets"][0]["epic"]
                self.epic_cache[search_term] = epic
                logger.info(f"Found epic '{epic}' for '{search_term}'")
                return epic
            logger.warning(f"No market found for '{search_term}'")
            return None
        except Exception as e:
            logger.error(f"Failed to get epic for '{search_term}': {e}")
            return None

    def get_candles(self, epic: str, resolution: str = "MINUTE_15", num_candles: int = 300) -> Optional[pd.DataFrame]:
        try:
            data = self._make_request("GET", f"/api/v1/prices/{epic}?resolution={resolution}&max={num_candles}")
            if not data or "prices" not in data:
                return None

            rows = [{
                "time": c["snapshotTime"],
                "Open": float(c["openPrice"]["bid"]),
                "High": float(c["highPrice"]["bid"]),
                "Low": float(c["lowPrice"]["bid"]),
                "Close": float(c["closePrice"]["bid"]),
            } for c in data["prices"]]

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            return df.dropna()

        except Exception as e:
            logger.error(f"Failed to get candles for {epic}: {e}")
            return None

    def get_live_price(self, epic: str) -> Optional[Dict]:
        try:
            data = self._make_request("GET", f"/api/v1/markets/{epic}")
            if not data or "snapshot" not in data:
                return None

            snap = data["snapshot"]
            bid = float(snap["bid"])
            ask = float(snap["offer"])
            return {
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2,
                "spread": round(ask - bid, 5),
                "tradeable": True,
            }
        except Exception as e:
            logger.error(f"Failed to get live price for {epic}: {e}")
            return None

    def place_order(self, epic: str, order_type: str, units: float,
                    entry: float, sl: float, tp: float) -> Optional[str]:
        try:
            direction = "BUY" if order_type == "BUY" else "SELL"
            pip_size = 0.01 if "JPY" in epic else 0.0001

            stop_distance = int(abs(entry - sl) / pip_size)
            limit_distance = int(abs(tp - entry) / pip_size)

            data = {
                "epic": epic,
                "direction": direction,
                "size": units,
                "stopDistance": max(stop_distance, 1),
                "limitDistance": max(limit_distance, 1),
                "guaranteedStop": False,
                "forceOpen": True
            }

            result = self._make_request("POST", "/api/v1/positions", data)
            if result and "dealReference" in result:
                logger.info(f"Order placed: {order_type} {units} {epic}")
                return result["dealReference"]
            return None

        except Exception as e:
            logger.error(f"Failed to place order for {epic}: {e}")
            return None

    # TWEAK: Added confirm_fill to verify order actually executed
    def confirm_fill(self, deal_ref: str) -> Optional[Dict]:
        """Confirm the deal was filled and return fill details."""
        try:
            data = self._make_request("GET", f"/api/v1/confirms/{deal_ref}")
            if data and data.get("status") in ["OPEN", "ACCEPTED"]:
                return data
            logger.warning(f"Deal {deal_ref} not confirmed: {data}")
            return None
        except Exception as e:
            logger.error(f"Failed to confirm fill for {deal_ref}: {e}")
            return None

    def get_account_balance(self) -> float:
        try:
            data = self._make_request("GET", "/api/v1/accounts")
            if data and data.get("accounts"):
                return float(data["accounts"][0]["balance"]["balance"])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def close_position(self, deal_ref: str) -> bool:
        """Close a position by deal reference."""
        try:
            result = self._make_request("DELETE", f"/api/v1/positions/{deal_ref}")
            return result is not None
        except Exception as e:
            logger.error(f"Failed to close position {deal_ref}: {e}")
            return False

# -------------------- Helper Functions --------------------
def ensure_csv(path: str, headers: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

def setup_files():
    init_db()
    ensure_csv(RESULTS_FILE, ["timestamp", "pair", "type", "entry", "sl", "tp",
                               "exit_price", "status", "profit_r", "pnl"])

def is_valid_number(v):
    return v is not None and not math.isnan(v) and math.isfinite(v)

def now_utc():
    return datetime.now(timezone.utc)

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
        logger.error(f"Telegram error: {e}")

# -------------------- Technical Indicators --------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = (
        (high - low)
        .combine((high - prev_close).abs(), max)
        .combine((low - prev_close).abs(), max)
    )
    return tr.ewm(alpha=1 / period, adjust=False).mean()

# -------------------- Session Filter --------------------
# TWEAK: Expanded to include NY afternoon session (17:00–20:00 UTC)
def in_optimal_session() -> bool:
    now = now_utc()
    if now.weekday() >= 5:  # Skip weekends
        return False
    hour = now.hour
    # Asian (00:00–03:00), London (07:00–12:00), NY overlap (12:00–17:00), NY afternoon (17:00–20:00)
    return 0 <= hour < 20

# -------------------- Global State --------------------
trades: List[Dict] = []
wins = losses = 0
last_trade_times: Dict[str, float] = {}
broker: Optional[CapitalClient] = None

# -------------------- State Persistence --------------------
def load_state():
    global trades, last_trade_times, wins, losses
    open_trades = db_load_open_trades()
    for row in open_trades:
        row["break_even_done"] = bool(row["break_even_done"])
        trades.append(row)
        last_trade_times[row["pair"]] = row["opened_at"]

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            for row in csv.DictReader(f):
                if row["status"] == "WIN":
                    wins += 1
                elif row["status"] == "LOSS":
                    losses += 1

def save_trade_result(trade: Dict, status: str, exit_price: float):
    global wins, losses
    risk = trade["risk_per_unit"]
    if trade["type"] == "BUY":
        profit = (exit_price - trade["entry"]) * trade["lot_size"]
        profit_r = (exit_price - trade["entry"]) / risk if risk else 0
    else:
        profit = (trade["entry"] - exit_price) * trade["lot_size"]
        profit_r = (trade["entry"] - exit_price) / risk if risk else 0

    with open(RESULTS_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            now_utc().isoformat(), trade["pair"], trade["type"],
            round(trade["entry"], 5), round(trade["sl"], 5), round(trade["tp"], 5),
            round(exit_price, 5), status, round(profit_r, 2), round(profit, 2)
        ])

    if status == "WIN":
        wins += 1
    else:
        losses += 1

    emoji = "✅" if status == "WIN" else "❌"
    send_telegram(
        f"{emoji} {status} | {trade['pair']} {trade['type']}\n"
        f"Exit: {round(exit_price, 5)} | PnL: {round(profit, 2)} | {round(profit_r, 2)}R"
    )

# -------------------- Trade Management --------------------
def active_trade_count():
    return sum(1 for t in trades if t["status"] == "OPEN")

def has_open_trade(pair: str) -> bool:
    return any(t["pair"] == pair and t["status"] == "OPEN" for t in trades)

def cooldown_ready(pair: str) -> bool:
    last = last_trade_times.get(pair)
    return last is None or (time.time() - last) >= PAIR_COOLDOWN_SECONDS

def calculate_position_size(entry: float, sl: float) -> float:
    equity = broker.get_account_balance() if broker else INITIAL_EQUITY
    if equity <= 0:
        equity = INITIAL_EQUITY
    risk_amt = equity * (RISK_PERCENT / 100)
    sl_dist = abs(entry - sl)
    return round(risk_amt / sl_dist, 2) if sl_dist > 0 else 0

# -------------------- Signal Generation --------------------
# TWEAK: Relaxed RSI bands slightly (was 52-68 / 32-48) to catch more valid trend signals
def build_signal(name: str, epic: str) -> Optional[Dict]:
    data_15m = broker.get_candles(epic, resolution="MINUTE_15", num_candles=300)
    data_1h = broker.get_candles(epic, resolution="HOUR", num_candles=300)
    live = broker.get_live_price(epic)

    if data_15m is None or data_1h is None or live is None:
        return None
    if len(data_15m) < 80 or len(data_1h) < 80:
        return None

    close_15m = data_15m["Close"]
    close_1h = data_1h["Close"]

    ema20_15m = close_15m.ewm(span=20, adjust=False).mean()
    ema50_15m = close_15m.ewm(span=50, adjust=False).mean()
    ema20_1h = close_1h.ewm(span=20, adjust=False).mean()
    ema50_1h = close_1h.ewm(span=50, adjust=False).mean()
    rsi = calculate_rsi(close_15m)
    atr = calculate_atr(data_15m)

    lp = float(close_15m.iloc[-1])
    prev_p = float(close_15m.iloc[-2])
    e20_15 = float(ema20_15m.iloc[-1])
    prev_e20 = float(ema20_15m.iloc[-2])
    e50_15 = float(ema50_15m.iloc[-1])
    e20_1h = float(ema20_1h.iloc[-1])
    e50_1h = float(ema50_1h.iloc[-1])
    rsi_val = float(rsi.iloc[-1])
    atr_val = float(atr.iloc[-1])
    high = float(data_15m["High"].iloc[-1])
    low = float(data_15m["Low"].iloc[-1])
    prev_high = float(data_15m["High"].iloc[-2])
    prev_low = float(data_15m["Low"].iloc[-2])
    close_1h_last = float(close_1h.iloc[-1])

    if not all(is_valid_number(v) for v in [lp, prev_p, e20_15, prev_e20, e50_15, e20_1h, e50_1h, rsi_val, atr_val]):
        return None
    if atr_val <= 0:
        return None

    spread_limit = atr_val * MAX_SPREAD_TO_ATR_RATIO
    if live["spread"] > spread_limit:
        logger.info(f"{name}: spread too high ({live['spread']:.5f} > {spread_limit:.5f})")
        return None

    trend_gap = abs(e20_15 - e50_15)
    if trend_gap < atr_val * 0.25:
        logger.info(f"{name}: trend too weak (gap {trend_gap:.5f} < {atr_val * 0.25:.5f})")
        return None

    sig = None

    # TWEAK: Relaxed RSI to 50-72 (BUY) and 28-50 (SELL) to capture strong trend continuations
    if (lp > e20_15 > e50_15
            and close_1h_last > e20_1h > e50_1h
            and 50 <= rsi_val <= 72
            and prev_p <= prev_e20 * 1.0015
            and high > prev_high):
        sig = "BUY"
    elif (lp < e20_15 < e50_15
          and close_1h_last < e20_1h < e50_1h
          and 28 <= rsi_val <= 50
          and prev_p >= prev_e20 * 0.9985
          and low < prev_low):
        sig = "SELL"

    if not sig:
        return None

    entry = round(live["ask"] if sig == "BUY" else live["bid"], 5)
    stop_dist = round(atr_val * ATR_STOP_MULTIPLIER, 5)
    target_dist = round(atr_val * ATR_TARGET_MULTIPLIER, 5)
    sl = round(entry - stop_dist if sig == "BUY" else entry + stop_dist, 5)
    tp = round(entry + target_dist if sig == "BUY" else entry - target_dist, 5)
    lot_size = calculate_position_size(entry, sl)

    if lot_size <= 0:
        return None

    return {
        "pair": name,
        "epic": epic,
        "type": sig,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": round(atr_val, 5),
        "rsi": round(rsi_val, 2),
        "lot_size": lot_size,
        "risk_per_unit": abs(entry - sl),
        "spread": live["spread"],
    }

def open_trade(signal: Dict):
    order_id = broker.place_order(
        signal["epic"], signal["type"], signal["lot_size"],
        signal["entry"], signal["sl"], signal["tp"]
    )
    if not order_id:
        logger.error(f"Order failed for {signal['pair']}")
        return

    # TWEAK: Confirm fill before storing trade
    time.sleep(1)  # brief pause for broker to process
    fill = broker.confirm_fill(order_id)
    if not fill:
        logger.error(f"Fill not confirmed for {signal['pair']} deal {order_id}. Trade NOT recorded.")
        send_telegram(f"⚠️ Order placed but fill unconfirmed: {signal['pair']} ref={order_id}")
        return

    trade = {
        "pair": signal["pair"],
        "epic": signal["epic"],
        "type": signal["type"],
        "entry": signal["entry"],
        "sl": signal["sl"],
        "tp": signal["tp"],
        "status": "OPEN",
        "opened_at": time.time(),
        "risk_per_unit": signal["risk_per_unit"],
        "break_even_done": False,
        "entry_atr": signal["atr"],
        "lot_size": signal["lot_size"],
        "deal_ref": order_id,
    }

    db_id = db_save_trade(trade)
    trade["db_id"] = db_id
    trades.append(trade)
    last_trade_times[signal["pair"]] = time.time()

    send_telegram(
        f"🔔 NEW TRADE | {signal['pair']} {signal['type']}\n"
        f"Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']}\n"
        f"Lot: {signal['lot_size']} | ATR: {signal['atr']} | RSI: {signal['rsi']}"
    )
    logger.info(f"Trade opened: {signal['pair']} {signal['type']} @ {signal['entry']}")

# -------------------- COMPLETED: update_trade_status --------------------
def update_trade_status(trade: Dict, live: Dict):
    """
    Manages open trade lifecycle:
    - Detects SL/TP hit via live price
    - Moves SL to break-even at 1R profit
    - Applies ATR-based trailing stop after break-even
    """
    if trade["status"] != "OPEN":
        return

    price = live["bid"] if trade["type"] == "BUY" else live["ask"]
    risk = trade["risk_per_unit"]
    if risk <= 0:
        return

    changed = False

    # ---- Check SL hit ----
    if trade["type"] == "BUY" and price <= trade["sl"]:
        status = "WIN" if price >= trade["entry"] else "LOSS"
        trade["status"] = "CLOSED"
        db_update_trade(trade)
        save_trade_result(trade, status, price)
        logger.info(f"SL hit: {trade['pair']} @ {price} → {status}")
        return

    if trade["type"] == "SELL" and price >= trade["sl"]:
        status = "WIN" if price <= trade["entry"] else "LOSS"
        trade["status"] = "CLOSED"
        db_update_trade(trade)
        save_trade_result(trade, status, price)
        logger.info(f"SL hit: {trade['pair']} @ {price} → {status}")
        return

    # ---- Check TP hit ----
    if trade["type"] == "BUY" and price >= trade["tp"]:
        trade["status"] = "CLOSED"
        db_update_trade(trade)
        save_trade_result(trade, "WIN", price)
        logger.info(f"TP hit: {trade['pair']} @ {price} → WIN")
        return

    if trade["type"] == "SELL" and price <= trade["tp"]:
        trade["status"] = "CLOSED"
        db_update_trade(trade)
        save_trade_result(trade, "WIN", price)
        logger.info(f"TP hit: {trade['pair']} @ {price} → WIN")
        return

    # ---- Break-even logic (at 1R profit) ----
    if not trade["break_even_done"]:
        if trade["type"] == "BUY":
            progress = (price - trade["entry"]) / risk
        else:
            progress = (trade["entry"] - price) / risk

        if progress >= BREAK_EVEN_TRIGGER_R:
            new_sl = trade["entry"]  # move SL to entry (break-even)
            if trade["type"] == "BUY" and new_sl > trade["sl"]:
                trade["sl"] = round(new_sl, 5)
                trade["break_even_done"] = True
                changed = True
                logger.info(f"Break-even set: {trade['pair']} SL → {trade['sl']}")
                send_telegram(f"🔒 Break-even: {trade['pair']} SL moved to entry {trade['sl']}")
            elif trade["type"] == "SELL" and new_sl < trade["sl"]:
                trade["sl"] = round(new_sl, 5)
                trade["break_even_done"] = True
                changed = True
                logger.info(f"Break-even set: {trade['pair']} SL → {trade['sl']}")
                send_telegram(f"🔒 Break-even: {trade['pair']} SL moved to entry {trade['sl']}")

    # ---- Trailing stop (after break-even) ----
    if trade["break_even_done"]:
        atr = trade.get("entry_atr", 0)
        trail_dist = atr * TRAILING_STOP_ATR_MULTIPLIER

        if trade["type"] == "BUY":
            new_trail_sl = round(price - trail_dist, 5)
            if new_trail_sl > trade["sl"]:
                trade["sl"] = new_trail_sl
                changed = True
                logger.info(f"Trailing SL updated: {trade['pair']} → {trade['sl']}")

        elif trade["type"] == "SELL":
            new_trail_sl = round(price + trail_dist, 5)
            if new_trail_sl < trade["sl"]:
                trade["sl"] = new_trail_sl
                changed = True
                logger.info(f"Trailing SL updated: {trade['pair']} → {trade['sl']}")

    if changed:
        db_update_trade(trade)

# -------------------- Main Scan Loop --------------------
def scan_pairs():
    if not in_optimal_session():
        logger.info("Outside trading session. Skipping scan.")
        return

    if active_trade_count() >= MAX_ACTIVE_TRADES:
        logger.info(f"Max trades ({MAX_ACTIVE_TRADES}) reached. Skipping scan.")
        return

    for name, search_term in pairs.items():
        if has_open_trade(name):
            continue
        if not cooldown_ready(name):
            continue

        epic = broker.get_epic(search_term)
        if not epic:
            continue

        signal = build_signal(name, epic)
        if signal:
            logger.info(f"Signal: {name} {signal['type']} | RSI={signal['rsi']} | ATR={signal['atr']}")
            open_trade(signal)
            time.sleep(1)

def check_open_trades():
    open_trades = [t for t in trades if t["status"] == "OPEN"]
    for trade in open_trades:
        live = broker.get_live_price(trade["epic"])
        if live:
            update_trade_status(trade, live)

def send_heartbeat():
    balance = broker.get_account_balance()
    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total > 0 else 0
    send_telegram(
        f"💓 Heartbeat\n"
        f"Balance: {balance:.2f} | Open: {active_trade_count()}\n"
        f"Wins: {wins} | Losses: {losses} | WR: {win_rate}%"
    )

def send_report():
    total = wins + losses
    win_rate = round(wins / total * 100, 1) if total > 0 else 0
    send_telegram(
        f"📊 Hourly Report\n"
        f"Trades: {total} | W: {wins} | L: {losses} | WR: {win_rate}%\n"
        f"Open positions: {active_trade_count()}"
    )

# -------------------- Entry Point --------------------
def main():
    global broker

    logger.info("=" * 50)
    logger.info("Forex Trading Bot Starting...")
    logger.info("=" * 50)

    setup_files()
    load_state()

    broker = CapitalClient(
        api_key=CAPITAL_API_KEY,
        login=CAPITAL_LOGIN,
        password=CAPITAL_PASSWORD,
        demo=True
    )

    last_scan = 0
    last_heartbeat = 0
    last_report = 0

    send_telegram("🚀 Bot started successfully.")

    while True:
        try:
            now = time.time()

            # Trade monitoring (every 20s)
            check_open_trades()

            # Pair scanning (every 15 min)
            if now - last_scan >= SCAN_INTERVAL_SECONDS:
                scan_pairs()
                last_scan = now

            # Heartbeat (every 30 min)
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                send_heartbeat()
                last_heartbeat = now

            # Hourly report
            if now - last_report >= REPORT_INTERVAL_SECONDS:
                send_report()
                last_report = now

            time.sleep(TRADE_CHECK_INTERVAL_SECONDS)

        except RuntimeError as e:
            logger.critical(f"Critical error — bot halting: {e}")
            send_telegram(f"🛑 Bot halted: {e}")
            break
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            send_telegram("🛑 Bot manually stopped.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
