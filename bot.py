import csv
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import pandas as pd
import requests

# -------------------- Configuration --------------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"

# Strategy Parameters
INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", "1000.0"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "2.0"))
MAX_ACTIVE_TRADES = int(os.getenv("MAX_ACTIVE_TRADES", "3"))
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "1.5"))
ATR_TARGET_MULTIPLIER = float(os.getenv("ATR_TARGET_MULTIPLIER", "2.4"))
BREAK_EVEN_TRIGGER_R = float(os.getenv("BREAK_EVEN_TRIGGER_R", "1.0"))
TRAILING_STOP_ATR_MULTIPLIER = float(os.getenv("TRAILING_STOP_ATR_MULTIPLIER", "1.2"))
MAX_SPREAD_TO_ATR_RATIO = float(os.getenv("MAX_SPREAD_TO_ATR_RATIO", "0.12"))
PAIR_COOLDOWN_SECONDS = int(os.getenv("PAIR_COOLDOWN_SECONDS", "14400"))

# Timing
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "900"))
TRADE_CHECK_INTERVAL_SECONDS = int(os.getenv("TRADE_CHECK_INTERVAL_SECONDS", "20"))
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))
REPORT_INTERVAL_SECONDS = int(os.getenv("REPORT_INTERVAL_SECONDS", "3600"))

STATE_FILE = "trade_state.csv"
RESULTS_FILE = "trade_results.csv"
LOG_FILE = "bot.log"

pairs = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "NZDUSD": "NZD/USD",
    "EURGBP": "EUR/GBP",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
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

# -------------------- Manual Retry Logic --------------------
def retry_on_failure(max_attempts=3, initial_wait=2, max_wait=10):
    """Simple retry decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, RuntimeError) as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, max_wait)
            return None
        return wrapper
    return decorator

# -------------------- Rate Limiter --------------------
class RateLimiter:
    def __init__(self, max_calls: int = 8, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * (self.max_calls / self.period)
        self.tokens = min(self.max_calls, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self):
        while True:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return
            time.sleep(0.1)

_rate_limiter = RateLimiter()

def rate_limited_request(method: str, url: str, **kwargs) -> requests.Response:
    _rate_limiter.acquire()
    response = requests.request(method, url, **kwargs)
    response.raise_for_status()
    return response

# -------------------- Helper Functions --------------------
def ensure_csv(path: str, headers: List[str]):
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def setup_files():
    ensure_csv(RESULTS_FILE, [
        "timestamp", "pair", "type", "entry", "sl", "tp",
        "exit_price", "status", "profit_r", "pnl", "equity_after"
    ])
    ensure_csv(STATE_FILE, [
        "pair", "type", "entry", "sl", "tp", "status", "opened_at",
        "risk_per_unit", "break_even_done", "symbol", "entry_atr", "lot_size"
    ])

def is_valid_number(value):
    return value is not None and not math.isnan(value) and math.isfinite(value)

def now_utc():
    return datetime.now(timezone.utc)

def send_telegram(message: str):
    if not TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured; message not sent: %s", message)
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Telegram send failed: %s", e)

# -------------------- Broker Abstraction --------------------
class Broker:
    def place_order(self, symbol: str, order_type: str, units: float,
                    entry: float, sl: float, tp: float) -> Optional[str]:
        raise NotImplementedError

class SimulationBroker(Broker):
    def place_order(self, symbol: str, order_type: str, units: float,
                    entry: float, sl: float, tp: float) -> Optional[str]:
        logger.info("SIMULATION: Placed %s order for %s units of %s at %s", 
                   order_type, units, symbol, entry)
        return "SIM-" + str(int(time.time()))

broker = SimulationBroker()

# -------------------- Twelve Data API --------------------
@retry_on_failure(max_attempts=3, initial_wait=2, max_wait=10)
def twelve_data_get(path: str, params: Optional[Dict] = None) -> Dict:
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("Missing TWELVE_DATA_API_KEY")
    query = dict(params or {})
    query["apikey"] = TWELVE_DATA_API_KEY
    response = rate_limited_request("GET", f"{TWELVE_DATA_BASE_URL}{path}", params=query, timeout=15)
    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(payload.get("message", "Twelve Data error"))
    return payload

def get_history(symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetch historical candles and return as DataFrame."""
    try:
        # Fixed: Use proper Twelve Data interval format
        outputsize_map = {"15min": 300, "1h": 300, "5min": 300}
        payload = twelve_data_get("/time_series", {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize_map.get(interval, 300),
            "timezone": "UTC",
            "format": "JSON",
        })
    except Exception as e:
        logger.error("History fetch failed for %s: %s", symbol, e)
        return None

    values = payload.get("values", [])
    if not values:
        logger.warning(f"No historical data returned for {symbol}")
        return None
        
    rows = []
    for candle in reversed(values):
        try:
            rows.append({
                "time": candle["datetime"],
                "Open": float(candle["open"]),
                "High": float(candle["high"]),
                "Low": float(candle["low"]),
                "Close": float(candle["close"]),
                "Volume": float(candle.get("volume", 0) or 0),
            })
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping malformed candle for {symbol}: {e}")
            continue
            
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    return df.dropna()

def get_live_price(symbol: str) -> Optional[Dict]:
    try:
        payload = twelve_data_get("/quote", {"symbol": symbol, "interval": "1min"})
    except Exception as e:
        logger.error("Pricing fetch failed for %s: %s", symbol, e)
        return None

    close = payload.get("close")
    if close is None:
        return None
    mid = float(close)
    ask = float(payload.get("ask", mid))
    bid = float(payload.get("bid", mid))
    if ask < bid:
        ask = mid
        bid = mid
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": round(max(ask - bid, 0.0), 5),
        "tradeable": True,
    }

# -------------------- Technical Indicators --------------------
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = (high - low).combine((high - prev_close).abs(), max)
    tr = tr.combine((low - prev_close).abs(), max)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# -------------------- Session Filter --------------------
def in_optimal_session() -> bool:
    now = now_utc()
    if now.weekday() >= 5:
        return False
    return 12 <= now.hour < 16

# -------------------- Global State --------------------
trades: List[Dict] = []
current_equity = INITIAL_EQUITY
wins = 0
losses = 0
last_trade_times: Dict[str, float] = {}

# -------------------- State Persistence --------------------
def load_state():
    global trades, last_trade_times, current_equity, wins, losses
    if not os.path.exists(STATE_FILE):
        return

    with open(STATE_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "OPEN":
                continue
            trade = {
                "pair": row["pair"],
                "symbol": row["symbol"],
                "type": row["type"],
                "entry": float(row["entry"]),
                "sl": float(row["sl"]),
                "tp": float(row["tp"]),
                "status": row["status"],
                "opened_at": float(row["opened_at"]),
                "risk_per_unit": float(row["risk_per_unit"]),
                "break_even_done": row["break_even_done"] == "1",
                "entry_atr": float(row.get("entry_atr", 0.0)),
                "lot_size": float(row.get("lot_size", 0.0)),
                "broker_order_id": row.get("broker_order_id", None),
            }
            trades.append(trade)
            last_trade_times[trade["pair"]] = trade["opened_at"]

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] == "WIN":
                    wins += 1
                elif row["status"] == "LOSS":
                    losses += 1
                equity_str = row.get("equity_after")
                if equity_str:
                    current_equity = float(equity_str)

    logger.info("Loaded %d open trades. Equity: %.2f, Wins: %d, Losses: %d",
                len([t for t in trades if t["status"] == "OPEN"]), current_equity, wins, losses)

def save_open_state():
    with open(STATE_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair", "type", "entry", "sl", "tp", "status", "opened_at",
            "risk_per_unit", "break_even_done", "symbol", "entry_atr", "lot_size", "broker_order_id"
        ])
        for t in trades:
            if t["status"] == "OPEN":
                writer.writerow([
                    t["pair"], t["type"], t["entry"], t["sl"], t["tp"],
                    t["status"], t["opened_at"], t["risk_per_unit"],
                    int(t["break_even_done"]), t["symbol"], t["entry_atr"],
                    t["lot_size"], t.get("broker_order_id", "")
                ])

def save_trade_result(trade: Dict, status: str, exit_price: float):
    global current_equity, wins, losses

    risk = trade["risk_per_unit"]
    lot_size = trade["lot_size"]
    if trade["type"] == "BUY":
        profit = (exit_price - trade["entry"]) * lot_size
        profit_r = (exit_price - trade["entry"]) / risk if risk > 0 else 0.0
    else:
        profit = (trade["entry"] - exit_price) * lot_size
        profit_r = (trade["entry"] - exit_price) / risk if risk > 0 else 0.0

    current_equity += profit

    with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            now_utc().isoformat(),
            trade["pair"],
            trade["type"],
            round(trade["entry"], 5),
            round(trade["sl"], 5),
            round(trade["tp"], 5),
            round(exit_price, 5),
            status,
            round(profit_r, 2),
            round(profit, 2),
            round(current_equity, 2)
        ])

    if status == "WIN":
        wins += 1
    else:
        losses += 1

    send_telegram(f"{status} {trade['pair']} at {round(exit_price,5)} | PnL: {round(profit,2)} | Equity: {round(current_equity,2)}")

# -------------------- Trade Management --------------------
def has_open_trade(pair: str) -> bool:
    return any(t["pair"] == pair and t["status"] == "OPEN" for t in trades)

def active_trade_count() -> int:
    return sum(1 for t in trades if t["status"] == "OPEN")

def cooldown_ready(pair: str) -> bool:
    last_time = last_trade_times.get(pair)
    if last_time is None:
        return True
    return (time.time() - last_time) >= PAIR_COOLDOWN_SECONDS

def calculate_position_size(entry: float, sl: float) -> float:
    risk_amount = current_equity * (RISK_PERCENT / 100.0)
    sl_distance = abs(entry - sl)
    if sl_distance <= 0:
        return 0.0
    return round(risk_amount / sl_distance, 2)

# -------------------- Signal Generation --------------------
def build_signal(name: str, symbol: str) -> Optional[Dict]:
    # Fixed: Use proper Twelve Data interval format "15min" instead of "15m"
    data_15m = get_history(symbol, period="10d", interval="15min")
    data_1h = get_history(symbol, period="20d", interval="1h")
    live = get_live_price(symbol)

    if data_15m is None or data_1h is None or live is None:
        return None
    if len(data_15m) < 80 or len(data_1h) < 80:
        logger.info(f"Insufficient data for {symbol}: 15min={len(data_15m) if data_15m is not None else 0}, 1h={len(data_1h) if data_1h is not None else 0}")
        return None

    close_15m = data_15m["Close"]
    close_1h = data_1h["Close"]

    ema20_15m = close_15m.ewm(span=20, adjust=False).mean()
    ema50_15m = close_15m.ewm(span=50, adjust=False).mean()
    ema20_1h = close_1h.ewm(span=20, adjust=False).mean()
    ema50_1h = close_1h.ewm(span=50, adjust=False).mean()
    rsi_15m = calculate_rsi(close_15m)
    atr_15m = calculate_atr(data_15m)

    latest_price = float(close_15m.iloc[-1])
    prev_price = float(close_15m.iloc[-2])
    latest_ema20_15m = float(ema20_15m.iloc[-1])
    prev_ema20_15m = float(ema20_15m.iloc[-2])
    latest_ema50_15m = float(ema50_15m.iloc[-1])
    latest_ema20_1h = float(ema20_1h.iloc[-1])
    latest_ema50_1h = float(ema50_1h.iloc[-1])
    latest_rsi = float(rsi_15m.iloc[-1])
    latest_atr = float(atr_15m.iloc[-1])
    candle_high = float(data_15m["High"].iloc[-1])
    candle_low = float(data_15m["Low"].iloc[-1])
    previous_high = float(data_15m["High"].iloc[-2])
    previous_low = float(data_15m["Low"].iloc[-2])

    values = [latest_price, prev_price, latest_ema20_15m, prev_ema20_15m, latest_ema50_15m,
              latest_ema20_1h, latest_ema50_1h, latest_rsi, latest_atr,
              candle_high, candle_low, previous_high, previous_low]
    if not all(is_valid_number(v) for v in values):
        return None
    if latest_atr <= 0:
        return None
    if not live["tradeable"]:
        return None
    if live["spread"] > latest_atr * MAX_SPREAD_TO_ATR_RATIO:
        logger.info(f"{symbol} spread too high: {live['spread']} > {latest_atr * MAX_SPREAD_TO_ATR_RATIO}")
        return None

    trend_gap = abs(latest_ema20_15m - latest_ema50_15m)
    if trend_gap < latest_atr * 0.25:
        return None

    signal_type = None

    if (latest_price > latest_ema20_15m > latest_ema50_15m and
        float(close_1h.iloc[-1]) > latest_ema20_1h > latest_ema50_1h and
        52 <= latest_rsi <= 68 and
        prev_price <= prev_ema20_15m * 1.0015 and
        candle_high > previous_high):
        signal_type = "BUY"
    elif (latest_price < latest_ema20_15m < latest_ema50_15m and
          float(close_1h.iloc[-1]) < latest_ema20_1h < latest_ema50_1h and
          32 <= latest_rsi <= 48 and
          prev_price >= prev_ema20_15m * 0.9985 and
          candle_low < previous_low):
        signal_type = "SELL"

    if signal_type is None:
        return None

    entry = round(live["ask"] if signal_type == "BUY" else live["bid"], 5)
    stop_distance = round(latest_atr * ATR_STOP_MULTIPLIER, 5)
    target_distance = round(latest_atr * ATR_TARGET_MULTIPLIER, 5)

    if stop_distance <= 0 or target_distance <= 0:
        return None

    if signal_type == "BUY":
        sl = round(entry - stop_distance, 5)
        tp = round(entry + target_distance, 5)
    else:
        sl = round(entry + stop_distance, 5)
        tp = round(entry - target_distance, 5)

    lot_size = calculate_position_size(entry, sl)
    if lot_size <= 0:
        return None

    return {
        "pair": name,
        "symbol": symbol,
        "type": signal_type,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "atr": round(latest_atr, 5),
        "rsi": round(latest_rsi, 2),
        "lot_size": lot_size,
        "risk_per_unit": round(abs(entry - sl), 5),
        "spread": live["spread"],
    }

def open_trade(signal: Dict):
    order_id = broker.place_order(
        symbol=signal["symbol"],
        order_type=signal["type"],
        units=signal["lot_size"],
        entry=signal["entry"],
        sl=signal["sl"],
        tp=signal["tp"]
    )
    if order_id is None:
        logger.error("Failed to place order for %s", signal["pair"])
        return

    trade = {
        "pair": signal["pair"],
        "symbol": signal["symbol"],
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
        "broker_order_id": order_id,
    }
    trades.append(trade)
    last_trade_times[signal["pair"]] = time.time()
    save_open_state()

    send_telegram(
        f"🔔 SNIPER SIGNAL\n\n"
        f"{signal['pair']} {signal['type']}\n"
        f"Entry: {signal['entry']}\n"
        f"SL: {signal['sl']}\n"
        f"TP: {signal['tp']}\n"
        f"ATR: {signal['atr']}\n"
        f"RSI: {signal['rsi']}\n"
        f"Spread: {signal['spread']}\n"
        f"Lot Size: {signal['lot_size']}\n"
        f"Equity: {round(current_equity,2)}"
    )

def update_trade_status(trade: Dict, latest_price: float):
    if trade["status"] != "OPEN":
        return

    risk = trade["risk_per_unit"]
    entry_atr = trade["entry_atr"]
    if risk <= 0 or entry_atr <= 0:
        return

    if trade["type"] == "BUY":
        progress_r = (latest_price - trade["entry"]) / risk
    else:
        progress_r = (trade["entry"] - latest_price) / risk

    if progress_r >= BREAK_EVEN_TRIGGER_R and not trade["break_even_done"]:
        if trade["type"] == "BUY":
            trade["sl"] = max(trade["sl"], trade["entry"])
        else:
            trade["sl"] = min(trade["sl"], trade["entry"])
        trade["break_even_done"] = True
        send_telegram(f"{trade['pair']} moved to break-even at {trade['sl']}")
        save_open_state()

    trail_distance = entry_atr * TRAILING_STOP_ATR_MULTIPLIER
    if trade["break_even_done"]:
        if trade["type"] == "BUY":
            new_sl = latest_price - trail_distance
            if new_sl > trade["sl"]:
                trade["sl"] = round(new_sl, 5)
                logger.info(f"{trade['pair']} trailing stop updated to {trade['sl']}")
        else:
            new_sl = latest_price + trail_distance
            if new_sl < trade["sl"]:
                trade["sl"] = round(new_sl, 5)
                logger.info(f"{trade['pair']} trailing stop updated to {trade['sl']}")
        save_open_state()

    if trade["type"] == "BUY":
        if latest_price >= trade["tp"]:
            trade["status"] = "WIN"
            save_trade_result(trade, "WIN", latest_price)
        elif latest_price <= trade["sl"]:
            trade["status"] = "LOSS"
            save_trade_result(trade, "LOSS", latest_price)
    else:
        if latest_price <= trade["tp"]:
            trade["status"] = "WIN"
            save_trade_result(trade, "WIN", latest_price)
        elif latest_price >= trade["sl"]:
            trade["status"] = "LOSS"
            save_trade_result(trade, "LOSS", latest_price)

def check_trades():
    for trade in list(trades):
        if trade["status"] != "OPEN":
            continue
        live = get_live_price(trade["symbol"])
        if live is None or not live["tradeable"]:
            continue
        price = live["bid"] if trade["type"] == "BUY" else live["ask"]
        if not is_valid_number(price):
            continue
        update_trade_status(trade, price)

    save_open_state()

# -------------------- Scanning --------------------
def scan_market():
    if not in_optimal_session():
        logger.info("Market scan skipped: outside preferred session.")
        return
    if active_trade_count() >= MAX_ACTIVE_TRADES:
        logger.info("Market scan skipped: max active trades reached.")
        return

    signals = []
    for name, symbol in pairs.items():
        if has_open_trade(name) or not cooldown_ready(name):
            continue
        sig = build_signal(name, symbol)
        if sig:
            signals.append(sig)
            logger.info(f"Signal generated for {name}")

    if not signals:
        logger.info("No valid signals this cycle.")
        return

    slots = MAX_ACTIVE_TRADES - active_trade_count()
    for sig in signals[:slots]:
        open_trade(sig)

# -------------------- Reporting --------------------
def send_heartbeat():
    open_count = active_trade_count()
    send_telegram(
        f"💓 Bot Alive\n"
        f"Open Trades: {open_count}\n"
        f"Equity: {round(current_equity,2)}\n"
        f"Time UTC: {now_utc().strftime('%Y-%m-%d %H:%M:%S')}"
    )

def send_performance():
    total = wins + losses
    winrate = (wins / total * 100) if total > 0 else 0
    send_telegram(
        f"📊 PERFORMANCE\n\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Win Rate: {round(winrate,2)}%\n"
        f"Open Trades: {active_trade_count()}\n"
        f"Current Equity: {round(current_equity,2)}"
    )

# -------------------- Main Loop --------------------
def run_bot():
    setup_files()
    load_state()
    logger.info("Bot started. Equity: %.2f, Wins: %d, Losses: %d", current_equity, wins, losses)
    
    # Send startup message
    send_telegram(f"🚀 Bot Started\nEquity: {round(current_equity,2)}\nTime: {now_utc().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    last_scan = 0
    last_heartbeat = 0
    last_report = 0
    last_trade_check = 0

    while True:
        try:
            now = time.time()

            if now - last_scan >= SCAN_INTERVAL_SECONDS:
                logger.info("Scanning market...")
                scan_market()
                last_scan = now

            if now - last_trade_check >= TRADE_CHECK_INTERVAL_SECONDS:
                check_trades()
                last_trade_check = now

            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                send_heartbeat()
                last_heartbeat = now

            if now - last_report >= REPORT_INTERVAL_SECONDS:
                send_performance()
                last_report = now

        except Exception as e:
            logger.exception("Unhandled error in main loop: %s", e)
            send_telegram(f"⚠️ Bot error: {e}")

        time.sleep(1)

if __name__ == "__main__":
    if not TWELVE_DATA_API_KEY:
        logger.error("TWELVE_DATA_API_KEY missing. Please set it in environment variables.")
        exit(1)
    run_bot()
