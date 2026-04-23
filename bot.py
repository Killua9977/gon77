import csv
import logging
import math
import os
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
MAX_SPREAD_TO_ATR_RATIO = float(os.getenv("MAX_SPREAD_TO_ATR_RATIO", "0.20"))  # Slightly relaxed
PAIR_COOLDOWN_SECONDS = int(os.getenv("PAIR_COOLDOWN_SECONDS", "14400"))

# Timing
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "900"))
TRADE_CHECK_INTERVAL_SECONDS = int(os.getenv("TRADE_CHECK_INTERVAL_SECONDS", "20"))
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))
REPORT_INTERVAL_SECONDS = int(os.getenv("REPORT_INTERVAL_SECONDS", "3600"))

STATE_FILE = "trade_state.csv"
RESULTS_FILE = "trade_results.csv"
LOG_FILE = "bot.log"

# Search terms for Capital.com (bot auto-discovers EPICs)
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

# -------------------- Capital.com Direct API Client --------------------
class CapitalClient:
    """Direct REST API client for Capital.com."""

    def __init__(self, api_key: str, login: str, password: str, demo: bool = True):
        self.api_key = api_key
        self.login = login
        self.password = password
        self.base_url = "https://demo-api-capital.backend-capital.com" if demo else "https://api-capital.backend-capital.com"
        self.cst = None
        self.security_token = None
        self.session = requests.Session()
        self.epic_cache = {}
        self.authenticate()

    def authenticate(self):
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
                error_msg = response.json().get('errorMessage', 'Unknown error')
                raise Exception(f"Authentication failed: {error_msg}")

            self.cst = response.headers.get("CST")
            self.security_token = response.headers.get("X-SECURITY-TOKEN")

            logger.info("✅ Connected to Capital.com Demo Account")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Capital.com: {e}")
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

            if response.status_code in [401, 403]:
                logger.info("Session expired, re-authenticating...")
                self.authenticate()
                return self._make_request(method, endpoint, data)

            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def get_epic(self, search_term: str) -> Optional[str]:
        if search_term in self.epic_cache:
            return self.epic_cache[search_term]

        try:
            endpoint = f"/api/v1/markets?searchTerm={search_term}"
            data = self._make_request("GET", endpoint)
            if data and 'markets' in data and len(data['markets']) > 0:
                epic = data['markets'][0]['epic']
                self.epic_cache[search_term] = epic
                logger.info(f"Found epic '{epic}' for search term '{search_term}'")
                return epic
            else:
                logger.warning(f"No market found for search term '{search_term}'")
                return None
        except Exception as e:
            logger.error(f"Failed to search for epic '{search_term}': {e}")
            return None

    def get_candles(self, epic: str, resolution: str = "MINUTE_15", num_candles: int = 300) -> Optional[pd.DataFrame]:
        try:
            endpoint = f"/api/v1/prices/{epic}?resolution={resolution}&max={num_candles}"
            data = self._make_request("GET", endpoint)

            if not data or 'prices' not in data:
                logger.warning(f"No candle data for {epic}")
                return None

            rows = []
            for candle in data['prices']:
                rows.append({
                    "time": candle['snapshotTime'],
                    "Open": float(candle['openPrice']['bid']),
                    "High": float(candle['highPrice']['bid']),
                    "Low": float(candle['lowPrice']['bid']),
                    "Close": float(candle['closePrice']['bid']),
                })

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
            endpoint = f"/api/v1/markets/{epic}"
            data = self._make_request("GET", endpoint)

            if not data or 'snapshot' not in data:
                logger.warning(f"No market data for {epic}")
                return None

            snapshot = data['snapshot']
            bid = float(snapshot['bid'])
            ask = float(snapshot['offer'])
            mid = (bid + ask) / 2

            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
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

            pip_size = 0.0001
            if "JPY" in epic:
                pip_size = 0.01

            if direction == "BUY":
                stop_distance = int(abs(entry - sl) / pip_size)
                limit_distance = int(abs(tp - entry) / pip_size)
            else:
                stop_distance = int(abs(sl - entry) / pip_size)
                limit_distance = int(abs(entry - tp) / pip_size)

            endpoint = "/api/v1/positions"
            data = {
                "epic": epic,
                "direction": direction,
                "size": units,
                "stopDistance": max(stop_distance, 1),
                "limitDistance": max(limit_distance, 1),
                "guaranteedStop": False,
                "forceOpen": True
            }

            result = self._make_request("POST", endpoint, data)

            if result and 'dealReference' in result:
                logger.info(f"Order placed: {order_type} {units} {epic}")
                return result['dealReference']

            return None

        except Exception as e:
            logger.error(f"Failed to place order for {epic}: {e}")
            return None

    def get_account_balance(self) -> float:
        try:
            endpoint = "/api/v1/accounts"
            data = self._make_request("GET", endpoint)

            if data and 'accounts' in data and len(data['accounts']) > 0:
                return float(data['accounts'][0]['balance']['balance'])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return 0.0

# -------------------- Helper Functions --------------------
def ensure_csv(path: str, headers: List[str]):
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)

def setup_files():
    ensure_csv(RESULTS_FILE, ["timestamp","pair","type","entry","sl","tp","exit_price","status","profit_r","pnl"])
    ensure_csv(STATE_FILE, ["pair","type","entry","sl","tp","status","opened_at","risk_per_unit","break_even_done","epic","entry_atr","lot_size","deal_ref"])

def is_valid_number(v):
    return v is not None and not math.isnan(v) and math.isfinite(v)

def now_utc():
    return datetime.now(timezone.utc)

def send_telegram(msg: str):
    if not TOKEN or not CHAT_ID:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                     data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# -------------------- Technical Indicators --------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = (high - low).combine((high - prev_close).abs(), max).combine((low - prev_close).abs(), max)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# -------------------- Session Filter (EXPANDED) --------------------
def in_optimal_session() -> bool:
    """Trade during Asian, London, and NY sessions (00:00–17:00 UTC)."""
    now = now_utc()
    
    # Skip weekends
    if now.weekday() >= 5:
        return False
    
    hour = now.hour
    
    # Covers Asian (00:00–03:00), London (08:00–12:00), London/NY overlap (12:00–16:00)
    # Skips only the quietest period (17:00–00:00)
    return 0 <= hour < 17

# -------------------- Global State --------------------
trades: List[Dict] = []
wins = losses = 0
last_trade_times: Dict[str, float] = {}
broker = None

# -------------------- State Persistence --------------------
def load_state():
    global trades, last_trade_times, wins, losses
    if not os.path.exists(STATE_FILE):
        return
    with open(STATE_FILE, 'r') as f:
        for row in csv.DictReader(f):
            if row["status"] != "OPEN":
                continue
            trades.append({
                "pair": row["pair"],
                "epic": row["epic"],
                "type": row["type"],
                "entry": float(row["entry"]),
                "sl": float(row["sl"]),
                "tp": float(row["tp"]),
                "status": row["status"],
                "opened_at": float(row["opened_at"]),
                "risk_per_unit": float(row["risk_per_unit"]),
                "break_even_done": row["break_even_done"] == "1",
                "entry_atr": float(row.get("entry_atr", 0)),
                "lot_size": float(row.get("lot_size", 0)),
                "deal_ref": row.get("deal_ref", "")
            })
            last_trade_times[row["pair"]] = float(row["opened_at"])
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            for row in csv.DictReader(f):
                if row["status"] == "WIN":
                    wins += 1
                elif row["status"] == "LOSS":
                    losses += 1

def save_open_state():
    with open(STATE_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["pair","type","entry","sl","tp","status","opened_at","risk_per_unit","break_even_done","epic","entry_atr","lot_size","deal_ref"])
        for t in trades:
            if t["status"] == "OPEN":
                w.writerow([t["pair"], t["type"], t["entry"], t["sl"], t["tp"], t["status"],
                           t["opened_at"], t["risk_per_unit"], int(t["break_even_done"]),
                           t["epic"], t["entry_atr"], t["lot_size"], t.get("deal_ref","")])

def save_trade_result(trade, status, exit_price):
    global wins, losses
    risk = trade["risk_per_unit"]
    if trade["type"] == "BUY":
        profit = (exit_price - trade["entry"]) * trade["lot_size"]
        profit_r = (exit_price - trade["entry"]) / risk if risk else 0
    else:
        profit = (trade["entry"] - exit_price) * trade["lot_size"]
        profit_r = (trade["entry"] - exit_price) / risk if risk else 0
    with open(RESULTS_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([now_utc().isoformat(), trade["pair"], trade["type"],
            round(trade["entry"],5), round(trade["sl"],5), round(trade["tp"],5),
            round(exit_price,5), status, round(profit_r,2), round(profit,2)])
    if status == "WIN":
        wins += 1
    else:
        losses += 1
    send_telegram(f"{status} {trade['pair']} @ {round(exit_price,5)} | PnL: {round(profit,2)}")

# -------------------- Trade Management --------------------
def active_trade_count():
    return sum(1 for t in trades if t["status"] == "OPEN")

def has_open_trade(pair):
    return any(t["pair"] == pair and t["status"] == "OPEN" for t in trades)

def cooldown_ready(pair):
    last = last_trade_times.get(pair)
    return last is None or (time.time() - last) >= PAIR_COOLDOWN_SECONDS

def calculate_position_size(entry, sl):
    equity = broker.get_account_balance() if broker else INITIAL_EQUITY
    if equity <= 0:
        equity = INITIAL_EQUITY
    risk_amt = equity * (RISK_PERCENT / 100)
    sl_dist = abs(entry - sl)
    return round(risk_amt / sl_dist, 2) if sl_dist > 0 else 0

# -------------------- Signal Generation --------------------
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

    if not all(is_valid_number(v) for v in [lp, prev_p, e20_15, prev_e20, e50_15, e20_1h, e50_1h, rsi_val, atr_val]):
        return None
    if atr_val <= 0:
        return None
    if live["spread"] > atr_val * MAX_SPREAD_TO_ATR_RATIO:
        logger.info(f"{name}: spread too high ({live['spread']} > {atr_val * MAX_SPREAD_TO_ATR_RATIO})")
        return None
    if abs(e20_15 - e50_15) < atr_val * 0.25:
        logger.info(f"{name}: trend too weak (gap {abs(e20_15 - e50_15)} < {atr_val * 0.25})")
        return None

    sig = None
    if (lp > e20_15 > e50_15 and float(close_1h.iloc[-1]) > e20_1h > e50_1h and
        52 <= rsi_val <= 68 and prev_p <= prev_e20 * 1.0015 and high > prev_high):
        sig = "BUY"
    elif (lp < e20_15 < e50_15 and float(close_1h.iloc[-1]) < e20_1h < e50_1h and
          32 <= rsi_val <= 48 and prev_p >= prev_e20 * 0.9985 and low < prev_low):
        sig = "SELL"

    if not sig:
        return None

    entry = round(live["ask"] if sig == "BUY" else live["bid"], 5)
    stop_dist = round(atr_val * ATR_STOP_MULTIPLIER, 5)
    target_dist = round(atr_val * ATR_TARGET_MULTIPLIER, 5)
    sl = round(entry - stop_dist, 5) if sig == "BUY" else round(entry + stop_dist, 5)
    tp = round(entry + target_dist, 5) if sig == "BUY" else round(entry - target_dist, 5)
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
        "spread": live["spread"]
    }

def open_trade(signal: Dict):
    order_id = broker.place_order(
        signal["epic"], signal["type"], signal["lot_size"],
        signal["entry"], signal["sl"], signal["tp"]
    )
    if not order_id:
        logger.error(f"Order failed for {signal['pair']}")
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
        "deal_ref": order_id
    }
    trades.append(trade)
    last_trade_times[signal["pair"]] = time.time()
    save_open_state()
    send_telegram(f"🔔 {signal['pair']} {signal['type']}\nEntry: {signal['entry']}\nSL: {signal['sl']}\nTP: {signal['tp']}")

def update_trade_status(trade: Dict, live: Dict):
    if trade["status"] != "OPEN":
        return

    price = live["bid"] if trade["type"] == "BUY" else live["ask"]
    risk = trade["risk_per_unit"]
    if risk <= 0:
        return

    if trade["type"] == "BUY":
        progress = (price - trade["entry"]) / risk
        if progress >= BREAK_EVEN_TRIGGER_R and not trade["break_even_done"]:
            trade["sl"] = max(trade["sl"], trade["entry"])
            trade["break_even_done"] = True
            send_telegram(f"{trade['pair']} moved to break-even at {trade['sl']}")
        if trade["break_even_done"]:
            new_sl = price - trade["entry_atr"] * TRAILING_STOP_ATR_MULTIPLIER
            if new_sl > trade["sl"]:
                trade["sl"] = round(new_sl, 5)
        if price >= trade["tp"]:
            trade["status"] = "WIN"
            save_trade_result(trade, "WIN", price)
        elif price <= trade["sl"]:
            trade["status"] = "LOSS"
            save_trade_result(trade, "LOSS", price)
    else:
        progress = (trade["entry"] - price) / risk
        if progress >= BREAK_EVEN_TRIGGER_R and not trade["break_even_done"]:
            trade["sl"] = min(trade["sl"], trade["entry"])
            trade["break_even_done"] = True
            send_telegram(f"{trade['pair']} moved to break-even at {trade['sl']}")
        if trade["break_even_done"]:
            new_sl = price + trade["entry_atr"] * TRAILING_STOP_ATR_MULTIPLIER
            if new_sl < trade["sl"]:
                trade["sl"] = round(new_sl, 5)
        if price <= trade["tp"]:
            trade["status"] = "WIN"
            save_trade_result(trade, "WIN", price)
        elif price >= trade["sl"]:
            trade["status"] = "LOSS"
            save_trade_result(trade, "LOSS", price)

def check_trades():
    for t in trades:
        if t["status"] != "OPEN":
            continue
        live = broker.get_live_price(t["epic"])
        if live:
            update_trade_status(t, live)
    save_open_state()

def scan_market():
    if not in_optimal_session():
        logger.info("Market scan skipped: outside trading session")
        return
    if active_trade_count() >= MAX_ACTIVE_TRADES:
        logger.info("Market scan skipped: max active trades reached")
        return

    logger.info("Scanning market for signals...")
    for name, search_term in pairs.items():
        if has_open_trade(name) or not cooldown_ready(name):
            continue
        try:
            epic = broker.get_epic(search_term)
            if not epic:
                logger.warning(f"Could not find epic for {name} ({search_term}), skipping")
                continue
            sig = build_signal(name, epic)
            if sig:
                open_trade(sig)
                if active_trade_count() >= MAX_ACTIVE_TRADES:
                    break
        except Exception as e:
            logger.error(f"Error scanning {name} ({search_term}): {e}")
            continue

def send_heartbeat():
    eq = broker.get_account_balance() if broker else 0
    send_telegram(f"💓 Alive | Trades: {active_trade_count()} | Equity: ${round(eq,2)}")

def send_performance():
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    send_telegram(f"📊 Wins: {wins} | Losses: {losses} | Win Rate: {round(wr,1)}%")

# -------------------- Main --------------------
def run_bot():
    global broker
    setup_files()
    load_state()

    if not all([CAPITAL_API_KEY, CAPITAL_LOGIN, CAPITAL_PASSWORD]):
        logger.error("Missing Capital.com credentials!")
        return

    broker = CapitalClient(CAPITAL_API_KEY, CAPITAL_LOGIN, CAPITAL_PASSWORD, demo=True)

    balance = broker.get_account_balance()
    logger.info(f"Bot started. Equity: ${balance}")
    send_telegram(f"🚀 Capital.com Demo Bot Started\nEquity: ${round(balance,2)}")

    last_scan = last_heartbeat = last_report = last_check = 0
    while True:
        try:
            now = time.time()
            if now - last_scan >= SCAN_INTERVAL_SECONDS:
                scan_market()
                last_scan = now
            if now - last_check >= TRADE_CHECK_INTERVAL_SECONDS:
                check_trades()
                last_check = now
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                send_heartbeat()
                last_heartbeat = now
            if now - last_report >= REPORT_INTERVAL_SECONDS:
                send_performance()
                last_report = now
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
            send_telegram(f"⚠️ Bot error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    run_bot()
