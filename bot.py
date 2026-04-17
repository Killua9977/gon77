import csv
import math
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

TOKEN = "8670855189:AAGJq69MG1e1GnURfSqr5nIj8THXpXwXxaw"
CHAT_ID = "8670855189"
TWELVE_DATA_API_KEY = "43ec0e0e2bc94a6d937879ee172621d8"
TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"

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

ACCOUNT_BALANCE = 100.0
RISK_PERCENT = 2.0
MAX_ACTIVE_TRADES = 3
SCAN_INTERVAL_SECONDS = 900
TRADE_CHECK_INTERVAL_SECONDS = 20
HEARTBEAT_INTERVAL_SECONDS = 1800
REPORT_INTERVAL_SECONDS = 3600
PAIR_COOLDOWN_SECONDS = 4 * 3600

ATR_STOP_MULTIPLIER = 1.5
ATR_TARGET_MULTIPLIER = 2.4
BREAK_EVEN_TRIGGER_R = 1.0
TRAILING_STOP_ATR_MULTIPLIER = 1.2
MAX_SPREAD_TO_ATR_RATIO = 0.12

STATE_FILE = "trade_state.csv"
RESULTS_FILE = "trade_results.csv"

trades = []
wins = 0
losses = 0
last_trade_times = {}


def ensure_csv(path, headers):
    if os.path.exists(path):
        return

    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def setup_files():
    ensure_csv(
        RESULTS_FILE,
        [
            "timestamp",
            "pair",
            "type",
            "entry",
            "sl",
            "tp",
            "exit_price",
            "status",
            "profit_r",
        ],
    )
    ensure_csv(
        STATE_FILE,
        [
            "pair",
            "type",
            "entry",
            "sl",
            "tp",
            "status",
            "opened_at",
            "risk_per_unit",
            "break_even_done",
            "symbol",
        ],
    )


def is_valid_number(value):
    return value is not None and not math.isnan(value) and math.isfinite(value)


def now_utc():
    return datetime.now(timezone.utc)


def require_twelve_data_key():
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("Missing TWELVE_DATA_API_KEY environment variable.")


def require_telegram_config():
    if not TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable.")
    if not CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_CHAT_ID environment variable.")


def twelve_data_get(path, params=None):
    require_twelve_data_key()
    query = dict(params or {})
    query["apikey"] = TWELVE_DATA_API_KEY
    response = requests.get(
        f"{TWELVE_DATA_BASE_URL}{path}",
        params=query,
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(payload.get("message", "Twelve Data request failed."))
    return payload


def send(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": message},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Telegram send failed: {exc}")


def save_trade_result(trade, status, exit_price):
    profit_r = 0.0
    risk = trade["risk_per_unit"]

    if risk > 0:
        if trade["type"] == "BUY":
            profit_r = (exit_price - trade["entry"]) / risk
        else:
            profit_r = (trade["entry"] - exit_price) / risk

    with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                now_utc().isoformat(),
                trade["pair"],
                trade["type"],
                round(trade["entry"], 5),
                round(trade["sl"], 5),
                round(trade["tp"], 5),
                round(exit_price, 5),
                status,
                round(profit_r, 2),
            ]
        )


def save_open_state():
    with open(STATE_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "pair",
                "type",
                "entry",
                "sl",
                "tp",
                "status",
                "opened_at",
                "risk_per_unit",
                "break_even_done",
                "symbol",
            ]
        )
        for trade in trades:
            if trade["status"] == "OPEN":
                writer.writerow(
                    [
                        trade["pair"],
                        trade["type"],
                        trade["entry"],
                        trade["sl"],
                        trade["tp"],
                        trade["status"],
                        trade["opened_at"],
                        trade["risk_per_unit"],
                        int(trade["break_even_done"]),
                        trade["symbol"],
                    ]
                )


def has_open_trade(pair):
    return any(
        trade["pair"] == pair and trade["status"] == "OPEN"
        for trade in trades
    )


def active_trade_count():
    return sum(1 for trade in trades if trade["status"] == "OPEN")


def calculate_position_size(entry, sl):
    risk_amount = ACCOUNT_BALANCE * (RISK_PERCENT / 100)
    sl_distance = abs(entry - sl)
    if sl_distance <= 0:
        return 0.0
    return round(risk_amount / sl_distance, 2)


def get_history(symbol, period, interval):
    try:
        outputsize_map = {
            "15m": 300,
            "1h": 300,
            "5m": 300,
        }
        outputsize = outputsize_map[interval]
        payload = twelve_data_get(
            "/time_series",
            params={
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "timezone": "UTC",
                "format": "JSON",
            },
        )
    except Exception as exc:
        print(f"History fetch failed for {symbol}: {exc}")
        return None

    values = payload.get("values", [])
    rows = []

    for candle in reversed(values):
        rows.append(
            {
                "time": candle["datetime"],
                "Open": float(candle["open"]),
                "High": float(candle["high"]),
                "Low": float(candle["low"]),
                "Close": float(candle["close"]),
                "Volume": float(candle.get("volume", 0) or 0),
            }
        )

    if not rows:
        return None

    data = pd.DataFrame(rows)
    data["time"] = pd.to_datetime(data["time"], utc=True)
    data.set_index("time", inplace=True)
    return data.dropna()


def get_live_price(symbol):
    try:
        payload = twelve_data_get(
            "/quote",
            params={
                "symbol": symbol,
                "interval": "1min",
            },
        )
    except Exception as exc:
        print(f"Pricing fetch failed for {symbol}: {exc}")
        return None

    close_price = payload.get("close")
    if close_price is None:
        return None

    mid = float(close_price)
    ask = float(payload.get("ask", mid))
    bid = float(payload.get("bid", mid))
    if ask < bid:
        ask = mid
        bid = mid

    spread = round(max(ask - bid, 0.0), 5)

    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "tradeable": True,
    }


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(data, period=14):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    prev_close = close.shift(1)

    true_range = (high - low).combine((high - prev_close).abs(), max)
    true_range = true_range.combine((low - prev_close).abs(), max)

    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def in_session():
    hour = now_utc().hour
    return 6 <= hour <= 20


def cooldown_ready(pair):
    last_trade_time = last_trade_times.get(pair)
    if last_trade_time is None:
        return True
    return time.time() - last_trade_time >= PAIR_COOLDOWN_SECONDS


def build_signal(name, symbol):
    data_15m = get_history(symbol, period="10d", interval="15m")
    data_1h = get_history(symbol, period="20d", interval="1h")
    live_price = get_live_price(symbol)

    if data_15m is None or data_1h is None or live_price is None:
        return None

    if len(data_15m) < 80 or len(data_1h) < 80:
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

    values = [
        latest_price,
        prev_price,
        latest_ema20_15m,
        prev_ema20_15m,
        latest_ema50_15m,
        latest_ema20_1h,
        latest_ema50_1h,
        latest_rsi,
        latest_atr,
        candle_high,
        candle_low,
        previous_high,
        previous_low,
    ]

    if not all(is_valid_number(value) for value in values):
        return None

    if latest_atr <= 0:
        return None

    if not live_price["tradeable"]:
        return None

    if live_price["spread"] > latest_atr * MAX_SPREAD_TO_ATR_RATIO:
        return None

    trend_gap = abs(latest_ema20_15m - latest_ema50_15m)
    if trend_gap < latest_atr * 0.25:
        return None

    signal_type = None

    if (
        latest_price > latest_ema20_15m > latest_ema50_15m
        and float(close_1h.iloc[-1]) > latest_ema20_1h > latest_ema50_1h
        and 52 <= latest_rsi <= 68
        and prev_price <= prev_ema20_15m * 1.0015
        and candle_high > previous_high
    ):
        signal_type = "BUY"

    elif (
        latest_price < latest_ema20_15m < latest_ema50_15m
        and float(close_1h.iloc[-1]) < latest_ema20_1h < latest_ema50_1h
        and 32 <= latest_rsi <= 48
        and prev_price >= prev_ema20_15m * 0.9985
        and candle_low < previous_low
    ):
        signal_type = "SELL"

    if signal_type is None:
        return None

    entry = round(live_price["ask"] if signal_type == "BUY" else live_price["bid"], 5)
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
        "spread": live_price["spread"],
    }


def open_trade(signal):
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
    }
    trades.append(trade)
    last_trade_times[trade["pair"]] = time.time()
    save_open_state()

    send(
        f"SNIPER SIGNAL\n\n"
        f"{trade['pair']} {trade['type']}\n"
        f"Entry: {trade['entry']}\n"
        f"SL: {trade['sl']}\n"
        f"TP: {trade['tp']}\n"
        f"ATR: {signal['atr']}\n"
        f"RSI: {signal['rsi']}\n"
        f"Spread: {signal['spread']}\n"
        f"Lot Size: {signal['lot_size']}"
    )


def update_trade_status(trade, latest_price):
    global wins, losses

    risk = trade["risk_per_unit"]
    if risk <= 0:
        return

    if trade["type"] == "BUY":
        progress_r = (latest_price - trade["entry"]) / risk
        trailing_sl = round(latest_price - (risk * TRAILING_STOP_ATR_MULTIPLIER), 5)

        if progress_r >= BREAK_EVEN_TRIGGER_R and not trade["break_even_done"]:
            trade["sl"] = max(trade["sl"], trade["entry"])
            trade["break_even_done"] = True
            send(f"{trade['pair']} moved to break-even at {trade['sl']}")

        if trade["break_even_done"]:
            trade["sl"] = max(trade["sl"], trailing_sl)

        if latest_price >= trade["tp"]:
            trade["status"] = "WIN"
            wins += 1
            save_trade_result(trade, "WIN", latest_price)
            send(f"WIN {trade['pair']} at {round(latest_price, 5)}")

        elif latest_price <= trade["sl"]:
            trade["status"] = "LOSS"
            losses += 1
            save_trade_result(trade, "LOSS", latest_price)
            send(f"LOSS {trade['pair']} at {round(latest_price, 5)}")

    else:
        progress_r = (trade["entry"] - latest_price) / risk
        trailing_sl = round(latest_price + (risk * TRAILING_STOP_ATR_MULTIPLIER), 5)

        if progress_r >= BREAK_EVEN_TRIGGER_R and not trade["break_even_done"]:
            trade["sl"] = min(trade["sl"], trade["entry"])
            trade["break_even_done"] = True
            send(f"{trade['pair']} moved to break-even at {trade['sl']}")

        if trade["break_even_done"]:
            trade["sl"] = min(trade["sl"], trailing_sl)

        if latest_price <= trade["tp"]:
            trade["status"] = "WIN"
            wins += 1
            save_trade_result(trade, "WIN", latest_price)
            send(f"WIN {trade['pair']} at {round(latest_price, 5)}")

        elif latest_price >= trade["sl"]:
            trade["status"] = "LOSS"
            losses += 1
            save_trade_result(trade, "LOSS", latest_price)
            send(f"LOSS {trade['pair']} at {round(latest_price, 5)}")


def check_trades():
    for trade in trades:
        if trade["status"] != "OPEN":
            continue

        live_price = get_live_price(trade["symbol"])
        if live_price is None or not live_price["tradeable"]:
            continue

        latest_price = live_price["bid"] if trade["type"] == "BUY" else live_price["ask"]
        if not is_valid_number(latest_price):
            continue

        update_trade_status(trade, latest_price)

    save_open_state()


def scan_market():
    if not in_session():
        print("Market scan skipped: outside preferred session.")
        return

    if active_trade_count() >= MAX_ACTIVE_TRADES:
        print("Market scan skipped: max active trades reached.")
        return

    signals = []

    for name, symbol in pairs.items():
        if has_open_trade(name) or not cooldown_ready(name):
            continue

        signal = build_signal(name, symbol)
        if signal is None:
            continue

        signals.append(signal)

    if not signals:
        print("No valid signals this cycle.")
        return

    for signal in signals[: max(0, MAX_ACTIVE_TRADES - active_trade_count())]:
        open_trade(signal)


def send_performance():
    total = wins + losses
    winrate = (wins / total * 100) if total > 0 else 0
    open_count = active_trade_count()

    send(
        f"PERFORMANCE\n\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Open Trades: {open_count}\n"
        f"Win Rate: {round(winrate, 2)}%"
    )


def send_heartbeat():
    send(
        f"Bot is alive\n"
        f"Open trades: {active_trade_count()}\n"
        f"Time UTC: {now_utc().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def run_bot():
    setup_files()
    require_telegram_config()
    require_twelve_data_key()
    print("Bot started... Sniper improved mode")

    last_scan = 0
    last_heartbeat = 0
    last_report = 0
    last_trade_check = 0

    while True:
        try:
            now = time.time()

            if now - last_scan >= SCAN_INTERVAL_SECONDS:
                print("Scanning market...")
                scan_market()
                last_scan = now

            if now - last_trade_check >= TRADE_CHECK_INTERVAL_SECONDS:
                check_trades()
                last_trade_check = now

            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                send_heartbeat()
                print("Heartbeat sent")
                last_heartbeat = now

            if now - last_report >= REPORT_INTERVAL_SECONDS:
                send_performance()
                print("Performance report sent")
                last_report = now

            print("Bot running...")

        except Exception as exc:
            print(f"Error: {exc}")

        time.sleep(10)


run_bot()
