import os
from dataclasses import replace
from datetime import date
from typing import Dict, Optional, Set, Tuple

import multi_strategy_bot_fixed as bot


REPORT_TUNING_NOTES: Dict[str, str] = {
    "US500": "Disabled by default from report: -AED171.13, PF 0.34, win rate 33.3%",
    "EURUSD": "Disabled by default from report: -AED164.49, PF 0.45, win rate 35.7%",
    "GBPUSD": "Kept on reduced risk: -AED41.62, PF 0.82, skip Monday/Wednesday and cut stale trades early",
    "USDJPY": "Priority pair: +AED4.04, PF 1.33, best market in the report",
}


PAIR_ENABLED: Dict[str, bool] = {
    "US500": bot.env_bool("ENABLE_US500", False),
    "EURUSD": bot.env_bool("ENABLE_EURUSD", False),
    "GBPUSD": bot.env_bool("ENABLE_GBPUSD", True),
    "USDJPY": bot.env_bool("ENABLE_USDJPY", True),
}

PAIR_RISK_MULTIPLIER: Dict[str, float] = {
    "US500": bot.env_float("US500_RISK_MULT", 0.0),
    "EURUSD": bot.env_float("EURUSD_RISK_MULT", 0.0),
    "GBPUSD": bot.env_float("GBPUSD_RISK_MULT", 0.40),
    "USDJPY": bot.env_float("USDJPY_RISK_MULT", 0.75),
}

PAIR_EARLY_BE_R: Dict[str, float] = {
    "US500": bot.env_float("US500_EARLY_BE_R", 0.0),
    "EURUSD": bot.env_float("EURUSD_EARLY_BE_R", 0.0),
    "GBPUSD": bot.env_float("GBPUSD_EARLY_BE_R", 0.60),
    "USDJPY": bot.env_float("USDJPY_EARLY_BE_R", 0.70),
}

PAIR_REENTRY_EXIT_MINS: Dict[str, int] = {
    "US500": bot.env_int("US500_REENTRY_EXIT_MINS", 0),
    "EURUSD": bot.env_int("EURUSD_REENTRY_EXIT_MINS", 0),
    "GBPUSD": bot.env_int("GBPUSD_REENTRY_EXIT_MINS", 20),
    "USDJPY": bot.env_int("USDJPY_REENTRY_EXIT_MINS", 30),
}

PAIR_STALE_CUTOFF: Dict[str, Optional[Tuple[int, int]]] = {
    "US500": None,
    "EURUSD": None,
    "GBPUSD": (10, 30),
    "USDJPY": None,
}

PAIR_MIN_R_AT_CUTOFF: Dict[str, float] = {
    "US500": 0.0,
    "EURUSD": 0.0,
    "GBPUSD": bot.env_float("GBPUSD_MIN_R_AT_CUTOFF", 0.15),
    "USDJPY": 0.0,
}

PAIR_ALLOWED_WEEKDAYS: Dict[str, Set[int]] = {
    "US500": {0, 1, 2, 3, 4},
    "EURUSD": {0, 1, 2, 3, 4},
    "GBPUSD": {1, 3, 4},
    "USDJPY": {0, 1, 2, 3, 4},
}


def pair_enabled(pair: str) -> bool:
    return PAIR_ENABLED.get(pair, True) and PAIR_RISK_MULTIPLIER.get(pair, 0.0) > 0.0


def configure_report_tuning():
    bot.RISK_PERCENT = bot.env_float("RISK_PERCENT", 0.60)
    bot.DAILY_LOSS_LIMIT_PCT = bot.env_float("DAILY_LOSS_LIMIT_PCT", 1.50)
    bot.PAIR_DAILY_LOSS_LIMIT_PCT = bot.env_float("PAIR_DAILY_LOSS_LIMIT_PCT", 0.75)
    bot.MAX_ACTIVE_TRADES = bot.env_int("MAX_ACTIVE_TRADES", 1)
    bot.MAX_PORTFOLIO_HEAT_PCT = bot.env_float("MAX_PORTFOLIO_HEAT_PCT", 0.90)
    bot.TP1_PARTIAL_CLOSE_PCT = bot.clamp(bot.env_float("TP1_PARTIAL_CLOSE_PCT", 0.35), 0.0, 0.90)
    bot.TP1_SL_MODE = os.getenv("TP1_SL_MODE", "BREAKEVEN").strip().upper() or "BREAKEVEN"
    bot.TP2_MODE = os.getenv("TP2_MODE", "TRAIL").strip().upper() or "TRAIL"
    bot.MAX_SPREAD_TO_RANGE_PCT = bot.env_float("MAX_SPREAD_TO_RANGE_PCT", 0.12)
    bot.MAX_ENTRY_DISTANCE_PCT = bot.env_float("MAX_ENTRY_DISTANCE_PCT", 0.15)
    bot.ALLOW_CORRELATED_TRADES = bot.env_bool("ALLOW_CORRELATED_TRADES", False)

    tuned_pairs = {
        "USDJPY": replace(
            bot.PAIR_CONFIGS["USDJPY"],
            max_spread=bot.env_float("USDJPY_MAX_SPREAD", 0.020),
        ),
        "GBPUSD": replace(
            bot.PAIR_CONFIGS["GBPUSD"],
            trade_end=(11, 0),
            max_spread=bot.env_float("GBPUSD_MAX_SPREAD", 0.00018),
            buffer=bot.env_float("GBPUSD_BUFFER", 0.0004),
        ),
        "EURUSD": replace(
            bot.PAIR_CONFIGS["EURUSD"],
            trade_end=(11, 0),
            max_spread=bot.env_float("EURUSD_MAX_SPREAD", 0.00018),
            buffer=bot.env_float("EURUSD_BUFFER", 0.0004),
        ),
        "US500": replace(
            bot.PAIR_CONFIGS["US500"],
            max_spread=bot.env_float("US500_MAX_SPREAD", 1.8),
        ),
    }
    bot.PAIR_CONFIGS = {
        "USDJPY": tuned_pairs["USDJPY"],
        "GBPUSD": tuned_pairs["GBPUSD"],
        "EURUSD": tuned_pairs["EURUSD"],
        "US500": tuned_pairs["US500"],
    }


def tuned_validate_runtime_config():
    _BASE_VALIDATE_RUNTIME_CONFIG()
    if not any(pair_enabled(pair) for pair in bot.PAIR_CONFIGS):
        raise RuntimeError("All pairs are disabled. Enable at least one pair before running the bot.")


def tuned_calculate_position_size(pair: str, entry: float, sl: float, equity: float) -> float:
    meta = bot.instrument_meta_for_pair(pair)
    risk_pct = bot.RISK_PERCENT * PAIR_RISK_MULTIPLIER.get(pair, 0.0)
    if risk_pct <= 0:
        return 0.0
    stop_distance = abs(entry - sl)
    if stop_distance <= 0 or meta.point_value <= 0:
        return 0.0
    risk_cash = equity * (risk_pct / 100.0)
    raw_size = risk_cash / (stop_distance * meta.point_value)
    rounded = bot.round_size(raw_size, meta.size_step)
    clamped = bot.clamp(rounded, meta.min_size, meta.max_size)
    return bot.round_size(clamped, meta.size_step)


def weekday_allowed(pair: str) -> bool:
    return bot.now_utc().weekday() in PAIR_ALLOWED_WEEKDAYS.get(pair, {0, 1, 2, 3, 4})


def tuned_signal_for_pair(pair: str, live: Dict[str, float], trade_date: str):
    if not pair_enabled(pair):
        return None
    if not weekday_allowed(pair):
        return None
    return _BASE_SIGNAL_FOR_PAIR(pair, live, trade_date)


def live_profit_r(trade: Dict[str, object], price: float) -> float:
    risk = float(trade["risk_per_unit"])
    if risk <= 0:
        return 0.0
    if trade["direction"] == "BUY":
        return (price - float(trade["entry"])) / risk
    return (float(trade["entry"]) - price) / risk


def stale_cutoff_due(trade: Dict[str, object]) -> bool:
    cutoff = PAIR_STALE_CUTOFF.get(str(trade["pair"]))
    if not cutoff:
        return False
    trade_day = date.fromisoformat(str(trade["trade_date"]))
    return bot.now_utc() >= bot.combine_utc(trade_day, cutoff)


def breakout_reentered(trade: Dict[str, object], price: float) -> bool:
    if trade["direction"] == "BUY":
        return price <= float(trade["range_high"])
    return price >= float(trade["range_low"])


def maybe_move_to_early_be(trade: Dict[str, object]):
    trigger_r = PAIR_EARLY_BE_R.get(str(trade["pair"]), 0.0)
    if trigger_r <= 0 or trade.get("tp1_locked") or trade.get("break_even_done"):
        return
    live = bot.broker.get_live_price(str(trade["epic"]))
    if not live:
        return
    price = live["bid"] if trade["direction"] == "BUY" else live["ask"]
    current_r = live_profit_r(trade, price)
    if current_r < trigger_r:
        return
    entry = float(trade["entry"])
    current_sl = float(trade["sl"])
    better = entry > current_sl if trade["direction"] == "BUY" else entry < current_sl
    if not better:
        return
    trade["sl"] = bot.round_price(entry, bot.PAIR_CONFIGS[str(trade["pair"])].decimals)
    trade["break_even_done"] = True
    bot.db_update_trade(trade)
    bot.db_log_event(int(trade["id"]), "EARLY_BE", float(trade["sl"]), f"trigger={trigger_r:.2f}R")
    bot.update_broker_sl(trade, float(trade["sl"]))
    bot.send_telegram(
        f"Early BE | {trade['pair']} {trade['direction']}\n"
        f"Triggered at {current_r:.2f}R, SL moved to entry {trade['sl']}"
    )


def tuned_check_open_trades(live_prices: Dict[str, Dict[str, float]]):
    for trade in list(bot.open_trades):
        if trade["status"] != "OPEN":
            continue
        live = live_prices.get(str(trade["pair"]))
        if not live:
            continue

        pair = str(trade["pair"])
        price = live["bid"] if trade["direction"] == "BUY" else live["ask"]
        maybe_move_to_early_be(trade)

        stop_hit = (trade["direction"] == "BUY" and price <= trade["sl"]) or (
            trade["direction"] == "SELL" and price >= trade["sl"]
        )
        if stop_hit:
            exit_price = bot.round_price(float(trade["sl"]), bot.PAIR_CONFIGS[pair].decimals)
            bot.finalize_trade(trade, exit_price, "SL", broker_close=True)
            continue

        if not trade.get("tp1_locked"):
            current_r = live_profit_r(trade, price)
            min_r = PAIR_MIN_R_AT_CUTOFF.get(pair, 0.0)
            if stale_cutoff_due(trade) and current_r < min_r:
                bot.finalize_trade(
                    trade,
                    bot.round_price(price, bot.PAIR_CONFIGS[pair].decimals),
                    "STALE_EXIT",
                    broker_close=True,
                )
                continue
            reentry_mins = PAIR_REENTRY_EXIT_MINS.get(pair, 0)
            trade_age_mins = (bot.time.time() - float(trade["opened_at"])) / 60.0
            if reentry_mins > 0 and trade_age_mins >= reentry_mins and breakout_reentered(trade, price):
                bot.finalize_trade(
                    trade,
                    bot.round_price(price, bot.PAIR_CONFIGS[pair].decimals),
                    "RANGE_REENTRY",
                    broker_close=True,
                )
                continue

        tp1_hit = (trade["direction"] == "BUY" and price >= trade["tp1"]) or (
            trade["direction"] == "SELL" and price <= trade["tp1"]
        )
        if tp1_hit and not trade.get("tp1_locked"):
            bot.handle_tp1(trade, live)

        tp2_hit = (trade["direction"] == "BUY" and price >= trade["tp2"]) or (
            trade["direction"] == "SELL" and price <= trade["tp2"]
        )
        if tp2_hit and not (bot.TP2_MODE == "TRAIL" and trade.get("tp2_locked")):
            bot.handle_tp2(trade, live)
            continue

        if bot.force_close_due(pair):
            bot.finalize_trade(
                trade,
                bot.round_price(price, bot.PAIR_CONFIGS[pair].decimals),
                "EOD_CLOSE",
                broker_close=True,
            )
    bot.open_trades[:] = [trade for trade in bot.open_trades if trade["status"] == "OPEN"]


def tuned_startup_report(start_balance: float):
    lines = []
    for pair, cfg in bot.PAIR_CONFIGS.items():
        meta = bot.market_meta[pair]
        status = "ENABLED" if pair_enabled(pair) else "DISABLED"
        risk_mult = PAIR_RISK_MULTIPLIER.get(pair, 0.0)
        lines.append(
            f"{pair} {status} | risk x{risk_mult:.2f} | end {cfg.trade_end[0]:02d}:{cfg.trade_end[1]:02d} UTC | {REPORT_TUNING_NOTES[pair]}"
        )
        lines.append(
            f"{pair} market | epic={bot.epics[pair]} point={meta.point_value:.6f} min={meta.min_size} step={meta.size_step}"
        )
    bot.send_telegram(
        f"Report-tuned bot started | {'DEMO' if bot.CAPITAL_DEMO else 'LIVE'}\n"
        f"Balance: {start_balance:.2f} {bot.account_currency}\n"
        f"Base risk: {bot.RISK_PERCENT}% | Max active trades: {bot.MAX_ACTIVE_TRADES}\n"
        f"Overall daily limit: {bot.DAILY_LOSS_LIMIT_PCT}% | Pair daily limit: {bot.PAIR_DAILY_LOSS_LIMIT_PCT}%\n"
        f"TP1 partial: {bot.TP1_PARTIAL_CLOSE_PCT * 100:.0f}% | TP1 mode: {bot.TP1_SL_MODE} | TP2 mode: {bot.TP2_MODE}\n"
        f"Breakout close confirmation: {bot.REQUIRE_BREAKOUT_CLOSE}\n"
        f"Report-driven tuning:\n" + "\n".join(lines)
    )


configure_report_tuning()

_BASE_VALIDATE_RUNTIME_CONFIG = bot.validate_runtime_config
_BASE_SIGNAL_FOR_PAIR = bot.signal_for_pair

bot.validate_runtime_config = tuned_validate_runtime_config
bot.calculate_position_size = tuned_calculate_position_size
bot.signal_for_pair = tuned_signal_for_pair
bot.check_open_trades = tuned_check_open_trades
bot.startup_report = tuned_startup_report


def main():
    bot.main()


if __name__ == "__main__":
    main()
