# Multi-Strategy Professional Trading Bot

Python trading bot for Capital.com with Telegram alerts and three built-in strategies.

## Strategies

- `US500`: Opening Range Breakout with pre-market bias filter
- `EURUSD` / `GBPUSD`: Asian Range Breakout at London open
- `USDJPY`: Previous-day high/low breakout

## Features

- One trade per pair per day
- Internal TP1 and TP2 management with stop-loss promotion
- Daily loss limit enforcement
- Startup reconciliation between broker positions and local SQLite state
- Pair-aware news filter so FX news only blocks relevant instruments
- Telegram trade, heartbeat, and warning notifications

## Files

- `strategy_v1.py`: main bot
- `multi_state.db`: local SQLite state database
- `multi_results.csv`: trade results log
- `multi_bot.log`: runtime log

## Requirements

- Python 3.10+
- Capital.com API credentials
- Telegram bot token and chat ID

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` and fill in your real values.

Required:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `CAPITAL_API_KEY`
- `CAPITAL_LOGIN`
- `CAPITAL_PASSWORD`

Important runtime variables:

- `CAPITAL_DEMO=true` to stay on demo
- `RISK_PERCENT=1.0`
- `DAILY_LOSS_LIMIT_PCT=3.0`
- `NEWS_BUFFER_MINS=30`
- `DATA_DIR=./data` locally or `/data` on Railway with a mounted volume

## Local Run

```bash
python strategy_v1.py
```

## Railway Deploy Notes

1. Create a persistent volume.
2. Mount it and set `DATA_DIR=/data`.
3. Set the start command to:

```bash
python strategy_v1.py
```

4. Add all environment variables from `.env.example`.
5. Keep `CAPITAL_DEMO=true` until you have verified behavior for an extended period.

## Safety Notes

- Do not commit your real `.env` file.
- Do not trust local SQLite persistence on Railway without a mounted volume.
- Untracked broker positions are intentionally blocked from new entries until you manually sync or close them.
- Position sizing still depends on your broker contract assumptions; review risk carefully before going live.
