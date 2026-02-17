# Keltner Channel Strategy Backtester

A Python-based backtesting framework for **Keltner Channel** trading strategies applied to equity indices. Tests two distinct signal philosophies — **Mean Reversion** and **Breakout** — across multiple position modes, with Buy & Hold benchmarking and synthetic market simulations via Geometric Brownian Motion (GBM).

---

## Overview

The Keltner Channel is a volatility-based envelope built around an Exponential Moving Average (EMA), with upper and lower bands set at a multiple of the Average True Range (ATR). This project exploits that structure in two opposing ways:

- **Mean Reversion (MR):** Assumes prices that breach the channel will snap back — buy when price re-enters from below the lower band, sell when it re-enters from above the upper band.
- **Breakout (BO):** Assumes a band breach signals momentum — buy when price closes above the upper band, sell when it closes below the lower band.

---

## Features

- Downloads historical OHLCV data via `yfinance`
- Builds Keltner Channel (EMA ± ATR multiplier)
- Generates and visualises MR and BO signals on price charts
- Backtests all 6 strategy combinations: `{MR, BO} × {long_only, short_only, long_short}`
- Computes full performance metrics: Total Return, CAGR, Sharpe, Sortino, Volatility, Max Drawdown
- Benchmarks against Buy & Hold
- Plots equity curves, drawdown charts, and annual signal frequency
- Stress-tests strategies on simulated GBM markets (neutral, bull, bear)

---

## Requirements

```
pandas
numpy
yfinance
matplotlib
```

Install with:

```bash
pip install pandas numpy yfinance matplotlib
```

---

## Configuration

All key parameters are set at the top of the script:

| Parameter      | Default      | Description                              |
|----------------|--------------|------------------------------------------|
| `TICKER`       | `^NSEI`      | Yahoo Finance ticker (Nifty 50 index)    |
| `START`        | `2015-01-01` | Backtest start date                      |
| `END`          | `2024-12-31` | Backtest end date                        |
| `EMA_PERIOD`   | `20`         | Lookback period for the EMA midline      |
| `ATR_PERIOD`   | `20`         | Lookback period for the ATR calculation  |
| `ATR_MULT`     | `2`          | ATR multiplier for band width            |

---

## How It Works

### 1. Keltner Channel Construction

```
EMA  = Exponential Moving Average of Close (period = EMA_PERIOD)
ATR  = Average True Range (period = ATR_PERIOD)
Upper Band = EMA + ATR_MULT × ATR
Lower Band = EMA − ATR_MULT × ATR
```

### 2. Signal Logic

**Mean Reversion**

| Signal   | Condition                                                                 |
|----------|---------------------------------------------------------------------------|
| MR Buy   | Previous close < previous lower band **AND** current close > current lower band |
| MR Sell  | Previous close > previous upper band **AND** current close < current upper band |

**Breakout**

| Signal   | Condition                                                                 |
|----------|---------------------------------------------------------------------------|
| BO Buy   | Previous close ≤ previous upper band **AND** current close > upper band  |
| BO Sell  | Previous close ≥ previous lower band **AND** current close < lower band  |

### 3. Backtest Engine

The `backtest_keltner()` function supports three position modes:

- `long_only` — holds long or flat
- `short_only` — holds short or flat
- `long_short` — always holds a position (long or short)

Positions are determined on each bar and applied to the *next* bar's return (no look-ahead bias). A flat transaction cost of **0.1%** is applied whenever the position changes.

### 4. Performance Metrics

Each backtest reports:

| Metric         | Description                                        |
|----------------|----------------------------------------------------|
| Initial Capital | Starting portfolio value (default ₹1,00,000)     |
| Final Value    | Ending portfolio value                             |
| Total Return   | Overall percentage gain/loss                       |
| CAGR           | Compound Annual Growth Rate                        |
| Sharpe Ratio   | Risk-adjusted return (annualised, 252 trading days)|
| Sortino Ratio  | Downside risk-adjusted return                      |
| Volatility     | Annualised standard deviation of daily returns     |
| Max Drawdown   | Worst peak-to-trough decline                       |

### 5. GBM Simulation

Three synthetic markets are stress-tested using Geometric Brownian Motion:

| Scenario     | Drift (μ) | Volatility (σ) |
|--------------|-----------|----------------|
| Neutral      | 0.00      | 0.20           |
| Bull Market  | +0.08     | 0.20           |
| Bear Market  | −0.08     | 0.20           |

This isolates strategy behaviour from the specific characteristics of the historical dataset.

---

## Output

Running the script produces the following charts in sequence:

1. **Keltner Channel — Mean Reversion Signals** (price chart with buy/sell markers)
2. **Keltner Channel — Breakout Signals** (price chart with buy/sell markers)
3. **Equity Curve Comparison** — MR long_only vs BO long_only vs Buy & Hold
4. **Drawdown Chart** — MR long_only strategy
5. **Signal Frequency per Year** — bar chart comparing MR and BO signal counts
6. **Simulated Market Charts** — one per GBM scenario (neutral, bull, bear)

Console output includes per-strategy metrics for all 6 combinations, plus the Buy & Hold benchmark.

---

## Project Structure

```
keltner_backtest.py   # Main script — all logic in a single file
README.md
```

---

## Notes & Limitations

- **Slippage** is not modelled; only a flat transaction cost is applied.
- **Position sizing** is fixed (all-in or all-out). No fractional or risk-based sizing.
- **Short selling** in `short_only` / `long_short` modes assumes unrestricted shorting, which may not reflect real market conditions for retail traders.
- The ATR-based transaction cost approximation (`position.diff().abs() × cost`) is a simplification — it charges cost on position *magnitude change*, not notional trade value.
- Results on `^NSEI` (Nifty 50) reflect a specific index and time period; performance on other instruments or date ranges may differ significantly.
