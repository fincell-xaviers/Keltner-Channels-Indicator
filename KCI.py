#Import libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

#Params
TICKER = "^NSEI"
START = "2015-01-01"
END = "2024-12-31"
EMA_PERIOD = 20
ATR_PERIOD = 20
ATR_MULT = 2

#Download data
df = yf.download(TICKER, start=START, end=END)
if df.empty:
    raise SystemError("No data fetched -- check ticker or internet connection.")
df.columns = [c[0].capitalize() if isinstance(c, tuple) else c.capitalize() for c in df.columns]
df = df[['Open','High','Low','Close','Volume']].copy()

#EMA and ATR
def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()
def atr(df_in: pd.DataFrame, period: int=14):
    #True Range
    high_low = df_in['High'] - df_in['Low']
    high_close = (df_in['High'] - df_in['Close'].shift(1)).abs()
    low_close = (df_in['Low'] - df_in['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()

#Build Keltner Channel
def build_keltner(df_in: pd.DataFrame, ema_period=20, atr_period=20, atr_mult=2):
    df = df_in.copy()
    df['EMA'] = ema(df['Close'], ema_period)
    df['ATR'] = atr(df, atr_period)
    df['Upper'] = df['EMA'] + atr_mult * df['ATR']
    df['Lower'] = df['EMA'] - atr_mult * df['ATR']
    #Drop rows where the channel can't be computed yet
    df = df.dropna(subset=['EMA','ATR','Upper','Lower']).copy()
    return df

data = build_keltner(df, EMA_PERIOD, ATR_PERIOD, ATR_MULT)

#Reindex each series to the data.index so their indices are guaranteed identical
close = data['Close'].reindex(data.index)
upper = data['Upper'].reindex(data.index)
lower = data['Lower'].reindex(data.index)
ema_s = data['EMA'].reindex(data.index)
#Attach back to DataFrame
data = data.assign(Close=close, Upper=upper, Lower=lower, EMA=ema_s)

#Quick diagnostics
print("Rows:", len(data))
print("Columns present:", list(data.columns))
print("Any NaNs?:", data[['Close','Upper','Lower']].isna().any().to_dict())
#SIGNAL LOGIC
#Mean-reversion signals
#Buy when previous close was below prev lower AND current close closes back above current lower
#Sell when previous close was above prev upper AND current close closes back below current upper
data['MR_Buy']  = (data['Close'].shift(1) < data['Lower'].shift(1)) & (data['Close'] > data['Lower'])
data['MR_Sell'] = (data['Close'].shift(1) > data['Upper'].shift(1)) & (data['Close'] < data['Upper'])

#Breakout signals
#Buy when price closes above Upper (breakout)
#Sell when price closes below Lower.
data['BO_Buy']  = (data['Close'].shift(1) <= data['Upper'].shift(1)) & (data['Close'] > data['Upper'])
data['BO_Sell'] = (data['Close'].shift(1) >= data['Lower'].shift(1)) & (data['Close'] < data['Lower'])

#Print counts for sanity
print("Mean-reversion Buys:", data['MR_Buy'].sum(), "Sells:", data['MR_Sell'].sum())
print("Breakout Buys:", data['BO_Buy'].sum(), "Sells:", data['BO_Sell'].sum())

#Plotting both strategy plots
plt.figure(figsize=(14,10))

#Mean-Reversion subplot
ax1 = plt.subplot(2,1,1)
ax1.plot(data.index, data['Close'], label='Close', linewidth=1, alpha=0.6)
ax1.plot(data.index, data['EMA'], label=f'EMA({EMA_PERIOD})', linewidth=1)
ax1.plot(data.index, data['Upper'], label='Upper', linewidth=0.8)
ax1.plot(data.index, data['Lower'], label='Lower', linewidth=0.8)
ax1.fill_between(data.index, data['Lower'], data['Upper'], alpha=0.08)

mr_buys = data.index[data['MR_Buy']]
mr_sells = data.index[data['MR_Sell']]
ax1.scatter(mr_buys, data.loc[mr_buys, 'Close'], marker="^", s=10, label='MR Buy', zorder=5)
ax1.scatter(mr_sells, data.loc[mr_sells, 'Close'], marker="v", s=10, label='MR Sell', zorder=5)
ax1.set_title('Keltner Channel — Mean Reversion Signals')
ax1.legend(loc='upper left')

#Breakout subplot
ax2 = plt.subplot(2,1,2, sharex=ax1)
ax2.plot(data.index, data['Close'], label='Close', linewidth=1, alpha=0.6)
ax2.plot(data.index, data['EMA'], label=f'EMA({EMA_PERIOD})', linewidth=1)
ax2.plot(data.index, data['Upper'], label='Upper', linewidth=0.8)
ax2.plot(data.index, data['Lower'], label='Lower', linewidth=0.8)
ax2.fill_between(data.index, data['Lower'], data['Upper'], alpha=0.08)

bo_buys = data.index[data['BO_Buy']]
bo_sells = data.index[data['BO_Sell']]
ax2.scatter(bo_buys, data.loc[bo_buys, 'Close'], marker="^", s=10, label='BO Buy', zorder=5)
ax2.scatter(bo_sells, data.loc[bo_sells, 'Close'], marker="v", s=10, label='BO Sell', zorder=5)
ax2.set_title('Keltner Channel — Breakout Signals')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
def backtest_keltner(data, strategy="BO", mode="long_only",
                           initial_capital=100000, transaction_cost=0.001):
    """
    Returns: (df_with_equity, stats_dict, buyhold_df, buyhold_stats)
    Stats include: Initial Capital, Final Value, Total Return, CAGR, Sharpe, Sortino, Volatility, Max Drawdown
    """
    df = data.copy().sort_index()
    if strategy.upper() == "BO":
        buy_col, sell_col = "BO_Buy", "BO_Sell"
    else:
        buy_col, sell_col = "MR_Buy", "MR_Sell"

    df["returns"] = df["Close"].pct_change().fillna(0)
    df["position"] = 0

    position = 0
    for i in range(1, len(df)):
        if mode == "long_only":
            if df.iloc[i][buy_col]:
                position = 1
            elif df.iloc[i][sell_col]:
                position = 0
        elif mode == "short_only":
            if df.iloc[i][sell_col]:
                position = -1
            elif df.iloc[i][buy_col]:
                position = 0
        elif mode == "long_short":
            if df.iloc[i][buy_col]:
                position = 1
            elif df.iloc[i][sell_col]:
                position = -1
        df.iloc[i, df.columns.get_loc("position")] = position

    #Strategy daily returns
    #We use position.shift(1) to avoid look-ahead)
    df["strategy_ret"] = df["position"].shift(1).fillna(0) * df["returns"]

    #Transaction cost when position changes (entry or exit) approximated as fraction of portfolio on that day
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["strategy_ret"] = df["strategy_ret"] - df["trade"] * transaction_cost

    #Equity curve (starting capital)
    df["equity"] = (1 + df["strategy_ret"]).cumprod() * initial_capital

    #Metrics calculator
    def metrics_from_equity(equity_series, daily_returns):
        total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
        n_days = len(equity_series)
        cagr = (1 + total_return) ** (252 / n_days) - 1
        vol = daily_returns.std() * np.sqrt(252)
        sharpe = (np.sqrt(252) * daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else np.nan
        downside_std = np.std(daily_returns[daily_returns < 0]) if len(daily_returns[daily_returns < 0])>0 else 0.0
        sortino = (np.sqrt(252) * daily_returns.mean() / downside_std) if downside_std>0 else np.nan
        rolling_max = equity_series.cummax()
        drawdown = equity_series / rolling_max - 1
        max_dd = drawdown.min()
        return {
            "Initial Capital": f"₹{equity_series.iloc[0]:,.0f}",
            "Final Value": f"₹{equity_series.iloc[-1]:,.0f}",
            "Total Return": f"{total_return:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Sharpe Ratio": round(sharpe, 2) if not np.isnan(sharpe) else "-",
            "Sortino Ratio": round(sortino, 2) if not np.isnan(sortino) else "-",
            "Volatility": f"{vol:.2%}",
            "Max Drawdown": f"{max_dd:.2%}",
        }

    #Strategy metrics
    strat_stats = metrics_from_equity(df["equity"], df["strategy_ret"])

    #Buy & hold benchmark for same period (buy at first close, hold)
    bh = df[["Close"]].copy()
    bh["bh_ret"] = bh["Close"].pct_change().fillna(0)
    bh["bh_equity"] = (1 + bh["bh_ret"]).cumprod() * initial_capital
    bh_stats = metrics_from_equity(bh["bh_equity"], bh["bh_ret"])

    return df, strat_stats, bh, bh_stats
modes = ["long_only", "short_only", "long_short"]
results = {}
for strategy in ["MR", "BO"]:
    for m in modes:
        df_res, stats_res, bh_df, bh_stats = backtest_keltner(data, strategy=strategy, mode=m)
        results[(strategy,m)] = (df_res, stats_res, bh_stats)
        print(f"\n{strategy} - {m}")
        for k,v in stats_res.items():
            print(f"{k}: {v}")
print("\nBuy & Hold benchmark final:", bh_stats["Final Value"], "Total Return:", bh_stats["Total Return"])

#Example: compare MR long_only vs BO long_only vs BuyHold
mr_df, mr_stats, mr_bh, mr_bh_stats = backtest_keltner(data, strategy="MR", mode="long_only")
bo_df, bo_stats, bo_bh, bo_bh_stats = backtest_keltner(data, strategy="BO", mode="long_only")

plt.figure(figsize=(12,6))
plt.plot(mr_df.index, mr_df["equity"], label="MR (long_only)")
plt.plot(bo_df.index, bo_df["equity"], label="BO (long_only)")
plt.plot(mr_bh.index, mr_bh["bh_equity"], label="Buy & Hold", linestyle="--", color="gray")
plt.title("Equity Curve Comparison")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()
#Drawdown (MR long_only)
dd = mr_df["equity"] / mr_df["equity"].cummax() - 1
plt.figure(figsize=(12,3))
plt.fill_between(dd.index, dd, 0, color="red", alpha=0.3)
plt.title("Drawdown — MR (long_only)")
plt.show()
def plot_signal_frequency(data):
    """
    Bar chart of number of Buy/Sell signals per year for both MR and BO.
    """
    df = data.copy()
    df["Year"] = df.index.year

    mr_buys = df.groupby("Year")["MR_Buy"].sum()
    mr_sells = df.groupby("Year")["MR_Sell"].sum()
    bo_buys = df.groupby("Year")["BO_Buy"].sum()
    bo_sells = df.groupby("Year")["BO_Sell"].sum()

    plt.figure(figsize=(12,5))
    width = 0.2
    years = np.array(sorted(df["Year"].unique()))

    plt.bar(years - width*1.5, mr_buys, width, label="MR Buys", color="skyblue")
    plt.bar(years - width*0.5, mr_sells, width, label="MR Sells", color="lightcoral")
    plt.bar(years + width*0.5, bo_buys, width, label="BO Buys", color="orange")
    plt.bar(years + width*1.5, bo_sells, width, label="BO Sells", color="green")

    plt.title("Signal Frequency per Year — MR vs BO")
    plt.xlabel("Year")
    plt.ylabel("Number of Signals")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

plot_signal_frequency(data)
def backtest_buy_hold(data, initial_capital=100000, rf_rate=0.06):
    """
    Computes Buy & Hold performance metrics on a price series (data['Close']).
    Uses same logic & metrics format as Keltner backtests.
    """
    df = data.copy()
    df = df.sort_index()

    df['Daily_Return'] = df['Close'].pct_change().fillna(0)
    df['Equity'] = (1 + df['Daily_Return']).cumprod() * initial_capital

    #Metrics
    total_return = df['Equity'].iloc[-1] / initial_capital - 1
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (df['Equity'].iloc[-1] / initial_capital) ** (1 / years) - 1

    #Risk metrics
    daily_rf = rf_rate / 252
    excess_daily = df['Daily_Return'] - daily_rf / 252
    sharpe = np.sqrt(252) * (df['Daily_Return'].mean() - daily_rf/252) / df['Daily_Return'].std(ddof=0)
    downside = df.loc[df['Daily_Return'] < 0, 'Daily_Return']
    sortino = np.sqrt(252) * (df['Daily_Return'].mean() - daily_rf/252) / downside.std(ddof=0)

    vol = df['Daily_Return'].std() * np.sqrt(252)

    #Drawdown
    rolling_max = df['Equity'].cummax()
    dd = df['Equity'] / rolling_max - 1
    max_dd = dd.min()

    #Store metrics
    stats = {
        "Initial Capital": f"₹{initial_capital:,.0f}",
        "Final Value": f"₹{df['Equity'].iloc[-1]:,.0f}",
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Volatility": f"{vol:.2%}",
        "Max Drawdown": f"{max_dd:.2%}",
    }

    return df, stats

bh_df, bh_stats = backtest_buy_hold(data)
print("Buy & Hold Performance:\n")
for k, v in bh_stats.items():
    print(f"{k}: {v}")

def simulate_gbm(mu=0.0, sigma=0.2, S0=100, days=2520, seed=42):
    """
    Simulates a price series using Geometric Brownian Motion (approx 10 years of daily data)
    """
    np.random.seed(seed)
    dt = 1/252
    prices = [S0]
    for _ in range(days-1):
        prices.append(prices[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal()))
    return pd.Series(prices, name="SimPrice")


def generate_keltner_signals(df):
    """
    Adds MR and BO signals to a DataFrame that already has Upper/Lower/Close/EMA
    """
    df['MR_Buy']  = (df['Close'].shift(1) < df['Lower'].shift(1)) & (df['Close'] > df['Lower'])
    df['MR_Sell'] = (df['Close'].shift(1) > df['Upper'].shift(1)) & (df['Close'] < df['Upper'])
    df['BO_Buy']  = (df['Close'].shift(1) <= df['Upper'].shift(1)) & (df['Close'] > df['Upper'])
    df['BO_Sell'] = (df['Close'].shift(1) >= df['Lower'].shift(1)) & (df['Close'] < df['Lower'])
    return df


def test_strategy_on_sim(mu, sigma, title):
    """
    Simulate market -> Build Keltner -> Generate signals -> Backtest -> Plot results
    """
    sim = simulate_gbm(mu, sigma)
    df_sim = pd.DataFrame({
        "Close": sim,
        "High": sim * (1 + 0.01*np.random.rand(len(sim))),
        "Low": sim * (1 - 0.01*np.random.rand(len(sim))),
        "Open": sim.shift(1).fillna(sim.iloc[0]),
        "Volume": np.random.randint(1000,2000,len(sim))
    })

    #Build Keltner Channel
    sim_data = build_keltner(df_sim)

    #Generate signals
    sim_data = generate_keltner_signals(sim_data)

    #Backtest (choose strategy & mode)
    sim_data, stats_sim, _, bh_stats_sim = backtest_keltner(sim_data, strategy="BO", mode="long_only")

    print(f"\nSimulation ({title}) Results:")
    for k, v in stats_sim.items():
        print(f"{k}: {v}")

    #Plot the simulated market
    plt.figure(figsize=(10,5))
    plt.plot(sim_data.index, sim_data["Close"], label="Simulated Price", alpha=0.7)
    plt.title(f"Simulated Market ({title}) — GBM μ={mu}, σ={sigma}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return stats_sim, bh_stats_sim
#Run 3 market simulations (Neutral, Bull, Bear)
neutral_stats, neutral_bh = test_strategy_on_sim(mu=0.0, sigma=0.2, title="Neutral Drift")
bull_stats, bull_bh = test_strategy_on_sim(mu=0.08, sigma=0.2, title="Positive Drift (Bull Market)")
bear_stats, bear_bh = test_strategy_on_sim(mu=-0.08, sigma=0.2, title="Negative Drift (Bear Market)")
