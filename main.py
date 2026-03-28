import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import config
import matplotlib.pyplot as plt
from src.data       import download
from src.strategy   import build_keltner, generate_signals, signal_summary
from src.backtest   import run_backtest, run_buy_hold
from src.simulation import simulate_gbm, build_ohlcv_from_sim
from src.plotting   import (plot_keltner_signals, plot_equity_curves,
                             plot_drawdown, plot_signal_frequency)

def main():
    # 1. Data
    raw  = download(config.TICKER, config.START, config.END)
    data = build_keltner(raw, config.EMA_PERIOD, config.ATR_PERIOD, config.ATR_MULT)
    data = generate_signals(data)
    signal_summary(data)

    # 2. Backtest all combinations
    for strategy in ["MR", "BO"]:
        for mode in ["long_only", "short_only", "long_short"]:
            _, stats = run_backtest(data, strategy=strategy, mode=mode,
                                    initial_capital=config.INITIAL_CAPITAL,
                                    transaction_cost=config.TRANSACTION_COST)
            print(f"\n{strategy} – {mode}")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    # 3. Plots
    mr_df, _ = run_backtest(data, "MR", "long_only", config.INITIAL_CAPITAL)
    bo_df, _ = run_backtest(data, "BO", "long_only", config.INITIAL_CAPITAL)
    bh_df, _ = run_buy_hold(data, config.INITIAL_CAPITAL)

    plot_keltner_signals(data, config.EMA_PERIOD); plt.show()
    plot_equity_curves(mr_df, bo_df, bh_df);       plt.show()
    plot_drawdown(mr_df, "MR long-only");           plt.show()
    plot_signal_frequency(data);                    plt.show()

    # 4. GBM simulation
    for s in config.GBM_SCENARIOS:
        sim  = simulate_gbm(mu=s["mu"], sigma=s["sigma"],
                            days=config.GBM_DAYS, seed=config.GBM_SEED)
        df_s = build_ohlcv_from_sim(sim, seed=config.GBM_SEED)
        df_s = build_keltner(df_s, config.EMA_PERIOD, config.ATR_PERIOD, config.ATR_MULT)
        df_s = generate_signals(df_s)
        _, stats = run_backtest(df_s, "BO", "long_only", config.INITIAL_CAPITAL)
        print(f"\nGBM {s['label']}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
