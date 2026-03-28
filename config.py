TICKER = "^NSEI"
START  = "2015-01-01"
END    = "2024-12-31"

EMA_PERIOD = 20
ATR_PERIOD = 20
ATR_MULT   = 2

INITIAL_CAPITAL  = 100_000
TRANSACTION_COST = 0.001
RISK_FREE_RATE   = 0.06

GBM_SEED = 42
GBM_DAYS = 2520
GBM_SCENARIOS = [
    {"mu":  0.00, "sigma": 0.20, "label": "Neutral Drift"},
    {"mu":  0.08, "sigma": 0.20, "label": "Bull Market"},
    {"mu": -0.08, "sigma": 0.20, "label": "Bear Market"},
]
```

---

**`requirements.txt`**
```
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
pytest>=8.0.0
```

---

**`.gitignore`**
```
__pycache__/
*.pyc
.env
.venv
results/
*.png
*.csv
.ipynb_checkpoints/
.pytest_cache/
