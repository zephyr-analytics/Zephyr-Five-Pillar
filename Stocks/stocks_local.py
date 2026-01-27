import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from collections import defaultdict
import pandas_market_calendars as mcal

# ====================================================
# CONFIG
# ====================================================
START_DATE = "2010-01-01"
INITIAL_CASH = 100_000

STOCK_COUNT = 10
EMA_PERIOD = 210
MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOMENTUM_LOOKBACKS)

MAX_WEIGHT = 0.25
MAX_LEVERAGE = 1.0
USE_VOL_SCALING = True

CORR_LOOKBACK = 21
CORR_KILL_THRESHOLD = 0.75
CORR_FLOOR = 0.30
CORR_CEILING = 0.80

STOCKS_CSV = "stocks.csv"

# ====================================================
# HELPERS
# ====================================================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_momentum(px):
    return np.mean([
        px.iloc[-1] / px.iloc[-(lb + 1)] - 1
        for lb in MOMENTUM_LOOKBACKS
    ])

def average_pairwise_correlation(returns):
    if returns.shape[1] < 2:
        return None
    corr = returns.corr()
    vals = corr.values[np.triu_indices_from(corr, k=1)]
    return None if np.isnan(vals.mean()) else vals.mean()

def corr_scaler(avg_corr):
    if not USE_VOL_SCALING or avg_corr is None:
        return 1.0

    if avg_corr <= CORR_FLOOR:
        scale = 1.0
    elif avg_corr >= CORR_CEILING:
        scale = 0.25
    else:
        scale = 1.0 - (avg_corr - CORR_FLOOR) / (CORR_CEILING - CORR_FLOOR)
        scale = max(0.25, scale)

    return min(MAX_LEVERAGE, scale)

# ====================================================
# LOAD UNIVERSE
# ====================================================
symbols = (
    pd.read_csv(STOCKS_CSV)["symbol"]
    .astype(str)
    .str.upper()
    .unique()
    .tolist()
)

# ====================================================
# DOWNLOAD DATA (TOTAL RETURN ≈ Adj Close)
# ====================================================
data = yf.download(
    symbols,
    start=START_DATE,
    auto_adjust=True,
    progress=False
)["Close"]

data = data.dropna(axis=1, how="all")

# ====================================================
# CALENDAR
# ====================================================
nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(start_date=START_DATE, end_date=data.index[-1])
trading_days = schedule.index

month_ends = (
    pd.Series(trading_days)
    .groupby(pd.Series(trading_days).dt.to_period("M"))
    .last()
)

# ====================================================
# PORTFOLIO STATE
# ====================================================
weights = {}
kill_switch_active = False

# ====================================================
# BACKTEST LOOP
# ====================================================
for date in trading_days:
    if date not in data.index:
        continue

    # -----------------------------
    # DAILY CORRELATION KILL SWITCH
    # -----------------------------
    if weights and not kill_switch_active:
        active = list(weights.keys())
        if len(active) >= 2:
            hist = data.loc[:date, active].tail(CORR_LOOKBACK + 1)
            rets = hist.pct_change().dropna()
            avg_corr = average_pairwise_correlation(rets)

            if avg_corr is not None and avg_corr >= CORR_KILL_THRESHOLD:
                weights = {}
                kill_switch_active = True
                print(
                    f"{date.date()} | CORR KILL | "
                    f"AvgCorr={avg_corr:.2f} → CASH (until rebalance)"
                )

    # -----------------------------
    # MONTHLY REBALANCE
    # -----------------------------
    if date not in month_ends.values:
        continue

    if kill_switch_active:
        print(
            f"{date.date()} | REBALANCE WINDOW | Kill switch reset"
        )
        kill_switch_active = False

    hist = data.loc[:date].tail(MAX_LOOKBACK + EMA_PERIOD + 1)
    if len(hist) < MAX_LOOKBACK + EMA_PERIOD:
        continue

    scores = {}

    for s in hist.columns:
        px = hist[s]
        if len(px) < MAX_LOOKBACK + 1:
            continue

        mom = compute_momentum(px.tail(MAX_LOOKBACK + 1))
        if mom <= 0:
            continue

        ema_val = ema(px, EMA_PERIOD).iloc[-1]
        if px.iloc[-1] <= ema_val:
            continue

        scores[s] = mom

    if not scores:
        continue

    top = sorted(scores, key=scores.get, reverse=True)[:STOCK_COUNT]

    # -----------------------------
    # CORRELATION SCALING
    # -----------------------------
    corr_hist = hist[top].tail(CORR_LOOKBACK + 1)
    corr_rets = corr_hist.pct_change().dropna()
    avg_corr = average_pairwise_correlation(corr_rets)
    scale = corr_scaler(avg_corr)

    # -----------------------------
    # RAW MOMENTUM WEIGHTS
    # -----------------------------
    raw = {s: scores[s] for s in top}
    total = sum(raw.values())
    base = {s: raw[s] / total for s in raw}

    # -----------------------------
    # APPLY MAX WEIGHT CAP
    # -----------------------------
    capped = base.copy()
    excess = 0.0

    for s, w in capped.items():
        if w > MAX_WEIGHT:
            excess += w - MAX_WEIGHT
            capped[s] = MAX_WEIGHT

    if excess > 0:
        uncapped = {s: w for s, w in capped.items() if w < MAX_WEIGHT}
        uncapped_total = sum(uncapped.values())
        if uncapped_total > 0:
            for s in uncapped:
                capped[s] += excess * (uncapped[s] / uncapped_total)

    final = {s: w * scale for s, w in capped.items()}
    weights = final

    total_alloc = sum(final.values())
    cash_weight = max(0.0, 1.0 - total_alloc)

    weights_str = ", ".join(
        f"{s}:{w:.2%}" for s, w in final.items()
    )

    print(
        f"{date.date()} | n={len(final)} | "
        f"AvgCorr={avg_corr:.2f} | Scale={scale:.2f} | "
        f"CASH:{cash_weight:.2%} | {weights_str}"
    )
