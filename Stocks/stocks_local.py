import pandas as pd
import numpy as np
import yfinance as yf

# =============================
# CONFIG (same as QC)
# =============================
STOCK_COUNT = 10
EMA_PERIOD = 210
MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOMENTUM_LOOKBACKS)

MAX_WEIGHT = 0.25

CORR_LOOKBACK = 21
CORR_KILL_THRESHOLD = 0.65
CORR_FLOOR = 0.45
CORR_CEILING = 0.40

STOCKS_CSV = "stocks.csv"

# =============================
# HELPERS
# =============================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_momentum(px):
    return np.mean([
        px.iloc[-1] / px.iloc[-(lb + 1)] - 1
        for lb in MOMENTUM_LOOKBACKS
    ])

def avg_pairwise_corr(returns):
    if returns.shape[1] < 2:
        return None
    c = returns.corr()
    v = c.values[np.triu_indices_from(c, k=1)]
    return None if np.isnan(v.mean()) else v.mean()

def corr_scaler(avg_corr):
    if avg_corr is None:
        return 1.0

    safe = CORR_FLOOR      # e.g. 0.40–0.45
    danger = CORR_CEILING  # e.g. 0.50–0.55
    min_scale = 0.25

    # Full exposure in safe zone
    if avg_corr <= safe:
        return 1.0

    # Minimum exposure in danger zone
    if avg_corr >= danger:
        return min_scale

    # Fast convex decay between safe and danger
    x = (avg_corr - safe) / (danger - safe)
    scale = 1.0 - x * x   # convex penalty

    return max(min_scale, scale)

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

# =============================
# LOAD UNIVERSE
# =============================
symbols = pd.read_csv(STOCKS_CSV)["Symbol"].str.upper().tolist()

# =============================
# FETCH DATA (TOTAL RETURN ≈ Adj Close) — CHUNKED & SAFE
# =============================
def fetch_adj_close(symbols, period="500d", chunk_size=100):
    frames = []

    for batch in chunked(symbols, chunk_size):
        df = yf.download(
            batch,
            period=period,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
        else:
            close = df.to_frame(name=batch[0])

        frames.append(close)

    return (
        pd.concat(frames, axis=1)
        .sort_index()
        .dropna(axis=1)
        if frames else pd.DataFrame()
    )

prices = fetch_adj_close(symbols)

# =============================
# SIGNAL COMPUTATION (LATEST ONLY)
# =============================
scores = {}

for s in prices.columns:
    px = prices[s]

    if len(px) < MAX_LOOKBACK + EMA_PERIOD:
        continue

    mom = compute_momentum(px.tail(MAX_LOOKBACK + 1))
    if mom <= 0:
        continue

    ema_val = ema(px, EMA_PERIOD).iloc[-1]
    if px.iloc[-1] <= ema_val:
        continue

    scores[s] = mom

if not scores:
    print("No eligible stocks.")
    exit()

top = sorted(scores, key=scores.get, reverse=True)[:STOCK_COUNT]

# =============================
# CORRELATION CHECK
# =============================
rets = prices[top].pct_change().dropna().tail(CORR_LOOKBACK)
avg_corr = avg_pairwise_corr(rets)

if avg_corr is not None and avg_corr >= CORR_KILL_THRESHOLD:
    print(
        f"KILL SWITCH ACTIVE | AvgCorr={avg_corr:.2f} → CASH"
    )
    exit()

# =============================
# WEIGHTS
# =============================
raw = {s: scores[s] for s in top}
total = sum(raw.values())
weights = {s: raw[s] / total for s in raw}

# Apply cap
excess = 0.0
for s in weights:
    if weights[s] > MAX_WEIGHT:
        excess += weights[s] - MAX_WEIGHT
        weights[s] = MAX_WEIGHT

if excess > 0:
    uncapped = {s: w for s, w in weights.items() if w < MAX_WEIGHT}
    tot = sum(uncapped.values())
    for s in uncapped:
        weights[s] += excess * (uncapped[s] / tot)

# Scale
scale = corr_scaler(avg_corr)
weights = {s: w * scale for s, w in weights.items()}

# =============================
# OUTPUT
# =============================
cash = 1.0 - sum(weights.values())

print(f"AvgCorr={avg_corr:.2f} | Scale={scale:.2f}")
print(f"CASH: {cash:.2%}")

for s, w in weights.items():
    print(f"{s}: {w:.2%}")
