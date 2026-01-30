import pandas as pd
import numpy as np
import yfinance as yf

# =============================
# CONFIG (QC-MATCHED)
# =============================
STOCK_COUNT = 10
EMA_PERIOD = 210
MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOMENTUM_LOOKBACKS)

WINRATE_LOOKBACK = 126
VOL_LOOKBACK = 126

MAX_WEIGHT = 0.25

CORR_LOOKBACK = 21
CORR_KILL_THRESHOLD = 0.65

# QC correlation scaling
CORR_Z_LOOKBACK = 252
MIN_SCALE = 0.40
CORR_K = 1.0
SCALE_RECOVERY_MONTHS = 6

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
    if returns is None or returns.shape[1] < 2:
        return None
    c = returns.corr()
    v = c.values[np.triu_indices_from(c, k=1)]
    m = v.mean()
    return None if np.isnan(m) else m

def compute_avg_corr_from_prices(prices_df, symbols, end_idx, lookback):
    syms = [s for s in symbols if s in prices_df.columns]
    if len(syms) < 2:
        return None

    start = end_idx - (lookback + 1)
    if start < 0:
        return None

    window = prices_df.iloc[start:end_idx + 1][syms]
    rets = window.pct_change(fill_method=None).dropna()
    rets = rets.dropna(axis=1)
    if rets.shape[1] < 2:
        return None

    return avg_pairwise_corr(rets)

def qc_corr_scaler(avg_corr, corr_history):
    if avg_corr is None or len(corr_history) < 5:
        return 1.0

    mu = np.mean(corr_history)
    sigma = np.std(corr_history)
    if sigma <= 1e-6:
        return 1.0

    z = max(0.0, (avg_corr - mu) / sigma)
    return MIN_SCALE + (1.0 - MIN_SCALE) * np.exp(-CORR_K * z)

def apply_weight_cap(weights, max_weight):
    weights = weights.copy()

    while True:
        excess = 0.0
        free = {}

        for s, w in weights.items():
            if w > max_weight:
                excess += w - max_weight
                weights[s] = max_weight
            else:
                free[s] = w

        if excess <= 1e-8 or not free:
            break

        free_total = sum(free.values())
        redistributed = False

        for s in list(free.keys()):
            add = excess * (free[s] / free_total)
            new_w = weights[s] + add
            if new_w > max_weight:
                weights[s] = max_weight
                redistributed = True
            else:
                weights[s] = new_w

        if not redistributed:
            break

    total = sum(weights.values())
    if total > 0:
        for s in weights:
            weights[s] /= total

    return weights

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

# =============================
# QC PER-ASSET EDGE (ADDED)
# =============================
def compute_edge(px, momentum):
    simple_returns = px.pct_change().dropna()
    log_returns = np.log1p(simple_returns)

    if len(log_returns) < max(WINRATE_LOOKBACK, VOL_LOOKBACK):
        return None

    win_rate = float(np.mean(log_returns.tail(WINRATE_LOOKBACK) > 0))

    vol = float(np.std(log_returns.tail(VOL_LOOKBACK)) * np.sqrt(252))
    if not np.isfinite(vol) or vol <= 0:
        return None

    confidence = np.clip(abs(momentum) / (vol + 1e-6), 0.0, 2.0)
    scale = max(0.1, 1.0 + confidence * np.sign(momentum))

    edge = win_rate * scale
    return edge if np.isfinite(edge) and edge > 0 else None

# =============================
# LOAD UNIVERSE
# =============================
symbols = pd.read_csv(STOCKS_CSV)["Symbol"].astype(str).str.upper().tolist()

# =============================
# FETCH DATA (TOTAL RETURN ≈ QC)
# =============================
def fetch_adj_close(symbols, period="1200d", chunk_size=100):
    frames = []
    for batch in chunked(symbols, chunk_size):
        df = yf.download(batch, period=period, auto_adjust=True, progress=False)
        if df.empty:
            continue
        close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df.to_frame(name=batch[0])
        frames.append(close)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1).sort_index()

prices = fetch_adj_close(symbols)

if prices.empty:
    raise SystemExit("No price data downloaded.")

prices = prices.loc[:, [c for c in prices.columns if c in set(symbols)]]

ema_df = prices.apply(lambda s: ema(s, EMA_PERIOD))

# =============================
# QC-FAITHFUL EVENT SIMULATION
# =============================
corr_history = []
kill_switch_active = False
current_scale = 1.0

holdings = {}
hold_syms = []

dates = prices.index
month_end_dates = dates.to_series().groupby([dates.year, dates.month]).max().values
month_end_set = set(month_end_dates)

min_bars = MAX_LOOKBACK + EMA_PERIOD + 5

for t_i, dt in enumerate(dates):
    if t_i < min_bars:
        continue

    # -------------------------
    # DailyRiskCheck
    # -------------------------
    if not kill_switch_active and len(hold_syms) >= 2:
        avg_corr_inv = compute_avg_corr_from_prices(prices, hold_syms, t_i, CORR_LOOKBACK)
        if avg_corr_inv is not None and avg_corr_inv >= CORR_KILL_THRESHOLD:
            holdings = {}
            hold_syms = []
            kill_switch_active = True

    # -------------------------
    # Monthly Rebalance
    # -------------------------
    if dt not in month_end_set:
        continue

    scores = {}

    for s in prices.columns:
        px = prices[s].iloc[:t_i + 1]
        if px.isna().iloc[-1]:
            continue

        if len(px.dropna()) < MAX_LOOKBACK + 1:
            continue

        tail_px = px.dropna().iloc[-(MAX_LOOKBACK + 1):]
        if len(tail_px) < MAX_LOOKBACK + 1:
            continue

        mom = compute_momentum(tail_px)
        if not np.isfinite(mom) or mom <= 0:
            continue

        ema_val = ema_df[s].iloc[t_i]
        if not np.isfinite(ema_val):
            continue

        if tail_px.iloc[-1] <= ema_val:
            continue

        scores[s] = mom

    if not scores:
        continue

    top = sorted(scores, key=scores.get, reverse=True)[:STOCK_COUNT]

    avg_corr_top = compute_avg_corr_from_prices(prices, top, t_i, CORR_LOOKBACK)

    if avg_corr_top is not None and np.isfinite(avg_corr_top):
        corr_history.append(avg_corr_top)
        if len(corr_history) > CORR_Z_LOOKBACK:
            corr_history.pop(0)

    if kill_switch_active:
        if avg_corr_top is None or avg_corr_top >= CORR_KILL_THRESHOLD:
            continue
        kill_switch_active = False

    target_scale = qc_corr_scaler(avg_corr_top, corr_history)

    if target_scale < current_scale:
        current_scale = target_scale
    else:
        alpha = 1.0 / SCALE_RECOVERY_MONTHS
        current_scale += alpha * (target_scale - current_scale)

    # -------------------------
    # QC EDGE-BASED WEIGHTS
    # -------------------------
    edges = {}

    for s in top:
        px = prices[s].iloc[:t_i + 1].dropna()
        edge = compute_edge(px, scores[s])
        if edge is not None:
            edges[s] = edge

    if not edges:
        continue

    total_edge = sum(edges.values())
    base_weights = {s: w / total_edge for s, w in edges.items()}

    weights = apply_weight_cap(base_weights, MAX_WEIGHT)
    final_weights = {s: w * current_scale for s, w in weights.items()}

    holdings = final_weights
    hold_syms = list(final_weights.keys())

# =============================
# OUTPUT (FINAL STATE)
# =============================
total_alloc = sum(holdings.values()) if holdings else 0.0
cash = max(0.0, 1.0 - total_alloc)

print(f"Scale={current_scale:.2f} | KillSwitch={'ON' if kill_switch_active else 'OFF'}")
print(f"CASH: {cash:.2%}")

if holdings:
    for s, w in sorted(holdings.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{s}: {w:.2%}")
else:
    print("No holdings (100% cash).")
