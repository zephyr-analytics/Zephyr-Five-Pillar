import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ============================
# USER CONFIG
# ============================
CSV_PATH = "stocks.csv"
START_LOOKBACK = 300
GROUP_VOL_TARGET = 0.10
CRYPTO_CAP = 0.10

ENABLE_STOCKS = True
ENABLE_SECTORS = True
ENABLE_SMA_FILTER = True

WINRATE_LOOKBACK = 63
VOL_LOOKBACK = 63
SMA_PERIOD = 168
BOND_SMA_PERIOD = 126
STOCK_EMA_PERIOD = 189

MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOMENTUM_LOOKBACKS)

SECTOR_COUNT = 4
STOCK_COUNT = 5

# ============================
# STATIC GROUPS (UNCHANGED)
# ============================
GROUPS = {
    "real": ["GLD", "DBC"],
    "corp_bonds": ["VCSH", "VCIT", "VCLT"],
    "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
    "high_yield_bonds": ["SHYG", "HYG"],
    "equities": ["VTI", "VEA", "VWO"],
    "sectors": [
        "IXP","RXI","KXI","IXC","IXG","IXJ",
        "EXI","IXN","MXI","REET","JXI"
    ],
    "crypto": ["IBIT", "ETHA"],
    "cash": ["BIL"]
}

# ============================
# LOAD STOCK UNIVERSE
# ============================
stock_universe = pd.read_csv(CSV_PATH)["Symbol"].str.upper().tolist()

# ============================
# FETCH DATA
# ============================
all_tickers = sorted(
    set(
        stock_universe
        + sum(GROUPS.values(), [])
    )
)

data = yf.download(
    all_tickers,
    period=f"{START_LOOKBACK}d",
    auto_adjust=True,
    progress=False
)["Close"]

data = data.dropna(how="all")

# ============================
# INDICATORS
# ============================
def sma(series, n):
    return series.rolling(n).mean()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

# ============================
# TREND FILTER
# ============================
def passes_trend(ticker):
    px = data[ticker]
    if ticker in stock_universe:
        return px.iloc[-1] > ema(px, STOCK_EMA_PERIOD).iloc[-1]
    if ticker in sum(
        [GROUPS[g] for g in ["corp_bonds","treasury_bonds","high_yield_bonds"]],
        []
    ):
        return px.iloc[-1] > sma(px, BOND_SMA_PERIOD).iloc[-1]
    if not ENABLE_SMA_FILTER:
        return True
    return px.iloc[-1] > sma(px, SMA_PERIOD).iloc[-1]

# ============================
# MOMENTUM
# ============================
def momentum(ticker):
    px = data[ticker]
    return np.mean([
        px.iloc[-1] / px.iloc[-(lb + 1)] - 1
        for lb in MOMENTUM_LOOKBACKS
    ])

def group_momentum(tickers):
    moms = [momentum(t) for t in tickers if t in data]
    return np.mean(moms) if moms else 0.0

# ============================
# DURATION LADDER
# ============================
def duration_regime(symbols):
    def m(t):
        if t not in data:
            return -np.inf
        px = data[t]
        return np.mean([
            px.iloc[-1]/px.iloc[-21]-1,
            px.iloc[-1]/px.iloc[-63]-1,
            px.iloc[-1]/px.iloc[-126]-1
        ])
    if len(symbols) == 3:
        s,i,l = symbols
        return [s,i,l] if m(l)>m(i)>m(s) else [s,i] if m(i)>m(s) else [s]
    if len(symbols) == 2:
        s,l = symbols
        return [s,l] if m(l)>m(s) else [s]
    return symbols

# ============================
# BUILD GROUPS
# ============================
risk_groups = {
    "real": GROUPS["real"],
    "corp_bonds": duration_regime(GROUPS["corp_bonds"]),
    "treasury_bonds": duration_regime(GROUPS["treasury_bonds"]),
    "high_yield_bonds": duration_regime(GROUPS["high_yield_bonds"]),
    "equities": GROUPS["equities"],
    "crypto": GROUPS["crypto"]
}

if ENABLE_SECTORS:
    sector_mom = {s: momentum(s) for s in GROUPS["sectors"] if s in data}
    risk_groups["sectors"] = sorted(sector_mom, key=sector_mom.get, reverse=True)[:SECTOR_COUNT]

if ENABLE_STOCKS:
    stock_mom = {s: momentum(s) for s in stock_universe if s in data}
    risk_groups["stocks"] = sorted(stock_mom, key=stock_mom.get, reverse=True)[:STOCK_COUNT]

# ============================
# EDGE + VOL
# ============================
edges, vols = {}, {}

for g, syms in risk_groups.items():
    eligible = [s for s in syms if s in data and passes_trend(s)]
    if not eligible:
        continue

    rets = data[eligible].pct_change().mean(axis=1).dropna()
    if len(rets) < WINRATE_LOOKBACK:
        continue

    p_win = (rets.tail(WINRATE_LOOKBACK) > 0).mean()
    mom = group_momentum(eligible)
    noise = rets.tail(VOL_LOOKBACK).std()

    confidence = np.clip(abs(mom) / (noise + 1e-6), 0, 2)
    scale = max(0.1, 1 + confidence * np.sign(mom))
    edge = p_win * scale

    vol = np.std(np.log1p(rets.tail(VOL_LOOKBACK))) * np.sqrt(252)
    if vol <= 0:
        continue

    edges[g] = edge
    vols[g] = vol

# ============================
# NORMALIZE
# ============================
edge_eff = {g: e + 0.01 for g,e in edges.items()}
total_edge = sum(edge_eff.values())

weights = {
    g: (e/total_edge) * min(1.0, GROUP_VOL_TARGET/vols[g])
    for g,e in edge_eff.items()
}

# ============================
# HIERARCHY GATING
# ============================
PARENT_CHILD = {
    "equities": ["stocks","sectors"],
    "treasury_bonds": ["corp_bonds"],
    "corp_bonds": ["high_yield_bonds"]
}

for parent, children in PARENT_CHILD.items():
    if parent not in weights:
        for c in children:
            weights.pop(c, None)

# ============================
# CRYPTO CAP
# ============================
if "crypto" in weights and weights["crypto"] > CRYPTO_CAP:
    excess = weights["crypto"] - CRYPTO_CAP
    weights["crypto"] = CRYPTO_CAP
    others = [g for g in weights if g != "crypto"]
    for g in others:
        weights[g] += excess * (weights[g] / sum(weights[o] for o in others))

cash_weight = max(0.0, 1 - sum(weights.values()))

# ============================
# FINAL ALLOCATION
# ============================
allocations = {}

for g,w in weights.items():
    syms = [s for s in risk_groups[g] if passes_trend(s)]
    if not syms:
        continue
    for s in syms:
        allocations[s] = w / len(syms)

allocations["BIL"] = cash_weight

out = (
    pd.Series(allocations)
    .sort_values(ascending=False)
    .to_frame("weight")
)

out.to_csv("signals.csv")

print("\nFINAL SIGNALS\n")
print(out)
