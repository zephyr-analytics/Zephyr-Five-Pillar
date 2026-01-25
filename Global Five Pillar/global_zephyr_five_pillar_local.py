import numpy as np
import pandas as pd
import yfinance as yf

# ============================
# USER CONFIG (MATCHES QC)
# ============================
GROUP_VOL_TARGET = 0.20
CRYPTO_CAP = 0.10

ENABLE_SMA_FILTER = True
ENABLE_SECTORS = True
ENABLE_TREASURY_KILL_SWITCH = True

WINRATE_LOOKBACK = 126
VOL_LOOKBACK = 126

SMA_PERIOD = 147
BOND_SMA_PERIOD = 126

MOMENTUM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOMENTUM_LOOKBACKS)

SECTOR_COUNT = 2

# ============================
# ASSET GROUPS (QC MATCH)
# ============================
GROUPS = {
    "real": ["GLD", "DBC"],
    "corp_bonds": ["VCSH", "VCIT", "VCLT"],
    "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
    "high_yield_bonds": ["SHYG", "HYG"],
    "equities": ["VTI", "VEA", "VWO"],
    "sectors": [
        "IXP", "RXI", "KXI", "IXC", "IXG", "IXJ",
        "EXI", "IXN", "MXI", "REET", "JXI"
    ],
    "crypto": ["IBIT", "ETHA"],
    "cash": ["BIL"]
}

BOND_GROUPS = {"corp_bonds", "treasury_bonds", "high_yield_bonds"}

# ============================
# FETCH DATA
# ============================
all_tickers = sorted(set(sum(GROUPS.values(), [])))

data = yf.download(
    all_tickers,
    period=f"{MAX_LOOKBACK + 50}d",
    auto_adjust=True,
    progress=False
)["Close"].dropna(how="all")

# ============================
# INDICATORS
# ============================
def sma(series, n):
    return series.rolling(n).mean()

# ============================
# TREND FILTER (QC LOGIC)
# ============================
def passes_trend(ticker):
    if not ENABLE_SMA_FILTER:
        return True

    px = data[ticker]

    if ticker in sum(
        [GROUPS[g] for g in BOND_GROUPS],
        []
    ):
        ma = sma(px, BOND_SMA_PERIOD)
    else:
        ma = sma(px, SMA_PERIOD)

    return ma.notna().iloc[-1] and px.iloc[-1] > ma.iloc[-1]

# ============================
# MOMENTUM (QC EXACT)
# ============================
def momentum(ticker):
    px = data[ticker]
    return np.mean([
        px.iloc[-1] / px.iloc[-(lb + 1)] - 1
        for lb in MOMENTUM_LOOKBACKS
    ])

def group_momentum(tickers):
    vals = [momentum(t) for t in tickers if t in data]
    return float(np.mean(vals)) if vals else 0.0

# ============================
# DURATION REGIME (QC EXACT)
# ============================
def duration_regime(symbols):
    def m(t):
        if t not in data or len(data[t]) < 127:
            return -np.inf
        px = data[t]
        return np.mean([
            px.iloc[-1]/px.iloc[-21]-1,
            px.iloc[-1]/px.iloc[-63]-1,
            px.iloc[-1]/px.iloc[-126]-1,
        ])

    if len(symbols) == 3:
        s,i,l = symbols
        return (
            [s,i,l] if m(l) > m(i) > m(s)
            else [s,i] if m(i) > m(s)
            else [s]
        )

    if len(symbols) == 2:
        s,l = symbols
        return [s,l] if m(l) > m(s) else [s]

    return symbols

# ============================
# BUILD RISK GROUPS
# ============================
risk_groups = {
    "real": GROUPS["real"],
    "corp_bonds": duration_regime(GROUPS["corp_bonds"]),
    "treasury_bonds": duration_regime(GROUPS["treasury_bonds"]),
    "high_yield_bonds": duration_regime(GROUPS["high_yield_bonds"]),
    "equities": GROUPS["equities"],
    "crypto": GROUPS["crypto"],
}

if ENABLE_SECTORS:
    sector_mom = {s: momentum(s) for s in GROUPS["sectors"] if s in data}
    risk_groups["sectors"] = sorted(
        sector_mom,
        key=sector_mom.get,
        reverse=True
    )[:SECTOR_COUNT]

# ============================
# EDGE + VOL (QC EXACT)
# ============================
edges, vols = {}, {}

for group, symbols in risk_groups.items():
    eligible = [s for s in symbols if s in data and passes_trend(s)]
    if not eligible:
        continue

    rets = data[eligible].pct_change().mean(axis=1).dropna()
    if len(rets) < WINRATE_LOOKBACK:
        continue

    win_rate = (rets.tail(WINRATE_LOOKBACK) > 0).mean()
    mom = group_momentum(eligible)
    noise = rets.tail(VOL_LOOKBACK).std()

    confidence = np.clip(abs(mom) / (noise + 1e-6), 0.0, 2.0)
    scale = max(0.1, 1.0 + confidence * np.sign(mom))
    edge = win_rate * scale

    vol = np.std(np.log1p(rets.tail(VOL_LOOKBACK))) * np.sqrt(252)
    if vol <= 0:
        continue

    edges[group] = edge
    vols[group] = vol

# ============================
# TREASURY KILL SWITCH
# ============================
if ENABLE_TREASURY_KILL_SWITCH and "treasury_bonds" not in edges:
    print("Treasury kill switch → 100% cash")
    pd.Series({"BIL": 1.0}, name="weight").to_csv("signals.csv")
    exit()

# ============================
# NORMALIZE + VOL TARGET
# ============================
effective_edges = {g: e + 0.01 for g, e in edges.items()}
total_edge = sum(effective_edges.values())

weights = {
    g: (effective_edges[g] / total_edge)
    * min(1.0, GROUP_VOL_TARGET / vols[g])
    for g in effective_edges
}

# ============================
# CRYPTO CAP (QC EXACT)
# ============================
if "crypto" in weights and weights["crypto"] > CRYPTO_CAP:
    excess = weights["crypto"] - CRYPTO_CAP
    weights["crypto"] = CRYPTO_CAP
    others = [g for g in weights if g != "crypto"]
    total_other = sum(weights[g] for g in others)
    for g in others:
        weights[g] += excess * (weights[g] / total_other)

cash_weight = max(0.0, 1.0 - sum(weights.values()))

# ============================
# FINAL ALLOCATION
# ============================
allocations = {}

for group, w in weights.items():
    symbols = [s for s in risk_groups[group] if passes_trend(s)]
    if not symbols:
        continue
    for s in symbols:
        allocations[s] = w / len(symbols)

allocations["BIL"] = cash_weight

out = (
    pd.Series(allocations)
    .sort_values(ascending=False)
    .to_frame("weight")
)
out_pct = out * 100
out_pct.columns = ["weight_pct"]

out_pct.to_csv("signals_pct.csv")

print("\nFINAL SIGNALS (%)\n")
print(out_pct.round(2))
