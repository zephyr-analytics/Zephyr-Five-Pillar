import numpy as np
import pandas as pd
import yfinance as yf

# ============================
# USER CONFIG (MATCHES QC)
# ============================
GROUP_VOL_TARGET = 0.15
CRYPTO_CAP = 0.10

ENABLE_SMA_FILTER = True
ENABLE_SECTORS = False
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
    "cash": ["SHV"]
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
# TREND FILTER (QC EXACT)
# ============================
def passes_trend(ticker):
    if not ENABLE_SMA_FILTER:
        return True

    px = data[ticker]

    if ticker in sum([GROUPS[g] for g in BOND_GROUPS], []):
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
# 6-MONTH RETURN (QC EXACT)
# ============================
def six_month_return(ticker):
    if ticker not in data or len(data[ticker]) < 127:
        return -np.inf
    px = data[ticker]
    return px.iloc[-1] / px.iloc[-127] - 1

# ============================
# DURATION REGIME (QC EXACT)
# ============================
def duration_regime(symbols):
    def m(t):
        if t not in data or len(data[t]) < 127:
            return -np.inf
        px = data[t]
        return np.mean([
            px.iloc[-1]/px.iloc[-22]-1,
            px.iloc[-1]/px.iloc[-64]-1,
            px.iloc[-1]/px.iloc[-127]-1,
        ])

    if len(symbols) == 3:
        s, i, l = symbols
        return (
            [s, i, l] if m(l) > m(i) > m(s)
            else [s, i] if m(i) > m(s)
            else [s]
        )

    if len(symbols) == 2:
        s, l = symbols
        return [s, l] if m(l) > m(s) else [s]

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
group_assets = {}

# >>> NEW: asset-level storage (QC match)
group_asset_edges = {}
group_asset_vols = {}

cash_symbol = GROUPS["cash"][0]
bil_6m = six_month_return(cash_symbol)

for group, symbols in risk_groups.items():
    eligible = []
    asset_edges = {}
    asset_vols = {}

    for s in symbols:
        if s not in data:
            continue
        if not passes_trend(s):
            continue
        if six_month_return(s) <= bil_6m:
            continue

        mom = momentum(s)
        if mom <= 0:
            continue

        rets = data[s].pct_change().dropna()
        if len(rets) < max(WINRATE_LOOKBACK, VOL_LOOKBACK):
            continue

        log_rets = np.log1p(rets)

        win_rate = (log_rets.tail(WINRATE_LOOKBACK) > 0).mean()
        vol = np.std(log_rets.tail(VOL_LOOKBACK)) * np.sqrt(252)

        if not np.isfinite(vol) or vol <= 0:
            continue

        confidence = np.clip(abs(mom) / (vol + 1e-6), 0.0, 2.0)
        scale = max(0.1, 1.0 + confidence * np.sign(mom))

        asset_edges[s] = win_rate * scale
        asset_vols[s] = vol
        eligible.append(s)

    if not eligible:
        continue

    group_assets[group] = eligible
    group_asset_edges[group] = asset_edges
    group_asset_vols[group] = asset_vols

    simple_rets = data[eligible].pct_change().mean(axis=1).dropna()
    if len(simple_rets) < max(WINRATE_LOOKBACK, VOL_LOOKBACK):
        continue

    log_rets = np.log1p(simple_rets)
    group_mom = group_momentum(eligible)

    win_rate = (log_rets.tail(WINRATE_LOOKBACK) > 0).mean()
    vol = np.std(log_rets.tail(VOL_LOOKBACK)) * np.sqrt(252)

    if not np.isfinite(vol) or vol <= 0:
        continue

    confidence = np.clip(abs(group_mom) / (vol + 1e-6), 0.0, 2.0)
    scale = max(0.1, 1.0 + confidence * np.sign(group_mom))

    edges[group] = win_rate * scale
    vols[group] = vol

# ============================
# TREASURY KILL SWITCH
# ============================
if ENABLE_TREASURY_KILL_SWITCH and "treasury_bonds" not in edges:
    pd.Series({"BIL": 1.0}, name="weight").to_csv("signals.csv")
    print("Treasury kill switch → 100% cash")
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
# FINAL ALLOCATION (QC EXACT)
# ============================
allocations = {}

for group, gw in weights.items():
    symbols = group_assets.get(group, [])
    if not symbols:
        continue

    edges_i = {s: group_asset_edges[group][s] + 0.01 for s in symbols}
    vols_i = {s: group_asset_vols[group][s] for s in symbols}

    edge_sum = sum(edges_i.values())

    raw = {
        s: (edges_i[s] / edge_sum)
        * min(1.0, GROUP_VOL_TARGET / vols_i[s])
        for s in symbols
    }

    norm = sum(raw.values())
    for s, w in raw.items():
        allocations[s] = gw * w / norm

allocations["BIL"] = cash_weight

out = (
    pd.Series(allocations)
    .sort_values(ascending=False)
    .to_frame("weight")
)

(out * 100).rename(columns={"weight": "weight_pct"}).to_csv("signals_pct.csv")

print("\nFINAL SIGNALS (%)\n")
print((out * 100).round(2))
