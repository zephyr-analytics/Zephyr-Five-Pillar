import numpy as np
import pandas as pd
import yfinance as yf

# ============================
# USER CONFIG (UNCHANGED)
# ============================
WINRATE_LOOKBACK = 63
VOL_LOOKBACK = 63
SMA_PERIOD = 126
GROUP_VOL_TARGET = 0.15

MOM_LOOKBACKS = [21, 63, 126, 189, 252]
MAX_LOOKBACK = max(MOM_LOOKBACKS)

ENABLE_SMA_FILTER = True

# ============================
# BOND UNIVERSE (FIXED)
# ============================
GROUPS = {
    "corp_bonds": ["VCSH", "VCIT", "VCLT"],
    "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
    "high_yield_bonds": ["SHYG", "HYG"],
    "cash": ["BIL"]
}

ALL_SYMBOLS = sorted(set(sum(GROUPS.values(), [])))

# ============================
# FETCH DATA
# ============================
data = yf.download(
    ALL_SYMBOLS,
    period=f"{MAX_LOOKBACK + 30}d",
    auto_adjust=True,
    progress=False
)["Close"].dropna(how="all")

# ============================
# INDICATORS
# ============================
def sma(px, n):
    return px.rolling(n).mean()

# ============================
# TREND FILTER (SMA FOR ALL)
# ============================
def passes_trend(t):
    if not ENABLE_SMA_FILTER:
        return True
    px = data[t]
    return px.iloc[-1] > sma(px, SMA_PERIOD).iloc[-1]

# ============================
# MOMENTUM HELPERS (EXACT)
# ============================
def duration_mom(t):
    px = data[t]
    return np.mean([
        px.iloc[-1] / px.iloc[-21] - 1,
        px.iloc[-1] / px.iloc[-63] - 1,
        px.iloc[-1] / px.iloc[-126] - 1,
    ])

def group_momentum(symbols):
    moms = []
    for s in symbols:
        px = data[s]
        if len(px) < 253:
            continue
        moms.append(np.mean([
            px.iloc[-1] / px.iloc[-21] - 1,
            px.iloc[-1] / px.iloc[-63] - 1,
            px.iloc[-1] / px.iloc[-126] - 1,
            px.iloc[-1] / px.iloc[-189] - 1,
            px.iloc[-1] / px.iloc[-252] - 1,
        ]))
    return np.mean(moms) if moms else 0.0

# ============================
# DURATION LADDER (EXACT)
# ============================
def duration_regime(symbols):
    if len(symbols) == 3:
        s,i,l = symbols
        return (
            [s,i,l] if duration_mom(l) > duration_mom(i) > duration_mom(s)
            else [s,i] if duration_mom(i) > duration_mom(s)
            else [s]
        )
    if len(symbols) == 2:
        s,l = symbols
        return [s,l] if duration_mom(l) > duration_mom(s) else [s]
    return symbols

# ============================
# BUILD RISK GROUPS
# ============================
risk_groups = {
    "corp_bonds": duration_regime(GROUPS["corp_bonds"]),
    "treasury_bonds": duration_regime(GROUPS["treasury_bonds"]),
    "high_yield_bonds": duration_regime(GROUPS["high_yield_bonds"])
}

# ============================
# EDGE + VOL
# ============================
edges, vols = {}, {}

for g, syms in risk_groups.items():
    eligible = [s for s in syms if passes_trend(s)]
    if not eligible:
        continue

    g_rets = data[eligible].pct_change().mean(axis=1).dropna()
    if len(g_rets) < WINRATE_LOOKBACK:
        continue

    p_win = (g_rets.tail(WINRATE_LOOKBACK) > 0).mean()
    mom = group_momentum(eligible)

    mom_std = g_rets.tail(VOL_LOOKBACK).std()
    confidence = np.clip(abs(mom) / (mom_std + 1e-6), 0, 2)

    scale = max(0.1, 1.0 + confidence * np.sign(mom))
    edge = p_win * scale

    vol = np.std(np.log1p(g_rets.tail(VOL_LOOKBACK))) * np.sqrt(252)
    if vol <= 0:
        continue

    edges[g] = edge
    vols[g] = vol

# ============================
# CASH FALLBACK
# ============================
if not edges:
    print("\nALL CASH\n")
    print("BIL: 1.0")
    exit()

# ============================
# NORMALIZE + VOL TARGET
# ============================
edge_eff = {g: e + 0.01 for g,e in edges.items()}
total_edge = sum(edge_eff.values())

weights = {
    g: (e / total_edge) * min(1.0, GROUP_VOL_TARGET / vols[g])
    for g,e in edge_eff.items()
}

# ============================
# HIERARCHICAL GATING
# corp → high yield
# ============================
if "corp_bonds" not in edges and "high_yield_bonds" in weights:
    weights.pop("high_yield_bonds")

risk_weight = sum(weights.values())
cash_weight = max(0.0, 1.0 - risk_weight)

# ============================
# FINAL ALLOCATION
# ============================
allocs = {}

for g,w in weights.items():
    syms = [s for s in risk_groups[g] if passes_trend(s)]
    for s in syms:
        allocs[s] = w / len(syms)

allocs["BIL"] = cash_weight

out = (
    pd.Series(allocs)
    .sort_values(ascending=False)
    .to_frame("weight")
)

out.to_csv("bond_signals.csv")

print("\nFINAL BOND SIGNALS\n")
print(out)
