# Zephyr-Five-Pillar Documentation

## Overview
Group-based multi-asset allocator with:
- SMA filter to avoid assets trading below their moving average.
- Momentum-based sector selection.
- Persistence (win-rate) scoring at the group level.
- Group-level volatility targeting.
- Unified crypto sleeve (BTC + ETH ETFs in local version).
- Residual cash.

Local implementation: `zephyr_five_pillar_local.py` (yfinance snapshot).
QuantConnect implementation: `zephyr_five_pillar.py` (scheduled monthly rebalance).

## Strategy Flow (Local Snapshot)
1. Pull price history for all assets.
2. Score sector ETFs with multi-horizon momentum and select top `sector_count`.
3. Build risk groups (real assets, bonds, equities, sectors, crypto).
4. For each group, keep only symbols above SMA.
5. Compute group persistence (win-rate) and group volatility.
6. Normalize edges with `epsilon`, then scale by `group_vol_target`.
7. Cap the crypto sleeve at `crypto_cap`, redistribute excess to other groups.
8. Allocate each group weight equally across its eligible symbols.
9. Hold residual cash.

If no groups qualify, the portfolio holds 100% cash.

## User Settings (Local)
All local settings live in `ZephyrFivePillar.__init__` in `zephyr_five_pillar_local.py`.

### Dates and Lookbacks
- `start` (str): Start date for price history (e.g., `"2012-01-01"`).
- `winrate_lookback` (int): Lookback for persistence (win-rate).
- `vol_lookback` (int): Lookback for realized volatility.
- `sma_period` (int): SMA length in trading days.

### Sector Selection
- `sector_count` (int): Number of top sectors to include each rebalance.
- `momentum_lookbacks` (list of ints): Trading-day lookbacks used for momentum scoring.

### Risk Controls
- `group_vol_target` (float): Annualized volatility target for each group (scales weights).
- `crypto_cap` (float): Max combined weight for the crypto sleeve.
- `epsilon` (float): Small constant added to each group edge before normalization.

### Universe Definitions
- `real`: Real assets (e.g., GLD, DBC)
- `bonds`: Short to long duration bonds
- `equities`: Broad market equities
- `sectors`: Sector ETFs used for momentum ranking
- `crypto`: BTC/ETH ETFs in the local model (e.g., IBIT, ETHA)

### Cash
- `cash_ticker` (str): Placeholder used for residual cash weights in output.

## Allocation Details
- Group edges are computed from average daily returns across eligible symbols.
- Volatility uses log returns over `vol_lookback`, annualized by `sqrt(252)`.
- Weights are allocated equally across eligible symbols within each group.

## QuantConnect Differences
- Uses `BTCUSD` and `ETHUSD` via `AddCrypto`, plus `BIL` as a cash proxy.
- Runs on a monthly schedule at month end, 5 minutes before market close.
- SMA filter can be toggled with `enable_sma_filter`.

## Notes
- All assets must have enough history for SMA, momentum, and volatility calculations.
- Local data uses `yfinance` with adjusted prices.

## Portfolio Formula
Pieces:
1. Sector selection:
   - `TopSectors = top_k( momentum(t), k=sector_count )`
2. Eligible symbols (SMA filter):
   - `Eligible(g) = { s in Group(g) | Price_s > SMA_s }`
3. Group returns:
   - `r_g(t) = mean( returns_s(t) for s in Eligible(g) )`
4. Group edge (persistence):
   - `edge_g = mean( r_g(t) > 0, t in last winrate_lookback )`
5. Group volatility:
   - `vol_g = std( log(1 + r_g(t)) for t in last vol_lookback ) * sqrt(252)`
6. Edge normalization with epsilon:
   - `w_raw_g = (edge_g + epsilon) / sum_g(edge_g + epsilon)`
7. Group vol targeting:
   - `w_scaled_g = w_raw_g * min(1, group_vol_target / vol_g)`
8. Crypto cap and redistribution:
   - `w_crypto = min(w_scaled_crypto, crypto_cap)`
   - `excess = w_scaled_crypto - w_crypto`
   - `w_other_g = w_scaled_g + excess * (w_scaled_g / sum_other w_scaled_g)`
9. Within-group allocation:
   - `w_s = w_group / |Eligible(g)| for s in Eligible(g)`
10. Cash:
   - `w_cash = max(0, 1 - sum_s w_s)`

Summary:
- `w_s = (w_group / |Eligible(g)|) for s in Eligible(g)`
- `w_group = cap_and_scale( normalize(edge_g + epsilon), vol_g, group_vol_target, crypto_cap )`
- `w_cash = 1 - sum_s w_s`
