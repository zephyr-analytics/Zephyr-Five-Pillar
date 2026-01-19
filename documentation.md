# Zephyr-Five-Pillar Documentation

## Overview
The strategy is a monthly rebalanced, multi-asset portfolio with:
- SMA trend filter to avoid assets trading below their moving average.
- Momentum-based sector rotation.
- Optional volatility targeting per asset.
- A fixed cash buffer.

Core implementation: `zephyr_five_pillar.py`.

## Strategy Flow (Monthly)
1. Pull history for all assets.
2. Score sector ETFs with multi-horizon momentum.
3. Select top `sector_count` sectors.
4. Build allocation groups (real assets, bonds, equities, sectors, bitcoin).
5. For each group, keep only symbols above SMA.
6. Allocate group weights equally among eligible symbols.
7. Optionally scale weights down using realized volatility.
8. Hold remaining cash as a buffer.

## User Settings (Adjustable)
All user settings live in `ZephyrFivePillars()` in `zephyr_five_pillar.py`.

### Capital and Dates
- `self.SetStartDate(2012, 1, 1)`: Backtest start date.
- `self.SetCash(100000)`: Starting capital.

### Risk and Allocation Controls
- `enable_vol_targeting` (bool): Turn volatility targeting on/off.
  - If `True`, each eligible asset weight is scaled by `min(1, target_vol / realized_vol)`.
- `target_vol` (float): Annualized volatility cap per asset. Example: `0.75` means 75% annualized.
- `cash_buffer` (float): Fraction of portfolio held in cash (0.0 to 1.0).
  - All group weights must sum to `1.0 - cash_buffer`.

### Group Weights
`category_target_weights` defines the total weight per group. Must sum to `1.0 - cash_buffer`.

```
self.category_target_weights = {
    "real": 0.20,
    "bonds": 0.25,
    "equities": 0.20,
    "sectors": 0.25,
    "bitcoin": 0.05
}
```

If a group has no eligible symbols (fails SMA), its weight is skipped and remains in cash.

### Universe Definitions
You can customize tickers per group:
- `real_asset_tickers`: Real assets (e.g., GLD, DBC)
- `bond_tickers`: Short to long duration bonds
- `equity_tickers`: Broad market equities
- `sector_tickers`: Sector ETFs used for momentum ranking
- `BTCUSD`: Added via `AddCrypto` for the bitcoin group

### Momentum and Filters
- `momentum_lookbacks` (list of ints): Trading-day lookbacks used for momentum scoring.
  - Default: `[21, 63, 126, 189, 252]`.
- `sector_count` (int): Number of top sectors selected each rebalance.
- `sma_period` (int): SMA length in trading days.
- `vol_lookback` (int): Lookback for realized volatility.

## Rebalance Schedule
Rebalances at month end, 5 minutes before market close for the first equity symbol in `self.symbols`.

```
self.Schedule.On(
    self.DateRules.MonthEnd(self.symbols[0]),
    self.TimeRules.BeforeMarketClose(self.symbols[0], 5),
    self.Rebalance
)
```

## Weight Validation
At rebalance time, the algorithm verifies:
```
sum(category_target_weights.values()) == 1.0 - cash_buffer
```
If not, the rebalance exits and logs an error.

## Example User Customizations
1) Conservative: raise bonds, lower equities, increase cash buffer.
2) Trend focus: increase `sma_period` or reduce `sector_count`.
3) Higher risk: disable vol targeting and lower `cash_buffer`.

## Notes
- All assets must have enough history for SMA, momentum, and volatility calculations.
- BTCUSD uses daily crypto data; ensure your data permissions support it.
