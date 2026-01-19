from AlgorithmImports import *
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ZephyrFivePillar(QCAlgorithm):
    """Monthly rebalanced multi-asset strategy with SMA and volatility targeting."""

    def initialize(self) -> None:
        """Configure the strategy, assets, indicators, and schedule.

        Returns
        -------
        None
        """
        self.set_start_date(2012, 1, 1)
        self.set_cash(100_000)
        self.set_security_initializer(lambda security: security.SetFeeModel(ZeroFeeModel()))
        # === User Options ===
        self.enable_vol_targeting = True
        self.target_vol = 0.75
        self.cash_buffer = 0.05
        self.sector_count = 4
        # === Custom Group Weights (must sum to 1 - cash_buffer) ===
        self.category_target_weights = {
            "real": 0.20,
            "bonds": 0.25,
            "equities": 0.20,
            "sectors": 0.25,
            "bitcoin": 0.05
        }

        # === Ticker Groups ===
        self.real_asset_tickers = ["GLD", "DBC"]
        self.bond_tickers = ["VCSH", "VCIT", "VCLT", "VGSH", "VGIT", "VGLT"]
        self.equity_tickers = ["SPY", "QQQ", "DIA"]
        self.sector_tickers = [
            "VOX", "VCR", "VDC", "VDE", "VFH", "VHT",
            "VIS", "VGT", "VAW", "VNQ", "VPU"
        ]

        self.bitcoin_weight = self.category_target_weights.get("bitcoin", 0.0)
        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)
        self.sma_period = 168
        self.vol_lookback = 21

        # === Add Assets ===
        all_tickers = self.real_asset_tickers + self.bond_tickers + self.equity_tickers + self.sector_tickers
        self.symbols = [self.AddEquity(t, Resolution.Daily).Symbol for t in all_tickers]
        self.btc_symbol = self.AddCrypto("BTCUSD", Resolution.Daily).Symbol
        self.all_symbols = self.symbols + [self.btc_symbol]

        # === Indicators ===
        self.smas = {
            s: self.SMA(s, self.sma_period, Resolution.Daily)
            for s in self.all_symbols
        }

        self.set_warm_up(max(self.sma_period, self.max_lookback, self.vol_lookback))

        self.schedule.On(
            self.DateRules.MonthEnd(self.symbols[0]),
            self.TimeRules.BeforeMarketClose(self.symbols[0], 5),
            self.rebalance
        )


    def rebalance(self) -> None:
        """Rebalance portfolio allocations at the scheduled interval.

        Returns
        -------
        None
        """
        if self.IsWarmingUp:
            return

        self.Liquidate()

        # === Pull batched history for all assets (momentum + vol) ===
        all_needed = list(set(self.symbols + [self.btc_symbol]))
        history = self.history(
            all_needed,
            self.max_lookback + 1,
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        # === Compute momentum for sectors ===
        sector_symbols = [s for s in self.symbols if s.Value in self.sector_tickers]
        sector_momentum = self.compute_momentum(sector_symbols, history)
        top_sectors = sorted(sector_momentum, key=sector_momentum.get, reverse=True)[:self.sector_count]

        # === Build group map ===
        group_map = {
            "real": [s for s in self.symbols if s.Value in self.real_asset_tickers],
            "bonds": [s for s in self.symbols if s.Value in self.bond_tickers],
            "equities": [s for s in self.symbols if s.Value in self.equity_tickers],
            "sectors": top_sectors,
            "bitcoin": [self.btc_symbol],
        }

        # === Validate group weights ===
        total_weight = sum(self.category_target_weights.values())
        expected = 1.0 - self.cash_buffer
        if abs(total_weight - expected) > 0.01:
            self.Error(f"Group weights must sum to {expected}. Current: {total_weight}")
            return

        total_allocated = 0
        for group, syms in group_map.items():
            group_weight = self.category_target_weights.get(group, 0.0)
            eligible = [s for s in syms if self.passes_sma(s)]
            if not eligible:
                self.Debug(f"{self.Time.date()} | {group} group: all failed SMA")
                continue

            raw_weight = group_weight / len(eligible)

            for s in eligible:
                weight = raw_weight

                if self.enable_vol_targeting:
                    vol = self.compute_realized_volatility(s, history)
                    if vol is None:
                        continue
                    vol_scalar = min(1.0, self.target_vol / vol)
                    weight *= vol_scalar
                    self.Debug(f"{self.Time.date()} | {s.Value} vol={round(vol, 3)} scaled_weight={round(weight, 3)}")

                self.set_holdings(s, weight)
                total_allocated += weight

        self.debug(f"{self.Time.date()} | Top Sectors: {[s.Value for s in top_sectors]} | "
                f"Allocated: {round(total_allocated, 3)} | Cash held: {round(1.0 - total_allocated, 3)}")


    def compute_momentum(self, symbols: List[Symbol], history: pd.DataFrame) -> Dict[Symbol, float]:
        """Compute average multi-horizon momentum scores for symbols.

        Parameters
        ----------
        symbols
            Symbols to score.
        history
            Historical price data, including a "close" column.

        Returns
        -------
        dict
            Mapping of symbol to momentum score.
        """
        momentum_scores = {}
        closes = history["close"].unstack(0).dropna()

        for s in symbols:
            if s not in closes.columns:
                continue
            px = closes[s]
            if len(px) < self.max_lookback + 1:
                continue

            rets = [
                px.iloc[-1] / px.iloc[-(lb + 1)] - 1
                for lb in self.momentum_lookbacks
            ]
            momentum_scores[s] = sum(rets) / len(rets)

        return momentum_scores


    def compute_realized_volatility(self, symbol: Symbol, history: pd.DataFrame) -> Optional[float]:
        """Compute annualized realized volatility for a symbol.

        Parameters
        ----------
        symbol
            Symbol to evaluate.
        history
            Historical price data, including a "close" column.

        Returns
        -------
        float or None
            Annualized realized volatility when enough data is available.
        """
        closes = history["close"].unstack(0)

        if symbol not in closes.columns:
            return None

        px = closes[symbol].dropna()
        if len(px) < self.vol_lookback + 1:
            return None

        log_returns = np.log(px / px.shift(1)).dropna()
        daily_vol = np.std(log_returns)
        ann_vol = daily_vol * np.sqrt(252)

        return ann_vol


    def passes_sma(self, symbol: Symbol) -> bool:
        """Check whether a symbol is trading above its SMA.

        Parameters
        ----------
        symbol
            Symbol to evaluate.

        Returns
        -------
        bool
            True if the price is above the SMA, otherwise False.
        """
        sma = self.smas[symbol]
        if not sma.IsReady:
            return False
        return self.securities[symbol].Price > sma.Current.Value
