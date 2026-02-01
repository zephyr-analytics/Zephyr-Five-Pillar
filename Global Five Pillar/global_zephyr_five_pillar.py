from AlgorithmImports import *
from typing import Dict, List
import numpy as np


class ZephyrFivePillar(QCAlgorithm):
    """
    Multi-asset, regime-aware allocation strategy using momentum,
    trend filtering, and volatility-adjusted win-rate confidence.

    Volatility is used ONLY to penalize noisy signals in edge construction.
    There is NO volatility targeting or exposure scaling.
    """

    # ====================================================
    # initialize
    # ====================================================
    def initialize(self) -> None:
        self.SetStartDate(2012, 1, 1)
        self.SetCash(100_000)

        # ============================
        # user options
        # ============================
        self.enable_sma_filter = True
        self.enable_sectors = True
        self.enable_treasury_kill_switch = True

        self.winrate_lookback = 126
        self.vol_lookback = 126

        self.sma_period = 147
        self.bond_sma_period = 126
        self.sector_count = 2
        self.crypto_cap = 0.10

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # ============================
        # asset groups
        # ============================
        self.group_tickers = {
            "real": ["GLD", "DBC"],
            "corp_bonds": ["VCSH", "VCIT", "VCLT"],
            "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
            "high_yield_bonds": ["SHYG", "HYG"],
            "equities": ["VTI", "VEA", "VWO"],
            "sectors": [
                "IXP", "RXI", "KXI", "IXC", "IXG", "IXJ",
                "EXI", "IXN", "MXI", "REET", "JXI"
            ],
            "crypto": ["BTCUSD", "ETHUSD"],
            "cash": ["SHV"]
        }

        self.symbols: Dict[str, List[Symbol]] = {}

        for group, tickers in self.group_tickers.items():
            self.symbols[group] = []
            for ticker in tickers:
                symbol = (
                    self.AddCrypto(ticker, Resolution.Daily).Symbol
                    if ticker.endswith("USD")
                    else self.AddEquity(ticker, Resolution.Daily).Symbol
                )
                self.symbols[group].append(symbol)
                self.Securities[symbol].FeeModel = ConstantFeeModel(0)

        self.all_symbols = [s for v in self.symbols.values() for s in v]

        # ----------------------------
        # bond universe
        # ----------------------------
        self.bond_symbols = {
            s for g in ["corp_bonds", "treasury_bonds", "high_yield_bonds"]
            for s in self.symbols[g]
        }

        # ============================
        # indicators
        # ============================
        self.smas = {
            s: self.SMA(s, self.sma_period, Resolution.Daily)
            for s in self.all_symbols if s not in self.bond_symbols
        }

        self.bond_smas = {
            s: self.SMA(s, self.bond_sma_period, Resolution.Daily)
            for s in self.bond_symbols
        }

        self.SetWarmUp(
            max(
                self.winrate_lookback,
                self.vol_lookback,
                self.max_lookback,
                self.bond_sma_period
            )
        )

        self.Schedule.On(
            self.DateRules.MonthEnd(self.all_symbols[0]),
            self.TimeRules.BeforeMarketClose(self.all_symbols[0], 5),
            self.rebalance
        )

    # ====================================================
    # trend filter
    # ====================================================
    def passes_trend(self, symbol: Symbol) -> bool:
        if not self.enable_sma_filter:
            return True

        sma = (
            self.bond_smas.get(symbol)
            if symbol in self.bond_symbols
            else self.smas.get(symbol)
        )

        return sma and sma.IsReady and self.Securities[symbol].Price > sma.Current.Value

    # ====================================================
    # NEW: treasury trend-only kill helper
    # ====================================================
    def all_treasuries_fail_trend(self) -> bool:
        """
        Kill risk only if ALL treasury assets fail the trend filter.
        """
        if not self.enable_sma_filter:
            return False

        for s in self.symbols["treasury_bonds"]:
            sma = self.bond_smas.get(s)
            if sma and sma.IsReady and self.Securities[s].Price > sma.Current.Value:
                return False  # at least one treasury passes trend

        return True

    # ====================================================
    # rebalance
    # ====================================================
    def rebalance(self) -> None:
        if self.IsWarmingUp:
            return

        self.Liquidate()

        history = self.history(
            self.all_symbols,
            self.max_lookback + 1,
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        closes = history["close"].unstack(0)
        cash_symbol = self.symbols["cash"][0]

        bil_6m = self.six_month_return(cash_symbol, closes)

        corp_bonds = self.get_duration_regime_for_group(
            closes, self.symbols["corp_bonds"]
        )
        treasury_bonds = self.get_duration_regime_for_group(
            closes, self.symbols["treasury_bonds"]
        )
        high_yield_bonds = self.get_duration_regime_for_group(
            closes, self.symbols["high_yield_bonds"]
        )

        sector_symbols = []
        if self.enable_sectors:
            sector_momentum = self.compute_momentum(self.symbols["sectors"], closes)
            sector_symbols = sorted(
                sector_momentum,
                key=sector_momentum.get,
                reverse=True
            )[:self.sector_count]

        risk_groups = {
            "real": self.symbols["real"],
            "corp_bonds": corp_bonds,
            "treasury_bonds": treasury_bonds,
            "high_yield_bonds": high_yield_bonds,
            "equities": self.symbols["equities"],
            "crypto": self.symbols["crypto"]
        }

        if sector_symbols:
            risk_groups["sectors"] = sector_symbols

        edges = {}
        group_assets = {}
        group_asset_edges = {}

        # ============================
        # group + asset edge construction
        # ============================
        for group, symbols in risk_groups.items():
            eligible = []
            asset_edges = {}

            for s in symbols:
                if s not in closes.columns:
                    continue
                if not self.passes_trend(s):
                    continue

                asset_momentum = self.compute_asset_momentum(s, closes)
                asset_6m = self.six_month_return(s, closes)

                if asset_6m <= bil_6m or asset_momentum <= 0:
                    continue

                asset_edge = max(0.0, asset_momentum)
                eligible.append(s)
                asset_edges[s] = asset_edge

            if not eligible:
                continue

            group_assets[group] = eligible
            group_asset_edges[group] = asset_edges

            group_simple_returns = closes[eligible].pct_change().mean(axis=1).dropna()
            if len(group_simple_returns) < max(self.winrate_lookback, self.vol_lookback):
                continue

            log_group = np.log1p(group_simple_returns)
            win_rate = float(
                np.mean(log_group.tail(self.winrate_lookback) > 0)
            )

            group_momentum = self.compute_group_momentum(eligible, closes)
            group_vol = float(
                np.std(log_group.tail(self.vol_lookback)) * np.sqrt(252)
            )

            if not np.isfinite(group_vol) or group_vol <= 0:
                continue

            confidence = group_momentum / (group_vol + 1e-6)

            edges[group] = win_rate * max(0.1, 1.0 + confidence)

        if not edges:
            self.SetHoldings(cash_symbol, 1.0)
            return

        # ====================================================
        # UPDATED TREASURY KILL SWITCH (trend-only)
        # ====================================================
        if self.enable_treasury_kill_switch and self.all_treasuries_fail_trend():
            self.SetHoldings(cash_symbol, 1.0)
            self.Debug("TREASURY KILL SWITCH: all treasuries failed trend")
            return

        # ============================
        # group weights
        # ============================
        effective_edges = {g: e + 0.01 for g, e in edges.items()}
        total_edge = sum(effective_edges.values())

        weights = {
            g: effective_edges[g] / total_edge
            for g in effective_edges
        }

        # crypto cap
        if "crypto" in weights and weights["crypto"] > self.crypto_cap:
            excess = weights["crypto"] - self.crypto_cap
            weights["crypto"] = self.crypto_cap
            others = [g for g in weights if g != "crypto"]
            total_other = sum(weights[g] for g in others)
            for g in others:
                weights[g] += excess * (weights[g] / total_other)

        cash_weight = max(0.0, 1.0 - sum(weights.values()))

        # ============================
        # intra-group allocation (EDGE ONLY)
        # ============================
        for group, group_weight in weights.items():
            symbols = group_assets.get(group, [])
            if not symbols:
                continue

            edges_i = {
                s: group_asset_edges[group][s] + 0.01
                for s in symbols
            }

            edge_sum = sum(edges_i.values())
            if edge_sum <= 0:
                continue

            for s, edge in edges_i.items():
                self.SetHoldings(s, group_weight * edge / edge_sum)

        self.SetHoldings(cash_symbol, cash_weight)

        self.Debug(
            "GROUPS | "
            + " | ".join(f"{g}:{w:.2f}" for g, w in sorted(weights.items()))
            + f" | cash:{cash_weight:.2f}"
        )

    # ====================================================
    # helpers (UNCHANGED)
    # ====================================================
    def six_month_return(self, symbol, closes) -> float:
        if symbol not in closes.columns or len(closes[symbol]) < 127:
            return -np.inf
        prices = closes[symbol]
        return float(prices.iloc[-1] / prices.iloc[-127] - 1)

    def compute_momentum(self, symbols, closes):
        scores = {}
        for symbol in symbols:
            if symbol not in closes.columns:
                continue
            prices = closes[symbol]
            if len(prices) < self.max_lookback + 1:
                continue
            scores[symbol] = float(np.mean([
                prices.iloc[-1] / prices.iloc[-(lb + 1)] - 1
                for lb in self.momentum_lookbacks
            ]))
        return scores

    def get_duration_regime_for_group(self, closes, symbols):
        def momentum(symbol):
            if symbol not in closes.columns or len(closes[symbol]) < 127:
                return -np.inf
            prices = closes[symbol]
            return np.mean([
                prices.iloc[-1] / prices.iloc[-22] - 1,
                prices.iloc[-1] / prices.iloc[-64] - 1,
                prices.iloc[-1] / prices.iloc[-127] - 1,
            ])

        if len(symbols) == 3:
            short, intermediate, long = symbols
            return (
                [short, intermediate, long]
                if momentum(long) > momentum(intermediate) > momentum(short)
                else [short, intermediate]
                if momentum(intermediate) > momentum(short)
                else [short]
            )

        if len(symbols) == 2:
            short, long = symbols
            return [short, long] if momentum(long) > momentum(short) else [short]

        return symbols

    def compute_group_momentum(self, symbols, closes):
        values = []
        for symbol in symbols:
            if symbol not in closes.columns or len(closes[symbol]) < 253:
                continue
            prices = closes[symbol]
            values.append(np.mean([
                prices.iloc[-1] / prices.iloc[-22] - 1,
                prices.iloc[-1] / prices.iloc[-64] - 1,
                prices.iloc[-1] / prices.iloc[-127] - 1,
                prices.iloc[-1] / prices.iloc[-190] - 1,
                prices.iloc[-1] / prices.iloc[-253] - 1,
            ]))
        return float(np.mean(values)) if values else 0.0

    def compute_asset_momentum(self, symbol, closes) -> float:
        if symbol not in closes.columns or len(closes[symbol]) < self.max_lookback + 1:
            return -np.inf
        prices = closes[symbol]
        return float(np.mean([
            prices.iloc[-1] / prices.iloc[-(lb + 1)] - 1
            for lb in self.momentum_lookbacks
        ]))
