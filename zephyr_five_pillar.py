from AlgorithmImports import *
from typing import Dict, List
import numpy as np
import pandas as pd


class ZephyrFivePillar(QCAlgorithm):

    def initialize(self) -> None:
        self.set_start_date(2012, 1, 1)
        self.set_cash(100_000)

        # ============================
        # User Options
        # ============================
        self.enable_sma_filter = True

        # Lookbacks
        self.winrate_lookback = 126
        self.vol_lookback = 126
        self.sma_period = 168

        # Sector selection
        self.sector_count = 4
        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # Group-level volatility target
        self.group_vol_target = 0.15

        self.crypto_cap = 0.10

        # ============================
        # Asset Groups
        # ============================
        self.group_tickers = {
            "real": ["GLD", "DBC"],
            "bonds": ["VCSH", "VCIT", "VCLT", "VGSH", "VGIT", "VGLT"],
            "equities": ["SPY", "QQQ", "DIA"],
            "sectors": [
                "VOX", "VCR", "VDC", "VDE", "VFH", "VHT",
                "VIS", "VGT", "VAW", "VNQ", "VPU"
            ],
            "crypto": ["BTCUSD", "ETHUSD"],
            "cash": ["BIL"]
        }

        self.symbols: Dict[str, List[Symbol]] = {}

        for g, tickers in self.group_tickers.items():
            self.symbols[g] = []
            for t in tickers:
                if t in ["BTCUSD", "ETHUSD"]:
                    sym = self.AddCrypto(t, Resolution.DAILY).Symbol
                else:
                    sym = self.AddEquity(t, Resolution.DAILY).Symbol
                self.symbols[g].append(sym)
                self.Securities[sym].FeeModel = ConstantFeeModel(0)

        self.all_symbols = [s for group in self.symbols.values() for s in group]

        # ============================
        # Indicators
        # ============================
        self.smas = {
            s: self.SMA(s, self.sma_period, Resolution.DAILY)
            for s in self.all_symbols
        }

        self.SetWarmUp(
            max(self.winrate_lookback, self.vol_lookback,
                self.max_lookback, self.sma_period)
        )

        self.Schedule.On(
            self.DateRules.MonthEnd(self.all_symbols[0]),
            self.TimeRules.BeforeMarketClose(self.all_symbols[0], 5),
            self.Rebalance
        )

    # ====================================================
    # Rebalance
    # ====================================================

    def Rebalance(self) -> None:
        if self.IsWarmingUp:
            return

        self.Liquidate()

        history = self.history(
            self.all_symbols,
            self.max_lookback + 1,
            Resolution.DAILY,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        closes = history["close"].unstack(0)

        # ------------------------------------------------
        # 1) Momentum-based sector selection
        # ------------------------------------------------
        sector_syms = self.symbols["sectors"]
        sector_mom = self.compute_momentum(sector_syms, closes)
        top_sectors = sorted(
            sector_mom, key=sector_mom.get, reverse=True
        )[:self.sector_count]

        risk_groups = {
            "real": self.symbols["real"],
            "bonds": self.symbols["bonds"],
            "equities": self.symbols["equities"],
            "sectors": top_sectors,
            "crypto": self.symbols["crypto"]
        }

        # ------------------------------------------------
        # 2) Persistence edges (NO CASH)
        # ------------------------------------------------
        edges: Dict[str, float] = {}
        vols: Dict[str, float] = {}

        for g, syms in risk_groups.items():
            eligible = [
                s for s in syms
                if s in closes.columns and self.passes_sma(s)
            ]
            if not eligible:
                continue

            g_rets = closes[eligible].pct_change().mean(axis=1).dropna()
            if len(g_rets) < max(self.winrate_lookback, self.vol_lookback):
                continue

            p_win = float(np.mean(g_rets.tail(self.winrate_lookback) > 0))
            edge = p_win

            g_vol = float(
                np.std(np.log1p(g_rets.tail(self.vol_lookback))) * np.sqrt(252)
            )
            if g_vol <= 0:
                continue

            edges[g] = edge
            vols[g] = g_vol

        if not edges:
            self.SetHoldings(self.symbols["cash"][0], 1.0)
            return

        # ------------------------------------------------
        # 3) Normalize across risk assets
        # ------------------------------------------------
        epsilon = 0.01
        edge_eff = {g: e + epsilon for g, e in edges.items()}
        total_edge = sum(edge_eff.values())
        w_raw = {g: e / total_edge for g, e in edge_eff.items()}

        # ------------------------------------------------
        # 4) Group-level vol targeting
        # ------------------------------------------------
        w_scaled: Dict[str, float] = {}

        for g, w in w_raw.items():
            vol_scalar = min(1.0, self.group_vol_target / vols[g])
            w_scaled[g] = w * vol_scalar

        # ------------------------------------------------
        # 5) Crypto sleeve cap (BTC + ETH combined)
        # ------------------------------------------------
        if "crypto" in w_scaled and w_scaled["crypto"] > self.crypto_cap:
            excess = w_scaled["crypto"] - self.crypto_cap
            w_scaled["crypto"] = self.crypto_cap

            others = [g for g in w_scaled if g != "crypto"]
            other_total = sum(w_scaled[g] for g in others)
            if other_total > 0:
                for g in others:
                    w_scaled[g] += excess * (w_scaled[g] / other_total)

        # ------------------------------------------------
        # 6) Residual → cash
        # ------------------------------------------------
        risk_weight = sum(w_scaled.values())
        cash_weight = max(0.0, 1.0 - risk_weight)

        # ------------------------------------------------
        # 7) Allocate
        # ------------------------------------------------
        for g, wg in w_scaled.items():
            syms = risk_groups[g]
            alloc = wg / len(syms)
            for s in syms:
                if self.passes_sma(s):
                    self.SetHoldings(s, alloc)

        self.SetHoldings(self.symbols["cash"][0], cash_weight)

        self.Debug(
            f"{self.Time.date()} | risk={round(risk_weight,3)} "
            f"cash={round(cash_weight,3)} | "
            + ", ".join(f"{g}:{round(w,3)}" for g, w in w_scaled.items())
        )

    # ====================================================
    # Helpers
    # ====================================================

    def compute_momentum(self, symbols: List[Symbol], closes: pd.DataFrame) -> Dict[Symbol, float]:
        scores = {}
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
            scores[s] = float(np.mean(rets))
        return scores

    def passes_sma(self, symbol: Symbol) -> bool:
        if not self.enable_sma_filter:
            return True
        sma = self.smas[symbol]
        return sma.IsReady and self.Securities[symbol].Price > sma.Current.Value
