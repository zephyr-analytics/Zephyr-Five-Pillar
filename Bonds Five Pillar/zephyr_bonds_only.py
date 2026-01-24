from AlgorithmImports import *
import numpy as np
import pandas as pd
from typing import Dict, List


class ZephyrBondOnly(QCAlgorithm):

    def Initialize(self) -> None:
        self.SetStartDate(2012, 1, 1)
        self.SetCash(100_000)

        # ============================
        # User Options (UNCHANGED)
        # ============================
        self.enable_sma_filter = True
        self.winrate_lookback = 63
        self.vol_lookback = 63
        self.sma_period = 126
        self.group_vol_target = 0.15

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # ============================
        # Bond-Only Asset Groups
        # ============================
        self.group_tickers = {
            "corp_bonds": ["VCSH", "VCIT", "VCLT"],
            "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
            "high_yield_bonds": ["SHYG", "HYG"],
            "cash": ["BIL"]
        }

        self.symbols: Dict[str, List[Symbol]] = {}

        for g, tickers in self.group_tickers.items():
            self.symbols[g] = []
            for t in tickers:
                sym = self.AddEquity(t, Resolution.Daily).Symbol
                self.symbols[g].append(sym)
                self.Securities[sym].FeeModel = ConstantFeeModel(0)

        self.all_symbols = [s for group in self.symbols.values() for s in group]

        # ============================
        # Indicators
        # ============================
        self.smas = {
            s: self.SMA(s, self.sma_period, Resolution.Daily)
            for s in self.all_symbols
        }

        self.SetWarmUp(
            max(self.winrate_lookback, self.vol_lookback, self.max_lookback)
        )

        self.Schedule.On(
            self.DateRules.MonthEnd(self.all_symbols[0]),
            self.TimeRules.BeforeMarketClose(self.all_symbols[0], 5),
            self.Rebalance
        )

    # ====================================================
    # Trend Filter (SMA for ALL assets)
    # ====================================================
    def PassesTrend(self, symbol: Symbol) -> bool:
        if not self.enable_sma_filter:
            return True

        sma = self.smas.get(symbol)
        return sma and sma.IsReady and self.Securities[symbol].Price > sma.Current.Value

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
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        closes = history["close"].unstack(0)

        # -----------------------------
        # Duration regimes
        # -----------------------------
        corp_syms = self.GetDurationRegimeForGroup(
            closes, self.symbols["corp_bonds"]
        )

        tsy_syms = self.GetDurationRegimeForGroup(
            closes, self.symbols["treasury_bonds"]
        )

        hy_syms = self.GetDurationRegimeForGroup(
            closes, self.symbols["high_yield_bonds"]
        )

        risk_groups = {
            "corp_bonds": corp_syms,
            "treasury_bonds": tsy_syms,
            "high_yield_bonds": hy_syms
        }

        edges, vols = {}, {}

        for g, syms in risk_groups.items():
            eligible = [
                s for s in syms
                if s in closes.columns and self.PassesTrend(s)
            ]
            if not eligible:
                continue

            g_rets = closes[eligible].pct_change().mean(axis=1).dropna()
            if len(g_rets) < self.winrate_lookback:
                continue

            p_win = float(np.mean(g_rets.tail(self.winrate_lookback) > 0))
            group_mom = self.ComputeGroupMomentum(eligible, closes)

            mom_std = float(np.std(g_rets.tail(self.vol_lookback)))
            confidence = abs(group_mom) / (mom_std + 1e-6)
            confidence = np.clip(confidence, 0.0, 2.0)

            scale = max(0.1, 1.0 + confidence * np.sign(group_mom))
            edge = p_win * scale

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

        # -----------------------------
        # Normalize + vol targeting
        # -----------------------------
        edge_eff = {g: e + 0.01 for g, e in edges.items()}
        total_edge = sum(edge_eff.values())

        w_raw = {g: e / total_edge for g, e in edge_eff.items()}

        w_scaled = {
            g: w * min(1.0, self.group_vol_target / vols[g])
            for g, w in w_raw.items()
        }

        # -----------------------------
        # Hierarchical gating
        # corp → high yield
        # -----------------------------
        if "corp_bonds" not in edges and "high_yield_bonds" in w_scaled:
            w_scaled.pop("high_yield_bonds")

        risk_weight = sum(w_scaled.values())
        cash_weight = max(0.0, 1.0 - risk_weight)

        # -----------------------------
        # Allocate
        # -----------------------------
        for g, wg in w_scaled.items():
            syms = [s for s in risk_groups[g] if self.PassesTrend(s)]
            if not syms:
                continue

            alloc = wg / len(syms)
            for s in syms:
                self.SetHoldings(s, alloc)

        self.SetHoldings(self.symbols["cash"][0], cash_weight)

        self.Debug(
            f"{self.Time.date()} | risk={round(risk_weight,3)} "
            f"cash={round(cash_weight,3)} | "
            + ", ".join(f"{g}:{round(w,3)}" for g, w in w_scaled.items())
        )

    # ====================================================
    # Helpers (UNCHANGED LOGIC)
    # ====================================================
    def GetDurationRegimeForGroup(self, closes, symbols):

        def mom(s):
            if s not in closes.columns:
                return -np.inf
            px = closes[s]
            if len(px) < 127:
                return -np.inf
            return np.mean([
                px.iloc[-1] / px.iloc[-21] - 1,
                px.iloc[-1] / px.iloc[-63] - 1,
                px.iloc[-1] / px.iloc[-126] - 1,
            ])

        # -----------------------------
        # 3-asset duration ladder
        # -----------------------------
        if len(symbols) == 3:
            s, i, l = symbols
            return (
                [s, i, l] if mom(l) > mom(i) > mom(s)
                else [s, i] if mom(i) > mom(s)
                else [s]
            )

        # -----------------------------
        # 2-asset ladder (HY case)
        # -----------------------------
        if len(symbols) == 2:
            s, l = symbols
            return [s, l] if mom(l) > mom(s) else [s]

        # -----------------------------
        # Fallback (single asset)
        # -----------------------------
        return symbols

    def ComputeGroupMomentum(self, symbols, closes):
        moms = []
        for s in symbols:
            if s not in closes.columns:
                continue
            px = closes[s]
            if len(px) < 253:
                continue
            moms.append(np.mean([
                px.iloc[-1] / px.iloc[-21] - 1,
                px.iloc[-1] / px.iloc[-63] - 1,
                px.iloc[-1] / px.iloc[-126] - 1,
                px.iloc[-1] / px.iloc[-189] - 1,
                px.iloc[-1] / px.iloc[-252] - 1,
            ]))
        return float(np.mean(moms)) if moms else 0.0
