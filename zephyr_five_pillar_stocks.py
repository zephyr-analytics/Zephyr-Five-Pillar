from AlgorithmImports import *
from typing import Dict, List
import numpy as np
import pandas as pd


# ====================================================
# Nasdaq-100 Constituents Universe
# ====================================================
class ConstituentsUniverseSelectionModel(ETFConstituentsUniverseSelectionModel):
    def __init__(self, universe_settings: UniverseSettings = None) -> None:
        symbol = Symbol.Create("SPY", SecurityType.Equity, Market.USA)
        super().__init__(
            symbol,
            universe_settings,
            lambda constituents: [c.Symbol for c in constituents]
        )


class ZephyrFivePillar(QCAlgorithm):

    def Initialize(self) -> None:
        self.SetStartDate(2012, 1, 1)
        self.SetCash(100_000)

        # ============================
        # User Options
        # ============================
        self.enable_sma_filter = True
        self.winrate_lookback = 63
        self.vol_lookback = 63
        self.sma_period = 168
        self.stock_ema_period = 168
        self.sector_count = 4
        self.stock_count = 10
        self.group_vol_target = 0.10
        self.crypto_cap = 0.10

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # ============================
        # Static Asset Groups
        # ============================
        self.group_tickers = {
            "real": ["GLD", "DBC"],
            "corp_bonds": ["VCSH", "VCIT", "VCLT"],
            "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
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
                    sym = self.AddCrypto(t, Resolution.Daily).Symbol
                else:
                    sym = self.AddEquity(t, Resolution.Daily).Symbol
                self.symbols[g].append(sym)
                self.Securities[sym].FeeModel = ConstantFeeModel(0)

        self.all_symbols = [s for group in self.symbols.values() for s in group]

        # ============================
        # Nasdaq-100 Dynamic Universe
        # ============================
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverseSelection(
            ConstituentsUniverseSelectionModel(self.UniverseSettings)
        )

        self.stock_symbols = set()

        # ============================
        # Indicators
        # ============================
        self.smas = {
            s: self.SMA(s, self.sma_period, Resolution.Daily)
            for s in self.all_symbols
        }

        self.stock_emas: Dict[Symbol, IndicatorBase] = {}

        self.SetWarmUp(
            max(self.winrate_lookback, self.vol_lookback,
                self.max_lookback, self.stock_ema_period)
        )

        self.Schedule.On(
            self.DateRules.MonthEnd(self.all_symbols[0]),
            self.TimeRules.BeforeMarketClose(self.all_symbols[0], 5),
            self.Rebalance
        )

    # ====================================================
    # Dynamic universe tracking
    # ====================================================
    def OnSecuritiesChanged(self, changes: SecurityChanges):
        for security in changes.AddedSecurities:
            if security.Symbol.SecurityType == SecurityType.Equity:
                self.stock_symbols.add(security.Symbol)

                # Stock EMA (189)
                if security.Symbol not in self.stock_emas:
                    self.stock_emas[security.Symbol] = self.EMA(
                        security.Symbol,
                        self.stock_ema_period,
                        Resolution.Daily
                    )

                # Ensure SMA exists too (safety)
                if security.Symbol not in self.smas:
                    self.smas[security.Symbol] = self.SMA(
                        security.Symbol,
                        self.sma_period,
                        Resolution.Daily
                    )

        for security in changes.RemovedSecurities:
            self.stock_symbols.discard(security.Symbol)
            self.stock_emas.pop(security.Symbol, None)

    # ====================================================
    # Trend logic (STOCKS vs EVERYTHING ELSE)
    # ====================================================
    def PassesTrend(self, symbol: Symbol) -> bool:
        # Nasdaq-100 stocks → 189 EMA
        if symbol in self.stock_symbols:
            ema = self.stock_emas.get(symbol)
            return ema and ema.IsReady and self.Securities[symbol].Price > ema.Current.Value

        # Everything else → SMA
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

        all_hist_syms = list(set(self.all_symbols) | self.stock_symbols)

        history = self.history(
            all_hist_syms,
            self.max_lookback + 1,
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        closes = history["close"].unstack(0)

        # -----------------------------
        # Duration regimes
        # -----------------------------
        corp_bond_syms = self.GetDurationRegimeForGroup(
            closes, self.symbols["corp_bonds"]
        )
        treasury_bond_syms = self.GetDurationRegimeForGroup(
            closes, self.symbols["treasury_bonds"]
        )

        # -----------------------------
        # Sector momentum
        # -----------------------------
        sector_mom = self.ComputeMomentum(self.symbols["sectors"], closes)
        top_sectors = sorted(
            sector_mom, key=sector_mom.get, reverse=True
        )[:self.sector_count]

        # -----------------------------
        # Nasdaq-100 Top 10
        # -----------------------------
        stock_mom = self.ComputeMomentum(list(self.stock_symbols), closes)
        top_stocks = sorted(
            stock_mom, key=stock_mom.get, reverse=True
        )[:self.stock_count]

        # -----------------------------
        # Risk groups
        # -----------------------------
        risk_groups = {
            "real": self.symbols["real"],
            "corp_bonds": corp_bond_syms,
            "treasury_bonds": treasury_bond_syms,
            "equities": self.symbols["equities"],
            "sectors": top_sectors,
            "nasdaq100": top_stocks,
            "crypto": self.symbols["crypto"]
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
            # ------------------------------------------------
            # Data-driven magnitude (signal-to-noise confidence)
            # ------------------------------------------------
            group_mom = self.ComputeGroupMomentum(eligible, closes)

            # Noise estimate from recent group returns
            mom_std = float(np.std(g_rets.tail(self.vol_lookback)))

            # Signal-to-noise confidence
            confidence = abs(group_mom) / (mom_std + 1e-6)
            confidence = np.clip(confidence, 0.0, 2.0)

            # Conviction scaling (never vetoes participation)
            scale = max(0.1, 1.0 + confidence * np.sign(group_mom))

            edge = p_win * scale

            g_vol = float(np.std(np.log1p(g_rets.tail(self.vol_lookback))) * np.sqrt(252))
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
        # Crypto cap
        # -----------------------------
        if "crypto" in w_scaled and w_scaled["crypto"] > self.crypto_cap:
            excess = w_scaled["crypto"] - self.crypto_cap
            w_scaled["crypto"] = self.crypto_cap
            others = [g for g in w_scaled if g != "crypto"]
            total_other = sum(w_scaled[g] for g in others)
            for g in others:
                w_scaled[g] += excess * (w_scaled[g] / total_other)

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

        # -----------------------------
        # DEBUG PRINT (RESTORED)
        # -----------------------------
        self.Debug(
            f"{self.Time.date()} | risk={round(risk_weight,3)} "
            f"cash={round(cash_weight,3)} | "
            + ", ".join(f"{g}:{round(w,3)}" for g, w in w_scaled.items())
        )

    # ====================================================
    # Helpers
    # ====================================================
    def ComputeMomentum(self, symbols, closes):
        scores = {}
        for s in symbols:
            if s not in closes.columns:
                continue
            px = closes[s]
            if len(px) < self.max_lookback + 1:
                continue
            scores[s] = float(np.mean([
                px.iloc[-1] / px.iloc[-(lb + 1)] - 1
                for lb in self.momentum_lookbacks
            ]))
        return scores

    def GetDurationRegimeForGroup(self, closes, symbols):
        if len(symbols) != 3:
            return symbols

        def mom(s):
            if s not in closes.columns:
                return -np.inf
            px = closes[s]
            return np.mean([
                px.iloc[-1] / px.iloc[-22] - 1,
                px.iloc[-1] / px.iloc[-64] - 1,
                px.iloc[-1] / px.iloc[-127] - 1,
            ])

        s, i, l = symbols
        return [s, i, l] if mom(l) > mom(i) > mom(s) else [s, i] if mom(i) > mom(s) else [s]

    def ComputeGroupMomentum(self, symbols, closes):
        moms = []
        for s in symbols:
            if s not in closes.columns:
                continue
            px = closes[s]
            if len(px) < 253:
                continue
            moms.append(np.mean([
                px.iloc[-1] / px.iloc[-22] - 1,
                px.iloc[-1] / px.iloc[-64] - 1,
                px.iloc[-1] / px.iloc[-127] - 1,
                px.iloc[-1] / px.iloc[-190] - 1,
                px.iloc[-1] / px.iloc[-253] - 1,
            ]))
        return float(np.mean(moms)) if moms else 0.0
