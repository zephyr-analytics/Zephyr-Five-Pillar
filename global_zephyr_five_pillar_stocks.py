from AlgorithmImports import *
from typing import Dict, List
import numpy as np
import pandas as pd
from collections import defaultdict


# ====================================================
# Sector-Neutral Large-Cap Universe
# Top 20 stocks per sector by market cap
# Modern Fundamental Universe (Python-safe)
# ====================================================
class SectorTop20UniverseSelectionModel(FundamentalUniverseSelectionModel):
    def __init__(self):
        super().__init__(self._select)

    def _select(self, fundamentals):

        sector_buckets = defaultdict(list)

        for f in fundamentals:
            if not f.has_fundamental_data:
                continue

            # Basic investability filters
            if f.price is None or f.price <= 5:
                continue
            if f.market_cap is None or f.market_cap <= 0:
                continue

            # Sector classification
            sector = f.asset_classification.morningstar_sector_code
            if sector is None:
                continue

            # -----------------------------
            # Quality filter: Positive Free Cash Flow
            # -----------------------------
            fcf = f.financial_statements.cash_flow_statement.free_cash_flow
            if fcf is None or fcf.Value is None or fcf.Value <= 0:
                continue

            sector_buckets[sector].append(f)

        selected = []

        # Top 20 by market cap per sector
        for sector, stocks in sector_buckets.items():
            stocks.sort(key=lambda f: f.market_cap, reverse=True)
            selected.extend(f.symbol for f in stocks[:20])

        return selected


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
        self.bond_sma_period = 126
        self.stock_ema_period = 189
        self.sector_count = 4
        self.stock_count = 5
        self.group_vol_target = 0.10
        self.crypto_cap = 0.10

        self.enable_stocks = True
        self.enable_sectors = True
        self.enable_treasury_kill_switch = True

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # ============================
        # Static Asset Groups
        # ============================
        self.group_tickers = {
            "real": ["GLD", "DBC"],
            "corp_bonds": ["VCSH", "VCIT", "VCLT"],
            "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
            "high_yield_bonds": ["SHYG", "HYG"],
            "equities": ["VTI", "VEA", "VWO"],
            "sectors": [
                "IXP","RXI","KXI","IXC","IXG","IXJ",
                "EXI","IXN","MXI","REET","JXI"
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

        # ----------------------------
        # Bond symbol set
        # ----------------------------
        self.bond_symbols = set(
            s for g in ["corp_bonds", "treasury_bonds", "high_yield_bonds"]
            for s in self.symbols[g]
        )

        # ============================
        # Universe
        # ============================
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.TOTAL_RETURN
        self.SetUniverseSelection(SectorTop20UniverseSelectionModel())

        self.stock_symbols = set()

        # ============================
        # Indicators
        # ============================
        self.smas = {
            s: self.SMA(s, self.sma_period, Resolution.Daily)
            for s in self.all_symbols if s not in self.bond_symbols
        }

        self.bond_smas = {
            s: self.SMA(s, self.bond_sma_period, Resolution.Daily)
            for s in self.bond_symbols
        }

        self.stock_emas: Dict[Symbol, IndicatorBase] = {}

        self.SetWarmUp(
            max(self.winrate_lookback, self.vol_lookback,
                self.max_lookback, self.stock_ema_period, self.bond_sma_period)
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

                if security.Symbol not in self.stock_emas:
                    self.stock_emas[security.Symbol] = self.EMA(
                        security.Symbol,
                        self.stock_ema_period,
                        Resolution.Daily
                    )

        for security in changes.RemovedSecurities:
            self.stock_symbols.discard(security.Symbol)
            self.stock_emas.pop(security.Symbol, None)

    # ====================================================
    # Trend logic
    # ====================================================
    def PassesTrend(self, symbol: Symbol) -> bool:

        # Stocks → EMA
        if symbol in self.stock_symbols:
            ema = self.stock_emas.get(symbol)
            return ema and ema.IsReady and self.Securities[symbol].Price > ema.Current.Value

        # Bonds → bond SMA
        if symbol in self.bond_symbols:
            if not self.enable_sma_filter:
                return True
            sma = self.bond_smas.get(symbol)
            return sma and sma.IsReady and self.Securities[symbol].Price > sma.Current.Value

        # Everything else → default SMA
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

        high_yield_bond_syms = self.GetDurationRegimeForGroup(
            closes, self.symbols["high_yield_bonds"]
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
            "high_yield_bonds": high_yield_bond_syms,
            "equities": self.symbols["equities"],
            "crypto": self.symbols["crypto"]
        }

        if self.enable_sectors:
            risk_groups["sectors"] = top_sectors

        if self.enable_stocks:
            risk_groups["stocks"] = top_stocks

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
        # ------------------------------------------------
        # Treasury kill switch
        # ------------------------------------------------
        if self.enable_treasury_kill_switch and "treasury_bonds" not in edges:
            self.Debug(
                f"{self.Time.date()} | TREASURY KILL SWITCH → ALL CASH"
            )
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
        # Hierarchical gating with cash handling
        # -----------------------------
        cash_sink = 0.0

        parent_children = {
            "equities": ["stocks", "sectors"],
            "treasury_bonds": ["corp_bonds"],
            "corp_bonds": ["high_yield_bonds"]
        }

        for parent, children in parent_children.items():
            if parent not in edges:
                for child in children:
                    if child in w_scaled:
                        cash_sink += w_scaled[child]
                        w_scaled.pop(child, None)

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
        if len(symbols) >= 2:
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
