from AlgorithmImports import *
from collections import defaultdict
import numpy as np


# ====================================================
# Sector-Neutral Large-Cap Stock Universe
# ====================================================
class SectorTopUniverse(FundamentalUniverseSelectionModel):
    def __init__(self, blacklist=None, require_positive_fcf=False):
        self.blacklist = set(blacklist or [])
        self.require_positive_fcf = require_positive_fcf
        super().__init__(self._select)

    def _select(self, fundamentals):
        sector_buckets = defaultdict(list)

        for f in fundamentals:
            if not f.has_fundamental_data:
                continue
            if f.symbol.Value in self.blacklist:
                continue
            exch = f.company_reference.primary_exchange_id
            if exch not in ("NYS", "NAS", "ASE"):
                continue
            if f.price is None or f.price <= 5:
                continue
            if f.market_cap is None or f.market_cap < 5_000_000_000:
                continue

            sector = f.asset_classification.morningstar_sector_code
            if sector is None:
                continue

            if self.require_positive_fcf:
                fcf = f.financial_statements.cash_flow_statement.free_cash_flow
                if fcf is None or fcf.Value is None or fcf.Value <= 0:
                    continue

            sector_buckets[sector].append(f)

        selected = []
        for _, stocks in sector_buckets.items():
            stocks.sort(key=lambda f: f.market_cap, reverse=True)
            selected.extend(s.symbol for s in stocks[:75])

        return selected


# ====================================================
# Stock-Only Momentum Strategy
# ====================================================
class StockOnlyMomentum(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetCash(100_000)

        # -----------------------------
        # Strategy parameters
        # -----------------------------
        self.stock_count = 10
        self.ema_period = 210
        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        self.max_weight = 0.25
        self.max_leverage = 1.0
        self.use_vol_scaling = True

        # Correlation controls
        self.corr_lookback = 21
        self.corr_kill_threshold = float(self.get_parameter("corr_threshold"))

        # Correlation scaler band
        self.corr_floor = float(self.get_parameter("corr_floor"))
        self.corr_ceiling = float(self.get_parameter("corr_ceiling"))

        # Risk state
        self.kill_switch_active = False

        # -----------------------------
        # Universe & settings
        # -----------------------------
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.TOTAL_RETURN
        self.SetUniverseSelection(SectorTopUniverse())

        self.stock_symbols = set()
        self.ema = {}

        self.SetWarmUp(self.max_lookback + self.ema_period)

        # Monthly rebalance
        self.Schedule.On(
            self.DateRules.MonthEnd(),
            self.TimeRules.BeforeMarketClose("SPY", 5),
            self.Rebalance
        )

        # Daily kill switch
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.BeforeMarketClose("SPY", 5),
            self.DailyRiskCheck
        )

    # ------------------------------------------------
    # Daily correlation kill switch
    # ------------------------------------------------
    def DailyRiskCheck(self):
        if self.IsWarmingUp or self.kill_switch_active:
            return

        invested = [s for s in self.stock_symbols if self.Portfolio[s].Invested]
        if len(invested) < 2:
            return

        avg_corr = self.ComputeAverageCorrelation(invested)
        if avg_corr is None:
            return

        if avg_corr >= self.corr_kill_threshold:
            self.Liquidate()
            self.kill_switch_active = True

            self.Debug(
                f"{self.Time.date()} | CORR KILL | "
                f"AvgCorr={avg_corr:.2f} → CASH (until rebalance)"
            )

    # ------------------------------------------------
    # Universe tracking
    # ------------------------------------------------
    def OnSecuritiesChanged(self, changes):
        for sec in changes.AddedSecurities:
            if sec.Symbol.SecurityType != SecurityType.Equity:
                continue

            sec.SetFeeModel(ConstantFeeModel(0))
            self.stock_symbols.add(sec.Symbol)
            self.ema[sec.Symbol] = self.EMA(
                sec.Symbol, self.ema_period, Resolution.Daily
            )

        for sec in changes.RemovedSecurities:
            self.stock_symbols.discard(sec.Symbol)
            self.ema.pop(sec.Symbol, None)

    # ------------------------------------------------
    # Monthly rebalance
    # ------------------------------------------------
    def Rebalance(self):
        if self.IsWarmingUp:
            return
        if self.kill_switch_active:
            self.Debug(
                f"{self.Time.date()} | REBALANCE WINDOW | "
                f"Kill switch reset"
            )
            self.kill_switch_active = False

        history = self.history(
            list(self.stock_symbols),
            self.max_lookback + 1,
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        if history.empty:
            return

        closes = history["close"].unstack(0)
        scores = {}

        for s in self.stock_symbols:
            if s not in closes.columns:
                continue
            if not self.Securities[s].IsTradable:
                continue

            px = closes[s]
            if len(px) < self.max_lookback + 1:
                continue

            momentum = np.mean([
                px.iloc[-1] / px.iloc[-(lb + 1)] - 1
                for lb in self.momentum_lookbacks
            ])

            if momentum <= 0:
                continue

            ema = self.ema.get(s)
            if ema is None or not ema.IsReady:
                continue

            if self.Securities[s].Price <= ema.Current.Value:
                continue

            scores[s] = momentum

        if not scores:
            return

        top = sorted(scores, key=scores.get, reverse=True)[:self.stock_count]

        # -----------------------------
        # Correlation-based scaling
        # -----------------------------
        avg_corr = self.ComputeAverageCorrelation(top)
        vol_scale = self.ComputeCorrScaler(avg_corr)

        # Raw momentum weights
        raw = {s: scores[s] for s in top}
        total = sum(raw.values())
        base_weights = {s: raw[s] / total for s in raw}

        # -----------------------------
        # Apply 20% cap with redistribution
        # -----------------------------
        capped = base_weights.copy()
        excess = 0.0

        for s, w in capped.items():
            if w > self.max_weight:
                excess += w - self.max_weight
                capped[s] = self.max_weight

        if excess > 0:
            uncapped = {s: w for s, w in capped.items() if w < self.max_weight}
            uncapped_total = sum(uncapped.values())
            if uncapped_total > 0:
                for s in uncapped:
                    capped[s] += excess * (uncapped[s] / uncapped_total)

        # Final scaled weights
        final_weights = {s: w * vol_scale for s, w in capped.items()}

        # -----------------------------
        # Execute trades
        # -----------------------------
        self.Liquidate()
        for s, w in final_weights.items():
            if w > 0:
                self.SetHoldings(s, w)

        # -----------------------------
        # Debug output (FULL VISIBILITY)
        # -----------------------------
        total_alloc = sum(final_weights.values())
        cash_weight = max(0.0, 1.0 - total_alloc)

        weights_str = ", ".join(
            f"{s.Value}:{w:.2%}" for s, w in final_weights.items()
        )

        self.Debug(
            f"{self.Time.date()} | n={len(final_weights)} | "
            f"AvgCorr={avg_corr:.2f} | Scale={vol_scale:.2f} | "
            f"CASH:{cash_weight:.2%} | {weights_str}"
        )


    # ------------------------------------------------
    # Correlation helpers
    # ------------------------------------------------
    def ComputeAverageCorrelation(self, symbols):
        if len(symbols) < 2:
            return None

        history = self.history(
            symbols,
            self.corr_lookback + 1,
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        if history.empty:
            return None

        closes = history["close"].unstack(0)
        returns = closes.pct_change().dropna()

        if returns.shape[1] < 2:
            return None

        corr = returns.corr()
        avg_corr = corr.values[np.triu_indices_from(corr, k=1)].mean()

        return None if np.isnan(avg_corr) else avg_corr

    def ComputeCorrScaler(self, avg_corr):
        if not self.use_vol_scaling or avg_corr is None:
            return 1.0

        if avg_corr <= self.corr_floor:
            scale = 1.0
        elif avg_corr >= self.corr_ceiling:
            scale = 0.25
        else:
            scale = 1.0 - (avg_corr - self.corr_floor) / (
                self.corr_ceiling - self.corr_floor
            )
            scale = max(0.25, scale)

        return min(self.max_leverage, scale)
