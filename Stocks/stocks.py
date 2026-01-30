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

        for sector in sorted(sector_buckets.keys()):
            stocks = sector_buckets[sector]
            stocks.sort(key=lambda f: f.market_cap, reverse=True)
            selected.extend(s.symbol for s in stocks[:75])

        return selected


# ====================================================
# Stock-Only Momentum Strategy
# ====================================================
class StockOnlyMomentum(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetCash(1_000_000)

        # -----------------------------
        # Strategy parameters
        # -----------------------------
        self.stock_count = 10
        self.ema_period = 210
        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)
        self.vol_lookback = 126
        self.winrate_lookback = 126
        self.max_weight = 0.25

        # Correlation controls
        self.corr_lookback = 21
        self.corr_kill_threshold = float(self.get_parameter("corr_threshold"))
        # self.corr_kill_threshold = 0.60

        # Market Risk Controls
        self.corr_z_lookback = 252
        self.corr_history = []
        # Risk Scale Factor
        self.min_scale = 0.40
        # Exponentail Decay Rate
        self.corr_k = 1.0
        self.corr_warmup_done = False

        # Hysteresis / slow re-risking
        self.current_scale = 1.0
        self.scale_recovery_months = 6

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

        self.Schedule.On(
            self.DateRules.MonthEnd(),
            self.TimeRules.BeforeMarketClose("SPY", 5),
            self.Rebalance
        )

        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 5),
            self.DailyRiskCheck
        )

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
                f"AvgCorr={avg_corr:.2f} → CASH"
            )

    def OnData(self, data):
        # ------------------------------------------------
        # Warm-up correlation history bootstrap
        # ------------------------------------------------
        if not self.IsWarmingUp or self.corr_warmup_done:
            return

        # Use currently available universe symbols
        symbols = sorted(self.stock_symbols)
        if len(symbols) < 2:
            return

        avg_corr = self.ComputeAverageCorrelation(symbols)
        if avg_corr is None or not np.isfinite(avg_corr):
            return

        self.corr_history.append(avg_corr)
        if len(self.corr_history) > self.corr_z_lookback:
            self.corr_history.pop(0)

        # Once we have enough history, lock it in
        if len(self.corr_history) >= min(30, self.corr_z_lookback):
            self.corr_warmup_done = True

            self.Debug(
                f"{self.Time.date()} | CORR WARMUP COMPLETE | "
                f"n={len(self.corr_history)} "
                f"μ={np.mean(self.corr_history):.3f} "
                f"σ={np.std(self.corr_history):.3f}"
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

        # ------------------------------------------------
        # Correlation (single evaluation)
        # ------------------------------------------------
        avg_corr = self.ComputeAverageCorrelation(top)

        if avg_corr is not None and np.isfinite(avg_corr):
            self.corr_history.append(avg_corr)
            if len(self.corr_history) > self.corr_z_lookback:
                self.corr_history.pop(0)

        # ------------------------------------------------
        # Kill switch check
        # ------------------------------------------------
        if self.kill_switch_active:
            if avg_corr is None or avg_corr >= self.corr_kill_threshold:
                self.Debug(
                    f"{self.Time.date()} | REBALANCE SKIPPED | "
                    f"Kill switch ACTIVE | AvgCorr={avg_corr}"
                )
                return

            self.Debug(
                f"{self.Time.date()} | REBALANCE RESUME | "
                f"Kill switch CLEARED | AvgCorr={avg_corr:.2f}"
            )
            self.kill_switch_active = False

        # ------------------------------------------------
        # Correlation-based scaling with hysteresis
        # ------------------------------------------------
        target_scale = self.ComputeCorrScaler(avg_corr)

        if target_scale < self.current_scale:
            self.current_scale = target_scale
        else:
            alpha = 1.0 / self.scale_recovery_months
            self.current_scale += alpha * (target_scale - self.current_scale)

        vol_scale = self.current_scale

        self.Debug(
            f"{self.Time.date()} | SCALE | "
            f"AvgCorr={avg_corr:.3f} "
            f"Target={target_scale:.2f} "
            f"Current={vol_scale:.2f}"
        )

        # ------------------------------------------------
        # Per-stock edge (group logic applied per asset)
        # ------------------------------------------------
        edges = {}

        for s in top:
            px = closes[s].dropna()
            if len(px) < max(self.winrate_lookback, self.vol_lookback) + 1:
                continue

            simple_returns = px.pct_change().dropna()
            log_returns = np.log1p(simple_returns)

            asset_momentum = scores[s]

            win_rate = float(
                np.mean(log_returns.tail(self.winrate_lookback) > 0)
            )

            asset_vol = float(
                np.std(log_returns.tail(self.vol_lookback)) * np.sqrt(252)
            )

            if not np.isfinite(asset_vol) or asset_vol <= 0:
                continue

            confidence = np.clip(
                abs(asset_momentum) / (asset_vol + 1e-6),
                0.0,
                2.0
            )

            scale = max(0.1, 1.0 + confidence * np.sign(asset_momentum))
            edge = win_rate * scale

            if np.isfinite(edge) and edge > 0:
                edges[s] = edge

        if not edges:
            return

        total = sum(edges.values())
        base_weights = {s: w / total for s, w in edges.items()}


        # ------------------------------------------------
        # Iterative cap with redistribution
        # ------------------------------------------------
        weights = base_weights.copy()

        while True:
            excess = 0.0
            free = {}

            for s in sorted(weights):
                w = weights[s]
                if w > self.max_weight:
                    excess += w - self.max_weight
                    weights[s] = self.max_weight
                else:
                    free[s] = w

            if excess <= 1e-8 or not free:
                break

            free_total = sum(free.values())
            redistributed = False

            for s in sorted(free):
                add = excess * (free[s] / free_total)
                new_w = weights[s] + add
                if new_w > self.max_weight:
                    weights[s] = self.max_weight
                    redistributed = True
                else:
                    weights[s] = new_w

            if not redistributed:
                break

        total = sum(weights.values())
        if total > 0:
            for s in sorted(weights):
                weights[s] /= total

        final_weights = {s: w * vol_scale for s, w in weights.items()}

        # ------------------------------------------------
        # Execute trades
        # ------------------------------------------------
        self.Liquidate()
        for s, w in final_weights.items():
            if w > 0:
                self.SetHoldings(s, w)

        # ------------------------------------------------
        # Debug output (FULL VISIBILITY)
        # ------------------------------------------------
        total_alloc = sum(final_weights.values())
        cash_weight = max(0.0, 1.0 - total_alloc)

        weights_str = ", ".join(
            f"{s.Value}:{w:.2%}" for s, w in final_weights.items()
        )

        self.Debug(
            f"{self.Time.date()} | n={len(final_weights)} | "
            f"AvgCorr={avg_corr:.2f} | "
            f"Scale={vol_scale:.2f} | "
            f"CASH:{cash_weight:.2%} | "
            f"{weights_str}"
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
        if avg_corr is None:
            return 1.0

        if len(self.corr_history) < 5:
            return 1.0

        mu = np.mean(self.corr_history)
        sigma = np.std(self.corr_history)

        if sigma <= 1e-6:
            return 1.0

        z = max(0.0, (avg_corr - mu) / sigma)
        scale = self.min_scale + (1.0 - self.min_scale) * np.exp(-self.corr_k * z)

        self.Debug(
            f"{self.Time.date()} | CORR Z | "
            f"μ={mu:.3f} σ={sigma:.3f} z={z:.2f} scale={scale:.2f}"
        )

        return scale
