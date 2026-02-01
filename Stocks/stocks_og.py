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

            # Explicit ticker exclusion
            if f.symbol.Value in self.blacklist:
                continue

            # ---------------------------------
            # Exclude OTC and non-major exchanges
            # ---------------------------------
            exch = f.company_reference.primary_exchange_id
            if exch not in ("NYS", "NAS", "ASE"):
                continue

            if f.price is None or f.price <= 5:
                continue

            # Market cap >= $5B
            if f.market_cap is None or f.market_cap < 5_000_000_000:
                continue

            sector = f.asset_classification.morningstar_sector_code
            if sector is None:
                continue

            # -----------------------------
            # Optional Free Cash Flow filter
            # -----------------------------
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
# Stock-Only Momentum Strategy WITH Vol Scaling
# ====================================================
class StockOnlyMomentum(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetCash(100_000)
        self.blacklist = { "GME", "AMC" }

        # -----------------------------
        # Strategy parameters
        # -----------------------------
        self.stock_count = 10
        self.ema_period = 210
        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # Volatility scaling parameters
        self.vol_lookback = 252
        self.target_vol = 1.0
        self.max_leverage = 1.0
        self.max_weight = 0.20

        # -----------------------------
        # Universe & data settings
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

    # ------------------------------------------------
    # Universe tracking
    # ------------------------------------------------
    def OnSecuritiesChanged(self, changes):
        for sec in changes.AddedSecurities:
            if sec.Symbol.SecurityType != SecurityType.Equity:
                continue

            self.stock_symbols.add(sec.Symbol)
            self.ema[sec.Symbol] = self.EMA(
                sec.Symbol, self.ema_period, Resolution.Daily
            )

        for sec in changes.RemovedSecurities:
            self.stock_symbols.discard(sec.Symbol)
            self.ema.pop(sec.Symbol, None)

    # ------------------------------------------------
    # Portfolio-level volatility scaling
    # ------------------------------------------------
    def ComputeVolScaler(self, symbols):
        if len(symbols) == 0:
            return 1.0

        history = self.history(
            symbols,
            self.vol_lookback + 1,
            Resolution.Daily,
            data_normalization_mode=DataNormalizationMode.TOTAL_RETURN
        )

        if history.empty:
            return 1.0

        closes = history["close"].unstack(0)

        if closes.shape[0] < self.vol_lookback + 1:
            return 1.0

        # Equal-weight portfolio returns
        returns = closes.pct_change().dropna()
        port_returns = returns.mean(axis=1)

        realized_vol = np.std(port_returns) * np.sqrt(252)

        if realized_vol <= 0:
            return 1.0

        scale = self.target_vol / realized_vol
        return min(self.max_leverage, scale)

    # ------------------------------------------------
    # Rebalance
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

            # -----------------------------
            # Momentum FIRST
            # -----------------------------
            momentum = float(np.mean([
                px.iloc[-1] / px.iloc[-(lb + 1)] - 1
                for lb in self.momentum_lookbacks
            ]))

            # Hard momentum gate
            if momentum <= 0:
                continue

            # -----------------------------
            # EMA confirmation SECOND
            # -----------------------------
            ema = self.ema.get(s)
            if ema is None or not ema.IsReady:
                continue

            if self.Securities[s].Price <= ema.Current.Value:
                continue

            scores[s] = momentum

        if not scores:
            return

        # -----------------------------
        # Select top momentum names
        # -----------------------------
        top = sorted(scores, key=scores.get, reverse=True)[:self.stock_count]
        top_scores = {s: scores[s] for s in top}

        # -----------------------------
        # Momentum-proportional base weights (long-only)
        # -----------------------------
        raw = {s: max(0.0, v) for s, v in top_scores.items()}
        total = sum(raw.values())

        if total <= 0:
            base_weights = {s: 1.0 / len(top) for s in top}
        else:
            base_weights = {s: m / total for s, m in raw.items()}

        # -----------------------------
        # Apply 20% weight cap
        # -----------------------------
        max_w = self.max_weight
        capped = base_weights.copy()
        excess = 0.0

        for s, w in capped.items():
            if w > max_w:
                excess += w - max_w
                capped[s] = max_w

        if excess > 0:
            uncapped = {s: w for s, w in capped.items() if w < max_w}
            uncapped_total = sum(uncapped.values())

            if uncapped_total > 0:
                for s in uncapped:
                    capped[s] += excess * (uncapped[s] / uncapped_total)

        # Final safety: no negative weights
        base_weights = {s: max(0.0, w) for s, w in capped.items()}


        # -----------------------------
        # Volatility scaling
        # -----------------------------
        vol_scale = self.ComputeVolScaler(top)

        self.Liquidate()

        for s, w in base_weights.items():
            self.SetHoldings(s, w * vol_scale)

        total_alloc = sum(w * vol_scale for w in base_weights.values())
        cash_weight = max(0.0, 1.0 - total_alloc)

        weights_str = ", ".join(
            f"{s.Value}:{w * vol_scale:.2%}"
            for s, w in base_weights.items()
        )

        self.Debug(
            f"{self.Time.date()} | n={len(base_weights)} | "
            f"volScale={vol_scale:.2f} | CASH:{cash_weight:.2%} | {weights_str}"
        )
