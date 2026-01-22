import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional


class ZephyrFivePillar:
    """
    Group-based multi-asset allocator with:
    - persistence (win-rate) scoring
    - sector momentum selection
    - group-level volatility targeting
    - SMA filter
    - unified crypto sleeve (BTC + ETH)
    - residual cash
    """

    def __init__(
        self,
        start="2012-01-01",
        winrate_lookback=126,
        vol_lookback=126,
        sma_period=168,
        sector_count=4,
        group_vol_target=0.15,
        crypto_cap=0.05,
        epsilon=0.01,
    ):
        self.start = start
        self.winrate_lookback = winrate_lookback
        self.vol_lookback = vol_lookback
        self.sma_period = sma_period
        self.sector_count = sector_count
        self.group_vol_target = group_vol_target
        self.crypto_cap = crypto_cap
        self.epsilon = epsilon

        # ----------------------------
        # Groups
        # ----------------------------
        self.groups = {
            "real": ["GLD", "DBC"],
            "bonds": ["VCSH", "VCIT", "VCLT", "VGSH", "VGIT", "VGLT"],
            "equities": ["SPY", "QQQ", "DIA"],
            "sectors": [
                "VOX", "VCR", "VDC", "VDE", "VFH", "VHT",
                "VIS", "VGT", "VAW", "VNQ", "VPU"
            ],
            "crypto": ["IBIT", "ETHA"],
        }

        self.cash_ticker = "CASH"

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        self.all_tickers = sorted(
            {t for g in self.groups.values() for t in g}
        )

        self.prices = None
        self.sma = None

    # ---------------------------------------------------
    # Data
    # ---------------------------------------------------

    def fetch_data(self):
        raw = yf.download(
            self.all_tickers,
            start=self.start,
            progress=False,
        )

        prices = raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices = prices.dropna(how="all").sort_index()

        self.prices = prices
        self.sma = prices.rolling(self.sma_period).mean()

    # ---------------------------------------------------
    # Helpers
    # ---------------------------------------------------

    def passes_sma(self, ticker, date) -> bool:
        return (
            ticker in self.sma.columns
            and self.prices.loc[date, ticker] > self.sma.loc[date, ticker]
        )

    def compute_momentum(self, tickers, date) -> Dict[str, float]:
        scores = {}
        for t in tickers:
            px = self.prices[t].loc[:date].dropna()
            if len(px) < self.max_lookback + 1:
                continue
            rets = [
                px.iloc[-1] / px.iloc[-(lb + 1)] - 1
                for lb in self.momentum_lookbacks
            ]
            scores[t] = float(np.mean(rets))
        return scores

    # ---------------------------------------------------
    # Rebalance
    # ---------------------------------------------------

    def rebalance(self, date) -> Dict[str, float]:
        weights: Dict[str, float] = {}

        # ----------------------------
        # 1) Sector selection
        # ----------------------------
        sector_mom = self.compute_momentum(self.groups["sectors"], date)
        top_sectors = sorted(
            sector_mom, key=sector_mom.get, reverse=True
        )[: self.sector_count]

        risk_groups = {
            "real": self.groups["real"],
            "bonds": self.groups["bonds"],
            "equities": self.groups["equities"],
            "sectors": top_sectors,
            "crypto": self.groups["crypto"],
        }

        edges = {}
        vols = {}

        # ----------------------------
        # 2) Persistence + vol (group)
        # ----------------------------
        for g, tickers in risk_groups.items():
            eligible = [
                t for t in tickers if self.passes_sma(t, date)
            ]
            if not eligible:
                continue

            g_rets = (
                self.prices[eligible]
                .pct_change()
                .mean(axis=1)
                .dropna()
            )

            if len(g_rets) < max(self.winrate_lookback, self.vol_lookback):
                continue

            p_win = float(
                np.mean(g_rets.tail(self.winrate_lookback) > 0)
            )

            edges[g] = p_win

            g_log = np.log1p(g_rets.tail(self.vol_lookback))
            vols[g] = float(g_log.std() * np.sqrt(252))

        if not edges:
            return {self.cash_ticker: 1.0}

        # ----------------------------
        # 3) Normalize edges (epsilon)
        # ----------------------------
        edge_eff = {g: e + self.epsilon for g, e in edges.items()}
        total_edge = sum(edge_eff.values())

        w_raw = {g: e / total_edge for g, e in edge_eff.items()}

        # ----------------------------
        # 4) Group-level vol targeting
        # ----------------------------
        w_scaled = {}
        for g, w in w_raw.items():
            vol_scalar = min(1.0, self.group_vol_target / vols[g])
            w_scaled[g] = w * vol_scalar

        # ----------------------------
        # 5) Crypto sleeve cap
        # ----------------------------
        if "crypto" in w_scaled and w_scaled["crypto"] > self.crypto_cap:
            excess = w_scaled["crypto"] - self.crypto_cap
            w_scaled["crypto"] = self.crypto_cap

            others = [g for g in w_scaled if g != "crypto"]
            other_total = sum(w_scaled[g] for g in others)
            if other_total > 0:
                for g in others:
                    w_scaled[g] += excess * (w_scaled[g] / other_total)

        # ----------------------------
        # 6) Allocate within groups
        # ----------------------------
        allocated = 0.0

        for g, wg in w_scaled.items():
            eligible = [
                t for t in risk_groups[g] if self.passes_sma(t, date)
            ]
            if not eligible:
                continue

            alloc = wg / len(eligible)
            for t in eligible:
                weights[t] = alloc
                allocated += alloc

        # ----------------------------
        # 7) Residual cash
        # ----------------------------
        weights[self.cash_ticker] = max(0.0, 1.0 - allocated)
        return weights

    # ---------------------------------------------------
    # Run (single snapshot)
    # ---------------------------------------------------

    def run(self) -> pd.DataFrame:
        self.fetch_data()
        date = self.prices.index[-1]
        w = self.rebalance(date)

        return pd.DataFrame(
            [{"date": date, "asset": k, "weight": v} for k, v in w.items()]
        )


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    strat = ZephyrFivePillar(start="2018-01-01")
    df = strat.run()

    print(f"\nMost recent portfolio ({df['date'].iloc[0].date()}):\n")
    for _, r in df.sort_values("weight", ascending=False).iterrows():
        print(f"{r['asset']:>8s} : {r['weight']:.3f}")
