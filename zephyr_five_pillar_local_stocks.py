import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List


class ZephyrFivePillarLocal:
    """
    Local Python mirror of QC ZephyrFivePillar
    """

    # ====================================================
    # Init
    # ====================================================
    def __init__(
        self,
        start="2012-01-01",
        enable_sma_filter=True,
        winrate_lookback=63,
        vol_lookback=63,
        sma_period=168,
        stock_ema_period=168,
        sector_count=4,
        stock_count=5,
        group_vol_target=0.10,
        crypto_cap=0.10,
    ):
        self.start = start
        self.enable_sma_filter = enable_sma_filter
        self.winrate_lookback = winrate_lookback
        self.vol_lookback = vol_lookback
        self.sma_period = sma_period
        self.stock_ema_period = stock_ema_period
        self.sector_count = sector_count
        self.stock_count = stock_count
        self.group_vol_target = group_vol_target
        self.crypto_cap = crypto_cap

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)

        # ------------------------------------------------
        # Static groups (QC identical)
        # ------------------------------------------------
        self.groups = {
            "real": ["GLD", "DBC"],
            "corp_bonds": ["VCSH", "VCIT", "VCLT"],
            "treasury_bonds": ["VGSH", "VGIT", "VGLT"],
            "equities": ["SPY", "QQQ", "DIA"],
            "sectors": [
                "VOX", "VCR", "VDC", "VDE", "VFH", "VHT",
                "VIS", "VGT", "VAW", "VNQ", "VPU"
            ],
            "crypto": ["IBIT", "ETHA"],
            "cash": ["BIL"],
        }

        # ------------------------------------------------
        # Dynamic stock universe (from CSV)
        # ------------------------------------------------
        self.stock_symbols = (
            pd.read_csv("stocks.csv")
            .loc[:, "Symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        self.all_tickers = sorted(
            set(
                sum(self.groups.values(), [])
                + self.stock_symbols
            )
        )

        self.prices = None
        self.sma = None
        self.stock_ema = None

    # ====================================================
    # Data
    # ====================================================
    def fetch_data(self):
        raw = yf.download(
            self.all_tickers,
            start=self.start,
            auto_adjust=True,
            progress=False,
        )

        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices = prices.dropna(how="all").sort_index()

        self.prices = prices

        # Indicators
        self.sma = prices.rolling(self.sma_period).mean()
        self.stock_ema = prices.ewm(span=self.stock_ema_period, adjust=False).mean()

    # ====================================================
    # Trend logic (QC exact)
    # ====================================================
    def passes_trend(self, ticker, date) -> bool:
        px = self.prices.at[date, ticker]

        # Nasdaq stocks → EMA
        if ticker in self.stock_symbols:
            ema = self.stock_ema.at[date, ticker]
            return px > ema

        # Everything else → SMA
        if not self.enable_sma_filter:
            return True

        sma = self.sma.at[date, ticker]
        return px > sma

    # ====================================================
    # Momentum helpers
    # ====================================================
    def compute_momentum(self, tickers, date) -> Dict[str, float]:
        scores = {}
        for t in tickers:
            if t not in self.prices.columns:
                continue
            px = self.prices[t].loc[:date].dropna()
            if len(px) < self.max_lookback + 1:
                continue

            scores[t] = float(np.mean([
                px.iloc[-1] / px.iloc[-(lb + 1)] - 1
                for lb in self.momentum_lookbacks
            ]))
        return scores

    def compute_group_momentum(self, tickers, date) -> float:
        moms = []
        for t in tickers:
            if t not in self.prices.columns:
                continue
            px = self.prices[t].loc[:date]
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

    # ====================================================
    # Duration regime (QC exact)
    # ====================================================
    def duration_regime(self, symbols, date):
        if len(symbols) != 3:
            return symbols

        def mom(t):
            px = self.prices[t].loc[:date]
            return np.mean([
                px.iloc[-1] / px.iloc[-22] - 1,
                px.iloc[-1] / px.iloc[-64] - 1,
                px.iloc[-1] / px.iloc[-127] - 1,
            ])

        s, i, l = symbols
        if mom(l) > mom(i) > mom(s):
            return [s, i, l]
        if mom(i) > mom(s):
            return [s, i]
        return [s]

    # ====================================================
    # Rebalance (QC identical)
    # ====================================================
    def rebalance(self, date) -> Dict[str, float]:
        weights = {}

        # -----------------------------
        # Duration regimes
        # -----------------------------
        corp = self.duration_regime(self.groups["corp_bonds"], date)
        treas = self.duration_regime(self.groups["treasury_bonds"], date)

        # -----------------------------
        # Sector momentum
        # -----------------------------
        sector_mom = self.compute_momentum(self.groups["sectors"], date)
        top_sectors = sorted(sector_mom, key=sector_mom.get, reverse=True)[:self.sector_count]

        # -----------------------------
        # Top stocks
        # -----------------------------
        stock_mom = self.compute_momentum(self.stock_symbols, date)
        top_stocks = sorted(stock_mom, key=stock_mom.get, reverse=True)[:self.stock_count]

        risk_groups = {
            "real": self.groups["real"],
            "corp_bonds": corp,
            "treasury_bonds": treas,
            "equities": self.groups["equities"],
            "sectors": top_sectors,
            "nasdaq100": top_stocks,
            "crypto": self.groups["crypto"],
        }

        edges, vols = {}, {}

        # -----------------------------
        # Persistence + confidence
        # -----------------------------
        for g, tickers in risk_groups.items():
            eligible = [t for t in tickers if self.passes_trend(t, date)]
            if not eligible:
                continue

            g_rets = (
                self.prices[eligible]
                .pct_change()
                .mean(axis=1)
                .dropna()
            )

            if len(g_rets) < self.winrate_lookback:
                continue

            p_win = float(np.mean(g_rets.tail(self.winrate_lookback) > 0))

            group_mom = self.compute_group_momentum(eligible, date)
            mom_std = float(np.std(g_rets.tail(self.vol_lookback)))

            confidence = abs(group_mom) / (mom_std + 1e-6)
            confidence = np.clip(confidence, 0.0, 2.0)

            scale = max(0.1, 1.0 + confidence * np.sign(group_mom))
            edge = p_win * scale

            g_vol = float(np.std(np.log1p(g_rets.tail(self.vol_lookback))) * np.sqrt(252))
            if g_vol <= 0:
                continue

            edges[g] = edge
            vols[g] = g_vol

        if not edges:
            return {"BIL": 1.0}

        # -----------------------------
        # Normalize + vol target
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

        # -----------------------------
        # Allocate
        # -----------------------------
        allocated = 0.0

        for g, wg in w_scaled.items():
            eligible = [t for t in risk_groups[g] if self.passes_trend(t, date)]
            if not eligible:
                continue

            alloc = wg / len(eligible)
            for t in eligible:
                weights[t] = alloc
                allocated += alloc

        weights["BIL"] = max(0.0, 1.0 - allocated)
        return weights

    # ====================================================
    # Run snapshot
    # ====================================================
    def run(self):
        self.fetch_data()
        date = self.prices.index[-1]
        w = self.rebalance(date)

        df = (
            pd.DataFrame.from_dict(w, orient="index", columns=["weight"])
            .sort_values("weight", ascending=False)
        )
        df["date"] = date
        return df


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    strat = ZephyrFivePillarLocal(start="2018-01-01")
    df = strat.run()

    print(f"\nMost recent portfolio ({df['date'].iloc[0].date()}):\n")
    for asset, row in df.iterrows():
        print(f"{asset:>8s} : {row['weight']:.3f}")
