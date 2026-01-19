import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional


class MonthlySmaFixedGroupWithBitcoin:
    """
    Monthly rebalanced multi-asset strategy with:
    - SMA filter
    - multi-horizon momentum
    - sector ranking
    - volatility targeting
    - Bitcoin inclusion
    """

    def __init__(
        self,
        start="2012-01-01",
        capital=100_000,
        enable_vol_targeting=True,
        target_vol=0.75,
        cash_buffer=0.05,
        sector_count=4,
    ):
        self.start = start
        self.capital = capital
        self.enable_vol_targeting = enable_vol_targeting
        self.target_vol = target_vol
        self.cash_buffer = cash_buffer
        self.sector_count = sector_count

        # === Group weights (must sum to 1 - cash_buffer) ===
        self.category_target_weights = {
            "real": 0.20,
            "bonds": 0.25,
            "equities": 0.20,
            "sectors": 0.25,
            "bitcoin": 0.05,
        }

        # === Tickers ===
        self.real_asset_tickers = ["GLD", "DBC"]
        self.bond_tickers = ["VCSH", "VCIT", "VCLT", "VGSH", "VGIT", "VGLT"]
        self.equity_tickers = ["SPY", "QQQ", "DIA"]
        self.sector_tickers = [
            "VOX", "VCR", "VDC", "VDE", "VFH", "VHT",
            "VIS", "VGT", "VAW", "VNQ", "VPU"
        ]

        self.bitcoin_ticker = "IBIT"

        self.momentum_lookbacks = [21, 63, 126, 189, 252]
        self.max_lookback = max(self.momentum_lookbacks)
        self.sma_period = 168
        self.vol_lookback = 21

        self.all_tickers = (
            self.real_asset_tickers
            + self.bond_tickers
            + self.equity_tickers
            + self.sector_tickers
            + [self.bitcoin_ticker]
        )

        self.prices = None
        self.sma = None

    # -------------------------------------------------------
    # Data
    # -------------------------------------------------------

    def fetch_data(self):
        raw = yf.download(
            self.all_tickers,
            start=self.start,
            auto_adjust=True,
            progress=False,
            group_by="column"
        )

        # ----------------------------
        # Normalize yfinance output
        # ----------------------------
        if isinstance(raw.columns, pd.MultiIndex):
            # Expected: ('Close', TICKER)
            if "Close" in raw.columns.get_level_values(0):
                prices = raw["Close"]
            else:
                raise ValueError("Unexpected MultiIndex format from yfinance")
        else:
            # Single index case
            prices = raw["Close"] if "Close" in raw.columns else raw

        prices = prices.sort_index()
        prices = prices.dropna(how="all")

        self.prices = prices
        self.sma = self.prices.rolling(self.sma_period).mean()

    # -------------------------------------------------------
    # Indicators
    # -------------------------------------------------------

    def passes_sma(self, ticker, date) -> bool:
        if ticker not in self.sma.columns:
            return False
        return self.prices.loc[date, ticker] > self.sma.loc[date, ticker]

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
            scores[t] = np.mean(rets)

        return scores

    def realized_vol(self, ticker, date) -> Optional[float]:
        px = self.prices[ticker].loc[:date].dropna()
        if len(px) < self.vol_lookback + 1:
            return None

        log_ret = np.log(px / px.shift(1)).dropna()
        return log_ret[-self.vol_lookback :].std() * np.sqrt(252)

    # -------------------------------------------------------
    # Rebalance
    # -------------------------------------------------------

    def rebalance(self, date) -> Dict[str, float]:
        weights = {}

        # --- Sector ranking ---
        sector_mom = self.compute_momentum(self.sector_tickers, date)
        top_sectors = sorted(
            sector_mom, key=sector_mom.get, reverse=True
        )[: self.sector_count]

        groups = {
            "real": self.real_asset_tickers,
            "bonds": self.bond_tickers,
            "equities": self.equity_tickers,
            "sectors": top_sectors,
            "bitcoin": [self.bitcoin_ticker],
        }

        allocated = 0.0

        for group, tickers in groups.items():
            group_weight = self.category_target_weights.get(group, 0.0)
            eligible = [
                t for t in tickers if self.passes_sma(t, date)
            ]

            if not eligible:
                continue

            base_w = group_weight / len(eligible)

            for t in eligible:
                w = base_w

                if self.enable_vol_targeting:
                    vol = self.realized_vol(t, date)
                    if vol:
                        w *= min(1.0, self.target_vol / vol)

                weights[t] = w
                allocated += w

        weights["CASH"] = max(0.0, 1.0 - allocated)
        return weights

    # -------------------------------------------------------
    # Run (weights only)
    # -------------------------------------------------------

    def run(self) -> pd.DataFrame:
        self.fetch_data()

        rebalance_dates = self.prices.resample("M").last().index
        records = []

        for d in rebalance_dates:
            if d not in self.prices.index:
                continue

            weights = self.rebalance(d)
            for asset, w in weights.items():
                records.append({
                    "date": d,
                    "asset": asset,
                    "weight": w,
                })

        return pd.DataFrame(records)


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    strategy = MonthlySmaFixedGroupWithBitcoin(
        start="2012-01-01",
        enable_vol_targeting=True,
        target_vol=0.75,
        sector_count=4,
    )

    weights_df = strategy.run()

    # --- Print most recent portfolio ---
    latest_date = weights_df["date"].max()
    latest_portfolio = (
        weights_df[weights_df["date"] == latest_date]
        .sort_values("weight", ascending=False)
    )

    print(f"\nMost recent portfolio ({latest_date.date()}):\n")
    for _, row in latest_portfolio.iterrows():
        print(f"{row['asset']:>8s} : {row['weight']:.3f}")
