from scipy.optimize import brentq
import math
import pandas as pd
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo


EXCHANGE_TIMEZONES = {
    "NMS": ZoneInfo("America/New_York"),
    "NYQ": ZoneInfo("America/New_York"),
    "PCX": ZoneInfo("America/New_York"),
    "CBOE": ZoneInfo("America/Chicago"),
    "LSE": ZoneInfo("Europe/London"),
    "FRA": ZoneInfo("Europe/Berlin"),
}


TENOR_TO_ID = {
    0.5: "DGS6MO",
    1.0: "DGS1",
    2.0: "DGS2",
    3.0: "DGS3",
    5.0: "DGS5",
    7.0: "DGS7",
    10.0: "DGS10",
}

TBILLS = {1 / 12: "DGS1MO", 0.25: "DGS3MO"}

COUPON_FRED = 2
DAY_CONVENTION = "ACT/ACT"


DATA_DIR = Path("../data/rates/processed")


class ZeroRatesEngine:
    @staticmethod
    def calcZeroRates(yields: pd.Series) -> dict:
        """Calculate discount factors from par yields. Returns dict mapping tenor to discount factor."""
        # Ensure yields sorted by tenor and numeric
        ys = pd.Series({float(k): float(v) for k, v in yields.items()}).sort_index()  # type: ignore

        discountRates = {}
        # First 6M discount factor from simple discounting
        if 0.5 in ys.index and not math.isnan(ys.loc[0.5]):
            discountRates[0.5] = 100 / (100 + 100 * ys.loc[0.5] / 2)
        else:
            # If 0.5 not available, assume flat zero rate from shortest par
            r = ys.iloc[0]
            discountRates[0.5] = math.exp(-r * 0.5)

        def price_error(x, prev_t, tenor_t, cpn, df_prev):
            # Sum of known coupons up to prev_t
            known_sum = cpn * sum(
                discountRates[t] for t in discountRates.keys() if t <= prev_t
            )
            steps = int((tenor_t - prev_t) * 2)
            # Future semiannual coupons discounted with forward x
            future_sum = 0.0
            for a in range(1, steps):
                future_sum += cpn * df_prev * math.exp(x * -(a * 0.5))
            final_leg = (100 + cpn) * df_prev * math.exp(x * -(tenor_t - prev_t))
            return known_sum + future_sum + final_leg - 100.0

        for tenor, rate in ys.items():
            if tenor in discountRates:
                continue
            prev = max(discountRates.keys())
            c = 100.0 * rate / 2.0

            # Find a bracketing interval with sign change
            a, b = -0.5, 5.0
            fa = price_error(a, prev, tenor, c, discountRates[prev])
            fb = price_error(b, prev, tenor, c, discountRates[prev])
            expand = 0
            while fa * fb > 0 and expand < 10:
                # Expand bounds
                a -= 0.5
                b += 0.5
                fa = price_error(a, prev, tenor, c, discountRates[prev])
                fb = price_error(b, prev, tenor, c, discountRates[prev])
                expand += 1

            if fa * fb > 0:
                # Fallback: assume flat forward equal to par rate
                forwardRate = max(rate, 0.0)
            else:
                forwardRate = brentq(
                    lambda x: price_error(x, prev, tenor, c, discountRates[prev]), a, b
                )

            # Populate intermediate half-year discount factors to reach this tenor
            steps = int((tenor - prev) * 2)
            for k in range(1, steps + 1):
                t = prev + k * 0.5
                discountRates[t] = discountRates[prev] * math.exp(
                    -forwardRate * (k * 0.5)  # type: ignore
                )

        # Filter to only bond tenors we track (excluding interpolated intermediates)
        desired_bond_tenors = sorted(TENOR_TO_ID.keys())
        filtered = {
            t: discountRates[t] for t in desired_bond_tenors if t in discountRates
        }
        return filtered

    @staticmethod
    def interpolate_zero_rate(df, tte_col="T"):
        csv_df = pd.read_csv(DATA_DIR / "discount_factors.csv")
        row = csv_df.iloc[-1][1:].values.tolist()
        row = np.insert(row, 0, 1)
        row = np.log(row)
        tenors = [0, 1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10]
        risk_free = -np.interp(df[tte_col], tenors, row) / df[tte_col]
        return risk_free
