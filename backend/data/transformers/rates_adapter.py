import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred

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

DATA_DIR = Path(__file__).parent.parent / "rates" / "processed"
fred = Fred(api_key=os.getenv("FRED_API_KEY"))


class RatesAdapter:
    @staticmethod
    def _csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    @staticmethod
    def _fred(series_id: str, start_date) -> pd.DataFrame:
        try:
            data = fred.get_series(series_id, observation_start=start_date)
            if data.empty:
                return pd.DataFrame()
            df = data.reset_index().rename(columns={"index": "date", 0: "value"})
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df.drop_duplicates("date").sort_values("date")
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _fetch_rates(start_date, series_dict):
        dfs = []
        for tenor, sid in series_dict.items():
            df = RatesAdapter._fred(sid, start_date)
            if df.empty:
                continue
            df[str(tenor)] = (df["value"] * 0.01).round(4)
            dfs.append(df[["date", str(tenor)]].ffill())

        if not dfs:
            return pd.DataFrame()

        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on="date", how="outer")
        return result.sort_values("date").drop_duplicates("date").ffill()

    @staticmethod
    def _build_discount_factors(par, tbill):
        from zero_rates import ZeroRatesEngine

        dates = sorted(
            set(par["date"]) | (set(tbill["date"]) if tbill is not None else set())
        )
        rows = []
        desired = sorted(list(TBILLS.keys()) + list(TENOR_TO_ID.keys()))

        for d in dates:
            prow = par[par["date"] == d]
            if prow.empty or pd.isna(prow.drop(columns=["date"]).iloc[0]).any():
                continue

            pseries = pd.Series(
                {float(c): float(prow.iloc[0][c]) for c in prow.columns if c != "date"}
            ).sort_index()
            dfs = ZeroRatesEngine.calcZeroRates(pseries)

            if tbill is not None:
                trow = tbill[tbill["date"] == d]
                if not trow.empty:
                    tr = trow.iloc[0]
                    for tenor in TBILLS.keys():
                        col = str(tenor)
                        if col in tr and not pd.isna(tr[col]):
                            dfs[tenor] = 1.0 / (1.0 + float(tr[col]) * tenor)

            row_dict = {"date": d}
            for t in desired:
                row_dict[str(t)] = dfs.get(t, np.nan)
            rows.append(row_dict)

        return pd.DataFrame(rows).sort_values("date").drop_duplicates("date")

    @staticmethod
    def generateZeroCurves():
        par = RatesAdapter._fetch_rates("2000-01-01", TENOR_TO_ID)
        tbill = RatesAdapter._fetch_rates("2000-01-01", TBILLS)
        df = RatesAdapter._build_discount_factors(par, tbill)
        df.to_csv(DATA_DIR / "discount_factors.csv", index=False)

    @staticmethod
    def updateZeroCurves():
        file_path = DATA_DIR / "discount_factors.csv"
        existing = RatesAdapter._csv(file_path)
        last_date = existing["date"].max() if not existing.empty else None
        start_date = (
            (last_date + timedelta(days=1))
            if last_date
            else datetime(2000, 1, 1).date()
        )

        par = RatesAdapter._fetch_rates(start_date, TENOR_TO_ID)
        if par.empty:
            print(f"No new data available from {start_date}")
            return

        tbill = RatesAdapter._fetch_rates(start_date, TBILLS)
        new_df = RatesAdapter._build_discount_factors(
            par, tbill if not tbill.empty else None
        )

        if new_df.empty:
            print("No valid discount factors generated")
            return

        combined = (
            (
                pd.concat([existing, new_df], ignore_index=True)
                if not existing.empty
                else new_df
            )
            .drop_duplicates("date")
            .sort_values("date")
        )

        combined.to_csv(file_path, index=False)
        print(f"Updated discount factors: added {len(new_df)} new records")

    @staticmethod
    def updateRates():
        RatesAdapter.updateZeroCurves()

    @staticmethod
    def getLastUpdateDate():
        df = RatesAdapter._csv(DATA_DIR / "discount_factors.csv")
        return df["date"].max() if not df.empty else None
