from __future__ import annotations
from datetime import datetime, date, time as time_cls
from typing import List, Optional, Union, Iterable
import pytz

ET_TZ = pytz.timezone("US/Eastern")
UTC_TZ = pytz.utc


def tte(
    expiry: List[datetime],
    now_utc: Optional[datetime] = datetime.now(UTC_TZ),
    market_close_et: time_cls = time_cls(16, 0, 0),
    min_tte: float = 1e-5,
):
    """
    Compute Time-To-Expiry (TTE) in years using 4:00 PM US/Eastern as option expiry time.

    - Accepts single `date`/`datetime` or an iterable/Pandas Series of them.
    - Uses `now_utc` if provided; otherwise current UTC.
    - Returns strictly positive values (clamped to `min_tte`).
    - Year basis: 365-day.
    """
    if now_utc is None:
        now_utc = datetime.now(UTC_TZ)

    def _one(e: Union[date, datetime]) -> float:
        exp_date = e.date() if isinstance(e, datetime) else e
        expiry_dt_naive = datetime.combine(exp_date, market_close_et)
        expiry_dt_et = ET_TZ.localize(expiry_dt_naive)
        expiry_dt_utc = expiry_dt_et.astimezone(UTC_TZ)
        diff_seconds = (expiry_dt_utc - now_utc).total_seconds()
        years = diff_seconds / (365.0 * 24.0 * 3600.0)
        return max(years, min_tte)

    # Vectorized path for Pandas Series
    try:
        import pandas as pd  # type: ignore

        if isinstance(expiry, pd.Series):
            return expiry.apply(lambda e: _one(e))
    except Exception:
        pass

    # Iterable path
    if isinstance(expiry, Iterable) and not isinstance(expiry, (str, bytes)):
        return [_one(e) for e in expiry]

    # Scalar path
    return _one(expiry)  # type: ignore[arg-type]
