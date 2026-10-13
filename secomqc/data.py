"""Loading and auditing the SECOM data set.

The widely used CSV mirror of SECOM (``uci-secom.csv``) stores timestamps that
were parsed month-first. Every date whose day is 12 or less had its day and
month swapped (19/07/2008 is correct, but 01/08/2008 became 8 January): 604 rows
were parsed that way, 582 of them changed (the other 22 have day == month).
After the swap is undone, the timestamps follow the file order (33 lots share
a timestamp with their neighbour) and span 19 July to 17 October 2008, as
documented in the original UCI release. Any chronological analysis on the
uncorrected mirror is wrong.

The rule is specific to this file: `load_secom` checks that the corrected dates
are in file order and inside the documented span, and refuses otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "uci-secom.csv"
LABEL_COL = "Pass/Fail"
TIME_COL = "Time"


def fix_swapped_timestamps(t: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Undo the day/month swap of the CSV mirror.

    Returns the corrected timestamps and a boolean mask of the rows that changed.
    """
    t = pd.to_datetime(t)
    candidate = t.dt.day <= 12
    fixed = t.copy()
    fixed[candidate] = pd.to_datetime(dict(
        year=t[candidate].dt.year, month=t[candidate].dt.day, day=t[candidate].dt.month,
        hour=t[candidate].dt.hour, minute=t[candidate].dt.minute, second=t[candidate].dt.second,
    ))
    return fixed, fixed != t


@dataclass
class Secom:
    """Chronologically ordered SECOM data."""

    X: pd.DataFrame          # 590 anonymised process measurements
    y: np.ndarray            # 1 = lot failed the final test, 0 = passed
    time: pd.Series          # corrected timestamps
    swapped: pd.Series       # rows whose timestamp was corrected

    @property
    def n(self) -> int:
        return len(self.y)


def load_secom(path: str | Path = DATA_PATH) -> Secom:
    raw = pd.read_csv(path)
    time, swapped = fix_swapped_timestamps(raw[TIME_COL])
    if not (time.is_monotonic_increasing and time.min() >= pd.Timestamp("2008-07-19")
                         and time.max() < pd.Timestamp("2008-10-18")):
        raise ValueError("timestamp correction did not give the documented SECOM order and span; "
                         "check that this is the uci-secom.csv mirror")
    order = np.argsort(time.values, kind="stable")
    raw, time, swapped = raw.iloc[order], time.iloc[order], swapped.iloc[order]
    X = raw.drop(columns=[TIME_COL, LABEL_COL]).reset_index(drop=True)
    X.columns = [f"s{c}" for c in X.columns]
    y = (raw[LABEL_COL].values == 1).astype(int)
    return Secom(X=X, y=y, time=time.reset_index(drop=True), swapped=swapped.reset_index(drop=True))


def audit(path: str | Path = DATA_PATH) -> dict:
    """Data-quality figures reported in the README."""
    raw = pd.read_csv(path)
    t_raw = pd.to_datetime(raw[TIME_COL])
    fixed, swapped = fix_swapped_timestamps(raw[TIME_COL])
    X = raw.drop(columns=[TIME_COL, LABEL_COL])
    miss = X.isna().mean()
    return {
        "lots": len(raw),
        "sensors": X.shape[1],
        "failed_lots": int((raw[LABEL_COL] == 1).sum()),
        "fail_rate": float((raw[LABEL_COL] == 1).mean()),
        "timestamps_swapped": int(swapped.sum()),
        "timestamps_swapped_share": float(swapped.mean()),
        "order_violations_raw": int((t_raw.diff().dt.total_seconds() < 0).sum()),
        "order_violations_fixed": int((fixed.diff().dt.total_seconds() < 0).sum()),
        "raw_span": f"{t_raw.min():%Y-%m-%d} to {t_raw.max():%Y-%m-%d}",
        "fixed_span": f"{fixed.min():%Y-%m-%d} to {fixed.max():%Y-%m-%d}",
        "constant_sensors": int((X.nunique(dropna=True) <= 1).sum()),
        "sensors_over_50pct_missing": int((miss > 0.5).sum()),
        "sensors_with_missing": int((miss > 0).sum()),
        "missing_share": float(X.isna().values.mean()),
    }
