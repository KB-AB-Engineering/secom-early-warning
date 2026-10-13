"""Leakage-safe preprocessing.

Every step is an sklearn transformer, so it is fitted on the training lots of a
fold only. Selecting sensors on the full data set before cross-validation is the
most common mistake in published SECOM studies; ``experiments`` measures how
much it inflates the results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin


class SensorFilter(BaseEstimator, TransformerMixin):
    """Drop sensors that are mostly missing, constant or near-duplicates."""

    def __init__(self, max_missing: float = 0.5, min_unique: int = 3, max_corr: float | None = 0.98):
        self.max_missing = max_missing
        self.min_unique = min_unique
        self.max_corr = max_corr

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        keep = X.columns[(X.isna().mean() <= self.max_missing) & (X.nunique(dropna=True) >= self.min_unique)]
        if self.max_corr is not None and len(keep) > 1:
            corr = X[keep].corr().abs().to_numpy(copy=True)
            np.fill_diagonal(corr, 0.0)
            dropped: set[int] = set()
            for i in range(len(keep)):
                if i in dropped:
                    continue
                for j in np.where(corr[i, i + 1:] > self.max_corr)[0] + i + 1:
                    dropped.add(int(j))
            keep = keep[[i for i in range(len(keep)) if i not in dropped]]
        self.columns_ = list(keep)
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_)


def rank_sensors(X: pd.DataFrame, y: np.ndarray, min_fail_obs: int = 5) -> pd.Series:
    """Two-sided Mann-Whitney p-value of each sensor, failed vs passed lots."""
    y = np.asarray(y)
    p = {}
    for c in X.columns:
        a = X[c].values[y == 1]
        b = X[c].values[y == 0]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < min_fail_obs or len(b) < min_fail_obs or np.ptp(np.r_[a, b]) == 0:
            continue
        p[c] = mannwhitneyu(a, b).pvalue
    return pd.Series(p, dtype=float).sort_values()


class MannWhitneyScreen(BaseEstimator, TransformerMixin):
    """Keep the k sensors that best separate failed from passed training lots."""

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, X: pd.DataFrame, y):
        ranking = rank_sensors(pd.DataFrame(X), y)
        self.pvalues_ = ranking
        self.columns_ = list(ranking.index[: self.k])
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_)
