"""Evaluation protocols and metrics.

The plant question is not "what is the AUC" but "if quality can inspect only a
fraction of the lots, how many of the bad ones does the model catch?". The
metrics below answer that, out of time, with uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


# --- folds ---------------------------------------------------------------------------
def chronological_folds(n: int, first_test: float = 0.4, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds: train on everything before a block, test on the block."""
    edges = np.linspace(first_test, 1.0, n_folds + 1)
    cuts = (edges * n).astype(int)
    cuts[-1] = n
    return [(np.arange(0, cuts[f]), np.arange(cuts[f], cuts[f + 1])) for f in range(n_folds)]


# --- metrics -----------------------------------------------------------------------
def capture_at_budget(y: np.ndarray, score: np.ndarray, budget: float) -> float:
    """Share of failed lots found when inspecting the top `budget` share of lots."""
    y, score = np.asarray(y), np.asarray(score)
    k = int(np.ceil(budget * len(y)))
    if y.sum() == 0:
        return float("nan")
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def summarise(y: np.ndarray, score: np.ndarray, budgets=(0.1, 0.2, 0.3)) -> dict:
    out = {"roc_auc": roc_auc_score(y, score), "avg_precision": average_precision_score(y, score),
           "base_rate": float(np.mean(y))}
    for b in budgets:
        cap = capture_at_budget(y, score, b)
        out[f"capture@{int(b * 100)}"] = cap
        out[f"lift@{int(b * 100)}"] = cap / b
    return out


def block_bootstrap(y: np.ndarray, score: np.ndarray, blocks: np.ndarray, metric: Callable,
                    n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % interval resampling whole days, so autocorrelated lots stay together."""
    rng = np.random.default_rng(seed)
    ids = np.unique(blocks)
    idx_by_block = {b: np.where(blocks == b)[0] for b in ids}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([idx_by_block[b] for b in pick])
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        vals.append(metric(y[idx], score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# --- protocols ---------------------------------------------------------------------
@dataclass
class OutOfTime:
    score: np.ndarray      # within-block percentile rank of the predicted risk (NaN before first test block)
    raw: np.ndarray        # predicted probability
    fold: np.ndarray       # fold id (-1 = never tested)
    models: list           # fitted pipeline of each fold

    @property
    def tested(self) -> np.ndarray:
        return self.fold >= 0


def out_of_time(make: Callable, X: pd.DataFrame, y: np.ndarray, folds) -> OutOfTime:
    """Fit on the past, score the next block.

    `score` ranks lots within each block: the inspection list is drawn up at the
    end of a block, as in a batch release. `raw` keeps the probabilities, usable
    lot by lot without waiting for the block to end."""
    n = len(y)
    score, raw, fold = np.full(n, np.nan), np.full(n, np.nan), np.full(n, -1)
    models = []
    for f, (tr, te) in enumerate(folds):
        m = make().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        raw[te] = p
        score[te] = pd.Series(p).rank(pct=True, method="average").values
        fold[te] = f
        models.append(m)
    return OutOfTime(score=score, raw=raw, fold=fold, models=models)


def random_cv_auc(make: Callable, X: pd.DataFrame, y: np.ndarray, leaky_columns: list | None = None,
                  n_splits: int = 5, n_repeats: int = 3, score_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-fold AUCs of shuffled stratified CV, the protocol used by most SECOM write-ups.

    With ``leaky_columns`` the model only sees sensors chosen on the full data set
    beforehand (selection leakage). With ``score_mask`` each fold is scored only on
    its test lots inside the mask (e.g. the same calendar period as another protocol)."""
    aucs = []
    Xs = X[leaky_columns] if leaky_columns is not None else X
    for rep in range(n_repeats):
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=rep).split(Xs, y):
            m = make().fit(Xs.iloc[tr], y[tr])
            if score_mask is not None:
                te = te[score_mask[te]]
                if y[te].sum() == 0:
                    continue
            aucs.append(roc_auc_score(y[te], m.predict_proba(Xs.iloc[te])[:, 1]))
    return np.asarray(aucs)
