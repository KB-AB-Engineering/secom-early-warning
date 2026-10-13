"""Which process parameters matter, and how much can we trust that list?"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def selection_stability(models: list) -> pd.DataFrame:
    """How often each sensor was kept by the in-fold screening, across folds."""
    counts = Counter(c for m in models for c in m.named_steps["screen"].columns_)
    df = pd.DataFrame({"sensor": list(counts), "folds_selected": list(counts.values())})
    df["share"] = df["folds_selected"] / len(models)
    return df.sort_values(["folds_selected", "sensor"], ascending=[False, True]).reset_index(drop=True)


def shap_importance(model, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Mean |SHAP| of each screened sensor for a fitted tree pipeline.

    Returns (importance table, SHAP matrix, transformed features)."""
    import shap

    pre = model[:-1]
    Xt = pre.transform(X)
    explainer = shap.TreeExplainer(model.named_steps["model"])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):          # older shap: one array per class
        sv = sv[1]
    elif sv.ndim == 3:                # newer shap: (n, features, classes)
        sv = sv[:, :, 1]
    imp = np.abs(sv).mean(axis=0)
    out = pd.DataFrame({"sensor": Xt.columns, "mean_abs_shap": imp})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True), sv, Xt


def fail_rate_by_quantile(x: pd.Series, y: np.ndarray, q: int = 5) -> pd.DataFrame:
    """Observed fail rate per quantile bin of a sensor (missing values in their own bin)."""
    x = pd.Series(np.asarray(x, dtype=float))
    bins = pd.qcut(x, q=q, duplicates="drop")
    labels = bins.astype(str).where(x.notna(), "missing")
    g = pd.DataFrame({"bin": labels, "y": y}).groupby("bin", sort=False)["y"]
    out = pd.DataFrame({"lots": g.size(), "fails": g.sum(), "fail_rate": g.mean()})
    order = [b for b in bins.cat.categories.astype(str) if b in out.index] + (["missing"] if "missing" in out.index else [])
    return out.loc[order]
