"""Turning a risk ranking into an inspection policy.

Each period, quality inspects the riskiest share b of lots. Inspecting a lot
costs c_inspect; a failed lot that is not inspected escapes downstream and costs
c_escape. The expected cost per lot is

    cost(b) = b * c_inspect + (1 - capture(b)) * fail_rate * c_escape

so the best budget depends only on the cost ratio c_escape / c_inspect.
"""

from __future__ import annotations

import numpy as np

from .evaluation import capture_at_budget


def capture_curve(y: np.ndarray, score: np.ndarray, budgets: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Capture at each budget. With `groups` the budget is applied inside each
    period and failures are pooled, as in practice."""
    y, score = np.asarray(y), np.asarray(score)
    if groups is None:
        return np.array([capture_at_budget(y, score, b) for b in budgets])
    caught = np.zeros(len(budgets))
    for g in np.unique(groups):
        m = groups == g
        k = np.ceil(budgets * m.sum()).astype(int)
        order = np.argsort(-score[m], kind="stable")
        cum = np.r_[0, np.cumsum(y[m][order])]
        caught += cum[np.minimum(k, m.sum())]
    return caught / y.sum()


def expected_cost(budgets: np.ndarray, capture: np.ndarray, fail_rate: float, ratio: float) -> np.ndarray:
    """Cost per lot in units of c_inspect, for c_escape = ratio * c_inspect."""
    return budgets + (1 - capture) * fail_rate * ratio


def best_budget(budgets: np.ndarray, capture: np.ndarray, fail_rate: float, ratio: float) -> tuple[float, float]:
    c = expected_cost(budgets, capture, fail_rate, ratio)
    i = int(np.argmin(c))
    return float(budgets[i]), float(c[i])
