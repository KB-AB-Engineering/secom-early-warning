"""Statistical process control: Shewhart, EWMA, CUSUM and Hotelling T².

All univariate charts work on a standardised series z = (x - mu) / sigma, where
mu and sigma come from a Phase I reference period. Chart parameters can be
calibrated by simulation to a common in-control average run length (ARL0), so
the three charts are compared at the same false-alarm rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

D2_N2 = 1.128  # E[|X1 - X2|] / sigma for normal data (moving range of 2)


# --- Phase I --------------------------------------------------------------------------
def phase1_params(x: np.ndarray, method: str = "moving_range") -> tuple[float, float]:
    """Centre line and sigma from a reference sample.

    moving_range: textbook individuals chart (MR-bar / d2), short-term sigma.
    std: overall standard deviation, long-term sigma.
    robust: median and 1.4826 * MAD, insensitive to outliers.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if method == "moving_range":
        return float(x.mean()), float(np.mean(np.abs(np.diff(x))) / D2_N2)
    if method == "std":
        return float(x.mean()), float(x.std(ddof=1))
    if method == "robust":
        med = float(np.median(x))
        return med, float(1.4826 * np.median(np.abs(x - med)))
    raise ValueError(method)


# --- charts ---------------------------------------------------------------------------
@dataclass
class ChartResult:
    statistic: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    alarm: np.ndarray

    def first_alarm(self) -> int | None:
        idx = np.flatnonzero(self.alarm)
        return int(idx[0]) if idx.size else None


def shewhart(z: np.ndarray, L: float = 3.0) -> ChartResult:
    z = np.asarray(z, dtype=float)
    up = np.full_like(z, L)
    return ChartResult(z, up, -up, np.abs(z) > L)


def ewma(z: np.ndarray, lam: float = 0.2, L: float = 2.962, reset: bool = False) -> ChartResult:
    """EWMA with exact (time-varying) limits, started at the target.

    With ``reset`` the chart restarts at the target after each signal, as an
    operator would after investigating; every alarm is then a separate signal."""
    z = np.asarray(z, dtype=float)
    stat, up, alarm = np.empty_like(z), np.empty_like(z), np.zeros(len(z), dtype=bool)
    prev, i = 0.0, 0
    for t, v in enumerate(z):
        i += 1
        prev = lam * v + (1 - lam) * prev
        width = L * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
        stat[t], up[t] = prev, width
        alarm[t] = abs(prev) > width
        if alarm[t] and reset:
            prev, i = 0.0, 0
    return ChartResult(stat, up, -up, alarm)


def cusum(z: np.ndarray, k: float = 0.5, h: float = 5.0, reset: bool = False) -> ChartResult:
    """Two-sided tabular CUSUM. `statistic` is the larger of C+ and C-.
    With ``reset`` both sums restart at zero after each signal."""
    z = np.asarray(z, dtype=float)
    cp = cm = 0.0
    stat = np.empty_like(z)
    alarm = np.zeros(len(z), dtype=bool)
    for t, v in enumerate(z):
        cp = max(0.0, cp + v - k)
        cm = max(0.0, cm - v - k)
        stat[t] = max(cp, cm)
        alarm[t] = stat[t] > h
        if alarm[t] and reset:
            cp = cm = 0.0
    up = np.full_like(z, h)
    return ChartResult(stat, up, np.zeros_like(z), alarm)


def count_signals(alarm: np.ndarray) -> int:
    """Number of alarm episodes (runs of consecutive alarm points)."""
    a = np.asarray(alarm, dtype=bool)
    return int(a[0] + np.sum(a[1:] & ~a[:-1])) if a.size else 0


# --- run lengths by simulation ------------------------------------------------------
def run_lengths(kind: str, param: float, shift: float, n_rep: int, max_len: int, rng,
                 lam: float = 0.2, k: float = 0.5, noise: np.ndarray | None = None,
                 transform=None) -> np.ndarray:
    """First-alarm times (1-based) of `n_rep` series with a mean shift. The noise
    is standard normal, or resampled with replacement from `noise` (a real
    sample). `transform`, if given, is applied after the shift (e.g. clipping).
    Censored series get max_len + 1."""
    rl = np.full(n_rep, max_len + 1)
    alive = np.ones(n_rep, dtype=bool)
    a = np.zeros(n_rep)
    b = np.zeros(n_rep)
    for t in range(1, max_len + 1):
        z = (rng.standard_normal(n_rep) if noise is None else rng.choice(noise, n_rep)) + shift
        if transform is not None:
            z = transform(z)
        if kind == "shewhart":
            hit = np.abs(z) > param
        elif kind == "ewma":
            a = lam * z + (1 - lam) * a
            width = param * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
            hit = np.abs(a) > width
        elif kind == "cusum":
            a = np.maximum(0.0, a + z - k)
            b = np.maximum(0.0, b - z - k)
            hit = (a > param) | (b > param)
        else:
            raise ValueError(kind)
        new = alive & hit
        rl[new] = t
        alive &= ~hit
        if not alive.any():
            break
    return rl


def simulated_arl(kind: str, param: float, shift: float = 0.0, n_rep: int = 4000,
                  max_len: int = 5000, seed: int = 0, **kw) -> float:
    rng = np.random.default_rng(seed)
    return float(run_lengths(kind, param, shift, n_rep, max_len, rng, **kw).mean())


def shewhart_arl(L: float, shift: float = 0.0) -> float:
    """Exact ARL of the individuals chart for normal data."""
    p = norm.sf(L - shift) + norm.cdf(-L - shift)
    return float(1 / p)


def calibrate(kind: str, target_arl0: float = 370.0, lo: float = 1.0, hi: float = 8.0,
              n_rep: int = 4000, seed: int = 0, iters: int = 18, noise: np.ndarray | None = None, **kw) -> float:
    """Find L (EWMA) or h (CUSUM) giving the target in-control ARL, by bisection
    on simulated run lengths with common random numbers.

    With `noise` (a standardised in-control sample) the run lengths are simulated
    by resampling that sample, so the limits fit the real, non-normal data."""
    if kind == "shewhart":
        if noise is not None:
            raise ValueError("a 1/370 tail quantile cannot be estimated from a few hundred lots")
        return float(norm.isf(1 / (2 * target_arl0)))
    for _ in range(iters):
        mid = (lo + hi) / 2
        rl = run_lengths(kind, mid, 0.0, n_rep, int(target_arl0 * 8), np.random.default_rng(seed), noise=noise, **kw)
        arl = float(rl.mean())
        if arl < target_arl0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_chart(kind: str, z: np.ndarray, param: float, lam: float = 0.2, k: float = 0.5,
              reset: bool = False) -> ChartResult:
    if kind == "shewhart":
        return shewhart(z, param)
    if kind == "ewma":
        return ewma(z, lam, param, reset=reset)
    if kind == "cusum":
        return cusum(z, k, param, reset=reset)
    raise ValueError(kind)


# --- multivariate --------------------------------------------------------------------
class HotellingT2:
    """Hotelling T² with the Phase I sample covariance.

    Variables are standardised with their Phase I mean and standard deviation
    first, so sensors on very different scales are handled alike. The chi²
    upper limit assumes multivariate normal data."""

    def __init__(self, alpha: float = 0.0027):
        self.alpha = alpha

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=1)
        self.scale_[self.scale_ == 0] = 1.0
        self.precision_ = np.linalg.inv(np.cov((X - self.mean_) / self.scale_, rowvar=False))
        self.ucl_ = float(chi2.ppf(1 - self.alpha, X.shape[1]))
        return self

    def statistic(self, X: np.ndarray) -> np.ndarray:
        D = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return np.einsum("ij,jk,ik->i", D, self.precision_, D)

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """Share of a point's T² attributable to each variable (MYT-style
        decomposition simplified to d_j * (Sigma^-1 d)_j)."""
        d = (np.asarray(x, dtype=float) - self.mean_) / self.scale_
        c = d * (self.precision_ @ d)
        return c / c.sum()


# --- retrospective change-point analysis ---------------------------------------------
def best_change_point(x: np.ndarray, min_seg: int = 30) -> tuple[int, float]:
    """Single mean-shift change point by least squares.

    Returns the index of the first point of the second segment and the share of
    the total sum of squares explained by the split."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    c = np.cumsum(x)
    c2 = np.cumsum(x ** 2)
    total = c2[-1] - c[-1] ** 2 / n
    i = np.arange(min_seg, n - min_seg + 1)
    left = c2[i - 1] - c[i - 1] ** 2 / i
    right = (c2[-1] - c2[i - 1]) - (c[-1] - c[i - 1]) ** 2 / (n - i)
    sse = left + right
    j = int(np.argmin(sse))
    return int(i[j]), float(1 - sse[j] / total) if total > 0 else 0.0


def change_point_pvalue(x: np.ndarray, blocks: np.ndarray, n_perm: int = 500, seed: int = 0,
                        min_seg: int = 30) -> float:
    """Permutation p-value of the best split, shuffling whole blocks (days) so the
    within-day autocorrelation is preserved under the null."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    _, obs = best_change_point(x, min_seg)
    ids = np.unique(blocks)
    groups = [x[blocks == b] for b in ids]
    hits = 0
    for _ in range(n_perm):
        perm = np.concatenate([groups[k] for k in rng.permutation(len(groups))])
        if best_change_point(perm, min_seg)[1] >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def change_point_interval(x: np.ndarray, blocks: np.ndarray, n_boot: int = 500, seed: int = 0,
                          min_seg: int = 30, level: float = 0.95) -> tuple[int, int, np.ndarray]:
    """Uncertainty of the change-point location by a day-block bootstrap within
    each segment: resample whole days (with replacement) before and after the
    estimated change separately, rebuild the series, re-estimate the change.
    Returns the interval of indices and all bootstrap locations."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    blocks = np.asarray(blocks)
    i, _ = best_change_point(x, min_seg)
    segs = []
    for part, b in ((x[:i], blocks[:i]), (x[i:], blocks[i:])):
        ids = list(dict.fromkeys(b))
        segs.append([part[b == d] for d in ids])
    locs = np.empty(n_boot, dtype=int)
    for r in range(n_boot):
        pieces = []
        for days in segs:
            pick = rng.integers(0, len(days), size=len(days))
            pieces.append(np.concatenate([days[j] for j in pick]))
        locs[r] = best_change_point(np.concatenate(pieces), min_seg)[0]
    a = (1 - level) / 2
    lo, hi = np.quantile(locs, [a, 1 - a])
    return int(lo), int(hi), locs


# --- heavy-tailed sensors --------------------------------------------------------------
class WinsorisedScale:
    """Robust standardisation for heavy-tailed sensors.

    Centre on the Phase I median, scale by the MAD-based sigma, clip at +/- c,
    then rescale so the clipped Phase I values have unit variance. A single wild
    value can then move an EWMA or CUSUM by at most c, while ordinary shifts of
    the bulk of the data pass through unchanged."""

    def __init__(self, c: float = 3.0):
        self.c = c

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        self.median_, self.sigma_ = phase1_params(x, "robust")
        zc = np.clip((x - self.median_) / self.sigma_, -self.c, self.c)
        self.mean_, self.std_ = float(zc.mean()), float(zc.std(ddof=1))
        return self

    def transform(self, x: np.ndarray, shift_robust_sigma: float = 0.0) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self.median_) / self.sigma_ + shift_robust_sigma
        return (np.clip(z, -self.c, self.c) - self.mean_) / self.std_
