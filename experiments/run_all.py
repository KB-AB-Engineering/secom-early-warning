"""Run every experiment and write the results used by the README and dashboard.

    python experiments/run_all.py          # about 4 minutes on a laptop
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, fisher_exact
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from secomqc import audit, load_secom  # noqa: E402
from secomqc import cost, evaluation as ev, explain, spc  # noqa: E402
from secomqc.data import fix_swapped_timestamps  # noqa: E402
from secomqc.models import make_model  # noqa: E402
from secomqc.preprocessing import SensorFilter, rank_sensors  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

PHASE1_SHARE = 0.4      # first 40 % of lots = reference period (also the first training window)
N_FOLDS = 5
BUDGETS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
CHARTS = ["shewhart", "ewma", "cusum"]
SIGMAS = ["std", "moving_range", "robust"]
SHIFTS = [0.0, 0.5, 1.0, 2.0, 3.0]  # in robust-sigma units (spread of the bulk of the data)
WINDOW = 150
N_KEY = 6               # sensors monitored with SPC


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)


def main() -> None:
    data = load_secom()
    X, y, t = data.X, data.y, data.time
    n = data.n
    day = t.dt.strftime("%Y-%m-%d").values
    folds = ev.chronological_folds(n, PHASE1_SHARE, N_FOLDS)

    # E1 ---------------------------------------------------------------- data audit
    log("E1 data audit")
    a = audit()
    a["fold_boundaries"] = [str(t.iloc[te[0]]) for _, te in folds]
    (OUT / "audit.json").write_text(json.dumps(a, indent=2))
    raw_t = pd.to_datetime(pd.read_csv(ROOT / "data" / "uci-secom.csv", usecols=["Time"])["Time"])
    fixed_t, swapped = fix_swapped_timestamps(raw_t)
    save(pd.DataFrame({"row": np.arange(len(raw_t)), "raw_time": raw_t, "fixed_time": fixed_t,
                       "swapped": swapped.astype(int)}), "timestamps.csv")

    # E2 ---------------------------------------------- evaluation protocols compared
    log("E2 protocols (leaky random CV, clean random CV, chronological)")
    oot = {m: ev.out_of_time(lambda m=m: make_model(m), X, y, folds)
           for m in ["random_forest", "gradient_boosting", "logistic", "no_skill"]}
    rf = oot["random_forest"]
    mask = rf.tested
    leaky_cols = list(rank_sensors(SensorFilter().fit(X).transform(X), y).index[:50])
    rf_make = lambda: make_model("random_forest")  # noqa: E731
    protocol_aucs = {
        "Random CV, sensors chosen on all data (leaky)": ev.random_cv_auc(rf_make, X, y, leaky_columns=leaky_cols),
        "Random CV, sensors chosen inside each fold": ev.random_cv_auc(rf_make, X, y),
        "Random CV, scored on the Sept-Oct lots only": ev.random_cv_auc(rf_make, X, y, score_mask=mask),
        "Chronological: train on the past, test the next block": np.array(
            [roc_auc_score(y[rf.fold == f], rf.raw[rf.fold == f]) for f in range(N_FOLDS)]),
    }
    rows = [{"protocol": k, "auc": v.mean(), "fold_min": v.min(), "fold_max": v.max(), "folds": len(v)}
            for k, v in protocol_aucs.items()]
    save(pd.DataFrame(rows), "protocols.csv")

    # E3 ---------------------------------------------------- out-of-time benchmark
    log("E3 out-of-time model benchmark")
    bench, per_fold = [], []
    for name, res in oot.items():
        s = summarise_with_ci(y[mask], res.score[mask], day[mask])
        s["model"] = name
        s["roc_auc_mean_of_folds"] = float(np.mean([roc_auc_score(y[res.fold == f], res.raw[res.fold == f])
                                                    for f in range(N_FOLDS)]))
        s["roc_auc_raw_pooled"] = roc_auc_score(y[mask], res.raw[mask])
        s["capture@20_raw_pooled"] = ev.capture_at_budget(y[mask], res.raw[mask], 0.2)
        bench.append(s)
        for f in range(N_FOLDS):
            m = res.fold == f
            per_fold.append({"model": name, "fold": f, "start": str(t[m].iloc[0]), "lots": int(m.sum()),
                             "fails": int(y[m].sum()), "roc_auc": roc_auc_score(y[m], res.score[m]),
                             "capture@20": ev.capture_at_budget(y[m], res.score[m], 0.2)})
    save(pd.DataFrame(bench), "benchmark.csv")
    save(pd.DataFrame(per_fold), "benchmark_per_fold.csv")
    pred = pd.DataFrame({"time": t, "fail": y, "fold": rf.fold})
    for name, res in oot.items():
        pred[f"risk_{name}"] = res.score
    save(pred, "oot_predictions.csv")

    # E3b -------------------------------------------------------- inspection policy
    log("E3b inspection budget and cost")
    curves = {"budget": BUDGETS}
    for name, res in oot.items():
        curves[name] = cost.capture_curve(y[mask], res.score[mask], BUDGETS, groups=res.fold[mask])
    curves = pd.DataFrame(curves)
    save(curves, "capture_curves.csv")
    # Honest version: choose the budget on the blocks already seen, apply it to the next one.
    pol = []
    for ratio in [5, 10, 20, 50, 100]:
        spent = {"model": 0.0, "nothing": 0.0, "everything": 0.0}
        lots = 0
        for f in range(1, N_FOLDS):
            past = (rf.fold >= 0) & (rf.fold < f)
            cur = rf.fold == f
            cap_past = cost.capture_curve(y[past], rf.score[past], BUDGETS, groups=rf.fold[past])
            b, _ = cost.best_budget(BUDGETS, cap_past, float(y[past].mean()), ratio)
            k = int(np.ceil(b * cur.sum()))
            order = np.argsort(-rf.score[cur], kind="stable")
            missed = y[cur].sum() - y[cur][order[:k]].sum()
            spent["model"] += k + missed * ratio
            spent["nothing"] += y[cur].sum() * ratio
            spent["everything"] += cur.sum()
            lots += cur.sum()
        pol.append({"escape_to_inspect_ratio": ratio,
                    **{f"cost_{key}": v / lots for key, v in spent.items()},
                    "saving_vs_best_simple_policy": 1 - spent["model"] / min(spent["nothing"], spent["everything"])})
    save(pd.DataFrame(pol), "cost_policy.csv")

    # E5 (first: E4 needs the regime boundary) ------------- retrospective change points
    log("E5 change points: when did the yield and the process change?")
    i_fail, _ = spc.best_change_point(y.astype(float))
    lo, hi, _ = spc.change_point_interval(y.astype(float), day, n_boot=500)
    cp = {"fail_change_time": str(t[i_fail]), "fail_change_interval": [str(t[lo]), str(t[hi])],
          "fail_rate_before": float(y[:i_fail].mean()), "fail_rate_after": float(y[i_fail:].mean()),
          "lots_before": int(i_fail), "p_value": spc.change_point_pvalue(y.astype(float), day, n_perm=1000),
          "p_value_note": "1/(1000+1) is the smallest value 1000 permutations can give"}
    rows = []
    kept = SensorFilter().fit(X).columns_
    for s_ in kept:
        ok = ~np.isnan(X[s_].values)
        x = X[s_].values[ok]
        i, r2 = spc.best_change_point(x)
        sd = np.std(x[:i], ddof=1)
        rows.append({"sensor": s_, "change_time": t[ok].iloc[i], "explained_share": r2,
                     "shift_sigma": (x[i:].mean() - x[:i].mean()) / sd if sd > 0 else np.nan,
                     "hours_from_yield_change": (t[ok].iloc[i] - t[i_fail]).total_seconds() / 3600})
    cps = pd.DataFrame(rows).sort_values("explained_share", ascending=False).reset_index(drop=True)
    strong = cps[cps.explained_share >= 0.05]
    near = strong[strong.hours_from_yield_change.abs() <= 72]
    # chance level: how many strong shifts fall in a random 6-day window?
    times = strong.change_time.values
    starts = pd.date_range(t.min(), t.max() - pd.Timedelta("6D"), freq="6h")
    counts = np.array([((times >= s0) & (times < s0 + pd.Timedelta("6D"))).sum() for s0 in starts])
    cp.update({"sensors_tested": len(cps), "sensors_with_strong_shift": int(len(strong)),
               "strong_shifts_within_3_days": near.sensor.tolist(),
               "strong_shifts_in_random_6_day_window_median": float(np.median(counts)),
               "share_of_windows_with_as_many": float((counts >= len(near)).mean())})
    save(cps, "change_points.csv")

    # E4 ------------------------------------------------------------ key parameters
    log("E4 key parameters (stability across folds + SHAP)")
    stab = explain.selection_stability(rf.models)
    final = rf.models[-1]                     # trained on the first 88 % of lots
    tr_last = folds[-1][0]
    imp, _, _ = explain.shap_importance(final, X.iloc[tr_last])
    key = imp.merge(stab, on="sensor", how="left").fillna({"folds_selected": 0, "share": 0})
    key["robust"] = key["folds_selected"] >= N_FOLDS - 1
    save(key, "key_parameters.csv")
    top = key[key.robust].head(N_KEY)["sensor"].tolist()
    if len(top) < N_KEY:
        top += [s_ for s_ in key.sensor if s_ not in top][: N_KEY - len(top)]
    # Within each regime, so the gradient is not just "high values = old regime".
    bins = []
    regimes = {"before the yield change": np.arange(i_fail), "after the yield change": np.arange(i_fail, n)}
    for s_ in top[:3]:
        for regime, idx in regimes.items():
            v = X[s_].iloc[idx]
            qs = np.unique(np.nanquantile(v, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
            qs[0], qs[-1] = -np.inf, np.inf
            cut = pd.cut(v, qs, labels=False)
            g = pd.DataFrame({"q": cut.values, "y": y[idx]}).dropna().groupby("q")["y"]
            for q, grp in g:
                bins.append({"sensor": s_, "regime": regime, "quintile": int(q) + 1, "lots": len(grp),
                             "fails": int(grp.sum()), "fail_rate": grp.mean()})
    save(pd.DataFrame(bins), "risk_bins.csv")

    # E4b out-of-sample check: sensors flagged by the first model (data up to 1 Sept),
    # thresholds from the post-change training lots, tested on the 6 following weeks.
    first_pick = list(rf.models[0].named_steps["screen"].columns_)   # ranked by Mann-Whitney p
    tr0, te0 = np.arange(i_fail, folds[0][0][-1] + 1), folds[0][1][0]
    test_idx = np.arange(te0, n)
    oos = []
    for rank, s_ in enumerate(first_pick, start=1):
        edges = np.nanquantile(X[s_].values[tr0], [0.2, 0.4, 0.6, 0.8])
        v = X[s_].values[test_idx]
        ok = ~np.isnan(v)
        q = np.digitize(v[ok], edges)
        yy = y[test_idx][ok]
        above = q == 4
        table = [[int(yy[above].sum()), int(above.sum() - yy[above].sum())],
                 [int(yy[~above].sum()), int((~above).sum() - yy[~above].sum())]]
        p_ = fisher_exact(table)[1]
        row = {"sensor": s_, "rank_in_first_model": rank, "key_parameter": s_ in top,
               "above_p80_lots": int(above.sum()), "above_p80_fail_rate": float(yy[above].mean()),
               "below_p80_lots": int((~above).sum()), "below_p80_fail_rate": float(yy[~above].mean()),
               "risk_ratio": float(yy[above].mean() / yy[~above].mean()), "fisher_p": float(p_),
               "bonferroni_p": float(min(1.0, p_ * len(first_pick)))}
        for k_ in range(5):
            row[f"fail_rate_q{k_ + 1}"] = float(yy[q == k_].mean()) if (q == k_).any() else np.nan
            row[f"lots_q{k_ + 1}"] = int((q == k_).sum())
        oos.append(row)
    save(pd.DataFrame(oos).sort_values("fisher_p"), "key_parameters_out_of_sample.csv")

    # change-point intervals of the key sensors and of the sensors near the yield change
    detail = []
    for s_ in dict.fromkeys(top + near.sensor.tolist()):
        ok = ~np.isnan(X[s_].values)
        x = X[s_].values[ok]
        i, _ = spc.best_change_point(x)
        lo_, hi_, _ = spc.change_point_interval(x, day[ok], n_boot=300)
        detail.append({"sensor": s_, "key_parameter": s_ in top, "change_time": str(t[ok].iloc[i]),
                       "interval_start": str(t[ok].iloc[lo_]), "interval_end": str(t[ok].iloc[hi_]),
                       "p_value": spc.change_point_pvalue(x, day[ok], n_perm=300)})
    save(pd.DataFrame(detail), "change_points_detail.csv")
    (OUT / "change_point_summary.json").write_text(json.dumps(cp, indent=2))

    # E6 ------------------------------------- SPC in the new, stable regime (after the change)
    log("E6 SPC: calibration, false alarms on real data, detection on real noise")
    params = {"shewhart": spc.calibrate("shewhart"), "ewma": spc.calibrate("ewma"), "cusum": spc.calibrate("cusum")}
    post = np.arange(i_fail, n)
    q1 = post[: int(0.3 * len(post))]          # Phase I: first 30 % of the new regime
    q2 = post[int(0.3 * len(post)):]           # Phase II: monitored
    (OUT / "spc_params.json").write_text(json.dumps({
        **params, "lambda": 0.2, "k": 0.5, "target_arl0": 370, "key_sensors": top,
        "phase1": [str(t[q1[0]]), str(t[q1[-1]])], "phase2": [str(t[q2[0]]), str(t[q2[-1]])]}, indent=2))
    alarms, noise, delays = [], [], []
    for s_ in top:
        v1, v2 = X[s_].values[q1], X[s_].values[q2]
        x1 = v1[(y[q1] == 0) & ~np.isnan(v1)]
        ok2 = ~np.isnan(v2)
        yy = y[q2][ok2]
        med = np.median(x1)
        mad_sigma = spc.phase1_params(x1, "robust")[1]
        noise.append({"sensor": s_, "lag1_autocorrelation": float(np.corrcoef(x1[:-1], x1[1:])[0, 1]),
                      "skewness": float(pd.Series(x1).skew()), "excess_kurtosis": float(pd.Series(x1).kurt()),
                      "std_over_robust_sigma": float(np.std(x1, ddof=1) / mad_sigma),
                      "phase2_median_shift_robust_sigma": float((np.median(v2[ok2]) - med) / mad_sigma)})
        for method in SIGMAS:
            mu, sd = spc.phase1_params(x1, method)
            # (a) the real Phase II, charts reset after each signal
            z2 = (v2[ok2] - mu) / sd
            for kind in CHARTS:
                res = spc.run_chart(kind, z2, params[kind], reset=True)
                n_sig = int(res.alarm.sum())
                good = int((yy == 0).sum())
                alarms.append({"sensor": s_, "sigma": method, "chart": kind, "signals": n_sig,
                               "good_lots": good, "lots_per_signal": good / n_sig if n_sig else np.inf,
                               "fail_rate_at_signal": float(yy[res.alarm].mean()) if n_sig else np.nan,
                               "fail_rate_overall": float(yy.mean())})
            # (b) heavy tails only: lots resampled independently from the Phase I good
            # lots (no drift, no ordering), shifts in robust-sigma units
            z1 = (x1 - mu) / sd
            for kind in CHARTS:
                for shift in SHIFTS:
                    rl = spc.run_lengths(kind, params[kind], shift * mad_sigma / sd, 4000, WINDOW,
                                          np.random.default_rng(1), noise=z1)
                    delays.append({"sensor": s_, "sigma": method, "chart": kind, "shift_robust_sigma": shift,
                                   "source": "resampled real noise", "phase1_lots": len(x1),
                                   "median_run_length": float(np.median(rl)),
                                   "detected_share": float((rl <= WINDOW).mean())})
        # (c) remedy for heavy tails: winsorised standardisation, normal-theory limits
        ws = spc.WinsorisedScale(3.0).fit(x1)
        r1 = (x1 - ws.median_) / ws.sigma_
        clip = lambda v, ws=ws: (np.clip(v, -ws.c, ws.c) - ws.mean_) / ws.std_  # noqa: E731
        for kind in ["ewma", "cusum"]:
            for shift in SHIFTS:
                rl = spc.run_lengths(kind, params[kind], shift, 4000, WINDOW, np.random.default_rng(1),
                                      noise=r1, transform=clip)
                delays.append({"sensor": s_, "sigma": "winsorised (clip at 3 robust sigma)", "chart": kind,
                               "shift_robust_sigma": shift, "source": "resampled real noise", "phase1_lots": len(x1),
                               "median_run_length": float(np.median(rl)), "detected_share": float((rl <= WINDOW).mean())})
            res = spc.run_chart(kind, ws.transform(v2[ok2]), params[kind], reset=True)
            n_sig = int(res.alarm.sum())
            good = int((yy == 0).sum())
            alarms.append({"sensor": s_, "sigma": "winsorised (clip at 3 robust sigma)", "chart": kind,
                           "signals": n_sig, "good_lots": good, "lots_per_signal": good / n_sig if n_sig else np.inf,
                           "fail_rate_at_signal": float(yy[res.alarm].mean()) if n_sig else np.nan,
                           "fail_rate_overall": float(yy.mean())})
    for kind in CHARTS:
        for shift in SHIFTS:
            rl = spc.run_lengths(kind, params[kind], shift, 4000, WINDOW, np.random.default_rng(1))
            delays.append({"sensor": "theory", "sigma": "normal", "chart": kind, "shift_robust_sigma": shift,
                           "source": "normal theory", "median_run_length": float(np.median(rl)),
                           "detected_share": float((rl <= WINDOW).mean())})
    save(pd.DataFrame(alarms), "spc_alarm_rates.csv")
    save(pd.DataFrame(noise), "spc_noise.csv")
    save(pd.DataFrame(delays), "spc_delays.csv")

    # E7 ------------------------------------------------ multivariate Hotelling T²
    log("E7 Hotelling T² on the key sensors")
    med = X[top].iloc[q1].median()
    Z = X[top].fillna(med)
    t2 = spc.HotellingT2().fit(Z.iloc[q1][y[q1] == 0].values)
    stat = t2.statistic(Z.values)
    alarm = stat > t2.ucl_
    phase = np.full(n, "before change", dtype=object)
    phase[q1], phase[q2] = "I", "II"
    save(pd.DataFrame({"time": t, "fail": y, "t2": stat, "ucl": t2.ucl_, "phase": phase}), "t2.csv")
    in2 = phase == "II"
    good1 = Z.iloc[q1][y[q1] == 0].values
    (OUT / "t2_summary.json").write_text(json.dumps({
        "sensors": top, "ucl": t2.ucl_,
        "phase1_good_lots_above_ucl": float((t2.statistic(good1) > t2.ucl_).mean()),
        "design_false_alarm_rate": 0.0027,
        "alarm_rate_phase2": float(alarm[in2].mean()), "alarms_phase2": int(alarm[in2].sum()),
        "fail_rate_when_alarm_phase2": float(y[in2 & alarm].mean()) if (in2 & alarm).any() else None,
        "fail_rate_when_quiet_phase2": float(y[in2 & ~alarm].mean()),
        "p_at_least_this_many_failures_by_chance": float(binom.sf(int(y[in2 & alarm].sum()) - 1, int(alarm[in2].sum()),
                                                                   float(y[in2].mean()))),
    }, indent=2))
    log("done")


def summarise_with_ci(y, score, blocks) -> dict:
    s = ev.summarise(y, score)
    s["roc_auc_lo"], s["roc_auc_hi"] = ev.block_bootstrap(y, score, blocks, roc_auc_score)
    cap = lambda yy, ss: ev.capture_at_budget(yy, ss, 0.2)  # noqa: E731
    s["capture@20_lo"], s["capture@20_hi"] = ev.block_bootstrap(y, score, blocks, cap)
    return s


if __name__ == "__main__":
    main()
