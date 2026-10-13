"""Figures for the README (reads results/, writes results/figures/)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from secomqc import load_secom  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_COLORS = {"random_forest": BLUE, "gradient_boosting": ORANGE, "logistic": AQUA, "no_skill": MUTED}
MODEL_NAMES = {"random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
               "logistic": "Logistic regression", "no_skill": "Random ranking"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
    "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_timestamps():
    ts = pd.read_csv(RES / "timestamps.csv", parse_dates=["raw_time", "fixed_time"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, col, title in [(axes[0], "raw_time", "As published in the CSV mirror"),
                           (axes[1], "fixed_time", "After undoing the day/month swap")]:
        sw = ts.swapped == 1
        ax.scatter(ts.row[~sw], ts[col][~sw], s=6, color=BLUE, label="unchanged", lw=0)
        ax.scatter(ts.row[sw], ts[col][sw], s=6, color=ORANGE, label="day/month swapped (37%)", lw=0)
        ax.set_title(title)
        ax.set_xlabel("row in file")
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0].legend(loc="lower right", markerscale=2.5, fontsize=8.5)
    fig.suptitle("582 of 1,567 timestamps are wrong in the popular SECOM CSV", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig1_timestamps.png")


def fig_yield_change():
    data = load_secom()
    cp = json.loads((RES / "change_point_summary.json").read_text())
    t_change = pd.Timestamp(cp["fail_change_time"])
    lo, hi = (pd.Timestamp(v) for v in cp["fail_change_interval"])
    tl = pd.DataFrame({"time": data.time, "fail": data.y, "s59": data.X["s59"]})
    daily = tl.set_index("time").resample("D").agg({"fail": ["mean", "size"]})
    daily.columns = ["rate", "lots"]
    daily = daily[daily.lots >= 5]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    for a_ in axes:
        a_.axvspan(lo, hi, color=GRID, alpha=0.9, lw=0, zorder=0)
    ax = axes[0]
    ax.bar(daily.index, daily.rate * 100, width=0.8, color=BLUE, zorder=2)
    ax.hlines([cp["fail_rate_before"] * 100], tl.time.min(), t_change, color=INK, lw=1.5, ls="--", zorder=3)
    ax.hlines([cp["fail_rate_after"] * 100], t_change, tl.time.max(), color=INK, lw=1.5, ls="--", zorder=3)
    box = dict(fc=SURFACE, ec="none", pad=1)
    ax.text(tl.time.min(), cp["fail_rate_before"] * 100 + 1.5, f"{cp['fail_rate_before']:.1%} failed", color=INK,
            fontsize=9, bbox=box, zorder=4)
    ax.text(hi + pd.Timedelta("1D"), cp["fail_rate_after"] * 100 + 1.5, f"{cp['fail_rate_after']:.1%} failed",
            color=INK, fontsize=9, bbox=box, zorder=4)
    ax.text(lo, 41, " 95% interval of the change", fontsize=8.5, color=INK2, va="top", bbox=box, zorder=4)
    ax.set_ylabel("failed lots per day (%)")
    ax.set_title(f"Yield stepped up around {t_change.day} {t_change:%B} (best estimate; 95% interval {lo.day} {lo:%b} to {hi.day} {hi:%b})")
    ax = axes[1]
    ok = tl.s59.notna()
    ax.scatter(tl.time[ok], tl.s59[ok], s=5, color=MUTED, lw=0, label="each lot", zorder=2)
    fail = ok & (tl.fail == 1)
    ax.scatter(tl.time[fail], tl.s59[fail], s=16, color=ORANGE, lw=0, label="failed lot", zorder=3)
    ax.plot(tl.time, tl.s59.rolling(51, center=True, min_periods=20).median(), color=BLUE,
            label="rolling median (51 lots)", zorder=4)
    q_lo, q_hi = np.nanpercentile(tl.s59, [1, 99])
    ax.set_ylim(q_lo - 2, q_hi + 2)
    ax.set_ylabel("sensor s59")
    ax.set_title("Key sensor s59 steps down in the same window (so do s103, s247 and others)")
    ax.legend(loc="upper right", ncol=3, markerscale=2)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    save(fig, "fig2_yield_change.png")


def fig_protocols():
    p = pd.read_csv(RES / "protocols.csv")
    labels = ["Shuffled CV, sensors\nchosen on all data (leaky)", "Shuffled CV, sensors\nchosen inside each fold",
              "Shuffled CV, scored on\nthe Sept-Oct lots only", "Chronological: train on\nthe past, test the next block"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    yy = np.arange(len(p))[::-1]
    colors = [MUTED, MUTED, MUTED, BLUE]
    ax.barh(yy, p.auc - 0.5, left=0.5, height=0.55, color=colors)
    ax.hlines(yy, p.fold_min, p.fold_max, color=INK, lw=1.2)
    ax.vlines(p.fold_min, yy - 0.12, yy + 0.12, color=INK, lw=1.2)
    ax.vlines(p.fold_max, yy - 0.12, yy + 0.12, color=INK, lw=1.2)
    for y0, a_, hi in zip(yy, p.auc, p.fold_max):
        ax.text(hi + 0.008, y0, f"{a_:.2f}", va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_yticks(yy, labels)
    ax.set_xlim(0.5, 0.96)
    ax.set_xlabel("ROC AUC, mean of the test folds (0.5 = random); bars show the range across folds")
    ax.set_title("Same random forest, four evaluation protocols")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig3_protocols.png")


def fig_capture():
    c = pd.read_csv(RES / "capture_curves.csv")
    b = pd.read_csv(RES / "benchmark.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--")
    for m in ["no_skill", "logistic", "gradient_boosting", "random_forest"]:
        ax.plot(c.budget * 100, c[m] * 100, color=MODEL_COLORS[m], lw=2 if m != "no_skill" else 1.5,
                label=f"{MODEL_NAMES[m]} (AUC {b.loc[m, 'roc_auc']:.2f})")
    ax.axvline(20, color=INK2, lw=1, ls=":")
    rf = b.loc["random_forest"]
    ax.annotate(f"inspect 20% of lots:\ncatch {rf['capture@20']:.0%} of failures\n(95% CI {rf['capture@20_lo']:.0%} to {rf['capture@20_hi']:.0%})",
                xy=(20, rf["capture@20"] * 100), xytext=(48, 6), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlabel("lots inspected, riskiest first (%)")
    ax.set_ylabel("failed lots caught (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Out-of-time triage, Sept 1 to Oct 17")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig4_capture.png")


def fig_key_parameters():
    k = pd.read_csv(RES / "key_parameters.csv").head(12)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    colors = [BLUE if r else MUTED for r in k.robust]
    ax.barh(k.sensor, k.mean_abs_shap, color=colors, height=0.65)
    for i, (v, f) in enumerate(zip(k.mean_abs_shap, k.folds_selected)):
        ax.text(v + 0.001, i, f"{int(f)}/5", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("mean |SHAP| on the training lots")
    ax.set_title("Key parameters (blue: kept in ≥ 4 of 5 folds)")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    oos = pd.read_csv(RES / "key_parameters_out_of_sample.csv").set_index("sensor")
    x = np.arange(5)
    for j, (sensor, color) in enumerate([("s103", BLUE), ("s59", ORANGE)]):
        r = oos.loc[sensor]
        rates = [r[f"fail_rate_q{k}"] * 100 for k in range(1, 6)]
        ax.bar(x + (j - 0.5) * 0.36, rates, width=0.34, color=color,
               label=f"{sensor} (above late-Aug 80th pct: {r.risk_ratio:.1f}x)")
    pred = pd.read_csv(RES / "oot_predictions.csv")
    base = pred.loc[pred.fold >= 0, "fail"].mean() * 100
    ax.axhline(base, color=INK2, lw=1, ls="--")
    ax.text(-0.45, base + 0.5, f"average {base:.1f}%", fontsize=8.5, color=INK2)
    ax.set_xticks(x, ["<20th\npct", "20-40th", "40-60th", "60-80th", ">80th\npct"])
    ax.set_xlabel("bins from late-August percentiles")
    ax.set_ylabel("failed lots, 1 Sept to 17 Oct (%)")
    ax.set_title("Out of sample: chosen on data up to 1 Sept, tested after")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig5_key_parameters.png")


def fig_spc():
    d = pd.read_csv(RES / "spc_delays.csv")
    g = d.groupby(["sigma", "chart", "shift_robust_sigma"])["detected_share"].mean().reset_index()
    series = [("normal", "ideal normal data", MUTED, "--"), ("std", "σ from std dev", BLUE, "-"),
              ("robust", "σ from MAD", ORANGE, "-"), ("winsorised (clip at 3 robust sigma)", "winsorised", AQUA, "-")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), sharey=True)
    for ax, chart, title in zip(axes, ["shewhart", "ewma", "cusum"], ["Shewhart (3σ)", "EWMA (λ = 0.2)", "CUSUM (k = 0.5)"]):
        for key, label, color, ls in series:
            v = g[(g.chart == chart) & (g.sigma == key)]
            if len(v):
                ax.plot(v.shift_robust_sigma, v.detected_share * 100, color=color, ls=ls, marker="o", ms=5, label=label)
        ax.set_title(title)
        ax.set_xlabel("step shift (robust σ); 0 = no shift")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("150-lot runs with a signal (%)")
    axes[1].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Heavy-tailed sensors (about 347 good lots each, resampled): winsorising restores the design; "
                 "at shift 0 every signal is false",
                 x=0.01, ha="left", fontweight="bold", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, "fig6_spc_detection.png")

    a = pd.read_csv(RES / "spc_alarm_rates.csv")
    a = a[a.chart == "ewma"].copy()
    a["lps"] = a.lots_per_signal.replace(np.inf, 2000).clip(upper=2000)
    rows = [("std", "σ from std dev", BLUE), ("robust", "σ from MAD", ORANGE),
            ("winsorised (clip at 3 robust sigma)", "winsorised", AQUA)]
    sensors = list(dict.fromkeys(a.sensor))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for j, (key, label, color) in enumerate(rows):
        v = a[a.sigma == key].set_index("sensor").reindex(sensors)
        ax.scatter(v.lps, np.arange(len(sensors)) + (j - 1) * 0.22, s=40, color=color, lw=0, label=label, zorder=3)
    ax.set_yticks(range(len(sensors)), sensors)
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000], ["20", "50", "100", "200", "500", "1000", "none"])
    ax.minorticks_off()
    ax.axvline(370, color=INK, lw=1.2, ls="--")
    ax.text(390, len(sensors) - 0.45, "design: 370", fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(sensors) - 0.2)
    ax.set_xlabel("good lots per EWMA signal, 3 Sept to 17 Oct (log scale; chart reset after each signal)")
    ax.set_title("After the change, several key sensors keep moving: frequent signals, rarely on failed lots", fontsize=10.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.36, 1.0), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig7_spc_signals.png")


def fig_cost():
    pol = pd.read_csv(RES / "cost_policy.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    r = pol.escape_to_inspect_ratio.astype(str)
    x = np.arange(len(pol))
    for j, (col, label, color) in enumerate([("cost_model", "inspect the riskiest lots (budget chosen on past blocks)", BLUE),
                                             ("cost_nothing", "inspect nothing", MUTED),
                                             ("cost_everything", "inspect everything", ORANGE)]):
        ax.bar(x + (j - 1) * 0.27, pol[col], width=0.25, color=color, label=label)
    ax.set_xticks(x, [f"{v}x" for v in r])
    ax.set_xlabel("cost of an escaped failure, in inspections")
    ax.set_ylabel("cost per lot (inspections)")
    ax.set_title("Applied honestly, the triage never beats the simpler of the two rules")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig8_cost.png")


if __name__ == "__main__":
    fig_timestamps()
    fig_yield_change()
    fig_protocols()
    fig_capture()
    fig_key_parameters()
    fig_spc()
    fig_cost()
