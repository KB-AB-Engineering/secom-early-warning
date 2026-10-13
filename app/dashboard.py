"""Quality early-warning dashboard.

    streamlit run app/dashboard.py

Reads the data set and the files written by experiments/run_all.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from secomqc import load_secom, spc  # noqa: E402
from secomqc.cost import best_budget  # noqa: E402
from secomqc.explain import fail_rate_by_quantile  # noqa: E402

RES = ROOT / "results"
BLUE, ORANGE, INK2, MUTED = "#2a78d6", "#eb6834", "#52514e", "#8a8984"

st.set_page_config(page_title="SECOM early warning", layout="wide")


@st.cache_data
def data():
    d = load_secom()
    return d.X, d.y, d.time


@st.cache_data
def results():
    read = lambda f: json.loads((RES / f).read_text())  # noqa: E731
    return {
        "audit": read("audit.json"), "cp": read("change_point_summary.json"), "spc": read("spc_params.json"),
        "key": pd.read_csv(RES / "key_parameters.csv"), "bench": pd.read_csv(RES / "benchmark.csv"),
        "pred": pd.read_csv(RES / "oot_predictions.csv", parse_dates=["time"]),
        "curves": pd.read_csv(RES / "capture_curves.csv"), "policy": pd.read_csv(RES / "cost_policy.csv"),
    }


@st.cache_data
def calibrated(kind: str, arl0: int) -> float:
    return spc.calibrate(kind, arl0, n_rep=2000)


X, y, t = data()
R = results()
page = st.sidebar.radio("View", ["Overview", "SPC monitor", "Lot triage", "Key parameters", "Data audit"])
st.sidebar.caption("SECOM: 1,567 production lots, 590 process sensors, July to October 2008.")


def kpi(col, label, value, help_text=None):
    col.metric(label, value, help=help_text)


if page == "Overview":
    st.title("Quality early warning on a real production line")
    cp = R["cp"]
    c = st.columns(4)
    kpi(c[0], "Lots", f"{len(y):,}")
    kpi(c[1], "Fail rate before 21 Aug", f"{cp['fail_rate_before']:.1%}")
    kpi(c[2], "Fail rate after", f"{cp['fail_rate_after']:.1%}")
    rf = R["bench"].set_index("model").loc["random_forest"]
    kpi(c[3], "Failures caught at 20% inspection", f"{rf['capture@20']:.0%}",
        "Out-of-time, Sept 1 to Oct 17, random forest")
    df = pd.DataFrame({"time": t, "fail": y}).set_index("time").resample("D")["fail"].agg(["mean", "size"]).reset_index()
    df = df[df["size"] >= 5]
    chart = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("time:T", title=None), y=alt.Y("mean:Q", title="failed lots per day", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("time:T", title="day"), alt.Tooltip("mean:Q", format=".1%", title="fail rate"),
                 alt.Tooltip("size:Q", title="lots")])
    rule = alt.Chart(pd.DataFrame({"time": [pd.Timestamp(cp["fail_change_time"])]})).mark_rule(
        color=INK2, strokeDash=[4, 4]).encode(x="time:T")
    st.altair_chart((chart + rule).properties(height=260), width="stretch")
    st.markdown(
        f"The fail rate dropped from **{cp['fail_rate_before']:.1%}** to **{cp['fail_rate_after']:.1%}** "
        f"around **{pd.Timestamp(cp['fail_change_time']).day} {pd.Timestamp(cp['fail_change_time']):%B}** (95% interval "
        f"{pd.Timestamp(cp['fail_change_interval'][0]).day} to {pd.Timestamp(cp['fail_change_interval'][1]).day} "
        f"{pd.Timestamp(cp['fail_change_interval'][1]):%B}). "
        "Several sensors, including the key parameters s59 and s103, step in the same window; "
        "the timing alone cannot say which change caused the other. Use **SPC monitor** to look at them.")

elif page == "SPC monitor":
    st.title("SPC monitor")
    key = R["spc"]["key_sensors"]
    others = [c for c in X.columns if c not in key and X[c].nunique() > 2]
    c1, c2, c3, c4 = st.columns(4)
    sensor = c1.selectbox("Sensor", key + others)
    kind = c2.selectbox("Chart", ["ewma", "cusum", "shewhart"], format_func=str.upper)
    sigma_method = c3.selectbox("Scaling", ["winsorised", "robust", "std", "moving_range"],
                                help="winsorised = clip at 3 robust sigma, then rescale (best compromise on heavy-tailed "
                                     "sensors). robust = 1.4826 x MAD. std inflates on heavy tails.")
    arl0 = c4.select_slider("Target lots between false alarms", [100, 200, 370, 500, 1000], value=370)
    dates = t.dt.date
    p1 = st.slider("Phase I (reference period)", min_value=dates.min(), max_value=dates.max(),
                   value=(pd.Timestamp(R["spc"]["phase1"][0]).date(), pd.Timestamp(R["spc"]["phase1"][1]).date()))
    param = calibrated(kind, arl0)
    x = X[sensor].values
    ok = ~np.isnan(x)
    in1 = ok & (dates >= p1[0]).values & (dates <= p1[1]).values & (y == 0)
    if in1.sum() < 20:
        st.warning("Phase I needs at least 20 good lots with a value.")
        st.stop()
    after = ok & (dates > p1[1]).values
    if sigma_method in ("winsorised", "robust") and spc.phase1_params(x[in1], "robust")[1] == 0:
        st.warning("This sensor's MAD is zero in Phase I (more than half the values are identical). "
                   "Choose 'std' or 'moving_range' scaling, or another sensor.")
        st.stop()
    if sigma_method == "winsorised":
        z = spc.WinsorisedScale(3.0).fit(x[in1]).transform(x[after])
    else:
        mu, sd = spc.phase1_params(x[in1], sigma_method)
        if sd == 0:
            st.warning("Phase I sigma is zero for this sensor; choose another scaling or sensor.")
            st.stop()
        z = (x[after] - mu) / sd
    res = spc.run_chart(kind, z, param, reset=True)
    df = pd.DataFrame({"time": t[after].values, "stat": res.statistic, "upper": res.upper, "lower": res.lower,
                       "alarm": res.alarm, "failed": y[after] == 1})
    yy = y[after]
    c = st.columns(5)
    kpi(c[0], "Lots monitored", f"{len(df):,}")
    kpi(c[1], "Signals", f"{int(res.alarm.sum())}", "The chart restarts after each signal, as after an investigation.")
    kpi(c[2], "Lots per signal", f"{len(df) / res.alarm.sum():.0f}" if res.alarm.any() else "none")
    kpi(c[3], "Fail rate at signals", f"{yy[res.alarm].mean():.1%}" if res.alarm.any() else "n/a")
    kpi(c[4], "Fail rate otherwise", f"{yy[~res.alarm].mean():.1%}" if (~res.alarm).any() else "n/a")
    base = alt.Chart(df).encode(x=alt.X("time:T", title=None, axis=alt.Axis(format="%d %b")))
    line = base.mark_line(color=BLUE, strokeWidth=1.5).encode(y=alt.Y("stat:Q", title=f"{kind.upper()} statistic (σ units)"))
    limits = base.mark_line(color=INK2, strokeDash=[4, 3], strokeWidth=1).encode(y="upper:Q") + \
        base.mark_line(color=INK2, strokeDash=[4, 3], strokeWidth=1).encode(y="lower:Q")
    alarms = base.transform_filter("datum.alarm").mark_circle(color=ORANGE, size=40).encode(
        y="stat:Q", tooltip=[alt.Tooltip("time:T"), alt.Tooltip("stat:Q", format=".2f"), "failed:N"])
    df["fail_mark"] = float(np.nanmin(np.r_[res.statistic, res.lower])) - 0.15 * float(np.nanmax(res.upper))
    base = alt.Chart(df).encode(x=alt.X("time:T", title=None, axis=alt.Axis(format="%d %b")))
    fails = base.transform_filter("datum.failed").mark_tick(color=INK2, thickness=1.5, size=10).encode(
        y="fail_mark:Q", tooltip=[alt.Tooltip("time:T", title="failed lot")])
    st.altair_chart((limits + line + alarms + fails).properties(height=330).interactive(), width="stretch")
    st.caption(f"Parameter {param:.3f} calibrated by simulation for {arl0} lots between false alarms on normal data. "
               "Orange dots: signals (chart restarted after each). Grey ticks at the bottom: failed lots. "
               "Phase I uses good lots only.")

elif page == "Lot triage":
    st.title("Lot triage: inspect the riskiest lots first")
    pred = R["pred"]
    pred = pred[pred.fold >= 0]
    curves = R["curves"]
    c1, c2 = st.columns(2)
    budget = c1.slider("Inspection budget (share of lots)", 0.05, 0.6, 0.2, 0.01)
    ratio = c2.slider("Cost of an escaped failure / cost of an inspection", 2, 100, 20)
    cap = float(np.interp(budget, curves.budget, curves.random_forest))
    fr = pred.fail.mean()
    b_best, _ = best_budget(curves.budget.values, curves.random_forest.values, fr, ratio)
    c = st.columns(4)
    kpi(c[0], "Failures caught", f"{cap:.0%}", "Out-of-time, budget applied within each test block")
    kpi(c[1], "Lift vs random", f"{cap / budget:.2f}x")
    kpi(c[2], "Best budget in hindsight", f"{b_best:.0%}",
        "Chosen on this same period, so optimistic. See results/cost_policy.csv for the honest version.")
    pol = R["policy"]
    row = pol.iloc[(pol.escape_to_inspect_ratio - ratio).abs().argmin()]
    kpi(c[3], f"Honest cost vs simpler rule ({int(row.escape_to_inspect_ratio)}x)",
        f"{-row.saving_vs_best_simple_policy:+.0%}",
        "Budget chosen on past blocks and applied to the next one, compared with inspecting nothing or everything "
        "(nearest computed cost ratio). Positive = the model costs more.")
    names = {"random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
             "logistic": "Logistic regression", "no_skill": "Random ranking"}
    long = curves.melt("budget", var_name="model", value_name="capture")
    long["model"] = long["model"].map(names)
    ch = alt.Chart(long).mark_line().encode(
        x=alt.X("budget:Q", axis=alt.Axis(format="%"), title="lots inspected"),
        y=alt.Y("capture:Q", axis=alt.Axis(format="%"), title="failures caught"),
        color=alt.Color("model:N", scale=alt.Scale(domain=list(names.values()),
                                                   range=[BLUE, ORANGE, "#1baf7a", MUTED])),
        tooltip=["model:N", alt.Tooltip("budget:Q", format=".0%"), alt.Tooltip("capture:Q", format=".0%")])
    st.altair_chart((ch + alt.Chart(pd.DataFrame({"b": [budget]})).mark_rule(color=INK2, strokeDash=[4, 4]).encode(x="b:Q"))
                    .properties(height=320), width="stretch")
    last = pred[pred.fold == pred.fold.max()].sort_values("risk_random_forest", ascending=False)
    k = int(np.ceil(budget * len(last)))
    st.subheader(f"Last block: the {k} lots to inspect")
    st.dataframe(last.head(k)[["time", "risk_random_forest", "fail"]].rename(
        columns={"risk_random_forest": "risk percentile in block", "fail": "failed (known afterwards)"}),
        width="stretch", hide_index=True)

elif page == "Key parameters":
    st.title("Key process parameters")
    key = R["key"].copy()
    key["robust"] = key["robust"].map({True: "yes", False: "no"})
    st.dataframe(key.head(20), width="stretch", hide_index=True)
    sensor = st.selectbox("Fail rate by value of", key.sensor.head(20))
    q = fail_rate_by_quantile(X[sensor], y, q=5).reset_index()
    ch = alt.Chart(q).mark_bar(color=BLUE, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("bin:N", sort=None, title=f"{sensor} quintile"),
        y=alt.Y("fail_rate:Q", axis=alt.Axis(format="%"), title="fail rate"),
        tooltip=["bin:N", "lots:Q", "fails:Q", alt.Tooltip("fail_rate:Q", format=".1%")])
    st.altair_chart(ch.properties(height=280), width="stretch")
    st.caption("Whole period, in-sample: values mix the two yield regimes. 'robust' = kept by the in-fold screening "
               "in at least 4 of the 5 chronological folds. For the out-of-sample check see the README.")

else:
    st.title("Data audit")
    a = R["audit"]
    c = st.columns(4)
    kpi(c[0], "Timestamps corrected", f"{a['timestamps_swapped']} ({a['timestamps_swapped_share']:.0%})")
    kpi(c[1], "Constant sensors", a["constant_sensors"])
    kpi(c[2], "Sensors > 50% missing", a["sensors_over_50pct_missing"])
    kpi(c[3], "Missing values", f"{a['missing_share']:.1%}")
    st.markdown(f"The CSV mirror spans **{a['raw_span']}**; after undoing the day/month swap the data span "
                f"**{a['fixed_span']}** and the timestamps follow the file order exactly "
                f"({a['order_violations_raw']} order violations before, {a['order_violations_fixed']} after).")
    st.image(str(RES / "figures" / "fig1_timestamps.png"))
