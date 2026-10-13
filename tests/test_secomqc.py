"""Verification tests: data fix, leakage safety, SPC theory, metrics, findings."""

import numpy as np
import pandas as pd
import pytest

from secomqc import audit, load_secom, spc
from secomqc.cost import capture_curve, expected_cost
from secomqc.data import fix_swapped_timestamps
from secomqc.evaluation import capture_at_budget, chronological_folds
from secomqc.models import MODELS, make_model
from secomqc.preprocessing import MannWhitneyScreen, SensorFilter


@pytest.fixture(scope="module")
def secom():
    return load_secom()


# --- data ---------------------------------------------------------------------------
def test_day_month_swap_is_undone():
    raw = pd.Series(["2008-07-19 11:55:00", "2008-01-08 02:02:00", "2008-10-12 06:00:00", "2008-08-25 10:00:00"])
    fixed, changed = fix_swapped_timestamps(raw)
    assert list(fixed.dt.strftime("%Y-%m-%d")) == ["2008-07-19", "2008-08-01", "2008-12-10", "2008-08-25"]
    assert list(changed) == [False, True, True, False]


def test_audit_matches_uci_documentation():
    a = audit()
    assert (a["lots"], a["sensors"], a["failed_lots"]) == (1567, 590, 104)
    assert a["timestamps_swapped"] == 582
    assert a["order_violations_fixed"] == 0 and a["order_violations_raw"] > 0
    assert a["fixed_span"] == "2008-07-19 to 2008-10-17"


def test_loaded_data_is_chronological(secom):
    assert secom.time.is_monotonic_increasing
    assert secom.X.shape == (1567, 590) and secom.y.sum() == 104


# --- preprocessing and leakage ------------------------------------------------------
def test_sensor_filter_drops_bad_columns():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({"good": a, "copy": a * 2 + 1, "const": 1.0, "missing": np.where(rng.random(200) < 0.7, np.nan, 1.0),
                      "other": rng.normal(size=200)})
    f = SensorFilter().fit(X)
    assert f.columns_ == ["good", "other"]


def test_screen_finds_the_informative_sensor():
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.1).astype(int)
    X = pd.DataFrame(rng.normal(size=(600, 30)), columns=[f"s{i}" for i in range(30)])
    X["s7"] += 1.5 * y
    assert MannWhitneyScreen(k=1).fit(X, y).columns_ == ["s7"]


def test_pipeline_selects_sensors_from_training_lots_only(secom):
    X, y = secom.X, secom.y
    tr = np.arange(600)
    m = make_model("logistic", k=10).fit(X.iloc[tr], y[tr])
    trained_on_all = make_model("logistic", k=10).fit(X, y)
    expected = MannWhitneyScreen(k=10).fit(SensorFilter().fit(X.iloc[tr]).transform(X.iloc[tr]), y[tr]).columns_
    assert m.named_steps["screen"].columns_ == expected
    assert m.named_steps["screen"].columns_ != trained_on_all.named_steps["screen"].columns_


def test_chronological_folds_never_look_ahead():
    folds = chronological_folds(1000, 0.4, 5)
    assert folds[0][1][0] == 400 and folds[-1][1][-1] == 999
    for tr, te in folds:
        assert tr.max() < te.min()
    assert sum(len(te) for _, te in folds) == 600


@pytest.mark.parametrize("name", MODELS)
def test_every_model_trains_and_scores(secom, name):
    m = make_model(name).fit(secom.X.iloc[:400], secom.y[:400])
    p = m.predict_proba(secom.X.iloc[400:450])[:, 1]
    assert p.shape == (50,) and np.all((p >= 0) & (p <= 1))


# --- metrics ------------------------------------------------------------------------
def test_capture_at_budget():
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    perfect = y.astype(float)
    assert capture_at_budget(y, perfect, 0.2) == 1.0
    assert capture_at_budget(y, -perfect, 0.2) == 0.0
    assert capture_at_budget(y, perfect, 0.1) == 0.5


def test_grouped_capture_curve_and_cost():
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.1).astype(int)
    s = rng.random(300) + y
    b = np.array([0.0, 0.1, 0.5, 1.0])
    assert np.allclose(capture_curve(y, s, b), capture_curve(y, s, b, groups=np.zeros(300)))
    assert capture_curve(y, s, b)[-1] == 1.0
    assert expected_cost(np.array([0.0]), np.array([0.0]), 0.05, 20)[0] == pytest.approx(1.0)


# --- SPC theory ---------------------------------------------------------------------
def test_shewhart_arl_matches_theory():
    assert spc.shewhart_arl(3.0) == pytest.approx(370.4, abs=0.1)
    assert spc.simulated_arl("shewhart", 3.0, n_rep=3000) == pytest.approx(370, rel=0.08)


def test_ewma_and_cusum_match_published_arl():
    # Montgomery, Introduction to SPC: EWMA(0.2, 2.962) ~ 500; CUSUM(k=0.5, h=5) ~ 465
    assert spc.simulated_arl("ewma", 2.962, n_rep=3000, max_len=6000) == pytest.approx(500, rel=0.08)
    assert spc.simulated_arl("cusum", 5.0, n_rep=3000, max_len=6000) == pytest.approx(465, rel=0.08)
    assert spc.simulated_arl("cusum", 4.0, shift=1.0, n_rep=3000) == pytest.approx(8.38, rel=0.05)


def test_calibration_recovers_textbook_parameters():
    assert spc.calibrate("ewma", 370, n_rep=2000) == pytest.approx(2.86, abs=0.05)
    assert spc.calibrate("cusum", 370, n_rep=2000) == pytest.approx(4.77, abs=0.08)


def test_chart_statistics_by_hand():
    z = np.array([1.0, 1.0, -2.0])
    e = spc.ewma(z, lam=0.5, L=3)
    assert np.allclose(e.statistic, [0.5, 0.75, -0.625])
    assert e.upper[0] == pytest.approx(3 * np.sqrt(0.5 / 1.5 * (1 - 0.25)))
    c = spc.cusum(np.array([2.0, 2.0, -1.0, -3.0]), k=0.5, h=2.5)
    assert np.allclose(c.statistic, [1.5, 3.0, 1.5, 3.0])
    assert list(c.alarm) == [False, True, False, True]


def test_moving_range_sigma_is_unbiased_for_normal_data():
    x = np.random.default_rng(3).normal(10, 2, 20000)
    mu, sd = spc.phase1_params(x, "moving_range")
    assert mu == pytest.approx(10, abs=0.05) and sd == pytest.approx(2, rel=0.02)


def test_hotelling_false_alarm_rate():
    rng = np.random.default_rng(4)
    t2 = spc.HotellingT2(alpha=0.01).fit(rng.normal(size=(3000, 5)))
    rate = (t2.statistic(rng.normal(size=(20000, 5))) > t2.ucl_).mean()
    assert rate == pytest.approx(0.01, abs=0.004)
    assert t2.contributions(np.array([3.0, 0, 0, 0, 0])).sum() == pytest.approx(1.0)


def test_change_point_detection():
    rng = np.random.default_rng(5)
    x = np.r_[rng.normal(0, 1, 300), rng.normal(1.5, 1, 300)]
    i, share = spc.best_change_point(x)
    assert abs(i - 300) <= 10 and share > 0.3
    blocks = np.repeat(np.arange(60), 10)
    assert spc.change_point_pvalue(x, blocks, n_perm=200) < 0.01
    assert spc.change_point_pvalue(rng.normal(size=600), blocks, n_perm=200) > 0.05


# --- headline finding -----------------------------------------------------------------
def test_yield_change_and_s59_step_fall_in_the_same_window(secom):
    i, _ = spc.best_change_point(secom.y.astype(float))
    t_fail = secom.time[i]
    assert t_fail.strftime("%Y-%m-%d") == "2008-08-21"
    ok = ~secom.X["s59"].isna().values
    j, _ = spc.best_change_point(secom.X["s59"].values[ok])
    hours = abs((secom.time[ok].iloc[j] - t_fail).total_seconds()) / 3600
    assert hours < 24


# --- robustness --------------------------------------------------------------------------
def test_hotelling_handles_sensors_on_very_different_scales():
    rng = np.random.default_rng(6)
    scales = np.array([1e-3, 1.0, 1e3, 5e5, 0.02, 40.0])
    ref = rng.normal(size=(2000, 6)) * scales
    t2 = spc.HotellingT2(alpha=0.01).fit(ref)
    assert t2.statistic(ref).mean() == pytest.approx(6, rel=0.05)
    new = rng.normal(size=(20000, 6)) * scales
    assert (t2.statistic(new) > t2.ucl_).mean() == pytest.approx(0.01, abs=0.004)
    shifted = new.copy()
    shifted[:, 0] += 5 * scales[0]          # a 5-sigma shift on the smallest-scale sensor
    assert (t2.statistic(shifted) > t2.ucl_).mean() > 0.7   # chi2(6, nc=25) gives about 0.85


def test_reset_turns_one_excursion_into_separate_signals():
    z = np.r_[np.zeros(20), np.full(60, 2.0)]
    no_reset = spc.cusum(z, k=0.5, h=4.0)
    with_reset = spc.cusum(z, k=0.5, h=4.0, reset=True)
    assert spc.count_signals(no_reset.alarm) == 1 and no_reset.alarm.sum() > 50
    assert with_reset.alarm.sum() == spc.count_signals(with_reset.alarm) > 1
    e = spc.ewma(z, lam=0.2, L=2.86, reset=True)
    assert e.alarm.sum() == spc.count_signals(e.alarm) > 1


def test_change_point_interval_covers_the_truth():
    rng = np.random.default_rng(7)
    x = np.r_[rng.normal(0, 1, 400), rng.normal(0.8, 1, 400)]
    lo, hi, _ = spc.change_point_interval(x, np.repeat(np.arange(80), 10), n_boot=200)
    assert lo <= 400 <= hi and hi - lo < 120


def test_winsorised_scale_limits_the_effect_of_outliers():
    rng = np.random.default_rng(8)
    x = rng.standard_t(df=2, size=3000) * 5 + 10
    ws = spc.WinsorisedScale(3.0).fit(x)
    z = ws.transform(x)
    assert np.abs(z).max() < 3.0 / ws.std_ + 1
    assert z.std() == pytest.approx(1.0, rel=0.02)
    assert ws.transform(np.array([1e9]))[0] == pytest.approx((3.0 - ws.mean_) / ws.std_)


def test_empirical_calibration_agrees_with_theory_on_normal_noise():
    z = np.random.default_rng(9).standard_normal(20000)
    assert spc.calibrate("ewma", 370, n_rep=2000, noise=z) == pytest.approx(2.86, abs=0.08)
    with pytest.raises(ValueError):
        spc.calibrate("shewhart", 370, noise=z)


def test_hotelling_uses_the_correlation_between_sensors():
    rng = np.random.default_rng(10)
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    t2 = spc.HotellingT2(alpha=0.01).fit(rng.multivariate_normal([0, 0], cov, 5000))
    # moving both sensors together is normal, moving them in opposite directions is not
    assert t2.statistic(np.array([[2.0, 2.0]]))[0] < t2.ucl_ < t2.statistic(np.array([[1.5, -1.5]]))[0]
