"""Phase 1B unit tests: metrics, policies, calibration, models (dummy/logreg without GBDT deps), sealing."""

import json

import numpy as np
import pandas as pd
import pytest

from frauddet.calibrate import Calibrator, fit_calibrators
from frauddet.experiment import SealedHoldout, choose_champion
from frauddet.metrics import at_threshold, business, calibration, discrimination, precision_at_budget, recall_at_fpr
from frauddet.models import Model, search_space
from frauddet.policy import alert_budget, cost_optimal, f1_max, fpr_budget, select_thresholds


def _scores(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.02).astype(int)
    p = np.clip(rng.normal(0.02 + 0.4 * y, 0.15), 0, 1)
    amount = rng.gamma(2, 40, n)
    return y, p, amount


def test_discrimination_and_operating_point_basics():
    y, p, _ = _scores()
    d = discrimination(y, p)
    assert 0.5 < d["pr_auc"] <= 1 and d["roc_auc"] > 0.9 and d["prevalence"] == pytest.approx(y.mean())
    r, t = recall_at_fpr(y, p, 0.01)
    assert at_threshold(y, p, t)["fpr"] <= 0.01 + 1e-9 and 0 < r <= 1
    pr, rc, thr = precision_at_budget(y, p, 0.01)
    assert abs(at_threshold(y, p, thr)["alert_rate"] - 0.01) < 0.002 and pr > y.mean()
    op = at_threshold(y, p, 0.5)
    assert op["tp"] + op["fn"] == y.sum() and op["fp"] + op["tn"] == (y == 0).sum()
    perfect = at_threshold(y, y.astype(float), 0.5)
    assert perfect["f1"] == 1.0 and perfect["mcc"] == 1.0


def test_calibration_metrics_and_calibrators_are_monotone():
    y, p, _ = _scores()
    c = calibration(y, p)
    assert c["brier"] < 0.1 and len(c["reliability"]) == 10 and 0 <= c["ece"] <= 1
    cals = fit_calibrators(p, y)
    for name, cal in cals.items():
        q = cal.transform(np.sort(p))
        assert np.all(np.diff(q) >= -1e-9), name                           # monotone: order never reversed
    # Platt is strictly monotone -> ranking metrics unchanged; isotonic may merge scores into plateaus
    assert discrimination(y, cals["platt"].transform(p))["roc_auc"] == pytest.approx(discrimination(y, p)["roc_auc"], abs=1e-6)
    assert calibration(y, cals["isotonic"].transform(p))["ece"] <= calibration(y, p)["ece"] + 1e-9
    d = cals["platt"].to_dict(); assert Calibrator.from_dict(d).transform(p[:5]).tolist() == cals["platt"].transform(p[:5]).tolist()


def test_policies_use_only_the_data_they_are_given_and_cost_matches_bahnsen():
    y, p, amount = _scores()
    assert at_threshold(y, p, fpr_budget(y, p, 0.005))["fpr"] <= 0.005 + 1e-9
    assert abs(at_threshold(y, p, alert_budget(p, 0.01))["alert_rate"] - 0.01) < 0.002
    t_f1 = f1_max(y, p)
    assert at_threshold(y, p, t_f1)["f1"] >= max(at_threshold(y, p, t)["f1"] for t in (0.2, 0.4, 0.6))
    t_cost = cost_optimal(y, p, amount, ca=2.0)
    b = business(y, p, t_cost, amount, ca=2.0)
    assert b["cost"] <= business(y, p, 0.5, amount, ca=2.0)["cost"] and b["savings"] > 0
    sel = select_thresholds(y, p, amount, ca=2.0)
    assert set(sel["thresholds"]) >= {"f1_max", "fpr_0.001", "fpr_0.005", "alert_0.005", "alert_0.01", "cost_ca2"}
    assert sel["selected_on"]["n"] == len(y)


def test_dummy_and_logreg_models_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 4)).astype(np.float32), columns=list("abcd"))
    y = (X["a"] + rng.normal(scale=0.5, size=600) > 1.2).astype(int).to_numpy()
    d = Model("dummy").fit(X, y)
    assert np.allclose(d.predict_proba(X), y.mean())
    m = Model("logreg", {"C": 1.0}).fit(X, y, sample_weight=np.ones(600))
    p = m.predict_proba(X)
    assert discrimination(y, p)["roc_auc"] > 0.9
    m.save(tmp_path / "m")
    m2 = Model.load(tmp_path / "m")
    assert np.allclose(m2.predict_proba(X), p) and m2.size_bytes == m.size_bytes and set(m.importance()) == set("abcd")
    assert len(search_space("xgboost", 4)) == 4 and len(search_space("lightgbm", 6)) == 6 and search_space("logreg", 3)[0] == {"C": 0.01}


def test_sealed_holdout_cannot_be_read_before_lock():
    y = np.array([0, 1, 0]); X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}); ctx = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "y": y})
    h = SealedHoldout(X, y, ctx)
    with pytest.raises(RuntimeError, match="locked"):
        h.unseal({})
    Xh, yh, _ = h.unseal({"champion": {"model": "logreg"}})
    assert len(yh) == 3 and h.unsealed_at is not None
    with pytest.raises(RuntimeError, match="already"):
        h.unseal({"champion": {}})


def test_champion_choice_prefers_validation_pr_auc_then_calibration():
    def cand(model, pr, brier):
        return {"model": model, "treatment": "none", "params": {}, "n_estimators": None, "cv_mean_pr_auc": pr,
                "validation": {c: {"discrimination": {"pr_auc": pr}, "calibration": {"log_loss": brier, "brier": brier}}
                               for c in ("none", "platt", "isotonic")}}
    locked = choose_champion([cand("dummy", 0.01, 0.5), cand("logreg", 0.6, 0.02), cand("lightgbm", 0.7, 0.05)], True)
    assert locked["champion"]["model"] == "lightgbm" and locked["ranking"][0]["model"] == "lightgbm"
    assert "dummy" not in {r["model"] for r in locked["ranking"]}


def test_report_renders_from_results_json():
    from pathlib import Path
    from frauddet.report1b import render, summary_rows
    f = Path(__file__).resolve().parents[1] / "experiments" / "ulb" / "temporal" / "results.json"
    if not f.exists():
        pytest.skip("no experiment results present")
    r = json.loads(f.read_text())
    org = render(r)
    assert "* Final holdout (evaluated once)" in org and r["holdout"]["unsealed_at"] > r["locked"]["locked_at"]
    assert len(summary_rows([r])) == 1 and r["bundle_sha256"]


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_gbdt_models_roundtrip_with_categoricals_and_nan(name, tmp_path):
    pytest.importorskip(name)
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({"a": rng.normal(size=n).astype(np.float32), "b": rng.normal(size=n).astype(np.float32),
                      "c": pd.Categorical(rng.choice(["u", "v", "w"], n))})
    X.loc[::9, "a"] = np.nan
    y = ((X["a"].fillna(0) + (X["c"] == "u")) > 1.0).astype(int).to_numpy()
    m = Model(name, search_space(name, 1)[0], n_jobs=2).fit(X.iloc[:1000], y[:1000], eval_set=(X.iloc[1000:], y[1000:]))
    p = m.predict_proba(X)
    assert m.best_iteration and discrimination(y, p)["roc_auc"] > 0.9
    m.save(tmp_path / name)
    m2 = Model.load(tmp_path / name)
    assert np.allclose(m2.predict_proba(X), p, atol=1e-6) and m2.size_bytes > 0
    assert set(m.importance()) == {"a", "b", "c"}
