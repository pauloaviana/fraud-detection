"""Imbalance treatment: weighting, training-fold-only resampling, natural evaluation parts, folds, costs."""

import numpy as np
import pandas as pd
import pytest

from frauddet.adapters import ADAPTERS
from frauddet.folds import forward_chaining_folds, stratified_folds, training_folds
from frauddet.imbalance import (
    CostMatrix, ExperimentConfig, ImbalanceSpec, check_natural, class_weights, cost_context,
    resample_training_fold, sample_weights,
)


def _data(n=2000, p=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < p).astype(np.int8)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    X.loc[y == 1, "a"] += 2.0
    return X, y


def test_class_and_example_weights():
    _, y = _data()
    cw = class_weights(y)
    assert cw[1] > cw[0] and abs(cw[0] * (y == 0).sum() - cw[1] * (y == 1).sum()) < 1e-6   # balanced classes
    w = sample_weights(y, ImbalanceSpec("class_weight"))
    assert abs(w.mean() - 1) < 1e-9 and w[y == 1].min() > w[y == 0].max()
    amount = np.full(len(y), 10.0); amount[np.flatnonzero(y == 1)[:3]] = 0.0
    w = sample_weights(y, ImbalanceSpec("example_weight", ca=2.0, amount_floor=1.0), amount)
    raw = np.where(y == 1, np.maximum(amount, 1.0), 2.0)
    assert np.allclose(w, raw / raw.mean())
    with pytest.raises(ValueError):
        sample_weights(y, ImbalanceSpec("example_weight"))


@pytest.mark.parametrize("method", ["random_under", "random_over", "smote", "enn", "smote_enn"])
def test_resampling_only_on_training_fold_and_reproducible(method):
    X, y = _data()
    spec = ImbalanceSpec(method, sampling_ratio=0.5 if method != "enn" else 1.0)
    Xr, yr, meta = resample_training_fold(X, y, spec)
    Xr2, yr2, _ = resample_training_fold(X, y, spec)
    assert np.array_equal(yr, yr2) and np.allclose(Xr.to_numpy(float), Xr2.to_numpy(float))
    assert meta["before"]["positives"] == int(y.sum()) and meta["method"] == method
    if method == "random_under":
        assert yr.sum() == y.sum() and abs(yr.mean() - 1 / 3) < 0.02
    if method in ("random_over", "smote"):
        assert abs(yr.mean() - 1 / 3) < 0.02 and (yr == 0).sum() == (y == 0).sum()
    if method == "smote":
        assert meta["synthetic_added"] > 0 and len(Xr) == len(yr)
    if method in ("enn", "smote_enn"):
        assert meta["removed_by_enn"] >= 0 and len(Xr) == len(yr)
    with pytest.raises(ValueError, match="training fold"):
        resample_training_fold(X, y, spec, fold_role="val")


def test_none_and_weighting_leave_rows_untouched():
    X, y = _data()
    for m in ("none", "class_weight", "example_weight"):
        Xr, yr, meta = resample_training_fold(X, y, ImbalanceSpec(m))
        assert Xr is X and np.array_equal(yr, y) and meta["after"] == meta["before"]


def test_smote_refuses_categoricals_or_nan():
    X, y = _data(300)
    Xc = X.copy(); Xc["c"] = pd.Categorical(["u", "v"] * 150)
    with pytest.raises(ValueError, match="categorical"):
        resample_training_fold(Xc, y, ImbalanceSpec("smote"))
    Xn = X.copy(); Xn.loc[0, "a"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        resample_training_fold(Xn, y, ImbalanceSpec("smote_enn"))
    Xr, yr, _ = resample_training_fold(Xn, y, ImbalanceSpec("random_over"))     # row-level methods are fine
    assert len(yr) > len(y)


def test_check_natural_detects_balanced_eval():
    _, y = _data()
    assert check_natural(y[:500], y[:500], "val")["natural"]
    with pytest.raises(RuntimeError):
        check_natural(np.r_[y[:500], 1, 1], y[:500], "val")


def test_folds_keep_natural_validation_prevalence():
    rng = np.random.default_rng(1)
    t = np.sort(rng.integers(0, 1000, 3000)).astype(float)
    y = (rng.random(3000) < 0.05).astype(int)
    folds = forward_chaining_folds(t, 3)
    for tr, va in folds:
        assert t[tr].max() < t[va].min()                    # strictly earlier training data
        assert len(np.intersect1d(tr, va)) == 0
    assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])   # expanding window
    sf = stratified_folds(y, 3)
    assert all(abs(y[va].mean() - y.mean()) < 0.02 for _, va in sf)
    _, meta = training_folds("temporal", t, y)
    assert meta["kind"] == "forward_chaining" and len(meta["folds"]) == 3
    _, meta = training_folds("stratified_ma2026", t, y)
    assert meta["kind"] == "stratified_kfold" and meta["seed"] == 42


def test_cost_matrix_matches_bahnsen():
    y = np.array([1, 1, 0, 0, 1]); amt = np.array([100.0, 5.0, 50.0, 7.0, 20.0])
    pred = np.array([1, 0, 1, 0, 0])
    cm = CostMatrix(ca=2.0)
    assert cm.costs(y, pred, amt).tolist() == [2.0, 5.0, 2.0, 0.0, 20.0]     # TP=Ca, FN=amt, FP=Ca, TN=0
    assert cm.total_cost(y, pred, amt) == 29.0 and cm.cost_no_model(y, amt) == 125.0
    assert cm.savings(y, pred, amt) == pytest.approx((125.0 - 29.0) / 125.0)
    assert cm.savings(y, np.zeros(5, int), amt) == 0.0                         # no model = no savings


def test_cost_context_keeps_raw_amount_and_experiment_metadata(tmp_path):
    c = ADAPTERS["ulb"].contract
    df = pd.DataFrame({"Time": [0.0, 5.0], "Amount": [0.0, 12.5], "Class": [0, 1]})
    for i in range(1, 29):
        df[f"V{i}"] = 0.0
    ctx = cost_context(df, c, df["Class"])
    assert ctx["amount"].tolist() == [0.0, 12.5] and ctx["y"].tolist() == [0, 1] and "row_id" not in ctx
    cfg = ExperimentConfig("ulb", "temporal", "tree", ImbalanceSpec("class_weight"), folds={"kind": "forward_chaining"})
    d = cfg.to_dict()
    assert d["imbalance"]["method"] == "class_weight" and d["selection"] == "none"
    cfg.save(tmp_path / "e.json")
    assert (tmp_path / "e.json").exists()
