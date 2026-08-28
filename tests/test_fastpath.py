"""Bit-parity of the row-native fast path against the pandas reference on the four locked bundles.
Skipped where artifacts/data are absent."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
from frauddet.adapters import get_adapter  # noqa: E402
from frauddet.adapters.ieee import canonical_name  # noqa: E402
from frauddet.calibrate import Calibrator  # noqa: E402
from frauddet.fastpath import compile_fast_scorer  # noqa: E402
from frauddet.history import compute_history  # noqa: E402
from frauddet.models import Model  # noqa: E402
from frauddet.serving import FeatureBundle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(dataset, protocol):
    bdir, mdir = ROOT / "artifacts" / dataset / protocol, ROOT / "experiments" / dataset / protocol
    if not (mdir / "model" / "model.json").exists():
        pytest.skip(f"{dataset}/{protocol}: locked artifacts absent")
    if dataset == "ulb" and not (ROOT / "data" / "creditcardfraud.csv").exists():
        pytest.skip("data absent")
    b = FeatureBundle.load(bdir)
    m = Model.load(mdir / "model")
    if m.name == "xgboost":
        pytest.importorskip("xgboost")
    return b, m, Calibrator.load(mdir / "calibrator.json")


def _reference(b, m, cal, frame):
    X = b.transform_batch(frame, view=m.view, with_history=False)
    if b.selector is not None:
        X = b.selector.transform(X)
    return X, cal.transform(m.predict_proba(X))


def _matrix(X, cols):
    """Reference float32 matrix: categoricals as their codes (exactly what LightGBM sees from a DataFrame)."""
    out = np.empty((len(X), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        col = X[c]
        out[:, j] = col.cat.codes.to_numpy() if isinstance(col.dtype, pd.CategoricalDtype) else col.to_numpy(dtype=np.float32)
    return out


def _rows_to_events(frame):
    return [{k: (None if (isinstance(v, float) and np.isnan(v)) or v is pd.NA else v) for k, v in r.items()}
            for r in frame.to_dict("records")]


@pytest.mark.parametrize("protocol", ["temporal", "stratified_ma2026"])
def test_ulb_fast_path_is_bit_identical(protocol):
    b, m, cal = _load("ulb", protocol)
    df = pd.read_csv(ROOT / "data" / "creditcardfraud.csv", nrows=2000).drop(columns=["Class"])
    X, p_ref = _reference(b, m, cal, df)
    fs = compile_fast_scorer(b, m, cal)
    vec = np.vstack([fs.features(ev) for ev in _rows_to_events(df)])
    ref = _matrix(X, fs.columns)
    assert np.array_equal(vec, ref, equal_nan=True)
    p = np.array([fs.score(ev) for ev in _rows_to_events(df)])
    assert np.array_equal(p, p_ref), float(np.max(np.abs(p - p_ref)))


def test_ieee_fast_path_is_bit_identical():
    b, m, cal = _load("ieee", "temporal")
    a = get_adapter("ieee", ROOT / "data")
    if not a.available("train"):
        pytest.skip("data absent")
    tx = a.load("train", nrows=1500)
    ident = a.load("train_identity", nrows=6000)
    ident.columns = [canonical_name(c) for c in ident.columns]
    df = tx.merge(ident, on="TransactionID", how="left")
    df["has_identity"] = df["TransactionID"].isin(ident["TransactionID"]).astype("int8")
    df = df.drop(columns=["isFraud"])
    X, p_ref = _reference(b, m, cal, df)
    fs = compile_fast_scorer(b, m, cal)
    events = _rows_to_events(df)
    vec = np.vstack([fs.features(ev) for ev in events])
    ref = _matrix(X, fs.columns)
    bad = np.flatnonzero(~np.all((vec == ref) | (np.isnan(vec) & np.isnan(ref)), axis=0))
    assert len(bad) == 0, [fs.columns[i] for i in bad[:10]]
    p = np.array([fs.score(ev) for ev in events])
    assert np.array_equal(p, p_ref), float(np.max(np.abs(p - p_ref)))


def test_sparkov_fast_path_and_sequential_state_are_bit_identical():
    b, m, cal = _load("sparkov", "temporal")
    a = get_adapter("sparkov", ROOT / "data")
    if not a.available("train"):
        pytest.skip("data absent")
    df = a.load("train")
    cut = json.load(open(ROOT / "artifacts" / "sparkov" / "temporal" / "split.json"))["boundaries"]["train"]
    val = df[df["unix_time"] > cut].sort_values("unix_time", kind="stable").head(400)
    hist = compute_history(df, b.history_spec)                       # offline reference (whole frame)
    X, p_ref = _reference(b, m, cal, hist.loc[val.index])
    # sequential online path: state restored from the shipped snapshot, fast row functions on top
    store = FeatureBundle.load(ROOT / "artifacts" / "sparkov" / "temporal").state      # v2 serving state
    fs = compile_fast_scorer(b, m, cal)
    raw_cols = [c for c in val.columns if c not in b.history_spec.feature_names() and c != "is_fraud"]
    ps, vecs = [], []
    for r in _rows_to_events(val[raw_cols]):
        feats = store.process(r, r["trans_num"])
        ev = {**r, **{k: np.float32(v) for k, v in feats.items()}}
        vecs.append(fs.features(ev).copy())
        ps.append(fs.score(ev))
    vec = np.vstack(vecs)
    ref = _matrix(X, fs.columns)
    bad = np.flatnonzero(~np.all((vec == ref) | (np.isnan(vec) & np.isnan(ref)), axis=0))
    assert len(bad) == 0, [fs.columns[i] for i in bad[:10]]
    p = np.array(ps)
    assert np.array_equal(p, p_ref), float(np.max(np.abs(p - p_ref)))       # max |Δp| == 0
