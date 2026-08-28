"""Unit tests for the preprocessing steps and pipelines (synthetic frames)."""

import json

import numpy as np
import pandas as pd
import pytest

from frauddet.adapters import ADAPTERS
from frauddet.labels import TARGETS
from frauddet.preprocessing import (
    DAY, MA2026_PROTOCOL, AmountDecimals, CategoricalEncoder, CyclicClock, FinalizeFeatures, MedianImputer,
    MissingIndicator, Pipeline, Standardize, build_pipeline,
)


def _ulb(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Time": np.sort(rng.integers(0, 2 * DAY, n)).astype(float),
                       "Amount": np.round(rng.gamma(2, 30, n), 2), "Class": (rng.random(n) < 0.05).astype(int)})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n)
    return df


def test_categorical_encoder_learns_on_train_only_and_maps_unseen_to_zero():
    enc = CategoricalEncoder(["c"]).fit(pd.DataFrame({"c": ["a", "b", "a", None]}))
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "zzz", None]}))
    assert out["c"].tolist() == [1, 2, 0, 0]
    assert enc.state["categories"]["c"] == ["a", "b"]


def test_missing_indicator_and_median_imputer_use_train_statistics():
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [1.0, 2.0, 3.0]})
    val = pd.DataFrame({"x": [np.nan, 10.0], "y": [np.nan, 100.0]})
    mi = MissingIndicator().fit(train)
    assert mi.state["columns"] == ["x"]                    # y had no nulls in train
    imp = MedianImputer().fit(train)
    out = imp.transform(mi.transform(val))
    assert out["x"].tolist() == [2.0, 10.0] and out["x__isna"].tolist() == [1, 0]
    assert out["y"].tolist() == [2.0, 100.0]               # imputable at serving even without indicator


def test_amount_decimals_and_cyclic_clock():
    df = pd.DataFrame({"a": [10.0, 10.5, 10.25, 10.125], "t": [0.0, 6 * 3600.0, DAY + 12 * 3600.0, 23 * 3600.0]})
    assert AmountDecimals("a").transform(df)["a_decimals"].tolist() == [0, 1, 2, 3]
    out = CyclicClock("t", DAY, "clk", bins=24).transform(df)
    assert out["clk_bin"].tolist() == [0, 6, 12, 23]
    assert np.allclose(out["clk_sin"], [0, 1, 0, np.sin(2 * np.pi * 23 / 24)], atol=1e-6)


def test_standardize_uses_train_moments():
    st = Standardize(["v"]).fit(pd.DataFrame({"v": [0.0, 2.0]}))
    assert st.state["stats"]["v"] == {"mean": 1.0, "std": 1.0}
    assert st.transform(pd.DataFrame({"v": [4.0]}))["v"].tolist() == [3.0]


def test_finalize_excludes_never_input_roles():
    df = _ulb(50)
    fin = FinalizeFeatures("ulb").fit(df)
    cols = fin.state["feature_columns"]
    assert "Class" not in cols and "Time" not in cols and "Amount" in cols and "V1" in cols
    with pytest.raises(KeyError):
        fin.transform(df.drop(columns=["V1"]))


def test_ulb_pipeline_roundtrip_and_serving_parity(tmp_path):
    df = _ulb()
    c = ADAPTERS["ulb"].contract
    pipe = build_pipeline(c).fit(df.iloc[:200], order_key="Time")
    TARGETS["ulb"].assert_no_label_leak(pipe.feature_columns, c)
    X = pipe.transform(df)
    assert "Time" not in X.columns and "Class" not in X.columns and "clk_day_sin" in X.columns
    assert not X.isna().any().any()
    path = pipe.save(tmp_path / "pipeline.json")
    json.loads(path.read_text())                                        # plain JSON, no pickles
    loaded = Pipeline.load(path)
    X2 = loaded.transform(df)
    pd.testing.assert_frame_equal(X, X2)
    for i in (0, 123, 299):                                             # one event == batch row
        pd.testing.assert_frame_equal(loaded.transform(df.iloc[[i]]).reset_index(drop=True),
                                      X.iloc[[i]].reset_index(drop=True))


def test_ma2026_pipeline_keeps_time_but_temporal_does_not():
    df = _ulb()
    c = ADAPTERS["ulb"].contract
    ma = build_pipeline(c, MA2026_PROTOCOL).fit(df)
    assert ma.feature_columns[0] == "Time" and len(ma.feature_columns) == 30
    tp = build_pipeline(c).fit(df)
    assert "Time" not in tp.feature_columns


def test_pipeline_refuses_transform_before_fit():
    with pytest.raises(RuntimeError):
        build_pipeline(ADAPTERS["ulb"].contract).transform(_ulb(10))
