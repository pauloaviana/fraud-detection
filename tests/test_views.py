"""Typed categoricals, model views, and the RF-Gini selector."""

import numpy as np
import pandas as pd
import pytest

from frauddet.preprocessing import CategoricalTyper, FrequencyEncoder, RowMissingCount, TokenSplit
from frauddet.views import ModelView, RFGiniSelector


def _typed(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n), "c": rng.choice(["a", "b", "c"], n)})
    df.loc[::7, "x"] = np.nan
    df.loc[::11, "c"] = None
    df.loc[:2, "c"] = "rare"
    return CategoricalTyper(["c"]).fit(df).transform(df)


def test_typer_is_categorical_not_ordinal_and_handles_unseen():
    train = pd.DataFrame({"c": ["b", "a", "a", None]})
    typ = CategoricalTyper(["c"]).fit(train)
    out = typ.transform(pd.DataFrame({"c": ["a", "zzz", None]}))
    assert isinstance(out["c"].dtype, pd.CategoricalDtype)
    assert out["c"].tolist() == ["a", "<UNK>", "<NA>"]
    assert list(out["c"].cat.categories[:2]) == ["<NA>", "<UNK>"]


def test_frequency_token_and_rowmissing_steps():
    tr = pd.DataFrame({"e": ["gmail.com", "gmail.com", "yahoo.co.uk", None], "V1": [1, None, 3, None], "V2": [None] * 4})
    fe = FrequencyEncoder(["e"]).fit(tr)
    assert fe.transform(pd.DataFrame({"e": ["gmail.com", "new.org", None]}))["e_freq"].tolist() == pytest.approx([2 / 3, 0, 0])
    ts = TokenSplit("e", "prov", index=0).transform(tr)
    assert ts["prov"].tolist()[:3] == ["gmail", "gmail", "yahoo"] and pd.isna(ts["prov"].iloc[3])
    assert TokenSplit("e", "tld", index=-1).transform(tr)["tld"].tolist()[2] == "uk"
    rm = RowMissingCount(r"V\d+", "n_missing_V").fit(tr)
    assert rm.transform(tr)["n_missing_V"].tolist() == [1, 2, 1, 2]


def test_tree_view_keeps_nan_and_native_categoricals():
    X = _typed()
    v = ModelView("tree").fit(X)
    out = v.transform(X)
    assert out["x"].isna().sum() == X["x"].isna().sum()                    # NaN preserved for GBDTs
    assert isinstance(out["c"].dtype, pd.CategoricalDtype)
    assert list(out.columns) == list(X.columns)


def test_linear_view_imputes_onehots_and_scales_with_train_stats(tmp_path):
    X = _typed()
    v = ModelView("linear", min_count=5).fit(X.iloc[:150])
    out = v.transform(X)
    assert not out.isna().any().any()
    assert {"c=a", "c=b", "c=c", "c=<RARE>"} <= set(out.columns) and "c=rare" not in out.columns
    assert abs(out["x"].iloc[:150].mean()) < 0.05                           # standardised on train
    v.save(tmp_path / "v.json")
    out2 = ModelView.load(tmp_path / "v.json").transform(X)
    pd.testing.assert_frame_equal(out, out2)
    one = ModelView.load(tmp_path / "v.json").transform(X.iloc[[5]]).reset_index(drop=True)
    pd.testing.assert_frame_equal(one, out.iloc[[5]].reset_index(drop=True))


def test_rf_gini_selector_fits_on_train_only_and_serialises(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f2"] + 0.5 * X["f4"] + rng.normal(scale=0.3, size=n) > 0.8).astype(int))
    sel = RFGiniSelector(k=2, n_estimators=50).fit(X.iloc[:400], y.iloc[:400])
    assert set(sel.selected) == {"f2", "f4"} and sel.state["fitted_rows"] == 400
    sel.save(tmp_path / "s.json")
    assert RFGiniSelector.load(tmp_path / "s.json").transform(X).columns.tolist() == sel.selected
