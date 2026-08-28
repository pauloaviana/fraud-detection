"""Data-backed tests for 1A.4 (skipped when data/ is absent). ULB is small enough to run end to end;
IEEE/Sparkov checks use column subsets."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frauddet.adapters import get_adapter
from frauddet.labels import TARGETS
from frauddet.prepare import prepare_frame, run
from frauddet.preprocessing import DAY, MA2026_PROTOCOL, CalendarFeatures, CyclicClock
from frauddet.splits import TEMPORAL, temporal_split, verify

DATA = Path(__file__).resolve().parents[1] / "data"


def _adapter(name):
    a = get_adapter(name, DATA)
    if not all(a.available(f.key) for f in a.contract.files):
        pytest.skip(f"{name}: data files not present")
    return a


def test_ulb_prepare_dedups_on_all_columns_and_temporal_split_holds(tmp_path):
    a = _adapter("ulb")
    df, notes = prepare_frame(a)
    assert notes["dedup"]["removed_rows"] == 1081 and len(df) == 283726
    sp = temporal_split(df, a.contract)
    assert verify(sp, df) == []
    assert sp.summary["train"]["positives"] + sp.summary["val"]["positives"] + sp.summary["holdout"]["positives"] == 473
    m = run(a, TEMPORAL, tmp_path)
    assert m["parts"]["holdout"]["prevalence"] < 0.003 and m["n_features"] == 32     # 28 V + Amount_log1p + 3 clock
    assert m["views"]["tree"] == 32 and m["views"]["linear"] == 32 and m["selection"] is None
    out = tmp_path / "ulb" / "temporal"
    assert (out / "experiment.json").exists() and (out / "cost-context-holdout.csv").exists()
    ctx = pd.read_csv(out / "cost-context-holdout.csv")
    assert ctx["y"].sum() == m["parts"]["holdout"]["positives"] and (ctx["amount"] >= 0).all()   # raw amount kept
    assert m["imbalance"]["natural_parts"]["holdout"]["natural"] is True


def test_ulb_ma2026_benchmark_reproduces_paper_counts(tmp_path):
    a = _adapter("ulb")
    m = run(a, MA2026_PROTOCOL, tmp_path)
    s = m["split"]["summary"]
    assert (s["train"]["rows"], s["train"]["positives"]) == (199364, 344)
    assert (s["test"]["rows"], s["test"]["positives"]) == (85443, 148)
    assert m["feature_columns"][0] == "Time" and m["n_features"] == 30
    assert m["selection"]["k"] == 15 and len(m["selection"]["selected"]) == 15
    assert len(m["selection"]["overlap_with_ma2026_top15"]) >= 10
    assert m["selection"]["mode"] == "rf_gini_refit" and m["selection"]["published_alternative"]["mode"] == "ma2026_published"


def test_ieee_daily_phase_is_anchored_by_d9():
    a = _adapter("ieee")
    df = a.load("train", usecols=["TransactionDT", "D9"])
    out = CyclicClock("TransactionDT", DAY, "clk_day", bins=24).transform(df)
    m = df["D9"].notna()
    assert np.allclose(out.loc[m, "clk_day_bin"], df.loc[m, "D9"] * 24, atol=1e-3)


def test_sparkov_calendar_from_datetime_not_unix_time():
    a = _adapter("sparkov")
    df = a.load("train", usecols=["trans_date_trans_time", "unix_time"], nrows=2000)
    cal = CalendarFeatures("trans_date_trans_time").transform(df)
    assert (cal["cal_dow"] == df["trans_date_trans_time"].dt.dayofweek).all()
    wrong = pd.to_datetime(df["unix_time"], unit="s").dt.dayofweek
    assert (cal["cal_dow"] != wrong).mean() > 0.9                 # unix_time weekday is the wrong one


def test_no_label_leak_in_any_feature_list():
    for name in ("sparkov", "ieee", "ulb"):
        a = _adapter(name)
        c = a.contract
        with pytest.raises(ValueError):
            TARGETS[name].assert_no_label_leak([c.target, "x"], c)


def test_sparkov_history_on_real_rows_is_causal_and_store_parity():
    from frauddet.history import EntityStateStore, HistorySpec, compute_history
    a = _adapter("sparkov")
    df = a.load("train", nrows=20000)
    spec = HistorySpec()
    cards = df["cc_num"].value_counts().index[:3]
    sub = df[df["cc_num"].isin(cards)].sort_values("unix_time", kind="stable").reset_index(drop=True)
    batch = compute_history(sub, spec)
    full = compute_history(df, spec)
    names = spec.feature_names()
    pd.testing.assert_frame_equal(full.loc[full["cc_num"].isin(cards), names].reset_index(drop=True),
                                  batch[names].reset_index(drop=True))              # per-entity: subset == full
    store = EntityStateStore(spec)
    seq = pd.DataFrame([store.process(r) for r in sub.to_dict("records")])[names].astype("float32")
    pd.testing.assert_frame_equal(seq, batch[names].reset_index(drop=True), check_exact=False, rtol=1e-4, atol=1e-4)
