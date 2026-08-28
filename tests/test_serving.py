"""Phase 1A.7: bundle save/load, leakage (fit-on-train), isolation between datasets, cold start, ordering /
idempotency, and offline-vs-online parity on a synthetic Sparkov-shaped stream (no data files needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from frauddet.adapters import ADAPTERS
from frauddet.history import DuplicateEvent, HistorySpec, OutOfOrderEvent, compute_history, snapshot_from_frame
from frauddet.preprocessing import build_pipeline
from frauddet.serving import FeatureBundle
from frauddet.splits import temporal_split
from frauddet.views import ModelView

SPEC = HistorySpec(windows_h=(1, 24, 168), vm_windows_d=(7,))


def sparkov_stream(n_cards=3, n=120, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2019-01-01")
    rows = []
    for k in range(n_cards):
        hours = np.sort(rng.uniform(0, 40 * 24, n))
        for i, h in enumerate(hours):
            rows.append({"trans_date_trans_time": base + pd.Timedelta(hours=float(h)), "cc_num": f"card{k}",
                         "merchant": rng.choice(["fraud_A", "fraud_B", "fraud_C"]), "category": rng.choice(["food", "gas"]),
                         "amt": float(rng.gamma(2, 20)), "gender": "F" if k % 2 else "M", "state": "NC",
                         "lat": 36.0 + k, "long": -81.0, "city_pop": 1000 * (k + 1), "merch_lat": 36.0 + k + rng.random(),
                         "merch_long": -81.0 + rng.random(), "dob": pd.Timestamp("1980-01-01") + pd.Timedelta(days=365 * k),
                         "trans_num": f"t{k}_{i}", "is_fraud": int(rng.random() < 0.05)})
    df = pd.DataFrame(rows)
    df["unix_time"] = (df["trans_date_trans_time"].astype("int64") // 10**9).astype("int64")
    return df.sort_values("unix_time", kind="stable").reset_index(drop=True)


def fitted_bundle(tmp_path):
    c = ADAPTERS["sparkov"].contract
    df = compute_history(sparkov_stream(), SPEC)
    sp = temporal_split(df, c, (0.7, 0.3), ("train", "val"))
    train = df.iloc[sp.parts["train"]]
    pipe = build_pipeline(c).fit(train, order_key="unix_time")
    X = pipe.transform(train)
    views = {"tree": ModelView("tree").fit(X), "linear": ModelView("linear", min_count=1).fit(X)}
    required = tuple(col for col in df.columns if col != "is_fraud" and col not in SPEC.feature_names())
    b = FeatureBundle(c, "temporal", pipe, views, None, SPEC, snapshot_from_frame(train, SPEC, "trans_num"), required)
    b.save(tmp_path / "bundle")
    return b, df, sp


def test_bundle_roundtrip_and_offline_online_parity(tmp_path):
    b, df, sp = fitted_bundle(tmp_path)
    raw = df.drop(columns=SPEC.feature_names())
    val = raw.iloc[sp.parts["val"]]
    for view in ("tree", "linear"):
        loaded = FeatureBundle.load(tmp_path / "bundle")     # fresh state per replay (a replay is a new stream)
        offline = b.transform_batch(raw, view=view).iloc[sp.parts["val"]].reset_index(drop=True)
        online = pd.concat([loaded.serve_event(ev, view=view, row_id=ev["trans_num"]) for ev in val.to_dict("records")],
                           ignore_index=True)
        pd.testing.assert_frame_equal(online, offline, check_exact=False, rtol=1e-4, atol=1e-4)
    assert loaded.serving.stateful and loaded.serving.outputs["tree"] == len(b.pipeline.feature_columns)
    assert json.loads((tmp_path / "bundle" / "bundle.json").read_text())["files"]["features.json"]


def test_streaming_state_snapshot_continues_correctly(tmp_path):
    b, df, sp = fitted_bundle(tmp_path)
    raw = df.drop(columns=SPEC.feature_names())
    val = raw.iloc[sp.parts["val"]]
    half = len(val) // 2
    loaded = FeatureBundle.load(tmp_path / "bundle")
    first = [loaded.serve_event(ev, view=None, row_id=ev["trans_num"]) for ev in val.iloc[:half].to_dict("records")]
    loaded.save(tmp_path / "bundle2")                       # snapshot mid-stream
    again = FeatureBundle.load(tmp_path / "bundle2")
    rest = [again.serve_event(ev, view=None, row_id=ev["trans_num"]) for ev in val.iloc[half:].to_dict("records")]
    online = pd.concat(first + rest, ignore_index=True)
    offline = b.transform_batch(raw, view=None).iloc[sp.parts["val"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(online, offline, check_exact=False, rtol=1e-4, atol=1e-4)


def test_cold_start_unseen_entity_and_categories(tmp_path):
    b, df, sp = fitted_bundle(tmp_path)
    ev = df.drop(columns=SPEC.feature_names()).iloc[-1].to_dict()
    ev.update(cc_num="never-seen", merchant="fraud_ZZZ", state="XX", trans_num="new1")
    t = b.serve_event(ev, view="tree", row_id="new1")
    assert t["h_n_prior"].iloc[0] == 0 and t["h_cnt_24h"].iloc[0] == 0 and np.isnan(t["h_hours_since_last"].iloc[0])
    assert str(t["merchant"].iloc[0]) == "<UNK>" and str(t["state"].iloc[0]) == "<UNK>"
    lin = b.views["linear"]
    ev2 = dict(ev, trans_num="new2", trans_date_trans_time=ev["trans_date_trans_time"] + pd.Timedelta(hours=1),
               unix_time=ev["unix_time"] + 3600)
    l = b.serve_event(ev2, view="linear", row_id="new2")
    assert not l.isna().any().any() and l.shape[1] == len(lin.state["output_columns"])
    assert b.state.last["never-seen"][1] == 2                       # the entity now has two recorded events


def test_ordering_and_idempotency_are_enforced(tmp_path):
    b, df, sp = fitted_bundle(tmp_path)
    raw = df.drop(columns=SPEC.feature_names())
    ev = raw.iloc[sp.parts["val"][0]].to_dict()
    b.serve_event(ev, view=None, row_id=ev["trans_num"])
    with pytest.raises(DuplicateEvent):
        b.serve_event(ev, view=None, row_id=ev["trans_num"])
    earlier = dict(ev, trans_num="x", unix_time=ev["unix_time"] - 10,
                   trans_date_trans_time=ev["trans_date_trans_time"] - pd.Timedelta(seconds=10))
    with pytest.raises(OutOfOrderEvent):
        b.serve_event(earlier, view=None, row_id="x")
    with pytest.raises(KeyError, match="required fields"):
        b.serve_event({"cc_num": "a"}, view=None)


def test_fitted_state_comes_only_from_training_part():
    c = ADAPTERS["ulb"].contract
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Time": np.arange(400.0), "Amount": rng.gamma(2, 30, 400), "Class": (rng.random(400) < 0.05).astype(int)})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=400)
    df.loc[300:, "V1"] += 100.0                              # a shift that only the validation part has
    train = df.iloc[:280]
    pipe = build_pipeline(c).fit(train, order_key="Time")
    st = [s for s in pipe.steps if type(s).__name__ == "Standardize"][0].state["stats"]["V1"]
    assert abs(st["mean"] - train["V1"].mean()) < 1e-6 and st["mean"] < 5      # unaffected by the val shift
    X = pipe.transform(df)
    assert X["V1"].iloc[300:].mean() > 50                                        # the shift is visible, not absorbed


def test_no_fitted_state_is_shared_between_datasets(tmp_path):
    ulb = build_pipeline(ADAPTERS["ulb"].contract)
    ulb2 = build_pipeline(ADAPTERS["ulb"].contract)
    sp = build_pipeline(ADAPTERS["sparkov"].contract)
    ie = build_pipeline(ADAPTERS["ieee"].contract)
    ids = [id(s) for p in (ulb, ulb2, sp, ie) for s in p.steps]
    assert len(ids) == len(set(ids))                                             # no shared step objects
    assert {p.dataset for p in (ulb, sp, ie)} == {"ulb", "sparkov", "ieee"}
    # loading a bundle for one dataset never touches another's artifacts: the index carries the dataset name
    b, *_ = fitted_bundle(tmp_path)
    idx = json.loads((tmp_path / "bundle" / "bundle.json").read_text())
    assert idx["dataset"] == "sparkov" and idx["contract_version"] == b.contract.contract_version
    (tmp_path / "bundle" / "view-tree.json").write_text("{}")                    # tampered artifact
    with pytest.raises(RuntimeError, match="does not match"):
        FeatureBundle.load(tmp_path / "bundle")


def test_extra_json_files_next_to_a_bundle_do_not_break_loading(tmp_path):
    b, *_ = fitted_bundle(tmp_path)
    (tmp_path / "bundle" / "manifest.json").write_text("{}")          # prepare writes these after save()
    (tmp_path / "bundle" / "experiment.json").write_text("{}")
    loaded = FeatureBundle.load(tmp_path / "bundle")
    assert set(json.loads((tmp_path / "bundle" / "bundle.json").read_text())["files"]) == {
        "features.json", "view-tree.json", "view-linear.json", "history-state.json"}
    assert loaded.serving.outputs == b.serving.outputs
