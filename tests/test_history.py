"""Causal history features: strictly-prior semantics, window arithmetic, von Mises sanity, and parity
between the batch computation and the serving-side EntityStateStore."""

import numpy as np
import pandas as pd
import pytest

from frauddet.history import (
    EntityStateStore, HistorySpec, compute_history, kappa_from_R, snapshot_from_frame, vm_halfwidth,
)

SPEC = HistorySpec(windows_h=(1, 24, 168), vm_windows_d=(7,), vm_min_prior=3)


def _events(rows):
    base = pd.Timestamp("2019-01-01 00:00:00")
    df = pd.DataFrame(rows, columns=["cc_num", "hours", "amt", "category", "merchant", "merch_lat", "merch_long"])
    df["trans_date_trans_time"] = base + pd.to_timedelta(df["hours"], unit="h")
    df["unix_time"] = (df["trans_date_trans_time"].astype("int64") // 10**9).astype(float)
    df["lat"], df["long"] = 40.0, -75.0
    return df.drop(columns=["hours"])


def test_windows_are_strictly_prior_and_exclusive_of_self():
    df = _events([
        ["A", 0.0, 10, "food", "m1", 40.0, -75.0],
        ["A", 0.5, 20, "food", "m2", 40.1, -75.0],
        ["A", 2.0, 30, "gas", "m1", 40.2, -75.0],
        ["B", 1.0, 99, "gas", "m9", 41.0, -75.0],
        ["A", 30.0, 40, "food", "m1", 40.0, -75.0],
    ])
    h = compute_history(df, SPEC)
    assert h["h_n_prior"].tolist() == [0, 1, 2, 0, 3]
    assert h["h_cnt_1h"].tolist() == [0, 1, 0, 0, 0]              # 2.0 h: both prior events are >1 h old
    assert h["h_amt_24h"].tolist() == [0, 10, 30, 0, 0]           # 30 h: nothing within 24 h
    assert h["h_amt_168h"].tolist() == [0, 10, 30, 0, 60]
    assert h["h_category_cnt_168h"].tolist() == [0, 1, 0, 0, 2]   # same category, prior only
    assert h["h_merchant_cnt_168h"].tolist() == [0, 0, 1, 0, 2]
    assert np.isnan(h["h_hours_since_last"].iloc[0]) and h["h_hours_since_last"].iloc[1] == pytest.approx(0.5)
    assert h["h_prev_amt"].iloc[4] == 30 and h["h_amt_ratio_prev"].iloc[4] == pytest.approx(40 / 30)
    assert np.isnan(h["h_speed_kmh"].iloc[0]) and h["h_speed_kmh"].iloc[1] > 0


def test_future_events_never_change_past_features():
    rng = np.random.default_rng(0)
    rows = [["A", float(hh), float(a), c, m, 40 + rng.random(), -75 + rng.random()]
            for hh, a, c, m in zip(np.sort(rng.uniform(0, 400, 60)), rng.gamma(2, 20, 60),
                                    rng.choice(["a", "b"], 60), rng.choice(["m1", "m2", "m3"], 60))]
    df = _events(rows)
    full = compute_history(df, SPEC)
    cut = 30
    part = compute_history(df.iloc[:cut].copy(), SPEC)
    pd.testing.assert_frame_equal(full.iloc[:cut].reset_index(drop=True), part.reset_index(drop=True))
    shuffled = compute_history(df.sample(frac=1, random_state=3), SPEC).sort_index()
    pd.testing.assert_frame_equal(full, shuffled)


def test_von_mises_flags_concentrated_times():
    # 20 prior events at ~09:00 then one at 09:10 (inside) and one at 21:00 (outside)
    rows = [["A", 24 * d + 9 + 0.1 * (d % 3), 10.0, "a", "m", 40.0, -75.0] for d in range(6)]
    rows += [["A", 24 * 6 + 9.17, 10.0, "a", "m", 40.0, -75.0], ["A", 24 * 6 + 21.0, 10.0, "a", "m", 40.0, -75.0]]
    h = compute_history(_events(rows), SPEC)
    assert h["vm7_in_ci90"].iloc[6] == 1.0 and h["vm7_in_ci90"].iloc[7] == 0.0
    assert h["vm7_R"].iloc[6] > 0.99 and h["vm7_dist_h"].iloc[7] == pytest.approx(12, abs=0.5)
    assert np.isnan(h["vm7_R"].iloc[1])                          # fewer than vm_min_prior events
    assert vm_halfwidth(0.9, kappa_from_R([0.0]))[0] == pytest.approx(np.pi)   # uniform -> whole circle


def test_store_parity_with_batch_and_snapshot_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for card in ("A", "B"):
        for hh, a, c, m in zip(np.sort(rng.uniform(0, 30 * 24, 80)), rng.gamma(2, 20, 80),
                               rng.choice(["a", "b", "c"], 80), rng.choice(["m1", "m2"], 80)):
            rows.append([card, float(hh), float(a), c, m, 40 + rng.random(), -75 + rng.random()])
    df = _events(rows).sort_values("unix_time", kind="stable").reset_index(drop=True)
    batch = compute_history(df, SPEC)
    store = EntityStateStore(SPEC)
    names = SPEC.feature_names()
    seq = pd.DataFrame([store.process(r) for r in df.to_dict("records")])[names].astype("float32")
    pd.testing.assert_frame_equal(seq, batch[names].reset_index(drop=True), check_exact=False, rtol=1e-4, atol=1e-4)
    # snapshot built from the frame == state left by the sequential replay, and it continues correctly
    snap = snapshot_from_frame(df.iloc[:100], SPEC)
    snap.save(tmp_path / "state.json")
    restored = EntityStateStore.load(tmp_path / "state.json")
    cont = pd.DataFrame([restored.process(r) for r in df.iloc[100:].to_dict("records")])[names].astype("float32")
    pd.testing.assert_frame_equal(cont, batch[names].iloc[100:].reset_index(drop=True), check_exact=False,
                                  rtol=1e-4, atol=1e-4)
