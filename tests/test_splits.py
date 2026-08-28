"""Unit tests for the split protocols (synthetic frames)."""

import numpy as np
import pandas as pd
import pytest

from frauddet.adapters import ADAPTERS
from frauddet.splits import MA2026, TEMPORAL, stratified_split_ma2026, temporal_split, verify

ULB = ADAPTERS["ulb"].contract


def _frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 500, n)).astype(float)         # heavy ties like ULB
    y = (rng.random(n) < 0.05).astype(int)
    df = pd.DataFrame({"Time": t, "Amount": rng.random(n), "Class": y})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n)
    return df.sample(frac=1, random_state=1).reset_index(drop=True)     # shuffled on purpose


def test_temporal_split_is_chronological_tie_safe_and_natural():
    df = _frame()
    sp = temporal_split(df, ULB, (0.7, 0.15, 0.15))
    assert verify(sp, df) == []
    t = df["Time"].to_numpy()
    for a, b in (("train", "val"), ("val", "holdout")):
        assert t[sp.parts[a]].max() < t[sp.parts[b]].min()          # strict: no shared second
    assert sum(len(v) for v in sp.parts.values()) == len(df)          # nothing dropped
    for name, idx in sp.parts.items():                                 # natural prevalence = raw slice
        assert sp.summary[name]["positives"] == int(df.iloc[idx]["Class"].sum())
        assert np.all(np.diff(t[idx]) >= 0)                            # chronological inside the part
    assert abs(sp.summary["train"]["fraction"] - 0.7) < 0.05


def test_temporal_split_gap_is_enforced():
    df = _frame()
    sp = temporal_split(df, ULB, (0.7, 0.3), ("train", "val"), gap_seconds=20)
    t = df["Time"].to_numpy()
    assert t[sp.parts["val"]].min() - t[sp.parts["train"]].max() > 20
    assert sp.spec.gap_seconds == 20
    v = verify(sp, df)
    assert all("gap" not in x for x in v)


def test_temporal_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        temporal_split(_frame(), ULB, (0.5, 0.4))


def test_ma2026_split_preserves_prevalence_and_is_reproducible():
    df = _frame(5000)
    a = stratified_split_ma2026(df, ULB)
    b = stratified_split_ma2026(df, ULB)
    assert np.array_equal(a.parts["test"], b.parts["test"])
    assert a.spec.protocol == MA2026 and a.spec.seed == 42
    assert verify(a, df) == []
    base = df["Class"].mean()
    assert abs(a.summary["test"]["prevalence"] - base) < 0.01


def test_split_serialises():
    sp = temporal_split(_frame(), ULB)
    d = sp.to_dict()
    assert d["spec"]["protocol"] == TEMPORAL and set(d["sizes"]) == {"train", "val", "holdout"}
    assert "train" in d["boundaries"]
