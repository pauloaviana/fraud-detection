"""Leakage-safe evaluation splits (Phase 1A.4).

Primary protocol: TEMPORAL. Rows are ordered by the contract's order key and cut
at *time values* (not row positions), so rows sharing a timestamp never straddle a
boundary; everything at or before a cut belongs to the earlier part. Parts are
train / val / holdout. Evaluation parts keep their natural prevalence — nothing
here resamples or stratifies the temporal protocol. An optional embargo gap after
each cut is supported but is a separate decision (default 0).

Secondary protocol (ULB only): the Ma et al. 2026 reproducibility benchmark —
stratified 70:30, seed 42, on the raw (non-deduplicated) rows, as in the paper.

Split membership is positional into the *prepared* frame (see prepare.py), and
the split object records its spec, cut values and a per-part summary so it can be
serialised next to the fitted pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .contracts import DatasetContract

TEMPORAL = "temporal"
MA2026 = "stratified_ma2026"


@dataclass(frozen=True)
class SplitSpec:
    dataset: str
    protocol: str
    order_key: str | None
    target: str | None
    part_names: tuple[str, ...]
    fractions: tuple[float, ...] | None = None   # temporal
    gap_seconds: float = 0.0                     # temporal embargo after each cut (separate question; 0)
    seed: int | None = None                      # stratified
    test_size: float | None = None               # stratified
    note: str = ""


@dataclass
class Split:
    spec: SplitSpec
    parts: dict[str, np.ndarray]        # part -> positional indices into the prepared frame (chronological)
    boundaries: dict[str, float]        # temporal: part -> inclusive upper cut on the order key
    summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    def frame(self, df: pd.DataFrame, part: str) -> pd.DataFrame:
        return df.iloc[self.parts[part]]

    def to_dict(self) -> dict[str, Any]:
        return {"spec": asdict(self.spec), "boundaries": self.boundaries, "summary": self.summary,
                "sizes": {k: int(len(v)) for k, v in self.parts.items()}}


# ------------------------------------------------------------------------------ temporal
def temporal_split(df: pd.DataFrame, contract: DatasetContract,
                   fractions: tuple[float, ...] = (0.70, 0.15, 0.15),
                   part_names: tuple[str, ...] = ("train", "val", "holdout"),
                   gap_seconds: float = 0.0) -> Split:
    if contract.order_key is None:
        raise ValueError(f"{contract.name}: no order key — temporal split impossible")
    if len(fractions) != len(part_names) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must match part_names and sum to 1")
    t = df[contract.order_key].to_numpy()
    if np.isnan(t.astype(float)).any():
        raise ValueError("order key has nulls")
    n = len(df)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    cuts: list[float] = []
    cum = 0.0
    for f in fractions[:-1]:
        cum += f
        pos = min(max(int(round(cum * n)) - 1, 0), n - 1)
        cuts.append(float(ts[pos]))            # cut at the time VALUE — ties stay together
    parts: dict[str, np.ndarray] = {}
    boundaries: dict[str, float] = {}
    lower = -np.inf
    for i, name in enumerate(part_names):
        upper = cuts[i] if i < len(cuts) else np.inf
        mask = (t <= upper) & (t > lower)
        if i > 0 and gap_seconds > 0:
            mask &= t > lower + gap_seconds
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise ValueError(f"temporal split: part {name!r} is empty (massive ties at the cut?)")
        parts[name] = idx[np.argsort(t[idx], kind="stable")]
        boundaries[name] = upper
        lower = upper
    spec = SplitSpec(contract.name, TEMPORAL, contract.order_key, contract.target, tuple(part_names),
                     fractions=tuple(fractions), gap_seconds=gap_seconds,
                     note="cut at order-key values; rows at a cut belong to the earlier part; evaluation parts "
                          "keep natural prevalence")
    split = Split(spec, parts, boundaries)
    split.summary = summarize(split, df)
    return split


# ------------------------------------------------------------------------------ Ma 2026 benchmark
def stratified_split_ma2026(df: pd.DataFrame, contract: DatasetContract, seed: int = 42,
                            test_size: float = 0.30) -> Split:
    """Ma et al. 2026 protocol: stratified 70:30 on the raw rows (no dedup), seed 42.
    Reproducibility benchmark only — NOT the primary protocol (chronology is ignored)."""
    y = df[contract.target].to_numpy()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    spec = SplitSpec(contract.name, MA2026, contract.order_key, contract.target, ("train", "test"),
                     seed=seed, test_size=test_size,
                     note="Ma et al. 2026 reproducibility benchmark: stratified, seed 42, raw rows; "
                          "prevalence preserved by stratification; ignores chronology by design")
    split = Split(spec, {"train": np.sort(tr), "test": np.sort(te)}, {})
    split.summary = summarize(split, df)
    return split


# ------------------------------------------------------------------------------ summary / invariants
def summarize(split: Split, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    tgt, ok = split.spec.target, split.spec.order_key
    total_pos = float(df[tgt].sum()) if tgt and tgt in df else None
    for name, idx in split.parts.items():
        part = df.iloc[idx]
        d: dict[str, Any] = {"rows": int(len(idx)), "fraction": round(len(idx) / len(df), 4)}
        if tgt and tgt in df:
            pos = int(part[tgt].sum())
            d.update(positives=pos, prevalence=round(pos / len(idx), 6),
                     share_of_positives=round(pos / total_pos, 4) if total_pos else None)
        if ok and ok in df:
            d.update(order_min=float(part[ok].min()), order_max=float(part[ok].max()))
        out[name] = d
    return out


def verify(split: Split, df: pd.DataFrame) -> list[str]:
    """Hard invariants. Returns a list of violations (empty = OK)."""
    v: list[str] = []
    names = list(split.parts)
    all_idx = np.concatenate([split.parts[n] for n in names])
    if len(np.unique(all_idx)) != len(all_idx):
        v.append("parts overlap")
    if all_idx.min() < 0 or all_idx.max() >= len(df):
        v.append("indices out of range")
    for n in names:
        if len(split.parts[n]) == 0:
            v.append(f"part {n} empty")
    if split.spec.protocol == TEMPORAL:
        ok = split.spec.order_key
        t = df[ok].to_numpy()
        for a, b in zip(names, names[1:]):
            ta, tb = t[split.parts[a]], t[split.parts[b]]
            if not ta.max() < tb.min():
                v.append(f"chronology violated between {a} and {b}")
            if split.spec.gap_seconds and tb.min() - ta.max() < split.spec.gap_seconds:
                v.append(f"gap violated between {a} and {b}")
        # natural prevalence: no resampling means the part is exactly a contiguous time slice
        for n in names:
            lo, hi = t[split.parts[n]].min(), t[split.parts[n]].max()
            expect = int(((t >= lo) & (t <= hi)).sum())
            if expect != len(split.parts[n]):
                v.append(f"part {n} is not a full contiguous time slice (rows dropped/resampled)")
    else:
        tgt = split.spec.target
        base = df[tgt].mean()
        for n in names:
            p = df.iloc[split.parts[n]][tgt].mean()
            if abs(p - base) > 0.05 * base + 1e-6:
                v.append(f"part {n}: prevalence {p:.5f} deviates from base {base:.5f}")
    return v
