"""Inner training folds (Phase 1A.6) — for tuning/selection inside the training part only.

* temporal protocol → forward-chaining (expanding-window) folds cut at order-key values: fold k trains on
  everything up to cut k and validates on the next slice. Validation slices keep natural prevalence.
* Ma-2026 protocol → StratifiedKFold(3, shuffle, seed 42) as in the paper (prevalence preserved by
  stratification). Resampling, if any, is applied by the caller to the fold's TRAIN indices only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def forward_chaining_folds(order: np.ndarray | pd.Series, n_folds: int = 3, min_train_frac: float = 0.4
                           ) -> list[tuple[np.ndarray, np.ndarray]]:
    t = np.asarray(order, dtype=float)
    n = len(t)
    ts = np.sort(t)
    cuts = [float(ts[min(max(int(round(f * n)) - 1, 0), n - 1)])
            for f in np.linspace(min_train_frac, 1.0, n_folds + 1)]
    folds = []
    for k in range(n_folds):
        tr = np.flatnonzero(t <= cuts[k])
        va = np.flatnonzero((t > cuts[k]) & (t <= cuts[k + 1]))
        if len(tr) == 0 or len(va) == 0:
            raise ValueError("forward-chaining fold is empty (ties at the cut?)")
        folds.append((tr, va))
    return folds


def stratified_folds(y: np.ndarray | pd.Series, n_folds: int = 3, seed: int = 42
                     ) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]


def training_folds(protocol: str, order: np.ndarray | pd.Series, y: np.ndarray | pd.Series, n_folds: int = 3,
                   seed: int = 42) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    if protocol == "temporal":
        folds = forward_chaining_folds(order, n_folds)
        kind = "forward_chaining"
    else:
        folds = stratified_folds(y, n_folds, seed)
        kind = "stratified_kfold"
    y = np.asarray(y)
    meta = {"kind": kind, "n_folds": n_folds, "seed": seed if kind == "stratified_kfold" else None,
            "folds": [{"train_rows": int(len(tr)), "val_rows": int(len(va)),
                       "val_positives": int(y[va].sum()), "val_prevalence": float(y[va].mean())}
                      for tr, va in folds]}
    return folds, meta
