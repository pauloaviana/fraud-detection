"""Decision policies (Phase 1B): the approve/decline layer, separate from the probability layer.

A policy turns calibrated scores into a threshold using TRAINING/VALIDATION information only:
* f1_max        — threshold maximising fraud-class F1 (Ma et al. 2026 use this on OOF predictions);
* fpr_budget    — highest recall subject to FPR <= budget (customer-friction constraint);
* alert_budget  — alert the top fraction of transactions (analyst capacity constraint);
* cost_optimal  — minimise Correa Bahnsen cost with C_a (business objective on the raw cost context).

``select_thresholds`` returns the thresholds and the operating statistics *on the data they were selected
from*; ``evaluate`` applies them elsewhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .imbalance import CostMatrix
from .metrics import at_threshold, business


def _candidates(p: np.ndarray, max_points: int = 2000) -> np.ndarray:
    q = np.unique(np.quantile(p, np.linspace(0, 1, max_points)))
    return np.unique(np.concatenate([q, [np.inf]]))


def f1_max(y, p) -> float:
    best, thr = -1.0, np.inf
    for t in _candidates(p):
        f = at_threshold(y, p, t)["f1"]
        if f > best:
            best, thr = f, t
    return float(thr)


def fpr_budget(y, p, fpr_max: float) -> float:
    from .metrics import recall_at_fpr
    return recall_at_fpr(np.asarray(y).astype(int), np.asarray(p, float), fpr_max)[1]


def alert_budget(p, rate: float) -> float:
    p = np.asarray(p, float)
    k = max(1, int(round(rate * len(p))))
    return float(np.sort(p)[::-1][k - 1])


def cost_optimal(y, p, amount, ca: float) -> float:
    y, p, a = np.asarray(y).astype(int), np.asarray(p, float), np.asarray(amount, float)
    cm = CostMatrix(ca)
    best, thr = np.inf, np.inf
    for t in _candidates(p):
        c = cm.total_cost(y, (p >= t).astype(int), a)
        if c < best:
            best, thr = c, t
    return float(thr)


def select_thresholds(y, p, amount=None, ca: float = 1.0, fprs=(0.001, 0.005), alerts=(0.005, 0.01)) -> dict[str, Any]:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    thr: dict[str, float] = {"f1_max": f1_max(y, p)}
    for f in fprs:
        thr[f"fpr_{f:g}"] = fpr_budget(y, p, f)
    for r in alerts:
        thr[f"alert_{r:g}"] = alert_budget(p, r)
    if amount is not None:
        thr[f"cost_ca{ca:g}"] = cost_optimal(y, p, amount, ca)
    stats = {}
    for name, t in thr.items():
        s = at_threshold(y, p, t)
        if amount is not None:
            s["business"] = business(y, p, t, amount, ca)
        stats[name] = s
    return {"thresholds": thr, "selected_on": {"n": int(len(y)), "positives": int(y.sum())}, "stats": stats}
