"""Evaluation metrics for rare-event fraud scoring (Phase 1B).

Three layers, kept apart:
* discrimination — threshold-free ranking quality (PR-AUC primary, ROC-AUC, recall at constrained false-
  positive rates, precision at alert budgets);
* operating point — everything that follows from ONE threshold (precision, recall, F1, MCC, confusion
  counts, alert rate, FP per 10k);
* calibration — Brier, log loss, expected calibration error, reliability table;
* business — Correa Bahnsen cost / savings from the preserved cost context (raw amounts), plus the
  fraud-vs-friction view (fraud amount caught vs legitimate customers declined).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, matthews_corrcoef,
                             roc_auc_score, roc_curve)

from .imbalance import CostMatrix


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ------------------------------------------------------------------------------ discrimination
def recall_at_fpr(y: np.ndarray, p: np.ndarray, fpr_max: float) -> tuple[float, float]:
    """Highest recall achievable with FPR <= fpr_max, and the threshold that attains it."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = fpr <= fpr_max
    if not ok.any():
        return 0.0, float("inf")
    i = int(np.flatnonzero(ok)[np.argmax(tpr[ok])])
    return float(tpr[i]), float(thr[i])


def precision_at_budget(y: np.ndarray, p: np.ndarray, alert_rate: float) -> tuple[float, float, float]:
    """Precision and recall when the top ``alert_rate`` fraction of transactions is alerted."""
    n = len(p)
    k = max(1, int(round(alert_rate * n)))
    order = np.argsort(-p, kind="stable")
    top = order[:k]
    tp = float(y[top].sum())
    return tp / k, tp / max(y.sum(), 1), float(p[order[k - 1]])


def discrimination(y, p, fprs=(0.001, 0.005, 0.01), budgets=(0.001, 0.005, 0.01)) -> dict[str, Any]:
    y, p = _np(y).astype(int), _np(p)
    out: dict[str, Any] = {"pr_auc": float(average_precision_score(y, p)),
                           "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
                           "positives": int(y.sum()), "n": int(len(y)), "prevalence": float(y.mean())}
    for f in fprs:
        r, t = recall_at_fpr(y, p, f)
        out[f"recall_at_fpr_{f:g}"] = r
    for b in budgets:
        pr, rc, _ = precision_at_budget(y, p, b)
        out[f"precision_at_alert_{b:g}"] = pr
        out[f"recall_at_alert_{b:g}"] = rc
    return out


# ------------------------------------------------------------------------------ operating point
def at_threshold(y, p, thr: float) -> dict[str, Any]:
    y, p = _np(y).astype(int), _np(p)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": f1, "mcc": float(matthews_corrcoef(y, pred)) if tp + fp and tp + fn else 0.0,
            "balanced_accuracy": (rec + spec) / 2, "specificity": spec,
            "alert_rate": float(pred.mean()), "fpr": fp / max(tn + fp, 1),
            "fp_per_10k": 1e4 * fp / len(y), "alerts_per_fraud_caught": (tp + fp) / tp if tp else float("inf")}


# ------------------------------------------------------------------------------ calibration
def calibration(y, p, n_bins: int = 10) -> dict[str, Any]:
    y, p = _np(y).astype(int), np.clip(_np(p), 1e-7, 1 - 1e-7)
    # equal-frequency bins (rare events: equal-width bins are almost empty above the base rate)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    table, ece = [], 0.0
    for b in bins:
        if len(b) == 0:
            continue
        conf, acc = float(p[b].mean()), float(y[b].mean())
        ece += len(b) / len(p) * abs(conf - acc)
        table.append({"n": int(len(b)), "mean_pred": conf, "observed": acc, "p_min": float(p[b].min()), "p_max": float(p[b].max())})
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)), "ece": float(ece),
            "mean_pred": float(p.mean()), "observed_rate": float(y.mean()), "reliability": table}


# ------------------------------------------------------------------------------ business
def business(y, p, thr: float, amount, ca: float = 1.0) -> dict[str, Any]:
    y, p, a = _np(y).astype(int), _np(p), _np(amount)
    pred = (p >= thr).astype(int)
    cm = CostMatrix(ca)
    caught = a[(pred == 1) & (y == 1)].sum(); missed = a[(pred == 0) & (y == 1)].sum()
    total_fraud = a[y == 1].sum()
    return {"threshold": float(thr), "ca": ca,
            "cost": cm.total_cost(y, pred, a), "cost_no_model": cm.cost_no_model(y, a), "savings": cm.savings(y, pred, a),
            "fraud_amount_total": float(total_fraud), "fraud_amount_caught": float(caught), "fraud_amount_missed": float(missed),
            "amount_recall": float(caught / total_fraud) if total_fraud else float("nan"),
            "legit_declined": int(((pred == 1) & (y == 0)).sum()), "legit_amount_declined": float(a[(pred == 1) & (y == 0)].sum()),
            "friction_rate": float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))}


def full_evaluation(y, p, thresholds: dict[str, float], amount=None, ca: float = 1.0) -> dict[str, Any]:
    out = {"discrimination": discrimination(y, p), "calibration": calibration(y, p), "operating_points": {}}
    for name, thr in thresholds.items():
        op = at_threshold(y, p, thr)
        if amount is not None:
            op["business"] = business(y, p, thr, amount, ca)
        out["operating_points"][name] = op
    return out
