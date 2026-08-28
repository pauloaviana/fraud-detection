"""Class-imbalance treatment (Phase 1A.6).

Baseline = the natural class distribution. Weighting (class or example/cost-based) is the preferred,
information-preserving treatment; resampling methods are *experimental alternatives* that must only ever
touch a training fold. Nothing here assumes resampling helps — Ma et al. (2026) found no-resampling XGBoost
best on ULB (PR-AUC 0.809 vs 0.741 with SMOTE-ENN), and every application is recorded in metadata.

Hard invariants enforced here:
* ``resample_training_fold`` refuses anything not tagged as a training fold;
* validation / evaluation parts are never passed through it (prepare/experiments only hand it the train
  fold), and ``check_natural`` documents their prevalence against the raw slice;
* every treatment returns a metadata dict (method, params, seed, rows/positives/prevalence before/after).

Cost information for later transaction-level business evaluation (Correa Bahnsen et al. 2016, Table 3:
C_FP = C_TP = C_a, C_FN = Amount_i, C_TN = 0) is preserved via ``cost_context`` (row id, order key, raw
amount, label per part) and computed by ``CostMatrix``. ``example_weight`` is a TRAINING HEURISTIC derived
from that context (it re-weights rows by their cost components); it is not equivalent to optimising
Bahnsen's cost/savings objective. Final business evaluation must use the preserved raw cost context.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .contracts import DatasetContract, FeatureFamily

METHODS = ("none", "class_weight", "example_weight", "random_under", "random_over", "smote", "enn", "smote_enn")
WEIGHTING = {"class_weight", "example_weight"}
RESAMPLING = {"random_under", "random_over", "smote", "enn", "smote_enn"}


@dataclass(frozen=True)
class ImbalanceSpec:
    method: str = "none"
    class_weight: str | float = "balanced"     # "balanced" or explicit positive:negative weight ratio
    ca: float = 1.0                            # administrative cost C_a (example weighting / cost matrix)
    amount_floor: float = 1.0                  # minimum FN cost so zero-amount positives keep weight
    sampling_ratio: float = 1.0                # minority/majority ratio targeted by resampling
    k_neighbors: int = 5                       # SMOTE
    enn_neighbors: int = 3                     # ENN
    enn_classes: str = "majority"              # which classes may be removed: "majority" (imblearn ENN default
                                               # sampling_strategy='auto') or "all" (imblearn SMOTEENN)
    enn_kind_sel: str = "all"                  # imblearn kind_sel: "all" = keep only if every neighbour agrees
                                               # (imblearn default); "mode" = keep if the majority agrees
    seed: int = 42

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown imbalance method {self.method!r}; known: {METHODS}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------ weighting
def class_weights(y: np.ndarray | pd.Series, mode: str | float = "balanced") -> dict[int, float]:
    y = np.asarray(y)
    n, n1 = len(y), int((y == 1).sum())
    n0 = n - n1
    if mode == "balanced":                       # sklearn convention: n / (2 * n_c)
        return {0: n / (2 * n0), 1: n / (2 * n1)}
    return {0: 1.0, 1: float(mode)}


def sample_weights(y: np.ndarray | pd.Series, spec: ImbalanceSpec,
                   amount: np.ndarray | pd.Series | None = None) -> np.ndarray:
    """Per-row weights for weighting methods (mean 1). 'example_weight' is a training heuristic derived from
    the Bahnsen cost context: a positive weighs its false-negative cost (amount, floored), a negative its
    false-positive cost C_a. It does not optimise the cost/savings objective itself — evaluate that on the
    raw cost context."""
    y = np.asarray(y)
    if spec.method == "class_weight":
        cw = class_weights(y, spec.class_weight)
        w = np.where(y == 1, cw[1], cw[0]).astype(float)
    elif spec.method == "example_weight":
        if amount is None:
            raise ValueError("example_weight needs the raw transaction amount")
        a = np.maximum(np.asarray(amount, float), spec.amount_floor)
        w = np.where(y == 1, a, spec.ca).astype(float)
    else:
        w = np.ones(len(y))
    return w / w.mean()


# ------------------------------------------------------------------------------ resampling (training fold only)
def _prevalence(y: np.ndarray) -> float:
    return float(np.mean(y == 1)) if len(y) else float("nan")


def _numeric_matrix(X: pd.DataFrame, method: str) -> np.ndarray:
    if any(isinstance(X[c].dtype, pd.CategoricalDtype) for c in X.columns):
        raise ValueError(f"{method}: categorical columns present — apply to a numeric (linear) view, or use "
                         "weighting / random_under / random_over")
    arr = X.to_numpy(dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"{method}: NaN present — apply to an imputed (linear) view, or use weighting / "
                         "random_under / random_over")
    return arr


def _smote(arr: np.ndarray, y: np.ndarray, spec: ImbalanceSpec, rng: np.random.Generator):
    from sklearn.neighbors import NearestNeighbors
    pos = np.flatnonzero(y == 1)
    n_neg = int((y == 0).sum())
    n_new = int(round(spec.sampling_ratio * n_neg)) - len(pos)
    if n_new <= 0 or len(pos) < 2:
        return arr, y, 0
    k = min(spec.k_neighbors, len(pos) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(arr[pos])
    _, idx = nn.kneighbors(arr[pos])
    base = rng.integers(0, len(pos), n_new)
    nb = idx[base, rng.integers(1, k + 1, n_new)]
    u = rng.random((n_new, 1))
    synth = arr[pos[base]] + u * (arr[pos[nb]] - arr[pos[base]])
    return np.vstack([arr, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)]), n_new


def _enn(arr: np.ndarray, y: np.ndarray, spec: ImbalanceSpec, classes: str):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=spec.enn_neighbors + 1, n_jobs=-1).fit(arr)
    _, idx = nn.kneighbors(arr)
    neigh = y[idx[:, 1:]]
    if spec.enn_kind_sel == "mode":
        agree = (neigh == y[:, None]).sum(axis=1) * 2 > neigh.shape[1]     # strict majority agrees
    else:                                                                  # "all": every neighbour agrees
        agree = (neigh == y[:, None]).all(axis=1)
    disagree = ~agree
    if classes == "majority":
        maj = 0 if (y == 0).sum() >= (y == 1).sum() else 1
        disagree &= y == maj
    keep = ~disagree
    return arr[keep], y[keep], int((~keep).sum())


def resample_training_fold(X: pd.DataFrame, y: pd.Series | np.ndarray, spec: ImbalanceSpec, *,
                           fold_role: str = "train") -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Apply the experimental resampling in ``spec`` to ONE training fold. Refuses any other role.
    Returns (X_res, y_res, metadata). Weighting methods and 'none' return the fold unchanged."""
    if fold_role != "train":
        raise ValueError(f"resampling is only allowed on a training fold, got role {fold_role!r}")
    y = np.asarray(y).astype(np.int8)
    before = {"rows": int(len(y)), "positives": int(y.sum()), "prevalence": _prevalence(y)}
    meta: dict[str, Any] = {"method": spec.method, "params": spec.to_dict(), "fold_role": fold_role, "before": before}
    rng = np.random.default_rng(spec.seed)
    if spec.method in ("none", *WEIGHTING):
        meta["after"] = before
        meta["note"] = "no rows changed (weighting is applied through sample weights)"
        return X, y, meta
    if spec.method == "random_under":
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_keep = min(len(neg), int(round(len(pos) / spec.sampling_ratio)))
        keep = np.sort(np.concatenate([pos, rng.choice(neg, n_keep, replace=False)]))
        Xr, yr = X.iloc[keep], y[keep]
    elif spec.method == "random_over":
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_new = max(0, int(round(spec.sampling_ratio * len(neg))) - len(pos))
        extra = rng.choice(pos, n_new, replace=True)
        keep = np.concatenate([np.arange(len(y)), extra])
        Xr, yr = X.iloc[keep], y[keep]
    else:
        arr = _numeric_matrix(X, spec.method)
        removed = added = 0
        if spec.method in ("smote", "smote_enn"):
            arr, y2, added = _smote(arr, y, spec, rng)
        else:
            y2 = y
        if spec.method in ("enn", "smote_enn"):
            classes = spec.enn_classes if spec.method == "enn" else "all"     # imblearn SMOTEENN uses 'all'
            arr, y2, removed = _enn(arr, y2, spec, classes)
        Xr, yr = pd.DataFrame(arr, columns=X.columns), y2
        meta["synthetic_added"], meta["removed_by_enn"] = added, removed
    meta["after"] = {"rows": int(len(yr)), "positives": int(yr.sum()), "prevalence": _prevalence(yr)}
    return Xr, yr, meta


def check_natural(y_part: np.ndarray | pd.Series, y_raw_slice: np.ndarray | pd.Series, name: str) -> dict[str, Any]:
    """Evaluation parts must be the untouched raw slice: same rows, same positives."""
    yp, yr = np.asarray(y_part), np.asarray(y_raw_slice)
    ok = len(yp) == len(yr) and int(yp.sum()) == int(yr.sum())
    if not ok:
        raise RuntimeError(f"{name}: evaluation part does not match its natural slice")
    return {"part": name, "rows": int(len(yp)), "positives": int(yp.sum()), "prevalence": _prevalence(yp), "natural": True}


# ------------------------------------------------------------------------------ business-cost context
def amount_column(contract: DatasetContract) -> str:
    claim = contract.claim(FeatureFamily.COST_SENSITIVE_EVAL)
    if not claim or not claim.basis:
        raise ValueError(f"{contract.name}: no amount column for cost evaluation")
    return claim.basis[0]


def cost_context(df: pd.DataFrame, contract: DatasetContract, y: pd.Series) -> pd.DataFrame:
    """Per-row information needed for transaction-level cost evaluation: row id (if any), order key,
    raw amount, label. Kept separately from the feature frame (which may transform or drop the amount)."""
    cols = {}
    if contract.row_id and contract.row_id in df.columns:
        cols["row_id"] = df[contract.row_id].to_numpy()
    cols["order"] = df[contract.order_key].to_numpy()
    cols["amount"] = df[amount_column(contract)].to_numpy(dtype=float)
    cols["y"] = np.asarray(y, dtype=np.int8)
    return pd.DataFrame(cols, index=df.index)


@dataclass(frozen=True)
class CostMatrix:
    """Correa Bahnsen et al. (2016), Table 3: C_TP = C_FP = C_a, C_FN = Amount_i, C_TN = 0."""

    ca: float = 1.0

    def costs(self, y, pred, amount) -> np.ndarray:
        y, p, a = np.asarray(y), np.asarray(pred), np.asarray(amount, float)
        return np.where(p == 1, self.ca, np.where(y == 1, a, 0.0))

    def total_cost(self, y, pred, amount) -> float:                       # eq. (1)
        return float(self.costs(y, pred, amount).sum())

    def cost_no_model(self, y, amount) -> float:                           # Cost_l: accept everything
        return float(np.asarray(amount, float)[np.asarray(y) == 1].sum())

    def savings(self, y, pred, amount) -> float:                           # eq. (5)
        base = self.cost_no_model(y, amount)
        return float((base - self.total_cost(y, pred, amount)) / base) if base > 0 else float("nan")


# ------------------------------------------------------------------------------ experiment metadata
@dataclass
class ExperimentConfig:
    dataset: str
    protocol: str
    view: str                                   # "tree" | "linear"
    imbalance: ImbalanceSpec = field(default_factory=ImbalanceSpec)
    selection: str = "none"                     # "none" | "rf_gini_refit" | "ma2026_published"
    folds: dict[str, Any] = field(default_factory=dict)
    cost_ca: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["frauddet_version"] = __version__
        return d

    def save(self, path: str | Path) -> Path:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))
        return Path(path)


# ------------------------------------------------------------------------------ CLI: fold-internal resampling demo
def main(argv: list[str] | None = None) -> int:
    """Apply an imbalance treatment on the saved artifacts of a dataset/protocol — to the training part and
    to each inner training fold only — and write the metadata (imbalance-<method>.json). Evaluation parts and
    inner validation folds are reported untouched. No model is trained.

        python -m frauddet.imbalance --dataset ulb --protocol stratified_ma2026 --method smote_enn
    """
    import argparse

    from .adapters import get_adapter
    from .folds import training_folds
    from .prepare import prepare_frame
    from .preprocessing import Pipeline
    from .views import ModelView

    p = argparse.ArgumentParser(description="Leakage-safe imbalance treatment on saved artifacts (Phase 1A.6)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", default="temporal")
    p.add_argument("--method", default="none", choices=METHODS)
    p.add_argument("--view", default="linear", choices=["tree", "linear"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--n-folds", type=int, default=3)
    a = p.parse_args(argv)
    adapter = get_adapter(a.dataset, a.data_dir)
    c = adapter.contract
    out = Path(a.artifacts) / a.dataset / a.protocol
    df, _ = prepare_frame(adapter, a.protocol)
    if c.entity_key:
        from .history import HistorySpec, compute_history
        df = compute_history(df, HistorySpec(entity=c.entity_key, order=c.order_key, event_time=c.event_time))
    member = pd.read_csv(out / "membership.csv")
    parts = {n: member.loc[member["part"] == n, "row"].to_numpy() for n in member["part"].unique()}
    first = list(parts)[0]
    from .labels import TARGETS
    y = TARGETS[c.name].binary(df).to_numpy()
    pipe = Pipeline.load(out / "features.json")
    view = ModelView.load(out / f"view-{a.view}.json")
    spec = ImbalanceSpec(a.method)
    tr_idx = parts[first]
    X_train = view.transform(pipe.transform(df.iloc[tr_idx]))
    y_train = y[tr_idx]
    amount = df[amount_column(c)].to_numpy(float)[tr_idx]
    report: dict[str, Any] = {"frauddet_version": __version__, "dataset": c.name, "protocol": a.protocol,
                              "view": a.view, "spec": spec.to_dict()}
    _, _, meta = resample_training_fold(X_train, y_train, spec)
    report["training_part"] = meta
    if spec.method in WEIGHTING:
        w = sample_weights(y_train, spec, amount)
        report["training_part"]["weight_stats"] = {"mean": float(w.mean()), "pos_mean": float(w[y_train == 1].mean()),
                                                   "neg_mean": float(w[y_train == 0].mean())}
    folds, fmeta = training_folds(a.protocol, df[c.order_key].to_numpy()[tr_idx], y_train, a.n_folds)
    report["inner_folds"] = {"kind": fmeta["kind"], "folds": []}
    for k, (ftr, fva) in enumerate(folds):
        _, _, m = resample_training_fold(X_train.iloc[ftr], y_train[ftr], spec)
        report["inner_folds"]["folds"].append({"fold": k, "train": m["before"], "train_after": m["after"],
                                               "val_untouched": {"rows": int(len(fva)), "positives": int(y_train[fva].sum()),
                                                                 "prevalence": _prevalence(y_train[fva])}})
    report["evaluation_parts_untouched"] = {n: {"rows": int(len(idx)), "positives": int(y[idx].sum()),
                                               "prevalence": _prevalence(y[idx])} for n, idx in parts.items() if n != first}
    path = out / f"imbalance-{a.method}.json"
    path.write_text(json.dumps(report, indent=1))
    tp = report["training_part"]
    print(f"[imbalance] {c.name}/{a.protocol} {a.method} on {a.view} view: training part {tp['before']['rows']}/{tp['before']['positives']} "
          f"→ {tp['after']['rows']}/{tp['after']['positives']}; evaluation parts untouched → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
