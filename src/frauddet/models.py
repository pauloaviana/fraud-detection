"""Model factory (Phase 1B): dummy prior, logistic regression, XGBoost, LightGBM.

Each ``Model`` has the same surface — ``fit(X, y, sample_weight, eval_set)``, ``predict_proba(X)``,
``save``/``load``, ``size_bytes``, ``importance`` — and declares which 1A model view it consumes
(linear for LR, tree for GBDTs with native NaN and categorical handling). Hyperparameter search spaces
are small, seeded, and deliberately conventional; early stopping uses the inner-fold validation slice
(training data) and never the validation part or the holdout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_VIEW = {"dummy": "tree", "logreg": "linear", "xgboost": "tree", "lightgbm": "tree"}
EARLY_STOP = 50
MAX_ROUNDS = 2000
SMALL_BATCH = 256


def search_space(name: str, n: int, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if name == "dummy":
        return [{}]
    if name == "logreg":
        return [{"C": c} for c in (0.01, 0.1, 1.0)][:max(n, 1)]
    if name == "xgboost":
        grid = {"learning_rate": [0.05, 0.1], "max_depth": [4, 6, 8], "min_child_weight": [1, 5],
                "subsample": [0.8], "colsample_bytree": [0.6, 0.8], "reg_lambda": [1.0, 5.0]}
    elif name == "lightgbm":
        grid = {"learning_rate": [0.03, 0.06], "num_leaves": [31, 63, 127], "min_child_samples": [20, 100],
                "feature_fraction": [0.6, 0.8], "bagging_fraction": [0.8], "lambda_l2": [0.0, 5.0]}
    else:
        raise ValueError(name)
    keys = list(grid)
    seen, out = set(), []
    while len(out) < n:
        cfg = {k: rng.choice(grid[k]) for k in keys}
        key = tuple(cfg[k] for k in keys)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


class Model:
    def __init__(self, name: str, params: dict[str, Any] | None = None, seed: int = 42, n_jobs: int = 6):
        if name not in MODEL_VIEW:
            raise ValueError(name)
        self.name, self.params, self.seed, self.n_jobs = name, dict(params or {}), seed, n_jobs
        self.view = MODEL_VIEW[name]
        self.best_iteration: int | None = None
        self._est = None
        self._columns: list[str] = []
        self.meta: dict[str, Any] = {}

    # -- fit ---------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y, sample_weight=None, eval_set: tuple | None = None,
            n_estimators: int | None = None) -> "Model":
        self._columns = list(X.columns)
        y = np.asarray(y).astype(int)
        if self.name == "dummy":
            self.meta = {"prior": float(np.average(y, weights=sample_weight))}
            return self
        if self.name == "logreg":
            from sklearn.linear_model import LogisticRegression
            est = LogisticRegression(C=self.params.get("C", 1.0), max_iter=self.params.get("max_iter", 300),
                                     solver="lbfgs")
            est.fit(X.to_numpy(dtype=np.float32), y, sample_weight=sample_weight)
            self._est = est
            self.meta = {"n_iter": int(est.n_iter_[0])}
            return self
        rounds = n_estimators or MAX_ROUNDS
        if self.name == "xgboost":
            import xgboost as xgb
            kw = dict(tree_method="hist", enable_categorical=True, max_bin=256, n_estimators=rounds,
                      random_state=self.seed, n_jobs=self.n_jobs, eval_metric="aucpr", **self.params)
            if eval_set is not None:
                kw["early_stopping_rounds"] = EARLY_STOP
            est = xgb.XGBClassifier(**kw)
            est.fit(X, y, sample_weight=sample_weight,
                    eval_set=[eval_set] if eval_set is not None else None, verbose=False)
            self._est = est
            self.best_iteration = int(est.best_iteration) + 1 if eval_set is not None else rounds
        else:
            import lightgbm as lgb
            est = lgb.LGBMClassifier(objective="binary", n_estimators=rounds, random_state=self.seed,
                                     n_jobs=self.n_jobs, verbose=-1, bagging_freq=1, **self.params)
            cb = [lgb.early_stopping(EARLY_STOP, verbose=False)] if eval_set is not None else []
            est.fit(X, y, sample_weight=sample_weight, eval_set=[eval_set] if eval_set is not None else None,
                    eval_metric="average_precision", callbacks=cb)
            self._est = est
            self.best_iteration = int(est.best_iteration_) if eval_set is not None and est.best_iteration_ else rounds
        self.meta = {"best_iteration": self.best_iteration}
        return self

    # -- predict -----------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.name == "dummy":
            return np.full(len(X), self.meta["prior"], dtype=float)
        X = X[self._columns]
        if self.name == "logreg":
            return self._est.predict_proba(X.to_numpy(dtype=np.float32))[:, 1]
        if self.name == "xgboost":
            it = (0, self.best_iteration) if self.best_iteration else None
            return self._est.predict_proba(X, iteration_range=it)[:, 1]
        # small batches (online scoring): a single thread avoids LightGBM's OpenMP team start-up (~7 ms/row)
        kw = {"num_threads": 1} if len(X) <= SMALL_BATCH else {}
        return self._est.predict_proba(X, num_iteration=self.best_iteration, **kw)[:, 1]

    # -- diagnostics -------------------------------------------------------------------
    def importance(self) -> dict[str, float]:
        if self.name == "logreg":
            return dict(zip(self._columns, map(float, np.abs(self._est.coef_[0]))))
        if self.name == "xgboost":
            sc = self._est.get_booster().get_score(importance_type="gain")
            return {c: float(sc.get(c, 0.0)) for c in self._columns}
        if self.name == "lightgbm":
            return dict(zip(self._columns, map(float, self._est.booster_.feature_importance("gain"))))
        return {}

    # -- io ----------------------------------------------------------------------------
    def save(self, directory: str | Path) -> dict[str, Any]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.name == "xgboost":
            self._est.get_booster().save_model(d / "model.xgb.json")
            files = ["model.xgb.json"]
        elif self.name == "lightgbm":
            self._est.booster_.save_model(d / "model.lgbm.txt", num_iteration=self.best_iteration)
            files = ["model.lgbm.txt"]
        elif self.name == "logreg":
            (d / "model.logreg.json").write_text(json.dumps({"coef": self._est.coef_[0].tolist(),
                                                              "intercept": float(self._est.intercept_[0]),
                                                              "classes": self._est.classes_.tolist()}))
            files = ["model.logreg.json"]
        else:
            (d / "model.dummy.json").write_text(json.dumps(self.meta))
            files = ["model.dummy.json"]
        spec = {"name": self.name, "params": self.params, "seed": self.seed, "columns": self._columns,
                "best_iteration": self.best_iteration, "meta": self.meta, "files": files, "view": self.view}
        (d / "model.json").write_text(json.dumps(spec))
        return spec

    @property
    def size_bytes(self) -> int:
        if self.name == "xgboost":
            raw = self._est.get_booster().save_raw("json")
            return len(raw)
        if self.name == "lightgbm":
            return len(self._est.booster_.model_to_string(num_iteration=self.best_iteration).encode())
        if self.name == "logreg":
            return int(self._est.coef_.nbytes + 8)
        return 8

    @classmethod
    def load(cls, directory: str | Path) -> "Model":
        d = Path(directory)
        spec = json.loads((d / "model.json").read_text())
        m = cls(spec["name"], spec["params"], spec["seed"])
        m._columns, m.best_iteration, m.meta = spec["columns"], spec["best_iteration"], spec["meta"]
        if m.name == "xgboost":
            import xgboost as xgb
            est = xgb.XGBClassifier(enable_categorical=True, tree_method="hist")
            est.load_model(d / "model.xgb.json")
            m._est = est
        elif m.name == "lightgbm":
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(d / "model.lgbm.txt"))
            m._est = _LGBWrap(booster)
        elif m.name == "logreg":
            from sklearn.linear_model import LogisticRegression
            s = json.loads((d / "model.logreg.json").read_text())
            est = LogisticRegression()
            est.coef_ = np.array([s["coef"]]); est.intercept_ = np.array([s["intercept"]])
            est.classes_ = np.array(s["classes"])
            m._est = est
        return m


class _LGBWrap:
    """Minimal predict_proba shim around a loaded LightGBM Booster."""

    def __init__(self, booster):
        self.booster_ = booster

    def predict_proba(self, X, num_iteration=None, **kw):
        p = self.booster_.predict(X, num_iteration=num_iteration, **kw)
        return np.column_stack([1 - p, p])
