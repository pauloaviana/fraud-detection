"""Model-facing views and optional feature selection (Phase 1A.5).

The feature layer (preprocessing.Pipeline) emits a *typed* frame: numeric columns with NaN preserved
and categorical columns as pandas ``category`` with a fixed, train-learned category set. Whether NaN
is imputed, whether categoricals are one-hot or native, and whether numerics are scaled is a property
of the model family, not of the data — so it lives here, fitted on the training part only:

* ``ModelView("tree")``   — numeric passthrough (NaN kept for XGBoost/LightGBM native handling);
                            categoricals stay ``category`` (native categorical support) — no ordinal
                            meaning is ever attached to the codes.
* ``ModelView("linear")`` — train-median imputation, one-hot for categories seen ≥ ``min_count`` times in
                            training (others → ``<RARE>``), standardisation with train moments; no NaN.
* ``RFGiniSelector``      — Random-Forest Gini-importance top-k (Ma et al. 2026), fitted on the training
                            part only; a selector, not a final model.

Everything serialises to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__

UNK, NA, RARE = "<UNK>", "<NA>", "<RARE>"


def _is_cat(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype)


class ModelView:
    def __init__(self, kind: str, min_count: int = 50):
        if kind not in ("tree", "linear"):
            raise ValueError(kind)
        self.kind, self.min_count = kind, min_count
        self.state: dict[str, Any] = {}

    # -- fit ------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "ModelView":
        cat_cols = [c for c in X.columns if _is_cat(X[c])]
        num_cols = [c for c in X.columns if c not in cat_cols]
        st: dict[str, Any] = {"kind": self.kind, "columns": list(X.columns), "categorical": {}, "numeric": num_cols}
        for c in cat_cols:
            st["categorical"][c] = list(map(str, X[c].cat.categories))
        if self.kind == "linear":
            medians, moments = {}, {}
            for c in num_cols:                       # column-wise: never materialise the frame in float64
                col = X[c].to_numpy(dtype=np.float32, copy=True)
                m = np.nanmedian(col) if np.isnan(col).any() else np.median(col)
                m = float(m) if np.isfinite(m) else 0.0
                np.nan_to_num(col, nan=m, copy=False)
                sd = float(col.std())
                medians[c] = m
                moments[c] = {"mean": float(col.mean()), "std": sd if sd > 0 else 1.0}
            st["medians"], st["moments"] = medians, moments
            st["onehot"] = {}
            for c in cat_cols:
                vc = X[c].astype("string").value_counts()
                keep = [str(v) for v, n in vc.items() if n >= self.min_count]
                st["onehot"][c] = keep
            st["output_columns"] = num_cols + [f"{c}={v}" for c in cat_cols for v in st["onehot"][c] + [RARE]]
        else:
            st["output_columns"] = list(X.columns)
        self.state = st
        return self

    # -- transform ----------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        st = self.state
        if not st:
            raise RuntimeError("view not fitted")
        missing = [c for c in st["columns"] if c not in X.columns]
        if missing:
            raise KeyError(f"frame lacks columns {missing[:8]}")
        if self.kind == "tree":
            cols = {}
            for c in st["columns"]:
                if c in st["categorical"]:
                    cols[c] = pd.Categorical(X[c].astype("string").fillna(NA), categories=st["categorical"][c])
                else:
                    cols[c] = X[c].to_numpy(dtype=np.float32)
            return pd.DataFrame(cols, index=X.index)
        num_cols = st["numeric"]
        arr = X[num_cols].to_numpy(dtype=np.float32)                       # one float32 block
        med = np.array([st["medians"][c] for c in num_cols], dtype=np.float32)
        mean = np.array([st["moments"][c]["mean"] for c in num_cols], dtype=np.float32)
        std = np.array([st["moments"][c]["std"] for c in num_cols], dtype=np.float32)
        nan = np.isnan(arr)
        if nan.any():
            arr = np.where(nan, med, arr)
        arr -= mean
        arr /= std
        blocks = [pd.DataFrame(arr, index=X.index, columns=num_cols)]
        for c, keep in st["onehot"].items():
            s = X[c].astype("string").fillna(NA)
            s = s.where(s.isin(keep), RARE)
            oh = pd.DataFrame({f"{c}={v}": (s == v).astype("int8") for v in keep + [RARE]}, index=X.index)
            blocks.append(oh)
        out = pd.concat(blocks, axis=1)
        return out[st["output_columns"]]

    # -- io -----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"frauddet_version": __version__, "kind": self.kind, "min_count": self.min_count, "state": self.state}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelView":
        v = cls(d["kind"], d["min_count"])
        v.state = d["state"]
        return v

    def save(self, path: str | Path) -> Path:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "ModelView":
        return cls.from_dict(json.loads(Path(path).read_text()))


class RFGiniSelector:
    """Top-k features by Random-Forest Gini importance, fitted on training data only (Ma et al. 2026 use
    the top 15 on ULB). Input must be numeric; NaN is accepted by sklearn ≥ 1.4 trees."""

    def __init__(self, k: int = 15, n_estimators: int = 100, random_state: int = 42, n_jobs: int = 4):
        self.k, self.n_estimators, self.random_state, self.n_jobs = k, n_estimators, random_state, n_jobs
        self.state: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RFGiniSelector":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state,
                                    n_jobs=self.n_jobs)
        rf.fit(X.to_numpy(dtype=float), y.to_numpy())
        imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        self.state = {"ranking": [[c, float(v)] for c, v in imp.items()], "selected": imp.index[:self.k].tolist(),
                      "fitted_rows": int(len(X)), "positives": int(y.sum())}
        return self

    @property
    def selected(self) -> list[str]:
        return self.state["selected"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected]

    def to_dict(self) -> dict[str, Any]:
        return {"frauddet_version": __version__, "k": self.k, "n_estimators": self.n_estimators,
                "random_state": self.random_state, "state": self.state}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RFGiniSelector":
        s = cls(d["k"], d["n_estimators"], d["random_state"])
        s.state = d["state"]
        return s

    def save(self, path: str | Path) -> Path:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "RFGiniSelector":
        return cls.from_dict(json.loads(Path(path).read_text()))


MA2026_SELECTED_15 = ["V17", "V12", "V14", "V10", "V16", "V11", "V9", "V18", "V7", "V4", "V26", "V3", "V21",
                      "V27", "V20"]
