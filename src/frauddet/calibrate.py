"""Probability calibration (Phase 1B) — fitted on out-of-fold training predictions only.

Methods: "none", "platt" (logistic regression on the logit of the score), "isotonic" (monotone
piecewise-constant map). All serialise to JSON. A calibrator is chosen with validation information only
(never the holdout) and never changes the ranking within a method (both maps are monotone), so PR-AUC /
ROC-AUC are unaffected while Brier / log loss / ECE and the meaning of a threshold are.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-7


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "none"):
        if method not in ("none", "platt", "isotonic"):
            raise ValueError(method)
        self.method = method
        self.state: dict[str, Any] = {}

    def fit(self, p_oof: np.ndarray, y: np.ndarray) -> "Calibrator":
        p_oof, y = np.asarray(p_oof, float), np.asarray(y).astype(int)
        if self.method == "platt":
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_oof).reshape(-1, 1), y)
            self.state = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_oof, y)
            self.state = {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}
        else:
            self.state = {}
        self.state["fitted_on"] = int(len(y))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        if self.method == "platt":
            z = self.state["a"] * _logit(p) + self.state["b"]
            return 1 / (1 + np.exp(-z))
        if self.method == "isotonic":
            return np.interp(p, self.state["x"], self.state["y"])
        return p

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "state": self.state}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibrator":
        c = cls(d["method"])
        c.state = d["state"]
        return c

    def save(self, path: str | Path) -> Path:
        Path(path).write_text(json.dumps(self.to_dict()))
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_calibrators(p_oof: np.ndarray, y: np.ndarray) -> dict[str, Calibrator]:
    return {m: Calibrator(m).fit(p_oof, y) for m in ("none", "platt", "isotonic")}
