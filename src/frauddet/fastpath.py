"""Row-native execution of the frozen feature pipeline for single-event scoring (Phase 3A).

The reference path (pandas ``Pipeline`` → ``ModelView`` → model) is the definition of every feature.
This module *compiles* a fitted bundle into per-row functions that perform exactly the same arithmetic
in the same dtypes (float64 intermediates, float32 outputs, identical category codes), so that the
feature vector and the probability are bit-identical to the reference — verified by the parity tests
(tests/test_fastpath.py) on the four locked bundles, and guarded at runtime by shadow checks in the
service. No feature definition is changed; unsupported steps make ``compile`` raise, and the service
falls back to the reference path.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from .calibrate import Calibrator
from .models import Model
from .preprocessing import DAY
from .serving import FeatureBundle

Row = dict[str, Any]


def _isna(v: Any) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, np.floating) and np.isnan(v):
        return True
    return False


def _fstr(v: Any) -> str | None:
    """pandas ``astype("string")`` rendering of a scalar (None for missing)."""
    if _isna(v):
        return None
    if isinstance(v, (float, np.floating)):
        return str(float(v))
    return str(v)


# ------------------------------------------------------------------------------ step compilers
def _strip_prefix(p):
    c, pre = p["column"], p["prefix"]

    def f(r: Row):
        v = _fstr(r.get(c))
        r[c] = None if v is None else v.removeprefix(pre)
    return f


def _log_amount(p):
    c, o = p["column"], p["output"]

    def f(r: Row):
        x = float(r[c])
        r[o] = float(np.log1p(np.clip(x, 0, None)))
    return f


def _amount_decimals(p):
    c, o = p["column"], p["output"]

    def f(r: Row):
        x = float(r[c])
        dec = 3
        for d in (2, 1, 0):
            if np.isclose(x, np.round(x, d), atol=1e-9):
                dec = d
        r[o] = np.int8(dec)
    return f


def _calendar(p):
    c, pre = p["column"], p["prefix"]

    def f(r: Row):
        ts = pd.Timestamp(r[c])
        hour, dow, month = ts.hour, ts.dayofweek, ts.month
        r[f"{pre}_hour"] = np.int8(hour); r[f"{pre}_dow"] = np.int8(dow); r[f"{pre}_month"] = np.int8(month)
        frac = (hour * 3600 + ts.minute * 60 + ts.second) / DAY
        r[f"{pre}_day_sin"] = np.float32(np.sin(2 * np.pi * frac)); r[f"{pre}_day_cos"] = np.float32(np.cos(2 * np.pi * frac))
        r[f"{pre}_dow_sin"] = np.float32(np.sin(2 * np.pi * dow / 7)); r[f"{pre}_dow_cos"] = np.float32(np.cos(2 * np.pi * dow / 7))
    return f


def _age(p):
    dob, ev, o = p["dob"], p["event_time"], p["output"]

    def f(r: Row):
        delta = pd.Timestamp(r[ev]) - pd.Timestamp(r[dob])
        r[o] = np.float32(delta.days / 365.25)
    return f


def _haversine(p):
    la, lo, la2, lo2, o = p["lat"], p["lon"], p["lat2"], p["lon2"], p["output"]

    def f(r: Row):
        a1, o1, a2, o2 = (np.radians(float(r[la])), np.radians(float(r[lo])), np.radians(float(r[la2])), np.radians(float(r[lo2])))
        a = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
        r[o] = np.float32(2 * 6371.0088 * np.arcsin(np.sqrt(a)))
    return f


def _cyclic(p):
    c, per, pre, bins = p["column"], p["period_seconds"], p["prefix"], p["bins"]

    def f(r: Row):
        t = float(r[c])
        phase = np.mod(t, per) / per
        r[f"{pre}_sin"] = np.float32(np.sin(2 * np.pi * phase)); r[f"{pre}_cos"] = np.float32(np.cos(2 * np.pi * phase))
        if bins:
            r[f"{pre}_bin"] = np.int16(np.floor(phase * bins))
    return f


def _token(p):
    c, o, pat, idx = p["column"], p["output"], re.compile(p["pattern"]), p["index"]

    def f(r: Row):
        v = _fstr(r.get(c))
        if v is None:
            r[o] = None
            return
        parts = pat.split(v)
        try:
            r[o] = parts[idx]
        except IndexError:
            r[o] = None
    return f


def _row_missing(p, state):
    cols, o = state["columns"], p["output"]

    def f(r: Row):
        r[o] = np.int16(sum(1 for c in cols if c in r and _isna(r[c])))
    return f


def _missing_indicator(state):
    cols = state["columns"]

    def f(r: Row):
        for c in cols:
            r[f"{c}__isna"] = np.int8(1 if _isna(r.get(c)) else 0)
    return f


def _frequency(state):
    freq = state["freq"]

    def f(r: Row):
        for c, fr in freq.items():
            v = _fstr(r.get(c))
            r[f"{c}_freq"] = np.float32(fr[v]) if v is not None and v in fr else np.float32(0.0)
    return f


def _typer(state):
    cats = {c: set(v) for c, v in state["categories"].items()}

    def f(r: Row):
        for c, allowed in cats.items():
            v = _fstr(r.get(c))
            s = "<NA>" if v is None else v
            r[c] = s if s in allowed else "<UNK>"
    return f


def _standardize(state):
    stats = state["stats"]

    def f(r: Row):
        for c, s in stats.items():
            r[c] = np.float32((float(r[c]) - s["mean"]) / s["std"])
    return f


def _noop(*_):
    return lambda r: None


def compile_steps(pipeline) -> list[Callable[[Row], None]]:
    fns = []
    for step in pipeline.steps:
        k, p, st = type(step).__name__, step.params, step.state
        if k == "StripPrefix":
            fns.append(_strip_prefix(p))
        elif k == "LogAmount":
            fns.append(_log_amount(p))
        elif k == "AmountDecimals":
            fns.append(_amount_decimals(p))
        elif k == "CalendarFeatures":
            fns.append(_calendar(p))
        elif k == "AgeAtEvent":
            fns.append(_age(p))
        elif k == "HaversineDistance":
            fns.append(_haversine(p))
        elif k == "CyclicClock":
            fns.append(_cyclic(p))
        elif k == "TokenSplit":
            fns.append(_token(p))
        elif k == "RowMissingCount":
            fns.append(_row_missing(p, st))
        elif k == "MissingIndicator":
            fns.append(_missing_indicator(st))
        elif k == "FrequencyEncoder":
            fns.append(_frequency(st))
        elif k == "CategoricalTyper":
            fns.append(_typer(st))
        elif k == "Standardize":
            fns.append(_standardize(st))
        elif k in ("FinalizeFeatures", "DropColumns"):
            fns.append(_noop())
        else:
            raise NotImplementedError(f"fast path: unsupported step {k}")
    return fns


# ------------------------------------------------------------------------------ compiled scorer
class FastScorer:
    """Event dict → float32 feature vector (tree view, optional selection) → probability."""

    def __init__(self, bundle: FeatureBundle, model: Model, calibrator: Calibrator):
        if model.view != "tree":
            raise NotImplementedError("fast path supports the tree view only")
        self.bundle, self.model, self.calibrator = bundle, model, calibrator
        self.steps = compile_steps(bundle.pipeline)
        view = bundle.views["tree"].state
        cols = list(view["columns"])
        if bundle.selector is not None:
            cols = [c for c in bundle.selector.selected if c in cols]
        self.columns = cols
        self.cat_codes: dict[str, dict[str, int]] = {c: {v: i for i, v in enumerate(view["categorical"][c])}
                                                     for c in cols if c in view["categorical"]}
        self.n = len(cols)
        self._predict = self._make_predict()

    def _make_predict(self):
        m = self.model
        if m.name == "lightgbm":
            booster = m._est.booster_
            it = m.best_iteration

            def pred(x):
                return float(booster.predict(x, num_iteration=it, num_threads=1)[0])
            return pred
        if m.name == "xgboost":
            booster = m._est.get_booster()
            booster.set_param({"nthread": 1})
            rng = (0, m.best_iteration) if m.best_iteration else (0, 0)

            def pred(x):
                return float(booster.inplace_predict(x, iteration_range=rng)[0])
            return pred
        if m.name == "dummy":
            prior = m.meta["prior"]
            return lambda x: float(prior)
        raise NotImplementedError(m.name)

    def features(self, event: Row) -> np.ndarray:
        r = dict(event)
        for fn in self.steps:
            fn(r)
        out = np.empty((1, self.n), dtype=np.float32)
        for j, c in enumerate(self.columns):
            if c in self.cat_codes:
                out[0, j] = self.cat_codes[c][r[c]]
            else:
                v = r.get(c)
                out[0, j] = np.nan if _isna(v) else v
        return out

    def score(self, event: Row) -> float:
        x = self.features(event)
        return float(self.calibrator.transform(np.array([self._predict(x)]))[0])


def compile_fast_scorer(bundle: FeatureBundle, model: Model, calibrator: Calibrator) -> FastScorer:
    return FastScorer(bundle, model, calibrator)
