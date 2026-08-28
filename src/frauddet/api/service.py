"""ScoringService: frozen 1A bundle + locked 1B model/calibrator/policy, loaded once, scored per event.

Probability layer: bundle.serve_event → model.predict_proba → calibrator.
Decision layer: policy.json threshold (name chosen by configuration) → APPROVED / SUSPECT.
Stateful bundles (Sparkov) serialise scoring behind a lock so the entity state sees events in order.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..calibrate import Calibrator
from ..fastpath import compile_fast_scorer
from ..contracts import Kind
from ..history import DuplicateEvent, OutOfOrderEvent
from ..models import Model
from ..serving import FeatureBundle
from .config import Settings
from .schemas import DERIVED_AT_SERVING, build_request_model, consumed_fields


class NotReady(RuntimeError):
    pass


class OrderingError(ValueError):
    """Wraps DuplicateEvent / OutOfOrderEvent for the API layer (409)."""


@dataclass
class Prediction:
    row_id: str | None
    probability: float
    decision: str
    policy: str
    threshold: float
    features_ms: float
    model_ms: float
    latency_ms: float


class ScoringService:
    def __init__(self, settings: Settings):
        self.settings = settings
        bdir, mdir = settings.bundle_dir, settings.model_dir
        for p in (bdir / "bundle.json", mdir / "model" / "model.json", mdir / "calibrator.json",
                  mdir / "policy.json", mdir / "locked.json"):
            if not p.exists():
                raise NotReady(f"missing serving artifact: {p}")
        self.bundle = FeatureBundle.load(bdir, strict_order=settings.strict_order)
        self.model = Model.load(mdir / "model")
        self.calibrator = Calibrator.load(mdir / "calibrator.json")
        policy = json.loads((mdir / "policy.json").read_text())
        self.thresholds: dict[str, float] = policy["thresholds"]
        if settings.policy not in self.thresholds:
            raise NotReady(f"policy {settings.policy!r} not in policy.json ({sorted(self.thresholds)})")
        self.threshold = float(self.thresholds[settings.policy])
        self.locked = json.loads((mdir / "locked.json").read_text())
        if self.locked["champion"]["model"] != self.model.name:
            raise NotReady("locked.json and model.json disagree on the model family")
        spec = json.loads((mdir / "model" / "model.json").read_text())
        mfile = mdir / "model" / spec["files"][0]
        self.model_id = f"{settings.dataset}/{settings.protocol}:{self.model.name}@{hashlib.sha256(mfile.read_bytes()).hexdigest()[:12]}"
        self.bundle_sha256 = hashlib.sha256((bdir / "bundle.json").read_bytes()).hexdigest()
        self.contract = self.bundle.contract
        self.required = tuple(self.bundle.required_fields)
        self.fields = consumed_fields(self.contract, self.bundle.pipeline, self.bundle.history_spec, self.required)
        self.request_model = build_request_model(self.contract, self.fields)
        self.derive = DERIVED_AT_SERVING.get(self.contract.name, {})
        self.stateful = self.bundle.history_spec is not None
        self.loaded_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self._lock = threading.Lock()
        self._kinds = {f: spec.kind for f in self.required if (spec := self.contract.spec_for(f)) is not None}
        # row-native fast path (bit-identical to the pandas reference; see fastpath.py). The first
        # ``shadow_checks`` requests run BOTH paths and compare; any mismatch disables the fast path.
        self.fast = None
        self.shadow_remaining = settings.shadow_checks
        self.shadow_mismatches = 0
        if settings.fast_path:
            try:
                self.fast = compile_fast_scorer(self.bundle, self.model, self.calibrator)
            except NotImplementedError as e:
                logging.getLogger("frauddet.api").warning("fast_path_unavailable", extra={"request_id": None, "reason": str(e)})
        self.hist_names = self.bundle.history_spec.feature_names() if self.bundle.history_spec is not None else []

    # -- event assembly ------------------------------------------------------------
    def _event(self, payload: dict[str, Any]) -> dict[str, Any]:
        ev: dict[str, Any] = {}
        for f in self.required:
            v = payload.get(f)
            kind = self._kinds.get(f)
            if v is None:
                ev[f] = np.nan if kind in (Kind.FLOAT, Kind.INT) else (pd.NaT if kind is Kind.DATETIME else None)
            elif kind is Kind.DATETIME:
                ev[f] = pd.Timestamp(v)
            else:
                ev[f] = v
        for name, fn in self.derive.items():
            ev[name] = fn(payload)
        return ev

    # -- scoring -------------------------------------------------------------------
    def predict(self, payload: dict[str, Any]) -> Prediction:
        t0 = time.perf_counter()
        ev = self._event(payload)
        row_id = payload.get(self.contract.row_id) if self.contract.row_id else None
        if self.fast is not None:
            p, t1, t2 = self._predict_fast(ev, row_id)
            if self.shadow_remaining > 0:                      # compare with the reference path, once per request
                self.shadow_remaining -= 1
                self._shadow(ev, row_id, p)
        else:
            p, t1, t2 = self._predict_reference(ev, row_id, update_state=True)
        p = min(max(p, 0.0), 1.0) if math.isfinite(p) else 0.0
        decision = "SUSPECT" if p >= self.threshold else "APPROVED"
        t3 = time.perf_counter()
        return Prediction(None if row_id is None else str(row_id), p, decision, self.settings.policy, self.threshold,
                          (t1 - t0) * 1e3, (t2 - t1) * 1e3 + (t3 - t2) * 1e3, (t3 - t0) * 1e3)

    def _predict_fast(self, ev, row_id):
        try:
            if self.stateful:
                with self._lock:
                    feats = self.bundle.state.process(ev, row_id)
                    self._last_feats = feats
                    x = self.fast.features({**ev, **{k: np.float32(v) for k, v in feats.items()}})
                    t1 = time.perf_counter()
                    raw = self.fast._predict(x)
            else:
                x = self.fast.features(ev)
                t1 = time.perf_counter()
                raw = self.fast._predict(x)
        except (DuplicateEvent, OutOfOrderEvent) as e:
            raise OrderingError(str(e)) from e
        p = float(self.calibrator.transform(np.array([raw]))[0])
        return p, t1, time.perf_counter()

    def _predict_reference(self, ev, row_id, update_state: bool):
        try:
            if self.stateful and update_state:
                with self._lock:
                    x = self.bundle.serve_event(ev, view=self.model.view, row_id=row_id)
            elif self.stateful:                                  # shadow: reuse the features just computed
                frame = pd.concat([pd.DataFrame([ev]),
                                   pd.DataFrame([self._last_feats])[self.hist_names].astype("float32")], axis=1)
                x = self.bundle.views[self.model.view].transform(self.bundle.pipeline.transform(frame))
            else:
                x = self.bundle.serve_event(ev, view=self.model.view)
        except (DuplicateEvent, OutOfOrderEvent) as e:
            raise OrderingError(str(e)) from e
        t1 = time.perf_counter()
        if self.bundle.selector is not None:
            x = self.bundle.selector.transform(x)
        p = float(self.calibrator.transform(self.model.predict_proba(x))[0])
        return p, t1, time.perf_counter()

    def _shadow(self, ev, row_id, p_fast: float) -> None:
        p_ref, _, _ = self._predict_reference(ev, row_id, update_state=False)
        if p_ref != p_fast:
            self.shadow_mismatches += 1
            self.fast = None                                     # fail safe: reference path from now on
            logging.getLogger("frauddet.api").error("fast_path_mismatch",
                                                    extra={"request_id": None, "p_fast": p_fast, "p_ref": p_ref,
                                                           "row_id_hash": None, "action": "fast path disabled"})

    def info(self) -> dict[str, Any]:
        return {"dataset": self.settings.dataset, "protocol": self.settings.protocol, "model_id": self.model_id,
                "bundle_sha256": self.bundle_sha256, "policy": self.settings.policy, "threshold": self.threshold,
                "calibrator": self.calibrator.method, "stateful": self.stateful, "loaded_at": self.loaded_at,
                "fields": self.fields, "champion": self.locked["champion"],
                "fast_path": self.fast is not None, "shadow_remaining": self.shadow_remaining,
                "shadow_mismatches": self.shadow_mismatches}
