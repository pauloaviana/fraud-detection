"""Serving-compatible feature bundle (Phase 1A.7).

A ``FeatureBundle`` is everything needed to turn raw events of ONE dataset/protocol into model-facing
features, offline (batch) or online (one event at a time), with identical results:

    bundle = FeatureBundle.load("artifacts/sparkov/temporal")
    X = bundle.transform_batch(frame, view="tree")            # offline: chronological frame (+ warm-up rows)
    x = bundle.serve_event(event, view="tree", row_id=...)    # online: uses and updates the entity state

Nothing in a bundle is shared with another dataset: contract, fitted pipeline, views, selector and
history state are loaded from that dataset's own artifact directory. The ``ServingContract`` states the
obligations of the caller (ordering, idempotency, warm-up) and the guarantees of the bundle (point-in-
time features, cold-start behaviour, no labels).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .contracts import DatasetContract
from .history import EntityStateStore, HistorySpec, compute_history, snapshot_from_frame
from .preprocessing import Pipeline
from .views import ModelView, RFGiniSelector

STATE_FORMAT = "frauddet.entity_state.v1"
BUNDLE_FILES = {"features.json", "view-tree.json", "view-linear.json", "selection.json", "history-state.json"}


@dataclass(frozen=True)
class ServingContract:
    dataset: str
    protocol: str
    contract_version: str
    order_key: str
    entity_key: str | None
    row_id: str | None
    event_time: str | None
    stateful: bool
    required_fields: tuple[str, ...]            # raw event fields the bundle needs (target excluded)
    outputs: dict[str, int]                     # feature_layer / tree / linear (+ selected) sizes
    state_retention_seconds: float | None       # events older than this are pruned from the entity buffer
    rules: tuple[str, ...] = (
        "point_in_time: features of an event use only events observed before it; labels never enter",
        "ordering: events of one entity must arrive in non-decreasing order-key order (global chronological "
        "order recommended); an earlier event is rejected (OutOfOrderEvent) unless strict_order is disabled",
        "idempotency: resubmitting an entity's last processed row id is rejected (DuplicateEvent); callers "
        "must not re-score an event to obtain different features",
        "update_after_score: the event is recorded after its features are computed, regardless of any label",
        "warm_up: before online scoring, replay (or restore a snapshot of) at least state_retention_seconds "
        "of prior events per entity; the offline snapshot after the training part is shipped as history-state.json",
        "cold_start: an unseen entity gets counts 0 and NaN for the other history features; unseen categories "
        "map to <UNK>, missing to <NA>; NaN is kept in the tree view and train-median-imputed in the linear view",
        "determinism: the same event sequence yields the same features; offline batch == online stream (tested)",
        "isolation: no fitted state is shared between datasets; every artifact carries the dataset name and "
        "the frozen-contract hash",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FeatureBundle:
    def __init__(self, contract: DatasetContract, protocol: str, pipeline: Pipeline, views: dict[str, ModelView],
                 selector: RFGiniSelector | None = None, history_spec: HistorySpec | None = None,
                 state: EntityStateStore | None = None, required_fields: tuple[str, ...] = (),
                 serving: ServingContract | None = None):
        self.contract, self.protocol, self.pipeline, self.views = contract, protocol, pipeline, views
        self.selector, self.history_spec, self.state = selector, history_spec, state
        if history_spec is not None and state is None:
            self.state = EntityStateStore(history_spec)
        self.required_fields = tuple(required_fields)
        self.serving = serving or self.make_contract()

    # -- contract --------------------------------------------------------------------
    def make_contract(self) -> ServingContract:
        c = self.contract
        outputs = {"feature_layer": len(self.pipeline.feature_columns),
                   **{k: len(v.state["output_columns"]) for k, v in self.views.items()}}
        if self.selector is not None:
            outputs["selected"] = len(self.selector.selected)
        return ServingContract(c.name, self.protocol, c.contract_version, c.order_key or "", c.entity_key, c.row_id,
                               c.event_time, self.history_spec is not None, self.required_fields, outputs,
                               self.history_spec.max_window_s if self.history_spec else None)

    # -- offline ---------------------------------------------------------------------
    def transform_batch(self, df: pd.DataFrame, view: str | None = "tree", with_history: bool = True,
                        apply_selection: bool = False) -> pd.DataFrame:
        """Batch path. For stateful bundles ``df`` must be the chronological frame including the warm-up
        rows that precede the rows of interest (history is computed from prior rows of the same entity)."""
        if self.history_spec is not None and with_history:
            df = compute_history(df, self.history_spec)
        X = self.pipeline.transform(df)
        return self._view(X, view, apply_selection)

    # -- online ----------------------------------------------------------------------
    def serve_event(self, event: dict[str, Any], view: str | None = "tree", row_id: Any = None,
                    apply_selection: bool = False) -> pd.DataFrame:
        """Online path: one event → one feature row, updating the entity state (stateful bundles)."""
        missing = [f for f in self.required_fields if f not in event]
        if missing:
            raise KeyError(f"event lacks required fields {missing[:8]}{'...' if len(missing) > 8 else ''}")
        row = dict(event)
        frame = pd.DataFrame([row])
        if self.state is not None and self.history_spec is not None:
            rid = row_id if row_id is not None else (row.get(self.contract.row_id) if self.contract.row_id else None)
            feats = self.state.process(row, rid)
            hist = pd.DataFrame([feats])[self.history_spec.feature_names()].astype("float32")   # same dtype as batch
            frame = pd.concat([frame, hist], axis=1)
        X = self.pipeline.transform(frame)
        return self._view(X, view, apply_selection)

    def warm_up(self, df: pd.DataFrame) -> None:
        """Replay a chronological frame into the entity state without producing features (bulk, vectorised)."""
        if self.history_spec is None:
            return
        snap = snapshot_from_frame(df, self.history_spec, self.contract.row_id)
        self.state = snap

    def _view(self, X: pd.DataFrame, view: str | None, apply_selection: bool) -> pd.DataFrame:
        if view is None:
            return X
        out = self.views[view].transform(X)
        if apply_selection and self.selector is not None:
            out = self.selector.transform(out)
        return out

    # -- io --------------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.pipeline.save(d / "features.json")
        for k, v in self.views.items():
            v.save(d / f"view-{k}.json")
        if self.selector is not None:
            self.selector.save(d / "selection.json")
        if self.state is not None:
            self.state.save(d / "history-state.json")          # new bundles are written in the current (v2) format
        index = {"frauddet_version": __version__, "dataset": self.contract.name, "protocol": self.protocol,
                 "contract_version": self.contract.contract_version,
                 "history_spec": asdict(self.history_spec) if self.history_spec else None,
                 "serving_contract": self.serving.to_dict(),
                 "files": {p.name: _sha(p) for p in sorted(d.iterdir()) if p.name in BUNDLE_FILES}}
        (d / "bundle.json").write_text(json.dumps(index, indent=1))
        return d / "bundle.json"

    @classmethod
    def load(cls, directory: str | Path, strict_order: bool = True, require_state: bool = True) -> "FeatureBundle":
        """Load a bundle. Stateful bundles need an entity-state snapshot in the current format: a
        ``history-state.v2.json`` (Phase 3A serving artifact, hash-checked via ``serving-extras.json``) is
        preferred; an older ``history-state.json`` that the store cannot continue from raises unless
        ``require_state=False`` (used by the upgrade tool)."""
        from .adapters import ADAPTERS
        d = Path(directory)
        index = json.loads((d / "bundle.json").read_text())
        contract = ADAPTERS[index["dataset"]].contract
        if contract.contract_version != index["contract_version"]:
            raise RuntimeError(f"bundle built for contract {index['contract_version']}, code has {contract.contract_version}")
        for name, sha in index["files"].items():
            if (d / name).exists() and _sha(d / name) != sha:
                raise RuntimeError(f"artifact {name} does not match the bundle index (modified?)")
        pipeline = Pipeline.load(d / "features.json")
        views = {k: ModelView.load(d / f"view-{k}.json") for k in ("tree", "linear") if (d / f"view-{k}.json").exists()}
        selector = RFGiniSelector.load(d / "selection.json") if (d / "selection.json").exists() else None
        hspec = state = None
        if index["history_spec"]:
            hspec = HistorySpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in index["history_spec"].items()})  # type: ignore[arg-type]
            v2, extras = d / "history-state.v2.json", d / "serving-extras.json"
            if v2.exists():
                if extras.exists():
                    want = json.loads(extras.read_text())["files"].get(v2.name)
                    if want and _sha(v2) != want:
                        raise RuntimeError("history-state.v2.json does not match serving-extras.json (modified?)")
                state = EntityStateStore.load(v2)
                state.strict_order = strict_order
            elif (d / "history-state.json").exists():
                try:
                    state = EntityStateStore.load(d / "history-state.json")
                    state.strict_order = strict_order
                except ValueError as e:
                    if require_state:
                        raise RuntimeError(f"{e}; run `python -m frauddet.state_upgrade --dataset {index['dataset']} "
                                           f"--protocol {index['protocol']}` to derive the v2 serving state") from e
        sc = index["serving_contract"]
        serving = ServingContract(**{**sc, "required_fields": tuple(sc["required_fields"]), "rules": tuple(sc["rules"])})
        return cls(contract, index["protocol"], pipeline, views, selector, hspec, state,
                   tuple(sc["required_fields"]), serving)
