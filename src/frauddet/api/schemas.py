"""Request / response schemas (Pydantic v2), derived from the frozen dataset contracts.

The request model for a dataset is BUILT from its frozen ``DatasetContract`` and the bundle's serving
contract, never hand-written: every field the fitted pipeline consumes becomes a typed field
(int / float / str / datetime, nullable exactly where the contract says so). Model config: strict typing,
unknown fields rejected, non-finite floats rejected. Field constraints and cross-field invariants that
the contracts imply (amounts ≥ 0, coordinates in range, naive timestamps, Sparkov's unix_time ↔ datetime
consistency, dob before the event) are attached per dataset. Unseen categorical *values* are accepted:
the frozen pipeline maps them to <UNK>.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from ..contracts import DatasetContract, Kind
from ..history import HistorySpec
from ..preprocessing import Pipeline

# ------------------------------------------------------------------------------ field constraints
_CONSTRAINTS: dict[str, dict[str, dict[str, Any]]] = {
    "sparkov": {"amt": {"ge": 0}, "lat": {"ge": -90, "le": 90}, "long": {"ge": -180, "le": 180},
                "merch_lat": {"ge": -90, "le": 90}, "merch_long": {"ge": -180, "le": 180},
                "city_pop": {"ge": 0}, "unix_time": {"gt": 0}},
    "ieee": {"TransactionAmt": {"ge": 0}, "TransactionDT": {"ge": 0}, "TransactionID": {"ge": 0}},
    "ulb": {"Time": {"ge": 0}, "Amount": {"ge": 0}},
}
_STRING_MIN_LEN = 1
# fields the API derives itself (never accepted from the client)
DERIVED_AT_SERVING: dict[str, dict[str, Any]] = {
    "ieee": {"has_identity": lambda ev: int(any(ev.get(k) is not None for k in ev if k.startswith("id_")
                                               or k in ("DeviceType", "DeviceInfo")))},
}


def consumed_fields(contract: DatasetContract, pipeline: Pipeline, history: HistorySpec | None,
                    required: tuple[str, ...]) -> list[str]:
    """Fields of the serving contract that the fitted pipeline / history actually read (contract-driven:
    step parameters, history spec, raw inputs kept as features, the row id). PII the pipeline does not
    touch (names, street, zip) is therefore never requested."""
    req = set(required)
    used: set[str] = set()
    for step in pipeline.steps:
        for v in step.params.values():
            vals = v if isinstance(v, (list, tuple)) else [v]
            used.update(x for x in vals if isinstance(x, str) and x in req)
    used.update(c for c in pipeline.feature_columns if c in req)
    if history is not None:
        used.update(x for x in (history.entity, history.order, history.event_time, history.amount, history.lat,
                                history.lon, history.merch_lat, history.merch_lon, *history.conds) if x in req)
    if contract.row_id and contract.row_id in req:
        used.add(contract.row_id)
    derived = set(DERIVED_AT_SERVING.get(contract.name, {}))
    return [f for f in required if f in used and f not in derived]


_PY = {Kind.INT: int, Kind.FLOAT: float, Kind.STRING: str, Kind.DATETIME: dt.datetime}


def build_request_model(contract: DatasetContract, fields: list[str]) -> type[BaseModel]:
    defs: dict[str, Any] = {}
    cons = _CONSTRAINTS.get(contract.name, {})
    for name in fields:
        spec = contract.spec_for(name)
        if spec is None:
            continue
        py = _PY[spec.kind]
        kw: dict[str, Any] = dict(cons.get(name, {}))
        kw["description"] = f"{spec.role.value}: {spec.description or name}"
        if spec.kind is Kind.STRING:
            kw["min_length"] = _STRING_MIN_LEN
        if spec.nullable:                                   # exactly the contract's nullability, nothing more
            defs[name] = (py | None, Field(default=None, **kw))
        else:
            defs[name] = (py, Field(**kw))

    def _cross_field(self):                                   # after-validation invariants per dataset
        if contract.name == "sparkov":
            ts: dt.datetime = self.trans_date_trans_time
            if ts.tzinfo is not None:
                raise ValueError("trans_date_trans_time must be naive (the dataset clock has no timezone)")
            utc = dt.datetime.fromtimestamp(self.unix_time, dt.timezone.utc).replace(tzinfo=None)
            tod = abs((utc - utc.replace(hour=0, minute=0, second=0)) - (ts - ts.replace(hour=0, minute=0, second=0)))
            if tod > dt.timedelta(seconds=1):
                raise ValueError("unix_time and trans_date_trans_time disagree on the time of day "
                                 "(the frozen contract uses the datetime for calendar features and unix_time for ordering)")
            if self.dob.tzinfo is not None or self.dob >= ts:
                raise ValueError("dob must be a naive date before the transaction")
        if contract.name == "ieee" and self.TransactionAmt is not None and not math.isfinite(self.TransactionAmt):
            raise ValueError("TransactionAmt must be finite")
        return self

    return create_model(
        f"{contract.name.capitalize()}Transaction",
        __config__=ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True,
                              title=f"{contract.title}"),
        __validators__={"_cross_field": model_validator(mode="after")(_cross_field)},  # type: ignore[dict-item]
        **defs,
    )


# ------------------------------------------------------------------------------ responses
class Timing(BaseModel):
    features_ms: float = Field(description="feature computation incl. entity state (stateful bundles) and model view")
    model_ms: float = Field(description="model inference + calibration + policy")


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    row_id: str | None = Field(default=None, description="client transaction id echoed back (Sparkov trans_num / IEEE TransactionID)")
    fraud_probability: float = Field(ge=0, le=1, description="calibrated probability of fraud")
    decision: Literal["APPROVED", "SUSPECT"]
    policy: str = Field(description="decision policy name (from the locked policy.json)")
    threshold: float
    latency_ms: float = Field(description="server-side scoring latency: event assembly + features + model + policy; "
                                          "excludes network, JSON parsing/validation and response serialisation")
    timing: Timing
    model_id: str = Field(description="dataset/protocol:<model family>@<model file sha256[:12]>")
    bundle_sha256: str = Field(description="sha256 of the 1A bundle.json (features/views/state hashes inside)")
    calibrator: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ready", "not_ready"]
    ready: bool
    service: str
    dataset: str | None = None
    protocol: str | None = None
    model_id: str | None = None
    bundle_sha256: str | None = None
    policy: str | None = None
    stateful: bool | None = None
    loaded_at: str | None = None
    fast_path: bool | None = None
    shadow_mismatches: int | None = None
    uptime_s: float
    reason: str | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    error: str
    detail: str | None = None
