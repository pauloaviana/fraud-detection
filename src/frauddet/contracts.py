"""Raw-data contracts (Phase 1A.1).

A contract describes a dataset *as shipped*: its files, its columns with declared
types, and the *role* each column is allowed to play downstream. Roles are the
mechanism that keeps identifiers apart from classifier inputs:

* ``ENTITY_KEY`` / ``ORDER_KEY`` / ``EVENT_TIME`` / ``ROW_ID`` / ``JOIN_KEY`` are
  identifiers: legitimate for ordering, joining and constructing historical
  features, never fed to a classifier directly.
* ``INPUT`` / ``OPAQUE`` / ``GROUP_KEY`` are direct-input candidates (``GROUP_KEY``
  doubles as a grouping criterion for historical aggregation).
* ``PII`` / ``EXCLUDED`` are never inputs.

A contract also carries the dataset's *capability claims*: which feature families
the data can genuinely support, with the columns that justify the claim.

Nothing in this module reads data, renames, derives or imputes anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    TARGET = "target"
    ROW_ID = "row_id"          # unique transaction identifier; never an input
    JOIN_KEY = "join_key"      # links files of one dataset (e.g. TransactionID in identity)
    ENTITY_KEY = "entity_key"  # stable customer/card identifier: history construction only
    ORDER_KEY = "order_key"    # monotone ordering and time deltas
    EVENT_TIME = "event_time"  # wall-clock timestamp with calendar semantics
    GROUP_KEY = "group_key"    # categorical context: grouping criterion and/or input
    INPUT = "input"            # direct classifier-input candidate with known semantics
    OPAQUE = "opaque"          # masked / anonymised numeric input; semantics unknown
    PII = "pii"                # personal data; not an input (derivations noted per column)
    EXCLUDED = "excluded"      # not usable (positional index, generator artefact, ...)
    LABEL_DERIVED = "label_derived"  # post-investigation / label-resolution info (chargeback dates,
                                     # investigator outcome, dispute status, ...): never an input.
                                     # None of the three datasets ship such columns; the role exists so
                                     # a production schema cannot smuggle them in as INPUT.


IDENTIFIER_ROLES = frozenset({Role.ROW_ID, Role.JOIN_KEY, Role.ENTITY_KEY, Role.ORDER_KEY, Role.EVENT_TIME})
INPUT_ROLES = frozenset({Role.INPUT, Role.OPAQUE, Role.GROUP_KEY})
LABEL_ROLES = frozenset({Role.TARGET, Role.LABEL_DERIVED})
NEVER_INPUT_ROLES = frozenset({Role.PII, Role.EXCLUDED}) | LABEL_ROLES | IDENTIFIER_ROLES


class Kind(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    DATETIME = "datetime"


_PANDAS_DTYPE = {Kind.INT: "int64", Kind.FLOAT: "float64", Kind.STRING: "string"}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: Kind
    role: Role
    nullable: bool = False
    description: str = ""
    notes: str = ""            # caveats, artefacts, leakage considerations
    dtype: str | None = None   # pandas dtype override (e.g. "float32")

    @property
    def pandas_dtype(self) -> str | None:
        if self.kind is Kind.DATETIME:
            return None        # parsed via parse_dates
        if self.dtype:
            return self.dtype
        if self.kind is Kind.INT and self.nullable:
            return "Int64"
        return _PANDAS_DTYPE[self.kind]


@dataclass(frozen=True)
class ColumnFamily:
    """A regex-defined group of same-typed columns (IEEE V1..V339, ULB V1..V28, ...)."""

    label: str
    pattern: str
    kind: Kind
    role: Role
    nullable: bool = True
    description: str = ""
    notes: str = ""
    dtype: str | None = None

    def matches(self, name: str) -> bool:
        return re.fullmatch(self.pattern, name) is not None

    def spec(self, name: str) -> ColumnSpec:
        return ColumnSpec(name, self.kind, self.role, self.nullable, self.description, self.notes, self.dtype)


@dataclass(frozen=True)
class FileSpec:
    key: str                    # e.g. "train", "test", "train_identity"
    member: str                 # csv file name
    container: str | None = None  # zip archive (relative to data dir) or None for a bare csv
    labeled: bool = True
    purpose: str = ""


class FeatureFamily(str, Enum):
    """Feature families a dataset may or may not be able to support (capability mapping)."""

    RAW_TRANSACTION = "raw_transaction"            # amount, transaction type/category as given
    ENTITY_HISTORY = "entity_history"              # per-entity aggregation / velocity (Bahnsen agg1)
    ENTITY_HISTORY_GROUPED = "entity_history_grouped"  # per-entity aggregation by context (Bahnsen agg2)
    PERIODIC_TIME_ENTITY = "periodic_time_entity"  # per-entity periodic time-of-day (von Mises)
    CALENDAR_TIME = "calendar_time"                # hour / weekday / month from a wall-clock timestamp
    RELATIVE_TIME = "relative_time"                # ordering, deltas, elapsed time
    CYCLIC_RELATIVE_TIME = "cyclic_relative_time"  # day-fraction of a relative clock (phase unknown)
    GEO_DISTANCE = "geo_distance"                  # customer <-> merchant / previous-location distance
    DEMOGRAPHICS = "demographics"                  # age, gender, population, occupation
    MERCHANT_CONTEXT = "merchant_context"          # merchant identity / category
    CARD_META = "card_meta"                        # network, debit/credit, issuer-ish codes
    EMAIL_DOMAIN = "email_domain"
    DEVICE_IDENTITY = "device_identity"            # device / browser / OS / identity-vendor fields
    OPAQUE_MASKED = "opaque_masked"                # anonymised numeric columns used as-is
    COST_SENSITIVE_EVAL = "cost_sensitive_eval"    # amount available for cost-matrix evaluation


class Support(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    PROVISIONAL = "provisional"   # technically possible but semantics undecided (next step)
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CapabilityClaim:
    family: FeatureFamily
    support: Support
    basis: tuple[str, ...]   # columns that justify (or would be needed for) the claim
    reason: str


@dataclass
class DatasetContract:
    name: str
    title: str
    source: str
    files: list[FileSpec]
    columns: list[ColumnSpec]
    families: list[ColumnFamily] = field(default_factory=list)
    role_in_suite: str = ""              # what this dataset is *for* in the experimental suite (README)
    contract_version: str = ""           # frozen version tag; changes require re-freezing the snapshot
    dedup_key: tuple[str, ...] | None = None  # columns whose joint equality defines a TRUE duplicate record;
                                              # None = row ids are unique, no duplicate concept. Applying
                                              # the deduplication is preprocessing, not part of the contract.
    target: str | None = None
    positive_label: int = 1
    row_id: str | None = None
    entity_key: str | None = None
    order_key: str | None = None
    event_time: str | None = None
    join_key: str | None = None
    capabilities: list[CapabilityClaim] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- lookup -----------------------------------------------------------------
    def file(self, key: str) -> FileSpec:
        for f in self.files:
            if f.key == key:
                return f
        raise KeyError(f"{self.name}: no file {key!r}")

    def spec_for(self, name: str) -> ColumnSpec | None:
        for c in self.columns:
            if c.name == name:
                return c
        for fam in self.families:
            if fam.matches(name):
                return fam.spec(name)
        return None

    def resolve(self, header: list[str]) -> tuple[list[ColumnSpec], list[str]]:
        """Map a file header onto specs. Returns (specs, unexpected column names)."""
        specs, unexpected = [], []
        for name in header:
            s = self.spec_for(name)
            if s is None:
                unexpected.append(name)
            else:
                specs.append(s)
        return specs, unexpected

    def columns_with_role(self, header: list[str], role: Role) -> list[str]:
        return [n for n in header if (s := self.spec_for(n)) is not None and s.role is role]

    def claim(self, family: FeatureFamily) -> CapabilityClaim | None:
        for c in self.capabilities:
            if c.family is family:
                return c
        return None

    # -- freeze snapshot (compared against contracts/frozen-contracts.json by tests) ----
    def snapshot(self, headers: dict[str, list[str]] | None = None) -> dict:
        """Roles, families, keys and capability claims in a stable, JSON-friendly form.

        ``headers`` (file key -> header) lets the snapshot list the resolved role of every real
        column; without it only explicit columns and family patterns are recorded.
        """
        cols = {c.name: c.role.value for c in self.columns}
        if headers:
            for key, header in headers.items():
                for name in header:
                    s = self.spec_for(name)
                    cols[name] = s.role.value if s else "UNEXPECTED"
        return {
            "name": self.name, "contract_version": self.contract_version, "role_in_suite": self.role_in_suite,
            "files": [{"key": f.key, "member": f.member, "labeled": f.labeled} for f in self.files],
            "keys": {"target": self.target, "row_id": self.row_id, "entity_key": self.entity_key,
                     "order_key": self.order_key, "event_time": self.event_time, "join_key": self.join_key,
                     "dedup_key": list(self.dedup_key) if self.dedup_key else None},
            "families": [{"label": f.label, "pattern": f.pattern, "kind": f.kind.value, "role": f.role.value}
                         for f in self.families],
            "column_roles": dict(sorted(cols.items())),
            "capabilities": {c.family.value: c.support.value for c in self.capabilities},
        }

    # -- self-checks (exercised by tests) ---------------------------------------
    def validate(self) -> list[str]:
        problems: list[str] = []
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            problems.append("duplicate explicit column names")
        for fam in self.families:
            re.compile(fam.pattern)
        for attr in ("target", "row_id", "entity_key", "order_key", "event_time", "join_key"):
            val = getattr(self, attr)
            if val is not None and self.spec_for(val) is None:
                problems.append(f"{attr}={val!r} is not a declared column")
        role_expect = {
            "target": Role.TARGET, "row_id": Role.ROW_ID, "entity_key": Role.ENTITY_KEY,
            "order_key": Role.ORDER_KEY, "event_time": Role.EVENT_TIME, "join_key": Role.JOIN_KEY,
        }
        for attr, role in role_expect.items():
            val = getattr(self, attr)
            if val is None or self.spec_for(val) is None:
                continue
            got = self.spec_for(val).role
            allowed = {role, Role.ROW_ID} if attr == "join_key" else {role}  # a row id may serve as join key
            if got not in allowed:
                problems.append(f"{attr}={val!r} does not carry role {role.value}")
        claimed = {c.family for c in self.capabilities}
        missing = set(FeatureFamily) - claimed
        if missing:
            problems.append(f"no capability claim for: {sorted(m.value for m in missing)}")
        if self.dedup_key:
            for col in self.dedup_key:
                if self.spec_for(col) is None:
                    problems.append(f"dedup_key cites unknown column {col!r}")
        for c in self.capabilities:
            if not c.reason:
                problems.append(f"capability {c.family.value} has no reason")
            for col in c.basis:
                if c.support is not Support.UNSUPPORTED and self.spec_for(col) is None:
                    problems.append(f"capability {c.family.value} cites unknown column {col!r}")
        return problems
