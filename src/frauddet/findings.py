"""Audit findings: a small, serialisable record type shared by adapters and the audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"    # a measured fact worth recording
    NOTE = "note"    # a limitation or caveat that constrains later phases
    WARN = "warn"    # something that must be handled explicitly (naming mismatch, duplicates, ...)
    RISK = "risk"    # possible leakage / target contamination — needs a decision


@dataclass
class Finding:
    severity: Severity
    code: str            # short stable identifier, e.g. "sparkov.unix_time_offset"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d
