"""Dataset adapters (read-only)."""

from __future__ import annotations

from pathlib import Path

from .base import RawAdapter
from .ieee import IEEEAdapter
from .sparkov import SparkovAdapter
from .ulb import ULBAdapter

ADAPTERS: dict[str, type[RawAdapter]] = {
    "sparkov": SparkovAdapter,
    "ieee": IEEEAdapter,
    "ulb": ULBAdapter,
}


def get_adapter(name: str, data_dir: str | Path) -> RawAdapter:
    try:
        return ADAPTERS[name](data_dir)
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(ADAPTERS)}") from None


__all__ = ["ADAPTERS", "RawAdapter", "SparkovAdapter", "IEEEAdapter", "ULBAdapter", "get_adapter"]
