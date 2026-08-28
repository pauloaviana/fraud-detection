"""Freeze the dataset contracts (Phase 1A.3).

Writes contracts/frozen-contracts.json: column roles (resolved against the real
file headers when data/ is present), key columns, families, dedup keys and the
capability matrix. tests/test_frozen_contracts.py fails whenever the live
contracts drift from this snapshot, so a change to a role or a capability claim
is a deliberate, reviewed act: edit the contract, re-run this module, commit both.

    python -m frauddet.freeze --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import ADAPTERS
from .labels import TARGETS

FROZEN_PATH = Path(__file__).resolve().parents[2] / "contracts" / "frozen-contracts.json"


def snapshot_all(data_dir: str | Path | None) -> dict:
    out: dict = {"contracts": {}, "labels": {}}
    for name, cls in ADAPTERS.items():
        headers = None
        if data_dir is not None:
            a = cls(data_dir)
            if all(a.available(f.key) for f in a.contract.files):
                headers = {f.key: a.header(f.key) for f in a.contract.files}
        out["contracts"][name] = cls.contract.snapshot(headers)
    for name, spec in TARGETS.items():
        out["labels"][name] = {"column": spec.column, "order_key": spec.order_key,
                               "mechanism": spec.provenance.mechanism.value,
                               "maturity_policy": spec.maturity_policy.value,
                               "documented_maturation_seconds": spec.documented_maturation_seconds,
                               "label_derived_columns": list(spec.label_derived_columns)}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Freeze dataset contracts to contracts/frozen-contracts.json")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=str(FROZEN_PATH))
    a = p.parse_args(argv)
    snap = snapshot_all(a.data_dir if Path(a.data_dir).exists() else None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    n = sum(len(c["column_roles"]) for c in snap["contracts"].values())
    print(f"[freeze] {len(snap['contracts'])} contracts, {n} column roles → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
