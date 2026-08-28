"""The live contracts must equal the frozen snapshot (Phase 1A.3). If this fails, a role, key,
family or capability claim changed: re-run `python -m frauddet.freeze` deliberately and review the diff."""

import json
from pathlib import Path

import pytest

from frauddet.adapters import ADAPTERS
from frauddet.contracts import Role, Support
from frauddet.freeze import FROZEN_PATH, snapshot_all

DATA = Path(__file__).resolve().parents[1] / "data"


def _frozen():
    if not FROZEN_PATH.exists():
        pytest.fail(f"missing {FROZEN_PATH}; run python -m frauddet.freeze")
    return json.loads(FROZEN_PATH.read_text())


def test_live_contracts_match_frozen_snapshot():
    frozen = _frozen()
    live = snapshot_all(DATA if DATA.exists() else None)
    for name in ADAPTERS:
        f, l = frozen["contracts"][name], live["contracts"][name]
        # when data/ is absent the live snapshot lacks header-resolved family columns; compare what it has
        for k in ("keys", "families", "capabilities", "contract_version", "role_in_suite"):
            assert f[k] == l[k], (name, k)
        for col, role in l["column_roles"].items():
            assert f["column_roles"].get(col) == role, (name, col)
        assert "UNEXPECTED" not in f["column_roles"].values(), name
    assert frozen["labels"] == live["labels"]


def test_frozen_decisions_of_1A3_and_1A4():
    frozen = _frozen()
    sp, ie, ulb = (frozen["contracts"][n] for n in ("sparkov", "ieee", "ulb"))
    assert all(c["contract_version"] == "1A.4" for c in (sp, ie, ulb))
    # Sparkov: engineering dataset; near-identifiers excluded
    assert "PIPELINE-ENGINEERING" in sp["role_in_suite"]
    assert sp["column_roles"]["city"] == Role.EXCLUDED.value and sp["column_roles"]["job"] == Role.EXCLUDED.value
    assert sp["keys"]["dedup_key"] is None
    # IEEE: opaque blocks stay opaque; no entity key; labels finalized
    assert ie["column_roles"]["V1"] == Role.OPAQUE.value and ie["column_roles"]["C1"] == Role.OPAQUE.value
    assert ie["column_roles"]["D9"] == Role.OPAQUE.value and ie["column_roles"]["card1"] == Role.OPAQUE.value
    assert ie["keys"]["entity_key"] is None and ie["keys"]["dedup_key"] is None
    assert ie["capabilities"]["entity_history"] == Support.UNSUPPORTED.value
    assert frozen["labels"]["ieee"]["maturity_policy"] == "finalized"
    assert frozen["labels"]["ieee"]["documented_maturation_seconds"] == 120 * 86400
    # ULB: true duplicate = all 31 columns incl. Time
    assert ulb["keys"]["dedup_key"][0] == "Time" and len(ulb["keys"]["dedup_key"]) == 31
    assert ulb["column_roles"]["Amount"] == Role.INPUT.value
    # 1A.4: cyclic time resolved from the data's own semantics
    assert ie["capabilities"]["cyclic_relative_time"] == Support.SUPPORTED.value     # anchored by D9
    assert ulb["capabilities"]["cyclic_relative_time"] == Support.PARTIAL.value      # relative phase only
    assert ie["capabilities"]["calendar_time"] == Support.UNSUPPORTED.value          # still no calendar
