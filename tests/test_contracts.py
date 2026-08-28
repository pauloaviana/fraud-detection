"""Pure unit tests for the raw-data contracts (no data files needed)."""

import pytest

from frauddet.adapters import ADAPTERS
from frauddet.adapters.ieee import canonical_name
from frauddet.contracts import (
    IDENTIFIER_ROLES, INPUT_ROLES, NEVER_INPUT_ROLES, FeatureFamily, Kind, Role, Support,
)

CONTRACTS = {name: cls.contract for name, cls in ADAPTERS.items()}


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_contract_is_self_consistent(name):
    assert CONTRACTS[name].validate() == []


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_every_feature_family_has_a_claim_with_reason(name):
    c = CONTRACTS[name]
    claimed = {k.family for k in c.capabilities}
    assert claimed == set(FeatureFamily)
    assert all(k.reason for k in c.capabilities)


@pytest.mark.parametrize("name", list(CONTRACTS))
def test_identifier_and_input_roles_are_disjoint(name):
    c = CONTRACTS[name]
    assert not (IDENTIFIER_ROLES & INPUT_ROLES)
    for col in c.columns:
        assert (col.role in NEVER_INPUT_ROLES) != (col.role in INPUT_ROLES)


def test_sparkov_roles():
    c = CONTRACTS["sparkov"]
    assert c.spec_for("cc_num").role is Role.ENTITY_KEY
    assert c.spec_for("unix_time").role is Role.ORDER_KEY
    assert c.spec_for("trans_date_trans_time").role is Role.EVENT_TIME
    assert c.spec_for("trans_num").role is Role.ROW_ID
    assert c.spec_for("Unnamed: 0").role is Role.EXCLUDED
    for pii in ("first", "last", "street", "zip", "dob"):
        assert c.spec_for(pii).role is Role.PII
    for near_id in ("city", "job"):                 # 1A.3: near-identifiers excluded
        assert c.spec_for(near_id).role is Role.EXCLUDED
    assert c.claim(FeatureFamily.ENTITY_HISTORY).support is Support.SUPPORTED


def test_ieee_families_and_naming():
    c = CONTRACTS["ieee"]
    assert c.spec_for("V1").role is Role.OPAQUE and c.spec_for("V339").kind is Kind.FLOAT
    assert c.spec_for("V340") is None and c.spec_for("V0") is None
    assert c.spec_for("C14") is not None and c.spec_for("C15") is None
    assert c.spec_for("D15") is not None and c.spec_for("D16") is None
    assert c.spec_for("id_01").kind is Kind.FLOAT and c.spec_for("id-01").kind is Kind.FLOAT
    assert c.spec_for("id_30").kind is Kind.STRING and c.spec_for("id-30").kind is Kind.STRING
    assert c.spec_for("id_39") is None
    assert canonical_name("id-07") == "id_07" and canonical_name("DeviceType") == "DeviceType"
    assert c.entity_key is None                      # no proxy entity key by decision
    assert c.claim(FeatureFamily.ENTITY_HISTORY).support is Support.UNSUPPORTED
    assert c.claim(FeatureFamily.CYCLIC_RELATIVE_TIME).support is Support.SUPPORTED   # 1A.4: anchored by D9
    assert c.claim(FeatureFamily.CALENDAR_TIME).support is Support.UNSUPPORTED
    assert c.spec_for("card1").role is Role.OPAQUE   # never an entity key


def test_ulb_families():
    c = CONTRACTS["ulb"]
    assert c.spec_for("V28") is not None and c.spec_for("V29") is None
    assert c.entity_key is None and c.row_id is None and c.event_time is None
    assert c.claim(FeatureFamily.CALENDAR_TIME).support is Support.UNSUPPORTED
    assert c.claim(FeatureFamily.OPAQUE_MASKED).support is Support.SUPPORTED


def test_resolve_reports_unexpected_columns():
    c = CONTRACTS["ulb"]
    specs, unexpected = c.resolve(["Time", "V1", "Amount", "Class", "bogus"])
    assert [s.name for s in specs] == ["Time", "V1", "Amount", "Class"]
    assert unexpected == ["bogus"]
