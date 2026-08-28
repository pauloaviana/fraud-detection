"""Unit tests for the label semantics layer (no data files needed)."""

import pandas as pd
import pytest

from frauddet.adapters import ADAPTERS
from frauddet.contracts import Role
from frauddet.labels import DAY, TARGETS, LabelMechanism, MaturityPolicy, get_target


def test_registry_matches_adapters_and_contracts():
    assert set(TARGETS) == set(ADAPTERS)
    for name, spec in TARGETS.items():
        c = ADAPTERS[name].contract
        assert spec.column == c.target
        assert spec.order_key == c.order_key
        assert c.spec_for(spec.column).role is Role.TARGET


def test_mechanisms_are_distinct_and_documented():
    mechs = [s.provenance.mechanism for s in TARGETS.values()]
    assert len(set(mechs)) == 3 and set(mechs) == set(LabelMechanism)
    for s in TARGETS.values():
        p = s.provenance
        assert p.sources and all(src.url.startswith("http") for src in p.sources)
        assert p.assumptions and p.positive_definition and p.negative_definition and p.maturation
        assert p.label_timestamp_available is False        # none of the datasets ship label times


def test_maturity_policy_consistency():
    s = TARGETS["ieee"]
    assert s.maturity_policy is MaturityPolicy.FINALIZED
    assert s.documented_maturation_seconds == 120 * DAY and s.finalized_reason
    s = TARGETS["ulb"]
    assert s.maturity_policy is MaturityPolicy.FINALIZED
    assert s.documented_maturation_seconds == 7 * DAY and s.finalized_reason
    s = TARGETS["sparkov"]
    assert s.maturity_policy is MaturityPolicy.NOT_APPLICABLE_SIMULATED
    assert s.documented_maturation_seconds is None and s.finalized_reason is None


def test_binary_validates():
    s = get_target("ulb")
    y = s.binary(pd.DataFrame({"Class": [0, 1, 0]}))
    assert y.tolist() == [0, 1, 0] and str(y.dtype) == "int8" and y.name == "y"
    with pytest.raises(ValueError, match="null"):
        s.binary(pd.DataFrame({"Class": [0, None, 1]}))
    with pytest.raises(ValueError, match="unexpected"):
        s.binary(pd.DataFrame({"Class": [0, 2]}))
    with pytest.raises(KeyError):
        s.binary(pd.DataFrame({"isFraud": [0, 1]}))


def test_matured_mask_ieee_is_finalized_not_a_filter():
    s = get_target("ieee")
    df = pd.DataFrame({"TransactionDT": [0, 10 * DAY, 100 * DAY, 200 * DAY]})
    assert s.matured_mask(df, 220 * DAY).all()          # 120-day window is metadata only
    assert s.matured_mask(df, 0).all()
    with pytest.raises(ValueError):
        s.matured_mask(df, 220 * DAY, rehearsal_lag_seconds=DAY)


def test_matured_mask_sparkov_and_ulb():
    sp = get_target("sparkov")
    df = pd.DataFrame({"unix_time": [0, 5 * DAY, 30 * DAY]})
    assert sp.matured_mask(df, 10 * DAY).all()
    assert sp.matured_mask(df, 10 * DAY, rehearsal_lag_seconds=7 * DAY).tolist() == [True, False, False]
    ulb = get_target("ulb")
    d2 = pd.DataFrame({"Time": [0.0, 1000.0]})
    assert ulb.matured_mask(d2, 0).all()
    with pytest.raises(ValueError):
        ulb.matured_mask(d2, 0, rehearsal_lag_seconds=DAY)


def test_label_leak_guard():
    s = get_target("ieee")
    c = ADAPTERS["ieee"].contract
    s.assert_no_label_leak(["TransactionAmt", "ProductCD"], c)
    with pytest.raises(ValueError, match="isFraud"):
        s.assert_no_label_leak(["TransactionAmt", "isFraud"], c)
    with pytest.raises(ValueError):
        s.assert_no_label_leak(["isFraud"])          # works without the contract too


def test_metadata_is_serialisable():
    import json
    for s in TARGETS.values():
        m = s.metadata()
        json.dumps(m)
        assert m["provenance"]["mechanism"] in {x.value for x in LabelMechanism}
