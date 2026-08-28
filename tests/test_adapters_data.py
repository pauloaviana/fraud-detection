"""Adapter tests against the real files in data/ (skipped when the data is absent). Header-level and
tiny-nrows loads only — fast, read-only."""

from pathlib import Path

import pandas as pd
import pytest

from frauddet.adapters import ADAPTERS, get_adapter
from frauddet.adapters.ieee import canonical_name
from frauddet.contracts import Kind

DATA = Path(__file__).resolve().parents[1] / "data"


def _adapter(name):
    a = get_adapter(name, DATA)
    if not all(a.available(f.key) for f in a.contract.files):
        pytest.skip(f"{name}: data files not present")
    return a


@pytest.mark.parametrize("name", list(ADAPTERS))
def test_every_header_column_is_covered_by_the_contract(name):
    a = _adapter(name)
    for f in a.contract.files:
        _specs, unexpected = a.contract.resolve(a.header(f.key))
        assert unexpected == [], (f.key, unexpected)


@pytest.mark.parametrize("name", list(ADAPTERS))
def test_small_load_honours_declared_dtypes(name):
    a = _adapter(name)
    for f in a.contract.files:
        df = a.load(f.key, nrows=50)
        assert len(df) == 50
        for col in df.columns:
            s = a.contract.spec_for(col)
            if s.kind is Kind.DATETIME:
                assert pd.api.types.is_datetime64_any_dtype(df[col]), col
            elif s.kind is Kind.STRING:
                assert pd.api.types.is_string_dtype(df[col]), col
            else:
                assert pd.api.types.is_numeric_dtype(df[col]), col
        assert list(df.columns) == a.header(f.key)          # raw names are preserved


def test_ieee_test_identity_header_matches_train_after_canonicalisation():
    a = _adapter("ieee")
    assert [canonical_name(c) for c in a.header("test_identity")] == a.header("train_identity")
    assert "id-01" in a.header("test_identity")              # raw names untouched


def test_ieee_test_transaction_has_no_label():
    a = _adapter("ieee")
    assert "isFraud" not in a.header("test") and "isFraud" in a.header("train")


def test_sparkov_key_columns():
    a = _adapter("sparkov")
    df = a.load("train", nrows=5)
    assert df["cc_num"].dtype == "string"                    # never parsed as a number
    assert df["trans_date_trans_time"].dt.tz is None         # naive: timezone unknown, not assumed
