"""IEEE-CIS / Vesta (Kaggle ieee-fraud-detection) — real e-commerce transactions, masked schema."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..contracts import (
    CapabilityClaim, ColumnFamily, ColumnSpec, DatasetContract, FeatureFamily as F, FileSpec, Kind as K,
    Role as R, Support as S,
)
from ..findings import Finding, Severity as Sev
from .base import RawAdapter

ZIP = "ieee-fraud-detection.zip"

# identity columns that hold strings (the rest of id_01..id_38 are numeric)
_ID_STRING = {12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38}
_ID_NUMERIC = sorted(set(range(1, 39)) - _ID_STRING)


def _alts(nums: list[int]) -> str:
    return "|".join(f"{n:02d}" for n in nums)


CONTRACT = DatasetContract(
    name="ieee",
    title="IEEE-CIS Fraud Detection (Vesta) — masked real transactions, ~6 months + ~6 months",
    source="https://www.kaggle.com/competitions/ieee-fraud-detection/",
    files=[
        FileSpec("train", "train_transaction.csv", ZIP, labeled=True, purpose="labeled transactions"),
        FileSpec("train_identity", "train_identity.csv", ZIP, labeled=False,
                 purpose="identity side-table for train (24% of transactions), join on TransactionID"),
        FileSpec("test", "test_transaction.csv", ZIP, labeled=False,
                 purpose="UNLABELED official test (Kaggle); inference / schema / drift only"),
        FileSpec("test_identity", "test_identity.csv", ZIP, labeled=False,
                 purpose="identity side-table for test; column names use 'id-NN' instead of 'id_NN'"),
    ],
    columns=[
        ColumnSpec("TransactionID", K.INT, R.ROW_ID, description="transaction id; join key to identity"),
        ColumnSpec("isFraud", K.INT, R.TARGET, description="fraud label (chargeback-derived, per Vesta)"),
        ColumnSpec("TransactionDT", K.INT, R.ORDER_KEY, description="seconds from an undisclosed reference",
                   notes="relative clock: no calendar semantics; a 24 h cycle exists (see D9) but its phase is unknown"),
        ColumnSpec("TransactionAmt", K.FLOAT, R.INPUT, description="amount in USD",
                   notes="a subset has 3 decimals (currency-converted), itself informative"),
        ColumnSpec("ProductCD", K.STRING, R.GROUP_KEY, description="product code (W,C,H,R,S); meaning masked"),
        ColumnSpec("card1", K.INT, R.OPAQUE, description="card code (13.5k values)",
                   notes="high-cardinality code; NOT an entity key (no proxy keys by decision)"),
        ColumnSpec("card2", K.FLOAT, R.OPAQUE, nullable=True, description="card code"),
        ColumnSpec("card3", K.FLOAT, R.OPAQUE, nullable=True, description="card code (issuer country-like)"),
        ColumnSpec("card4", K.STRING, R.INPUT, nullable=True, description="card network"),
        ColumnSpec("card5", K.FLOAT, R.OPAQUE, nullable=True, description="card code"),
        ColumnSpec("card6", K.STRING, R.INPUT, nullable=True, description="debit / credit"),
        ColumnSpec("addr1", K.FLOAT, R.OPAQUE, nullable=True, description="masked billing region code"),
        ColumnSpec("addr2", K.FLOAT, R.OPAQUE, nullable=True, description="masked billing country code"),
        ColumnSpec("dist1", K.FLOAT, R.OPAQUE, nullable=True, description="distance between undisclosed endpoints"),
        ColumnSpec("dist2", K.FLOAT, R.OPAQUE, nullable=True, description="distance between undisclosed endpoints"),
        ColumnSpec("P_emaildomain", K.STRING, R.INPUT, nullable=True, description="purchaser email domain"),
        ColumnSpec("R_emaildomain", K.STRING, R.INPUT, nullable=True, description="recipient email domain"),
        ColumnSpec("DeviceType", K.STRING, R.INPUT, nullable=True, description="desktop / mobile"),
        ColumnSpec("DeviceInfo", K.STRING, R.INPUT, nullable=True, description="device string (high cardinality)"),
    ],
    families=[
        ColumnFamily("C", r"C([1-9]|1[0-4])", K.FLOAT, R.OPAQUE, dtype="float32",
                     description="vendor counts (e.g. addresses associated with the card)",
                     notes="pre-aggregated by Vesta; point-in-time computation not documented"),
        ColumnFamily("D", r"D([1-9]|1[0-5])", K.FLOAT, R.OPAQUE, dtype="float32",
                     description="vendor timedeltas in days (e.g. since previous transaction)",
                     notes="D9 is a day fraction (hour/24); others masked"),
        ColumnFamily("M", r"M[1-9]", K.STRING, R.OPAQUE, description="match flags T/F (M4: M0/M1/M2)"),
        ColumnFamily("V", r"V([1-9]|[1-9]\d|[12]\d\d|3[0-3]\d)", K.FLOAT, R.OPAQUE, dtype="float32",
                     description="Vesta engineered features (ranking, counting, entity relations)",
                     notes="missing in blocks; semantics masked"),
        ColumnFamily("id_num", rf"id[_-]({_alts(_ID_NUMERIC)})", K.FLOAT, R.OPAQUE, dtype="float32",
                     description="identity vendor numeric scores / codes"),
        ColumnFamily("id_str", rf"id[_-]({_alts(sorted(_ID_STRING))})", K.STRING, R.INPUT,
                     description="identity categorical: OS (id_30), browser (id_31), resolution (id_33), flags"),
    ],
    target="isFraud", row_id="TransactionID", join_key="TransactionID", order_key="TransactionDT",
    role_in_suite=("PRODUCTION REALITY CHECK: real masked transactions; use every defensible 1A feature the "
                   "masked schema and relative chronology support; no invented entity, time-of-day or device "
                   "semantics."),
    contract_version="1A.4", dedup_key=None,
    capabilities=[
        CapabilityClaim(F.RAW_TRANSACTION, S.SUPPORTED, ("TransactionAmt", "ProductCD"), "amount and product code"),
        CapabilityClaim(F.ENTITY_HISTORY, S.UNSUPPORTED, ("card1", "addr1"),
                        "no customer/card identifier; card1+addr1 style proxies are ruled out by decision"),
        CapabilityClaim(F.ENTITY_HISTORY_GROUPED, S.UNSUPPORTED, (), "requires an entity key"),
        CapabilityClaim(F.PERIODIC_TIME_ENTITY, S.UNSUPPORTED, (), "requires an entity key"),
        CapabilityClaim(F.CALENDAR_TIME, S.UNSUPPORTED, ("TransactionDT",),
                        "relative clock only; absolute date/hour would be invented"),
        CapabilityClaim(F.RELATIVE_TIME, S.SUPPORTED, ("TransactionDT",), "monotone relative seconds"),
        CapabilityClaim(F.CYCLIC_RELATIVE_TIME, S.SUPPORTED, ("TransactionDT", "D9"),
                        "1A.4: daily phase is ANCHORED by the data — floor((TransactionDT mod 86400)/3600) equals "
                        "D9*24 on 100 % of rows where D9 is present, so TransactionDT mod 86400 is Vesta's own "
                        "hour-of-day for every row (not a calendar/timezone claim). A 7-day cycle is real "
                        "(hourly-volume autocorrelation 0.83 at lag 168 h) but the weekday phase is unknown: "
                        "day-of-cycle index only. Raw TransactionDT is never an input."),
        CapabilityClaim(F.GEO_DISTANCE, S.UNSUPPORTED, ("dist1", "dist2"),
                        "no coordinates; dist1/dist2 are opaque and usable only as OPAQUE inputs"),
        CapabilityClaim(F.DEMOGRAPHICS, S.UNSUPPORTED, (), "no demographic fields"),
        CapabilityClaim(F.MERCHANT_CONTEXT, S.PARTIAL, ("ProductCD",), "product code only; no merchant id"),
        CapabilityClaim(F.CARD_META, S.SUPPORTED, ("card4", "card6", "card1", "card2", "card3", "card5"),
                        "network and debit/credit explicit; card1/2/3/5 are opaque codes"),
        CapabilityClaim(F.EMAIL_DOMAIN, S.SUPPORTED, ("P_emaildomain", "R_emaildomain"),
                        "purchaser 84% present, recipient 23%"),
        CapabilityClaim(F.DEVICE_IDENTITY, S.PARTIAL, ("DeviceType", "DeviceInfo", "id_30", "id_31"),
                        "identity table covers only 24% of transactions, skewed by ProductCD"),
        CapabilityClaim(F.OPAQUE_MASKED, S.SUPPORTED, ("C1", "D1", "M1", "V1"), "C/D/M/V/id blocks"),
        CapabilityClaim(F.COST_SENSITIVE_EVAL, S.SUPPORTED, ("TransactionAmt",), "amount available"),
    ],
    notes=[
        "Official test is unlabeled: all evaluation must happen inside train_transaction.",
        "Test identity file uses 'id-NN' column names; the adapter keeps raw names and exposes canonical_name().",
        "1A.3 — labels: the released labeled set is FINALIZED; Vesta's 120-day chargeback rule is provenance "
        "metadata (labels.IEEE_TARGET.documented_maturation_seconds), never a row filter.",
        "1A.3 — label-derived history: 'prior fraud on any linkage key (card/addr/email)' is label-derived and "
        "excluded from ordinary inference features (75 % of frauds have an earlier fraud on the same key — "
        "that is Vesta's propagation rule, not a signal).",
        "1A.3 — C/D/M/V blocks stay OPAQUE. Vesta's own description (Kaggle discussion 101203, paraphrased): "
        "C = counting (e.g. addresses associated with the card), D = timedeltas (e.g. days since previous "
        "transaction), M = match flags (e.g. names on card and address), V = 'engineered rich features, "
        "including ranking, counting and other entity relations' — all with 'actual meaning masked'. That "
        "justifies the family-level kinds, not per-column semantics; D9 = k/24 is observed, its use is the "
        "deferred cyclic-time decision. Point-in-time correctness of C/D/V is undocumented.",
        "TransactionID is unique in every file (no duplicate concept).",
        "1A.4 — missingness: indicator columns for every column with a null in the training part, train-median "
        "imputation for numerics, missing/unseen categories -> code 0; identity is left-joined (has_identity "
        "recorded; presence is scoring-time information). No stronger semantics for C/D/M/V.",
    ],
)

_ID_RE = re.compile(r"^id-(\d\d)$")


def canonical_name(col: str) -> str:
    """Map test_identity's 'id-NN' to train's 'id_NN'. Pure function; raw frames are never renamed."""
    m = _ID_RE.match(col)
    return f"id_{m.group(1)}" if m else col


class IEEEAdapter(RawAdapter):
    contract = CONTRACT

    def checks(self, frames: dict[str, pd.DataFrame]) -> list[Finding]:
        out: list[Finding] = []
        tr = frames["train"]
        ident = frames.get("train_identity")
        te = frames.get("test")

        # header consistency across files
        h_tr, h_te = self.header("train"), self.header("test")
        diff = sorted(set(h_tr) ^ set(h_te))
        out.append(Finding(Sev.INFO if diff == ["isFraud"] else Sev.WARN, "ieee.transaction_headers",
                           "train/test transaction headers differ only by isFraud" if diff == ["isFraud"]
                           else "train/test transaction headers differ", {"symmetric_difference": diff}))
        h_id_tr, h_id_te = self.header("train_identity"), self.header("test_identity")
        renamed = [c for c in h_id_te if canonical_name(c) != c]
        out.append(Finding(Sev.WARN, "ieee.identity_header_naming",
                           f"test_identity uses hyphenated names for {len(renamed)} columns (id-NN vs id_NN); "
                           "canonical_name() reconciles them without renaming raw frames",
                           {"train_identity_equals_canonical_test_identity":
                                [canonical_name(c) for c in h_id_te] == h_id_tr}))

        # chronology
        out.append(Finding(Sev.INFO, "ieee.train_sorted", "train is sorted by TransactionDT",
                           {"monotone": bool(tr["TransactionDT"].is_monotonic_increasing)}))
        if te is not None:
            out.append(Finding(Sev.INFO, "ieee.time_ranges",
                               "relative clock ranges; official test starts ~30 days after train ends",
                               {"train_days": [float(tr["TransactionDT"].min() / 86400),
                                               float(tr["TransactionDT"].max() / 86400)],
                                "test_days": [float(te["TransactionDT"].min() / 86400),
                                              float(te["TransactionDT"].max() / 86400)],
                                "gap_days": float((te["TransactionDT"].min() - tr["TransactionDT"].max()) / 86400)}))

        # identity join
        if ident is not None:
            ids = ident["TransactionID"]
            has_id = tr["TransactionID"].isin(set(ids))
            fr_with, fr_without = float(tr.loc[has_id, "isFraud"].mean()), float(tr.loc[~has_id, "isFraud"].mean())
            cov_by_pcd = tr.assign(_h=has_id).groupby("ProductCD")["_h"].mean().round(3).to_dict()
            out.append(Finding(Sev.NOTE, "ieee.identity_join",
                               "identity rows cover 24% of transactions; presence is strongly associated with "
                               "fraud and with ProductCD (vendor collection policy, available at scoring time)",
                               {"identity_rows": int(len(ident)), "join_key_unique": bool(ids.is_unique),
                                "all_in_train": bool(ids.isin(set(tr["TransactionID"])).all()),
                                "coverage": round(float(has_id.mean()), 4),
                                "fraud_rate_with_identity": round(fr_with, 4),
                                "fraud_rate_without_identity": round(fr_without, 4),
                                "coverage_by_ProductCD": cov_by_pcd}))

        # amount
        dec = tr["TransactionAmt"].astype(str).str.split(".").str[1].str.len().fillna(0).astype(int)
        med = tr.groupby("isFraud")["TransactionAmt"].median()
        out.append(Finding(Sev.NOTE, "ieee.amount_decimals",
                           "TransactionAmt has 1–3 decimals; 3-decimal amounts indicate currency conversion",
                           {"decimals_counts": {int(k): int(v) for k, v in dec.value_counts().sort_index().items()},
                            "fraud_rate_by_decimals": {int(k): round(float(v), 4) for k, v in
                                                       tr.groupby(dec)["isFraud"].mean().items()},
                            "median_amt_legit": float(med.loc[0]), "median_amt_fraud": float(med.loc[1])}))

        # D9 = hour fraction
        d9 = tr["D9"].dropna()
        vals = np.sort(d9.unique())
        out.append(Finding(Sev.NOTE, "ieee.d9_hour_fraction",
                           "D9 takes 24 values equal to k/24 — a masked hour-of-day on the relative clock",
                           {"n_values": int(len(vals)), "all_k_over_24": bool(np.allclose(vals * 24, np.round(vals * 24), atol=1e-3)),
                            "present_fraction": round(float(tr["D9"].notna().mean()), 4)}))

        # ProductCD / card4 / card6
        out.append(Finding(Sev.INFO, "ieee.product_and_card",
                           "fraud rate varies strongly by ProductCD",
                           {"fraud_rate_by_ProductCD": tr.groupby("ProductCD")["isFraud"].mean().round(4).to_dict(),
                            "card4_counts": tr["card4"].value_counts(dropna=False).to_dict(),
                            "card6_counts": tr["card6"].value_counts(dropna=False).to_dict()}))

        # missingness blocks in V
        vcols = [c for c in tr.columns if re.fullmatch(r"V\d+", c)]
        nulls = tr[vcols].isna().sum()
        blocks: dict[int, list[str]] = {}
        for c, n in nulls.items():
            blocks.setdefault(int(n), []).append(c)
        summary = {f"{v[0]}..{v[-1]} ({len(v)} cols)": round(n / len(tr), 4)
                   for n, v in sorted(blocks.items())}
        out.append(Finding(Sev.NOTE, "ieee.v_missing_blocks",
                           f"V columns are missing in {len(blocks)} distinct blocks (same null count within a block)",
                           {"null_fraction_by_block": summary}))

        # no entity key: cardinalities of candidate codes (information only)
        combo = tr[["card1", "card2", "addr1"]].astype(str).agg("|".join, axis=1).nunique()
        out.append(Finding(Sev.NOTE, "ieee.no_entity_key",
                           "no customer identifier; card1 / (card1,card2,addr1) cardinalities recorded for "
                           "reference only — proxies are not used",
                           {"card1_unique": int(tr["card1"].nunique()), "card1_card2_addr1_unique": int(combo),
                            "addr1_unique": int(tr["addr1"].nunique())}))

        # prevalence over time (drift inside train)
        block = (tr["TransactionDT"] // (30 * 86400)).astype(int)
        out.append(Finding(Sev.INFO, "ieee.prevalence_by_30d",
                           "fraud prevalence per 30-day block of the relative clock",
                           {"fraud_rate_pct": {int(k): round(float(v) * 100, 2)
                                               for k, v in tr.groupby(block)["isFraud"].mean().items()}}))
        return out
