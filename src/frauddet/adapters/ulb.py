"""ULB / MLG European credit-card dataset (Kaggle mlg-ulb/creditcardfraud) — PCA-anonymised, 2 days."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import (
    CapabilityClaim, ColumnFamily, ColumnSpec, DatasetContract, FeatureFamily as F, FileSpec, Kind as K,
    Role as R, Support as S,
)
from ..findings import Finding, Severity as Sev
from .base import RawAdapter

ALL_COLUMNS = ("Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class")

CONTRACT = DatasetContract(
    name="ulb",
    title="ULB European card transactions, September 2013 (2 days), PCA-anonymised",
    source="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/",
    files=[FileSpec("train", "creditcardfraud.csv", None, labeled=True,
                    purpose="the whole dataset (no official split)")],
    columns=[
        ColumnSpec("Time", K.FLOAT, R.ORDER_KEY, description="seconds since the first transaction",
                   notes="spans 172,792 s = 48 h; heavy ties (2.3 rows per distinct second on average)"),
        ColumnSpec("Amount", K.FLOAT, R.INPUT, description="transaction amount",
                   notes="1A.3: Amount = 0 rows (1,825; fraud-enriched at 1.5 %) are KEPT — no corruption "
                         "evidence (no inf/NaN, ordinary V spreads, spread evenly over both days); zero-amount "
                         "authorisations are a known card behaviour"),
        ColumnSpec("Class", K.INT, R.TARGET, description="fraud label"),
    ],
    families=[
        ColumnFamily("V", r"V([1-9]|1\d|2[0-8])", K.FLOAT, R.OPAQUE, nullable=False,
                     description="PCA components of undisclosed original features",
                     notes="Time and Amount were not transformed"),
    ],
    target="Class", order_key="Time",
    role_in_suite=("EXTREME-IMBALANCE BENCHMARK: PCA schema supports only amount + opaque inputs; used for "
                   "leakage-safe preprocessing, imbalance handling, feature selection and later model evaluation "
                   "(incl. the Ma-2026 reproducibility benchmark)."),
    contract_version="1A.4",
    dedup_key=ALL_COLUMNS,   # a TRUE duplicate = identical on all 31 columns, Time included (see notes)
    capabilities=[
        CapabilityClaim(F.RAW_TRANSACTION, S.PARTIAL, ("Amount",), "amount only; no type/category"),
        CapabilityClaim(F.ENTITY_HISTORY, S.UNSUPPORTED, (), "no customer/card identifier of any kind"),
        CapabilityClaim(F.ENTITY_HISTORY_GROUPED, S.UNSUPPORTED, (), "no entity key, no grouping context"),
        CapabilityClaim(F.PERIODIC_TIME_ENTITY, S.UNSUPPORTED, (), "no entity key"),
        CapabilityClaim(F.CALENDAR_TIME, S.UNSUPPORTED, ("Time",), "relative clock; start-of-day unknown"),
        CapabilityClaim(F.RELATIVE_TIME, S.SUPPORTED, ("Time",), "monotone seconds over 48 h"),
        CapabilityClaim(F.CYCLIC_RELATIVE_TIME, S.PARTIAL, ("Time",),
                        "1A.4: hour of the 24 h cycle counted from the dataset start (Time mod 86400). A diurnal "
                        "volume cycle is observed (autocorrelation 0.48 at lag 24 h over only two cycles) but its "
                        "phase is unknown, so this is a relative phase, not an hour of day. Weak by construction; "
                        "raw Time is never an input in the primary protocol (Ma-2026 benchmark keeps it, as Ma do)."),
        CapabilityClaim(F.GEO_DISTANCE, S.UNSUPPORTED, (), "no location data"),
        CapabilityClaim(F.DEMOGRAPHICS, S.UNSUPPORTED, (), "no demographic data"),
        CapabilityClaim(F.MERCHANT_CONTEXT, S.UNSUPPORTED, (), "no merchant data"),
        CapabilityClaim(F.CARD_META, S.UNSUPPORTED, (), "no card metadata"),
        CapabilityClaim(F.EMAIL_DOMAIN, S.UNSUPPORTED, (), "n/a"),
        CapabilityClaim(F.DEVICE_IDENTITY, S.UNSUPPORTED, (), "n/a"),
        CapabilityClaim(F.OPAQUE_MASKED, S.SUPPORTED, ("V1", "V28"), "V1..V28 are the substance of the dataset"),
        CapabilityClaim(F.COST_SENSITIVE_EVAL, S.SUPPORTED, ("Amount",), "amount available"),
    ],
    notes=[
        "1A.3 — duplicates: a TRUE duplicate is a row identical on all 31 columns *including Time* "
        "(773 groups, 1,081 redundant rows, 19 of them fraud, labels never conflict). Rows identical on "
        "V1–V28 + Amount (+ Class) but at different Time (4,383 groups, 12,456 rows) are DISTINCT events: tiny "
        "recurring amounts (1.00, 1.98, 0.89, 9.99 …), consecutive gaps median 81 min, only 0.5 % under 2 s, "
        "no fraud — i.e. repeated identical purchases whose PCA'd source features carry no time. They are kept. "
        "Applying the dedup (keep-first on dedup_key) is a preprocessing step, not done here.",
        "1A.3 — Amount = 0 rows are kept (see the Amount column note).",
        "Only 492 positives; any evaluation slice carries high variance.",
        "Labels are a finalized benchmark extract (see labels.ULB_TARGET); no label timestamps exist.",
    ],
)


class ULBAdapter(RawAdapter):
    contract = CONTRACT

    def checks(self, frames: dict[str, pd.DataFrame]) -> list[Finding]:
        out: list[Finding] = []
        df = frames["train"]
        t = df["Time"]
        out.append(Finding(Sev.NOTE, "ulb.time_structure",
                           "Time is a relative clock over exactly two days with heavy ties",
                           {"span_hours": float(t.max() / 3600), "unique_values": int(t.nunique()),
                            "monotone": bool(t.is_monotonic_increasing),
                            "rows_per_hour": [int(x) for x in df.groupby((t // 3600).astype(int)).size()],
                            "fraud_per_hour": [int(x) for x in df.groupby((t // 3600).astype(int))["Class"].sum()]}))

        # --- duplicate taxonomy (1A.3 evidence) ---------------------------------------
        feat = [c for c in df.columns if c not in ("Time", "Class")]
        exact = df.duplicated(keep=False)
        exact_groups = df[exact].groupby(list(df.columns)).size()
        key = feat + ["Class"]
        same = df.duplicated(subset=key, keep=False)
        g = df[same].groupby(key)
        n_time = g["Time"].nunique()
        diff_t = n_time[n_time > 1]
        in_diff = df[same].set_index(key).index.isin(diff_t.index)
        dt = df[same][in_diff]
        gaps = dt.sort_values("Time").groupby(key)["Time"].diff().dropna()
        gaps = gaps[gaps > 0]
        out.append(Finding(Sev.WARN, "ulb.duplicate_taxonomy",
                           "TRUE duplicates = identical on all 31 columns incl. Time (dedup_key). Same-features-"
                           "different-Time groups are distinct recurring events and are kept",
                           {"exact": {"groups": int(len(exact_groups)), "rows": int(exact.sum()),
                                      "redundant_rows": int(df.duplicated().sum()),
                                      "redundant_fraud_rows": int((df.duplicated() & (df["Class"] == 1)).sum()),
                                      "group_sizes": {int(k): int(v) for k, v in exact_groups.value_counts().sort_index().items()},
                                      "label_conflicts": 0},
                            "same_features_different_time": {
                                "groups": int(len(diff_t)), "rows": int(len(dt)),
                                "fraud_rows": int((dt["Class"] == 1).sum()),
                                "consecutive_gap_s_quantiles": {str(q): float(v) for q, v in gaps.quantile([.1, .5, .9]).items()},
                                "share_gap_under_2s": round(float((gaps < 2).mean()), 4),
                                "median_amount": float(dt["Amount"].median()),
                                "top_amounts": {str(k): int(v) for k, v in dt["Amount"].value_counts().head(5).items()},
                                "largest_group": int(diff_t.max())},
                            "share_of_all_rows_sharing_their_second": round(float((df.groupby("Time")["Time"].transform("size") > 1).mean()), 3)}))

        zero = df["Amount"] == 0
        med = df.groupby("Class")["Amount"].median()
        out.append(Finding(Sev.NOTE, "ulb.amount",
                           "Amount = 0 rows are kept (1A.3): fraud-enriched but no corruption evidence",
                           {"zero_amount_rows": int(zero.sum()), "zero_amount_fraud_rate": round(float(df.loc[zero, "Class"].mean()), 4),
                            "zero_amount_inf_or_nan": int(np.isinf(df.loc[zero, feat]).sum().sum() + df.loc[zero, feat].isna().sum().sum()),
                            "zero_amount_per_day": [int(x) for x in df[zero].groupby((t[zero] // 86400).astype(int)).size()],
                            "overall_fraud_rate": round(float(df["Class"].mean()), 5),
                            "median_amt_legit": float(med.loc[0]), "median_amt_fraud": float(med.loc[1]),
                            "max_amt_fraud": float(df.loc[df["Class"] == 1, "Amount"].max())}))

        v = df[[c for c in df.columns if c.startswith("V")]]
        out.append(Finding(Sev.INFO, "ulb.pca_sanity",
                           "V columns are centred (PCA scores); variance decreases with index",
                           {"max_abs_mean": float(np.abs(v.mean()).max()),
                            "std_V1_V2_V27_V28": [round(float(v[c].std()), 3) for c in ("V1", "V2", "V27", "V28")]}))
        day = (t >= 86400).astype(int)
        out.append(Finding(Sev.INFO, "ulb.by_day",
                           "rows and positives per calendar-like day of the relative clock (information only)",
                           {"rows": [int(x) for x in df.groupby(day).size()],
                            "fraud": [int(x) for x in df.groupby(day)["Class"].sum()]}))
        return out
