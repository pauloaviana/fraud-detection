"""Sparkov (Kaggle kartik2112/fraud-detection) — synthetic, production-like card transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import (
    CapabilityClaim, ColumnSpec, DatasetContract, FeatureFamily as F, FileSpec, Kind as K, Role as R,
    Support as S,
)
from ..findings import Finding, Severity as Sev
from .base import RawAdapter

_ENTITY_CONSTANT = ("first", "last", "gender", "street", "city", "state", "zip", "lat", "long",
                    "city_pop", "job", "dob")

CONTRACT = DatasetContract(
    name="sparkov",
    title="Sparkov simulated credit-card transactions (2019-01-01 .. 2020-12-31)",
    source="https://www.kaggle.com/datasets/kartik2112/fraud-detection/",
    files=[
        FileSpec("train", "fraudTrain.csv", "sparkov.zip", labeled=True,
                 purpose="labeled, 2019-01-01 .. 2020-06-21 12:13"),
        FileSpec("test", "fraudTest.csv", "sparkov.zip", labeled=True,
                 purpose="labeled, 2020-06-21 12:14 .. 2020-12-31 (official split, chronologically later)"),
    ],
    columns=[
        ColumnSpec("Unnamed: 0", K.INT, R.EXCLUDED, description="positional index written by the generator",
                   notes="restarts at 0 in each file; carries no information"),
        ColumnSpec("trans_date_trans_time", K.DATETIME, R.EVENT_TIME, description="transaction timestamp",
                   notes="no timezone; a single clock for customers in 51 states — hour is not local hour"),
        ColumnSpec("cc_num", K.STRING, R.ENTITY_KEY, description="simulated card number",
                   notes="1:1 with (first,last,dob,street); history construction only, never an input"),
        ColumnSpec("merchant", K.STRING, R.GROUP_KEY, description="merchant name",
                   notes="every value carries a 'fraud_' prefix — Faker generator artefact, not a label"),
        ColumnSpec("category", K.STRING, R.GROUP_KEY, description="merchant category (14 values)"),
        ColumnSpec("amt", K.FLOAT, R.INPUT, description="transaction amount"),
        ColumnSpec("first", K.STRING, R.PII, description="customer first name"),
        ColumnSpec("last", K.STRING, R.PII, description="customer last name"),
        ColumnSpec("gender", K.STRING, R.INPUT, description="customer gender (F/M)",
                   notes="constant per entity; sensitive attribute — it selects the generator profile, so it is "
                         "a legitimate DEMOGRAPHICS basis here; direct use in a real system is a policy decision"),
        ColumnSpec("street", K.STRING, R.PII, description="customer street address",
                   notes="equivalent to the entity key"),
        ColumnSpec("city", K.STRING, R.EXCLUDED, description="customer home city",
                   notes="1A.3: excluded — 894 values for 983 cards, constant per entity: a near-identifier with "
                         "no semantics beyond lat/long/state"),
        ColumnSpec("state", K.STRING, R.GROUP_KEY, description="customer home state",
                   notes="constant per entity"),
        ColumnSpec("zip", K.STRING, R.PII, description="customer home ZIP",
                   notes="constant per entity; redundant with lat/long"),
        ColumnSpec("lat", K.FLOAT, R.INPUT, description="customer home latitude",
                   notes="constant per entity; basis for customer<->merchant distance"),
        ColumnSpec("long", K.FLOAT, R.INPUT, description="customer home longitude",
                   notes="constant per entity"),
        ColumnSpec("city_pop", K.INT, R.INPUT, description="population of the customer's city",
                   notes="constant per entity"),
        ColumnSpec("job", K.STRING, R.EXCLUDED, description="customer occupation (~500 values)",
                   notes="1A.3: excluded — 494 values, constant per entity: near-identifier; generator assigns it "
                         "independently of behaviour"),
        ColumnSpec("dob", K.DATETIME, R.PII, description="customer date of birth",
                   notes="age at event time is derivable (DEMOGRAPHICS); dob itself is not an input"),
        ColumnSpec("trans_num", K.STRING, R.ROW_ID, description="transaction id (32 hex chars)"),
        ColumnSpec("unix_time", K.INT, R.ORDER_KEY, description="epoch seconds",
                   notes="equals trans_date_trans_time read as UTC minus exactly 7 years: wall-clock matches, "
                         "weekday does not — ordering and deltas only, never calendar features"),
        ColumnSpec("merch_lat", K.FLOAT, R.INPUT, description="merchant latitude (drawn per transaction)"),
        ColumnSpec("merch_long", K.FLOAT, R.INPUT, description="merchant longitude (drawn per transaction)"),
        ColumnSpec("is_fraud", K.INT, R.TARGET, description="fraud label (simulated)"),
    ],
    target="is_fraud", row_id="trans_num", entity_key="cc_num", order_key="unix_time",
    event_time="trans_date_trans_time",
    role_in_suite=("PIPELINE-ENGINEERING dataset: exercises the complete 1A feature pipeline (entity history, "
                   "velocity, context, periodic time). NOT evidence of realistic fraud signal — labels and "
                   "fraud patterns are generator artefacts."),
    contract_version="1A.4", dedup_key=None,
    capabilities=[
        CapabilityClaim(F.RAW_TRANSACTION, S.SUPPORTED, ("amt", "category"), "amount and category present"),
        CapabilityClaim(F.ENTITY_HISTORY, S.SUPPORTED, ("cc_num", "unix_time", "amt"),
                        "stable card key with 6..3123 transactions per card over 18 months"),
        CapabilityClaim(F.ENTITY_HISTORY_GROUPED, S.SUPPORTED, ("cc_num", "category", "merchant", "state"),
                        "grouping criteria available (category, merchant; state is constant per entity)"),
        CapabilityClaim(F.PERIODIC_TIME_ENTITY, S.SUPPORTED, ("cc_num", "trans_date_trans_time"),
                        "per-card time-of-day history available; clock is not local time (tz unknown)"),
        CapabilityClaim(F.CALENDAR_TIME, S.PARTIAL, ("trans_date_trans_time",),
                        "wall-clock timestamp present but timezone unknown: hour/weekday are 'dataset clock', "
                        "not the customer's local time"),
        CapabilityClaim(F.RELATIVE_TIME, S.SUPPORTED, ("unix_time",), "monotone epoch seconds"),
        CapabilityClaim(F.CYCLIC_RELATIVE_TIME, S.SUPPORTED, ("trans_date_trans_time",),
                        "superseded by CALENDAR_TIME (absolute clock available)"),
        CapabilityClaim(F.GEO_DISTANCE, S.SUPPORTED, ("lat", "long", "merch_lat", "merch_long"),
                        "customer home and per-transaction merchant coordinates"),
        CapabilityClaim(F.DEMOGRAPHICS, S.SUPPORTED, ("gender", "dob", "city_pop"),
                        "gender, age (from dob) and city_pop are the generator's profile selectors; job and city "
                        "are excluded as near-identifiers (1A.3)"),
        CapabilityClaim(F.MERCHANT_CONTEXT, S.SUPPORTED, ("merchant", "category"), "693 merchants, 14 categories"),
        CapabilityClaim(F.CARD_META, S.UNSUPPORTED, (), "no network / card-type / issuer fields"),
        CapabilityClaim(F.EMAIL_DOMAIN, S.UNSUPPORTED, (), "no email fields"),
        CapabilityClaim(F.DEVICE_IDENTITY, S.UNSUPPORTED, (), "no device / channel fields"),
        CapabilityClaim(F.OPAQUE_MASKED, S.UNSUPPORTED, (), "no masked columns"),
        CapabilityClaim(F.COST_SENSITIVE_EVAL, S.SUPPORTED, ("amt",), "amount available per transaction"),
    ],
    notes=[
        "Synthetic (Sparkov generator + Faker): fraud is injected by the generator; patterns such as the "
        "night-hour and high-amount concentration are generator behaviour, not transferable findings.",
        "Generator artefacts on record (1A.1–1A.3): one 1-day fraud episode per card for ~98 % of cards; episode "
        "dates contain fraud rows only; ~85 % of fraud at 22:00–03:59; fraud-profile amounts 3–100× larger by "
        "category; 2–16 fraud tx/day vs 1–6; 'fraud_' merchant prefix; unix_time shifted −7 years; every "
        "customer attribute constant per card. Any metric on this dataset measures the pipeline, not fraud.",
        "Official train/test files are chronologically contiguous and non-overlapping; trans_num is unique "
        "(no duplicate concept).",
        "1A.4 — calendar features (hour, weekday, month, cyclic encodings) come from trans_date_trans_time only; "
        "unix_time is the order key for splitting and deltas and is never an input. fraudTest.csv is the sealed "
        "official labeled test; the primary split lives inside fraudTrain.csv.",
    ],
)


class SparkovAdapter(RawAdapter):
    contract = CONTRACT

    def checks(self, frames: dict[str, pd.DataFrame]) -> list[Finding]:
        out: list[Finding] = []
        tr, te = frames["train"], frames.get("test")

        # unix_time vs wall-clock timestamp
        epoch_utc = tr["trans_date_trans_time"].astype("int64") // 10**9
        offset_days = ((epoch_utc - tr["unix_time"]) / 86400.0).round(3)
        offs = sorted(offset_days.unique().tolist())
        wd_mismatch = float((tr["trans_date_trans_time"].dt.dayofweek
                             != pd.to_datetime(tr["unix_time"], unit="s").dt.dayofweek).mean())
        out.append(Finding(Sev.WARN, "sparkov.unix_time_offset",
                           "unix_time is the wall-clock timestamp shifted by a constant −7 years "
                           "(2555/2556 days); weekday differs for most rows",
                           {"offset_days_values": offs, "weekday_mismatch_fraction": round(wd_mismatch, 4)}))

        # merchant prefix artefact
        frac = float(tr["merchant"].str.startswith("fraud_").mean())
        out.append(Finding(Sev.NOTE, "sparkov.merchant_prefix",
                           f"{frac:.1%} of merchant names start with 'fraud_' — generator artefact, "
                           "not a label; strip only as a cosmetic normalisation", {"fraction": frac}))

        # entity key <-> customer attributes
        per_card = tr.groupby("cc_num", sort=False)[list(_ENTITY_CONSTANT)].nunique().max()
        cards_per_person = tr.groupby(["first", "last", "dob"], sort=False)["cc_num"].nunique().max()
        out.append(Finding(Sev.NOTE, "sparkov.entity_constant_attributes",
                           "every customer attribute is constant per cc_num and each person has one card: "
                           "these attributes identify the entity; as direct inputs they invite memorisation",
                           {"max_distinct_per_cc_num": {k: int(v) for k, v in per_card.items()},
                            "max_cards_per_person": int(cards_per_person)}))

        # chronology and continuity
        for key, df in (("train", tr), ("test", te)):
            if df is None:
                continue
            mono = bool(df["unix_time"].is_monotonic_increasing)
            if not mono:
                out.append(Finding(Sev.WARN, f"sparkov.{key}.not_sorted", f"{key} is not sorted by unix_time"))
        if te is not None:
            gap = int(te["unix_time"].min() - tr["unix_time"].max())
            out.append(Finding(Sev.INFO, "sparkov.split_continuity",
                               "official test starts right after train ends (chronological split)",
                               {"gap_seconds": gap, "train_end": str(tr["trans_date_trans_time"].max()),
                                "test_start": str(te["trans_date_trans_time"].min())}))
            tr_cards, te_cards = set(tr["cc_num"]), set(te["cc_num"])
            out.append(Finding(Sev.NOTE, "sparkov.entity_overlap",
                               "official test contains cards never seen in train (cold start)",
                               {"train_cards": len(tr_cards), "test_cards": len(te_cards),
                                "shared": len(tr_cards & te_cards), "test_only": len(te_cards - tr_cards),
                                "test_only_rows": int(te["cc_num"].isin(te_cards - tr_cards).sum()),
                                "new_merchants_in_test": len(set(te["merchant"]) - set(tr["merchant"])),
                                "categories_equal": set(te["category"]) == set(tr["category"])}))

        # generator behaviour: fraud by hour / amount
        hour = tr["trans_date_trans_time"].dt.hour
        by_hour = tr.groupby(hour)["is_fraud"].mean()
        night = float(by_hour.loc[[22, 23, 0, 1, 2, 3]].mean())
        day = float(by_hour.drop([22, 23, 0, 1, 2, 3]).mean())
        out.append(Finding(Sev.RISK, "sparkov.fraud_hour_concentration",
                           "fraud rate at 22:00–03:59 is ~an order of magnitude above other hours — "
                           "generator behaviour; hour-of-day will look unrealistically predictive",
                           {"night_rate": round(night, 5), "other_hours_rate": round(day, 5),
                            "ratio": round(night / day, 1),
                            "by_hour_pct": {int(h): round(v * 100, 3) for h, v in by_hour.items()}}))
        med = tr.groupby("is_fraud")["amt"].median()
        out.append(Finding(Sev.RISK, "sparkov.fraud_amount_shift",
                           "fraud amounts are far larger than legitimate ones (generator behaviour)",
                           {"median_amt_legit": float(med.loc[0]), "median_amt_fraud": float(med.loc[1])}))

        # fraud burstiness per card (generator injects fraud episodes)
        fr = tr[tr["is_fraud"] == 1]
        per_card_fraud = fr.groupby("cc_num").size()
        span = fr.groupby("cc_num")["unix_time"].agg(lambda s: (s.max() - s.min()) / 86400.0)
        out.append(Finding(Sev.NOTE, "sparkov.fraud_episodes",
                           "fraud comes in short per-card episodes (many frauds per card within days)",
                           {"cards_with_fraud": int(per_card_fraud.size),
                            "frauds_per_card_median": float(per_card_fraud.median()),
                            "frauds_per_card_max": int(per_card_fraud.max()),
                            "episode_span_days_median": float(np.round(span.median(), 2)),
                            "episode_span_days_p90": float(np.round(span.quantile(0.9), 2))}))
        return out
