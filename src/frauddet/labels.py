"""Label semantics (Phase 1A.2).

Three datasets, three *different* label-generating mechanisms that merely share
a {0,1} encoding:

* Sparkov  — the label is a generator flag: fraud rows are produced by a separate
             "fraud profile" and stamped is_fraud=1 at generation time. No reporting,
             no investigation, no maturation exist.
* IEEE-CIS — Vesta's chargeback-based label with linkage propagation; the
             released labels are the finalized state after Vesta's 120-day window
             (chargeback timestamps are not shipped, so the window is provenance,
             not a row filter).
* ULB      — a finalized benchmark extract of Worldline's operational labels
             (investigator feedback on alerts + customer reports within a reaction
             window); no per-row label source or timestamp is published.

`TargetSpec` is the common interface: ``binary(df)`` gives the {0,1} target,
``matured_mask(df, as_of)`` is the (documented) maturity view — all-true for
finalized labels, with an optional *rehearsal* lag for the simulated dataset —
and ``assert_no_label_leak(columns)`` refuses any feature list that contains
label or post-resolution columns. Provenance and assumptions
travel with the spec as metadata and are never flattened away.

Nothing here splits, engineers features, rebalances or trains.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import pandas as pd

from .contracts import LABEL_ROLES, DatasetContract

DAY = 86_400


class LabelMechanism(str, Enum):
    SIMULATED = "simulated"                              # generator flag
    CHARGEBACK_REPORTED = "chargeback_reported"          # issuer chargeback + linkage propagation
    INVESTIGATOR_AND_DELAYED_REPORTS = "investigator_and_delayed_reports"  # alert feedback + customer reports


class MaturityPolicy(str, Enum):
    NOT_APPLICABLE_SIMULATED = "not_applicable_simulated"   # label exists at generation; nothing matures
    FINALIZED = "finalized"   # released labels are the final state; the maturation window (if any)
                              # is provenance metadata, never a row filter; reason documented


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    note: str = ""


@dataclass(frozen=True)
class LabelProvenance:
    mechanism: LabelMechanism
    positive_definition: str
    negative_definition: str
    maturation: str                       # how/when a label becomes final in the originating process
    propagation: str                      # does one label influence others? (linkage, episodes, ...)
    noise: str                            # known ways the label can be wrong
    label_timestamp_available: bool       # is a per-row label/resolution time shipped?
    sources: tuple[Source, ...]
    assumptions: tuple[str, ...]          # what we assume because the data does not say


@dataclass(frozen=True)
class TargetSpec:
    dataset: str
    column: str
    order_key: str                        # dataset clock used for maturity arithmetic
    provenance: LabelProvenance
    maturity_policy: MaturityPolicy
    documented_maturation_seconds: int | None  # the originating process's window, in order_key units;
                                               # metadata only — never applied as a row filter
    finalized_reason: str | None          # required when FINALIZED
    label_derived_columns: tuple[str, ...]  # columns carrying label / post-resolution information
    positive_value: int = 1
    negative_value: int = 0

    # -- common interface ---------------------------------------------------------
    def binary(self, df: pd.DataFrame) -> pd.Series:
        """The {0,1} target as int8. Raises on nulls or values outside {negative, positive}."""
        if self.column not in df.columns:
            raise KeyError(f"{self.dataset}: target column {self.column!r} not in frame (unlabeled file?)")
        y = df[self.column]
        if y.isna().any():
            raise ValueError(f"{self.dataset}: {int(y.isna().sum())} null labels — nulls are not 'legit'")
        allowed = {self.negative_value, self.positive_value}
        bad = set(pd.unique(y)) - allowed
        if bad:
            raise ValueError(f"{self.dataset}: unexpected label values {sorted(bad)} (allowed {sorted(allowed)})")
        return (y == self.positive_value).astype("int8").rename("y")

    def matured_mask(self, df: pd.DataFrame, as_of: float | int, *,
                     rehearsal_lag_seconds: int | None = None) -> pd.Series:
        """True where the row's label may be used at clock value ``as_of``.

        FINALIZED / SIMULATED: always True — the shipped labels are final; a documented maturation
        window is metadata, not a filter (see ``finalized_reason`` / ``documented_maturation_seconds``).
        SIMULATED only: ``rehearsal_lag_seconds`` imposes an *artificial* lag for pipeline rehearsal
        (Sparkov is the engineering dataset; this is not evidence about real label timing and is
        refused for finalized labels).
        """
        if rehearsal_lag_seconds is not None:
            if self.maturity_policy is not MaturityPolicy.NOT_APPLICABLE_SIMULATED:
                raise ValueError(f"{self.dataset}: rehearsal lag is only meaningful for simulated labels")
            return (df[self.order_key] + rehearsal_lag_seconds) <= as_of
        return pd.Series(True, index=df.index)

    def assert_no_label_leak(self, columns: list[str] | tuple[str, ...],
                             contract: DatasetContract | None = None) -> None:
        """Refuse a feature list that contains the label or any label-derived column."""
        offenders = [c for c in columns if c in self.label_derived_columns]
        if contract is not None:
            offenders += [c for c in columns if (s := contract.spec_for(c)) is not None
                          and s.role in LABEL_ROLES and c not in offenders]
        if offenders:
            raise ValueError(f"{self.dataset}: label / post-resolution columns cannot be inference "
                             f"features: {offenders}")

    def metadata(self) -> dict[str, Any]:
        d = asdict(self)
        d["provenance"]["mechanism"] = self.provenance.mechanism.value
        d["maturity_policy"] = self.maturity_policy.value
        return d


# ------------------------------------------------------------------------------ sources
_AWS_TFI = Source("AWS Fraud Detector — Transaction Fraud Insights",
                  "https://docs.aws.amazon.com/frauddetector/latest/ug/transaction-fraud-insights.html",
                  "\"ensure that the records that are used to train the model have had sufficient time to mature "
                  "... for chargeback fraud, it often takes 60 days or more to correctly identify fraudulent events. "
                  "For the best model performance, ensure that all records in your training dataset are mature.\"")
_AWS_DATASET = Source("AWS Fraud Detector — Event dataset",
                      "https://docs.aws.amazon.com/frauddetector/latest/ug/create-event-dataset.html",
                      "\"The maturity period is dependent on your business, and can take anywhere from two weeks to "
                      "three months ... determined by the chargeback period of the credit card or time taken by an "
                      "investigator to make determination.\" EVENT_LABEL requires LABEL_TIMESTAMP.")
_STRIPE = Source("Stripe — primer on ML for fraud protection",
                 "https://stripe.com/en-br/guides/primer-on-machine-learning-for-fraud-protection",
                 "labels come from cardholder disputes (chargebacks); blocked payments never get an outcome "
                 "(selective labels). No maturation window is stated.")
_ADYEN = Source("Adyen — risk field reference",
                "https://docs.adyen.com/risk-management/configure-your-risk-profile/risk-field-reference/",
                "all documented risk fields are authorization-time fields; chargeback / dispute / NOF data are "
                "post-transaction and documented elsewhere — never decision inputs.")
_UBER = Source("Uber — Mastermind", "https://www.uber.com/au/en/blog/mastermind/",
               "rule-execution architecture; no label semantics stated.")

# ------------------------------------------------------------------------------ Sparkov
_SPARKOV_GEN = "https://github.com/namebrandon/Sparkov_Data_Generation"
SPARKOV_PROVENANCE = LabelProvenance(
    mechanism=LabelMechanism.SIMULATED,
    positive_definition=(
        "Row produced by the customer's *fraud profile* during a generated fraud episode. Generator "
        "(datagen_transaction.py, current master): per customer `fraud_flag = random.randint(0,100)`; "
        "`if fraud_flag < 99:` one episode of `fraud_interval = random.randint(1,1)` day starting at a random "
        "date; `is_fraud = 1; temp_tx_data = fraud_profile.sample_from(is_fraud)`. The fraud profile "
        "(profiles/fraud_<profile>.json) has 2–16 tx/day (normal 1–6), much larger gamma amounts per category "
        "(e.g. shopping_net mean 1000 vs 100, misc_net 800 vs 100, grocery_pos 350 vs 200) and, in "
        "`sample_time`, an 80 % chance to restrict the hour to [0,4) or [22,24)."),
    negative_definition=(
        "Row produced by the customer's normal profile with `is_fraud = 0`; rows whose date falls in the "
        "customer's fraud dates are dropped (`if (is_fraud == 0 and t[1] not in fraud_dates) or is_fraud == 1`), "
        "so episode dates contain fraud rows only."),
    maturation="None. The label is decided at generation time; there is no report, dispute or investigation.",
    propagation=("Episode-level: all rows of one card on the episode date(s) are fraud by construction; the "
                 "label is a property of the (card, date) draw, not of the row."),
    noise=("No label noise in the sense of misreporting. Instead the label is *perfectly* aligned with "
           "generator artefacts (night hours, large amounts, burst counts) — an engineering dataset, not evidence "
           "of realistic fraud."),
    label_timestamp_available=False,
    sources=(Source("Sparkov_Data_Generation (generator source)", _SPARKOV_GEN,
                    "datagen_transaction.py, profile_weights.py::sample_time, profiles/fraud_*.json"),
             Source("Kaggle: kartik2112/fraud-detection", "https://www.kaggle.com/datasets/kartik2112/fraud-detection/",
                    "dataset page (JS-rendered; not readable here) — generated 2020-08-05 with the generator above")),
    assumptions=(
        "The Kaggle files were produced by a 2020 revision of the generator; the current master was read. "
        "Empirical checks in label_audit confirm the mechanism (one episode per card, no legit rows on episode "
        "dates, ~80 % night hours).",
        "Because ~98 % of customers get an episode somewhere in the generation window, 'cards with fraud' is not "
        "a risk attribute — it is a property of the generator."),
)
SPARKOV_TARGET = TargetSpec(
    dataset="sparkov", column="is_fraud", order_key="unix_time", provenance=SPARKOV_PROVENANCE,
    maturity_policy=MaturityPolicy.NOT_APPLICABLE_SIMULATED, documented_maturation_seconds=None,
    finalized_reason=None, label_derived_columns=("is_fraud",),
)

# ------------------------------------------------------------------------------ IEEE-CIS
IEEE_PROVENANCE = LabelProvenance(
    mechanism=LabelMechanism.CHARGEBACK_REPORTED,
    positive_definition=(
        "Vesta (competition host, Kaggle discussion 101203): \"The logic of our labeling is define reported "
        "chargeback on the card as fraud transaction (isFraud=1) and transactions posterior to it with either "
        "user account, email address or billing address directly linked to these attributes as fraud too.\""),
    negative_definition=(
        "\"If none of above is reported and found beyond 120 days, then we define as legit transaction "
        "(isFraud=0).\""),
    maturation=("Vesta's process: a label is final once 120 days have elapsed without a reported chargeback. "
                "The released training labels are that finalized state for every row (the file postdates the "
                "window for all of them). Because chargeback timestamps are not shipped, the window is recorded "
                "as provenance only — it is not applied as a row filter or an availability time."),
    propagation=("Linkage propagation: a chargeback on one transaction marks *posterior* transactions sharing "
                 "the user account, email address or billing address as fraud. Hence labels are not independent "
                 "across rows; any feature encoding 'prior fraud on this email/address' is label-derived history, "
                 "not an ordinary feature."),
    noise=("Host (paraphrased): fraud that is never reported, or reported after the claim period, stays "
           "labeled legit; the host considers such cases negligible. Chargebacks also include friendly fraud "
           "and non-fraud disputes are not separated."),
    label_timestamp_available=False,
    sources=(Source("Vesta host statement, Kaggle discussion 101203",
                    "https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203",
                    "key sentences verified verbatim via search snippet; page itself is JS-rendered"),
             Source("Kaggle competition data page", "https://www.kaggle.com/competitions/ieee-fraud-detection/",
                    "TransactionDT: \"timedelta from a given reference datetime (not an actual timestamp)\""),
             _AWS_TFI, _AWS_DATASET),
    assumptions=(
        "The 120 days are Vesta's rule on their own calendar; on TransactionDT's clock that is 120 × 86 400 s, "
        "recorded as documented_maturation_seconds for reference.",
        "Per-row chargeback dates are not shipped; the true resolution time of each label is unknown. The "
        "labels are therefore treated as finalized rather than simulated as 'known at t + 120 d'.",
        "The official test window (days 213–396) postdates train by 30 days; its labels were never released.",
    ),
)
IEEE_TARGET = TargetSpec(
    dataset="ieee", column="isFraud", order_key="TransactionDT", provenance=IEEE_PROVENANCE,
    maturity_policy=MaturityPolicy.FINALIZED, documented_maturation_seconds=120 * DAY,
    finalized_reason=("Released labels are the finalized outcome of Vesta's 120-day chargeback window; "
                      "chargeback timestamps are unavailable, so the window cannot be replayed per row and is "
                      "kept as provenance metadata rather than a filter."),
    label_derived_columns=("isFraud",),
)

# ------------------------------------------------------------------------------ ULB
ULB_PROVENANCE = LabelProvenance(
    mechanism=LabelMechanism.INVESTIGATOR_AND_DELAYED_REPORTS,
    positive_definition=(
        "Class = 1 \"in case of fraud\" (dataset card). The originating process (Worldline / ULB-MLG, "
        "Dal Pozzolo et al. 2015): investigators \"check the alerts by calling the cardholders, and then provide "
        "the FDS with feedbacks indicating whether the alerts were related to fraudulent or genuine "
        "transactions\"; frauds missed by the system are \"reported by customers themselves ... within a maximum "
        "time-interval of δ days\"."),
    negative_definition=(
        "\"all the transactions that customers do not report as frauds are considered genuine\" — after the "
        "reaction window (δ = 7 days assumed in the paper); i.e. not-reported ⇒ Class = 0."),
    maturation=("In the operational process labels mature over δ ≈ 7 days (verification latency). The Kaggle "
                "extract (2 days of September 2013, 284,807 rows, 492 frauds) is a *finalized* snapshot: labels "
                "were fixed long before publication (2015); no per-row label time or source (investigator vs "
                "customer report) is published."),
    propagation="Not documented for the extract. Alert feedback is per transaction; card-level effects unknown.",
    noise=("Unreported fraud within the window is labeled genuine; investigator feedback covers only the alerted "
           "fraction. PCA anonymisation removes any way to audit label consistency beyond duplicates."),
    label_timestamp_available=False,
    sources=(Source("Kaggle: mlg-ulb/creditcardfraud (via OpenML mirror d/1597)",
                    "https://www.openml.org/api/v1/json/data/1597",
                    "\"collected and analysed during a research collaboration of Worldline and the Machine Learning "
                    "Group of ULB\"; Class \"takes value 1 in case of fraud and 0 otherwise\""),
             Source("Dal Pozzolo, Boracchi, Caelen, Alippi, Bontempi (2015) — Credit card fraud detection and "
                    "concept-drift adaptation with delayed supervised information (IJCNN)",
                    "https://boracchi.faculty.polimi.it/docs/2015_04_Credit_Card_Fraud_Detection_DalPozzolo_Boracchi_Caelen_Alippi_Bontempi.pdf",
                    "same Worldline data stream (2013 dataset: 2013-09-05 .. 2014-01-18, ~160k tx/day, ~304 "
                    "frauds/day); δ = 7 assumed"),
             Source("Dal Pozzolo, Caelen, Johnson, Bontempi (2015) — Calibrating probability with undersampling "
                    "(CIDM)", "https://doi.org/10.1109/SSCI.2015.33", "the dataset's citation of record"),
             _AWS_DATASET),
    assumptions=(
        "The extract's labels are the operational labels after the reaction window had fully elapsed; treated as "
        "final. This cannot be verified from the file.",
        "The 48-hour span is shorter than any plausible maturation window (δ = 7 d, AWS 2 weeks–3 months), so "
        "point-in-time maturity cannot be enforced on this dataset even in principle.",
        "Time is a relative clock; no calendar or label timestamps exist and none are invented.",
    ),
)
ULB_TARGET = TargetSpec(
    dataset="ulb", column="Class", order_key="Time", provenance=ULB_PROVENANCE,
    maturity_policy=MaturityPolicy.FINALIZED, documented_maturation_seconds=7 * DAY,
    finalized_reason=("Finalized benchmark extract: labels were fixed before the 2015 release; no per-row label "
                      "timestamps are shipped, and the 48 h span is shorter than the ≈7-day reaction window "
                      "(documented_maturation_seconds records δ = 7 d for reference only)."),
    label_derived_columns=("Class",),
)

TARGETS: dict[str, TargetSpec] = {t.dataset: t for t in (SPARKOV_TARGET, IEEE_TARGET, ULB_TARGET)}


def get_target(name: str) -> TargetSpec:
    try:
        return TARGETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(TARGETS)}") from None


__all__ = ["DAY", "LabelMechanism", "MaturityPolicy", "Source", "LabelProvenance", "TargetSpec",
           "TARGETS", "get_target", "SPARKOV_TARGET", "IEEE_TARGET", "ULB_TARGET"]
