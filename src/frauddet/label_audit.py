"""Empirical label inspection (Phase 1A.2) — read-only.

Checks that each dataset's labels behave as their documented mechanism says
they should, and records the documented maturation windows as metadata. Writes
reports/labels.org (+ labels.json).

    python -m frauddet.label_audit --data-dir data --out reports
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .adapters import get_adapter
from .audit import _jsonable, _org_table
from .findings import Finding, Severity as Sev
from .labels import DAY, TARGETS, TargetSpec


def _basic(spec: TargetSpec, df: pd.DataFrame) -> tuple[pd.Series, list[Finding]]:
    y = spec.binary(df)
    return y, [Finding(Sev.INFO, f"{spec.dataset}.label_values",
                       "label column validated: no nulls, values ⊆ {0,1}",
                       {"positives": int(y.sum()), "rows": int(len(y)), "prevalence": round(float(y.mean()), 6)})]


# ------------------------------------------------------------------------------ Sparkov
def inspect_sparkov(adapter) -> list[Finding]:
    spec = TARGETS["sparkov"]
    out: list[Finding] = []
    frames = {k: adapter.load(k, usecols=["trans_date_trans_time", "cc_num", "category", "amt", "unix_time",
                                          "is_fraud"]) for k in ("train", "test")}
    for key, df in frames.items():
        y, f = _basic(spec, df)
        out += f
        fr = df[y == 1]
        date = df["trans_date_trans_time"].dt.normalize()
        # episodes: distinct fraud dates per card, and their span
        fd = fr.assign(d=date[y == 1]).groupby("cc_num")["d"].agg(["nunique", "min", "max"])
        span_days = (fd["max"] - fd["min"]).dt.days + 1
        # legit rows on a card's fraud dates (generator drops them → expect 0)
        fraud_keys = set(zip(fr["cc_num"], date[y == 1]))
        on_fraud_date = pd.Series([(c, d) in fraud_keys for c, d in zip(df["cc_num"], date)], index=df.index)
        legit_on_fraud_dates = int(((y == 0) & on_fraud_date).sum())
        hour = df["trans_date_trans_time"].dt.hour
        night = hour.isin([22, 23, 0, 1, 2, 3])
        out.append(Finding(Sev.NOTE, f"sparkov.{key}.episode_structure",
                           "labels follow the generator: one short episode per card, episode dates contain no "
                           "legit rows, ~80 % of fraud rows at 22:00–03:59",
                           {"cards": int(df["cc_num"].nunique()), "cards_with_fraud": int(len(fd)),
                            "fraud_dates_per_card": {"median": float(fd["nunique"].median()), "max": int(fd["nunique"].max())},
                            "episode_span_days": {"median": float(span_days.median()), "max": int(span_days.max())},
                            "legit_rows_on_fraud_dates": legit_on_fraud_dates,
                            "fraud_rows_per_card_date": round(float(fr.groupby(["cc_num", date[y == 1]]).size().mean()), 2),
                            "legit_rows_per_card_date": round(float(df[y == 0].groupby(["cc_num", date[y == 0]]).size().mean()), 2),
                            "night_share_fraud": round(float(night[y == 1].mean()), 4),
                            "night_share_legit": round(float(night[y == 0].mean()), 4)}))
        amt = df.groupby([df["category"], y])["amt"].median().unstack()
        out.append(Finding(Sev.INFO, f"sparkov.{key}.amount_by_category",
                           "median amount per category, legit vs fraud (fraud profile means are 3–10× larger)",
                           {c: [round(float(amt.loc[c, 0]), 2), round(float(amt.loc[c, 1]), 2)] for c in amt.index}))
    all_cards = pd.concat([frames["train"]["cc_num"], frames["test"]["cc_num"]]).nunique()
    cards_fraud = pd.concat([frames[k].loc[frames[k]["is_fraud"] == 1, "cc_num"] for k in frames]).nunique()
    out.append(Finding(Sev.NOTE, "sparkov.cards_with_episode_overall",
                       "share of cards with a fraud episode across train+test (generator: fraud_flag < 99 ⇒ ~98 % "
                       "of customers get one episode within the generation window)",
                       {"cards": int(all_cards), "cards_with_fraud": int(cards_fraud),
                        "share": round(cards_fraud / all_cards, 4)}))
    out.append(Finding(Sev.NOTE, "sparkov.maturity",
                       "no maturation exists (label decided at generation); maturity is NOT enforced. A rehearsal "
                       "lag can be imposed via TargetSpec.matured_mask(rehearsal_lag_seconds=...) for pipeline "
                       "engineering only", {"policy": spec.maturity_policy.value}))
    return out


# ------------------------------------------------------------------------------ IEEE
def inspect_ieee(adapter) -> list[Finding]:
    spec = TARGETS["ieee"]
    out: list[Finding] = []
    df = adapter.load("train", usecols=["TransactionID", "isFraud", "TransactionDT", "card1", "addr1",
                                        "P_emaildomain", "ProductCD"])
    y, f = _basic(spec, df)
    out += f
    t = df["TransactionDT"]
    end = int(t.max())
    # documented 120-day window: information only (labels are finalized; not a row filter)
    window = spec.documented_maturation_seconds
    inside = (t + window) > end
    assert spec.matured_mask(df, as_of=end).all()
    out.append(Finding(Sev.NOTE, "ieee.maturation_window_documented",
                       "Vesta's 120-day window is provenance metadata, not a filter: all released labels are "
                       "treated as final (chargeback timestamps unavailable). For reference, this many rows of the "
                       "labeled file lie within 120 days of its end",
                       {"documented_window_days": window // DAY, "file_end_day": round(end / DAY, 2),
                        "rows_within_window_of_end": int(inside.sum()), "fraud_within_window_of_end": int(y[inside].sum()),
                        "rows_older_than_window": int((~inside).sum()),
                        "prevalence_older": round(float(y[~inside].mean()), 5),
                        "prevalence_within": round(float(y[inside].mean()), 5),
                        "matured_mask_all_true": True}))
    # linkage-propagation signature: prior fraud on the same (card1, addr1, P_emaildomain)
    key = df[["card1", "addr1", "P_emaildomain"]].astype("string").fillna("<NA>").agg("|".join, axis=1)
    order = np.argsort(t.to_numpy(), kind="stable")
    ks, ys = key.to_numpy()[order], y.to_numpy()[order]
    prior = pd.Series(ys).groupby(ks).cumsum().to_numpy() - ys           # frauds strictly earlier in the key
    has_prior = np.zeros(len(df), dtype=bool)
    has_prior[order] = prior > 0
    rate_prior, rate_none = float(y[has_prior].mean()), float(y[~has_prior].mean())
    out.append(Finding(Sev.RISK, "ieee.linkage_propagation_signature",
                       "fraud rate given an earlier fraud on the same (card1, addr1, P_emaildomain) is far above "
                       "base rate — consistent with Vesta's 'posterior linked transactions are fraud too'. 'Prior "
                       "fraud on a linkage key' is label-derived history, not an ordinary feature",
                       {"rows_with_prior_fraud_on_key": int(has_prior.sum()),
                        "fraud_rate_with_prior_fraud": round(rate_prior, 4),
                        "fraud_rate_without": round(rate_none, 4),
                        "share_of_all_fraud_with_prior_fraud_on_key": round(float(y[has_prior].sum() / y.sum()), 4),
                        "key": "card1|addr1|P_emaildomain (proxy for Vesta's account/email/billing linkage; "
                               "not used as an entity key)"}))
    blk = (t // (30 * DAY)).astype(int)
    out.append(Finding(Sev.INFO, "ieee.prevalence_by_30d",
                       "prevalence per 30-day block of the relative clock",
                       {int(k): round(float(v), 5) for k, v in y.groupby(blk).mean().items()}))
    return out


# ------------------------------------------------------------------------------ ULB
def inspect_ulb(adapter) -> list[Finding]:
    spec = TARGETS["ulb"]
    out: list[Finding] = []
    df = adapter.load("train")
    y, f = _basic(spec, df)
    out += f
    feat = df.drop(columns=["Class"])
    dup = feat.duplicated(keep=False)
    conflicts = df[dup].groupby(list(feat.columns))["Class"].nunique()
    n_conflict_groups = int((conflicts > 1).sum())
    out.append(Finding(Sev.WARN if n_conflict_groups else Sev.INFO, "ulb.duplicate_label_consistency",
                       "rows identical on all features: do any carry conflicting labels? (label-noise check; "
                       "deduplication itself is NOT decided here)",
                       {"rows_in_identical_feature_groups": int(dup.sum()),
                        "groups_with_conflicting_labels": n_conflict_groups,
                        "fraud_rows_in_identical_groups": int(y[dup].sum())}))
    day = (df["Time"] // DAY).astype(int)
    out.append(Finding(Sev.NOTE, "ulb.maturity_unenforceable",
                       "48 h span vs ≈7-day reaction window (Dal Pozzolo 2015, δ = 7) and AWS's 2 weeks–3 months: "
                       "maturity cannot be enforced; labels treated as finalized (see finalized_reason)",
                       {"span_hours": float(df["Time"].max() / 3600), "fraud_per_day": [int(v) for v in y.groupby(day).sum()],
                        "rows_per_day": [int(v) for v in y.groupby(day).size()],
                        "reference_operational_rate": "~304 frauds / ~160k tx per day (2013 stream, Dal Pozzolo 2015)",
                        "finalized_reason": spec.finalized_reason}))
    return out


INSPECTORS = {"sparkov": inspect_sparkov, "ieee": inspect_ieee, "ulb": inspect_ulb}


# ------------------------------------------------------------------------------ rendering
def render_org(report: dict[str, Any]) -> str:
    L = ["#+TITLE: Label semantics and provenance — Phase 1A.2", f"#+DATE: {report['generated'][:10]}",
         "#+OPTIONS: ^:nil toc:2", "",
         "Three label-generating mechanisms that only share a {0,1} encoding. The common interface is "
         "~frauddet.labels.TargetSpec~ (binary target, maturity mask, label-leak guard); provenance and "
         "assumptions stay attached as metadata. Read-only: nothing split, engineered, rebalanced or trained.", "",
         "* Summary",
         _org_table(["dataset", "column", "mechanism", "maturity policy", "documented window (metadata)",
                     "label timestamps", "label-derived cols"],
                    [[n, s["column"], s["provenance"]["mechanism"], s["maturity_policy"],
                      f"{s['documented_maturation_seconds'] // DAY} d" if s["documented_maturation_seconds"] else "—",
                      "no", ", ".join(s["label_derived_columns"])] for n, s in report["targets"].items()]), "",
         "* Guard: post-investigation information",
         "~TargetSpec.assert_no_label_leak(columns, contract)~ refuses any feature list containing the target, a "
         "column listed in ~label_derived_columns~, or a contract column with role ~target~ / ~label_derived~. None "
         "of the three datasets ship resolution fields (chargeback date, dispute status, investigator outcome); the "
         "~label_derived~ role exists so a production schema cannot expose them as inputs. Label-derived *history* "
         "(e.g. 'prior fraud on this key') is likewise not an inference feature unless it respects maturation.", ""]
    for name, s in report["targets"].items():
        p = s["provenance"]
        L += [f"* {name} — {p['mechanism']}",
              "** Positive (=1)", p["positive_definition"], "** Negative (=0)", p["negative_definition"],
              "** Maturation", p["maturation"], f"Policy: ~{s['maturity_policy']}~"
              + (f"; documented window {s['documented_maturation_seconds'] // DAY} days (metadata only, on ~{s['order_key']}~ units)"
                 if s["documented_maturation_seconds"] else "")
              + (f". Finalized because: {s['finalized_reason']}" if s["finalized_reason"] else "."),
              "** Propagation", p["propagation"], "** Noise", p["noise"],
              "** Assumptions"] + [f"- {a}" for a in p["assumptions"]] + [
              "** Sources"] + [f"- [[{src['url']}][{src['title']}]]" + (f" — {src['note']}" if src["note"] else "")
                              for src in p["sources"]] + ["** Empirical inspection"]
        order = {"risk": 0, "warn": 1, "note": 2, "info": 3}
        for f in sorted(report["inspection"].get(name, []), key=lambda f: order[f["severity"]]):
            L += [f"*** {f['severity'].upper()} — {f['code']}", f["message"], "#+begin_example",
                  json.dumps(f["detail"], indent=1, ensure_ascii=False, default=str), "#+end_example"]
        L.append("")
    L += ["* Industry guidance consulted",
          "- AWS Transaction Fraud Insights: chargeback fraud \"often takes 60 days or more to correctly identify\"; "
          "\"ensure that all records in your training dataset are mature\".",
          "- AWS Event dataset: maturity \"anywhere from two weeks to three months\", set by the chargeback period or "
          "investigator determination; a label requires a LABEL_TIMESTAMP — which none of our datasets provide.",
          "- Stripe primer: labels are cardholder disputes; blocked payments never receive an outcome (selective labels).",
          "- Adyen risk-field reference: only authorization-time fields are risk inputs; chargeback/dispute data are "
          "post-transaction.",
          "- Uber Mastermind: rule-execution architecture; no label semantics.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Label semantics inspection (Phase 1A.2)")
    p.add_argument("--dataset", default="all", choices=["all", *TARGETS])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="reports")
    a = p.parse_args(argv)
    names = list(TARGETS) if a.dataset == "all" else [a.dataset]
    report: dict[str, Any] = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
                              "frauddet_version": __version__,
                              "targets": {n: TARGETS[n].metadata() for n in TARGETS}, "inspection": {}}
    for n in names:
        print(f"[labels] {n}: inspecting ...", flush=True)
        report["inspection"][n] = [f.to_dict() for f in INSPECTORS[n](get_adapter(n, a.data_dir))]
    report = _jsonable(report)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    (out / "labels.org").write_text(render_org(report))
    print(f"[labels] → {out / 'labels.org'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
