"""Feature / capability manifest per dataset (Phase 1A.7).

Reads a prepared artifact directory and writes ``feature-manifest.json`` there plus
``reports/manifest-<dataset>.org``: capability matrix, every feature with its provenance (raw input,
derived-by-step, causal history family, view-only), categorical columns, NaN policy, view sizes,
imbalance/experiment defaults and the serving contract.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS
from .audit import _org_table

_DERIVED = [
    (r"^h_n_prior$", "history", "number of prior events of the entity"),
    (r"^h_(category|merchant)_(cnt|amt)_\d+h$", "history/agg2", "prior events of the entity with the same context, window"),
    (r"^h_(cnt|amt)_\d+h$", "history/agg1", "prior events of the entity, window"),
    (r"^h_amt_over_.*mean_\d+h$|^h_amt_z_\d+h$|^h_amt_ratio_prev$|^h_prev_amt$", "history/deviation", "amount vs the entity's prior behaviour"),
    (r"^h_hours_since_last$", "history/last", "hours since the entity's previous event"),
    (r"^h_dist_prev_km$|^h_speed_kmh$|^h_dist_home_.*$", "history/geo", "geographic behaviour vs prior events"),
    (r"^vm\d+_.*$", "history/periodic", "von Mises time-of-day features over prior events"),
    (r"^cal_.*$", "derived/calendar", "CalendarFeatures on the event datetime (dataset clock)"),
    (r"^clk_.*$", "derived/cyclic", "CyclicClock on the relative order key"),
    (r"^.*_log1p$", "derived/amount", "LogAmount"),
    (r"^.*_decimals$", "derived/amount", "AmountDecimals"),
    (r"^.*_freq$", "derived/frequency", "FrequencyEncoder (train frequencies)"),
    (r"^.*__isna$", "derived/missing", "MissingIndicator (columns with nulls in train)"),
    (r"^n_missing_.*$", "derived/missing", "RowMissingCount"),
    (r"^age_years$", "derived/demographics", "AgeAtEvent (dob at event time)"),
    (r"^dist_km$", "derived/geo", "HaversineDistance home→merchant"),
    (r"^(P|R)_email_(provider|tld)$|^(os|browser|device)_family$", "derived/token", "TokenSplit family token"),
    (r"^has_identity$", "derived/join", "identity row present (prepare)"),
]


def provenance(column: str, contract) -> tuple[str, str]:
    for pat, fam, how in _DERIVED:
        if re.fullmatch(pat, column):
            return fam, how
    s = contract.spec_for(column)
    if s is not None:
        return f"raw/{s.role.value}", s.description or "as shipped"
    return "derived/other", "derived column"


def build_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    d = Path(artifact_dir)
    m = json.loads((d / "manifest.json").read_text())
    bundle = json.loads((d / "bundle.json").read_text()) if (d / "bundle.json").exists() else {}
    exp = json.loads((d / "experiment.json").read_text()) if (d / "experiment.json").exists() else {}
    contract = ADAPTERS[m["dataset"]].contract
    feats = [{"name": c, "family": provenance(c, contract)[0], "how": provenance(c, contract)[1],
              "categorical": c in m["categorical_columns"]} for c in m["feature_columns"]]
    fam_counts: dict[str, int] = {}
    for f in feats:
        fam_counts[f["family"]] = fam_counts.get(f["family"], 0) + 1
    out = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"), "dataset": m["dataset"], "protocol": m["protocol"],
        "contract_version": m["contract_version"], "role_in_suite": m["role_in_suite"],
        "frozen_contracts_sha256": m["frozen_contracts_sha256"],
        "capabilities": {k.family.value: k.support.value for k in contract.capabilities},
        "keys": {"target": contract.target, "row_id": contract.row_id, "entity_key": contract.entity_key,
                 "order_key": contract.order_key, "event_time": contract.event_time},
        "label": {"mechanism": m["label_mechanism"], "maturity_policy": m["maturity_policy"]},
        "split": m["split"]["summary"], "split_spec": m["split"]["spec"],
        "features": feats, "n_features": m["n_features"], "families": fam_counts,
        "categorical_columns": m["categorical_columns"],
        "nan_policy": "NaN preserved in the feature layer and the tree view; train-median imputed in the linear view",
        "views": m["views"], "selection": m.get("selection"),
        "imbalance": {"baseline": "none (natural distribution)", "experiment": exp.get("imbalance"),
                      "natural_parts": m.get("imbalance", {}).get("natural_parts")},
        "serving_contract": bundle.get("serving_contract"),
        "artifacts": bundle.get("files", {}),
    }
    (d / "feature-manifest.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return out


def render_org(man: dict[str, Any]) -> str:
    L = [f"#+TITLE: Feature & capability manifest — {man['dataset']} / {man['protocol']}",
         f"#+DATE: {man['generated'][:10]}", "#+OPTIONS: ^:nil toc:2", "",
         man["role_in_suite"], "",
         f"Contract {man['contract_version']} (frozen sha {man['frozen_contracts_sha256'][:12]}); label mechanism "
         f"{man['label']['mechanism']}, maturity {man['label']['maturity_policy']}. Keys: {man['keys']}.", "",
         "* Capabilities", _org_table(["feature family", "support"], [[k, v] for k, v in man["capabilities"].items()]), "",
         "* Split (natural prevalence in every part)",
         _org_table(["part", "rows", "positives", "prevalence", "order min", "order max"],
                    [[p, s["rows"], s.get("positives"), s.get("prevalence"), s.get("order_min"), s.get("order_max")]
                     for p, s in man["split"].items()]), "",
         f"* Features ({man['n_features']}; {len(man['categorical_columns'])} categorical)",
         _org_table(["family", "count"], [[k, v] for k, v in sorted(man["families"].items())]), "",
         man["nan_policy"], "", "** Feature list",
         _org_table(["feature", "family", "categorical", "how"],
                    [[f["name"], f["family"], "yes" if f["categorical"] else "", f["how"]] for f in man["features"]]), "",
         "* Model views", _org_table(["view", "columns"], [[k, v] for k, v in man["views"].items()])]
    if man.get("selection"):
        s = man["selection"]
        L += ["", f"* Feature selection: {s['mode']} (k = {s['k']}, fitted on {s['fitted_on']})", s["note"],
              "Selected: " + ", ".join(s["selected"])]
        if "published_alternative" in s:
            L.append(f"Published alternative ({s['published_alternative']['mode']}): " + ", ".join(s["published_alternative"]["selected"]))
    L += ["", "* Imbalance", f"Baseline: {man['imbalance']['baseline']}; experiment default: {man['imbalance']['experiment']}",
          _org_table(["part", "rows", "positives", "prevalence", "natural"],
                     [[p, v["rows"], v["positives"], round(v["prevalence"], 6), v["natural"]]
                      for p, v in (man["imbalance"]["natural_parts"] or {}).items()])]
    sc = man.get("serving_contract")
    if sc:
        L += ["", "* Serving contract",
              f"stateful: {sc['stateful']}; required fields: {len(sc['required_fields'])}; outputs: {sc['outputs']}; "
              f"state retention: {sc['state_retention_seconds']} s"] + [f"- {r}" for r in sc["rules"]]
    L += ["", "* Artifacts", _org_table(["file", "sha256"], [[k, v[:16]] for k, v in man["artifacts"].items()]), ""]
    return "\n".join(L)


def write_manifest(artifact_dir: str | Path, reports: str | Path = "reports") -> Path:
    man = build_manifest(artifact_dir)
    suffix = "" if man["protocol"] == "temporal" else f"-{man['protocol']}"
    path = Path(reports) / f"manifest-{man['dataset']}{suffix}.org"
    path.write_text(render_org(man))
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Write feature/capability manifests from prepared artifacts")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--reports", default="reports")
    a = p.parse_args(argv)
    for d in sorted(Path(a.artifacts).glob("*/*")):
        if (d / "manifest.json").exists():
            print(f"[manifest] {d} → {write_manifest(d, a.reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
