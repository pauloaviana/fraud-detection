"""Prepare → causal history → split → fit feature layer → model views (Phase 1A.4 + 1A.5).

    python -m frauddet.prepare --dataset all --data-dir data --artifacts artifacts --reports reports

Stages, per dataset and protocol:
  1. assemble the labeled frame (ULB: keep-first dedup on dedup_key; IEEE: identity left-join, canonical
     names, has_identity), sorted by the order key;
  2. Sparkov only — causal per-entity history (history.compute_history): label-free, fit-free, uses only
     prior events, so it is computed on the whole chronological frame BEFORE splitting;
  3. primary temporal split (ULB also: Ma-2026 stratified benchmark);
  4. learned feature layer (preprocessing.build_pipeline) fitted on the training part only → typed frame
     (NaN preserved, categoricals as ``category``);
  5. model views fitted on the training features: tree (NaN + native categoricals) and linear (impute,
     one-hot, standardise);
  6. ULB Ma-2026 protocol only: RF-Gini top-15 selector fitted on the training part.
Artifacts under artifacts/<dataset>/<protocol>/ (all JSON + membership.csv); report reports/prepare-<ds>.org.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .adapters import ADAPTERS, RawAdapter, get_adapter
from .adapters.ieee import canonical_name
from .audit import _jsonable, _org_table
from .freeze import FROZEN_PATH
from .folds import training_folds
from .history import HistorySpec, compute_history, snapshot_from_frame
from .imbalance import ExperimentConfig, ImbalanceSpec, check_natural, class_weights, cost_context
from .labels import TARGETS
from .manifest import write_manifest
from .preprocessing import MA2026_PROTOCOL, build_pipeline
from .serving import FeatureBundle
from .splits import TEMPORAL, stratified_split_ma2026, temporal_split, verify
from .views import MA2026_SELECTED_15, ModelView, RFGiniSelector


# ------------------------------------------------------------------------------ frame assembly
def prepare_frame(adapter: RawAdapter, protocol: str = TEMPORAL) -> tuple[pd.DataFrame, dict[str, Any]]:
    c = adapter.contract
    notes: dict[str, Any] = {}
    if c.name == "ieee":
        tx = adapter.load("train")
        ident = adapter.load("train_identity")
        ident.columns = [canonical_name(col) for col in ident.columns]
        df = tx.merge(ident, on="TransactionID", how="left", validate="one_to_one")
        df["has_identity"] = df["TransactionID"].isin(ident["TransactionID"]).astype("int8")
        notes["identity_join"] = {"identity_rows": int(len(ident)), "coverage": round(float(df["has_identity"].mean()), 4)}
        row_id = "TransactionID"
    else:
        df = adapter.load("train")
        row_id = c.row_id
    if c.dedup_key and protocol == TEMPORAL:
        before = len(df)
        df = df.drop_duplicates(subset=list(c.dedup_key), keep="first").reset_index(drop=True)
        notes["dedup"] = {"key": list(c.dedup_key), "removed_rows": before - len(df), "rows": len(df)}
    elif c.dedup_key:
        notes["dedup"] = {"key": list(c.dedup_key), "removed_rows": 0, "rows": len(df),
                          "note": "benchmark protocol keeps raw rows as in Ma et al. 2026"}
    df = df.sort_values(c.order_key, kind="stable").reset_index(drop=True)
    notes["row_id"] = row_id
    return df, notes


# ------------------------------------------------------------------------------ run one dataset/protocol
def run(adapter: RawAdapter, protocol: str, artifacts: Path, fractions=(0.70, 0.15, 0.15),
        gap_seconds: float = 0.0, select_k: int | None = None) -> dict[str, Any]:
    c = adapter.contract
    spec = TARGETS[c.name]
    df, notes = prepare_frame(adapter, protocol)
    y = spec.binary(df)

    # 2. causal history (Sparkov): before the split, on the whole chronological frame, label-free
    hspec = None
    if c.entity_key:
        hspec = HistorySpec(entity=c.entity_key, order=c.order_key, event_time=c.event_time)
        df = compute_history(df, hspec)
        notes["history"] = {"features": len(hspec.feature_names()), "windows_h": list(hspec.windows_h),
                            "vm_windows_d": list(hspec.vm_windows_d), "vm_alpha": hspec.vm_alpha,
                            "conds": list(hspec.conds)}

    # 3. split
    split = temporal_split(df, c, fractions, gap_seconds=gap_seconds) if protocol == TEMPORAL \
        else stratified_split_ma2026(df, c)
    violations = verify(split, df)
    if violations:
        raise RuntimeError(f"{c.name}/{protocol}: split invariants violated: {violations}")
    first = split.spec.part_names[0]

    # 4. learned feature layer on the training part only
    pipe = build_pipeline(c, protocol)
    train_df = df.iloc[split.parts[first]]
    pipe.fit(train_df, order_key=c.order_key)
    spec.assert_no_label_leak(pipe.feature_columns, c)
    y_parts = {name: y.iloc[split.parts[name]] for name in split.spec.part_names}

    # 5. model views fitted on the training features; parts are transformed one at a time (memory)
    X_train = pipe.transform(train_df)
    tree = ModelView("tree").fit(X_train)
    linear = ModelView("linear").fit(X_train)

    # 6. optional leakage-safe selection (Ma-2026 benchmark by default), on the training part only
    selection = None
    k = select_k if select_k is not None else (15 if protocol == MA2026_PROTOCOL else None)
    if k:
        sel = RFGiniSelector(k=k).fit(linear.transform(X_train), y_parts[first])
        selection = {"mode": "rf_gini_refit", "k": k, "selected": sel.selected, "ranking_top": sel.state["ranking"][:k],
                     "overlap_with_ma2026_top15": sorted(set(sel.selected) & set(MA2026_SELECTED_15)),
                     "fitted_on": first,
                     "note": "protocol reproduction with RE-FITTED selection (ranks 13-15 are seed/depth "
                             "sensitive, importances ~0.01); not an exact published-feature reproduction"}
        if protocol == MA2026_PROTOCOL:
            selection["published_alternative"] = {"mode": "ma2026_published", "selected": MA2026_SELECTED_15,
                                                  "note": "exact published feature list (Ma et al. 2026, Table 3)"}

    # cost context (raw amount, order, label) per part — evaluation parts are the untouched natural slices
    out = artifacts / c.name / protocol
    out.mkdir(parents=True, exist_ok=True)
    ctx_all = cost_context(df, c, y)
    natural = {}
    for name in split.spec.part_names:
        ctx = ctx_all.iloc[split.parts[name]]
        ctx.to_csv(out / f"cost-context-{name}.csv", index=False)
        natural[name] = check_natural(y_parts[name], y.iloc[split.parts[name]], name)

    parts_out: dict[str, Any] = {}
    for name in split.spec.part_names:
        X = X_train if name == first else pipe.transform(split.frame(df, name))
        Xt = tree.transform(X)
        tree_cols = int(Xt.shape[1])
        del Xt
        Xl = linear.transform(X)
        if Xl.isna().any().any():
            raise RuntimeError(f"{c.name}/{protocol}/{name}: NaN in the linear view")
        parts_out[name] = {"rows": int(len(X)), "positives": int(y_parts[name].sum()),
                           "prevalence": round(float(y_parts[name].mean()), 6),
                           "features": int(X.shape[1]), "nan_cells_feature_layer": int(X.isna().sum().sum()),
                           "tree_view_columns": tree_cols, "linear_view_columns": int(Xl.shape[1])}
        del Xl, X
    del X_train

    # ---- experiment metadata: natural baseline; weighting values; folds; resampling only as alternatives
    _, fold_meta = training_folds(protocol, df[c.order_key].to_numpy()[split.parts[first]], y_parts[first])
    exp = ExperimentConfig(
        c.name, protocol, view="tree", imbalance=ImbalanceSpec("none"),
        selection="rf_gini_refit" if selection else "none", folds=fold_meta, cost_ca=1.0,
        notes=["baseline = natural class distribution; evaluation parts are never balanced",
               "weighting available: class_weight (balanced) / example_weight (Bahnsen cost matrix)",
               "resampling (random_under, random_over, smote, enn, smote_enn) only as explicit alternatives, applied "
               "strictly inside training folds via imbalance.resample_training_fold; not assumed beneficial "
               "(Ma et al. 2026: no-resampling XGBoost best on ULB)",
               "cost context per part in cost-context-<part>.csv (row id, order, raw amount, label)"])
    exp_d = exp.to_dict()
    exp_d["class_weight_balanced_on_train"] = class_weights(y_parts[first])
    exp_d["natural_parts"] = natural
    (out / "experiment.json").write_text(json.dumps(_jsonable(exp_d), indent=1))

    # ---- artifacts: the serving bundle (pipeline + views + selector + history spec/state + contract)
    hist_cols = set(hspec.feature_names()) if hspec else set()
    required = tuple(col for col in df.columns if col != c.target and col not in hist_cols)
    bundle = FeatureBundle(c, protocol, pipe, {"tree": tree, "linear": linear}, sel if selection else None, hspec,
                           snapshot_from_frame(train_df, hspec, c.row_id) if hspec is not None else None, required)
    bundle.save(out)
    (out / "split.json").write_text(json.dumps(_jsonable(split.to_dict()), indent=1))
    rid = notes["row_id"]
    member = pd.DataFrame({"row": np.concatenate([split.parts[n] for n in split.spec.part_names]),
                           "part": np.concatenate([[n] * len(split.parts[n]) for n in split.spec.part_names])})
    if rid and rid in df.columns:
        member.insert(0, rid, df[rid].to_numpy()[member["row"].to_numpy()])
    member.to_csv(out / "membership.csv", index=False)
    cat_cols = list(tree.state["categorical"])
    manifest = {
        "frauddet_version": __version__, "dataset": c.name, "protocol": protocol,
        "contract_version": c.contract_version, "role_in_suite": c.role_in_suite,
        "frozen_contracts_sha256": hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest() if FROZEN_PATH.exists() else None,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "prepare_notes": notes, "target": spec.column, "label_mechanism": spec.provenance.mechanism.value,
        "maturity_policy": spec.maturity_policy.value,
        "split": split.to_dict(), "parts": parts_out,
        "feature_columns": pipe.feature_columns, "n_features": len(pipe.feature_columns),
        "categorical_columns": cat_cols, "n_history_features": len(hspec.feature_names()) if hspec else 0,
        "steps": [s.to_dict()["kind"] for s in pipe.steps],
        "views": {"tree": len(tree.state["output_columns"]), "linear": len(linear.state["output_columns"])},
        "selection": selection,
        "imbalance": {"baseline": "none (natural distribution)", "metadata": "experiment.json",
                      "natural_parts": natural},
        "artifacts": sorted(p.name for p in out.iterdir()),
    }
    (out / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=1, ensure_ascii=False))
    reports_dir = artifacts.parent / "reports" if (artifacts.parent / "reports").exists() else Path("reports")
    if reports_dir.exists():
        write_manifest(out, reports_dir)
    return _jsonable(manifest)


# ------------------------------------------------------------------------------ report
def render_org(name: str, runs: list[dict[str, Any]]) -> str:
    L = [f"#+TITLE: Prepared splits, feature layer and views — {name}", f"#+DATE: {dt.date.today()}",
         "#+OPTIONS: ^:nil", ""]
    for m in runs:
        L += [f"* protocol: {m['protocol']}", f"{m['role_in_suite']}",
              f"Contract {m['contract_version']}; label mechanism {m['label_mechanism']}; maturity {m['maturity_policy']}. "
              f"Prepare notes: {json.dumps(m['prepare_notes'])}", "",
              "** Split", m["split"]["spec"]["note"],
              _org_table(["part", "rows", "fraction", "positives", "prevalence", "share of positives", "order min", "order max"],
                         [[p, s["rows"], s["fraction"], s.get("positives"), s.get("prevalence"), s.get("share_of_positives"),
                           s.get("order_min"), s.get("order_max")] for p, s in m["split"]["summary"].items()]),
              "", f"** Feature layer ({m['n_features']} features, {m['n_history_features']} causal history, "
                  f"{len(m['categorical_columns'])} categorical): " + " → ".join(m["steps"]),
              f"Fitted on the {m['split']['spec']['part_names'][0]} part only. NaN is preserved in the feature layer "
              f"(imputation is a view choice); categoricals are typed, not ordinal.",
              _org_table(["part", "rows", "NaN cells (feature layer)", "tree-view cols", "linear-view cols"],
                         [[p, s["rows"], s["nan_cells_feature_layer"], s["tree_view_columns"], s["linear_view_columns"]]
                          for p, s in m["parts"].items()]),
              "Categorical columns: " + (", ".join(m["categorical_columns"]) or "none"),
              "Feature columns: " + ", ".join(m["feature_columns"][:80]) + (" …" if m["n_features"] > 80 else ""), ""]
        if m.get("selection"):
            s = m["selection"]
            L += [f"** Feature selection (RF-Gini top-{s['k']}, fitted on {s['fitted_on']})",
                  _org_table(["rank", "feature", "importance"], [[i + 1, f, round(v, 4)] for i, (f, v) in enumerate(s["ranking_top"])]),
                  f"Overlap with Ma et al. 2026's 15: {len(s['overlap_with_ma2026_top15'])}/15 — {', '.join(s['overlap_with_ma2026_top15'])}",
                  f"Labelled as: {s['note']}" + (f". Published alternative available: {s['published_alternative']['mode']}"
                                                 if "published_alternative" in s else ""), ""]
        L += ["** Imbalance treatment", "Baseline: natural class distribution; evaluation parts untouched (verified). "
              "Weighting and resampling alternatives are recorded in experiment.json; resampling only inside training folds.",
              _org_table(["part", "rows", "positives", "prevalence", "natural"],
                         [[p, v["rows"], v["positives"], round(v["prevalence"], 6), v["natural"]]
                          for p, v in m["imbalance"]["natural_parts"].items()]), "",
              f"Artifacts: {', '.join(m['artifacts'])}", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare, split, fit feature layer and views (Phase 1A.4/1A.5)")
    p.add_argument("--dataset", default="all", choices=["all", *ADAPTERS])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--reports", default="reports")
    p.add_argument("--gap-seconds", type=float, default=0.0)
    p.add_argument("--select-k", type=int, default=None, help="RF-Gini top-k for every protocol (default: Ma only)")
    a = p.parse_args(argv)
    names = list(ADAPTERS) if a.dataset == "all" else [a.dataset]
    Path(a.reports).mkdir(parents=True, exist_ok=True)
    for name in names:
        adapter = get_adapter(name, a.data_dir)
        protocols = [TEMPORAL, MA2026_PROTOCOL] if name == "ulb" else [TEMPORAL]
        runs = []
        for proto in protocols:
            print(f"[prepare] {name}/{proto} ...", flush=True)
            m = run(adapter, proto, Path(a.artifacts), gap_seconds=a.gap_seconds, select_k=a.select_k)
            runs.append(m)
            print(f"[prepare] {name}/{proto}: " + "; ".join(f"{k}={v['rows']}/{v['positives']}" for k, v in m["parts"].items())
                  + f"; {m['n_features']} features ({m['n_history_features']} history), views tree={m['views']['tree']} linear={m['views']['linear']}")
        (Path(a.reports) / f"prepare-{name}.org").write_text(render_org(name, runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
