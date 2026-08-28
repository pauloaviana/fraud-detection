"""Read-only raw-data audit (Phase 1A.1).

For every file of a dataset: schema vs contract, observed dtypes, missingness,
duplicates, cardinalities, target prevalence, ordering/time keys, entity-key
statistics, amount summary and a univariate *leakage screen* (per-column AUC /
category fraud rates — a screen, not a model). Dataset-specific checks come
from the adapter. Output: one .org report and one .json per dataset, plus a
cross-dataset capability mapping. Nothing is split, imputed, scaled, resampled
or engineered here.

    python -m frauddet.audit --dataset all --data-dir data --out reports
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import __version__
from .adapters import ADAPTERS, RawAdapter, get_adapter
from .contracts import (
    IDENTIFIER_ROLES, ColumnSpec, DatasetContract, FeatureFamily, Kind, Role, Support,
)
from .findings import Finding, Severity as Sev

# Which files to load per dataset, and which columns (None = all). Light loads
# (a column subset) only get schema / row-count / time-range treatment.
LOAD_PLAN: dict[str, dict[str, list[str] | None]] = {
    "sparkov": {"train": None, "test": None},
    "ieee": {
        "train": None,
        "train_identity": None,
        "test": ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"],
        "test_identity": ["TransactionID"],
    },
    "ulb": {"train": None},
}

AUC_SUSPICIOUS = 0.85
MIN_CATEGORY_SUPPORT = 100


# --------------------------------------------------------------------------- helpers
def _family_label(contract: DatasetContract, name: str) -> str | None:
    if any(c.name == name for c in contract.columns):
        return None
    for fam in contract.families:
        if fam.matches(name):
            return fam.label
    return None


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def _auc(y: np.ndarray, x: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2 or len(x) == 0:
        return None
    a = roc_auc_score(y, x)
    return float(max(a, 1.0 - a))


# --------------------------------------------------------------------------- per-file audit
def audit_frame(contract: DatasetContract, key: str, df: pd.DataFrame,
                target: pd.Series | None = None, light: bool = False) -> dict[str, Any]:
    fs = contract.file(key)
    header = list(df.columns)
    specs, unexpected = contract.resolve(header)
    spec_of: dict[str, ColumnSpec] = {s.name: s for s in specs}
    n = len(df)
    out: dict[str, Any] = {
        "file": {"key": key, "member": fs.member, "container": fs.container, "purpose": fs.purpose,
                 "labeled": fs.labeled, "light": light, "rows": n, "columns": len(header),
                 "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 2**20, 1)},
        "schema": [], "unexpected_columns": unexpected, "findings": [],
    }
    findings: list[Finding] = out["findings"]
    if unexpected:
        findings.append(Finding(Sev.WARN, f"{contract.name}.{key}.unexpected_columns",
                                "columns not covered by the contract", {"columns": unexpected}))

    nunique = df.nunique(dropna=True)
    nulls = df.isna().sum()
    for name in header:
        s = spec_of.get(name)
        out["schema"].append({
            "column": name, "family": _family_label(contract, name),
            "kind": s.kind.value if s else None, "role": s.role.value if s else None,
            "declared_dtype": (s.pandas_dtype or "datetime64") if s else None,
            "observed_dtype": str(df[name].dtype),
            "nulls": int(nulls[name]), "null_frac": round(float(nulls[name]) / n, 4) if n else None,
            "nunique": int(nunique[name]), "notes": s.notes if s else "",
        })
    out["roles"] = {r.value: contract.columns_with_role(header, r) for r in Role}

    # ordering / time keys
    if contract.order_key in header:
        t = df[contract.order_key]
        out["order_key"] = {"column": contract.order_key, "min": _jsonable(t.min()), "max": _jsonable(t.max()),
                            "span_seconds": _jsonable(t.max() - t.min()), "unique": int(t.nunique()),
                            "monotone_non_decreasing": bool(t.is_monotonic_increasing)}
        if not t.is_monotonic_increasing:
            findings.append(Finding(Sev.WARN, f"{contract.name}.{key}.order_key_not_sorted",
                                    "file is not sorted by its order key"))
    if contract.event_time in header:
        e = df[contract.event_time]
        out["event_time"] = {"column": contract.event_time, "min": str(e.min()), "max": str(e.max()),
                             "tz": str(e.dt.tz)}
    if light:
        return out

    # target
    if fs.labeled and contract.target in header:
        target = df[contract.target]
    if target is not None:
        y = target.to_numpy()
        vc = pd.Series(y).value_counts(dropna=False)
        out["target"] = {"column": contract.target, "counts": {str(k): int(v) for k, v in vc.items()},
                         "positives": int(np.nansum(y == contract.positive_label)),
                         "prevalence": round(float(np.nanmean(y == contract.positive_label)), 6),
                         "nulls": int(pd.isna(y).sum())}

    # duplicates
    ignore = [c for c in header if (s := spec_of.get(c)) and s.role in
              {Role.ROW_ID, Role.ORDER_KEY, Role.EVENT_TIME, Role.EXCLUDED}]
    dup_exact = int(df.duplicated().sum())
    dup_ign = int(df.drop(columns=ignore).duplicated().sum()) if ignore else dup_exact
    out["duplicates"] = {"exact_redundant_rows": dup_exact, "ignoring": ignore,
                         "redundant_rows_ignoring_identifiers": dup_ign}
    if contract.row_id in header:
        d = int(df[contract.row_id].duplicated().sum())
        out["duplicates"]["duplicated_row_ids"] = d
        if d:
            findings.append(Finding(Sev.WARN, f"{contract.name}.{key}.duplicate_row_ids", "row ids are not unique",
                                    {"count": d}))
    if dup_exact:
        findings.append(Finding(Sev.WARN, f"{contract.name}.{key}.duplicate_rows",
                                "exact duplicate rows present", {"redundant_rows": dup_exact}))

    # constant columns
    const = [c for c in header if nunique[c] <= 1]
    if const:
        findings.append(Finding(Sev.NOTE, f"{contract.name}.{key}.constant_columns", "columns with ≤1 distinct value",
                                {"columns": const}))

    # low-cardinality value tables (+ fraud rate)
    low: dict[str, Any] = {}
    for c in header:
        if 1 < nunique[c] <= 25 and c != contract.target:
            keyed = df[c].astype("string").fillna("<NA>").to_numpy()   # one key space for counts and rates
            vc = pd.Series(keyed).value_counts()
            tab = {str(k): {"rows": int(v)} for k, v in vc.items()}
            if target is not None:
                fr = pd.Series(target.to_numpy()).groupby(keyed).mean()
                for k, v in fr.items():
                    tab.setdefault(str(k), {})["fraud_rate"] = round(float(v), 5)
            low[c] = tab
    out["low_cardinality_values"] = low

    # entity key
    if contract.entity_key in header:
        g = df.groupby(contract.entity_key, sort=False).size()
        ent: dict[str, Any] = {"column": contract.entity_key, "entities": int(g.size),
                               "tx_per_entity": {q: float(g.quantile(q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)}}
        if target is not None:
            fe = pd.Series(target.to_numpy()).groupby(df[contract.entity_key].to_numpy()).sum()
            ent["entities_with_fraud"] = int((fe > 0).sum())
            ent["fraud_share_top_decile_entities"] = round(float(fe.sort_values(ascending=False)
                                                                 .head(max(1, fe.size // 10)).sum() / fe.sum()), 4)
        out["entity"] = ent

    # amount (column cited by the cost-sensitive claim)
    claim = contract.claim(FeatureFamily.COST_SENSITIVE_EVAL)
    if claim and claim.basis and claim.basis[0] in header:
        a = df[claim.basis[0]].astype(float)
        amt: dict[str, Any] = {"column": claim.basis[0], "overall": a.describe().round(3).to_dict(),
                               "zero_rows": int((a == 0).sum()), "negative_rows": int((a < 0).sum())}
        if target is not None:
            amt["by_class"] = {str(k): v.round(3).to_dict() for k, v in a.groupby(target.to_numpy()).describe().iterrows()}
        out["amount"] = amt

    # leakage screen
    if target is not None:
        out["leakage_screen"] = _leakage_screen(contract, header, spec_of, df, target, findings)
    return out


def _leakage_screen(contract, header, spec_of, df, target, findings) -> dict[str, Any]:
    y_all = target.to_numpy().astype(float)
    rows: list[dict[str, Any]] = []
    base = float(np.nanmean(y_all))
    for c in header:
        s = spec_of.get(c)
        if c == contract.target or (s and s.role is Role.TARGET):
            continue
        col = df[c]
        rec: dict[str, Any] = {"column": c, "role": s.role.value if s else None, "nunique": int(col.nunique())}
        notna = col.notna().to_numpy()
        if (~notna).any():
            rec["fraud_rate_when_null"] = round(float(np.nanmean(y_all[~notna])), 5)
            rec["fraud_rate_when_present"] = round(float(np.nanmean(y_all[notna])), 5)
        if s and s.kind is Kind.DATETIME:
            x = col.astype("int64").to_numpy() / 1e9
            rec["auc"] = _auc(y_all[notna], x[notna])
        elif pd.api.types.is_numeric_dtype(col.dtype):
            x = pd.to_numeric(col).astype(float).to_numpy()
            rec["auc"] = _auc(y_all[notna], x[notna]) if rec["nunique"] > 1 else None
        else:
            keyed = col.astype("string").to_numpy()
            grp = pd.DataFrame({"k": keyed[notna], "y": y_all[notna]}).groupby("k")["y"].agg(["size", "mean"])
            big = grp[grp["size"] >= MIN_CATEGORY_SUPPORT]
            if len(big):
                top = big["mean"].idxmax()
                rec["max_category_fraud_rate"] = round(float(big.loc[top, "mean"]), 5)
                rec["max_category"] = str(top)
                rec["max_category_rows"] = int(big.loc[top, "size"])
            rec["identifier_like"] = bool(rec["nunique"] > 0.5 * len(df))
        rows.append(rec)

    def score(r):
        return max(r.get("auc") or 0.0, (r.get("max_category_fraud_rate") or 0.0) / max(base, 1e-9) / 100)
    rows.sort(key=score, reverse=True)
    suspicious = [r for r in rows if (r.get("auc") or 0) >= AUC_SUSPICIOUS]
    if suspicious:
        ident = [r["column"] for r in suspicious if r["role"] in {x.value for x in IDENTIFIER_ROLES}]
        inputs = [r["column"] for r in suspicious if r["column"] not in ident]
        findings.append(Finding(Sev.RISK, f"{contract.name}.univariate_auc",
                                f"columns with univariate AUC ≥ {AUC_SUSPICIOUS}: unusually strong single "
                                "columns — check for target contamination or generator artefacts",
                                {"input_candidates": inputs, "identifiers": ident,
                                 "auc": {r["column"]: round(r["auc"], 4) for r in suspicious}}))
    return {"base_rate": round(base, 6), "columns": rows}


# --------------------------------------------------------------------------- dataset driver
def audit_dataset(adapter: RawAdapter, plan: dict[str, list[str] | None] | None = None) -> dict[str, Any]:
    c = adapter.contract
    plan = plan or LOAD_PLAN[c.name]
    t0 = time.time()
    frames: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {
        "dataset": c.name, "title": c.title, "source": c.source, "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "frauddet_version": __version__, "contract_notes": c.notes,
        "files": {}, "findings": [],
    }
    for key, usecols in plan.items():
        if not adapter.available(key):
            report["findings"].append(Finding(Sev.WARN, f"{c.name}.{key}.missing_file",
                                              "file not found", {"path": str(adapter.path(key))}).to_dict())
            continue
        frames[key] = adapter.load(key, usecols=usecols)
    # side tables get the target through the join key
    for key, df in frames.items():
        target = None
        light = plan.get(key) is not None
        if not c.file(key).labeled and c.join_key and "train" in frames and c.join_key in df.columns \
                and c.target in frames["train"].columns and not light:
            target = frames["train"].set_index(c.join_key)[c.target].reindex(df[c.join_key]).reset_index(drop=True)
        report["files"][key] = audit_frame(c, key, df, target=target, light=light)
    generic = [f for fr in report["files"].values() for f in fr["findings"]]
    specific = adapter.checks(frames) if frames else []
    report["findings"] = [f.to_dict() for f in generic + specific]
    for fr in report["files"].values():
        fr["findings"] = [f.to_dict() for f in fr["findings"]]
    report["capabilities"] = [{"family": k.family.value, "support": k.support.value, "basis": list(k.basis),
                               "reason": k.reason} for k in c.capabilities]
    report["elapsed_s"] = round(time.time() - t0, 1)
    return _jsonable(report)


# --------------------------------------------------------------------------- org rendering
def _org_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(v):
        s = "" if v is None else str(v)
        return s.replace("|", "\\vert{}").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def _collapse_schema(schema: list[dict[str, Any]]) -> list[list[Any]]:
    rows, seen = [], {}
    for s in schema:
        fam = s["family"]
        if fam is None:
            rows.append([s["column"], s["kind"], s["role"], s["declared_dtype"], s["observed_dtype"],
                        s["nunique"], f'{s["null_frac"]:.1%}' if s["null_frac"] is not None else "", s["notes"]])
            continue
        if fam not in seen:
            seen[fam] = {"first": s["column"], "last": s["column"], "n": 0, "kind": s["kind"], "role": s["role"],
                         "dtype": s["declared_dtype"], "obs": s["observed_dtype"], "nulls": [], "notes": s["notes"]}
            rows.append(seen[fam])
        g = seen[fam]
        g["last"], g["n"] = s["column"], g["n"] + 1
        g["nulls"].append(s["null_frac"] or 0.0)
    return [[f'{r["first"]}..{r["last"]} ({r["n"]})', r["kind"], r["role"], r["dtype"], r["obs"], "",
             f'{min(r["nulls"]):.1%}–{max(r["nulls"]):.1%}', r["notes"]] if isinstance(r, dict) else r
            for r in rows]


def render_org(rep: dict[str, Any]) -> str:
    L: list[str] = [f"#+TITLE: Raw-data audit — {rep['dataset']}", f"#+DATE: {rep['generated'][:10]}",
                    "#+OPTIONS: toc:2 ^:nil", "",
                    f"{rep['title']}. Source: {rep['source']}. Generated by frauddet {rep['frauddet_version']} "
                    f"in {rep['elapsed_s']} s. Read-only audit: no rows were split, imputed, scaled or resampled.", ""]
    if rep["contract_notes"]:
        L += ["* Contract notes"] + [f"- {n}" for n in rep["contract_notes"]] + [""]
    L.append("* Files")
    for key, fr in rep["files"].items():
        f = fr["file"]
        L += [f"** {key} — {f['member']}" + (f" (in {f['container']})" if f["container"] else ""),
              f"{f['purpose']}. Rows {f['rows']:,}, columns {f['columns']}, {f['memory_mb']} MB in memory"
              + (" — light load (column subset)." if f["light"] else ".")]
        if fr["unexpected_columns"]:
            L.append(f"Unexpected columns: {fr['unexpected_columns']}")
        L += ["*** Schema and roles",
              _org_table(["column(s)", "kind", "role", "declared", "observed", "nunique", "nulls", "notes"],
                         _collapse_schema(fr["schema"]))]
        ids = {r: v for r, v in fr["roles"].items() if v and Role(r) in IDENTIFIER_ROLES}
        if ids:
            L += ["*** Identifiers (history / ordering / joining only — never classifier inputs)",
                  _org_table(["role", "columns"], [[r, ", ".join(v)] for r, v in ids.items()])]
        if "order_key" in fr:
            o = fr["order_key"]
            L += ["*** Ordering key",
                  f"~{o['column']}~: min {o['min']}, max {o['max']}, span {o['span_seconds']} s, "
                  f"{o['unique']:,} distinct, sorted: {o['monotone_non_decreasing']}"]
        if "event_time" in fr:
            e = fr["event_time"]
            L.append(f"Event time ~{e['column']}~: {e['min']} .. {e['max']} (tz: {e['tz']})")
        if "target" in fr:
            t = fr["target"]
            L += ["*** Target", f"~{t['column']}~: {t['counts']} — positives {t['positives']:,}, "
                                f"prevalence {t['prevalence']:.4%}, nulls {t['nulls']}"]
        if "duplicates" in fr:
            d = fr["duplicates"]
            L += ["*** Duplicates", f"Exact redundant rows: {d['exact_redundant_rows']:,}; ignoring "
                                    f"{d['ignoring'] or 'nothing'}: {d['redundant_rows_ignoring_identifiers']:,}"
                  + (f"; duplicated row ids: {d['duplicated_row_ids']:,}" if "duplicated_row_ids" in d else "")]
        if "entity" in fr:
            e = fr["entity"]
            L += ["*** Entity key", f"~{e['column']}~: {e['entities']:,} entities; transactions per entity "
                                    f"(min/p10/median/p90/max) {[e['tx_per_entity'][k] for k in ('0.0','0.1','0.5','0.9','1.0')]}"
                  + (f"; entities with ≥1 fraud {e['entities_with_fraud']:,}; share of fraud in the top decile "
                     f"of entities {e['fraud_share_top_decile_entities']:.1%}" if "entities_with_fraud" in e else "")]
        if "amount" in fr:
            a = fr["amount"]
            rows = [["all"] + [a["overall"].get(k) for k in ("count", "mean", "50%", "max")]]
            for k, v in a.get("by_class", {}).items():
                rows.append([f"class {k}"] + [v.get(kk) for kk in ("count", "mean", "50%", "max")])
            L += ["*** Amount", f"~{a['column']}~: zero rows {a['zero_rows']:,}, negative rows {a['negative_rows']:,}",
                  _org_table(["slice", "count", "mean", "median", "max"], rows)]
        if fr.get("low_cardinality_values"):
            L.append("*** Low-cardinality columns (rows, fraud rate)")
            for col, tab in fr["low_cardinality_values"].items():
                L += [f"**** {col}", _org_table(["value", "rows", "fraud rate"],
                                                [[k, v.get("rows"), f"{v['fraud_rate']:.3%}" if "fraud_rate" in v else ""]
                                                 for k, v in tab.items()])]
        if "leakage_screen" in fr:
            ls = fr["leakage_screen"]
            top = ls["columns"][:30]
            L += ["*** Leakage screen (univariate; top 30)",
                  f"Base rate {ls['base_rate']:.4%}. AUC is max(AUC, 1−AUC) on non-null rows; for string columns the "
                  f"highest fraud rate among categories with ≥{MIN_CATEGORY_SUPPORT} rows.",
                  _org_table(["column", "role", "nunique", "AUC", "max-cat fraud rate", "max-cat (rows)",
                              "fraud rate null / present"],
                             [[r["column"], r["role"], r["nunique"],
                               f"{r['auc']:.3f}" if r.get("auc") is not None else "",
                               f"{r['max_category_fraud_rate']:.3%}" if "max_category_fraud_rate" in r else "",
                               f"{r.get('max_category', '')} ({r.get('max_category_rows', '')})" if "max_category" in r else "",
                               f"{r['fraud_rate_when_null']:.3%} / {r['fraud_rate_when_present']:.3%}"
                               if "fraud_rate_when_null" in r else ""] for r in top])]
        L.append("")
    L.append("* Findings")
    order = {"risk": 0, "warn": 1, "note": 2, "info": 3}
    for f in sorted(rep["findings"], key=lambda f: order[f["severity"]]):
        L += [f"** {f['severity'].upper()} — {f['code']}", f["message"]]
        if f["detail"]:
            L += ["#+begin_example", json.dumps(f["detail"], indent=1, ensure_ascii=False, default=str), "#+end_example"]
    L += ["", "* Capability claims",
          _org_table(["feature family", "support", "basis", "reason"],
                     [[c["family"], c["support"], ", ".join(c["basis"]), c["reason"]] for c in rep["capabilities"]]), ""]
    return "\n".join(L)


def render_capability_mapping(contracts: list[DatasetContract]) -> str:
    L = ["#+TITLE: Capability mapping — feature families per dataset", f"#+DATE: {dt.date.today()}",
         "#+OPTIONS: ^:nil", "",
         "Which feature families each dataset can *genuinely* support, from the raw-data contracts. "
         "PROVISIONAL = technically possible, semantics undecided (next step). "
         "Identifiers (entity/order/event-time/row/join keys) are listed separately from classifier inputs.", "",
         "* Matrix",
         _org_table(["feature family"] + [c.name for c in contracts],
                    [[fam.value] + [(c.claim(fam).support.value if c.claim(fam) else "?") for c in contracts]
                     for fam in FeatureFamily]), ""]
    L.append("* Identifiers vs inputs")
    L.append(_org_table(["dataset", "row id", "entity key", "order key", "event time", "join key"],
                        [[c.name, c.row_id, c.entity_key, c.order_key, c.event_time, c.join_key] for c in contracts]))
    for c in contracts:
        L += ["", f"* {c.name} — {c.title}",
              _org_table(["feature family", "support", "basis", "reason"],
                         [[k.family.value, k.support.value, ", ".join(k.basis), k.reason] for k in c.capabilities])]
    counts = {s.value: sum(1 for c in contracts for k in c.capabilities if k.support is s) for s in Support}
    L += ["", f"Totals across datasets: {counts}", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only raw-data audit (Phase 1A.1)")
    p.add_argument("--dataset", default="all", choices=["all", *ADAPTERS])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="reports")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(ADAPTERS) if a.dataset == "all" else [a.dataset]
    for name in names:
        adapter = get_adapter(name, a.data_dir)
        print(f"[audit] {name}: loading {list(LOAD_PLAN[name])} ...", flush=True)
        rep = audit_dataset(adapter)
        (out / f"audit-{name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
        (out / f"audit-{name}.org").write_text(render_org(rep))
        print(f"[audit] {name}: {len(rep['findings'])} findings, {rep['elapsed_s']} s → {out / f'audit-{name}.org'}")
    (out / "capability-mapping.org").write_text(
        render_capability_mapping([ADAPTERS[n].contract for n in ADAPTERS]))
    print(f"[audit] capability mapping → {out / 'capability-mapping.org'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
