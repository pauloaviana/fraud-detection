"""Render Phase 1B results (experiments/<dataset>/<protocol>/results.json) as an org report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit import _org_table


def _f(x, nd=4):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render(r: dict[str, Any]) -> str:
    spec, lk = r["spec"], r["locked"]
    L = [f"#+TITLE: Phase 1B results — {spec['dataset']} / {spec['protocol']}", f"#+DATE: {r['started'][:10]}",
         "#+OPTIONS: ^:nil toc:2", "",
         f"1A bundle sha256 {r['bundle_sha256'][:16]} ({', '.join(r['bundle_files'])}); frauddet {r['frauddet_version']}; "
         f"env {r['environment']}. Parts: {r['parts']}; holdout {r['holdout']['rows']} rows / {r['holdout']['positives']} positives, "
         f"unsealed once at {r['holdout']['unsealed_at']} (after lock at {lk['locked_at']}). Run time {r['seconds']} s.", "",
         "* Design",
         f"Models {spec['models']}; treatments {spec['treatments']} + {spec['extra_treatments']}; inner folds "
         f"{r['cv'][0]['folds']['kind']} × {spec['n_folds']}; up to {spec['max_configs']} configs per model; C_a = {spec['ca']}; "
         f"selection features: {len(r['selection_features']) if r['selection_features'] else 'all'}.",
         "Imbalance applied to fold-training slices only; early stopping on the fold-validation slice; calibrators and "
         "thresholds from out-of-fold training predictions; champion by validation PR-AUC (calibrator by validation log loss); "
         "holdout evaluated once for the locked candidate.", "",
         "* Cross-validation (best config per model × treatment)",
         _org_table(["model", "treatment", "mean PR-AUC", "std", "best params", "iterations", "s/config"],
                    [[c["model"], c["treatment"], _f(c["best"]["mean_pr_auc"]), _f(c["best"]["std_pr_auc"]),
                      json.dumps(c["best"]["params"]), c["best"]["best_iterations"], c["best"]["seconds"]] for c in r["cv"]]), ""]
    has_val = any("validation" in c for c in r["candidates"])
    if has_val:
        rows = []
        for c in r["candidates"]:
            v = c["validation"]
            best_cal = min(v, key=lambda k: v[k]["calibration"]["log_loss"])
            d = v["none"]["discrimination"]; cal = v[best_cal]["calibration"]
            op = v[best_cal]["operating_points"]["f1_max"]
            rows.append([c["model"], c["treatment"], _f(d["pr_auc"]), _f(d["roc_auc"]), _f(d["recall_at_fpr_0.005"], 3),
                         _f(d["precision_at_alert_0.01"], 3), best_cal, _f(cal["brier"], 5), _f(cal["ece"], 4),
                         _f(op["f1"], 3), _f(op["mcc"], 3), _f(op["business"]["savings"], 3) if "business" in op else "",
                         c["model_size_bytes"]])
        L += ["* Validation (models refit on the full training part)",
              _org_table(["model", "treatment", "PR-AUC", "ROC-AUC", "R@FPR0.5%", "P@1%alerts", "best cal", "Brier", "ECE",
                          "F1(f1max)", "MCC", "savings(f1max)", "size B"], rows), ""]
    L += ["* Locked configuration",
          f"Champion: {lk['champion']} — selected by {lk['selection_metric']} = {_f(lk['selection_value'])}; calibrator {lk['calibrator']}.",
          "Ranking: " + "; ".join(f"{x['model']}/{x['treatment']} {_f(x['pr_auc'])}" for x in lk["ranking"]),
          "Thresholds (selected on OOF training predictions): " + ", ".join(f"{k}={_f(v, 5)}" for k, v in lk["thresholds"].items()), ""]
    h = r["holdout"]["champion"]
    d, c = h["discrimination"], h["calibration"]
    L += ["* Final holdout (evaluated once)",
          _org_table(["PR-AUC", "ROC-AUC", "R@FPR0.1%", "R@FPR0.5%", "R@FPR1%", "P@0.5%alerts", "P@1%alerts", "Brier", "log loss", "ECE"],
                     [[_f(d["pr_auc"]), _f(d["roc_auc"]), _f(d["recall_at_fpr_0.001"], 3), _f(d["recall_at_fpr_0.005"], 3),
                       _f(d["recall_at_fpr_0.01"], 3), _f(d["precision_at_alert_0.005"], 3), _f(d["precision_at_alert_0.01"], 3),
                       _f(c["brier"], 5), _f(c["log_loss"], 5), _f(c["ece"], 4)]]),
          f"Uncalibrated Brier {_f(r['holdout']['raw_uncalibrated']['calibration']['brier'], 5)}, ECE "
          f"{_f(r['holdout']['raw_uncalibrated']['calibration']['ece'], 4)}." +
          (f" Dummy baseline PR-AUC {_f(r['holdout']['dummy']['discrimination']['pr_auc'])}." if "dummy" in r["holdout"] else ""), "",
          "** Operating points on the holdout (thresholds fixed before unsealing)",
          _org_table(["policy", "thr", "alert rate", "precision", "recall", "F1", "MCC", "FP/10k", "TP", "FP", "FN",
                      "savings", "amount recall", "legit declined"],
                     [[k, _f(op["threshold"], 5), _f(op["alert_rate"], 4), _f(op["precision"], 3), _f(op["recall"], 3), _f(op["f1"], 3),
                       _f(op["mcc"], 3), _f(op["fp_per_10k"], 1), op["tp"], op["fp"], op["fn"],
                       _f(op["business"]["savings"], 3), _f(op["business"]["amount_recall"], 3), op["business"]["legit_declined"]]
                      for k, op in h["operating_points"].items()]), ""]
    if "online_parity" in r["holdout"]:
        op = r["holdout"]["online_parity"]
        L += [f"Sequential-scoring parity on {op['events']} holdout events (state warmed with train+val): max |Δp| = "
              f"{op['max_abs_diff']:.2e}, identical within 1e-6: {op['identical_within_1e-6']}.", ""]
    lt = r.get("latency", {})
    if "error" in lt or not lt:
        L += ["* Inference characteristics", f"not available: {lt.get('error', 'not measured')}", ""]
        lt = None
    if lt:
        L += ["* Inference characteristics",
          _org_table(["model size (B)", "feature ms/event p50", "p95", "model ms/event p50", "p95", "batch rows/s (model)", "batch rows/s (features+view)"],
                     [[lt["model_size_bytes"], _f(lt["feature_ms_per_event"]["p50"], 2), _f(lt["feature_ms_per_event"]["p95"], 2),
                       _f(lt["model_ms_per_event"]["p50"], 3), _f(lt["model_ms_per_event"]["p95"], 3),
                       _f(lt["batch_rows_per_second"]["model"], 0), _f(lt["batch_rows_per_second"]["feature_layer_and_view"], 0)]]), ""]
    imp = r.get("importance", {})
    if "error" in imp or not imp:
        return "\n".join(L) + f"\n* Feature diagnostics\nnot available: {imp.get('error', 'not computed')}\n"
    L += [f"* Feature diagnostics ({imp['type']}, champion)",
          _org_table(["feature", "importance"], [[f, _f(v, 2)] for f, v in imp["top"][:20]])]
    if "permutation_pr_auc_drop_validation" in imp:
        L += ["Permutation drop in validation PR-AUC (top features):",
              _org_table(["feature", "ΔPR-AUC"], [[f, _f(v, 4)] for f, v in imp["permutation_pr_auc_drop_validation"]])]
    return "\n".join(L) + "\n"


def summary_rows(results: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for r in results:
        lk, h = r["locked"], r["holdout"]["champion"]
        d, c = h["discrimination"], h["calibration"]
        cost_key = next((k for k in h["operating_points"] if k.startswith("cost")), None)
        fpr_key = "fpr_0.005"
        ops = h["operating_points"]
        lt = r.get("latency", {})
        ok = lt and "error" not in lt
        rows.append([f"{r['spec']['dataset']}/{r['spec']['protocol']}", f"{lk['champion']['model']}/{lk['champion']['treatment']}",
                     lk["calibrator"], _f(r["holdout"].get("dummy", {}).get("discrimination", {}).get("pr_auc")),
                     _f(d["pr_auc"]), _f(d["roc_auc"]), _f(d["recall_at_fpr_0.005"], 3), _f(d["precision_at_alert_0.01"], 3),
                     _f(c["brier"], 5), _f(c["ece"], 4),
                     _f(ops[fpr_key]["business"]["savings"], 3) if fpr_key in ops else "",
                     _f(ops[cost_key]["business"]["savings"], 3) if cost_key else "",
                     _f(lt["feature_ms_per_event"]["p50"], 2) if ok else "", _f(lt["model_ms_per_event"]["p50"], 3) if ok else "",
                     lt["model_size_bytes"] if ok else ""])
    return rows


SUMMARY_HEADERS = ["bundle", "champion", "cal", "dummy PR-AUC", "PR-AUC", "ROC-AUC", "R@FPR0.5%", "P@1%", "Brier", "ECE",
                   "savings@FPR0.5%", "savings@cost", "feat ms", "model ms", "size B"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", default="experiments")
    p.add_argument("--reports", default="reports")
    a = p.parse_args(argv)
    results = []
    for f in sorted(Path(a.experiments).glob("*/*/results.json")):
        r = json.loads(f.read_text())
        results.append(r)
        suffix = "" if r["spec"]["protocol"] == "temporal" else f"-{r['spec']['protocol']}"
        out = Path(a.reports) / f"1B-{r['spec']['dataset']}{suffix}.org"
        out.write_text(render(r))
        print(f"[report1b] {f} → {out}")
    if results:
        out = Path(a.reports) / "1B-summary.org"
        out.write_text("#+TITLE: Phase 1B — holdout summary across the suite\n#+OPTIONS: ^:nil\n\n"
                       "Different domains: rows are NOT a leaderboard. Sparkov is synthetic (pipeline validation only).\n\n"
                       + _org_table(SUMMARY_HEADERS, summary_rows(results)) + "\n")
        print(f"[report1b] summary → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
