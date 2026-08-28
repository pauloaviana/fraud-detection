"""Phase 1B experiment runner: train → CV-select → validate → lock → evaluate the holdout ONCE.

    python -m frauddet.experiment --dataset ulb --protocol temporal

Protocol per bundle (the 1A bundle is read-only; its hash is recorded):
  1. rebuild the prepared frame exactly as 1A did (prepare_frame + causal history), select parts by
     membership.csv, take the feature-layer frame per part and the model view per model family;
  2. the holdout is SEALED at load time: no code path can read it before ``lock()`` has been called;
  3. for every (model, imbalance treatment, hyperparameter config): inner training folds from 1A
     (forward-chaining or stratified), imbalance applied to the fold's training slice only, early stopping
     on the fold's validation slice; score = mean fold PR-AUC; out-of-fold predictions kept for the best
     config of each (model, treatment);
  4. refit the best config on the full training part; calibrators (none/platt/isotonic) fitted on the OOF
     predictions; thresholds selected on the OOF predictions; everything evaluated on the VALIDATION part;
  5. the champion (validation PR-AUC, then calibration and cost as tie-breaks) and its calibrator and
     threshold policies are LOCKED; the holdout is then unsealed once and evaluated for the locked
     candidate (and the dummy baseline for context);
  6. latency / throughput / size and importance diagnostics on the locked model (validation, never holdout).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .adapters import get_adapter
from .calibrate import Calibrator, fit_calibrators
from .folds import training_folds
from .history import compute_history
from .imbalance import ImbalanceSpec, resample_training_fold, sample_weights
from .labels import TARGETS
from .metrics import calibration, discrimination, full_evaluation
from .models import MODEL_VIEW, Model, search_space
from .policy import select_thresholds
from .prepare import prepare_frame
from .serving import FeatureBundle
from .views import RFGiniSelector

MA2026 = "stratified_ma2026"


@dataclass
class ExperimentSpec:
    dataset: str
    protocol: str = "temporal"
    models: tuple[str, ...] = ("dummy", "logreg", "xgboost", "lightgbm")
    treatments: tuple[str, ...] = ("none", "class_weight")
    extra_treatments: dict[str, tuple[str, ...]] = field(default_factory=dict)   # model -> extra treatments
    n_folds: int = 3
    max_configs: int = 6
    seed: int = 42
    ca: float = 1.0
    n_jobs: int = 6
    use_selection: bool = False          # Ma protocol: restrict to the 15 selected features
    parity_events: int = 200             # Sparkov: online-vs-offline check on that many holdout events

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SealedHoldout:
    """The final holdout. Any access before ``unseal()`` raises; unsealing is recorded and happens once."""

    def __init__(self, X: pd.DataFrame, y: np.ndarray, ctx: pd.DataFrame):
        self._X, self._y, self._ctx = X, y, ctx
        self.unsealed_at: str | None = None

    def unseal(self, locked: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        if self.unsealed_at is not None:
            raise RuntimeError("holdout already unsealed once")
        if not locked or "champion" not in locked:
            raise RuntimeError("configuration must be locked before unsealing the holdout")
        self.unsealed_at = dt.datetime.now().isoformat(timespec="seconds")
        return self._X, self._y, self._ctx

    @property
    def size(self) -> int:
        return len(self._y)


@dataclass
class BundleData:
    bundle: FeatureBundle
    bundle_sha: str
    parts: dict[str, pd.DataFrame]      # feature-layer typed frames
    y: dict[str, np.ndarray]
    ctx: dict[str, pd.DataFrame]        # cost context (row id, order, amount, y)
    holdout: SealedHoldout | None
    raw_parts: dict[str, pd.DataFrame]  # raw prepared rows (for online parity checks)
    part_names: list[str]


def load_bundle_data(dataset: str, protocol: str, data_dir="data", artifacts="artifacts") -> BundleData:
    adir = Path(artifacts) / dataset / protocol
    bundle = FeatureBundle.load(adir)
    bundle_sha = hashlib.sha256((adir / "bundle.json").read_bytes()).hexdigest()
    adapter = get_adapter(dataset, data_dir)
    df, _ = prepare_frame(adapter, protocol)
    if bundle.history_spec is not None:
        df = compute_history(df, bundle.history_spec)         # 1A semantics: prior events only, whole frame
    member = pd.read_csv(adir / "membership.csv")
    names = list(dict.fromkeys(member["part"]))
    y_all = TARGETS[dataset].binary(df).to_numpy()
    parts, ys, ctxs, raws = {}, {}, {}, {}
    for n in names:
        rows = member.loc[member["part"] == n, "row"].to_numpy()
        raws[n] = df.iloc[rows]
        parts[n] = bundle.pipeline.transform(raws[n])
        ys[n] = y_all[rows]
        ctxs[n] = pd.read_csv(adir / f"cost-context-{n}.csv")
        assert int(ctxs[n]["y"].sum()) == int(ys[n].sum()), f"{n}: cost context does not match labels"
    final = names[-1]
    holdout = SealedHoldout(parts.pop(final), ys.pop(final), ctxs.pop(final))
    raw_hold = raws.pop(final)
    if bundle.history_spec is not None:            # stateful: raw rows needed for warm-up / online parity
        raws["__holdout__"] = raw_hold
    else:                                          # stateless: keep only what latency measurement reads
        val = names[1] if len(names) > 2 else names[0]
        raws = {val: raws[val].head(10000).copy()}
    del df
    return BundleData(bundle, bundle_sha, parts, ys, ctxs, holdout, raws, names)


# ------------------------------------------------------------------------------ helpers
def _view(data: BundleData, part_X: pd.DataFrame, model_name: str, selector: RFGiniSelector | None) -> pd.DataFrame:
    X = data.bundle.views[MODEL_VIEW[model_name]].transform(part_X)
    if selector is not None:
        cols = [c for c in selector.selected if c in X.columns]
        X = X[cols]
    return X


def _weights(y, treatment: str, amount, spec_kw) -> np.ndarray | None:
    spec = ImbalanceSpec(treatment, **spec_kw)
    if treatment in ("class_weight", "example_weight"):
        return sample_weights(y, spec, amount)
    return None


_RESAMPLE_CACHE: dict[tuple, tuple] = {}


def _fold_train(X, y, treatment: str, seed: int, cache_key: tuple | None = None):
    """Training-fold view after the imbalance treatment. Resampled folds are cached per
    (treatment, view, fold) so several models/configs reuse the same resampled slice."""
    spec = ImbalanceSpec(treatment, seed=seed)
    if treatment in ("none", "class_weight", "example_weight"):
        return X, y, {"method": treatment}
    key = (treatment, *cache_key) if cache_key else None
    if key is not None and key in _RESAMPLE_CACHE:
        return _RESAMPLE_CACHE[key]
    Xr, yr, meta = resample_training_fold(X, y, spec)
    if key is not None:
        _RESAMPLE_CACHE[key] = (Xr, yr, meta)
    return Xr, yr, meta


# ------------------------------------------------------------------------------ CV search
def run_cv(data: BundleData, spec: ExperimentSpec, model_name: str, treatment: str,
           selector: RFGiniSelector | None) -> dict[str, Any]:
    train_name = data.part_names[0]
    Xtr_full = _view(data, data.parts[train_name], model_name, selector)
    ytr = data.y[train_name]
    amount = data.ctx[train_name]["amount"].to_numpy()
    order = data.ctx[train_name]["order"].to_numpy()
    folds, fmeta = training_folds(spec.protocol, order, ytr, spec.n_folds, spec.seed)
    configs = search_space(model_name, spec.max_configs, spec.seed)
    results = []
    for cfg in configs:
        fold_scores, best_its, oof = [], [], np.full(len(ytr), np.nan)
        t0 = time.time()
        for k, (tr, va) in enumerate(folds):
            Xf, yf, rmeta = _fold_train(Xtr_full.iloc[tr], ytr[tr], treatment, spec.seed,
                                        cache_key=(spec.dataset, spec.protocol, MODEL_VIEW[model_name], bool(selector), k))
            w = _weights(yf, treatment, amount[tr] if len(yf) == len(tr) else None, {"ca": spec.ca})
            m = Model(model_name, cfg, spec.seed, spec.n_jobs).fit(Xf, yf, w, eval_set=(Xtr_full.iloc[va], ytr[va]))
            p = m.predict_proba(Xtr_full.iloc[va])
            oof[va] = p
            fold_scores.append(discrimination(ytr[va], p)["pr_auc"])
            best_its.append(m.best_iteration)
        results.append({"params": cfg, "fold_pr_auc": fold_scores, "mean_pr_auc": float(np.mean(fold_scores)),
                        "std_pr_auc": float(np.std(fold_scores)), "best_iterations": best_its,
                        "seconds": round(time.time() - t0, 1), "oof": oof})
    best = max(results, key=lambda r: r["mean_pr_auc"])
    oof_mask = ~np.isnan(best["oof"])
    return {"model": model_name, "treatment": treatment, "folds": fmeta,
            "configs": [{k: v for k, v in r.items() if k != "oof"} for r in results],
            "best": {k: v for k, v in best.items() if k != "oof"},
            "oof_pred": best["oof"][oof_mask], "oof_y": ytr[oof_mask], "oof_amount": amount[oof_mask]}


# ------------------------------------------------------------------------------ refit + validation
def refit_and_validate(data: BundleData, spec: ExperimentSpec, cv: dict[str, Any],
                       selector: RFGiniSelector | None) -> dict[str, Any]:
    train_name, val_name = data.part_names[0], data.part_names[1] if len(data.part_names) > 2 else None
    model_name, treatment = cv["model"], cv["treatment"]
    Xtr = _view(data, data.parts[train_name], model_name, selector)
    ytr = data.y[train_name]
    amount = data.ctx[train_name]["amount"].to_numpy()
    Xf, yf, _ = _fold_train(Xtr, ytr, treatment, spec.seed,
                            cache_key=(spec.dataset, spec.protocol, MODEL_VIEW[model_name], bool(selector), "full"))
    w = _weights(yf, treatment, amount if len(yf) == len(ytr) else None, {"ca": spec.ca})
    its = [i for i in cv["best"]["best_iterations"] if i]
    n_est = int(round(np.mean(its))) if its else None
    t0 = time.time()
    model = Model(model_name, cv["best"]["params"], spec.seed, spec.n_jobs).fit(Xf, yf, w, n_estimators=n_est)
    fit_s = time.time() - t0
    cals = fit_calibrators(cv["oof_pred"], cv["oof_y"])
    out: dict[str, Any] = {"model": model_name, "treatment": treatment, "params": cv["best"]["params"],
                           "n_estimators": n_est, "fit_seconds": round(fit_s, 1), "model_size_bytes": model.size_bytes,
                           "cv_mean_pr_auc": cv["best"]["mean_pr_auc"], "calibrators": {}}
    if val_name is None:      # Ma protocol: no validation part; selection uses CV only
        out["thresholds_oof"] = {c: select_thresholds(cv["oof_y"], cal.transform(cv["oof_pred"]), cv["oof_amount"], spec.ca)
                                 for c, cal in cals.items()}
        return out, model, cals
    Xva = _view(data, data.parts[val_name], model_name, selector)
    yva, ava = data.y[val_name], data.ctx[val_name]["amount"].to_numpy()
    p_raw = model.predict_proba(Xva)
    out["validation"] = {}
    out["thresholds_oof"] = {}
    for cname, cal in cals.items():
        p_oof_c = cal.transform(cv["oof_pred"])
        thr = select_thresholds(cv["oof_y"], p_oof_c, cv["oof_amount"], spec.ca)
        out["thresholds_oof"][cname] = thr
        p_val = cal.transform(p_raw)
        out["validation"][cname] = full_evaluation(yva, p_val, thr["thresholds"], ava, spec.ca)
    return out, model, cals


# ------------------------------------------------------------------------------ champion, lock, holdout
def choose_champion(candidates: list[dict[str, Any]], has_validation: bool) -> dict[str, Any]:
    """Validation PR-AUC first; calibration (Brier) and cost as tie-breaks; calibrator by validation log loss."""
    scored = []
    for c in candidates:
        if c["model"] == "dummy":
            continue
        if has_validation:
            v = c["validation"]
            best_cal = min(v, key=lambda k: v[k]["calibration"]["log_loss"])
            scored.append((v["none"]["discrimination"]["pr_auc"], -v[best_cal]["calibration"]["brier"], c, best_cal))
        else:
            scored.append((c["cv_mean_pr_auc"], 0.0, c, "none"))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    pr, _, c, cal = scored[0]
    return {"champion": {"model": c["model"], "treatment": c["treatment"], "params": c["params"],
                         "n_estimators": c["n_estimators"]},
            "calibrator": cal, "selection_metric": "validation pr_auc" if has_validation else "cv mean pr_auc",
            "selection_value": pr,
            "ranking": [{"model": t[2]["model"], "treatment": t[2]["treatment"], "pr_auc": t[0], "calibrator": t[3]}
                        for t in scored]}


def run(spec: ExperimentSpec, data_dir="data", artifacts="artifacts", out_root="experiments") -> dict[str, Any]:
    t_start = time.time()
    data = load_bundle_data(spec.dataset, spec.protocol, data_dir, artifacts)
    out = Path(out_root) / spec.dataset / spec.protocol
    out.mkdir(parents=True, exist_ok=True)
    selector = data.bundle.selector if spec.use_selection else None
    has_val = len(data.part_names) > 2
    import sklearn
    env = {"python": platform.python_version(), "sklearn": sklearn.__version__}
    for lib in ("xgboost", "lightgbm"):
        try:
            env[lib] = __import__(lib).__version__
        except Exception:
            env[lib] = None
    log = {"frauddet_version": __version__, "spec": spec.to_dict(), "bundle_sha256": data.bundle_sha,
           "bundle_files": json.loads((Path(artifacts) / spec.dataset / spec.protocol / "bundle.json").read_text())["files"],
           "environment": env, "started": dt.datetime.now().isoformat(timespec="seconds"),
           "parts": {n: {"rows": int(len(data.y[n])), "positives": int(data.y[n].sum())} for n in data.y},
           "holdout": {"rows": data.holdout.size, "sealed": True}, "selection_features": selector.selected if selector else None,
           "cv": [], "candidates": []}

    models_cals: dict[tuple[str, str], tuple[Model, dict[str, Calibrator]]] = {}
    ckpt_root = out / "checkpoint"
    for model_name in spec.models:
        treatments = list(spec.treatments) + list(spec.extra_treatments.get(model_name, ()))
        if model_name == "dummy":
            treatments = ["none"]
        for treatment in treatments:
            ck = ckpt_root / f"{model_name}__{treatment}"
            if (ck / "cand.json").exists():                      # resume: candidate already finished
                cv = json.loads((ck / "cv.json").read_text())
                oof = np.load(ck / "oof.npz")
                cv.update(oof_pred=oof["pred"], oof_y=oof["y"], oof_amount=oof["amount"])
                cand = json.loads((ck / "cand.json").read_text())
                model = Model.load(ck / "model")
                cals = fit_calibrators(cv["oof_pred"], cv["oof_y"])
                print(f"[1B] {spec.dataset}/{spec.protocol} {model_name}/{treatment}: resumed from checkpoint", flush=True)
            else:
                print(f"[1B] {spec.dataset}/{spec.protocol} {model_name}/{treatment}: CV ...", flush=True)
                cv = run_cv(data, spec, model_name, treatment, selector)
                cand, model, cals = refit_and_validate(data, spec, cv, selector)
                ck.mkdir(parents=True, exist_ok=True)
                (ck / "cv.json").write_text(json.dumps({k: v for k, v in cv.items() if not k.startswith("oof")}, default=_json_default))
                np.savez_compressed(ck / "oof.npz", pred=cv["oof_pred"], y=cv["oof_y"], amount=cv["oof_amount"])
                (ck / "cand.json").write_text(json.dumps(cand, default=_json_default))
                model.save(ck / "model")
            log["cv"].append({k: v for k, v in cv.items() if not k.startswith("oof")})
            models_cals[(model_name, treatment)] = (model, cals)
            import gc
            gc.collect()
            log["candidates"].append(cand)
            v = cand.get("validation", {}).get("none", {}).get("discrimination", {}).get("pr_auc")
            print(f"[1B]   cv pr_auc={cv['best']['mean_pr_auc']:.4f}" + (f" val pr_auc={v:.4f}" if v is not None else ""), flush=True)

    # ---- lock
    locked = choose_champion(log["candidates"], has_val)
    key = (locked["champion"]["model"], locked["champion"]["treatment"])
    model, cals = models_cals[key]
    cal = cals[locked["calibrator"]]
    cand = next(c for c in log["candidates"] if (c["model"], c["treatment"]) == key)
    policy = cand["thresholds_oof"][locked["calibrator"]]
    locked["thresholds"] = policy["thresholds"]
    locked["locked_at"] = dt.datetime.now().isoformat(timespec="seconds")
    log["locked"] = locked
    (out / "locked.json").write_text(json.dumps(locked, indent=1, default=float))
    model.save(out / "model")
    cal.save(out / "calibrator.json")
    (out / "policy.json").write_text(json.dumps(policy, indent=1, default=float))

    # ---- final holdout, once (immediately after the lock; nothing below can change the locked candidate)
    Xh, yh, ctxh = data.holdout.unseal(locked)
    Xh_v = _view(data, Xh, model.name, selector)
    p_h = cal.transform(model.predict_proba(Xh_v))
    log["holdout"] = {"unsealed_at": data.holdout.unsealed_at, "rows": int(len(yh)), "positives": int(yh.sum()),
                      "champion": full_evaluation(yh, p_h, locked["thresholds"], ctxh["amount"].to_numpy(), spec.ca),
                      "raw_uncalibrated": {"calibration": calibration(yh, model.predict_proba(Xh_v))}}
    dummy = models_cals.get(("dummy", "none"))
    if dummy:
        pd_ = dummy[0].predict_proba(_view(data, Xh, "dummy", None))
        log["holdout"]["dummy"] = {"discrimination": discrimination(yh, pd_), "calibration": calibration(yh, pd_)}
    log["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    log["seconds"] = round(time.time() - t_start, 1)
    _write_results(out, log)          # completed expensive work is now on disk
    print(f"[1B] {spec.dataset}/{spec.protocol}: locked {key} cal={locked['calibrator']} "
          f"holdout pr_auc={log['holdout']['champion']['discrimination']['pr_auc']:.4f} → {out / 'results.json'}", flush=True)

    # ---- diagnostics (validation only), latency, parity — each guarded so a failure cannot lose the run
    def _guarded(name, fn):
        try:
            log[name] = fn()
        except Exception as e:                       # noqa: BLE001
            log[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[1B]   {name} failed: {type(e).__name__}: {e}", flush=True)
        _write_results(out, log)

    def _importance():
        imp = model.importance()
        top = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:30]
        res = {"type": "gain" if model.name in ("xgboost", "lightgbm") else "abs_coef", "top": top}
        if has_val:
            val_name = data.part_names[1]
            Xva = _view(data, data.parts[val_name], model.name, selector)
            base = discrimination(data.y[val_name], cal.transform(model.predict_proba(Xva)))["pr_auc"]
            rng = np.random.default_rng(spec.seed)
            perm = []
            for f, _ in top[:15]:
                Xp = Xva.copy()
                pidx = rng.permutation(len(Xp))
                col = Xp[f]
                if isinstance(col.dtype, pd.CategoricalDtype):        # keep the category dtype (LightGBM/XGBoost)
                    Xp[f] = pd.Categorical(col.to_numpy()[pidx], categories=col.cat.categories)
                else:
                    Xp[f] = col.to_numpy()[pidx]
                perm.append([f, base - discrimination(data.y[val_name], cal.transform(model.predict_proba(Xp)))["pr_auc"]])
            res["permutation_pr_auc_drop_validation"] = perm
        return res

    _guarded("importance", _importance)
    _guarded("latency", lambda: measure_latency(data, model, cal, selector, spec))
    if data.bundle.history_spec is not None and spec.parity_events:
        _guarded("online_parity", lambda: online_parity(data, model, cal, spec))
        log["holdout"]["online_parity"] = log.pop("online_parity")
        _write_results(out, log)
    return log


def _write_results(out: Path, log: dict[str, Any]) -> None:
    (out / "results.json").write_text(json.dumps(log, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)


# ------------------------------------------------------------------------------ latency & parity
def measure_latency(data: BundleData, model: Model, cal: Calibrator, selector, spec: ExperimentSpec,
                    n_single: int = 200, n_batch: int = 10000) -> dict[str, Any]:
    val_name = data.part_names[1] if len(data.part_names) > 2 else data.part_names[0]
    raw = data.raw_parts[val_name].head(n_batch)
    b = data.bundle
    out: dict[str, Any] = {"model_size_bytes": model.size_bytes, "n_single": n_single, "n_batch": int(len(raw))}
    # feature computation, one event at a time (stateful path for Sparkov), no model
    events = raw.head(n_single)
    if b.history_spec is not None:
        b2 = FeatureBundle.load(Path("artifacts") / spec.dataset / spec.protocol)   # fresh state (train snapshot)
    else:
        b2 = b
    t = []
    rows = []
    for ev in events.drop(columns=[c for c in b.history_spec.feature_names() if c in events.columns] if b.history_spec else []).to_dict("records"):
        t0 = time.perf_counter()
        x = b2.serve_event(ev, view=model.view, row_id=ev.get(b.contract.row_id) if b.contract.row_id else None)
        t.append(time.perf_counter() - t0)
        rows.append(x)
    t = np.array(t) * 1e3
    out["feature_ms_per_event"] = {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)), "mean": float(t.mean())}
    # model inference on one row
    t = []
    for x in rows:
        xs = x[[c for c in selector.selected if c in x.columns]] if selector else x
        t0 = time.perf_counter()
        cal.transform(model.predict_proba(xs))
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    out["model_ms_per_event"] = {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)), "mean": float(t.mean())}
    # batch throughput (feature layer + view + model on the already-prepared frame)
    Xb = _view(data, data.parts[val_name].head(n_batch), model.name, selector)
    t0 = time.perf_counter(); cal.transform(model.predict_proba(Xb)); dt_model = time.perf_counter() - t0
    t0 = time.perf_counter(); b.pipeline.transform(raw); b.views[model.view].transform(data.parts[val_name].head(n_batch)); dt_feat = time.perf_counter() - t0
    out["batch_rows_per_second"] = {"model": float(len(Xb) / dt_model), "feature_layer_and_view": float(len(raw) / dt_feat)}
    return out


def online_parity(data: BundleData, model: Model, cal: Calibrator, spec: ExperimentSpec) -> dict[str, Any]:
    """Sparkov: score the first N holdout events through the streaming path (state warmed with train+val)
    and compare with the batch predictions. Uses holdout events, never their labels, and happens after lock."""
    b = data.bundle
    train, val = data.part_names[0], data.part_names[1]
    warm = pd.concat([data.raw_parts[train], data.raw_parts[val]])
    raw_cols = [c for c in warm.columns if c not in b.history_spec.feature_names()]
    b.warm_up(warm[raw_cols])
    hold = data.raw_parts["__holdout__"].head(spec.parity_events)
    online = pd.concat([b.serve_event(ev, view=model.view, row_id=ev[b.contract.row_id]) for ev in hold[raw_cols].to_dict("records")],
                       ignore_index=True)
    p_on = cal.transform(model.predict_proba(online))
    p_off = cal.transform(model.predict_proba(b.views[model.view].transform(b.pipeline.transform(hold))))
    return {"events": int(len(hold)), "max_abs_diff": float(np.max(np.abs(p_on - p_off))),
            "identical_within_1e-6": bool(np.allclose(p_on, p_off, atol=1e-6))}


# ------------------------------------------------------------------------------ CLI
def default_spec(dataset: str, protocol: str) -> ExperimentSpec:
    if protocol == MA2026:
        return ExperimentSpec(dataset, protocol, treatments=("none", "smote_enn"), n_folds=3, max_configs=6,
                              use_selection=True)
    extra = {}
    if dataset == "ulb":
        extra = {"logreg": ("smote_enn",), "xgboost": ("smote_enn",), "lightgbm": ("smote_enn",)}
        return ExperimentSpec(dataset, protocol, extra_treatments=extra, max_configs=6)
    # D36: LightGBM is the sole non-trivial model for Sparkov and IEEE (dummy baseline kept)
    return ExperimentSpec(dataset, protocol, models=("dummy", "lightgbm"),
                          extra_treatments={"lightgbm": ("random_under",)}, max_configs=6)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 1B experiments on frozen 1A bundles")
    p.add_argument("--dataset", required=True)
    p.add_argument("--protocol", default="temporal")
    p.add_argument("--max-configs", type=int, default=None)
    p.add_argument("--models", default=None, help="comma-separated subset")
    a = p.parse_args(argv)
    spec = default_spec(a.dataset, a.protocol)
    if a.max_configs:
        spec.max_configs = a.max_configs
    if a.models:
        spec.models = tuple(a.models.split(","))
    run(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
