"""Synthetic serving artifacts for CI / container smoke tests (NOT a model of anything).

Builds, from random ULB-shaped rows, a real 1A bundle (frozen contract, fitted pipeline, views, bundle.json)
and a real 1B artifact set (LightGBM model, calibrator, policy, locked.json) in the same layout the service
expects, so the API, container and CI can be exercised end to end without the datasets. The data is
synthetic and the numbers are meaningless; the code paths are the production ones.

    python -m frauddet.demo_artifacts --out /tmp/demo   # -> /tmp/demo/artifacts/ulb/temporal, /tmp/demo/experiments/ulb/temporal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import ADAPTERS
from .calibrate import Calibrator
from .models import Model
from .policy import select_thresholds
from .preprocessing import build_pipeline
from .serving import FeatureBundle
from .splits import temporal_split
from .views import ModelView


def synthetic_ulb(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Time": np.sort(rng.integers(0, 2 * 86400, n)).astype(float),
                       "Amount": np.round(rng.gamma(2, 30, n), 2)})
    y = (rng.random(n) < 0.04).astype(int)
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n) + (0.8 * y if i in (4, 10, 12, 14, 17) else 0)
    df["Class"] = y
    return df


def build(out: Path, n: int = 4000, seed: int = 0) -> dict:
    c = ADAPTERS["ulb"].contract
    df = synthetic_ulb(n, seed)
    sp = temporal_split(df, c, (0.7, 0.15, 0.15))
    train = df.iloc[sp.parts["train"]]
    pipe = build_pipeline(c).fit(train, order_key="Time")
    X = pipe.transform(train)
    views = {"tree": ModelView("tree").fit(X), "linear": ModelView("linear").fit(X)}
    required = tuple(col for col in df.columns if col != "Class")
    bdir = out / "artifacts" / "ulb" / "temporal"
    FeatureBundle(c, "temporal", pipe, views, None, None, None, required).save(bdir)
    Xt = views["tree"].transform(X)
    y = train["Class"].to_numpy()
    model = Model("lightgbm", {"num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 20}, n_jobs=2).fit(
        Xt, y, n_estimators=60)
    p = model.predict_proba(Xt)
    cal = Calibrator("platt").fit(p, y)
    policy = select_thresholds(y, cal.transform(p), train["Amount"].to_numpy(), 1.0)
    mdir = out / "experiments" / "ulb" / "temporal"
    mdir.mkdir(parents=True, exist_ok=True)
    model.save(mdir / "model")
    cal.save(mdir / "calibrator.json")
    (mdir / "policy.json").write_text(json.dumps(policy, indent=1, default=float))
    locked = {"champion": {"model": "lightgbm", "treatment": "none", "params": model.params, "n_estimators": 60},
              "calibrator": "platt", "thresholds": policy["thresholds"], "note": "SYNTHETIC demo artifact for smoke tests"}
    (mdir / "locked.json").write_text(json.dumps(locked, indent=1, default=float))
    return {"bundle": str(bdir), "model": str(mdir), "rows": n}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=4000)
    a = p.parse_args(argv)
    print(json.dumps(build(Path(a.out), a.rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
