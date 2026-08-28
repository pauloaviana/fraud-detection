"""Benchmarks (Phase 3A): in-process scoring latency (reference vs fast path) and HTTP load tests.

    python -m frauddet.bench service --dataset sparkov --events 500        # p50/p95/p99 per path
    python -m frauddet.bench http --url http://127.0.0.1:8000 --dataset ieee --events 2000 --concurrency 1 8 32

Events are real rows of the dataset (labels dropped), so both paths and the server see production-shaped
input. The HTTP test reports throughput and end-to-end client latency percentiles per concurrency level.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .api.config import Settings
from .api.service import ScoringService


def real_events(dataset: str, protocol: str, n: int, data_dir="data", artifacts="artifacts") -> list[dict[str, Any]]:
    from .adapters import get_adapter
    from .adapters.ieee import canonical_name
    a = get_adapter(dataset, data_dir)
    svc_fields = ScoringService(Settings(dataset=dataset, protocol=protocol, shadow_checks=0)).fields
    if dataset == "sparkov":
        df = a.load("train")
        cut = json.load(open(Path(artifacts) / dataset / protocol / "split.json"))["boundaries"]["train"]
        df = df[df["unix_time"] > cut].sort_values("unix_time", kind="stable").head(n)
    elif dataset == "ieee":
        tx = a.load("train", nrows=n)
        ident = a.load("train_identity", nrows=max(n * 4, 20000))
        ident.columns = [canonical_name(c) for c in ident.columns]
        df = tx.merge(ident, on="TransactionID", how="left")
    else:
        df = a.load("train", nrows=n)
    out = []
    for r in df.to_dict("records"):
        ev = {}
        for k in svc_fields:
            v = r.get(k)
            if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                ev[k] = None
            elif isinstance(v, pd.Timestamp):
                ev[k] = v.isoformat()
            elif isinstance(v, (np.integer,)):
                ev[k] = int(v)
            elif isinstance(v, (np.floating,)):
                ev[k] = float(v)
            else:
                ev[k] = v
        out.append(ev)
    return out


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def bench_service(dataset: str, protocol: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    res = {}
    for name, fast in (("reference", False), ("fast", True)):
        svc = ScoringService(Settings(dataset=dataset, protocol=protocol, fast_path=fast, shadow_checks=0))
        lat, feat, mod = [], [], []
        for ev in events:
            tx = svc.request_model.model_validate_json(json.dumps(ev, default=float)).model_dump()
            p = svc.predict(tx)
            lat.append(p.latency_ms); feat.append(p.features_ms); mod.append(p.model_ms)
        lat, feat, mod = lat[5:], feat[5:], mod[5:]           # drop warm-up
        res[name] = {"n": len(lat), "p50": _pct(lat, 50), "p95": _pct(lat, 95), "p99": _pct(lat, 99),
                     "features_p50": _pct(feat, 50), "model_p50": _pct(mod, 50), "mean": statistics.mean(lat)}
    res["speedup_p50"] = res["reference"]["p50"] / res["fast"]["p50"]
    return res


def bench_http(url: str, events: list[dict[str, Any]], concurrency: int, warmup: int = 100) -> dict[str, Any]:
    """Client-side load test. ``warmup`` requests are sent first (sequentially, untimed) so that the service's
    start-up shadow checks and JIT costs do not pollute the percentiles; for stateful bundles the warm-up
    events are consumed from the sequence (they are not replayed)."""
    import httpx
    payloads = [json.dumps(ev, default=float).encode() for ev in events]
    if warmup:
        with httpx.Client(base_url=url, timeout=30) as c:
            for b in payloads[:warmup]:
                c.post("/predict", content=b, headers={"content-type": "application/json"})
        payloads = payloads[warmup:]
    lat: list[float] = []
    errors = [0]
    lock = threading.Lock()
    idx = [0]

    def worker():
        with httpx.Client(base_url=url, timeout=30) as c:
            while True:
                with lock:
                    i = idx[0]
                    if i >= len(payloads):
                        return
                    idx[0] += 1
                t0 = time.perf_counter()
                r = c.post("/predict", content=payloads[i], headers={"content-type": "application/json"})
                dt = (time.perf_counter() - t0) * 1e3
                with lock:
                    lat.append(dt)
                    if r.status_code != 200:
                        errors[0] += 1

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    return {"concurrency": concurrency, "requests": len(lat), "errors": errors[0], "wall_s": round(wall, 2),
            "throughput_rps": round(len(lat) / wall, 1), "p50_ms": round(_pct(lat, 50), 2),
            "p95_ms": round(_pct(lat, 95), 2), "p99_ms": round(_pct(lat, 99), 2)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("service"); s.add_argument("--dataset", required=True); s.add_argument("--protocol", default="temporal")
    s.add_argument("--events", type=int, default=500)
    h = sub.add_parser("http"); h.add_argument("--url", required=True); h.add_argument("--dataset", required=True)
    h.add_argument("--protocol", default="temporal"); h.add_argument("--events", type=int, default=2000)
    h.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    a = p.parse_args(argv)
    events = real_events(a.dataset, a.protocol, a.events)
    if a.cmd == "service":
        print(json.dumps({"dataset": a.dataset, "protocol": a.protocol, **bench_service(a.dataset, a.protocol, events)}, indent=1))
    else:
        for i, c in enumerate(a.concurrency):
            # stateful bundles need in-order per-entity delivery (concurrency > 1 would raise 409s); events are
            # consumed once, so each concurrency level scores a fresh slice of the sequence
            n = len(events) // len(a.concurrency)
            print(json.dumps({"dataset": a.dataset, **bench_http(a.url, events[i * n:(i + 1) * n], c, warmup=100 if i == 0 else 0)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
