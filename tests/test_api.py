"""Phase 2A API tests: schema generation (unit), decision policy, /health and /predict contracts, invalid
inputs, offline/serving parity, and an end-to-end integration run. Uses synthetic demo artifacts built
from the real 1A/1B code paths, so no data files are needed; when the real ULB artifacts are present the
parity test also runs against them."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("lightgbm")
from fastapi.testclient import TestClient  # noqa: E402

from frauddet.api.app import create_app  # noqa: E402
from frauddet.api.config import Settings  # noqa: E402
from frauddet.api.logging import JsonFormatter, safe_id  # noqa: E402
from frauddet.api.schemas import build_request_model, consumed_fields  # noqa: E402
from frauddet.api.service import ScoringService  # noqa: E402
from frauddet.demo_artifacts import build, synthetic_ulb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    build(out, n=3000)
    return Settings(dataset="ulb", protocol="temporal", artifacts_dir=out / "artifacts", experiments_dir=out / "experiments",
                    policy="f1_max")


@pytest.fixture(scope="module")
def client(demo):
    with TestClient(create_app(demo)) as c:
        yield c


def event(**over):
    ev = {"Time": 1000.0, "Amount": 42.0, **{f"V{i}": round(0.05 * i, 3) for i in range(1, 29)}}
    ev.update(over)
    return ev


# ------------------------------------------------------------------------------ unit: schema & policy
def test_request_model_is_built_from_the_frozen_contract(demo):
    svc = ScoringService(demo)
    assert set(svc.fields) == {"Time", "Amount", *[f"V{i}" for i in range(1, 29)]}      # consumed fields only
    M = svc.request_model
    assert M.model_config["extra"] == "forbid" and M.model_config["strict"] is True
    ok = M.model_validate(event())
    assert ok.Amount == 42.0
    assert M.model_validate_json(json.dumps(event(Amount=42))).Amount == 42.0     # int for a float field is fine in JSON
    for bad, frag in ((event(Amount=-1), "greater than or equal"), (event(Time=-5), "greater than or equal"),
                      (event(bogus=1), "Extra inputs"), (event(V1="0.1"), "valid number"), (event(Amount=float("inf")), "finite"),
                      (event(Class=0), "Extra inputs")):
        with pytest.raises(Exception) as e:
            M.model_validate(bad)
        assert frag.lower() in str(e.value).lower()
    with pytest.raises(Exception):
        M.model_validate({k: v for k, v in event().items() if k != "V7"})              # missing required


def test_sparkov_schema_has_cross_field_invariants_and_no_pii():
    from frauddet.adapters import ADAPTERS
    from frauddet.history import HistorySpec
    from frauddet.preprocessing import build_pipeline
    c = ADAPTERS["sparkov"].contract
    pipe = build_pipeline(c)
    required = ("Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last", "gender",
                "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time",
                "merch_lat", "merch_long")
    pipe.steps[-1].state = {"feature_columns": ["merchant", "category", "amt", "gender", "state", "lat", "long", "city_pop",
                                                "merch_lat", "merch_long"]}
    fields = consumed_fields(c, pipe, HistorySpec(), required)
    assert "first" not in fields and "street" not in fields and "zip" not in fields and "Unnamed: 0" not in fields
    assert {"cc_num", "trans_num", "unix_time", "trans_date_trans_time", "dob", "amt", "merchant"} <= set(fields)
    M = build_request_model(c, fields)
    good = {"trans_date_trans_time": "2019-01-01T00:00:18", "cc_num": "1234", "merchant": "fraud_X", "category": "food",
            "amt": 10.0, "gender": "F", "state": "NC", "lat": 36.0, "long": -81.0, "city_pop": 100, "dob": "1980-01-01T00:00:00",
            "trans_num": "abc", "unix_time": 1325376018, "merch_lat": 36.1, "merch_long": -81.1}
    good = {k: v for k, v in good.items() if k in fields}
    V = lambda d: M.model_validate_json(json.dumps(d))          # the API validates in JSON mode (strict)
    V(good)
    with pytest.raises(Exception, match="time of day"):
        V({**good, "unix_time": 1325376018 + 7200})
    with pytest.raises(Exception, match="dob"):
        V({**good, "dob": "2030-01-01T00:00:00"})
    with pytest.raises(Exception, match="naive"):
        V({**good, "trans_date_trans_time": "2019-01-01T00:00:18+00:00"})
    with pytest.raises(Exception):
        V({**good, "lat": 100.0})
    with pytest.raises(Exception, match="valid integer"):
        V({**good, "unix_time": "1325376018"})                    # strict: no string -> int coercion


def test_decision_policy_is_a_separate_threshold_layer(demo):
    svc = ScoringService(demo)
    assert svc.settings.policy in svc.thresholds and 0 < svc.threshold < 1
    p = svc.predict(event())
    assert (p.decision == "SUSPECT") == (p.probability >= svc.threshold)
    other = ScoringService(Settings(**{**demo.__dict__, "policy": "alert_0.01"}))
    assert other.threshold != svc.threshold


def test_logging_is_json_and_ids_are_hashed():
    rec = logging.LogRecord("frauddet.api", logging.INFO, __file__, 1, "prediction", (), None)
    rec.request_id, rec.latency_ms = "abc", 1.5
    doc = json.loads(JsonFormatter().format(rec))
    assert doc["event"] == "prediction" and doc["request_id"] == "abc" and doc["latency_ms"] == 1.5 and "ts" in doc
    assert safe_id("t123") != "t123" and len(safe_id("t123")) == 12 and safe_id(None) is None


# ------------------------------------------------------------------------------ API contract
def test_health_ready_and_live(client):
    r = client.get("/health")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ready" and b["ready"] and b["dataset"] == "ulb" and b["model_id"].startswith("ulb/temporal:lightgbm@")
    assert client.get("/health/live").json() == {"status": "alive"}
    assert "X-Request-ID" in r.headers


def test_health_not_ready_when_artifacts_missing(tmp_path):
    bad = Settings(dataset="ulb", protocol="temporal", artifacts_dir=tmp_path, experiments_dir=tmp_path)
    with TestClient(create_app(bad)) as c:
        r = c.get("/health")
        assert r.status_code == 503 and r.json()["status"] == "not_ready" and "missing serving artifact" in r.json()["reason"]
        r = c.post("/predict", json=event())
        assert r.status_code == 503 and r.json()["error"] == "not_ready"
        assert c.get("/health/live").status_code == 200


def test_predict_contract(client):
    r = client.post("/predict", json=event(), headers={"X-Request-ID": "req-1"})
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "req-1"
    b = r.json()
    assert set(b) == {"request_id", "row_id", "fraud_probability", "decision", "policy", "threshold", "latency_ms",
                      "timing", "model_id", "bundle_sha256", "calibrator"}
    assert b["request_id"] == "req-1" and 0 <= b["fraud_probability"] <= 1 and b["decision"] in ("APPROVED", "SUSPECT")
    assert b["latency_ms"] > 0 and b["timing"]["features_ms"] > 0 and len(b["bundle_sha256"]) == 64
    assert (b["decision"] == "SUSPECT") == (b["fraud_probability"] >= b["threshold"])


@pytest.mark.parametrize("payload, expect", [
    (event(Amount=-3), "greater_than_equal"), (event(surprise=1), "extra_forbidden"), (event(V2="abc"), "float_type"),
    ({k: v for k, v in event().items() if k != "Amount"}, "missing"), (event(Class=1), "extra_forbidden"),
    (event(Time=None), "float_type"),
])
def test_predict_rejects_invalid_inputs_with_422(client, payload, expect):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["request_id"] and any(e["type"] == expect for e in body["detail"]), body["detail"]
    assert all("input" not in e for e in body["detail"])          # payload values are not echoed


def test_predict_rejects_nan_and_bad_json(client):
    r = client.post("/predict", content='{"Amount": NaN, "Time": 1}', headers={"content-type": "application/json"})
    assert r.status_code == 422
    r = client.post("/predict", content="not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["detail"][0]["type"] == "json_invalid"


# ------------------------------------------------------------------------------ parity & integration
def test_serving_matches_offline_pipeline_on_synthetic_rows(demo):
    from frauddet.calibrate import Calibrator
    from frauddet.models import Model
    from frauddet.serving import FeatureBundle
    svc = ScoringService(demo)
    df = synthetic_ulb(300, seed=7)
    b = FeatureBundle.load(demo.bundle_dir); m = Model.load(demo.model_dir / "model"); cal = Calibrator.load(demo.model_dir / "calibrator.json")
    offline = cal.transform(m.predict_proba(b.transform_batch(df.drop(columns=["Class"]), view="tree")))
    online = np.array([svc.predict(row).probability for row in df.drop(columns=["Class"]).to_dict("records")])
    assert np.allclose(online, offline, atol=1e-6)


def test_serving_parity_against_real_ulb_artifacts_if_present():
    real = Settings(dataset="ulb", protocol="stratified_ma2026", artifacts_dir=ROOT / "artifacts", experiments_dir=ROOT / "experiments")
    if not (real.model_dir / "model" / "model.json").exists() or not (ROOT / "data" / "creditcardfraud.csv").exists():
        pytest.skip("real ULB artifacts/data not present")
    from frauddet.calibrate import Calibrator
    from frauddet.models import Model
    from frauddet.serving import FeatureBundle
    svc = ScoringService(real)
    df = pd.read_csv(ROOT / "data" / "creditcardfraud.csv", nrows=200)
    b = FeatureBundle.load(real.bundle_dir); m = Model.load(real.model_dir / "model"); cal = Calibrator.load(real.model_dir / "calibrator.json")
    X = b.transform_batch(df.drop(columns=["Class"]), view=m.view)
    if b.selector is not None:
        X = b.selector.transform(X)
    offline = cal.transform(m.predict_proba(X))
    online = np.array([svc.predict(row).probability for row in df.drop(columns=["Class"]).to_dict("records")])
    assert np.allclose(online, offline, atol=1e-6)


def test_end_to_end_sequence_with_logs(client, caplog):
    caplog.set_level(logging.INFO, logger="frauddet.api")
    ids = []
    for i in range(20):
        r = client.post("/predict", json=event(Time=1000.0 + i, Amount=10.0 * i), headers={"X-Request-ID": f"e2e-{i}"})
        assert r.status_code == 200
        ids.append(r.json()["request_id"])
    assert ids == [f"e2e-{i}" for i in range(20)]
    preds = [r for r in caplog.records if r.getMessage() == "prediction"]
    assert len(preds) >= 20
    doc = json.loads(JsonFormatter().format(preds[-1]))
    assert {"request_id", "latency_ms", "decision", "model_id", "bundle_sha256", "policy"} <= set(doc)
    assert "Amount" not in doc and "V1" not in doc                     # payload never logged
    assert client.get("/health").json()["ready"]
