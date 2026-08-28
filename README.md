# Real-Time Fraud Detection

[![CI](https://github.com/pauloaviana/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/pauloaviana/fraud-detection/actions/workflows/ci.yml)

In production, monitor feature/score drift, missingness, alert rate, delayed-label **PR-AUC / recall at fixed FPR / calibration**, API latency/errors, and state consistency. Degradation should trigger controlled retraining and challenger evaluation, not automatic replacement.

Real-time fraud scoring pipeline with leakage-safe preprocessing, frozen artifacts, explicit decision policies, and a FastAPI serving layer. The repository includes the model and preprocessing artifacts required to run the supported bundles.

## Run with Docker

```bash
git clone https://github.com/pauloaviana/fraud-detection.git
cd fraud-detection
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

Select a bundle:

```bash
FRAUDDET_DATASET=ieee FRAUDDET_PROTOCOL=temporal docker compose up -d --build
FRAUDDET_DATASET=sparkov FRAUDDET_PROTOCOL=temporal docker compose up -d --build
```

Other supported protocols include `ulb/temporal` and `ulb/stratified_ma2026`.

API:
- `GET /health/live` — liveness
- `GET /health` — readiness and loaded artifact metadata
- `POST /predict` — fraud probability + policy decision
- `/docs` — OpenAPI schema

Stop with:

```bash
docker compose down
```

## Technical choices

**Model — LightGBM.** Gradient-boosted trees fit this problem well because the inputs are mostly structured/tabular, with nonlinear interactions, missing values, high-cardinality fields, and strong class imbalance. LightGBM is the primary production family because it delivered strong real-data discrimination while remaining compact and fast for single-transaction inference. The reference configuration also avoids resampling, reducing pipeline complexity and avoiding synthetic changes to the training distribution. The locked ULB temporal experiment keeps its XGBoost + Platt winner rather than forcing one model family across every benchmark.

**Metrics — PR-AUC first.** Fraud is rare, so accuracy can look excellent while the detector is useless. PR-AUC therefore serves as the main ranking metric because it focuses on performance for the positive class. ROC-AUC is kept as a secondary discrimination metric, while precision, recall, F1/MCC, recall at fixed FPR or alert budgets, Brier score, and calibration error cover operational and probabilistic quality. The model outputs a probability; the approve/suspect threshold is a separate policy so fraud capture can be traded against customer friction and investigation cost.

**API — FastAPI + Pydantic v2.** The serving layer is intentionally small and typed. FastAPI gives low-overhead HTTP serving, OpenAPI documentation, and simple health/readiness endpoints; Pydantic enforces strict schemas generated from the frozen data contracts. Models and preprocessing artifacts are loaded once at startup, prediction responses expose model/bundle identity and latency, and the optimized row-native path is checked against the reference preprocessing path for train/serve parity.

## Milestones

- [x] **Data & preprocessing** — audits, leakage-safe splits, causal features, frozen preprocessing
- [x] **Model training & evaluation** — imbalance experiments, calibration, threshold policies, holdout evaluation
- [x] **Real-time inference API** — strict schemas, configurable bundles, stateful scoring, train/serve parity
- [x] **Productionization & MLOps** — Docker, CI, structured logs, artifact integrity, fast-path/load testing
- [ ] **Online learning & concept drift** — drift detection, delayed-label monitoring, challenger/retraining policy
