"""FastAPI application: GET /health, GET /health/live, POST /predict.

Resources are loaded once in the lifespan; readiness reflects whether they loaded. Every request gets a
correlation id (X-Request-ID, honoured if the client sends one) that appears in the response headers, the
response body and the structured logs. Validation errors keep FastAPI's standard 422 body (with the
request id added); ordering/idempotency violations of a stateful bundle return 409.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal, cast

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .. import __version__
from .config import Settings
from .logging import configure_logging, safe_id
from .schemas import ErrorResponse, HealthResponse, PredictResponse, Timing
from .service import OrderingError, ScoringService

log = logging.getLogger("frauddet.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        app.state.settings = settings
        app.state.started = time.time()
        app.state.service = None
        app.state.load_error = None
        try:
            app.state.service = ScoringService(settings)
            log.info("startup", extra={"request_id": None, "status": "ready", **_identity(app)})
        except Exception as e:                                  # noqa: BLE001 — readiness reports it
            app.state.load_error = f"{type(e).__name__}: {e}"
            log.error("startup_failed", extra={"request_id": None, "reason": app.state.load_error,
                                               "dataset": settings.dataset, "protocol": settings.protocol})
        yield
        log.info("shutdown", extra={"request_id": None})

    app = FastAPI(title="frauddet scoring API", version=__version__, lifespan=lifespan,
                  description="Scores one transaction with the frozen Phase 1A feature bundle and the locked "
                              "Phase 1B model. Probability and decision are separate layers.")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", None)
        errors = exc.errors()
        for e in errors:
            e.pop("input", None)         # never echo payload values back or into logs
            e.pop("url", None)
        log.warning("validation_error", extra={"request_id": rid, "n_errors": len(errors),
                                               "fields": [".".join(str(p) for p in e.get("loc", ())) for e in errors][:20]})
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                            content={"request_id": rid, "detail": errors})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None)
        log.error("unhandled_error", extra={"request_id": rid, "error": type(exc).__name__}, exc_info=exc)
        return JSONResponse(status_code=500, content=ErrorResponse(request_id=rid or "", error="internal_error").model_dump())

    # -- health -------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}},
             summary="Readiness: 200 when the bundle, model, calibrator and policy are loaded and usable")
    async def health(request: Request):
        svc: ScoringService | None = request.app.state.service
        body = HealthResponse(status="ready" if svc else "not_ready", ready=svc is not None,
                              service=settings.service_name, uptime_s=round(time.time() - request.app.state.started, 3),
                              **(_identity(request.app) if svc else {}),
                              reason=None if svc else request.app.state.load_error)
        return JSONResponse(status_code=200 if svc else status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump())

    @app.get("/health/live", summary="Liveness: the process answers")
    async def live():
        return {"status": "alive"}

    # -- predict ------------------------------------------------------------------
    @app.post("/predict", response_model=PredictResponse, responses={409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
              summary="Score one transaction")
    async def predict(request: Request):
        rid = request.state.request_id
        svc: ScoringService | None = request.app.state.service
        if svc is None:
            return JSONResponse(status_code=503, content=ErrorResponse(request_id=rid, error="not_ready",
                                                                        detail=request.app.state.load_error).model_dump())
        body = await request.body()
        try:
            # JSON mode: strict types (no "1" -> 1), datetimes as RFC 3339 strings, extra=forbid, no inf/nan,
            # cross-field invariants; malformed JSON surfaces as a standard json_invalid error
            tx = svc.request_model.model_validate_json(body)
        except ValidationError as e:
            raise RequestValidationError(e.errors())
        try:
            pred = svc.predict(tx.model_dump())
        except OrderingError as e:
            log.warning("ordering_conflict", extra={"request_id": rid, "row_id_hash": safe_id(getattr(tx, svc.contract.row_id, None)
                                                                                              if svc.contract.row_id else None)})
            return JSONResponse(status_code=status.HTTP_409_CONFLICT,
                                content=ErrorResponse(request_id=rid, error="event_order_conflict", detail=str(e)).model_dump())
        log.info("prediction", extra={"request_id": rid, "row_id_hash": safe_id(pred.row_id), "decision": pred.decision,
                                      "probability": round(pred.probability, 6), "policy": pred.policy,
                                      "threshold": pred.threshold, "latency_ms": round(pred.latency_ms, 3),
                                      "features_ms": round(pred.features_ms, 3), "model_ms": round(pred.model_ms, 3),
                                      "model_id": svc.model_id, "bundle_sha256": svc.bundle_sha256[:12],
                                      "dataset": settings.dataset, "protocol": settings.protocol})
        return PredictResponse(request_id=rid, row_id=pred.row_id, fraud_probability=pred.probability,
                               decision=cast(Literal["APPROVED", "SUSPECT"], pred.decision), policy=pred.policy, threshold=pred.threshold,
                               latency_ms=round(pred.latency_ms, 3),
                               timing=Timing(features_ms=round(pred.features_ms, 3), model_ms=round(pred.model_ms, 3)),
                               model_id=svc.model_id, bundle_sha256=svc.bundle_sha256, calibrator=svc.calibrator.method)

    return app


def _identity(app: FastAPI) -> dict[str, Any]:
    svc: ScoringService | None = app.state.service
    if svc is None:
        return {}
    return {"dataset": svc.settings.dataset, "protocol": svc.settings.protocol, "model_id": svc.model_id,
            "bundle_sha256": svc.bundle_sha256, "policy": svc.settings.policy, "stateful": svc.stateful,
            "loaded_at": svc.loaded_at, "fast_path": svc.fast is not None, "shadow_mismatches": svc.shadow_mismatches}


app = create_app()
