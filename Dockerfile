# syntax=docker/dockerfile:1.7
# Production image for the frauddet scoring API.
#   docker build --build-arg DATASET=ulb --build-arg PROTOCOL=temporal \
#                --build-arg ARTIFACTS=artifacts --build-arg EXPERIMENTS=experiments -t frauddet-api .
# Stage 1 builds a virtualenv from pinned requirements; stage 2 is a slim runtime with only the venv,
# the package sources, the selected 1A bundle and the locked 1B model (.dockerignore drops data, tests,
# reports and the large non-serving artifact files). Non-root user, read-only friendly, no secrets,
# HEALTHCHECK against the readiness endpoint. libgomp1 is the OpenMP runtime LightGBM needs.

ARG PYTHON_VERSION=3.13
ARG DATASET=ulb
ARG PROTOCOL=temporal
ARG ARTIFACTS=artifacts
ARG EXPERIMENTS=experiments

# ---------------------------------------------------------------- build
FROM python:${PYTHON_VERSION}-slim-bookworm AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements-serving.txt /tmp/requirements-serving.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: --prefix /opt/venv -r /tmp/requirements-serving.txt \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
 && find /opt/venv -name "*.pyc" -delete

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ARG DATASET
ARG PROTOCOL
ARG ARTIFACTS
ARG EXPERIMENTS
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
ENV PYTHONPATH=/opt/venv/lib/python3.13/site-packages:/app/src \
    PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/app \
    FRAUDDET_DATASET=${DATASET} FRAUDDET_PROTOCOL=${PROTOCOL} \
    FRAUDDET_ARTIFACTS=/app/artifacts FRAUDDET_EXPERIMENTS=/app/experiments \
    FRAUDDET_HOST=0.0.0.0 FRAUDDET_PORT=8000 FRAUDDET_WORKERS=1
WORKDIR /app
COPY --from=build --chown=root:root /opt/venv /opt/venv
COPY --chown=root:root src/frauddet /app/src/frauddet
# only the selected bundle and locked model (large non-serving files are excluded by .dockerignore)
COPY --chown=root:root ${ARTIFACTS}/${DATASET}/${PROTOCOL}/ /app/artifacts/${DATASET}/${PROTOCOL}/
COPY --chown=root:root ${EXPERIMENTS}/${DATASET}/${PROTOCOL}/model /app/experiments/${DATASET}/${PROTOCOL}/model
COPY --chown=root:root ${EXPERIMENTS}/${DATASET}/${PROTOCOL}/calibrator.json \
     ${EXPERIMENTS}/${DATASET}/${PROTOCOL}/policy.json \
     ${EXPERIMENTS}/${DATASET}/${PROTOCOL}/locked.json \
     /app/experiments/${DATASET}/${PROTOCOL}/
RUN chmod -R a-w /app /opt/venv
USER app:app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"]
CMD ["python", "-m", "frauddet.api"]
