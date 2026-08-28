"""Run the service: python -m frauddet.api  (host/port/workers via env)."""

from __future__ import annotations

import os

import uvicorn


def main() -> int:
    uvicorn.run("frauddet.api.app:app", host=os.environ.get("FRAUDDET_HOST", "0.0.0.0"),
                port=int(os.environ.get("FRAUDDET_PORT", "8000")),
                workers=int(os.environ.get("FRAUDDET_WORKERS", "1")),   # stateful bundles need ONE process
                log_config=None, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
