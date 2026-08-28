"""Service configuration from environment variables (no secrets; nothing here is sensitive)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    dataset: str = "sparkov"                       # which frozen bundle / locked model to serve
    protocol: str = "temporal"
    artifacts_dir: Path = Path("artifacts")        # 1A bundles: <artifacts>/<dataset>/<protocol>/
    experiments_dir: Path = Path("experiments")    # 1B outputs: <experiments>/<dataset>/<protocol>/
    policy: str = "f1_max"                         # decision policy name from policy.json
    strict_order: bool = True                      # stateful bundles: reject out-of-order events
    fast_path: bool = True                         # row-native execution of the frozen steps (bit-identical)
    shadow_checks: int = 50                        # first N requests also run the pandas reference and compare
    log_level: str = "INFO"
    service_name: str = "frauddet-api"
    version: str = field(default="")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        return cls(
            dataset=e.get("FRAUDDET_DATASET", cls.dataset),
            protocol=e.get("FRAUDDET_PROTOCOL", cls.protocol),
            artifacts_dir=Path(e.get("FRAUDDET_ARTIFACTS", str(cls.artifacts_dir))),
            experiments_dir=Path(e.get("FRAUDDET_EXPERIMENTS", str(cls.experiments_dir))),
            policy=e.get("FRAUDDET_POLICY", cls.policy),
            strict_order=e.get("FRAUDDET_STRICT_ORDER", "1") not in ("0", "false", "False"),
            fast_path=e.get("FRAUDDET_FAST_PATH", "1") not in ("0", "false", "False"),
            shadow_checks=int(e.get("FRAUDDET_SHADOW_CHECKS", "50")),
            log_level=e.get("FRAUDDET_LOG_LEVEL", cls.log_level),
            service_name=e.get("FRAUDDET_SERVICE_NAME", cls.service_name),
            version=e.get("FRAUDDET_VERSION", ""),
        )

    @property
    def bundle_dir(self) -> Path:
        return self.artifacts_dir / self.dataset / self.protocol

    @property
    def model_dir(self) -> Path:
        return self.experiments_dir / self.dataset / self.protocol
