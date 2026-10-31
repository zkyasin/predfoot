"""Physical KPI configuration, loaded from ``configs/kpi.yaml``.

The YAML file says which values SkillCorner publishes and which ones are assumptions of this project.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from predfoot.paths import REPO_ROOT

#: Default location of the KPI configuration.
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "kpi.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SmoothingConfig(_Frozen):
    """Savitzky–Golay smoothing of positions, applied to each gap-free segment."""

    method: Literal["savgol", "none"] = "savgol"
    window_s: float = Field(default=1.0, gt=0)
    polyorder: int = Field(default=2, ge=1)


class ExplosiveAccelerationConfig(_Frozen):
    """Explosive acceleration criteria (published by SkillCorner, see the YAML file)."""

    accel_ms2: float = Field(default=3.0, gt=0)
    min_duration_s: float = Field(default=0.7, ge=0)
    start_below_kmh: float = Field(default=9.0, gt=0)


class Psv99Config(_Frozen):
    """Peak sprint velocity: a high percentile of the speed during activities above a threshold."""

    activity_kmh: float = Field(default=15.0, gt=0)
    percentile: float = Field(default=99.0, gt=0, le=100)
    min_activity_s: float = Field(default=1.0, ge=0)
    statistic: Literal["samples", "run_peaks"] = "samples"


class KpiConfig(_Frozen):
    """All the definitions the KPI module needs."""

    native_hz: int = Field(default=25, gt=0)
    output_hz: int = Field(default=10, gt=0)
    smoothing: SmoothingConfig = SmoothingConfig()
    hsr_kmh: float = Field(default=20.0, gt=0)
    sprint_kmh: float = Field(default=25.0, gt=0)
    band_edges_kmh: tuple[float, ...] = (15.0, 20.0, 25.0)
    min_effort_s: float = Field(default=1.0, ge=0)
    explosive_acceleration: ExplosiveAccelerationConfig = ExplosiveAccelerationConfig()
    psv99: Psv99Config = Psv99Config()

    @model_validator(mode="after")
    def _check_thresholds(self) -> KpiConfig:
        edges = self.band_edges_kmh
        if not edges or any(b <= a for a, b in pairwise(edges)) or edges[0] <= 0:
            raise ValueError("band_edges_kmh must be positive and strictly increasing")
        if self.sprint_kmh <= self.hsr_kmh:
            raise ValueError("sprint_kmh must be above hsr_kmh")
        return self


def load_kpi_config(path: Path | str | None = None) -> KpiConfig:
    """Read and validate a KPI configuration (``configs/kpi.yaml`` by default)."""
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return KpiConfig.model_validate(raw)
