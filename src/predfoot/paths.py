"""Where datasets, caches and generated reports live.

Data never goes to git. Point ``PREDFOOT_DATA_DIR`` at a shared folder so that every git worktree reuses the
same downloads; without it, data goes to ``<repo>/data`` (git-ignored).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root when the package is installed in editable mode (the default with ``uv sync``).
REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Root of all downloaded datasets."""
    env = os.environ.get("PREDFOOT_DATA_DIR")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data"


def index_dir() -> Path:
    """Parquet tables derived from raw labels (clips, persons, lines)."""
    return data_dir() / "index"


def cache_dir() -> Path:
    """Model outputs cached by content hash (keypoints, calibrations)."""
    return data_dir() / "cache"


def reports_dir() -> Path:
    """Markdown reports with aggregate numbers only; these are committed."""
    return REPO_ROOT / "reports"
