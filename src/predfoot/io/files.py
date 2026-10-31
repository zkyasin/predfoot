"""Small file helpers: atomic writes and content hashes.

Outputs are written to a temporary sibling, flushed to disk, then renamed over the target. A rename is
atomic on POSIX file systems, so a crash leaves either the old file or the new one, never half a file; this
is what makes resuming a killed run safe.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically."""
    atomic_write_chunks(path, [data])


def atomic_write_chunks(path: Path, chunks: Iterable[bytes]) -> None:
    """Write the concatenation of ``chunks`` to ``path`` atomically, without holding it all in memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with tmp.open("wb") as fh:
        for chunk in chunks:
            fh.write(chunk)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 ``text`` to ``path`` atomically."""
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as indented JSON (no NaN), atomically, ending with a newline."""
    atomic_write_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's content."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """Hex SHA-256 of the canonical JSON form of ``obj`` (sorted keys, compact separators)."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
