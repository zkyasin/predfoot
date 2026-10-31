"""SHA-256 manifest of the downloaded files (``PREDFOOT_DATA_DIR/MANIFEST.sha256``).

One line per file, ``<sha256>  <size>  <path>``, with the path relative to the data directory (POSIX
separators) and lines sorted by path. Download commands may run concurrently: every update re-reads the file
and merges into it under an exclusive lock (``MANIFEST.sha256.lock``, POSIX ``flock``), then replaces it
atomically.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import threading
import time
import zlib
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType

MANIFEST_NAME = "MANIFEST.sha256"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ManifestEntry:
    """Expected size and digest of one file."""

    path: str  # relative to the data directory, POSIX separators
    size: int
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def crc32_file(path: Path, chunk_size: int = 1 << 20) -> int:
    """CRC-32 of a file, as stored in zip archives."""
    crc = 0
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            crc = zlib.crc32(block, crc)
    return crc


def git_blob_sha1(path: Path, chunk_size: int = 1 << 20) -> str:
    """Git object id of a file (what the Hugging Face API reports for files not stored with LFS)."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def check_relpath(rel: str) -> str:
    """Normalise a manifest path and reject anything that could point outside the data directory."""
    path = PurePosixPath(rel)
    if not rel or "\\" in rel or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"manifest paths must be relative and stay inside the data directory: {rel!r}")
    return path.as_posix()


def parse_manifest(text: str) -> dict[str, ManifestEntry]:
    """Parse manifest text; raise ``ValueError`` on a malformed line."""
    entries: dict[str, ManifestEntry] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3 or not _SHA256.fullmatch(parts[0]) or not parts[1].isdigit():
            raise ValueError(f"{MANIFEST_NAME} line {lineno} is malformed: {line!r}")
        rel = check_relpath(parts[2])
        entries[rel] = ManifestEntry(rel, int(parts[1]), parts[0])
    return entries


def format_manifest(entries: Iterable[ManifestEntry]) -> str:
    return "".join(f"{e.sha256}  {e.size}  {e.path}\n" for e in sorted(entries, key=lambda e: e.path))


class Manifest:
    """The manifest of one data directory: look up entries and record new files.

    New entries are buffered and merged into the file every ``flush_every`` entries or ``flush_interval_s``
    seconds, and on :meth:`flush` or when the context manager exits. Thread-safe.
    """

    def __init__(self, root: Path, *, flush_every: int = 500, flush_interval_s: float = 30.0) -> None:
        self.root = root.resolve()
        self.path = self.root / MANIFEST_NAME
        self._lock_path = self.root / f"{MANIFEST_NAME}.lock"
        self._flush_every = flush_every
        self._flush_interval_s = flush_interval_s
        self._mutex = threading.Lock()
        self._pending: dict[str, ManifestEntry] = {}
        self._last_flush = time.monotonic()
        self._entries = self._read()

    def __enter__(self) -> Manifest:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.flush()

    def relpath(self, path: Path) -> str:
        return check_relpath(path.resolve().relative_to(self.root).as_posix())

    def entries(self) -> dict[str, ManifestEntry]:
        with self._mutex:
            return {**self._entries, **self._pending}

    def get(self, path: Path) -> ManifestEntry | None:
        rel = self.relpath(path)
        with self._mutex:
            return self._pending.get(rel) or self._entries.get(rel)

    def add(self, path: Path, size: int, sha256: str) -> ManifestEntry:
        if not _SHA256.fullmatch(sha256):
            raise ValueError(f"not a hex sha256: {sha256!r}")
        entry = ManifestEntry(self.relpath(path), size, sha256)
        with self._mutex:
            self._pending[entry.path] = entry
            due = (
                len(self._pending) >= self._flush_every
                or time.monotonic() - self._last_flush >= self._flush_interval_s
            )
        if due:
            self.flush()
        return entry

    def flush(self) -> None:
        with self._mutex:
            pending, self._pending = self._pending, {}
            self._last_flush = time.monotonic()
            if not pending:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            with self._file_lock():
                merged = {**self._read(), **pending}
                tmp = self.path.with_name(f"{MANIFEST_NAME}.tmp.{os.getpid()}")
                tmp.write_text(format_manifest(merged.values()))
                os.replace(tmp, self.path)
            self._entries = merged

    def _read(self) -> dict[str, ManifestEntry]:
        return parse_manifest(self.path.read_text()) if self.path.exists() else {}

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        with self._lock_path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class Problem:
    """One manifest entry that does not match the file on disk."""

    path: str
    kind: str  # "missing", "size" or "sha256"
    detail: str


def verify_manifest(root: Path, *, jobs: int = 4, prefix: str = "") -> tuple[int, list[Problem]]:
    """Check every manifest entry under ``prefix``: the file exists, has the size and the SHA-256 recorded.

    Returns the number of entries checked and the problems found.
    """
    root = root.resolve()
    path = root / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {root}")
    entries = [e for e in parse_manifest(path.read_text()).values() if e.path.startswith(prefix)]

    def check(entry: ManifestEntry) -> Problem | None:
        file = root / entry.path
        if not file.is_file():
            return Problem(entry.path, "missing", "file not found")
        size = file.stat().st_size
        if size != entry.size:
            return Problem(entry.path, "size", f"{size} bytes, expected {entry.size}")
        sha = sha256_file(file)
        if sha != entry.sha256:
            return Problem(entry.path, "sha256", f"{sha}, expected {entry.sha256}")
        return None

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        problems = [p for p in pool.map(check, entries) if p is not None]
    return len(entries), problems
