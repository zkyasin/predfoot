"""Idempotent, resumable downloads into the data directory, recorded in the manifest.

A file at its final path is always complete: downloads go through ``.part`` files renamed at the end. A rerun
skips a file when it exists with the expected size and the manifest has a matching entry; a complete file
without an entry (run interrupted before the manifest was flushed) is checked against the remote checksum it
came with, then recorded, without downloading it again.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from predfoot.io.manifest import Manifest, crc32_file, sha256_file
from predfoot.io.remote_zip import ExtractedMember, RangeFetcher, RemoteError, RemoteZip, download_file


@dataclass
class SyncStats:
    """What a sync did."""

    downloaded: int = 0
    skipped: int = 0
    recovered: int = 0  # complete on disk but missing from the manifest
    bytes_written: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.downloaded} downloaded ({self.bytes_written / 1e6:.1f} MB), {self.skipped} already "
            f"complete, {self.recovered} recovered, {len(self.failed)} failed"
        )


class Progress:
    """Throttled progress lines on stderr."""

    def __init__(self, label: str, total: int, every_s: float = 10.0) -> None:
        self.label, self.total, self.every_s = label, total, every_s
        self.done = 0
        self.bytes = 0
        self._start = self._last = time.monotonic()

    def __call__(self, n_bytes: int) -> None:
        self.done += 1
        self.bytes += n_bytes
        now = time.monotonic()
        if now - self._last >= self.every_s or self.done == self.total:
            self._last = now
            rate = self.bytes / 1e6 / max(now - self._start, 1e-9)
            print(
                f"[{self.label}] {self.done}/{self.total} files, {self.bytes / 1e6:.0f} MB, {rate:.1f} MB/s",
                file=sys.stderr,
                flush=True,
            )


def sync_zip_members(
    remote: RemoteZip,
    targets: Mapping[str, Path],
    manifest: Manifest,
    *,
    workers: int = 8,
    max_gap: int = 1 << 16,
    label: str = "zip",
) -> SyncStats:
    """Extract the members ``{name: destination}`` that are not complete on disk yet."""
    stats = SyncStats()
    todo: dict[str, Path] = {}
    for name, dest in targets.items():
        info = remote.extent(name).info
        if dest.is_file() and dest.stat().st_size == info.file_size:
            entry = manifest.get(dest)
            if entry is not None and entry.size == info.file_size:
                stats.skipped += 1
                continue
            if crc32_file(dest) == info.CRC:
                manifest.add(dest, info.file_size, sha256_file(dest))
                stats.recovered += 1
                continue
        todo[name] = dest
    if not todo:
        return stats
    progress = Progress(label, len(todo))

    def done(member: ExtractedMember) -> None:
        manifest.add(member.path, member.size, member.sha256)
        stats.downloaded += 1
        stats.bytes_written += member.size
        progress(member.size)

    try:
        remote.extract(todo, workers=workers, max_gap=max_gap, on_done=done)
    except Exception as exc:
        failures = getattr(exc, "failures", None)
        if failures is None:
            raise
        stats.failed.update({name: repr(err) for name, err in failures.items()})
    finally:
        manifest.flush()
    return stats


def sync_file(
    fetcher: RangeFetcher,
    dest: Path,
    manifest: Manifest,
    *,
    expected_sha256: str | None = None,
    verify: Callable[[Path], bool] | None = None,
    chunk_size: int = 16 << 20,
) -> SyncStats:
    """Download one whole file unless it is already complete.

    ``expected_sha256`` (when the remote publishes one) is checked before the file is renamed into place.
    ``verify(path)`` is an extra integrity check (for example a git blob id), applied to fresh downloads and
    to complete files missing from the manifest; a fresh download that fails it is deleted.
    """
    stats = SyncStats()
    size = fetcher.size
    if dest.is_file() and dest.stat().st_size == size:
        entry = manifest.get(dest)
        if entry is not None and entry.size == size and expected_sha256 in (None, entry.sha256):
            stats.skipped += 1
            return stats
        sha = sha256_file(dest)
        if expected_sha256 in (None, sha) and (verify is None or verify(dest)):
            manifest.add(dest, size, sha)
            manifest.flush()
            stats.recovered += 1
            return stats
    last_print = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if now - last_print[0] >= 10 or done == total:
            last_print[0] = now
            print(f"[{dest.name}] {done / 1e6:.0f}/{total / 1e6:.0f} MB", file=sys.stderr, flush=True)

    size, sha = download_file(
        fetcher, dest, chunk_size=chunk_size, expected_sha256=expected_sha256, progress=progress
    )
    if verify is not None and not verify(dest):
        dest.unlink()
        raise RemoteError(f"{dest.name} failed its integrity check after download")
    manifest.add(dest, size, sha)
    manifest.flush()
    stats.downloaded += 1
    stats.bytes_written += size
    return stats
