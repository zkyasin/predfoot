"""Guards of the preregistered test run (tickets T3.4 and T6.1): the freeze lock and the single evaluation.

``scripts/freeze_prereg.py`` fits the frozen artifacts on dev only, from a clean tree whose preregistration is
final (no draft marker), and writes ``docs/preregistration.lock.json``: SHA-256 of the preregistration and of
every artifact, the code commit, the date and the configuration digests. ``scripts/run_test_once.py`` calls
:func:`verify_lock` before every stage and :func:`claim_once` before the only stage that reads test labels:

- the lock exists and has the expected format;
- the preregistration and every artifact have the hashes of the lock, and the artifact folder holds exactly
  the locked files;
- the code commit of the lock is an ancestor of HEAD and nothing under ``src/``, ``scripts/`` or ``configs/``
  changed since;
- the working tree is clean (git status shows nothing: no edit, no untracked file);
- the evaluation marker does not exist yet; it is written first, before any label is read, so a crash or a
  second invocation can never produce a second evaluation.

All functions take the repository root, so tests run them on throw-away repositories.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from predfoot.io.manifest import sha256_file

PREREG_PATH = Path("docs/preregistration.md")
LOCK_PATH = Path("docs/preregistration.lock.json")
ARTIFACTS_ROOT = Path("artifacts/prereg")
PROTECTED = ("src", "scripts", "configs")
LOCK_FORMAT = "predfoot-prereg-lock/1"
#: Text that marks a preregistration as not approved yet; the freeze refuses it.
DRAFT_MARKERS = ("DRAFT", "pending owner approval")
_VERSION = re.compile(r"^- Version:\s*(\d+)\s*$", re.MULTILINE)


class PreregError(RuntimeError):
    """A guard of the preregistered test run refused to go on."""


def git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise PreregError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def head_commit(root: Path) -> str:
    return git(root, "rev-parse", "HEAD").strip()


def dirty_paths(root: Path, paths: Sequence[str] = ()) -> list[str]:
    """Paths that git status reports (modified, staged, deleted or untracked), optionally under ``paths``."""
    out = git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths)
    return [line[3:] for line in out.splitlines() if line.strip()]


def require_clean_tree(root: Path) -> None:
    dirty = dirty_paths(root)
    if dirty:
        shown = ", ".join(dirty[:5]) + (" …" if len(dirty) > 5 else "")
        raise PreregError(
            f"the working tree is not clean ({len(dirty)} paths: {shown}); commit or stash first"
        )


def prereg_version(text: str) -> int:
    """The ``- Version: N`` line of the preregistration."""
    m = _VERSION.search(text)
    if m is None:
        raise PreregError("the preregistration has no '- Version: N' line")
    return int(m.group(1))


def require_final_text(text: str) -> None:
    """Refuse a preregistration that still carries a draft marker (the owner has not approved it)."""
    found = [m for m in DRAFT_MARKERS if m in text]
    if found:
        raise PreregError(f"the preregistration is not final: it still contains {found}")


def artifact_hashes(root: Path, art_dir: Path) -> dict[str, str]:
    """SHA-256 of every file under ``art_dir`` (paths relative to ``root``, POSIX, sorted)."""
    base = root / art_dir
    if not base.is_dir():
        return {}
    files = sorted(p for p in base.rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


@dataclass
class Lock:
    """Content of ``docs/preregistration.lock.json``."""

    preregistration: dict[str, Any]  # path, sha256, version
    artifacts_dir: str
    artifacts: dict[str, str]  # relative path -> sha256
    code_commit: str
    frozen_at_utc: str
    config_digests: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)
    test_split: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    protected_paths: list[str] = field(default_factory=lambda: list(PROTECTED))
    format: str = LOCK_FORMAT

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Lock:
        try:
            d = json.loads(text)
            if d.get("format") != LOCK_FORMAT:
                raise PreregError(f"lock format {d.get('format')!r}, expected {LOCK_FORMAT!r}")
            return cls(**d)
        except (json.JSONDecodeError, TypeError) as err:
            raise PreregError(f"unreadable lock: {err}") from err

    def readme_line(self) -> str:
        sha = self.preregistration["sha256"]
        return (
            f"Preregistration v{self.preregistration['version']} frozen on {self.frozen_at_utc[:10]} at"
            " commit"
            f" `{self.code_commit[:12]}`: `docs/preregistration.md` SHA-256 `{sha[:16]}…`, artifacts and"
            " hashes in `docs/preregistration.lock.json`."
        )


def make_lock(
    root: Path,
    art_dir: Path,
    *,
    code_commit: str,
    frozen_at_utc: str,
    config_digests: Mapping[str, Any],
    inputs: Mapping[str, Any] | None = None,
    test_split: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    prereg_path: Path = PREREG_PATH,
) -> Lock:
    """The lock of the current preregistration and of every file under ``art_dir``."""
    text = (root / prereg_path).read_text(encoding="utf-8")
    return Lock(
        preregistration={
            "path": prereg_path.as_posix(),
            "sha256": sha256_file(root / prereg_path),
            "version": prereg_version(text),
        },
        artifacts_dir=art_dir.as_posix(),
        artifacts=artifact_hashes(root, art_dir),
        code_commit=code_commit,
        frozen_at_utc=frozen_at_utc,
        config_digests=dict(config_digests),
        inputs=dict(inputs or {}),
        test_split=dict(test_split or {}),
        environment=dict(environment or {}),
    )


def verify_hashes(root: Path, lock: Lock) -> None:
    """The preregistration and the artifacts are exactly the locked files."""
    prereg = root / lock.preregistration["path"]
    if not prereg.is_file():
        raise PreregError(f"{lock.preregistration['path']} is missing")
    if sha256_file(prereg) != lock.preregistration["sha256"]:
        raise PreregError("the preregistration changed since the freeze (SHA-256 mismatch)")
    found = artifact_hashes(root, Path(lock.artifacts_dir))
    missing = sorted(set(lock.artifacts) - set(found))
    extra = sorted(set(found) - set(lock.artifacts))
    if missing or extra:
        raise PreregError(f"the frozen artifacts differ from the lock: missing {missing}, unexpected {extra}")
    changed = sorted(p for p, sha in lock.artifacts.items() if found[p] != sha)
    if changed:
        raise PreregError(f"frozen artifacts modified since the freeze: {changed}")


def verify_code(root: Path, lock: Lock) -> None:
    """The lock commit is an ancestor of HEAD and nothing protected changed since."""
    commit = lock.code_commit
    try:
        git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    except PreregError as err:
        raise PreregError(f"the lock commit {commit[:12]} is not in this repository") from err
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "HEAD"], capture_output=True
    )
    if ancestor.returncode != 0:
        raise PreregError(f"the lock commit {commit[:12]} is not an ancestor of HEAD")
    changed = git(root, "diff", "--name-only", commit, "HEAD", "--", *lock.protected_paths).split()
    if changed:
        shown = ", ".join(changed[:5]) + (" …" if len(changed) > 5 else "")
        raise PreregError(f"code changed since the lock commit {commit[:12]}: {shown}")


def read_lock(root: Path, lock_path: Path = LOCK_PATH) -> Lock:
    path = root / lock_path
    if not path.is_file():
        raise PreregError(
            f"no lock at {lock_path}: freeze the preregistration first (scripts/freeze_prereg.py)"
        )
    return Lock.from_json(path.read_text(encoding="utf-8"))


def verify_lock(root: Path, lock_path: Path = LOCK_PATH, *, require_clean: bool = True) -> Lock:
    """Every check of the module docstring but the marker; returns the lock. Raises :class:`PreregError`."""
    lock = read_lock(root, lock_path)
    verify_hashes(root, lock)
    verify_code(root, lock)
    if require_clean:
        require_clean_tree(root)
    return lock


def claim_once(markers: Sequence[Path], payload: Mapping[str, Any]) -> None:
    """Write every evaluation marker, refusing when any already exists (a second evaluation).

    Markers are created exclusively (O_EXCL), so two concurrent invocations cannot both pass.
    """
    existing = [m for m in markers if m.exists()]
    if existing:
        raise PreregError(
            f"the test evaluation already started ({existing[0]}): a second evaluation is refused"
        )
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    for marker in markers:
        marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        except FileExistsError as err:
            raise PreregError(f"the test evaluation already started ({marker})") from err
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
