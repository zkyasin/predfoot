"""Guards of the preregistered test run (tickets T3.4 and T6.1): the freeze lock and the single evaluation.

``scripts/freeze_prereg.py`` fits the frozen artifacts on dev only, from a clean tree whose preregistration is
final (no draft marker), and writes ``docs/preregistration.lock.json``: SHA-256 of the preregistration and of
every artifact, the code commit, the date, the configuration digests, the pinned test clip list and the
environment. The lock and the artifacts are then committed together (the *freeze commit*).
``scripts/run_test_once.py`` calls :func:`verify_lock` and :func:`verify_environment` before every stage and
:func:`claim_once` before the only stage that reads test labels:

- the lock exists and has the expected format;
- the preregistration and every artifact have the hashes of the lock, and the artifact folder holds exactly
  the locked files;
- the code commit of the lock is an ancestor of HEAD and nothing under the protected paths (``src/``,
  ``scripts/``, ``configs/``, ``pyproject.toml``, ``uv.lock``, ``.python-version``) changed since;
- the lock is anchored in git history: the lock, the preregistration and every artifact are tracked; the
  freeze commit (the latest commit that added the lock) adds only the lock and the artifacts on top of the
  lock's code commit; nothing among them changed after it. Editing the preregistration and the lock together,
  or pointing the lock at a newer code commit, is refused; a re-freeze (lock and artifacts deleted, then
  frozen and committed again) stays possible until stage C has claimed the evaluation;
- the working tree is clean: git status shows nothing, no tracked file hides its changes (assume-unchanged or
  skip-worktree) and no ignored file sits under the protected paths (bytecode caches and Finder's
  ``.DS_Store`` aside);
- the interpreter and the package versions are those recorded at the freeze;
- the evaluation marker does not exist yet; it is written first, before any label is read, so a crash or a
  second invocation can never produce a second evaluation.

All functions take the repository root, so tests run them on throw-away repositories.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

from predfoot.io.manifest import sha256_file

PREREG_PATH = Path("docs/preregistration.md")
LOCK_PATH = Path("docs/preregistration.lock.json")
ARTIFACTS_ROOT = Path("artifacts/prereg")
#: Paths whose content decides what the test run computes: code, configuration and the software stack.
PROTECTED = ("src", "scripts", "configs", "pyproject.toml", "uv.lock", ".python-version")
#: Packages whose versions the lock records and every stage compares (torch runs the networks of stage A).
ENV_PACKAGES = ("numpy", "scipy", "scikit-learn", "polars", "opencv-python-headless", "torch", "torchvision")
LOCK_FORMAT = "predfoot-prereg-lock/1"
#: Text that marks a preregistration as not approved yet; the freeze refuses it.
DRAFT_MARKERS = ("DRAFT", "pending owner approval")
_VERSION = re.compile(r"^- Version:\s*(\d+)\s*$", re.MULTILINE)
_BYTECODE = (".pyc", ".pyo", ".pyd")
#: Folder metadata that macOS Finder writes (git-ignored): never importable, so it cannot shadow a module.
_OS_METADATA = (".DS_Store",)


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


def hidden_paths(root: Path) -> list[str]:
    """What git status cannot show: tracked files marked assume-unchanged or skip-worktree (their edits are
    invisible), and ignored files under the protected paths (an ignored module can shadow a tracked one);
    bytecode caches and Finder's ``.DS_Store`` are not counted."""
    flagged = [
        line[2:] for line in git(root, "ls-files", "-v").splitlines() if line[:1].islower() or line[:1] == "S"
    ]
    out = git(root, "status", "--porcelain", "--ignored=matching", "--untracked-files=all", "--", *PROTECTED)
    ignored = []
    for line in out.splitlines():
        if not line.startswith("!! "):
            continue
        path = line[3:].strip('"')
        posix = PurePosixPath(path)
        if "__pycache__" in posix.parts or path.endswith(_BYTECODE) or posix.name in _OS_METADATA:
            continue
        ignored.append(path)
    return [*(f"{p} (changes hidden from git status)" for p in flagged), *(f"{p} (ignored)" for p in ignored)]


def require_clean_tree(root: Path) -> None:
    dirty = dirty_paths(root) + hidden_paths(root)
    if dirty:
        shown = ", ".join(dirty[:5]) + (" …" if len(dirty) > 5 else "")
        raise PreregError(
            f"the working tree is not clean ({len(dirty)} paths: {shown}); commit or stash the changes and"
            " delete untracked or ignored files first"
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
    """SHA-256 of every file under ``art_dir`` (paths relative to ``root``, POSIX, sorted); hidden files such
    as the ``.DS_Store`` that Finder writes are skipped (no artifact is a dotfile)."""
    base = root / art_dir
    if not base.is_dir():
        return {}
    files = sorted(
        p
        for p in base.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(base).parts)
    )
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def artifacts_digest(artifacts: Mapping[str, str]) -> str:
    """One SHA-256 over the locked artifact paths and hashes."""
    text = "".join(f"{path} {sha}\n" for path, sha in sorted(artifacts.items()))
    return hashlib.sha256(text.encode()).hexdigest()


def environment() -> dict[str, str]:
    """Python and the versions of :data:`ENV_PACKAGES` ("missing" when a package is not installed)."""
    env = {"python": platform.python_version()}
    for pkg in ENV_PACKAGES:
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = "missing"
    return env


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

    def readme_line(self, lock_sha256: str | None = None) -> str:
        """The commitment quoted outside the repository: preregistration, artifacts and (once written) the
        lock file itself, so that a re-freeze cannot produce the same line."""
        sha = self.preregistration["sha256"]
        line = (
            f"Preregistration v{self.preregistration['version']} frozen on {self.frozen_at_utc[:10]} at"
            f" commit `{self.code_commit[:12]}`: `docs/preregistration.md` SHA-256 `{sha[:16]}…`, artifacts"
            f" digest `{artifacts_digest(self.artifacts)[:16]}…`"
        )
        if lock_sha256:
            line += f", `docs/preregistration.lock.json` SHA-256 `{lock_sha256[:16]}…`"
        return line + "; hashes in `docs/preregistration.lock.json`."


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


def lock_sha256(root: Path, lock_path: Path = LOCK_PATH) -> str:
    """SHA-256 of the lock file: the identity of a freeze, recorded in every output of the test run."""
    return sha256_file(root / lock_path)


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


def protected_changes(root: Path, commit: str) -> list[str]:
    """Files under :data:`PROTECTED` that differ between ``commit`` and HEAD."""
    return git(root, "diff", "--no-renames", "--name-only", commit, "HEAD", "--", *PROTECTED).split()


def verify_code(root: Path, lock: Lock) -> None:
    """The lock commit is an ancestor of HEAD and nothing protected changed since (the protected set is this
    module's, not the lock's own list)."""
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
    changed = protected_changes(root, commit)
    if changed:
        shown = ", ".join(changed[:5]) + (" …" if len(changed) > 5 else "")
        raise PreregError(f"code changed since the lock commit {commit[:12]}: {shown}")


def freeze_commit(root: Path, lock_path: Path = LOCK_PATH) -> str:
    """The latest commit reachable from HEAD that added the lock file."""
    commit = git(root, "log", "--diff-filter=A", "--format=%H", "-1", "--", lock_path.as_posix()).strip()
    if not commit:
        raise PreregError(f"{lock_path} is not committed: commit the lock and the artifacts together first")
    return commit


def verify_freeze_commit(root: Path, lock: Lock, lock_path: Path = LOCK_PATH) -> str:
    """The lock is anchored in git history (see the module docstring); returns the freeze commit."""
    frozen = [lock_path.as_posix(), lock.preregistration["path"], *lock.artifacts]
    tracked = set(git(root, "ls-files", "--", *frozen).splitlines())
    untracked = [p for p in frozen if p not in tracked]
    if untracked:
        raise PreregError(f"not tracked by git: {untracked}; commit the lock and the artifacts together")
    commit = freeze_commit(root, lock_path)
    after = git(
        root,
        "diff",
        "--no-renames",
        "--name-only",
        commit,
        "HEAD",
        "--",
        lock_path.as_posix(),
        lock.preregistration["path"],
        lock.artifacts_dir,
    ).split()
    if after:
        raise PreregError(f"changed since the freeze commit {commit[:12]}: {after}")
    allowed = {lock_path.as_posix(), *lock.artifacts}
    between = git(root, "diff", "--no-renames", "--name-only", lock.code_commit, commit).split()
    extra = [p for p in between if p not in allowed]
    if extra:
        raise PreregError(
            f"the freeze commit {commit[:12]} adds more than the lock and the artifacts to the lock's code"
            f" commit {lock.code_commit[:12]}: {extra}"
        )
    return commit


def verify_environment(lock: Lock, current: Mapping[str, str] | None = None) -> None:
    """The running interpreter and packages are those recorded at the freeze."""
    if not lock.environment:
        raise PreregError("the lock records no environment")
    now = dict(environment() if current is None else current)
    diff = {
        k: (str(v), now.get(k, "missing"))
        for k, v in sorted(lock.environment.items())
        if now.get(k, "missing") != v
    }
    if diff:
        shown = ", ".join(f"{k} {a} at the freeze, {b} now" for k, (a, b) in diff.items())
        raise PreregError(f"the environment differs from the freeze: {shown}")


def read_lock(root: Path, lock_path: Path = LOCK_PATH) -> Lock:
    path = root / lock_path
    if not path.is_file():
        raise PreregError(
            f"no lock at {lock_path}: freeze the preregistration first (scripts/freeze_prereg.py)"
        )
    return Lock.from_json(path.read_text(encoding="utf-8"))


def verify_lock(root: Path, lock_path: Path = LOCK_PATH, *, require_clean: bool = True) -> Lock:
    """Every git and hash check of the module docstring (not the environment, not the marker); returns the
    lock. Raises :class:`PreregError`."""
    lock = read_lock(root, lock_path)
    verify_hashes(root, lock)
    verify_code(root, lock)
    verify_freeze_commit(root, lock, lock_path)
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
