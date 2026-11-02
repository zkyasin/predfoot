"""Upload the built showcase site (site/dist/) to its static Hugging Face Space.

Before uploading, the script checks what it is about to publish and refuses, listing every problem:
- a folder that scripts/build_site.py did not write (its index.html carries the build's generator mark);
- for the default site/dist, a build older than site/page.html or its media: every file must match a fresh
  build, so a stale page cannot replace the public one;
- a link of the page to another site that does not answer 200 to an anonymous request. GitHub answers 404
  for a private repository and for a file not yet on main, so this keeps the Space from going public before
  the repository and its evidence do. ``--skip-link-check`` lifts this check, for a private Space or a
  preview.

It then creates the Space when it does not exist yet (sdk static) and uploads the folder in one commit, which
also deletes the remote files the build no longer produces (huggingface_hub always keeps .gitattributes);
hidden files such as .DS_Store are left out. The visibility of an existing Space is left as it is. Needs a
Hugging Face token with write access to the namespace (``hf auth login``, or ``HF_TOKEN`` in the environment).

Usage:
    uv run python scripts/build_site.py
    uv run python scripts/deploy_space.py [--repo-id Ya-SinZ/predfoot] [--dist site/dist] [--private]
                                          [--skip-link-check]
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import typer
from huggingface_hub import HfApi

from predfoot.paths import REPO_ROOT

REPO_ID = "Ya-SinZ/predfoot"
DIST = REPO_ROOT / "site" / "dist"
#: What scripts/build_site.py writes and the Space cannot run without.
REQUIRED = ("index.html", "README.md")
#: The mark scripts/build_site.py writes into index.html (its GENERATOR).
GENERATOR_META = '<meta name="generator" content="predfoot scripts/build_site.py">'
#: Never uploaded: hidden files at any depth.
IGNORE = [".*", "*/.*"]
USER_AGENT = "predfoot-deploy-link-check"


class DeployError(Exception):
    """The folder cannot be published as it is; the message lists every problem."""


def load_build_site() -> ModuleType:
    """scripts/build_site.py as a module (scripts/ is not a package), shared with any earlier import of it."""
    key = "scripts_build_site"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name("build_site.py"))
    if spec is None or spec.loader is None:
        raise ImportError("build_site")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def git_sha() -> str:
    """Short commit of the working tree, "-dirty" with uncommitted changes, or "unknown" outside git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    if not sha:
        return "unknown"
    return f"{sha}-dirty" if status.strip() else sha


def check_build(dist: Path) -> None:
    """Refuse a folder without the files the Space needs or without the build's generator mark."""
    missing = [name for name in REQUIRED if not (dist / name).is_file()]
    if missing:
        raise DeployError(f"{dist} has no {', '.join(missing)}: run scripts/build_site.py first")
    if GENERATOR_META not in (dist / "index.html").read_text(encoding="utf-8", errors="replace"):
        raise DeployError(
            f"{dist}/index.html was not written by scripts/build_site.py: refusing to upload it"
        )


def stale_files(dist: Path, expected: dict[str, bytes]) -> list[str]:
    """How ``dist`` differs from a fresh build given as {relative path: bytes}; hidden files are ignored."""
    present = {
        p.relative_to(dist).as_posix()
        for p in dist.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(dist).parts)
    }
    problems = [f"{name}: missing" for name in sorted(expected.keys() - present)]
    problems += [f"{name}: not in a fresh build" for name in sorted(present - expected.keys())]
    problems += [
        f"{name}: differs from a fresh build"
        for name in sorted(expected.keys() & present)
        if (dist / name).read_bytes() != expected[name]
    ]
    return problems


def fresh_build() -> dict[str, bytes]:
    """Every file scripts/build_site.py would write now from site/page.html and assets/showcase/."""
    build_site = load_build_site()
    try:
        site = build_site.plan(build_site.PAGE.read_text(encoding="utf-8"), build_site.MEDIA)
    except build_site.SiteError as exc:
        raise DeployError(f"site/page.html does not build: {exc}") from exc
    files = {
        "index.html": site.index_html.encode("utf-8"),
        "README.md": build_site.space_readme().encode("utf-8"),
    }
    return files | {name: src.read_bytes() for name, src in site.media.items()}


def external_links(index_html: str) -> list[str]:
    """The distinct links of the page to other sites (http, https, protocol-relative), in page order."""
    references, _ = load_build_site().scan(index_html)
    links = []
    for ref in references:
        url = ref.url
        if url.startswith("//"):
            url = "https:" + url
        if not ref.load and url.startswith(("http://", "https://")) and url not in links:
            links.append(url)
    return links


def link_status(url: str, timeout: float = 20.0) -> int:
    """HTTP status of an anonymous request (HEAD, or GET when HEAD is refused), redirects followed; 0 when the
    request fails before any answer."""
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(
                request, timeout=timeout
            ) as response:  # only http(s) links reach here
                return int(response.status)
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in (403, 405, 501):
                continue
            return int(exc.code)
        except (urllib.error.URLError, OSError):
            return 0
    return 0


def broken_links(index_html: str, status: Callable[[str], int] | None = None) -> list[str]:
    """Every link of the page to another site that does not answer 200 (to ``status``, by default
    :func:`link_status`), with what it answered."""
    ask = status or link_status
    problems = []
    for url in external_links(index_html):
        code = ask(url)
        if code != 200:
            problems.append(f"{url}: {'no answer' if code == 0 else f'HTTP {code}'}")
    return problems


def preflight(
    dist: Path,
    *,
    expected: dict[str, bytes] | None = None,
    check_links: bool = True,
    status: Callable[[str], int] | None = None,
) -> list[str]:
    """Every reason not to publish ``dist``: not a build, stale against ``expected``, broken outside links."""
    check_build(dist)
    problems = []
    if expected is not None:
        problems += [f"stale build, run scripts/build_site.py: {p}" for p in stale_files(dist, expected)]
    if check_links:
        index_html = (dist / "index.html").read_text(encoding="utf-8")
        problems += [f"link not public yet: {p}" for p in broken_links(index_html, status)]
    return problems


def deploy(dist: Path, repo_id: str = REPO_ID, *, private: bool = False, api: Any = None) -> str:
    """Create the Space if needed and replace its files with ``dist``; returns the commit URL."""
    check_build(dist)
    hub = api if api is not None else HfApi()
    hub.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", private=private, exist_ok=True)
    info = hub.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=dist,
        commit_message=f"Showcase page built from predfoot {git_sha()}",
        delete_patterns="*",
        ignore_patterns=IGNORE,
    )
    return str(getattr(info, "commit_url", info))


app = typer.Typer(add_completion=False)


@app.command()
def main(
    repo_id: str = typer.Option(REPO_ID, help="Space id, owner/name."),
    dist: Path = typer.Option(DIST, help="Folder written by scripts/build_site.py."),  # noqa: B008
    private: bool = typer.Option(False, help="Create the Space private (an existing one keeps its setting)."),
    skip_link_check: bool = typer.Option(
        False, help="Upload even if a link to another site does not answer 200 (private Space, preview)."
    ),
) -> None:
    try:
        expected = fresh_build() if dist.resolve() == DIST.resolve() else None
        problems = preflight(dist, expected=expected, check_links=not skip_link_check)
        if problems:
            raise DeployError("\n".join(problems))
        url = deploy(dist, repo_id, private=private)
    except DeployError as exc:
        typer.echo(f"deploy_space: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Uploaded {dist} → https://huggingface.co/spaces/{repo_id} ({url})")


if __name__ == "__main__":
    app()
