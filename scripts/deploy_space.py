"""Upload the built showcase site (site/dist/) to its static Hugging Face Space.

Creates the Space when it does not exist yet (sdk static), then uploads site/dist/ in one commit, which also
deletes the remote files the build no longer produces (huggingface_hub always keeps .gitattributes). The
visibility of an existing Space is left as it is. Needs a Hugging Face token with write access to the
namespace (``hf auth login``, or ``HF_TOKEN`` in the environment).

Usage:
    uv run python scripts/build_site.py
    uv run python scripts/deploy_space.py [--repo-id Ya-SinZ/predfoot] [--dist site/dist] [--private]
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import typer
from huggingface_hub import HfApi

from predfoot.paths import REPO_ROOT

REPO_ID = "Ya-SinZ/predfoot"
DIST = REPO_ROOT / "site" / "dist"
#: What scripts/build_site.py writes and the Space cannot run without.
REQUIRED = ("index.html", "README.md")


def git_sha() -> str:
    """Short commit of the working tree, or "unknown" outside git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def deploy(dist: Path, repo_id: str = REPO_ID, *, private: bool = False, api: Any = None) -> str:
    """Create the Space if needed and replace its files with ``dist``; returns the commit URL."""
    missing = [name for name in REQUIRED if not (dist / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{dist} has no {', '.join(missing)}: run scripts/build_site.py first")
    hub = api if api is not None else HfApi()
    hub.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", private=private, exist_ok=True)
    info = hub.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=dist,
        commit_message=f"Showcase page built from predfoot {git_sha()}",
        delete_patterns="*",
    )
    return str(getattr(info, "commit_url", info))


app = typer.Typer(add_completion=False)


@app.command()
def main(
    repo_id: str = typer.Option(REPO_ID, help="Space id, owner/name."),
    dist: Path = typer.Option(DIST, help="Folder written by scripts/build_site.py."),  # noqa: B008
    private: bool = typer.Option(False, help="Create the Space private (an existing one keeps its setting)."),
) -> None:
    try:
        url = deploy(dist, repo_id, private=private)
    except FileNotFoundError as exc:
        typer.echo(f"deploy_space: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Uploaded {dist} → https://huggingface.co/spaces/{repo_id} ({url})")


if __name__ == "__main__":
    app()
