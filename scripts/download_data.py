"""Download the datasets and weights predfoot needs into PREDFOOT_DATA_DIR.

Every command is idempotent (complete files are skipped) and resumable (interrupted files restart from their
``.part``), and records each file's size and SHA-256 in ``PREDFOOT_DATA_DIR/MANIFEST.sha256``; check them with
``scripts/verify_data.py``. Options that take several values accept them space-separated
(``--splits train valid test``) or repeated (``--splits train --splits valid``).

    uv run python scripts/download_data.py sn-gsr-labels --splits train valid test
    uv run python scripts/download_data.py sn-gsr-frames --clips SNGS-056 SNGS-060 --every 5
    uv run python scripts/download_data.py pnlcalib-weights
    uv run python scripts/download_data.py soccertrack-raw --match 117092
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Annotated
from urllib.parse import quote

import typer
from huggingface_hub import HfApi, get_token
from huggingface_hub.hf_api import RepoFile

from predfoot.io import sngsr
from predfoot.io.download import SyncStats, sync_file, sync_zip_members
from predfoot.io.manifest import Manifest, check_relpath, git_blob_sha1
from predfoot.io.remote_zip import HttpRangeFetcher, RemoteZip
from predfoot.paths import data_dir

PNLCALIB_RELEASE = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"
PNLCALIB_FILES = ("SV_kp", "SV_lines")
SOCCERTRACK_REPO = "atomscott/soccertrack-v2"
SOCCERTRACK_EXCLUDE_SUFFIXES = ("_tracker_box_data.xml",)  # 318 MB of tracker boxes, not needed
MULTI_VALUE_OPTIONS = ("--splits", "--clips")

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__.split("\n\n")[0])


def expand_multi_value_options(argv: list[str]) -> list[str]:
    """Rewrite ``--splits a b`` as ``--splits a --splits b`` (click options take one value each)."""
    out: list[str] = []
    current: str | None = None
    for token in argv:
        if token in MULTI_VALUE_OPTIONS:
            current = token
            continue
        if current is not None and not token.startswith("-"):
            out += [current, token]
            continue
        current = None
        out.append(token)
    return out


def _report(label: str, stats: SyncStats) -> None:
    typer.echo(f"{label}: {stats.summary()}")
    for name, error in sorted(stats.failed.items())[:20]:
        typer.echo(f"  FAILED {name}: {error}", err=True)


@app.command("sn-gsr-labels")
def sn_gsr_labels(
    splits: Annotated[list[str] | None, typer.Option(help="Splits to fetch (train, valid, test).")] = None,
    workers: Annotated[int, typer.Option(help="Parallel range requests.")] = 8,
) -> None:
    """Every Labels-GameState.json and sequences_info.json -> sn-gsr/<split>/<clip>/."""
    root = data_dir()
    failed = 0
    with Manifest(root) as manifest:
        for split in splits or list(sngsr.LABELED_SPLITS):
            if split not in sngsr.LABELED_SPLITS:
                raise typer.BadParameter(f"{split!r} is not one of {sngsr.LABELED_SPLITS}")
            remote = RemoteZip(HttpRangeFetcher(sngsr.split_url(split), token=get_token()))
            names = remote.namelist()
            members = sngsr.labels_members(names)
            if not members:
                raise RuntimeError(f"no labels found in {split}.zip: has the layout changed?")
            targets = {member: sngsr.labels_path(split, clip, root) for clip, member in members.items()}
            if sngsr.SEQUENCES_INFO_NAME in names:
                targets[sngsr.SEQUENCES_INFO_NAME] = sngsr.sequences_info_path(split, root)
            stats = sync_zip_members(remote, targets, manifest, workers=workers, label=f"{split} labels")
            _report(f"{split}: {len(members)} clips", stats)
            failed += len(stats.failed)
    raise typer.Exit(1 if failed else 0)


@app.command("sn-gsr-frames")
def sn_gsr_frames(
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids, e.g. SNGS-056.")] = None,
    every: Annotated[int, typer.Option(help="Keep frames 1, 1 + every, 1 + 2 every...")] = 5,
    all_dev: Annotated[bool, typer.Option(help="Every train and valid clip with labels on disk.")] = False,
    allow_locked: Annotated[bool, typer.Option(help="Allow clips of the locked test split.")] = False,
    workers: Annotated[int, typer.Option(help="Parallel range requests.")] = 16,
) -> None:
    """Frames of the chosen clips -> sn-gsr/<split>/<clip>/img1/ (the labels tell each clip's split)."""
    root = data_dir()
    known = sngsr.local_clips(root)
    wanted = list(clips or [])
    if all_dev:
        wanted += [clip for clip, split in known.items() if split in sngsr.DEV_SPLITS]
    if not wanted:
        raise typer.BadParameter("give --clips or --all-dev")
    by_split: dict[str, list[str]] = defaultdict(list)
    for clip in dict.fromkeys(wanted):
        split = known.get(clip)
        if split is None:
            raise typer.BadParameter(f"{clip}: labels not on disk, run sn-gsr-labels first")
        if split in sngsr.LOCKED_SPLITS and not allow_locked:
            raise typer.BadParameter(f"{clip} belongs to the locked {split} split")
        by_split[split].append(clip)
    failed = 0
    with Manifest(root) as manifest:
        for split, split_clips in sorted(by_split.items()):
            remote = RemoteZip(HttpRangeFetcher(sngsr.split_url(split), token=get_token()))
            members = sngsr.frame_members(remote.namelist(), split_clips, every)
            missing = set(split_clips) - {clip for clip, _ in members}
            if missing:
                raise RuntimeError(f"no frames found for {sorted(missing)} in {split}.zip")
            targets = {
                member: sngsr.frame_path(split, clip, frame, root)
                for (clip, frame), member in members.items()
            }
            stats = sync_zip_members(remote, targets, manifest, workers=workers, label=f"{split} frames")
            _report(f"{split}: {len(split_clips)} clips, {len(targets)} frames", stats)
            failed += len(stats.failed)
    raise typer.Exit(1 if failed else 0)


@app.command("pnlcalib-weights")
def pnlcalib_weights() -> None:
    """PnLCalib keypoint and line weights (CC BY 4.0) -> weights/pnlcalib/."""
    root = data_dir()
    with Manifest(root) as manifest:
        for name in PNLCALIB_FILES:
            stats = sync_file(
                HttpRangeFetcher(f"{PNLCALIB_RELEASE}/{name}"), root / "weights" / "pnlcalib" / name, manifest
            )
            _report(name, stats)


@app.command("soccertrack-raw")
def soccertrack_raw(
    match: Annotated[str, typer.Option(help="Match id under raw/ in atomscott/soccertrack-v2.")] = "117092",
) -> None:
    """SoccerTrack v2 raw/<match>/ (CC BY 4.0, gated: accept the terms first) -> soccertrack-v2/raw/<match>/.

    Every file except the tracker boxes XML; LFS files are checked against their SHA-256, the others against
    their git blob id.
    """
    if not match.isdigit():
        raise typer.BadParameter(f"not a match id: {match!r}")
    root = data_dir()
    token = get_token()
    prefix = f"raw/{match}"
    items = [
        item
        for item in HfApi(token=token).list_repo_tree(
            SOCCERTRACK_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
        )
        if isinstance(item, RepoFile)
    ]
    if not items:
        raise RuntimeError(f"nothing under {prefix} in {SOCCERTRACK_REPO}")
    with Manifest(root) as manifest:
        for item in sorted(items, key=lambda i: i.path):
            path = PurePosixPath(check_relpath(item.path))
            if path.parts[:2] != ("raw", match) or path.name.endswith(SOCCERTRACK_EXCLUDE_SUFFIXES):
                continue
            url = f"https://huggingface.co/datasets/{SOCCERTRACK_REPO}/resolve/main/{quote(item.path)}"
            expected = item.lfs.sha256 if item.lfs is not None else None
            blob_id = item.blob_id

            def same_blob(p: Path, blob_id: str = blob_id) -> bool:
                return git_blob_sha1(p) == blob_id

            stats = sync_file(
                HttpRangeFetcher(url, token=token),
                root / "soccertrack-v2" / path,
                manifest,
                expected_sha256=expected,
                verify=None if expected else same_blob,
            )
            _report(item.path, stats)


def main() -> None:
    app(args=expand_multi_value_options(sys.argv[1:]), prog_name="download_data.py")


if __name__ == "__main__":
    main()
