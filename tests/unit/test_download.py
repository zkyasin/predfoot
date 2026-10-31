import importlib.util
import io
import zipfile

import pytest

from predfoot.io.download import sync_file, sync_zip_members
from predfoot.io.manifest import Manifest, sha256_file
from predfoot.io.remote_zip import BytesFetcher, RemoteError, RemoteZip
from predfoot.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location("download_data", REPO_ROOT / "scripts" / "download_data.py")
assert _spec is not None
assert _spec.loader is not None
download_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_data)


@pytest.fixture
def archive(rng):
    members = {f"SNGS-001/img1/{f:06d}.jpg": rng.bytes(2000 + f) for f in range(1, 11)}
    members["SNGS-001/Labels-GameState.json"] = b'{"info": {}}' * 100
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return members, buf.getvalue()


def test_rerunning_downloads_nothing(tmp_path, archive):
    members, data = archive
    fetcher = BytesFetcher(data)
    remote = RemoteZip(fetcher)
    targets = {name: tmp_path / name for name in members}
    with Manifest(tmp_path) as manifest:
        first = sync_zip_members(remote, targets, manifest)
    assert first.downloaded == len(members)
    fetcher.calls.clear()
    with Manifest(tmp_path) as manifest:
        second = sync_zip_members(remote, targets, manifest)
    assert (second.downloaded, second.skipped) == (0, len(members))
    assert fetcher.calls == []
    assert set(Manifest(tmp_path).entries()) == set(members)


def test_only_missing_or_damaged_files_are_fetched_again(tmp_path, archive):
    members, data = archive
    fetcher = BytesFetcher(data)
    remote = RemoteZip(fetcher)
    targets = {name: tmp_path / name for name in members}
    with Manifest(tmp_path) as manifest:
        sync_zip_members(remote, targets, manifest)
    gone, shrunk = "SNGS-001/img1/000003.jpg", "SNGS-001/img1/000007.jpg"
    (tmp_path / gone).unlink()
    (tmp_path / shrunk).write_bytes(b"truncated")
    fetcher.calls.clear()
    with Manifest(tmp_path) as manifest:
        stats = sync_zip_members(remote, targets, manifest, max_gap=0)
    assert (stats.downloaded, stats.skipped) == (2, len(members) - 2)
    assert len(fetcher.calls) == 2
    assert (tmp_path / shrunk).read_bytes() == members[shrunk]


def test_complete_files_missing_from_the_manifest_are_recovered_by_crc(tmp_path, archive):
    members, data = archive
    fetcher = BytesFetcher(data)
    remote = RemoteZip(fetcher)
    fetcher.calls.clear()
    good, forged = "SNGS-001/img1/000001.jpg", "SNGS-001/img1/000002.jpg"
    for name in (good, forged):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(members[name])
    (tmp_path / forged).write_bytes(bytes(len(members[forged])))  # right size, wrong content
    with Manifest(tmp_path) as manifest:
        stats = sync_zip_members(remote, {n: tmp_path / n for n in (good, forged)}, manifest)
    assert (stats.recovered, stats.downloaded) == (1, 1)
    assert (tmp_path / forged).read_bytes() == members[forged]


def test_sync_file_skips_recovers_and_verifies(tmp_path):
    payload = bytes(range(256)) * 10
    dest = tmp_path / "weights" / "SV_kp"
    with Manifest(tmp_path) as manifest:
        assert sync_file(BytesFetcher(payload), dest, manifest, chunk_size=1000).downloaded == 1
    fetcher = BytesFetcher(payload)
    with Manifest(tmp_path) as manifest:
        assert sync_file(fetcher, dest, manifest).skipped == 1
    assert fetcher.calls == []
    (tmp_path / "MANIFEST.sha256").unlink()
    with Manifest(tmp_path) as manifest:
        assert sync_file(BytesFetcher(payload), dest, manifest).recovered == 1
        assert manifest.get(dest).sha256 == sha256_file(dest)


def test_sync_file_deletes_a_download_that_fails_its_check(tmp_path):
    dest = tmp_path / "f.csv"
    with Manifest(tmp_path) as manifest, pytest.raises(RemoteError, match="integrity"):
        sync_file(BytesFetcher(b"a,b\n1,2\n"), dest, manifest, verify=lambda p: False)
    assert not dest.exists()
    assert manifest.get(dest) is None


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["sn-gsr-labels", "--splits", "train", "valid"],
            ["sn-gsr-labels", "--splits", "train", "--splits", "valid"],
        ),
        (
            ["sn-gsr-frames", "--clips", "SNGS-056", "SNGS-060", "--every", "5"],
            ["sn-gsr-frames", "--clips", "SNGS-056", "--clips", "SNGS-060", "--every", "5"],
        ),
        (
            ["sn-gsr-labels", "--splits", "test", "--splits", "valid"],
            ["sn-gsr-labels", "--splits", "test", "--splits", "valid"],
        ),
        (["pnlcalib-weights"], ["pnlcalib-weights"]),
    ],
)
def test_multi_value_options_accept_space_separated_values(argv, expected):
    assert download_data.expand_multi_value_options(argv) == expected
