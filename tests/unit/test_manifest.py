import hashlib

import pytest

from predfoot.io.manifest import (
    MANIFEST_NAME,
    Manifest,
    ManifestEntry,
    format_manifest,
    git_blob_sha1,
    parse_manifest,
    sha256_file,
    verify_manifest,
)


def write(root, rel, data: bytes):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def record(manifest, path):
    return manifest.add(path, path.stat().st_size, sha256_file(path))


def test_format_and_parse_round_trip():
    entries = [ManifestEntry("b/x y.json", 12, "a" * 64), ManifestEntry("a/1.jpg", 3, "0" * 64)]
    text = format_manifest(entries)
    assert text.splitlines()[0].endswith("a/1.jpg")  # sorted by path
    assert parse_manifest(text) == {e.path: e for e in entries}


@pytest.mark.parametrize(
    "line",
    [
        f"{'a' * 64}  12  /etc/passwd",
        f"{'a' * 64}  12  ../outside",
        f"{'a' * 64}  12  sn-gsr/../../outside",
        f"{'a' * 64}  12  a\\b",
        f"{'a' * 63}  12  ok.json",
        f"{'a' * 64}  -1  ok.json",
        f"{'a' * 64}  ok.json",
    ],
)
def test_malformed_or_escaping_lines_are_rejected(line):
    with pytest.raises(ValueError, match=r"malformed|inside the data directory"):
        parse_manifest(line + "\n")


def test_entries_are_buffered_then_flushed(tmp_path):
    path = write(tmp_path, "sn-gsr/valid/SNGS-056/Labels-GameState.json", b"{}")
    with Manifest(tmp_path) as manifest:
        entry = record(manifest, path)
        assert entry.path == "sn-gsr/valid/SNGS-056/Labels-GameState.json"
        assert manifest.get(path) == entry
        assert not (tmp_path / MANIFEST_NAME).exists()
    assert Manifest(tmp_path).get(path) == entry


def test_concurrent_writers_merge_instead_of_overwriting(tmp_path):
    a, b = Manifest(tmp_path), Manifest(tmp_path)  # two download commands started at the same time
    record(a, write(tmp_path, "a.bin", b"a"))
    record(b, write(tmp_path, "b.bin", b"b"))
    a.flush()
    b.flush()
    assert set(Manifest(tmp_path).entries()) == {"a.bin", "b.bin"}


def test_verify_detects_missing_resized_and_corrupted_files(tmp_path):
    files = {name: write(tmp_path, name, bytes(range(200))) for name in ("ok", "gone", "short", "flipped")}
    with Manifest(tmp_path) as manifest:
        for path in files.values():
            record(manifest, path)
    assert verify_manifest(tmp_path) == (4, [])
    files["gone"].unlink()
    files["short"].write_bytes(bytes(range(199)))
    data = bytearray(files["flipped"].read_bytes())
    data[100] ^= 0x01  # one corrupted byte, same size
    files["flipped"].write_bytes(bytes(data))
    n, problems = verify_manifest(tmp_path)
    assert n == 4
    assert {(p.path, p.kind) for p in problems} == {
        ("gone", "missing"),
        ("short", "size"),
        ("flipped", "sha256"),
    }


def test_digests_match_reference_tools(tmp_path):
    path = write(tmp_path, "hello.txt", b"hello\n")
    assert sha256_file(path) == hashlib.sha256(b"hello\n").hexdigest()
    assert git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"  # git hash-object hello.txt
