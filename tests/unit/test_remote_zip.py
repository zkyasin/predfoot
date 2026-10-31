import hashlib
import io
import itertools
import re
import zipfile
from collections.abc import Callable, Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from predfoot.io.remote_zip import (
    BytesFetcher,
    HttpRangeFetcher,
    HttpResponse,
    MemberExtractionError,
    RangeFile,
    RemoteChangedError,
    RemoteError,
    RemoteZip,
    coalesce_ranges,
    download_file,
    parse_content_range,
)

# ------------------------------------------------------------------------------------------------ helpers


def make_zip(members: Mapping[str, bytes], *, deflate: bool = True, zip64: bool = False) -> bytes:
    """An in-memory archive; JPEG-like members are stored, the others deflated (like real datasets)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            if name.endswith("/"):
                zf.writestr(name, b"")
                continue
            method = zipfile.ZIP_DEFLATED if deflate and not name.endswith(".jpg") else zipfile.ZIP_STORED
            info = zipfile.ZipInfo(name)
            info.compress_type = method
            with zf.open(info, "w", force_zip64=zip64) as f:
                f.write(data)
    return buf.getvalue()


def clip_members(rng_bytes: Callable[[int], bytes], n_frames: int = 12) -> dict[str, bytes]:
    members: dict[str, bytes] = {"sequences_info.json": b'{"version": "1.3"}'}
    for clip in ("SNGS-001", "SNGS-002"):
        members[f"{clip}/"] = b""
        members[f"{clip}/Labels-GameState.json"] = b'{"info": {"id": "%s"}}' % clip.encode() * 500
        for frame in range(1, n_frames + 1):
            members[f"{clip}/img1/{frame:06d}.jpg"] = rng_bytes(3000 + 17 * frame)
    return members


@pytest.fixture
def rng_bytes(rng):
    return lambda n: rng.integers(0, 256, size=n, dtype="uint8").tobytes()


# --------------------------------------------------------------------------------------------- coalescing


def test_adjacent_ranges_are_merged_and_far_ones_are_not():
    ranges = [(0, 10), (10, 20), (20, 30), (100, 110)]
    groups = coalesce_ranges(ranges, max_gap=0, max_span=1000)
    assert groups == [(0, 30, [0, 1, 2]), (100, 110, [3])]


def test_small_gaps_are_bridged_up_to_max_gap():
    ranges = [(0, 10), (15, 20), (40, 50)]
    assert coalesce_ranges(ranges, max_gap=5, max_span=1000) == [(0, 20, [0, 1]), (40, 50, [2])]
    assert len(coalesce_ranges(ranges, max_gap=4, max_span=1000)) == 3


def test_max_span_caps_a_group_and_long_ranges_stand_alone():
    ranges = [(0, 10), (10, 20), (20, 30), (30, 200), (200, 210)]
    groups = coalesce_ranges(ranges, max_gap=0, max_span=20)
    assert groups == [(0, 20, [0, 1]), (20, 30, [2]), (30, 200, [3]), (200, 210, [4])]


def test_input_order_does_not_matter_and_indices_point_back():
    ranges = [(100, 110), (0, 10), (10, 20)]
    assert coalesce_ranges(ranges, max_gap=0, max_span=1000) == [(0, 20, [1, 2]), (100, 110, [0])]


@settings(max_examples=200, deadline=None)
@given(
    sizes=st.lists(st.tuples(st.integers(0, 50), st.integers(1, 80)), min_size=0, max_size=40),
    max_gap=st.integers(0, 60),
    max_span=st.integers(1, 300),
)
def test_coalescing_invariants(sizes, max_gap, max_span):
    # Non-overlapping ranges laid out left to right with random gaps, like members of an archive.
    ranges, pos = [], 0
    for gap, length in sizes:
        pos += gap
        ranges.append((pos, pos + length))
        pos += length
    groups = coalesce_ranges(ranges, max_gap=max_gap, max_span=max_span)
    seen = sorted(i for _, _, idx in groups for i in idx)
    assert seen == list(range(len(ranges)))  # every range in exactly one group
    for k, (start, stop, idx) in enumerate(groups):
        members = [ranges[i] for i in idx]
        assert start == min(a for a, _ in members)
        assert stop == max(b for _, b in members)
        assert len(idx) == 1 or stop - start <= max_span
        for (_, b), (a, _) in itertools.pairwise(members):
            assert a - b <= max_gap
        if k + 1 < len(groups):
            assert groups[k + 1][0] >= stop


# -------------------------------------------------------------------------------------------- range file


@settings(max_examples=100, deadline=None)
@given(
    data=st.binary(min_size=0, max_size=3000),
    ops=st.lists(
        st.tuples(st.sampled_from([0, 1, 2]), st.integers(-3500, 3500), st.integers(-1, 1200)), max_size=20
    ),
    readahead=st.integers(1, 2048),
)
def test_range_file_behaves_like_bytesio(data, ops, readahead):
    ref, remote = io.BytesIO(data), RangeFile(BytesFetcher(data), readahead=readahead)
    for whence, offset, n in ops:
        target = offset + {0: 0, 1: ref.tell(), 2: len(data)}[whence]
        if target < 0:  # real files raise here (BytesIO clamps); zipfile expects OSError
            with pytest.raises(OSError, match="negative seek"):
                remote.seek(offset, whence)
            continue
        expected = ref.seek(offset, whence)
        assert remote.seek(offset, whence) == expected
        assert remote.read(n) == ref.read(n)
        assert remote.tell() == ref.tell()


def test_readahead_serves_small_reads_from_one_request():
    fetcher = BytesFetcher(bytes(range(256)) * 100)
    f = RangeFile(fetcher, readahead=1024)
    f.seek(100)
    assert f.read(10) == fetcher.data[100:110]
    assert f.read(10) == fetcher.data[110:120]
    f.seek(50, io.SEEK_CUR)
    assert f.read(4) == fetcher.data[170:174]
    assert fetcher.calls == [(100, 1124)]


# -------------------------------------------------------------------------------------------- remote zip


@pytest.mark.parametrize("zip64", [False, True])
def test_listing_matches_zipfile(rng_bytes, zip64):
    data = make_zip(clip_members(rng_bytes), zip64=zip64)
    remote = RemoteZip(BytesFetcher(data))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert remote.namelist() == zf.namelist()


@pytest.mark.parametrize("zip64", [False, True])
def test_extracted_members_are_byte_identical(tmp_path, rng_bytes, zip64):
    members = clip_members(rng_bytes)
    remote = RemoteZip(BytesFetcher(make_zip(members, zip64=zip64)))
    names = [n for n in members if not n.endswith("/")]
    done = remote.extract({n: tmp_path / n for n in names}, workers=4)
    assert sorted(m.name for m in done) == sorted(names)
    for m in done:
        assert m.path.read_bytes() == members[m.name]
        assert m.size == len(members[m.name])
        assert m.sha256 == hashlib.sha256(members[m.name]).hexdigest()
    assert not list(tmp_path.rglob("*.part"))


def test_adjacent_members_share_one_request_and_strided_ones_do_not(tmp_path, rng_bytes):
    members = clip_members(rng_bytes, n_frames=12)
    fetcher = BytesFetcher(make_zip(members))
    remote = RemoteZip(fetcher)
    fetcher.calls.clear()
    consecutive = [f"SNGS-001/img1/{f:06d}.jpg" for f in range(1, 7)]
    remote.extract({n: tmp_path / "a" / n for n in consecutive}, max_gap=0)
    assert len(fetcher.calls) == 1
    fetcher.calls.clear()
    strided = [f"SNGS-001/img1/{f:06d}.jpg" for f in (1, 6, 11)]
    remote.extract({n: tmp_path / "b" / n for n in strided}, max_gap=0)
    assert len(fetcher.calls) == 3
    fetched = sum(b - a for a, b in fetcher.calls)
    assert fetched < 3 * 3300  # only the selected members (plus their local headers) were read


def test_members_larger_than_max_span_are_streamed(tmp_path, rng_bytes):
    members = {"big.bin": rng_bytes(50_000), "small.json": b"[1, 2, 3]" * 10}
    fetcher = BytesFetcher(make_zip(members))
    remote = RemoteZip(fetcher)
    fetcher.calls.clear()
    remote.extract({n: tmp_path / n for n in members}, max_span=4096, chunk_size=4096)
    assert (tmp_path / "big.bin").read_bytes() == members["big.bin"]
    assert max(b - a for a, b in fetcher.calls) <= 4096
    assert remote.read("small.json") == members["small.json"]


def test_a_corrupted_member_fails_alone_and_leaves_no_file(tmp_path, rng_bytes):
    members = clip_members(rng_bytes, n_frames=3)
    data = bytearray(make_zip(members))
    target = "SNGS-002/img1/000002.jpg"
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as zf:
        info = zf.getinfo(target)
    data[info.header_offset + 30 + len(target) + 100] ^= 0xFF  # flip one byte of the member's data
    remote = RemoteZip(BytesFetcher(bytes(data)))
    names = [n for n in members if n.endswith(".jpg")]
    with pytest.raises(MemberExtractionError) as err:
        remote.extract({n: tmp_path / n for n in names}, max_gap=0)
    assert set(err.value.failures) == {target}
    assert not (tmp_path / target).exists()
    assert (tmp_path / "SNGS-002/img1/000001.jpg").read_bytes() == members["SNGS-002/img1/000001.jpg"]


# -------------------------------------------------------------------------------------- HTTP range fetcher


class FakeServer:
    """An origin that redirects to a signed CDN URL; the signature changes at every resolution."""

    origin = "https://hub.test/datasets/x/resolve/main/train.zip"

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.etag = '"v1"'
        self.signature = 0
        self.expired: set[str] = set()
        self.failures: list[Callable[[], HttpResponse]] = []  # applied to the next CDN requests
        self.requests: list[tuple[str, dict[str, str]]] = []

    def cdn_url(self) -> str:
        return f"https://cdn.test/blob?sig={self.signature}"

    def __call__(self, method, url, headers, *, max_body, timeout):
        self.requests.append((url, dict(headers)))
        if url == self.origin:
            self.signature += 1
            return HttpResponse(302, {"location": self.cdn_url()}, b"Found.")
        if self.failures:
            return self.failures.pop(0)()
        if url in self.expired:
            return HttpResponse(403, {}, b"Request has expired")
        first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        last = min(last, len(self.data) - 1)
        body = self.data[first : last + 1][:max_body]
        headers_out = {"content-range": f"bytes {first}-{last}/{len(self.data)}", "etag": self.etag}
        return HttpResponse(206, headers_out, body)


def make_fetcher(server: FakeServer, sleeps: list[float], **kwargs) -> HttpRangeFetcher:
    return HttpRangeFetcher(
        server.origin, transport=server, sleep=sleeps.append, jitter=False, backoff_s=1.0, **kwargs
    )


def test_resolves_once_and_reuses_the_final_url():
    server, sleeps = FakeServer(bytes(range(200))), []
    fetcher = make_fetcher(server, sleeps, token="hf_secret")
    assert fetcher.size == 200
    assert fetcher.fetch(10, 20) == bytes(range(10, 20))
    assert fetcher.fetch(150, 200) == bytes(range(150, 200))
    assert [url for url, _ in server.requests].count(server.origin) == 1
    assert fetcher.n_resolves == 1
    assert sleeps == []


def test_token_goes_to_the_origin_only():
    server, sleeps = FakeServer(b"x" * 100), []
    fetcher = make_fetcher(server, sleeps, token="hf_secret")
    fetcher.fetch(0, 10)
    for url, headers in server.requests:
        if url == server.origin:
            assert headers["Authorization"] == "Bearer hf_secret"
        else:
            assert "Authorization" not in headers


def test_expired_signature_is_resolved_again():
    server, sleeps = FakeServer(bytes(range(100))), []
    fetcher = make_fetcher(server, sleeps)
    fetcher.fetch(0, 10)
    server.expired.add(server.cdn_url())
    assert fetcher.fetch(10, 30) == bytes(range(10, 30))
    assert fetcher.n_resolves == 2
    assert sleeps == []  # the first refresh is immediate


def test_transient_errors_back_off_exponentially():
    server, sleeps = FakeServer(bytes(range(100))), []
    fetcher = make_fetcher(server, sleeps)
    fetcher.size  # noqa: B018 (resolve first)
    server.failures = [lambda: HttpResponse(503, {}, b""), lambda: HttpResponse(429, {}, b"")]

    def reset():
        raise ConnectionResetError("reset by peer")

    server.failures.append(reset)
    assert fetcher.fetch(5, 15) == bytes(range(5, 15))
    assert sleeps == [1.0, 2.0, 4.0]


def test_short_bodies_are_retried():
    server, sleeps = FakeServer(bytes(range(100))), []
    fetcher = make_fetcher(server, sleeps)
    fetcher.size  # noqa: B018
    server.failures = [lambda: HttpResponse(206, {"content-range": "bytes 0-9/100"}, b"abc")]
    assert fetcher.fetch(0, 10) == bytes(range(10))
    assert sleeps == [1.0]


def test_gives_up_after_max_retries():
    server, sleeps = FakeServer(bytes(range(100))), []
    fetcher = make_fetcher(server, sleeps, max_retries=3)
    fetcher.size  # noqa: B018
    server.failures = [lambda: HttpResponse(500, {}, b"")] * 10
    with pytest.raises(RemoteError, match="giving up"):
        fetcher.fetch(0, 10)
    assert sleeps == [1.0, 2.0, 4.0]


def test_a_changed_remote_file_is_detected():
    server, sleeps = FakeServer(bytes(range(100))), []
    fetcher = make_fetcher(server, sleeps)
    fetcher.fetch(0, 10)
    server.expired.add(server.cdn_url())
    server.etag = '"v2"'
    with pytest.raises(RemoteChangedError):
        fetcher.fetch(10, 20)


def test_missing_file_fails_without_retrying():
    sleeps: list[float] = []

    def transport(method, url, headers, *, max_body, timeout):
        return HttpResponse(404, {}, b"Entry not found")

    fetcher = HttpRangeFetcher("https://hub.test/nope", transport=transport, sleep=sleeps.append)
    with pytest.raises(RemoteError, match="not found"):
        fetcher.size  # noqa: B018
    assert sleeps == []


def test_servers_ignoring_ranges_are_rejected():
    def transport(method, url, headers, *, max_body, timeout):
        return HttpResponse(200, {"content-length": "1000000"}, b"x" * max_body)

    with pytest.raises(RemoteError, match="range requests"):
        HttpRangeFetcher("https://hub.test/f", transport=transport).size  # noqa: B018


def test_content_range_parsing():
    assert parse_content_range("bytes 0-0/11169497921") == (0, 0, 11169497921)
    assert parse_content_range("bytes */0") == (None, None, 0)
    with pytest.raises(RemoteError):
        parse_content_range("items 0-1/2")


def test_remote_zip_over_http_with_a_mid_run_expiry(tmp_path, rng_bytes):
    members = clip_members(rng_bytes, n_frames=4)
    server, sleeps = FakeServer(make_zip(members)), []
    remote = RemoteZip(make_fetcher(server, sleeps))
    server.expired.add(server.cdn_url())
    names = [n for n in members if n.endswith(".json")]
    remote.extract({n: tmp_path / n for n in names}, workers=2)
    for n in names:
        assert (tmp_path / n).read_bytes() == members[n]


# ------------------------------------------------------------------------------------ whole-file download


def test_download_resumes_from_a_partial_file(tmp_path):
    payload = bytes(range(256)) * 40
    server, sleeps = FakeServer(payload), []
    fetcher = make_fetcher(server, sleeps)
    dest = tmp_path / "weights" / "SV_kp"
    dest.parent.mkdir()
    # A previous run stopped after 3000 bytes.
    (dest.parent / "SV_kp.part").write_bytes(payload[:3000])
    stamp = f'{{"url": "{server.origin}", "size": {len(payload)}, "etag": "\\"v1\\""}}'
    (dest.parent / "SV_kp.part.json").write_text(stamp)
    server.requests.clear()
    size, sha = download_file(fetcher, dest, chunk_size=4096)
    assert dest.read_bytes() == payload
    assert (size, sha) == (len(payload), hashlib.sha256(payload).hexdigest())
    ranges = [h["Range"] for u, h in server.requests if u != server.origin and h["Range"] != "bytes=0-0"]
    assert ranges[0] == "bytes=3000-7095"
    assert not list(dest.parent.glob("*.part*"))


def test_download_restarts_when_the_remote_changed(tmp_path):
    payload = b"new content " * 100
    fetcher = make_fetcher(FakeServer(payload), [])
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(b"old content")
    (tmp_path / "f.bin.part.json").write_text('{"url": "x", "size": 11, "etag": "old"}')
    download_file(fetcher, dest, chunk_size=100)
    assert dest.read_bytes() == payload


def test_download_rejects_a_wrong_digest(tmp_path):
    fetcher = make_fetcher(FakeServer(b"payload" * 10), [])
    with pytest.raises(RemoteError, match="sha256"):
        download_file(fetcher, tmp_path / "f.bin", expected_sha256="0" * 64)
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------------------------- network


@pytest.mark.network
def test_lists_a_real_sn_gsr_split():
    remote = RemoteZip(
        HttpRangeFetcher("https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/valid.zip")
    )
    labels = [n for n in remote.namelist() if n.endswith("Labels-GameState.json")]
    assert len(labels) == 58
    assert remote.fetcher.size > 10e9


class GatedOrigin:
    """A gated repo serving a small (non-LFS) file directly from the origin: no redirect, token required."""

    def __init__(self, data: bytes, token: str) -> None:
        self.data, self.token = data, token
        self.requests: list[dict[str, str]] = []

    def __call__(self, method, url, headers, *, max_body, timeout):
        self.requests.append(dict(headers))
        if headers.get("Authorization") != f"Bearer {self.token}":
            return HttpResponse(401, {}, b"Access to dataset is restricted")
        first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        last = min(last, len(self.data) - 1)
        body = self.data[first : last + 1][:max_body]
        return HttpResponse(206, {"content-range": f"bytes {first}-{last}/{len(self.data)}"}, body)


def test_files_served_by_the_origin_itself_keep_the_token():
    server = GatedOrigin(b"0123456789" * 10, token="hf_secret")
    fetcher = HttpRangeFetcher(
        "https://hub.test/f.json", token="hf_secret", transport=server, sleep=lambda s: None
    )
    assert fetcher.fetch(10, 20) == b"0123456789"
    assert all(h.get("Authorization") == "Bearer hf_secret" for h in server.requests)


def test_a_persistent_refusal_after_refreshing_fails_fast():
    server, sleeps = FakeServer(bytes(range(100))), []
    fetcher = make_fetcher(server, sleeps)
    fetcher.size  # noqa: B018
    server.failures = [lambda: HttpResponse(403, {}, b"denied")] * 20
    server.requests.clear()
    with pytest.raises(RemoteError, match=r"refused|403"):
        fetcher.fetch(0, 10)
    assert len(server.requests) <= 6
    assert sleeps == []
