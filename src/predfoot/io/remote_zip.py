"""Read remote files, and members of remote zip archives, over HTTP range requests.

SN-GSR ships each split as one 5-11 GB zip, but predfoot only needs the labels and a fraction of the frames. A
zip keeps its central directory at the end, so a reader can list the members with a few small range requests
and then fetch exactly the bytes of the members it wants.

Layers, each testable without network:

- :class:`Transport`: one HTTP request, redirects not followed (default :class:`HttpClientTransport`);
- :class:`HttpRangeFetcher`: follows redirects once to the final (often signed, expiring) URL, resolves it
  again on 401/403/404/410, retries transient failures with exponential backoff, checks every
  ``Content-Range``;
- :class:`RangeFile`: a seekable read-only file over any :class:`RangeFetcher`, enough for :mod:`zipfile`;
- :class:`RemoteZip`: lists members, coalesces neighbouring members into single range requests and extracts
  them in a thread pool, checking sizes and CRC-32;
- :func:`download_file`: resumable whole-file download through the same fetcher.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import random
import re
import struct
import threading
import time
import zipfile
import zlib
from collections.abc import Buffer, Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin, urlsplit

from predfoot import __version__

USER_AGENT = f"predfoot/{__version__}"

_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_TRANSIENT = frozenset({408, 425, 429, 500, 502, 503, 504})
#: On a resolved URL these usually mean that its signature expired: resolve again from the original URL.
_REFRESH = frozenset({401, 403, 404, 410})
_CONTENT_RANGE = re.compile(r"^bytes (?:(\d+)-(\d+)|\*)/(\d+)$")
_NETWORK_ERRORS: tuple[type[BaseException], ...] = (OSError, http.client.HTTPException)

_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_LOCAL_SIGNATURE = b"PK\x03\x04"


class RemoteError(OSError):
    """A remote file could not be read."""


class RemoteChangedError(RemoteError):
    """The remote file changed (size or ETag) while it was being read."""


# --------------------------------------------------------------------------------------------- HTTP transport


@dataclass(frozen=True)
class HttpResponse:
    """Status, headers (lower-case names) and the body, possibly capped by the caller."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class Transport(Protocol):
    """Performs one HTTP request without following redirects."""

    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], *, max_body: int, timeout: float
    ) -> HttpResponse:
        """Return the response, reading at most ``max_body`` bytes of a 2xx body."""
        ...


class HttpClientTransport:
    """HTTP/1.1 over :mod:`http.client`: no redirects, one keep-alive connection per thread and host."""

    #: Bytes read from a non-2xx body, enough to drain the short messages that come with redirects and errors.
    ERROR_BODY_LIMIT = 1 << 16

    def __init__(self, user_agent: str = USER_AGENT) -> None:
        self._user_agent = user_agent
        self._local = threading.local()

    def _pool(self) -> dict[tuple[str, str, int | None], http.client.HTTPConnection]:
        pool: dict[tuple[str, str, int | None], http.client.HTTPConnection] | None = getattr(
            self._local, "pool", None
        )
        if pool is None:
            pool = {}
            self._local.pool = pool
        return pool

    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], *, max_body: int, timeout: float
    ) -> HttpResponse:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"unsupported URL: {url!r}")
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        key = (parts.scheme, parts.hostname, parts.port)
        pool = self._pool()
        all_headers = {"User-Agent": self._user_agent, **headers}
        while True:
            conn = pool.pop(key, None)
            reused = conn is not None
            if conn is None:
                factory = (
                    http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
                )
                conn = factory(parts.hostname, parts.port, timeout=timeout)
            try:
                conn.request(method, target, headers=all_headers)
                resp = conn.getresponse()
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                conn.close()
                if reused:  # the server closed an idle keep-alive connection: retry once on a new one
                    continue
                raise
            except BaseException:
                conn.close()
                raise
            try:
                limit = max_body if 200 <= resp.status < 300 else self.ERROR_BODY_LIMIT
                body = b"" if method == "HEAD" else resp.read(limit)
            except BaseException:
                conn.close()
                raise
            if resp.isclosed() and not resp.will_close:
                pool[key] = conn  # body fully read: the connection can carry the next request
            else:
                conn.close()
            return HttpResponse(resp.status, {k.lower(): v for k, v in resp.getheaders()}, body)


# ---------------------------------------------------------------------------------------------- range fetcher


class RangeFetcher(Protocol):
    """Random access to the bytes of one remote file."""

    @property
    def size(self) -> int:
        """Total size in bytes."""
        ...

    def fetch(self, start: int, stop: int) -> bytes:
        """Return exactly the bytes ``[start, stop)``, or raise."""
        ...


def parse_content_range(value: str | None) -> tuple[int | None, int | None, int]:
    """Parse ``bytes a-b/total`` (or ``bytes */total``) into ``(a, b, total)``, b inclusive."""
    match = _CONTENT_RANGE.match((value or "").strip())
    if match is None:
        raise RemoteError(f"unexpected Content-Range: {value!r}")
    first, last, total = match.groups()
    return (int(first) if first else None, int(last) if last else None, int(total))


class HttpRangeFetcher:
    """Byte ranges of one remote file, robust to expiring signed URLs and transient failures. Thread-safe.

    The original URL is resolved once, with a one-byte range request that follows redirects by hand: the token
    (if any) is only sent to the original host, never to the CDN the redirect points to. Later range requests
    go straight to the final URL. When that URL stops working (expired signature), it is resolved again; if
    the file's size or ETag changed in the meantime, :class:`RemoteChangedError` is raised because byte
    offsets computed earlier would no longer be valid.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        transport: Transport | None = None,
        max_retries: int = 8,
        backoff_s: float = 1.0,
        max_backoff_s: float = 60.0,
        timeout_s: float = 60.0,
        max_redirects: int = 10,
        sleep: Callable[[float], None] = time.sleep,
        jitter: bool = True,
    ) -> None:
        self.url = url
        self._token = token
        self._transport: Transport = transport if transport is not None else HttpClientTransport()
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self.timeout_s = timeout_s
        self.max_redirects = max_redirects
        self._sleep = sleep
        self._jitter = jitter
        self._lock = threading.Lock()
        self._final_url: str | None = None
        self._size: int | None = None
        self._etag: str | None = None
        self._stats_lock = threading.Lock()
        self.n_requests = 0
        self.n_bytes = 0
        self.n_resolves = 0

    # -- public API

    @property
    def size(self) -> int:
        self._current_url()
        assert self._size is not None
        return self._size

    @property
    def etag(self) -> str | None:
        self._current_url()
        return self._etag

    def resolve(self) -> str:
        """Resolve the original URL again (following redirects) and return the final URL."""
        with self._lock:
            return self._resolve_locked()

    def fetch(self, start: int, stop: int) -> bytes:
        size = self.size
        if not 0 <= start <= stop <= size:
            raise ValueError(f"range [{start}, {stop}) outside a file of {size} bytes")
        if start == stop:
            return b""
        want = stop - start
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            url = self._current_url()
            try:
                resp = self._request(url, {"Range": f"bytes={start}-{stop - 1}"}, max_body=want)
            except _NETWORK_ERRORS as exc:
                last = exc
                self._backoff(attempt)
                continue
            if resp.status == 206:
                first, end, total = parse_content_range(resp.headers.get("content-range"))
                if total != size:
                    raise RemoteChangedError(f"{self.url}: size changed from {size} to {total} bytes")
                if (first, end) == (start, stop - 1) and len(resp.body) == want:
                    with self._stats_lock:
                        self.n_bytes += want
                    return resp.body
                last = RemoteError(f"got {len(resp.body)} bytes ({first}-{end}) for {start}-{stop - 1}")
                self._backoff(attempt)
            elif resp.status == 200 and start == 0 and stop == size and len(resp.body) == size:
                return resp.body
            elif resp.status in _REFRESH or resp.status in _REDIRECTS:
                last = RemoteError(f"HTTP {resp.status} on the resolved URL of {self.url}")
                self._invalidate(url)
                if attempt:  # the first refresh is immediate: an expired signature is the common case
                    self._backoff(attempt - 1)
            elif resp.status in _TRANSIENT:
                last = RemoteError(f"HTTP {resp.status} for {self.url}")
                self._backoff(attempt)
            else:
                raise RemoteError(f"HTTP {resp.status} for bytes {start}-{stop - 1} of {self.url}")
        raise RemoteError(
            f"giving up on bytes {start}-{stop - 1} of {self.url} after {self.max_retries + 1} attempts"
        ) from last

    # -- internals

    def _request(self, url: str, headers: Mapping[str, str], *, max_body: int) -> HttpResponse:
        with self._stats_lock:
            self.n_requests += 1
        return self._transport("GET", url, headers, max_body=max_body, timeout=self.timeout_s)

    def _backoff(self, attempt: int) -> None:
        if attempt >= self.max_retries:
            return  # the caller's loop ends and raises
        delay = min(self.max_backoff_s, self.backoff_s * 2.0**attempt)
        if self._jitter:
            delay *= random.uniform(0.5, 1.0)
        self._sleep(delay)

    def _current_url(self) -> str:
        with self._lock:
            if self._final_url is None:
                return self._resolve_locked()
            return self._final_url

    def _invalidate(self, url: str) -> None:
        with self._lock:
            if self._final_url == url:
                self._final_url = None

    def _follow(self) -> tuple[HttpResponse, str]:
        url = self.url
        origin = urlsplit(self.url)
        for _ in range(self.max_redirects + 1):
            here = urlsplit(url)
            headers = {"Range": "bytes=0-0"}
            if self._token and (here.scheme, here.netloc) == ("https", origin.netloc):
                headers["Authorization"] = f"Bearer {self._token}"
            resp = self._request(url, headers, max_body=1)
            if resp.status in _REDIRECTS and "location" in resp.headers:
                url = urljoin(url, resp.headers["location"])
                continue
            return resp, url
        raise RemoteError(f"too many redirects for {self.url}")

    def _resolve_locked(self) -> str:
        last: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp, url = self._follow()
            except RemoteError:
                raise
            except _NETWORK_ERRORS as exc:
                last = exc
                self._backoff(attempt)
                continue
            if resp.status in _TRANSIENT:
                last = RemoteError(f"HTTP {resp.status} while resolving {self.url}")
                self._backoff(attempt)
                continue
            if resp.status in (206, 416):
                _, _, total = parse_content_range(resp.headers.get("content-range"))
            elif resp.status == 200:
                raise RemoteError(f"{self.url} does not support range requests")
            elif resp.status == 404:
                raise RemoteError(f"not found: {self.url}")
            elif resp.status in (401, 403):
                raise RemoteError(
                    f"HTTP {resp.status} for {self.url}: missing token, or dataset terms not accepted"
                )
            else:
                raise RemoteError(f"HTTP {resp.status} while resolving {self.url}")
            etag = resp.headers.get("etag")
            if self._size is not None and (
                total != self._size or (self._etag and etag and etag != self._etag)
            ):
                raise RemoteChangedError(f"{self.url} changed since it was first opened")
            self._final_url, self._size, self._etag = url, total, etag
            self.n_resolves += 1
            return url
        raise RemoteError(f"could not resolve {self.url}") from last


class BytesFetcher:
    """In-memory :class:`RangeFetcher`, for tests and small local archives. Records every call."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.calls: list[tuple[int, int]] = []

    @property
    def size(self) -> int:
        return len(self.data)

    def fetch(self, start: int, stop: int) -> bytes:
        if not 0 <= start <= stop <= len(self.data):
            raise ValueError(f"range [{start}, {stop}) outside a file of {len(self.data)} bytes")
        self.calls.append((start, stop))
        return self.data[start:stop]


# ------------------------------------------------------------------------------------------- seekable file


class RangeFile(io.RawIOBase):
    """Seekable read-only file over a :class:`RangeFetcher`, with a small read-ahead buffer.

    Each read outside the buffer costs one range request, so it suits a few large reads (zip central
    directory), not byte-by-byte parsing.
    """

    def __init__(self, fetcher: RangeFetcher, readahead: int = 1 << 16) -> None:
        super().__init__()
        self._fetcher = fetcher
        self._size = fetcher.size
        self._readahead = readahead
        self._pos = 0
        self._buf = b""
        self._buf_start = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if pos < 0:
            raise OSError(f"negative seek position {pos}")
        self._pos = pos
        return pos

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        stop = min(self._pos + size, self._size)
        if stop <= self._pos:
            return b""
        data = self._range(self._pos, stop)
        self._pos = stop
        return data

    def readall(self) -> bytes:
        return self.read(-1)

    def readinto(self, buffer: Buffer) -> int:
        view = memoryview(buffer).cast("B")
        data = self.read(len(view))
        view[: len(data)] = data
        return len(data)

    def _range(self, start: int, stop: int) -> bytes:
        buf_stop = self._buf_start + len(self._buf)
        if self._buf_start <= start and stop <= buf_stop:
            return self._buf[start - self._buf_start : stop - self._buf_start]
        fetch_stop = min(max(stop, start + self._readahead), self._size)
        self._buf = self._fetcher.fetch(start, fetch_stop)
        self._buf_start = start
        return self._buf[: stop - start]


# -------------------------------------------------------------------------------------------- coalescing


def coalesce_ranges(
    ranges: Sequence[tuple[int, int]], *, max_gap: int, max_span: int
) -> list[tuple[int, int, list[int]]]:
    """Group byte ranges ``[start, stop)`` into fewer, larger requests.

    Ranges are taken in order of ``start``. A range joins the current group when the gap to the group is at
    most ``max_gap`` bytes and the group would still span at most ``max_span`` bytes; a range longer than
    ``max_span`` forms a group of its own. Returns ``(start, stop, indices)`` per group, sorted by ``start``,
    where ``indices`` point into ``ranges``.
    """
    if max_gap < 0 or max_span <= 0:
        raise ValueError("max_gap must be >= 0 and max_span > 0")
    groups: list[tuple[int, int, list[int]]] = []
    for i in sorted(range(len(ranges)), key=lambda k: ranges[k]):
        start, stop = ranges[i]
        if stop < start:
            raise ValueError(f"invalid range {ranges[i]}")
        if groups:
            g_start, g_stop, members = groups[-1]
            if start - g_stop <= max_gap and max(stop, g_stop) - g_start <= max_span:
                groups[-1] = (g_start, max(stop, g_stop), [*members, i])
                continue
        groups.append((start, stop, [i]))
    return groups


# --------------------------------------------------------------------------------------------- remote zip


@dataclass(frozen=True)
class MemberExtent:
    """Bytes of the archive that hold one member: local header, data and optional data descriptor."""

    info: zipfile.ZipInfo
    start: int  # offset of the local header
    stop: int  # offset of the next local header, or of the central directory


@dataclass(frozen=True)
class ExtractedMember:
    """One member written to disk."""

    name: str
    path: Path
    size: int
    sha256: str


ReadFn = Callable[[int, int], Buffer]


def iter_member_data(read: ReadFn, extent: MemberExtent, *, chunk_size: int = 1 << 22) -> Iterator[bytes]:
    """Yield the decompressed content of one member; raise :class:`zipfile.BadZipFile` on any inconsistency.

    ``read(start, stop)`` returns archive bytes. Sizes and CRC-32 come from the central directory and are
    checked at the end, so a consumer must not trust the content before the iterator is exhausted.
    """
    info = extent.info
    if info.flag_bits & 0x1:
        raise NotImplementedError(f"{info.filename}: encrypted members are not supported")
    header = bytes(read(extent.start, extent.start + _LOCAL_HEADER.size))
    if len(header) != _LOCAL_HEADER.size:
        raise zipfile.BadZipFile(f"{info.filename}: truncated local header")
    fields = _LOCAL_HEADER.unpack(header)
    if fields[0] != _LOCAL_SIGNATURE:
        raise zipfile.BadZipFile(f"{info.filename}: bad local header signature")
    name_len, extra_len = fields[9], fields[10]
    name_start = extent.start + _LOCAL_HEADER.size
    local_name = bytes(read(name_start, name_start + name_len))
    flag_utf8 = bool(info.flag_bits & 0x800)
    if local_name.decode("utf-8" if flag_utf8 else "cp437") != info.orig_filename:
        raise zipfile.BadZipFile(f"{info.filename}: local header names {local_name!r}")
    data_start = name_start + name_len + extra_len
    data_stop = data_start + info.compress_size
    if data_stop > extent.stop:
        raise zipfile.BadZipFile(f"{info.filename}: data overruns the next member")
    if info.compress_type == zipfile.ZIP_STORED:
        inflater = None
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-15)
    else:
        raise NotImplementedError(f"{info.filename}: compression method {info.compress_type}")

    crc, size = 0, 0

    def account(out: bytes) -> bytes:
        nonlocal crc, size
        crc = zlib.crc32(out, crc)
        size += len(out)
        if size > info.file_size:
            raise zipfile.BadZipFile(f"{info.filename}: more data than the declared {info.file_size} bytes")
        return out

    for pos in range(data_start, data_stop, chunk_size):
        raw: Buffer = read(pos, min(pos + chunk_size, data_stop))
        if inflater is None:
            yield account(bytes(raw))
            continue
        while True:  # bounded output per call, so a corrupt stream cannot exhaust memory
            out = inflater.decompress(raw, chunk_size)
            if out:
                yield account(out)
            raw = inflater.unconsumed_tail
            if not raw:
                break
    if inflater is not None:
        tail = inflater.flush()
        if tail:
            yield account(tail)
        if not inflater.eof:
            raise zipfile.BadZipFile(f"{info.filename}: truncated deflate stream")
    if size != info.file_size or crc != info.CRC:
        raise zipfile.BadZipFile(f"{info.filename}: size or CRC-32 mismatch")


def write_atomic(chunks: Iterable[Buffer], dest: Path) -> tuple[int, str]:
    """Write chunks to ``dest`` through a ``.part`` file renamed at the end; return (size, sha256)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with part.open("wb") as f:
            for chunk in chunks:
                f.write(chunk)
                digest.update(chunk)
                size += memoryview(chunk).nbytes
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return size, digest.hexdigest()


class MemberExtractionError(RemoteError):
    """Some members could not be extracted; the others were written."""

    def __init__(self, failures: Mapping[str, BaseException]) -> None:
        self.failures = dict(failures)
        names = ", ".join(sorted(self.failures)[:5])
        super().__init__(f"{len(self.failures)} member(s) failed, e.g. {names}")


class RemoteZip:
    """A zip archive read through a :class:`RangeFetcher`: list members and extract a selection of them."""

    def __init__(self, fetcher: RangeFetcher) -> None:
        self.fetcher = fetcher
        with zipfile.ZipFile(RangeFile(fetcher)) as zf:
            infos = zf.infolist()
            central_directory = zf.start_dir
        offsets = sorted({info.header_offset for info in infos})
        next_offset = dict(zip(offsets, [*offsets[1:], central_directory], strict=True))
        self._infos = infos
        self._extents = {
            info.filename: MemberExtent(info, info.header_offset, next_offset[info.header_offset])
            for info in infos
        }

    def infolist(self) -> list[zipfile.ZipInfo]:
        return list(self._infos)

    def namelist(self) -> list[str]:
        return [info.filename for info in self._infos]

    def extent(self, name: str) -> MemberExtent:
        return self._extents[name]

    def read(self, name: str) -> bytes:
        """Return the content of one member (meant for small members)."""
        ext = self._extents[name]
        blob = self.fetcher.fetch(ext.start, ext.stop)
        return b"".join(iter_member_data(_slicer(blob, ext.start), ext))

    def extract(
        self,
        targets: Mapping[str, Path],
        *,
        workers: int = 8,
        max_gap: int = 1 << 16,
        max_span: int = 32 << 20,
        chunk_size: int = 1 << 22,
        on_done: Callable[[ExtractedMember], None] | None = None,
    ) -> list[ExtractedMember]:
        """Extract members ``{name: destination}`` with coalesced range requests in a thread pool.

        Each file is written atomically after its size and CRC-32 check. ``on_done`` runs in the calling
        thread as members complete. Failures do not stop the other members; they are raised together at the
        end as :class:`MemberExtractionError`.
        """
        extents = [self._extents[name] for name in targets]
        groups = coalesce_ranges([(e.start, e.stop) for e in extents], max_gap=max_gap, max_span=max_span)

        def run(
            start: int, stop: int, members: list[MemberExtent]
        ) -> list[tuple[str, ExtractedMember | Exception]]:
            read: ReadFn
            if stop - start <= max_span:
                read = _slicer(self.fetcher.fetch(start, stop), start)
            else:  # one large member: stream it chunk by chunk
                read = self.fetcher.fetch
            done: list[tuple[str, ExtractedMember | Exception]] = []
            for ext in members:
                name, dest = ext.info.filename, targets[ext.info.filename]
                try:
                    size, sha = write_atomic(iter_member_data(read, ext, chunk_size=chunk_size), dest)
                except Exception as exc:  # one bad member must not sink its neighbours
                    done.append((name, exc))
                else:
                    done.append((name, ExtractedMember(name, dest, size, sha)))
            return done

        results: list[ExtractedMember] = []
        failures: dict[str, BaseException] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(run, start, stop, [extents[i] for i in idx]): [extents[i] for i in idx]
                for start, stop, idx in groups
            }
            for future in as_completed(futures):
                try:
                    batch = future.result()
                except Exception as exc:  # the group's range request itself failed
                    failures.update((ext.info.filename, exc) for ext in futures[future])
                    continue
                for name, outcome in batch:
                    if isinstance(outcome, Exception):
                        failures[name] = outcome
                        continue
                    results.append(outcome)
                    if on_done is not None:
                        on_done(outcome)
        if failures:
            raise MemberExtractionError(failures)
        return results


def _slicer(blob: bytes, offset: int) -> ReadFn:
    view = memoryview(blob)

    def read(start: int, stop: int) -> memoryview:
        if start < offset or stop > offset + len(view):
            raise zipfile.BadZipFile(f"read [{start}, {stop}) outside the fetched block")
        return view[start - offset : stop - offset]

    return read


# ------------------------------------------------------------------------------------- whole-file download


def download_file(
    fetcher: RangeFetcher,
    dest: Path,
    *,
    chunk_size: int = 16 << 20,
    expected_sha256: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, str]:
    """Download a whole remote file to ``dest``, resuming a ``.part`` file left by an interrupted run.

    The size and ETag seen when the download started are kept next to the ``.part`` file; if they differ on
    resume, the download restarts from zero. The file is renamed into place only when complete (and, if
    ``expected_sha256`` is given, when its digest matches). Returns ``(size, sha256)``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    meta = dest.with_name(dest.name + ".part.json")
    size = fetcher.size
    stamp = {"url": getattr(fetcher, "url", None), "size": size, "etag": getattr(fetcher, "etag", None)}
    offset = 0
    if part.exists():
        try:
            previous = json.loads(meta.read_text())
        except (OSError, ValueError):
            previous = None
        if previous == stamp and part.stat().st_size <= size:
            offset = part.stat().st_size
        else:
            part.unlink()
    meta.write_text(json.dumps(stamp))
    digest = hashlib.sha256()
    if offset:
        with part.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    with part.open("ab") as f:
        while offset < size:
            data = fetcher.fetch(offset, min(offset + chunk_size, size))
            f.write(data)
            digest.update(data)
            offset += len(data)
            if progress is not None:
                progress(offset, size)
    sha = digest.hexdigest()
    if expected_sha256 is not None and sha != expected_sha256.lower():
        part.unlink()
        meta.unlink(missing_ok=True)
        raise RemoteError(f"{dest.name}: sha256 {sha} does not match the expected {expected_sha256}")
    os.replace(part, dest)
    meta.unlink(missing_ok=True)
    return size, sha
