"""ASGI middleware of the HTTP API: Prometheus request metrics and a limit on request body sizes."""

from __future__ import annotations

import time
from collections.abc import Mapping

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Latency buckets in seconds: from a localisation (milliseconds) to a CPU calibration (seconds).
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
PROMETHEUS_CONTENT_TYPE = CONTENT_TYPE_LATEST
#: Methods counted under their own name. Clients can send any token as a method; the rest count as "other".
METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


def route_path(scope: Scope) -> str:
    """The path that the router matches, without a trailing slash.

    Servers such as uvicorn put the ``root_path`` (the prefix of a proxy that mounts the app under, say,
    /predfoot) in front of ``path``; Starlette's router strips it with the same rule as here.
    """
    path: str = scope["path"]
    root: str = scope.get("root_path", "")
    if root and path.startswith(root) and (len(path) == len(root) or path[len(root)] == "/"):
        path = path[len(root) :]
    return path.rstrip("/") or "/"


class RequestMetrics:
    """A request counter and a latency histogram, in a registry of their own (one per app)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "predfoot_http_requests",
            "HTTP requests by route, method and status code.",
            ("route", "method", "status"),
            registry=self.registry,
        )
        self.latency = Histogram(
            "predfoot_http_request_duration_seconds",
            "HTTP request latency by route.",
            ("route",),
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

    def observe(self, route: str, method: str, status: int, seconds: float) -> None:
        self.requests.labels(route, method, str(status)).inc()
        self.latency.labels(route).observe(seconds)

    def render(self) -> bytes:
        """The metrics in the Prometheus text format."""
        return generate_latest(self.registry)


class MetricsMiddleware:
    """Counts every HTTP request and times it. Paths outside ``routes`` and methods outside :data:`METHODS`
    share the label "other", so unknown URLs and made-up methods cannot grow the number of series."""

    def __init__(self, app: ASGIApp, metrics: RequestMetrics, routes: frozenset[str]) -> None:
        self.app = app
        self.metrics = metrics
        self.routes = routes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        status = 500  # what the client sees if the app raises before answering

        async def send_and_record(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_and_record)
        finally:
            route = route_path(scope)
            method = scope["method"]
            self.metrics.observe(
                route if route in self.routes else "other",
                method if method in METHODS else "other",
                status,
                time.perf_counter() - start,
            )


class BodySizeLimitMiddleware:
    """Answers 413 to request bodies above a per-path limit: at once when Content-Length announces one, and
    otherwise as soon as the streamed body passes the limit, so an oversized upload is never read in full."""

    def __init__(self, app: ASGIApp, limits: Mapping[str, int], default: int) -> None:
        self.app = app
        self.limits = dict(limits)
        self.default = default

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limits.get(route_path(scope), self.default)
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            await JSONResponse({"detail": _too_large(limit)}, status_code=413)(scope, receive, send)
            return
        received = 0

        async def receive_within_limit() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, _too_large(limit))
            return message

        await self.app(scope, receive_within_limit, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _too_large(limit: int) -> str:
    return f"request body larger than {limit} bytes"
