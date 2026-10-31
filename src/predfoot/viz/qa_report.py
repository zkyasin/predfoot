"""Per-run QA report: one self-contained HTML file next to ``tracking.jsonl`` (T5.2).

``predfoot run`` writes ``qa_report.html`` at the end of every run; ``predfoot qa OUT`` rebuilds it from the
files of a finished run. The report reads ``tracking.jsonl`` (required), and ``match.json``, ``meta.json``,
``metrics.json`` and ``minimap.png`` when they exist, and shows:

- the run summary: frames, published share, abstentions, tracks, positions, provenance and stages;
- the validation of the outputs (:func:`predfoot.io.validate.validate_tracking`), plus consistency checks
  between ``metrics.json``, ``match.json`` and ``tracking.jsonl``;
- a per-frame timeline of calibration health, abstentions and p90 radii;
- the distribution of the p90 radii;
- the live frames with the largest radii, and the abstained spans;
- short definitions of the fields.

The file opens offline, in any browser: CSS and SVG are inline, the minimap is embedded as a base64 PNG, there
is no JavaScript and no external reference, and a Content-Security-Policy forbids loading anything else.
Every string read from the run files is HTML-escaped. The report holds no timestamps or timings (those stay
in ``metrics.json``), so a resumed run writes the same bytes.
"""

from __future__ import annotations

import base64
import html
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from predfoot import __version__
from predfoot.io.files import atomic_write_text
from predfoot.io.skillcorner import EXPORT_FPS
from predfoot.io.validate import Report, validate_tracking

REPORT_NAME = "qa_report.html"
#: At most this many problems are listed by the validation section.
MAX_ISSUES = 20
#: Rows of the worst-frames table, and abstained spans listed.
WORST_FRAMES = 15
MAX_SPANS = 20
#: The timeline groups frames into at most this many columns (a full match has 54 000 export frames).
MAX_COLUMNS = 600
HISTOGRAM_BINS = 30
#: A minimap larger than this is not embedded.
MAX_EMBEDDED_PNG_BYTES = 5 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: Nothing may be loaded: no script, no font, no stylesheet, no image other than the embedded minimap.
CONTENT_SECURITY_POLICY = "default-src 'none'; style-src 'unsafe-inline'; img-src data:"

# Chart geometry, in SVG user units; the SVG scales to the width of the page.
_W = 960
_LEFT, _RIGHT = 64, 20
_PLOT_W = _W - _LEFT - _RIGHT


# --- reading the run ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameRow:
    """What the report needs from one record of ``tracking.jsonl``."""

    frame: int
    timestamp: str | None
    abstain: bool
    health: float | None
    n_players: int
    n_located: int
    radius_median: float | None
    radius_max: float | None
    worst_track: int | None


@dataclass
class TrackingSummary:
    """One streamed pass over ``tracking.jsonl``: a row per record and every published p90 radius."""

    rows: list[FrameRow] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)
    track_ids: set[int] = field(default_factory=set)
    unreadable: int = 0  # lines that are not a JSON object; the validation section says why

    @property
    def n_frames(self) -> int:
        return len(self.rows)

    @property
    def n_abstained(self) -> int:
        return sum(r.abstain for r in self.rows)

    @property
    def n_located(self) -> int:
        return sum(r.n_located for r in self.rows)

    @property
    def rows_in_live_frames(self) -> int:
        return sum(r.n_players for r in self.rows if not r.abstain)


def _finite(value: Any) -> float | None:
    """``value`` as a float if it is a finite JSON number (booleans are not numbers here), else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _frame_row(record: Mapping[str, Any], index: int, summary: TrackingSummary) -> FrameRow:
    raw_players = record.get("player_data")
    players = [p for p in raw_players if isinstance(p, dict)] if isinstance(raw_players, list) else []
    located = 0
    radii: list[tuple[float, int | None]] = []
    for p in players:
        track = _int(p.get("player_id"))
        if track is not None:
            summary.track_ids.add(track)
        if _finite(p.get("x")) is None or _finite(p.get("y")) is None:
            continue
        located += 1
        radius = _finite(p.get("p90_radius_m"))
        if radius is not None and radius >= 0:
            radii.append((radius, track))
    summary.radii.extend(r for r, _ in radii)
    worst = max(radii, key=lambda rt: rt[0]) if radii else None
    frame = _int(record.get("frame"))
    timestamp = record.get("timestamp")
    health = _finite(record.get("calib_health"))
    return FrameRow(
        frame=index if frame is None else frame,
        timestamp=timestamp if isinstance(timestamp, str) else None,
        abstain=record.get("abstain") is True,
        health=health if health is not None and 0.0 <= health <= 1.0 else None,
        n_players=len(players),
        n_located=located,
        radius_median=float(np.median([r for r, _ in radii])) if radii else None,
        radius_max=None if worst is None else worst[0],
        worst_track=None if worst is None else worst[1],
    )


def summarise_tracking(path: Path) -> TrackingSummary:
    """Read ``tracking.jsonl`` line by line; malformed lines are counted, never fatal."""
    summary = TrackingSummary()
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                summary.unreadable += 1
                continue
            if not isinstance(record, dict):
                summary.unreadable += 1
                continue
            summary.rows.append(_frame_row(record, len(summary.rows), summary))
    return summary


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """(object, None), or (None, why it could not be used)."""
    if not path.is_file():
        return None, "missing"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as err:
        return None, f"unreadable ({type(err).__name__}: {err})"
    if not isinstance(obj, dict):
        return None, "not a JSON object"
    return obj, None


def _get(obj: Any, *keys: str) -> Any:
    """Nested lookup that returns None as soon as a level is missing or not an object."""
    value = obj
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


@dataclass(frozen=True)
class RunFiles:
    """Everything the report shows, read from an output folder."""

    out_dir: Path
    summary: TrackingSummary
    validation: Report | None
    validation_error: str | None
    match: dict[str, Any] | None
    meta: dict[str, Any] | None
    metrics: dict[str, Any] | None
    problems: dict[str, str]  # file name → why it could not be used
    export_fps: float
    minimap_png: bytes | None


def read_run(out_dir: Path) -> RunFiles:
    """Read the outputs of a run. Raises ``FileNotFoundError`` when ``tracking.jsonl`` is missing."""
    out_dir = Path(out_dir)
    tracking = out_dir / "tracking.jsonl"
    if not tracking.is_file():
        raise FileNotFoundError(f"{tracking} not found: is {out_dir} the output folder of predfoot run?")
    problems: dict[str, str] = {}
    loaded: dict[str, dict[str, Any] | None] = {}
    for name in ("match.json", "meta.json", "metrics.json"):
        obj, problem = _load_json(out_dir / name)
        loaded[name] = obj
        if problem:
            problems[name] = problem
    fps = _finite(_get(loaded["meta.json"], "export_fps"))
    export_fps = fps if fps is not None and fps > 0 else float(EXPORT_FPS)
    validation, validation_error = None, None
    try:
        validation = validate_tracking(
            tracking, match=loaded["match.json"], fps=export_fps, max_issues=MAX_ISSUES
        )
    except (KeyError, TypeError, ValueError, OSError) as err:  # e.g. a match.json without integer ids
        validation_error = f"{type(err).__name__}: {err}"
    png = None
    minimap = out_dir / "minimap.png"
    if minimap.is_file():
        data = minimap.read_bytes()
        if len(data) <= MAX_EMBEDDED_PNG_BYTES and data.startswith(PNG_SIGNATURE):
            png = data
        else:
            problems["minimap.png"] = "not embedded (not a PNG, or larger than 5 MiB)"
    return RunFiles(
        out_dir=out_dir,
        summary=summarise_tracking(tracking),
        validation=validation,
        validation_error=validation_error,
        match=loaded["match.json"],
        meta=loaded["meta.json"],
        metrics=loaded["metrics.json"],
        problems=problems,
        export_fps=export_fps,
        minimap_png=png,
    )


# --- formatting helpers ------------------------------------------------------------------------------


def _e(value: object) -> str:
    """HTML-escaped text of any value (quotes too, so it is also safe inside attributes)."""
    return html.escape(str(value), quote=True)


def _num(value: float | None, digits: int = 2, unit: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{unit}"


def _pct(part: int, whole: int) -> str:
    return "—" if whole == 0 else f"{100.0 * part / whole:.1f} %"


def _f(x: float) -> str:
    """SVG coordinate, one decimal."""
    return f"{x:.1f}"


def _line(cls: str, x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<line class="{cls}" x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}"/>'


def _text(cls: str, x: float, y: float, text: str, anchor: str = "start") -> str:
    return f'<text class="{cls}" x="{_f(x)}" y="{_f(y)}" text-anchor="{anchor}">{_e(text)}</text>'


def _rect(cls: str, x: float, y: float, w: float, h: float, tip: str | None = None) -> str:
    attrs = f'class="{cls}" x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}"'
    return f"<rect {attrs}/>" if tip is None else f"<rect {attrs}><title>{_e(tip)}</title></rect>"


def _time_label(seconds: float, duration: float) -> str:
    if duration < 60:
        return f"{seconds:g} s"
    total = round(seconds)
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _nice_step(span: float, target: int) -> float:
    """A 1-2-5 step that cuts ``span`` into about ``target`` intervals."""
    if span <= 0:
        return 1.0
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0):
        if raw <= m * magnitude:
            return m * magnitude
    return 10.0 * magnitude


def _nice_ceiling(value: float, target: int = 4) -> tuple[float, float]:
    """(upper bound, tick step) of an axis from 0 to at least ``value``."""
    step = _nice_step(max(value, 1e-9), target)
    return max(step, math.ceil(value / step - 1e-9) * step), step


def _short(sha: Any, n: int = 12) -> str:
    return sha[:n] if isinstance(sha, str) and sha else "—"


def _ranges(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """Maximal runs [start, end] (inclusive) of True values."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


# --- charts ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Column:
    """Consecutive frames drawn as one timeline column."""

    first: int  # row indices, inclusive
    last: int
    abstained: int
    health_min: float | None
    radius_max_median: float | None


def _columns(rows: Sequence[FrameRow]) -> list[_Column]:
    size = max(1, math.ceil(len(rows) / MAX_COLUMNS))
    out = []
    for start in range(0, len(rows), size):
        group = rows[start : start + size]
        healths = [r.health for r in group if r.health is not None]
        radii = [r.radius_median for r in group if r.radius_median is not None]
        out.append(
            _Column(
                first=start,
                last=start + len(group) - 1,
                abstained=sum(r.abstain for r in group),
                health_min=min(healths) if healths else None,
                radius_max_median=max(radii) if radii else None,
            )
        )
    return out


def _polylines(points: Sequence[tuple[float, float] | None]) -> list[str]:
    """SVG polylines through consecutive points; a None breaks the line (a gap in the data)."""
    out: list[str] = []
    current: list[tuple[float, float]] = []
    for point in [*points, None]:
        if point is not None:
            current.append(point)
            continue
        if len(current) == 1:  # an isolated value still shows, as a short dash
            x, y = current[0]
            current = [(x - 1.5, y), (x + 1.5, y)]
        if current:
            coords = " ".join(f"{_f(x)},{_f(y)}" for x, y in current)
            out.append(f'<polyline class="series-line" points="{coords}"/>')
        current = []
    return out


def _y_axis(top: float, height: float, hi: float, step: float, fmt: Callable[[float], str]) -> list[str]:
    parts = []
    for k in range(round(hi / step) + 1):
        value = k * step
        y = top + height - height * value / hi
        parts.append(_line("grid", _LEFT, y, _W - _RIGHT, y))
        parts.append(_text("tick", _LEFT - 8, y + 4, fmt(value), "end"))
    return parts


def _centre_note(top: float, height: float, text: str) -> str:
    return _text("note", _LEFT + _PLOT_W / 2, top + height / 2 + 4, text, "middle")


def timeline_svg(rows: Sequence[FrameRow], fps: float) -> str:
    """Two panels on one time axis, calibration health (0-1) and the median p90 radius of each frame, with
    the abstained frames shaded across both. ``rows`` must not be empty."""
    columns = _columns(rows)
    n = len(columns)
    panel_h, top1, gap = 110.0, 34.0, 44.0
    top2 = top1 + panel_h + gap
    bottom = top2 + panel_h
    frame0 = rows[0].frame
    duration = len(rows) / fps

    def x_of(i: float) -> float:
        return _LEFT + _PLOT_W * i / n

    parts = [
        f'<svg class="chart" viewBox="0 0 {_W} {_f(bottom + 40)}" role="img" aria-labelledby="timeline-t">',
        '<title id="timeline-t">Calibration health and median p90 radius per frame; abstained frames are '
        "shaded</title>",
    ]
    for a, b in _ranges([c.abstained > 0 for c in columns]):
        for top in (top1, top2):
            parts.append(_rect("band", x_of(a), top, x_of(b + 1) - x_of(a), panel_h))

    parts.append(_text("panel-title", _LEFT, top1 - 12, "Calibration health (0-1)"))
    parts += _y_axis(top1, panel_h, 1.0, 0.5, lambda v: f"{v:g}")
    health = [
        None if c.health_min is None else (x_of(i + 0.5), top1 + panel_h * (1.0 - c.health_min))
        for i, c in enumerate(columns)
    ]
    if any(p is not None for p in health):
        parts += _polylines(health)
    else:
        parts.append(
            _centre_note(top1, panel_h, "No health score in this run: calib_health is null in every frame")
        )

    parts.append(_text("panel-title", _LEFT, top2 - 12, "Median p90 radius of the frame (m)"))
    radii = [c.radius_max_median for c in columns if c.radius_max_median is not None]
    if radii:
        hi, step = _nice_ceiling(max(radii))
        parts += _y_axis(top2, panel_h, hi, step, lambda v: f"{v:g}")
        parts += _polylines(
            [
                None
                if c.radius_max_median is None
                else (x_of(i + 0.5), top2 + panel_h * (1.0 - c.radius_max_median / hi))
                for i, c in enumerate(columns)
            ]
        )
    else:
        parts.append(_line("axis", _LEFT, bottom, _W - _RIGHT, bottom))
        parts.append(_centre_note(top2, panel_h, "No published p90 radius"))

    step = _nice_step(duration, 8)
    k = 0
    while (t := k * step) <= duration + 1e-9:
        x = _LEFT + _PLOT_W * t * fps / len(rows)
        parts.append(_line("axis", x, bottom, x, bottom + 5))
        parts.append(_text("tick", x, bottom + 20, _time_label(round(t, 6), duration), "middle"))
        k += 1

    # Hover targets, one per column: the native tooltip carries the numbers (no JavaScript).
    for i, c in enumerate(columns):
        first, last = rows[c.first], rows[c.last]
        frames = f"frame {first.frame}" if c.first == c.last else f"frames {first.frame}-{last.frame}"
        when = first.timestamp or f"{(first.frame - frame0) / fps:.2f} s"
        tip = (
            f"{frames} at {when}: abstained {c.abstained}/{c.last - c.first + 1}; health "
            f"{_num(c.health_min, 3)}; median p90 radius {_num(c.radius_max_median, 2, ' m')}"
        )
        parts.append(_rect("hit", x_of(i), top1, x_of(i + 1) - x_of(i), bottom - top1, tip))
    parts.append("</svg>")
    return "\n".join(parts)


def _bar_path(x: float, w: float, y: float, base: float) -> str:
    """Column with a 4 px rounded data end, square at the baseline."""
    r = min(4.0, w / 2, base - y)
    return (
        f"M{_f(x)},{_f(base)} V{_f(y + r)} Q{_f(x)},{_f(y)} {_f(x + r)},{_f(y)} H{_f(x + w - r)} "
        f"Q{_f(x + w)},{_f(y)} {_f(x + w)},{_f(y + r)} V{_f(base)} Z"
    )


def histogram_svg(radii: Sequence[float]) -> str:
    """Distribution of the published p90 radii, with the median and the 90th percentile marked. ``radii``
    must not be empty."""
    values = np.asarray(radii, dtype=np.float64)
    p50, p90, p995, vmax = (float(v) for v in np.quantile(values, [0.5, 0.9, 0.995, 1.0]))
    # A few huge radii would squash every other value into the first bar: they get a bar of their own.
    overflow = p995 > 0 and vmax > 1.5 * p995
    hi, _ = _nice_ceiling(p995 if overflow else max(1.05 * vmax, 1e-3), 5)  # 5 %: the maximum stays inside
    counts, edges = np.histogram(np.clip(values, 0.0, hi), bins=HISTOGRAM_BINS, range=(0.0, hi))
    n_over = int(np.sum(values > hi)) if overflow else 0
    counts[-1] -= n_over  # the clipped values landed in the last bin
    slots = HISTOGRAM_BINS + (1 if n_over else 0)
    top, plot_h = 30.0, 180.0
    base = top + plot_h
    ymax, ystep = _nice_ceiling(float(max(int(counts.max()), n_over, 1)), 4)
    slot_w = _PLOT_W / slots
    bar_w = min(24.0, slot_w - 2.0)  # thin bars, a surface gap between neighbours

    def x_of(value: float) -> float:
        return _LEFT + _PLOT_W * (value / hi) * HISTOGRAM_BINS / slots

    parts = [
        f'<svg class="chart" viewBox="0 0 {_W} {_f(base + 46)}" role="img" aria-labelledby="radii-t">',
        '<title id="radii-t">Histogram of the p90 radii of every published position</title>',
    ]
    parts += _y_axis(top, plot_h, ymax, ystep, lambda v: f"{v:,.0f}")
    bars = [(i, int(c), f"{edges[i]:.2f}-{edges[i + 1]:.2f} m") for i, c in enumerate(counts)]
    if n_over:
        bars.append((HISTOGRAM_BINS, n_over, f"above {hi:g} m"))
    for i, count, label in bars:
        tip = f"{label}: {count} position{'' if count == 1 else 's'}"
        if count > 0:
            x = _LEFT + i * slot_w + (slot_w - bar_w) / 2
            path = _bar_path(x, bar_w, base - plot_h * count / ymax, base)
            parts.append(f'<path class="bar" d="{path}"><title>{_e(tip)}</title></path>')
        # a hit target as tall as the plot, so that small bars are easy to hover
        parts.append(_rect("hit", _LEFT + i * slot_w, top, slot_w, plot_h, tip))
    parts.append(_line("axis", _LEFT, base, _W - _RIGHT, base))
    over_x = _LEFT + (HISTOGRAM_BINS + 0.5) * slot_w
    x_step = _nice_step(hi, 6)
    k = 0
    while (v := k * x_step) <= hi + 1e-9:
        if not n_over or over_x - x_of(v) > 32:  # leave room for the label of the overflow bar
            parts.append(_text("tick", x_of(v), base + 18, f"{v:g}", "middle"))
        k += 1
    if n_over:
        parts.append(_text("tick", over_x, base + 18, f">{hi:g}", "middle"))
    parts.append(_text("tick", _LEFT + _PLOT_W / 2, base + 38, "p90 radius (m)", "middle"))
    markers = [(name, value) for name, value in (("median", p50), ("p90", p90)) if value <= hi]
    for _, value in markers:
        parts.append(_line("marker", x_of(value), top - 6, x_of(value), base))
    labels = [(x_of(value), f"{name} {value:.2f} m") for name, value in markers]
    if len(labels) == 2 and labels[1][0] - labels[0][0] < 120:  # too close for two labels: one for both
        labels = [(labels[1][0], f"median {p50:.2f} m · p90 {p90:.2f} m")]
    for x, label in labels:
        near_right = x > _W - _RIGHT - 180
        parts.append(
            _text(
                "marker-label",
                x - 4 if near_right else x + 4,
                top - 10,
                label,
                "end" if near_right else "start",
            )
        )
    parts.append("</svg>")
    return "\n".join(parts)


# --- sections ----------------------------------------------------------------------------------------


def _tile(label: str, value: str, note: str = "") -> str:
    extra = f'<div class="tile-note">{_e(note)}</div>' if note else ""
    return (
        f'<div class="tile"><div class="tile-label">{_e(label)}</div>'
        f'<div class="tile-value">{_e(value)}</div>{extra}</div>'
    )


def _kv_table(rows: Iterable[tuple[str, str]], caption: str) -> str:
    body = "".join(f'<tr><th scope="row">{_e(k)}</th><td>{_e(v)}</td></tr>' for k, v in rows)
    return f'<table class="kv"><caption>{_e(caption)}</caption><tbody>{body}</tbody></table>'


def _status(ok: bool, label: str) -> str:
    """Status badge: an icon and a label, never colour alone."""
    icon, cls = ("✓", "ok") if ok else ("✗", "fail")
    return (
        f'<span class="status status-{cls}"><span class="status-icon" aria-hidden="true">{icon}</span> '
        f"{_e(label)}</span>"
    )


def _section(key: str, title: str, body: Iterable[str]) -> str:
    return "\n".join(
        [
            f'<section id="{key}" aria-labelledby="{key}-h">',
            f'<h2 id="{key}-h">{_e(title)}</h2>',
            *body,
            "</section>",
        ]
    )


def _conformal_text(conformal: Any) -> str:
    if conformal is None:
        return "none"
    if isinstance(conformal, Mapping):
        scalars = [
            f"{k}: {_short(v) if k == 'sha256' else v}"
            for k, v in conformal.items()
            if not isinstance(v, Mapping | list)
        ]
        return ", ".join(scalars) or "yes"
    return str(conformal)


def _summary_section(run: RunFiles) -> str:
    s = run.summary
    meta = run.meta
    live = s.n_frames - s.n_abstained
    tiles = [
        _tile("Export frames", f"{s.n_frames:,}", f"at {run.export_fps:g} fps"),
        _tile("Published frames", _pct(live, s.n_frames), f"{live:,} of {s.n_frames:,} do not abstain"),
        _tile("Abstentions", f"{s.n_abstained:,}", f"{_pct(s.n_abstained, s.n_frames)} of the frames"),
        _tile("Tracks", f"{len(s.track_ids):,}"),
        _tile(
            "Located players",
            _pct(s.n_located, s.rows_in_live_frames),
            f"{s.n_located:,} of {s.rows_in_live_frames:,} rows of live frames",
        ),
    ]
    native = _int(_get(run.metrics, "frames", "native"))
    if native is not None:
        native_fps = _finite(_get(meta, "native_fps"))
        rate = "" if native_fps is None else f"at {native_fps:g} fps"
        tiles.insert(1, _tile("Native frames", f"{native:,}", rate))
    image = _get(meta, "image_size")
    pitch = _get(meta, "pitch")
    provenance = [
        ("Clip", str(_get(meta, "clip_id") or _get(run.metrics, "clip_id") or "—")),
        ("predfoot version", str(_get(meta, "predfoot_version") or "—")),
        ("Git commit", _short(_get(meta, "git_sha"))),
        ("Config hash", _short(_get(meta, "config_hash"))),
        ("Stages version", str(_get(meta, "config", "stages") or "—")),
        ("Input", f"{_get(meta, 'input', 'file') or '—'} (sha256 {_short(_get(meta, 'input', 'sha256'))})"),
        ("Image size", f"{image[0]} × {image[1]} px" if isinstance(image, list) and len(image) == 2 else "—"),
        (
            "Pitch",
            f"{_get(pitch, 'length')} × {_get(pitch, 'width')} m" if isinstance(pitch, Mapping) else "—",
        ),
        ("Time resampling", str(_get(meta, "time_resampling") or "—")),
    ]
    stages = _get(meta, "stages")
    stage_rows = [(str(k), str(v)) for k, v in stages.items()] if isinstance(stages, Mapping) else []
    if meta is not None:
        stage_rows.append(("conformal", _conformal_text(meta.get("conformal"))))
    body = [
        f'<div class="tiles">{"".join(tiles)}</div>',
        '<div class="two-col">',
        _kv_table(provenance, "Provenance (meta.json)"),
        _kv_table(stage_rows, "Stages (meta.json)") if stage_rows else "",
        "</div>",
    ]
    if run.problems:
        items = "".join(f"<li><code>{_e(k)}</code>: {_e(v)}</li>" for k, v in sorted(run.problems.items()))
        body.append(f'<p class="note">Files not used:</p><ul class="problems">{items}</ul>')
    return _section("summary", "Run summary", body)


def consistency_checks(run: RunFiles) -> list[tuple[bool, str]]:
    """Checks that ``metrics.json`` and ``match.json`` describe the tracking file next to them."""
    s = run.summary
    checks: list[tuple[bool, str]] = []
    expected = [
        (("frames", "exported"), s.n_frames, "export frames"),
        (("frames", "abstained"), s.n_abstained, "abstained frames"),
        (("players", "positions"), s.n_located, "published positions"),
        (("players", "tracks"), len(s.track_ids), "tracks"),
        (("players", "unlocated_in_live_frames"), s.rows_in_live_frames - s.n_located, "unlocated rows"),
    ]
    for keys, actual, what in expected:
        recorded = _int(_get(run.metrics, *keys))
        if recorded is not None:
            name = "metrics.json " + ".".join(keys)
            checks.append((recorded == actual, f"{name} = {recorded}; tracking.jsonl has {actual} {what}"))
    players = _get(run.match, "players")
    if isinstance(players, list):
        listed = {_int(p.get("id")) for p in players if isinstance(p, Mapping)}
        missing = len(s.track_ids - {i for i in listed if i is not None})
        checks.append(
            (missing == 0, f"match.json lists {len(listed)} players; tracks missing from it: {missing}")
        )
    if s.unreadable:
        checks.append((False, f"{s.unreadable} line(s) of tracking.jsonl are not JSON objects"))
    return checks


def _validation_section(run: RunFiles) -> str:
    body = []
    report = run.validation
    if report is None:
        body.append(f"<p>{_status(False, 'Validation could not run')}: {_e(run.validation_error or '')}</p>")
    elif report.ok:
        with_match = "with match.json" if run.match is not None else "without match.json"
        body.append(
            f"<p>{_status(True, 'Valid')}: the {report.n_records:,} records of tracking.jsonl pass the frame "
            f"schema 1.0 and the semantic rules ({_e(with_match)}, timestamps at {run.export_fps:g} fps).</p>"
        )
    else:
        more = "+" if report.truncated else ""
        body.append(
            f"<p>{_status(False, 'Failed')}: {len(report.issues)}{more} problem(s) in "
            f"{report.n_records:,} records; the first ones:</p>"
        )
        body.append(f'<ul class="issues">{"".join(f"<li>{_e(i)}</li>" for i in report.issues)}</ul>')
    checks = consistency_checks(run)
    if checks:
        items = "".join(
            f"<li>{_status(ok, 'ok' if ok else 'mismatch')} {_e(text)}</li>" for ok, text in checks
        )
        body.append(f'<h3>Consistency of the run files</h3><ul class="checks">{items}</ul>')
    return _section("validation", "Validation", body)


def _timeline_section(run: RunFiles) -> str:
    rows = run.summary.rows
    if not rows:
        return _section("timeline", "Timeline", ['<p class="note">tracking.jsonl has no records.</p>'])
    size = math.ceil(len(rows) / MAX_COLUMNS)
    grouping = (
        "One column per export frame."
        if size == 1
        else f"Each column groups {size} export frames: their lowest health, their largest median radius."
    )
    body = [
        '<p class="legend"><span class="swatch band-swatch" aria-hidden="true"></span> abstained (no '
        f"position published) · {_e(grouping)} Hover a column for its numbers.</p>",
        f'<div class="chart-wrap">{timeline_svg(rows, run.export_fps)}</div>',
    ]
    spans = _ranges([r.abstain for r in rows])
    if not spans:
        body.append('<p class="note">No frame abstains.</p>')
        return _section("timeline", "Timeline", body)
    items = []
    for a, b in spans[:MAX_SPANS]:
        ra, rb = rows[a], rows[b]
        if a == b:
            label = f"frame {ra.frame}" + (f" ({ra.timestamp})" if ra.timestamp else "")
        else:
            label = f"frames {ra.frame}-{rb.frame}"
            if ra.timestamp and rb.timestamp:
                label += f" ({ra.timestamp}-{rb.timestamp})"
        items.append(f"<li>{_e(label)}: {b - a + 1} frame{'s' if b > a else ''}</li>")
    if len(spans) > MAX_SPANS:
        items.append(f"<li>… and {len(spans) - MAX_SPANS} more spans</li>")
    body.append(f'<h3>Abstained spans ({len(spans)})</h3><ul class="spans">{"".join(items)}</ul>')
    return _section("timeline", "Timeline", body)


def _radii_section(run: RunFiles) -> str:
    radii = run.summary.radii
    if not radii:
        note = (
            '<p class="note">No published p90 radius: every frame abstains, or no position carries an '
            "uncertainty.</p>"
        )
        return _section("radii", "Distribution of p90 radii", [note])
    q = np.quantile(np.asarray(radii), [0.1, 0.5, 0.9, 0.99, 1.0])
    stats = [
        ("Positions", f"{len(radii):,}"),
        ("Mean", _num(float(np.mean(radii)), 3, " m")),
        *((k, _num(float(v), 3, " m")) for k, v in zip(("p10", "p50", "p90", "p99", "Max"), q, strict=True)),
    ]
    head = "".join(f'<th scope="col" class="num">{_e(k)}</th>' for k, _ in stats)
    cells = "".join(f'<td class="num">{_e(v)}</td>' for _, v in stats)
    table = (
        '<div class="table-wrap"><table><caption>p90 radius of the published positions</caption>'
        f"<thead><tr>{head}</tr></thead><tbody><tr>{cells}</tr></tbody></table></div>"
    )
    chart = f'<div class="chart-wrap">{histogram_svg(radii)}</div>'
    return _section("radii", "Distribution of p90 radii", [chart, table])


def worst_frames(rows: Sequence[FrameRow], n: int = WORST_FRAMES) -> list[FrameRow]:
    """Live frames that list players, worst first: those where no listed player could be located, then the
    largest median p90 radius (ties: the largest radius, then the earliest frame)."""

    def key(r: FrameRow) -> tuple[int, float, float, int]:
        if r.radius_median is None:
            return (0 if r.n_located == 0 else 2, 0.0, 0.0, r.frame)
        return (1, -r.radius_median, -(r.radius_max or 0.0), r.frame)

    return sorted((r for r in rows if not r.abstain and r.n_players > 0), key=key)[:n]


_WORST_COLUMNS = (
    ("Frame", True),
    ("Time", False),
    ("Health", True),
    ("Located / listed", True),
    ("Median p90 (m)", True),
    ("Max p90 (m)", True),
    ("Track with max", True),
)


def _worst_section(run: RunFiles) -> str:
    rows = worst_frames(run.summary.rows)
    if not rows:
        note = '<p class="note">No live frame lists a player: nothing to rank.</p>'
        return _section("worst-frames", "Worst frames", [note])

    def cell(text: str, numeric: bool, tag: str = "td") -> str:
        attrs = (' scope="col"' if tag == "th" else "") + (' class="num"' if numeric else "")
        return f"<{tag}{attrs}>{_e(text)}</{tag}>"

    head = "".join(cell(name, numeric, "th") for name, numeric in _WORST_COLUMNS)
    body = []
    for r in rows:
        values = (
            str(r.frame),
            r.timestamp or "—",
            _num(r.health, 3),
            f"{r.n_located} / {r.n_players}",
            _num(r.radius_median, 2),
            _num(r.radius_max, 2),
            "—" if r.worst_track is None else str(r.worst_track),
        )
        body.append(
            "<tr>"
            + "".join(cell(v, num) for v, (_, num) in zip(values, _WORST_COLUMNS, strict=True))
            + "</tr>"
        )
    intro = (
        "<p>Frames that publish positions, ranked by the median p90 radius of their players; frames where no "
        "listed player could be located come first. Abstained frames are listed under the timeline.</p>"
    )
    table = (
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )
    return _section("worst-frames", "Worst frames", [intro, table])


def _minimap_section(run: RunFiles) -> str:
    if run.minimap_png is None:
        return ""
    data = base64.b64encode(run.minimap_png).decode("ascii")
    return _section(
        "minimap",
        "Minimap",
        [
            "<p>minimap.png of the run: evenly spaced frames with each player's 90 % ellipse; an abstained "
            "frame is greyed out.</p>",
            '<img class="minimap" alt="Minimap: evenly spaced frames with the 90 % ellipses of the players" '
            f'src="data:image/png;base64,{data}">',
        ],
    )


DEFINITIONS: tuple[tuple[str, str], ...] = (
    (
        "p90_radius_m",
        "Radius in metres of the smallest circle around the published position that contains its 90 % "
        "region, the ellipse {p : (p − x̂)ᵀ Σ⁻¹ (p − x̂) ≤ 4.605} of pos_cov_m2. Smaller means more "
        "certain.",
    ),
    (
        "pos_cov_m2",
        "Calibrated 2 × 2 covariance of the position in m², in the internal pitch frame (origin at the "
        "centre spot, x towards the right-hand goal of the main camera, y towards the far touchline). Its "
        "χ²₂(0.90) = 4.605 ellipse is the 90 % prediction region.",
    ),
    (
        "calib_health",
        "Score of the frame's calibration in [0, 1], 1 meaning healthy; null when the run has no health "
        "model.",
    ),
    (
        "abstain",
        "True when the frame publishes no position because its calibration failed or was judged unhealthy: "
        "every x, y, p90_radius_m and pos_cov_m2 of the frame is then null.",
    ),
    (
        "image_corners_projection",
        "Ground footprint of the image: the four image corners projected on the pitch, in metres, the top "
        "ones clipped 5 m beyond the far touchline, as in SkillCorner open data. Null when the frame "
        "abstains.",
    ),
    ("is_detected", "Whether the player was seen in the frame rather than filled in."),
    ("Published frames", "Share of the export frames that do not abstain."),
    (
        "Located players",
        "Share of the player rows of non-abstained frames that have a position. A listed player has none "
        "when the foot point lies above the horizon or outside the region where the camera model is valid.",
    ),
    (
        "Median p90 radius of the frame",
        "Median of the p90 radii of the frame's located players. Every player of a frame shares the same "
        "camera, so it follows the quality of the frame's calibration.",
    ),
    (
        "Validation",
        "The frame JSON Schema 1.0 plus the rules a schema cannot express: symmetric positive semi-definite "
        "covariances, image corners clipped and turning counter-clockwise, unique player ids listed in "
        "match.json, increasing frames and timestamps at the export rate, no NaN.",
    ),
)


def _definitions_section() -> str:
    items = "".join(f"<dt>{_e(term)}</dt><dd>{_e(text)}</dd>" for term, text in DEFINITIONS)
    return _section("definitions", "Definitions", [f'<dl class="defs">{items}</dl>'])


# Light and dark themes from one set of roles; the series colour is slot 1 of the validated default palette.
_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--series:#2a78d6;--band:#ecebe6;--good:#0ca30c;
--critical:#d03b3b;--border:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink-2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--series:#3987e5;--band:#2c2c2a;
--border:rgba(255,255,255,.10)}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 48px}
header h1{font-size:24px;margin:0 0 4px}
header p{margin:0;color:var(--ink-2)}
section{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;
margin-top:16px}
h2{font-size:18px;margin:0 0 12px}
h3{font-size:15px;margin:16px 0 8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.tile{border:1px solid var(--border);border-radius:10px;padding:12px}
.tile-label{color:var(--ink-2);font-size:13px}
.tile-value{font-size:24px;font-weight:600}
.tile-note{color:var(--muted);font-size:12px}
.two-col{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
caption{text-align:left;color:var(--ink-2);font-size:13px;padding-bottom:6px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
.kv th{color:var(--ink-2);font-weight:500;width:38%}
.kv td{overflow-wrap:anywhere}
th.num,td.num{text-align:right}
.table-wrap{overflow-x:auto}
.status{font-weight:600;white-space:nowrap}
.status-icon{display:inline-block;width:1.3em;height:1.3em;line-height:1.3em;text-align:center;
border-radius:50%;color:#fff;font-size:12px}
.status-ok .status-icon{background:var(--good)}
.status-fail .status-icon{background:var(--critical)}
ul.issues li,ul.checks li,ul.problems li{overflow-wrap:anywhere}
ul.checks{list-style:none;padding-left:0}
.note,.legend{color:var(--ink-2)}
.legend{font-size:13px}
.swatch{display:inline-block;width:14px;height:10px;border-radius:2px}
.band-swatch{background:var(--band);border:1px solid var(--axis)}
svg.chart{width:100%;height:auto;display:block}
.chart-wrap{overflow-x:auto}
.chart-wrap svg.chart{min-width:640px}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .tick,svg .note{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
svg .panel-title{fill:var(--ink-2);font-size:13px;font-weight:600}
svg .series-line{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .bar{fill:var(--series)}
svg .band{fill:var(--band)}
svg .marker{stroke:var(--ink-2);stroke-width:1}
svg .marker-label{fill:var(--ink-2);font-size:12px}
svg .hit{fill:transparent}
svg .hit:hover{fill:var(--ink);fill-opacity:.06}
img.minimap{max-width:100%;height:auto;border-radius:8px}
dl.defs dt{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;font-weight:600;
margin-top:10px}
dl.defs dd{margin:2px 0 0;color:var(--ink-2)}
footer{color:var(--muted);font-size:12px;margin-top:16px}
"""


def render_qa_report(run: RunFiles) -> str:
    """The report as an HTML document."""
    clip = _get(run.meta, "clip_id") or _get(run.metrics, "clip_id") or run.out_dir.name
    title = _e(f"QA report · {clip}")
    sections = [
        _summary_section(run),
        _validation_section(run),
        _timeline_section(run),
        _radii_section(run),
        _worst_section(run),
        _minimap_section(run),
        _definitions_section(),
    ]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f'<meta http-equiv="Content-Security-Policy" content="{_e(CONTENT_SECURITY_POLICY)}">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{title}</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            f"<h1>{title}</h1>",
            "<p>Outputs of predfoot run: tracking.jsonl, match.json, meta.json, metrics.json.</p>",
            "</header>",
            *[s for s in sections if s],
            f"<footer>Generated by predfoot {_e(__version__)} from the files of the run. Self-contained: no "
            "script, no external resource.</footer>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def write_qa_report(out_dir: Path, path: Path | None = None) -> Path:
    """Build the report of the run in ``out_dir`` and write it atomically to ``path`` (by default
    ``out_dir/qa_report.html``). Returns the path written."""
    run = read_run(out_dir)
    target = Path(out_dir) / REPORT_NAME if path is None else Path(path)
    atomic_write_text(target, render_qa_report(run))
    return target
