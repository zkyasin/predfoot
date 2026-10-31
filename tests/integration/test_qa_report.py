"""QA report (T5.2): written by ``predfoot run``, rebuilt by ``predfoot qa``, self-contained and escaped."""

import json
from collections import Counter
from html.parser import HTMLParser

import pytest
from typer.testing import CliRunner

from predfoot.cli import app
from predfoot.pipeline.run import RunConfig, run_clip
from predfoot.viz.qa_report import REPORT_NAME, read_run, worst_frames
from tests.cli_output import PLAIN_ENV, plain
from tests.fixtures import FIXTURES

CLIP = FIXTURES / "synthetic_clip.json"
SECTIONS = {"summary", "validation", "timeline", "radii", "worst-frames", "definitions"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class StrictParser(HTMLParser):
    """html.parser with the checks it does not make itself: every element closed, in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: Counter[str] = Counter()
        self.attrs: list[tuple[str, str, str | None]] = []
        self.text: list[str] = []

    def _record(self, tag, attrs):
        self.tags[tag] += 1
        self.attrs += [(tag, k, v) for k, v in attrs]

    def handle_starttag(self, tag, attrs):
        self._record(tag, attrs)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):  # <line .../> in SVG
        self._record(tag, attrs)

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(f"</{tag}> closes {self.stack[-3:]}")

    def handle_data(self, data):
        self.text.append(data)

    @property
    def plain_text(self) -> str:
        """Text content with whitespace runs collapsed, as a reader sees it."""
        return " ".join("".join(self.text).split())

    @property
    def ids(self) -> list[str]:
        return [v for _, k, v in self.attrs if k == "id" and v is not None]


def parse(html: str) -> StrictParser:
    parser = StrictParser()
    parser.feed(html)
    parser.close()
    assert parser.errors == [], parser.errors[:5]
    assert parser.stack == [], parser.stack
    assert len(parser.ids) == len(set(parser.ids)), "duplicate ids"
    return parser


def assert_self_contained(html: str) -> None:
    lowered = html.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "url(" not in lowered  # no image, stylesheet or font fetched from the CSS
    assert "@import" not in lowered
    parser = parse(html)
    for tag in ("script", "link", "iframe", "object", "embed", "base", "form"):
        assert parser.tags[tag] == 0, tag
    for tag, name, value in parser.attrs:
        assert not name.startswith("on"), f"event handler {name} on <{tag}>"
        if name in ("src", "href", "xlink:href", "action"):
            assert (value or "").startswith("data:image/png;base64,"), (tag, name, (value or "")[:40])


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("qa")
    run_clip(CLIP, out, RunConfig(block_size=7))
    return out


def test_run_writes_a_report_with_every_section(run_dir):
    html = (run_dir / REPORT_NAME).read_text(encoding="utf-8")
    parser = parse(html)
    assert set(parser.ids) >= SECTIONS | {"minimap"}
    text = parser.plain_text
    assert "QA report · synthetic-broadcast-001" in text
    assert "Valid: the 30 records of tracking.jsonl pass" in text
    assert "frames 16-19 (00:00:01.60-00:00:01.90): 4 frames" in text  # native frames 40-49 failed
    assert "86.7 %" in text  # 26 of 30 frames publish positions
    assert "No health score in this run" in text  # the stub stages have no health model
    assert "mismatch" not in text  # metrics.json, match.json and tracking.jsonl agree
    assert parser.tags["svg"] == 2  # timeline and histogram
    assert parser.tags["img"] == 1  # the embedded minimap


def test_report_is_self_contained(run_dir):
    assert_self_contained((run_dir / REPORT_NAME).read_text(encoding="utf-8"))


def test_qa_command_rebuilds_the_same_bytes(run_dir, tmp_path):
    target = tmp_path / "again.html"
    result = CliRunner(env=PLAIN_ENV).invoke(app, ["qa", str(run_dir), "--output", str(target)])
    assert result.exit_code == 0, result.output
    assert "30 frames, 4 abstained; outputs valid" in plain(result)
    assert target.read_bytes() == (run_dir / REPORT_NAME).read_bytes()


def test_qa_command_needs_a_run_folder(tmp_path):
    result = CliRunner(env=PLAIN_ENV).invoke(app, ["qa", str(tmp_path)])
    assert result.exit_code == 2
    assert "tracking.jsonl not found" in plain(result)


def _clip_dict():
    return json.loads(CLIP.read_text(encoding="utf-8"))


def test_report_when_every_frame_abstains(tmp_path):
    clip = _clip_dict()
    for frame in clip["frames"]:
        frame["homography"] = None
    path = tmp_path / "abstains.json"
    path.write_text(json.dumps(clip), encoding="utf-8")
    out = tmp_path / "out"
    result = run_clip(path, out)
    assert result.n_abstained == result.n_export == 30
    html = (out / REPORT_NAME).read_text(encoding="utf-8")
    assert_self_contained(html)
    parser = parse(html)
    assert set(parser.ids) >= SECTIONS
    text = parser.plain_text
    assert "Valid: the 30 records" in text
    assert "0.0 %" in text  # published frames
    assert "frames 0-29 (00:00:00.00-00:00:02.90): 30 frames" in text
    assert "No published p90 radius: every frame abstains" in text
    assert "No live frame lists a player" in text
    assert parser.tags["svg"] == 1  # the timeline; no histogram without radii


MARKUP = '<script>alert("x")</script> & <b>bold</b>'


def test_strings_from_the_run_files_are_escaped(tmp_path):
    clip = _clip_dict()
    clip["clip_id"] = MARKUP
    path = tmp_path / "markup.json"
    path.write_text(json.dumps(clip), encoding="utf-8")
    run_clip(path, tmp_path / "out")
    html = (tmp_path / "out" / REPORT_NAME).read_text(encoding="utf-8")
    assert "<script" not in html
    assert "<b>" not in html
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &lt;b&gt;bold&lt;/b&gt;" in html
    text = parse(html).plain_text
    assert f"QA report · {MARKUP}" in text  # shown as text, not interpreted


def test_a_broken_tracking_file_is_reported_not_fatal(run_dir, tmp_path):
    for name in ("match.json", "meta.json", "metrics.json"):
        (tmp_path / name).write_bytes((run_dir / name).read_bytes())
    lines = (run_dir / "tracking.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[3])
    record["player_data"][0]["x"] = '<img src=x onerror="alert(1)">'
    lines[3] = json.dumps(record)
    lines.insert(5, "{not json")
    (tmp_path / "tracking.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = CliRunner(env=PLAIN_ENV).invoke(app, ["qa", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "validation FAILED" in plain(result)
    html = (tmp_path / REPORT_NAME).read_text(encoding="utf-8")
    assert_self_contained(html)  # the injected markup stays text
    text = parse(html).plain_text
    assert "Failed:" in text
    assert "line 6: invalid JSON" in text
    assert 'onerror="alert(1)">' in text  # the validator's message about record 4, escaped
    assert "1 line(s) of tracking.jsonl are not JSON objects" in text
    assert "mismatch" in text  # metrics.json counts 343 positions; the edited file has 342


def test_missing_side_files_are_listed(run_dir, tmp_path):
    (tmp_path / "tracking.jsonl").write_bytes((run_dir / "tracking.jsonl").read_bytes())
    (tmp_path / "meta.json").write_text("[1, 2]", encoding="utf-8")
    run = read_run(tmp_path)
    assert run.problems == {
        "match.json": "missing",
        "meta.json": "not a JSON object",
        "metrics.json": "missing",
    }
    assert run.validation is not None
    assert run.validation.ok  # validated without match.json, at 10 fps


def test_report_of_the_real_stages_shows_their_conformal_model(tmp_path):
    from tests.integration.test_golden_real import GOLDEN, golden

    golden.run_golden(GOLDEN, tmp_path)
    text = parse((tmp_path / REPORT_NAME).read_text(encoding="utf-8")).plain_text
    assert "kind: mondrian, sha256: 892b11963c54, alpha: 0.1, pooled: 6.5" in text
    assert "Mondrian conformal quantile of each player's bin" in text
    assert "frame 3 (00:00:00.30): 1 frame" in text


def test_worst_frames_come_first(run_dir):
    run = read_run(run_dir)
    worst = worst_frames(run.summary.rows, n=5)
    assert all(not r.abstain for r in worst)
    medians = [r.radius_median for r in worst]
    assert medians == sorted(medians, reverse=True)
