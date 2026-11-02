"""scripts/build_site.py and scripts/deploy_space.py without network or a Hugging Face account."""

import importlib.util
import json
import re
import sys

import pytest
import yaml
from typer.testing import CliRunner

from predfoot.paths import REPO_ROOT


def _load(name):
    key = f"scripts_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


build_site = _load("build_site")
deploy = _load("deploy_space")

FONTS = "https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next&display=swap"
PAGE = f"""<title>Demo</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
@import url("{FONTS}");
:root {{ --paper: #fff; }}
body {{ background: var(--paper); }}
</style>
<main id="top">
  <video poster="assets/clip.jpg" autoplay muted loop playsinline>
    <source src="assets/clip.mp4" type="video/mp4">
  </video>
  <a href="https://github.com/zkyasin/predfoot" target="_blank" rel="noopener">Code</a> <a href="#top">Top</a>
</main>
"""


def media_dir(tmp_path, sizes=None):
    """A media folder: the two files PAGE uses, plus one it does not."""
    folder = tmp_path / "media"
    folder.mkdir()
    for name, size in (sizes or {"clip.jpg": 10, "clip.mp4": 20, "unused.mp4": 30}).items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"x" * size)
    return folder


# The wrapper


def test_wrap_adds_the_artifact_skeleton_and_moves_the_head_elements():
    html = build_site.wrap(PAGE)
    assert html.startswith("<!doctype html>\n")
    assert '<html lang="en">' in html
    head, body = html.split("</head>")
    assert '<meta charset="utf-8">' in head
    assert re.search(r'<meta name="viewport" content="[^"]*viewport-fit=cover', head)
    for rule in (
        "color-scheme: light",
        "padding-top: env(safe-area-inset-top)",
        "padding-bottom: env(safe-area-inset-bottom)",
        "margin: 0",
        "14px",
        "img { max-width: 100%; }",
        "[hidden] { display: none !important; }",
    ):
        assert rule in head
    # the page's leading title, meta and style go to the head, after the reset; the rest is the body
    assert head.index("[hidden]") < head.index("<title>Demo</title>") < head.index("--paper")
    assert '<meta name="claude:viewport-fit-sides" content="cover">' in head
    assert body.strip().startswith('<body>\n<main id="top">')
    assert body.rstrip().endswith("</body>\n</html>")
    assert html.count("<title>") == 1


@pytest.mark.parametrize("tag", ["<!doctype html>", "<html lang='en'>", "<head>", "<body class='x'>"])
def test_wrap_refuses_a_page_with_document_tags(tag):
    with pytest.raises(build_site.SiteError, match="page content"):
        build_site.wrap(tag + PAGE)


def test_wrap_keeps_a_header_element_in_the_body():
    html = build_site.wrap("<title>x</title>\n<header>Band</header>")
    assert "<header>Band</header>" in html.split("<body>")[1]


# References


def test_scan_finds_every_kind_of_reference():
    page = """<img src="assets/a.png" srcset="assets/a.png 1x, assets/a@2x.png 2x">
<video poster="assets/p.jpg"><source src="assets/v.mp4"></video>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=X">
<div style="background: url('assets/bg.png')"></div>
<style>@import "https://fonts.googleapis.com/css2?family=Y"; .x { background: url(assets/t.svg); }</style>
<svg><use href="#icon"/></svg>
<a href="https://example.org">out</a> <a href="#icon">in</a>"""
    refs, ids = build_site.scan(page)
    loads = {r.url for r in refs if r.load}
    links = {r.url for r in refs if not r.load}
    assert loads == {
        "assets/a.png",
        "assets/a@2x.png",
        "assets/p.jpg",
        "assets/v.mp4",
        "https://fonts.googleapis.com/css2?family=X",
        "assets/bg.png",
        "https://fonts.googleapis.com/css2?family=Y",
        "assets/t.svg",
        "#icon",
    }
    assert links == {"https://example.org", "#icon"}
    assert ids == set()


def test_check_references_maps_media_and_ignores_links(tmp_path):
    media = build_site.check_references(PAGE, media_dir(tmp_path))
    folder = tmp_path / "media"
    assert media == {"assets/clip.jpg": folder / "clip.jpg", "assets/clip.mp4": folder / "clip.mp4"}


@pytest.mark.parametrize(
    ("snippet", "message"),
    [
        ('<video src="assets/missing.mp4"></video>', "missing"),
        ('<img src="photo.jpg">', "under assets/"),
        ('<img src="../assets/clip.jpg">', "under assets/"),
        ('<img src="/assets/clip.jpg">', "under assets/"),
        ('<script src="https://cdn.example.com/lib.js"></script>', "only Google Fonts"),
        ('<img src="//images.example.com/x.png">', "only Google Fonts"),
        ("<style>.x { background: url(https://example.com/x.png); }</style>", "only Google Fonts"),
        ('<a href="#nowhere">jump</a>', "no element on the page has this id"),
        ('<a href="javascript:alert(1)">x</a>', "unsupported URL scheme"),
        ('<a href="docs/technical.md">overview</a>', "under assets/"),
        ('<a href="https://example.org">out</a>', "must open a new tab"),
        ('<a href="https://example.org" target="_self">out</a>', "must open a new tab"),
        ('<svg><a href="//example.org/x"><text>x</text></a></svg>', "must open a new tab"),
    ],
)
def test_check_references_refuses_what_breaks_on_the_space(tmp_path, snippet, message):
    with pytest.raises(build_site.SiteError, match=message):
        build_site.check_references(PAGE + snippet, media_dir(tmp_path))


def test_check_references_lists_every_problem_at_once(tmp_path):
    page = '<img src="assets/one.png"><img src="assets/two.png"><a href="#gone">x</a>'
    with pytest.raises(build_site.SiteError) as caught:
        build_site.check_references(page, media_dir(tmp_path))
    assert str(caught.value).count("\n") == 2


def test_scan_marks_the_links_that_open_a_new_tab():
    refs, _ = build_site.scan(
        '<a href="https://a.example" target="_blank">a</a><a href="https://b.example">b</a>'
        '<img src="assets/clip.jpg" target="_blank">'
    )
    assert {r.url: r.new_tab for r in refs} == {"https://a.example": True, "https://b.example": False,
                                                 "assets/clip.jpg": False}  # fmt: skip


def test_check_references_accepts_data_uris_mail_links_and_queries(tmp_path):
    page = (
        '<img src="data:image/png;base64,AAAA"><a href="mailto:x@example.org">mail</a>'
        '<img src="assets/clip.jpg?v=2">'
    )
    media = build_site.check_references(page, media_dir(tmp_path))
    assert media == {"assets/clip.jpg": tmp_path / "media" / "clip.jpg"}


# Size budget and output


def test_plan_counts_the_page_and_its_media_against_the_budget(tmp_path):
    media = media_dir(tmp_path, {"clip.jpg": 1000, "clip.mp4": 2000})
    planned = build_site.plan(PAGE, media, max_bytes=10_000)
    assert planned.total_bytes == len(planned.index_html.encode()) + 3000
    exact = build_site.plan(PAGE, media, max_bytes=planned.total_bytes)
    assert exact.total_bytes == planned.total_bytes
    with pytest.raises(build_site.SiteError, match="budget"):
        build_site.plan(PAGE, media, max_bytes=planned.total_bytes - 1)


def test_default_budget_is_four_megabytes():
    assert build_site.MAX_BYTES == 4_000_000


def test_write_copies_only_referenced_media_and_replaces_its_own_build(tmp_path):
    out = tmp_path / "dist"
    build_site.write(build_site.plan(PAGE, media_dir(tmp_path)), out)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "README.md",
        "assets/clip.jpg",
        "assets/clip.mp4",
        "index.html",
    ]
    (out / "assets" / "stale.mp4").write_bytes(b"old")
    build_site.write(build_site.plan(PAGE.replace("assets/clip.jpg", ""), tmp_path / "media"), out)
    assert not (out / "assets" / "stale.mp4").exists()
    assert not (out / "assets" / "clip.jpg").exists()


def test_write_refuses_a_folder_it_did_not_write(tmp_path):
    out = tmp_path / "dist"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(build_site.SiteError, match="refusing"):
        build_site.write(build_site.plan(PAGE, media_dir(tmp_path)), out)
    assert (out / "notes.txt").read_text() == "keep me"


def test_space_readme_front_matter():
    text = build_site.space_readme()
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---\n")[1])
    assert front["title"] == "predfoot"
    assert front["sdk"] == "static"
    assert front["app_file"] == "index.html"
    assert front["license"] == "apache-2.0"
    assert front["pinned"] is False
    assert front["colorFrom"] in {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
    assert front["colorTo"] in {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
    assert front["emoji"]
    assert 0 < len(front["short_description"]) < 60
    assert "not affiliated with SkillCorner" in text
    # ADR-0002: the footage credit beside the downloadable clips says how the frames were changed
    assert "SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2), Scott et al." in text
    assert all(change in text for change in ("reprojected", "cropped", "downscaled", "overlaid"))


def test_cli_builds_and_fails_with_a_readable_message(tmp_path):
    page = tmp_path / "page.html"
    page.write_text(PAGE)
    media = media_dir(tmp_path)
    runner = CliRunner()
    ok = runner.invoke(
        build_site.app, ["--page", str(page), "--media", str(media), "--out", str(tmp_path / "dist")]
    )
    assert ok.exit_code == 0, ok.output
    assert (tmp_path / "dist" / "index.html").is_file()
    (media / "clip.mp4").unlink()
    failed = runner.invoke(
        build_site.app, ["--page", str(page), "--media", str(media), "--out", str(tmp_path / "dist")]
    )
    assert failed.exit_code == 1
    assert "clip.mp4" in failed.output


# The real page


SHOWCASE_MEDIA = {
    "hero.mp4",
    "hero.jpg",
    "smoothing.mp4",
    "smoothing.jpg",
    "smoothing-tall.mp4",
    "smoothing-tall.jpg",
    "vendor/d3.min.js",
    "vendor/motion.js",
    "vendor/lenis.min.js",
    "vendor/LICENSES.txt",
}


def test_showcase_page_builds_with_its_media(tmp_path):
    page = build_site.PAGE.read_text(encoding="utf-8")
    assert page.startswith("<title>predfoot</title>\n")
    assert page.splitlines()[1] == '<meta name="claude:viewport-fit-sides" content="cover">'
    media = media_dir(tmp_path, dict.fromkeys(SHOWCASE_MEDIA, 100))
    planned = build_site.plan(page, media)
    assert {name.removeprefix("assets/") for name in planned.media} == SHOWCASE_MEDIA


#: Every showcase file stays below this, under the 1 MB that pre-commit refuses.
MAX_ASSET_BYTES = 900_000


def test_showcase_page_builds_with_the_committed_media():
    planned = build_site.plan(build_site.PAGE.read_text(encoding="utf-8"), build_site.MEDIA)
    assert {name.removeprefix("assets/") for name in planned.media} == SHOWCASE_MEDIA
    assert max(src.stat().st_size for src in planned.media.values()) < MAX_ASSET_BYTES
    assert planned.total_bytes <= build_site.MAX_BYTES


def test_readme_images_are_committed_and_light():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    images = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', readme) + re.findall(r"!\[[^\]]*\]\(([^)\s]+)", readme)
    local = [image for image in images if "://" not in image]
    assert {"assets/showcase/hero.jpg", "assets/showcase/smoothing.gif"} <= set(local)
    for image in local:
        assert (REPO_ROOT / image).is_file(), image
        assert (REPO_ROOT / image).stat().st_size < MAX_ASSET_BYTES, image


def test_showcase_page_keeps_its_contract():
    page = build_site.PAGE.read_text(encoding="utf-8")
    css = page[page.index("<style>") : page.index("</style>")]
    # the side gutter clears a phone's system bar, as the viewport-fit-sides meta promises
    assert "padding-left: max(16px, env(safe-area-inset-left, 0px))" in css
    assert "padding-right: max(16px, env(safe-area-inset-right, 0px))" in css
    # both themes, and the explicit dark toggle
    assert '@media (prefers-color-scheme: dark) {\n  :root:not([data-theme="light"])' in css
    assert ':root[data-theme="dark"]' in css
    assert css.count("color-scheme: dark;") == 2
    assert "@media (prefers-reduced-motion: reduce)" in css
    for video in re.findall(r"<video\b[^>]*>", page):
        for attr in ("autoplay", "muted", "loop", "playsinline", "poster=", "aria-label="):
            assert attr in video
    text = re.sub(r"<[^>]+>", "", page)
    assert "Independent project, not affiliated with SkillCorner." in text
    assert "SoccerTrack v2, Scott et al., CC BY 4.0" in text
    # ADR-0002: public pixels say how the frames were changed
    assert all(change in text for change in ("cropped", "downscaled", "overlaid"))


def test_showcase_scripts_are_vendored_and_licensed():
    page = build_site.PAGE.read_text(encoding="utf-8")
    scripts = re.findall(r'<script src="([^"]+)"', page)
    assert scripts
    assert all(src.startswith("assets/vendor/") for src in scripts)
    licences = (build_site.MEDIA / "vendor" / "LICENSES.txt").read_text(encoding="utf-8")
    for src in scripts:
        assert src.removeprefix("assets/vendor/") in licences


def test_accuracy_chart_data_matches_the_test_report():
    page = build_site.PAGE.read_text(encoding="utf-8")
    raw = re.search(r'<script type="application/json" id="data-errors">(.*?)</script>', page)
    assert raw is not None
    errors = json.loads(raw.group(1))
    # reports/test_run.md, position error against the raw labels: p50 and p90 of each camera
    for camera, (p50, p90) in {"perframe": (0.51, 1.82), "ptz": (0.40, 1.16)}.items():
        values = errors[camera]
        assert len(values) == 100
        assert values == sorted(values)
        # the dots sit at the percentiles 0.5, 1.5, ..., 99.5: p50 and p90 fall between two of them
        assert abs((values[49] + values[50]) / 2 - p50) <= 0.01
        assert abs((values[89] + values[90]) / 2 - p90) <= 0.01


def test_phones_get_the_stacked_smoothing_clip_and_its_poster():
    page = build_site.PAGE.read_text(encoding="utf-8")
    video = page[page.index('id="smoothing-video"') :]
    video = video[: video.index("</video>")]
    sources = re.findall(r'<source src="([^"]+)" type="video/mp4"(?: media="([^"]+)")?>', video)
    # browsers that ignore media take the first source: the wide clip
    assert sources == [("assets/smoothing.mp4", "(min-width: 601px)"), ("assets/smoothing-tall.mp4", "")]
    assert 'poster="assets/smoothing.jpg"' in video
    link = re.search(
        r'<link rel="preload" as="image" href="([^"]+)" media="([^"]+)" data-poster-for="([^"]+)">', page
    )
    assert link is not None
    assert link.groups() == ("assets/smoothing-tall.jpg", "(max-width: 600px)", "smoothing-video")


# Deploy


class FakeHub:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return type("CommitInfo", (), {"commit_url": "https://huggingface.co/spaces/o/n/commit/abc"})()


def built(tmp_path, page=PAGE):
    dist = tmp_path / "dist"
    build_site.write(build_site.plan(page, media_dir(tmp_path)), dist)
    return dist


def test_deploy_creates_the_static_space_and_replaces_its_files(tmp_path):
    dist = built(tmp_path)
    hub = FakeHub()
    url = deploy.deploy(dist, "owner/name", api=hub)
    assert url.endswith("/commit/abc")
    (create, create_kwargs), (upload, upload_kwargs) = hub.calls
    assert create == "create_repo"
    assert create_kwargs == {
        "repo_id": "owner/name",
        "repo_type": "space",
        "space_sdk": "static",
        "private": False,
        "exist_ok": True,
    }
    assert upload == "upload_folder"
    assert upload_kwargs["repo_id"] == "owner/name"
    assert upload_kwargs["repo_type"] == "space"
    assert upload_kwargs["folder_path"] == dist
    assert upload_kwargs["delete_patterns"] == "*"
    assert upload_kwargs["ignore_patterns"] == [".*", "*/.*"]


def test_deploy_refuses_an_unbuilt_folder(tmp_path):
    hub = FakeHub()
    with pytest.raises(deploy.DeployError, match=r"build_site\.py first"):
        deploy.deploy(tmp_path, "owner/name", api=hub)
    assert hub.calls == []


def test_deploy_refuses_a_folder_the_build_did_not_write(tmp_path):
    (tmp_path / "index.html").write_text("<p>hand-made</p>")
    (tmp_path / "README.md").write_text("---\nsdk: static\n---\n")
    hub = FakeHub()
    with pytest.raises(deploy.DeployError, match=r"not written by scripts/build_site\.py"):
        deploy.deploy(tmp_path, "owner/name", api=hub)
    assert hub.calls == []


def test_the_generator_mark_is_the_one_the_build_writes():
    assert deploy.GENERATOR_META in build_site.wrap(PAGE)


def test_external_links_are_the_pages_links_to_other_sites():
    page = PAGE + (
        '<a href="//example.org/a" target="_blank">a</a><a href="https://github.com/zkyasin/predfoot" '
        'target="_blank">again</a><a href="mailto:x@example.org">mail</a>'
    )
    links = deploy.external_links(build_site.wrap(page))
    assert links == ["https://github.com/zkyasin/predfoot", "https://example.org/a"]


def test_preflight_lists_stale_files_and_links_that_are_not_public(tmp_path):
    dist = built(tmp_path)
    (dist / ".DS_Store").write_bytes(b"finder")  # hidden files are neither compared nor uploaded
    fresh = {p.relative_to(dist).as_posix(): p.read_bytes() for p in dist.rglob("*") if p.is_file()}
    del fresh[".DS_Store"]
    answers = {"https://github.com/zkyasin/predfoot": 200}
    assert deploy.preflight(dist, expected=fresh, status=answers.__getitem__) == []
    fresh["assets/clip.mp4"] = b"newer"
    fresh["assets/new.jpg"] = b"added"
    (dist / "assets" / "old.jpg").write_bytes(b"stale")
    answers["https://github.com/zkyasin/predfoot"] = 404
    problems = deploy.preflight(dist, expected=fresh, status=answers.__getitem__)
    assert problems == [
        "stale build, run scripts/build_site.py: assets/new.jpg: missing",
        "stale build, run scripts/build_site.py: assets/old.jpg: not in a fresh build",
        "stale build, run scripts/build_site.py: assets/clip.mp4: differs from a fresh build",
        "link not public yet: https://github.com/zkyasin/predfoot: HTTP 404",
    ]
    assert deploy.preflight(dist, expected=None, check_links=False) == []


def test_the_fresh_build_is_what_build_site_writes(tmp_path):
    out = tmp_path / "dist"
    site = build_site.plan(build_site.PAGE.read_text(encoding="utf-8"), build_site.MEDIA)
    build_site.write(site, out)
    assert deploy.stale_files(out, deploy.fresh_build()) == []


def test_link_status_asks_anonymously_and_falls_back_to_get(monkeypatch):
    import urllib.error

    seen = []

    class Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        seen.append((request.get_method(), request.full_url, request.get_header("Authorization")))
        if request.get_method() == "HEAD":
            raise urllib.error.HTTPError(request.full_url, 405, "no HEAD", {}, None)
        return Answer()

    monkeypatch.setattr(deploy.urllib.request, "urlopen", fake_urlopen)
    assert deploy.link_status("https://example.org/x") == 200
    assert seen == [("HEAD", "https://example.org/x", None), ("GET", "https://example.org/x", None)]

    def private(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)

    monkeypatch.setattr(deploy.urllib.request, "urlopen", private)
    assert deploy.link_status("https://github.com/o/private") == 404

    def offline(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(deploy.urllib.request, "urlopen", offline)
    assert deploy.link_status("https://example.org/x") == 0


def test_git_sha_marks_uncommitted_changes(monkeypatch):
    outputs = {"rev-parse": "abc1234\n", "status": " M site/page.html\n"}

    def fake_run(cmd, **kwargs):
        return type("Done", (), {"stdout": outputs[cmd[1]]})()

    monkeypatch.setattr(deploy.subprocess, "run", fake_run)
    assert deploy.git_sha() == "abc1234-dirty"
    outputs["status"] = ""
    assert deploy.git_sha() == "abc1234"


def test_cli_refuses_links_that_are_not_public_unless_told_to_skip(tmp_path, monkeypatch):
    dist = built(tmp_path)
    hub = FakeHub()
    monkeypatch.setattr(deploy, "HfApi", lambda: hub)
    monkeypatch.setattr(deploy, "link_status", lambda url: 404)
    monkeypatch.setattr(deploy.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no network in tests"))
    runner = CliRunner()
    refused = runner.invoke(deploy.app, ["--dist", str(dist), "--repo-id", "owner/name"])
    assert refused.exit_code == 1
    assert "link not public yet: https://github.com/zkyasin/predfoot: HTTP 404" in refused.output
    assert hub.calls == []
    skipped = runner.invoke(deploy.app, ["--dist", str(dist), "--repo-id", "owner/name", "--skip-link-check"])
    assert skipped.exit_code == 0, skipped.output
    assert [name for name, _ in hub.calls] == ["create_repo", "upload_folder"]
