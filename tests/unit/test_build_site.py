"""scripts/build_site.py and scripts/deploy_space.py without network or a Hugging Face account."""

import importlib.util
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
  <a href="https://github.com/zkyasin/predfoot">Code</a> <a href="#top">Top</a>
</main>
"""


def media_dir(tmp_path, sizes=None):
    """A media folder: the two files PAGE uses, plus one it does not."""
    folder = tmp_path / "media"
    folder.mkdir()
    for name, size in (sizes or {"clip.jpg": 10, "clip.mp4": 20, "unused.mp4": 30}).items():
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


SHOWCASE_MEDIA = {"hero.mp4", "hero.jpg", "smoothing.mp4", "smoothing.jpg"}


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
    assert "Independent project, not affiliated with SkillCorner." in page
    assert "SoccerTrack v2, Scott et al., CC BY 4.0" in page


# Deploy


class FakeHub:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return type("CommitInfo", (), {"commit_url": "https://huggingface.co/spaces/o/n/commit/abc"})()


def test_deploy_creates_the_static_space_and_replaces_its_files(tmp_path):
    dist = tmp_path / "dist"
    build_site.write(build_site.plan(PAGE, media_dir(tmp_path)), dist)
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


def test_deploy_refuses_an_unbuilt_folder(tmp_path):
    hub = FakeHub()
    with pytest.raises(FileNotFoundError, match=r"build_site\.py"):
        deploy.deploy(tmp_path, "owner/name", api=hub)
    assert hub.calls == []
