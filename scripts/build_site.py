"""Build the static showcase site: site/page.html → site/dist/ (index.html, assets/, README.md).

site/page.html is page content in the form the claude.ai Artifact host expects: no doctype, html, head or body
tags, starting with its <title>. The build wraps it in the same skeleton the Artifact host adds (charset,
viewport with viewport-fit=cover, a small reset), so the Artifact preview and the Hugging Face Space show the
same page. Leading <title>, <meta>, <link> and <style> elements move into <head>; the rest goes in <body>.

The page may only load media from ``assets/<name>``; the build copies each of them, and nothing else, from
assets/showcase/<name>. It also writes the README.md whose front matter configures the static Space.

The build refuses, before writing anything, a page that references a file that does not exist, loads anything
from outside except Google Fonts, links to an anchor that is not on the page, links to another site without
opening a new tab (the Space shows the page inside a frame, which GitHub and most sites refuse to load), or
weighs more than ``--max-bytes`` together with its media.

Usage:
    uv run python scripts/build_site.py                          # → site/dist/ (git-ignored)
    uv run python scripts/build_site.py --media DIR --out DIR    # other media folder, other output
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import typer
import yaml

from predfoot.io.files import atomic_write_text
from predfoot.paths import REPO_ROOT

PAGE = REPO_ROOT / "site" / "page.html"
MEDIA = REPO_ROOT / "assets" / "showcase"
OUT = REPO_ROOT / "site" / "dist"
#: Page plus media, in bytes: the showcase must open quickly on a phone.
MAX_BYTES = 4_000_000
#: Where the page finds its media, relative to index.html.
ASSET_DIR = "assets"
#: The only hosts the page may load anything from (Google Fonts: the stylesheet and its font files).
FONT_HOSTS = frozenset({"fonts.googleapis.com", "fonts.gstatic.com"})
#: Written into index.html; a folder whose index.html carries it was written by this script.
GENERATOR = "predfoot scripts/build_site.py"

#: Front matter of the Space's README.md (https://huggingface.co/docs/hub/spaces-config-reference).
SPACE_CONFIG: dict[str, str | bool] = {
    "title": "predfoot",
    "emoji": "⚽",
    "colorFrom": "green",
    "colorTo": "gray",
    "sdk": "static",
    "app_file": "index.html",
    "pinned": False,
    "license": "apache-2.0",
    "short_description": "Player positions from TV football, with error zones",
}
SPACE_BODY = """# predfoot

The showcase page of [predfoot](https://github.com/zkyasin/predfoot): where every player stands, read from a
TV picture, and how far off it could be. The code, the method and every result are in the repository.

This Space is built by `scripts/build_site.py` from `site/page.html`: change the page there, not here.

Footage: [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2), Scott et al.,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), reprojected to a virtual camera, cropped,
downscaled and overlaid. Test matches: SoccerNet SN-GSR (Somers et al.), shown only as numbers and pitch
diagrams. Every credit: [ATTRIBUTION.md](https://github.com/zkyasin/predfoot/blob/main/ATTRIBUTION.md).
"""

_RESET = (
    ":root { color-scheme: light; padding-top: env(safe-area-inset-top);"
    " padding-bottom: env(safe-area-inset-bottom); }\n"
    'body { margin: 0; font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }\n'
    "img { max-width: 100%; }\n"
    "[hidden] { display: none !important; }"
)
# The run of head elements the page starts with (comments and whitespace between them included).
_LEADING_HEAD = re.compile(
    r"\A(?:\s+|<!--.*?-->|<title\b[^>]*>.*?</title\s*>|<meta\b[^>]*>|<link\b[^>]*>|<style\b[^>]*>.*?</style\s*>)*",
    re.IGNORECASE | re.DOTALL,
)
_DOCUMENT_TAGS = re.compile(r"<!doctype|<html[\s>]|<head[\s>]|<body[\s>]", re.IGNORECASE)
_CSS_URL = re.compile(r"""url\(\s*(['"]?)(.*?)\1\s*\)""", re.IGNORECASE | re.DOTALL)
_CSS_IMPORT = re.compile(r"""@import\s+(['"])(.*?)\1""", re.IGNORECASE)
# Attributes that make the browser fetch something (href of <a> and <area> only navigates).
_LOAD_ATTRS = frozenset({"src", "poster", "data", "href", "xlink:href", "action", "formaction"})
_SRCSET_ATTRS = frozenset({"srcset", "imagesrcset"})
_NAVIGATION_TAGS = frozenset({"a", "area"})


class SiteError(Exception):
    """The page cannot be published as it is; the message lists every problem."""


@dataclass(frozen=True)
class Reference:
    """One URL the page uses: ``load`` when the browser fetches it, otherwise a link the visitor follows;
    ``new_tab`` when that link opens a new browsing context (``target="_blank"``)."""

    where: str
    url: str
    load: bool
    new_tab: bool = False


class _Scan(HTMLParser):
    """Every URL of a page, the ids it defines and its CSS (style elements and attributes)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[Reference] = []
        self.ids: set[str] = set()
        self.css: list[str] = []
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        new_tab = tag in _NAVIGATION_TAGS and any(n == "target" and v == "_blank" for n, v in attrs)
        for name, value in attrs:
            if value is None:
                continue
            if name == "id" or (name == "name" and tag == "a"):
                self.ids.add(value)
            elif name in _LOAD_ATTRS:
                load = not (name == "href" and tag in _NAVIGATION_TAGS)
                self.references.append(Reference(f"<{tag} {name}>", value.strip(), load, new_tab))
            elif name in _SRCSET_ATTRS:
                for candidate in value.split(","):
                    if candidate.strip():
                        self.references.append(Reference(f"<{tag} {name}>", candidate.split()[0], True))
            elif name == "style":
                self.css.append(value)
        if tag == "style":
            self._in_style = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.css.append(data)


def scan(page: str) -> tuple[list[Reference], set[str]]:
    """The URLs a page uses (markup and CSS) and the ids it defines."""
    parser = _Scan()
    parser.feed(page)
    parser.close()
    references = list(parser.references)
    for css in parser.css:
        references += [Reference("CSS url()", m.group(2).strip(), True) for m in _CSS_URL.finditer(css)]
        references += [Reference("CSS @import", m.group(2).strip(), True) for m in _CSS_IMPORT.finditer(css)]
    return references, parser.ids


def _asset_path(url: str) -> str | None:
    """The path below assets/ that a relative URL names, or None when it points anywhere else."""
    path = PurePosixPath(unquote(urlsplit(url).path))
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != ASSET_DIR:
        return None
    return path.relative_to(ASSET_DIR).as_posix()


def check_references(page: str, media_dir: Path) -> dict[str, Path]:
    """Map each media file the page uses (``assets/<name>``) to its source in ``media_dir``.

    Raises SiteError listing every reference that would break on the Space or load from outside."""
    references, ids = scan(page)
    media: dict[str, Path] = {}
    problems: list[str] = []
    for ref in references:
        url = ref.url
        parts = urlsplit(url)
        if not url or url.startswith("data:"):
            continue
        if url.startswith("#"):
            if url[1:] and unquote(url[1:]) not in ids:
                problems.append(f"{ref.where} {url}: no element on the page has this id")
        elif parts.scheme in ("http", "https") or url.startswith("//"):
            if ref.load and parts.hostname not in FONT_HOSTS:
                problems.append(f"{ref.where} {url}: the page may load only Google Fonts from outside")
            elif not ref.load and not ref.new_tab:
                problems.append(
                    f'{ref.where} {url}: a link to another site must open a new tab (target="_blank"), since'
                    " the Space shows the page inside a frame that most sites refuse to load"
                )
        elif parts.scheme:
            if ref.load or parts.scheme.lower() == "javascript":
                problems.append(f"{ref.where} {url}: unsupported URL scheme")
        else:
            name = _asset_path(url)
            if name is None:
                problems.append(f"{ref.where} {url}: local files must be under {ASSET_DIR}/")
            elif not (media_dir / name).is_file():
                problems.append(f"{ref.where} {url}: missing {media_dir / name}")
            else:
                media[f"{ASSET_DIR}/{name}"] = media_dir / name
    if problems:
        raise SiteError("\n".join(dict.fromkeys(problems)))
    return media


def wrap(page: str) -> str:
    """The page as a complete document, in the skeleton the claude.ai Artifact host adds."""
    if _DOCUMENT_TAGS.search(page):
        raise SiteError("the page is page content: it must not have its own doctype, html, head or body")
    leading = _LEADING_HEAD.match(page)
    split = leading.end() if leading else 0
    head, body = page[:split].strip(), page[split:].strip()
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f'<meta name="generator" content="{GENERATOR}">\n'
        f"<style>\n{_RESET}\n</style>\n"
        f"{head}\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def space_readme() -> str:
    """README.md of the Space: its configuration as front matter, then a short description."""
    front = yaml.safe_dump(SPACE_CONFIG, sort_keys=False, allow_unicode=True)
    return f"---\n{front}---\n\n{SPACE_BODY}"


@dataclass(frozen=True)
class Site:
    """Everything the Space needs, checked: the document, the media to copy and the total weight."""

    index_html: str
    media: dict[str, Path]
    total_bytes: int


def plan(page: str, media_dir: Path, max_bytes: int = MAX_BYTES) -> Site:
    """Check the page and its media and compute the site, without writing anything."""
    media = check_references(page, media_dir)
    index_html = wrap(page)
    total = len(index_html.encode("utf-8")) + sum(src.stat().st_size for src in media.values())
    if total > max_bytes:
        sizes = ", ".join(f"{name} {src.stat().st_size / 1e6:.2f} MB" for name, src in sorted(media.items()))
        raise SiteError(
            f"page plus media weigh {total / 1e6:.2f} MB, above the {max_bytes / 1e6:.2f} MB budget ({sizes})"
        )
    return Site(index_html, media, total)


def write(site: Site, out_dir: Path) -> None:
    """Write the site to ``out_dir``, replacing an earlier build; refuse a folder this script did not write.

    Raises SiteError for a non-empty folder without the generator mark in its index.html."""
    if out_dir.exists():
        index = out_dir / "index.html"
        ours = index.is_file() and GENERATOR in index.read_text(encoding="utf-8", errors="replace")
        if not ours and any(out_dir.iterdir()):
            raise SiteError(f"{out_dir} is not empty and not a build_site.py output: refusing to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name, src in site.media.items():
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    atomic_write_text(out_dir / "README.md", space_readme())
    atomic_write_text(out_dir / "index.html", site.index_html)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    page: Path = typer.Option(PAGE, help="Page content (no doctype, html, head or body)."),  # noqa: B008
    media: Path = typer.Option(MEDIA, help="Folder the page's assets/<name> files come from."),  # noqa: B008
    out: Path = typer.Option(OUT, help="Output folder, replaced on every build."),  # noqa: B008
    max_bytes: int = typer.Option(MAX_BYTES, help="Budget for the page plus its media, in bytes."),
) -> None:
    try:
        site = plan(page.read_text(encoding="utf-8"), media, max_bytes)
        write(site, out)
    except SiteError as exc:
        typer.echo(f"build_site: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"index.html  {len(site.index_html.encode('utf-8')) / 1e3:8.1f} kB")
    for name, src in sorted(site.media.items()):
        typer.echo(f"{name}  {src.stat().st_size / 1e3:8.1f} kB")
    typer.echo(f"total {site.total_bytes / 1e6:.2f} MB of {max_bytes / 1e6:.2f} MB → {out}")


if __name__ == "__main__":
    app()
