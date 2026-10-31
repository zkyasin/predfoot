"""Helpers for CLI tests: typer renders errors in rich panels whose colours and wrapping depend on the
terminal (GitHub Actions forces colours), so assertions compare plain, unwrapped text."""

import re

from click.testing import Result

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BOX = re.compile(r"[\u2500-\u257f]")
#: Wide, colourless terminal for CliRunner(env=...), so rich panels neither wrap nor truncate messages.
PLAIN_ENV = {"COLUMNS": "240", "NO_COLOR": "1", "TERM": "dumb"}


def plain(result: Result) -> str:
    """Output of a CLI invocation without ANSI codes, panel borders or line wrapping."""
    text = _BOX.sub(" ", _ANSI.sub("", result.output))
    return " ".join(text.split())
