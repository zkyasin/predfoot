"""Fail when an installed dependency carries a licence the project cannot ship with (ADR-0002).

Run with ``uv run python scripts/check_licenses.py``. It inspects the active environment, so CI checks what it
installs (core + dev group + the serve extra); the other extras can be checked locally after
``uv sync --all-extras``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

_LESSER = re.compile(r"LGPL[\w.+-]*|(GNU )?Lesser General Public License[\w .+-]*", re.IGNORECASE)
_DENIED = re.compile(
    r"\bA?GPL|GNU (Affero )?General Public|non[- ]?commercial|\bCC[- ]BY[- ]NC", re.IGNORECASE
)
# MPL-2.0 is accepted: its copyleft is per file and only applies if we modify those files (we never do).
_REVIEW = re.compile(r"LGPL|Lesser General Public|UNKNOWN", re.IGNORECASE)

#: Packages reviewed by hand: lower-case name -> reason. Keep this short and justified.
EXCEPTIONS: dict[str, str] = {
    "predfoot": "this project (Apache-2.0)",
}


def classify(licence: str) -> str:
    """Return "denied", "review" or "ok" for a licence string reported by pip-licenses."""
    text = licence.strip()
    if _DENIED.search(_LESSER.sub("", text)):
        return "denied"
    if not text or _REVIEW.search(text):
        return "review"
    return "ok"


def main() -> int:
    raw = subprocess.run(
        [sys.executable, "-m", "piplicenses", "--format=json", "--from=mixed"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    problems = []
    for pkg in json.loads(raw):
        if pkg["Name"].lower() in EXCEPTIONS:
            continue
        verdict = classify(pkg["License"])
        if verdict != "ok":
            problems.append(f"{verdict.upper():<7} {pkg['Name']}=={pkg['Version']}: {pkg['License']}")
    if problems:
        print("\n".join(problems))
        print(
            "Read the package's licence file, then add a justified entry to EXCEPTIONS if it is acceptable."
        )
        return 1
    print("Licences OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
