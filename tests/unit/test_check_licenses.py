import importlib.util

import pytest

from predfoot.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location("check_licenses", REPO_ROOT / "scripts" / "check_licenses.py")
assert _spec is not None
assert _spec.loader is not None
check_licenses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_licenses)


@pytest.mark.parametrize(
    ("licence", "verdict"),
    [
        ("MIT License", "ok"),
        ("BSD-3-Clause", "ok"),
        ("Apache Software License", "ok"),
        ("GNU General Public License v3 (GPLv3)", "denied"),
        ("AGPL-3.0", "denied"),
        ("GPL-2.0-or-later", "denied"),
        ("CC BY-NC 4.0", "denied"),
        ("GNU Lesser General Public License v3 (LGPLv3)", "review"),
        ("LGPL-2.1", "review"),
        ("Mozilla Public License 2.0 (MPL 2.0)", "ok"),
        ("MPL-2.0 AND MIT", "ok"),
        ("UNKNOWN", "review"),
        ("", "review"),
    ],
)
def test_classify(licence, verdict):
    assert check_licenses.classify(licence) == verdict
