import importlib.util
import json

from predfoot.io import sngsr
from predfoot.paths import REPO_ROOT
from tests.unit.test_sngsr import labels

_spec = importlib.util.spec_from_file_location("data_audit", REPO_ROOT / "scripts" / "data_audit.py")
assert _spec is not None
assert _spec.loader is not None
data_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_audit)


def test_audit_reports_dev_clips_and_only_the_structure_of_test(tmp_path):
    for split, clip, game in [
        ("train", "SNGS-061", "4"),
        ("valid", "SNGS-056", "3"),
        ("test", "SNGS-120", "7"),
        ("test", "SNGS-121", "8"),
    ]:
        path = sngsr.labels_path(split, clip, tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(labels(clip=clip, game_id=game)))
    sngsr.build_index(tmp_path, workers=1)
    text = data_audit.render(tmp_path / "index", "pytest")
    assert "SNGS-061" in text
    assert "SNGS-056" in text
    assert "SNGS-120" not in text  # no per-clip line for the locked split
    assert "SNGS-121" not in text
    test_row = next(line for line in text.splitlines() if line.startswith("| test |"))
    assert "| 2 | 7, 8 |" in test_row  # clip count and game ids
    assert "matches_research" in text
