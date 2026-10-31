"""What the real stages record in meta.json: their description and the provenance of their conformal model."""

import hashlib
import json

from predfoot.pipeline.real import real_stages
from predfoot.uncertainty.conformal import MondrianConformal


def test_real_stages_describe_themselves_for_meta_json():
    laplace = real_stages({}, conformal=None, version="test")
    assert laplace.conformal is None
    assert laplace.description is not None
    assert "uncalibrated Laplace" in laplace.description["localization"]
    model = MondrianConformal(alpha=0.1, edges={"mpp": [0.05]}, quantiles={"0": 6.0}, pooled=7.0)
    stages = real_stages({}, conformal=model, version="test")
    assert stages.description is not None
    assert "Mondrian" in stages.description["localization"]
    assert stages.conformal == {
        "kind": "mondrian",
        "sha256": hashlib.sha256(model.to_json().encode("utf-8")).hexdigest(),
        "alpha": 0.1,
        "pooled": 7.0,
        "edges": {"mpp": [0.05]},
        "quantiles": {"0": 6.0},
    }


def test_infinite_quantiles_stay_valid_json_in_meta():
    inf = float("inf")
    model = MondrianConformal(alpha=0.1, edges={"mpp": [0.05]}, quantiles={"0": inf}, pooled=inf)
    provenance = real_stages({}, conformal=model, version="test").conformal
    assert provenance is not None
    assert provenance["pooled"] == "inf"
    assert provenance["quantiles"] == {"0": "inf"}
    json.dumps(provenance, allow_nan=False)  # meta.json is written without NaN or Infinity
