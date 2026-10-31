"""Property tests: whatever the writer accepts, the validator accepts, and the reader gives it back."""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from predfoot.contracts import Localization
from predfoot.io import skillcorner as sc
from predfoot.io.timebase import format_timestamp
from predfoot.io.validate import validate_record
from predfoot.pipeline.synthetic import look_at_homography

SIZE = (1920, 1080)
finite = st.floats(min_value=-200.0, max_value=200.0, allow_nan=False)


@st.composite
def covariances(draw):
    """Any PSD matrix, rank-deficient ones included, at scales from 1 mm² to 100 m²."""
    a = np.array(draw(st.lists(st.floats(-1.0, 1.0), min_size=4, max_size=4))).reshape(2, 2)
    if draw(st.booleans()):
        a[:, 1] = 0.0  # rank one
    return draw(st.floats(1e-3, 10.0)) ** 2 * (a @ a.T)


@st.composite
def localizations(draw, track_id, abstain):
    if abstain or draw(st.integers(0, 9)) == 0:
        return Localization(track_id=track_id, xy=None)
    return Localization(
        track_id=track_id,
        xy=np.array([draw(finite), draw(finite)]),
        cov_xy=draw(st.none() | covariances()),
        p90_radius_m=draw(st.none() | st.floats(0.0, 50.0)),
        is_detected=draw(st.booleans()),
    )


@st.composite
def frames(draw):
    k = draw(st.integers(0, 60_000))
    abstain = draw(st.booleans())
    ids = draw(st.lists(st.integers(0, 10**6), max_size=25, unique=True))
    corners = None
    if not abstain and draw(st.booleans()):
        H = look_at_homography(
            (draw(st.floats(-30.0, 30.0)), draw(st.floats(-70.0, -36.0)), draw(st.floats(8.0, 40.0))),
            (draw(st.floats(-60.0, 60.0)), draw(st.floats(-30.0, 34.0)), 0.0),
            focal_px=draw(st.floats(800.0, 6000.0)),
            image_size=SIZE,
        )
        corners = sc.image_corners_projection(H, SIZE)
    return sc.TrackingFrame(
        frame=k,
        timestamp=format_timestamp(k / sc.EXPORT_FPS),
        players=tuple(draw(localizations(i, abstain)) for i in ids),
        corners=corners,
        calib_health=draw(st.none() | st.floats(0.0, 1.0)),
        abstain=abstain,
    )


@settings(max_examples=300, deadline=None)
@given(frames())
def test_writer_output_always_validates_and_reads_back(frame):
    record = sc.frame_to_record(frame)
    assert validate_record(record) == []
    back = sc.record_to_frame(record)
    assert back.abstain == frame.abstain
    for ours, theirs in zip(sorted(frame.players, key=lambda p: p.track_id), back.players, strict=True):
        assert ours.track_id == theirs.track_id
        assert (ours.xy is None) == (theirs.xy is None)
        if ours.xy is not None:
            assert np.max(np.abs(theirs.xy - ours.xy)) <= 0.005 + 1e-9
        if ours.cov_xy is not None:
            assert np.max(np.abs(theirs.cov_xy - ours.cov_xy)) <= 5e-5 + 1e-9
        if ours.p90_radius_m is not None:
            assert math.isclose(theirs.p90_radius_m, ours.p90_radius_m, abs_tol=5e-4 + 1e-9)
