import numpy as np
import pytest

from predfoot.contracts import Localization
from predfoot.io.timebase import (
    ExportSlot,
    export_schedule,
    format_timestamp,
    interpolate_localization,
    parse_timestamp,
)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0.0, "00:00:00.00"),
        (0.1, "00:00:00.10"),
        (2.9, "00:00:02.90"),
        (59.99, "00:00:59.99"),
        (61.5, "00:01:01.50"),
        (3600 + 2 * 60 + 3.04, "01:02:03.04"),
    ],
)
def test_timestamp_format_matches_skillcorner(seconds, text):
    assert format_timestamp(seconds) == text
    assert parse_timestamp(text) == pytest.approx(seconds)


def test_timestamps_of_a_10_fps_grid_are_exact():
    # 0.1 is not representable in binary; the formatter must not drift over a full half.
    stamps = [format_timestamp(k / 10) for k in range(28_990)]
    assert stamps[-1] == "00:48:18.90"
    assert len(set(stamps)) == len(stamps)


def test_negative_time_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        format_timestamp(-0.01)


def test_malformed_timestamp_is_rejected():
    with pytest.raises(ValueError, match="HH:MM:SS"):
        parse_timestamp("1:2:3")


def test_schedule_is_a_plain_decimation_when_rates_divide():
    slots = export_schedule(n_native=30, native_fps=30, export_fps=10)
    assert [s.i0 for s in slots] == list(range(0, 30, 3))
    assert all(s.w == 0.0 and s.i1 == s.i0 for s in slots)


def test_schedule_25_to_10_fps_interpolates_between_bracketing_frames():
    slots = export_schedule(n_native=75, native_fps=25, export_fps=10)
    assert len(slots) == 30  # t = 0.0 … 2.9 s; t = 3.0 s would need native frame 75
    assert slots[0] == ExportSlot(export_frame=0, i0=0, i1=0, w=0.0)
    assert slots[1] == ExportSlot(export_frame=1, i0=2, i1=3, w=0.5)
    assert slots[2] == ExportSlot(export_frame=2, i0=5, i1=5, w=0.0)
    assert slots[-1] == ExportSlot(export_frame=29, i0=72, i1=73, w=0.5)
    for s in slots:  # the export time lies between the two native frames
        t = s.export_frame / 10
        assert s.i0 / 25 <= t + 1e-12
        assert t <= s.i1 / 25 + 1e-12


def test_schedule_handles_ntsc_rates_without_float_drift():
    slots = export_schedule(n_native=3000, native_fps=29.97, export_fps=10)
    assert all(0.0 <= s.w < 1.0 for s in slots)
    assert all(s.i1 < 3000 for s in slots)
    assert slots[-1].export_frame == len(slots) - 1


@pytest.mark.parametrize(
    ("n", "fps_in", "fps_out", "message"),
    [(0, 25, 10, "at least one"), (10, 0, 10, "positive"), (10, 25, -1, "positive")],
)
def test_schedule_rejects_bad_arguments(n, fps_in, fps_out, message):
    with pytest.raises(ValueError, match=message):
        export_schedule(n_native=n, native_fps=fps_in, export_fps=fps_out)


def _loc(xy, var, r):
    return Localization(
        track_id=3,
        xy=None if xy is None else np.array(xy, dtype=float),
        cov_xy=None if var is None else var * np.eye(2),
        p90_radius_m=r,
    )


def test_interpolation_is_linear_in_position_covariance_and_radius():
    out = interpolate_localization(_loc([0.0, 0.0], 0.1, 1.0), _loc([10.0, -4.0], 0.3, 2.0), w=0.25)
    np.testing.assert_allclose(out.xy, [2.5, -1.0])
    np.testing.assert_allclose(out.cov_xy, 0.15 * np.eye(2))
    assert out.p90_radius_m == pytest.approx(1.25)
    assert out.track_id == 3


def test_interpolated_covariance_stays_positive_semi_definite(rng):
    for _ in range(100):
        a, b = rng.normal(size=(2, 2, 2))
        la = Localization(track_id=1, xy=np.zeros(2), cov_xy=a @ a.T, p90_radius_m=1.0)
        lb = Localization(track_id=1, xy=np.ones(2), cov_xy=b @ b.T, p90_radius_m=1.0)
        cov = interpolate_localization(la, lb, w=float(rng.uniform())).cov_xy
        assert cov is not None
        assert np.linalg.eigvalsh(cov).min() >= -1e-12


def test_interpolation_with_a_missing_position_gives_no_position():
    out = interpolate_localization(_loc(None, None, None), _loc([1.0, 1.0], 0.1, 1.0), w=0.5)
    assert out.xy is None
    assert out.cov_xy is None
    assert out.p90_radius_m is None


def test_interpolation_rejects_different_tracks():
    a = _loc([0.0, 0.0], 0.1, 1.0)
    b = Localization(track_id=4, xy=np.zeros(2))
    with pytest.raises(ValueError, match="track"):
        interpolate_localization(a, b, w=0.5)
