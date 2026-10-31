import io
import math

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from PIL import Image

from predfoot.contracts import Localization
from predfoot.io.skillcorner import TrackingFrame
from predfoot.viz.minimap import (
    CHI2_2_090,
    draw_frame,
    ellipse_axes,
    render_minimap,
    select_panels,
)


def test_chi2_quantile():
    assert pytest.approx(4.60517, abs=1e-5) == CHI2_2_090


@pytest.mark.parametrize(
    ("cov", "angle"),
    [
        (np.diag([4.0, 1.0]), 0.0),
        (np.diag([1.0, 4.0]), 90.0),
        (np.array([[2.0, 1.0], [1.0, 2.0]]), 45.0),  # positive correlation tilts counter-clockwise (y up)
        (np.array([[2.0, -1.0], [-1.0, 2.0]]), -45.0),
    ],
)
def test_ellipse_orientation(cov, angle):
    _, _, a = ellipse_axes(cov)
    assert a == pytest.approx(angle)


def test_ellipse_boundary_is_the_90_percent_mahalanobis_contour(rng):
    for _ in range(50):
        m = rng.normal(size=(2, 2))
        cov = m @ m.T + 0.01 * np.eye(2)
        width, height, angle = ellipse_axes(cov)
        t = math.radians(angle)
        major = np.array([math.cos(t), math.sin(t)])
        minor = np.array([-math.sin(t), math.cos(t)])
        inv = np.linalg.inv(cov)
        for point in (major * width / 2, minor * height / 2):
            assert point @ inv @ point == pytest.approx(CHI2_2_090, rel=1e-9)
        assert width >= height


def test_degenerate_covariance_gives_a_flat_ellipse():
    width, height, _ = ellipse_axes(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert height == pytest.approx(0.0, abs=1e-9)
    assert width == pytest.approx(2 * math.sqrt(CHI2_2_090 * 2.0))


def _frame(k, abstain=False, n=4):
    players = tuple(
        Localization(
            track_id=i,
            xy=None if abstain else np.array([-30.0 + 15 * i, 10.0 - 5 * i]),
            cov_xy=None if abstain else np.array([[0.5, 0.2], [0.2, 0.3]]),
            p90_radius_m=None if abstain else 1.5,
        )
        for i in range(n)
    )
    corners = None if abstain else np.array([[-40.0, 39.0], [-14.0, -36.0], [14.0, -36.0], [40.0, 39.0]])
    return TrackingFrame(
        frame=k, timestamp=f"00:00:0{k // 10}.{k % 10}0", players=players, corners=corners, abstain=abstain
    )


def test_select_panels_spreads_frames_and_shows_an_abstention():
    flags = [k in (3, 4) for k in range(30)]
    picked = select_panels(flags, 6)
    assert len(picked) == 6
    assert picked == sorted(picked)
    assert picked[0] == 0
    assert picked[-1] == 29
    assert any(flags[i] for i in picked)
    assert select_panels(flags[:3], 6) == [0, 1, 2]


def test_live_frame_draws_one_ellipse_per_player():
    ax = Figure().add_subplot()
    draw_frame(ax, _frame(5))
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse) and p.get_gid() == "p90"]
    assert len(ellipses) == 4
    assert not ax.yaxis_inverted()  # far touchline (y > 0) at the top
    assert not ax.xaxis_inverted()
    assert ax.get_aspect() == 1.0


def test_abstained_frame_is_greyed_out_without_positions():
    ax = Figure().add_subplot()
    draw_frame(ax, _frame(5, abstain=True))
    assert not [p for p in ax.patches if p.get_gid() == "p90"]
    assert any(t.get_text() == "abstained" for t in ax.texts)
    assert any(p.get_gid() == "abstain-veil" for p in ax.patches)


def test_render_writes_a_deterministic_png(tmp_path):
    frames = [_frame(k, abstain=k == 2) for k in range(6)]
    a = render_minimap(frames, tmp_path / "a.png", title="test clip")
    b = render_minimap(frames, tmp_path / "b.png", title="test clip")
    data = a.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data == b.read_bytes()
    width, height = Image.open(io.BytesIO(data)).size
    assert width > height > 300


def test_render_needs_frames(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        render_minimap([], tmp_path / "x.png")
