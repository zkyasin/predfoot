"""Planar homography between the pitch (z = 0, metres) and the image (pixels).

``H`` maps homogeneous pitch points to homogeneous image points and is normalised so that H[2, 2] = 1. The
parameter vector θ holds the first 8 entries of ``H`` in row-major order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

N_PARAMS = 8


def theta_from_h(H: ArrayLike) -> FloatArray:
    h = np.asarray(H, dtype=np.float64)
    return (h / h[2, 2]).reshape(-1)[:N_PARAMS].copy()


def h_from_theta(theta: ArrayLike) -> FloatArray:
    return np.append(np.asarray(theta, dtype=np.float64), 1.0).reshape(3, 3)


def normalise(H: ArrayLike) -> FloatArray:
    h = np.asarray(H, dtype=np.float64)
    return h / h[2, 2]


def _homogeneous(p: ArrayLike) -> FloatArray:
    p = np.asarray(p, dtype=np.float64)
    return np.concatenate([p, np.ones((*p.shape[:-1], 1))], axis=-1)


def project(H: ArrayLike, xy: ArrayLike) -> FloatArray:
    """Pitch points (..., 2) → image points (..., 2)."""
    q = _homogeneous(xy) @ np.asarray(H, dtype=np.float64).T
    return q[..., :2] / q[..., 2:3]


def backproject(H: ArrayLike, uv: ArrayLike) -> tuple[FloatArray, BoolArray]:
    """Image points (..., 2) → pitch points (..., 2), plus a validity mask.

    A pixel is valid when its viewing ray hits the ground in front of the camera. With H normalised so that
    the centre spot has positive depth, that is when the homogeneous scale of H⁻¹ũ is positive; pixels above
    the horizon get a negative scale and would otherwise map to plausible-looking points behind the camera.
    """
    G = np.linalg.inv(normalise(H))
    p = _homogeneous(uv) @ G.T
    valid = p[..., 2] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = p[..., :2] / p[..., 2:3]
    return xy, valid


def hartley_normaliser(points: ArrayLike) -> FloatArray:
    """Similarity T such that T·points has zero mean and mean distance √2 to the origin."""
    p = np.asarray(points, dtype=np.float64)
    centre = p.mean(axis=0)
    scale = np.sqrt(2.0) / max(np.linalg.norm(p - centre, axis=1).mean(), 1e-12)
    return np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1.0]])


def dlt(xy: ArrayLike, uv: ArrayLike, weights: ArrayLike | None = None) -> FloatArray:
    """Normalised direct linear transform from ≥ 4 pitch → image correspondences (least squares beyond 4)."""
    xy = np.asarray(xy, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    if len(xy) < 4:
        raise ValueError("a homography needs at least 4 correspondences")
    t_src, t_dst = hartley_normaliser(xy), hartley_normaliser(uv)
    a = _homogeneous(xy) @ t_src.T
    b = _homogeneous(uv) @ t_dst.T
    zeros = np.zeros_like(a)
    rows_u = np.hstack([zeros, -b[:, 2:3] * a, b[:, 1:2] * a])
    rows_v = np.hstack([b[:, 2:3] * a, zeros, -b[:, 0:1] * a])
    m = np.empty((2 * len(a), 9))
    m[0::2], m[1::2] = rows_u, rows_v
    if weights is not None:
        m *= np.repeat(np.sqrt(np.asarray(weights, dtype=np.float64)), 2)[:, None]
    _, _, vt = np.linalg.svd(m)
    h_norm = vt[-1].reshape(3, 3)
    return normalise(np.linalg.inv(t_dst) @ h_norm @ t_src)


def jacobian_backproject_uv(H: ArrayLike, uv: ArrayLike) -> FloatArray:
    """∂(x, y)/∂(u, v) of :func:`backproject`, shape (..., 2, 2): metres per pixel at each pixel."""
    G = np.linalg.inv(normalise(H))
    p = _homogeneous(uv) @ G.T
    d_pi = _d_pi(p)  # (..., 2, 3)
    return d_pi @ G[:, :2]


def jacobian_backproject_theta(H: ArrayLike, uv: ArrayLike) -> FloatArray:
    """∂(x, y)/∂θ of :func:`backproject`, shape (..., 2, 8).

    With G = H⁻¹ and p = Gũ, dG = −G dH G gives dp = −G dH p; for θ_k = H_ij that is −G[:, i] p_j.
    """
    h = normalise(H)
    G = np.linalg.inv(h)
    p = _homogeneous(uv) @ G.T  # (..., 3)
    d_pi = _d_pi(p)  # (..., 2, 3)
    cols = []
    for k in range(N_PARAMS):
        i, j = divmod(k, 3)
        dp = -G[:, i] * p[..., j : j + 1]  # (..., 3)
        cols.append(np.einsum("...ab,...b->...a", d_pi, dp))
    return np.stack(cols, axis=-1)


def jacobian_project_theta(H: ArrayLike, xy: ArrayLike) -> FloatArray:
    """∂(u, v)/∂θ of :func:`project`, shape (..., 2, 8); used by the calibration refinement."""
    h = normalise(H)
    xh = _homogeneous(xy)  # (..., 3)
    q = xh @ h.T
    d_pi = _d_pi(q)  # (..., 2, 3)
    cols = []
    for k in range(N_PARAMS):
        i, j = divmod(k, 3)
        dq = np.zeros_like(q)
        dq[..., i] = xh[..., j]
        cols.append(np.einsum("...ab,...b->...a", d_pi, dq))
    return np.stack(cols, axis=-1)


def _d_pi(p: FloatArray) -> FloatArray:
    """Jacobian of the perspective division π(a, b, c) = (a/c, b/c), shape (..., 2, 3)."""
    c = p[..., 2]
    out = np.zeros((*p.shape[:-1], 2, 3))
    out[..., 0, 0] = 1 / c
    out[..., 1, 1] = 1 / c
    out[..., 0, 2] = -p[..., 0] / c**2
    out[..., 1, 2] = -p[..., 1] / c**2
    return out


def image_line(H: ArrayLike, a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Normalised image line (n_u, n_v, c), n_u² + n_v² = 1, through the projections of pitch points a, b."""
    ua, ub = project(H, np.asarray(a, dtype=np.float64)[:2]), project(H, np.asarray(b, dtype=np.float64)[:2])
    line = np.cross(np.append(ua, 1.0), np.append(ub, 1.0))
    return line / np.linalg.norm(line[:2])
