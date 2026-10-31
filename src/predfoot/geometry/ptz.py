"""Pan-tilt-zoom camera with a fixed projection centre: the 4-parameter model of a main broadcast camera.

Inside a clip, the main camera turns about one projection centre C, has square pixels and its principal point
at the image centre. Its ground-plane homography then has 4 degrees of freedom per frame,
φ = (pan ψ, tilt τ, roll ρ, l = log f), and 3 shared by the whole clip (C). Textbook geometry only: the
closed-form decomposition uses the two orthogonality constraints of the rotation columns with a known
principal point and square pixels (Zhang 2000; Hartley & Zisserman, ch. 8).

Conventions (ADR-0001): world = internal pitch frame with z up; camera = OpenCV axes (x right, y down,
z forward). R0 is a horizontal view along +y, equal to the rotation of
:func:`predfoot.geometry.camera.look_at` for that view, and R(ψ, τ, ρ) = R_roll(ρ) R_tilt(τ) R_pan(ψ) R0:
ψ > 0 turns right, τ > 0 looks down and ρ is look_at's image roll. H(φ, C) = K(eˡ) R M(C) with
M(C) = [[1, 0, −Cx], [0, 1, −Cy], [0, 0, −Cz]]; before the normalisation H[2, 2] = 1, H[2, 2] is the depth
of the centre spot, which is positive.

A single frame cannot separate C from some model errors: a 5 % aspect-ratio error, for instance, is absorbed
by moving the free centre by about a metre. With C fixed for the clip, the same frame is off-model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from predfoot.contracts import FloatArray
from predfoot.geometry import homography as hg

#: Principal point of a 1920 × 1080 frame, OpenCV convention (pixel centres on integers), as in look_at.
PP_1080P: tuple[float, float] = ((1920 - 1) / 2, (1080 - 1) / 2)
#: World → camera rotation of a horizontal view along +y.
R0: FloatArray = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _rotations(pan: ArrayLike, tilt: ArrayLike, roll: ArrayLike) -> FloatArray:
    """Batched R(ψ, τ, ρ), shape (..., 3, 3)."""
    p, t, r = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (pan, tilt, roll)))
    zero, one = np.zeros_like(p), np.ones_like(p)

    def mat(rows: list[list[FloatArray]]) -> FloatArray:
        return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)

    cp, sp, ct, st, cr, sr = np.cos(p), np.sin(p), np.cos(t), np.sin(t), np.cos(r), np.sin(r)
    r_pan = mat([[cp, zero, -sp], [zero, one, zero], [sp, zero, cp]])
    r_tilt = mat([[one, zero, zero], [zero, ct, -st], [zero, st, ct]])
    r_roll = mat([[cr, -sr, zero], [sr, cr, zero], [zero, zero, one]])
    return np.asarray(r_roll @ r_tilt @ r_pan @ R0, dtype=np.float64)


def rotation(pan: float, tilt: float, roll: float) -> FloatArray:
    """World → camera rotation R(ψ, τ, ρ) (3 × 3), angles in radians."""
    return _rotations(pan, tilt, roll)


def angles(R: ArrayLike) -> tuple[float, float, float]:
    """(pan, tilt, roll) of a rotation, inverse of :func:`rotation` for |tilt| < π/2."""
    m = np.asarray(R, dtype=np.float64) @ R0.T
    return (
        float(np.arctan2(m[2, 0], m[2, 2])),
        float(np.arcsin(np.clip(m[2, 1], -1.0, 1.0))),
        float(np.arctan2(-m[0, 1], m[1, 1])),
    )


def _centre_matrix(centre: ArrayLike) -> FloatArray:
    c = np.asarray(centre, dtype=np.float64)
    return np.array([[1.0, 0.0, -c[0]], [0.0, 1.0, -c[1]], [0.0, 0.0, -c[2]]])


def _centring(pp: tuple[float, float]) -> FloatArray:
    """T = K⁻¹ up to the focal length: moves the principal point to the origin."""
    return np.array([[1.0, 0.0, -pp[0]], [0.0, 1.0, -pp[1]], [0.0, 0.0, 1.0]])


def homographies(phi: ArrayLike, centre: ArrayLike, pp: tuple[float, float] = PP_1080P) -> FloatArray:
    """Batched H(φ, C) for φ of shape (..., 4), normalised so that H[2, 2] = 1; shape (..., 3, 3)."""
    p = np.asarray(phi, dtype=np.float64)
    f = np.exp(p[..., 3])
    rot = _rotations(p[..., 0], p[..., 1], p[..., 2])
    k = np.zeros((*p.shape[:-1], 3, 3))
    k[..., 0, 0] = f
    k[..., 1, 1] = f
    k[..., 0, 2] = pp[0]
    k[..., 1, 2] = pp[1]
    k[..., 2, 2] = 1.0
    h = k @ rot @ _centre_matrix(centre)
    return np.asarray(h / h[..., 2:3, 2:3], dtype=np.float64)


def homography(phi: ArrayLike, centre: ArrayLike, pp: tuple[float, float] = PP_1080P) -> FloatArray:
    """Ground-plane homography (3 × 3, H[2, 2] = 1) of the PTZ camera φ at centre C."""
    return homographies(np.asarray(phi, dtype=np.float64).reshape(4), centre, pp)


def thetas(phi: ArrayLike, centre: ArrayLike, pp: tuple[float, float] = PP_1080P) -> FloatArray:
    """Batched θ (first 8 entries of H, row-major) for φ of shape (..., 4); shape (..., 8)."""
    h = homographies(phi, centre, pp)
    return h.reshape(*h.shape[:-2], 9)[..., : hg.N_PARAMS]


def theta(phi: ArrayLike, centre: ArrayLike, pp: tuple[float, float] = PP_1080P) -> FloatArray:
    """θ of the PTZ camera φ at centre C, shape (8,)."""
    return hg.theta_from_h(homography(phi, centre, pp))


def jacobian_theta_phi(
    phi: ArrayLike, centre: ArrayLike, pp: tuple[float, float] = PP_1080P, step: float = 1e-6
) -> FloatArray:
    """∂θ/∂φ (8 × 4) by central differences (angles in radians and log f are well scaled)."""
    p = np.asarray(phi, dtype=np.float64).reshape(4)
    shifts = np.concatenate([np.eye(4), -np.eye(4)]) * step
    th = thetas(p + shifts, centre, pp)
    return np.asarray((th[:4] - th[4:]).T / (2 * step), dtype=np.float64)


def decompose(H: ArrayLike, pp: tuple[float, float] = PP_1080P) -> tuple[FloatArray, FloatArray] | None:
    """Closed-form (φ, C) of a homography, assuming square pixels and the principal point ``pp``.

    With Hc = T H ∝ diag(f, f, 1) [r1 r2 t], the columns r1, r2 of a rotation are orthogonal with equal
    norms, which gives two linear equations in w = 1/f² (solved in least squares). Returns None when w ≤ 0
    (no real focal length: the homography is not a pinhole view of the plane with these intrinsics).
    """
    hc = _centring(pp) @ hg.normalise(H)
    h1, h2, h3 = hc[:, 0], hc[:, 1], hc[:, 2]
    a1, b1 = h1[0] * h2[0] + h1[1] * h2[1], h1[2] * h2[2]
    a2, b2 = h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2, h1[2] ** 2 - h2[2] ** 2
    den = a1**2 + a2**2
    if not den > 0:
        return None
    w = -(a1 * b1 + a2 * b2) / den
    if not w > 0:
        return None
    f = 1.0 / np.sqrt(w)
    k_inv = np.diag([1.0 / f, 1.0 / f, 1.0])
    lam = 2.0 / (np.linalg.norm(k_inv @ h1) + np.linalg.norm(k_inv @ h2))
    r1, r2, t = lam * (k_inv @ h1), lam * (k_inv @ h2), lam * (k_inv @ h3)
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    rot = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    centre = -rot.T @ t
    return np.array([*angles(rot), np.log(f)]), np.asarray(centre, dtype=np.float64)


def phi_given_centre(H: ArrayLike, centre: ArrayLike, pp: tuple[float, float] = PP_1080P) -> FloatArray:
    """Closed-form φ of a homography for a known centre C: the initial value of the PTZ projection.

    A = T H M(C)⁻¹ is proportional to diag(f, f, 1) R; its sign is fixed by the positive depth of the centre
    spot, f is the ratio of the row norms and R the nearest rotation to the row-normalised A.
    """
    c = np.asarray(centre, dtype=np.float64)
    a = _centring(pp) @ hg.normalise(H) @ np.linalg.inv(_centre_matrix(c))
    if -(a[2] @ c) < 0:
        a = -a
    norms = np.linalg.norm(a, axis=1)
    f = (norms[0] + norms[1]) / (2 * norms[2])
    u, _, vt = np.linalg.svd(a / norms[:, None])
    rot = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    return np.array([*angles(rot), np.log(f)])
