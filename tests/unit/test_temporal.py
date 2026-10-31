import time

import numpy as np
import pytest

from predfoot.calib import temporal as tm
from predfoot.calib.solver import SolverConfig, calibrate
from predfoot.contracts import CalibResult, CalibStatus, FootObs
from predfoot.geometry import homography as hg
from predfoot.geometry import ptz
from predfoot.pipeline.real import real_stages
from predfoot.uncertainty.delta import position_covariance

from .synthetic_camera_track import (
    CENTRE,
    PLANAR_KP,
    common_step_rms,
    cv_process,
    gaussian_calibs,
    laplace_cov,
    probe_errors,
    ptz_path,
    visible,
)

CFG = tm.SmootherConfig()  # native 25 fps; the synthetic clips use one frame in five (0.2 s)
PARAMS = tm.NoiseParams(kappa2=1.0, q_pan=5e-3, q_tilt=1e-4, q_logf=1e-3)
C2 = np.array([-35.0, -48.0, 10.0])
GRID_FEET = np.array([[x * 1920, y * 1080] for x in (0.1, 0.5, 0.9) for y in (0.45, 0.7, 0.95)])


def frames_of(n):
    return 5 * np.arange(n)


def true_centre():
    return tm.CentreEstimate(CENTRE.copy(), 1e-4 * np.eye(3), 150, 1.0, 3.4, True, "clip")


def measure(calibs, duplicate=None, centre=None):
    n = len(calibs)
    dup = np.zeros(n, bool) if duplicate is None else duplicate
    return tm.measure_clip("synthetic", frames_of(n), calibs, dup, centre or true_centre(), CFG)


def h_true(phi):
    return [ptz.homography(p, CENTRE) for p in phi]


def norms(e):
    return np.linalg.norm(e, axis=2)


def steady_pan_path(n=150):
    """A gentle pan around 20° (keeps the camera on the pitch when tests add ±30° jumps)."""
    t, phi = ptz_path(n, whip_at_s=1e6)
    phi[:, 0] = np.radians(20 + 5 * np.sin(2 * np.pi * t / 12))
    return t, phi


def slice_meas(meas, sl):
    return tm.ClipMeasurements(
        meas.clip_id,
        meas.frames[sl],
        meas.times[sl],
        meas.centre,
        meas.z[sl],
        meas.sigma_phi[sl],
        meas.d2[sl],
        meas.raw_ok[sl],
        meas.duplicate[sl],
        meas.projected[sl],
        meas.misfit[sl],
        meas.accepted[sl],
        meas.kappa2_clip,
    )


def toy_clip(rng, n, missing=(), sigma_scale=0.3, clip_id="toy"):
    """Measurements built directly in φ space (no geometry), with random SPD covariances."""
    a = rng.normal(scale=sigma_scale, size=(n, 4, 4))
    sigma = a @ np.swapaxes(a, 1, 2) + 0.1 * np.eye(4)
    z = np.cumsum(rng.normal(scale=0.05, size=(n, 4)), axis=0)
    ok = np.ones(n, bool)
    ok[list(missing)] = False
    z[~ok] = np.nan
    no = np.zeros(n, bool)
    return tm.ClipMeasurements(
        clip_id,
        frames_of(n),
        frames_of(n) / CFG.fps,
        true_centre(),
        z,
        np.where(ok[:, None, None], sigma, np.eye(4)),
        np.where(ok, 3.0, np.nan),
        ok,
        no,
        ok.copy(),
        no.copy(),
        ok.copy(),
        1.0,
    )


# --- measurement step ---------------------------------------------------------------------------------


def test_projection_is_exact_on_a_ptz_homography():
    _, phi = ptz_path(n=40)
    for p in phi[::8]:
        h = ptz.homography(p, CENTRE)
        cov = laplace_cov(h, PLANAR_KP[visible(h, PLANAR_KP)], 2.0)
        out = tm.project_to_ptz(hg.theta_from_h(h), cov, CENTRE, p + np.array([0.01, -0.01, 0.002, 0.05]))
        assert out is not None
        phi_hat, sigma, d2 = out
        assert d2 < 1e-10
        np.testing.assert_allclose(phi_hat, p, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(sigma) > 0)


def test_projection_covariance_matches_monte_carlo(rng):
    _, phi = ptz_path(n=1)
    h = ptz.homography(phi[0], CENTRE)
    cov = laplace_cov(h, PLANAR_KP[visible(h, PLANAR_KP)], 2.0)
    theta = hg.theta_from_h(h)
    _, sigma, _ = tm.project_to_ptz(theta, cov, CENTRE, phi[0])
    root = np.linalg.cholesky(cov)
    draws = []
    for _ in range(500):
        out = tm.project_to_ptz(theta + root @ rng.standard_normal(8), cov, CENTRE, phi[0])
        assert out is not None
        draws.append(out[0])
    np.testing.assert_allclose(np.var(draws, axis=0, ddof=1), np.diag(sigma), rtol=0.15)


def test_centre_survives_outliers_and_a_cut_to_another_camera(rng):
    _, phi = ptz_path()
    cut = list(range(60, 70))
    calibs = gaussian_calibs(phi, CENTRE, rng, outliers=(20, 45, 90, 120), other_camera=(cut, C2))
    assert all(calibs[k].ok for k in cut)
    est = tm.estimate_centre(calibs, np.zeros(len(calibs), bool), CFG)
    assert est is not None
    assert est.reliable
    assert np.linalg.norm(est.centre - CENTRE) < 0.3
    meas = measure(calibs, centre=est)
    assert meas.misfit[cut].all()
    assert meas.misfit.sum() <= len(cut) + 2
    assert not meas.accepted[cut].any()


def test_too_few_frames_give_no_centre(rng):
    _, phi = ptz_path(n=12)
    calibs = gaussian_calibs(phi, CENTRE, rng, failed=range(0, 5))
    assert tm.estimate_centre(calibs, np.zeros(12, bool), CFG) is None


def test_game_centre_and_fallback_order():
    def est(c, reliable, source="clip"):
        return tm.CentreEstimate(np.asarray(c, float), 0.01 * np.eye(3), 100, 1.0, 3.0, reliable, source)

    a, b, c = est([0, -50, 10], True), est([2, -52, 12], True), est([30, -10, 3], False)
    game = tm.game_centre([a, b, c, None])
    assert game is not None
    assert game.source == "game"
    np.testing.assert_allclose(game.centre, [1, -51, 11])
    assert tm.game_centre([c]) is None
    assert tm.choose_centre(a, game) is a
    assert tm.choose_centre(c, game) is game
    assert tm.choose_centre(c, None) is c
    assert tm.choose_centre(None, None) is None


# --- filter and smoother ------------------------------------------------------------------------------


def dense_gls(dt, has, z, r, params, cfg):
    """Posterior of the whole trajectory by one dense least-squares solve (independent of the filter)."""
    n, dim = len(dt), 8
    qc = np.array([params.q_pan, params.q_tilt, cfg.q_roll, params.q_logf])
    info, vec = np.zeros((n * dim, n * dim)), np.zeros(n * dim)

    def add(rows, b, cov):
        w = np.linalg.inv(cov)
        nonlocal info, vec
        info += rows.T @ w @ rows
        vec += rows.T @ w @ b

    rows = np.zeros((8, n * dim))
    rows[:, :8] = np.eye(8)
    p0 = np.zeros((8, 8))
    p0[:4, :4], p0[4:, 4:] = r[0], np.diag(np.square(cfg.v0))
    add(rows, np.concatenate([z[0], np.zeros(4)]), p0)
    for k in range(1, n):
        lam = 1.0 if (has[k] and has[k - 1]) else cfg.gap_noise_factor
        f = np.block([[np.eye(4), dt[k] * np.eye(4)], [np.zeros((4, 4)), np.eye(4)]])
        q = lam * np.diag(qc)
        qq = np.block([[dt[k] ** 3 / 3 * q, dt[k] ** 2 / 2 * q], [dt[k] ** 2 / 2 * q, dt[k] * q]])
        rows = np.zeros((8, n * dim))
        rows[:, 8 * k : 8 * k + 8] = np.eye(8)
        rows[:, 8 * (k - 1) : 8 * k] = -f
        add(rows, np.zeros(8), qq)
        if has[k]:
            rows = np.zeros((4, n * dim))
            rows[:, 8 * k : 8 * k + 4] = np.eye(4)
            add(rows, z[k], r[k])
    cov = np.linalg.inv(info)
    mean = cov @ vec
    return mean.reshape(n, 8), np.stack([cov[8 * k : 8 * k + 8, 8 * k : 8 * k + 8] for k in range(n)])


def test_filter_and_smoother_equal_the_dense_gls_solution(rng):
    meas = toy_clip(rng, 30, missing=(9, 17))
    params = tm.NoiseParams(kappa2=1.0, q_pan=0.7, q_tilt=0.3, q_logf=1.2)
    cfg = tm.SmootherConfig(q_roll=0.5, v0=(1.0, 0.8, 0.6, 1.2))
    mask = meas.accepted.copy()
    batch = tm.stack_clips([meas], [mask], [np.zeros(30, np.int64)], cfg)
    np.testing.assert_array_equal(np.flatnonzero(batch.lam[0] == cfg.gap_noise_factor), [9, 10, 17, 18])
    fwd = tm.forward(batch, params, cfg, store=True)
    ms, ps = tm.rts(fwd, batch)
    r = tm.measurement_cov(meas, params, cfg)
    mean, cov = dense_gls(batch.dt[0], mask, np.nan_to_num(meas.z), r, params, cfg)
    np.testing.assert_allclose(ms[0], mean, rtol=1e-8, atol=1e-8 * np.abs(mean).max())
    np.testing.assert_allclose(ps[0], cov, rtol=1e-8, atol=1e-8 * np.abs(cov).max())


def test_batched_nll_is_the_sum_of_the_clips(rng):
    clips = [toy_clip(rng, n, missing=m) for n, m in ((40, (5, 6)), (25, ()), (60, (29,)))]
    masks = [c.accepted for c in clips]
    segs = []
    for c, m in zip(clips, masks, strict=True):
        s = tm.segments(c, tm.measurement_cov(c, PARAMS, CFG), m, CFG, forced=(len(c) // 2,))
        assert s.max() == 1  # two segments per clip
        segs.append(s)
    total = tm.forward(tm.stack_clips(clips, masks, segs, CFG), PARAMS, CFG)
    parts = [
        tm.forward(tm.stack_clips([c], [m], [s], CFG), PARAMS, CFG)
        for c, m, s in zip(clips, masks, segs, strict=True)
    ]
    assert total.nll == pytest.approx(sum(p.nll for p in parts), rel=1e-9)
    assert total.n_terms == sum(p.n_terms for p in parts)


# --- robust smoothing on synthetic clips --------------------------------------------------------------


def test_ptz_consistent_outliers_are_repaired(rng):
    _, phi = ptz_path()
    n = len(phi)
    outliers = np.sort(rng.choice(np.arange(5, n - 5), size=8, replace=False))
    calibs = gaussian_calibs(phi, CENTRE, rng, outliers=outliers)
    meas = measure(calibs)
    track = tm.smooth_clip(meas, PARAMS, CFG)
    res, flags = tm.to_calib_results(calibs, meas, track, PARAMS, CFG)
    assert flags.repaired[outliers].all()
    assert flags.repaired.sum() - len(outliers) <= 0.05 * n
    e_raw = norms(probe_errors([c.H for c in calibs], h_true(phi)))
    e_sm = norms(probe_errors([r.H for r in res], h_true(phi)))
    assert np.nanquantile(e_sm, 0.9) <= 0.5 * np.nanquantile(e_raw, 0.9)


def test_a_long_failure_gap_splits_and_stays_failed(rng):
    _, phi = ptz_path()
    failed = range(50, 56)  # last measurement at 49, next at 56: 1.4 s
    calibs = gaussian_calibs(phi, CENTRE, rng, failed=failed)
    meas = measure(calibs)
    track = tm.smooth_clip(meas, PARAMS, CFG)
    res, flags = tm.to_calib_results(calibs, meas, track, PARAMS, CFG)
    assert track.segment[49] != track.segment[56]
    assert (track.segment[list(failed)] == -1).all()
    assert not any(res[k].ok for k in failed)
    assert not flags.filled.any()


def test_a_short_failure_gap_is_filled(rng):
    _, phi = ptz_path()
    calibs = gaussian_calibs(phi, CENTRE, rng, failed=(40,))  # 0.4 s between measurements
    meas = measure(calibs)
    track = tm.smooth_clip(meas, PARAMS, CFG)
    res, flags = tm.to_calib_results(calibs, meas, track, PARAMS, CFG)
    assert flags.filled[40]
    assert res[40].ok
    assert res[40].n_inliers == 0
    tr = [np.trace(track.cov[k, :4, :4]) for k in (39, 40, 41)]
    assert tr[1] > tr[0]
    assert tr[1] > tr[2]
    d = track.mean[40, :4] - phi[40]
    assert d @ np.linalg.solve(track.cov[40, :4, :4], d) <= tm.NEES4_99


def test_a_persistent_jump_splits_and_each_side_is_smoothed_alone(rng):
    _, phi = steady_pan_path()
    phi[75:, 0] -= np.radians(30.0)
    calibs = gaussian_calibs(phi, CENTRE, rng)
    meas = measure(calibs)
    track = tm.smooth_clip(meas, PARAMS, CFG)
    assert track.segment[74] != track.segment[75]
    assert track.segment[74] >= 0
    assert track.segment[75] >= 0
    for sl in (slice(0, 75), slice(75, None)):
        side = tm.smooth_clip(slice_meas(meas, sl), PARAMS, CFG)
        np.testing.assert_allclose(side.mean, track.mean[sl], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(side.cov, track.cov[sl], rtol=1e-10, atol=1e-16)
        np.testing.assert_array_equal(side.rejected, track.rejected[sl])


def test_a_lone_spike_does_not_split_and_is_repaired(rng):
    _, phi = steady_pan_path()
    phi_obs = phi.copy()
    phi_obs[75, 0] -= np.radians(30.0)
    calibs = gaussian_calibs(phi_obs, CENTRE, rng)
    meas = measure(calibs)
    track = tm.smooth_clip(meas, PARAMS, CFG)
    _, flags = tm.to_calib_results(calibs, meas, track, PARAMS, CFG)
    assert len(np.unique(track.segment)) == 1
    assert flags.repaired[75]
    assert flags.repaired.sum() <= 3


def test_a_three_frame_excursion_becomes_its_own_segment(rng):
    seed = int(rng.integers(2**31))
    _, phi = steady_pan_path()
    excursion = (60, 61, 62)
    calibs = gaussian_calibs(phi, CENTRE, np.random.default_rng(seed), outliers=excursion)
    clean = gaussian_calibs(phi, CENTRE, np.random.default_rng(seed))
    meas, meas_clean = measure(calibs), measure(clean)
    track = tm.smooth_clip(meas, PARAMS, CFG)
    track_clean = tm.smooth_clip(meas_clean, PARAMS, CFG)
    seg = track.segment
    assert seg[60] == seg[61] == seg[62]
    assert len({seg[59], seg[60], seg[63]}) == 3
    assert not track.rejected[list(excursion)].any()
    res, _ = tm.to_calib_results(calibs, meas, track, PARAMS, CFG)
    res_clean, _ = tm.to_calib_results(clean, meas_clean, track_clean, PARAMS, CFG)
    after = slice(63, None)
    e = norms(probe_errors([r.H for r in res], h_true(phi)))[after]
    e_clean = norms(probe_errors([r.H for r in res_clean], h_true(phi)))[after]
    assert np.sqrt(np.nanmean(e**2)) <= 1.2 * np.sqrt(np.nanmean(e_clean**2))


def test_whip_pan_gap_is_consistent(rng):
    """Regression for the λ rule: a 0.6 s gap in the middle of the whip pan stays consistent."""
    _, phi = ptz_path()
    calibs = gaussian_calibs(phi, CENTRE, rng, failed=(74, 75))
    meas = measure(calibs)
    params, _ = tm.fit_noise([meas], CFG)
    track = tm.smooth_clip(meas, params, CFG)
    for k in (74, 75):
        d = track.mean[k, :4] - phi[k]
        assert d @ np.linalg.solve(track.cov[k, :4, :4], d) <= tm.NEES4_99


# --- edge cases and contract --------------------------------------------------------------------------


def test_all_frames_failed():
    calibs = [CalibResult(CalibStatus.FEW_POINTS) for _ in range(20)]
    res, flags, track = tm.smooth_results(frames_of(20), calibs, np.zeros(20, bool), CFG, PARAMS)
    assert not any(r.ok for r in res)
    assert not any(v.any() for k, v in flags.as_dict().items() if k != "segment")
    assert (track.segment == -1).all()


def test_one_accepted_frame_is_projected():
    meas0 = toy_clip(np.random.default_rng(1), 5, missing=(0, 1, 3, 4))
    phi = np.array([0.1, np.radians(17), 0.005, np.log(2500)])
    z = meas0.z.copy()
    z[2] = phi
    meas = tm.ClipMeasurements(**{**meas0.__dict__, "z": z})
    track = tm.smooth_clip(meas, PARAMS, CFG)
    assert track.segment.tolist() == [-1, -1, 0, -1, -1]
    assert track.short[2]
    np.testing.assert_array_equal(track.mean[2, :4], z[2])
    np.testing.assert_array_equal(track.cov[2, :4, :4], track.R[2])
    raw = [CalibResult(CalibStatus.FEW_POINTS) for _ in range(5)]
    raw[2] = CalibResult(CalibStatus.OK, H=ptz.homography(phi, CENTRE), cov_theta=np.eye(8), n_inliers=9)
    res, flags = tm.to_calib_results(raw, meas, track, PARAMS, CFG)
    assert flags.projected.tolist() == [False, False, True, False, False]
    assert [r.ok for r in res] == [False, False, True, False, False]


def test_duplicates_are_copied_not_measured(rng):
    _, phi = ptz_path(n=60)
    calibs = gaussian_calibs(phi, CENTRE, rng)
    calibs[31] = calibs[30]
    dup = np.zeros(60, bool)
    dup[31] = True
    res, flags, track = tm.smooth_results(frames_of(60), calibs, dup, CFG, PARAMS)
    assert flags.duplicate.tolist() == dup.tolist()
    assert res[31] is res[30]
    assert not track.used[31]
    assert flags.smoothed[30]


def test_smoothing_is_deterministic(rng):
    _, phi = ptz_path(n=80)
    calibs = gaussian_calibs(phi, CENTRE, rng, outliers=(10, 50))
    dup = np.zeros(80, bool)
    a = tm.smooth_results(frames_of(80), calibs, dup, CFG, PARAMS)
    b = tm.smooth_results(frames_of(80), calibs, dup, CFG, PARAMS)
    for ra, rb in zip(a[0], b[0], strict=True):
        np.testing.assert_array_equal(ra.H, rb.H)
        np.testing.assert_array_equal(ra.cov_theta, rb.cov_theta)
    np.testing.assert_array_equal(a[2].mean, b[2].mean)


def test_outputs_honour_the_calibration_contract(rng):
    _, phi = ptz_path(n=60)
    calibs = gaussian_calibs(phi, CENTRE, rng, failed=(20,), outliers=(40,))
    meas = measure(calibs)
    track = tm.smooth_clip(meas, PARAMS, CFG)
    res, flags = tm.to_calib_results(calibs, meas, track, PARAMS, CFG)
    raw_ok = np.array([c.ok for c in calibs])
    one_flag = np.sum(
        [flags.smoothed, flags.repaired, flags.projected, flags.passthrough, flags.duplicate], 0
    )
    np.testing.assert_array_equal(one_flag[raw_ok], 1)
    stages = real_stages({}, None, "t")
    feet = [FootObs(track_id=i, uv=uv) for i, uv in enumerate(GRID_FEET)]
    for r in res:
        assert isinstance(r, CalibResult)
        assert r.status is CalibStatus.OK
        assert r.dof is None
        assert r.H[2, 2] == pytest.approx(1.0)
        np.testing.assert_allclose(r.cov_theta, r.cov_theta.T, atol=0)
        assert np.linalg.eigvalsh(r.cov_theta).min() >= -1e-12 * np.abs(r.cov_theta).max()
        _, cov, valid = position_covariance(r.H, r.cov_theta, GRID_FEET)
        assert valid.all()
        assert (np.linalg.eigvalsh(cov)[:, 0] > 0).all()
        locs = stages.localize(r, feet)
        assert all(np.isfinite(loc.xy).all() and np.isfinite(loc.p90_radius_m) for loc in locs)


def test_projection_only_ablation(rng):
    _, phi = ptz_path(n=40)
    calibs = gaussian_calibs(phi, CENTRE, rng, other_camera=(range(10, 14), C2))
    meas = measure(calibs)
    res, flags = tm.projection_results(calibs, meas, CFG)
    assert flags.projected.sum() == meas.accepted.sum()
    assert flags.passthrough[10:14].all()
    for k in np.flatnonzero(flags.projected):
        np.testing.assert_allclose(res[k].H, ptz.homography(meas.z[k], CENTRE))
        assert res[k].dof is None
    for k in range(10, 14):
        np.testing.assert_array_equal(res[k].H, calibs[k].H)


# --- statistics of the noise fit ----------------------------------------------------------------------


def cv_clips(rng, n_clips, n, params, cfg):
    """Clips whose φ is a constant-velocity process with the given noise, measured with R = κ² Σ."""
    sigma = np.diag(np.square([1.5e-3, 6e-4, 3e-4, 1.5e-3]))
    q = (params.q_pan, params.q_tilt, cfg.q_roll, params.q_logf)
    clips, truths = [], []
    root = np.linalg.cholesky(params.kappa2 * sigma)
    for c in range(n_clips):
        phi = cv_process(rng, n, 0.2, q)
        z = phi + rng.standard_normal((n, 4)) @ root.T
        ok = np.ones(n, bool)
        no = np.zeros(n, bool)
        clips.append(
            tm.ClipMeasurements(
                f"cv-{c}", frames_of(n), frames_of(n) / cfg.fps, true_centre(), z, np.tile(sigma, (n, 1, 1)),
                np.full(n, 3.0), ok, no, ok.copy(), no.copy(), ok.copy(), 1.0,
            )
        )  # fmt: skip
        truths.append(phi)
    return clips, truths


def test_noise_fit_recovers_the_true_parameters(rng):
    true = tm.NoiseParams(kappa2=1.8, q_pan=2e-3, q_tilt=2e-4, q_logf=4e-3)
    clips, _ = cv_clips(rng, 20, 150, true, CFG)
    fitted, diag = tm.fit_noise(clips, CFG)
    assert diag["converged"]
    np.testing.assert_allclose(fitted.log_vector(), true.log_vector(), atol=np.log(1.25))


def test_truncated_likelihood_is_unbounded_below(rng):
    """Why fit_noise does not minimise the gate-truncated NLL: shrinking every variance keeps lowering it."""
    true = tm.NoiseParams(kappa2=1.8, q_pan=2e-3, q_tilt=2e-4, q_logf=4e-3)
    clips, _ = cv_clips(rng, 2, 150, true, CFG)
    masks = [c.accepted for c in clips]
    segs = [
        tm.segments(c, tm.measurement_cov(c, true, CFG), m, CFG) for c, m in zip(clips, masks, strict=True)
    ]
    batch = tm.stack_clips(clips, masks, segs, CFG)
    shrink = (1.0, 1e-4, 1e-8)

    def scaled(s):
        return tm.NoiseParams.from_log_vector(true.log_vector() + np.log(s))

    truncated = [tm.forward(batch, scaled(s), CFG, nll_gate=tm.GATE4).nll for s in shrink]
    gaussian = [tm.forward(batch, scaled(s), CFG).nll for s in shrink]
    assert truncated[0] > truncated[1] > truncated[2]
    assert gaussian[0] < gaussian[1] < gaussian[2]


def test_noise_fit_is_robust_to_gross_outliers(rng):
    true = tm.NoiseParams(kappa2=1.8, q_pan=2e-3, q_tilt=2e-4, q_logf=4e-3)
    clips, _ = cv_clips(rng, 10, 150, true, CFG)
    dirty = []
    for c in clips:
        z = c.z.copy()
        bad = rng.choice(np.arange(3, 147), size=8, replace=False)  # 5 % of the frames
        z[bad, 0] += 30 * np.sqrt(true.kappa2) * 1.5e-3 * rng.choice([-1.0, 1.0], size=len(bad))
        dirty.append(tm.ClipMeasurements(**{**c.__dict__, "z": z}))
    fitted, diag = tm.fit_noise(dirty, CFG)
    assert diag["deleted"] >= 70
    np.testing.assert_allclose(fitted.log_vector()[:2], true.log_vector()[:2], atol=np.log(1.3))


def test_smoothed_state_covers_the_truth_at_the_nominal_level(rng):
    true = tm.NoiseParams(kappa2=1.8, q_pan=2e-3, q_tilt=2e-4, q_logf=4e-3)
    clips, truths = cv_clips(rng, 20, 150, true, CFG)
    nees = []
    for meas, phi in zip(clips, truths, strict=True):
        track = tm.smooth_clip(meas, true, CFG)
        d = track.mean[:, :4] - phi
        nees += [d[k] @ np.linalg.solve(track.cov[k, :4, :4], d[k]) for k in range(len(phi))]
    cover = np.mean(np.array(nees) <= 7.779)  # χ²₄ at 0.90
    assert 0.85 <= cover <= 0.95


# --- headline: the real solver on a synthetic clip ----------------------------------------------------


def test_headline_jitter_and_tails_with_the_real_solver(rng):
    t0 = time.perf_counter()
    n = 100
    _, phi = ptz_path(n=n, whip_at_s=12.0)
    pts = np.vstack([PLANAR_KP, rng.uniform([-52.0, -34.0], [52.0, 34.0], size=(3, 2))])
    slow = 3.0 * rng.standard_normal((len(pts), 2))
    calibs = []
    solver = SolverConfig(ransac_iterations=60)
    for k in range(n):
        if k:
            slow = 0.8 * slow + np.sqrt(1 - 0.64) * 3.0 * rng.standard_normal(slow.shape)
        h = ptz.homography(phi[k], CENTRE)
        keep = visible(h, pts)
        uv = hg.project(h, pts) + 3.0 * rng.standard_normal((len(pts), 2)) + slow
        res, _ = calibrate(pts[keep], uv[keep], np.random.default_rng(1000 + k), config=solver)
        calibs.append(res)
    dup = np.zeros(n, bool)
    centre = tm.estimate_centre(calibs, dup, CFG)
    assert centre is not None
    meas = tm.measure_clip("headline", frames_of(n), calibs, dup, centre, CFG)
    params, _ = tm.fit_noise([meas], CFG)
    track = tm.smooth_clip(meas, params, CFG)
    res, _ = tm.to_calib_results(calibs, meas, track, params, CFG)
    both = np.array([c.ok and r.ok for c, r in zip(calibs, res, strict=True)])
    e_raw = probe_errors([c.H if b else None for c, b in zip(calibs, both, strict=True)], h_true(phi))
    e_sm = probe_errors([r.H if b else None for r, b in zip(res, both, strict=True)], h_true(phi))
    ratio = {q: np.nanquantile(norms(e_sm), q) / np.nanquantile(norms(e_raw), q) for q in (0.5, 0.99)}
    step = common_step_rms(e_sm) / common_step_rms(e_raw)
    elapsed = time.perf_counter() - t0
    print(f"headline ratios: p50 {ratio[0.5]:.3f}, p99 {ratio[0.99]:.3f}, step {step:.3f}, {elapsed:.1f}s")
    assert ratio[0.5] <= 0.5
    assert ratio[0.99] <= 0.3
    assert step <= 0.2
