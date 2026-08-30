"""Synthetic-ladder tests for the mark-based x geometry.

Every case builds a ladder whose TRUTH is known -- the designed angles, the
fleet-calibrated distortion, an injected seam step, an off-centre exposure -- and
checks that the fitted warp puts each mark back on its designed angle and
recovers the cx offset.  No raster, no I/O: this is the layer where the geometry
is decided.

The warp form is NOT a free choice here.  ``docs/mark_fleet_2026-08-30.md``
measured one fixed distortion curve per sensor across 5 missions, 24 scanner
sessions and 14 397 marks; each frame contributes only phase and period.  These
tests therefore check that a frame drawn from the calibrated shape is recovered
with TWO free parameters -- including on a 7-mark 30-deg frame, where a per-frame
shape fit is impossible.
"""

import json

import numpy as np
import pytest

from hipp.kh9pc.restitution.mark_geometry import (
    CAMERA_SIGN,
    DISTORTION_SHAPE,
    LadderError,
    MarkTrain,
    canvas_px_per_deg,
    deg_per_mark_from_pattern,
    fit_mark_warp,
    load_distortion_shape,
    seam_step_qa,
    tier_degrees,
)

W90 = 342247          # 90 deg sector tier
W30 = 114082          # 30 deg sector tier
K_ORIGIN = 7.0        # arbitrary origin of the detector's relative ladder index
SCALE = 3808.0        # source px per degree (19040 px per 5 deg mark)
X0 = 170000.0         # source column of the alpha = 0 mark
NOISE_FLOOR = 3.0     # measured per-rail detection noise (fleet sec 2)


def shape_mosaic(alpha, camera="F", coeffs=None):
    """The calibrated distortion in the MOSAIC frame, in source px.

    Even orders carry the camera sign (the A image is stored 180 deg rotated), the
    cubic does not -- a sign flip applied to both the angle and the displacement.
    """
    a2, a3, a4 = coeffs if coeffs is not None else DISTORTION_SHAPE[camera]
    s = CAMERA_SIGN[camera]
    a = np.asarray(alpha, float)
    return s * (a2 * a ** 2 + a4 * a ** 4) + a3 * a ** 3


def truth_x(alpha, camera="F", scale=SCALE, x0=X0, extra_quad=0.0):
    a = np.asarray(alpha, float)
    return x0 + scale * a + shape_mosaic(a, camera) + extra_quad * a ** 2


def make_ladder(kk_lo=-9, kk_hi=8, deg_per_mark=5.0, camera="F", noise=0.0, seed=0,
                step_px=0.0, step_after_kk=None, side="bottom", k_origin=K_ORIGIN,
                scale=SCALE, x0=X0, extra_quad=0.0):
    """A ladder drawn from the calibrated shape, with optional noise / seam step."""
    kk = np.arange(kk_lo, kk_hi + 1, dtype=float)
    a = kk * deg_per_mark
    x = truth_x(a, camera, scale, x0, extra_quad)
    if step_after_kk is not None:
        x = x + np.where(kk > step_after_kk, step_px, 0.0)
    if noise:
        x = x + np.random.default_rng(seed).normal(0.0, noise, size=kk.shape)
    return MarkTrain(side, kk + k_origin, x, deg_per_mark), kk, x


# --------------------------------------------------------------------------- basics
def test_canvas_constants_are_tier_consistent():
    assert tier_degrees(W90) == 90.0 and tier_degrees(W30) == 30.0
    assert canvas_px_per_deg(W30) == pytest.approx(canvas_px_per_deg(W90), rel=1e-5)
    assert 5.0 * canvas_px_per_deg(W90) == pytest.approx(19014.0, rel=5e-5)


def test_deg_per_mark_comes_from_the_pattern_table():
    assert deg_per_mark_from_pattern("regulare_sparse") == 5.0
    assert deg_per_mark_from_pattern("regulare_mid") == 1.0
    assert deg_per_mark_from_pattern("segmented_mid") == 1.0
    with pytest.raises(LadderError):
        deg_per_mark_from_pattern("serialized_time_word")


def test_unknown_canvas_width_is_refused():
    with pytest.raises(LadderError):
        tier_degrees(345000)


def test_the_calibrated_shape_has_the_measured_amplitude_and_mirror():
    """~-160 px at the sweep edge in the camera frame; mirrored between F and A."""
    a = 45.0
    for cam in ("F", "A"):
        a2, a3, a4 = DISTORTION_SHAPE[cam]
        camera_frame = a2 * a ** 2 + a3 * a ** 3 + a4 * a ** 4
        assert -200.0 < camera_frame < -100.0, (cam, camera_frame)
    # in MOSAIC x the quadratic term flips sign between the sensors (fleet 3c)
    assert shape_mosaic(20.0, "F") < 0 < shape_mosaic(20.0, "A")


def test_shape_loader_refuses_a_polynomial_carrying_phase_or_period(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"F": [1e-6, 1e-4, -0.09, 0.0, 0.0]}))
    assert load_distortion_shape(p)["F"][0] == pytest.approx(-0.09)
    p.write_text(json.dumps({"F": [1e-6, 1e-4, -0.09, 12.0, 0.0]}))
    with pytest.raises(LadderError, match="phase and period"):
        load_distortion_shape(p)


# ------------------------------------------------------- exact-angle recovery
@pytest.mark.parametrize("camera", ["F", "A"])
def test_a_frame_drawn_from_the_calibrated_shape_is_recovered_exactly(camera):
    train, kk, x = make_ladder(camera=camera)
    m = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0, camera)), camera=camera)
    assert m.k_ref == pytest.approx(K_ORIGIN)
    assert m.tier_width == W90
    assert m.cx == m.canvas_width / 2.0
    assert m.canvas_x_of_k(m.k_ref) == pytest.approx(m.cx)
    # the nominal sweep sits symmetrically inside the widened canvas, at +-45 deg
    lo, hi = m.nominal_sweep_columns
    assert float(m.alpha_deg(lo)) == pytest.approx(-45.0)
    assert float(m.alpha_deg(hi)) == pytest.approx(+45.0)
    assert hi - lo == pytest.approx(W90)
    err = m.placement_error_canvas_px(train.k, train.x)
    assert np.abs(err).max() < 0.05, f"max placement error {np.abs(err).max():.4f} canvas px"
    assert m.resid_rms_px < 1e-6
    # the distortion was real: a uniform grid leaves the measured 40-50 px class
    assert 30.0 < m.resid_rms_uniform_px < 80.0
    assert m.is_monotonic()


def test_the_A_shape_is_not_interchangeable_with_the_F_shape():
    """The mirror is real: the wrong sensor's shape doubles the distortion.

    The residual gate catches it, which makes an F/A mix-up a loud failure rather
    than a quietly worse product.
    """
    train, kk, x = make_ladder(camera="A")
    good = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0, "A")), camera="A")
    assert good.resid_rms_px < 1e-6
    with pytest.raises(LadderError, match="not a clean"):
        fit_mark_warp([train], W90, 0.0, float(truth_x(0.0, "A")), camera="F")
    loose = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0, "A")), camera="F",
                          max_resid_px=1e9)
    assert loose.resid_rms_px > 2.0 * good.resid_rms_uniform_px


def test_a_seven_mark_thirty_degree_frame_still_solves():
    """ops196 class: a 5 deg ladder over a 30 deg sweep gives ~7 marks.

    A per-frame SHAPE fit is impossible here; two parameters against the shared
    shape are not.
    """
    train, kk, x = make_ladder(kk_lo=-3, kk_hi=3, x0=57041.0)
    assert len(train.k) == 7
    m = fit_mark_warp([train], W30, 0.0, float(truth_x(0.0, x0=57041.0)))
    assert np.abs(m.placement_error_canvas_px(train.k, train.x)).max() < 0.05
    assert m.n_marks == 7


def test_one_degree_ladder_recovers_exactly():
    train, kk, x = make_ladder(kk_lo=-44, kk_hi=44, deg_per_mark=1.0)
    m = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)))
    assert m.deg_per_mark == 1.0 and m.unwrap_half_ambiguity_deg == 0.5
    assert np.abs(m.placement_error_canvas_px(train.k, train.x)).max() < 0.05


def test_the_period_is_fitted_per_frame_not_assumed():
    """Phase and period genuinely vary; the shared shape must not fight them."""
    train, kk, x = make_ladder(scale=SCALE * 1.003)
    m = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0, scale=SCALE * 1.003)))
    assert m.scale_src_per_deg == pytest.approx(SCALE * 1.003, rel=1e-6)
    assert np.abs(m.placement_error_canvas_px(train.k, train.x)).max() < 0.05


# ------------------------------------------------------------------ cx recovery
@pytest.mark.parametrize("alpha_off,label", [
    (0.126, "fleet interior NMAD"),
    (0.389, "fleet interior worst (ops323 A010)"),
    (-0.389, "fleet interior worst, mirrored"),
    (0.8, "nepal PolyStrategy class"),
    (1.223, "nepal worst (F057)"),
    (6.391, "first frame (ops323 F001)"),
    (-6.959, "first frame (ops323 A001)"),
])
def test_off_centre_exposure_cx_is_recovered(alpha_off, label):
    """The exposure sits `alpha_off` deg off the sweep centre; the marks say so."""
    train, kk, x = make_ladder()
    x_centre = float(truth_x(alpha_off))
    m = fit_mark_warp([train], W90, alpha_off, x_centre)
    landed = float(m.canvas_x_of_source(x_centre))
    assert m.alpha_deg(landed) == pytest.approx(alpha_off, abs=1e-3), label
    assert landed - m.cx == pytest.approx(alpha_off * m.px_per_deg, abs=5.0), label
    assert np.abs(m.placement_error_canvas_px(train.k, train.x)).max() < 0.05


def test_cx_is_measured_against_the_marks_not_the_canvas_centre():
    """The survey's F001/A001 result: canvas centre and marks disagree; marks win."""
    train, kk, x = make_ladder()
    x_centre = float(truth_x(5.0))
    m = fit_mark_warp([train], W90, 5.0, x_centre)
    assert m.k_ref == pytest.approx(K_ORIGIN)
    assert float(m.canvas_x_of_source(x_centre)) - m.cx == pytest.approx(5.0 * m.px_per_deg,
                                                                        abs=5.0)


# ------------------------------------------------------------------- unwrapping
def test_marginal_unwrap_is_flagged_loudly():
    train, kk, x = make_ladder()
    m = fit_mark_warp([train], W90, 0.0, float(truth_x(2.0)))     # prior 2 deg wrong
    assert abs(m.unwrap_phase) > 0.35
    assert any(n.startswith("UNWRAP_MARGINAL") for n in m.notes), m.notes


def test_a_bad_prior_moves_placement_by_exactly_one_period_and_nothing_else():
    train, kk, x = make_ladder()
    good = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)))
    bad = fit_mark_warp([train], W90, 0.0, float(truth_x(3.6)))    # > half a mark
    assert bad.k_ref - good.k_ref == pytest.approx(1.0)
    shift = bad.canvas_x_of_k(train.k) - good.canvas_x_of_k(train.k)
    assert np.allclose(shift, -good.period_canvas)


def test_k_ref_override_bypasses_the_prior():
    train, kk, x = make_ladder()
    m = fit_mark_warp([train], W90, 0.0, float(truth_x(3.6)), k_ref_override=K_ORIGIN)
    assert m.k_ref == K_ORIGIN
    assert any("fixed at" in n for n in m.notes)


# ------------------------------------------------------------------- two rails
def test_both_rails_are_combined():
    bot, kk, x = make_ladder(side="bottom")
    top, _, _ = make_ladder(side="top", k_origin=K_ORIGIN + 3, noise=NOISE_FLOOR, seed=1)
    m = fit_mark_warp([bot, top], W90, 0.0, float(truth_x(0.0)))
    assert set(m.sides_used) == {"top", "bottom"}
    assert m.n_marks > len(kk)
    assert np.abs(m.placement_error_canvas_px(bot.k, bot.x)).max() < 2.0


def test_a_rail_locked_on_the_wrong_train_is_dropped():
    bot, kk, x = make_ladder(side="bottom")
    bad = MarkTrain("top", bot.k, bot.x + 0.5 * 5.0 * SCALE, 5.0)
    m = fit_mark_warp([bot, bad], W90, 0.0, float(truth_x(0.0)))
    assert m.sides_used == ("bottom",)
    assert any("out of phase" in n for n in m.notes), m.notes


# ----------------------------------------------------------- detection noise
def test_measurement_noise_is_not_amplified():
    """With the measured 3 px per-mark scatter the placement error stays ~1 px."""
    train, kk, x = make_ladder(noise=NOISE_FLOOR, seed=11)
    m = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)))
    truth = truth_x(kk * 5.0)
    err_vs_truth = np.abs(m.source_x_of_k(train.k) - truth)
    # two free parameters over 18 marks: the fit averages the noise DOWN
    assert err_vs_truth.mean() < NOISE_FLOOR
    assert m.resid_rms_px < 2.0 * NOISE_FLOOR


# ------------------------------------------------------------------ seam steps
def test_seam_steps_are_reported_as_qa_and_never_enter_the_warp():
    train, kk, x = make_ladder(kk_lo=-44, kk_hi=44, deg_per_mark=1.0,
                               step_px=8.0, step_after_kk=0.5)
    seam = float(truth_x(0.5)) + 0.5 * SCALE
    prior = float(truth_x(0.0))
    plain = fit_mark_warp([train], W90, 0.0, prior)
    withseam = fit_mark_warp([train], W90, 0.0, prior, seam_x_src=[seam])
    # the warp is bit-identical: seams inform QA, not geometry
    grid = np.linspace(0.0, W90, 501)
    assert np.array_equal(plain.source_x(grid), withseam.source_x(grid))
    qa = withseam.seam_qa
    assert qa["resolvable"] and qa["n_seams_inside"] == 1
    assert qa["max_abs_step_px"] == pytest.approx(8.0, abs=1.0)
    assert qa["rms_after_steps_px"] < plain.resid_rms_px


def test_seam_qa_declines_when_the_ladder_cannot_resolve_a_step():
    """5 deg ladders carry ~2 marks per section: the honest answer is 'cannot'."""
    train, kk, x = make_ladder()
    seams = [float(truth_x(a)) for a in (-20.0, -10.0, 0.0, 10.0, 20.0, 30.0)]
    m = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)), seam_x_src=seams)
    assert m.seam_qa["resolvable"] is False
    assert "not resolvable" in m.seam_qa["reason"]


def test_seam_qa_is_callable_standalone():
    train, kk, x = make_ladder(kk_lo=-44, kk_hi=44, deg_per_mark=1.0)
    m = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)))
    qa = seam_step_qa(m, train.k, train.x, [float(truth_x(10.0))])
    assert qa["resolvable"] and qa["max_abs_step_px"] < 1.0


# ------------------------------------------------------- per-frame refinement
def test_per_frame_refinement_declines_on_a_frame_that_does_not_need_it():
    train, kk, x = make_ladder(kk_lo=-44, kk_hi=44, deg_per_mark=1.0,
                               noise=NOISE_FLOOR, seed=5)
    m = fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)), per_frame_refine="auto")
    assert m.refine == (0.0, 0.0, 0.0)
    assert any("declined" in n for n in m.notes), m.notes


def test_per_frame_refinement_applies_only_when_earned():
    """A frame with a genuine extra quadratic and enough marks gets the escape."""
    train, kk, x = make_ladder(kk_lo=-44, kk_hi=44, deg_per_mark=1.0, extra_quad=0.05)
    prior = float(truth_x(0.0, extra_quad=0.05))
    off = fit_mark_warp([train], W90, 0.0, prior)
    on = fit_mark_warp([train], W90, 0.0, prior, per_frame_refine="auto")
    assert off.refine == (0.0, 0.0, 0.0)
    assert on.refine != (0.0, 0.0, 0.0)
    assert on.resid_rms_px < 0.1 * off.resid_rms_px
    assert any("PER_FRAME_REFINE applied" in n for n in on.notes)


# --------------------------------------------------------------------- refusals
def test_too_few_marks_is_refused():
    train, kk, x = make_ladder(kk_lo=0, kk_hi=2)
    with pytest.raises(LadderError, match="marks"):
        fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)))


def test_short_coverage_is_refused():
    train, kk, x = make_ladder(kk_lo=0, kk_hi=4)
    with pytest.raises(LadderError, match="covers"):
        fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)), min_marks=5, min_cover_frac=0.5)


def test_a_scrambled_train_is_refused_not_fitted():
    rng = np.random.default_rng(3)
    kk = np.arange(-9, 9, dtype=float)
    x = np.sort(rng.uniform(0.0, 340000.0, size=kk.size))
    with pytest.raises(LadderError, match="not a clean"):
        fit_mark_warp([MarkTrain("bottom", kk + K_ORIGIN, x, 5.0)], W90, 0.0, 170000.0)


def test_mixed_ladder_classes_are_refused():
    a, _, _ = make_ladder(side="bottom")
    b, _, _ = make_ladder(side="top", deg_per_mark=1.0)
    with pytest.raises(LadderError, match="ladder class"):
        fit_mark_warp([a, b], W90, 0.0, 170000.0)


def test_an_unknown_camera_is_refused():
    train, kk, x = make_ladder()
    with pytest.raises(LadderError, match="camera"):
        fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)), camera="X")


def test_a_shape_that_folds_the_canvas_is_refused():
    train, kk, x = make_ladder()
    huge = {"F": (-500.0, 0.0, 0.0)}          # 500 px/deg^2 beats the 3808 px/deg scale
    with pytest.raises(LadderError, match="monotonic|not a clean"):
        fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)), shape=huge, max_resid_px=1e9)


def test_a_missing_sensor_in_the_shape_table_is_refused():
    train, kk, x = make_ladder()
    with pytest.raises(LadderError, match="calibrated distortion shape"):
        fit_mark_warp([train], W90, 0.0, float(truth_x(0.0)), shape={"F": (0.0, 0.0, 0.0)},
                      camera="A")
