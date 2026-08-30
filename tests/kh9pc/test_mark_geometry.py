"""Synthetic-ladder tests for the mark-based x geometry.

Every case builds a ladder whose TRUTH is known -- the designed angles, the
drift, an injected seam step, an off-centre exposure -- and checks that the
fitted warp puts each mark back at its exact designed angle and recovers the
cx offset.  No raster, no I/O: this is the layer where the geometry is decided.
"""

import numpy as np
import pytest

from hipp.kh9pc.restitution.mark_geometry import (
    LadderError,
    MarkTrain,
    SeamModel,
    canvas_px_per_deg,
    deg_per_mark_from_pattern,
    fit_mark_warp,
    tier_degrees,
)

W90 = 342247          # 90 deg sector tier
W30 = 114082          # 30 deg sector tier
K_ORIGIN = 7.0        # arbitrary origin of the detector's relative ladder index


def make_ladder(
    kk_lo=-9, kk_hi=8, period_src=19040.0, x0=170000.0, curv=1.59,
    deg_per_mark=5.0, noise=0.0, seed=0, step_px=0.0, step_after_kk=None,
    side="bottom", k_origin=K_ORIGIN,
):
    """A ladder with a known quadratic drift and an optional seam step.

    ``x_src(kk) = x0 + period_src*kk + curv*kk**2`` plus ``step_px`` for every mark
    beyond ``step_after_kk``.  ``curv=1.59`` with kk in [-9, 8] reproduces the
    measured ops323 sagitta (~129 px) and a ~0.30 % end-to-end period drift.
    """
    kk = np.arange(kk_lo, kk_hi + 1, dtype=float)
    x = x0 + period_src * kk + curv * kk ** 2
    if step_after_kk is not None:
        x = x + np.where(kk > step_after_kk, step_px, 0.0)
    if noise:
        x = x + np.random.default_rng(seed).normal(0.0, noise, size=kk.shape)
    train = MarkTrain(side=side, k=kk + k_origin, x=x, deg_per_mark=deg_per_mark,
                      period_src=period_src)
    return train, kk, x


def truth_source_x(kk, period_src=19040.0, x0=170000.0, curv=1.59):
    return x0 + period_src * np.asarray(kk, float) + curv * np.asarray(kk, float) ** 2


# --------------------------------------------------------------------------- basics
def test_canvas_constants_are_tier_consistent():
    assert tier_degrees(W90) == 90.0
    assert tier_degrees(W30) == 30.0
    # px/deg is the same physical constant at every tier to 1e-5
    assert canvas_px_per_deg(W30) == pytest.approx(canvas_px_per_deg(W90), rel=1e-5)
    # ... and agrees with the printed 5.24 in = 5 deg mark pitch to 3e-5
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


# ------------------------------------------------------- exact-angle recovery
def test_constant_period_ladder_lands_on_exact_angles():
    """No drift: every mark must land on its designed angle to floating point."""
    train, kk, x = make_ladder(curv=0.0)
    m = fit_mark_warp([train], W90, prior_alpha_deg=0.0,
                      prior_x_src=float(truth_source_x(0.0, curv=0.0)))
    assert m.k_ref == pytest.approx(K_ORIGIN)
    assert m.cx == W90 / 2.0
    err = m.placement_error_canvas_px(train.k, train.x)
    assert np.abs(err).max() < 0.05
    # the alpha = 0 mark lands exactly at cx
    assert m.canvas_x_of_k(m.k_ref) == pytest.approx(m.cx)
    assert m.alpha_deg(m.cx) == pytest.approx(0.0)


def test_quadratic_drift_is_removed_and_marks_land_exactly():
    """The measured drift class (0.3 %, 129 px sagitta) must vanish in the warp."""
    train, kk, x = make_ladder()
    m = fit_mark_warp([train], W90, prior_alpha_deg=0.0, prior_x_src=float(truth_source_x(0.0)))
    err = m.placement_error_canvas_px(train.k, train.x)
    assert np.abs(err).max() < 0.05, f"max placement error {np.abs(err).max():.3f} canvas px"
    assert m.resid_rms_px < 0.01
    # and the drift really was there: a single constant period leaves the measured
    # 40-50 px class of residual
    assert 30.0 < m.resid_rms_constant_period_px < 80.0
    assert m.drift_pct == pytest.approx(0.30, abs=0.02)
    assert m.is_monotonic()


def test_one_degree_ladder_recovers_exactly():
    train, kk, x = make_ladder(kk_lo=-44, kk_hi=44, period_src=3808.0, curv=0.064,
                               deg_per_mark=1.0, x0=170000.0)
    m = fit_mark_warp([train], W90, prior_alpha_deg=0.0,
                      prior_x_src=float(truth_source_x(0.0, 3808.0, 170000.0, 0.064)))
    assert m.deg_per_mark == 1.0
    assert m.unwrap_half_ambiguity_deg == 0.5
    err = m.placement_error_canvas_px(train.k, train.x)
    assert np.abs(err).max() < 0.05


def test_thirty_degree_tier_uses_the_same_angular_scale():
    train, kk, x = make_ladder(kk_lo=-3, kk_hi=2, x0=57000.0)
    m = fit_mark_warp([train], W30, prior_alpha_deg=0.0,
                      prior_x_src=float(truth_source_x(0.0, x0=57000.0)),
                      min_cover_frac=0.4)
    assert m.canvas_width == W30
    assert m.px_per_deg == pytest.approx(canvas_px_per_deg(W90), rel=1e-5)
    assert np.abs(m.placement_error_canvas_px(train.k, train.x)).max() < 0.05


# ------------------------------------------------------------------ seam steps
def test_seam_step_is_modelled_explicitly_not_smoothed():
    """A +40 px join error must come out as an explicit step, marks still exact."""
    step, after = 40.0, 1.5
    train, kk, x = make_ladder(step_px=step, step_after_kk=after)
    seam_x = float(truth_source_x(after) + 0.5 * 19040.0)
    seams = SeamModel(x_src=np.array([seam_x]), source="test")
    m = fit_mark_warp([train], W90, prior_alpha_deg=0.0, prior_x_src=float(truth_source_x(0.0)),
                      min_marks_per_section=3)
    # without the seam there is no step term and the join error contaminates the
    # smooth fit -- it is NOT quietly absorbed to a clean-looking residual
    assert m.seam_steps_px.size == 0
    assert m.resid_rms_px > 5.0

    m2 = fit_mark_warp([train], W90, prior_alpha_deg=0.0, prior_x_src=float(truth_source_x(0.0)),
                       seams=seams, min_marks_per_section=3)
    assert m2.seam_steps_px.size == 1
    assert m2.seam_steps_px[0] == pytest.approx(step, abs=0.5)
    assert m2.resid_rms_px < 0.01
    err = m2.placement_error_canvas_px(train.k, train.x)
    assert np.abs(err).max() < 0.05, f"max placement error {np.abs(err).max():.3f} canvas px"
    # the warp is discontinuous at the seam by exactly the step, and still monotonic
    assert m2.is_monotonic()


def test_seam_with_too_few_marks_is_dropped_not_silently_smoothed():
    train, kk, x = make_ladder(step_px=40.0, step_after_kk=6.5)   # only 2 marks past it
    seams = SeamModel(x_src=np.array([float(truth_source_x(7.0) - 9000.0)]), source="test")
    m = fit_mark_warp([train], W90, prior_alpha_deg=0.0, prior_x_src=float(truth_source_x(0.0)),
                      seams=seams, min_marks_per_section=3)
    assert m.seam_steps_px.size == 0
    assert any("not separable" in n for n in m.notes), m.notes


def test_a_seam_step_that_folds_the_image_is_refused():
    train, kk, x = make_ladder(step_px=-30000.0, step_after_kk=1.5)
    seams = SeamModel(x_src=np.array([float(truth_source_x(1.5) + 9000.0)]), source="test")
    with pytest.raises(LadderError, match="monotonic"):
        fit_mark_warp([train], W90, prior_alpha_deg=0.0, prior_x_src=float(truth_source_x(0.0)),
                      seams=seams, max_resid_px=1e9)


# ------------------------------------------------------------------ cx recovery
@pytest.mark.parametrize("alpha_off", [6.4, -6.4, 2.4])
def test_off_centre_exposure_cx_is_recovered(alpha_off):
    """The A001 class: the exposure sits `alpha_off` deg off the sweep centre.

    The marks must place it there on the canvas -- i.e. the exposure centre lands
    at cx + alpha_off * px_per_deg, not at cx.
    """
    train, kk, x = make_ladder()
    kk_centre = alpha_off / 5.0
    x_centre = float(truth_source_x(kk_centre))
    m = fit_mark_warp([train], W90, prior_alpha_deg=alpha_off, prior_x_src=x_centre)
    landed = float(m.canvas_x_of_source(x_centre))
    assert m.alpha_deg(landed) == pytest.approx(alpha_off, abs=1e-3)
    assert landed - m.cx == pytest.approx(alpha_off * m.px_per_deg, abs=5.0)
    # the marks are still exact
    assert np.abs(m.placement_error_canvas_px(train.k, train.x)).max() < 0.05


def test_cx_error_is_measured_when_the_prior_says_centred_but_the_film_is_not():
    """Prior 'centred' + a ladder whose alpha = 0 mark is 1 period away.

    This is the survey's F001/A001 measurement: the delivered canvas centre and the
    marks' alpha = 0 disagree, and the marks win.
    """
    train, kk, x = make_ladder()
    # truth: alpha = 0 sits at kk = 0; claim the exposure centre is at kk = 1 and
    # that it is 5 deg off, which is the correct prior for that geometry
    x_centre = float(truth_source_x(1.0))
    m = fit_mark_warp([train], W90, prior_alpha_deg=5.0, prior_x_src=x_centre)
    assert m.k_ref == pytest.approx(K_ORIGIN)
    assert float(m.canvas_x_of_source(x_centre)) - m.cx == pytest.approx(5.0 * m.px_per_deg, abs=5.0)


# ------------------------------------------------------------------- unwrapping
def test_marginal_unwrap_is_flagged_loudly():
    train, kk, x = make_ladder()
    m = fit_mark_warp([train], W90, prior_alpha_deg=0.0,
                      prior_x_src=float(truth_source_x(0.40)))   # prior 2 deg wrong
    assert abs(m.unwrap_phase) > 0.35
    assert any(n.startswith("UNWRAP_MARGINAL") for n in m.notes), m.notes


def test_a_bad_prior_moves_placement_by_exactly_one_period():
    train, kk, x = make_ladder()
    good = fit_mark_warp([train], W90, 0.0, float(truth_source_x(0.0)))
    bad = fit_mark_warp([train], W90, 0.0, float(truth_source_x(0.7)))   # > half a mark
    assert bad.k_ref - good.k_ref == pytest.approx(1.0)
    shift = bad.canvas_x_of_k(train.k) - good.canvas_x_of_k(train.k)
    assert np.allclose(shift, -good.period_canvas)


def test_k_ref_override_bypasses_the_prior():
    train, kk, x = make_ladder()
    m = fit_mark_warp([train], W90, 0.0, float(truth_source_x(0.7)), k_ref_override=K_ORIGIN)
    assert m.k_ref == K_ORIGIN
    assert any("fixed at" in n for n in m.notes)


# ------------------------------------------------------------------- two rails
def test_both_rails_are_combined():
    bot, kk, x = make_ladder(side="bottom")
    top, _, _ = make_ladder(side="top", k_origin=K_ORIGIN + 3, noise=2.0, seed=1)
    m = fit_mark_warp([bot, top], W90, 0.0, float(truth_source_x(0.0)))
    assert set(m.sides_used) == {"top", "bottom"}
    assert m.n_marks > len(kk)
    assert np.abs(m.placement_error_canvas_px(bot.k, bot.x)).max() < 1.0


def test_a_rail_locked_on_the_wrong_train_is_dropped():
    bot, kk, x = make_ladder(side="bottom")
    bad = MarkTrain(side="top", k=bot.k, x=bot.x + 0.5 * 19040.0, deg_per_mark=5.0)
    m = fit_mark_warp([bot, bad], W90, 0.0, float(truth_source_x(0.0)))
    assert m.sides_used == ("bottom",)
    assert any("out of phase" in n for n in m.notes), m.notes


# --------------------------------------------------------------------- refusals
def test_too_few_marks_is_refused():
    train, kk, x = make_ladder(kk_lo=0, kk_hi=3)
    with pytest.raises(LadderError, match="marks fitted"):
        fit_mark_warp([train], W90, 0.0, float(truth_source_x(0.0)))


def test_short_coverage_is_refused():
    train, kk, x = make_ladder(kk_lo=0, kk_hi=6)
    with pytest.raises(LadderError, match="covers"):
        fit_mark_warp([train], W90, 0.0, float(truth_source_x(0.0)), min_marks=6,
                      min_cover_frac=0.5)


def test_a_scrambled_train_is_refused_not_fitted():
    rng = np.random.default_rng(3)
    kk = np.arange(-9, 9, dtype=float)
    x = np.sort(rng.uniform(0.0, 340000.0, size=kk.size))
    train = MarkTrain("bottom", kk + K_ORIGIN, x, 5.0)
    with pytest.raises(LadderError, match="not a clean"):
        fit_mark_warp([train], W90, 0.0, 170000.0)


def test_mixed_ladder_classes_are_refused():
    a, _, _ = make_ladder(side="bottom")
    b, _, _ = make_ladder(side="top", deg_per_mark=1.0)
    with pytest.raises(LadderError, match="ladder class"):
        fit_mark_warp([a, b], W90, 0.0, 170000.0)


# ------------------------------------------------------------------ warp forms
def test_interp_form_matches_the_smooth_form_on_noise_free_marks():
    train, kk, x = make_ladder()
    prior = float(truth_source_x(0.0))
    a = fit_mark_warp([train], W90, 0.0, prior, warp_form="smooth")
    b = fit_mark_warp([train], W90, 0.0, prior, warp_form="interp")
    grid = np.linspace(a.canvas_x_of_k(train.k).min(), a.canvas_x_of_k(train.k).max(), 500)
    assert np.abs(a.source_x(grid) - b.source_x(grid)).max() < 2.0
    assert np.abs(b.placement_error_canvas_px(train.k, train.x)).max() < 0.05


def test_interp_form_keeps_the_seam_step_explicit():
    train, kk, x = make_ladder(step_px=40.0, step_after_kk=1.5)
    seams = SeamModel(x_src=np.array([float(truth_source_x(1.5) + 9000.0)]), source="test")
    m = fit_mark_warp([train], W90, 0.0, float(truth_source_x(0.0)), seams=seams,
                      warp_form="interp", min_marks_per_section=3)
    assert m.seam_steps_px[0] == pytest.approx(40.0, abs=0.5)
    assert np.abs(m.placement_error_canvas_px(train.k, train.x)).max() < 0.05


def test_noise_is_averaged_down_by_the_smooth_form():
    """With realistic 4 px detection scatter the smooth fit beats interpolation."""
    train, kk, x = make_ladder(noise=4.0, seed=11)
    truth = truth_source_x(kk)                       # noise-free positions
    prior = float(truth_source_x(0.0))
    sm = fit_mark_warp([train], W90, 0.0, prior, warp_form="smooth")
    it = fit_mark_warp([train], W90, 0.0, prior, warp_form="interp")
    grid = sm.canvas_x_of_k(train.k)
    e_sm = np.abs(sm.source_x(grid) - truth)
    e_it = np.abs(it.source_x(grid) - truth)
    assert e_sm.mean() < e_it.mean()
