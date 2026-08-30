"""
Copyright (c) 2026 HIPP developers
Description: Mark-based x-geometry for KH-9 PC restitution -- the pure-math core.

    The film rails carry a printed scan-angle ladder (5.24 in = 5 deg on missions
    <= 1213, 1.048 in = 1 deg from 1214).  That ladder is the CAMERA'S OWN ANGULAR
    RULER: mark ``k`` was exposed at scan angle ``(k - k_ref) * deg_per_mark``,
    where ``k_ref`` indexes the mark printed at alpha = 0.  This module turns a set
    of detected marks into

      * the alpha = 0 column on the canonical canvas (the true cx), and
      * an x-resampling that puts every mark at its EXACT designed angle,

    so that on the delivered canvas the scan angle maps LINEARLY to the column by
    construction and ``cx = width / 2`` is true for every frame regardless of where
    the exposure landed on the film.

    Nothing here reads a raster; everything is arrays in, arrays out, so the
    geometry is testable against synthetic ladders with a known drift, a known
    seam step and a known cx offset.

    Warp form (chosen from the 2026-08-30 fleet mark survey, 183 frames / 7 blocks;
    the medians below were independently recomputed from the fleet mark tables):

      residual of the detected marks to ...        median rms
        a constant period (one P for the frame)     46 px
        a QUADRATIC in the ladder index k            4.5 - 5.4 px
        a cubic in k                                 3.3 - 3.8 px
        quadratic + explicit per-section steps       2.2 - 3.5 px

    so the drift is smooth and second order, and the structure left over is the
    section-mosaic seam STEP (the fleet's ``var_explained_by_steps`` is 0.64-0.83 on
    the 1 deg missions).  The model is therefore a low-order polynomial in k (the
    smooth intra-section drift) PLUS a free constant per mosaic section (the seam
    steps -- a MOSAICKING-QUALITY metric, never smoothed across; dshean 2026-08-30,
    per_frame_canvas_design sec 11 addendum).

    Interpolating straight through every mark is deliberately NOT the default: the
    per-mark scatter about the smooth model is 2-5 px and an interpolant injects all
    of it into the delivered geometry as local wiggle, whereas the fit averages it
    down.  ``warp_form="interp"`` is kept for the A/B and applies the same explicit
    section offsets, so it does not smooth across a seam either.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hipp.kh9pc.fiducial_patterns import Patterns, theorical_spacing_from_pattern
from hipp.kh9pc.kh9_image_spec import IMAGE_WIDTHS_PX

logger = logging.getLogger(__name__)

#: Sector tiers are 30 deg apart: IMAGE_WIDTHS_PX[i] spans (i + 1) * 30 deg.
TIER_DEGREES: tuple[float, ...] = tuple(30.0 * (i + 1) for i in range(len(IMAGE_WIDTHS_PX)))

#: Degrees per mark for the two printed ladders, keyed by designed px spacing, so
#: the constants stay single sourced in ``fiducial_patterns``: SPARSE_SPACING
#: 19014 px = 5.24 in = 5 deg, MID_SPACING 3803 px = 1.048 in = 1 deg.
_DEG_PER_MARK_BY_SPACING: dict[int, float] = {
    theorical_spacing_from_pattern("regulare_sparse"): 5.0,
    theorical_spacing_from_pattern("regulare_mid"): 1.0,
}


class LadderError(Exception):
    """The scan-angle ladder could not be fitted confidently.

    Raised rather than silently falling back: a restitution whose x geometry is not
    pinned by the marks is NOT a mark restitution (fallback-refusal ruling, dshean
    2026-08-25).
    """


def deg_per_mark_from_pattern(pattern: Patterns) -> float:
    """Designed scan angle between consecutive marks of *pattern*, in degrees.

    Raises ``LadderError`` for patterns with no regular designed spacing
    (``*_dense``, ``serialized_time_word``) -- those trains are diagnostics, not an
    angular ruler.
    """
    try:
        spacing = theorical_spacing_from_pattern(pattern)
    except ValueError as exc:
        raise LadderError(f"pattern {pattern!r} has no designed scan-angle spacing") from exc
    return _DEG_PER_MARK_BY_SPACING[spacing]


def tier_degrees(canvas_width_px: int) -> float:
    """Nominal sector width in degrees for a canonical canvas width.

    ``canvas_width_px`` must be one of ``kh9_image_spec.IMAGE_WIDTHS_PX``; the tier
    is a physical class (30/60/90/120 deg) known to the caller, never inferred from
    a mosaic width.
    """
    try:
        return TIER_DEGREES[IMAGE_WIDTHS_PX.index(int(canvas_width_px))]
    except ValueError as exc:
        raise LadderError(
            f"canvas width {canvas_width_px} is not a sector tier {IMAGE_WIDTHS_PX}") from exc


def canvas_px_per_deg(canvas_width_px: int, sweep_deg: float | None = None) -> float:
    """Canvas columns per degree of scan angle.

    Derived from the CANVAS, not from the printed mark pitch, so the canvas edges
    sit at exactly +-sweep/2 -- which is what ``cam_gen`` assumes when it pins the
    USGS footprint corners at the image corners (cam_gen.cc:660-669).  The two
    canonical constants agree to 3e-5 (SPARSE_SPACING/5 = 3802.8 vs 342247/90 =
    3802.744 px/deg), so the choice costs 0.3 px on a mark period.

    ``sweep_deg`` overrides the nominal tier width in degrees.  The marks measure
    the EXPOSED sweep at 90.11-90.19 deg (fleet median 2026-08-30) rather than
    90.000, i.e. the film is exposed ~0.15 % beyond the nominal sector; whether the
    USGS footprint describes 90.000 deg or the measured sweep is an OPEN question,
    and this argument is the knob that settles it by A/B.  It is not a free
    parameter: changing it rescales every frame's placement identically.
    """
    return float(canvas_width_px) / float(sweep_deg if sweep_deg else tier_degrees(canvas_width_px))


@dataclass(frozen=True)
class MarkTrain:
    """One detected ladder train on one film rail.

    Attributes
    ----------
    side:
        ``"top"`` or ``"bottom"`` film rail.
    k:
        Integer ladder index of each mark, RELATIVE (the period fit's own origin).
        Absolute indexing happens in :func:`fit_mark_warp`, against the prior.
    x:
        Source column of each mark in full-raster (mosaic) pixels.
    deg_per_mark:
        Designed scan angle between consecutive indices.
    period_src:
        Fitted period in source px (diagnostic only; the warp never uses it).
    """

    side: str
    k: NDArray[np.floating]
    x: NDArray[np.floating]
    deg_per_mark: float
    period_src: float = float("nan")

    def __post_init__(self) -> None:
        if len(self.k) != len(self.x):
            raise LadderError(f"{self.side} train: k and x differ in length")


@dataclass
class SeamModel:
    """Mosaic section boundaries used to break the smooth drift model.

    ``x_src`` are seam columns in SOURCE (mosaic) pixels.  They come from the merge
    provenance, never from the image: a seam is a fact about how the sections were
    joined, not something to re-detect.
    """

    x_src: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    source: str = "none"
    fallback: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, bool))


@dataclass
class MarkWarpModel:
    """Canvas-x -> source-x mapping that places every mark at its designed angle.

    The forward (design) direction is exact by construction::

        canvas_x(k) = canvas_width / 2 + (k - k_ref) * period_canvas

    and the inverse -- the one ``remap_tif_blockwise`` calls -- is the fitted drift
    model::

        source_x(canvas_x) = poly(kk) + step(kk),  kk = (canvas_x - W/2) / period_canvas

    where ``poly`` is the smooth intra-section drift and ``step`` the piecewise
    constant seam offset.  ``kk`` is exactly ``k - k_ref``, so the model is a
    function of SCAN ANGLE, not of pixel position.
    """

    k_ref: float
    deg_per_mark: float
    px_per_deg: float
    canvas_width: int
    coeffs: NDArray[np.floating]                     # ascending powers of kk
    section_edges_kk: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    section_offsets: NDArray[np.floating] = field(default_factory=lambda: np.zeros(1))
    # --- diagnostics; none of these is read by source_x ---
    warp_form: str = "smooth"
    n_marks: int = 0
    n_marks_rejected: int = 0
    resid_rms_px: float = float("nan")
    resid_max_px: float = float("nan")
    resid_rms_constant_period_px: float = float("nan")
    period_src_px: float = float("nan")
    drift_pct: float = float("nan")
    seam_steps_px: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    seam_x_src: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    unwrap_phase: float = float("nan")
    unwrap_prior_deg: float = float("nan")
    unwrap_half_ambiguity_deg: float = float("nan")
    sides_used: tuple[str, ...] = ()
    knots_canvas: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    knots_source: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    notes: tuple[str, ...] = ()

    # ---- forward: design geometry -------------------------------------------------
    @property
    def period_canvas(self) -> float:
        """Designed mark period on the canonical canvas, in canvas px."""
        return self.deg_per_mark * self.px_per_deg

    @property
    def cx(self) -> float:
        """The alpha = 0 column on the delivered canvas -- the true cx."""
        return self.canvas_width / 2.0

    def canvas_x_of_k(self, k: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Designed canvas column of absolute ladder index *k*."""
        return self.cx + (np.asarray(k, float) - self.k_ref) * self.period_canvas

    def alpha_deg(self, canvas_x: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Scan angle of a canvas column in degrees -- linear, by construction."""
        return (np.asarray(canvas_x, float) - self.cx) / self.px_per_deg

    # ---- inverse: what the remap calls --------------------------------------------
    def _kk(self, canvas_x: NDArray[np.floating]) -> NDArray[np.floating]:
        return (np.asarray(canvas_x, float) - self.cx) / self.period_canvas

    def _section_of(self, kk: NDArray[np.floating]) -> NDArray[np.integer]:
        if self.section_edges_kk.size == 0:
            return np.zeros(np.shape(kk), int)
        return np.searchsorted(self.section_edges_kk, kk, side="right")

    def source_x(self, canvas_x: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Source column for a canonical-canvas column (the inverse warp)."""
        kk = self._kk(canvas_x)
        step = self.section_offsets[self._section_of(kk)] if self.section_offsets.size > 1 else 0.0
        if self.warp_form == "interp":
            # knots hold the drift with the section offsets REMOVED, so the step
            # stays explicit and the interpolant never spans a seam smoothly.
            return np.interp(np.asarray(canvas_x, float), self.knots_canvas, self.knots_source) + step
        return np.polyval(self.coeffs[::-1], kk) + step

    def canvas_x_of_source(self, x_src: NDArray[np.floating] | float, samples: int = 8192
                           ) -> NDArray[np.floating]:
        """Numeric inverse of :meth:`source_x` over the canvas (monotone by gate)."""
        grid = np.linspace(0.0, float(self.canvas_width), samples)
        return np.interp(np.asarray(x_src, float), self.source_x(grid), grid)

    def is_monotonic(self, samples: int = 8192) -> bool:
        """True when the inverse warp is strictly increasing across the canvas.

        Section steps make ``source_x`` piecewise: a step that reverses the mapping
        would fold the image, so it is refused rather than rendered.
        """
        src = self.source_x(np.linspace(0.0, float(self.canvas_width), samples))
        return bool(np.all(np.diff(src) > 0))

    # ---- verification -------------------------------------------------------------
    def mark_residuals(self, k, x) -> NDArray[np.floating]:
        """Source-px error of the model at the given (absolute k, source x) marks."""
        return np.asarray(x, float) - self.source_x(self.canvas_x_of_k(k))

    def placement_error_canvas_px(self, k, x) -> NDArray[np.floating]:
        """Where each mark LANDS minus where it is DESIGNED to land, in canvas px.

        This is the quantity the mark warp exists to drive to zero: the delivered
        column of a mark, minus ``canvas_x_of_k``.
        """
        return self.canvas_x_of_source(np.asarray(x, float)) - self.canvas_x_of_k(k)

    def to_dict(self) -> dict:
        """JSON-safe summary for the per-frame QC record."""
        return {
            "k_ref": self.k_ref, "deg_per_mark": self.deg_per_mark,
            "px_per_deg": self.px_per_deg, "canvas_width": self.canvas_width,
            "cx": self.cx, "period_canvas_px": self.period_canvas,
            "period_src_px": self.period_src_px, "drift_pct": self.drift_pct,
            "warp_form": self.warp_form, "poly_coeffs": [float(c) for c in self.coeffs],
            "n_marks": self.n_marks, "n_marks_rejected": self.n_marks_rejected,
            "resid_rms_px": self.resid_rms_px, "resid_max_px": self.resid_max_px,
            "resid_rms_constant_period_px": self.resid_rms_constant_period_px,
            "seam_steps_px": [float(s) for s in self.seam_steps_px],
            "seam_x_src": [float(s) for s in self.seam_x_src],
            "unwrap_phase": self.unwrap_phase, "unwrap_prior_deg": self.unwrap_prior_deg,
            "unwrap_half_ambiguity_deg": self.unwrap_half_ambiguity_deg,
            "sides_used": list(self.sides_used), "notes": list(self.notes),
        }


def _drop_unidentifiable_seams(sec_counts: NDArray[np.integer], seam_x: NDArray[np.floating],
                               min_marks: int) -> tuple[NDArray[np.floating], list[float]]:
    """Merge sections that cannot identify their own offset, weakest seam first.

    Returns the retained seam positions and the dropped ones.  A dropped seam's
    step is NOT modelled -- it stays in the residual and is reported as a mosaic
    quality flag, which is the honest outcome: with < ``min_marks`` marks on a side
    the step and the drift are not separable.
    """
    counts = list(sec_counts)
    seams = list(seam_x)
    dropped: list[float] = []
    while len(counts) > 1:
        weakest = min(range(len(seams)), key=lambda i: min(counts[i], counts[i + 1]))
        if min(counts[weakest], counts[weakest + 1]) >= min_marks:
            break
        counts[weakest] = counts[weakest] + counts[weakest + 1]
        del counts[weakest + 1]
        dropped.append(seams.pop(weakest))
    return np.asarray(seams, float), sorted(dropped)


def _fit_drift(kk, x, section, n_sections, poly_degree, robust_iters, clip_sigma):
    """Least squares of ``x ~ poly(kk) + step[section]`` with MAD clipping.

    Returns ``(poly_coeffs_ascending, section_offsets, inlier_mask)``.  Section 0's
    offset is pinned to 0 (absorbed by the polynomial constant).
    """
    n = len(kk)
    cols = [kk ** p for p in range(poly_degree + 1)]
    cols += [(section == s).astype(float) for s in range(1, n_sections)]
    A = np.column_stack(cols)
    n_par = A.shape[1]
    keep = np.ones(n, bool)
    for _ in range(max(1, robust_iters)):
        sol, *_ = np.linalg.lstsq(A[keep], x[keep], rcond=None)
        r = x - A @ sol
        med = np.median(r[keep])
        mad = 1.4826 * np.median(np.abs(r[keep] - med))
        new = np.abs(r - med) <= max(clip_sigma * mad, 3.0)
        if new.sum() < n_par + 1 or np.array_equal(new, keep):
            break
        keep = new
    sol, *_ = np.linalg.lstsq(A[keep], x[keep], rcond=None)
    coeffs = sol[: poly_degree + 1]
    offs = np.concatenate([[0.0], sol[poly_degree + 1:]]) if n_sections > 1 else np.zeros(1)
    return coeffs, offs, keep


def fit_mark_warp(
    trains: list[MarkTrain],
    canvas_width: int,
    prior_alpha_deg: float,
    prior_x_src: float,
    *,
    sweep_deg: float | None = None,
    seams: SeamModel | None = None,
    poly_degree: int = 2,
    warp_form: str = "smooth",
    min_marks: int = 6,
    min_cover_frac: float = 0.5,
    min_marks_per_section: int = 3,
    max_resid_px: float = 60.0,
    unwrap_warn_frac: float = 0.35,
    rail_agree_px: float = 40.0,
    k_ref_override: float | None = None,
    robust_iters: int = 6,
    clip_sigma: float = 4.0,
) -> MarkWarpModel:
    """Fit the canvas-x -> source-x mark warp for one frame.

    Parameters
    ----------
    trains:
        One or more :class:`MarkTrain` (top and/or bottom rail).  Trains whose
        ladder phase disagrees with the reference by more than a quarter mark are
        dropped and recorded (a rail that locked onto the wrong train).
    canvas_width:
        Canonical canvas width in px -- a sector tier.  ``cx = canvas_width / 2``.
    prior_alpha_deg, prior_x_src:
        The a-priori scan angle (deg) believed to sit at source column
        ``prior_x_src``.  Interior frame: 0 deg at the exposure centre.  First
        frame: the regime offset (~6.5 deg, per-camera sign; first_frame_census
        2026-08-30).  Used ONLY to resolve the ladder's integer ambiguity; it never
        enters the geometry, and a wrong unwrap shifts placement by exactly one
        mark period, nothing else.
    seams:
        Mosaic section boundaries in source px, from the merge provenance.  ``None``
        = one section, and any step is absorbed into the smooth model (recorded).
    warp_form:
        ``"smooth"`` (default: polynomial drift + per-section steps) or ``"interp"``
        (piecewise linear through every inlier mark, with the same explicit section
        offsets; exact at the marks but carrying the full detection scatter).
    k_ref_override:
        Absolute index of the alpha = 0 mark, when a printed label has been read or
        a ruling fixes it.  Bypasses the prior unwrap entirely.

    Raises
    ------
    LadderError
        Too few marks, too little sweep coverage, a residual above ``max_resid_px``,
        a folded (non-monotonic) warp, or trains that cannot be reconciled.
        Refusing is the ruling: a silently degraded x geometry is worse than no
        product.
    """
    trains = [t for t in trains if len(t.k) >= 3]
    if not trains:
        raise LadderError("no ladder train with >= 3 marks")
    degs = {t.deg_per_mark for t in trains}
    if len(degs) != 1:
        raise LadderError(f"trains disagree on the ladder class: {sorted(degs)}")
    deg_per_mark = degs.pop()

    px_per_deg = canvas_px_per_deg(canvas_width, sweep_deg)
    period_canvas = deg_per_mark * px_per_deg
    notes: list[str] = []

    # ---- 1. put every train on ONE relative index --------------------------------
    # Each train's k came from its own period fit, so origins differ by an integer.
    # Align on the provisional linear model of the train with the most marks.
    trains = sorted(trains, key=lambda t: (-len(t.k), t.side))
    ref = trains[0]
    p_ref = np.polyfit(np.asarray(ref.k, float), np.asarray(ref.x, float), 1)  # x = p0*k + p1
    if not np.isfinite(p_ref).all() or p_ref[0] <= 0:
        raise LadderError(f"{ref.side} train has a non-increasing period")
    aligned: list[tuple[MarkTrain, float]] = [(ref, 0.0)]
    for t in trains[1:]:
        shift = float(np.median((np.asarray(t.x, float) - p_ref[1]) / p_ref[0] - np.asarray(t.k, float)))
        if abs(shift - round(shift)) > 0.25:
            notes.append(f"{t.side} rail ladder is {shift - round(shift):+.2f} mark out of phase "
                         f"with the {ref.side} rail -- rail dropped")
            logger.warning("mark warp: dropping the %s rail (phase %+.2f mark)", t.side, shift - round(shift))
            continue
        aligned.append((t, float(round(shift))))

    k_all = np.concatenate([np.asarray(t.k, float) + s for t, s in aligned])
    x_all = np.concatenate([np.asarray(t.x, float) for t, _ in aligned])
    side_all = np.concatenate([np.full(len(t.k), t.side, object) for t, _ in aligned])
    order = np.argsort(x_all)
    k_all, x_all, side_all = k_all[order], x_all[order], side_all[order]
    sides = tuple(dict.fromkeys(str(s) for s in side_all))

    if len(k_all) < min_marks:
        raise LadderError(f"only {len(k_all)} marks fitted (need {min_marks})")
    span_canvas = (x_all[-1] - x_all[0]) * period_canvas / p_ref[0]
    cover = span_canvas / float(canvas_width)
    if cover < min_cover_frac:
        raise LadderError(f"the ladder covers {cover:.2f} of the sweep (need {min_cover_frac:.2f})")

    # ---- 2. absolute index: unwrap the ladder against the prior -------------------
    k_at_prior = (float(prior_x_src) - p_ref[1]) / p_ref[0]
    k_ref_real = k_at_prior - float(prior_alpha_deg) / deg_per_mark
    half_amb = 0.5 * deg_per_mark
    if k_ref_override is not None:
        k_ref = float(k_ref_override)
        phase = float("nan")
        notes.append(f"k_ref fixed at {k_ref:g} by the caller (prior would have given "
                     f"{np.round(k_ref_real):g})")
    else:
        k_ref = float(np.round(k_ref_real))
        phase = k_ref_real - k_ref
        if abs(phase) > unwrap_warn_frac:
            notes.append(
                f"UNWRAP_MARGINAL: ladder phase {phase:+.3f} mark ({phase * deg_per_mark:+.3f} deg) "
                f"against a +-{half_amb:.2f} deg ambiguity -- placement may be one mark out; "
                "a label-derived k_ref settles it")
            logger.warning("mark unwrap is marginal: phase %+.3f mark (%+.3f deg), half-ambiguity "
                           "%.2f deg", phase, phase * deg_per_mark, half_amb)
    kk = k_all - k_ref

    # ---- 3. sections from the merge seams ----------------------------------------
    # Seam positions are converted to ladder index with the PROVISIONAL linear model
    # (accurate to a few tens of source px, i.e. ~1e-3 mark) and everything -- fit
    # and evaluation -- is then indexed in kk, so both paths agree exactly.
    seam_src = np.asarray(seams.x_src, float) if seams is not None else np.empty(0)
    seam_src = np.sort(seam_src[(seam_src > x_all[0]) & (seam_src < x_all[-1])])
    seam_kk = (seam_src - p_ref[1]) / p_ref[0] - k_ref if seam_src.size else np.empty(0)
    if seam_kk.size:
        counts = np.bincount(np.searchsorted(seam_kk, kk, side="right"), minlength=seam_kk.size + 1)
        seam_kk, dropped = _drop_unidentifiable_seams(counts, seam_kk, min_marks_per_section)
        if dropped:
            notes.append(f"{len(dropped)} of {len(dropped) + seam_kk.size} interior seams have "
                         f"< {min_marks_per_section} marks on a side: the step is not separable "
                         "from the drift there, left unmodelled (mosaic-quality flag)")
    sec = np.searchsorted(seam_kk, kk, side="right") if seam_kk.size else np.zeros(len(kk), int)
    n_sections = int(seam_kk.size) + 1

    # ---- 4. the drift fit ---------------------------------------------------------
    deg = int(np.clip(poly_degree, 1, max(1, len(k_all) - n_sections - 1)))
    if deg != poly_degree:
        notes.append(f"polynomial degree reduced {poly_degree} -> {deg} for {len(k_all)} marks "
                     f"over {n_sections} sections")
    coeffs, offs, inl = _fit_drift(kk, x_all, sec, n_sections, deg, robust_iters, clip_sigma)
    resid = x_all - (np.polyval(coeffs[::-1], kk) + offs[sec])
    rms = float(np.sqrt((resid[inl] ** 2).mean()))
    rmax = float(np.abs(resid[inl]).max())
    # what a single constant period would have left -- the number the warp beats
    p1 = np.polyfit(kk[inl], x_all[inl], 1)
    rms_const = float(np.sqrt(((x_all[inl] - np.polyval(p1, kk[inl])) ** 2).mean()))

    if rms > max_resid_px:
        raise LadderError(
            f"ladder residual {rms:.1f} px rms (max {rmax:.1f}) over {len(k_all)} marks exceeds "
            f"{max_resid_px:.0f} px -- this is not a clean {deg_per_mark:g} deg ladder")

    if len(sides) > 1:
        for side in sides:
            m = (side_all == side) & inl
            if m.any():
                rr = float(np.sqrt((resid[m] ** 2).mean()))
                if rr > rail_agree_px:
                    notes.append(f"{side} rail residual {rr:.1f} px vs joint {rms:.1f} px -- the "
                                 f"rails disagree beyond {rail_agree_px:.0f} px")

    # ---- 5. optional interpolating form (same explicit section offsets) -----------
    knots_c = np.empty(0)
    knots_s = np.empty(0)
    if warp_form == "interp":
        ku, first = np.unique(k_all[inl], return_index=True)
        xs = (x_all[inl] - offs[sec[inl]])[first]
        if len(ku) < 3:
            raise LadderError("interp warp form needs >= 3 distinct inlier marks")
        c = canvas_width / 2.0 + (ku - k_ref) * period_canvas
        s_lo = (xs[1] - xs[0]) / (c[1] - c[0])
        s_hi = (xs[-1] - xs[-2]) / (c[-1] - c[-2])
        knots_c = np.concatenate([[0.0], c, [float(canvas_width)]])
        knots_s = np.concatenate([[xs[0] + s_lo * (0.0 - c[0])], xs,
                                  [xs[-1] + s_hi * (float(canvas_width) - c[-1])]])
        if not np.all(np.diff(knots_c) > 0):        # a mark outside the canvas edge
            keep = np.concatenate([[True], np.diff(knots_c) > 0])
            knots_c, knots_s = knots_c[keep], knots_s[keep]

    drift = float(2.0 * coeffs[2] / coeffs[1] * (kk[inl].max() - kk[inl].min()) * 100.0) \
        if deg >= 2 and coeffs[1] else float("nan")

    model = MarkWarpModel(
        k_ref=k_ref, deg_per_mark=deg_per_mark, px_per_deg=px_per_deg,
        canvas_width=int(canvas_width), coeffs=np.asarray(coeffs, float),
        section_edges_kk=np.asarray(seam_kk, float), section_offsets=np.asarray(offs, float),
        warp_form=warp_form, n_marks=int(inl.sum()), n_marks_rejected=int((~inl).sum()),
        resid_rms_px=rms, resid_max_px=rmax, resid_rms_constant_period_px=rms_const,
        period_src_px=float(coeffs[1]) if deg >= 1 else float("nan"), drift_pct=drift,
        seam_steps_px=np.diff(offs) if offs.size > 1 else np.empty(0),
        seam_x_src=np.asarray(seam_kk * p_ref[0] + p_ref[1] + k_ref * p_ref[0], float),
        unwrap_phase=float(phase), unwrap_prior_deg=float(prior_alpha_deg),
        unwrap_half_ambiguity_deg=float(half_amb), sides_used=sides,
        knots_canvas=knots_c, knots_source=knots_s, notes=tuple(notes),
    )
    if not model.is_monotonic():
        raise LadderError("the fitted mark warp is not monotonic across the canvas -- refusing "
                          "(a folded x mapping would duplicate terrain)")
    return model
