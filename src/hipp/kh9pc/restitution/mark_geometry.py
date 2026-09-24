"""
Copyright (c) 2026 HIPP developers
Description: Mark-based x-geometry for KH-9 PC restitution -- the pure-math core.

    The film rails carry a printed scan-angle ladder (5.24 in = 5 deg on missions
    <= 1213, 1.048 in = 1 deg from 1214).  That ladder is the CAMERA'S OWN ANGULAR
    RULER: mark ``k`` was exposed at scan angle ``(k - k_ref) * deg_per_mark``,
    where ``k_ref`` indexes the mark printed at alpha = 0.  This module turns a set
    of detected marks into

      * the alpha = 0 column on the canonical canvas (the true cx), and
      * an x-resampling that puts every mark at its designed angle,

    so that on the delivered canvas the scan angle maps LINEARLY to the column by
    construction and ``cx = width / 2`` is true for every frame regardless of where
    the exposure landed on the film.

    Nothing here reads a raster; everything is arrays in, arrays out, so the
    geometry is testable against synthetic ladders with a known truth.

    THE WARP FORM IS FIXED BY THE FLEET, NOT FITTED PER FRAME
    ---------------------------------------------------------
    ``docs/mark_fleet_2026-08-30.md`` (186 frames, 342 rails, 14 397 marks at
    native resolution, 5 missions 1973-1982, 24 scanner sessions) measured the
    along-scan distortion and found ONE curve per sensor:

      * residual against a uniform grid: 44.5 px, falling to 4.6 px after a single
        QUADRATIC and 3.3 px after a cubic -- smooth, no piecewise structure, no
        case for a spline;
      * the curvature is the same number everywhere -- fleet median
        -0.0745 px/deg^2, NMAD 0.0041 across every mission, epoch and scanner
        session -- and it crosses merge seams unbroken, so it is a property of the
        CAMERA, not of the scan or the mosaic;
      * F and A carry the same distortion mirrored (the A image is stored 180 deg
        rotated, VERIFIED from the printed angle labels), so the shape is
        calibrated per sensor in the CAMERA scan frame;
      * pooled over a sensor's rails, the scatter about the shared shape is
        3.5-3.7 px rms against a 3.0 px measurement noise floor.

    So the shape is a FIXED per-sensor quartic in alpha (orders 2-4), and each
    frame's own marks supply only the two things the shared shape cannot carry --
    PHASE (the alpha = 0 column) and PERIOD (the scale).  A per-frame shape fit is
    refuted: it would re-fit noise, and on a 30-deg block with a 5-deg ladder
    (ops196: 7 marks per rail) it is not even possible.  Two free parameters are.

    SEAM STEPS ARE QA, NOT GEOMETRY
    -------------------------------
    Residual join errors do appear as steps at the merge seams, but the fleet's
    true statistics over 1279 seams are median 3.3 px, p90 7.5 px, max 23.8 px --
    at the noise floor and an order of magnitude below the 40-50 px smooth drift.
    They stay OUT of the warp (a fixed low-order polynomial has no freedom to chase
    a step anyway) and are reported by :func:`seam_step_qa` as a merge-stage
    metric.  Only missions >= 1214 have enough marks per section to resolve them.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hipp.kh9pc.fiducial_patterns import Patterns, theorical_spacing_from_pattern
from hipp.kh9pc.kh9_image_spec import IMAGE_WIDTHS_PX

logger = logging.getLogger(__name__)

#: Sector tiers are 30 deg apart: IMAGE_WIDTHS_PX[i] spans (i + 1) * 30 deg.
TIER_DEGREES: tuple[float, ...] = tuple(30.0 * (i + 1) for i in range(len(IMAGE_WIDTHS_PX)))

#: Degrees per mark for the two printed ladders, keyed by designed px spacing so
#: the constants stay single sourced in ``fiducial_patterns``: SPARSE_SPACING
#: 19014 px = 5.24 in = 5 deg, MID_SPACING 3803 px = 1.048 in = 1 deg.
_DEG_PER_MARK_BY_SPACING: dict[int, float] = {
    theorical_spacing_from_pattern("regulare_sparse"): 5.0,
    theorical_spacing_from_pattern("regulare_mid"): 1.0,
}

#: Scan angle increases with mosaic x on F and decreases on A -- the A image is
#: stored 180 deg rotated (VERIFIED 2026-08-30 from the printed angle labels,
#: which run "1-45 ... 4+00" with increasing x on F and "1+45 ... 4+00" on A).
CAMERA_SIGN: dict[str, float] = {"F": +1.0, "A": -1.0}

#: Fleet-calibrated along-scan distortion, per sensor, in the CAMERA scan frame:
#: ``dx_px = a2*alpha^2 + a3*alpha^3 + a4*alpha^4`` with alpha in DEGREES.  Orders
#: 0 and 1 are not identifiable and are not part of the shape: a constant and a
#: term linear in alpha are exactly the per-frame phase and period.
#: Source: mark_fleet_2026-08-30 sec 3d, joint least squares over 134 F rails /
#: 6691 marks and 144 A rails / 6551 marks; scatter about the shape 3.7 / 3.5 px
#: rms against a 3.0 px noise floor.  Amplitude ~-160 px at alpha = +-45 deg.
DISTORTION_SHAPE: dict[str, tuple[float, float, float]] = {
    #        a2            a3             a4
    "F": (-0.090687, +1.0113e-4, +8.864e-6),
    "A": (-0.086740, -2.2053e-4, +7.316e-6),
}
#: Delivered canvas width as a multiple of the nominal sector tier.  The film is
#: exposed slightly beyond the nominal sector: the fleet marks measure the exposed
#: sweep at a median 1.0021 of the tier (p90 1.0074), with 171 of 172 frames under
#: 1.0123 and one ops323 outlier at 1.043.  A strict-tier canvas would clip real
#: content -- the harmful direction -- while extra nodata columns cost nothing but
#: disk (dshean 2026-08-30).  1.015 keeps every non-outlier frame whole.
DEFAULT_CANVAS_WIDEN: float = 1.015

DISTORTION_SHAPE_PROVENANCE = (
    "mark_fleet_2026-08-30 sec 3d (186 frames, 278 quality-gated rails, "
    "13 242 marks; 5 missions 1973-1982, 24 scanner sessions)")


class LadderError(Exception):
    """The scan-angle ladder could not be fitted confidently.

    Raised rather than silently falling back: a restitution whose x geometry is not
    pinned by the marks is NOT a mark restitution (fallback-refusal ruling, dshean
    2026-08-25).
    """


def load_distortion_shape(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """Read a re-calibrated shape from ``mark_distortion_model.json``.

    The file stores ``numpy.polyval`` order (descending powers, degree 4) per
    sensor; only orders 2-4 are used, and a non-zero order 0/1 is refused because
    it would silently double-count the per-frame phase and period.
    """
    d = json.loads(Path(path).read_text())
    out: dict[str, tuple[float, float, float]] = {}
    for cam, c in d.items():
        c = [float(v) for v in c]
        if len(c) != 5:
            raise LadderError(f"{path}: sensor {cam} is not a degree-4 polynomial: {c}")
        if abs(c[3]) > 1e-12 or abs(c[4]) > 1e-12:
            raise LadderError(f"{path}: sensor {cam} carries order 0/1 terms {c[3:]}, which are "
                              "the per-frame phase and period -- refusing to double-count them")
        out[cam] = (c[2], c[1], c[0])
    return out


def deg_per_mark_from_pattern(pattern: Patterns) -> float:
    """Designed scan angle between consecutive marks of *pattern*, in degrees.

    Raises ``LadderError`` for patterns with no regular designed spacing
    (``*_dense``, ``serialized_time_word``) -- those trains carry the time record,
    not an angular ruler.
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


def expected_canvas_width(tier_width_px: int, canvas_widen: float | None = None) -> int:
    """THE delivered mark-canvas width for a sector tier: ``round(tier * widen)``.

    The single width contract (review H1, 2026-08-30): the strategy sizes its canvas
    with this function (via :func:`fit_mark_warp`), so the ``block_prep`` spec gate
    and kh9_01's identity-dims check must call IT rather than re-deriving
    ``tier * 1.015`` -- an independent re-derivation that rounds differently would
    quarantine every mark frame on a 1 px mismatch.  ``None`` means
    ``DEFAULT_CANVAS_WIDEN``; pass ``1.0`` for the strict-tier A/B arm, whose width
    is exactly the tier.
    """
    widen = DEFAULT_CANVAS_WIDEN if canvas_widen is None else float(canvas_widen)
    if not np.isfinite(widen) or widen < 1.0:
        raise LadderError(f"canvas_widen {widen!r} is not a finite factor >= 1 -- a canvas "
                          "narrower than the tier would clip the nominal sweep itself")
    return int(round(int(tier_width_px) * widen))


def canvas_px_per_deg(canvas_width_px: int, sweep_deg: float | None = None) -> float:
    """Canvas columns per degree of scan angle.

    Derived from the CANVAS, not from the printed mark pitch, so the canvas edges
    sit at exactly +-sweep/2 -- which is what ``cam_gen`` assumes when it pins the
    USGS footprint corners at the image corners (cam_gen.cc:660-669).  The two
    canonical constants agree to 1.5e-5 (SPARSE_SPACING/5 = 3802.800 vs 342247/90
    = 3802.744 px/deg), so the choice costs 0.28 px on a 19014 px mark period.

    ``sweep_deg`` overrides the nominal tier width in degrees.  The marks measure
    the EXPOSED sweep at 90.11-90.19 deg (fleet median 2026-08-30) rather than
    90.000; whether the USGS footprint describes 90.000 deg or the measured sweep
    is an OPEN question, and this argument is the knob that settles it by A/B.  It
    is not a free parameter: changing it rescales every frame's placement.
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
        Fitted period in source px (diagnostic only; the warp refits it).
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
class MarkWarpModel:
    """Canvas-x -> source-x mapping that places the marks at their designed angles.

    Forward (design), exact by construction::

        canvas_x(k) = canvas_width / 2 + (k - k_ref) * period_canvas
        alpha(canvas_x) = (canvas_x - canvas_width / 2) / px_per_deg

    Inverse -- the direction ``remap_tif_blockwise`` calls::

        source_x(canvas_x) = x0 + scale_src_per_deg * alpha + shape(alpha) + refine(alpha)

    ``shape`` is the FIXED fleet-calibrated per-sensor distortion, expressed in the
    mosaic frame (even orders carry the camera sign, the cubic does not, because
    the A image is stored 180 deg rotated).  ``x0`` and ``scale_src_per_deg`` are
    the only per-frame parameters; ``refine`` is an opt-in quadratic escape that is
    zero unless a frame earns it.
    """

    k_ref: float
    deg_per_mark: float
    px_per_deg: float
    tier_width: int                          # the NOMINAL sweep, in canvas px
    canvas_width: int                        # tier_width widened so nothing clips
    x0: float                                # source x at alpha = 0
    scale_src_per_deg: float                 # source px per degree of scan angle
    camera: str = "F"
    shape: tuple[float, float, float] = (0.0, 0.0, 0.0)     # a2, a3, a4 (camera frame)
    refine: tuple[float, float, float] = (0.0, 0.0, 0.0)    # opt-in a2, a3, a4 on top
    # --- diagnostics; none of these is read by source_x ---
    n_marks: int = 0
    n_marks_rejected: int = 0
    resid_rms_px: float = float("nan")
    resid_max_px: float = float("nan")
    resid_rms_uniform_px: float = float("nan")
    #: fitted SOURCE px per degree over CANVAS px per degree.  This carries the scan
    #: pitch (source px are ~0.16 % larger than canvas px at 6.9887 um), so it is NOT
    #: the period-vs-design ratio -- multiply by the scan scale sx for that, which
    #: only a caller that knows the pitch can do (MarkStrategy.mark_qc does).
    scale_ratio_src_per_canvas: float = float("nan")
    unwrap_phase: float = float("nan")
    unwrap_prior_deg: float = float("nan")
    unwrap_half_ambiguity_deg: float = float("nan")
    #: machine-readable twin of the UNWRAP_MARGINAL note: True when the fitted
    #: phase sat far enough from a slot that the placement may be one mark out.
    #: A consumer must not have to grep ``notes`` for it (review LOW, 2026-08-30).
    unwrap_marginal: bool = False
    sides_used: tuple[str, ...] = ()
    cover_frac: float = float("nan")
    alpha_span_deg: tuple[float, float] = (float("nan"), float("nan"))
    # the marks AS USED: k is on the model's own (reference-train) index, which is
    # what k_ref counts in.  A caller must verify against these, never against a
    # single train's raw k -- two rails' detector origins differ by an integer.
    marks_k: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    marks_x: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    marks_side: tuple[str, ...] = ()
    marks_inlier: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, bool))
    shape_provenance: str = DISTORTION_SHAPE_PROVENANCE
    seam_qa: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    # ---- forward: design geometry -------------------------------------------------
    @property
    def period_canvas(self) -> float:
        """Designed mark period on the canonical canvas, in canvas px."""
        return self.deg_per_mark * self.px_per_deg

    @property
    def cx(self) -> float:
        """The alpha = 0 column on the delivered canvas -- the true cx.

        The canvas is widened symmetrically about the nominal sweep, so cx stays
        exactly the centre and alpha stays linear in the column; only the ANGULAR
        RANGE grows.
        """
        return self.canvas_width / 2.0

    @property
    def canvas_widen(self) -> float:
        """Delivered canvas width as a multiple of the nominal sector tier."""
        return self.canvas_width / float(self.tier_width)

    @property
    def nominal_sweep_columns(self) -> tuple[float, float]:
        """Canvas columns of the NOMINAL sweep edges (alpha = +- tier/2).

        On a widened canvas the USGS footprint corners no longer coincide with the
        image corners, so ``cam_gen`` must be given these columns explicitly via
        ``--pixel-values`` instead of relying on its corner-pinning default
        (cam_gen.cc:660-669).  They are exact by construction.
        """
        half = self.tier_width / 2.0
        return (self.cx - half, self.cx + half)

    def canvas_x_of_k(self, k) -> NDArray[np.floating]:
        """Designed canvas column of absolute ladder index *k*."""
        return self.cx + (np.asarray(k, float) - self.k_ref) * self.period_canvas

    def alpha_deg(self, canvas_x) -> NDArray[np.floating]:
        """Scan angle of a canvas column in degrees -- linear, by construction."""
        return (np.asarray(canvas_x, float) - self.cx) / self.px_per_deg

    # ---- the fixed shape ----------------------------------------------------------
    def shape_px(self, alpha_deg) -> NDArray[np.floating]:
        """The calibrated distortion at a MOSAIC-frame scan angle, in source px.

        In the camera frame the shape is ``a2 A^2 + a3 A^3 + a4 A^4`` with
        ``A = sign * alpha``; the displacement itself is also mirrored, so in the
        mosaic frame the even orders pick up the sign and the cubic does not.
        """
        a = np.asarray(alpha_deg, float)
        s = CAMERA_SIGN.get(self.camera, 1.0)
        out = np.zeros_like(a)
        for (a2, a3, a4) in (self.shape, self.refine):
            out = out + s * (a2 * a ** 2 + a4 * a ** 4) + a3 * a ** 3
        return out

    # ---- inverse: what the remap calls --------------------------------------------
    def source_x(self, canvas_x) -> NDArray[np.floating]:
        """Source column for a canonical-canvas column (the inverse warp)."""
        a = self.alpha_deg(canvas_x)
        return self.x0 + self.scale_src_per_deg * a + self.shape_px(a)

    def source_x_of_k(self, k) -> NDArray[np.floating]:
        """Where the model expects ladder index *k* to sit in the source."""
        return self.source_x(self.canvas_x_of_k(k))

    def canvas_x_of_source(self, x_src, samples: int = 8192, pad_frac: float = 0.15
                           ) -> NDArray[np.floating]:
        """Numeric inverse of :meth:`source_x` (monotone by gate).

        The grid runs PAST both canvas edges: a frame whose exposure reaches beyond
        the canvas must report the angle it actually reaches, not a value clamped to
        the edge -- that number is the exposed-sweep QA, and clamping it would hide
        exactly the frames worth looking at.  Delivery clips separately.
        """
        w = float(self.canvas_width)
        grid = np.linspace(-pad_frac * w, (1.0 + pad_frac) * w, samples)
        return np.interp(np.asarray(x_src, float), self.source_x(grid), grid)

    def is_monotonic(self, samples: int = 8192) -> bool:
        """True when the inverse warp is strictly increasing across the canvas."""
        src = self.source_x(np.linspace(0.0, float(self.canvas_width), samples))
        return bool(np.all(np.diff(src) > 0))

    # ---- verification -------------------------------------------------------------
    def mark_residuals(self, k, x) -> NDArray[np.floating]:
        """Source-px error of the model at the given (absolute k, source x) marks."""
        return np.asarray(x, float) - self.source_x_of_k(k)

    def placement_error_canvas_px(self, k, x) -> NDArray[np.floating]:
        """Where each mark LANDS minus where it is DESIGNED to land, in canvas px.

        The quantity the mark warp exists to drive to zero.  Its floor is the
        3.0 px per-rail measurement noise, not zero.
        """
        return self.canvas_x_of_source(np.asarray(x, float)) - self.canvas_x_of_k(k)

    def mark_table(self) -> list[dict]:
        """Per-mark QA rows: index, angle, source and canvas columns, residuals."""
        if self.marks_k.size == 0:
            return []
        a = (self.marks_k - self.k_ref) * self.deg_per_mark
        resid = self.mark_residuals(self.marks_k, self.marks_x)
        place = self.placement_error_canvas_px(self.marks_k, self.marks_x)
        return [{"k": float(k), "k_rel": float(k - self.k_ref), "alpha_deg": float(al),
                 "side": s, "x_src": float(x), "x_canvas_designed": float(cd),
                 "resid_src_px": float(r), "placement_err_canvas_px": float(pe),
                 "inlier": bool(i)}
                for k, al, s, x, cd, r, pe, i in zip(
                    self.marks_k, a, self.marks_side, self.marks_x,
                    self.canvas_x_of_k(self.marks_k), resid, place, self.marks_inlier)]

    def to_dict(self) -> dict:
        """JSON-safe summary for the per-frame QC record."""
        return {
            "k_ref": self.k_ref, "deg_per_mark": self.deg_per_mark,
            "px_per_deg": self.px_per_deg, "tier_width": self.tier_width,
            "canvas_width": self.canvas_width, "canvas_widen": self.canvas_widen,
            "nominal_sweep_columns": list(self.nominal_sweep_columns),
            "cx": self.cx, "period_canvas_px": self.period_canvas,
            "x0_src": self.x0, "scale_src_per_deg": self.scale_src_per_deg,
            "scale_ratio_src_per_canvas": self.scale_ratio_src_per_canvas,
            "camera": self.camera,
            "shape": list(self.shape), "refine": list(self.refine),
            "shape_provenance": self.shape_provenance,
            "n_marks": self.n_marks, "n_marks_rejected": self.n_marks_rejected,
            "resid_rms_px": self.resid_rms_px, "resid_max_px": self.resid_max_px,
            "resid_rms_uniform_px": self.resid_rms_uniform_px,
            "cover_frac": self.cover_frac, "alpha_span_deg": list(self.alpha_span_deg),
            "unwrap_phase": self.unwrap_phase, "unwrap_prior_deg": self.unwrap_prior_deg,
            "unwrap_half_ambiguity_deg": self.unwrap_half_ambiguity_deg,
            "unwrap_marginal": bool(self.unwrap_marginal),
            "sides_used": list(self.sides_used), "seam_qa": self.seam_qa,
            "notes": list(self.notes),
        }


def seam_step_qa(model: MarkWarpModel, k, x, seam_x_src, *, min_marks_per_section: int = 3,
                 noise_px: float = 3.0) -> dict:
    """Measure section-join steps in the mark residual.  A MERGE metric, not geometry.

    Fits ``resid ~ const + slope*alpha + sum_j step_j 1(x > seam_j)`` on the marks
    and reports the step statistics.  Nothing here feeds the warp: the fleet's true
    steps are median 3.3 px / p90 7.5 / max 23.8 over 1279 seams -- real, but at the
    measurement noise floor and an order of magnitude under the smooth drift, so
    correcting them is not the thing to fix.  Missions <= 1213 carry ~2 marks per
    section and cannot resolve individual steps; the function says so instead of
    returning noise.
    """
    k = np.asarray(k, float)
    x = np.asarray(x, float)
    seams = np.sort(np.asarray(seam_x_src, float))
    seams = seams[(seams > x.min()) & (seams < x.max())]
    out: dict = {"n_seams_inside": int(seams.size), "resolvable": False}
    if seams.size == 0:
        return out
    sec = np.searchsorted(seams, x, side="right")
    counts = np.bincount(sec, minlength=seams.size + 1)
    if counts.min() < min_marks_per_section:
        out["reason"] = (f"{int((counts < min_marks_per_section).sum())} of {seams.size + 1} "
                         f"sections carry < {min_marks_per_section} marks -- individual steps "
                         "are not resolvable at this ladder class")
        return out
    resid = model.mark_residuals(k, x)
    a = model.alpha_deg(model.canvas_x_of_k(k))
    cols = [np.ones_like(a), a] + [(sec == s).astype(float) for s in range(1, seams.size + 1)]
    A = np.column_stack(cols)
    sol, *_ = np.linalg.lstsq(A, resid, rcond=None)
    steps = np.diff(np.concatenate([[0.0], sol[2:]]))
    r = resid - A @ sol
    out.update(resolvable=True, seam_x_src=[float(s) for s in seams],
               steps_px=[float(s) for s in steps],
               median_abs_step_px=float(np.median(np.abs(steps))),
               p90_abs_step_px=float(np.percentile(np.abs(steps), 90)),
               max_abs_step_px=float(np.abs(steps).max()),
               n_significant_3sigma=int((np.abs(steps) > 3.0 * noise_px).sum()),
               rms_after_steps_px=float(np.sqrt((r ** 2).mean())))
    return out


def _robust_line(k, x, robust_iters: int = 6, clip_sigma: float = 4.0):
    """MAD-clipped least squares of ``x ~ p0*k + p1``; returns ``(p0, p1)``.

    The reference train's period/phase line is load-bearing three times over --
    it aligns the second rail onto the first's index, it converts the prior
    column into a ladder index for the unwrap, and it scales the coverage gate --
    so a plain ``polyfit`` lets one levered mark (a text hit, a start-of-frame
    glyph the guard zone did not reach) move the whole frame's placement by a
    mark period.  Clipping is the same MAD rule the phase/period fit uses.
    """
    k = np.asarray(k, float)
    x = np.asarray(x, float)
    A = np.column_stack([k, np.ones_like(k)])
    keep = np.ones(k.size, bool)
    sol = np.polyfit(k, x, 1)
    for _ in range(max(1, robust_iters)):
        sol, *_ = np.linalg.lstsq(A[keep], x[keep], rcond=None)
        r = x - A @ sol
        med = np.median(r[keep])
        mad = 1.4826 * np.median(np.abs(r[keep] - med))
        new = np.abs(r - med) <= max(clip_sigma * mad, 3.0)
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    sol, *_ = np.linalg.lstsq(A[keep], x[keep], rcond=None)
    return float(sol[0]), float(sol[1])


def _fit_phase_and_period(alpha, x, shape_px, robust_iters, clip_sigma):
    """Least squares of ``x ~ x0 + scale*alpha + shape(alpha)`` with MAD clipping."""
    y = np.asarray(x, float) - np.asarray(shape_px, float)
    A = np.column_stack([np.ones_like(alpha), alpha])
    keep = np.ones(len(alpha), bool)
    for _ in range(max(1, robust_iters)):
        sol, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        r = y - A @ sol
        med = np.median(r[keep])
        mad = 1.4826 * np.median(np.abs(r[keep] - med))
        new = np.abs(r - med) <= max(clip_sigma * mad, 3.0)
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    sol, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return float(sol[0]), float(sol[1]), keep


def fit_mark_warp(
    trains: list[MarkTrain],
    tier_width: int,
    prior_alpha_deg: float,
    prior_x_src: float,
    *,
    camera: str = "F",
    canvas_widen: float = DEFAULT_CANVAS_WIDEN,
    sweep_deg: float | None = None,
    shape: dict[str, tuple[float, float, float]] | None = None,
    seam_x_src=None,
    min_marks: int = 5,
    min_cover_frac: float = 0.3,
    max_resid_px: float = 40.0,
    resid_vs_uniform_margin_px: float | None = None,
    unwrap_warn_frac: float = 0.35,
    k_ref_override: float | None = None,
    per_frame_refine: str = "off",
    refine_min_marks: int = 40,
    refine_resid_factor: float = 3.0,
    noise_px: float = 3.0,
    robust_iters: int = 6,
    clip_sigma: float = 4.0,
) -> MarkWarpModel:
    """Fit the canvas-x -> source-x mark warp for one frame.

    Only TWO parameters are fitted per frame -- the phase (alpha = 0 column) and
    the period (scale).  The distortion SHAPE is the fleet-calibrated per-sensor
    constant (``DISTORTION_SHAPE``); a per-frame shape fit is refuted by the fleet
    (it would re-fit noise, and a 30-deg block on a 5-deg ladder has only ~7 marks).

    Parameters
    ----------
    trains:
        One or more :class:`MarkTrain` (top and/or bottom rail).  A train whose
        ladder phase disagrees with the reference by more than a quarter mark is
        dropped and recorded (a rail that locked onto the wrong train).
    tier_width:
        The NOMINAL sector width in px -- one of ``IMAGE_WIDTHS_PX``.  It sets the
        angular scale (``px_per_deg``); the delivered canvas is this widened by
        ``canvas_widen``, symmetrically, so ``cx = canvas_width / 2`` is still
        exactly alpha = 0.
    max_resid_px, resid_vs_uniform_margin_px:
        The two residual gates.  ``max_resid_px`` is the ABSOLUTE ceiling and is
        sweep-blind: a wrong-sensor shape leaves 107 px rms over a 90 deg sweep but
        only ~12 px over a 30 deg one, so on the short tiers it passes a 40 px
        ceiling while still being twice as wrong as removing no shape at all.  The
        RELATIVE gate closes that: the calibrated shape has to EARN its place, so a
        fit is refused when its residual exceeds what a plain uniform grid leaves
        (``resid_rms_uniform_px``, already computed) by more than
        ``resid_vs_uniform_margin_px`` -- ``None`` means ``noise_px``, which keeps a
        frame whose distortion is genuinely negligible (both residuals at the noise
        floor, their difference sampling noise) out of the refusal.  Judged on the
        SHARED-shape residual, before any ``per_frame_refine``: a free quadratic can
        absorb a wrong shape, and the question this gate asks is whether the shared
        shape belongs on this frame.
    canvas_widen:
        How much wider than the nominal sweep the delivered canvas is.  The film is
        exposed slightly BEYOND the nominal sector -- the marks measure the exposed
        sweep at a median 1.0021 of the tier, p90 1.0074, and 171 of 172 fleet
        frames under 1.0123 -- so a strict-tier canvas would clip real content, the
        harmful direction under dshean's conservative-crop principle.  Extra nodata
        columns are not.  ``1.0`` is the strict-tier A/B arm.
    prior_alpha_deg, prior_x_src:
        The a-priori scan angle (deg, MOSAIC frame) believed to sit at source column
        ``prior_x_src``.  Interior frame: 0 deg at the exposure centre -- the fleet
        measured that prior good to NMAD 0.126 deg with a worst case of 0.389 deg
        on collimation-anchored frames, so a 5-deg ladder is unambiguous and a
        1-deg ladder has ~0.1 deg of margin.  A first frame takes the measured
        regime offset (+6.35 deg in mosaic x for BOTH cameras, ops323).  Used ONLY
        to resolve the integer ambiguity; a wrong unwrap shifts placement by exactly
        one mark period and changes nothing else.
    camera:
        ``"F"`` or ``"A"``.  Selects the calibrated shape and its mirror.
    seam_x_src:
        Merge-seam columns, for the :func:`seam_step_qa` metric only.  Seam steps
        are NOT modelled in the warp (fleet sec 4).
    per_frame_refine:
        ``"off"`` (default) or ``"auto"`` -- the opt-in escape of fleet sec 8.4: a
        quadratic correction ON TOP of the shared shape, allowed only for a frame
        with >= ``refine_min_marks`` marks whose residual exceeds
        ``refine_resid_factor`` x the noise floor.

    Raises
    ------
    LadderError
        Too few marks, too little sweep coverage, a residual above ``max_resid_px``
        or above the uniform-grid residual it is supposed to beat, a folded warp, or
        trains that cannot be reconciled.  Refusing is the ruling: a silently
        degraded x geometry is worse than no product.
    """
    trains = [t for t in trains if len(t.k) >= 3]
    if not trains:
        raise LadderError("no ladder train with >= 3 marks")
    degs = {t.deg_per_mark for t in trains}
    if len(degs) != 1:
        raise LadderError(f"trains disagree on the ladder class: {sorted(degs)}")
    deg_per_mark = degs.pop()
    if camera not in CAMERA_SIGN:
        raise LadderError(f"unknown camera {camera!r} (expected F or A)")

    tier_width = int(tier_width)
    px_per_deg = canvas_px_per_deg(tier_width, sweep_deg)
    canvas_width = expected_canvas_width(tier_width, canvas_widen)   # THE width contract (H1)
    period_canvas = deg_per_mark * px_per_deg
    table = DISTORTION_SHAPE if shape is None else shape
    if camera not in table:
        raise LadderError(f"no calibrated distortion shape for camera {camera!r}")
    shape_coeffs = tuple(float(v) for v in table[camera])
    notes: list[str] = []

    # ---- 1. put every train on ONE relative index --------------------------------
    trains = sorted(trains, key=lambda t: (-len(t.k), t.side))
    ref = trains[0]
    p_ref = _robust_line(ref.k, ref.x, robust_iters, clip_sigma)   # x = p0*k + p1
    if not np.isfinite(p_ref).all() or p_ref[0] <= 0:
        raise LadderError(f"{ref.side} train has a non-increasing period")
    aligned: list[tuple[MarkTrain, float]] = [(ref, 0.0)]
    for t in trains[1:]:
        shift = float(np.median((np.asarray(t.x, float) - p_ref[1]) / p_ref[0]
                                - np.asarray(t.k, float)))
        if abs(shift - round(shift)) > 0.25:
            notes.append(f"{t.side} rail ladder is {shift - round(shift):+.2f} mark out of phase "
                         f"with the {ref.side} rail -- rail dropped")
            logger.warning("mark warp: dropping the %s rail (phase %+.2f mark)",
                           t.side, shift - round(shift))
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
    cover = (x_all[-1] - x_all[0]) * period_canvas / p_ref[0] / float(tier_width)
    if cover < min_cover_frac:
        raise LadderError(f"the ladder covers {cover:.2f} of the sweep (need {min_cover_frac:.2f})")

    # ---- 2. absolute index: unwrap the ladder against the prior -------------------
    k_at_prior = (float(prior_x_src) - p_ref[1]) / p_ref[0]
    k_ref_real = k_at_prior - float(prior_alpha_deg) / deg_per_mark
    half_amb = 0.5 * deg_per_mark
    if k_ref_override is not None:
        k_ref = float(k_ref_override)
        phase = float("nan")
        notes.append(f"k_ref fixed at {k_ref:g} by the caller (the prior would have given "
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

    alpha = (k_all - k_ref) * deg_per_mark

    # ---- 3. phase + period against the FIXED shape --------------------------------
    def _shape(a, coeffs):
        s = CAMERA_SIGN[camera]
        a2, a3, a4 = coeffs
        return s * (a2 * a ** 2 + a4 * a ** 4) + a3 * a ** 3

    x0, scale, inl = _fit_phase_and_period(alpha, x_all, _shape(alpha, shape_coeffs),
                                           robust_iters, clip_sigma)
    resid = x_all - (x0 + scale * alpha + _shape(alpha, shape_coeffs))
    rms = float(np.sqrt((resid[inl] ** 2).mean()))
    rmax = float(np.abs(resid[inl]).max())
    # what a uniform grid with NO shape would have left -- the number the warp beats
    x0u, scu, inu = _fit_phase_and_period(alpha, x_all, np.zeros_like(alpha),
                                          robust_iters, clip_sigma)
    rms_uniform = float(np.sqrt(((x_all - (x0u + scu * alpha))[inu] ** 2).mean()))
    rms_shared_shape = rms          # before any per-frame refine; the gate below judges THIS

    refine = (0.0, 0.0, 0.0)
    if per_frame_refine == "auto":
        if inl.sum() >= refine_min_marks and rms > refine_resid_factor * noise_px:
            # a QUADRATIC correction on top of the shared shape, never a free reshape
            A = np.column_stack([np.ones_like(alpha), alpha, alpha ** 2])
            sol, *_ = np.linalg.lstsq(A[inl], (x_all - _shape(alpha, shape_coeffs))[inl],
                                      rcond=None)
            s = CAMERA_SIGN[camera]
            refine = (float(sol[2]) * s, 0.0, 0.0)      # stored in the camera frame
            x0, scale = float(sol[0]), float(sol[1])
            resid = x_all - (x0 + scale * alpha + _shape(alpha, shape_coeffs)
                             + _shape(alpha, refine))
            rms = float(np.sqrt((resid[inl] ** 2).mean()))
            rmax = float(np.abs(resid[inl]).max())
            notes.append(f"PER_FRAME_REFINE applied: {inl.sum()} marks, residual about the shared "
                         f"shape exceeded {refine_resid_factor:g}x the {noise_px:g} px noise floor")
            logger.warning("mark warp: per-frame quadratic refinement applied (opt-in escape)")
        else:
            notes.append(f"per_frame_refine=auto declined ({int(inl.sum())} marks, {rms:.1f} px "
                         f"rms): the shared shape already fits at the noise floor")

    if rms > max_resid_px:
        raise LadderError(
            f"ladder residual {rms:.1f} px rms (max {rmax:.1f}) over {len(k_all)} marks exceeds "
            f"{max_resid_px:.0f} px against the calibrated {camera} shape -- this is not a clean "
            f"{deg_per_mark:g} deg ladder")
    # The absolute ceiling above is sweep-blind, so on a 30 deg tier a wrong-sensor
    # shape (~12 px rms) slips under a 40 px bar while being twice as wrong as
    # removing NO shape.  The calibrated shape must earn its place: it may not leave
    # more residual than the uniform grid it replaces (review H3, 2026-08-30).
    margin = noise_px if resid_vs_uniform_margin_px is None else float(resid_vs_uniform_margin_px)
    if rms_shared_shape > rms_uniform + margin:
        raise LadderError(
            f"the calibrated {camera} shape makes the fit WORSE than a plain uniform grid: "
            f"{rms_shared_shape:.1f} px rms with it against {rms_uniform:.1f} px without (margin "
            f"{margin:.1f} px) over {len(k_all)} marks spanning "
            f"{alpha.max() - alpha.min():.1f} deg -- the wrong sensor's shape, or not this "
            f"camera's {deg_per_mark:g} deg ladder")

    model = MarkWarpModel(
        k_ref=k_ref, deg_per_mark=deg_per_mark, px_per_deg=px_per_deg,
        tier_width=tier_width, canvas_width=canvas_width, x0=x0,
        scale_src_per_deg=scale, camera=camera,
        shape=shape_coeffs, refine=refine,
        n_marks=int(inl.sum()), n_marks_rejected=int((~inl).sum()),
        resid_rms_px=rms, resid_max_px=rmax, resid_rms_uniform_px=rms_uniform,
        scale_ratio_src_per_canvas=float(scale / px_per_deg), unwrap_phase=float(phase),
        unwrap_prior_deg=float(prior_alpha_deg), unwrap_half_ambiguity_deg=float(half_amb),
        unwrap_marginal=bool(np.isfinite(phase) and abs(phase) > unwrap_warn_frac),
        sides_used=sides, cover_frac=float(cover),
        alpha_span_deg=(float(alpha.min()), float(alpha.max())),
        marks_k=k_all, marks_x=x_all, marks_side=tuple(str(s) for s in side_all),
        marks_inlier=inl,
        shape_provenance=(DISTORTION_SHAPE_PROVENANCE if shape is None else "caller-supplied"),
        notes=tuple(notes),
    )
    if seam_x_src is not None:
        model.seam_qa = seam_step_qa(model, k_all, x_all, seam_x_src, noise_px=noise_px)
    if not model.is_monotonic():
        raise LadderError("the fitted mark warp is not monotonic across the canvas -- refusing "
                          "(a folded x mapping would duplicate terrain)")
    return model
