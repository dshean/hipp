"""
Copyright (c) 2026 HIPP developers
Description: Mark-based restitution -- the printed scan-angle ladder carries the x
    geometry, the collimation lines (or the film edges) still carry y, and the
    exposure edges are demoted to a conservative crop plus QA.

    Confirmed architecture, dshean 2026-08-30 (kh9pc_stereo
    docs/per_frame_canvas_design_2026-08-30.md secs 10-12):

      "we want to get to the point where left/right edges are not important for
       success, and we are conservatively cropping to eliminate unexposed areas,
       but acceptable to allow more empty/unexposed columns ... and we've nailed
       the scan angle constraints, so the cx/alpha is properly set for the
       variable image widths.  At the same time, the deviations of apparent
       exposed width are a valuable qa/qc check."

    The canonical order (dshean 2026-08-30, design sec 13 -- "do this simply and
    efficiently"), with NO masking or nodata before the product step:

      1. section seam matching (with checks)  -> the seam QA metrics
      2. merge / mosaic                       -> the raw film mosaic
      3. collimation lines + top/bottom edges  on RAW film  (the parent strategy)
      4. rectify from the line fit            -> STRAIGHT RAILS
         (:func:`rectified_rail_strip`; the same vertical shear the final warp
         applies, done for the two rail bands step 5 reads, so restitution still
         runs exactly one full-frame warp)
      5. scan-angle marks at a CONSTANT row offset on the straight rails --
         no tilted-band machinery, no per-block line evaluation
      6. only then: placement on the nominal canvas, conservative exposure crop,
         nodata.

    What this module delivers, per frame:

      * a canvas that is the nominal sector tier widened symmetrically (1.5 % by
        default) so no measured exposure is clipped, with ``cx = W/2`` the true
        alpha = 0 column and the nominal-sweep columns recorded for ``cam_gen
        --pixel-values``;
      * content PLACED at its measured sweep phase -- a short or rolled frame comes
        out as nodata in the right place, not re-centred (the A010/A055/first-frame
        failure class);
      * an x-resampling onto the FLEET-CALIBRATED per-sensor distortion shape, with
        this frame's own marks supplying only phase and period, which removes the
        measured ~160 px along-scan distortion at source instead of leaving it for
        ba2 motion terms and CSM jitter to absorb (docs/mark_fleet_2026-08-30.md);
      * y unchanged -- the collimation-line (or film-edge) rectification of the
        parent strategy, composed, not replaced;
      * merge seam steps measured and reported as a MOSAIC-QUALITY metric only
        (fleet: median 3.3 px, p90 7.5, max 23.8 over 1279 seams -- at the noise
        floor, an order of magnitude under the smooth distortion);
      * a conservative exposure crop biased toward KEEPING questionable columns:
        nodata makes no matches, over-cropping real content is the harmful
        direction.

    Two concrete strategies share one implementation:

      ``MarkStrategy``     -- on top of :class:`CollimationStrategy` (missions >= 1206)
      ``MarkPolyStrategy`` -- on top of :class:`PolyStrategy` (mission 1205, no lines);
                              anchored on the FILM-FRAME boundary the parent's
                              rupture scan already locks, not on the crop window.

    Both refuse loudly when the ladder cannot be fitted (``LadderError`` ->
    ``is_failed``); a frame whose x geometry is not pinned by the marks is not a
    mark restitution.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hipp.kh9pc.fiducial_patterns import theorical_spacing_from_pattern
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.kh9pc.restitution.base import Transformation, scan_scale
from hipp.kh9pc.restitution.collimation_strategy import CollimationStrategy
from hipp.kh9pc.restitution.mark_geometry import (
    LadderError,
    MarkTrain,
    MarkWarpModel,
    deg_per_mark_from_pattern,
    fit_mark_warp,
)
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy

logger = logging.getLogger(__name__)

#: timing_marks' train labels -> the designed spacing pattern they correspond to.
_LABEL_TO_PATTERN = {"sparse": "regulare_sparse", "mid": "regulare_mid"}

#: Measured first-frame regime offset, +6.35 deg in MOSAIC x for BOTH cameras
#: (fleet sec 5; MEASURED on ops323 from the printed labels, INFERRED elsewhere).
FIRST_FRAME_OFFSET_DEG = 6.35

_ENTITY_RE = re.compile(r"^(D3C)(\d{4})-(\d)(\d{5})([FA])(\d{3})$")


def camera_from_filepath(filepath: str | Path) -> str:
    """``"F"`` or ``"A"`` from the standardised entity id in the filename stem."""
    m = _ENTITY_RE.match(Path(filepath).stem)
    if m is None:
        raise LadderError(f"cannot parse a KH-9 entity id from {filepath!r}")
    return m.group(5)


def frame_number_from_filepath(filepath: str | Path) -> str:
    m = _ENTITY_RE.match(Path(filepath).stem)
    if m is None:
        raise LadderError(f"cannot parse a KH-9 entity id from {filepath!r}")
    return m.group(6)


@dataclass
class SeamPositions:
    """Section-join columns in source px, from the merge provenance."""

    x_src: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    source: str = "none"
    fallback: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, bool))


@dataclass
class MarkOptions:
    """Knobs for mark-based placement.  Defaults are the confirmed design.

    Attributes
    ----------
    prior_alpha_deg, prior_x_src:
        The a-priori scan angle at a known source column, used ONLY to resolve the
        ladder's integer ambiguity.  ``None`` for ``prior_alpha_deg`` means "derive
        it": 0 deg for an interior frame, ``FIRST_FRAME_OFFSET_DEG`` for a frame
        numbered 001.  The fleet measured the exposure-centre prior good to NMAD
        0.126 deg (worst 0.389) on collimation-anchored frames.
    sweep_deg:
        Override the nominal tier width in degrees.  Open question: the marks
        measure the exposed sweep at 90.11-90.19 deg against a nominal 90.000.
        Leave ``None`` until an A/B settles it.
    band_dy_px, band_half_px:
        The rail search band as (top, bottom) OUTWARD offsets from the anchor line,
        plus a half-height.  Outward means away from the image centre, so both are
        positive for the collimation anchor and the ruling's "-600 top / +850
        bottom" (signed rows) is ``(600, 850)`` here; a NEGATIVE value searches
        inward, which is where the ladder sits relative to the FILM EDGE anchor the
        no-collimation missions use.  Measured ladder rows on the
        collimation-anchored frames: 834-879 outward at the bottom, 563-643 at the
        top.
    band_fallback_half_px:
        Widened half-height retried, loudly, when the ruled band yields no
        acceptable ladder (the PolyStrategy anchor is not the collimation line, so
        its ladder sits elsewhere).  ``None`` disables the retry.
    margin_max_dn, margin_max_bright_frac:
        The rail band must LOOK like unexposed film -- dark and flat.  The fleet
        found the PolyStrategy anchor putting the band on terrain, where a chance
        alignment of mountains fitted a "ladder" (ops251 F050).  A band that is too
        bright or too textured is refused before any ladder is fitted.
    exposure_pad_px:
        Outward bias on the conservative crop, in canvas px.  The shared edge
        detector errs INWARD by construction (base._featureless_edge returns the
        last content sample); under this architecture the bias flips, because empty
        columns are harmless and clipped content is not.  300 px is 0.09 % of a
        90 deg sweep against film margins measured at 2-17 % of the mosaic.
    """

    prior_alpha_deg: float | None = None
    prior_x_src: float | None = None
    sweep_deg: float | None = None
    canvas_widen: float | None = None        # None -> mark_geometry.DEFAULT_CANVAS_WIDEN
    k_ref_override: float | None = None
    # ladder fit (see mark_geometry: the SHAPE is fleet-calibrated, not per frame)
    per_frame_refine: str = "off"
    min_marks: int = 5
    min_cover_frac: float = 0.3
    max_resid_px: float = 40.0
    unwrap_warn_frac: float = 0.35
    period_ratio_tol: float = 0.02
    noise_px: float = 3.0
    distortion_shape_json: str | Path | None = None
    # detection
    sides: tuple[str, ...] = ("top", "bottom")
    band_dy_px: tuple[float, float] = (600.0, 850.0)
    band_half_px: float = 60.0
    band_fallback_half_px: float | None = 350.0
    score_min: float = 0.45
    block_w: int = 16384
    margin_max_dn: float = 120.0
    margin_max_bright_frac: float = 0.20
    redaction_max_std: float = 1.5           # USGS redaction fill is UNIFORM, film is noisy
    guard_zone_periods: float = 1.5          # near a frame start, require both neighbours
    #: A scan can contain part of the NEXT frame, rail marks included (ops327 A003;
    #: ops323 A003 is the extreme case).  The neighbour's ladder has the same PERIOD
    #: but its own PHASE, so an unbounded fit can lock onto it or blend the two.
    #: Marks are bounded to this frame's exposure window plus a pad, and a coherent
    #: out-of-window train is reported as a neighbour rather than silently dropped.
    frame_extent_pad_periods: float = 0.5
    # straightness: on the rectified strip a mark train MUST run along a constant row.
    # Any curvature is residual collimation-line / edge-fit error, not film (dshean
    # 2026-08-30) -- it would have caught the ops196 A002/F002 straightening failures.
    straightness_warn_px: float = 8.0
    straightness_fail_px: float = 40.0
    # seams -- QA only, never geometry
    use_seams: bool = True
    seam_provenance: str | Path | None = None
    seam_step_warn_px: float = 25.0
    # delivery
    exposure_pad_px: int = 300
    nodata: int = 0
    max_x_scale_mismatch_px: float = 2.0
    #: Refuse any strategy whose rail-band anchor is NOT tied to the film -- a
    #: fixed-height crop window can put the band inside the exposure, where terrain
    #: fits a "ladder" (the ops251 F050 failure, fleet sec 2).  Both strategies here
    #: anchor on the film (collimation lines / the rupture-scan film-frame boundary),
    #: so this is a guard for a future anchor rather than a live refusal.
    require_margin_anchor: bool = True


def seam_positions_from_provenance(path: str | Path) -> SeamPositions:
    """Section-join columns in mosaic px from ``<entity>_merge_provenance.json``.

    ``tools/merge_scene_sections.py`` records, per seam, the RANSAC translation
    ``tx`` of the section against its predecessor and the resulting ``overlap``.
    Cumulative ``tx`` is the left edge of each section in mosaic coordinates
    (``write_mosaic`` shifts the whole chain by ``-offset_x``, which is 0 for a
    left-to-right chain that starts at the origin); the join is taken at the middle
    of the overlap.  A chain whose cumulative positions do not stay inside the
    mosaic is rejected rather than guessed at -- these positions drive a QA metric,
    and a QA number attributed to the wrong column is worse than none.
    """
    path = Path(path)
    prov = json.loads(path.read_text())
    seams = prov.get("seams") or []
    width = (prov.get("merged_size") or [0, 0])[0]
    xs: list[float] = []
    fb: list[bool] = []
    x = 0.0
    for s in seams:
        x += float(s.get("tx", 0.0))
        ov = s.get("overlap")
        xs.append(x + (0.5 * float(ov) if ov else 0.0))
        fb.append(bool(s.get("fallback")))
    a = np.asarray(xs, float)
    if a.size and (not np.all(np.diff(a) > 0) or a.min() <= 0 or (width and a.max() >= width)):
        logger.warning("%s: seam chain is not a monotone left-to-right cover of [0, %s) "
                       "(%s) -- seam QA unavailable for this frame", path.name, width,
                       np.round(a, 0).tolist())
        return SeamPositions(source=f"{path.name} REJECTED")
    return SeamPositions(x_src=a, source=path.name, fallback=np.asarray(fb, bool))


def band_rows_for_side(side: str, opts: MarkOptions) -> tuple[float, float]:
    """(inner, outer) rail-band offsets from the anchor line, signed OUTWARD."""
    dy = opts.band_dy_px[0] if side == "top" else opts.band_dy_px[1]
    return (dy - opts.band_half_px, dy + opts.band_half_px)


def rectified_rail_strip(raster, line, side: str, inner_px: float, outer_px: float,
                         x0: int, x1: int):
    """Read one rail band as a STRAIGHT strip: row r is source row ``line(x) + r``.

    Step 4 of the canonical order (dshean 2026-08-30 sec 13), restricted to the two
    rail bands that step 5 actually reads.  It is the same per-column vertical shear
    the final restitution applies, so on the strip the printed ladder sits at a
    CONSTANT row -- which is the whole point: no tilted-band machinery, no per-block
    line evaluation, no masking, no nodata.  Doing it for the rails only rather than
    for the whole frame keeps restitution to one full-frame warp.

    Returns ``(strip, row0)``: ``strip[r, i]`` is the film at source column
    ``x0 + i`` and offset ``row0 + r`` outward from the line.
    """
    import rasterio
    from rasterio.windows import Window

    sign = -1.0 if side == "top" else 1.0
    xs = np.arange(x0, x1, dtype=np.float64)
    base = np.asarray(line(xs), dtype=np.float64)
    off = np.arange(float(inner_px), float(outer_px), 1.0)         # outward offsets, 1 px
    if off.size < 4:
        return np.zeros((0, x1 - x0), np.uint8), float(inner_px)
    rows = base[None, :] + sign * off[:, None]                     # (H_strip, W_blk)
    with rasterio.open(raster) as src:
        r_lo = max(0, int(np.floor(rows.min())) - 2)
        r_hi = min(src.height, int(np.ceil(rows.max())) + 3)
        if r_hi - r_lo < 4:
            return np.zeros((0, x1 - x0), np.uint8), float(inner_px)
        win = src.read(1, window=Window(x0, r_lo, x1 - x0, r_hi - r_lo))
    if win.dtype != np.uint8:
        w = win.astype(np.float32)
        lo, hi = float(np.nanmin(w)), float(np.nanmax(w))
        win = np.clip((w - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    # the offsets step by exactly 1 px, so the fractional shift is CONSTANT down a
    # column: the resample is a per-column 1-D linear interpolation, which numpy
    # does directly (cv2.remap caps both dimensions at SHRT_MAX and a rail strip is
    # wider than that).
    ri = np.floor(rows).astype(np.int64) - r_lo
    frac = (rows - np.floor(rows)).astype(np.float32)
    np.clip(ri, 0, win.shape[0] - 2, out=ri)
    cols = np.broadcast_to(np.arange(x1 - x0), rows.shape)
    a = win[ri, cols].astype(np.float32)
    b = win[ri + 1, cols].astype(np.float32)
    strip = np.rint(a + frac * (b - a)).astype(np.uint8)
    return strip, float(inner_px)


def strip_is_film_margin(strip, opts: MarkOptions) -> tuple[bool, dict]:
    """Does a straight rail strip look like unexposed film rather than terrain?

    The fleet's biggest single failure was an anchor that put the band INSIDE the
    exposed image, where mountains fitted a "ladder" (ops251 F050, VERIFIED from
    native crops).  Unexposed film margin is dark and flat; exposed terrain is not.
    """
    img = np.asarray(strip, dtype=np.float32)
    a = img[img > 0]                           # mosaic nodata is not evidence either way
    stats: dict = {"n_px": int(a.size)}
    if a.size < 4096:
        stats["verdict"] = "no readable band (all nodata)"
        return False, stats
    stats["median_dn"] = float(np.median(a))
    stats["bright_frac"] = float((a > opts.margin_max_dn).mean())
    # REDACTION vs film: a USGS redaction box is a uniform fill, unexposed film is
    # noisy at a similar DN.  Column-wise std separates them, and a redacted stretch
    # is a "no marks HERE" fact, not a bad anchor -- mission 1205 carries them in the
    # rail region (dshean 2026-08-30).
    col_std = img.std(axis=0)
    col_med = np.median(img, axis=0)
    flat = (col_std <= opts.redaction_max_std) & (col_med <= opts.margin_max_dn)
    stats["redacted_frac"] = float(flat.mean())
    ok = (stats["median_dn"] <= opts.margin_max_dn
          and stats["bright_frac"] <= opts.margin_max_bright_frac)
    if ok and stats["redacted_frac"] > 0.5:
        stats["verdict"] = f"film margin, {stats['redacted_frac']:.0%} REDACTED (uniform fill)"
    elif ok:
        stats["verdict"] = "film margin"
    else:
        stats["verdict"] = "EXPOSED CONTENT (band anchored on terrain)"
    return ok, stats


def detect_marks_on_straight_rail(raster, anchors, side: str, band_px: tuple[float, float],
                                  opts: MarkOptions, kinds: tuple[str, ...]
                                  ) -> tuple[list, dict]:
    """Template-match the glyphs on the RECTIFIED rail strip.

    Reuses ``timing_marks``' synthetic template bank and matcher; only the substrate
    changes -- a straight strip instead of a tilted band -- so ``dy`` is exactly the
    constant offset from the line and row clustering is trivial.
    """
    from hipp.kh9pc.restitution.timing_marks import Mark, _match_block, template_bank

    bank = template_bank(anchors.pitch_um[0], kinds)
    nms_px = int(0.6 * min(t.image.shape[0] for t in bank))
    inner, outer = band_px
    sign = -1.0 if side == "top" else 1.0
    line = anchors.line(side)
    marks: list[Mark] = []
    info = {"band_px": [inner, outer], "blocks": 0, "margin": None}
    import rasterio
    with rasterio.open(raster) as src:
        width = src.width
    ov = opts.block_w // 32
    margin_ok = None
    for x0 in range(0, width, opts.block_w):
        x1 = min(width, x0 + opts.block_w + ov)
        strip, row0 = rectified_rail_strip(raster, line, side, inner, outer, x0, x1)
        if strip.size == 0 or strip.shape[0] < 8:
            continue
        info["blocks"] += 1
        ok, stats = strip_is_film_margin(strip, opts)
        if info["margin"] is None or (stats.get("median_dn") or 0) > (
                info["margin"].get("median_dn") or 0):
            info["margin"] = stats            # keep the WORST (brightest) block's verdict
        margin_ok = ok if margin_ok is None else (margin_ok and ok)
        if not ok:
            continue
        for xl, yl, score, tid in _match_block(strip, bank, opts.score_min, nms_px):
            gx = x0 + xl
            if not (x0 <= gx < min(width, x0 + opts.block_w)):
                continue                       # the overlap belongs to the next block
            dy = sign * (row0 + yl)
            marks.append(Mark(x=float(gx), y=float(line(np.array([gx]))[0]) + dy, side=side,
                              score=float(score), kind=bank[tid].kind,
                              size_mm=bank[tid].size_mm, dy=float(dy)))
    info["margin_ok"] = bool(margin_ok)
    info["n_marks"] = len(marks)
    return marks, info


def train_straightness(marks, opts: MarkOptions, label: str = "") -> dict:
    """How straight a mark train runs on the RECTIFIED strip -- the y/rectification check.

    After the collimation-line rectification a printed train MUST lie along a
    constant row.  Any curvature in it is residual collimation-line / edge-fit error,
    not film (dshean 2026-08-30): this is the check that would have caught the
    ops196 A002/F002 straightening failures automatically.  The DENSE timing train is
    the better probe of the two -- hundreds of samples against the ladder's handful.

    Returns rms / max deviation about a robust straight line in ``dy``, plus the
    residual curvature (the quadratic coefficient scaled to the frame).
    """
    x = np.asarray([m.x for m in marks], float)
    dy = np.asarray([m.dy for m in marks], float)
    out = {"label": label, "n": int(x.size)}
    if x.size < 8:
        out["verdict"] = "too few marks to judge straightness"
        return out
    c = np.polyfit(x, dy, 1)
    for _ in range(4):
        r = dy - np.polyval(c, x)
        s_ = max(1.4826 * np.median(np.abs(r - np.median(r))), 0.5)
        k = np.abs(r - np.median(r)) < 3.0 * s_
        if k.sum() < 8:
            break
        c = np.polyfit(x[k], dy[k], 1)
    r = dy - np.polyval(c, x)
    q = np.polyfit(x, dy, 2)
    span = float(x.max() - x.min())
    out.update(rms_px=float(np.sqrt((r ** 2).mean())), max_px=float(np.abs(r).max()),
               tilt_px=float(c[0] * span), sagitta_px=float(q[0] * (span / 2.0) ** 2),
               dy_median=float(np.median(dy)))
    out["verdict"] = ("FAIL" if out["rms_px"] > opts.straightness_fail_px else
                      "WARN" if out["rms_px"] > opts.straightness_warn_px else "OK")
    return out


def detect_angle_ladder(raster, anchors, spec: KH9ImageSpec, opts: MarkOptions,
                        band_px=None) -> tuple[list[MarkTrain], list, list, dict]:
    """Detect rail marks on the straight rails and keep this mission's angle ladder.

    Row selection is by FIT, not by position: the train's class must be the pattern
    this mission prints on that rail (``KH9ImageSpec.{top,bottom}_fiducial_patterns``)
    and its fitted period must match the design within ``period_ratio_tol``; among
    the survivors the most inliers with the smallest residual wins.  The dense and
    time-word trains share the band and are rejected here by CLASS -- which is safe:
    the fleet found the per-mission expected class agreed with the fitted class on
    338 of 338 trains.

    Returns ``(trains, all_marks, chosen_timing_marks_trains, info)``.
    """
    from hipp.kh9pc.restitution import timing_marks as tm

    kinds = ("disk",) if spec.fiducial_type == "disk" else ("wheel",)
    trains_out: list[MarkTrain] = []
    chosen: list = []
    marks_all: list = []
    info: dict = {"band_px": {}, "detect": {}, "straightness": {}}
    for side in opts.sides:
        band = band_px if band_px is not None else band_rows_for_side(side, opts)
        info["band_px"][side] = band
        marks, dinfo = detect_marks_on_straight_rail(raster, anchors, side, band, opts, kinds)
        info["detect"][side] = dinfo
        if not dinfo["margin_ok"]:
            logger.warning("%s rail band is not unexposed film (%s; median DN %s, bright frac %s)"
                           " -- marks from it are not trusted", side,
                           (dinfo["margin"] or {}).get("verdict"),
                           (dinfo["margin"] or {}).get("median_dn"),
                           (dinfo["margin"] or {}).get("bright_frac"))
            continue
        marks_all.extend(marks)
        try:
            expected = deg_per_mark_from_pattern(
                spec.bottom_fiducial_patterns[0] if side == "bottom"
                else spec.top_fiducial_patterns[0])
        except LadderError:
            info["detect"][side]["ladder"] = "this rail prints no angle ladder"
            continue
        info["detect"][side]["expected_deg_per_mark"] = expected
        # bound the marks to THIS frame before any fit: a scan can carry part of the
        # next frame's rail, whose ladder has the same period and a different phase
        lo_x, hi_x = float(anchors.edges[0]), float(anchors.edges[1])
        pad = opts.frame_extent_pad_periods * theorical_spacing_from_pattern(
            "regulare_sparse" if expected == 5.0 else "regulare_mid")
        inside = [m for m in marks if lo_x - pad <= m.x <= hi_x + pad]
        outside = [m for m in marks if not (lo_x - pad <= m.x <= hi_x + pad)]
        if outside:
            info["detect"][side]["outside_frame_extent"] = len(outside)
        marks = inside
        best = None
        dense = None
        for row in tm.cluster_rows(marks, side):
            t = tm.fit_train(marks, side, row, anchors)
            if t is None:
                continue
            if t.label == "dense":
                if dense is None or t.n_inliers > dense.n_inliers:
                    dense = t
                continue
            if _LABEL_TO_PATTERN.get(t.label) is None:
                continue
            if deg_per_mark_from_pattern(_LABEL_TO_PATTERN[t.label]) != expected:
                continue                    # a real train, but not the angle ladder
            if not t.period_deg or abs(t.period_deg / expected - 1.0) > opts.period_ratio_tol:
                continue
            if best is None or (t.n_inliers, -t.resid_rms_px) > (best.n_inliers,
                                                                 -best.resid_rms_px):
                best = t
        info["detect"][side]["ladder"] = (
            None if best is None else
            {"row": best.row, "dy_px": best.dy_px, "label": best.label,
             "period_px": best.period_px, "n_inliers": best.n_inliers})
        # the DENSE timing train is picked too -- not for the warp (it carries the
        # time record, not angles) but because it is by far the best straightness
        # probe: hundreds of samples where the ladder has a handful.
        if dense is not None:
            dms = [m for m in marks if m.side == side and m.row == dense.row and m.inlier]
            info["straightness"][f"{side}_dense"] = train_straightness(
                dms, opts, f"{side} dense train")
            info["detect"][side]["dense_row"] = dense.row
            info["detect"][side]["dense_period_px"] = dense.period_px
        if best is None:
            continue
        ms = [m for m in marks if m.side == side and m.row == best.row and m.inlier]
        info["straightness"][f"{side}_ladder"] = train_straightness(
            ms, opts, f"{side} {expected:g} deg ladder")
        info["detect"][side]["ladder_row"] = best.row
        # START-OF-OPERATION / START-OF-FRAME marks live near a frame start and can
        # alias onto a ladder slot (dshean 2026-08-30), so inside a guard zone at
        # either end of the exposure a mark must be backed by a RUN of ladder slots.
        if opts.guard_zone_periods and len(ms) >= 3:
            ks_ = np.sort(np.asarray([m.k for m in ms], float))
            guard = opts.guard_zone_periods * float(best.period_px)
            lo_x, hi_x = float(anchors.edges[0]), float(anchors.edges[1])
            keep = []
            for m in ms:
                if not ((m.x - lo_x) < guard or (hi_x - m.x) < guard):
                    keep.append(m)
                    continue
                # inside the guard zone a mark must be backed by a RUN of ladder
                # slots continuing toward the frame interior.  Requiring a
                # neighbour on both sides would eat the legitimate outermost mark;
                # requiring two consecutive slots inward drops the isolated
                # start-of-operation / start-of-frame glyph, which is the confuser.
                inward = +1 if (m.x - lo_x) < guard else -1
                run = all(np.any(np.abs(ks_ - (m.k + inward * j)) < 0.25) for j in (1, 2))
                if run:
                    keep.append(m)
            dropped = len(ms) - len(keep)
            if dropped:
                info["detect"][side]["guard_zone_dropped"] = dropped
                logger.warning("%s rail: dropped %d isolated mark(s) inside %.0f px of a frame "
                               "edge -- no run of ladder slots continuing inward (start-of-"
                               "operation / start-of-frame marks alias there)",
                               side, dropped, guard)
            ms = keep
        if len(ms) < 3:
            continue
        if outside and best is not None:
            for m in outside:
                m.row = -1
            nb = tm.cluster_rows(outside, side)
            for row in nb:
                t = tm.fit_train(outside, side, row, anchors)
                if t is None or t.label != best.label or t.n_inliers < 3:
                    continue
                dphase = ((t.x0 - best.x0) / best.period_px) % 1.0
                dphase = min(dphase, 1.0 - dphase)
                info["detect"][side]["neighbour_ladder"] = {
                    "n_marks": t.n_inliers, "phase_offset_periods": float(dphase),
                    "x_first": t.x_first, "x_last": t.x_last}
                logger.warning("%s rail: a second %s train of %d marks sits outside this frame's "
                               "exposure window at a phase offset of %.3f period -- the NEXT "
                               "frame's ladder; excluded, not blended",
                               side, t.label, t.n_inliers, dphase)
                break
        trains_out.append(MarkTrain(side=side,
                                    k=np.array([float(m.k) for m in ms]),
                                    x=np.array([float(m.x) for m in ms]),
                                    deg_per_mark=float(expected),
                                    period_src=float(best.period_px)))
        chosen.append(best)
    return trains_out, marks_all, chosen, info


class _MarkPlacementMixin:
    """Mark detection, warp fitting and canvas placement, shared by both strategies.

    The parent strategy stays in charge of y: this mixin composes its
    transformation rather than replacing it, and verifies that the parent's warp is
    separable in x (a pure scan-pitch scale) before doing so.
    """

    # set by _fit_marks (plain class attributes -- the mixin is NOT a dataclass, so
    # these never become dataclass fields; immutable defaults only)
    mark_warp_: MarkWarpModel | None = None
    mark_error_: str | None = None
    mark_marks_: list | None = None
    mark_trains_: list | None = None
    mark_info_: dict | None = None

    # ---- pieces each concrete strategy supplies ---------------------------------
    def _mark_line_models(self) -> dict:
        raise NotImplementedError

    def _mark_detector(self):
        raise NotImplementedError

    def _mark_anchor_is_film_referenced(self) -> bool:
        """True when the anchor line is tied to the film, not to a fixed window."""
        raise NotImplementedError

    # ---- detection --------------------------------------------------------------
    def _mark_anchors(self):
        from hipp.kh9pc.restitution.timing_marks import RailAnchors

        models = self._mark_line_models()

        def mk(m):
            return lambda x: np.asarray(
                m.predict(np.asarray(x, dtype=float).reshape(-1, 1)), dtype=float).ravel()

        vd = self._mark_detector()
        pitch = self.scan_pitch_um or (7.0, 7.0)
        return RailAnchors(mk(models["top"]), mk(models["bottom"]),
                           tuple(int(v) for v in vd.edges_), (float(pitch[0]), float(pitch[1])),
                           f"{type(self).__name__} fit")

    def _detect_trains(self, opts: MarkOptions, band_px):
        return detect_angle_ladder(
            self.raster_filepath_, self._mark_anchors(),
            KH9ImageSpec.from_raster_filepath(self.raster_filepath_), opts, band_px)

    # ---- the warp ---------------------------------------------------------------
    def _mark_seams(self, opts: MarkOptions) -> SeamPositions:
        if not opts.use_seams:
            return SeamPositions(source="disabled")
        p = opts.seam_provenance
        if p is None:
            p = self.raster_filepath_.with_name(
                f"{self.raster_filepath_.stem}_merge_provenance.json")
        p = Path(p)
        if not p.exists():
            logger.info("no merge provenance at %s -- seam-step QA unavailable for this frame "
                        "(the warp does not use seams)", p)
            return SeamPositions(source="missing")
        try:
            return seam_positions_from_provenance(p)
        except Exception as exc:                                  # noqa: BLE001
            logger.warning("merge provenance %s unreadable (%r) -- seam QA skipped", p, exc)
            return SeamPositions(source="unreadable")

    def _mark_prior(self, opts: MarkOptions) -> tuple[float, float]:
        """(prior_alpha_deg, prior_x_src) for the ladder unwrap."""
        vd = self._mark_detector()
        left, right = (float(v) for v in vd.edges_)
        x = opts.prior_x_src
        if x is None:
            x = float(getattr(vd, "crop_center_", None) or 0.5 * (left + right))
        a = opts.prior_alpha_deg
        if a is None:
            first = frame_number_from_filepath(self.raster_filepath_) == "001"
            a = FIRST_FRAME_OFFSET_DEG if first else 0.0
            if first:
                logger.warning("%s: first frame -- unwrapping against the measured regime offset "
                               "%+.2f deg (MEASURED on ops323, INFERRED here)",
                               self.logging_prefix, a)
        return float(a), float(x)

    def _fit_marks(self) -> None:
        """Detect the ladder and fit the warp.  Records the failure, never hides it."""
        opts: MarkOptions = self.mark
        self.mark_warp_ = None
        self.mark_error_ = None
        self.mark_marks_, self.mark_trains_, self.mark_info_ = [], [], {}
        if opts.require_margin_anchor and not self._mark_anchor_is_film_referenced():
            self.mark_error_ = (
                "this strategy's anchor is a fixed-height crop window, not the film margin, so "
                "the rail band can land on exposed terrain (fleet 2026-08-30 sec 2: ops251 F050's "
                "'marks' were mountains). Supply a film-margin anchor, or set "
                "MarkOptions(require_margin_anchor=False) knowingly")
            logger.error("%s: MARK LADDER REFUSED -- %s", self.logging_prefix, self.mark_error_)
            return
        shape = None
        if opts.distortion_shape_json:
            from hipp.kh9pc.restitution.mark_geometry import load_distortion_shape
            shape = load_distortion_shape(opts.distortion_shape_json)
        prior_alpha, prior_x = self._mark_prior(opts)
        spec = KH9ImageSpec.from_raster_filepath(self.raster_filepath_)
        tier_w = int(spec.expected_size[0])          # the NOMINAL sweep, not the canvas
        camera = camera_from_filepath(self.raster_filepath_)
        seams = self._mark_seams(opts)
        attempts: list = [None]
        if opts.band_fallback_half_px:
            dy = 0.5 * (opts.band_dy_px[0] + opts.band_dy_px[1])
            attempts.append((dy - opts.band_fallback_half_px,
                             dy + opts.band_fallback_half_px))
        for i, band in enumerate(attempts):
            try:
                trains, marks, chosen, info = self._detect_trains(opts, band)
                if not trains:
                    raise LadderError("no row in the rail band fits this mission's angle ladder")
                warp = fit_mark_warp(
                    trains, tier_w, prior_alpha, prior_x, camera=camera,
                    sweep_deg=opts.sweep_deg, shape=shape,
                    **({} if opts.canvas_widen is None
                       else {"canvas_widen": opts.canvas_widen}),
                    seam_x_src=(seams.x_src if seams.x_src.size else None),
                    min_marks=opts.min_marks, min_cover_frac=opts.min_cover_frac,
                    max_resid_px=opts.max_resid_px, unwrap_warn_frac=opts.unwrap_warn_frac,
                    k_ref_override=opts.k_ref_override,
                    per_frame_refine=opts.per_frame_refine, noise_px=opts.noise_px)
            except LadderError as exc:
                if i + 1 < len(attempts):
                    logger.warning("%s: the ruled rail band gave no ladder (%s) -- retrying the "
                                   "wide %s px band", self.logging_prefix, exc, attempts[i + 1])
                    continue
                self.mark_error_ = str(exc)
                logger.error("%s: MARK LADDER REFUSED -- %s", self.logging_prefix, exc)
                return
            st = [v for v in (info.get("straightness") or {}).values() if "rms_px" in v]
            if not st:
                logger.warning("%s: no mark train carried enough marks to judge STRAIGHTNESS "
                               "(%s) -- the y rectification is unchecked on this frame",
                               self.logging_prefix,
                               "; ".join(f"{k}: {v.get('verdict')}"
                                         for k, v in (info.get("straightness") or {}).items())
                               or "no trains at all")
            worst = max(st, key=lambda v: v["rms_px"]) if st else None
            if worst and worst["verdict"] == "FAIL":
                self.mark_error_ = (
                    f"the mark trains are NOT straight after rectification: {worst['label']} "
                    f"deviates {worst['rms_px']:.1f} px rms ({worst['max_px']:.1f} px max, "
                    f"sagitta {worst['sagitta_px']:+.1f} px) over {worst['n']} marks. That is "
                    "residual collimation-line / edge-fit error, not film -- the y "
                    "rectification is wrong for this frame")
                logger.error("%s: STRAIGHTNESS FAIL -- %s", self.logging_prefix, self.mark_error_)
                return
            if worst and worst["verdict"] == "WARN":
                logger.warning("%s: STRAIGHTNESS WARN -- %s deviates %.1f px rms (%.1f px max) "
                               "after rectification; the collimation fit is marginal",
                               self.logging_prefix, worst["label"], worst["rms_px"],
                               worst["max_px"])
            self.mark_warp_, self.mark_marks_, self.mark_trains_, self.mark_info_ = \
                warp, marks, chosen, info
            self.mark_info_["band_attempt"] = "wide" if band else "ruled"
            self.mark_info_["seam_source"] = seams.source
            for note in warp.notes:
                logger.warning("%s: %s", self.logging_prefix, note)
            qa = warp.seam_qa or {}
            if qa.get("resolvable") and qa.get("max_abs_step_px", 0.0) > opts.seam_step_warn_px:
                logger.warning("%s: SEAM_STEP_WARN max |step| %.1f px (fleet norm: median 3.3, "
                               "p90 7.5, max 23.8 over 1279 seams) -- MOSAIC quality, not camera "
                               "geometry, and not corrected here",
                               self.logging_prefix, qa["max_abs_step_px"])
            logger.info("%s: %s ladder %g deg/mark, %d marks on %s, residual %.1f px rms about the "
                        "calibrated shape (a uniform grid leaves %.1f px), cx = %.1f, "
                        "period/design %.5f", self.logging_prefix, camera, warp.deg_per_mark,
                        warp.n_marks, "+".join(warp.sides_used), warp.resid_rms_px,
                        warp.resid_rms_uniform_px, warp.cx,
                        warp.scale_ratio_src_per_canvas
                        * scan_scale(self.scan_pitch_um, self.raster_filepath_)[0])
            return

    # ---- placement --------------------------------------------------------------
    def _check_base_is_x_separable(self, base: Transformation, sx: float) -> None:
        """The parent warp must be a pure scan-pitch scale in x for the composition.

        Both parents build their TPS from control points whose destination x is
        ``x_src * sx`` (poly_strategy.py:290, collimation_strategy.py:1027), so this
        holds by construction -- but it is the assumption the whole composition
        rests on, so it is measured, not trusted.
        """
        w, h = base.output_size
        gx = np.linspace(0.05 * w, 0.95 * w, 25) + base.crop_offset[0]
        gy = np.linspace(0.05 * h, 0.95 * h, 5) + base.crop_offset[1]
        X, Y = np.meshgrid(gx, gy)
        out = base.deformation(np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32))
        err = np.abs(out[:, 0] - X.ravel() / sx)
        if err.max() > self.mark.max_x_scale_mismatch_px:
            parent = next(c.__name__ for c in type(self).__mro__[1:]
                          if c.__name__.endswith("Strategy"))
            raise LadderError(
                f"the {parent} warp is not a pure scale in x (max |x_src - x_dst/sx| = "
                f"{err.max():.2f} px): the mark composition would corrupt y; refusing")

    def _compose_transformation(self, base: Transformation) -> Transformation:
        """Replace the parent's x mapping with the mark warp; keep its y exactly."""
        warp = self.mark_warp_
        assert warp is not None
        sx, _ = scan_scale(self.scan_pitch_um, self.raster_filepath_)
        self._check_base_is_x_separable(base, sx)
        _, out_h = base.output_size
        crop_top = base.crop_offset[1]
        base_def = base.deformation

        def deformation(coords: NDArray[np.float32]) -> NDArray[np.float32]:
            # coords arrive with crop_offset already added: (canvas_x, dst_y)
            xs = warp.source_x(coords[:, 0])
            probe = np.column_stack([xs * sx, coords[:, 1]]).astype(coords.dtype)
            ys = base_def(probe)[:, 1]
            return np.column_stack([xs, ys]).astype(coords.dtype)

        return Transformation(self.raster_filepath_, deformation,
                              crop_offset=(0.0, crop_top),
                              output_size=(warp.canvas_width, out_h))

    def _mark_keep_columns(self) -> tuple[int, int]:
        """Conservative kept-column range on the canvas, biased OUTWARD.

        The exposure edges no longer drive geometry; they only say which columns can
        hold film.  Everything outside becomes nodata, with ``exposure_pad_px`` of
        slack because empty columns cost nothing and clipped content does.
        """
        warp = self.mark_warp_
        vd = self._mark_detector()
        left, right = (float(v) for v in vd.edges_)
        lo = float(warp.canvas_x_of_source(left)) - self.mark.exposure_pad_px
        hi = float(warp.canvas_x_of_source(right)) + self.mark.exposure_pad_px
        return (int(max(0, np.floor(lo))), int(min(warp.canvas_width, np.ceil(hi))))

    def _apply_exposure_nodata(self, output_path: Path) -> None:
        from osgeo import gdal, gdal_array

        lo, hi = self._mark_keep_columns()
        ds = gdal.Open(str(output_path), gdal.GA_Update)
        if ds is None:
            logger.warning("could not reopen %s to apply the exposure crop", output_path)
            return
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(float(self.mark.nodata))
        dtype = gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)
        h = ds.RasterYSize
        for x0, w in ((0, lo), (hi, ds.RasterXSize - hi)):
            if w <= 0:
                continue
            for y0 in range(0, h, 4096):
                rows = min(4096, h - y0)
                band.WriteArray(np.full((rows, w), self.mark.nodata, dtype=dtype), x0, y0)
        band.FlushCache()
        ds.FlushCache()
        ds = None
        logger.info("exposure crop: kept canvas columns [%d, %d) of %d (%.2f deg .. %.2f deg), "
                    "%d px of outward slack", lo, hi, self.mark_warp_.canvas_width,
                    float(self.mark_warp_.alpha_deg(lo)), float(self.mark_warp_.alpha_deg(hi)),
                    self.mark.exposure_pad_px)

    # ---- public surface ---------------------------------------------------------
    def transform(self, output_path) -> None:
        """Write the restituted image, then apply the conservative exposure crop."""
        super().transform(output_path)
        try:
            self._apply_exposure_nodata(Path(output_path))
        except Exception as exc:                                  # noqa: BLE001
            logger.error("exposure crop failed on %s (%r) -- the product carries film margin; "
                         "REVIEW", output_path, exc)

    def mark_qc(self) -> dict:
        """Per-frame mark record for the QC json and the restitution sheet."""
        if self.mark_warp_ is None:
            return {"mark_ladder": "FAILED", "mark_error": self.mark_error_,
                    "mark_detect": (self.mark_info_ or {}).get("detect"),
                    "mark_straightness": (self.mark_info_ or {}).get("straightness")}
        w = self.mark_warp_
        vd = self._mark_detector()
        left, right = (float(v) for v in vd.edges_)
        lo, hi = self._mark_keep_columns()
        a0 = float(w.alpha_deg(w.canvas_x_of_source(left)))
        a1 = float(w.alpha_deg(w.canvas_x_of_source(right)))
        nominal = w.tier_width / w.px_per_deg
        return {
            "mark_ladder": "OK", "mark_warp": w.to_dict(),
            "mark_band_px": (self.mark_info_ or {}).get("band_px"),
            "mark_band_attempt": (self.mark_info_ or {}).get("band_attempt"),
            "mark_detect": (self.mark_info_ or {}).get("detect"),
            # the y/rectification check: a printed train must run along a constant
            # row after the shear; curvature here is collimation-fit error
            "mark_straightness": (self.mark_info_ or {}).get("straightness"),
            "mark_seam_source": (self.mark_info_ or {}).get("seam_source"),
            "mark_trains": [{"side": t.side, "row": t.row, "dy_px": t.dy_px, "label": t.label,
                             "period_px": t.period_px, "period_deg": t.period_deg,
                             "n_inliers": t.n_inliers, "n_marks": t.n_marks,
                             "resid_rms_px": t.resid_rms_px} for t in (self.mark_trains_ or [])],
            "exposure_edges_src": [left, right],
            # QA, never geometry: how far the exposed sweep runs past the nominal
            # sector, and where the canvas centre sits against the marks' alpha = 0
            "exposure_alpha_deg": [a0, a1],
            "exposed_sweep_deg": abs(a1 - a0),
            "exposed_sweep_excess_pct": 100.0 * (abs(a1 - a0) / nominal - 1.0),
            "canvas_centring_error_deg": float(w.alpha_deg(0.5 * (
                w.canvas_x_of_source(left) + w.canvas_x_of_source(right)))),
            "keep_columns": [lo, hi], "canvas_cx": w.cx,
            # the frame's fitted mark period against the DESIGN, with the scan pitch
            # taken out (source px are ~0.16 % larger than canvas px at 6.9887 um)
            "period_ratio_vs_design": float(
                w.scale_ratio_src_per_canvas
                * scan_scale(self.scan_pitch_um, self.raster_filepath_)[0]),
            # cam_gen must pin the USGS footprint corners HERE, not at the image
            # corners: the canvas is wider than the nominal sweep on purpose
            "nominal_sweep_columns": list(w.nominal_sweep_columns),
            "tier_width": w.tier_width, "canvas_width": w.canvas_width,
            "canvas_widen": w.canvas_widen,
        }


@dataclass
class MarkStrategy(_MarkPlacementMixin, CollimationStrategy):
    """Collimation-line y rectification + mark-derived x placement.

    Missions >= 1206 (collimation lines present).  Everything the parent does is
    kept -- the line pair search, the pairing gate, the joint parallel refit, the
    edge refit and every QC figure -- and only the delivered x mapping changes.
    """

    mark: MarkOptions = field(default_factory=MarkOptions)

    def _mark_line_models(self) -> dict:
        return {"top": self._results["top"].model, "bottom": self._results["bottom"].model}

    def _mark_detector(self):
        return self.poly_strategy.vertical_detector

    def _mark_anchor_is_film_referenced(self) -> bool:
        return True                 # the fitted collimation lines are printed on the film

    @property
    def is_failed(self) -> bool:
        return bool(super().is_failed) or (self.is_fitted and self.mark_warp_ is None)

    def _fit(self, raster_filepath: Path) -> "MarkStrategy":
        super()._fit(raster_filepath)
        if not CollimationStrategy.is_failed.fget(self):
            self._fit_marks()
        return self

    def _compute_transformation(self) -> Transformation:
        if self.mark_warp_ is None:
            raise LadderError(f"no mark ladder for {self.raster_filepath_.name}: "
                              f"{self.mark_error_}")
        if self.output_width:
            logger.warning("%s: output_width=%d is IGNORED under mark placement -- the canvas is "
                           "the nominal sector tier so that cx = W/2 is the alpha = 0 column "
                           "(unset OUTPUT_WIDTH for mark restitution)",
                           self.logging_prefix, self.output_width)
        return self._compose_transformation(super()._compute_transformation())


@dataclass
class MarkPolyStrategy(_MarkPlacementMixin, PolyStrategy):
    """Film-edge y rectification + mark-derived x placement, for the no-line missions.

    Mission 1205 (nepal ops251) carries no usable collimation lines (dshean ruling
    2026-08-28), so the anchor is the FILM-FRAME BOUNDARY that ``PolyStrategy``'s
    rupture scan already locks -- ``_results[side].model``, the outer film edge, not
    the delivered crop.  That distinction is the whole finding: the fleet detector
    anchored its band on the restitution crop extent (a fixed 21771-row window),
    which lands ~1000 px inside the exposure on several nepal frames, and F050's
    "marks" were mountains.  Re-anchored on the measured film edge the same frames
    show clean printed labels ("251 050", "2-30", "02 APR") -- the 2026-08-30 18:00
    overturn of the "1205 has no labels" null.

    The band offsets are the one UNCALIBRATED piece: the ladder sits between the
    film edge and where a collimation line would be, i.e. INWARD of this anchor
    rather than outward, so the default searches a wide inward window and lets the
    class/period gate find the train.  Tighten it from the first real 1205 run.
    """

    #: ~500 px (top) / ~900 px (bottom) INWARD of the film edge -- dshean's own
    #: reading of the ops251 margins, 2026-08-30.  Negative = inward in the outward
    #: sign convention.  The wide fallback stays armed until this is confirmed on a
    #: real 1205 run.
    mark: MarkOptions = field(default_factory=lambda: MarkOptions(
        band_dy_px=(-500.0, -900.0), band_half_px=150.0, band_fallback_half_px=400.0))

    def _mark_line_models(self) -> dict:
        # PolyStrategy._process_side fits the FILM-FRAME boundary (the rupture scan
        # the collimation search is itself anchored on), not the exposure edge and
        # not the crop window -- which is exactly the margin anchor the marks need.
        return {"top": self._results["top"].model, "bottom": self._results["bottom"].model}

    def _mark_detector(self):
        return self.vertical_detector

    def _mark_anchor_is_film_referenced(self) -> bool:
        return True                 # the rupture scan locks the film-frame boundary

    @property
    def is_failed(self) -> bool:
        return bool(super().is_failed) or (self.is_fitted and self.mark_warp_ is None)

    def _fit(self, raster_filepath: Path) -> "MarkPolyStrategy":
        super()._fit(raster_filepath)
        if not PolyStrategy.is_failed.fget(self):
            self._fit_marks()
        return self

    def _compute_transformation(self) -> Transformation:
        if self.mark_warp_ is None:
            raise LadderError(f"no mark ladder for {self.raster_filepath_.name}: "
                              f"{self.mark_error_}")
        if self.output_width:
            logger.warning("%s: output_width=%d is IGNORED under mark placement (see MarkStrategy)",
                           self.logging_prefix, self.output_width)
        return self._compose_transformation(super()._compute_transformation())


def plot_mark_ladder(strategy, strip_cols: int = 2600):
    """QC panel: the ladder on the straight rail, its residual, and the alpha = 0 column.

    Three panels, all in the frame the geometry is decided in:

      1. the RECTIFIED rail strip with every detected mark marked -- if the
         straightening is right the ladder is a horizontal row of glyphs;
      2. the mark residual against the fitted model, with the fleet-calibrated shape
         the model removed drawn behind it, so the size of the correction and the
         size of what is left are on the same axes;
      3. placement in DEGREES: the marks' alpha = 0 column, the delivered canvas
         centre, the exposure edges and the nominal sweep.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warp = getattr(strategy, "mark_warp_", None)
    entity = Path(strategy.raster_filepath_).stem
    fig, axes = plt.subplots(5, 1, figsize=(13, 15), dpi=110,
                             gridspec_kw={"height_ratios": [1.0, 1.1, 1.1, 1.3, 0.7]})
    if warp is None:
        axes[0].text(0.5, 0.5, f"MARK LADDER REFUSED\n{getattr(strategy, 'mark_error_', '')}",
                     ha="center", va="center", wrap=True, fontsize=11, color="firebrick",
                     transform=axes[0].transAxes)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{entity} -- scan-angle ladder", fontsize=13)
        fig.tight_layout()
        return fig

    opts: MarkOptions = strategy.mark
    side = warp.sides_used[0] if warp.sides_used else "bottom"
    band = (strategy.mark_info_ or {}).get("band_px", {}).get(side) or band_rows_for_side(side, opts)
    anchors = strategy._mark_anchors()
    import rasterio
    with rasterio.open(strategy.raster_filepath_) as src:
        width = src.width
    strip, row0 = rectified_rail_strip(strategy.raster_filepath_, anchors.line(side), side,
                                       band[0], band[1], 0, width)

    # 1. the straight strip, x max-pooled so ~50 px glyphs survive the squeeze
    ax = axes[0]
    if strip.size:
        pool = max(1, strip.shape[1] // strip_cols)
        w = (strip.shape[1] // pool) * pool
        img = strip[:, :w].reshape(strip.shape[0], w // pool, pool).max(axis=2)
        lo, hi = np.percentile(img, (2, 99.8))
        ax.imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1), aspect="auto",
                  interpolation="antialiased", interpolation_stage="rgba",
                  extent=[0, strip.shape[1], band[1], band[0]])
        m = np.asarray([x for x, s in zip(warp.marks_x, warp.marks_side) if s == side])
        ax.plot(m, np.full(m.shape, 0.5 * (band[0] + band[1])), "|", ms=14, mew=1.2,
                color="tab:orange", label=f"{len(m)} detected marks")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_ylabel("px outward\nfrom the line", fontsize=8)
    ax.set_title(f"{side} rail, rectified to straight (band {band[0]:.0f}-{band[1]:.0f} px outward)",
                 fontsize=9)

    # 2. mark ROW POSITIONS vs x -- the y / rectification check.  After the shear a
    #    printed train MUST run along a constant row; curvature here is residual
    #    collimation-line / edge-fit error, not film (dshean 2026-08-30).
    ax = axes[1]
    det = (strategy.mark_info_ or {}).get("detect", {})
    st = (strategy.mark_info_ or {}).get("straightness", {})
    allm = strategy.mark_marks_ or []
    for sd in ("top", "bottom"):
        for cls, key, col in (("ladder", "ladder_row", "tab:blue"),
                              ("dense", "dense_row", "tab:grey")):
            row = (det.get(sd) or {}).get(key)
            if row is None:
                continue
            mm = [m for m in allm if m.side == sd and m.row == row]
            if len(mm) < 3:
                continue
            xx = np.array([m.x for m in mm])
            yy = np.array([m.dy for m in mm])
            v = st.get(f"{sd}_{cls}", {})
            ax.plot(xx, yy - np.median(yy), ".", ms=2.5, color=col, alpha=0.8,
                    label=f"{sd} {cls} (n={len(mm)}, {v.get('rms_px', float('nan')):.1f} px rms, "
                          f"{v.get('verdict', '?')})")
    ax.axhline(0.0, color="k", lw=0.6)
    for y in (-opts.straightness_warn_px, opts.straightness_warn_px):
        ax.axhline(y, color="tab:orange", lw=0.6, ls=":")
    ax.grid(alpha=0.3)
    ax.set_xlabel("mosaic column (px)", fontsize=9)
    ax.set_ylabel("row offset from\nthe train median (px)", fontsize=8)
    ax.legend(fontsize=7, ncol=2, loc="upper center", framealpha=0.85)
    ax.set_title("STRAIGHTNESS -- mark row positions after rectification "
                 "(curvature = collimation-fit error, dotted = warn level)", fontsize=9)

    # 3. mark SEPARATIONS vs x -- the x / warp check, in raw spacings
    ax = axes[2]
    ax.axhline(warp.period_canvas, color="tab:orange", lw=0.8, ls="--",
               label=f"designed {warp.period_canvas:.0f} px")
    for sd in ("top", "bottom"):
        for cls, key, col in (("ladder", "ladder_row", "tab:blue"),
                              ("dense", "dense_row", "tab:grey")):
            row = (det.get(sd) or {}).get(key)
            if row is None:
                continue
            xx = np.sort(np.array([m.x for m in allm if m.side == sd and m.row == row]))
            if xx.size < 3:
                continue
            d = np.diff(xx)
            keep = d < 1.6 * np.median(d)          # skip gaps at missing slots
            ax.plot(0.5 * (xx[:-1] + xx[1:])[keep], d[keep], ".", ms=2.5, color=col,
                    alpha=0.8, label=f"{sd} {cls} (median {np.median(d[keep]):.1f} px)")
    ax.grid(alpha=0.3)
    ax.set_xlabel("mosaic column (px)", fontsize=9)
    ax.set_ylabel("separation (px)", fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper center", framealpha=0.85)
    ax.set_title(f"SPACING -- consecutive mark separations (the drift the warp removes; "
                 f"fitted period {warp.scale_src_per_deg * warp.deg_per_mark:.1f} px)", fontsize=9)

    # 4. the residual, with the shape that was removed behind it
    ax = axes[3]
    a = warp.alpha_deg(warp.canvas_x_of_k(warp.marks_k))
    resid = warp.mark_residuals(warp.marks_k, warp.marks_x)
    grid = np.linspace(a.min(), a.max(), 400)
    ax.plot(grid, warp.shape_px(grid) - np.interp(0.0, grid, warp.shape_px(grid)), "-",
            lw=1.2, color="0.55",
            label=f"calibrated {warp.camera} shape, removed ({warp.resid_rms_uniform_px:.0f} px rms)")
    for keep, style, lab in ((warp.marks_inlier, "o", "inlier"),
                             (~warp.marks_inlier, "x", "rejected")):
        if keep.any():
            ax.plot(a[keep], resid[keep], style, ms=4, mfc="none", color="tab:blue"
                    if lab == "inlier" else "firebrick", label=f"{lab} ({int(keep.sum())})")
    ax.axhline(0.0, color="k", lw=0.6)
    for y in (-opts.noise_px, opts.noise_px):
        ax.axhline(y, color="tab:green", lw=0.6, ls=":")
    qa = warp.seam_qa or {}
    for i, sx in enumerate(qa.get("seam_x_src", [])):
        ax.axvline(float(warp.alpha_deg(warp.canvas_x_of_source(sx))), color="tab:purple",
                   lw=0.5, alpha=0.6, label="merge seam" if i == 0 else None)
    ax.grid(alpha=0.3)
    ax.set_xlabel("scan angle alpha from the mark grid (deg)", fontsize=9)
    ax.set_ylabel("mark residual (source px)", fontsize=9)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", framealpha=0.85)
    ax.set_title(f"residual about the model: {warp.resid_rms_px:.2f} px rms, "
                 f"{warp.resid_max_px:.1f} px max  (noise floor {opts.noise_px:.1f} px, dotted)",
                 fontsize=9)

    # 5. placement, in degrees
    ax = axes[4]
    qc = strategy.mark_qc()
    e0, e1 = qc["exposure_alpha_deg"]
    half = warp.tier_width / warp.px_per_deg / 2.0
    ax.axvspan(e0, e1, color="tab:blue", alpha=0.15, label="exposed film")
    for v, c, lab in ((0.0, "tab:orange", "alpha = 0 from the marks == canvas cx"),
                      (-half, "k", "nominal sweep"), (half, "k", None),
                      (float(warp.alpha_deg(qc["keep_columns"][0])), "tab:green", "delivered crop"),
                      (float(warp.alpha_deg(qc["keep_columns"][1])), "tab:green", None)):
        ax.axvline(v, color=c, lw=1.4 if lab else 1.0, ls="-" if c != "k" else "--", label=lab)
    ax.set_xlim(-half * 1.12, half * 1.12)
    ax.set_yticks([])
    ax.set_xlabel("scan angle (deg)", fontsize=9)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", framealpha=0.85)
    ax.set_title(f"placement: exposed sweep {qc['exposed_sweep_deg']:.3f} deg "
                 f"({qc['exposed_sweep_excess_pct']:+.2f} % of nominal), canvas centring error "
                 f"{qc['canvas_centring_error_deg']:+.3f} deg", fontsize=9)

    seam_txt = ("seam steps n/a" if not qa.get("resolvable") else
                f"seam steps med {qa['median_abs_step_px']:.1f} / p90 {qa['p90_abs_step_px']:.1f} / "
                f"max {qa['max_abs_step_px']:.1f} px")
    fig.suptitle(
        f"{entity}  --  {warp.camera} {warp.deg_per_mark:g} deg ladder, {warp.n_marks} marks on "
        f"{'+'.join(warp.sides_used)}   |   canvas {warp.canvas_width} "
        f"(tier {warp.tier_width} x {warp.canvas_widen:.4f}), cx {warp.cx:.1f}, "
        f"period/canvas {warp.scale_ratio_src_per_canvas:.5f}   |   {seam_txt}", fontsize=11)
    if warp.notes:
        fig.text(0.005, 0.005, "  |  ".join(warp.notes)[:400], fontsize=6.5, color="firebrick")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig
