"""
Copyright (c) 2026 HIPP developers
Description: CollimationStrategy — refines the polynomial edge estimate using the physical
    collimation lines printed on the KH-9 film. Collimation lines are narrow bright bands
    whose peak can be located column-by-column with high accuracy, providing a stronger
    geometric reference than the film-edge rupture alone. The fixed known distance between
    collimation lines is used to set the output height precisely.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.warp import Resampling
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter1d
from skimage.transform import ThinPlateSplineTransform
from sklearn.linear_model import RANSACRegressor

from hipp.image import SubImage, remap_tif_blockwise
from hipp.kh9pc.restitution.base import _InwardEnvelopeModel, _featureless_edge, fit_ransac_poly, tps_from_estimate
from hipp.kh9pc.restitution.base import DEFAULT_OUTPUT_HEIGHT, RestitutionStrategy, Transformation
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy

logger = logging.getLogger(__name__)


@dataclass
class CollimationResult:
    """Fitted collimation line model and diagnostics for one side (top or bottom).

    Attributes
    ----------
    peaks_local:
        (N, 2) detected collimation peak coordinates in sub-image pixel space.
    peaks_global:
        (N, 2) same peaks converted to full-raster pixel coordinates.
    distortion:
        (N, 2) array of ``[x, deviation_from_mean]`` along the fitted collimation curve.
    inlier_ratio:
        Fraction of column peaks classified as inliers by RANSAC.
    model:
        Fitted ``RANSACRegressor`` wrapping the polynomial pipeline.
    sub_image:
        Narrow strip around the collimation line used for detection (kept for QC).
    """

    peaks_local: NDArray[np.integer]
    peaks_global: NDArray[np.integer]
    distortion: NDArray[np.floating]
    inlier_ratio: float
    model: RANSACRegressor
    sub_image: SubImage


@dataclass
class CollimationStrategy(RestitutionStrategy):
    """Restitution strategy using physical collimation lines printed on the film.

    Builds on a fitted ``PolyStrategy`` to narrow the search to a thin strip around
    each expected edge, then locates the collimation line peak column-by-column
    with sub-pixel accuracy. The result is fitted with a higher-degree RANSAC
    polynomial, and the fixed physical distance between lines (``collimation_line_dist``)
    sets the output height precisely rather than relying on the detected edge positions.
    Fails if the inlier ratio on either side falls below ``min_inliers_threshold``.

    .. warning:: Configure via CONSTRUCTOR KWARGS, never by assigning class
       attributes: annotated dataclass fields freeze their ``__init__``
       defaults at class creation, so ``CollimationStrategy.attr = x``
       silently never reaches new instances (found 2026-07-25 — the per-side
       crop offsets set that way by every production driver were inert and
       all fits ran the symmetric ``crop_offset_from_line`` default).
    """

    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)
    polynomial_degree: int = 5
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    grid_shape: tuple[int, int] = (100, 50)
    stride: int = 10
    refinement_fraction: float = 0.10
    # 0.10 (was 0.03): the line-to-edge inset is NOT constant across missions —
    # measured D3C1217 (1982): top line ~1150 px inside the edge; D3C1210
    # (1975): BOTH lines ~1300 px inside with a ~450 px cross-scan slope. The
    # old 3%-height strip (~740 px) missed those and locked secondary bands;
    # 10% (~2400-2500 px) covers every measured case with margin, and the
    # line's ~5x prominence over any mark-train/secondary band keeps the
    # per-column detection on it (David 2026-07-21: size the window to fit the
    # line instead of rescuing a too-small window).
    max_width_peak: int = 200
    collimation_line_dist: int = (
        21770  # known physical distance between top/bottom collimation lines at nominal scan resolution
    )
    separation_tolerance: float = 0.02  # accepted |top-bottom separation - collimation_line_dist| as a fraction
    # of collimation_line_dist. Real per-frame scan-scale deviation is ~0.3-0.9%
    # (D3C1217 F004: +0.28% line spacing, +0.9% detected width), so 2% separates
    # scan-scale variation from a genuinely wrong line lock.
    min_inliers_threshold: float = 0.5
    # Per-side INWARD shift of the detection strip start (px, default 0 = strip
    # anchored at the poly edge as before). Escape hatch for frames where a
    # bright OUTER secondary band inside the strip out-competes the true
    # collimation line in the per-column peak search and the separation
    # validator fails on an outer-band lock (first case: D3C1210-200323F013,
    # 2026-07-24 — BOTH sides locked outer bands, separation 24143 vs 21770
    # (+11%)). Set per frame by the driver (like crop_offset_from_line_*);
    # the separation validator still gates the result, so a wrong inset can
    # only produce a loud FAIL, never a silent wrong lock.
    search_inset_top: int = 0
    search_inset_bottom: int = 0
    # Exposure edge = the collimation line + a conservative fixed OUTWARD offset
    # (David 2026-07-22: "set the edge some distance outward from the
    # collimation line to remove the black frame; rather cut a few good pixels
    # in some sections than let redacted or black frame into the cropped
    # images"). The offset is the measured line->exposure-edge distance: across
    # the correctly-refit WA fits (D3C1217 F002/F003/A003/A004/A005 top+bottom
    # and F004 top; D3C1210 F020/F021/A022 top+bottom) the per-column edge sits
    # 182-377 px OUTWARD of its collimation line (top ~188-210, bottom
    # 182-377, mission-dependent). ``edge_offset_from_line`` = 180 is at/below
    # the smallest measured value, so the DERIVED edge (line + offset) always
    # lands at or INSIDE the true edge -> errs inward. The per-column texture
    # refit may only REFINE the edge to a transition whose distance OUTWARD from
    # THAT column's line falls in [edge_band_inner, edge_band_outer]: anything
    # nearer than 90 px (line halo / secondary band) or farther than 550 px is
    # rejected. 550 clears the largest measured edge (~380 px) with margin yet
    # is far short of the nearest margin artifact -- the film-frame boundary
    # (~955 px from the line) and F004's burned-in USGS logo (~1198 px from the
    # line, the row-24170 transition that bent the old free RANSAC fit down) --
    # so no margin artifact can ever enter the fit or bend the edge outward.
    edge_offset_from_line: int = 180
    edge_band_inner: int = 90
    edge_band_outer: int = 550
    # Round 2 (David 2026-07-22): the PRIMARY per-column edge test is simple and
    # DN-based -- walking OUTWARD from the line, the edge is the last real-content
    # row before a SUSTAINED FEATURELESS run: near-black (DN <= edge_black_dn;
    # the merged section-mosaic black frame is essentially 0, measured ~9) OR a
    # flat run (rolling std <= a small absolute value; the unexposed grey film
    # margin above the top edge is featureless at DN ~40, not black). This
    # accepts nearly every content column (F004: ~96-98/100 in-band vs 30-66/100
    # for the round-1 variance gate, which REFUSED obviously-decidable columns).
    # ``_variance_edge`` is kept only as a SECONDARY inner tiebreak. Because the
    # merged frames are mosaics of scan sections whose black frame starts at
    # DIFFERENT rows, the delivered edge is the INWARD ENVELOPE: the RANSAC
    # polynomial CLAMPED per column to the innermost detected transition
    # (``_InwardEnvelopeModel``), so it steps inward at section boundaries and
    # never crosses outward past a black block.
    edge_black_dn: int = 15
    # ---- DELIVERED CROP (David 2026-07-22 ~23:20 ruling) ----
    # "The collimation lines are the only true geometric markers ... rather
    # than trying to find an edge with a model, use a fixed offset from the
    # collimation line that cuts off all of the black rectangles; the model
    # line should be PARALLEL to the collimation line." The delivered
    # rectangle is cut at line -/+ this offset (OUTWARD, both edges) in the
    # warped frame -- no per-column edge/envelope model feeds the crop (those
    # remain as QC and a black-leak sentinel). Calibration (detqc 24883964,
    # ALL 10 frames of both WA missions): the per-column line->content-end
    # distance dips to ~87 px at section-mosaic steps (the round-1 "182-377
    # px" range was refit MEDIANS, not per-column minima — a 170 px offset
    # tripped the sentinel on every frame, worst 83 px intrusion on 99/100
    # columns of A021 top). 75 sits 12 px inside the smallest measured
    # content end: no measured section's black block survives, at a cost of
    # up to ~300 px of valid film on columns where content extends farthest
    # ("rather crop some valid pixels than include black rectangle in the
    # output" — the no-black constraint is categorical, area is secondary).
    crop_offset_from_line: int = 75
    # Per-side overrides (2026-07-23 v4 calibration: offsets are per date AND
    # per side — the exposed band beyond the line is asymmetric). None ->
    # fall back to the symmetric crop_offset_from_line. Set by the pipeline
    # driver from <date_dir>/crop_offset_v4.json (content-envelope p75 capped
    # by black p1 - margin). Crop policy under iteration with Luc + Amaury.
    crop_offset_from_line_top: int | None = None
    crop_offset_from_line_bottom: int | None = None
    output_width: int | None = None
    output_height: int | None = DEFAULT_OUTPUT_HEIGHT

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, CollimationResult] = {}
        self.__transformation_: Transformation | None = None
        self._separation_ok: bool = True

    @property
    def is_failed(self) -> bool:
        """True if either line's inlier ratio is below ``min_inliers_threshold`` or the
        top-bottom separation could not be reconciled with ``collimation_line_dist``."""
        if not self._separation_ok:
            return True
        return min(self.top_.inlier_ratio, self.bottom_.inlier_ratio) < self.min_inliers_threshold

    @property
    def top_(self) -> CollimationResult:
        """Fitted top collimation line result. Raises if ``fit()`` has not been called."""
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> CollimationResult:
        """Fitted bottom collimation line result. Raises if ``fit()`` has not been called."""
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def transformation_(self) -> Transformation:
        """TPS Transformation from collimation curves to horizontal lines (computed lazily)."""
        if self.__transformation_ is None:
            self.__transformation_ = self._compute_transformation()
        return self.__transformation_

    def _fit(self, raster_filepath: Path) -> Self:
        """Run PolyStrategy first, then detect collimation peaks in an edge-anchored strip.

        The strip extends ``refinement_fraction`` of the raster height INWARD
        from each poly edge — sized to cover the full measured range of
        line-to-edge insets (see the field comment), so a locally-corrupted
        edge estimate (e.g. a digital redaction band) still leaves the line
        inside the search strip. The two detected lines are then validated
        against the known physical separation ``collimation_line_dist``
        (invariant across fore/aft frames, unlike mark-train/edge distances);
        an unreconciled separation marks the fit FAILED rather than silently
        producing a wrong rectification.
        """
        if not self.poly_strategy.is_fitted or raster_filepath != self.poly_strategy.raster_filepath_:
            self.poly_strategy.fit(raster_filepath)
            if self.poly_strategy.is_failed:
                raise DetectionError(
                    f"poly edge model failed on {raster_filepath.name} — cannot "
                    "anchor collimation strip placement (review C2/M-1, 2026-08-23)")

        col_off, col_end = self.poly_strategy.vertical_detector.edges_
        window_width = self.poly_strategy.vertical_detector.detected_width_
        col_center = (col_off + col_end) // 2

        with rasterio.open(raster_filepath) as src:
            window_height = int(src.height * self.refinement_fraction)

            top_edge = int(self.poly_strategy.top_.model.predict(np.array([[col_center]])).flat[0])
            bot_edge = int(self.poly_strategy.bottom_.model.predict(np.array([[col_center]])).flat[0])

            # Both collimation lines lie inside the effective image content, not outside it.
            # top window starts at top_edge and extends downward; bottom window ends at bot_edge and extends upward.
            for side, window in {
                "top": Window(col_off, top_edge + self.search_inset_top,
                              window_width, window_height),
                "bottom": Window(col_off,
                                 bot_edge - window_height - self.search_inset_bottom,
                                 window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, resampling=Resampling.average, out_shape=self._out_shape(window))
                self._results[side] = self._process_side(sub_image, side)

            self._validate_separation()
            if not self._separation_ok:
                # PAIR RESCUE (dshean 2026-08-23, ops196 F003: the top
                # search locked a white furniture band at the mosaic top;
                # the USGS logo can do the same to the bottom). The lines
                # are a manufactured PAIR 21770 px apart, so re-anchor
                # each side's window from the OTHER side's fitted line +
                # the physical separation and keep the reconciliation
                # that restores the constant. A furniture band has no
                # partner at the right distance; the true line does.
                cands = {}
                # the partner line lies within +-tol of the anchor BY
                # PHYSICS, so the rescue window is exactly that tall --
                # a confuser >tol away (F003's furniture band, +1755 px)
                # cannot be inside it. A rescue fit must also clear the
                # inlier floor: a garbage window yields sparse peaks that
                # can land inside the separation tolerance by chance
                # (observed 0.17 inliers on the first F003 test).
                tol_px = int(self.separation_tolerance * self.collimation_line_dist)
                rescue_h = 2 * tol_px + 60
                for bad, good, sign in (("top", "bottom", -1), ("bottom", "top", 1)):
                    anchor = self._line_row(good) + sign * self.collimation_line_dist
                    row0 = int(anchor - rescue_h // 2)
                    row0 = max(0, min(row0, src.height - rescue_h))
                    win = Window(col_off, row0, window_width, rescue_h)
                    sub = SubImage(src, win, resampling=Resampling.average,
                                   out_shape=self._out_shape(win))
                    prev = self._results[bad]
                    self._results[bad] = self._process_side(sub, bad)
                    self._validate_separation()
                    if (self._separation_ok
                            and self._results[bad].inlier_ratio
                            >= self.poly_strategy.min_inliers_threshold):
                        cands[bad] = (self._results[bad],
                                      self._results[bad].inlier_ratio)
                    self._results[bad] = prev
                if cands:
                    bad = max(cands, key=lambda k: cands[k][1])
                    self._results[bad] = cands[bad][0]
                    self._validate_separation()
                    logger.warning(
                        "[CollimationStrategy] pair rescue: %s line re-anchored "
                        "from the %s line + physical separation (inliers %.2f)",
                        bad, "bottom" if bad == "top" else "top",
                        self._results[bad].inlier_ratio)
            if self._separation_ok:
                self._refit_edges_from_lines(src, col_off, window_width, window_height)

        return self

    def _refit_edges_from_lines(self, src: rasterio.DatasetReader, col_off: int, window_width: int, window_height: int) -> None:
        """Re-detect the EXPOSURE edges in windows ANCHORED at the fitted
        collimation lines, replacing the PolyStrategy edge results.

        The initial PolyStrategy scan is a DN-threshold rupture search from the
        raster boundary inward — it fires only on near-black background, and
        the film margin between the exposed area and the frame is NOT black,
        so it locks the film FRAME boundary (D3C1210 A022) or a redaction
        boundary (D3C1217 F004) instead of the exposed-area edge, no matter
        how the window is placed (verified: a line-anchored threshold scan
        reproduced the same locks +-3 px). The exposure edge is a TEXTURE
        transition — scene content is noisy, unexposed margin is smooth (the
        same variance principle that discriminates redaction fill from film) —
        so the refit scans each column OUTWARD from the line and takes the
        first SUSTAINED drop of rolling variance below a fraction of the
        content variance. Redacted samples are spliced out first (redaction
        fill is uniform and would fake an edge).

        The refit is ANCHORED to the line, not free: the edge is the line model
        plus a conservative fixed outward offset, and a per-column texture
        transition may only REFINE it when it lands in a tight band
        [edge_band_inner, edge_band_outer] px OUTWARD of that column's line
        (David 2026-07-22). A transition beyond the band -- F004's USGS logo
        burned into the bottom-left margin sits ~1198 px out and, being bright
        anti-aliased text, keeps the rolling variance HIGH past the true edge
        so the first sustained drop lands at the logo's lower rim; at the LEFT
        end of the span it had full leverage on the old free RANSAC fit and
        bent the bottom edge down ~1000 px -- is REJECTED. If too few columns
        survive the band gate (or the fit is low-inlier), the derived
        line+offset edge stands ON ITS OWN, erring INWARD (cutting a few good
        pixels) rather than admitting margin. The result REPLACES
        poly_strategy's DN-threshold edge (which locks the film-FRAME boundary,
        far outside the exposed area) so the poly_edges QC figure and
        downstream consumers see the line-anchored exposure edge."""
        from hipp.kh9pc.redaction_mask import redacted_region_mask
        from hipp.kh9pc.restitution.base import fit_ransac_poly
        from hipp.kh9pc.restitution.poly_strategy import PolyResult

        stride = self.poly_strategy.stride
        # Windows must contain the WHOLE line (it can slope ~450 px across the
        # scan — D3C1210; a median-anchored strip end CUT the line at half the
        # columns and the scan start missed it, David 2026-07-21) plus the
        # full outward reach. Per column, the scan starts at the line MODEL's
        # predicted row, never at the strip boundary. x uses PIXEL-CENTER
        # convention matching SubImage's resampled-column binning.
        ncols = self.poly_strategy.grid_shape[0]
        x_cols = col_off + (np.arange(ncols) + 0.5) * window_width / ncols
        for side in ("top", "bottom"):
            line_rows = self._results[side].model.predict(
                x_cols.reshape(-1, 1)).ravel()
            # Line-anchored DERIVED edge (line + conservative offset OUTWARD): the
            # authority when the texture refit finds too few in-band columns, and
            # the band centre the refit refines within. It errs INWARD by
            # construction (see the edge_offset_from_line comment).
            derived = self._derived_edge_result(x_cols, line_rows, side, col_off, window_width)
            lmin, lmax = int(line_rows.min()), int(line_rows.max())
            if side == "top":
                row0 = max(lmin - window_height, 0)
                row1 = min(lmax + 5 * stride, src.height)   # small pad past the line
            else:
                row0 = max(lmin - 5 * stride, 0)
                row1 = min(lmax + window_height, src.height)
            window = Window(col_off, row0, window_width, row1 - row0)
            if window.height < 12 * stride:
                logger.warning(
                    "[CollimationStrategy] no room to refit the %s exposure edge "
                    "(line at raster boundary) - using the derived line+offset edge", side)
                self.poly_strategy._results[side] = derived
                continue
            sub_image = SubImage(src, window, out_shape=(
                1, max(int(window.height) // stride, 1),
                self.poly_strategy.grid_shape[0]))
            redacted = redacted_region_mask(
                sub_image.band, max_dn=self.poly_strategy.background_threshold,
                dilate=3)
            nrows = sub_image.band.shape[0]
            row_scale = float(sub_image._scale[1])   # full-res px per strided local row
            res = []
            rejected_band = 0
            for c in range(sub_image.band.shape[1]):
                # per-column scan start = the line's own row at this column
                line_local = int(round(sub_image.to_local_y(line_rows[c])))
                line_local = min(max(line_local, 0), nrows - 1)
                # OUTWARD-ordered column vector (index 0 = line side) so the
                # detectors always "walk outward from the collimation line".
                if side == "top":
                    step = -1
                    out_vec = sub_image.band[: line_local + 1, c][::-1]
                    out_red = redacted[: line_local + 1, c][::-1]
                else:
                    step = 1
                    out_vec = sub_image.band[line_local:, c]
                    out_red = redacted[line_local:, c]
                # PRIMARY: last content row before a sustained featureless run
                # (near-black or flat). SECONDARY tiebreak: the round-1 variance
                # edge. Take the INNERMOST (most conservative) of the two — never
                # let the variance refusal reject a column the DN test decides.
                prim = _featureless_edge(out_vec, black_dn=self.edge_black_dn)
                sec = _variance_edge(out_vec, out_red, from_end=False)
                cands = [r for r in (prim, sec) if r is not None]
                if not cands:
                    continue
                r_out = min(cands)                       # innermost outward index
                local_row = line_local + step * r_out
                # BAND GATE (round 1, David 2026-07-22): keep only transitions a
                # plausible line->edge distance OUTWARD of THIS column's line, in
                # full-res px. Nearer than edge_band_inner = line halo / smooth
                # near-line content (premature flat trigger); farther than
                # edge_band_outer = redaction / black frame / F004's ~1198 px
                # logo rim. Out-of-band columns do not vote.
                dist = abs(local_row - line_local) * row_scale
                if not (self.edge_band_inner <= dist <= self.edge_band_outer):
                    rejected_band += 1
                    continue
                res.append((c, local_row))
            if len(res) < max(10, sub_image.band.shape[1] // 10):
                logger.warning(
                    "[CollimationStrategy] %s exposure-edge refit: only %d/%d "
                    "columns in-band (%d rejected out-of-band) - using the "
                    "derived line+offset edge",
                    side, len(res), sub_image.band.shape[1], rejected_band)
                self.poly_strategy._results[side] = derived
                continue
            ruptures_local = np.array(res)
            ruptures_global = sub_image.to_global(ruptures_local)
            model = fit_ransac_poly(
                ruptures_global[:, 0], ruptures_global[:, 1],
                degree=self.poly_strategy.polynomial_degree,
                residual_threshold=self.poly_strategy.ransac_residual_threshold,
                max_trials=self.poly_strategy.ransac_max_trials)
            inlier_ratio = float(model.inlier_mask_.mean())
            # Quality gate (adversarial review 2026-07-21): never install a
            # low-inlier refit — fall back to the derived line+offset edge, which
            # is always a clean line-anchored curve.
            if inlier_ratio < self.poly_strategy.min_inliers_threshold:
                logger.warning(
                    "[CollimationStrategy] %s exposure-edge refit REJECTED "
                    "(inliers %.2f < %.2f) - using the derived line+offset edge",
                    side, inlier_ratio, self.poly_strategy.min_inliers_threshold)
                self.poly_strategy._results[side] = derived
                continue
            # GEOMETRY vs CROP separation (David 2026-07-22 round 3): the edge
            # MODEL is the clean SMOOTH poly2 fit to the accepted inliers -- the
            # only thing that may ever feed a geometric warp, so it must NOT
            # step (a step in the edge = a step in local vertical scale = shear).
            # The conservative INWARD ENVELOPE (which steps inward at scan-section
            # black-frame boundaries so no black frame enters the crop) is a
            # SEPARATE ``crop_model`` product, used for the valid-pixel crop/mask
            # and drawn distinctly in QC -- never for the geometry.
            crop_model = _InwardEnvelopeModel.from_inliers(
                model, ruptures_global, side)
            x = np.linspace(col_off, col_off + window_width,
                            self.poly_strategy.grid_shape[0])
            y_pred = model.predict(x.reshape(-1, 1)).ravel()
            result = PolyResult(
                ruptures_local=ruptures_local,
                ruptures_global=ruptures_global.astype(int),
                distortion=np.column_stack([x, y_pred - y_pred.mean()]),
                inlier_ratio=inlier_ratio,
                model=model,
                sub_image=sub_image,
                crop_model=crop_model)
            derived_med = int(np.median(derived.model.predict(x.reshape(-1, 1))))
            new = int(np.median(y_pred))
            logger.info(
                "[CollimationStrategy] %s exposure edge refit (smooth poly2 model "
                "+ inward-envelope crop) from its line: median row %d (derived "
                "%d, %+d px), inliers %.2f, %d/%d columns in-band",
                side, new, derived_med, new - derived_med, inlier_ratio,
                len(res), sub_image.band.shape[1])
            self.poly_strategy._results[side] = result

    def _derived_edge_result(
        self,
        x_cols: NDArray[np.floating],
        line_rows: NDArray[np.floating],
        side: str,
        col_off: int,
        window_width: int,
    ) -> "PolyResult":
        """Exposure-edge PolyResult DERIVED from the collimation line model plus
        the conservative fixed offset (OUTWARD): edge(col) = line(col) -/+
        edge_offset_from_line for top/bottom. This is the fallback when the
        per-column texture refit has too few in-band columns (or is low-inlier);
        it errs INWARD -- cutting a few good pixels rather than admitting margin
        artifacts -- and reproduces the line's cross-scan slope. Synthetic
        ruptures lie exactly on the shifted line, so it fits with inliers=1 and
        the poly_edges QC figure shows a clean line-anchored edge."""
        from hipp.kh9pc.restitution.base import fit_ransac_poly
        from hipp.kh9pc.restitution.poly_strategy import PolyResult

        sign = -1.0 if side == "top" else 1.0   # top edge is OUTWARD = smaller row
        edge_rows = line_rows + sign * self.edge_offset_from_line
        model = fit_ransac_poly(
            x_cols, edge_rows,
            degree=self.poly_strategy.polynomial_degree,
            residual_threshold=self.poly_strategy.ransac_residual_threshold,
            max_trials=self.poly_strategy.ransac_max_trials)
        x = np.linspace(col_off, col_off + window_width, self.poly_strategy.grid_shape[0])
        y_pred = model.predict(x.reshape(-1, 1)).ravel()
        ruptures_global = np.column_stack([x_cols, edge_rows]).astype(int)
        return PolyResult(
            ruptures_local=ruptures_global.copy(),
            ruptures_global=ruptures_global,
            distortion=np.column_stack([x, y_pred - y_pred.mean()]),
            inlier_ratio=1.0,
            model=model,
            sub_image=self._results[side].sub_image)   # reuse the line strip for QC background

    def _out_shape(self, window: Window) -> tuple[int, int, int]:
        """Decimated read shape for a search window (rows strided, columns gridded)."""
        return (1, max(int(window.height) // self.stride, 1), self.grid_shape[0])

    def _line_row(self, side: str) -> int:
        """Median full-raster row of a fitted collimation line across the detected span."""
        left, right = self.poly_strategy.vertical_detector.edges_
        x = np.linspace(left, right, self.grid_shape[0])
        return int(np.median(self._results[side].model.predict(x.reshape(-1, 1))))

    def _validate_separation(self) -> None:
        """Validate the detected line separation against ``collimation_line_dist``.

        The spacing between the two collimation lines is a printed physical
        constant; the observed per-frame deviation is only the scan-scale
        variation (~0.2-0.9% measured across D3C1210/D3C1217). A separation
        outside ``separation_tolerance`` means at least one detection locked a
        secondary band, so the fit is marked FAILED (``is_failed``) instead of
        silently rectifying to a wrong height.
        """
        dist = self.collimation_line_dist
        tol = self.separation_tolerance * dist
        sep = self._line_row("bottom") - self._line_row("top")
        self._separation_ok = abs(sep - dist) <= tol
        if self._separation_ok:
            logger.info(
                "[CollimationStrategy] line separation %d vs expected %d (%+d px, tol %.0f) - OK",
                sep, dist, sep - dist, tol,
            )
        else:
            logger.error(
                "[CollimationStrategy] line separation %d deviates from expected %d "
                "by %+d px (tol %.0f) - marking fit FAILED",
                sep, dist, sep - dist, tol,
            )

    def _process_side(self, sub_image: SubImage, side: str) -> CollimationResult:
        """Detect collimation peaks column-by-column, fit a RANSAC polynomial, and compute the distortion curve."""
        _, w = sub_image.band.shape

        peaks_local = np.zeros((w, 2), dtype=int)
        for col in range(w):
            vec = sub_image.band[:, col]
            idx = detect_collimation_peak(vec, max_peak_width=self.max_width_peak // self.stride)
            peaks_local[col, 0] = col
            peaks_local[col, 1] = idx

        peaks_global = sub_image.to_global(peaks_local).astype(int)

        model = fit_ransac_poly(
            peaks_global[:, 0],
            peaks_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        inlier_ratio = float(model.inlier_mask_.mean())

        y_global_pred = model.predict(peaks_global[:, 0].reshape(-1, 1))
        y_distortion = y_global_pred - y_global_pred.mean()
        distortion = np.column_stack([peaks_global[:, 0], y_distortion])

        return CollimationResult(
            peaks_local=peaks_local,
            peaks_global=peaks_global,
            distortion=distortion,
            inlier_ratio=inlier_ratio,
            model=model,
            sub_image=sub_image,
        )

    def _crop_bound_model(self, side: str) -> object:
        """Conservative content-edge model for one edge: the exposure-edge
        refit's inward-envelope ``crop_model`` when present, else its (already
        conservative) derived line+offset ``model``. Since the fixed-offset
        crop ruling (David 2026-07-22) this feeds ONLY the black-leak sentinel
        and QC figures -- never the delivered crop, never the geometry."""
        r = self.poly_strategy._results[side]
        return r.crop_model if getattr(r, "crop_model", None) is not None else r.model

    def _compute_transformation(self) -> Transformation:
        """Build a TPS Transformation using the fixed physical collimation line separation."""
        left, right = self.poly_strategy.vertical_detector.edges_
        detected_width = self.poly_strategy.vertical_detector.detected_width_
        output_width = self.output_width or detected_width

        x = np.linspace(left, right, self.grid_shape[0])

        y_top_src = self.top_.model.predict(x.reshape(-1, 1))
        y_bot_src = self.bottom_.model.predict(x.reshape(-1, 1))

        top = int(np.median(y_top_src))
        bot = top + self.collimation_line_dist
        detected_height = bot - top

        y_top_dst = np.full_like(x, top)
        y_bot_dst = np.full_like(x, bot)

        src = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_src, y_bot_src))))
        dst = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_dst, y_bot_dst))))

        # inverse source destination (important). GEOMETRY: the warp uses ONLY
        # the smooth collimation line models -> no shear (David 2026-07-22 r3).
        deformation = tps_from_estimate(dst, src)

        # ---- SPEC-SNAPPED DELIVERED CROP (dshean 2026-08-23: "fine to
        # snap collimation crop to match the others") ----
        # Output canvas = KH9ImageSpec.expected_size, CENTERED on the
        # straightened line pair (y) and the detected content span (x) --
        # the FiducialStrategy convention -- so every frame in a block is
        # one geometry class (the 2026-07-22 fixed-offset +-75 px crop
        # delivered 21920 rows vs the spec 21771 and per-frame widths;
        # all six casa Collimation frames failed the per-frame spec gate,
        # rebuild24 25024328). crop_offset_from_line* now feed only the
        # QC overlay. Geometry/warp above is untouched.
        from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
        ew, eh = KH9ImageSpec.from_raster_filepath(self.raster_filepath_).expected_size
        y_center = (top + bot) / 2.0
        crop_top = int(y_center - eh / 2)
        crop_bot = crop_top + eh
        output_height = eh
        output_width = self.output_width or ew

        # Black-leak sentinel (non-fatal): the per-column content envelope no
        # longer drives the crop, but wherever it detects content ending
        # INSIDE the fixed rectangle, that section's black block would
        # survive -- warn with the worst intrusion so an out-of-family frame
        # (offset calibrated on measured WA frames) is caught in QC.
        sep = max(float(np.median(y_bot_src) - np.median(y_top_src)), 1.0)
        scale = self.collimation_line_dist / sep
        for side, line_src, line_dst, bound in (
                ("top", y_top_src, top, crop_top),
                ("bottom", y_bot_src, bot, crop_bot)):
            try:
                env = line_dst + scale * (
                    self._crop_bound_model(side).predict(x.reshape(-1, 1)).ravel()
                    - np.asarray(line_src).ravel())
            except Exception:
                continue
            intr = (env - bound) if side == "top" else (bound - env)
            worst = float(np.max(intr))
            if worst > 0:
                logger.warning(
                    "[CollimationStrategy] %s content envelope ends INSIDE the "
                    "fixed-offset crop by up to %d px on %d/%d sampled columns "
                    "-- black may survive; review poly_edges QC",
                    side, int(np.ceil(worst)), int((intr > 0).sum()), intr.size)

        logger.info(
            "[CollimationStrategy] spec-snapped delivered crop: canvas %dx%d "
            "centered on lines (line rows %d/%d -> crop rows [%d, %d])",
            output_width, output_height, top, bot, crop_top, crop_bot)

        x_center = (left + right) / 2.0
        crop_offset = (int(x_center - output_width / 2), crop_top)

        return Transformation(
            self.raster_filepath_,
            deformation,
            crop_offset=crop_offset,
            output_size=(output_width, output_height),
        )

    def transform(self, output_path: str | Path) -> None:
        """Write the restituted image using the collimation TPS warp."""
        tf = self.transformation_

        remap_tif_blockwise(
            tf.raster_filepath,
            output_path,
            tf.inverse_remap,
            tf.output_size,
            block_size=2**13,
            lowres_step=100,
        )


def _variance_edge(
    vec: NDArray[np.number],
    redacted: NDArray[np.bool_],
    from_end: bool,
    win: int = 9,
    sustain: int = 8,
    contrast_min: float = 2.0,
) -> int | None:
    """Exposure-edge row of one strip column via rolling-std transition.

    The column runs from the film side to the collimation-line side
    (``from_end=True`` = line at the END of the vector, i.e. the TOP strip;
    scan starts at the line end and moves outward). Content near the line is
    textured; the unexposed margin beyond the exposure edge is smooth. The
    threshold is the GEOMETRIC MEAN of the column's own two reference levels
    — median rolling std of the line-side third (content) and of the
    outward-end sixth (deep margin/frame) — and the column is REFUSED
    (None) when the contrast between them is below ``contrast_min``: an
    absolute floor misfired on smooth content (ice/ocean/cloud — the
    strided-average read shrinks content std ~3x, so a fixed 1.5-DN floor
    sat AT the content level and 3 chance-low windows faked an edge at the
    line; adversarial review 2026-07-21). ``sustain`` requires ~80 full-res
    px of continuous smoothness so transient content dips cannot fire. The
    edge is the FIRST position (scanning outward) with a sustained
    below-threshold run. Redacted samples (uniform fill — would fake a
    smooth margin) are spliced out; the returned index is in ORIGINAL
    column coordinates, or None when no edge is detectable.
    """
    keep = np.flatnonzero(~np.asarray(redacted, dtype=bool))
    if keep.size < 6 * win:
        return None
    v = np.asarray(vec, dtype=float)[keep]
    if from_end:
        v = v[::-1]
    # rolling std via cumulative sums (window `win`, valid positions)
    c1 = np.cumsum(np.insert(v, 0, 0.0))
    c2 = np.cumsum(np.insert(v * v, 0, 0.0))
    n = len(v) - win + 1
    mean = (c1[win:] - c1[:-win]) / win
    var = np.maximum((c2[win:] - c2[:-win]) / win - mean * mean, 0.0)
    std = np.sqrt(var[:n])
    content_lvl = float(np.median(std[: max(n // 3, win)]))
    outer_lvl = float(np.median(std[-max(n // 6, win):]))
    if content_lvl <= 0 or content_lvl < contrast_min * max(outer_lvl, 1e-6):
        return None                      # no content/margin contrast — refuse
    thresh = float(np.sqrt(content_lvl * max(outer_lvl, 1e-4)))
    below = std < thresh
    run = 0
    for i in range(n):
        run = run + 1 if below[i] else 0
        if run == sustain:
            j = i - sustain + 1          # first window of the sustained run
            idx = j                      # window start = transition row
            if from_end:
                idx = len(v) - 1 - idx
            return int(keep[idx])
    return None


def detect_collimation_peak(x: NDArray[np.number], max_peak_width: int, sigma: int = 2) -> int:
    """Locate the center of the collimation line in a 1D column profile.

    Smooths the profile with a Gaussian, then finds the gradient peak pair
    (rising + falling edge) whose separation is within ``max_peak_width``.
    Returns the index of the intensity maximum between the two gradient peaks,
    or the global maximum as a fallback when no compact peak is found.
    """
    smooth = gaussian_filter1d(x, sigma=sigma)

    grad = np.gradient(smooth)

    idx_max = np.argmax(grad)
    idx_min = np.argmin(grad)

    if abs(idx_max - idx_min) < max_peak_width and idx_max != idx_min:
        w_start = min(idx_max, idx_min)
        w_end = max(idx_max, idx_min)
        idx = np.argmax(smooth[w_start:w_end]) + w_start
    else:
        idx = np.argmax(smooth)  # fallback

    return int(idx)
