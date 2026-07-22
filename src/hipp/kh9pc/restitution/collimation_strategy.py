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
from hipp.kh9pc.restitution.base import fit_ransac_poly, tps_from_estimate
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
                "top": Window(col_off, top_edge, window_width, window_height),
                "bottom": Window(col_off, bot_edge - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, resampling=Resampling.average, out_shape=self._out_shape(window))
                self._results[side] = self._process_side(sub_image, side)

            self._validate_separation()
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
        fill is uniform and would fake an edge). RANSAC-fit as usual; the
        refit REPLACES poly_strategy's results so the poly_edges QC figure and
        downstream consumers see line-anchored exposure edges (David
        2026-07-21: redo the edge finding at a limited window from the line;
        do not trust poly_edges as originally plotted)."""
        from hipp.kh9pc.redaction_mask import redacted_region_mask
        from hipp.kh9pc.restitution.base import fit_ransac_poly
        from hipp.kh9pc.restitution.poly_strategy import PolyResult

        stride = self.poly_strategy.stride
        # Windows must contain the WHOLE line (it can slope ~450 px across the
        # scan — D3C1210; a median-anchored strip end CUT the line at half the
        # columns and the scan start missed it, David 2026-07-21) plus the
        # full outward reach. Per column, the scan starts at the line MODEL's
        # predicted row, never at the strip boundary.
        x_cols = np.linspace(col_off, col_off + window_width,
                             self.poly_strategy.grid_shape[0])
        for side in ("top", "bottom"):
            line_rows = self._results[side].model.predict(
                x_cols.reshape(-1, 1)).ravel()
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
                    "(line at raster boundary) - keeping the initial fit", side)
                continue
            sub_image = SubImage(src, window, out_shape=(
                1, max(int(window.height) // stride, 1),
                self.poly_strategy.grid_shape[0]))
            redacted = redacted_region_mask(
                sub_image.band, max_dn=self.poly_strategy.background_threshold,
                dilate=3)
            nrows = sub_image.band.shape[0]
            res = []
            for c in range(sub_image.band.shape[1]):
                # per-column scan start = the line's own row at this column
                line_local = int(round(sub_image.to_local_y(line_rows[c])))
                line_local = min(max(line_local, 0), nrows - 1)
                col_vec = sub_image.band[:, c]
                red_vec = redacted[:, c]
                if side == "top":
                    # rows [0 .. line_local]: line at the END, scan upward
                    r = _variance_edge(col_vec[: line_local + 1],
                                       red_vec[: line_local + 1], from_end=True)
                else:
                    # rows [line_local ..]: line at the START, scan downward
                    r = _variance_edge(col_vec[line_local:],
                                       red_vec[line_local:], from_end=False)
                    if r is not None:
                        r += line_local
                if r is not None:
                    res.append((c, r))
            if len(res) < max(10, sub_image.band.shape[1] // 10):
                logger.warning(
                    "[CollimationStrategy] %s exposure-edge refit: only %d/%d "
                    "columns yielded a texture edge - keeping the initial fit",
                    side, len(res), sub_image.band.shape[1])
                continue
            ruptures_local = np.array(res)
            ruptures_global = sub_image.to_global(ruptures_local)
            model = fit_ransac_poly(
                ruptures_global[:, 0], ruptures_global[:, 1],
                degree=self.poly_strategy.polynomial_degree,
                residual_threshold=self.poly_strategy.ransac_residual_threshold,
                max_trials=self.poly_strategy.ransac_max_trials)
            x = np.linspace(col_off, col_off + window_width,
                            self.poly_strategy.grid_shape[0])
            y_pred = model.predict(x.reshape(-1, 1)).ravel()
            result = PolyResult(
                ruptures_local=ruptures_local,
                ruptures_global=ruptures_global.astype(int),
                distortion=np.column_stack([x, y_pred - y_pred.mean()]),
                inlier_ratio=float(model.inlier_mask_.mean()),
                model=model,
                sub_image=sub_image)
            old = int(np.median(self.poly_strategy._results[side].model.predict(x.reshape(-1, 1))))
            new = int(np.median(y_pred))
            logger.info(
                "[CollimationStrategy] %s exposure edge refit (texture) from its "
                "line: median row %d -> %d (%+d px), inliers %.2f",
                side, old, new, new - old, result.inlier_ratio)
            self.poly_strategy._results[side] = result

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
        output_height = self.output_height or detected_height

        y_top_dst = np.full_like(x, top)
        y_bot_dst = np.full_like(x, bot)

        src = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_src, y_bot_src))))
        dst = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_dst, y_bot_dst))))

        # inverse source destination (important)
        deformation = tps_from_estimate(dst, src)

        # ---- CENTERING TO OUTPUT ----
        pad_x = (output_width - detected_width) / 2
        pad_y = (output_height - detected_height) / 2

        crop_offset = (int(left - pad_x), int(top - pad_y))

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
    sustain: int = 3,
    rel_frac: float = 0.2,
    abs_floor: float = 1.5,
) -> int | None:
    """Exposure-edge row of one strip column via rolling-std transition.

    The column runs from the film side to the collimation-line side
    (``from_end=True`` = line at the END of the vector, i.e. the TOP strip;
    scan starts at the line end and moves outward). Content near the line is
    textured; the unexposed margin beyond the exposure edge is smooth. The
    edge is the FIRST position (scanning outward) where the rolling std stays
    below ``max(abs_floor, rel_frac * content_std)`` for ``sustain``
    consecutive windows; content_std is the median rolling std of the third
    of the column nearest the line. Redacted samples (uniform fill — would
    fake a smooth margin) are spliced out; the returned index is in ORIGINAL
    column coordinates, or None when no sustained transition exists (edge
    outside the window, or an all-content column).
    """
    keep = np.flatnonzero(~np.asarray(redacted, dtype=bool))
    if keep.size < 4 * win:
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
    content_std = float(np.median(std[: max(n // 3, win)]))
    thresh = max(abs_floor, rel_frac * content_std)
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
