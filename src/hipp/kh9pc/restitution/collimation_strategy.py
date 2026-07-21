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
    refinement_fraction: float = 0.03
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
        """Run PolyStrategy first, then detect collimation peaks in a narrow refinement strip.

        The two detected lines are cross-validated against the known physical
        separation ``collimation_line_dist``: when they disagree beyond
        ``separation_tolerance`` (a corrupted edge estimate — e.g. a digital
        redaction band — parks one search strip away from its line), the bad
        side is re-detected in a wider window anchored at the OTHER side's line
        plus/minus the known separation. This is invariant to the mark-train /
        edge distances, which differ between fore and aft frames.
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

            self._validate_separation(src, col_off, window_width, window_height)

        return self

    def _out_shape(self, window: Window) -> tuple[int, int, int]:
        """Decimated read shape for a search window (rows strided, columns gridded)."""
        return (1, max(int(window.height) // self.stride, 1), self.grid_shape[0])

    def _line_row(self, side: str) -> int:
        """Median full-raster row of a fitted collimation line across the detected span."""
        left, right = self.poly_strategy.vertical_detector.edges_
        x = np.linspace(left, right, self.grid_shape[0])
        return int(np.median(self._results[side].model.predict(x.reshape(-1, 1))))

    def _validate_separation(self, src: rasterio.DatasetReader, col_off: int, window_width: int, window_height: int) -> None:
        """Cross-check the detected line separation against ``collimation_line_dist``;
        re-detect the corrupted side anchored on the good side when they disagree."""
        dist = self.collimation_line_dist
        tol = self.separation_tolerance * dist
        sep = self._line_row("bottom") - self._line_row("top")
        if abs(sep - dist) <= tol:
            self._separation_ok = True
            return

        logger.warning(
            "[CollimationStrategy] line separation %d deviates from expected %d by %+d px "
            "(tol %.0f) - re-detecting each side anchored on the other",
            sep, dist, sep - dist, tol,
        )

        candidates: list[tuple[float, str, CollimationResult, int]] = []
        for bad, good, sign in (("top", "bottom", -1), ("bottom", "top", +1)):
            center = self._line_row(good) + sign * dist
            # Wider (2x) window CENTERED on the expected row: tolerates the
            # anchor's own scan-scale deviation (~+-3% of raster height covered).
            row0 = max(center - window_height, 0)
            row1 = min(center + window_height, src.height)
            if row1 - row0 < 2 * self.stride:
                continue
            window = Window(col_off, row0, window_width, row1 - row0)
            sub_image = SubImage(src, window, resampling=Resampling.average, out_shape=self._out_shape(window))
            result = self._process_side(sub_image, bad)
            x = np.linspace(col_off, col_off + window_width, self.grid_shape[0])
            cand_row = int(np.median(result.model.predict(x.reshape(-1, 1))))
            cand_sep = (self._line_row(good) - cand_row) if bad == "top" else (cand_row - self._line_row(good))
            logger.info(
                "[CollimationStrategy] candidate re-detect %s (anchor %s): separation %d "
                "(%+d px from expected), inliers %.2f",
                bad, good, cand_sep, cand_sep - dist, result.inlier_ratio,
            )
            candidates.append((abs(cand_sep - dist), bad, result, cand_sep))

        # Acceptance needs BOTH separation agreement and a healthy re-detected
        # fit: a spurious anchor (the corrupted side) can place the re-detect
        # window over content where RANSAC "finds" a line at almost exactly the
        # expected distance from the WRONG anchor — separation error alone
        # scored such a 0.21-inlier lock at +3 px on D3C1217-200742F004 while
        # the true line (anchored on the good side, +61 px) sat in the other
        # candidate. Inlier filtering keeps only re-detects that landed on a
        # real continuous line.
        ok = [c for c in candidates
              if c[0] <= tol and c[2].inlier_ratio >= self.min_inliers_threshold]
        ok.sort(key=lambda c: c[0])
        if ok:
            err, bad, result, cand_sep = ok[0]
            logger.info(
                "[CollimationStrategy] adopted re-detected %s line: separation %d "
                "(%+d px from expected), inliers %.2f",
                bad, cand_sep, cand_sep - dist, result.inlier_ratio,
            )
            self._results[bad] = result
            self._separation_ok = True
        else:
            logger.error(
                "[CollimationStrategy] separation could not be reconciled with the known "
                "line distance - marking fit FAILED (candidates: %s)",
                "; ".join(f"{c[1]}: {c[0]:.0f} px off, inliers {c[2].inlier_ratio:.2f}"
                          for c in candidates) or "none",
            )
            self._separation_ok = False

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
