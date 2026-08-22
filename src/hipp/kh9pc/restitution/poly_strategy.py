"""
Copyright (c) 2026 HIPP developers
Description: PolyStrategy — polynomial edge fitting for KH-9 PC restitution. Samples
    rupture points along the top and bottom film edges in a downsampled grid, fits a
    RANSAC polynomial per edge, then applies a Thin Plate Spline warp to straighten
    the curved edges.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window
from skimage.transform import ThinPlateSplineTransform
from sklearn.linear_model import RANSACRegressor

from hipp.image import SubImage, remap_tif_blockwise
from hipp.kh9pc.redaction_mask import detect_ruptures_skip_redacted, redacted_region_mask
from hipp.kh9pc.restitution.base import _InwardEnvelopeModel, detect_content_edges, fit_ransac_poly, tps_from_estimate
from hipp.kh9pc.restitution.base import DEFAULT_OUTPUT_HEIGHT, RestitutionStrategy, Transformation
from hipp.kh9pc.restitution.vertical_detector import VerticalDetector


@dataclass
class PolyResult:
    """Fitted polynomial edge model and diagnostics for one side (top or bottom).

    Attributes
    ----------
    ruptures_local:
        (N, 2) detected rupture coordinates in the sub-image pixel space.
    ruptures_global:
        (N, 2) same ruptures converted to full-raster pixel coordinates.
    distortion:
        (M, 2) array of ``[x, deviation_from_mean]`` sampled along the fitted curve,
        used to visualise the edge curvature.
    inlier_ratio:
        Fraction of rupture points classified as inliers by RANSAC.
    model:
        Fitted ``RANSACRegressor`` wrapping the polynomial pipeline. This is the
        SMOOTH edge that may feed a geometric warp -- it must never step.
    sub_image:
        Downsampled strip from which ruptures were extracted (kept for QC).
    crop_model:
        Optional conservative INWARD-envelope model (``_InwardEnvelopeModel``)
        for the valid-pixel CROP/mask only -- it steps inward at scan-section
        black-frame boundaries so no black frame enters the crop, and is NEVER
        used for geometry (David 2026-07-22 round 3). ``None`` when there is no
        separate conservative crop (e.g. the strip-placement edge).
    """

    ruptures_local: NDArray[np.integer]
    ruptures_global: NDArray[np.integer]
    distortion: NDArray[np.floating]
    inlier_ratio: float
    model: RANSACRegressor
    sub_image: SubImage
    crop_model: object | None = None


@dataclass
class PolyStrategy(RestitutionStrategy):
    """Restitution strategy based on polynomial edge fitting.

    Detects rupture points along the top and bottom film edges in a downsampled
    horizontal grid, fits a RANSAC polynomial per edge, then warps the image with
    a Thin Plate Spline transform to map the curved edges to horizontal target lines.
    Fails if the inlier ratio on either edge falls below ``min_inliers_threshold``.
    """

    vertical_detector: VerticalDetector = field(default_factory=VerticalDetector)
    background_threshold: int = 20
    height_fraction: float = 0.15
    stride: int = 10
    polynomial_degree: int = 2
    ransac_residual_threshold: float = 80.0
    ransac_max_trials: int = 1000
    grid_shape: tuple[int, int] = (100, 50)
    min_inliers_threshold: float = 0.5
    output_width: int | None = None
    # None -> canonical KH9ImageSpec height (ed2c30e port); set only to override
    output_height: int | None = None

    def __post_init__(self) -> None:
        super().__init__()
        self._results: dict[str, PolyResult] = {}
        self.__transformation_: Transformation | None = None

    @property
    def is_failed(self) -> bool:
        """True if either edge inlier ratio is below ``min_inliers_threshold``."""
        return min(self.top_.inlier_ratio, self.bottom_.inlier_ratio) < self.min_inliers_threshold

    @property
    def top_(self) -> PolyResult:
        """Fitted top edge result. Raises if ``fit()`` has not been called."""
        if "top" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["top"]

    @property
    def bottom_(self) -> PolyResult:
        """Fitted bottom edge result. Raises if ``fit()`` has not been called."""
        if "bottom" not in self._results:
            raise RuntimeError("Call fit() before")
        return self._results["bottom"]

    @property
    def transformation_(self) -> Transformation:
        """TPS Transformation from curved edges to horizontal lines (computed lazily)."""
        if self.__transformation_ is None:
            self.__transformation_ = self._compute_transformation()
        return self.__transformation_

    def _fit(self, raster_filepath: Path) -> Self:
        """Detect and fit polynomial models for the top and bottom edges."""
        if not self.vertical_detector.is_fitted or raster_filepath != self.vertical_detector.raster_filepath_:
            self.vertical_detector.fit(raster_filepath)

        col_off, _ = self.vertical_detector.edges_
        window_width = self.vertical_detector.detected_width_

        with rasterio.open(raster_filepath) as src:
            window_height = int(src.height * self.height_fraction)
            out_shape = (1, window_height // self.stride, self.grid_shape[0])

            for side, window in {
                "top": Window(col_off, 0, window_width, window_height),
                "bottom": Window(col_off, src.height - window_height, window_width, window_height),
            }.items():
                sub_image = SubImage(src, window, out_shape)
                self._results[side] = self._process_side(sub_image, side)

        return self

    def _process_side(self, sub_image: SubImage, side: str) -> PolyResult:
        """Sample column-wise ruptures, fit a RANSAC polynomial, and compute the distortion curve.

        This edge is consumed for STRIP PLACEMENT by ``CollimationStrategy`` (the
        collimation-line search window is anchored at, and sized from, this edge:
        the line sits ~1150 px inside the FILM-FRAME boundary this rupture scan
        locks). It MUST stay byte-identical to keep line detection unchanged; the
        conservative content-end/inward-envelope edge for the no-line delivery is
        applied separately in ``_conservative_edge_model`` at transform time
        (regression 2026-07-22: moving this scan to the content end shifted the
        top strip ~780 px inward and collapsed line detection).

        DIGITALLY REDACTED runs (redaction rectangles / hard margin fill —
        constant near-black DN, unlike noisy dark scanned film) are masked
        first and the rupture scan skips through them, so the edge cannot
        snap to a redaction boundary instead of the true film edge."""
        # mask ceiling = the rupture threshold itself: redaction fill at
        # DN 9-19 defeated the default ceiling while still triggering
        # ruptures (F004 QC round 2 — fit blended true edge + redaction
        # boundary). Uniformity (range test) remains the discriminator.
        redacted = redacted_region_mask(
            sub_image.band, max_dn=self.background_threshold, dilate=3)
        res = []
        for i in range(sub_image.band.shape[1]):
            ruptures = detect_ruptures_skip_redacted(
                sub_image.band[:, i], self.background_threshold, redacted[:, i],
                reverse_scan=(side == "top"))
            if len(ruptures) > 0:
                res.append((i, ruptures[0]))

        if not res:
            raise RuntimeError(f"No rupture detected on the {side} edge.")

        ruptures_local = np.array(res)
        ruptures_global = sub_image.to_global(ruptures_local)

        model = fit_ransac_poly(
            ruptures_global[:, 0],
            ruptures_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials,
        )

        inlier_ratio = float(model.inlier_mask_.mean())

        x_sample = np.linspace(
            sub_image.window.col_off, sub_image.window.col_off + sub_image.window.width, self.grid_shape[0]
        )
        y_global_pred = model.predict(x_sample.reshape(-1, 1)).ravel()
        y_distortion = y_global_pred - y_global_pred.mean()
        distortion = np.column_stack([x_sample, y_distortion])

        return PolyResult(
            ruptures_local=ruptures_local,
            ruptures_global=ruptures_global.astype(int),
            distortion=distortion,
            inlier_ratio=inlier_ratio,
            model=model,
            sub_image=sub_image,
        )

    def _content_edge_model(self, side: str) -> tuple[object, object | None]:
        """``(smooth_model, crop_model)`` for the DELIVERED no-collimation-line
        output (David r1-3). The SMOOTH poly2 fit to the content-end inliers is
        the ONLY edge that feeds the warp -- geometry must not step (a step in
        the edge is a step in local vertical scale = shear). The INWARD-ENVELOPE
        ``crop_model`` bounds the conservative valid-pixel rectangle and never
        feeds geometry. Each column of the stored search band is scanned from the
        content (inner) side outward (``detect_content_edges``): the edge is the
        content end, NOT the DN-threshold film-frame boundary out in the black
        margin. Returns ``(round-1 strip-placement model, None)`` when too few
        columns yield a content edge."""
        result = self._results[side]
        band = result.sub_image.band
        edges = detect_content_edges(band, side, black_dn=self.background_threshold)
        if len(edges) < max(10, band.shape[1] // 10):
            return result.model, None
        ruptures_global = result.sub_image.to_global(np.array(edges)).astype(int)
        smooth = fit_ransac_poly(
            ruptures_global[:, 0], ruptures_global[:, 1],
            degree=self.polynomial_degree,
            residual_threshold=self.ransac_residual_threshold,
            max_trials=self.ransac_max_trials)
        crop = _InwardEnvelopeModel.from_inliers(smooth, ruptures_global, side)
        return smooth, crop

    def _compute_transformation(self) -> Transformation:
        """Build a TPS Transformation that maps the fitted curved edges to horizontal target lines."""
        left, right = self.vertical_detector.edges_
        detected_width = self.vertical_detector.detected_width_
        # Canonical output size ALWAYS (targeted port of origin/main
        # ed2c30e): the detected-size fallback produced non-canonical
        # rasters (17.4-18k heights on 2026-vintage casa frames, job
        # 25016563) that break every downstream fixed-dims assumption.
        from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
        _spec_w, _spec_h = KH9ImageSpec.from_raster_filepath(self.raster_filepath_).expected_size
        output_width = self.output_width or _spec_w

        x = np.linspace(left, right, self.grid_shape[0])

        # GEOMETRY: the SMOOTH content-edge poly2 -- never the stepped inward
        # envelope -- so the warp introduces no shear at scan-section steps
        # (David 2026-07-22 round 3).
        top_model, top_crop = self._content_edge_model("top")
        bot_model, bot_crop = self._content_edge_model("bottom")
        y_top_src = top_model.predict(x.reshape(-1, 1)).ravel()
        y_bot_src = bot_model.predict(x.reshape(-1, 1)).ravel()

        top, bot = int(np.median(y_top_src)), int(np.median(y_bot_src))

        y_top_dst = np.full_like(x, top)
        y_bot_dst = np.full_like(x, bot)

        src = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_src, y_bot_src))))
        dst = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_dst, y_bot_dst))))

        # inverse source destination (important)
        deformation = tps_from_estimate(dst, src)

        # ---- CONSERVATIVE CROP RECTANGLE (David r3) ----
        # The output is a plain rectangle (remap_tif_blockwise has no per-pixel
        # mask), so the valid region is the CLOSEST conservative rectangle: inset
        # from the smooth edge to the innermost content edge (``crop_model``) so
        # no scan-section black block enters. In the warped frame the smooth edge
        # sits at ``top``/``bot``; per-section content deviates by the envelope,
        # so shrink the crop by that residual. Pixel cost = the section-step
        # amplitude (a few tens of px). Output height follows the conservative
        # detected height, not the outward-padded standard, so the pad cannot
        # re-admit frame.
        top_inset = int(max(0.0, float(np.max(top_crop.predict(x.reshape(-1, 1)).ravel() - y_top_src)))) if top_crop else 0
        bot_inset = int(max(0.0, float(np.max(y_bot_src - bot_crop.predict(x.reshape(-1, 1)).ravel())))) if bot_crop else 0
        crop_top, crop_bot = top + top_inset, bot - bot_inset
        detected_height = crop_bot - crop_top
        # canonical spec height unless the caller explicitly overrides
        # (second half of the ed2c30e port; the old min(default, detected)
        # collapsed to the detected height whenever the film sat tilted or
        # short in the scan)
        output_height = self.output_height or _spec_h

        pad_x = (output_width - detected_width) / 2
        pad_y = (output_height - detected_height) / 2

        crop_offset = (int(left - pad_x), int(crop_top - pad_y))

        return Transformation(
            self.raster_filepath_,
            deformation,
            crop_offset=crop_offset,
            output_size=(output_width, output_height),
        )

    def transform(self, output_path: str | Path) -> None:
        """Write the restituted image using the polynomial TPS warp."""
        tf = self.transformation_

        remap_tif_blockwise(
            tf.raster_filepath,
            output_path,
            tf.inverse_remap,
            tf.output_size,
            block_size=2**13,
            lowres_step=100,
        )
