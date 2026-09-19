"""
Copyright (c) 2026 HIPP developers
Description: PolyStrategy — polynomial edge fitting for KH-9 PC restitution. Samples
    rupture points along the top and bottom film edges in a downsampled grid, fits a
    RANSAC polynomial per edge, then applies a Thin Plate Spline warp to straighten
    the curved edges.
"""

import logging

import os
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
from hipp.kh9pc.restitution.base import scan_scale
from hipp.kh9pc.restitution.base import DEFAULT_OUTPUT_HEIGHT, RestitutionStrategy, Transformation
from hipp.kh9pc.restitution.vertical_detector import VerticalDetector

logger = logging.getLogger(__name__)


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
    # (x_um, y_um) scanner pitch of this scan session; None -> raster tags or 1:1
    scan_pitch_um: tuple[float, float] | None = None
    # 2026-09-18: which model class fed each side's delivered geometry -- "oracle" | "content" |
    # "rupture". Populated by _content_edge_model(); read by _compute_transformation() for the
    # mixed-class guard and written to the QC record so a datum shift is attributable.
    _edge_class_: dict = field(default_factory=dict)

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
        # Reuse a detector that is already fitted on THIS raster (2026-08-28): the kh9pc_stereo gate
        # and worker pre-fit the detector and overwrite ruled edge positions (EDGE_OVERRIDES);
        # the unconditional fit() here silently discarded that (nepal ops251 A055).
        if not self.vertical_detector.is_fitted or Path(raster_filepath) != Path(self.vertical_detector.raster_filepath_):
            self.vertical_detector.scan_pitch_um = self.scan_pitch_um   # sweep width in this scan's px
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

    def _content_edge_model(self, side: str, skip_oracle: bool = False) -> tuple[object, object | None]:
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
        # Edge-oracle primary (dshean 2026-08-23, review-accepted on
        # casa_block): clean-slate per-strip middle-out detector replaces
        # the DN-threshold content walk that locked onto outer margin
        # structure on 2026-vintage scans (restitution review C1). The
        # legacy path below remains the fallback when the oracle's own
        # verdict does not pass.
        if getattr(self, "_edge_oracle_path_", None) != self.raster_filepath_:
            try:
                from .edge_oracle import fit_format_edges
                self._edge_oracle_ = fit_format_edges(self.raster_filepath_)
            except Exception as _exc:   # audit H-5: NEVER a silent fallback
                logger.warning(
                    "edge oracle failed on %s: %r — falling back to legacy "
                    "DN-threshold content edges (review C1 risk)",
                    self.raster_filepath_, _exc)
                self._edge_oracle_ = {}
            self._edge_oracle_path_ = self.raster_filepath_
        _ef = self._edge_oracle_.get(side)
        # _force_oracle_ (policy "promote"): accept this side's oracle fit even though its own verdict did
        # not pass, because the OTHER side's oracle did and a like-for-like midpoint beats a mixed one.
        _forced = side in getattr(self, "_force_oracle_", set())
        if _ef is not None and (_ef.passed or _forced) and not skip_oracle:
            self._edge_class_[side] = "oracle"
            return _ef, None

        result = self._results[side]
        band = result.sub_image.band
        edges = detect_content_edges(band, side, black_dn=self.background_threshold)
        if len(edges) < max(10, band.shape[1] // 10):
            # 2026-09-18: the RUPTURE model is a DIFFERENT physical feature from the oracle/content
            # frame edge (it sits ~1e3 px further out, in the film border). The delivered datum is
            # (crop_top+crop_bot)/2, so taking this on ONE side only moves the whole canvas by half
            # the class separation -- measured on nepal ops251 as ~1000 rows between two runs that
            # differed only in the source mosaic margins (A056 -1038, A057 -981, F053 +998, F054 +917),
            # with nothing in any log to attribute it. Record the class so it is attributable, and let
            # the caller refuse a MIXED pair.
            self._edge_class_[side] = "rupture"
            logger.warning(
                "%s %s edge: content walk found %d edges (< %d needed) -- falling back to the RUPTURE "
                "model, a different feature from the oracle/content frame edge; the delivered datum is "
                "only sound if BOTH sides fall back together",
                getattr(self, "raster_filepath_", "?"), side, len(edges), max(10, band.shape[1] // 10))
            return result.model, None
        self._edge_class_[side] = "content"
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
        # canvas scale: source px -> canvas px at CANVAS_PITCH_UM (base.scan_scale)
        sx, sy = scan_scale(self.scan_pitch_um, self.raster_filepath_)
        logger.info("[PolyStrategy] scan scale to canvas: sx=%.5f sy=%.5f", sx, sy)

        x = np.linspace(left, right, self.grid_shape[0])

        # GEOMETRY: the SMOOTH content-edge poly2 -- never the stepped inward
        # envelope -- so the warp introduces no shear at scan-section steps
        # (David 2026-07-22 round 3).
        top_model, top_crop = self._content_edge_model("top")
        bot_model, bot_crop = self._content_edge_model("bottom")
        y_top_src = top_model.predict(x.reshape(-1, 1)).ravel()
        y_bot_src = bot_model.predict(x.reshape(-1, 1)).ravel()

        top, bot = int(np.median(y_top_src)), int(np.median(y_bot_src))

        # ---- EDGE-MODEL CLASS CONSISTENCY (2026-09-18) ----
        # The delivered vertical datum is (crop_top+crop_bot)/2, so the two sides MUST measure the same
        # feature. MEASURED on nepal ops251 (14 frames x 2 mosaic generations, analysis/nepal_canvas_
        # 2026-09-17/edge_class_{cut,nocut}.txt): the "rupture" fallback never fires, but the oracle and
        # the content walk disagree by 1500-2000 rows -- the oracle finds the FRAME edge, the content walk
        # the end of CONTENT inside it. Five of 14 frames drew one side from each, and every one of those
        # five delivered an empty band at exactly one end; the only banded frame with a matched pair is
        # A055, whose film genuinely ends early. Between two runs differing only in the source margins,
        # class flips moved the datum by ~900-1000 rows with every gate green.
        # Policy "consistent" (default): if the sides disagree, DEMOTE the higher-preference side so both
        # use the same class (oracle > content > rupture); a symmetric error then cancels in the midpoint.
        # "native" keeps the old per-side choice and REFUSES a mixed pair unless it is accepted knowingly.
        # Policy "oracle": use the edge oracle on BOTH sides whenever a fit exists, even where its own
        # verdict did not pass -- the content walk is never the delivered geometry. MEASURED on nepal
        # ops251, 2026-09-18 (analysis/nepal_canvas_2026-09-17/edge_model_audit.txt, all 14 frames, each
        # side, against an edge measured from the raster alone -- per-row median DN, half-way crossing
        # between the film-margin plateau and the interior): the CONTENT walk lands 1200-2900 rows INSIDE
        # the true edge on every frame (top mid-frame 3143-3774 against measured edges of 1002-2056),
        # while the ORACLE sits within about +-400 rows. A055 is the one frame whose oracle failed its own
        # verdict on BOTH sides, so it fell to the content walk on both, no mixed pair fired, "promote"
        # never triggered, and the delivered datum shipped +1041 rows off with every gate green. Its
        # oracle fit is good (top 1175 vs measured 1159, bottom 23074 vs 22989), so forcing it here moves
        # A055's datum error from +1041 to roughly +50 and leaves the other 13 frames bit-identical
        # (they already deliver oracle on both sides). Not the default: this is a per-block ruling until
        # the oracle verdict itself is re-tuned, and a site whose oracle is genuinely unreliable still
        # wants the verdict to bite.
        _policy = os.environ.get("POLY_EDGE_CLASS_POLICY", "promote")
        if _policy not in ("promote", "consistent", "native", "oracle", "rupture"):
            raise ValueError("POLY_EDGE_CLASS_POLICY must be "
                             "'promote'|'consistent'|'native'|'oracle'|'rupture' (got %r)" % _policy)
        _rank = {"oracle": 0, "content": 1, "rupture": 2}
        # Policy "rupture": deliver the RANSAC rupture model on both sides. dshean 2026-09-18,
        # from the restitution sheets: "the green dots and the dashed light blue rupture model are
        # the correct edges of the exposed area of the frame".
        #
        # The test he set is consistency, not agreement with a nominal -- the exposed frame must map
        # to a CONSTANT rectangle. MEASURED on ops251, all 14 frames, separation in canvas px
        # (analysis/nepal_canvas_2026-09-17/rupture_width_profile.txt):
        #
        #                 across frames              within a frame
        #   rupture   median 22059, spread 148 px    median spread  96 px, worst  181, tilt <= +-161
        #   oracle    median 21596, spread 745 px    median spread 615 px, worst 1328, tilt <= +-1188
        #
        # The rupture lines hold the separation to 0.67 % across the block and 0.44 % within a
        # frame; the oracle is 5x and 6x worse. A055 -- the frame with the real exposure gap -- is
        # the BEST of the block by this measure (15 px, 0.07 %), so the gap is nodata inside a
        # correctly measured frame, exactly as the 2026-08-28 edge ruling says it should be.
        #
        # The rupture separation is 22059 px = 154.4 mm, not the 21771 px / 152.4 mm collimation
        # pair -- about 1 mm outside the printed line on each side, which is where a film-frame
        # edge belongs. So a block delivered on this model MUST also set KH9_IMAGE_HEIGHT_PX to
        # that separation, or 154.4 mm of film is squeezed into a 152.4 mm canvas and every camera
        # carries a 1.32 % along-track scale error.
        if _policy == "rupture":
            for _side in ("top", "bottom"):
                _m = self._results[_side].model
                self._edge_class_[_side] = "rupture"
                if _side == "top":
                    top_model, top_crop = _m, None
                    y_top_src = top_model.predict(x.reshape(-1, 1)).ravel()
                else:
                    bot_model, bot_crop = _m, None
                    y_bot_src = bot_model.predict(x.reshape(-1, 1)).ravel()
            top, bot = int(np.median(y_top_src)), int(np.median(y_bot_src))
            logger.warning(
                "%s: POLY_EDGE_CLASS_POLICY=rupture -- delivering the rupture model on both sides "
                "(rows %.1f / %.1f, separation %d src px); the canvas height must measure THIS "
                "feature (KH9_IMAGE_HEIGHT_PX), not the collimation pair",
                getattr(self, "raster_filepath_", "?"), float(np.median(y_top_src)),
                float(np.median(y_bot_src)), int(bot - top))
        if _policy == "oracle":
            _need = {_s for _s in ("top", "bottom")
                     if self._edge_class_.get(_s) != "oracle"
                     and getattr(self, "_edge_oracle_", {}).get(_s) is not None}
            _noora = {_s for _s in ("top", "bottom")
                      if getattr(self, "_edge_oracle_", {}).get(_s) is None}
            if _noora:
                logger.warning(
                    "%s: POLY_EDGE_CLASS_POLICY=oracle but no oracle fit on %s -- those sides keep the "
                    "content/rupture walk and the delivered datum is NOT oracle-consistent",
                    getattr(self, "raster_filepath_", "?"), sorted(_noora))
            if _need:
                logger.warning(
                    "%s: POLY_EDGE_CLASS_POLICY=oracle -- forcing the oracle on %s (was %s); the oracle "
                    "verdict did not pass there but the content walk measures a different, deeper feature",
                    getattr(self, "raster_filepath_", "?"), sorted(_need),
                    {_s: self._edge_class_.get(_s) for _s in sorted(_need)})
                self._force_oracle_ = set(getattr(self, "_force_oracle_", set())) | _need
                for _side in sorted(_need):
                    _m, _c = self._content_edge_model(_side)
                    if self._edge_class_.get(_side) != "oracle":
                        raise RuntimeError(
                            "%s: POLY_EDGE_CLASS_POLICY=oracle could not force the %s edge to the oracle "
                            "(got %r) -- refusing rather than delivering the content walk silently"
                            % (getattr(self, "raster_filepath_", "?"), _side, self._edge_class_.get(_side)))
                    if _side == "top":
                        top_model, top_crop = _m, _c
                        y_top_src = top_model.predict(x.reshape(-1, 1)).ravel()
                    else:
                        bot_model, bot_crop = _m, _c
                        y_bot_src = bot_model.predict(x.reshape(-1, 1)).ravel()
                top, bot = int(np.median(y_top_src)), int(np.median(y_bot_src))
        _cls = dict(self._edge_class_)
        if len(set(_cls.values())) > 1:
            if _policy in ("consistent", "promote"):
                # MEASURED nepal ops251 (analysis/nepal_canvas_2026-09-17/edge_policy_ab.txt): the ORACLE
                # frame-edge separation reproduces the canonical canvas height (detected_height 21138-22055
                # vs spec 21771, pad_y 14-314 px), while the content walk sits ~1800 px inside it and leaves
                # pad_y 1840-2966 -- i.e. demoting to "content" makes the empty band WIDER, not narrower.
                # So "promote" (default) takes the HIGHER-preference class, "consistent" the lower.
                _target = (min if _policy == "promote" else max)(_cls.values(), key=lambda c: _rank.get(c, 99))
                logger.warning(
                    "%s: edge-model classes disagree (top=%s bottom=%s) -- demoting both sides to %r so "
                    "the delivered datum is a like-for-like midpoint (POLY_EDGE_CLASS_POLICY=native keeps "
                    "the per-side choice)", getattr(self, "raster_filepath_", "?"),
                    _cls.get("top"), _cls.get("bottom"), _target)
                for _side in ("top", "bottom"):
                    if self._edge_class_.get(_side) != _target:
                        if _target == "oracle":
                            _force_prev = getattr(self, "_force_oracle_", set())
                            self._force_oracle_ = set(_force_prev) | {_side}
                            _m, _c = self._content_edge_model(_side)
                        else:
                            _m, _c = self._content_edge_model(_side, skip_oracle=True)
                        if self._edge_class_.get(_side) != _target:
                            raise RuntimeError(
                                "%s: could not demote %s edge to %r (got %r) -- refusing rather than "
                                "delivering a mixed-class datum" % (getattr(self, "raster_filepath_", "?"),
                                _side, _target, self._edge_class_.get(_side)))
                        if _side == "top":
                            top_model, top_crop = _m, _c
                            y_top_src = top_model.predict(x.reshape(-1, 1)).ravel()
                        else:
                            bot_model, bot_crop = _m, _c
                            y_bot_src = bot_model.predict(x.reshape(-1, 1)).ravel()
                top, bot = int(np.median(y_top_src)), int(np.median(y_bot_src))
                _cls = dict(self._edge_class_)
            else:
                msg = ("%s: MIXED edge-model classes top=%s bottom=%s -- the two sides measure different "
                       "features, so the delivered datum is shifted by ~half their separation. Set "
                       "POLY_ALLOW_MIXED_EDGE_CLASS=1 to accept knowingly.")
                args = (getattr(self, "raster_filepath_", "?"), _cls.get("top"), _cls.get("bottom"))
                if os.environ.get("POLY_ALLOW_MIXED_EDGE_CLASS", "0") != "1":
                    raise RuntimeError(msg % args)
                logger.warning(msg % args)
        self._edge_class_mixed_ = len(set(_cls.values())) > 1
        # PHYSICAL CHECK: the canonical canvas height is the collimation-line separation (152.4 mm at
        # 7 um = 21771 px). MEASURED: an oracle/oracle pair reproduces it to within ~630 px on 13 of 14
        # nepal ops251 frames, so a detected height far from spec means an edge is on the wrong feature.
        _spec_gap = float(np.median(y_bot_src) - np.median(y_top_src)) * float(sy) - float(_spec_h)
        logger.info("%s edge-model classes: top=%s bottom=%s (rows %.1f / %.1f, policy=%s, "
                    "edge separation - spec = %+.0f canvas px)",
                    getattr(self, "raster_filepath_", "?"), _cls.get("top"), _cls.get("bottom"),
                    float(np.median(y_top_src)), float(np.median(y_bot_src)), _policy, _spec_gap)
        self._edge_sep_minus_spec_ = _spec_gap
        if abs(_spec_gap) > float(os.environ.get("POLY_EDGE_SEP_TOL_PX", "1500")):
            logger.warning(
                "%s: detected edge separation is %+.0f px from the %d px spec (classes top=%s bottom=%s) -- "
                "the canvas will carry ~%.0f px of empty pad at EACH end; treat this frame's vertical datum "
                "as unverified", getattr(self, "raster_filepath_", "?"), _spec_gap, int(_spec_h),
                _cls.get("top"), _cls.get("bottom"), abs(_spec_gap) / 2.0)

        # destination (canvas) coordinates: straightened edges, scaled to the canvas pitch
        y_top_dst = np.full_like(x, top * sy)
        y_bot_dst = np.full_like(x, bot * sy)

        src = np.column_stack((np.concatenate((x, x)), np.concatenate((y_top_src, y_bot_src))))
        dst = np.column_stack((np.concatenate((x * sx, x * sx)), np.concatenate((y_top_dst, y_bot_dst))))

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
        crop_top, crop_bot = (top + top_inset) * sy, (bot - bot_inset) * sy   # canvas px
        detected_height = crop_bot - crop_top
        detected_width = detected_width * sx
        left = left * sx
        # canonical spec height unless the caller explicitly overrides
        # (second half of the ed2c30e port; the old min(default, detected)
        # collapsed to the detected height whenever the film sat tilted or
        # short in the scan)
        output_height = self.output_height or _spec_h

        pad_x = (output_width - detected_width) / 2
        pad_y = (output_height - detected_height) / 2
        # 2026-09-18: expose the padding for the class-policy A/B. pad_y is SYMMETRIC -- the canvas
        # extends pad_y beyond the detected content at BOTH ends -- so a smaller detected_height means
        # a wider empty band wherever the film does not reach. This is the number that decides whether
        # a mixed pair should be demoted (content walk, further in) or promoted (oracle frame edge).
        self._detected_height_ = float(detected_height)
        self._pad_y_ = float(pad_y)
        # canvas-space geometry for transform_extended(): the straightened line rows and the
        # exposure edges, all in canvas px BEFORE the crop offset is applied (2026-09-19)
        self._top_dst_ = float(top * sy)
        self._bot_dst_ = float(bot * sy)
        self._left_dst_ = float(left)
        self._right_dst_ = float(left + detected_width)
        self._crop_top_ = float(crop_top)
        self._crop_bot_ = float(crop_bot)

        # x: strength-weighted crop centre from the vertical detector (== left - pad_x when
        # both edges are equally strong); y: detected height centred in the spec height
        _c = getattr(self.vertical_detector, "crop_center_", None)
        x0 = (_c * sx - output_width / 2) if _c is not None else (left - pad_x)
        crop_offset = (int(x0), int(crop_top - pad_y))

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


    def transform_extended(self, output_path: str | Path, rail_px: int = 1600) -> dict:
        """Write the SAME warp over a window extended by ``rail_px`` rows above and below the
        delivered canvas, so the film margins -- scan-angle marks, time track, titling data --
        are rectified too. Split of "restitution" and "crop" (dshean 2026-09-19): the crop is
        already only a translation inside ``Transformation.inverse_remap`` (coords + crop_offset),
        so the delivered product is the integer sub-window rows [rail_px, rail_px + H) of this
        file, bit-identical -- no second warp. The marks are then straight rows at known canvas
        rows with a uniform period, which is both an easier detection and a check on the
        restitution itself (a bent row or a stepped period = a bad seam or edge model).

        Returns the sidecar dict that is also written as ``<output>.json``: rect_offset /
        rect_size (canvas px), the delivered window inside it, the straightened line rows and
        exposure edges in rect-canvas px, and the canvas pitch.
        """
        import dataclasses
        import json as _json
        tf = self.transformation_
        rail_px = int(rail_px)
        ext = dataclasses.replace(
            tf,
            crop_offset=(tf.crop_offset[0], tf.crop_offset[1] - rail_px),
            output_size=(tf.output_size[0], tf.output_size[1] + 2 * rail_px),
        )
        remap_tif_blockwise(
            ext.raster_filepath,
            output_path,
            ext.inverse_remap,
            ext.output_size,
            block_size=2**13,
            lowres_step=100,
        )
        from hipp.kh9pc.kh9_image_spec import CANVAS_PITCH_UM
        ox, oy = ext.crop_offset
        side = {
            "rail_px": rail_px,
            "rect_offset": [int(ox), int(oy)],
            "rect_size": [int(ext.output_size[0]), int(ext.output_size[1])],
            "delivered_window": {"col_off": 0, "row_off": rail_px,
                                 "width": int(tf.output_size[0]), "height": int(tf.output_size[1])},
            "delivered_crop_offset": [int(tf.crop_offset[0]), int(tf.crop_offset[1])],
            "line_rows_rect": {"top": self._top_dst_ - oy, "bottom": self._bot_dst_ - oy},
            "edges_rect": [self._left_dst_ - ox, self._right_dst_ - ox],
            "canvas_pitch_um": [CANVAS_PITCH_UM, CANVAS_PITCH_UM],
            "edge_classes": dict(getattr(self, "_edge_class_", {})),
            "source_raster": str(tf.raster_filepath),
        }
        Path(str(output_path) + ".json").write_text(_json.dumps(side, indent=2) + "\n")
        return side
