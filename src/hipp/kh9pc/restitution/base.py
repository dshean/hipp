"""
Copyright (c) 2026 HIPP developers
Description: Abstract base classes and shared utilities for KH-9 PC restitution strategies.
    Defines the FittingClass / RestitutionStrategy protocol, the Transformation dataclass,
    and the low-level helpers (RANSAC polynomial fitting, rupture detection) shared across
    all concrete strategy implementations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_HEIGHT: int = 22064
"""Standard output height in pixels for restituted KH-9 PC images (22064 px at nominal scan resolution)."""


def tps_from_estimate(src_pts, dst_pts):
    """skimage-version-agnostic ThinPlateSplineTransform estimation.

    ``from_estimate`` landed after scikit-image 0.25; the older API is
    ``estimate(src, dst) -> bool`` mutating the instance. hipp must run on
    both (HECC env carries 0.25.0 — 2026-07-19)."""
    from skimage.transform import ThinPlateSplineTransform

    if hasattr(ThinPlateSplineTransform, "from_estimate"):
        return ThinPlateSplineTransform.from_estimate(src_pts, dst_pts)
    tf = ThinPlateSplineTransform()
    if not tf.estimate(src_pts, dst_pts):
        raise RuntimeError("ThinPlateSplineTransform estimation failed")
    return tf


class DetectionError(Exception):
    """Raised when no valid detections are found during fitting."""


class FittingClass(ABC):
    """Base class for objects that are fitted against a raster file.

    Subclasses implement ``_fit`` to perform the actual work and expose
    ``is_failed`` to signal whether the result is usable. The public ``fit``
    method wraps ``_fit`` with logging and records the source raster path so
    QC code can always trace results back to their input.
    """

    def __init__(self) -> None:
        self.__raster_filepath_: Path | None = None

    @property
    def raster_filepath_(self) -> Path:
        """Path of the raster this instance was fitted on. Raises if ``fit()`` has not been called."""
        if self.__raster_filepath_ is None:
            raise RuntimeError("Call fit() before.")
        return self.__raster_filepath_

    @property
    def is_fitted(self) -> bool:
        """True after ``fit()`` has been called at least once."""
        return self.__raster_filepath_ is not None

    @property
    def logging_prefix(self) -> str:
        """Standard log prefix ``"[ClassName] raster.tif"`` for this instance."""
        return f"[{self.__class__.__name__}] {self.raster_filepath_.name}"

    @property
    @abstractmethod
    def is_failed(self) -> bool:
        """True if the last ``fit()`` produced an unusable result."""
        ...

    def fit(self, raster_filepath: str | Path) -> Self:
        """Run ``_fit`` on *raster_filepath*, log start/end, and record the source path."""
        self.__raster_filepath_ = Path(raster_filepath)
        logger.info("%s - start fit...", self.logging_prefix)
        fit_res = self._fit(self.__raster_filepath_)
        logger.info("%s - finish fit : [%s]", self.logging_prefix, "FAILED" if self.is_failed else "SUCCESS")
        return fit_res

    @abstractmethod
    def _fit(self, raster_filepath: Path) -> Self:
        """Perform the actual fitting work; called by ``fit()``."""
        ...


class RestitutionStrategy(FittingClass):
    """A ``FittingClass`` that can also produce a restituted output image.

    Concrete strategies (Flat, Poly, Collimation, Fiducial) fit edge/fiducial
    detections and then expose ``transformation_`` (the geometric model) and
    ``transform`` (the method that applies it to write the output GeoTIFF).
    """

    @abstractmethod
    def transform(self, output_path: str | Path) -> None:
        """Apply the fitted transformation and write the restituted image to *output_path*."""
        ...

    @property
    @abstractmethod
    def transformation_(self) -> "Transformation":
        """The fitted ``Transformation`` object describing the geometric correction."""
        ...


@dataclass
class Transformation:
    """Geometric transformation from output pixel space back to input raster space.

    Used with ``remap_tif_blockwise``: for every output pixel coordinate the
    inverse remap first un-crops (adds ``crop_offset``) then applies the
    ``deformation`` callable to obtain the corresponding source coordinate.

    Attributes
    ----------
    raster_filepath:
        Source raster to read pixel values from.
    deformation:
        Callable mapping (N, 2) output coords → (N, 2) source coords (inverse warp).
    crop_offset:
        ``(x, y)`` translation added before deformation to go from the cropped
        output space back to the full raster coordinate system.
    output_size:
        ``(width, height)`` of the restituted output image in pixels.
    """

    raster_filepath: Path
    deformation: Callable[[NDArray[np.float32]], NDArray[np.float32]]
    crop_offset: tuple[float, float] = (0, 0)
    output_size: tuple[int, int] = (0, 0)

    def inverse_remap(self, coords: NDArray[np.float32]) -> NDArray[np.float32]:
        """Translate coords by ``crop_offset`` then apply ``deformation``."""
        coords = coords + np.array([self.crop_offset[0], self.crop_offset[1]], dtype=coords.dtype)
        return self.deformation(coords)


def fit_ransac_poly(
    x: NDArray[np.generic],
    y: NDArray[np.generic],
    degree: int = 3,
    residual_threshold: float = 100,
    max_trials: int = 100,
) -> RANSACRegressor:
    """Fit a polynomial regression with RANSAC on 1D data. Returns the fitted RANSACRegressor."""
    poly_model = make_pipeline(
        PolynomialFeatures(degree=degree),
        StandardScaler(),
        LinearRegression(),
    )

    min_samples = min(degree * 3, len(x))
    ransac = RANSACRegressor(
        poly_model, residual_threshold=residual_threshold, min_samples=min_samples, max_trials=max_trials
    )
    ransac.fit(x.reshape(-1, 1), y)
    return ransac


def detect_ruptures(vec: NDArray[np.number], threshold: float, reverse_scan: bool = False) -> NDArray[np.integer]:
    """Detect indices where the signal drops below a threshold (falling edges).

    If reverse_scan is True, scan from the end and return indices in original coordinates.
    """
    if reverse_scan:
        vec = vec[::-1]

    idx = np.where((vec[1:] <= threshold) & (vec[:-1] > threshold))[0] + 1

    if reverse_scan:
        idx = len(vec) - 1 - idx

    return idx


# ---- conservative content-edge detection (shared across ALL strategies) ----
# David 2026-07-22: the delivered exposure edge must ERR INWARD and never admit
# black frame / margin / section-mosaic steps -- an invariant for EVERY strategy
# (Collimation with a line anchor, Poly/Flat without one), so these primitives
# live in base and are reused everywhere.


def _featureless_edge(
    vec: NDArray[np.number],
    black_dn: float = 15.0,
    flat_std: float = 1.5,
    win: int = 5,
    sustain: int = 10,
) -> int | None:
    """Outward index of the exposure edge in one column vector ordered from the
    CONTENT side outward toward the frame (index 0 = deepest content).

    Simple and DN-first (David 2026-07-22): walking outward, the edge is the
    first row that begins a SUSTAINED FEATURELESS run -- either near-black
    (``DN <= black_dn``; the merged section-mosaic black frame is essentially 0,
    measured ~9) OR flat (rolling std over ``win`` samples ``<= flat_std``; the
    unexposed grey film margin is featureless at DN ~40, not black). Real
    exposed content is textured and non-black, so it does not trigger;
    ``sustain`` requires the featureless run to persist (~a hundred full-res px
    at the strided read) so a small dark/smooth patch of content cannot fire.
    Returns the run-start index, or None when no featureless run is found. Must
    ACCEPT nearly every content column, not refuse low-contrast-but-obvious ones.
    Thresholds are for the strided-average band read.
    """
    v = np.asarray(vec, dtype=float)
    n = v.size
    if n < win + sustain:
        return None
    c1 = np.cumsum(np.insert(v, 0, 0.0))
    c2 = np.cumsum(np.insert(v * v, 0, 0.0))
    mean = (c1[win:] - c1[:-win]) / win
    std = np.sqrt(np.maximum((c2[win:] - c2[:-win]) / win - mean * mean, 0.0))
    # std has length n-win+1; pad to n by repeating the last (deep-frame) value
    std = np.concatenate([std, np.full(n - std.size, std[-1] if std.size else 0.0)])
    featureless = (v <= black_dn) | (std <= flat_std)
    run = 0
    for i in range(n):
        if featureless[i]:
            run += 1
            if run == sustain:
                # return the LAST CONTENT sample (run start - 1), one step INWARD
                # of the transition, so the strided-read quantisation (~10 px per
                # sample at the transition, a content/frame average) can never
                # place the delivered edge one row out INTO the frame. Errs inward
                # by construction (David 2026-07-22). None when the run starts at
                # index 0 (no content on the inner side).
                edge = i - sustain
                return edge if edge >= 0 else None
        else:
            run = 0
    return None


def detect_content_edges(
    band: NDArray[np.number], side: str, black_dn: float = 15.0,
) -> list[tuple[int, int]]:
    """Per-column exposure-edge rows (local, strided) for a BOUNDARY-anchored
    search window whose CONTENT is on the INNER side (the no-line Poly/Flat
    fallbacks: top window content at the bottom, bottom window content at the
    top). Each column is scanned from the content side OUTWARD to the frame with
    ``_featureless_edge``, so the edge is the content end -- NOT the DN-threshold
    film-frame boundary, which sits out in the black margin. Returns
    ``[(col, local_row), ...]`` for the columns that yielded an edge.
    """
    nrows = band.shape[0]
    res: list[tuple[int, int]] = []
    for c in range(band.shape[1]):
        if side == "top":                       # content at the LARGE-row (inner) end
            r = _featureless_edge(band[::-1, c], black_dn=black_dn)
            if r is not None:
                res.append((c, nrows - 1 - r))
        else:                                    # content at the SMALL-row (inner) end
            r = _featureless_edge(band[:, c], black_dn=black_dn)
            if r is not None:
                res.append((c, r))
    return res


class _InwardEnvelopeModel:
    """Exposure-edge model = a RANSAC polynomial CLAMPED to the INWARD envelope
    of the per-column detected transitions (David 2026-07-22).

    The merged frames are mosaics of scan sections whose black frame starts at
    DIFFERENT rows; a single smooth curve fitted through them can pass OUTSIDE a
    shallower section's black start and admit a sliver of frame. ``predict``
    returns ``min(poly, bound)`` for the bottom edge (content above, black below
    -> conservative = smaller row) and ``max(poly, bound)`` for the top, so the
    delivered edge steps inward at section boundaries and never crosses past a
    black block. ``bound`` is a rolling inward extremum of the RANSAC-INLIER
    transitions over +-``radius`` columns: a coherent shallow section pulls the
    edge inward, while isolated noisy columns (already RANSAC outliers) cannot.
    Duck-types the ``RANSACRegressor`` interface (``predict``/``inlier_mask_``)
    the QC figure and downstream consumers use.
    """

    def __init__(self, poly: RANSACRegressor, xs: NDArray[np.floating],
                 bound: NDArray[np.floating], side: str) -> None:
        self._poly = poly
        self.inlier_mask_ = poly.inlier_mask_
        self._xs = np.asarray(xs, dtype=float)
        self._bound = np.asarray(bound, dtype=float)
        self._bottom = side == "bottom"

    @classmethod
    def from_inliers(cls, poly: RANSACRegressor, ruptures_global: NDArray[np.integer],
                     side: str, radius: int = 2) -> "_InwardEnvelopeModel":
        inl = poly.inlier_mask_
        xs = ruptures_global[inl, 0].astype(float)
        ys = ruptures_global[inl, 1].astype(float)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        bound = ys.copy()
        n = ys.size
        for i in range(n):
            lo, hi = max(0, i - radius), min(n, i + radius + 1)
            bound[i] = ys[lo:hi].min() if side == "bottom" else ys[lo:hi].max()
        return cls(poly, xs, bound, side)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        x = np.asarray(X, dtype=float).reshape(-1)
        p = np.asarray(self._poly.predict(x.reshape(-1, 1))).ravel()
        if self._xs.size == 0:
            return p
        b = np.interp(x, self._xs, self._bound)
        return np.minimum(p, b) if self._bottom else np.maximum(p, b)
