"""Real-data regression guard for collimation LINE detection and its strip
placement.

The round-2 exposure-edge work regressed line detection on every real frame
(separation off by ~30%, inliers collapsed) because ``PolyStrategy._process_side``
was changed to return the content edge instead of the FILM-FRAME boundary, and
``CollimationStrategy`` anchors and sizes its collimation-line search strip on
that edge -- moving it ~780 px inward mislocated the strip. The unit suite missed
it because nothing exercised real-frame line detection. These tests pin both:
(1) the strip-placement edge stays at the frame boundary, and (2) the collimation
line separation on real film stays within tolerance of the physical constant.

The fixture ``data/f004_line_detection_fixture.npz`` holds the strided search
bands (real KH-9 PC film) extracted from the D3C1217-200742F004 restitution
joblib: the two PolyStrategy edge-search bands and the two collimation-line
strips, each with the global-row scale/offset needed to reconstruct absolute row
positions.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.restitution.collimation_strategy import CollimationStrategy
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy

FIXTURE = Path(__file__).parent / "data" / "f004_line_detection_fixture.npz"
COLLIMATION_LINE_DIST = 21770


def _sub_from_band(band: np.ndarray, tmp_path: Path, name: str) -> SubImage:
    """A SubImage backed by a temp raster equal to ``band`` (identity to_global,
    so model predictions come back in band-row coordinates)."""
    p = tmp_path / f"{name}.tif"
    with rasterio.open(p, "w", driver="GTiff", width=band.shape[1], height=band.shape[0], count=1, dtype="uint8") as d:
        d.write(band, 1)
    return SubImage(rasterio.open(p), Window(0, 0, band.shape[1], band.shape[0]))


def test_line_separation_on_real_film(tmp_path: Path) -> None:
    """Run the collimation line detector on the two REAL line strips and check
    the top/bottom separation matches the physical constant within the
    strategy's tolerance -- the invariant a regression in line detection or its
    inputs breaks (job 24882612: separation collapsed to ~14500-15900)."""
    f = np.load(FIXTURE)
    strat = CollimationStrategy()
    g = {}
    inliers = {}
    for side in ("top", "bottom"):
        sub = _sub_from_band(f[f"line_{side}_band"], tmp_path, f"line_{side}")
        res = strat._process_side(sub, side)
        ncol = f[f"line_{side}_band"].shape[1]
        local = float(np.median(res.model.predict(np.arange(ncol).reshape(-1, 1))))
        g[side] = local * f[f"line_{side}_scale"][0] + f[f"line_{side}_off"][0]
        inliers[side] = res.inlier_ratio
    separation = g["bottom"] - g["top"]
    tol = strat.separation_tolerance * COLLIMATION_LINE_DIST
    assert abs(separation - COLLIMATION_LINE_DIST) <= tol, (
        f"line separation {separation:.0f} deviates from {COLLIMATION_LINE_DIST} by "
        f"{separation - COLLIMATION_LINE_DIST:+.0f} px (tol {tol:.0f})"
    )
    assert min(inliers.values()) >= strat.min_inliers_threshold, f"line inliers collapsed: {inliers}"


def test_strip_placement_edge_stays_at_frame_boundary(tmp_path: Path) -> None:
    """PolyStrategy's edge -- which CollimationStrategy uses to place and size
    the line strip -- must stay at the FILM-FRAME boundary, NOT the content edge.
    On this real top band the frame boundary is ~161 px and the content edge is
    ~940 px; the regression moved it to the latter, shifting the strip ~780 px
    inward. Guards against re-coupling the strip-placement edge to the
    conservative content-end delivery edge."""
    f = np.load(FIXTURE)
    ps = PolyStrategy()
    edges = {}
    for side in ("top", "bottom"):
        sub = _sub_from_band(f[f"poly_{side}_band"], tmp_path, f"poly_{side}")
        res = ps._process_side(sub, side)
        # median of the RAW detected ruptures (deterministic; the RANSAC model is
        # unseeded) -- these are exactly the points the strip is anchored on.
        local = float(np.median(res.ruptures_global[:, 1]))
        edges[side] = local * f[f"poly_{side}_scale"][0] + f[f"poly_{side}_off"][0]
    # The top edge must be the frame boundary (~161 px), far OUTSIDE the content
    # edge (~940). The regression returned the content edge, shifting the strip
    # ~780 px inward so the line sat at the strip's very top and the peak search
    # locked a deeper band. Pin it near the frame boundary.
    assert edges["top"] < 500, f"top strip-placement edge moved off the frame boundary to {edges['top']:.0f}"
    assert edges["bottom"] > COLLIMATION_LINE_DIST, f"bottom strip-placement edge implausible: {edges['bottom']:.0f}"
