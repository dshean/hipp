"""Regression tests for the line-anchored exposure-edge refit
(``CollimationStrategy._refit_edges_from_lines``).

Reproduces the WA 1982 F004 failure mode: a bright textured block (the burned-in
USGS logo) sits ~700-1000 px BEYOND the true exposure edge in the leftmost
columns. Being high-variance anti-aliased text, it keeps the per-column rolling
variance HIGH past the true edge, so the texture detector reports the edge at the
block's far rim (~1000 px outward of the collimation line, verified on the real
frame at ~1168 px). At the LEFT END of the fitted span those points had full
leverage and randomly (RANSAC is unseeded) bent the free polynomial edge down by
~1000 px. David 2026-07-22: anchor the edge to the collimation line + a
conservative fixed OUTWARD offset and let the texture refit only REFINE within a
tight band [edge_band_inner, edge_band_outer]; reject anything farther out, and
if too few columns survive, the derived line+offset edge stands on its own
(erring inward). These tests assert the block can never enter the fit.

Round 2 (David 2026-07-22): the primary per-column detector is simple and
DN-based (``_featureless_edge`` -- last content row before a sustained near-black
or flat run), which accepts nearly every content column instead of refusing the
low-contrast-but-obvious ones the round-1 variance gate rejected; and the
delivered edge is the INWARD ENVELOPE (``_InwardEnvelopeModel``) so it steps
inward at scan-section black-frame boundaries and never crosses past a black
block.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.redaction_mask import redacted_region_mask
from hipp.kh9pc.restitution.base import fit_ransac_poly
from hipp.kh9pc.restitution.collimation_strategy import (
    CollimationResult,
    CollimationStrategy,
    _featureless_edge,
    _variance_edge,
)

STRIDE = 10
H, W = 3600, 1200
LINE_TOP, EDGE_TOP = 900, 600      # top collimation line / true exposure edge (offset 300 px outward = up)
LINE_BOT, EDGE_BOT = 1900, 2200    # bottom collimation line / true exposure edge (offset 300 px outward = down)
OFFSET = 300
MARGIN_DN = 60                     # smooth unexposed film margin
LOGO_COLS = int(W * 0.10)          # leftmost 10 % of columns carry the artifact
LOGO_REACH = 700                   # block extends 700 px beyond the true bottom edge


def _write_raster(path: Path, *, logo: bool = True, smooth_content: bool = False) -> None:
    """F004-like restituted frame: two bright collimation lines, high-contrast
    textured content between the exposure edges, smooth margin beyond. When
    ``logo`` the leftmost columns carry a bright TEXTURED block abutting and
    extending 700 px beyond the true bottom edge (the USGS-logo analogue). When
    ``smooth_content`` the content carries no texture, so the refit finds no
    edge and must fall back to the derived line+offset edge."""
    rng = np.random.default_rng(0)
    img = np.full((H, W), MARGIN_DN, np.uint8)
    if smooth_content:
        img[EDGE_TOP:EDGE_BOT] = 150                      # flat -> no texture edge
    else:
        img[EDGE_TOP:EDGE_BOT] = rng.integers(20, 240, size=(EDGE_BOT - EDGE_TOP, W)).astype(np.uint8)
    img[LINE_TOP - 4 : LINE_TOP + 4] = 250
    img[LINE_BOT - 4 : LINE_BOT + 4] = 250
    if logo:
        img[EDGE_BOT : EDGE_BOT + LOGO_REACH, :LOGO_COLS] = rng.integers(
            20, 240, size=(LOGO_REACH, LOGO_COLS)
        ).astype(np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", width=W, height=H, count=1, dtype="uint8"
    ) as dst:
        dst.write(img, 1)


def _strategy_with_lines(src: rasterio.DatasetReader, **kwargs) -> CollimationStrategy:
    """CollimationStrategy pre-loaded with the (known) collimation line models so
    ``_refit_edges_from_lines`` can be exercised directly."""
    x_cols = (np.arange(100) + 0.5) * W / 100

    def line_result(row: int) -> CollimationResult:
        model = fit_ransac_poly(x_cols, np.full(100, float(row)), degree=5, residual_threshold=80, max_trials=1000)
        sub = SubImage(src, Window(0, row - 150, W, 300), out_shape=(1, 300 // STRIDE, 100))
        z = np.zeros((100, 2))
        return CollimationResult(z.astype(int), z.astype(int), z, 1.0, model, sub)

    strat = CollimationStrategy(**kwargs)
    strat._results = {"top": line_result(LINE_TOP), "bottom": line_result(LINE_BOT)}
    strat._separation_ok = True
    return strat


def _edge_dist(strat: CollimationStrategy, side: str) -> np.ndarray:
    """Per-column outward distance (px) from the collimation line to the refit edge."""
    cols = np.linspace(0, W, 100).reshape(-1, 1)
    edge = strat.poly_strategy._results[side].model.predict(cols).ravel()
    line = strat._results[side].model.predict(cols).ravel()
    return (line - edge) if side == "top" else (edge - line)


def test_variance_detector_is_fooled_by_the_logo(tmp_path: Path) -> None:
    """Precondition for the regression: the raw texture detector reports the
    logo columns' edge ~1000 px OUTWARD of the line (far beyond edge_band_outer),
    while clean columns report the true edge at the 300 px offset. This is the
    exact failure that used to leak into the free RANSAC fit."""
    src_path = tmp_path / "f004.tif"
    _write_raster(src_path)
    with rasterio.open(src_path) as src:
        row0, row1 = LINE_BOT - 5 * STRIDE, min(LINE_BOT + 1700, H)
        sub = SubImage(src, Window(0, row0, W, row1 - row0), out_shape=(1, (row1 - row0) // STRIDE, 100))
        line_local = int(round(sub.to_local_y(LINE_BOT)))
        red = redacted_region_mask(sub.band, max_dn=20, dilate=3)

        def dist(col: int):
            r = _variance_edge(sub.band[:, col][line_local:], red[:, col][line_local:], from_end=False)
            return None if r is None else sub.to_global_y(r + line_local) - LINE_BOT

        for c in range(3):  # logo columns
            assert dist(c) is not None and dist(c) > 550, f"logo col {c} not fooled: {dist(c)}"
        for c in (50, 80):  # clean columns
            assert dist(c) is not None and abs(dist(c) - OFFSET) < 120, f"clean col {c}: {dist(c)}"


def test_logo_cannot_bend_the_refit_edge(tmp_path: Path) -> None:
    """With the band gate the fitted bottom edge stays within a tight band of the
    line at EVERY column -- the leftmost (logo) columns land at the true offset,
    not the logo rim -- so the black frame / logo can never enter the crop."""
    src_path = tmp_path / "f004.tif"
    _write_raster(src_path)
    with rasterio.open(src_path) as src:
        strat = _strategy_with_lines(src)
        strat._refit_edges_from_lines(src, 0, W, 1700)
        for side in ("top", "bottom"):
            d = _edge_dist(strat, side)
            assert d.min() >= strat.edge_band_inner - 5, f"{side} edge dived inside the line: {d.min()}"
            assert d.max() <= strat.edge_band_outer + 5, f"{side} edge admitted margin: {d.max()}"
        d_bot = _edge_dist(strat, "bottom")
        # leftmost (logo) columns must sit near the true 300 px offset, NOT the
        # ~1000 px logo rim that fooled the detector.
        assert d_bot[0] < OFFSET + 120, f"left edge pulled toward the logo: {d_bot[0]}"


def test_gate_is_load_bearing(tmp_path: Path) -> None:
    """Disabling the gate (huge edge_band_outer) admits the logo columns' far
    transitions into the fit input; the default gate excludes them. This is the
    deterministic difference the fix makes (independent of RANSAC's random
    consensus)."""
    src_path = tmp_path / "f004.tif"
    _write_raster(src_path)

    def accepted_max_dist(band_outer: int) -> float:
        with rasterio.open(src_path) as src:
            strat = _strategy_with_lines(src, edge_band_outer=band_outer)
            strat._refit_edges_from_lines(src, 0, W, 1700)
            rb = strat.poly_strategy._results["bottom"]
            rg = np.asarray(rb.ruptures_global)
            line = strat._results["bottom"].model.predict(rg[:, 0].reshape(-1, 1)).ravel()
            return float((rg[:, 1] - line).max())

    assert accepted_max_dist(10**9) > 900, "gate-off must admit the ~1000 px logo transitions"
    assert accepted_max_dist(550) <= 550 + 5, "default gate must exclude the logo transitions"


def test_starved_refit_falls_back_to_derived_edge(tmp_path: Path) -> None:
    """When the texture refit finds no valid columns (smooth content), the edge
    is the derived line + conservative offset -- erring INWARD -- NOT the old
    DN-threshold film-frame boundary and NOT any far artifact."""
    src_path = tmp_path / "smooth.tif"
    _write_raster(src_path, logo=False, smooth_content=True)
    with rasterio.open(src_path) as src:
        strat = _strategy_with_lines(src)
        strat._refit_edges_from_lines(src, 0, W, 1700)
        for side in ("top", "bottom"):
            d = _edge_dist(strat, side)
            assert np.allclose(d, strat.edge_offset_from_line, atol=2), (
                f"{side} did not fall back to line+offset: {d.min()}..{d.max()}"
            )


# ---- round 2: DN featureless detector + inward section envelope ----

def _write_black_margin_raster(path: Path) -> None:
    """Bottom edge = textured content ending in a near-black frame (DN 5), the
    merged-scan case David points at. No logo; a plain content->black transition
    the DN detector must accept in essentially every column."""
    rng = np.random.default_rng(3)
    img = np.full((H, W), 5, np.uint8)                         # near-black frame
    img[EDGE_TOP:EDGE_BOT] = rng.integers(30, 240, size=(EDGE_BOT - EDGE_TOP, W)).astype(np.uint8)
    img[LINE_TOP - 4 : LINE_TOP + 4] = 250
    img[LINE_BOT - 4 : LINE_BOT + 4] = 250
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=1, dtype="uint8") as dst:
        dst.write(img, 1)


def test_featureless_detector_accepts_nearly_all_content_columns(tmp_path: Path) -> None:
    """The DN featureless detector must ACCEPT nearly every content column (the
    round-1 variance gate refused the obviously-decidable ones). On a plain
    content->black bottom edge it accepts ~all columns at the true offset, far
    more than the variance detector."""
    src_path = tmp_path / "blackmargin.tif"
    _write_black_margin_raster(src_path)
    with rasterio.open(src_path) as src:
        row0, row1 = LINE_BOT - 5 * STRIDE, min(LINE_BOT + 1700, H)
        sub = SubImage(src, Window(0, row0, W, row1 - row0), out_shape=(1, (row1 - row0) // STRIDE, 100))
        ll = int(round(sub.to_local_y(LINE_BOT)))
        red = redacted_region_mask(sub.band, max_dn=20, dilate=3)
        n = sub.band.shape[1]
        feat_ok = var_ok = 0
        for c in range(n):
            out = sub.band[ll:, c]
            rf = _featureless_edge(out)
            if rf is not None and 90 <= (sub.to_global_y(rf + ll) - LINE_BOT) <= 550:
                feat_ok += 1
            rv = _variance_edge(out, red[ll:, c], from_end=False)
            if rv is not None and 90 <= (sub.to_global_y(rv + ll) - LINE_BOT) <= 550:
                var_ok += 1
        frac = feat_ok / n
        assert frac >= 0.90, f"featureless detector accepted only {feat_ok}/{n} in-band"
        assert feat_ok > var_ok, f"featureless ({feat_ok}) must beat variance ({var_ok})"


def _write_stepped_raster(path: Path, black_left: int, black_right: int) -> None:
    """Two scan sections (left / right halves) whose near-black frame starts at
    DIFFERENT bottom rows -- the section-mosaicking step. Content is textured
    down to each section's black start."""
    rng = np.random.default_rng(5)
    img = np.full((H, W), 5, np.uint8)
    img[EDGE_TOP : LINE_BOT + 700] = rng.integers(30, 240, size=(LINE_BOT + 700 - EDGE_TOP, W)).astype(np.uint8)
    img[black_left:, : W // 2] = 5
    img[black_right:, W // 2 :] = 5
    img[:EDGE_TOP] = 5
    img[LINE_TOP - 4 : LINE_TOP + 4] = 250
    img[LINE_BOT - 4 : LINE_BOT + 4] = 250
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=1, dtype="uint8") as dst:
        dst.write(img, 1)


def test_stepped_sections_geometry_smooth_crop_conservative(tmp_path: Path) -> None:
    """Two sections whose black frame starts at different rows (David 2026-07-22
    round 3): the GEOMETRY model must stay SMOOTH through the step (a step in the
    edge is shear in the warp), while the separate CONSERVATIVE crop line steps
    inward and never crosses below the shallower section's black start."""
    from types import SimpleNamespace

    black_left, black_right = 2300, 2150          # right section's black is shallower
    src_path = tmp_path / "stepped.tif"
    _write_stepped_raster(src_path, black_left, black_right)
    with rasterio.open(src_path) as src:
        strat = _strategy_with_lines(src)
        strat.collimation_line_dist = LINE_BOT - LINE_TOP   # synthetic line spacing -> warp scale ~1
        strat._refit_edges_from_lines(src, 0, W, 1700)
        res = strat.poly_strategy._results["bottom"]
        # the delivered crop needs the vertical detector's content-column span
        strat.poly_strategy.vertical_detector = SimpleNamespace(edges_=(0, W), detected_width_=W)
        strat._FittingClass__raster_filepath_ = src_path   # normally set by fit()
        tf = strat.transformation_
    cols = np.linspace(0, W, 200)
    geom = res.model.predict(cols.reshape(-1, 1)).ravel()          # feeds the warp
    crop = res.crop_model.predict(cols.reshape(-1, 1)).ravel()     # valid-pixel bound
    # GEOMETRY smooth: a degree-2 poly has a tiny second difference everywhere
    # (no step -> no local vertical-scale discontinuity -> no shear).
    assert np.abs(np.diff(geom, 2)).max() < 5, "geometry model stepped -> would shear the warp"
    # the smooth geometry DOES average across the step (that is why a crop/mask,
    # not the geometry, must enforce conservatism)
    assert geom[cols >= W // 2].max() > black_right + 5, "smooth model should cross the shallow step"
    # the CONSERVATIVE crop line excludes the shallow section's black in every column
    assert crop[cols >= W // 2].max() <= black_right + 3, f"crop admitted black: {crop[cols >= W // 2].max()}"
    assert crop[cols < W // 2].max() <= black_left + 3, f"left crop admitted black: {crop[cols < W // 2].max()}"

    # DELIVERED collimation crop (round 4): the output rectangle's bottom bound is
    # trimmed to the SHALLOWEST content edge, so the shallow section's black is
    # excluded from the product -- and it is the innermost (shallow) edge, not the
    # deep one (which would admit the shallow section's black).
    crop_bot = tf.crop_offset[1] + tf.output_size[1]              # warped bottom row of the product
    assert crop_bot <= black_right + 5, f"delivered crop admits shallow black: {crop_bot} > {black_right}"
    assert crop_bot < black_left - 30, f"delivered crop did not take the innermost bottom edge: {crop_bot}"
    # map that bottom row back to source at a right-section column: it must land
    # at/above the shallow content edge, never down in the black.
    src_row = float(tf.deformation(np.array([[3 * W // 4, crop_bot]], dtype=np.float32))[0, 1])
    assert src_row <= black_right + 8, f"delivered crop bottom maps into black: {src_row}"
