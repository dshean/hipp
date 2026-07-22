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
