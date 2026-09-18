"""
Copyright (c) 2026 HIPP developers
Description: Quality control figures for all KH-9 PC restitution strategies.
    Each ``plot_*`` function targets one fitted object and returns a matplotlib Figure.
    ``get_figures`` dispatches to the right set of plots based on the strategy type,
    and ``save_figures`` writes them all to disk.
"""

import logging
from pathlib import Path

from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rasterio.warp import Resampling
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.restitution.base import FittingClass, Transformation
from hipp.kh9pc.restitution.collimation_strategy import CollimationStrategy
from hipp.kh9pc.restitution.fiducial_strategy import FiducialStrategy
from hipp.kh9pc.restitution.flat_strategy import FlatStrategy
from hipp.kh9pc.restitution.mixed_strategy import MixedStrategy
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy
from hipp.kh9pc.restitution.vertical_detector import VerticalDetector

logger = logging.getLogger(__name__)


# --- Vertical ---
def plot_vertical_ruptures(detector: VerticalDetector) -> Figure:
    """Band profiles with detected rupture positions for left and right edges."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, result in zip(axes, ["left", "right"], [detector.left_, detector.right_]):
        ax.plot(result.profile, color="gray")
        ax.axvline(x=result.edge_local, color="red", label=f"edge (local={result.edge_local})")
        ax.set_title(f"{side} profile \n(global col={result.position}, ratio={result.gradient_ratio:.2f})")
        ax.set_xlabel("local column index")
        ax.set_ylabel("intensity")
        ax.legend()

    fig.suptitle(detector.raster_filepath_.stem, fontsize=12, fontweight="bold")
    return fig


def plot_vertical_edges(
    detector: VerticalDetector,
    window_width: int = 10000,
    out_height: int = 800,
) -> Figure:
    """Thumbnails around the left and right edge positions."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    with rasterio.open(detector.raster_filepath_) as src:
        for ax, side, edge_col in zip(axes, ["left", "right"], detector.edges_):
            window = Window(edge_col - window_width // 2, 0, window_width, src.height)
            scale = out_height / window.height
            out_shape = (1, out_height, int(window.width * scale))
            sub_image = SubImage(src, window=window, out_shape=out_shape)

            w = sub_image.window
            extent = [w.col_off, w.col_off + w.width, w.row_off + w.height, w.row_off]
            ax.imshow(sub_image.band, cmap="gray", aspect="auto", extent=extent)
            # dshean 2026-08-27: show WHAT decided the edge -- every row-coherent DN-step candidate in
            # view (thin, labelled with its z and single/trunc flags), the true exposure edges when
            # the crop is anchored elsewhere, and the expected-width-derived position for the other
            # side -- so a derived edge, a film boundary and an exposure edge can be told apart.
            x0, x1 = extent[0], extent[1]
            for c in getattr(detector, "candidates_", []) or []:
                if x0 <= c["x"] <= x1 and abs(c["x"] - edge_col) > 8:
                    flag = ("S" if c[f"single_{side}"] else "") + ("T" if c[f"trunc_{side}"] else "")
                    ax.axvline(c["x"], color="0.7" if c.get("pool") == "coarse" else "0.85", lw=0.7, ls="--")
                    ax.text(c["x"], extent[3] + 0.04 * (extent[2] - extent[3]), f"z{c['z']:+.0f}{flag}", color="w",
                            fontsize=6, rotation=90, va="top", ha="right")
            true_edges = getattr(detector, "true_edges_", None)
            if true_edges:
                te = true_edges[0] if side == "left" else true_edges[1]
                if abs(te - edge_col) > 8 and x0 <= te <= x1:
                    ax.axvline(te, color="orange", lw=1.2, ls="--", label=f"true exposure edge {int(te)}")
            exp_w = getattr(detector, "expected_width_", None)
            if exp_w:
                other = detector.edges_[1] - exp_w if side == "left" else detector.edges_[0] + exp_w
                if abs(other - edge_col) > 8 and x0 <= other <= x1:
                    ax.axvline(other, color="deepskyblue", lw=1.0, ls=":", label=f"expected width from the other edge {int(other)}")
            ax.axvline(x=edge_col, color="red", label=f"chosen {side} edge {int(edge_col)}")
            ax.set_title(f"{side} edge (col={edge_col})")
            ax.set_xlabel("column (full-res px)")
            ax.set_ylabel("row (full-res px)")
            ax.legend(fontsize=6, loc="lower right", framealpha=0.6)

    src_desc = getattr(detector, "edge_source_", None) or "?"
    fig.suptitle(f"{detector.raster_filepath_.stem}   edges from {src_desc}"
                 f"{'   (grey dashed = DN-step candidates: S strict single, T truncation)' if getattr(detector, 'candidates_', None) else ''}",
                 fontsize=9, fontweight="bold")
    return fig


# --- Flat ---
def plot_flat_ruptures(detector: FlatStrategy) -> Figure:
    """Band profiles (collapsed horizontally) with detected rupture row for top and bottom."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, side, result in zip(axes, ["top", "bottom"], [detector.top_, detector.bottom_]):
        profile = result.sub_image.band.flatten()
        ax.plot(profile, color="steelblue", linewidth=1)
        ax.axvline(result.rupture_local, color="red", linewidth=1.5, label=f"rupture={result.rupture_local}")
        ax.set_title(f"{side} band profile")
        ax.set_xlabel("row index (downsampled)")
        ax.set_ylabel("intensity")
        ax.legend(fontsize=8)

    return fig


def plot_flat_edges(detector: FlatStrategy, margin_fraction: float = 0.03) -> Figure:
    """Thumbnails around the top and bottom edge positions with detected line overlaid."""
    left, _ = detector.vertical_detector.edges_
    roi_w = detector.vertical_detector.detected_width_

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    with rasterio.open(detector.raster_filepath_) as src:
        margin = int(margin_fraction * src.height)

        for ax, side, result in zip(axes, ["top", "bottom"], [detector.top_, detector.bottom_]):
            row_off = max(0, result.position - margin)
            row_end = min(src.height, result.position + margin)
            win_h = row_end - row_off
            thumb = src.read(
                1,
                window=Window(left, row_off, roi_w, win_h),
                out_shape=(512, 512),
                resampling=Resampling.average,
            )
            extent = [left, left + roi_w, row_end, row_off]
            ax.imshow(thumb, cmap="gray", aspect="auto", extent=extent)
            ax.axhline(result.position, color="yellow", linewidth=1.5)
            ax.set_title(f"{side} edge — position={result.position} px")
            ax.set_xlabel("column (full-res px)")
            ax.set_ylabel("row (full-res px)")

    return fig


# --- Poly ---


def plot_poly_edges(detector: PolyStrategy, crop_is_delivered: bool = True) -> Figure:
    """Subimage thumbnails with RANSAC inliers/outliers, the SMOOTH edge model
    (blue, the geometry that feeds the warp) and, when present, the conservative
    inward-envelope line (dashed orange). The two are drawn separately so
    geometry (must be smooth -- a step is shear) and conservatism (steps inward
    to exclude black frame) can be reviewed apart (David 2026-07-22 round 3).
    ``crop_is_delivered``: True when the envelope actually cuts the product
    (pure Poly/Mixed fallback); False under CollimationStrategy, where the
    delivered crop is the fixed line-offset (2026-07-22 ruling) and the
    envelope is QC/sentinel only."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    # Resolve the DELIVERED geometry before reading it (2026-09-18). POLY_EDGE_CLASS_POLICY is
    # applied inside _compute_transformation, so on a strategy whose transformation has not been
    # built yet, _edge_class_ holds the NATIVE per-side picks and the curve drawn below is a model
    # the warp never used -- while the legend still calls it "DELIVERED". Touching the cached
    # property makes the figure agree with the product.
    try:
        detector.transformation_      # cached; does not re-fit
    except Exception:  # noqa: BLE001 - a figure must never take the run down
        pass
    _cls_all = getattr(detector, "_edge_class_", {})
    if _cls_all:
        _mixed = ("rupture" in _cls_all.values()) and (len(set(_cls_all.values())) > 1)
        fig.suptitle("edge-model classes: top=%s bottom=%s%s"
                     % (_cls_all.get("top", "?"), _cls_all.get("bottom", "?"),
                        "   *** MIXED -- delivered datum shifted ***" if _mixed else ""),
                     fontsize=9, color=("red" if _mixed else "black"))

    for ax, side, result in zip(axes, ["top", "bottom"], [detector.top_, detector.bottom_]):
        # GLOBAL full-res raster coordinates on both axes (diagnostic default:
        # window placement/size must be readable at a glance, David 2026-07-21)
        w = result.sub_image.window
        extent = [w.col_off, w.col_off + w.width, w.row_off + w.height, w.row_off]
        ax.imshow(result.sub_image.band, cmap="gray", aspect="auto", extent=extent)

        inlier_mask = result.model.inlier_mask_
        pts = result.ruptures_global
        ax.scatter(pts[~inlier_mask, 0], pts[~inlier_mask, 1], s=12, c="red", label="outliers")
        ax.scatter(pts[inlier_mask, 0], pts[inlier_mask, 1], s=12, c="green", label="inliers")

        order = np.argsort(result.ruptures_global[:, 0].astype(float))
        x_global = result.ruptures_global[order, 0].astype(float)
        # 2026-09-18: `result.model` is the detector's RUPTURE model. It is the delivered geometry ONLY
        # when _content_edge_model() fell back to it; normally the warp uses the edge-oracle or the
        # content-walk model instead. Labelling it "model (geometry)" hid exactly that (nepal ops251:
        # four frames moved their delivered datum ~1000 rows between runs, every gate green). Draw the
        # DELIVERED model separately and name the class it came from.
        ax.plot(x_global, result.model.predict(x_global.reshape(-1, 1)).ravel(),
                color="tab:blue", linewidth=1, linestyle=":", label="rupture model (detector)")
        _cls = getattr(detector, "_edge_class_", {}).get(side)
        try:
            _delivered, _ = detector._content_edge_model(side)
        except Exception:  # noqa: BLE001 - a figure must never take the run down
            _delivered = None
        if _delivered is not None:
            ax.plot(x_global, np.asarray(_delivered.predict(x_global.reshape(-1, 1))).ravel(),
                    color="blue", linewidth=1.6, label="DELIVERED geometry (%s)" % (_cls or "?"))
        crop_model = getattr(result, "crop_model", None)
        if crop_model is not None:
            cm = np.asarray(crop_model.predict(x_global.reshape(-1, 1))).ravel()
            ax.plot(x_global, cm, color="darkorange", linewidth=1.2, linestyle="--",
                    label="content envelope (conservative)")
            # innermost row of the envelope (deepest for the top edge,
            # shallowest for the bottom). Under the pure Poly/Mixed fallback
            # this is the actual row the product is cut at; under
            # CollimationStrategy the delivered crop is the fixed line-offset
            # (David 2026-07-22) and this row is QC/sentinel only.
            crop_row = float(cm.max()) if side == "top" else float(cm.min())
            lbl = (f"delivered crop = row {int(crop_row)}" if crop_is_delivered
                   else f"innermost content row {int(crop_row)} (QC)")
            ax.axhline(crop_row, color="red", linewidth=1.0, linestyle=":", label=lbl)

        ax.set_title(f"{side} edge")
        ax.set_xlabel("column (full-res px)")
        ax.set_ylabel("row (full-res px)")
        ax.legend(loc="best", fontsize=8)

    return fig


# Fixed axes so distortion figures compare ACROSS frames (dshean
# 2026-08-23); measured over the casa_block: curves span +-600 px
# frame-to-frame, top-bottom difference stays within ~105 px.
_DISTORTION_YLIM_PX = 700.0
_DISTORTION_DIFF_YLIM_PX = 150.0


def _plot_distortion_pair(top_res, bottom_res) -> Figure:
    """Two panels (dshean 2026-08-23 spec): top & bottom distortion curves
    on a FIXED y axis, and their difference (top - bottom, the differential
    signal camera geometry cannot absorb) on its own fixed y axis."""
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, constrained_layout=True)

    tx, ty = top_res.distortion[:, 0], top_res.distortion[:, 1]
    bx, by = bottom_res.distortion[:, 0], bottom_res.distortion[:, 1]
    ax.plot(tx, ty, label=f"top (peak {np.nanmax(np.abs(ty)):.0f} px)")
    ax.plot(bx, by, label=f"bottom (peak {np.nanmax(np.abs(by)):.0f} px)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(_DISTORTION_YLIM_PX, -_DISTORTION_YLIM_PX)  # inverted, fixed
    ax.legend(fontsize=8)
    ax.set_title("global distortion (top & bottom)")
    ax.set_ylabel("distortion (px)")

    diff = ty - np.interp(tx, bx, by)  # bottom resampled onto top columns
    axd.plot(tx, diff, color="purple",
             label=f"top - bottom (peak {np.nanmax(np.abs(diff)):.0f} px)")
    axd.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axd.set_ylim(_DISTORTION_DIFF_YLIM_PX, -_DISTORTION_DIFF_YLIM_PX)
    axd.legend(fontsize=8)
    axd.set_title("differential distortion (top - bottom)")
    axd.set_xlabel("column (px)")
    axd.set_ylabel("difference (px)")

    return fig


def plot_poly_distortions(detector: PolyStrategy) -> Figure:
    """Residual distortion curves (deviation from mean) for top and bottom polynomial fits."""
    return _plot_distortion_pair(detector.top_, detector.bottom_)


# --- Collimation ---


def _collimation_context_panel(ax, detector: CollimationStrategy, src, side: str,
                               context_px: int = 1500, out_cols: int = 1400, out_rows: int = 700) -> None:
    """One margin: full context from the raster edge (row 0 / bottom row) through the
    line + context_px, scan-canvas NODATA shaded, detector search band outlined,
    RANSAC inliers/outliers, model and delivered crop. dshean 2026-08-25."""
    result = detector.top_ if side == "top" else detector.bottom_
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#ffb3b3")
    w = result.sub_image.window
    cols = result.peaks_global[:, 0]
    line_rows = np.asarray(result.model.predict(cols.reshape(-1, 1))).ravel()
    # display window capped at panel_rows (dshean 2026-08-26: 3000 rows, not the full 4000-row
    # search slab -- less valid exposed area, more margin detail); the detection slab is unchanged
    panel_rows = 3000
    if side == "top":
        r0 = 0
        r1 = int(min(src.height, max(min(w.row_off + w.height, panel_rows), np.nanmax(line_rows) + context_px)))
    else:
        r0 = int(max(0, min(max(w.row_off, src.height - panel_rows), np.nanmin(line_rows) - context_px)))
        r1 = int(src.height)
    win = Window(int(w.col_off), r0, int(w.width), r1 - r0)
    band = SubImage(src, window=win, out_shape=(1, out_rows, out_cols)).band.astype(np.float32)
    nod = band <= 0
    valid = band[~nod]
    vmin, vmax = (np.percentile(valid, [2, 98]) if valid.size else (0, 255))
    extent = [win.col_off, win.col_off + win.width, win.row_off + win.height, win.row_off]
    ax.imshow(np.ma.masked_where(nod, band), cmap=cmap, aspect="auto",
              extent=extent, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.add_patch(patches.Rectangle((w.col_off, w.row_off), w.width, w.height,
                                   fill=False, ec="cyan", lw=0.8, ls="--", label="detector search band"))
    inliers = result.model.inlier_mask_
    peaks = result.peaks_global
    ax.scatter(peaks[~inliers, 0], peaks[~inliers, 1], s=10, c="red", label="outliers")
    ax.scatter(peaks[inliers, 0], peaks[inliers, 1], s=10, c="green", label="inliers")
    ax.plot(cols, line_rows, color="blue", linewidth=1, label="model")
    off = getattr(detector, "crop_offset_from_line", None)
    if off is not None:
        sign = -1.0 if side == "top" else 1.0
        ax.plot(cols, line_rows + sign * off, color="red", linewidth=1.0, linestyle=":",
                label=f"delivered crop = line {'-' if side == 'top' else '+'} {off} px")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(f"{side} collimation line (rows {r0}-{r1}; inliers {result.inlier_ratio:.2f})", fontsize=10)
    ax.set_xlabel("column (full-res px)")
    ax.set_ylabel("row (full-res px)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(patches.Patch(fc="#ffb3b3", ec="none", label="nodata (scan canvas)"))
    ax.legend(handles=handles, loc="best", fontsize=7)


def _poly_context_panel(ax, poly: PolyStrategy, src, side: str, crop_rows: tuple | None = None,
                        context_px: int = 1500, out_cols: int = 1400, out_rows: int = 700) -> None:
    """One margin of a PolyStrategy frame, the CollimationStrategy panel's counterpart.

    Added 2026-09-18 (dshean: "I need to see both top and bottom on
    D3C1205-200251A055_restitution_sheet.png indicated to advise further"). Every nepal frame
    is PolyStrategy, and until now the sheet drew top/bottom margins only for
    CollimationStrategy -- a PolyStrategy sheet had a blank "no collimation-line fit" box
    where this evidence belongs, so the ~1000-row A055 datum error was invisible on the one
    page that gets reviewed.

    Draws, in FULL-RES source rows: the margin from the raster edge inward, nodata shaded,
    the RANSAC rupture inliers/outliers, the DELIVERED edge model (after
    POLY_EDGE_CLASS_POLICY, labelled with the class that actually shipped), the rupture
    model for contrast, and the delivered crop row.
    """
    result = poly._results[side] if side in getattr(poly, "_results", {}) else (
        poly.top_ if side == "top" else poly.bottom_)
    cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#ffb3b3")
    w = result.sub_image.window
    panel_rows = 3600
    if side == "top":
        r0, r1 = 0, int(min(src.height, max(w.row_off + w.height, panel_rows)))
    else:
        r0, r1 = int(max(0, min(w.row_off, src.height - panel_rows))), int(src.height)
    win = Window(0, r0, int(src.width), r1 - r0)
    band = SubImage(src, window=win, out_shape=(1, out_rows, out_cols)).band.astype(np.float32)
    nod = band <= 0
    valid = band[~nod]
    vmin, vmax = (np.percentile(valid, [2, 98]) if valid.size else (0, 255))
    extent = [0, src.width, win.row_off + win.height, win.row_off]
    ax.imshow(np.ma.masked_where(nod, band), cmap=cmap, aspect="auto",
              extent=extent, vmin=vmin, vmax=vmax, interpolation="nearest")

    pts = result.ruptures_global
    xs = np.linspace(float(pts[:, 0].min()), float(pts[:, 0].max()), 200)
    inl = result.model.inlier_mask_
    ax.scatter(pts[~inl, 0], pts[~inl, 1], s=8, c="red", label="rupture outliers")
    ax.scatter(pts[inl, 0], pts[inl, 1], s=8, c="green", label="rupture inliers")
    ax.plot(xs, np.asarray(result.model.predict(xs.reshape(-1, 1))).ravel(),
            color="tab:blue", lw=1.0, ls=":", label="rupture model (detector)")
    # The DELIVERED model is whatever the live policy resolved -- read the recorded class,
    # never re-derive it here (a figure that re-picks can name a model the warp never used).
    cls = getattr(poly, "_edge_class_", {}).get(side)
    try:
        delivered, _ = poly._content_edge_model(side)
        ax.plot(xs, np.asarray(delivered.predict(xs.reshape(-1, 1))).ravel(),
                color="blue", lw=1.8, label="DELIVERED geometry (%s)" % (cls or "?"))
    except Exception:  # noqa: BLE001 - a figure must never take the run down
        pass
    if crop_rows is not None and crop_rows[0 if side == "top" else 1] is not None:
        cr = float(crop_rows[0 if side == "top" else 1])
        ax.axhline(cr, color="red", lw=1.4, ls="--", label="delivered crop row %d (source px)" % int(cr))
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title("%s margin (source rows %d-%d; class %s)" % (side, r0, r1, cls or "?"), fontsize=10)
    ax.set_xlabel("column (full-res px)"); ax.set_ylabel("row (full-res px)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(patches.Patch(fc="#ffb3b3", ec="none", label="nodata (scan canvas)"))
    ax.legend(handles=handles, loc="best", fontsize=7)


def plot_collimation_edges(detector: CollimationStrategy, context_px: int = 1500,
                           out_cols: int = 1400, out_rows: int = 700) -> Figure:
    """Both margins, full context (see _collimation_context_panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
    with rasterio.open(detector.raster_filepath_) as src:
        for ax, side in zip(axes, ["top", "bottom"]):
            _collimation_context_panel(ax, detector, src, side, context_px, out_cols, out_rows)
    return fig


def _read_decimated(src, window: Window, out_w: int, out_h: int):
    return SubImage(src, window=window, out_shape=(1, max(1, out_h), max(1, out_w))).band.astype(np.float32)


def plot_restitution_sheet(fitting_class: FittingClass, entity: str | None = None,
                           extra_text: str | None = None, product: str | Path | None = None,
                           context_px: int = 1500) -> Figure:
    """ONE page per frame = what gets reviewed. Layout (dshean 2026-08-25, merging
    the useful parts of the 08-22 harness sheet, all driven by the PRODUCTION fit):
      1 raw mosaic + delivered crop polygon (lines -/+ crop offset, vertical edges)
      2/3 top / bottom margin in full context from the raster edge (nodata shaded,
          search band, picks, model, delivered crop) + row median-DN profile
      4 left / right vertical-edge panels + column median-DN profile
      5 native-resolution zoom tiles on the line at 4 x positions (top, bottom)
      6 final restituted product (when ``product`` exists)
      7 verdict / metrics text (strategy, separation vs expected, inliers, pairing,
        pitch, crop offset, generation tag)."""
    import datetime as _dt
    sel = getattr(fitting_class, "selected_strategy_", fitting_class)
    name = type(sel).__name__
    is_col = isinstance(sel, CollimationStrategy)
    gen = getattr(sel, "generation_tag", None) or getattr(fitting_class, "generation_tag", None)
    fig = plt.figure(figsize=(13, 36), constrained_layout=True)
    gs = fig.add_gridspec(9, 4, height_ratios=[0.9, 1.5, 1.5, 1.6, 0.7, 2.6, 0.7, 0.9, 0.55],
                          width_ratios=[0.18, 1, 1, 1])
    rf = Path(str(getattr(sel, "raster_filepath_", "")))
    title = (f"{entity or rf.stem}  --  restitution evidence  --  {name}\n"
             f"generation: {gen or 'untagged'}  |  rendered {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    fig.suptitle(title, fontsize=12)
    lines = [f"strategy: {name}"]
    cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#ffb3b3")
    try:
        with rasterio.open(rf) as src:
            H, W = src.height, src.width
            poly = sel.poly_strategy if hasattr(sel, "poly_strategy") else (sel if isinstance(sel, PolyStrategy) else None)
            left = right = None
            if poly is not None and getattr(poly, "is_fitted", False):
                try:
                    left, right = poly.vertical_detector.edges_
                except Exception:  # noqa: BLE001
                    pass
            off = getattr(sel, "crop_offset_from_line", 75)
            # ---- 1: raw mosaic overview + crop polygon ----
            ax = fig.add_subplot(gs[0, :])
            ov = _read_decimated(src, Window(0, 0, W, H), 2400, max(60, int(2400 * H / W)))
            nod = ov <= 0; v = ov[~nod]; vmin, vmax = (np.percentile(v, [2, 98]) if v.size else (0, 255))
            ax.imshow(np.ma.masked_where(nod, ov), cmap=cmap, aspect="auto", extent=[0, W, H, 0], vmin=vmin, vmax=vmax, interpolation="nearest")
            if is_col:
                xs = np.linspace(left if left is not None else 0, right if right is not None else W, 200)
                yt = np.asarray(sel.top_.model.predict(xs.reshape(-1, 1))).ravel(); yb = np.asarray(sel.bottom_.model.predict(xs.reshape(-1, 1))).ravel()
                ax.plot(xs, yt, color="lime", lw=1.0, label="collimation lines (fit)"); ax.plot(xs, yb, color="lime", lw=1.0)
                ax.plot(xs, yt - off, color="red", lw=0.9, ls=":", label=f"delivered crop (lines -/+ {off} px)"); ax.plot(xs, yb + off, color="red", lw=0.9, ls=":")
            if left is not None:
                ax.axvline(left, color="cyan", lw=0.9, ls="--", label=f"vertical edges (left {int(left)}, right {int(right)})"); ax.axvline(right, color="cyan", lw=0.9, ls="--")
            ax.set_title(f"raw mosaic {W}x{H} + delivered crop", fontsize=10); ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            # ---- 2/3: margins with row profiles ----
            if is_col:
                for k, side in enumerate(["top", "bottom"]):
                    axp = fig.add_subplot(gs[1 + k, 0]); axm = fig.add_subplot(gs[1 + k, 1:], sharey=axp)
                    _collimation_context_panel(axm, sel, src, side, context_px)
                    r0, r1 = (int(axm.get_ylim()[1]), int(axm.get_ylim()[0]))
                    prof = _read_decimated(src, Window(0, max(0, r0), W, max(1, r1 - r0)), 512, max(1, r1 - r0))
                    med = np.array([np.median(row[row > 0]) if (row > 0).any() else np.nan for row in prof])
                    axp.plot(med, np.linspace(r0, r1, med.size), color="k", lw=0.8); axp.set_ylim(r1, r0); axp.set_xlabel("median DN", fontsize=8)
                    axp.set_ylabel("row (full-res px)", fontsize=8); axp.tick_params(labelsize=7); axm.tick_params(labelleft=False)
                    axm.set_ylabel("")
            elif poly is not None and getattr(poly, "is_fitted", False):
                # PolyStrategy top/bottom margins (2026-09-18). This grid slot used to be a blank
                # "no collimation-line fit" box, so a PolyStrategy sheet -- every nepal frame --
                # carried NO top/bottom edge evidence and a ~1000-row datum error (ops251 A055)
                # was invisible on the page that gets reviewed.
                _crop_rows = (None, None)
                try:
                    from hipp.kh9pc.restitution.base import scan_scale
                    _tr = poly.transformation_          # cached; does not re-fit
                    _sy = scan_scale(poly.scan_pitch_um, poly.raster_filepath_)[1]
                    # crop_offset / output_size are CANVAS px -- source row = canvas px / sy.
                    # Comparing them to source rows directly biases the bottom edge ~53 px on
                    # a 7.0/6.984 um scan (caught 2026-09-18).
                    _crop_rows = (_tr.crop_offset[1] / _sy,
                                  (_tr.crop_offset[1] + _tr.output_size[1]) / _sy)
                except Exception:  # noqa: BLE001 - a figure must never take the run down
                    pass
                for k, side in enumerate(["top", "bottom"]):
                    axp = fig.add_subplot(gs[1 + k, 0]); axm = fig.add_subplot(gs[1 + k, 1:], sharey=axp)
                    _poly_context_panel(axm, poly, src, side, crop_rows=_crop_rows)
                    r0, r1 = (int(axm.get_ylim()[1]), int(axm.get_ylim()[0]))
                    prof = _read_decimated(src, Window(0, max(0, r0), W, max(1, r1 - r0)), 512, max(1, r1 - r0))
                    med = np.array([np.median(row[row > 0]) if (row > 0).any() else np.nan for row in prof])
                    axp.plot(med, np.linspace(r0, r1, med.size), color="k", lw=0.8)
                    axp.set_ylim(r1, r0); axp.set_xlabel("median DN", fontsize=8)
                    axp.set_ylabel("row (full-res px)", fontsize=8); axp.tick_params(labelsize=7)
                    axm.tick_params(labelleft=False); axm.set_ylabel("")
            else:
                ax0 = fig.add_subplot(gs[1:3, :]); ax0.axis("off"); ax0.text(0.5, 0.5, f"no edge fit ({name})", ha="center", va="center", fontsize=12)
            # ---- 4: vertical edges with column profiles ----
            # ---- 4: left / right exposure edges (dshean 2026-08-26: +-1500 px, not 10 k;
            #         the column-median DN profile BELOW each panel on a shared x axis) ----
            gs4 = gs[3, :].subgridspec(2, 2, height_ratios=[3, 1])
            for k, (side, col) in enumerate([("left", left), ("right", right)]):
                ax = fig.add_subplot(gs4[0, k])
                if col is None:
                    ax.axis("off"); ax.text(0.5, 0.5, f"{side} edge: n/a", ha="center", va="center"); continue
                w0 = int(max(0, col - 1500)); w1 = int(min(W, col + 1500))
                band = _read_decimated(src, Window(w0, 0, w1 - w0, H), 600, 800)
                nod = band <= 0; v = band[~nod]; vmin, vmax = (np.percentile(v, [2, 98]) if v.size else (0, 255))
                ax.imshow(np.ma.masked_where(nod, band), cmap=cmap, aspect="auto", extent=[w0, w1, H, 0], vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.axvline(col, color="cyan", lw=1.0, ls="--"); ax.set_title(f"{side} vertical edge (col {int(col)}, +-1500 px)", fontsize=9); ax.tick_params(labelsize=7)
                cm = np.array([np.median(c[c > 0]) if (c > 0).any() else np.nan for c in band.T])
                axc = fig.add_subplot(gs4[1, k], sharex=ax)
                axc.plot(np.linspace(w0, w1, cm.size), cm, color="k", lw=0.8); axc.axvline(col, color="cyan", lw=0.8, ls="--")
                axc.set_ylabel("col median DN", fontsize=7); axc.tick_params(labelsize=7); axc.set_xlim(w0, w1)
            # ---- 5: native-resolution zoom tiles on the lines: TOP tiles above the corner
            #         block, BOTTOM tiles below it (dshean 2026-08-26) ----
            if is_col:
                x0 = left if left is not None else 0; x1 = right if right is not None else W
                xs4 = np.linspace(x0 + 0.1 * (x1 - x0), x1 - 0.1 * (x1 - x0), 4)
                for k, side in enumerate(["top", "bottom"]):
                    gsr = gs[4 if side == "top" else 6, :].subgridspec(1, 4)
                    r_ = sel.top_ if side == "top" else sel.bottom_
                    for n, xc in enumerate(xs4):
                        yc = float(np.asarray(r_.model.predict(np.array([[xc]]))).ravel()[0])
                        c0, rr0 = int(max(0, xc - 512)), int(max(0, yc - 512)); win = Window(c0, rr0, min(1024, W - c0), min(1024, H - rr0))
                        tile = src.read(1, window=win).astype(np.float32)
                        axz = fig.add_subplot(gsr[0, n]); nod = tile <= 0; v = tile[~nod]
                        vmin, vmax = (np.percentile(v, [2, 98]) if v.size else (0, 255))
                        axz.imshow(np.ma.masked_where(nod, tile), cmap=cmap, extent=[c0, c0 + win.width, rr0 + win.height, rr0], vmin=vmin, vmax=vmax, interpolation="nearest")
                        axz.axhline(yc, color="lime", lw=0.8); axz.axhline(yc - off if side == "top" else yc + off, color="red", lw=0.8, ls=":")
                        axz.set_title(f"{side} line @x={xc/1e3:.0f}k NATIVE", fontsize=7); axz.set_xticks([]); axz.set_yticks([])
                # ---- 5b: the four crop CORNERS at native resolution, 2x2 in their relative
                #          positions (dshean 2026-08-26): vertical exposure edge (cyan), collimation
                #          line (lime), delivered crop (red) ----
                gs5b = gs[5, :].subgridspec(2, 2)
                corners = [("top-left", left, sel.top_, "top", 0, 0), ("top-right", right, sel.top_, "top", 0, 1),
                           ("bottom-left", left, sel.bottom_, "bottom", 1, 0), ("bottom-right", right, sel.bottom_, "bottom", 1, 1)]
                for lab, xc_, r_, side, gr, gc in corners:
                    axz = fig.add_subplot(gs5b[gr, gc])
                    if xc_ is None:
                        axz.axis("off"); axz.text(0.5, 0.5, f"{lab}: n/a", ha="center", va="center", fontsize=8); continue
                    xc_ = float(min(max(xc_, 0), W)); yc = float(np.asarray(r_.model.predict(np.array([[xc_]]))).ravel()[0])
                    half = 1024
                    c0 = int(min(max(0, xc_ - half), max(0, W - 2 * half))); rr0 = int(min(max(0, yc - half), max(0, H - 2 * half)))
                    win = Window(c0, rr0, min(2 * half, W - c0), min(2 * half, H - rr0))
                    tile = src.read(1, window=win).astype(np.float32); nod = tile <= 0; v = tile[~nod]
                    vmin, vmax = (np.percentile(v, [2, 98]) if v.size else (0, 255))
                    axz.imshow(np.ma.masked_where(nod, tile), cmap=cmap, extent=[c0, c0 + win.width, rr0 + win.height, rr0], vmin=vmin, vmax=vmax, interpolation="nearest")
                    axz.axvline(xc_, color="cyan", lw=1.0, ls="--"); axz.axhline(yc, color="lime", lw=0.9)
                    axz.axhline(yc - off if side == "top" else yc + off, color="red", lw=0.9, ls=":")
                    axz.set_title(f"{lab} corner NATIVE (x {xc_/1e3:.1f}k, y {yc:.0f})", fontsize=8); axz.tick_params(labelsize=7)
            # ---- 6: final product ----
            axf = fig.add_subplot(gs[7, :])
            if product is not None and Path(product).is_file():
                with rasterio.open(product) as pr:
                    PW, PH = pr.width, pr.height
                    pv = _read_decimated(pr, Window(0, 0, PW, PH), 2400, max(60, int(2400 * PH / PW)))
                nod = pv <= 0; v = pv[~nod]; vmin, vmax = (np.percentile(v, [2, 98]) if v.size else (0, 255))
                axf.imshow(np.ma.masked_where(nod, pv), cmap=cmap, aspect="auto", extent=[0, PW, PH, 0], vmin=vmin, vmax=vmax, interpolation="nearest")
                axf.set_title(f"FINAL restituted product {PW}x{PH} (7.000 um canvas)", fontsize=10); axf.tick_params(labelsize=7)
            else:
                axf.axis("off"); axf.text(0.5, 0.5, "restituted product not written yet (gate / dry run)", ha="center", va="center", fontsize=10, color="0.4")
            # ---- 7: text ----
            if is_col:
                try:
                    x = np.linspace(left, right, 100).reshape(-1, 1)
                    sep = float(np.median(sel.bottom_.model.predict(x) - sel.top_.model.predict(x)))
                    exp = sel._expected_separation_px() if hasattr(sel, "_expected_separation_px") else float(sel.collimation_line_dist)
                    lines.append(f"line separation {sep:.1f} px vs expected {exp:.1f} px ({100 * (sep / exp - 1):+.2f} %)")
                except Exception as e:  # noqa: BLE001
                    lines.append(f"line separation: n/a ({e})")
                lines.append(f"inlier ratios top {sel.top_.inlier_ratio:.2f} / bottom {sel.bottom_.inlier_ratio:.2f}  (floor {sel.min_inliers_threshold}, paired floor {getattr(sel, 'min_inliers_threshold_paired', 'n/a')})")
                pr_ = getattr(sel, "_pairing_", None) or {}
                if pr_:
                    lines.append(f"pair-constrained detection: {pr_.get('paired_cols')}/{pr_.get('cols')} columns paired (tol {pr_.get('tol_px', 0):.0f} px); re-picked {pr_.get('repicked_cols', 0)}")
                lines.append(f"joint parallel refit: {'ok' if getattr(sel, '_joint_refit_ok', False) else 'NOT applied'}; separation check: {'ok' if getattr(sel, '_separation_ok', True) else 'FAILED'}; is_failed: {sel.is_failed}")
                if getattr(sel, "scan_pitch_um", None):
                    lines.append(f"scan pitch (x, y) um: {sel.scan_pitch_um[0]:.4f}, {sel.scan_pitch_um[1]:.4f}")
            if left is not None:
                vd = poly.vertical_detector if poly is not None else None
                lines.append(f"vertical exposure edges: left {int(left)} right {int(right)} (width {int(right - left)} px source; from {getattr(vd, 'edge_source_', 'n/a')}, strengths {getattr(vd.left_, 'gradient_ratio', 0):.1f}/{getattr(vd.right_, 'gradient_ratio', 0):.1f} DN)" if vd is not None else f"vertical edges: left {int(left)} right {int(right)}")
                lines.append(f"  coherent DN steps found: {getattr(vd, 'n_candidates_', 'n/a')}; strength-weighted centre {getattr(vd, 'crop_center_', float('nan')):.0f} px source")
                for _side, _v in (getattr(sel, "line_extent_", {}) or {}).items():
                    lines.append(f"  {_side} collimation line x-extent {int(_v[0])}..{int(_v[1])} (width {int(_v[1] - _v[0])} px, presence {_v[2]:.2f})")
                lines.append(f"crop x placed by: {getattr(sel, 'crop_x_source_', 'detector')}")
            try:
                tr = sel.transformation_
                lines.append(f"delivered crop offset (canvas px): {tuple(int(v) for v in getattr(tr, 'crop_offset', (0, 0)))}")
            except Exception as e:  # noqa: BLE001
                lines.append(f"delivered crop offset: n/a ({str(e)[:60]})")
            if poly is not None and getattr(poly, "is_fitted", False) and not poly.is_failed:
                lines.append(f"poly exposure edges: ok (inliers {poly.top_.inlier_ratio:.2f}/{poly.bottom_.inlier_ratio:.2f})")
            elif poly is not None:
                lines.append("poly exposure edges: FAILED / not fitted")
    except Exception as e:  # noqa: BLE001
        lines.append(f"sheet error: {e}")
    if extra_text:
        lines.append(extra_text)
    axt = fig.add_subplot(gs[8, :]); axt.axis("off")
    axt.text(0.01, 0.95, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    return fig


def plot_collimation_distortions(detector: CollimationStrategy) -> Figure:
    """Residual distortion curves for top and bottom collimation line fits
    (same fixed-axis + difference-panel layout as the poly figure -- the
    figure-feedback-propagates rule)."""
    return _plot_distortion_pair(detector.top_, detector.bottom_)


# --- Fiducial ---


_PATTERN_COLORS: dict[str, str] = {
    "regulare_sparse": "red",
    "regulare_mid": "orange",
    "regular_dense": "gold",
    "segmented_mid": "limegreen",
    "segmented_dense": "cyan",
    "serialized_time_word": "violet",
}


def _coord_index(centers_xy: np.ndarray) -> dict[tuple[float, float], int]:
    """Build a (cx, cy) → row-index lookup for fast pattern membership queries."""
    return {(float(cx), float(cy)): i for i, (cx, cy) in enumerate(centers_xy)}


def plot_fiducial_filtering(detector: FiducialStrategy) -> Figure:
    """Pattern detection diagnostics: spatial scatter and feature space for top and bottom sides.

    Each row corresponds to one side (top / bottom). The left column shows detections in
    global image space (cx vs cy) with the fitted polynomial edge overlaid. The right column
    shows the raw feature space (matching score vs residual to the edge model).

    Valid patterns are highlighted with distinct colours; unmatched detections appear in light gray.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"Fiducial pattern detection ({detector.raster_filepath_.stem})", fontsize=12, fontweight="bold")

    sides = ("top", "bottom")
    results = (detector.top_, detector.bottom_)
    edge_models = [detector.poly_strategy.top_.model, detector.poly_strategy.bottom_.model]
    cmap = plt.get_cmap("tab10")
    _noise = (0.85, 0.85, 0.85, 1.0)

    for row, (side, result, edge_model) in enumerate(zip(sides, results, edge_models)):
        ax_spatial, ax_feat = axes[row]

        centers_xy = result.centers_xy
        features = result.features
        coord_idx = _coord_index(centers_xy)

        ax_spatial.scatter(centers_xy[:, 0], centers_xy[:, 1], c=[_noise], s=10, linewidths=0)
        ax_feat.scatter(features[:, 0], features[:, 1], c=[_noise], s=10, linewidths=0)

        legend_handles: list[Line2D] = []

        for i, (name, pattern) in enumerate(result.patterns.items()):
            if pattern.count == 0:
                continue
            color = cmap(i % 10)
            score = pattern.score
            star = " ★" if score > detector.min_score_threshold else ""

            indices = [coord_idx[k] for pt in pattern.points if (k := (float(pt[0]), float(pt[1]))) in coord_idx]
            if indices:
                idx = np.array(indices)
                ax_spatial.scatter(centers_xy[idx, 0], centers_xy[idx, 1], c=[color], s=25, linewidths=0)
                ax_feat.scatter(features[idx, 0], features[idx, 1], c=[color], s=25, linewidths=0)

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=6,
                    label=f"{name}{star}  score={score:.3f}  n={pattern.count}",
                )
            )

        x_grid = np.linspace(0, float(centers_xy[:, 0].max()), 300)
        y_pred = edge_model.predict(x_grid.reshape(-1, 1)).ravel()
        ax_spatial.plot(x_grid, y_pred, color="steelblue", linewidth=1.0, linestyle="--")

        ax_spatial.invert_yaxis()
        ax_spatial.set_title(f"{side} — spatial  ({len(centers_xy)} detections)")
        ax_spatial.set_xlabel("cx (px)")
        ax_spatial.set_ylabel("cy (px)")

        ax_feat.legend(
            handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, borderaxespad=0
        )
        ax_feat.set_title(f"{side} — feature space")
        ax_feat.set_xlabel("score")
        ax_feat.set_ylabel("residual (px)")

    return fig


def plot_fiducial_distortions(detector: FiducialStrategy) -> Figure:
    """Fiducial center y-deviation from mean, per valid pattern, for top and bottom sides.

    In the ideal case all points lie at 0. Divergence reveals scan distortion.
    """
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    fig.suptitle(f"Fiducial distortion — {detector.raster_filepath_.stem}", fontsize=12, fontweight="bold")

    for side, result in zip(["top", "bottom"], [detector.top_, detector.bottom_]):
        for name, pattern in result.patterns.items():
            if pattern.score <= detector.min_score_threshold or pattern.count < 8:
                continue
            x = pattern.points[:, 0].astype(np.float64)
            y = pattern.points[:, 1].astype(np.float64)
            ax.scatter(
                x,
                y - y.mean(),
                s=8,
                marker="x" if side == "bottom" else "o",
                label=f"{side} · {name}  (n={len(x)})",
            )

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("distortion (px)")

    return fig


def plot_fiducial_detected_profiles(detector: FiducialStrategy, window_height_fraction: float = 0.08) -> Figure:
    """Detected fiducial centers overlaid on the top and bottom image strips, one scatter per valid pattern."""
    sides_results = [detector.top_, detector.bottom_]
    n_insets = max(
        max(sum(1 for p in r.patterns.values() if p.score > detector.min_score_threshold) for r in sides_results),
        1,
    )

    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    fig.suptitle(f"Fiducial detected profiles — {detector.raster_filepath_.stem}", fontsize=18, fontweight="bold")
    gs = fig.add_gridspec(2, 1 + n_insets, width_ratios=[14] + [1] * n_insets)
    main_axes = [fig.add_subplot(gs[row, 0]) for row in range(2)]
    inset_slots = [[fig.add_subplot(gs[row, col + 1]) for col in range(n_insets)] for row in range(2)]

    with rasterio.open(detector.raster_filepath_) as src:
        edge_models = [detector.poly_strategy.top_.model, detector.poly_strategy.bottom_.model]
        # Display windows ANCHORED on the content-edge model so the rail,
        # the fiducial marks, the collimation line, and the start of image
        # content are all visible for every column despite frame tilt
        # (David 2026-08-22/23: fixed-fraction windows cut all of these
        # off). Rail side gets the larger pad; falls back to the legacy
        # fixed fraction if a model is unusable.
        PAD_RAIL, PAD_CONTENT = 4500, 2500
        windows = []
        x_probe = np.linspace(0, src.width, 200).reshape(-1, 1)
        for side_i, model in enumerate(edge_models):
            try:
                yy = model.predict(x_probe).ravel()
                lo, hi = float(np.min(yy)), float(np.max(yy))
                if side_i == 0:      # top: rail/marks above the edge
                    y0, y1 = lo - PAD_RAIL, hi + PAD_CONTENT
                else:                # bottom: rail/marks below the edge
                    y0, y1 = lo - PAD_CONTENT, hi + PAD_RAIL
            except Exception:
                wh = int(src.height * window_height_fraction)
                y0, y1 = (0, wh) if side_i == 0 else (src.height - wh, src.height)
            y0 = max(0, int(y0))
            y1 = min(src.height, int(y1))
            if y1 - y0 < 100:        # degenerate model -> legacy window
                wh = int(src.height * window_height_fraction)
                y0, y1 = (0, wh) if side_i == 0 else (src.height - wh, src.height)
            windows.append(Window(0, y0, src.width, y1 - y0))

        def _spacing_info(cx: np.ndarray) -> str:
            if len(cx) >= 2:
                dists = np.diff(np.sort(cx.astype(np.float64)))
                return f"spacing  mean={float(dists.mean()):.1f}px  std={float(dists.std()):.1f}px"
            return "spacing  n/a"

        for row, (ax, side, window, result, edge_model) in enumerate(
            zip(main_axes, ["top", "bottom"], windows, sides_results, edge_models)
        ):
            sub_img = SubImage(src, window, (1, 1024, 4096))
            ax.imshow(sub_img.band, cmap="gray", aspect="auto")

            ax_handles: list[Line2D] = []
            inset_data: list[tuple[str, np.ndarray]] = []

            for name, pattern in result.patterns.items():
                if pattern.score <= detector.min_score_threshold:
                    continue
                color = _PATTERN_COLORS.get(name, "white")
                score = pattern.score
                centers = pattern.points.astype(np.float64)

                if len(centers) > 0:
                    centers_local = sub_img.to_local(centers)
                    ax.scatter(centers_local[:, 0], centers_local[:, 1], c=color, s=20, zorder=3)
                    ax_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=color,
                            markersize=7,
                            label=f"{side} {name}  |  score={score:.3f}  |  fiducials={pattern.count}  |  {_spacing_info(centers[:, 0])}",
                        )
                    )
                    mp = mean_patch_from_centers(src, centers)
                    if mp is not None:
                        inset_data.append((color, mp))

            x_edge = np.linspace(0, src.width, 500)
            edge_local = sub_img.to_local(np.column_stack([x_edge, edge_model.predict(x_edge.reshape(-1, 1)).ravel()]))
            (edge_line,) = ax.plot(
                edge_local[:, 0], edge_local[:, 1], color="steelblue",
                linewidth=1.0, linestyle="--",
                label="poly strip-placement edge model (rail-window anchor)")
            ax_handles.append(edge_line)

            ax.legend(handles=ax_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=15, frameon=True)
            ax.axis("off")

            for col, inset_ax in enumerate(inset_slots[row]):
                if col < len(inset_data):
                    color, patch = inset_data[col]
                    inset_ax.imshow(patch, cmap="gray")
                    for spine in inset_ax.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(2)
                    inset_ax.set_xticks([])
                    inset_ax.set_yticks([])
                else:
                    inset_ax.set_visible(False)

    return fig


def plot_fiducial_detected_boxes(detector: FiducialStrategy) -> tuple[Figure, Figure]:
    """One figure per side showing every detected fiducial box as a cropped patch.

    Boxes are colour-coded by pattern; unmatched detections appear in gray.
    """
    figures: list[Figure] = []
    cmap = plt.get_cmap("tab10")

    for side, side_result in zip(("top", "bottom"), (detector.top_, detector.bottom_)):
        boxes = side_result.boxes
        scores = side_result.scores
        centers_xy = side_result.centers_xy
        n = len(boxes)

        coord_to_pattern: dict[tuple[float, float], tuple[str, Any]] = {}
        for i, (name, pattern) in enumerate(side_result.patterns.items()):
            color = cmap(i % 10)
            for pt in pattern.points:
                coord_to_pattern[(float(pt[0]), float(pt[1]))] = (name, color)

        _noise_color = (0.85, 0.85, 0.85, 1.0)

        grid = max(1, int(np.ceil(np.sqrt(n))))
        fig, axes_2d = plt.subplots(grid, grid, figsize=(grid * 2, grid * 2), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Detected fiducial boxes — {side}  ({n} boxes)", fontsize=11, fontweight="bold")
        axes = axes_2d.flatten()

        with rasterio.open(detector.raster_filepath_) as src:
            for ax, box, score, (cx, cy) in zip(axes, boxes, scores, centers_xy):
                x, y, w, h = box
                band = src.read(1, window=Window(x, y, w, h))
                ax.imshow(band, cmap="gray", interpolation="nearest")

                match = coord_to_pattern.get((float(cx), float(cy)))
                if match is not None:
                    pattern_name, color = match
                    label_str = pattern_name
                else:
                    color = _noise_color
                    label_str = "unmatched"

                ax.set_title(f"{label_str}  {score:.3f}", fontsize=7, color=color)
                ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        figures.append(fig)

    return figures[0], figures[1]


def mean_patch_from_centers(
    src: str | Path | rasterio.DatasetReader,
    centers: np.ndarray,
    half_size: int = 50,
) -> np.ndarray | None:
    """Compute the mean image patch (band 1) around a set of pixel centers.

    Uses an incremental float64 accumulator so peak memory is O(patch_size²)
    regardless of the number of centers. Out-of-bounds regions are zero-padded
    before averaging. Centers that fall entirely outside the raster are silently skipped.
    """
    if not isinstance(src, rasterio.DatasetReader):
        with rasterio.open(src) as opened:
            return mean_patch_from_centers(opened, centers, half_size)

    size = 2 * half_size
    accumulator = np.zeros((size, size), dtype=np.float64)
    count = 0

    x0s: np.ndarray = centers[:, 0].astype(np.intp) - half_size
    y0s: np.ndarray = centers[:, 1].astype(np.intp) - half_size

    for x0, y0 in zip(x0s, y0s):
        x0c = max(0, int(x0))
        y0c = max(0, int(y0))
        x1c = min(src.width, int(x0) + size)
        y1c = min(src.height, int(y0) + size)
        if x1c <= x0c or y1c <= y0c:
            continue
        patch = src.read(1, window=Window(x0c, y0c, x1c - x0c, y1c - y0c))
        accumulator[y0c - y0 : y0c - y0 + patch.shape[0], x0c - x0 : x0c - x0 + patch.shape[1]] += patch
        count += 1

    return (accumulator / count).astype(np.float32) if count > 0 else None


# --- Transform ---


def plot_deformation_grid(
    transform: Transformation,
    num: int = 20,
    figsize: tuple[int, int] = (6, 6),
) -> Figure:
    """Visualize the deformation field by plotting warped grid lines."""
    with rasterio.open(transform.raster_filepath) as src:
        w, h = src.width, src.height

    xs = np.linspace(0, w - 1, num, dtype=np.float32)
    ys = np.linspace(0, h - 1, num, dtype=np.float32)

    fig, ax = plt.subplots(figsize=figsize)

    for y in ys:
        line = np.stack([xs, np.full_like(xs, y)], axis=-1)
        warped_line = transform.deformation(line)
        ax.plot(warped_line[:, 0], warped_line[:, 1], color="gray", lw=0.8, alpha=0.7)

    for x in xs:
        line = np.stack([np.full_like(ys, x), ys], axis=-1)
        warped_line = transform.deformation(line)
        ax.plot(warped_line[:, 0], warped_line[:, 1], color="gray", lw=0.8, alpha=0.7)

    ax.set_title("Warped deformation grid")
    ax.invert_yaxis()

    return fig


def plot_crop_area(transform: Transformation, figsize: tuple[int, int] = (6, 6)) -> Figure:
    """Visualize the crop region within the original image frame."""
    fig, ax = plt.subplots(figsize=figsize)

    with rasterio.open(transform.raster_filepath) as src:
        w, h = src.width, src.height

    crop_x, crop_y = transform.crop_offset
    crop_w, crop_h = transform.output_size

    ax.add_patch(patches.Rectangle((0, 0), w, h, fill=False, edgecolor="black", linewidth=2, label="Original image"))
    ax.add_patch(
        patches.Rectangle((crop_x, crop_y), crop_w, crop_h, fill=True, alpha=0.3, color="orange", label="Crop region")
    )
    ax.scatter(crop_x, crop_y, color="red", marker="+", s=10, label="Crop origin (0,0 in crop space)")

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("auto")
    ax.set_box_aspect(h / (w / 2))
    ax.invert_yaxis()
    ax.set_title(f"Crop visualization\ncrop_offset = ({crop_x}, {crop_y}), size = ({crop_w}, {crop_h})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    return fig


# --- Dispatch ---


def _safe_fig(fn, *args, **kw):
    """audit M-4: an exception inside a figure must not end the generator (which
    silently dropped every later figure of the frame) -- return None, log, continue."""
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        logger.warning("QC figure %s failed: %s", getattr(fn, "__name__", fn), e)
        return None


def save_figures(fitting_class: FittingClass, output_dir: str | Path) -> None:
    """Save all QC figures for a fitted strategy to ``output_dir/<figure_name>/<stem>.png``.

    Each figure type gets its own sub-directory named after the plot (e.g.
    ``poly_edges/``, ``deformation_grid/``). Errors on individual figures are
    logged as warnings so a single bad plot never aborts the whole QC run.
    """
    output_dir = Path(output_dir)
    gen = get_figures(fitting_class)
    while True:
        try:
            name, fig = next(gen)
            if fig is None:
                continue
            (output_dir / name).mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / name / f"{fitting_class.raster_filepath_.stem}.png")
            plt.close(fig)
        except StopIteration:
            break
        except Exception as e:
            logger.warning("Skipping QC figure: %s", e)


def get_figures(
    fitting_class: FittingClass,
    plot_transformation: bool = True,
    poly_crop_is_delivered: bool = True,
) -> Iterator[tuple[str, Figure]]:
    """Yield (name, figure) pairs for all QC plots of a fitted FittingClass instance.

    ``poly_crop_is_delivered`` is threaded to ``plot_poly_edges`` so the red
    crop row is labeled honestly: it is the delivered cut only for the pure
    Poly/Mixed fallback, not under CollimationStrategy (fixed line-offset crop,
    David 2026-07-22)."""
    if isinstance(fitting_class, VerticalDetector):
        yield "vertical_edges", plot_vertical_edges(fitting_class)
        yield "vertical_ruptures", plot_vertical_ruptures(fitting_class)
        return
    if isinstance(fitting_class, (FlatStrategy, PolyStrategy, FiducialStrategy)) and not isinstance(fitting_class, CollimationStrategy):
        yield "sheet", _safe_fig(plot_restitution_sheet, fitting_class)   # audit M-5: every strategy gets the one-page sheet
    if isinstance(fitting_class, FlatStrategy):
        yield from get_figures(fitting_class.vertical_detector, plot_transformation=False)
        yield "flat_edges", plot_flat_edges(fitting_class)
        yield "flat_ruptures", plot_flat_ruptures(fitting_class)
        if plot_transformation:
            yield "crop_area", plot_crop_area(fitting_class.transformation_)
        return
    if isinstance(fitting_class, PolyStrategy):
        yield from get_figures(fitting_class.vertical_detector, plot_transformation=False)
        yield "poly_edges", plot_poly_edges(fitting_class, crop_is_delivered=poly_crop_is_delivered)
        yield "poly_distortions", plot_poly_distortions(fitting_class)
        if plot_transformation:
            yield "deformation_grid", plot_deformation_grid(fitting_class.transformation_)
            yield "crop_area", plot_crop_area(fitting_class.transformation_)
        return
    if isinstance(fitting_class, CollimationStrategy):
        yield from get_figures(fitting_class.poly_strategy, plot_transformation=False,
                               poly_crop_is_delivered=False)
        yield "sheet", _safe_fig(plot_restitution_sheet, fitting_class)
        yield "collimation_edges", _safe_fig(plot_collimation_edges, fitting_class)
        yield "collimation_distortions", _safe_fig(plot_collimation_distortions, fitting_class)
        if plot_transformation:
            yield "deformation_grid", plot_deformation_grid(fitting_class.transformation_)
            yield "crop_area", plot_crop_area(fitting_class.transformation_)
        return
    if isinstance(fitting_class, FiducialStrategy):
        yield from get_figures(fitting_class.poly_strategy, plot_transformation=False)
        yield "fiducial_filtering", plot_fiducial_filtering(fitting_class)
        yield "fiducial_distortions", plot_fiducial_distortions(fitting_class)
        yield "fiducial_detected_profiles", plot_fiducial_detected_profiles(fitting_class)
        # yield from zip(("fiducial_boxes_top", "fiducial_boxes_bottom"), plot_fiducial_detected_boxes(fitting_class))
        if plot_transformation:
            yield "deformation_grid", plot_deformation_grid(fitting_class.transformation_)
            yield "crop_area", plot_crop_area(fitting_class.transformation_)
        return
    if isinstance(fitting_class, MixedStrategy):
        yield from get_figures(fitting_class.selected_strategy_, plot_transformation=plot_transformation)
