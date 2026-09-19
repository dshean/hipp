"""
Copyright (c) 2026 HIPP developers
Description: PROTOTYPE (kh9pc, 2026-08-26) -- detection of the periodic mark
    trains on the two film-edge margins ("rails") of a KH-9 PC mosaic and the
    fit of each train (period, phase, count, gaps) as an independent measurement
    of the sweep geometry.

    Anchoring (dshean 2026-08-26): the marks sit at a FIXED offset from the
    collimation line, inside the film edge, so the search band is placed from
    the fitted collimation-line models (CollimationStrategy results or a
    restitution joblib) and only that narrow band is read (window reads, block
    by block along x -- never a whole-frame or decimated-frame read).

    What is on the rails (NRO TCS-20055/69 "Recorded Data" + the 2026-08-26
    zooms of F013 / cg A010 / iceland A025 / ops196 A004 / ops327 A004):
      * a DENSE train, one mark every ~9.1 mm (~1300 px at 7 um): regular on
        the time-track edge (500-cycle pulse), gappy = serialized TIME WORD on
        the titling edge, starting at the centre of format (first mark seen
        400-490 px past the detector's sweep center on F013 and A010);
      * a MID (1.048 in, 3803 px at 7 um; missions >= 1214) or SPARSE (5.24 in,
        19014 px; missions <= 1213) scan-angle train in a second row;
      * glyphs: plain disks ~0.42 mm (D3C1210), wagon wheels ~0.40 mm
        (D3C1214/1216/1217; the 1214 ring is open on one side);
      * the row offsets from the collimation line are 4.2-5.9 mm on every
        rail measured; the band searched here is 3.2-6.5 mm outward.

    Outputs per frame: <entity>_timing_marks.json / .csv (every mark: x, y,
    side, row, type, score, train index, residual) and a QC figure.
    Nothing here writes into a data product; the CLI writes only to --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

# rasterio / GDAL are imported lazily inside the functions that read the mosaic
# so the detection core (cv2 + numpy) and the train fit stay importable and
# testable in a ~100 MB process (the pfe front ends cap a user at 1.8 GiB).

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Physical constants (mm on film; px = mm * 1000 / pitch_um)
# ----------------------------------------------------------------------------
CANVAS_PITCH_UM = 7.0
FOCAL_MM = 1524.0                 # 60.00 in nominal (measured 59.984 F / 59.975 A, ba2)
BAND_INNER_MM = 3.2               # search band, outward from the collimation line
BAND_OUTER_MM = 6.5               # stays inside the film-edge glow (>= 7.1 mm on F013/A025)
BAND_TILT_PAD_PX = 64             # line tilt inside one block (A010: 0.0027 px/px -> 44 px / 16 k)
# nominal periods (mm) used only as PRIORS / labels; the fit is data-driven
PERIOD_CLASSES_MM = {"dense": 9.08, "mid": 26.62, "sparse": 133.10}
PERIOD_CLASS_TOL = 0.20           # +-20 % around a class for the label (hipp DENSE_MAX 0.41 in = 10.4 mm)
# glyph sizes (mm) for the synthetic template bank
DISK_DIAM_MM = (0.22, 0.28, 0.34, 0.42, 0.50)   # 2026-09-19: +0.22/0.28 -- mission 1205 bottom-rail scan-angle marks are small dots
WHEEL_DIAM_MM = (0.36, 0.41, 0.46)
WHEEL_STROKE_MM = 0.055
TEMPLATE_PAD_PX = 10    # MEASURED 2026-09-19, ops251, 14 frames, pad 10 vs 40 (a dot with ~0.8 diameters of
                        # black around it, dshean's "larger template"): raw hits halve (~1000 -> ~550) but
                        # refine_marks() was already rejecting those, so the sparse trains are identical
                        # (top 156 vs 155 marks block-wide, bottom 132 vs 124) while the padded template LOSES
                        # dense time-track marks on four frames (A052 248 -> 171, A053 244 -> 160, F053 173 ->
                        # 143, F057 195 -> 152) -- the wider window is more sensitive to the film-edge glow
                        # gradient under the bottom rail. Isolation is enforced AFTER matching, at native
                        # resolution, by refine_marks(); the template stays tight.
TEMPLATE_BLUR_PX = 1.5


# ----------------------------------------------------------------------------
# Anchors: line models + edges + pitch
# ----------------------------------------------------------------------------
@dataclass
class RailAnchors:
    """Everything needed to place the rail bands: per-side line model y(x) in
    GLOBAL raster px, exposure edges (left, right), scan pitch (x_um, y_um)."""

    top: Callable[[NDArray], NDArray]
    bottom: Callable[[NDArray], NDArray]
    edges: tuple[int, int]
    pitch_um: tuple[float, float]
    source: str
    constant_rows: bool = False   # True -> unknown tilt: widen the band + local line probe

    def line(self, side: str) -> Callable[[NDArray], NDArray]:
        return self.top if side == "top" else self.bottom


def _const(row: float) -> Callable[[NDArray], NDArray]:
    return lambda x: np.full(np.shape(x), float(row), dtype=float)


def anchors_from_joblib(path: str | Path) -> tuple[RailAnchors, Path | None]:
    """Line models / edges / pitch from a fitted restitution joblib (MixedStrategy,
    CollimationStrategy or a bare PolyStrategy). Returns (anchors, raster path).

    The per-side models taken here are ``_results[side].model`` -- the RANSAC RUPTURE
    models. On a no-collimation-line block those are also the delivered geometry
    (POLY_EDGE_CLASS_POLICY=rupture, 2026-09-18), so the rails are placed off the same
    feature the product is cut on.""",
    import joblib

    s = joblib.load(path)
    sel = s
    if hasattr(s, "strategies"):
        try:
            sel = s.selected_strategy_
        except Exception:
            sel = s
    res = getattr(sel, "_results", None)
    if not res or "top" not in res or "bottom" not in res or not hasattr(res["top"], "model"):
        raise ValueError(f"{path}: no top/bottom line models ({type(sel).__name__}) -- need a Collimation or Poly fit")
    top_m, bot_m = res["top"].model, res["bottom"].model

    def _mk(m):
        return lambda x: np.asarray(m.predict(np.asarray(x, dtype=float).reshape(-1, 1)), dtype=float).ravel()

    # 2026-09-18: the prototype assumed a Mixed/Collimation wrapper that carries a
    # .poly_strategy. A block with no collimation lines (nepal ops251, mission 1205) selects a
    # bare PolyStrategy, which IS the poly strategy and has the detector directly.
    _poly = getattr(sel, "poly_strategy", None) or sel
    vd = getattr(_poly, "vertical_detector", None)
    if vd is None:
        raise ValueError(f"{path}: {type(sel).__name__} carries no vertical_detector -- cannot place the rails")
    edges = tuple(int(v) for v in vd.edges_)
    pitch = getattr(sel, "scan_pitch_um", None)
    if pitch is None:
        pitch = (CANVAS_PITCH_UM, CANVAS_PITCH_UM)
        logger.warning("joblib carries no scan_pitch_um -> assuming %.3f um", CANVAS_PITCH_UM)
    raster = None
    try:
        raster = Path(sel.raster_filepath_)
    except Exception:
        pass
    return RailAnchors(_mk(top_m), _mk(bot_m), edges, (float(pitch[0]), float(pitch[1])),
                       f"joblib:{Path(path).name} ({type(sel).__name__}, edge_source={getattr(vd, 'edge_source_', '?')})"), raster


def anchors_from_qc_json(path: str | Path) -> RailAnchors:
    """Median line rows from the worker's restit_qc json. The Collimation crop is
    centred on the line pair: crop_offset[1] = median(top line row) - (eh - 21770)/2
    with eh = 21771 -> top = crop_offset_y + 0.5; bottom = top + line_sep_px.
    No tilt information -> constant rows, the band is widened and each block
    probes the line locally."""
    j = json.loads(Path(path).read_text())
    if j.get("strategy") != "CollimationStrategy":
        raise ValueError(f"{path}: strategy {j.get('strategy')} has no line pair")
    top = float(j["crop_offset"][1]) + 0.5
    bot = top + float(j["line_sep_px"])
    edges = tuple(int(v) for v in j["exposure_edges_src"])
    pitch = j.get("scan_pitch_um") or (CANVAS_PITCH_UM, CANVAS_PITCH_UM)
    return RailAnchors(_const(top), _const(bot), edges, (float(pitch[0]), float(pitch[1])),
                       f"qc-json:{Path(path).name} (median rows, tilt unknown)", constant_rows=True)


# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------
def _render(mask_fn: Callable[[NDArray, NDArray], NDArray], size_px: float, ss: int = 4) -> NDArray[np.uint8]:
    n = int(round(size_px)) + 2 * TEMPLATE_PAD_PX
    yy, xx = np.mgrid[0:n * ss, 0:n * ss]
    c = (n * ss - 1) / 2.0
    img = mask_fn((xx - c) / ss, (yy - c) / ss).astype(np.float32)
    img = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), TEMPLATE_BLUR_PX)
    img = img / max(float(img.max()), 1e-6)
    return (img * 255).astype(np.uint8)


def make_disk_template(diam_px: float) -> NDArray[np.uint8]:
    r = diam_px / 2.0
    return _render(lambda x, y: np.hypot(x, y) <= r, diam_px)


def make_wheel_template(diam_px: float, stroke_px: float) -> NDArray[np.uint8]:
    r = diam_px / 2.0

    def m(x, y):
        rr = np.hypot(x, y)
        ring = (rr <= r) & (rr >= r - stroke_px)
        cross = (rr <= r) & ((np.abs(x) <= stroke_px / 2) | (np.abs(y) <= stroke_px / 2))
        return ring | cross

    return _render(m, diam_px)


@dataclass
class Template:
    kind: str          # "disk" | "wheel"
    size_mm: float
    image: NDArray[np.uint8]


def template_bank(pitch_x_um: float, kinds: tuple[str, ...] = ("disk", "wheel")) -> list[Template]:
    px = lambda mm: mm * 1000.0 / pitch_x_um
    bank: list[Template] = []
    if "disk" in kinds:
        bank += [Template("disk", d, make_disk_template(px(d))) for d in DISK_DIAM_MM]
    if "wheel" in kinds:
        bank += [Template("wheel", d, make_wheel_template(px(d), px(WHEEL_STROKE_MM))) for d in WHEEL_DIAM_MM]
    return bank


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
@dataclass
class Mark:
    x: float
    y: float
    side: str
    score: float
    kind: str
    size_mm: float
    dy: float                 # y - line(x), signed (negative above the top line)
    row: int = -1             # row cluster id per side (-1 = unassigned)
    train: str = ""           # class label of the row's train ("dense"/"mid"/"sparse"/"other")
    k: int = -999999          # index on the fitted grid
    resid: float = float("nan")
    inlier: bool = False


def _match_block(block: NDArray[np.uint8], bank: list[Template], score_min: float, nms_px: int
                 ) -> list[tuple[float, float, float, int]]:
    """(x_local, y_local, score, template_id) of the local maxima of the max-over-
    templates normalised cross-correlation."""
    H, W = block.shape
    resp = np.full((H, W), -1.0, dtype=np.float32)
    tid_map = np.full((H, W), -1, dtype=np.int16)
    for tid, t in enumerate(bank):
        th, tw = t.image.shape
        if th >= H or tw >= W:
            continue
        r = cv2.matchTemplate(block, t.image, cv2.TM_CCOEFF_NORMED)
        oy, ox = th // 2, tw // 2
        v = resp[oy:oy + r.shape[0], ox:ox + r.shape[1]]
        idv = tid_map[oy:oy + r.shape[0], ox:ox + r.shape[1]]
        better = r > v
        v[better] = r[better]
        idv[better] = tid
    k = max(3, int(nms_px) | 1)
    dil = cv2.dilate(resp, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero((resp >= score_min) & (resp == dil))
    out = []
    for y, x in zip(ys, xs):
        xr = float(x)
        if 0 < x < W - 1:      # parabolic sub-pixel refinement in x
            a, b, c = float(resp[y, x - 1]), float(resp[y, x]), float(resp[y, x + 1])
            den = a - 2 * b + c
            if den < 0:
                xr += 0.5 * (a - c) / den
        out.append((xr, float(y), float(resp[y, x]), int(tid_map[y, x])))
    return out


def _probe_line_row(src: "rasterio.DatasetReader", x0: int, w: int, y_guess: float, half: int) -> float | None:
    from rasterio.windows import Window
    """Local collimation-line row inside one block (constant-row anchors only):
    brightest row of the x-decimated row-median profile within +-half px."""
    y_a = int(max(0, y_guess - half))
    y_b = int(min(src.height, y_guess + half))
    if y_b - y_a < 50:
        return None
    dec = max(1, w // 2048)
    a = src.read(1, window=Window(x0, y_a, w, y_b - y_a), out_shape=(y_b - y_a, max(1, w // dec)))
    prof = np.median(a.astype(np.float32), axis=1)
    prof = np.convolve(prof, np.ones(5) / 5.0, mode="same")
    return float(y_a + int(np.argmax(prof)))


def detect_marks(
    mosaic: str | Path,
    anchors: RailAnchors,
    sides: tuple[str, ...] = ("top", "bottom"),
    kinds: tuple[str, ...] = ("disk", "wheel"),
    block_w: int = 16384,
    overlap: int = 512,
    score_min: float = 0.45,
    x_range: tuple[int, int] | None = None,
    band_mm: tuple[float, float] = (BAND_INNER_MM, BAND_OUTER_MM),
    keep_overview: int = 2400,
) -> tuple[list[Mark], dict]:
    """Slide blocks along x, read only the rail band placed from the line model,
    match the template bank, return marks in global px plus per-side overview
    strips (x max-pooled to <= keep_overview columns, for the QC figure)."""
    px_x = 1000.0 / anchors.pitch_um[0]
    px_y = 1000.0 / anchors.pitch_um[1]
    bank = template_bank(anchors.pitch_um[0], kinds)
    nms_px = int(0.6 * min(t.image.shape[0] for t in bank))
    inner, outer = band_mm[0] * px_y, band_mm[1] * px_y
    pad = BAND_TILT_PAD_PX + (450 if anchors.constant_rows else 0)
    marks: list[Mark] = []
    info: dict = {"blocks": {}, "overview": {}, "band_px": (inner, outer), "pad_px": pad,
                  "templates": [(t.kind, t.size_mm, t.image.shape) for t in bank]}

    import rasterio
    from rasterio.windows import Window

    with rasterio.open(mosaic) as src:
        W, H = src.width, src.height
        xa, xb = (0, W) if x_range is None else (max(0, x_range[0]), min(W, x_range[1]))
        for side in sides:
            line = anchors.line(side)
            sign = -1.0 if side == "top" else 1.0
            ov_cols: list[NDArray] = []
            ov_meta: list[tuple[int, int, int]] = []   # (x0, w, y0) per block for the overview
            n_blocks = 0
            t0 = time.time()
            for x0 in range(xa, xb, block_w):
                x1 = min(xb, x0 + block_w)
                rx0 = max(0, x0 - overlap)
                rx1 = min(W, x1 + overlap)
                xc = 0.5 * (rx0 + rx1)
                y_line = float(line(np.array([xc]))[0])
                if anchors.constant_rows:
                    y_probe = _probe_line_row(src, rx0, rx1 - rx0, y_line, 450)
                    if y_probe is not None:
                        y_line = y_probe
                y_lo = y_line + sign * outer - pad if side == "top" else y_line + inner - pad
                y_hi = y_line - inner + pad if side == "top" else y_line + outer + pad
                y_lo, y_hi = int(max(0, math.floor(y_lo))), int(min(H, math.ceil(y_hi)))
                if y_hi - y_lo < 100:
                    info["blocks"][f"{side}:{x0}"] = "band outside raster"
                    continue
                block = src.read(1, window=Window(rx0, y_lo, rx1 - rx0, y_hi - y_lo))
                if block.dtype != np.uint8:
                    block = cv2.normalize(block.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                for xl, yl, sc, tid in _match_block(block, bank, score_min, nms_px):
                    gx, gy = rx0 + xl, y_lo + yl
                    if not (x0 <= gx < x1):        # owned by the neighbouring block's core
                        continue
                    t = bank[tid]
                    marks.append(Mark(gx, gy, side, sc, t.kind, t.size_mm, gy - float(line(np.array([gx]))[0])))
                n_blocks += 1
                # overview: max-pool along x so 50-px marks survive a 140:1 compression
                f = max(1, int(math.ceil((xb - xa) / keep_overview)))
                core = block[:, (x0 - rx0):(x0 - rx0) + (x1 - x0)]
                wc = core.shape[1] // f * f
                if wc > 0:
                    pooled = core[:, :wc].reshape(core.shape[0], wc // f, f).max(axis=2)
                    ov_cols.append(pooled)
                    ov_meta.append((x0, wc, y_lo))
            info["blocks"][side] = {"n_blocks": n_blocks, "seconds": round(time.time() - t0, 1)}
            if ov_cols:
                hmax = max(c.shape[0] for c in ov_cols)
                cols = [np.pad(c, ((0, hmax - c.shape[0]), (0, 0))) for c in ov_cols]
                info["overview"][side] = {"image": np.concatenate(cols, axis=1), "x0": ov_meta[0][0],
                                          "factor": f, "y0_per_block": ov_meta, "height": hmax}
    return marks, info


# ----------------------------------------------------------------------------
# Rows and trains
# ----------------------------------------------------------------------------
@dataclass
class Sections:
    """Scan-section placement from the merge provenance: section i starts at x_start[i]
    (cumulative seam tx) and carries a cumulative y offset cum_ty[i]. The merged mosaic is a
    STAIRCASE of sections (dshean 2026-09-19: "spacing between different mark classes is
    variable due to seam merging"); F056 accumulates 658 px of ty over nine seams in steps
    of up to 115 px, against a 30-px row tolerance. Rows and lattices are therefore fitted in
    section-local coordinates, and the per-section offsets that come back ARE the seam
    errors -- which is what lets the marks check the seams."""
    x_start: list
    cum_ty: list
    source: str = ""

    def index(self, x):
        xs = np.asarray(x, float)
        return np.clip(np.searchsorted(np.asarray(self.x_start), xs, side="right") - 1, 0, len(self.x_start) - 1)

    def ty(self, x):
        return np.asarray(self.cum_ty)[self.index(x)]


def load_sections(mosaic):
    """<stem>_merge_provenance.json beside the mosaic -> Sections, else None (one section)."""
    mosaic = Path(mosaic)
    for c in (mosaic.with_name(mosaic.stem + "_merge_provenance.json"), mosaic.with_name(mosaic.stem + ".json")):
        if c.exists():
            try:
                p = json.loads(c.read_text())
                xs, tys, x, y = [0.0], [0.0], 0.0, 0.0
                for q in p.get("seams", []):
                    x += float(q["tx"]); y += float(q["ty"])
                    xs.append(x); tys.append(y)
                return Sections(xs, tys, str(c.name))
            except Exception as exc:  # noqa: BLE001
                logger.warning("merge provenance %s unreadable (%r) -- one section", c, exc)
                return None
    return None


def refine_marks(mosaic: Path, marks: list[Mark], pitch_um: tuple[float, float],
                 min_mm: float = 0.15, max_mm: float = 0.75, iso_frac: float = 0.35) -> list[Mark]:
    """Re-measure every template hit at NATIVE resolution and keep only isolated dots.

    dshean 2026-09-19: the template hits included fragments of the label text; real
    scan-angle and timing marks are "isolated, single dot with near-black film on all
    sides", and their size differs by rail and by mission (large timing dots, large
    top-rail angle dots, small bottom-rail angle dots on 1205; wagon wheels elsewhere).
    So: centre -> brightness centroid; size -> measured FWHM diameter (stored in
    size_mm, replacing the template's nominal); reject when the annulus 1.4-2.6 R
    around the dot carries anything brighter than bg + iso_frac x (peak - bg), or the
    diameter is outside [min_mm, max_mm]."""
    import rasterio
    from rasterio.windows import Window
    out: list[Mark] = []
    px_mm = 1000.0 / pitch_um[0]
    with rasterio.open(mosaic) as d:
        W, H = d.width, d.height
        for m in marks:
            h = int(max(60, round(max_mm * px_mm * 1.6)))
            x0, y0 = int(round(m.x)) - h, int(round(m.y)) - h
            if x0 < 0 or y0 < 0 or x0 + 2 * h > W or y0 + 2 * h > H:
                continue
            a = d.read(1, window=Window(x0, y0, 2 * h, 2 * h)).astype(np.float32)
            bg = float(np.median(a))
            pk = float(a[h - 4:h + 5, h - 4:h + 5].max())
            if pk - bg < 25:
                continue
            yy, xx = np.mgrid[0:2 * h, 0:2 * h]
            bright = a > bg + 0.5 * (pk - bg)
            # connected blob at the centre only (flood from the peak)
            lbl = np.zeros_like(bright, dtype=np.uint8)
            n, cc = cv2.connectedComponents(bright.astype(np.uint8), lbl, connectivity=8)
            cid = cc[h, h]
            if cid == 0:
                continue
            blob = cc == cid
            area = int(blob.sum())
            diam = 2.0 * math.sqrt(area / math.pi)
            if not (min_mm * px_mm <= diam <= max_mm * px_mm):
                continue
            wgt = np.where(blob, a - bg, 0.0)
            cx = float((wgt * xx).sum() / wgt.sum()); cy = float((wgt * yy).sum() / wgt.sum())
            R = diam / 2.0
            rr = np.hypot(xx - cx, yy - cy)
            ann = a[(rr >= 1.4 * R) & (rr <= 2.6 * R)]
            if ann.size == 0 or ann.max() > bg + iso_frac * (pk - bg):
                continue          # something bright next to it: a glyph, a box, a neighbour
            m.x, m.y = x0 + cx, y0 + cy
            m.size_mm = round(diam / px_mm, 3)
            out.append(m)
    return out


def cluster_rows_lines(marks: list[Mark], side: str, line_fn, min_count: int = 4, tol_px: int = 30,
                       max_rows: int = 6, seed: int = 0, sections=None) -> dict[int, float]:
    """Cluster one side's marks into STRAIGHT rows in mosaic (x, y) -- RANSAC lines.

    Replaces the dy-histogram clustering (constant offset from the restitution line).
    2026-09-19, ops251 A052: the dense train is straight on the film but the rupture
    line diverges from it by 332 px across the frame, so a constant-dy row lost every
    mark near the ends (pink on the rail sheets) and the sparse train never formed.
    A row of marks is a physically straight line relative to the camera; the line
    model is the thing allowed to be wrong. Returns {row_id: median dy} and sets
    m.row; m.dy stays y - line(x) at the mark, for reporting."""
    ms = [m for m in marks if m.side == side]
    for m in ms:
        m.row = -1
    if len(ms) < min_count:
        return {}
    rng = np.random.default_rng(seed)
    xs = np.array([m.x for m in ms]); ys = np.array([m.y for m in ms])
    if sections is not None:
        ys = ys - sections.ty(xs)          # de-staircase: rows are straight in section-local y
    free = np.ones(len(ms), bool)
    rows: dict[int, float] = {}
    rid = 0
    while free.sum() >= min_count and rid < max_rows:
        idx = np.flatnonzero(free)
        best, best_in = None, None
        for _ in range(min(400, len(idx) * 4)):
            i, j = rng.choice(idx, 2, replace=False)
            if abs(xs[i] - xs[j]) < 500:
                continue
            b = (ys[j] - ys[i]) / (xs[j] - xs[i]); a = ys[i] - b * xs[i]
            if abs(b) > 0.05:          # > ~3 deg: not a rail row
                continue
            r = np.abs(ys[idx] - (a + b * xs[idx]))
            inl = idx[r <= tol_px]
            if best is None or inl.size > best_in.size:
                best, best_in = (a, b), inl
        if best is None or best_in.size < min_count:
            break
        # refine with a least-squares line on the inliers, re-select
        A = np.vstack([np.ones(best_in.size), xs[best_in]]).T
        a, b = np.linalg.lstsq(A, ys[best_in], rcond=None)[0]
        r = np.abs(ys[idx] - (a + b * xs[idx]))
        inl = idx[r <= tol_px]
        for k in inl:
            ms[k].row = rid
        free[inl] = False
        rows[rid] = float(np.median([ms[k].dy for k in inl]))
        rid += 1
    return rows


def cluster_rows(marks: list[Mark], side: str, bin_px: int = 8, min_count: int = 3, assign_px: int = 45) -> dict[int, float]:
    """Cluster the signed line offset dy of one side's marks into rows (histogram
    peaks). Returns {row_id: dy_centre}; marks get .row set (-1 = unassigned)."""
    ms = [m for m in marks if m.side == side]
    if not ms:
        return {}
    dy = np.array([m.dy for m in ms])
    lo, hi = math.floor(dy.min()) - bin_px, math.ceil(dy.max()) + bin_px
    edges = np.arange(lo, hi + bin_px, bin_px)
    h, _ = np.histogram(dy, bins=edges)
    hs = np.convolve(h, np.ones(3), mode="same")
    peaks = []
    for i in range(len(hs)):
        if hs[i] >= min_count and hs[i] == hs[max(0, i - 6):i + 7].max():
            sel = (dy >= edges[i] - 1.5 * bin_px) & (dy < edges[i] + 2.5 * bin_px)
            if sel.sum() >= min_count:
                peaks.append(float(np.median(dy[sel])))
    peaks = sorted(set(round(p, 1) for p in peaks))
    # merge peaks closer than assign_px
    merged: list[float] = []
    for p in peaks:
        if merged and abs(p - merged[-1]) < assign_px:
            merged[-1] = 0.5 * (merged[-1] + p)
        else:
            merged.append(p)
    rows = {i: c for i, c in enumerate(merged)}
    for m in ms:
        m.row = -1
        if rows:
            i = min(rows, key=lambda r: abs(m.dy - rows[r]))
            if abs(m.dy - rows[i]) <= assign_px:
                m.row = i
    return rows


@dataclass
class Train:
    side: str
    row: int
    dy_px: float
    dy_mm: float
    label: str
    period_px: float
    period_mm: float
    period_deg: float
    x0: float                    # x of grid index 0 (= the highest-score mark used as reference)
    n_marks: int
    n_inliers: int
    n_outliers: int
    k_min: int
    k_max: int
    n_slots: int                 # k_max - k_min + 1 = expected count between first and last mark
    n_missing: int
    missing_k: list[int]
    resid_rms_px: float
    resid_max_px: float
    x_first: float
    x_last: float
    first_minus_left_edge: float
    right_edge_minus_last: float
    n_beyond_left: int
    n_beyond_right: int
    centre_x: float              # (left + right) / 2 exposure edges
    k_at_centre: float           # fractional grid index of the sweep center
    nearest_mark_dx: float       # x(nearest mark) - centre_x  (px)
    nearest_mark_dx_mm: float
    nearest_mark_dx_deg: float
    coded: bool                  # gappy train starting near the centre = serialized time word
    kind_majority: str
    presence_from_first: str     # '1'/'0' per slot from the first mark (<= 96 slots)
    section_dx: dict = field(default_factory=dict)   # {section: median x offset from the global grid} = seam x errors


def _period_candidates(d: NDArray, pitch_x: float) -> list[float]:
    cands = [PERIOD_CLASSES_MM[c] * 1000.0 / pitch_x for c in PERIOD_CLASSES_MM]
    if d.size:
        dd = d[(d > 20) & np.isfinite(d)]
        if dd.size:
            bins = np.geomspace(20, max(dd.max() * 1.05, 30), num=max(10, int(80 * math.log10(max(dd.max(), 30) / 20))))
            h, e = np.histogram(dd, bins=bins)
            i = int(np.argmax(h))
            near = dd[(dd >= e[i] * 0.97) & (dd <= e[i + 1] * 1.03)]
            if near.size:
                mode = float(np.median(near))
                cands += [mode, mode / 2.0, mode / 3.0]
    return cands


def fit_train(marks: list[Mark], side: str, row: int, anchors: RailAnchors, max_iter: int = 8,
              sections=None) -> Train | None:
    ms = sorted([m for m in marks if m.side == side and m.row == row], key=lambda m: m.x)
    if len(ms) < 6:      # 2026-09-19: 3 marks fit any period; a rail row has >= 6 (sparse: 18 slots per 90 deg)
        return None
    xs = np.array([m.x for m in ms])
    sc = np.array([m.score for m in ms])
    pitch_x = anchors.pitch_um[0]
    d = np.diff(xs)

    def _nominal(P_px: float) -> bool:
        mm = P_px * pitch_x / 1000.0
        return any(abs(mm - c) <= PERIOD_CLASS_TOL * c for c in PERIOD_CLASSES_MM.values())

    best = None
    for P in _period_candidates(d, pitch_x):
        if P <= 20:
            continue
        mult = d / P
        # 2026-09-19: the residual test is ABSOLUTE px, not a fraction that grows with the
        # multiple. With 0.08 x k, a 5-mark scan-angle row (spacing 19 k px) "supported" a
        # 1301-px dense period at k = 14 with a 1.1-period tolerance, and won the ranking on
        # A053's top rail. A row is on lattice P when every gap is within a few tens of px of
        # an integer multiple of P, whatever k is.
        tol = max(60.0, 0.04 * P)
        ok = np.abs(d - np.round(mult) * P) <= tol
        ok &= np.round(mult) >= 1
        support = int(ok.sum())
        # Every sub-multiple of the true period also has full integer support, so
        # rank: most support, then a candidate inside a nominal class (breaks the
        # P-vs-2P tie of a coded train with all-even gaps), then the LARGER P
        # (the fundamental, not a sub-harmonic).
        key = (support, _nominal(P), P)
        if best is None or key > best[0]:
            best = (key, P)
    P = best[1]
    ref = int(np.argmax(sc))
    x0 = xs[ref]

    def _lsq(mask, kk):
        A = np.column_stack([np.ones(int(mask.sum())), kk[mask]])
        sol, *_ = np.linalg.lstsq(A, xs[mask], rcond=None)
        return float(sol[0]), float(sol[1])

    # Grid snap with a TIGHT tolerance before any least squares (a single text
    # hit can lever a 4-point fit so that no residual looks large), growing the
    # radius outward from the reference mark so a prior period off by ~1 % does
    # not mis-index the far marks (index 130 x 10 px would be a whole period).
    snap_tol = lambda Pc: max(0.08 * Pc, 6.0)
    radius = 6.0 * P
    span = float(np.max(np.abs(xs - x0)))
    inl = np.zeros(len(xs), dtype=bool)
    for _ in range(64):
        k = np.round((xs - x0) / P)
        r = xs - (x0 + P * k)
        inl = (np.abs(xs - x0) <= radius) & (np.abs(r) <= snap_tol(P))
        if inl.sum() >= 2 and len(np.unique(k[inl])) >= 2:
            x0, P = _lsq(inl, k)
        if radius >= span:
            break
        radius *= 2.0
    # final MAD clipping on the converged grid
    for _ in range(max_iter):
        k = np.round((xs - x0) / P)
        r = xs - (x0 + P * k)
        mad = 1.4826 * float(np.median(np.abs(r[inl] - np.median(r[inl])))) if inl.sum() > 2 else 0.0
        tol = min(max(3.0 * mad, 6.0), snap_tol(P))
        inl_new = np.abs(r) <= tol
        if inl_new.sum() >= 2 and len(np.unique(k[inl_new])) >= 2:
            x0_new, P_new = _lsq(inl_new, k)
        else:
            break
        converged = np.array_equal(inl_new, inl) and abs(P_new - P) < 1e-3
        x0, P, inl = x0_new, P_new, inl_new
        if converged:
            break
    # Per-section phase (2026-09-19): the lattice is uniform on the FILM; in the merged mosaic
    # each section sits at its own seam tx, so the residual against one global grid steps at
    # every seam (F056: a +-40 px arc on both rails). Measure a median offset per section from
    # the current inliers, re-test every mark against its section's shifted grid, and report the
    # offsets -- they are the seam x errors the marks can check.
    sec_dx = {}
    off = np.zeros(len(xs))
    if sections is not None and inl.sum() >= 4:
        si = sections.index(xs)
        k0 = np.round((xs - x0) / P)
        r0 = xs - (x0 + P * k0)
        for sidx in np.unique(si):
            sel = inl & (si == sidx)
            if sel.sum() >= 2:
                sec_dx[int(sidx)] = float(np.median(r0[sel]))
        if sec_dx:
            off = np.array([sec_dx.get(int(v), 0.0) for v in si])
            k1 = np.round((xs - off - x0) / P)
            r1 = xs - off - (x0 + P * k1)
            mad = 1.4826 * float(np.median(np.abs(r1[inl]))) if inl.sum() > 2 else 0.0
            # the 500-cycle time track is variable-spacing BY DESIGN (NRO TCS-20055/69 p.7), so a
            # tight lattice tolerance throws away real marks: >= 60 px for a dense-class period,
            # >= 40 px for the scan-angle lattice, and never tighter than 3 x MAD.
            P_mm_here = P * pitch_x / 1000.0
            floor = 60.0 if abs(P_mm_here - PERIOD_CLASSES_MM["dense"]) <= PERIOD_CLASS_TOL * PERIOD_CLASSES_MM["dense"] else 40.0
            tol_s = max(3.0 * mad, floor)
            inl = np.abs(r1) <= tol_s
    k = np.round((xs - off - x0) / P).astype(int)
    r = xs - off - (x0 + P * k)
    for m, ki, ri, ii in zip(ms, k, r, inl):
        m.k, m.resid, m.inlier = int(ki), float(ri), bool(ii)
    ki = k[inl]
    kmin, kmax = int(ki.min()), int(ki.max())
    present = set(int(v) for v in ki)
    missing = [kk for kk in range(kmin, kmax + 1) if kk not in present]
    left, right = anchors.edges
    cx = 0.5 * (left + right)
    xi = xs[inl]
    j = int(np.argmin(np.abs(xi - cx)))
    P_mm = P * pitch_x / 1000.0
    dy = float(np.median([m.dy for m in ms]))
    cls = "other"
    for name, mm in PERIOD_CLASSES_MM.items():
        if abs(P_mm - mm) <= PERIOD_CLASS_TOL * mm:
            cls = name
    kinds = [m.kind for m, ii in zip(ms, inl) if ii]
    kind_major = max(set(kinds), key=kinds.count) if kinds else "?"
    n_slots = kmax - kmin + 1
    starts_near_centre = abs(xi.min() - cx) <= 2.5 * P
    coded = cls == "dense" and len(missing) >= 0.10 * n_slots and starts_near_centre
    pres = "".join("1" if kk in present else "0" for kk in range(kmin, min(kmax, kmin + 95) + 1))
    rr = r[inl]
    return Train(
        side=side, row=row, section_dx={int(a): round(float(b), 1) for a, b in sec_dx.items()}, dy_px=dy, dy_mm=dy * anchors.pitch_um[1] / 1000.0, label=cls,
        period_px=P, period_mm=P_mm, period_deg=math.degrees(P_mm / FOCAL_MM), x0=float(x0),
        n_marks=len(ms), n_inliers=int(inl.sum()), n_outliers=int((~inl).sum()),
        k_min=kmin, k_max=kmax, n_slots=n_slots, n_missing=len(missing), missing_k=missing[:200],
        resid_rms_px=float(np.sqrt(np.mean(rr ** 2))), resid_max_px=float(np.abs(rr).max()),
        x_first=float(xi.min()), x_last=float(xi.max()),
        first_minus_left_edge=float(xi.min() - left), right_edge_minus_last=float(right - xi.max()),
        n_beyond_left=int((xi < left).sum()), n_beyond_right=int((xi > right).sum()),
        centre_x=cx, k_at_centre=float((cx - x0) / P),
        nearest_mark_dx=float(xi[j] - cx), nearest_mark_dx_mm=float((xi[j] - cx) * pitch_x / 1000.0),
        nearest_mark_dx_deg=math.degrees(float((xi[j] - cx) * pitch_x / 1000.0) / FOCAL_MM),
        coded=bool(coded), kind_majority=kind_major, presence_from_first=pres,
    )


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def analyse(mosaic: Path, anchors: RailAnchors, out_dir: Path, entity: str, tag: str = "",
            sides=("top", "bottom"), kinds=("disk", "wheel"), score_min: float = 0.45,
            x_range: tuple[int, int] | None = None, block_w: int = 16384, figure: bool = True,
            tier_px: int | None = None, expected_kind: str | None = None) -> dict:
    t0 = time.time()
    marks, info = detect_marks(mosaic, anchors, sides=sides, kinds=kinds, block_w=block_w,
                               score_min=score_min, x_range=x_range)
    n_raw = len(marks)
    marks = refine_marks(mosaic, marks, anchors.pitch_um)
    logger.info("refine_marks: %d template hits -> %d isolated dots (native-res centroid, FWHM size, annulus test)",
                n_raw, len(marks))
    info["n_template_hits"] = n_raw
    sections = load_sections(mosaic)
    if sections is not None:
        logger.info("sections: %d from %s, cumulative ty %s px", len(sections.x_start), sections.source,
                    [int(v) for v in sections.cum_ty])
        info["sections"] = {"x_start": sections.x_start, "cum_ty": sections.cum_ty, "source": sections.source}
    trains: list[Train] = []
    rows_by_side: dict[str, dict[int, float]] = {}
    for side in sides:
        rows = cluster_rows_lines(marks, side, anchors.line(side), sections=sections)
        rows_by_side[side] = rows
        for row in rows:
            t = fit_train(marks, side, row, anchors, sections=sections)
            if t is not None:
                trains.append(t)
    # Class rules (dshean 2026-09-19): the 500-cycle time track is on ONE edge only (NRO TCS-20055/69
    # p.7), so a "dense" train may stand on one side only -- the side with the fuller one; and a dense
    # train that fills under half its slots is staircase debris, not a track.
    for t in trains:
        # demote a dense row only when a much fuller dense row (>= 3x) exists on the same rail --
        # a half-filled row is staircase debris only if the real track is also there
        fuller = [u for u in trains if u is not t and u.side == t.side and u.label == "dense"
                  and u.n_inliers >= 3 * max(t.n_inliers, 1)]
        if t.label == "dense" and fuller:
            t.label = "other"
    dense_sides = {}
    for t in trains:
        if t.label == "dense" and (t.side not in dense_sides or t.n_inliers > dense_sides[t.side].n_inliers):
            dense_sides[t.side] = t
    if len(dense_sides) > 1:
        keep = max(dense_sides.values(), key=lambda t: t.n_inliers).side
        for t in trains:
            if t.label == "dense" and t.side != keep:
                t.label = "other"
        logger.info("dense time track kept on the %s rail only", keep)
    left, right = anchors.edges
    sweep_px = (right - left)
    summary = {
        "entity": entity, "mosaic": str(mosaic), "tag": tag, "anchor_source": anchors.source,
        "scan_pitch_um": list(anchors.pitch_um), "exposure_edges_src": [left, right],
        "sweep_centre_x": 0.5 * (left + right), "sweep_width_px": sweep_px,
        "tier_px_canvas": tier_px,
        "sweep_width_mm": sweep_px * anchors.pitch_um[0] / 1000.0,
        "sweep_width_deg": math.degrees(sweep_px * anchors.pitch_um[0] / 1000.0 / FOCAL_MM),
        "expected_kind": expected_kind,
        "band_mm": [BAND_INNER_MM, BAND_OUTER_MM], "score_min": score_min, "x_range": list(x_range) if x_range else None,
        "line_rows_at_edges_and_centre": {
            s: [float(v) for v in anchors.line(s)(np.array([left, 0.5 * (left + right), right]))] for s in sides},
        "n_marks": len(marks), "rows": {s: {str(k): v for k, v in r.items()} for s, r in rows_by_side.items()},
        "trains": [asdict(t) for t in trains],
        "blocks": info["blocks"], "seconds": round(time.time() - t0, 1),
    }
    for t in trains:
        exp_n = sweep_px / t.period_px
        summary["trains"][trains.index(t)]["expected_marks_across_sweep"] = exp_n
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{entity}_timing_marks.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    with open(out_dir / f"{entity}_timing_marks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "side", "row", "train", "kind", "size_mm", "score", "dy", "k", "resid", "inlier"])
        for m in sorted(marks, key=lambda m: (m.side, m.row, m.x)):
            t_label = next((t.label for t in trains if t.side == m.side and t.row == m.row), "")
            w.writerow([f"{m.x:.2f}", f"{m.y:.1f}", m.side, m.row, t_label, m.kind, m.size_mm, f"{m.score:.3f}",
                        f"{m.dy:.1f}", m.k, f"{m.resid:.2f}", int(m.inlier)])
    if figure:
        try:
            plot_timing_marks(mosaic, anchors, marks, trains, info, summary, out_dir / f"{entity}_timing_marks.png")
        except Exception:
            logger.exception("figure failed (non-fatal)")
    for t in trains:
        logger.info("TRAIN %s row%d dy=%+.0f px (%.2f mm) %s P=%.2f px = %.3f mm = %.4f deg n=%d/%d slots missing=%d rms=%.2f px "
                    "first-left=%.0f right-last=%.0f nearest-to-center dx=%+.0f px (%+.3f deg) coded=%s kind=%s",
                    t.side, t.row, t.dy_px, t.dy_mm, t.label, t.period_px, t.period_mm, t.period_deg, t.n_inliers,
                    t.n_slots, t.n_missing, t.resid_rms_px, t.first_minus_left_edge, t.right_edge_minus_last,
                    t.nearest_mark_dx, t.nearest_mark_dx_deg, t.coded, t.kind_majority)
    return summary


# ----------------------------------------------------------------------------
# QC figure (docs/figure_style.md: aspect-true zooms, in-panel labels, 2-98 %
# stretch, provenance suptitle, nodata drawn distinctly, dpi 200)
# ----------------------------------------------------------------------------
_TRAIN_COLORS = {"dense": "#00d5ff", "mid": "#ffa500", "sparse": "#ff40ff", "other": "#a0a0a0", "": "#707070"}


def plot_timing_marks(mosaic: Path, anchors: RailAnchors, marks: list[Mark], trains: list[Train], info: dict,
                      summary: dict, out_png: Path, zoom_w: int = 1400) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import Window

    sides = [s for s in ("top", "bottom") if s in info["overview"]]
    left, right = anchors.edges
    cx = 0.5 * (left + right)
    n_ov = len(sides)
    from matplotlib.ticker import MultipleLocator
    sec = info.get("sections")
    fig = plt.figure(figsize=(24, 4.2 * n_ov + 2.6 * n_ov + 9.5), dpi=200)
    gs = fig.add_gridspec(2 * n_ov + 3, 3, height_ratios=[1.6] * n_ov + [1.0] * n_ov + [1.3, 1.5, 1.2],
                          hspace=0.32, wspace=0.05)
    label_of = {(t.side, t.row): t.label for t in trains}
    wide_axes = []

    def _xaxis(ax):
        # dshean 2026-09-19: "x axes don't line up ... need more x subticks so I can provide guidance"
        ax.xaxis.set_major_locator(MultipleLocator(25000)); ax.xaxis.set_minor_locator(MultipleLocator(5000))
        ax.tick_params(axis="x", which="major", labelsize=7, length=5)
        ax.tick_params(axis="x", which="minor", length=2.5)
        ax.grid(axis="x", which="major", alpha=0.25); ax.grid(axis="x", which="minor", alpha=0.10)
        if sec:
            for xb in sec["x_start"][1:]:
                ax.axvline(xb, color="#888888", lw=0.6, alpha=0.7)
        wide_axes.append(ax)

    # --- overview strips (x max-pooled) ---
    for i, side in enumerate(sides):
        ov = info["overview"][side]
        img, f = ov["image"], ov["factor"]
        ax = fig.add_subplot(gs[i, :])
        lo, hi = np.percentile(img, (2, 98))
        ax.imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1), aspect="auto",
                  extent=[ov["x0"], ov["x0"] + img.shape[1] * f, img.shape[0], 0], interpolation="nearest")
        for m in marks:
            if m.side != side:
                continue
            blk = next((b for b in ov["y0_per_block"] if b[0] <= m.x < b[0] + b[1]), None)
            y_loc = m.y - blk[2] if blk else np.nan
            c = _TRAIN_COLORS.get(label_of.get((side, m.row), ""), "#707070")
            ax.plot(m.x, y_loc, marker="o" if m.inlier else "x", ms=4 if m.inlier else 5, mfc="none", mec=c, color=c, lw=0.8)
        for xe, lab in ((left, "left edge"), (right, "right edge")):
            ax.axvline(xe, color="w", ls="--", lw=0.8)
        ax.axvline(cx, color="yellow", ls=":", lw=1.0)
        ax.text(0.005, 0.95, f"{side} rail band, x max-pooled {f}:1 (marks survive), rows = line {['-', '+'][side == 'bottom']}"
                f"[{BAND_INNER_MM}, {BAND_OUTER_MM}] mm; white dashed = exposure edges, yellow = sweep center",
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color="w",
                bbox=dict(facecolor="k", alpha=0.5, lw=0))
        ax.set_yticks([])
        ax.set_xlabel("raster x (px)", fontsize=8)
        ax.tick_params(labelsize=7)
        _xaxis(ax)
    # --- y position vs x per side (dshean 2026-09-19: "you need a y position value plot for each") ---
    for i, side in enumerate(sides):
        ax = fig.add_subplot(gs[n_ov + i, :])
        row_mean = {}
        for m in marks:
            if m.side == side and m.row >= 0 and m.inlier:
                row_mean.setdefault(m.row, []).append(m.dy)
        row_mean = {r: float(np.mean(v)) for r, v in row_mean.items()}
        for m in marks:
            if m.side != side or m.row < 0 or m.row not in row_mean:
                continue        # unassigned hits are not marks; they stay off the y panels
            c = _TRAIN_COLORS.get(label_of.get((side, m.row), ""), "#707070")
            ax.plot(m.x, m.dy - row_mean[m.row], marker="o" if m.inlier else "x", ms=3.5 if m.inlier else 4.5,
                    mfc="none", mec=c, color=c, lw=0)
        for r, mu in row_mean.items():
            ax.text(0.005, 0.95 - 0.1 * list(row_mean).index(r), f"row {r} ({label_of.get((side, r), '?')}): mean dy {mu:+.0f} px",
                    transform=ax.transAxes, fontsize=7, va="top")
        ax.axhline(0, color="k", lw=0.5)
        if sec:
            xx = np.linspace(0, max(sec["x_start"][-1] * 1.05, right), 600)
            sidx = np.clip(np.searchsorted(np.asarray(sec["x_start"]), xx, side="right") - 1, 0, len(sec["x_start"]) - 1)
            st = np.asarray(sec["cum_ty"])[sidx] * (-1 if side == "top" else 1)
            ax.plot(xx, st - float(np.mean(st)), color="#888888", lw=0.8, ls="--",
                    label="seam ty staircase (merge provenance), mean removed")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
        ax.axvline(left, color="k", ls="--", lw=0.8); ax.axvline(right, color="k", ls="--", lw=0.8)
        ax.set_ylabel(f"{side}: mark y - row mean (px)", fontsize=8)
        ax.set_xlabel("raster x (px)", fontsize=8); ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=0.3)
        if side == "top":
            ax.invert_yaxis()
        _xaxis(ax)
    # --- residuals ---
    ax = fig.add_subplot(gs[2 * n_ov, :])
    for t in trains:
        ms = [m for m in marks if m.side == t.side and m.row == t.row]
        c = _TRAIN_COLORS.get(t.label, "#707070")
        xi = [m.x for m in ms if m.inlier]
        ri = [m.resid for m in ms if m.inlier]
        ax.plot(xi, ri, ".", ms=3, color=c, ls="-" if t.side == "top" else "--", lw=0.4,
                label=f"{t.side} {t.label} P={t.period_px:.2f} px ({t.period_mm:.3f} mm, {t.period_deg:.4f} deg) "
                      f"n={t.n_inliers}/{t.n_slots} missing={t.n_missing} rms={t.resid_rms_px:.2f} px "
                      f"nearest-to-center {t.nearest_mark_dx:+.0f} px{' CODED' if t.coded else ''}")
        xo = [m.x for m in ms if not m.inlier]
        ax.plot(xo, [0] * len(xo), "x", ms=4, color=c, alpha=0.6)
    ax.axvline(left, color="k", ls="--", lw=0.8); ax.axvline(right, color="k", ls="--", lw=0.8); ax.axvline(cx, color="orange", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("x residual vs uniform grid (px)", fontsize=8); ax.set_xlabel("raster x (px)", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    _xaxis(ax)
    if trains:
        # legend BELOW the axes (dshean 2026-09-19: "the legend ... covers up data")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, framealpha=0.95, borderaxespad=0.0)
    else:
        ax.text(0.5, 0.5, "NO TRAIN FITTED", transform=ax.transAxes, ha="center", fontsize=14, color="r")
    for _a in wide_axes[1:]:
        _a.sharex(wide_axes[0])
    if wide_axes:
        wide_axes[0].set_xlim(-2000, max((sec["x_start"][-1] * 1.08) if sec else 0, right + 8000))

    # --- native zooms at left edge / centre / right edge, first side with an overview ---
    # zooms follow the SPARSE (scan-angle) train where one exists: the marks nearest the left edge,
    # the sweep center and the right edge (dshean 2026-09-19: A053's fixed-x windows showed the
    # start-of-frame slate and an empty centre)
    t_sp = sorted([t for t in trains if t.label == "sparse"], key=lambda t: -t.n_inliers)
    zoom_side = t_sp[0].side if t_sp else (sides[0] if sides else "top")
    zx = [left + 2000, cx - zoom_w // 2, right - 2000 - zoom_w]
    if t_sp:
        sx = np.array(sorted(m.x for m in marks if m.side == zoom_side and m.row == t_sp[0].row and m.inlier))
        if sx.size:
            zx = [int(sx[np.argmin(np.abs(sx - tgt))] - zoom_w // 2) for tgt in (left, cx, right)]
    band_in, band_out = info["band_px"]
    with rasterio.open(mosaic) as src:
        for j, x0 in enumerate(zx):
            ax = fig.add_subplot(gs[2 * n_ov + 1, j])
            x0 = int(min(max(0, x0), src.width - zoom_w))
            yl = float(anchors.line(zoom_side)(np.array([x0 + zoom_w / 2]))[0])
            y0 = yl - band_out - 30 if zoom_side == "top" else yl + band_in - 30
            y0 = int(max(0, y0)); h = int(band_out - band_in + 60)
            h = min(h, src.height - y0)
            if h <= 10 or x0 < 0:
                ax.text(0.5, 0.5, f"{zoom_side} zoom {j}: window outside the raster (x0={x0}, y0={y0}, h={h})",
                        transform=ax.transAxes, ha="center", fontsize=8, color="r"); ax.set_xticks([]); ax.set_yticks([]); continue
            a = src.read(1, window=Window(x0, y0, zoom_w, h))
            lo, hi = np.percentile(a, (2, 98))
            ax.imshow(a, cmap="gray", vmin=lo, vmax=max(hi, lo + 1), aspect="equal",
                      extent=[x0, x0 + zoom_w, y0 + h, y0], interpolation="nearest")
            for m in marks:
                if m.side == zoom_side and x0 <= m.x < x0 + zoom_w:
                    c = _TRAIN_COLORS.get(label_of.get((zoom_side, m.row), ""), "#707070")
                    ax.add_patch(plt.Circle((m.x, m.y), 45, fill=False, ec=c, lw=1.0, ls="-" if m.inlier else ":"))
                    ax.text(m.x, m.y - 50, f"{m.kind[0]}{m.score:.2f} k{m.k}", color=c, fontsize=6, ha="center")
            ax.text(0.01, 0.97, f"{zoom_side} native 1:1 x {x0}..{x0 + zoom_w} ({['nearest left edge', 'nearest sweep center', 'nearest right edge'][j]}{' sparse mark' if t_sp else ''})",
                    transform=ax.transAxes, va="top", fontsize=7, color="w", bbox=dict(facecolor="k", alpha=0.5, lw=0))
            ax.tick_params(labelsize=6)
            if j == 1:
                ax.axvline(cx, color="yellow", ls=":", lw=1)

    # --- spacing histogram + mean patch per train ---
    ax = fig.add_subplot(gs[2 * n_ov + 2, 0])
    for t in trains:
        xs = np.sort([m.x for m in marks if m.side == t.side and m.row == t.row and m.inlier])
        if len(xs) > 2:
            d = np.diff(xs)
            ax.hist(d / t.period_px, bins=np.arange(0.5, min(8, d.max() / t.period_px) + 1.0, 0.1), histtype="step",
                    color=_TRAIN_COLORS.get(t.label, "#707070"), label=f"{t.side} {t.label}")
    ax.set_xlabel("consecutive spacing / fitted period", fontsize=8); ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=7)
    for j, t in enumerate(trains[:2]):
        ax = fig.add_subplot(gs[2 * n_ov + 2, 1 + j])
        ms = [m for m in marks if m.side == t.side and m.row == t.row and m.inlier]
        patch = _mean_patch(mosaic, ms[:40], 120)
        if patch is not None:
            ax.imshow(patch, cmap="gray", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"mean patch {t.side} {t.label} ({t.kind_majority}, n<=40, dy={t.dy_mm:+.2f} mm)", fontsize=8)
    fig.suptitle(
        f"KH-9 PC rail mark trains -- {summary['entity']}  |  {Path(str(mosaic)).name}  |  pitch {anchors.pitch_um[0]:.4f}/{anchors.pitch_um[1]:.4f} um  "
        f"|  edges {left}..{right} (sweep {summary['sweep_width_deg']:.2f} deg, tier {summary.get('tier_px_canvas')})  |  anchor {anchors.source}\n"
        f"trains: " + ("; ".join(f"{t.side} {t.label} P={t.period_mm:.3f} mm n={t.n_inliers}/{t.n_slots} miss={t.n_missing}"
                                  f" dx_c={t.nearest_mark_dx:+.0f}px{' CODED' if t.coded else ''}" for t in trains) or "none")
        + f"  |  {summary.get('tag', '')}  {time.strftime('%Y-%m-%d %H:%M')}  timing_marks.py prototype",
        fontsize=9)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _mean_patch(mosaic: Path, ms: list[Mark], size: int) -> NDArray | None:
    if not ms:
        return None
    import rasterio
    from rasterio.windows import Window

    acc = np.zeros((size, size), dtype=np.float64)
    n = 0
    with rasterio.open(mosaic) as src:
        for m in ms:
            x0, y0 = int(round(m.x - size / 2)), int(round(m.y - size / 2))
            if x0 < 0 or y0 < 0 or x0 + size > src.width or y0 + size > src.height:
                continue
            acc += src.read(1, window=Window(x0, y0, size, size)).astype(np.float64)
            n += 1
    return acc / n if n else None


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _parse_pair(s: str | None, cast=float):
    if s is None:
        return None
    a, b = s.split(",")
    return (cast(a), cast(b))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mosaic", type=Path, help="merged frame tif (default: the raster the joblib was fitted on)")
    ap.add_argument("--joblib", type=Path, help="fitted restitution joblib (line models, edges, pitch) -- preferred anchor")
    ap.add_argument("--qc-json", type=Path, help="worker restit_qc json (median line rows; tilt unknown -> widened band + local probe)")
    ap.add_argument("--top-row", type=float, help="manual constant top line row (with --bottom-row, --edges, --pitch)")
    ap.add_argument("--bottom-row", type=float)
    ap.add_argument("--edges", help="override exposure edges 'left,right' (raster px)")
    ap.add_argument("--pitch", help="override scan pitch 'x_um,y_um'")
    ap.add_argument("--out", type=Path, required=True, help="output dir (json/csv/png); never a data-product dir")
    ap.add_argument("--entity", help="entity id (default: mosaic stem)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sides", default="top,bottom")
    ap.add_argument("--kinds", default="disk,wheel", help="template kinds to match")
    ap.add_argument("--score-min", type=float, default=0.45)
    ap.add_argument("--block-w", type=int, default=16384)
    ap.add_argument("--x-range", help="restrict to 'x0,x1' raster px (cheap centre probe: e.g. centre +- 10000)")
    ap.add_argument("--exact-rows", action="store_true",
                    help="with --top-row/--bottom-row: the rows are exact (e.g. from a rectified-canvas sidecar) -- "
                         "no per-block line probe, no widened band (2026-09-19)")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--threads", type=int, default=4, help="cv2 threads")
    ap.add_argument("-v", "--verbose", action="count", default=1)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    cv2.setNumThreads(max(1, a.threads))

    raster = a.mosaic
    if a.joblib:
        anchors, jr = anchors_from_joblib(a.joblib)
        raster = raster or jr
    elif a.qc_json:
        anchors = anchors_from_qc_json(a.qc_json)
    elif a.top_row is not None and a.bottom_row is not None:
        if not (a.edges and a.pitch):
            ap.error("--top-row/--bottom-row need --edges and --pitch")
        anchors = RailAnchors(_const(a.top_row), _const(a.bottom_row), _parse_pair(a.edges, int), _parse_pair(a.pitch),
                              "manual rows" + (" (exact)" if a.exact_rows else ""), constant_rows=not a.exact_rows)
    else:
        ap.error("one of --joblib, --qc-json, or --top-row/--bottom-row is required")
    if a.edges:
        anchors.edges = _parse_pair(a.edges, int)
    if a.pitch:
        anchors.pitch_um = _parse_pair(a.pitch)
    if raster is None or not Path(raster).exists():
        ap.error(f"mosaic not found: {raster}")
    raster = Path(raster)
    entity = a.entity or raster.stem
    tier_px = expected_kind = None
    try:
        from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
        tier_px = KH9ImageSpec.expected_size_from_file(raster)[0]
        expected_kind = {"disk": "disk", "wagon_wheel": "wheel"}[KH9ImageSpec.fiducial_type_from_mission(KH9ImageSpec.mission_from_filepath(raster))]
    except Exception as e:  # non-KH9 names (synthetic tests) are fine
        logger.info("no KH9ImageSpec tier/kind for %s (%s)", raster.name, e)
    x_range = _parse_pair(a.x_range, int) if a.x_range else None
    logger.info("timing_marks %s anchor=%s edges=%s pitch=%s tier=%s expected_kind=%s x_range=%s",
                entity, anchors.source, anchors.edges, anchors.pitch_um, tier_px, expected_kind, x_range)
    summary = analyse(raster, anchors, a.out, entity, tag=a.tag, sides=tuple(a.sides.split(",")),
                      kinds=tuple(a.kinds.split(",")), score_min=a.score_min, x_range=x_range, block_w=a.block_w,
                      figure=not a.no_figure, tier_px=tier_px, expected_kind=expected_kind)
    n_tr = len(summary["trains"])
    print(f"TIMING_MARKS_{'OK' if n_tr else 'NONE'}: {entity} marks={summary['n_marks']} trains={n_tr} "
          f"seconds={summary['seconds']} out={a.out}")
    return 0 if n_tr else 2


if __name__ == "__main__":
    sys.exit(main())
