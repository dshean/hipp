"""Synthetic end-to-end test of the rail mark-train prototype
(``hipp.kh9pc.restitution.timing_marks``).

A small "mosaic" (9000 x 2000 px, tiled uint8 GeoTIFF) carries two tilted
collimation lines, a titling-edge dense train of disks with a coded gap pattern
(time word) starting just past the sweep centre, a time-track dense train of
wagon wheels on the other rail, a mid (1.048 in) train of wheels in a second
row, some text-like clutter, and noise. The detector gets the line models and
edges (the CollimationStrategy contract) and must recover periods, phases,
glyph kinds, and the injected gaps.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    if os.environ.get("TM_TEST_PATHLOAD") == "1":      # front-end run: skip the strategy-package imports (sklearn/skimage)
        raise ImportError("path-load requested")
    from hipp.kh9pc.restitution import timing_marks as tm
except ImportError:
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "timing_marks", Path(__file__).resolve().parents[2] / "src" / "hipp" / "kh9pc" / "restitution" / "timing_marks.py")
    tm = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = tm          # dataclasses resolve cls.__module__ through sys.modules
    _spec.loader.exec_module(tm)

FIGURE = os.environ.get("TM_TEST_FIGURE", "1") == "1"   # =0 on the memory-capped front ends

PITCH = 7.0
W, H = 9000, 2000
TOP0, BOT0, TILT = 900.0, 1700.0, 0.004          # line rows at x=0 and slope (px/px)
LEFT, RIGHT = 1500, 7500                        # exposure edges -> centre 4500
P_DENSE = 1300.0                                # ~9.1 mm at 7 um
P_MID = 3803.0                                  # 1.048 in
TOP_DENSE_DY, TOP_MID_DY = -820.0, -640.0       # offsets from the top line (px)
BOT_DENSE_DY, BOT_MID_DY = +600.0, +780.0
GAPS = {2, 3, 5, 8}                             # missing slots in the top (coded) train


def _line(y0):
    return lambda x: y0 + TILT * np.asarray(x, dtype=float)


def _paste(img, tmpl, cx, cy):
    th, tw = tmpl.shape
    x0, y0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
    if x0 < 0 or y0 < 0 or x0 + tw > img.shape[1] or y0 + th > img.shape[0]:
        return
    sub = img[y0:y0 + th, x0:x0 + tw]
    np.maximum(sub, tmpl, out=sub)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    rng = np.random.default_rng(7)
    # uint8 noise built row-chunk-wise: a float64 (H, W) temporary is ~280 MB and
    # the pfe front ends enforce a 6 GiB per-user cgroup (2026-08-26: the first
    # version of this fixture was OOM-killed there)
    img = np.empty((H, W), dtype=np.uint8)
    for r0 in range(0, H, 200):
        img[r0:r0 + 200] = (40 + 8 * rng.standard_normal((min(200, H - r0), W))).clip(0, 255).astype(np.uint8)
    xs = np.arange(W, dtype=float)
    for y0 in (TOP0, BOT0):                       # bright collimation lines, 6 px wide
        yy = (y0 + TILT * xs).astype(int)
        for d in range(-3, 4):
            img[yy + d, np.arange(W)] = 230
    # exposure: brighter textured content between the lines inside the edges
    for r0 in range(int(TOP0) + 60, int(BOT0) - 60, 200):
        r1 = min(r0 + 200, int(BOT0) - 60)
        img[r0:r1, LEFT:RIGHT] = (90 + 30 * rng.standard_normal((r1 - r0, RIGHT - LEFT))).clip(0, 255).astype(np.uint8)
    disk = tm.make_disk_template(60)
    wheel = tm.make_wheel_template(58, 8)
    truth = {"top_dense": [], "bot_dense": [], "bot_mid": []}
    # titling edge (top): coded dense train starting 450 px past the centre, gaps = time word bits
    cx = 0.5 * (LEFT + RIGHT)
    x_start = cx + 450
    k = 0
    while x_start + k * P_DENSE < W - 200:
        x = x_start + k * P_DENSE
        if k not in GAPS:
            _paste(img, disk, x, _line(TOP0)(x) + TOP_DENSE_DY)
            truth["top_dense"].append(x)
        k += 1
    # time-track edge (bottom): regular dense train of wheels across the whole film
    x = 300.0
    while x < W - 200:
        _paste(img, wheel, x, _line(BOT0)(x) + BOT_DENSE_DY)
        truth["bot_dense"].append(x)
        x += P_DENSE
    # mid train of wheels (second row, bottom)
    x = 700.0
    while x < W - 200:
        _paste(img, wheel, x, _line(BOT0)(x) + BOT_MID_DY)
        truth["bot_mid"].append(x)
        x += P_MID
    # text-like clutter on the titling edge left of the centre (rectangles)
    for xc in (2000, 2400, 2800, 3600, 4000):
        img[int(_line(TOP0)(xc) + TOP_MID_DY) - 90:int(_line(TOP0)(xc) + TOP_MID_DY) + 90, xc:xc + 40] = 220
    import rasterio

    p = tmp_path_factory.mktemp("tm") / "synthetic_mosaic.tif"
    with rasterio.open(p, "w", driver="GTiff", width=W, height=H, count=1, dtype="uint8",
                       tiled=True, blockxsize=256, blockysize=256, compress="lzw") as d:
        d.write(img, 1)
    anchors = tm.RailAnchors(_line(TOP0), _line(BOT0), (LEFT, RIGHT), (PITCH, PITCH), "synthetic")
    return p, anchors, truth


def _train(summary, side, label):
    ts = [t for t in summary["trains"] if t["side"] == side and t["label"] == label]
    assert ts, f"no {side} {label} train in {[(t['side'], t['label']) for t in summary['trains']]}"
    return ts[0]


def test_trains_recovered(synthetic, tmp_path):
    p, anchors, truth = synthetic
    summary = tm.analyse(p, anchors, tmp_path, "synthetic", tag="pytest", block_w=4096, figure=FIGURE)
    if FIGURE:
        assert (tmp_path / "synthetic_timing_marks.png").exists()
    assert (tmp_path / "synthetic_timing_marks.csv").exists()

    td = _train(summary, "top", "dense")
    assert abs(td["period_px"] - P_DENSE) < 0.5
    assert td["n_inliers"] == len(truth["top_dense"])
    assert td["kind_majority"] == "disk"
    assert td["coded"] is True
    # gaps: the missing grid indices relative to the first mark are exactly GAPS
    missing_rel = {kk - td["k_min"] for kk in td["missing_k"]}
    assert missing_rel == GAPS
    # first mark sits 450 px past the sweep centre
    assert abs(td["nearest_mark_dx"] - 450) < 2.0
    assert abs(td["dy_px"] - TOP_DENSE_DY) < 6

    bd = _train(summary, "bottom", "dense")
    assert abs(bd["period_px"] - P_DENSE) < 0.5
    assert bd["kind_majority"] == "wheel"
    assert bd["n_missing"] == 0
    assert bd["n_beyond_left"] >= 1 and bd["n_beyond_right"] >= 1     # film-wide train
    assert bd["coded"] is False
    assert bd["resid_rms_px"] < 1.0

    bm = _train(summary, "bottom", "mid")
    assert abs(bm["period_px"] - P_MID) < 1.0
    assert bm["n_inliers"] == len(truth["bot_mid"])
    assert abs(bm["dy_px"] - BOT_MID_DY) < 6


def test_qc_json_anchor_constant_rows(synthetic, tmp_path):
    """The qc-json path only knows median rows: the widened band + per-block line
    probe must still find the dense trains under the injected tilt."""
    p, anchors, truth = synthetic
    top_med = float(np.median(_line(TOP0)(np.array([LEFT, RIGHT]))))
    bot_med = float(np.median(_line(BOT0)(np.array([LEFT, RIGHT]))))
    const = tm.RailAnchors(tm._const(top_med), tm._const(bot_med), (LEFT, RIGHT), (PITCH, PITCH), "const", constant_rows=True)
    summary = tm.analyse(p, const, tmp_path, "synthetic_const", block_w=4096, figure=False)
    bd = _train(summary, "bottom", "dense")
    assert abs(bd["period_px"] - P_DENSE) < 0.5
    assert bd["n_inliers"] == len(truth["bot_dense"])


@pytest.mark.skipif(os.environ.get("TM_TEST_SKIP_CLI") == "1", reason="CLI imports the kh9pc package (memory-capped front end)")
def test_cli_manual_rows(synthetic, tmp_path):
    p, anchors, truth = synthetic
    rc = tm.main(["--mosaic", str(p), "--top-row", str(TOP0 + TILT * 4500), "--bottom-row", str(BOT0 + TILT * 4500),
                  "--edges", f"{LEFT},{RIGHT}", "--pitch", "7,7", "--out", str(tmp_path / "cli"), "--no-figure",
                  "--block-w", "4096", "--x-range", "3000,7000"])
    assert rc == 0
    j = (tmp_path / "cli" / "synthetic_mosaic_timing_marks.json").read_text()
    assert "\"trains\"" in j


# ----------------------------------------------------------------------------
# Pure-function tests (cv2 + numpy only; run on the memory-capped front ends
# with TM_TEST_PATHLOAD=1 -k "match_block or fit_train")
# ----------------------------------------------------------------------------
def _strip_with_marks(rng, width=6000, height=400, period=1300.0, gaps=(), kind="disk", y=200.0):
    img = np.empty((height, width), dtype=np.uint8)
    for r0 in range(0, height, 100):
        img[r0:r0 + 100] = (40 + 8 * rng.standard_normal((min(100, height - r0), width))).clip(0, 255).astype(np.uint8)
    tmpl = tm.make_disk_template(60) if kind == "disk" else tm.make_wheel_template(58, 8)
    truth = []
    k, x = 0, 400.0
    while x < width - 100:
        if k not in gaps:
            _paste(img, tmpl, x, y)
            truth.append(x)
        k += 1
        x += period
    return img, truth


def test_match_block_synthetic_strip():
    rng = np.random.default_rng(3)
    img, truth = _strip_with_marks(rng, gaps={2, 5}, kind="wheel")
    bank = tm.template_bank(7.0)
    nms = int(0.6 * min(t.image.shape[0] for t in bank))
    det = tm._match_block(img, bank, 0.45, nms)
    strong = sorted([d for d in det if d[2] >= 0.6], key=lambda d: d[0])
    assert len(strong) == len(truth), (len(strong), len(truth))
    for (x, y, sc, tid), xt in zip(strong, truth):
        assert abs(x - xt) < 1.0 and abs(y - 200) < 2.0
        assert bank[tid].kind == "wheel"


def test_fit_train_coded_and_regular():
    left, right = 1500, 7500
    anchors = tm.RailAnchors(tm._const(900.0), tm._const(1700.0), (left, right), (7.0, 7.0), "unit")
    P = 1300.0
    # titling edge: coded train starting 450 px past the centre with gaps
    cx = 0.5 * (left + right)
    marks = []
    for k in range(0, 5):
        if k in (1, 3):
            continue
        marks.append(tm.Mark(cx + 450 + k * P + 0.3 * (-1) ** k, 80.0, "top", 0.9, "disk", 0.42, dy=-820.0))
    # one outlier (text) in the same row
    marks.append(tm.Mark(cx - 2000.0, 80.0, "top", 0.5, "disk", 0.42, dy=-822.0))
    # time-track edge: regular film-wide train + a mid train in a second row
    xs = np.arange(300.0, 9000.0, P)
    marks += [tm.Mark(x, 2300.0, "bottom", 0.85, "wheel", 0.41, dy=600.0) for x in xs]
    marks += [tm.Mark(x, 2480.0, "bottom", 0.8, "wheel", 0.41, dy=780.0) for x in np.arange(700.0, 9000.0, 3803.0)]
    rows_top = tm.cluster_rows(marks, "top")
    rows_bot = tm.cluster_rows(marks, "bottom")
    assert len(rows_top) == 1 and len(rows_bot) == 2
    t = tm.fit_train(marks, "top", 0, anchors)
    assert t is not None and t.label == "dense" and abs(t.period_px - P) < 0.5
    assert t.n_outliers == 1 and t.n_missing == 2 and sorted(kk - t.k_min for kk in t.missing_k) == [1, 3]
    assert t.coded is True and abs(t.nearest_mark_dx - 450) < 1.0
    fits = {tm.fit_train(marks, "bottom", r, anchors).label: tm.fit_train(marks, "bottom", r, anchors) for r in rows_bot}
    assert set(fits) == {"dense", "mid"}
    assert fits["dense"].n_missing == 0 and fits["dense"].coded is False and fits["dense"].n_beyond_left >= 1
    assert abs(fits["mid"].period_px - 3803.0) < 1.0
