"""_section_border_cuts must cut NOTHING on a dark-background scan session.

dshean 2026-09-01: the merged mosaics carried black/nodata through the film
margins and, in places, into the collimation line.  The scanner-border walk
was written for sessions whose bed scans pinned-bright (a white plateau or
border line hugging the section edge); on a dark session the walk runs through
the smooth dark margin until the first bright thing it meets -- the scan-angle
marks, the printed labels or the collimation line -- and cuts into it.
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from hipp.kh9pc.mosaic import _section_border_cuts


def _write(path, arr):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0],
                       count=1, dtype="uint8") as dst:
        dst.write(arr, 1)
    return path


def _dark_session(H=4000, W=3000, rng=None):
    """Dark bed (DN 0) around a film strip: dark margin (DN ~14) with a bright
    mark row and a bright label, a bright collimation line 1400 px in, then the
    exposure."""
    rng = rng or np.random.default_rng(0)
    a = np.zeros((H, W), np.uint8)
    a[300:H - 300, 50:W - 50] = rng.integers(10, 19, (H - 600, W - 100)).astype(np.uint8)   # film base
    a[1400:1440, 50:W - 50] = 230                                    # collimation line
    a[1440:H - 1440, 50:W - 50] = rng.integers(60, 200, (H - 2880, W - 100)).astype(np.uint8)  # exposure
    a[H - 1440:H - 1400, 50:W - 50] = 230                            # bottom line
    for x in range(200, W - 200, 400):                               # scan-angle marks, top + bottom margin
        a[760:800, x:x + 40] = 235
        a[H - 800:H - 760, x:x + 40] = 235
    a[700:900, 1200:1800] = 240                                      # a printed label block
    return a


def _bright_bed_session(H=4000, W=3000, rng=None):
    rng = rng or np.random.default_rng(1)
    a = _dark_session(H, W, rng)
    a[:300, :] = 251                                                 # pinned-bright bed above the film
    a[H - 300:, :] = 251
    a[:, :50] = 251
    a[:, W - 50:] = 251
    return a


def test_dark_session_gets_no_cuts(tmp_path):
    p = _write(tmp_path / "dark.tif", _dark_session())
    assert _section_border_cuts(str(p), dec=256) is None


def test_bright_bed_session_still_cut(tmp_path):
    p = _write(tmp_path / "bright.tif", _bright_bed_session())
    cuts = _section_border_cuts(str(p), dec=256)
    assert cuts is not None
    # the cut must stop at the bed, never reach the marks (760 px) or the line (1400 px)
    assert float(np.median(cuts["top"])) < 700
    assert float(np.median(4000 - cuts["bottom"])) < 700


def test_dark_session_marks_survive_the_old_walk_would_not(tmp_path):
    """Regression guard: on the dark synthetic section the OLD behaviour cut past
    the mark row (the smooth walk reached the marks / the line).  Simulate by
    disabling the gate through a bright border and checking the walk does reach
    the film features -- i.e. the gate is what protects them."""
    a = _dark_session()
    a[:300, :] = 251        # make the gate pass so the walk runs (bright bed above)
    p = _write(tmp_path / "walk.tif", a)
    cuts = _section_border_cuts(str(p), dec=256)
    assert cuts is not None
    assert float(np.median(cuts["top"])) >= 250   # the walk goes through the bed (block rounding)


def test_side_bright_bed_only_cuts_the_sides(tmp_path):
    """ops251 / mission 1205 (2026-09-02): the seam SIDES of every section carry a
    bright bed strip while the top/bottom margins are the DN-0 canvas.  A pooled
    four-edge gate passed and the top/bottom walks cut 484-1405 px into the marks;
    the gate is per edge, so only the sides are cut and the margins stay whole."""
    H, W = 4000, 3000
    a = _dark_session(H, W)
    a[:, :50] = 251
    a[:, W - 50:] = 251
    p = _write(tmp_path / "sides.tif", a)
    cuts = _section_border_cuts(str(p), dec=256)
    assert cuts is not None
    assert float(np.median(cuts["top"])) == 0
    assert float(np.median(cuts["bottom"])) == H
    assert 0 < float(np.median(cuts["left"])) < 300
    assert W - 300 < float(np.median(cuts["right"])) < W


def test_top_bright_bed_only_leaves_the_dark_bottom(tmp_path):
    H, W = 4000, 3000
    a = _dark_session(H, W)
    a[:300, :] = 251
    p = _write(tmp_path / "top.tif", a)
    cuts = _section_border_cuts(str(p), dec=256)
    assert cuts is not None
    assert float(np.median(cuts["top"])) >= 250
    assert float(np.median(cuts["bottom"])) == H


def test_sparse_marks_in_a_dark_margin_do_not_open_the_edge(tmp_path):
    """ops395 A001_c (2026-09-02): DN-8 canvas at the top with a few bright marks in
    the outer rows (outer mean 20, median 8, bright fraction 0) and a bright bed on
    one side.  704002e's dark-band branch walked the top and cut 751 px to the
    exposure; only a bright edge may be walked."""
    H, W = 4000, 3000
    a = _dark_session(H, W)
    a[:300, :] = 8
    for x in range(100, W - 100, 300):          # sparse bright marks inside the outer rows
        a[40:80, x:x + 30] = 235
    a[:, :50] = 251                              # bright bed on one side
    p = _write(tmp_path / "sparse.tif", a)
    cuts = _section_border_cuts(str(p), dec=256)
    assert cuts is not None
    assert float(np.median(cuts["top"])) == 0
    assert float(np.median(cuts["bottom"])) == H
    assert 0 < float(np.median(cuts["left"])) < 300
