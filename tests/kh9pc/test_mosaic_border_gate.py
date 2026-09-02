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
