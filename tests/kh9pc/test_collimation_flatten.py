"""Synthetic-raster tests for hipp.kh9pc.collimation_flatten."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from hipp.kh9pc.collimation_flatten import flatten_collimation_band

H, W = 1200, 300
INTERIOR_DN = 120
MARGIN_DN = 20
LINE_GLOW_DN = 110  # additive brightness of the collimation line over the ground
HALO_START = 157
HALO_END = 260
LINE_ROWS = slice(147, HALO_START)


def _synthetic(path: Path) -> None:
    """Restituted-KH9-like uint8 raster: dark film margin, bright collimation
    line as an ADDITIVE glow over textured ground, decaying halo, flat interior
    — mirrored top/bottom. Ground texture = column ramp so recovery of the
    background under the line is testable."""
    texture = (np.arange(W) % 40).astype(np.uint8)  # 0..39 column ramp
    img = np.tile(texture, (H, 1)) + INTERIOR_DN - 20
    img = img.astype(np.uint8)

    def apply_edge(rows: np.ndarray, arr: np.ndarray) -> None:
        arr[rows < 147] = MARGIN_DN
        arr[LINE_ROWS] = np.clip(arr[LINE_ROWS].astype(int) + LINE_GLOW_DN, 0, 255).astype(np.uint8)
        halo_rows = np.arange(HALO_START, HALO_END)
        decay = np.linspace(100, 0, halo_rows.size)
        for r, d in zip(halo_rows, decay):
            arr[r] = np.clip(arr[r].astype(int) + int(d), 0, 255).astype(np.uint8)

    apply_edge(np.arange(H), img)
    flipped = img[::-1]
    apply_edge(np.arange(H), flipped)
    img = flipped[::-1]
    img[200, 10] = 255  # lone clipped highlight in the top halo
    img[50, 10] = 250  # bright pixel in the dark margin: brightening must clip, not wrap
    profile = {"driver": "GTiff", "width": W, "height": H, "count": 1, "dtype": "uint8", "blockysize": 64}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(img, 1)


@pytest.fixture
def flattened(tmp_path: Path) -> tuple[np.ndarray, list, Path]:
    src = tmp_path / "in.tif"
    out = tmp_path / "out.tif"
    _synthetic(src)
    stats = flatten_collimation_band(src, out)
    with rasterio.open(out) as ds:
        return ds.read(1), stats, out


def test_no_nodata_introduced_by_default(flattened: tuple[np.ndarray, list, Path]) -> None:
    arr, _, out = flattened
    assert (arr > 0).all()  # every pixel preserved
    with rasterio.open(out) as ds:
        assert ds.nodata is None  # tag untouched when saturation is disabled


def test_background_recovered_under_line(flattened: tuple[np.ndarray, list, Path]) -> None:
    """The additive glow is removed and the ground texture under the line
    reappears at its true level (column ramp preserved)."""
    arr, _, _ = flattened
    for rows in (LINE_ROWS, slice(H - HALO_START, H - 147)):
        line = arr[rows].astype(int)
        assert abs(np.median(line) - INTERIOR_DN) <= 3
        # texture survives: column ramp spread ~40 DN retained
        assert np.percentile(line, 95) - np.percentile(line, 5) >= 30


def test_halo_flattened_to_interior_level(flattened: tuple[np.ndarray, list, Path]) -> None:
    arr, _, _ = flattened
    for row in (170, 200, 240, H - 171, H - 201, H - 241):
        assert abs(np.median(arr[row]) - INTERIOR_DN) <= 3, f"row {row}"


def test_margin_brightened_to_reference(flattened: tuple[np.ndarray, list, Path]) -> None:
    """Correction reaches the physical edge: the dark film margin is lifted so
    its row medians land on the interior reference."""
    arr, _, _ = flattened
    for rows in (slice(0, 147), slice(H - 147, H)):
        margin = arr[rows].astype(int)
        med = np.median(margin, axis=1)
        assert np.abs(med - INTERIOR_DN).max() <= 3
    # row 0 itself is corrected (all the way out to the edge)
    assert abs(np.median(arr[0]) - INTERIOR_DN) <= 3
    assert abs(np.median(arr[-1]) - INTERIOR_DN) <= 3


def test_brightening_clips_instead_of_wrapping(flattened: tuple[np.ndarray, list, Path]) -> None:
    arr, _, _ = flattened
    assert arr[50, 10] == 255  # 250 + ~100 shift must clip at the dtype ceiling


def test_interior_untouched(flattened: tuple[np.ndarray, list, Path]) -> None:
    arr, _, _ = flattened
    assert (arr[500:700].astype(int) - (np.arange(W) % 40) - (INTERIOR_DN - 20) == 0).all()


def test_saturation_optin(tmp_path: Path) -> None:
    src = tmp_path / "in.tif"
    out = tmp_path / "out.tif"
    _synthetic(src)
    stats = flatten_collimation_band(src, out, saturation_dn=250)
    with rasterio.open(out) as ds:
        arr = ds.read(1)
        assert ds.nodata == 0
    assert arr[200, 10] == 0  # the lone clipped highlight
    assert stats[0].saturated_px >= 1


def test_stats(flattened: tuple[np.ndarray, list, Path]) -> None:
    _, stats, _ = flattened
    assert [s.side for s in stats] == ["top", "bottom"]
    for s in stats:
        assert s.reference_dn == pytest.approx(INTERIOR_DN, abs=2)
        assert s.max_excess_dn == pytest.approx(LINE_GLOW_DN, abs=4)
        assert s.max_deficit_dn == pytest.approx(INTERIOR_DN - MARGIN_DN, abs=4)
        assert s.rows_darkened > 50  # line + halo
        assert s.rows_brightened >= 147  # film margin out to the edge
        assert s.saturated_px == 0  # saturation disabled by default


def test_refuses_existing_output(tmp_path: Path) -> None:
    src = tmp_path / "in.tif"
    out = tmp_path / "out.tif"
    _synthetic(src)
    out.touch()
    with pytest.raises(FileExistsError):
        flatten_collimation_band(src, out)
    flatten_collimation_band(src, out, overwrite=True)  # and this must succeed


def test_taper_smooth_transition_into_interior(tmp_path: Path) -> None:
    """A smooth brightness anomaly reaching into the taper zone is corrected,
    the correction fades toward band_px, and there is NO seam at the boundary."""
    src = tmp_path / "in.tif"
    out = tmp_path / "out.tif"
    rows = np.arange(H, dtype=float)
    bump = 60.0 * np.exp(-0.5 * ((rows - 250.0) / 40.0) ** 2)  # peak +60 DN at row 250
    img = np.clip(INTERIOR_DN + bump[:, None] + np.zeros((H, W)), 0, 255).astype(np.uint8)
    profile = {"driver": "GTiff", "width": W, "height": H, "count": 1, "dtype": "uint8", "blockysize": 64}
    with rasterio.open(src, "w", **profile) as dst:
        dst.write(img, 1)
    flatten_collimation_band(src, out)
    with rasterio.open(out) as ds:
        arr = ds.read(1)
    # bump removed where the taper is still ~full strength
    assert abs(int(np.median(arr[250])) - INTERIOR_DN) <= 4
    # untouched at/after the band boundary
    assert (arr[400:] == INTERIOR_DN).all()
    # no seam: consecutive row-median jumps stay small from bump through boundary
    meds = np.median(arr[200:420].astype(int), axis=1)
    assert np.abs(np.diff(meds)).max() <= 5


def test_match_spread_restores_contrast(tmp_path: Path) -> None:
    """Contrast compressed under the line is re-stretched when match_spread=True
    (and left compressed by default)."""
    src = tmp_path / "in.tif"
    out_plain = tmp_path / "plain.tif"
    out_gain = tmp_path / "gain.tif"
    texture = (np.arange(W) % 40).astype(float)  # spread ~40 DN
    img = np.tile(texture, (H, 1)) + INTERIOR_DN - 20
    # line rows: contrast crushed 5x + bright glow
    img[147:157] = (texture * 0.2)[None, :] + INTERIOR_DN + 100
    img = np.clip(img, 0, 255).astype(np.uint8)
    profile = {"driver": "GTiff", "width": W, "height": H, "count": 1, "dtype": "uint8", "blockysize": 64}
    with rasterio.open(src, "w", **profile) as dst:
        dst.write(img, 1)

    def spread(path: Path) -> float:
        with rasterio.open(path) as ds:
            line = ds.read(1)[150].astype(float)
        return float(np.percentile(line, 95) - np.percentile(line, 5))

    flatten_collimation_band(src, out_plain)
    flatten_collimation_band(src, out_gain, match_spread=True)
    assert spread(out_plain) <= 12  # still compressed (5x crush of ~36 -> ~7)
    assert spread(out_gain) >= 25  # re-stretched by the capped gain (4x)
    # medians land on the reference either way
    for p in (out_plain, out_gain):
        with rasterio.open(p) as ds:
            assert abs(np.median(ds.read(1)[150].astype(float)) - np.median(ds.read(1)[600].astype(float))) <= 4
