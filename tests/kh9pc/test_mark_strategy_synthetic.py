"""Pixels-in, geometry-out test of mark-based restitution.

Renders synthetic film rails carrying a KNOWN scan-angle ladder -- both printed
classes (5 deg disks on missions <= 1213, 1 deg wagon wheels from 1214), both
sensors' mirrored distortion, an injected mosaic seam step, decoy dense trains on
the same rail, and a tilted rail -- then runs the production detector and warp fit
and checks the marks come back on their designed angles.

Rail layout follows the real frames (2026-08-30 fleet survey): on the bottom rail
the angle ladder sits ~200 px OUTSIDE the dense train (ops323 dense +647 / ladder
+843) and on the top rail ~200 px inside it, so the ruled +-60 px band around the
ladder excludes the decoy by position while the wide-band case has to exclude it
by CLASS.

The film-margin gate is exercised too: the fleet's biggest single failure was an
anchor that put the band on terrain, where mountains fitted a "ladder"
(ops251 F050).

This is the layer where detection meets geometry; the pure-math cases live in
``test_mark_geometry.py``.
"""

import json
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
cv2 = pytest.importorskip("cv2")

from hipp.kh9pc.kh9_image_spec import KH9ImageSpec                              # noqa: E402
from hipp.kh9pc.restitution.mark_geometry import (                              # noqa: E402
    CAMERA_SIGN,
    DISTORTION_SHAPE,
    LadderError,
    fit_mark_warp,
)
from hipp.kh9pc.restitution.mark_strategy import (                              # noqa: E402
    MarkOptions,
    camera_from_filepath,
    detect_angle_ladder,
    rectified_rail_strip,
    seam_positions_from_provenance,
    strip_is_film_margin,
)
from hipp.kh9pc.restitution.timing_marks import (                               # noqa: E402
    RailAnchors,
    make_disk_template,
    make_wheel_template,
)

pytestmark = pytest.mark.slow

# a 30 deg block: the SOURCE mosaic carries ~25 % more film than the 114082 px
# sweep it holds (ops196 A003 is 141128 px for a 114082 sweep), so the exposure and
# its ladder sit inside a wider raster -- exactly the placement problem.
TIER = 114082
SRC_W = 141128
SRC_H = 2000
BG = 40                          # unexposed film margin DN (dark and flat)
TOP_Y, BOT_Y = 700.0, 1300.0
TILT = 0.001                     # px/px; A010 measured 0.0027 over a 90 deg frame
LADDER_DY = (300.0, 400.0)       # (top, bottom) OUTWARD offsets of the ladder row
DECOY_DY = (500.0, 200.0)        # the dense train, ~200 px away as on real film
PITCH = (6.9887, 6.9887)
X_CENTRE = 60526.0
SCALE = 3808.0                   # source px per degree of scan angle


def line_y(x, y0):
    return y0 + TILT * np.asarray(x, float)


def shape_mosaic(alpha, camera):
    a2, a3, a4 = DISTORTION_SHAPE[camera]
    s = CAMERA_SIGN[camera]
    a = np.asarray(alpha, float)
    return s * (a2 * a ** 2 + a4 * a ** 4) + a3 * a ** 3


def truth_x(alpha, camera="F"):
    a = np.asarray(alpha, float)
    return X_CENTRE + SCALE * a + shape_mosaic(a, camera)


def _stamp(img, glyph, xc, yc):
    th, tw = glyph.shape
    x0, y0 = int(round(xc - tw / 2)), int(round(yc - th / 2))
    if x0 < 0 or y0 < 0 or x0 + tw > img.shape[1] or y0 + th > img.shape[0]:
        return False
    sub = img[y0:y0 + th, x0:x0 + tw]
    np.maximum(sub, glyph, out=sub)
    return True


def build_rail_image(path, camera="F", kind="disk", deg_per_mark=5.0, n_half=3,
                     step_px=0.0, step_after_kk=None, seed=0, decoys=True,
                     bright_band=False, neighbour_phase=None):
    """Write a synthetic mosaic; return ``(kk, alpha, x_truth)`` of the ladder marks.

    Marks follow the FLEET-CALIBRATED distortion for *camera*, so a correct fit has
    exactly two free parameters and a zero residual.
    """
    rng = np.random.default_rng(seed)
    px_per_mm = 1000.0 / PITCH[0]
    glyph = (make_disk_template(0.42 * px_per_mm) if kind == "disk"
             else make_wheel_template(0.41 * px_per_mm, 0.055 * px_per_mm))
    kk = np.arange(-n_half, n_half + 1, dtype=float)
    alpha = kk * deg_per_mark
    xs = truth_x(alpha, camera)
    if step_after_kk is not None:
        xs = xs + np.where(kk > step_after_kk, step_px, 0.0)

    with rasterio.open(path, "w", driver="GTiff", width=SRC_W, height=SRC_H, count=1,
                       dtype="uint8", tiled=True, blockxsize=512, blockysize=512,
                       compress="LZW", BIGTIFF="IF_SAFER") as dst:
        const = np.full((256, SRC_W), BG, np.uint8)
        for y0 in range(0, SRC_H, 256):
            rows = min(256, SRC_H - y0)
            dst.write(const[:rows], 1, window=rasterio.windows.Window(0, y0, SRC_W, rows))
        # only the two rail strips carry texture -- everything else stays constant,
        # so the fixture compresses and the build stays a few seconds
        for ybase, sgn, dy_l, dy_d in ((TOP_Y, -1.0, LADDER_DY[0], DECOY_DY[0]),
                                       (BOT_Y, +1.0, LADDER_DY[1], DECOY_DY[1])):
            lo = int(ybase + min(sgn * dy_l, sgn * dy_d) - 150)
            hi = int(ybase + TILT * SRC_W + max(sgn * dy_l, sgn * dy_d) + 150)
            base = 190 if bright_band else BG            # bright = exposed terrain
            strip = np.full((hi - lo, SRC_W), base, np.int16)
            amp = 40 if bright_band else 3
            strip = np.clip(strip + rng.integers(-amp, amp + 1, strip.shape,
                                                 dtype=np.int16), 0, 255).astype(np.uint8)
            for x in xs:                                            # the angle ladder
                _stamp(strip, glyph, x, line_y(x, ybase) + sgn * dy_l - lo)
            if neighbour_phase is not None:
                # the NEXT frame's ladder: same period, its own phase, beyond this
                # frame's exposure window (ops327 A003 / ops323 A003 class)
                P = SCALE * deg_per_mark
                for j in range(4):
                    xn = 123000.0 + neighbour_phase * P + j * P
                    if 0 < xn < SRC_W:
                        _stamp(strip, glyph, xn, line_y(xn, ybase) + sgn * dy_l - lo)
            if decoys:                                              # the dense train
                for x in np.arange(6000.0, SRC_W - 6000.0, 1300.0):
                    _stamp(strip, glyph, x, line_y(x, ybase) + sgn * dy_d - lo)
            dst.write(strip, 1, window=rasterio.windows.Window(0, lo, SRC_W, hi - lo))
    return kk, alpha, xs


def anchors():
    return RailAnchors(lambda x: line_y(x, TOP_Y), lambda x: line_y(x, BOT_Y),
                       (3024, 118029), PITCH, "synthetic")


def options(**kw):
    o = MarkOptions(band_dy_px=LADDER_DY, band_half_px=60.0, band_fallback_half_px=None,
                    min_cover_frac=0.3, min_marks=5)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _canvas_truth(warp, kk):
    """Where the drawn marks are DESIGNED to land, from truth alone."""
    return warp.cx + kk * warp.period_canvas


@pytest.fixture(scope="module", autouse=True)
def sector_env():
    """The sector tier is physics known to the caller, never guessed from a width."""
    old = os.environ.get("KH9_SECTOR_WIDTH_PX")
    os.environ["KH9_SECTOR_WIDTH_PX"] = str(TIER)
    yield
    if old is None:
        os.environ.pop("KH9_SECTOR_WIDTH_PX", None)
    else:
        os.environ["KH9_SECTOR_WIDTH_PX"] = old


@pytest.fixture(scope="module")
def disk_frame(tmp_path_factory):
    p = tmp_path_factory.mktemp("disk") / "D3C1210-200323F013.tif"   # 1210: disks, 5 deg, F
    return (p, *build_rail_image(p, camera="F"))


@pytest.fixture(scope="module")
def disk_frame_a(tmp_path_factory):
    p = tmp_path_factory.mktemp("diska") / "D3C1210-200323A002.tif"  # the mirrored sensor
    return (p, *build_rail_image(p, camera="A"))


@pytest.fixture(scope="module")
def wheel_frame(tmp_path_factory):
    p = tmp_path_factory.mktemp("wheel") / "D3C1216-200533F015.tif"  # 1216: wheels, 1 deg
    return (p, *build_rail_image(p, camera="F", kind="wheel", deg_per_mark=1.0, n_half=14))


# ------------------------------------------------------------------- identity
def test_camera_is_read_from_the_entity_id():
    assert camera_from_filepath("/x/D3C1210-200323F013.tif") == "F"
    assert camera_from_filepath("/x/D3C1216-200533A015.tif") == "A"
    with pytest.raises(LadderError):
        camera_from_filepath("/x/not_an_entity.tif")


# ---------------------------------------------------------------- 5 deg / disks
def test_five_degree_disk_ladder_recovers_exact_angles(disk_frame):
    p, kk, alpha, xs = disk_frame
    spec = KH9ImageSpec.from_raster_filepath(p)
    assert spec.expected_size[0] == TIER and spec.fiducial_type == "disk"

    trains, marks, chosen, info = detect_angle_ladder(p, anchors(), spec, options())
    assert {t.side for t in trains} == {"top", "bottom"}, info
    for t in trains:
        assert t.deg_per_mark == 5.0
        assert len(t.k) >= 5, f"{t.side}: only {len(t.k)} marks"
    assert all(c.label == "sparse" for c in chosen), [c.label for c in chosen]
    # the ladder sits at a CONSTANT row on the rectified strip: the detected dy of
    # every mark of a rail agrees to a couple of px across a 141 px rail tilt
    for t in trains:
        dy = np.array([m.dy for m in marks if m.side == t.side and m.x in set(t.x)])
        if dy.size > 2:
            assert dy.max() - dy.min() <= 4.0, f"{t.side} dy spread {dy.max() - dy.min():.1f} px"

    warp = fit_mark_warp(trains, TIER, 0.0, float(xs[kk == 0][0]), camera="F")
    assert warp.tier_width == TIER and warp.cx == warp.canvas_width / 2.0
    assert warp.deg_per_mark == 5.0
    lo, hi = warp.nominal_sweep_columns          # what cam_gen --pixel-values takes
    assert float(warp.alpha_deg(lo)) == pytest.approx(-15.0)
    assert float(warp.alpha_deg(hi)) == pytest.approx(+15.0)
    # the DETECTED marks land on their designed angles ...
    err = warp.placement_error_canvas_px(warp.marks_k, warp.marks_x)
    assert np.abs(err).max() < 5.0, f"max placement error {np.abs(err).max():.2f} canvas px"
    # ... and so do the TRUTH positions the image was drawn from
    assert np.abs(warp.canvas_x_of_source(xs) - _canvas_truth(warp, kk)).max() < 5.0
    # only phase and period were free, and the frame's period is recovered
    assert warp.scale_src_per_deg == pytest.approx(SCALE, rel=1e-4)
    assert warp.resid_rms_px < 3.0


def test_the_mirrored_A_sensor_recovers_too(disk_frame_a):
    p, kk, alpha, xs = disk_frame_a
    assert camera_from_filepath(p) == "A"
    spec = KH9ImageSpec.from_raster_filepath(p)
    trains, _, _, info = detect_angle_ladder(p, anchors(), spec, options())
    assert trains, info
    warp = fit_mark_warp(trains, TIER, 0.0, float(xs[kk == 0][0]), camera="A")
    assert np.abs(warp.canvas_x_of_source(xs) - _canvas_truth(warp, kk)).max() < 5.0
    assert warp.resid_rms_px < 3.0


def test_detected_mark_positions_match_the_drawn_truth(disk_frame):
    p, kk, alpha, xs = disk_frame
    spec = KH9ImageSpec.from_raster_filepath(p)
    trains, _, _, info = detect_angle_ladder(p, anchors(), spec, options())
    for t in trains:
        d = np.abs(t.x[:, None] - xs[None, :]).min(axis=1)
        assert d.max() < 2.0, f"{t.side}: worst detection offset {d.max():.2f} px"


def test_a_wide_band_rejects_the_dense_decoy_by_class(disk_frame):
    """With both trains in view the ladder must still be the one selected."""
    p, kk, alpha, xs = disk_frame
    spec = KH9ImageSpec.from_raster_filepath(p)
    wide = (150.0, 650.0)                              # covers ladder AND decoy
    trains, marks, chosen, info = detect_angle_ladder(p, anchors(), spec, options(),
                                                      band_px=wide)
    assert len(marks) > 150, "the decoy train should be visible in the wide band"
    assert chosen and all(c.label == "sparse" for c in chosen), [c.label for c in chosen]
    warp = fit_mark_warp(trains, TIER, 0.0, float(xs[kk == 0][0]), camera="F")
    assert np.abs(warp.canvas_x_of_source(xs) - _canvas_truth(warp, kk)).max() < 5.0


@pytest.mark.parametrize("alpha_off", [0.389, -0.389, 1.223, 6.391])
def test_off_centre_exposure_is_placed_at_its_measured_phase(disk_frame, alpha_off):
    """From pixels, at the fleet's measured cx scales: interior worst 0.389 deg,
    nepal PolyStrategy class 1.223 deg, first frame 6.391 deg."""
    p, kk, alpha, xs = disk_frame
    spec = KH9ImageSpec.from_raster_filepath(p)
    trains, _, _, _ = detect_angle_ladder(p, anchors(), spec, options())
    x_centre = float(truth_x(alpha_off, "F"))
    warp = fit_mark_warp(trains, TIER, alpha_off, x_centre, camera="F")
    landed = float(warp.canvas_x_of_source(x_centre))
    assert warp.alpha_deg(landed) == pytest.approx(alpha_off, abs=0.01)
    assert landed - warp.cx == pytest.approx(alpha_off * warp.px_per_deg, abs=40.0)


# ------------------------------------------------------------ 1 deg / wheels
def test_one_degree_wheel_ladder_recovers_exact_angles(wheel_frame):
    p, kk, alpha, xs = wheel_frame
    spec = KH9ImageSpec.from_raster_filepath(p)
    assert spec.fiducial_type == "wagon_wheel"

    trains, marks, chosen, info = detect_angle_ladder(p, anchors(), spec, options())
    assert trains, info
    for t in trains:
        assert t.deg_per_mark == 1.0
        assert len(t.k) >= 15, f"{t.side}: only {len(t.k)} marks"
    warp = fit_mark_warp(trains, TIER, 0.0, float(xs[kk == 0][0]), camera="F")
    assert warp.unwrap_half_ambiguity_deg == 0.5
    assert np.abs(warp.placement_error_canvas_px(warp.marks_k, warp.marks_x)).max() < 5.0
    assert np.abs(warp.canvas_x_of_source(xs) - _canvas_truth(warp, kk)).max() < 5.0


# ---------------------------------------------------------- film-margin gate
def test_the_rectified_strip_shears_a_tilted_rail_flat(tmp_path):
    """Step 4 of the canonical order, measured directly on a small raster.

    A bright row drawn at exactly ``line(x) + 60`` must land on ONE strip row for
    every column, whatever the tilt -- that is what deletes the tilted-band
    machinery from mark finding.
    """
    w, h, tilt = 40000, 500, 0.004          # 160 px of tilt across the raster
    path = tmp_path / "shear.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1,
                       dtype="uint8", tiled=True, compress="LZW") as dst:
        img = np.full((h, w), 30, np.uint8)
        cols = np.arange(w)
        rows = np.rint(100.0 + tilt * cols + 60.0).astype(int)
        img[rows, cols] = 240
        dst.write(img, 1)
    strip, row0 = rectified_rail_strip(path, lambda x: 100.0 + tilt * np.asarray(x, float),
                                       "bottom", 40.0, 90.0, 0, w)
    assert strip.shape == (50, w) and row0 == 40.0
    peak = np.argmax(strip, axis=0)
    # the drawn row is quantised to integer source rows, so the sheared strip sees
    # it at 19 or 20 -- a 1 px spread is the fixture, anything more is a shear error
    assert peak.max() - peak.min() <= 1, f"bright row spread {peak.min()}..{peak.max()}"
    assert 20 in (int(peak.min()), int(peak.max()))
    # ... and the un-sheared raster really was tilted, so the test is not vacuous
    assert tilt * w > 100.0


def test_a_dark_flat_strip_passes_the_film_margin_gate(disk_frame):
    p, kk, alpha, xs = disk_frame
    for side, y in (("top", TOP_Y), ("bottom", BOT_Y)):
        strip, _ = rectified_rail_strip(p, lambda x, y=y: line_y(x, y), side,
                                        240.0, 360.0, 0, 40000)
        ok, stats = strip_is_film_margin(strip, options())
        assert ok, (side, stats)
        assert stats["median_dn"] < 120.0


def test_a_band_on_exposed_terrain_is_refused(tmp_path):
    """The ops251 F050 failure: the band lands inside the image and 'marks' are
    mountains.  A bright, textured band never reaches the ladder fit."""
    p = tmp_path / "D3C1210-200323F019.tif"
    build_rail_image(p, bright_band=True)
    spec = KH9ImageSpec.from_raster_filepath(p)
    strip, _ = rectified_rail_strip(p, lambda x: line_y(x, BOT_Y), "bottom",
                                    340.0, 460.0, 0, 40000)
    ok, stats = strip_is_film_margin(strip, options())
    assert not ok and "EXPOSED CONTENT" in stats["verdict"], stats
    trains, marks, chosen, info = detect_angle_ladder(p, anchors(), spec, options())
    assert not trains and not marks
    assert all(v["margin_ok"] is False for v in info["detect"].values())


# ------------------------------------------------------------------ seam step
def test_a_seam_step_is_reported_as_qa_and_never_changes_the_warp(tmp_path):
    p = tmp_path / "D3C1216-200533F016.tif"      # 1 deg ladder: enough marks per section
    step, after = 8.0, 0.5
    kk, alpha, xs = build_rail_image(p, kind="wheel", deg_per_mark=1.0, n_half=14,
                                     step_px=step, step_after_kk=after, decoys=False)
    seam_x = float(xs[kk == 1][0] - 0.5 * SCALE)
    prov = tmp_path / "D3C1216-200533F016_merge_provenance.json"
    prov.write_text(json.dumps({
        "entity": "D3C1216-200533F016", "merged_size": [SRC_W, SRC_H],
        "seams": [{"section": "b.tif", "tx": seam_x - 4000.0, "overlap": 8000.0,
                   "n_inliers": 900, "fallback": False}]}))
    seams = seam_positions_from_provenance(prov)
    assert seams.x_src.size == 1
    assert seams.x_src[0] == pytest.approx(seam_x, abs=1.0)

    spec = KH9ImageSpec.from_raster_filepath(p)
    trains, _, _, _ = detect_angle_ladder(p, anchors(), spec, options())
    x0 = float(xs[kk == 0][0])
    plain = fit_mark_warp(trains, TIER, 0.0, x0, camera="F")
    qa = fit_mark_warp(trains, TIER, 0.0, x0, camera="F", seam_x_src=seams.x_src)
    grid = np.linspace(0.0, TIER, 501)
    assert np.array_equal(plain.source_x(grid), qa.source_x(grid)), \
        "seams must not change the delivered geometry"
    assert qa.seam_qa["resolvable"]
    assert qa.seam_qa["max_abs_step_px"] == pytest.approx(step, abs=4.0)


def test_a_seam_chain_that_does_not_cover_the_mosaic_is_rejected(tmp_path):
    prov = tmp_path / "bad_merge_provenance.json"
    prov.write_text(json.dumps({"merged_size": [SRC_W, SRC_H],
                                "seams": [{"tx": -5000.0, "overlap": 100.0}]}))
    s = seam_positions_from_provenance(prov)
    assert s.x_src.size == 0 and "REJECTED" in s.source


# ------------------------------------------------------------------ refusals
def test_a_blank_rail_is_refused_loudly(tmp_path):
    p = tmp_path / "D3C1210-200323F017.tif"
    with rasterio.open(p, "w", driver="GTiff", width=SRC_W, height=SRC_H, count=1,
                       dtype="uint8", tiled=True, compress="LZW", BIGTIFF="IF_SAFER") as dst:
        for y0 in range(0, SRC_H, 256):
            rows = min(256, SRC_H - y0)
            dst.write(np.full((rows, SRC_W), BG, np.uint8), 1,
                      window=rasterio.windows.Window(0, y0, SRC_W, rows))
    spec = KH9ImageSpec.from_raster_filepath(p)
    trains, marks, chosen, info = detect_angle_ladder(p, anchors(), spec, options())
    assert not trains
    with pytest.raises(LadderError):
        fit_mark_warp(trains, TIER, 0.0, X_CENTRE, camera="F")


def test_a_dense_train_alone_is_not_promoted_to_the_angle_ladder(tmp_path):
    p = tmp_path / "D3C1210-200323F018.tif"
    build_rail_image(p, n_half=0)                   # one ladder mark; decoys still drawn
    spec = KH9ImageSpec.from_raster_filepath(p)
    wide = (150.0, 650.0)
    trains, _, chosen, info = detect_angle_ladder(p, anchors(), spec, options(), band_px=wide)
    assert all(c.label == "sparse" for c in chosen), [c.label for c in chosen]
    with pytest.raises(LadderError):
        fit_mark_warp(trains, TIER, 0.0, X_CENTRE, camera="F")


# ------------------------------------------------- the next frame's ladder
def test_a_neighbour_frames_ladder_is_excluded_not_blended(tmp_path):
    """A scan can carry part of the next frame, rail marks included (ops327 A003).

    Its ladder has the SAME period and its OWN phase, so an unbounded fit could lock
    onto it or blend the two.  The fit must take this frame's, and say so.
    """
    p = tmp_path / "D3C1210-200323F020.tif"
    kk, alpha, xs = build_rail_image(p, neighbour_phase=0.37, decoys=False)
    spec = KH9ImageSpec.from_raster_filepath(p)
    trains, marks, chosen, info = detect_angle_ladder(p, anchors(), spec, options())
    assert trains, info
    # the neighbour's marks were seen, bounded out, and reported -- not blended
    seen = [v.get("outside_frame_extent", 0) for v in info["detect"].values()]
    assert max(seen) > 0, info["detect"]
    for t in trains:
        d = np.abs(t.x[:, None] - xs[None, :]).min(axis=1)
        assert d.max() < 2.0, "a neighbour mark entered this frame's train"
    warp = fit_mark_warp(trains, TIER, 0.0, float(xs[kk == 0][0]), camera="F")
    # the phase is this frame's: the drawn marks land on their designed angles
    assert np.abs(warp.canvas_x_of_source(xs) - _canvas_truth(warp, kk)).max() < 5.0


def test_the_straightness_check_reports_a_constant_row(disk_frame):
    """The y/rectification check: after the shear a train runs along a constant row."""
    p, kk, alpha, xs = disk_frame
    spec = KH9ImageSpec.from_raster_filepath(p)
    trains, marks, chosen, info = detect_angle_ladder(p, anchors(), spec, options())
    st = info["straightness"]
    assert st, info
    for k, v in st.items():
        if "rms_px" in v:
            assert v["verdict"] == "OK", (k, v)
            assert v["rms_px"] < 4.0, (k, v)
