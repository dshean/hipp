"""The conservative inward-envelope edge is an invariant across ALL strategies,
not only ``CollimationStrategy`` (David 2026-07-22: "some images may not have the
collimation lines, so need fallback ... I'd rather crop some valid pixels than
include black rectangle in the output").

When a frame has no collimation lines, MixedStrategy falls to ``PolyStrategy`` /
``FlatStrategy``. These used the DN-threshold film-FRAME boundary (out in the
black margin) and a single smooth curve / horizontal line that could cross a
scan-section black step. Here they must instead find the per-column CONTENT end
and clamp to the inward envelope, so the delivered edge never admits black frame.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.restitution.base import detect_content_edges
from hipp.kh9pc.restitution.flat_strategy import FlatStrategy
from hipp.kh9pc.restitution.poly_strategy import PolyResult, PolyStrategy


def _delivered_model(sub: SubImage, side: str):
    """The DELIVERY-time conservative edge model for the no-line PolyStrategy
    fallback (``_conservative_edge_model``), driven only by the search band."""
    ps = PolyStrategy()
    z = np.zeros((1, 2), dtype=int)
    ps._results = {side: PolyResult(z, z, np.zeros((1, 2)), 1.0, None, sub)}
    return ps._conservative_edge_model(side)

HW, W = 2000, 1200          # a boundary-anchored search window (no collimation line)


def _bottom_window(path: Path, black_left: int, black_right: int, margin_dn: int = 5) -> None:
    """Bottom window: textured content at the top (inner), a near-black frame
    below, starting at DIFFERENT rows in the two half-width scan sections."""
    rng = np.random.default_rng(2)
    img = np.full((HW, W), margin_dn, np.uint8)
    img[: max(black_left, black_right)] = rng.integers(30, 240, size=(max(black_left, black_right), W)).astype(np.uint8)
    img[black_left:, : W // 2] = margin_dn
    img[black_right:, W // 2 :] = margin_dn
    with rasterio.open(path, "w", driver="GTiff", width=W, height=HW, count=1, dtype="uint8") as d:
        d.write(img, 1)


def _top_window(path: Path, content_left: int, content_right: int) -> None:
    """Top window: a near-black frame above, textured content starting at
    DIFFERENT rows below in the two sections (content_* = content-start row)."""
    rng = np.random.default_rng(4)
    img = np.full((HW, W), 5, np.uint8)
    img[content_left:, : W // 2] = rng.integers(30, 240, size=(HW - content_left, W // 2)).astype(np.uint8)
    img[content_right:, W // 2 :] = rng.integers(30, 240, size=(HW - content_right, W - W // 2)).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=W, height=HW, count=1, dtype="uint8") as d:
        d.write(img, 1)


def test_poly_bottom_fallback_steps_inward(tmp_path: Path) -> None:
    """No-line PolyStrategy bottom edge: the shallow section's black is never
    crossed, though a single smooth polynomial would."""
    black_left, black_right = 1400, 1300               # right section's frame is shallower
    src = tmp_path / "b.tif"
    _bottom_window(src, black_left, black_right)
    with rasterio.open(src) as ds:
        sub = SubImage(ds, Window(0, 0, W, HW), out_shape=(1, HW // 10, 100))
        model = _delivered_model(sub, "bottom")   # DELIVERY edge (not strip placement)
    cols = np.linspace(0, W, 200)
    edge = model.predict(cols.reshape(-1, 1)).ravel()
    right, left = edge[cols >= W // 2], edge[cols < W // 2]
    assert right.max() <= black_right + 3, f"right edge admitted black: {right.max()}"
    assert left.max() <= black_left + 3, f"left edge admitted black: {left.max()}"
    poly = model._poly.predict(cols.reshape(-1, 1)).ravel()
    assert poly[cols >= W // 2].max() > black_right + 10, "control: unclamped poly would admit black"


def test_poly_top_fallback_steps_inward(tmp_path: Path) -> None:
    """No-line PolyStrategy top edge (mirrored): the delivered edge never sits
    above (outside) a section's content start, so no frame is admitted."""
    content_left, content_right = 600, 700             # right section's content starts deeper
    src = tmp_path / "t.tif"
    _top_window(src, content_left, content_right)
    with rasterio.open(src) as ds:
        sub = SubImage(ds, Window(0, 0, W, HW), out_shape=(1, HW // 10, 100))
        model = _delivered_model(sub, "top")
    cols = np.linspace(0, W, 200)
    edge = model.predict(cols.reshape(-1, 1)).ravel()
    right = edge[cols >= W // 2]
    # top crop keeps rows >= edge; to exclude frame the right edge must be >= its content start
    assert right.min() >= content_right - 3, f"top edge admitted frame above content: {right.min()}"
    poly = model._poly.predict(cols.reshape(-1, 1)).ravel()
    assert poly[cols >= W // 2].min() < content_right - 10, "control: unclamped poly would admit frame"


def test_flat_fallback_position_is_innermost(tmp_path: Path) -> None:
    """No-line FlatStrategy delivers ONE horizontal row = the innermost content
    edge, so no column's black frame is admitted."""
    black_left, black_right = 1400, 1300
    src = tmp_path / "f.tif"
    _bottom_window(src, black_left, black_right)
    with rasterio.open(src) as ds:
        sub = SubImage(ds, Window(0, 0, W, HW), out_shape=(1, HW // 10, 50))
        fr = FlatStrategy()._process_side(sub, "bottom")
    assert fr.position <= black_right + 3, f"flat bottom admitted black: {fr.position}"
    assert fr.position >= black_right - 60, f"flat bottom over-cut far past the shallow content end: {fr.position}"


def test_content_end_excludes_flat_grey_margin(tmp_path: Path) -> None:
    """The content-end scan stops at content->featureless even when the margin is
    a flat GREY (DN ~40, not black) -- the unexposed film margin case -- so the
    delivered edge is the true content end, not deep in the grey/black."""
    rng = np.random.default_rng(6)
    img = np.full((HW, W), 5, np.uint8)
    img[:1300] = rng.integers(30, 240, size=(1300, W)).astype(np.uint8)   # content
    img[1300:1700] = 42                                                    # flat grey margin
    # rows 1700+ stay near-black frame
    src = tmp_path / "g.tif"
    with rasterio.open(src, "w", driver="GTiff", width=W, height=HW, count=1, dtype="uint8") as d:
        d.write(img, 1)
    with rasterio.open(src) as ds:
        sub = SubImage(ds, Window(0, 0, W, HW), out_shape=(1, HW // 10, 100))
        edges = detect_content_edges(sub.band, "bottom", black_dn=20)
        rows = np.array([sub.to_global_y(r) for _, r in edges])
    assert len(edges) >= 90, f"content-end refused too many columns: {len(edges)}/100"
    assert abs(np.median(rows) - 1300) <= 60, f"edge not at the content->grey boundary: {np.median(rows)}"
