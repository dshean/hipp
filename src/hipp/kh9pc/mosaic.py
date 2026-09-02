"""
Copyright (c) 2026 HIPP developers
Description: Image mosaicking via sequential ORB keypoint matching and RANSAC-based
    Euclidean alignment. Replaces the external ASP ``image_mosaic`` tool with a
    pure-Python implementation using rasterio WarpedVRT for block-wise compositing.
"""

import logging
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window
from skimage.measure import ransac
from skimage.transform import EuclideanTransform

from hipp.image import LogProgressBar


@dataclass
class ImageAlignment:
    """Alignment result for a single image in a sequential alignment chain.

    Attributes
    ----------
    image_path : Path
        Path to the image file.
    relative_transform : np.ndarray
        3x3 homogeneous transformation matrix relative to the previous image
        (identity for the first/reference image).
    absolute_transform : np.ndarray
        3x3 homogeneous transformation matrix in the global/mosaic coordinate system,
        accumulated from the reference image.
    n_matches : int
        Total number of ORB keypoint matches found before RANSAC filtering
        (0 for the reference image).
    n_inliers : int
        Number of inlier matches kept after RANSAC filtering
        (0 for the reference image).
    """

    image_path: Path
    relative_transform: np.ndarray
    absolute_transform: np.ndarray
    n_matches: int
    n_inliers: int
    # seam diagnostics (2026-08-26): the RANSAC seam as measured (tx, ty px; rot deg) and
    # whether it was replaced by the frame's median step (see validate_seams)
    tx: float = 0.0
    ty: float = 0.0
    rot_deg: float = 0.0
    fallback: bool = False
    reason: str = ""
    # per-seam ORB matches kept for diagnostics (dshean 2026-08-26: weak overlaps are
    # spotted from the matches, not from seam lines): global coords in each section
    match_points_a: "np.ndarray | None" = None
    match_points_b: "np.ndarray | None" = None
    match_inliers: "np.ndarray | None" = None


logger = logging.getLogger(__name__)


####################################################################################################################################
#                                                   MAIN FUNCTIONS
####################################################################################################################################
def image_mosaic(
    image_paths: Sequence[str | Path],
    output_tif: str | Path,
    overwrite: bool = False,
    resampling: int = Resampling.cubic,
    overlap_width: int = 3000,
    bloc_height: int = 512,
    nfeature_per_block: int = 500,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 3.0,
) -> None:
    """Stitch a sequence of image tiles into a single GeoTIFF mosaic.

    Tiles are assumed to be ordered left-to-right. Alignment is computed
    sequentially: each tile is matched to its left neighbour via ORB keypoints
    extracted from the overlapping strip, then a RANSAC Euclidean transform is
    estimated and accumulated into an absolute transform from the first tile.
    The aligned tiles are then composited block-by-block; earlier tiles take
    priority over later ones at overlap regions.

    Skips writing if ``output_tif`` already exists and ``overwrite`` is False.

    Parameters
    ----------
    image_paths:
        Ordered list of tile paths (left to right).
    output_tif:
        Destination GeoTIFF path.
    overlap_width:
        Width in pixels of the overlap strip used for keypoint matching.
    bloc_height:
        Height of each horizontal block used during keypoint extraction.
    nfeature_per_block:
        Maximum ORB features detected per block.
    ransac_max_trials:
        Maximum RANSAC iterations for the Euclidean transform estimation.
    ransac_residual_threshold:
        Inlier distance threshold (pixels) for RANSAC.
    """
    # standardize paths
    output_tif = Path(output_tif)

    # manage overwrite
    if output_tif.exists() and not overwrite:
        logger.info("Skipping image_mosaic: %s (already exists, overwrite=False)", str(output_tif))
        return

    alignments = compute_sequential_alignments(
        image_paths,
        overlap_width=overlap_width,
        bloc_height=bloc_height,
        nfeature_per_block=nfeature_per_block,
        ransac_max_trials=ransac_max_trials,
        ransac_residual_threshold=ransac_residual_threshold,
    )

    alignments = level_gauge(alignments)
    write_mosaic(alignments, output_tif, resampling=resampling)
    return alignments



def level_gauge(alignments: list[ImageAlignment]) -> list[ImageAlignment]:
    """Re-gauge the chain so its MEAN rotation is zero (dshean 2026-08-25).
    Sequential joins are chained from section 0, so every section inherits the
    running sum of the small per-join rotations and the far end tilts -- the
    canvas then grows (~5 % on a 12-section frame, docs/mosaic_gauge_2026-08-24.md).
    Rotating every absolute transform by minus the mean angle keeps all relative
    geometry and shrinks the canvas; the collimation lines end up close to
    horizontal, so the restitution has less rigid rotation to absorb."""
    import dataclasses
    th = np.array([np.arctan2(a.absolute_transform[1, 0], a.absolute_transform[0, 0]) for a in alignments])
    mean = float(th.mean())
    c, s_ = np.cos(-mean), np.sin(-mean)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    out = [dataclasses.replace(a, absolute_transform=R @ a.absolute_transform) for a in alignments]
    logger.info("level gauge: chain rotation min/max %.4f/%.4f deg, mean %.4f deg removed",
                float(np.degrees(th.min())), float(np.degrees(th.max())), float(np.degrees(mean)))
    return out


def compute_sequential_alignments(
    image_paths: Sequence[str | Path],
    overlap_width: int = 3000,
    bloc_height: int = 512,
    nfeature_per_block: int = 500,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 3.0,
) -> list[ImageAlignment]:
    """Compute sequential alignments between images.

    Detects ORB keypoints between consecutive images, estimates RANSAC Euclidean
    transforms, and accumulates absolute transformations from the reference image.
    """
    # standardize path
    paths: list[Path] = [Path(f) for f in image_paths]

    identity = np.eye(3)
    alignments: list[ImageAlignment] = [
        ImageAlignment(
            image_path=paths[0],
            relative_transform=identity,
            absolute_transform=identity,
            n_matches=0,
            n_inliers=0,
        )
    ]

    for i in range(len(paths) - 1):
        logger.info("Matching '%s' with '%s'", str(paths[i]), str(paths[i + 1]))

        points_a, points_b = _extract_global_matches_from_overlap(
            paths[i],
            paths[i + 1],
            overlap_width,
            bloc_height,
            nfeature_per_block,
        )

        model_robust, inliers = ransac(
            (np.array(points_b, dtype=np.float32), np.array(points_a, dtype=np.float32)),
            EuclideanTransform,
            min_samples=3,
            residual_threshold=ransac_residual_threshold,
            max_trials=ransac_max_trials,
        )

        n_inliers = int(np.sum(inliers))
        logger.info("Inliers after RANSAC: %d/%d", n_inliers, len(points_a))

        relative_transform: np.ndarray = model_robust.params
        absolute_transform: np.ndarray = alignments[i].absolute_transform @ relative_transform

        rot = float(np.degrees(np.arctan2(relative_transform[1, 0], relative_transform[0, 0])))
        alignments.append(
            ImageAlignment(
                image_path=Path(paths[i + 1]),
                relative_transform=relative_transform,
                absolute_transform=absolute_transform,
                n_matches=len(points_a),
                n_inliers=n_inliers,
                tx=float(relative_transform[0, 2]), ty=float(relative_transform[1, 2]), rot_deg=rot,
                match_points_a=np.asarray(points_a, dtype=np.float32),
                match_points_b=np.asarray(points_b, dtype=np.float32),
                match_inliers=np.asarray(inliers, dtype=bool),
            )
        )

    return validate_seams(alignments)


def validate_seams(alignments: list[ImageAlignment], min_inliers: int = 30, strong_inliers: int = 500,
                   max_rot_deg: float = 0.3, max_dty: float = 600.0, max_dtx: float = 2000.0,
                   margin_refine: bool = False) -> list[ImageAlignment]:
    """Guard every seam with the scanner's physics (dshean 2026-08-26, ops323 A003: a
    featureless ocean overlap gave RANSAC a noise fit -- 2 inliers, 3.9 deg -- that placed
    the last section ~17 k px to the right and ~150 rows low; the canvas grew to 366 k for
    a 343.5 k sweep and the duplicated strip manufactured an exposure edge).

    2026-08-26 relaunch lesson: the first version replaced STRONG seams too (6-15 k
    inliers, rotation 0.10-0.15 deg -- the good-seam population spans +-0.1 deg) and
    moved sections by 450-1350 px; the margin-profile ncc refinement locked one
    timing-mark period (~1300 px) off. Rule now: a seam with >= strong_inliers is kept
    as measured whatever its rotation / step (the RANSAC fit on thousands of matches IS
    the measurement); only a WEAK seam (< strong_inliers) is replaced by the frame's
    median step (pure translation) when it has too few inliers, a rotation above
    max_rot_deg, a ty far from the median, or a tx more than max_dtx from the median
    step. The median step comes from the strong seams. margin_refine is OFF by default
    (period-ambiguous on the timing-mark train)."""
    import dataclasses
    if len(alignments) < 2:
        return alignments
    seams = alignments[1:]
    strong = [a for a in seams if a.n_inliers >= strong_inliers]
    good = strong or [a for a in seams if a.n_inliers >= min_inliers and abs(a.rot_deg) <= max_rot_deg]
    if len(good) >= 2:
        tx_med = float(np.median([a.tx for a in good])); ty_med = float(np.median([a.ty for a in good]))
    elif len(good) == 1:
        tx_med, ty_med = good[0].tx, good[0].ty
    else:
        logger.error("seam validation: no trustworthy seam in this frame (%d seams) -- mosaic left as measured; REVIEW",
                     len(seams))
        return alignments
    out = [alignments[0]]
    for a in seams:
        why = []
        if a.n_inliers >= strong_inliers:
            if abs(a.rot_deg) > 0.1 or abs(a.tx - tx_med) > 1000:
                logger.info("seam %s: strong (%d inliers) but rot %.3f deg / tx %+.0f vs median step -- kept as measured",
                            a.image_path.name, a.n_inliers, a.rot_deg, a.tx - tx_med)
        else:
            if a.n_inliers < min_inliers: why.append(f"inliers {a.n_inliers} < {min_inliers}")
            if abs(a.rot_deg) > max_rot_deg: why.append(f"rot {a.rot_deg:.3f} deg")
            if abs(a.ty - ty_med) > max_dty: why.append(f"ty {a.ty:.0f} vs median {ty_med:.0f}")
            if abs(a.tx - tx_med) > max_dtx: why.append(f"tx {a.tx:.0f} vs median step {tx_med:.0f}")
        if why:
            tx_use, ty_use, how = tx_med, ty_med, "median step"
            if margin_refine:
                # margin-profile refinement (period-ambiguous on the timing-mark train; off by default)
                try:
                    prev_path = out[-1].image_path
                    with rasterio.open(str(prev_path)) as sa, rasterio.open(str(a.image_path)) as sb:
                        Wa, Ha = sa.width, sa.height; Hb = sb.height
                        band_h = 1400
                        ov = int(max(1000, Wa - tx_med + 500))
                        wa_ = min(ov + 2 * 1500, Wa)
                        pa = sa.read(1, window=Window(Wa - wa_, Ha - band_h, wa_, band_h)).astype(np.float32)
                        pb = sb.read(1, window=Window(0, Hb - band_h, min(ov, sb.width), band_h)).astype(np.float32)
                        prof_a = np.median(pa, axis=0); prof_b = np.median(pb, axis=0)
                        prof_a -= prof_a.mean(); prof_b -= prof_b.mean()
                        cc = np.correlate(prof_a, prof_b, mode="valid")
                        norm = np.sqrt(np.convolve(prof_a ** 2, np.ones(prof_b.size), mode="valid") * (prof_b ** 2).sum()) + 1e-6
                        ncc = cc / norm
                        lag = int(np.argmax(ncc)); score = float(ncc[lag])
                        tx_ref = (Wa - wa_) + lag
                        if score >= 0.5 and abs(tx_ref - tx_med) <= 1500:
                            tx_use, how = float(tx_ref), f"margin-profile ncc {score:.2f}"
                except Exception as e_:  # noqa: BLE001
                    logger.warning("seam %s: margin-profile refinement failed (%s) -- median step kept", a.image_path.name, e_)
            rel = np.array([[1.0, 0.0, tx_use], [0.0, 1.0, ty_use], [0.0, 0.0, 1.0]])
            logger.warning("seam %s: %s -> FALLBACK (tx %.0f, ty %.0f via %s)",
                           a.image_path.name, "; ".join(why), tx_use, ty_use, how)
            # the record carries the translation USED (provenance + seam figures read tx/ty);
            # the measured seam survives in `reason` (audit r2 2026-08-26)
            a = dataclasses.replace(a, relative_transform=rel, tx=float(tx_use), ty=float(ty_use), rot_deg=0.0, fallback=True,
                                    reason="; ".join(why) + f" [{how}; measured tx {a.tx:.0f} ty {a.ty:.0f} rot {a.rot_deg:.3f}]")
        a = dataclasses.replace(a, absolute_transform=out[-1].absolute_transform @ a.relative_transform)
        out.append(a)
    logger.info("seams: %d measured, %d strong, %d fallback; median step tx %.0f ty %.0f", len(seams), len(strong),
                sum(1 for a in out[1:] if a.fallback), tx_med, ty_med)
    return out


def write_mosaic(
    alignments: list[ImageAlignment],
    output_tif: str | Path,
    resampling: int = Resampling.cubic,
) -> None:
    """Warp and merge all aligned images into a single output GeoTIFF.

    Images are warped into the output pixel space using WarpedVRT and merged
    block-by-block. NONZERO pixels from LATER images OVERWRITE earlier ones
    (audit M-9: the code has always been later-nonzero-wins; the previous
    docstring claimed the opposite).

    If any image extends above or to the left of the first image (negative coordinates
    after transformation), an offset is automatically applied to all transforms so that
    the full mosaic fits within the canvas without clipping.

    """
    # normalize path
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(exist_ok=True, parents=True)

    output_width, output_height, offset_x, offset_y = _compute_canvas(alignments)

    T_offset = np.array([[1, 0, -offset_x], [0, 1, -offset_y], [0, 0, 1]], dtype=float)

    fake_crs = rasterio.CRS.from_epsg(3857)
    dst_transform = rasterio.Affine.identity()

    profile = {
        "width": output_width,
        "height": output_height,
        "compress": "lzw",
        "driver": "GTiff",
        "BIGTIFF": "YES",
        "count": 1,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "dtype": "uint8",
    }

    n_blocks = (output_width // 256 + 1) * (output_height // 256 + 1)

    logger.info("Mosaicing %d images → %s (%d×%d px)", len(alignments), str(output_tif), output_width, output_height)

    widths = []
    for al in alignments:
        with rasterio.open(al.image_path) as _s:
            widths.append(_s.width)
    with rasterio.open(output_tif, "w+", **profile) as dst:
        for i, alignment in enumerate(alignments):
            logger.info("[%d/%d] %s", i + 1, len(alignments), alignment.image_path.name)
            pbar = LogProgressBar(f"mosaicing {alignment.image_path.name}", n_blocks, logger)

            adjusted_transform = T_offset @ alignment.absolute_transform
            # C5 (2026-08-23): the merge mask `warped != 0` implicitly
            # assumes a DARK scan background. 2026-vintage rescans have a
            # PINNED-BRIGHT background (DN~251) around the film in every
            # scan part, which `!= 0` pastes as valid content -- the white
            # staircase along the merged-mosaic edges. Restrict each part
            # to its detected film-content bbox (mapped through the part's
            # transform) so scan background never enters the mosaic.
            bbox = _part_content_bbox(alignment.image_path)
            if bbox is None:
                logger.info("%s: dark scan background (self-masking) or no "
                            "bright surround -- legacy full-part paste",
                            alignment.image_path.name)
            # interior cut budget from the measured overlaps (see _section_border_cuts)
            ovl_prev = (widths[i - 1] - alignments[i].tx) if i > 0 else None
            ovl_next = (widths[i] - alignments[i + 1].tx) if i + 1 < len(alignments) else None
            cuts = _section_border_cuts(alignment.image_path, first=(i == 0), last=(i == len(alignments) - 1),
                                        left_cut_px=None if ovl_prev is None else max(64, 0.35 * ovl_prev),
                                        right_cut_px=None if ovl_next is None else max(64, 0.35 * ovl_next))
            if ovl_prev is not None or ovl_next is not None:
                logger.info("%s: overlap prev %s next %s px -> interior cut caps %s / %s px", alignment.image_path.name,
                            None if ovl_prev is None else int(ovl_prev), None if ovl_next is None else int(ovl_next),
                            None if ovl_prev is None else int(max(64, 0.35 * ovl_prev)), None if ovl_next is None else int(max(64, 0.35 * ovl_next)))
            inv_t = np.linalg.inv(adjusted_transform)

            with rasterio.open(alignment.image_path) as src:
                with WarpedVRT(
                    src,
                    src_transform=rasterio.Affine(*adjusted_transform.flatten()[:6]),
                    src_crs=fake_crs,
                    dst_crs=fake_crs,
                    resampling=resampling,
                    width=output_width,
                    height=output_height,
                    transform=dst_transform,
                ) as vrt:
                    for block_idx, (_, window) in enumerate(dst.block_windows(1)):
                        pbar.update(block_idx)
                        warped = vrt.read(1, window=window)
                        # no nodata metadata is set: valid pixels can legitimately be 0 (dark areas)
                        mask = warped != 0
                        if (bbox is not None or cuts is not None) and mask.any():
                            # map this block's dst pixels back to part pixel coords
                            yy, xx = np.mgrid[window.row_off:window.row_off + window.height,
                                              window.col_off:window.col_off + window.width]
                            sx = inv_t[0, 0] * xx + inv_t[0, 1] * yy + inv_t[0, 2]
                            sy = inv_t[1, 0] * xx + inv_t[1, 1] * yy + inv_t[1, 2]
                            if bbox is not None:
                                x0, y0, x1, y1 = bbox
                                mask &= (sx >= x0) & (sx < x1) & (sy >= y0) & (sy < y1)
                            if cuts is not None:
                                # scanner border (white plateau / dark band / border line)
                                # of THIS section -> nodata, per column and per row
                                ix = np.clip(sx.astype(int), 0, cuts["top"].size - 1)
                                iy = np.clip(sy.astype(int), 0, cuts["left"].size - 1)
                                mask &= (sy >= cuts["top"][ix]) & (sy < cuts["bottom"][ix]) \
                                        & (sx >= cuts["left"][iy]) & (sx < cuts["right"][iy])
                        if not mask.any():
                            continue
                        existing = dst.read(1, window=window)
                        dst.write(np.where(mask, warped, existing), 1, window=window)
            pbar.close()

    logger.info("Mosaic written to %s", str(output_tif))


####################################################################################################################################
#                                                   STANDALONE FUNCTIONS
####################################################################################################################################


def image_mosaic_asp(
    image_paths: list[str | Path],
    output_image_path: str | Path,
    threads: int = 0,
    cleanup: bool = True,
    dryrun: bool = False,
) -> None:
    """Mosaic tiles via the external ASP ``image_mosaic`` CLI tool.

    Kept as a fallback/reference; prefer ``image_mosaic`` for the pure-Python implementation.
    Pass ``cleanup=True`` (default) to remove the auxiliary log and ``.aux.xml`` files ASP leaves behind.
    """
    str_image_paths = list(sorted([str(f) for f in image_paths]))

    cmd = [
        "image_mosaic",
        *str_image_paths,
        "--ot",
        "byte",
        "--overlap-width",
        "3000",
        "--threads",
        str(threads),
        "-o",
        str(output_image_path),
    ]

    logger.info("Running: %s", " ".join(cmd))

    if not dryrun:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error("image_mosaic_asp failed for %s: %s", output_image_path, e)

    if cleanup:
        for f in glob(f"{output_image_path}-log-image_mosaic-*.txt") + glob(f"{output_image_path}.aux.xml"):
            os.remove(f)


####################################################################################################################################
#                                                   PRIVATE FUNCTIONS
####################################################################################################################################



def _section_border_cuts(image_path, max_cut_px: int = 2000, smooth_step: float = 6.0,
                         left_cut_px: int | None = None, right_cut_px: int | None = None,
                         bright_dn: float = 200.0, bright_run_px: int = 100, dec: int = 1024,
                         first: bool = False, last: bool = False, outer_cut_px: int = 800):
    """Scanner-border cuts of one scan section, derived from its OUTER rows/cols
    (dshean 2026-08-25: "figure out the nodata value for each section from the
    outer rows/cols"). Per column (top/bottom) and per row (left/right), walk
    inward from the edge while the decimated profile stays SMOOTH (|step| <=
    smooth_step: a white plateau, a pinned background or a dark band with its
    gradient are all smooth; film base/grain and marks are not); if a short
    bright run (<= bright_run_px, the thin white border line) immediately follows
    that smooth run, include it. Never cut deeper than max_cut_px (absolute) of the
    dimension (3 % ~ 750 rows) -- the collimation lines sit >= 1400 px in.
    Returns full-resolution cut arrays (top[W], bottom[W], left[H], right[H]) in
    section pixel coords, or None when nothing is cut. Verified on the archive
    USGS sections: F013_a top plateau 0-284 (DN 233), bottom white line
    24314-24349 (DN 232) + dark band DN 17 to the edge."""
    with rasterio.open(image_path) as s:
        H, W = s.height, s.width
        oh, ow = min(dec, H), min(dec, W)
        # block AVERAGES (nearest decimation keeps the film grain, +-6-10 DN per
        # sample, which breaks the smooth-run walk after a few blocks)
        a = s.read(1, out_shape=(oh, ow), resampling=Resampling.average).astype(np.float32)
    fy, fx = H / oh, W / ow
    # GATE (dshean 2026-09-01, "black/nodata written onto valid exposed pixels"):
    # these cuts model the PINNED-BRIGHT scanner bed of some sessions (a white
    # plateau / border line, DN >= bright_dn, hugging the section edge).  On a
    # DARK-background session there is no such border: the walk from the edge
    # runs through the smooth dark canvas and film margin until the first bright
    # thing it meets -- the scan-angle marks, the printed labels, or the
    # collimation line itself -- and the "thin bright border line" extension then
    # cuts INTO it.  Measured on every 2026-08-27 block (all dark sessions): top
    # cuts 435-1129 px mean / up to 1997 px max, bottom 284-964 / 1997, i.e. the
    # film margins zeroed column by column and the line eaten in places.  The
    # DN-0 canvas of a dark session self-masks through the `!= 0` merge rule, so
    # nothing needs cutting there.  Same test as _part_content_bbox: no bright
    # bed in the outermost rows/cols -> no cuts.
    # PER EDGE (2026-09-02, ops251 / mission 1205 re-merge): a pooled four-edge test
    # passed every 1205 section -- their SEAM sides carry a bright bed strip (right
    # edge 72-88 % >= DN 200) while the top/bottom margins are the DN-0 canvas -- and
    # the top/bottom walks then cut 484-1405 px top / 242-1955 px bottom into the
    # marks again (all 14 entities).  Each edge is judged on its OWN outer rows/cols:
    # walked when they carry the bright bed, or -- on a bright-bed session -- when
    # they are a dark BAND (mean DN >= 10: F013 bottom, DN 17 under the white line)
    # rather than the DN-0 canvas; a DN-0 margin self-masks through the merge rule.
    edges = dict(top=a[:8], bottom=a[-8:], left=a[:, :8], right=a[:, -8:])
    bright = {k: float((v >= bright_dn).mean()) for k, v in edges.items()}
    session_bright = float(np.concatenate([v.ravel() for v in edges.values()]).__ge__(bright_dn).mean()) >= 0.05
    walk = {k: bright[k] >= 0.05 or (session_bright and float(np.mean(edges[k])) >= 10.0) for k in edges}
    if not any(walk.values()):
        logger.info("%s: dark scan background (no pinned-bright bed in the outer rows/cols) "
                    "-- no scanner-border cuts", Path(image_path).name)
        return None
    if not all(walk.values()):
        logger.info("%s: per-edge gate: %s dark (no cut); %s walked (bright frac %s)", Path(image_path).name,
                    ",".join(k for k in edges if not walk[k]), ",".join(k for k in edges if walk[k]),
                    " ".join(f"{k}={bright[k]:.2f}" for k in edges))

    def _walk(prof2d, n_full, f, cut_px):
        # prof2d: (n_edge_axis, n_other) rows ordered edge -> inward
        # audit M-6 (2026-08-25): the cap is ABSOLUTE px -- a scanner border is a physical
        # ~300-1100 px (border survey, 26 session x writer combos) whatever the section
        # size; 3 % of a 31.8 k-px section (954 px) clipped real borders by ~140 px
        n = prof2d.shape[0]
        max_cut = max(1, min(n, int(cut_px / f)))
        run_blocks = max(1, int(bright_run_px / f))
        cuts = np.zeros(prof2d.shape[1], dtype=np.float32)
        for j in range(prof2d.shape[1]):
            v = prof2d[:, j]
            i = 1
            while i < max_cut and abs(v[i] - v[i - 1]) <= smooth_step:
                i += 1
            if i <= 1:
                continue
            # thin bright border line right after the background run (F013: a
            # ~50-px mixed transition zone sits between the dark band and the
            # white line, so look a few blocks past the run for the bright start)
            k = i
            start = None
            # the bright border LINE follows a dark BAND (DN >= ~10, F013: 12-29);
            # never extend after a near-zero margin -- on dark sessions the DN-0
            # unexposed margin can run up to the collimation line itself
            if float(np.mean(v[:i])) >= 10.0:
                for t in range(i, min(max_cut, i + 3 + run_blocks)):
                    if v[t] >= bright_dn:
                        start = t; break
            if start is not None:
                k = start
                lim = min(max_cut, start + run_blocks)
                while k < lim and v[k] >= bright_dn:
                    k += 1
            cuts[j] = k * f
        return cuts

    top = _walk(a, H, fy, max_cut_px) if walk["top"] else np.zeros(ow, np.float32)            # rows from the top, per column
    bot = H - _walk(a[::-1], H, fy, max_cut_px) if walk["bottom"] else np.full(ow, H, np.float32)  # rows from the bottom, per column
    # the frame's outer ends (first section's left, last section's right) are not
    # covered by a neighbour: a smooth run there can be uniform content (cloud),
    # so cap them at outer_cut_px; interior section edges lie inside the ~3000-px
    # overlaps and may be cut to max_cut_px (the neighbour supplies the pixels)
    # 2026-08-26 (dshean: "unexpected nodata in the overlap area / seams"): an interior edge
    # may only be cut as deep as its measured overlap allows -- the caller passes
    # left_cut_px / right_cut_px = min(max_cut_px, 0.35 x overlap with that neighbour);
    # with both sides bounded this way >= 30 % of every overlap stays covered by both
    # sections and the partial-height notch (0.5-1.7 k px, 40-74 % of rows nodata, right
    # of every seam on the 2026-08-26 relaunch mosaics) cannot form.
    lcap = outer_cut_px if first else (max_cut_px if left_cut_px is None else int(min(max_cut_px, left_cut_px)))
    rcap = outer_cut_px if last else (max_cut_px if right_cut_px is None else int(min(max_cut_px, right_cut_px)))
    left = _walk(a.T, W, fx, lcap) if walk["left"] else np.zeros(oh, np.float32)
    right = W - _walk(a.T[::-1], W, fx, rcap) if walk["right"] else np.full(oh, W, np.float32)
    if not (top.any() or (bot < H).any() or left.any() or (right < W).any()):
        return None
    xs = np.arange(W); ys = np.arange(H)
    xd = (np.arange(ow) + 0.5) * fx; yd = (np.arange(oh) + 0.5) * fy
    cuts = dict(top=np.interp(xs, xd, top), bottom=np.interp(xs, xd, bot),
                left=np.interp(ys, yd, left), right=np.interp(ys, yd, right))
    logger.info("%s: scanner border nodata (px, median) top %d bottom %d left %d right %d",
                Path(image_path).name, int(np.median(cuts["top"])), int(H - np.median(cuts["bottom"])),
                int(np.median(cuts["left"])), int(W - np.median(cuts["right"])))
    return cuts


def _part_content_bbox(image_path, dark_dn: float = 8.0,
                       bright_dn: float = 249.0, min_frac: float = 0.15,
                       margin_px: int = 32):
    """Film-content bbox of one scan part, in part pixel coords.

    The scan background is uniform and pinned (bright ~DN 251 on
    2026-vintage rescans, dark ~0 on 2018-vintage); film content is
    strictly interior DN. Decimated occupancy scan -> (x0, y0, x1, y1)
    shrunk by ``margin_px`` so residual background at the boundary stays
    outside. Returns None when no plausible content region is found
    (caller falls back to legacy full-part paste, loudly).
    """
    with rasterio.open(image_path) as s:
        oh = min(1024, s.height)
        ow = min(1024, s.width)
        a = s.read(1, out_shape=(oh, ow))
        fy, fx = s.height / a.shape[0], s.width / a.shape[1]
    # audit H-2: act ONLY on bright-background scans (2026 vintage). A dark
    # background already self-masks via the != 0 merge rule, and on dark
    # parts the interior test can crop the rails (timing marks, titling)
    # out of the mosaic. C5b (dshean 2026-08-23, ops196 F003): the border
    # MEDIAN misclassifies bright-background parts whose film reaches the
    # border (3 of F003's 4 parts pasted whole, white band included) --
    # the vintage test is the PINNED-BRIGHT FRACTION: dark-vintage scans
    # have essentially none, bright-vintage backgrounds plenty.
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).astype(float)
    if (border >= bright_dn).mean() < 0.05:
        return None
    margin_px = max(margin_px, int(2 * max(fx, fy)))
    interior = (a > dark_dn) & (a < bright_dn)
    # C5b (dshean 2026-08-23, ops196 F003): a pinned-bright background
    # band's noisy edge pixels can exceed min_frac occupancy and drag
    # whole background rows into the bbox (measured: band rows median
    # DN 251 with 34-42% of pixels non-pinned). A content row/col must
    # also have an interior MEDIAN -- film rows measure median ~30-80.
    row_med = np.median(a, axis=1)
    col_med = np.median(a, axis=0)
    rows = np.flatnonzero((interior.mean(axis=1) > min_frac)
                          & (row_med > dark_dn) & (row_med < bright_dn))
    cols = np.flatnonzero((interior.mean(axis=0) > min_frac)
                          & (col_med > dark_dn) & (col_med < bright_dn))
    if rows.size < 4 or cols.size < 4:
        return None
    return (cols[0] * fx + margin_px, rows[0] * fy + margin_px,
            (cols[-1] + 1) * fx - margin_px, (rows[-1] + 1) * fy - margin_px)


def _compute_canvas(alignments: list[ImageAlignment]) -> tuple[int, int, float, float]:
    """Compute output canvas dimensions and the offset needed to shift all images into positive coordinates.

    Returns
    -------
    width : int
    height : int
    offset_x : float
        Horizontal shift to apply so the leftmost pixel lands at x=0.
    offset_y : float
        Vertical shift to apply so the topmost pixel lands at y=0.
    """
    all_corners: list[np.ndarray] = []
    for alignment in alignments:
        with rasterio.open(alignment.image_path) as src:
            w, h = src.width, src.height
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=float).T
        transformed = (alignment.absolute_transform @ corners)[:2]
        all_corners.append(transformed)

    stacked = np.hstack(all_corners)
    min_x, min_y = stacked[0].min(), stacked[1].min()
    width = int(np.ceil(stacked[0].max() - min_x))
    height = int(np.ceil(stacked[1].max() - min_y))
    return width, height, min_x, min_y


def _extract_global_matches_from_overlap(
    image_a_path: str | Path,
    image_b_path: str | Path,
    overlap_width: int = 3000,
    bloc_height: int = 1024,
    nfeature_per_block: int = 500,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Extract matched keypoints between the overlapping edge of two images, in horizontal blocks.

    Assumes image A is on the left and image B is on the right.
    """
    points_a, points_b = [], []

    with rasterio.open(image_a_path) as src_a, rasterio.open(image_b_path) as src_b:
        width_a = src_a.width
        height_a = src_a.height
        height_b = src_b.height

        if height_a != height_b:
            logger.warning(
                "Image heights differ for block-wise matching (%d != %d); matching only over the common height.",
                height_a,
                height_b,
            )
        height = min(height_a, height_b)

        for i in range(0, height, bloc_height):
            current_block_height = min(bloc_height, height - i)

            window_a = Window(
                col_off=width_a - overlap_width, row_off=i, width=overlap_width, height=current_block_height
            )
            window_b = Window(col_off=0, row_off=i, width=overlap_width, height=current_block_height)

            block_a = src_a.read(1, window=window_a)
            block_b = src_b.read(1, window=window_b)

            pts_a, pts_b = _match_orb_keypoints(block_a, block_b, nfeatures=nfeature_per_block)

            points_a.extend([(pt[0] + (width_a - overlap_width), pt[1] + i) for pt in pts_a])
            points_b.extend([(pt[0], pt[1] + i) for pt in pts_b])

    return points_a, points_b


def _match_orb_keypoints(
    image_a: cv2.typing.MatLike, image_b: cv2.typing.MatLike, nfeatures: int = 500
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Detect ORB keypoints and return matched coordinates between two grayscale images.

    Returns
    -------
    pts_a : list of tuple[float, float]
        Matched keypoint coordinates from image A.
    pts_b : list of tuple[float, float]
        Matched keypoint coordinates from image B.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)  # type: ignore[attr-defined]

    kp_a, des_a = orb.detectAndCompute(image_a, None)
    kp_b, des_b = orb.detectAndCompute(image_b, None)

    if des_a is None or des_b is None:
        return [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_a, des_b), key=lambda x: x.distance)

    pts_a = [kp_a[m.queryIdx].pt for m in matches]
    pts_b = [kp_b[m.trainIdx].pt for m in matches]

    return pts_a, pts_b
