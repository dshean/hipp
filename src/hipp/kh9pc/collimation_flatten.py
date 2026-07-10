"""
Copyright (c) 2026 HIPP developers
Description: Radiometric flattening of the collimation-line band along the top and
    bottom edges of restituted KH-9 PC images. Same-sensor stereo pairs overlap
    exactly in this band, and the bright collimation line plus its halo cause
    correlator blunders downstream. This module derives a per-row median DN curve
    from each edge band and subtracts the excess brightness over the adjacent
    interior level, PRESERVING the background pixels — ground signal shows through
    the line, so nothing is masked by default. Optionally, truly saturated pixels
    (clipped highlights with no recoverable information) can be set to nodata via
    ``saturation_dn``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window

logger = logging.getLogger(__name__)

# Rows examined for the per-row brightness curve at each edge. The collimation
# line sits ~147-170 px from the edge (position shifts slightly per image with
# restitution) and its halo decays to the interior level within ~400 px
# (measured on mission 1214 imagery); rows with no excess are left untouched,
# so a generous band is safe.
DEFAULT_BAND_PX: int = 400

# Innermost rows of the examined band used as the local interior reference level.
_REF_PX: int = 64

# Rows over which the correction tapers to zero at the interior end of the band
# (cosine ramp), so there is no seam where the modified rows meet the untouched
# image and per-row medians deep in the band (which track TERRAIN, not the
# collimation artifact) are not applied at full strength.
DEFAULT_TAPER_PX: int = 96

# Cap on the optional per-row contrast gain (match_spread=True). The line core
# compresses spread 5-10x (NMAD 5-15 DN vs 55-80 interior on mission-1214
# samples); an uncapped gain there would amplify film grain as much as signal.
DEFAULT_GAIN_CAP: float = 4.0


@dataclass
class EdgeFlattenStats:
    """Per-edge summary of what the flattening changed (for logging/QC)."""

    side: str
    reference_dn: float
    max_excess_dn: float   # largest brightness removed (collimation line rows)
    max_deficit_dn: float  # largest brightness added (dark film margin rows)
    rows_darkened: int
    rows_brightened: int
    max_gain: float        # largest contrast gain applied (1.0 unless match_spread)
    saturated_px: int


def _flatten_band(
    band: NDArray[np.floating],
    saturation_dn: float | None,
    nodata: float,
    dtype_max: float,
    side: str,
    taper_px: int = DEFAULT_TAPER_PX,
    match_spread: bool = False,
    gain_cap: float = DEFAULT_GAIN_CAP,
) -> tuple[NDArray[np.floating], EdgeFlattenStats]:
    """Flatten one edge band (rows ordered outer edge first, regardless of side).

    1. per-row median DN over valid (non-nodata) pixels -> brightness curve;
    2. reference level = median of the curve over the innermost ``_REF_PX`` rows;
    3. shift every row's DN by (reference - curve) so each row's median lands ON
       the reference — bright rows (collimation line + halo) are darkened, dark
       rows (film margin) are brightened, ALL the way out to the physical edge;
    4. optionally (``match_spread``) also scale each row about its median by
       nmad_reference / nmad_row (clamped to [1, gain_cap]) to restore the
       contrast compressed under the bright exposure;
    5. the whole correction fades to zero over the innermost ``taper_px`` rows
       (cosine ramp) — smooth transition into the unmodified image, and per-row
       medians that track terrain rather than the artifact are not applied;
    6. results are clipped to [nodata + 1, dtype_max] so no valid pixel becomes
       nodata or wraps; only if ``saturation_dn`` is given do INPUT pixels
       at/above it (clipped highlights) become ``nodata``.
    """
    n_rows = band.shape[0]
    valid = band != nodata
    curve = np.full(n_rows, np.nan)
    nmad = np.full(n_rows, np.nan)
    for r in range(n_rows):
        row_valid = band[r][valid[r]]
        if row_valid.size:
            curve[r] = np.median(row_valid)
            nmad[r] = 1.4826 * np.median(np.abs(row_valid - curve[r]))
    reference = float(np.nanmedian(curve[-_REF_PX:]))
    nmad_ref = float(np.nanmedian(nmad[-_REF_PX:]))
    shift = np.nan_to_num(reference - curve, nan=0.0)  # >0 brightens, <0 darkens

    gain = np.ones(n_rows)
    if match_spread and nmad_ref > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.clip(np.nan_to_num(nmad_ref / nmad, nan=1.0, posinf=gain_cap), 1.0, gain_cap)

    # cosine taper: full correction until the last taper_px rows, then fade to 0
    taper = np.ones(n_rows)
    if taper_px > 0:
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper_px)))
        taper[n_rows - taper_px :] = ramp
    shift_t = shift * taper
    gain_t = 1.0 + (gain - 1.0) * taper

    out = band.copy()
    for r in np.flatnonzero((shift_t != 0) | (gain_t != 1.0)):
        row = out[r]
        row_valid = valid[r]
        corrected = (row[row_valid] - curve[r]) * gain_t[r] + curve[r] + shift_t[r]
        row[row_valid] = np.clip(corrected, nodata + 1, dtype_max)

    n_saturated = 0
    if saturation_dn is not None:
        saturated = valid & (band >= saturation_dn)
        out[saturated] = nodata
        n_saturated = int(saturated.sum())

    stats = EdgeFlattenStats(
        side=side,
        reference_dn=reference,
        max_excess_dn=float(-shift_t.min()) if shift_t.size else 0.0,
        max_deficit_dn=float(shift_t.max()) if shift_t.size else 0.0,
        rows_darkened=int((shift_t < 0).sum()),
        rows_brightened=int((shift_t > 0).sum()),
        max_gain=float(gain_t.max()),
        saturated_px=n_saturated,
    )
    return out, stats


def flatten_collimation_band(
    input_path: str | Path,
    output_path: str | Path,
    band_px: int = DEFAULT_BAND_PX,
    taper_px: int = DEFAULT_TAPER_PX,
    match_spread: bool = False,
    gain_cap: float = DEFAULT_GAIN_CAP,
    saturation_dn: float | None = None,
    nodata: float = 0.0,
    overwrite: bool = False,
) -> list[EdgeFlattenStats]:
    """Write a radiometrically-flattened copy of a restituted KH-9 PC image.

    The interior of the image is copied unchanged (blockwise, so arbitrarily
    large rasters stream through limited memory). Within the top and bottom
    ``band_px`` rows, every row's DN is shifted so its median lands on the local
    interior reference level: the collimation line and halo are darkened, the
    dark film margin is brightened — the whole band is corrected out to the
    physical edge. Background pixels are preserved everywhere, including under
    the collimation line (ground signal shows through it). Derive and apply this
    to EACH image of a same-sensor stereo pair (curves are per-image).

    Parameters
    ----------
    input_path:
        Restituted KH-9 PC image (``preprocess_kh9pc`` output).
    output_path:
        Flattened copy to create.
    band_px:
        Rows at each edge examined for the brightness curve.
    taper_px:
        Rows over which the correction fades to zero at the interior end of
        the band (cosine ramp) — no seam into the unmodified image.
    match_spread:
        Also scale each row about its median by nmad_ref/nmad_row (clamped to
        [1, gain_cap]) to restore the contrast compressed under the bright
        exposure. Default off — the line core's low NMAD means the gain
        amplifies film grain as much as ground signal; A/B before adopting.
    gain_cap:
        Upper clamp for the match_spread gain.
    saturation_dn:
        Optional: pixels at/above this DN become ``nodata`` (clipped highlights
        carry no recoverable information). Default None = disabled, every pixel
        is preserved and no nodata is introduced or tagged.
    nodata:
        Nodata value used when ``saturation_dn`` is set (default 0, the KH-9 PC
        convention). The output's nodata tag is only set in that case.
    overwrite:
        If False (default), raise if ``output_path`` exists.

    Returns
    -------
    list[EdgeFlattenStats]
        Stats for the top and bottom edges.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists (use overwrite=True)")

    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        if 2 * band_px >= height:
            raise ValueError(f"image height {height} too small for band_px {band_px}")
        dtype = np.dtype(src.dtypes[0])
        dtype_max = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else float("inf")

        # Phase 1: read + flatten the two edge bands (band_px x width fits in memory:
        # ~140 MB for a 345k px wide Byte scan).
        top_raw = src.read(1, window=Window(0, 0, width, band_px)).astype(np.float64)
        bot_raw = src.read(1, window=Window(0, height - band_px, width, band_px)).astype(np.float64)
        top, top_stats = _flatten_band(
            top_raw, saturation_dn, nodata, dtype_max, "top", taper_px, match_spread, gain_cap
        )
        # bottom band is processed outer-edge-first, so flip in and back out
        bot_flip, bot_stats = _flatten_band(
            bot_raw[::-1], saturation_dn, nodata, dtype_max, "bottom", taper_px, match_spread, gain_cap
        )
        bot = bot_flip[::-1]
        for s in (top_stats, bot_stats):
            logger.info(
                "%s: %s edge ref=%.0f DN, removed up to %.0f DN over %d rows, added up to %.0f DN "
                "over %d rows, max contrast gain %.1fx, %d saturated px -> nodata",
                input_path.name,
                s.side,
                s.reference_dn,
                s.max_excess_dn,
                s.rows_darkened,
                s.max_deficit_dn,
                s.rows_brightened,
                s.max_gain,
                s.saturated_px,
            )

        # Phase 2: blockwise copy, substituting edge-band rows where they overlap.
        profile = src.profile
        # Full-size KH-9 PC scans exceed the classic-TIFF 4 GiB limit even
        # LZW-compressed (measured: the write dies at exactly 2^32 bytes
        # without this); IF_SAFER only switches to BigTIFF when needed.
        profile.update(BIGTIFF="IF_SAFER")
        if not profile.get("compress"):
            profile.update(compress="lzw")
        if saturation_dn is not None:
            profile.update(nodata=nodata)
        with rasterio.open(output_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                block = src.read(1, window=window)
                r0, r1 = int(window.row_off), int(window.row_off + window.height)
                c0, c1 = int(window.col_off), int(window.col_off + window.width)
                if r0 < band_px:  # overlaps top band
                    lo, hi = r0, min(r1, band_px)
                    block[lo - r0 : hi - r0] = top[lo:hi, c0:c1].astype(dtype)
                if r1 > height - band_px:  # overlaps bottom band
                    lo, hi = max(r0, height - band_px), r1
                    b0 = lo - (height - band_px)
                    block[lo - r0 : hi - r0] = bot[b0 : b0 + (hi - lo), c0:c1].astype(dtype)
                dst.write(block, 1, window=window)

    return [top_stats, bot_stats]
