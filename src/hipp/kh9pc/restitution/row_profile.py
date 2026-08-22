"""
Copyright (c) 2026 HIPP developers
Description: Polarity-free row-profile classification of a merged KH-9 PC
    strip. Finds the exposed-film (content) rows, the uniform margins
    (dark 2018-era OR bright 2026-era scanner surround), and collimation-
    line candidate rows the way a human does: by TEXTURE and structure,
    never by absolute DN thresholds.

    Motivation (2026-08-21 audit): every existing detector encodes
    "background is dark" as an absolute DN cutoff, which inverted on the
    2026-vintage rescans (bright scanner surround, DN-251-pinned margin
    furniture) and collapsed the whole strategy cascade. The exposed area
    is trivially visible to a human because it is TEXTURED while margins
    are UNIFORM — properties that survive any polarity or era.

    This module is a WINDOW-PLACEMENT oracle for the existing strategies
    (Poly/Fiducial/Collimation): it yields global content-edge rows and
    line candidates with uncertainty, so their mature per-column refiners
    and template matchers can search small anchored windows instead of
    blind height-fraction strips.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray


@dataclass
class RowProfile:
    """Decimated per-row statistics and derived classification.

    All row indices are FULL-RESOLUTION raster rows.
    """

    row: NDArray[np.int_]            # full-res row of each profile sample
    median: NDArray[np.floating]     # per-row median DN across sampled cols
    spread: NDArray[np.floating]     # per-row robust spread (IQR/1.349)
    frac_dark: NDArray[np.floating]  # fraction of samples <= dark ceiling
    frac_sat: NDArray[np.floating]   # fraction of samples >= saturation floor
    textured: NDArray[np.bool_]      # classified textured-content rows
    content_top: int                 # first full-res row of the content run
    content_bottom: int              # last full-res row of the content run
    polarity_top: str                # 'dark' | 'bright' | 'mixed' margin above
    polarity_bottom: str             # same, below
    line_rows: list[int] = field(default_factory=list)  # collimation candidates
    step: int = 1                    # profile row stride (full-res rows)


def _classify_margin(median: NDArray, sel: NDArray) -> str:
    if not sel.any():
        return "mixed"
    m = float(np.median(median[sel]))
    if m < 40:
        return "dark"
    if m > 200:
        return "bright"
    return "mixed"


def compute_row_profile(
    raster_filepath: str | Path,
    col_samples: int = 512,
    row_step: int = 8,
    dark_dn: float = 25.0,
    sat_dn: float = 250.0,
    texture_spread_min: float = 6.0,
    min_content_frac: float = 0.30,
    smooth_rows: int = 5,
) -> RowProfile:
    """One strided pass over the strip -> classified row profile.

    Reads ``col_samples`` full-height columns spread across the strip
    (decimated in y by ``row_step``), so cost is O(col_samples * H/row_step)
    regardless of strip width (~seconds for a 350k x 25k mosaic).

    Texture test: a row is CONTENT when its cross-column robust spread
    exceeds ``texture_spread_min`` AND it is not pinned uniform (mostly
    dark or mostly saturated). Both dark-margin (2018) and bright-margin
    (2026) rows fail the test identically.
    """
    raster_filepath = Path(raster_filepath)
    with rasterio.open(raster_filepath) as src:
        W, H = src.width, src.height
        cols = np.linspace(int(0.02 * W), int(0.98 * W), col_samples).astype(int)
        n_rows = max(1, H // row_step)
        # sample the grid via a decimated full read of the sampled columns:
        # rasterio can't stride columns directly, so read a decimated
        # overview of the full strip and index the sampled columns.
        band = src.read(
            1, out_shape=(n_rows, min(W, col_samples * 4)),
            resampling=rasterio.enums.Resampling.nearest,
        )
        # map our sampled columns into the decimated grid
        cgrid = np.linspace(0, band.shape[1] - 1, col_samples).astype(int)
        samp = band[:, cgrid].astype(np.float32)

    rows = (np.arange(samp.shape[0]) * (H / samp.shape[0])).astype(int)
    med = np.median(samp, axis=1)
    q75, q25 = np.percentile(samp, [75, 25], axis=1)
    spread = (q75 - q25) / 1.349
    frac_dark = (samp <= dark_dn).mean(axis=1)
    frac_sat = (samp >= sat_dn).mean(axis=1)

    textured = (spread > texture_spread_min) & (frac_dark < 0.8) & (frac_sat < 0.8)
    if smooth_rows > 1:  # despeckle the classification
        k = np.ones(smooth_rows)
        textured = np.convolve(textured.astype(float), k / k.sum(),
                               mode="same") > 0.5

    # MIDDLE-OUT content span (David 2026-08-21): the strip center is
    # always exposed film; walk outward and accept a content edge only at
    # a TERMINAL margin — a non-textured run that continues to the raster
    # end with at most thin structured interruptions (timing marks /
    # collimation lines are <=~150 full-res rows each). Interior
    # low-texture spans (calm dark water, bright textureless snow/ice —
    # David's caveat) are NOT terminal: texture resumes beyond them, so
    # the walk continues through. "Edge-in" searching cannot make this
    # distinction; middle-out gets it for free.
    n = len(textured)
    max_furniture_rows = max(1, int(200 / row_step))  # thin structured runs ok

    def edge_out(start: int, direction: int) -> int:
        """Walk from `start` toward the raster end; return the profile
        index of the last CONTENT row before the terminal margin."""
        i = start
        last_content = start
        end = -1 if direction < 0 else n
        while i != end:
            if textured[i]:
                last_content = i
                i += direction
                continue
            # candidate margin run [i .. j)
            j = i
            while j != end and not textured[j]:
                j += direction
            if j == end:
                return last_content          # terminal margin: done
            # non-terminal: textured resumes at j. If the textured
            # resumption is only a thin structured sliver whose beyond-
            # side is again non-textured to the end, treat it as margin
            # furniture; else it is content (water/snow gap passed).
            k = j
            while k != end and textured[k]:
                k += direction
            if abs(k - j) <= max_furniture_rows:
                probe = k
                while probe != end and not textured[probe]:
                    probe += direction
                if probe == end:
                    return last_content      # furniture inside terminal margin
            i = j                            # genuine content resumes
        return last_content

    center = n // 2
    if not textured[center]:
        # recenter on the nearest textured row (center could be a water span)
        cand = np.flatnonzero(textured)
        if len(cand) == 0:
            raise ValueError(
                f"row-profile: no textured rows at all in {raster_filepath.name}")
        center = int(cand[np.argmin(np.abs(cand - center))])
    lo = edge_out(center, -1)
    hi = edge_out(center, +1)
    if (hi - lo) < min_content_frac * n:
        raise ValueError(
            f"row-profile: implausibly small content span in "
            f"{raster_filepath.name} ({hi - lo}/{n} profile rows)")
    content_top, content_bottom = int(rows[lo]), int(rows[min(hi, len(rows) - 1)])

    # margin polarity above/below the content
    above = np.arange(len(rows)) < lo
    below = np.arange(len(rows)) > hi
    pol_top = _classify_margin(med, above)
    pol_bot = _classify_margin(med, below)

    # collimation-line candidates: narrow bright median spikes in the
    # margin zones (prominence over the local margin level, width-bounded)
    line_rows: list[int] = []
    for sel in (above, below):
        idx = np.flatnonzero(sel)
        if len(idx) < 8:
            continue
        m = med[idx]
        base = np.median(m)
        prom = m - base
        # a line is a compact positive spike: > 5x local MAD and wider
        # margins (the pinned-bright slab) are excluded by a run-length cap
        mad = max(1.0, float(np.median(np.abs(m - base))))
        cand = prom > 5 * mad
        runs = []
        s = None
        for j, c in enumerate(np.append(cand, False)):
            if c and s is None:
                s = j
            elif not c and s is not None:
                runs.append((s, j - 1))
                s = None
        for s0, s1 in runs:
            if (s1 - s0 + 1) * row_step <= 120:  # thin line, not a slab
                peak = idx[s0 + int(np.argmax(m[s0:s1 + 1]))]
                line_rows.append(int(rows[peak]))

    return RowProfile(
        row=rows, median=med, spread=spread, frac_dark=frac_dark,
        frac_sat=frac_sat, textured=textured,
        content_top=content_top, content_bottom=content_bottom,
        polarity_top=pol_top, polarity_bottom=pol_bot,
        line_rows=sorted(line_rows), step=row_step,
    )
