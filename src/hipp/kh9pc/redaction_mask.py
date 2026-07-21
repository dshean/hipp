"""
Copyright (c) 2026 HIPP developers
Description: DIGITAL-REDACTION masking for KH-9 PC edge/mark detection.
    Declassified frames carry digitally redacted rectangles (and hard scan
    margins) at a constant near-black DN. Real scanned film, however dark,
    keeps grain noise — so a per-column range test inside a detector's search
    band separates redacted regions (uniform) from content (noisy) without
    scanning the whole image. Detection scans then skip through redacted runs
    instead of snapping to a redaction boundary (observed: PolyStrategy
    top-edge fits following a redaction box instead of the film edge).
"""

import numpy as np
from numpy.typing import NDArray


def redacted_region_mask(
    band: NDArray[np.number],
    max_dn: float = 6.0,
    max_range: float = 2.0,
    min_run: int = 3,
) -> NDArray[np.bool_]:
    """Boolean mask of DIGITALLY REDACTED samples in a (rows, cols) search band.

    A sample is masked when it belongs to a per-column run of >= ``min_run``
    consecutive samples that are all <= ``max_dn`` AND whose within-run range
    is <= ``max_range``. Digital redaction (and hard margin fill) is
    exactly-constant near-black; scanned film grain exceeds the range test
    even in deep shadow. Partial-width redactions fall out naturally
    (per-column decision).
    """
    band = np.asarray(band)
    dark = band <= max_dn
    mask = np.zeros(band.shape, dtype=bool)
    for col in range(band.shape[1]):
        d = dark[:, col]
        if not d.any():
            continue
        # maximal consecutive dark runs in this column
        edges = np.flatnonzero(np.diff(np.concatenate(([0], d.astype(np.int8), [0]))))
        for r0, r1 in zip(edges[::2], edges[1::2]):
            if r1 - r0 < min_run:
                continue
            seg = band[r0:r1, col]
            if float(seg.max()) - float(seg.min()) <= max_range:
                mask[r0:r1, col] = True
    return mask


def detect_ruptures_skip_redacted(
    vec: NDArray[np.number],
    threshold: float,
    redacted: NDArray[np.bool_],
    reverse_scan: bool = False,
) -> NDArray[np.integer]:
    """`detect_ruptures` that SKIPS digitally redacted samples.

    The redacted samples are removed before the falling-edge scan (splicing
    their neighbours together), so a threshold crossing can only occur at a
    genuine content/background transition; returned indices are in the
    ORIGINAL vector coordinates.
    """
    from hipp.kh9pc.restitution.base import detect_ruptures

    keep = ~np.asarray(redacted, dtype=bool)
    if keep.sum() < 2:
        return np.array([], dtype=int)
    idxmap = np.flatnonzero(keep)
    r = detect_ruptures(np.asarray(vec)[keep], threshold, reverse_scan=reverse_scan)
    return idxmap[np.asarray(r, dtype=int)] if len(r) else np.array([], dtype=int)
