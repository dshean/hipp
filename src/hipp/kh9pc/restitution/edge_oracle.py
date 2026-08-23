"""
Copyright (c) 2026 HIPP developers
Description: Clean-slate format-edge oracle for KH-9 PC merged mosaics
    (dshean 2026-08-23 design, review-accepted on casa_block):

    Per narrow column strip (tilt within a strip is negligible), the
    CONTENT DN distribution is characterized from the central band; the
    per-strip row-median profile is then walked OUTWARD from center. The
    FRAME EDGE is the first sustained drop to sub-content-dark (the
    unexposed film border) and is the MANDATORY primitive; a thin bright
    spike just inside it (the continuous edge line, where present) is
    optional supporting evidence — not all frames carry it, and content
    near the exposure edge can hide it locally. First-excursion-then-drop
    ordering, never amplitude, does the disambiguation. Detections feed a
    robust poly fit per side, with per-strip anatomy validation (textured
    content inward, margin outward).

    This replaces DN-threshold edge walks (black_dn) that inverted on the
    2026-vintage rescans and locked onto outer margin structure.
"""
# FIX-SIBLING: qa/restit_edge_harness.py (kh9pc_stereo repo) carries a
# copy of this detection core for standalone evidence figures — apply
# detector fixes to BOTH until the harness imports from here.

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

STRIP_BINS = 32          # decimated x-bins per strip
X_BINS = 2048            # decimated x-bins for the full pass
ROW_STEP = 8             # y decimation of the profile pass
TEXTURE_MIN = 6.0        # robust cross-column spread threshold


@dataclass
class EdgeFit:
    """Per-side oracle result in GLOBAL mosaic pixel coordinates."""

    coeffs: np.ndarray       # np.polyval poly2 coefficients, edge row vs col
    valid_frac: float        # strips passing anatomy validation
    support_frac: float      # fit inliers among valid strips
    n_strips: int
    line_frac: float         # strips where the optional edge line was seen

    def predict(self, x):
        """sklearn-model-compatible: accepts (N,1) or (N,) column coords."""
        x = np.asarray(x, float).ravel()
        return np.polyval(self.coeffs, x)

    @property
    def passed(self) -> bool:
        # thresholds MATCH the review-accepted harness PASS verdict
        # (audit H-3): WARN-grade fits must not silently feed the warp
        return self.valid_frac >= 0.8 and self.support_frac >= 0.7


def _strip_profiles(src):
    W, H = src.width, src.height
    n_rows = max(1, H // ROW_STEP)
    band = src.read(1, out_shape=(n_rows, X_BINS),
                    resampling=Resampling.average).astype(np.float32)
    rows = (np.arange(band.shape[0]) * (H / band.shape[0])).astype(int)
    n_strips = X_BINS // STRIP_BINS
    xc = ((np.arange(n_strips) + 0.5) * STRIP_BINS / X_BINS * W).astype(int)
    med = np.empty((n_strips, band.shape[0]), np.float32)
    spr = np.empty_like(med)
    for i in range(n_strips):
        s = band[:, i * STRIP_BINS:(i + 1) * STRIP_BINS]
        med[i] = np.median(s, axis=1)
        q75, q25 = np.percentile(s, [75, 25], axis=1)
        spr[i] = (q75 - q25) / 1.349
    return W, H, rows, xc, med, spr


def _strip_detect(rows, med, spr):
    """v4.2 detection for one strip: edge mandatory, line optional."""
    textured = spr > TEXTURE_MIN
    k = np.ones(5)
    textured = np.convolve(textured.astype(float), k / k.sum(), "same") > 0.5
    n = len(textured)
    c0, c1 = int(0.30 * n), int(0.70 * n)
    m_c = float(np.median(med[c0:c1]))
    s_c = max(3.0, 1.4826 * float(np.median(np.abs(med[c0:c1] - m_c))))

    def scan_outward(seq):
        spike_hi, spike_lo = m_c + 3 * s_c, m_c + 1.5 * s_c
        dark_thr = m_c - 2.5 * s_c
        i, nn = 0, len(seq)
        while i < nn:
            if seq[i] > spike_hi:
                j = i
                while j < nn and seq[j] > spike_lo:
                    j += 1
                if (j - i) * ROW_STEP <= 300:
                    kk, dark = j, 0
                    while kk < nn and (kk - j) * ROW_STEP < 800:
                        if seq[kk] < dark_thr:
                            dark += 1
                        kk += 1
                    if dark * ROW_STEP >= 100:
                        pk = i + int(np.argmax(seq[i:j]))
                        e = j
                        while e < nn and seq[e] >= dark_thr:
                            e += 1
                        return pk, min(e, nn - 1)
                i = j
            elif seq[i] < dark_thr:
                kk, dark = i, 0
                while kk < nn and (kk - i) * ROW_STEP < 800:
                    if seq[kk] < dark_thr:
                        dark += 1
                    kk += 1
                if dark * ROW_STEP >= 300:
                    return None, i
                i += 1                       # audit M-4: never skip spikes
            else:
                i += 1
        return None

    out = {}
    for side, start in (("top", c0), ("bottom", c1)):
        seq = med[start::-1] if side == "top" else med[start:]
        hit = scan_outward(seq)
        if hit is None:
            out[side] = (None, None, False)
            continue
        pk_rel, e_rel = hit
        if side == "top":
            e_i = max(0, start - e_rel)
            l_i = None if pk_rel is None else start - pk_rel
        else:
            e_i = min(n - 1, start + e_rel)
            l_i = None if pk_rel is None else start + pk_rel
        anchor = l_i if l_i is not None else e_i
        vin = (textured[anchor:min(n, anchor + int(1200 / ROW_STEP))].mean()
               if side == "top" else
               textured[max(0, anchor - int(1200 / ROW_STEP)):anchor].mean())
        # Audit finding M-5 (outward-must-be-untextured) was REVERTED
        # 2026-08-23 02:2x: on real post-C5 mosaics the outward side of the
        # frame edge legitimately carries marks/titling that read as
        # textured, and the check rejected ~40% of GOOD strips (oracle gate
        # F007/A011, support 88-97% with valid ~50-67%). Inward-textured
        # validation is the review-accepted v4.2 behavior.
        out[side] = (int(rows[e_i]),
                     None if l_i is None else int(rows[l_i]),
                     bool(vin > 0.7))
    return out


def _robust_poly(x, y, deg=2, iters=3):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(y)
    c = None
    for _ in range(iters):
        if keep.sum() < deg + 2:
            break
        c = np.polyfit(x[keep], y[keep], deg)
        r = y - np.polyval(c, x)
        mad = max(np.median(np.abs(r[keep])), 5.0)
        keep = np.isfinite(y) & (np.abs(r) < 4 * mad)
    return c, keep


def fit_format_edges(raster_filepath: str | Path) -> dict[str, EdgeFit]:
    """Oracle entry point: per-side EdgeFit in global mosaic coords."""
    with rasterio.open(raster_filepath) as src:
        W, H, rows, xc, med, spr = _strip_profiles(src)
    picks = {"top": [], "bottom": []}
    valid = {"top": [], "bottom": []}
    lines = {"top": 0, "bottom": 0}
    for i in range(len(xc)):
        det = _strip_detect(rows, med[i], spr[i])
        # SNR-adaptive 3-strip retry (audit H-4: review-accepted harness
        # behavior; without it marginal strips flip to the legacy fallback)
        if any((e is None or not v) for e, l, v in det.values()) \
                and 0 < i < len(xc) - 1:
            det3 = _strip_detect(rows, med[i - 1:i + 2].mean(axis=0),
                                 spr[i - 1:i + 2].mean(axis=0))
            for side in det:
                e, l, v = det[side]
                e3, l3, v3 = det3.get(side, (None, None, False))
                if (e is None or not v) and e3 is not None and v3:
                    det[side] = (e3, l3, v3)
        for side, (e, l, v) in det.items():
            picks[side].append(np.nan if e is None else e)
            valid[side].append(v)
            if l is not None:
                lines[side] += 1
    out = {}
    for side in ("top", "bottom"):
        y = np.array(picks[side], float)
        v = np.array(valid[side], bool)
        yv = np.where(v, y, np.nan)
        c, keep = _robust_poly(np.asarray(xc, float), yv)
        if c is None:
            continue
        out[side] = EdgeFit(
            coeffs=c,
            valid_frac=float(v.mean()) if v.size else 0.0,
            support_frac=float(keep.sum() / max(1, v.sum())),
            n_strips=len(xc),
            line_frac=lines[side] / max(1, len(xc)),
        )
    return out
