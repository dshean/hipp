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
STRIP_PX = 2048          # FIXED physical strip width (dshean 2026-08-23:
                         # fixed strip COUNTS scaled with scan sector --
                         # 1.8k px strips on 30-deg vs 5.3k on 90-deg)
ROW_STEP = 8             # y decimation of the profile pass
TEXTURE_MIN = 6.0        # robust cross-column spread threshold


@dataclass
class EdgeFit:
    """Per-side oracle result in GLOBAL mosaic pixel coordinates."""

    coeffs: np.ndarray       # np.polyval poly2 coefficients, edge row vs col
    valid_frac: float        # strips passing anatomy validation, of CONTENT strips
    support_frac: float      # fit inliers among valid strips
    n_strips: int
    line_frac: float         # content strips where the optional edge line was seen
    content_frac: float = 1.0  # strips inside the contiguous content span

    def predict(self, x):
        """sklearn-model-compatible: accepts (N,1) or (N,) column coords."""
        x = np.asarray(x, float).ravel()
        return np.polyval(self.coeffs, x)

    @property
    def passed(self) -> bool:
        # Calibrated 2026-08-23 against dshean's full-block visual review of
        # all 48 casa_block sides (min approved valid=53%, support=80%): the
        # line-anchored detector demotes inside-line picks, so valid_frac
        # runs lower by design while support_frac carries fit quality.
        # WARN-grade fits must not silently feed the warp (audit H-3).
        return self.valid_frac >= 0.5 and self.support_frac >= 0.8


def _strip_profiles(src):
    W, H = src.width, src.height
    n_rows = max(1, H // ROW_STEP)
    n_strips = max(8, W // STRIP_PX)
    x_bins = n_strips * STRIP_BINS
    band = src.read(1, out_shape=(n_rows, x_bins),
                    resampling=Resampling.average).astype(np.float32)
    rows = (np.arange(band.shape[0]) * (H / band.shape[0])).astype(int)
    xc = ((np.arange(n_strips) + 0.5) * STRIP_BINS / x_bins * W).astype(int)
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

    def scan_outward(seq, tex):
        spike_hi, spike_lo = m_c + 3 * s_c, m_c + 1.5 * s_c
        dark_thr = m_c - 2.5 * s_c
        i, nn = 0, len(seq)
        while i < nn:
            if seq[i] > spike_hi:
                j = i
                while j < nn and seq[j] > spike_lo:
                    j += 1
                if (j - i) * ROW_STEP <= 300:
                    # CONSECUTIVE dark run (2026-08-23, fix-sibling of the
                    # harness 09:4x change)
                    kk, dark, best = j, 0, 0
                    while kk < nn and (kk - j) * ROW_STEP < 800:
                        dark = dark + 1 if seq[kk] < dark_thr else 0
                        best = max(best, dark)
                        kk += 1
                    beyond_ok = True
                    bey = tex[min(nn, kk):min(nn, kk + int(4000 / ROW_STEP))]
                    if bey.size and bey.mean() > 0.25:
                        beyond_ok = False
                    if best * ROW_STEP >= 100 and beyond_ok:
                        pk = i + int(np.argmax(seq[i:j]))
                        e = j
                        while e < nn and seq[e] >= dark_thr:
                            e += 1
                        return pk, min(e, nn - 1)
                i = j
            elif seq[i] < dark_thr:
                kk, dark, best = i, 0, 0
                while kk < nn and (kk - i) * ROW_STEP < 800:
                    dark = dark + 1 if seq[kk] < dark_thr else 0
                    best = max(best, dark)
                    kk += 1
                if best * ROW_STEP >= 300:
                    # TERMINAL check: content resuming beyond the run means
                    # it was shadow/water, not the border (fix-sibling)
                    beyond = tex[min(nn, kk):min(nn, kk + int(4000 / ROW_STEP))]
                    if beyond.size and beyond.mean() > 0.25:
                        i = kk
                        continue
                    return None, i
                i += 1                       # audit M-4: never skip spikes
            else:
                i += 1
        return None

    out = {}
    for side, start in (("top", c0), ("bottom", c1)):
        seq = med[start::-1] if side == "top" else med[start:]
        tex = textured[start::-1] if side == "top" else textured[start:]
        hit = scan_outward(seq, tex)
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


def _content_span(content, max_gap=2):
    """Longest contiguous run of content strips, bridging gaps <= max_gap.

    dshean 2026-08-23 ruling (ops196 narrow-sector): strips outside this
    span are empty canvas fill -- or a slice of the NEXT frame if the
    scan was cut badly -- and must neither dilute valid_frac's
    denominator nor feed the fits. Contiguity is what excludes a foreign
    frame slice: it sits beyond a black gap, outside the longest run.
    """
    idx = np.flatnonzero(content)
    if idx.size == 0:
        return np.ones_like(content, bool)   # degrade to no exclusion
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p <= max_gap + 1:
            p = i
        else:
            runs.append((s, p))
            s = p = i
    runs.append((s, p))
    s, p = max(runs, key=lambda r: r[1] - r[0])
    m = np.zeros_like(content, bool)
    m[s:p + 1] = True
    return m


def _coverage_deg(x, y, extent_px):
    """Fit degree earned by x-coverage (dshean 2026-08-23, ops196 A001:
    picks clustered in 15% of the frame let a poly2 extrapolate wildly).
    >=50% of the content extent -> 2; >=25% -> 1; else 0 (constant)."""
    xf = np.asarray(x, float)[np.isfinite(np.asarray(y, float))]
    if xf.size < 2 or extent_px <= 0:
        return 0
    cov = (xf.max() - xf.min()) / float(extent_px)
    return 2 if cov >= 0.5 else (1 if cov >= 0.25 else 0)


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
    # content span (dshean 2026-08-23): textured central band marks real
    # film content; film grain clears TEXTURE_MIN, digital canvas fill is
    # uniform. Strips outside the longest contiguous run are excluded.
    n_band = med.shape[1]
    cb0, cb1 = int(0.30 * n_band), int(0.70 * n_band)
    span = _content_span((spr[:, cb0:cb1] > TEXTURE_MIN).mean(axis=1) > 0.2)
    picks = {"top": [], "bottom": []}
    valid = {"top": [], "bottom": []}
    lines_y = {"top": [], "bottom": []}
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
            lines_y[side].append(np.nan if l is None else l)
    out = {}
    n_span = max(1, int(span.sum()))
    for side in ("top", "bottom"):
        y = np.array(picks[side], float)
        # span exclusion BEFORE fitting: an out-of-span strip (canvas
        # fill or a foreign frame slice) must not bend the fit either
        v = np.array(valid[side], bool) & span
        yv = np.where(v, y, np.nan)
        xs = np.asarray(xc, float)[span]
        extent_px = float(xs.max() - xs.min()) if xs.size >= 2 else 0.0
        c, keep = _robust_poly(np.asarray(xc, float), yv,
                               deg=_coverage_deg(xc, yv, extent_px))
        if c is None:
            continue
        # LINE-ANCHORED EDGE (David 2026-08-23, fix-sibling of the harness
        # change): where the line fit is solid, edge curve = line fit +
        # robust median per-strip offset; outlier edge picks cannot bend it
        yl = np.array(lines_y[side], float)
        ylv = np.where(v, yl, np.nan)
        cl, _ = _robust_poly(np.asarray(xc, float), ylv,
                             deg=_coverage_deg(xc, ylv, extent_px))
        if cl is not None:
            # inside-the-line edge picks are anatomically impossible ->
            # suspect: excluded and their strips demoted (fix-sibling)
            yl_fit = np.polyval(cl, np.asarray(xc, float))
            tol = 40.0
            inside = (y > yl_fit + tol) if side == "top" else (y < yl_fit - tol)
            v = v & ~(np.isfinite(y) & inside)
            both = v & np.isfinite(yl) & np.isfinite(y)
            if both.sum() >= 6:
                xb = np.asarray(xc, float)[both]
                offs = y[both] - np.polyval(cl, xb)
                d = float(np.median(offs))
                mad = float(np.median(np.abs(offs - d)))
                ce = cl.copy()
                ce[-1] += d
                resid = y - np.polyval(ce, np.asarray(xc, float))
                c = ce
                keep = np.isfinite(y) & (np.abs(resid) < max(4 * mad, 60))
        yl_arr = np.array(lines_y[side], float)
        out[side] = EdgeFit(
            coeffs=c,
            valid_frac=float(v.sum() / n_span),
            support_frac=float((keep & v).sum() / max(1, v.sum())),
            n_strips=len(xc),
            line_frac=float((np.isfinite(yl_arr) & span).sum() / n_span),
            content_frac=float(span.sum() / max(1, len(xc))),
        )
    return out
