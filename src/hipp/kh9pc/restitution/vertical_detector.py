"""
Copyright (c) 2026 HIPP developers
Description: VerticalDetector — detects the left and right film frame edges of a KH-9 PC
    scan. For each side, a downsampled 1-row intensity profile is searched for the longest
    constant-minimum plateau (the film background/leader), then the strongest intensity
    gradient just after that plateau's end is taken as the edge. Used as the first step by
    all restitution strategies.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import os
import numpy as np
from numpy.typing import NDArray
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from hipp.image import SubImage
from hipp.kh9pc.kh9_image_spec import KH9ImageSpec
from hipp.kh9pc.restitution.base import FittingClass, DetectionError


logger = logging.getLogger(__name__)


@dataclass
class VerticalEdgeResult:
    """Detected edge: global position, local edge index, sub-image, and intensity profile."""

    position: int
    edge_local: int
    gradient_ratio: float
    sub_image: SubImage
    profile: NDArray[np.floating]


@dataclass
class VerticalDetector(FittingClass):
    """Detects the left and right film frame edges from a KH-9 PC raster."""

    vertical_padding: float = 0.25
    search_window_width: int = 10000
    downsample_scale: float = 0.01
    gradient_ratio_threshold: float = 0.3

    def __post_init__(self) -> None:
        """Initialise fitted-attribute slots."""
        super().__init__()
        self._results: dict[str, VerticalEdgeResult] = {}
        self._failed: bool = False

    @property
    def is_failed(self) -> bool:
        """True if the last fit() call failed to detect one or both edges."""
        return self._failed

    @property
    def left_(self) -> VerticalEdgeResult:
        """Detected left edge. Raises if fit() has not been called or failed."""
        if "left" not in self._results:
            raise RuntimeError("left edge not available — call fit() first")
        return self._results["left"]

    @property
    def right_(self) -> VerticalEdgeResult:
        """Detected right edge. Raises if fit() has not been called or failed."""
        if "right" not in self._results:
            raise RuntimeError("right edge not available — call fit() first")
        return self._results["right"]

    @property
    def edges_(self) -> tuple[int, int]:
        """(left_position, right_position) in full-raster pixel coordinates."""
        return self.left_.position, self.right_.position

    @property
    def detected_width_(self) -> int:
        """Width between detected edges in full-raster pixels."""
        return self.right_.position - self.left_.position

    # dshean 2026-08-25 (tonight): MIDDLE-OUT exposure-edge detection. The old
    # edge-in search (longest dark plateau from the raster edge, first strong
    # gradient after it) locks on the film's physical start / nodata boundary and
    # then finds the "right edge" wherever a gradient sits near left+expected --
    # fresh F013 mosaic: left=302, right=337649, width 337347 for a 342247 canvas.
    # Now: from the frame centre walk OUTWARD along the interior column profile
    # (median DN + texture) until a sustained run of UNEXPOSED film begins; the
    # exposed area ends there. The next frame's exposure beyond that gap is never
    # reached, and near-black frames are caught by the sweep-width cross-check:
    # both edges must be expected_width x (7/pitch) apart (tol width_tol); if only
    # one edge is trustworthy, the other is derived from it and the width.
    scan_pitch_um: tuple[float, float] | None = None
    width_tol: float = 0.008          # hard gate for a pair (fraction), OR width_tol_px, whichever is larger
    width_tol_px: int = 1500          # 2026-08-26: sweep variation is ~+-1000 px on BOTH tiers (90-deg +-0.3 %,
    width_sigma: float = 0.002        #   30-deg +-1 %: ops327 A004 +0.67 %, ops196 A004 +0.95 %), so absolute px
    width_sigma_px: int = 700         # Gaussian width prior for ranking pairs: max(fraction, px)
    edge_margin_px: int = 64          # 2026-08-26 (dshean: A002/F002 exposures start at 200-600 px): only the very edge is excluded; the width prior handles the black-margin|film step
    boundary_suspect_px: int = 600    # single-edge fallback: a step this close to the raster edge is suspect (scan cut through film)
    boundary_factor: float = 2.0      # ... unless it is this much stronger than every interior candidate
    flank_gap_px: int = 192           # candidate validation: skip this much beside the step ...
    flank_width_px: int = 2048        # ... then measure a band this wide on each side
    exposed_tex_min: float = 1.0      # inside band: median per-column row-std (DN) floor (dark ocean is ~2)
    pair_ratio: float = 1.5           # pair eligibility: inside texture >= this x outside (mild; the width decides)
    single_ratio: float = 3.0         # single-edge eligibility: strict contrast
    flat_tex_ratio: float = 0.5       # outside band must have <= this fraction of the inside texture
    nodata_frac: float = 0.9          # outside band this much DN 0 = truncation, not an edge
    z_min: float = 6.0                # row-coherent step significance (standard errors)
    candidate_min_sep_px: int = 512   # local-maximum window for step candidates (full-res px)
    tex_step_min: float = 8.0         # texture-profile step (DN of per-column row-std) that makes a candidate
    tex_step_weight: float = 1.0      # its strength on the z scale for pair ranking

    def _refine_edge(self, src: rasterio.DatasetReader, coarse: int, side: str, sign: float | None = None) -> int:
        """Full-resolution refinement: strongest gradient of the coarse step's SIGN within
        +-256 px (v10.2, audit r2: the old left->argmax / right->argmin prior assumed dark
        outside / bright inside, which the pair search dropped -- F002's right edge steps
        UP into brighter film base). sign None falls back to the side prior."""
        r0 = int(self.vertical_padding * src.height); h = int(src.height * (1 - 2 * self.vertical_padding))
        c0 = max(0, coarse - 256); w = min(512, src.width - c0)
        band = SubImage(src, window=Window(c0, r0, w, h), out_shape=(1, 1, w)).band.flatten().astype(np.float32)
        g = np.gradient(band)
        # v10.2b (2026-08-26): using the coarse step's sign (audit r2) was tried and REVERTED --
        # F026's left edge (grey unexposed base) refined 497 px inward and failed the golden
        # while no other frame improved; the side prior (dark outside -> bright inside) stays.
        # `sign` is accepted and ignored so the callers keep passing it for the record.
        up = side == "left"
        k = int(np.argmax(g)) if up else int(np.argmin(g))
        return c0 + k

    def _coherent_step_profile(self, src: rasterio.DatasetReader, f: int = 16):
        """Row-coherent horizontal step profile of the whole frame at 1/f (overviews
        make this a sub-second read; without them GDAL decodes every tile once).

        z[c] = mean over rows of the x-gradient at decimated column c, divided by
        its standard error: scene content has gradients of random sign row to row
        (mean ~ 0), a sweep boundary steps every row the same way (|z| large even
        for a 15-DN step against dark ocean). Columns where either side is nodata
        (DN 0) in most rows are masked -- the nodata boundary is not an exposure
        edge. Returns (z, valid_fraction) per decimated column."""
        H, W = int(src.height), int(src.width)
        r0 = int(self.vertical_padding * H); h = int(H * (1 - 2 * self.vertical_padding))
        a = src.read(1, window=Window(0, r0, W, h), out_shape=(max(8, h // f), max(8, W // f)),
                     resampling=Resampling.average).astype(np.float32)
        # NODATA is geometric, not a DN (2026-08-26): unexposed black film still carries the
        # margin marks / format edge, so a column is nodata only if it has NO signal over the
        # full height -- interior band AND the top/bottom margins (read at 1/f, averaged: a nodata column is exactly 0)
        mrows = max(64, int(0.10 * H))       # 10 %: the format line / collimation line sits ~1000-1400 rows in (A001: 4 % missed it)
        top = src.read(1, window=Window(0, 0, W, mrows), out_shape=(max(4, mrows // f), max(8, W // f)), resampling=Resampling.average)
        bot = src.read(1, window=Window(0, H - mrows, W, mrows), out_shape=(max(4, mrows // f), max(8, W // f)), resampling=Resampling.average)
        colmax = np.maximum.reduce([a.max(axis=0), top.max(axis=0).astype(np.float32), bot.max(axis=0).astype(np.float32)])
        self._nodata_col_ = colmax <= 0.5
        self._coltex_ = a.std(axis=0)
        # DN 0 is DATA here: unexposed film scans black on some sessions (ops323 A001's
        # right edge is a 30 -> 0 step) -- masking zeros hid it. The scanner boundary
        # (black margin -> film) is excluded by position instead (edge_margin_px).
        gx = np.diff(a, axis=1)
        n = float(a.shape[0])
        mean = gx.mean(axis=0)
        se = gx.std(axis=0, ddof=1) / np.sqrt(n) + 1e-3
        self._dec_ = a                      # kept for the flanking-band validation of candidates
        return mean / se, np.ones(gx.shape[1], dtype=np.float32)

    def _flank(self, c: int, side: str, f: int, to_boundary: bool = False):
        """Texture / DN / nodata fraction of a band next to decimated column boundary c
        (side 'left' = columns < c, 'right' = columns >= c), skipping flank_gap_px.
        to_boundary=True extends the band to the raster edge and reports the 90th
        percentile of per-column texture (a true edge has NO exposure anywhere outside)."""
        a = self._dec_
        g = max(1, self.flank_gap_px // f); w = max(2, self.flank_width_px // f)
        if side == "left":
            lo, hi = (0 if to_boundary else max(0, c + 1 - g - w)), max(0, c + 1 - g)
        else:
            lo, hi = min(a.shape[1], c + 1 + g), (a.shape[1] if to_boundary else min(a.shape[1], c + 1 + g + w))
        band = a[:, lo:hi]
        if band.size == 0 or band.shape[1] < 2:
            return dict(tex=0.0, tex90=0.0, dn=0.0, nodata=1.0, n=0)
        nod = self._nodata_col_[lo:hi]
        tex = band.std(axis=0)[~nod] if (~nod).any() else np.zeros(1)
        return dict(tex=float(np.median(tex)), tex90=float(np.percentile(tex, 90)), dn=float(np.median(band)),
                    nodata=float(nod.mean()), n=int(band.shape[1]))

    def _validate_candidates(self, peaks, z, f: int):
        """dshean 2026-08-26: an exposure edge has valid exposed area on its INSIDE and
        none on its OUTSIDE; a step next to nodata is a truncation (the frame edge is
        missing -> column 0 / W), never an edge. Raster-end cuts are the loudest steps in
        a frame (F002 base->nodata z -724, F026 bed->border z +161) and out-scored the
        true edges (18/232, 39/21) under any strength ranking. Returns per candidate:
        valid_as_left, valid_as_right, truncation_left, truncation_right, and the flank stats."""
        out = []
        for pi in peaks:
            L = self._flank(int(pi), "left", f); R = self._flank(int(pi), "right", f)
            Lb = self._flank(int(pi), "left", f, to_boundary=True); Rb = self._flank(int(pi), "right", f, to_boundary=True)
            trunc_l = L["nodata"] >= self.nodata_frac            # geometric nodata on the left of the step
            trunc_r = R["nodata"] >= self.nodata_frac
            # exposed = textured RELATIVE to the other side (A003's right edge: dark ocean tex
            # 2.0 inside vs 0.1 base outside; an absolute 2.5 floor rejected it), with a low
            # absolute floor against pure noise
            # PAIR eligibility is MILD (the sweep-width constraint discriminates; F026's washed-out
            # exposure over a grainy base is only ~2x its outside): inside textured >= pair_ratio x
            # the outside, outside not exposed-like; a nodata-adjacent step may still pair (F002's
            # exposure starts 192 px from the canvas edge)
            def rel(inn, out, ratio):
                return inn["tex"] >= self.exposed_tex_min and inn["tex"] >= ratio * max(out["tex"], 1e-3) and inn["nodata"] < 0.5
            pair_left = rel(R, L, self.pair_ratio) or (trunc_l and R["tex"] >= self.exposed_tex_min)
            pair_right = rel(L, R, self.pair_ratio) or (trunc_r and L["tex"] >= self.exposed_tex_min)
            # SINGLE-edge eligibility is STRICT: strong contrast (single_ratio), no exposure anywhere
            # outside up to the raster boundary (A001's coastline step at 196 k had land beyond it),
            # not nodata-adjacent (truncation: the frame edge is missing -> column 0 / W)
            single_left = rel(R, L, self.single_ratio) and not trunc_l and Lb["tex90"] < max(self.exposed_tex_min, self.flat_tex_ratio * R["tex"])
            single_right = rel(L, R, self.single_ratio) and not trunc_r and Rb["tex90"] < max(self.exposed_tex_min, self.flat_tex_ratio * L["tex"])
            exposed_r, exposed_l = rel(R, L, 1.0), rel(L, R, 1.0)
            out.append(dict(x=int((pi + 1) * f), z=float(z[pi]), L=L, R=R, Lb=Lb, Rb=Rb,
                            as_left=bool(pair_left), as_right=bool(pair_right),
                            single_left=bool(single_left), single_right=bool(single_right),
                            trunc_left=bool(trunc_l and exposed_r), trunc_right=bool(trunc_r and exposed_l)))
        return out

    def _texture_candidates(self, peaks, az, win: int, m: int):
        """Second-chance candidate pool from windowed texture steps (unpicklable-closure
        bug 2026-08-26: joblib.dump(strategy) failed while this lived as a stored closure).
        Mutates az in place so texture candidates carry a rankable strength."""
        texs = self._coltex_.astype(np.float64)
        cs = np.concatenate([[0.0], np.cumsum(texs)])
        wm = int(win)
        c_ = np.arange(wm, texs.size - wm)
        dtex = np.zeros(texs.size - 1)
        dtex[c_ - 1] = np.abs((cs[c_ + wm] - cs[c_]) / wm - (cs[c_] - cs[c_ - wm]) / wm)
        from scipy.ndimage import maximum_filter1d as _mf
        tpk = np.flatnonzero((dtex >= self.tex_step_min) & (dtex == _mf(dtex, size=2 * win + 1)))
        tpk = tpk[(tpk + 1 >= m) & (tpk + 1 <= dtex.size - m)]
        added = []
        for q in tpk:
            if peaks.size == 0 or np.abs(peaks - q).min() > win:
                added.append(int(q))
                az[q] = max(az[q], float(dtex[q]) * self.tex_step_weight)
        return np.array(sorted(set(list(peaks) + added)), dtype=int)

    def _fit(self, raster_filepath: Path) -> "VerticalDetector":
        """Exposure edges = the PAIR of row-coherent DN steps the expected sweep width
        apart (dshean 2026-08-25). The two earlier criteria were wrong, not their
        direction: edge-in locked the nodata / scanner boundary (F013 left=302),
        middle-out assumed unexposed film is dark -- ops323 F026's unexposed base is
        DN ~55 under a DN ~75 exposure (over-ran both edges by 3.3 k px) and dark
        ocean reads as unexposed (iceland A025). A sweep boundary is a step that
        every row shares; the pair constraint (expected width x 7/pitch, width_tol)
        rejects scanner furniture, the USGS logo and content edges. A truncated
        scan (block-end frames: ops323 A001/F001) has one edge only: the strongest
        single step + expected width, clamped to the raster."""
        self._failed = False
        self._results = {}
        self.edge_source_ = None
        self.crop_center_ = None
        self.multiframe_ = False
        self.short_scan_ = False
        self.true_edges_ = None
        self._tex_retry_ = False
        image_spec = KH9ImageSpec.from_raster_filepath(raster_filepath)
        expected = float(image_spec.expected_size[0])
        if self.scan_pitch_um and self.scan_pitch_um[0] > 0:
            expected *= 7.0 / float(self.scan_pitch_um[0])     # canvas px (7 um) -> this scan's px
        f = 16
        with rasterio.open(raster_filepath) as src:
            try:
                z, vf = self._coherent_step_profile(src, f)
                W = int(src.width)
                # candidates: local |z| maxima above z_min, at least min_sep apart
                az = np.abs(z)
                from scipy.ndimage import maximum_filter1d
                win = max(3, int(self.candidate_min_sep_px / f))
                peaks = np.flatnonzero((az >= self.z_min) & (az == maximum_filter1d(az, size=2 * win + 1)))
                # never the scanner boundary: a step within edge_margin_px of the raster edge
                # is the black margin meeting the film (A010 x=314, F026 x=366, A001 x=372,
                # F001's film end at W-406), not an exposure edge -- a truncated scan's
                # exposure runs INTO the boundary and has no edge there to find
                m = int(self.edge_margin_px / f)
                peaks = peaks[(peaks + 1 >= m) & (peaks + 1 <= az.size - m)]
                # texture-onset candidates (2026-08-26, ops327 A005: a DARK exposure -- median
                # DN 1-8 -- starts with texture 18-27 but a DN-step z of only 5): a coherent
                # step in the per-column texture profile marks an exposure edge the DN misses
                # v10 (2026-08-26): texture-onset candidates are a SECOND-CHANCE pool only --
                # merging them up front shifted two golden frames (A002 801->1234, ops196 A004
                # 3781->4643: near-edge texture steps won on width fit). DN candidates resolve
                # the frame first; only an unresolved frame (ops327 A005: a dark exposure whose
                # onset ramps over ~2000 px, DN-step z 5) retries with the texture pool merged.
                if peaks.size == 0:
                    peaks = self._texture_candidates(peaks, az, win, m)
                if peaks.size == 0:
                    raise DetectionError("no row-coherent DN step anywhere in the frame")
                x_peak = (peaks + 1.0) * f            # diff index c -> boundary between columns c and c+1
                tol = max(self.width_tol * expected, float(self.width_tol_px))
                sigma = max(self.width_sigma * expected, float(self.width_sigma_px))
                # pairs (left, right) whose separation matches the sweep; score = |z_l| + |z_r|
                # (+ 25 % when the signs are the natural dark->exposed->dark pattern)
                val = self._validate_candidates(peaks, z, f)
                if os.environ.get("KH9_EDGE_DEBUG"):
                    for v in sorted(val, key=lambda v: -abs(v["z"]))[:24]:
                        logger.info("[VerticalDetector] cand x=%7d z=%+7.1f | L tex %5.1f dn %5.1f nod %.2f t90b %5.1f | R tex %5.1f dn %5.1f nod %.2f t90b %5.1f | pairL=%d pairR=%d singleL=%d singleR=%d truncL=%d truncR=%d",
                                    v["x"], v["z"], v["L"]["tex"], v["L"]["dn"], v["L"]["nodata"], v["Lb"]["tex90"], v["R"]["tex"], v["R"]["dn"], v["R"]["nodata"], v["Rb"]["tex90"],
                                    v["as_left"], v["as_right"], v["single_left"], v["single_right"], v["trunc_left"], v["trunc_right"])
                ok_left = np.array([v["as_left"] for v in val]); ok_right = np.array([v["as_right"] for v in val])
                trunc_left = any(v["trunc_left"] for v in val); trunc_right = any(v["trunc_right"] for v in val)
                best = None
                for i, pi in enumerate(peaks):
                    if not ok_left[i]:
                        continue
                    for k, pk in enumerate(peaks):
                        if not ok_right[k]:
                            continue
                        sep = x_peak[k] - x_peak[i]
                        if sep <= 0 or abs(sep - expected) > tol:
                            continue
                        # rank by width agreement first (measured sweeps sit within +-0.3 % of
                        # expected: Gaussian prior, sigma = width_sigma), strength second --
                        # cg A010: the scanner boundary at x=314 (z 288, +0.60 %) must lose to
                        # the true edge at 1459 (+0.26 %); ops323 F001: the film end (-0.71 %) too
                        sc = (az[pi] + az[pk]) * float(np.exp(-0.5 * ((sep - expected) / sigma) ** 2))
                        # no sign prior (2026-08-26, ops323 F002): beyond the sweep the film base can
                        # be brighter than a dark exposure end (F026, F002 right +232), so a
                        # dark->bright->dark bonus tipped F002 to a next-frame cut at the raster end
                        if best is None or sc > best[0]:
                            best = (sc, i, k)
                strength = {}; sign_l = sign_r = None
                if best is not None:
                    _, i, k = best
                    coarse_l, coarse_r = int(x_peak[i]), int(x_peak[k])
                    strength = {"left": float(az[peaks[i]]), "right": float(az[peaks[k]])}
                    sign_l, sign_r = float(np.sign(z[peaks[i]])), float(np.sign(z[peaks[k]]))
                    src_desc = "both"
                else:
                    # no pair: strongest step decides the side by its sign and position
                    # single edge: prefer a candidate away from the raster boundary -- a step
                    # within boundary_suspect_px of the edge is usually the scan cutting through
                    # film (A001 x=384 z 42 vs the true right edge 339152 z 41) unless it is far
                    # stronger (> boundary_factor x) than every interior candidate
                    # no valid pair: the strongest VALID single edge (as a left or a right edge);
                    # a truncation on the other side (exposure running into nodata) is the
                    # "frame edge missing -> column 0 / W" case (dshean 2026-08-26)
                    cand = [(az[peaks[q]], q, "left") for q in range(peaks.size) if val[q]["single_left"]] + \
                           [(az[peaks[q]], q, "right") for q in range(peaks.size) if val[q]["single_right"]]
                    # MULTI-FRAME SCAN (dshean 2026-08-26, ops323 A003: a crop boundary in the
                    # middle of the scan is unacceptable): a valid LEFT and a valid RIGHT edge
                    # separated by MORE than the sweep + tol means the contiguous exposure holds
                    # more than one frame. Report BOTH true edges, flag for review, and anchor
                    # the fixed-width crop on the LEFT exposure edge (film reads left to right).
                    _lefts = [q for _, q, r in cand if r == "left"]
                    _rights = [q for _, q, r in cand if r == "right"]
                    if _lefts and _rights:
                        ql = min(_lefts, key=lambda q: x_peak[q]); qr = max(_rights, key=lambda q: x_peak[q])
                        span = x_peak[qr] - x_peak[ql]
                        if span > expected + tol:
                            left = int(x_peak[ql]); right = int(x_peak[qr])
                            self.multiframe_ = True
                            strength = {"left": float(az[peaks[ql]]), "right": float(az[peaks[qr]])}
                            src_desc = "multiframe-left-anchored"
                            logger.warning("[VerticalDetector] MULTIFRAME_SCAN: contiguous exposure %d..%d (%.0f px = %+.1f %% of the sweep) "
                                           "holds more than one frame -- true edges reported, crop anchored on the LEFT edge; REVIEW",
                                           left, right, span, 100 * (span / expected - 1))
                            coarse_l, coarse_r = left, right
                            left = self._refine_edge(src, coarse_l, "left", float(np.sign(z[peaks[ql]]))) if 0 <= coarse_l < W else coarse_l
                            right = self._refine_edge(src, coarse_r, "right", float(np.sign(z[peaks[qr]]))) if 0 < coarse_r <= W else coarse_r
                            left_c = int(min(max(left, 0), W - 1)); right_c = int(min(max(right, left_c + 1), W))
                            left, right = left_c, right_c
                            self.edge_source_ = src_desc
                            self.crop_center_ = left + expected / 2.0
                            for side, pos in (("left", left), ("right", right)):
                                sub = self._sub_image(src, pos - self.search_window_width // 2, self.search_window_width)
                                prof = sub.band.flatten()
                                self._results[side] = VerticalEdgeResult(position=pos, edge_local=int(sub.to_local_x(pos)) if hasattr(sub, "to_local_x") else 0,
                                                                         gradient_ratio=float(strength[side]), sub_image=sub, profile=prof)
                            self.n_candidates_ = int(peaks.size)
                            logger.info("%s - left=%d, right=%d, detected width=%d (expected=%.0f at this pitch, diff=%+d px, edges from %s, %d coherent steps)",
                                        self.logging_prefix, left, right, right - left, expected, int(right - left - expected), src_desc, peaks.size)
                            logger.info("%s - crop centre %.0f (left-anchored)", self.logging_prefix, self.crop_center_)
                            return self
                    # a strict single sitting against the raster edge is itself boundary-suspect
                    # (A001: x=384 with the scanner strip making its outside look like film) --
                    # when EVERY strict single is that close to a raster edge, rank it together
                    # with the truncation candidates by distance from the edge (v5 tie-break)
                    def _edge_dist(q, role):
                        return float(x_peak[q]) if role == "left" else float(W - x_peak[q])
                    if cand and all(_edge_dist(q, role) <= self.boundary_suspect_px for _, q, role in cand):
                        bnd = [(_edge_dist(q, role), az[peaks[q]], q, role) for _, q, role in cand] + \
                              [(float(x_peak[q]), az[peaks[q]], q, "left") for q in range(peaks.size) if val[q]["trunc_left"]] + \
                              [(float(W - x_peak[q]), az[peaks[q]], q, "right") for q in range(peaks.size) if val[q]["trunc_right"]]
                        d, zed, j, role = max(bnd)
                        logger.warning("[VerticalDetector] every strict single is raster-edge-adjacent -- tie-break by edge distance: %s x=%d (%d px from the edge, z %.1f)",
                                       role, int(x_peak[j]), int(d), zed)
                        cand = [(zed, j, role)]
                    if not cand:
                        # BOTH sides run into a scan boundary (ops323 A001: exposure from the raster's
                        # left edge to the last section's end at 339152, 1.2 % short of a sweep).
                        # dshean 2026-08-26: the frame end is the boundary the operator STOPPED at --
                        # a section end well inside the canvas -- not the raster's x=0 where scanning
                        # merely began; rank boundary candidates by their distance from the raster edge
                        bnd = [(float(x_peak[q]), q, "left") for q in range(peaks.size) if val[q]["trunc_left"]] + \
                              [(float(W - x_peak[q]), q, "right") for q in range(peaks.size) if val[q]["trunc_right"]]
                        if not bnd and not getattr(self, "_tex_retry_", False):
                            newpeaks = self._texture_candidates(peaks, az, win, m)
                            if newpeaks.size > peaks.size:
                                self._tex_retry_ = True
                                logger.warning("[VerticalDetector] unresolved with DN candidates -- retrying with %d texture-onset candidates",
                                               newpeaks.size - peaks.size)
                                peaks = newpeaks
                                x_peak = (peaks + 1.0) * f
                                val = self._validate_candidates(peaks, z, f)
                                ok_left = np.array([v["as_left"] for v in val]); ok_right = np.array([v["as_right"] for v in val])
                                best = None
                                for i, pi in enumerate(peaks):
                                    if not ok_left[i]:
                                        continue
                                    for k, pk in enumerate(peaks):
                                        if not ok_right[k]:
                                            continue
                                        sep = x_peak[k] - x_peak[i]
                                        if sep <= 0 or abs(sep - expected) > tol:
                                            continue
                                        sc = (az[pi] + az[pk]) * float(np.exp(-0.5 * ((sep - expected) / sigma) ** 2))
                                        if best is None or sc > best[0]:
                                            best = (sc, i, k)
                                if best is not None:
                                    _, i, k = best
                                    coarse_l, coarse_r = int(x_peak[i]), int(x_peak[k])
                                    strength = {"left": float(az[peaks[i]]), "right": float(az[peaks[k]])}
                                    src_desc = "both"
                                    left = self._refine_edge(src, coarse_l, "left", float(np.sign(z[peaks[i]])))
                                    right = self._refine_edge(src, coarse_r, "right", float(np.sign(z[peaks[k]])))
                                    if abs((right - left) - expected) > tol:
                                        left, right = coarse_l, coarse_r
                                    self.crop_center_ = (left + right) / 2.0        # v10.2: the measured sweep's midpoint
                                    left = int(min(max(left, 0), W - 1)); right = int(min(max(right, left + 1), W))
                                    self.edge_source_ = src_desc
                                    for side, pos in (("left", left), ("right", right)):
                                        sub = self._sub_image(src, pos - self.search_window_width // 2, self.search_window_width)
                                        prof = sub.band.flatten()
                                        self._results[side] = VerticalEdgeResult(position=pos, edge_local=int(sub.to_local_x(pos)) if hasattr(sub, "to_local_x") else 0,
                                                                                 gradient_ratio=float(strength[side]), sub_image=sub, profile=prof)
                                    self.n_candidates_ = int(peaks.size)
                                    logger.info("%s - left=%d, right=%d, detected width=%d (expected=%.0f at this pitch, diff=%+d px, edges from %s [texture retry], %d coherent steps)",
                                                self.logging_prefix, left, right, right - left, expected, int(right - left - expected), src_desc, peaks.size)
                                    return self
                        if not bnd:
                            raise DetectionError("no coherent step qualifies as an exposure edge (exposed inside / flat outside)")
                        d, j, role = max(bnd)
                        logger.warning("[VerticalDetector] both sides run into a scan boundary -- taking the %s boundary %d px from the raster edge as the frame end",
                                       role, int(d))
                        cand = [(az[peaks[j]], j, role)]
                    _, j, role = max(cand)
                    pj = peaks[j]; xj = int(x_peak[j])
                    if role == "left": sign_l = float(np.sign(z[pj]))
                    else: sign_r = float(np.sign(z[pj]))
                    logger.warning("[VerticalDetector] no valid pair -- single %s edge x=%d (z %.1f); truncation left=%s right=%s",
                                   role, xj, az[pj], trunc_left, trunc_right)
                    # the side is decided by POSITION (2026-08-26, ops323 F002: the strongest step
                    # sits in the right half with a positive sign -- beyond the sweep the film can
                    # be brighter than a dark exposure end, cf. F026's grey base); the sign is logged
                    if role == "left":
                        coarse_l, coarse_r = xj, xj + int(round(expected)); src_desc = "left+expected"
                        strength = {"left": float(az[pj]), "right": 0.0}
                    else:
                        coarse_l, coarse_r = xj - int(round(expected)), xj; src_desc = "right+expected"
                        strength = {"left": 0.0, "right": float(az[pj])}
                    # SHORT_SCAN reporting (2026-08-26, A001: the crop anchors on one edge but the
                    # TRUE opposite exposure edge -- e.g. 384 -- must be reported, never clamped away)
                    _opp = [q for q in range(peaks.size)
                            if (val[q]["as_right"] and role == "left" and x_peak[q] > xj)
                            or (val[q]["as_left"] and role == "right" and x_peak[q] < xj)]
                    if _opp:
                        qo = max(_opp, key=lambda q: az[peaks[q]])
                        span = abs(x_peak[qo] - xj)
                        if span < expected - tol:
                            self.short_scan_ = True
                            src_desc += "|short-scan"
                            self.true_edges_ = (int(min(xj, x_peak[qo])), int(max(xj, x_peak[qo])))
                            logger.warning("[VerticalDetector] SHORT_SCAN: true exposure edges %d..%d (%.0f px = %.1f %% of the sweep); crop anchored on the %s edge; REVIEW",
                                           self.true_edges_[0], self.true_edges_[1], span, 100 * span / expected, role)
                    logger.warning("[VerticalDetector] no coherent step pair %.0f+-%.0f px apart -- %s (z=%.1f); "
                                   "block-end truncated scan or a missing edge", expected, tol, src_desc, az[pj])
                meas_l = src_desc == "both" or src_desc.startswith("left")     # which sides were MEASURED (the other is anchor + sweep)
                meas_r = src_desc == "both" or src_desc.startswith("right")
                left = self._refine_edge(src, coarse_l, "left", sign_l) if (meas_l and 0 <= coarse_l < W) else coarse_l
                right = self._refine_edge(src, coarse_r, "right", sign_r) if (meas_r and 0 < coarse_r <= W) else coarse_r
                if src_desc == "both" and abs((right - left) - expected) > tol:
                    # the refinement drifted (a stronger local gradient within +-256 px): keep the coarse pair
                    logger.warning("[VerticalDetector] refinement broke the pair (%d vs %.0f) -- keeping the coarse pair", right - left, expected)
                    left, right = coarse_l, coarse_r
                if not meas_r: right = left + int(round(expected))        # derived side follows the refined anchor
                if not meas_l: left = right - int(round(expected))
                # v10.2 crop centre (unclamped -- it may sit outside the raster): the measured sweep's
                # midpoint for a pair (one width per block => symmetric margins, cx at the sweep
                # centre on every frame); the measured edge + half a sweep for a single edge
                self.crop_center_ = (left + right) / 2.0
                left_c = int(min(max(left, 0), W - 1)); right_c = int(min(max(right, left_c + 1), W))
                if (left_c, right_c) != (left, right):
                    logger.info("[VerticalDetector] sweep extends beyond the raster (left %d, right %d of %d) -- clamped; the crop pads the missing strip", left, right, W)
                left, right = left_c, right_c
                self.edge_source_ = src_desc
                for side, pos in (("left", left), ("right", right)):
                    sub = self._sub_image(src, pos - self.search_window_width // 2, self.search_window_width)
                    prof = sub.band.flatten()
                    self._results[side] = VerticalEdgeResult(position=pos, edge_local=int(sub.to_local_x(pos)) if hasattr(sub, "to_local_x") else 0,
                                                             gradient_ratio=float(strength.get(side, 0.0)), sub_image=sub, profile=prof)
                self.n_candidates_ = int(peaks.size)
            except DetectionError as e:
                self._failed = True
                logger.warning("%s - failed to detect edges: %s", self.logging_prefix, e)
                return self
        logger.info(
            "%s - left=%d, right=%d, detected width=%d (expected=%.0f at this pitch, diff=%+d px, edges from %s, %d coherent steps)",
            self.logging_prefix, self.left_.position, self.right_.position, self.detected_width_,
            expected, int(self.detected_width_ - expected), self.edge_source_, self.n_candidates_)
        logger.info("%s - crop centre %.0f (mid-point %.0f; strength-weighted)", self.logging_prefix,
                    self.crop_center_, (self.left_.position + self.right_.position) / 2.0)
        return self

    def _detect_edge(self, sub_image: SubImage, side: str) -> VerticalEdgeResult:
        """Locate the edge within one sub-image, reversing the scan direction for the right side."""
        profile = sub_image.band.flatten()
        signal = profile[::-1] if side == "right" else profile

        _, plateau_end_idx = find_longest_min_segment(signal)
        result = find_first_strong_gradient(signal, plateau_end_idx, self.gradient_ratio_threshold)
        if result is None:
            raise DetectionError("No edge detected")
        edge_idx, gradient_ratio = result

        if side == "right":
            edge_idx = len(signal) - 1 - edge_idx

        position = int(sub_image.to_global_x(edge_idx))
        return VerticalEdgeResult(
            position=position,
            edge_local=edge_idx,
            gradient_ratio=gradient_ratio,
            sub_image=sub_image,
            profile=profile,
        )

    def _sub_image(self, src: rasterio.DatasetReader, col_off: int, window_width: int) -> SubImage:
        window = Window(
            max(0, col_off),
            int(self.vertical_padding * src.height),
            window_width,
            int(src.height * (1 - 2 * self.vertical_padding)),
        )
        return SubImage(src, window=window, out_shape=(1, 1, int(window.width * self.downsample_scale)))


def find_longest_min_segment(signal: NDArray[np.floating]) -> tuple[int, int]:
    """Find the longest contiguous segment where the signal equals its minimum value.

    Returns the (start, end) index pair of that segment.
    """
    signal = np.asarray(signal)

    mask = signal == signal.min()

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    longest = np.argmax(ends - starts)
    return int(starts[longest]), int(ends[longest])


def find_first_strong_gradient(
    signal: NDArray[np.floating], plateau_end_idx: int, ratio_threshold: float = 0.3
) -> tuple[int, float] | None:
    """Scan forward from plateau_end_idx for the peak of the first gradient rise above the threshold.

    plateau_end_idx is expected to be the end of the background plateau, where the signal is
    still flat (zero gradient); the rising edge into image content follows right after it.
    threshold = ratio_threshold * gradient.max(). Once the threshold is crossed, the scan keeps
    climbing to the following index as long as its gradient is higher, so it lands on the peak
    of the rise rather than its first crossing.

    Returns (index, gradient_ratio) of the peak, or None if the threshold is never crossed.
    """
    signal = np.asarray(signal)

    gradient = np.gradient(signal)
    gradient_max = gradient.max()
    threshold = ratio_threshold * gradient_max

    for i in range(plateau_end_idx, len(signal)):
        if gradient[i] > threshold:
            while i + 1 < len(signal) and gradient[i + 1] >= gradient[i] and gradient[i] < gradient_max:
                i += 1
            return i, gradient[i] / gradient_max

    return None
