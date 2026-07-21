"""Digital-redaction masking: uniform near-black regions are masked, noisy
dark film is not, and the rupture scan skips redacted runs instead of
snapping to a redaction boundary."""

import numpy as np

from hipp.kh9pc.redaction_mask import detect_ruptures_skip_redacted, redacted_region_mask


def _band():
    rng = np.random.default_rng(7)
    band = rng.integers(40, 200, size=(40, 30)).astype(np.uint8)   # bright film
    # noisy dark film patch (deep shadow, grain variance) — must NOT mask
    band[5:15, 0:10] = rng.integers(1, 6, size=(10, 10))
    # digital redaction: exactly-constant 0 rectangle — must mask
    band[20:32, 5:20] = 0
    # short constant run below min_run — must NOT mask
    band[0:2, 25] = 0
    return band


def test_redacted_region_mask_selectivity():
    band = _band()
    m = redacted_region_mask(band, max_dn=6, max_range=2, min_run=3)
    assert m[20:32, 5:20].all(), "uniform redaction rectangle must be masked"
    assert not m[5:15, 0:10].any(), "noisy dark film must NOT be masked"
    assert not m[0:2, 25].any(), "runs shorter than min_run must NOT be masked"


def test_rupture_scan_skips_redaction():
    # column layout (top-side scan runs REVERSED, i.e. from index end upward):
    # rows 0-4 scanner background (dark noisy), 5-9 bright film base,
    # 10-19 redaction (constant 0), 20-39 bright content.
    vec = np.full(40, 120, dtype=np.uint8)
    vec[0:5] = np.array([3, 5, 2, 4, 3])
    vec[10:20] = 0
    redacted = np.zeros(40, bool)
    redacted[10:20] = True

    from hipp.kh9pc.restitution.base import detect_ruptures

    naive = detect_ruptures(vec, 20, reverse_scan=True)
    assert naive[0] == 19, "naive scan snaps to the redaction boundary"

    skipped = detect_ruptures_skip_redacted(vec, 20, redacted, reverse_scan=True)
    assert skipped[0] == 4, "masked scan must find the true film edge above the redaction"


def test_variance_edge_basic_and_redacted():
    """_variance_edge finds the texture edge nearest the line, both
    orientations, and is not fooled by a uniform redaction band."""
    import numpy as np
    from hipp.kh9pc.restitution.collimation_strategy import _variance_edge

    rng = np.random.default_rng(7)
    n = 200
    # bottom-side strip layout: line/content at the START (rows 0..119 noisy),
    # smooth margin beyond (rows 120.. flat ~60 DN)
    col = np.concatenate([
        rng.normal(120, 15, 120),           # scene content
        np.full(80, 60.0) + rng.normal(0, 0.3, 80),  # unexposed margin
    ])
    red = np.zeros(n, dtype=bool)
    r = _variance_edge(col, red, from_end=False)
    assert r is not None and 110 <= r <= 135

    # top-side strip: same column REVERSED (line at the END), edge mirrors
    r2 = _variance_edge(col[::-1], red, from_end=True)
    assert r2 is not None and n - 135 <= r2 <= n - 110

    # a uniform redaction band INSIDE content must not fake the edge
    col_red = col.copy()
    col_red[40:70] = 12.0                   # uniform fill
    red2 = np.zeros(n, dtype=bool)
    red2[40:70] = True
    r3 = _variance_edge(col_red, red2, from_end=False)
    assert r3 is not None and 110 <= r3 <= 135
