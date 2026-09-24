"""MixedStrategy cascade semantics around TERMINAL failures (review H2, 2026-08-30).

A mark strategy that refuses a frame must not silently become a fallback product:
the fallback (plain CollimationStrategy / PolyStrategy) delivers a canvas at a
DIFFERENT width (tier vs tier x widen) under a class name the worker accepts, which
is exactly the quiet degradation the loud-refusal design forbids.  The contract is
``cascade_on_failure = False`` on the strategy class; ``MixedStrategy`` stops the
cascade there and reports overall failure with the refusing strategy on
``terminal_failure_``.

These are interface tests on stub strategies -- no rasters, no detector fits -- plus
one assertion that the real classes declare the attribute the way the contract needs.
"""

from pathlib import Path

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("cv2")

from hipp.kh9pc.restitution.base import RestitutionStrategy                     # noqa: E402
from hipp.kh9pc.restitution.collimation_strategy import CollimationStrategy     # noqa: E402
from hipp.kh9pc.restitution.fiducial_strategy import FiducialStrategy           # noqa: E402
from hipp.kh9pc.restitution.flat_strategy import FlatStrategy                   # noqa: E402
from hipp.kh9pc.restitution.mark_strategy import MarkPolyStrategy, MarkStrategy  # noqa: E402
from hipp.kh9pc.restitution.mixed_strategy import MixedStrategy                 # noqa: E402
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy                   # noqa: E402


class _StubDetector:
    """Pretends the shared VerticalDetector is already fitted on the frame."""

    def __init__(self, path):
        self._path = Path(path)

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def raster_filepath_(self) -> Path:
        return self._path


class _StubPoly:
    def __init__(self, path):
        self.vertical_detector = _StubDetector(path)


class _Stub(RestitutionStrategy):
    """Minimal concrete strategy: fit is a no-op with a scripted outcome."""

    def __init__(self, failed: bool, raises: bool = False):
        super().__init__()
        self._failed = failed
        self._raises = raises

    def _fit(self, raster_filepath: Path) -> "_Stub":
        if self._raises:
            raise RuntimeError("synthetic fit explosion")
        return self

    @property
    def is_failed(self) -> bool:
        return self._failed

    def transform(self, output_path) -> None:
        raise NotImplementedError

    @property
    def transformation_(self):
        raise NotImplementedError


class _TerminalStub(_Stub):
    cascade_on_failure = False
    mark_error_ = "no clean 5 deg ladder (synthetic)"


def test_a_terminal_refusal_stops_the_cascade(tmp_path):
    """H2: the fallback is not selected AND not even tried after a terminal refusal."""
    p = tmp_path / "frame.tif"
    term = _TerminalStub(failed=True)
    fallback = _Stub(failed=False)
    ms = MixedStrategy(strategies=[term, fallback], poly_strategy=_StubPoly(p))
    ms.fit(p)
    assert ms.is_failed
    assert ms.terminal_failure_ is term
    assert not fallback.is_fitted                      # the cascade never reached it
    with pytest.raises(RuntimeError):
        _ = ms.selected_strategy_


def test_a_terminal_strategy_that_raises_stops_the_cascade_too(tmp_path):
    """The refusal path and the exception path must behave identically (sibling)."""
    p = tmp_path / "frame.tif"
    term = _TerminalStub(failed=True, raises=True)
    fallback = _Stub(failed=False)
    ms = MixedStrategy(strategies=[term, fallback], poly_strategy=_StubPoly(p))
    ms.fit(p)
    assert ms.is_failed
    assert ms.terminal_failure_ is term
    assert not fallback.is_fitted


def test_an_ordinary_failure_still_cascades(tmp_path):
    """Defaults untouched: a plain strategy failure falls through as it always did."""
    p = tmp_path / "frame.tif"
    bad = _Stub(failed=True)
    fallback = _Stub(failed=False)
    ms = MixedStrategy(strategies=[bad, fallback], poly_strategy=_StubPoly(p))
    ms.fit(p)
    assert not ms.is_failed
    assert ms.selected_strategy_ is fallback
    assert ms.terminal_failure_ is None
    assert ms.failed_strategies == [bad]


def test_an_instance_can_opt_back_into_cascading_knowingly(tmp_path):
    """The escape is per INSTANCE, so the class contract stays terminal."""
    p = tmp_path / "frame.tif"
    term = _TerminalStub(failed=True)
    term.cascade_on_failure = True
    fallback = _Stub(failed=False)
    ms = MixedStrategy(strategies=[term, fallback], poly_strategy=_StubPoly(p))
    ms.fit(p)
    assert not ms.is_failed
    assert ms.selected_strategy_ is fallback
    assert _TerminalStub.cascade_on_failure is False   # the class did not move


def test_the_mark_classes_declare_terminal_failure_and_the_parents_do_not():
    """The real classes carry the contract; the plain parents keep the old cascade."""
    assert MarkStrategy.cascade_on_failure is False
    assert MarkPolyStrategy.cascade_on_failure is False
    for cls in (CollimationStrategy, PolyStrategy, FiducialStrategy, FlatStrategy):
        assert getattr(cls, "cascade_on_failure", True) is True, cls.__name__
