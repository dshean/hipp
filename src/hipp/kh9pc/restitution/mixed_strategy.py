"""
Copyright (c) 2026 HIPP developers
Description: MixedStrategy — automatic strategy selection. Tries FiducialStrategy,
    CollimationStrategy, PolyStrategy, and FlatStrategy in order and selects the first
    non-failed result. Sub-strategy instances share a single PolyStrategy and
    VerticalDetector to avoid redundant computation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from hipp.kh9pc.restitution.collimation_strategy import CollimationStrategy
from hipp.kh9pc.restitution.fiducial_strategy import FiducialStrategy
from hipp.kh9pc.restitution.flat_strategy import FlatStrategy
from hipp.kh9pc.restitution.poly_strategy import PolyStrategy
from hipp.kh9pc.restitution.base import RestitutionStrategy, Transformation

logger = logging.getLogger(__name__)


@dataclass
class MixedStrategy(RestitutionStrategy):
    """Composite strategy that tries sub-strategies in priority order and picks the first that succeeds.

    Default order: ``FiducialStrategy → CollimationStrategy → PolyStrategy → FlatStrategy``.
    All sub-strategy instances share the same ``PolyStrategy`` (and transitively the same
    ``VerticalDetector``) so common fitting work is done only once. If every strategy
    fails, ``is_failed`` returns True and ``transform`` raises.

    A strategy whose class sets ``cascade_on_failure = False`` (the mark strategies do;
    review H2, 2026-08-30) is TERMINAL: when it fails, the cascade STOPS instead of
    falling through to the next strategy, and the whole MixedStrategy reports failure.
    Falling through would deliver a fallback product at a DIFFERENT canvas width under a
    name the caller may accept -- a mark refusal silently becoming a collimation product
    is exactly the failure mode the loud-refusal design exists to prevent. The refusing
    strategy is kept on ``terminal_failure_`` so the caller can report WHY.
    """

    strategies: list[RestitutionStrategy] = field(
        default_factory=lambda: [FiducialStrategy(), CollimationStrategy(), PolyStrategy(), FlatStrategy()]
    )
    poly_strategy: PolyStrategy = field(default_factory=PolyStrategy)
    # (x_um, y_um) scan pitch, propagated to every sub-strategy that has the field
    scan_pitch_um: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        super().__init__()
        for strat in list(self.strategies) + [self.poly_strategy]:
            if hasattr(strat, "scan_pitch_um") and self.scan_pitch_um is not None:
                strat.scan_pitch_um = self.scan_pitch_um
        self.__selected_strategy_: RestitutionStrategy | None = None
        self.terminal_failure_: RestitutionStrategy | None = None

        for i, strat in enumerate(self.strategies):
            if hasattr(strat, "vertical_detector"):
                setattr(strat, "vertical_detector", self.poly_strategy.vertical_detector)
            if hasattr(strat, "poly_strategy"):
                setattr(strat, "poly_strategy", self.poly_strategy)
            # replace any standalone PolyStrategy with the shared instance to avoid recomputation
            if isinstance(strat, PolyStrategy) and strat is not self.poly_strategy:
                self.strategies[i] = self.poly_strategy

    @property
    def is_failed(self) -> bool:
        """True if no strategy produced a usable result. Raises if ``fit()`` was not called."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before")
        return self.__selected_strategy_ is None

    @property
    def selected_strategy_(self) -> RestitutionStrategy:
        """The first strategy that succeeded. Raises if none did or fit was not called."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before")

        if self.__selected_strategy_ is None:
            raise RuntimeError("All strategies failed")

        return self.__selected_strategy_

    @property
    def failed_strategies(self) -> list[RestitutionStrategy]:
        """Strategies that were tried and failed before the selected one."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before")

        if self.__selected_strategy_ is None:
            return self.strategies

        idx = self.strategies.index(self.__selected_strategy_)
        return self.strategies[:idx]

    @property
    def transformation_(self) -> Transformation:
        """Delegates to the selected strategy's transformation."""
        return self.selected_strategy_.transformation_

    def transform(self, output_path: str | Path) -> None:
        """Delegates to the selected strategy's ``transform``."""
        self.selected_strategy_.transform(output_path)

    def _fit(self, raster_filepath: Path) -> "MixedStrategy":
        """Fit the shared VerticalDetector once, then try each strategy in order."""
        self.__selected_strategy_ = None
        #: the strategy that failed AND declared itself terminal (``cascade_on_failure
        #: = False``), stopping the cascade -- None on success or an ordinary failure
        self.terminal_failure_: RestitutionStrategy | None = None

        vd = self.poly_strategy.vertical_detector
        if not vd.is_fitted or raster_filepath != vd.raster_filepath_:
            # 2026-08-26: the SHARED detector is fitted here first; every strategy's poly
            # then finds it fitted and never sets the pitch -> the worker ran the sweep
            # width unscaled (0.2 % off; F013 paired 4828..346641 instead of ..348290) while
            # the gate, which builds CollimationStrategy directly, was correct (audit H7 class)
            if self.scan_pitch_um is not None:
                vd.scan_pitch_um = self.scan_pitch_um
                if hasattr(self.poly_strategy, "scan_pitch_um"):
                    self.poly_strategy.scan_pitch_um = self.scan_pitch_um
            vd.fit(raster_filepath)

        for strat in self.strategies:
            try:
                if not strat.is_fitted or raster_filepath != strat.raster_filepath_:
                    strat.fit(raster_filepath)
            except Exception:
                logger.warning("%s failed for %s", type(strat).__name__, raster_filepath.name, exc_info=True)
                if not getattr(strat, "cascade_on_failure", True):
                    self.terminal_failure_ = strat
                    logger.error(
                        "%s REFUSED %s and is terminal (cascade_on_failure=False) -- NOT falling "
                        "through to a fallback strategy: a silent fallback would deliver a product "
                        "at a different canvas width (review H2, 2026-08-30)",
                        type(strat).__name__, raster_filepath.name)
                    break
                continue
            if not strat.is_failed:
                self.__selected_strategy_ = strat
                break
            if not getattr(strat, "cascade_on_failure", True):
                self.terminal_failure_ = strat
                logger.error(
                    "%s REFUSED %s (%s) and is terminal (cascade_on_failure=False) -- NOT falling "
                    "through to a fallback strategy: a silent fallback would deliver a product at "
                    "a different canvas width (review H2, 2026-08-30)",
                    type(strat).__name__, raster_filepath.name,
                    getattr(strat, "mark_error_", None) or "see the strategy's own log")
                break

        return self
