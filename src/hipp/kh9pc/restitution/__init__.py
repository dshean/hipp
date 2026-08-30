"""
Copyright (c) 2026 HIPP developers
Description: Restitution strategies for KH-9 PC images — geometric correction from
    raw mosaicked scans to standardised output frames. Strategies are listed in order
    of increasing fallback: FiducialStrategy → CollimationStrategy → PolyStrategy → FlatStrategy.
    MixedStrategy chains them automatically and selects the first that succeeds.
    MarkStrategy / MarkPolyStrategy are opt-in siblings (RESTIT_STRATEGY=mark) that keep the
    Collimation / Poly y rectification and take x from the printed scan-angle ladder.
"""

from .collimation_strategy import CollimationStrategy
from .fiducial_strategy import FiducialStrategy
from .flat_strategy import FlatStrategy
from .mark_strategy import MarkOptions, MarkPolyStrategy, MarkStrategy
from .mixed_strategy import MixedStrategy
from .poly_strategy import PolyStrategy
from .vertical_detector import VerticalDetector

__all__ = [
    "CollimationStrategy",
    "FiducialStrategy",
    "MarkStrategy",
    "MarkPolyStrategy",
    "MarkOptions",
    "FlatStrategy",
    "PolyStrategy",
    "MixedStrategy",
    "VerticalDetector",
]
