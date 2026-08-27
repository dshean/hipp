"""
Copyright (c) 2026 HIPP developers
Description: KH-9 Hexagon panoramic camera image specification: per-mission lookup of
    expected image dimensions, collimation line presence, fiducial type, and fiducial
    pattern names. All properties are derived from the standardised entity ID filename.
"""

from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import re
import rasterio


from hipp.kh9pc.fiducial_patterns import Patterns

# Nominal widths for 1, 2, 3, and 4-frame scans at 0.007 mm/px resolution.
IMAGE_WIDTHS_PX: list[int] = [114082, 228165, 342247, 456329]
IMAGE_HEIGHT_PX: int = 21771
# Physical pitch of the canvas: the collimation lines are 152.4 mm (6.000 in)
# apart = 21770 px, and a 90 deg sector = f * pi/2 = 2394 mm = 342247 px, i.e.
# the canonical dims are defined at 7.000 um/px. Scanner sessions run at
# 6.955-6.988 um (TIFF X/YResolution, non-square at the 1e-4 level); pass the
# scan pitch to the strategies so every restituted image is at CANVAS_PITCH_UM.
CANVAS_PITCH_UM: float = 7.0


@dataclass
class KH9ImageSpec:
    """Mission-specific image specification for a KH-9 Hexagon panoramic camera scan.

    Derived entirely from the entity ID embedded in the filename (e.g. ``D3C1210-…``).
    Covers missions 1201–1219. Key differences across missions:

    - **Collimation lines** appear starting from mission 1206.
    - **Fiducial type** switches from ``"disk"`` to ``"wagon_wheel"`` at mission 1214.
    - **Fiducial patterns** vary in density/layout per mission range.
    """

    expected_size: tuple[int, int]
    collimation_line: bool
    fiducial_type: Literal["disk", "wagon_wheel"]
    top_fiducial_patterns: tuple[Patterns, Patterns]
    bottom_fiducial_patterns: tuple[Patterns, Patterns]

    @classmethod
    def from_raster_filepath(cls, filepath: str | Path) -> "KH9ImageSpec":
        """Build the spec by parsing mission number and actual width from the raster file."""
        mission = KH9ImageSpec.mission_from_filepath(filepath)
        expected_size = KH9ImageSpec.expected_size_from_file(filepath)
        collimation_line = KH9ImageSpec.collimation_from_mission(mission)
        fiducial_type = KH9ImageSpec.fiducial_type_from_mission(mission)
        top_fiducial_patterns = KH9ImageSpec.top_fiducial_patterns_from_mission(mission)
        bottom_fiducial_patterns = KH9ImageSpec.bottom_fiducial_patterns_from_mission(mission)

        return cls(expected_size, collimation_line, fiducial_type, top_fiducial_patterns, bottom_fiducial_patterns)

    @staticmethod
    def mission_from_filepath(filepath: str | Path) -> int:
        """Parse and return the 4-digit KH-9 mission number from the entity ID filename stem."""
        pattern = re.compile(r"^(D3C)(\d{4})-(\d)(\d{5})([FA])(\d{3})$")
        stem = Path(filepath).stem
        m = pattern.match(stem)
        if m is None:
            raise ValueError(
                f"Cannot parse KH-9 image ID from {filepath!r}. Expected D3C{{mission}}-{{n}}{{roll}}{{F|A}}{{frame}}."
            )
        mission = int(m.group(2))
        return mission

    @staticmethod
    def collimation_from_mission(mission: int) -> bool:
        """Return True if the mission has collimation lines (missions 1206 and later)."""
        if mission < 1201 or mission > 1219:
            raise ValueError("Unrecgnized mission")
        collimation_line = mission >= 1206
        return collimation_line

    @staticmethod
    def fiducial_type_from_mission(mission: int) -> Literal["disk", "wagon_wheel"]:
        """Return the fiducial marker type: ``"disk"`` up to mission 1213, ``"wagon_wheel"`` from 1214."""
        if mission < 1201 or mission > 1219:
            raise ValueError("Unrecgnized mission")
        fiducial_type: Literal["disk", "wagon_wheel"] = "disk" if mission <= 1213 else "wagon_wheel"
        return fiducial_type

    @staticmethod
    def top_fiducial_patterns_from_mission(mission: int) -> tuple[Patterns, Patterns]:
        """Return the (primary, secondary) fiducial pattern names for the top film edge."""
        if mission < 1201 or mission > 1219:
            raise ValueError("Unrecgnized mission")
        top_fiducial_patterns: tuple[Patterns, Patterns]
        if mission <= 1213:
            top_fiducial_patterns = ("regulare_sparse", "serialized_time_word")
        elif mission <= 1217:
            top_fiducial_patterns = ("segmented_mid", "serialized_time_word")
        else:
            top_fiducial_patterns = ("segmented_mid", "segmented_dense")
        return top_fiducial_patterns

    @staticmethod
    def bottom_fiducial_patterns_from_mission(mission: int) -> tuple[Patterns, Patterns]:
        """Return the (primary, secondary) fiducial pattern names for the bottom film edge."""
        if mission < 1201 or mission > 1219:
            raise ValueError("Unrecgnized mission")

        bottom_fiducial_patterns: tuple[Patterns, Patterns]
        if mission <= 1213:
            bottom_fiducial_patterns = ("regulare_sparse", "regular_dense")
        else:
            bottom_fiducial_patterns = ("regulare_mid", "regular_dense")

        return bottom_fiducial_patterns

    @staticmethod
    def expected_size_from_file(filepath: str | Path) -> tuple[int, int]:
        """Return the expected (width, height) by snapping the actual width to the NEAREST
        known nominal width (sector tiers 30/60/90/120 deg are 33 % apart).

        kh9pc 2026-08-25: the old snap-DOWN picked the 60-deg tier (228165) for a
        90-deg mosaic that came out 293 px under 342247 (ops323 F001: sections at
        6.988 um and a slightly short scan) and the vertical detector then anchored a
        228 k sweep inside a 342 k frame.  A mosaic can legitimately sit a few 0.1 %
        either side of its tier (scan pitch, canvas gauge); the tier is a class, not
        a lower bound.  Widths more than 15 % from every tier are an error."""
        with rasterio.open(filepath) as src:
            width = src.width

        # kh9pc 2026-08-26: a mosaic's width cannot identify the sector tier on its own --
        # 30-deg sector scans carry ~25 % of extra film (ops327 A004 142846 px, ops395
        # F001 142462 px for a 114082 sweep) and a 5-section scan (~175 k) would even
        # sit nearer the 60-deg tier. The block's tier is physics known to the caller
        # (prep wrapper TIER); pass it as KH9_SECTOR_WIDTH_PX. Without it, fall back to
        # the nearest tier in log space and warn when the raster is > 1.3x the tier.
        import logging, os
        env = os.environ.get("KH9_SECTOR_WIDTH_PX")
        if env:
            tier = int(env)
            if tier not in IMAGE_WIDTHS_PX:
                raise ValueError(f"KH9_SECTOR_WIDTH_PX={tier} is not a known sector width {sorted(IMAGE_WIDTHS_PX)}")
            if width < 0.95 * tier or width > 2.0 * tier:
                raise ValueError(f"Image width {width} is not plausible for the declared sector width {tier} (0.95..2.0x).")
            return (tier, IMAGE_HEIGHT_PX)
        import math
        nearest = min(IMAGE_WIDTHS_PX, key=lambda w: abs(math.log(width / w)))
        if width > 1.3 * nearest or width < 0.95 * nearest:
            logging.getLogger(__name__).warning(
                "KH9ImageSpec: raster width %d is %.2fx the nearest sector width %d -- set KH9_SECTOR_WIDTH_PX from the block tier",
                width, width / nearest, nearest)
        return (nearest, IMAGE_HEIGHT_PX)
