"""
Copyright (c) 2026 HIPP developers
Description: Click CLI exposing ``preproc``, ``batch-preproc`` and
    ``flatten-collimation`` commands for KH-9 PC preprocessing.
"""

# mypy: disable-error-code="misc"
import logging
import sys
from pathlib import Path

import click

from hipp.kh9pc.collimation_flatten import DEFAULT_BAND_PX, flatten_collimation_band
from hipp.kh9pc.pipeline import batch_preprocess_kh9pc, preprocess_kh9pc


def _configure_logging(verbosity: int) -> None:
    """Set the ``hipp`` logger level based on a verbosity count (0=WARNING, 1=INFO, 2+=DEBUG)."""
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
    )
    logging.getLogger("hipp").setLevel(level)


@click.group()
def main() -> None:
    """KH-9 Panoramic Camera preprocessing tools."""


@main.command()
@click.option(
    "--input",
    "-i",
    "input_files",
    multiple=True,
    required=True,
    metavar="FILE",
    help="Input archive (.tgz) or tile files (.tif)",
)
@click.option(
    "--output-dir", "-o", required=True, type=Path, metavar="DIR", help="Output directory for restituted images"
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.option("--keep-work", is_flag=True, help="Keep intermediate working files")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v INFO, -vv DEBUG)")
def preproc(input_files: tuple[str, ...], output_dir: Path, overwrite: bool, keep_work: bool, verbose: int) -> None:
    """Preprocess a single KH-9 PC scan."""
    _configure_logging(verbose)
    preprocess_kh9pc(
        input=list(input_files) if len(input_files) > 1 else input_files[0],
        output_dir=output_dir,
        overwrite=overwrite,
        keep_work=keep_work,
    )


@main.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=Path,
    metavar="DIR",
    help="Directory containing input archives or tile subdirectories",
)
@click.option(
    "--output-dir", "-o", required=True, type=Path, metavar="DIR", help="Output directory for restituted images"
)
@click.option("--n-jobs", "-j", default=1, show_default=True, help="Number of parallel jobs")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.option("--keep-work", is_flag=True, help="Keep intermediate working files")
@click.option("--dry-run", is_flag=True, help="Log what would be processed without running")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v INFO, -vv DEBUG)")
def batch_preproc(
    input_dir: Path, output_dir: Path, n_jobs: int, overwrite: bool, keep_work: bool, dry_run: bool, verbose: int
) -> None:
    """Batch preprocess multiple KH-9 PC scans."""
    _configure_logging(verbose)
    batch_preprocess_kh9pc(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=overwrite,
        keep_work=keep_work,
        n_jobs=n_jobs,
        dry_run=dry_run,
    )


@main.command(name="flatten-collimation")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_file", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--band-px",
    default=DEFAULT_BAND_PX,
    show_default=True,
    help="Rows at each edge examined for the brightness curve",
)
@click.option(
    "--taper-px",
    default=DEFAULT_TAPER_PX,
    show_default=True,
    help="Rows over which the correction fades to zero at the interior end of the band",
)
@click.option(
    "--match-spread",
    is_flag=True,
    help="Also restore per-row contrast (gain = nmad_ref/nmad_row, clamped to [1, gain-cap])",
)
@click.option("--gain-cap", default=DEFAULT_GAIN_CAP, show_default=True, help="Upper clamp for the match-spread gain")
@click.option(
    "--saturation-dn",
    type=float,
    default=None,
    help="Optional: pixels >= this DN -> nodata (clipped highlights). Default: disabled, all pixels preserved",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing output")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v INFO, -vv DEBUG)")
def flatten_collimation(
    input_file: Path,
    output_file: Path,
    band_px: int,
    taper_px: int,
    match_spread: bool,
    gain_cap: float,
    saturation_dn: float | None,
    overwrite: bool,
    verbose: int,
) -> None:
    """Radiometrically flatten the collimation band of a restituted KH-9 PC image.

    Subtracts the per-row median brightness excess of the collimation line and
    its halo at the top and bottom image edges, preserving the background pixels
    (ground signal shows through the line). Run this on BOTH images of a
    same-sensor stereo pair before correlation.
    """
    _configure_logging(verbose if verbose else 1)
    flatten_collimation_band(
        input_file,
        output_file,
        band_px=band_px,
        taper_px=taper_px,
        match_spread=match_spread,
        gain_cap=gain_cap,
        saturation_dn=saturation_dn,
        overwrite=overwrite,
    )
