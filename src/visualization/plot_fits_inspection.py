from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm

from data_access.load_fits import fits_label
from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve


@dataclass(frozen=True)
class FigureResult:
    fits_path: Path
    output_dir: Path
    figures: tuple[Path, ...]
    warnings: tuple[str, ...] = ()
    error: str | None = None


def generate_inspection_figures(
    fits_path: Path,
    output_root: Path,
    *,
    dpi: int = 300,
) -> FigureResult:
    fits_path = fits_path.expanduser().resolve()
    output_dir = output_root.expanduser().resolve() / fits_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Path] = []
    warnings: list[str] = []

    try:
        with fits.open(fits_path, memmap=False) as hdul:
            lightcurve_hdu = _find_lightcurve_hdu(hdul)
            pixel_hdu = _find_pixel_hdu(hdul)
            aperture_hdu = _find_aperture_hdu(hdul)

            if lightcurve_hdu is not None:
                figures.extend(_plot_raw_lightcurve(fits_path, lightcurve_hdu, output_dir, dpi=dpi))
                figures.extend(_plot_clean_flattened_lightcurve(fits_path, output_dir, dpi=dpi))

            if pixel_hdu is not None:
                figures.extend(_plot_pixel_frame(fits_path, pixel_hdu, output_dir, dpi=dpi))

            if aperture_hdu is not None:
                figures.append(_plot_inverted_aperture(fits_path, aperture_hdu, output_dir, dpi=dpi))

            if not figures:
                warnings.append("No supported LIGHTCURVE, PIXELS, or APERTURE data found.")
    except Exception as exc:
        return FigureResult(
            fits_path=fits_path,
            output_dir=output_dir,
            figures=tuple(figures),
            warnings=tuple(warnings),
            error=f"{type(exc).__name__}: {exc}",
        )

    return FigureResult(
        fits_path=fits_path,
        output_dir=output_dir,
        figures=tuple(figures),
        warnings=tuple(warnings),
    )


def _find_lightcurve_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        names = getattr(getattr(data, "columns", None), "names", None)
        if names and "TIME" in names and ("SAP_FLUX" in names or "PDCSAP_FLUX" in names):
            return hdu
    return None


def _find_pixel_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        names = getattr(getattr(data, "columns", None), "names", None)
        if not names or "FLUX" not in names:
            continue
        try:
            first_flux = data["FLUX"][0]
        except Exception:
            continue
        if np.asarray(first_flux).ndim == 2:
            return hdu
    return None


def _find_aperture_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        if hdu.name.upper() == "APERTURE" and getattr(hdu, "data", None) is not None:
            return hdu
    return None


def _plot_raw_lightcurve(
    fits_path: Path,
    hdu,
    output_dir: Path,
    *,
    dpi: int,
) -> Iterable[Path]:
    data = hdu.data
    time = np.asarray(data["TIME"], dtype=float)
    flux_column = "PDCSAP_FLUX" if "PDCSAP_FLUX" in data.columns.names else "SAP_FLUX"
    flux = np.asarray(data[flux_column], dtype=float)

    mask = np.isfinite(time) & np.isfinite(flux)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(time[mask], flux[mask], s=2, alpha=0.55, color="#102a83", linewidths=0)
    ax.set_title(f"{fits_label(fits_path)} | raw {flux_column}", fontsize=10)
    ax.set_xlabel("Time (BTJD days)")
    ax.set_ylabel(flux_column)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()

    output_path = output_dir / "lightcurve_raw_scatter.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return [output_path]


def _plot_clean_flattened_lightcurve(
    fits_path: Path,
    output_dir: Path,
    *,
    dpi: int,
) -> Iterable[Path]:
    clean = load_clean_flattened_lightcurve(fits_path)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(clean.time, clean.flux, s=2, alpha=0.6, color="#0b6b4f", linewidths=0)
    ax.axhline(1.0, color="#222222", lw=0.8, alpha=0.65)
    ax.set_title(
        f"{fits_label(fits_path)} | flattened, systematics-masked",
        fontsize=10,
    )
    ax.set_xlabel("Time (BTJD days)")
    ax.set_ylabel("Normalized flux")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()

    output_path = output_dir / "lightcurve_flattened_clean_scatter.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return [output_path]


def _plot_pixel_frame(
    fits_path: Path,
    hdu,
    output_dir: Path,
    *,
    dpi: int,
) -> Iterable[Path]:
    cube = np.asarray(hdu.data["FLUX"])
    valid_idx = _first_valid_frame_index(cube)
    frame = np.asarray(cube[valid_idx], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    norm = simple_norm(frame, stretch="sqrt", percent=99.5)
    image = ax.imshow(frame, origin="lower", cmap="viridis", norm=norm, interpolation="nearest")
    ax.set_title(f"{fits_label(fits_path)} | cadence {valid_idx}", fontsize=10)
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Flux")
    fig.tight_layout()

    output_path = output_dir / "target_pixel_frame.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return [output_path]


def _plot_inverted_aperture(
    fits_path: Path,
    hdu,
    output_dir: Path,
    *,
    dpi: int,
) -> Path:
    aperture = np.asarray(hdu.data)
    selected = _science_aperture_mask(aperture)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(selected, origin="lower", cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"{fits_label(fits_path)} | inverted aperture mask", fontsize=10)
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    fig.tight_layout()

    output_path = output_dir / "aperture_mask_inverted.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _first_valid_frame_index(cube: np.ndarray) -> int:
    for idx, frame in enumerate(cube):
        if np.isfinite(frame).any():
            return idx
    return 0


def _science_aperture_mask(aperture: np.ndarray) -> np.ndarray:
    aperture_int = np.nan_to_num(aperture, nan=0).astype(np.int64, copy=False)
    bit_two_mask = (aperture_int & 2) > 0
    if bit_two_mask.any():
        return bit_two_mask.astype(int)
    return (aperture_int > 0).astype(int)
