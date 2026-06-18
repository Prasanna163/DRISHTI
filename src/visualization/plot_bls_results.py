from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from data_access.load_fits import fits_label
from detection.run_bls import BlsSearchResult, bin_phase_curve, fold_lightcurve


def plot_bls_periodogram(
    fits_path: Path,
    result: BlsSearchResult,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result.period, result.power, linewidth=1.0, color="#243b6b")
    ax.axvline(result.best_period, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("BLS power")
    ax.set_title(
        f"{fits_label(fits_path)} | best P={result.best_period:.5f} d",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{fits_path.stem}_bls_periodogram.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_phase_folded(
    fits_path: Path,
    result: BlsSearchResult,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase, folded_flux = fold_lightcurve(
        result.time,
        result.flux,
        result.best_period,
        result.best_t0,
    )
    half_duration_phase = 0.5 * result.best_duration / result.best_period

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(phase, folded_flux, s=4, alpha=0.35, color="#0b6b4f", linewidths=0)
    ax.axhline(1.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(-half_duration_phase, color="gray", linestyle=":", linewidth=1)
    ax.axvline(+half_duration_phase, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized flux")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_title(
        f"{fits_label(fits_path)} | folded BLS P={result.best_period:.5f} d",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{fits_path.stem}_phase_folded.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_binned_phase_folded(
    fits_path: Path,
    result: BlsSearchResult,
    output_dir: Path,
    *,
    dpi: int = 300,
    bins: int = 150,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase, folded_flux = fold_lightcurve(
        result.time,
        result.flux,
        result.best_period,
        result.best_t0,
    )
    phase_bins, flux_bins = bin_phase_curve(phase, folded_flux, bins=bins)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(phase, folded_flux, s=3, alpha=0.15, color="#0b6b4f", linewidths=0)
    ax.plot(phase_bins, flux_bins, color="black", linewidth=2)
    ax.axhline(1.0, color="gray", linewidth=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized flux")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_title(
        f"{fits_label(fits_path)} | binned folded BLS P={result.best_period:.5f} d",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{fits_path.stem}_phase_folded_binned.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path

