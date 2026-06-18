from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detection.run_bls import bin_phase_curve, fold_lightcurve, run_bls_search
from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve


DEFAULT_RECOVERY = ROOT / "outputs" / "tables" / "tce_recovery_results.csv"
DEFAULT_FITS_DIR = ROOT / "data" / "raw" / "tce_products" / "lc"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "plots" / "tce_recovery"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate plots for the TCE recovery validation run.")
    parser.add_argument("--recovery", type=Path, default=DEFAULT_RECOVERY, help="Recovery result CSV.")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR, help="Downloaded LC FITS folder.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Plot output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Plot only the first N recovery rows.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    parser.add_argument("--min-period", type=float, default=0.5, help="Minimum BLS period in days.")
    parser.add_argument("--max-period", type=float, default=13.0, help="Maximum BLS period in days.")
    parser.add_argument("--n-periods", type=int, default=20000, help="Number of BLS period grid points.")
    parser.add_argument("--summary-only", action="store_true", help="Only regenerate summary plots.")
    parser.add_argument(
        "--diagnostics",
        choices=["controlled", "all", "none"],
        default="controlled",
        help=(
            "Target diagnostics to write. controlled writes all non-direct classes "
            "plus the strongest direct recoveries."
        ),
    )
    parser.add_argument(
        "--top-direct-diagnostics",
        type=int,
        default=10,
        help="Number of strongest direct recoveries to plot when --diagnostics controlled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recovery = pd.read_csv(args.recovery)
    if args.limit is not None:
        recovery = recovery.head(args.limit)

    summary_dir = args.output_dir / "summary"
    target_dir = args.output_dir / "targets"
    summary_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = plot_summary_figures(recovery, summary_dir, dpi=args.dpi)

    target_paths = []
    if args.summary_only or args.diagnostics == "none":
        print(f"Summary plots: {len(summary_paths)}")
        for path in summary_paths:
            print(f"  {path.resolve()}")
        print(f"Output directory: {args.output_dir.resolve()}")
        return 0

    diagnostic_rows = select_diagnostic_rows(
        recovery,
        mode=args.diagnostics,
        top_direct=args.top_direct_diagnostics,
    )
    print(f"Generating {len(diagnostic_rows)} target diagnostic plot(s)...", flush=True)
    for row in tqdm(list(diagnostic_rows.itertuples(index=False)), desc="Diagnostic plots", unit="plot", file=sys.stdout):
        fits_path = find_lc_fits(args.fits_dir, int(row.tic_id), int(row.sector))
        if fits_path is None:
            continue
        clean_lc = load_clean_flattened_lightcurve(fits_path)
        bls = run_bls_search(
            clean_lc,
            min_period=args.min_period,
            max_period=args.max_period,
            n_periods=args.n_periods,
        )
        target_paths.append(plot_target_diagnostic(row, clean_lc, bls, target_dir, dpi=args.dpi))

    print(f"Summary plots: {len(summary_paths)}")
    for path in summary_paths:
        print(f"  {path.resolve()}")
    print(f"Target diagnostics: {len(target_paths)}")
    print(f"Output directory: {args.output_dir.resolve()}")
    return 0


def plot_summary_figures(recovery: pd.DataFrame, output_dir: Path, *, dpi: int) -> list[Path]:
    paths = []
    recovery = recovery.copy()
    if "recovery_class" not in recovery.columns:
        recovery["recovery_class"] = np.where(
            recovery["recovered_true_false"].astype(bool),
            "direct_recovered",
            "not_recovered",
        )
    colors = recovery["recovery_class"].map(class_color).to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("TCE Recovery Summary", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(recovery["official_period"], recovery["our_bls_period"], c=colors, s=55, edgecolor="black", linewidth=0.4)
    low = min(recovery["official_period"].min(), recovery["our_bls_period"].min()) * 0.9
    high = max(recovery["official_period"].max(), recovery["our_bls_period"].max()) * 1.1
    ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--", label="1:1")
    ax.plot([low, high], [low / 2, high / 2], color="#777777", linewidth=0.8, linestyle=":", label="half-period")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Official period (days)")
    ax.set_ylabel("Our BLS period (days)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    x = np.arange(len(recovery))
    ax.bar(x, recovery["best_period_error_percent"], color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Best period error (%)")
    ax.set_xlabel("Recovery row")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.bar(x, recovery["epoch_match_score"], color=colors)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Epoch match score")
    ax.set_xlabel("Recovery row")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    sector_counts = recovery.groupby(["sector", "recovery_class"]).size().unstack(fill_value=0)
    class_order = [
        "direct_recovered",
        "alias_recovered",
        "period_recovered_epoch_mismatch",
        "period_recovered_bad_duration",
        "period_recovered_needs_vetting",
        "not_recovered",
        "download_failed",
        "processing_failed",
    ]
    for recovery_class in class_order:
        if recovery_class not in sector_counts.columns:
            sector_counts[recovery_class] = 0
    sectors = sector_counts.index.astype(int).to_numpy()
    positions = np.arange(len(sectors))
    bottom = np.zeros(len(sectors))
    for recovery_class in class_order:
        counts = sector_counts[recovery_class].to_numpy()
        if counts.sum() == 0:
            continue
        ax.bar(
            positions,
            counts,
            bottom=bottom,
            color=class_color(recovery_class),
            label=recovery_class,
        )
        bottom += counts
    ax.set_xticks(positions)
    ax.set_xticklabels([str(sector) for sector in sectors])
    ax.set_xlabel("Sector")
    ax.set_ylabel("Target rows")
    ax.set_title("Recovery class by sector")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    path = output_dir / "tce_recovery_summary.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{int(row.tic_id)} S{int(row.sector)}" for row in recovery.itertuples(index=False)]
    ax.scatter(recovery["official_snr"], recovery["our_snr"], c=colors, s=60, edgecolor="black", linewidth=0.4)
    for idx, label in enumerate(labels):
        ax.annotate(label, (recovery["official_snr"].iloc[idx], recovery["our_snr"].iloc[idx]), fontsize=6, alpha=0.75)
    ax.set_xlabel("Official SNR")
    ax.set_ylabel("Our BLS SNR estimate")
    ax.set_title("Official SNR vs Our BLS SNR")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / "tce_recovery_snr_comparison.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    paths.append(path)

    return paths


def select_diagnostic_rows(recovery: pd.DataFrame, *, mode: str, top_direct: int) -> pd.DataFrame:
    if mode == "all":
        return recovery.copy()
    if mode == "none":
        return recovery.head(0).copy()

    if "recovery_class" not in recovery.columns:
        return recovery.copy()

    non_direct = recovery[recovery["recovery_class"] != "direct_recovered"].copy()
    direct = recovery[recovery["recovery_class"] == "direct_recovered"].copy()
    direct = direct.sort_values("official_snr", ascending=False, kind="mergesort").head(top_direct)
    selected = pd.concat([non_direct, direct], ignore_index=True)
    return selected.sort_values(["sector", "tic_id"], kind="mergesort").reset_index(drop=True)


def class_color(recovery_class: str) -> str:
    colors = {
        "direct_recovered": "#19764b",
        "alias_recovered": "#2f6fba",
        "period_recovered_epoch_mismatch": "#c77c1a",
        "period_recovered_bad_duration": "#8a5fbf",
        "period_recovered_needs_vetting": "#8c6d31",
        "not_recovered": "#b33a3a",
        "download_failed": "#555555",
        "processing_failed": "#000000",
    }
    return colors.get(str(recovery_class), "#777777")


def plot_target_diagnostic(row, clean_lc, bls, output_dir: Path, *, dpi: int) -> Path:
    tic_id = int(row.tic_id)
    sector = int(row.sector)
    official_period = float(row.official_period)
    official_epoch = float(row.official_epoch)
    official_duration_days = float(row.official_duration_hours) / 24.0
    recovery_class = getattr(row, "recovery_class", "")
    if not recovery_class:
        recovery_class = "direct_recovered" if bool(row.recovered_true_false) else "not_recovered"
    status_color = class_color(recovery_class)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"TIC {tic_id} | Sector {sector} | {recovery_class}",
        fontsize=14,
        fontweight="bold",
        color=status_color,
    )

    ax = axes[0, 0]
    ax.scatter(clean_lc.time, clean_lc.flux, s=3, alpha=0.45, color="#314f7d", linewidths=0)
    shade_transit_windows(ax, clean_lc.time, official_epoch, official_period, official_duration_days)
    y_low, y_high = flux_limits(clean_lc.flux)
    ax.set_ylim(y_low, y_high)
    ax.set_xlabel("BTJD")
    ax.set_ylabel("Flattened flux")
    ax.set_title("Cleaned light curve with official transit windows")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(bls.period, bls.power, color="#263b63", linewidth=1)
    ax.axvline(official_period, color="#1f78b4", linestyle="-", linewidth=1.5, label="official period")
    ax.axvline(bls.best_period, color="#c43b3b", linestyle="--", linewidth=1.5, label="our BLS period")
    ax.axvline(official_period / 2.0, color="#1f78b4", linestyle=":", linewidth=1, label="official half")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("BLS power")
    ax.set_title(
        f"Periodogram | official={official_period:.5f} d | BLS={bls.best_period:.5f} d",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    plot_folded(ax, clean_lc.time, clean_lc.flux, official_period, official_epoch, official_duration_days)
    ax.set_title(f"Folded on official ephemeris | P={official_period:.5f} d", fontsize=10)

    ax = axes[1, 1]
    plot_folded(ax, clean_lc.time, clean_lc.flux, bls.best_period, bls.best_t0, bls.best_duration)
    ax.set_title(f"Folded on our BLS ephemeris | P={bls.best_period:.5f} d", fontsize=10)

    fig.tight_layout()
    safe_class = recovery_class.replace("/", "_")
    output_path = output_dir / safe_class / f"TIC_{tic_id}_S{sector:04d}_recovery_diagnostic.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_folded(
    ax,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
) -> None:
    phase, folded_flux = fold_lightcurve(time, flux, period, t0)
    phase_bins, flux_bins = bin_phase_curve(phase, folded_flux, bins=150)
    half_duration_phase = 0.5 * duration_days / period

    ax.scatter(phase, folded_flux, s=3, alpha=0.16, color="#456b91", linewidths=0)
    ax.plot(phase_bins, flux_bins, color="black", linewidth=1.8)
    ax.axhline(1.0, color="#777777", linewidth=1)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(-half_duration_phase, color="#8c8c8c", linestyle=":", linewidth=1)
    ax.axvline(half_duration_phase, color="#8c8c8c", linestyle=":", linewidth=1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(*flux_limits(folded_flux))
    ax.set_xlabel("Phase")
    ax.set_ylabel("Flattened flux")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.grid(True, alpha=0.25)


def shade_transit_windows(
    ax,
    time: np.ndarray,
    epoch: float,
    period: float,
    duration_days: float,
) -> None:
    if len(time) == 0 or period <= 0 or duration_days <= 0:
        return
    first = int(math.ceil((np.nanmin(time) - epoch) / period))
    last = int(math.floor((np.nanmax(time) - epoch) / period))
    for transit_number in range(first, last + 1):
        center = epoch + transit_number * period
        ax.axvspan(
            center - 0.5 * duration_days,
            center + 0.5 * duration_days,
            color="#3f8fc5",
            alpha=0.14,
            linewidth=0,
        )


def flux_limits(flux: np.ndarray) -> tuple[float, float]:
    finite = flux[np.isfinite(flux)]
    if len(finite) == 0:
        return 0.98, 1.02
    low, high = np.nanpercentile(finite, [0.5, 99.5])
    center = np.nanmedian(finite)
    half_range = max(abs(high - center), abs(center - low), 0.002)
    return center - half_range * 1.2, center + half_range * 1.2


def find_lc_fits(fits_dir: Path, tic_id: int, sector: int) -> Path | None:
    pattern = f"*-s{sector:04d}-{tic_id:016d}-*_lc.fits"
    matches = sorted(fits_dir.rglob(pattern))
    return matches[0] if matches else None


if __name__ == "__main__":
    raise SystemExit(main())
