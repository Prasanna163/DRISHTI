from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_access.load_fits import find_fits
from detection.run_bls import run_bls_search
from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve
from visualization.plot_bls_results import (
    plot_binned_phase_folded,
    plot_bls_periodogram,
    plot_phase_folded,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BLS on cleaned, flattened TESS light-curve FITS files."
    )
    parser.add_argument("input", type=Path, help="A .fits file or folder containing .fits files.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("outputs/candidates/bls_candidates.csv"),
        help="CSV path for BLS candidate summary.",
    )
    parser.add_argument(
        "--periodogram-dir",
        type=Path,
        default=Path("outputs/plots/periodograms"),
        help="Directory for BLS periodogram plots.",
    )
    parser.add_argument(
        "--phase-dir",
        type=Path,
        default=Path("outputs/plots/phase_folded"),
        help="Directory for BLS phase-folded plots.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument("--min-period", type=float, default=0.5, help="Minimum BLS period in days.")
    parser.add_argument("--max-period", type=float, default=13.0, help="Maximum BLS period in days.")
    parser.add_argument("--n-periods", type=int, default=20000, help="Number of periods in the BLS grid.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fits_files = [path for path in find_fits(args.input) if path.name.lower().endswith("_lc.fits")]

    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.periodogram_dir.mkdir(parents=True, exist_ok=True)
    args.phase_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fits_path in fits_files:
        print(f"Running BLS: {fits_path.name}")
        clean_lc = load_clean_flattened_lightcurve(fits_path)
        result = run_bls_search(
            clean_lc,
            min_period=args.min_period,
            max_period=args.max_period,
            n_periods=args.n_periods,
        )
        periodogram = plot_bls_periodogram(
            fits_path,
            result,
            args.periodogram_dir,
            dpi=args.dpi,
        )
        folded = plot_phase_folded(
            fits_path,
            result,
            args.phase_dir,
            dpi=args.dpi,
        )
        binned = plot_binned_phase_folded(
            fits_path,
            result,
            args.phase_dir,
            dpi=args.dpi,
        )

        rows.append(
            {
                "fits_file": fits_path.name,
                "best_period_days": result.best_period,
                "best_t0_btjd": result.best_t0,
                "best_duration_days": result.best_duration,
                "best_bls_power": result.best_power,
                "n_clean_cadences": len(result.time),
                "periodogram_plot": str(periodogram),
                "phase_folded_plot": str(folded),
                "phase_folded_binned_plot": str(binned),
            }
        )

    fieldnames = [
        "fits_file",
        "best_period_days",
        "best_t0_btjd",
        "best_duration_days",
        "best_bls_power",
        "n_clean_cadences",
        "periodogram_plot",
        "phase_folded_plot",
        "phase_folded_binned_plot",
    ]
    with args.candidates.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} light-curve FITS file(s).")
    print(f"Wrote candidate summary: {args.candidates.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

