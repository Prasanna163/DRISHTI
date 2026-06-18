from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_access.load_fits import find_fits
from detection.run_bls import run_bls_search
from features.quantitative_products import (
    build_cleaned_lightcurve_table,
    candidate_row,
    folded_binned_table,
    folded_table,
    periodogram_table,
)
from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export quantitative light-curve, BLS, folded, and summary products."
    )
    parser.add_argument("input", type=Path, help="A .fits file or folder containing .fits files.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output root.")
    parser.add_argument("--min-period", type=float, default=0.5, help="Minimum BLS period in days.")
    parser.add_argument("--max-period", type=float, default=13.0, help="Maximum BLS period in days.")
    parser.add_argument("--n-periods", type=int, default=20000, help="Number of periods in the BLS grid.")
    parser.add_argument("--phase-bins", type=int, default=150, help="Number of folded phase bins.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lc_files = [path for path in find_fits(args.input) if path.name.lower().endswith("_lc.fits")]

    dirs = {
        "cleaned": args.output_root / "lightcurves_cleaned",
        "tables": args.output_root / "tables",
        "candidates": args.output_root / "candidates",
        "periodograms": args.output_root / "periodograms_csv",
        "folded": args.output_root / "folded",
        "folded_binned": args.output_root / "folded_binned",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    summaries = []
    candidates = []

    for fits_path in lc_files:
        print(f"Exporting products: {fits_path.name}")
        product = build_cleaned_lightcurve_table(fits_path)
        label = product.meta.label

        cleaned_path = dirs["cleaned"] / f"{label}_cleaned.csv.gz"
        product.table.to_csv(cleaned_path, index=False, compression="gzip")
        summaries.append(product.summary)

        clean_lc = load_clean_flattened_lightcurve(fits_path)
        bls = run_bls_search(
            clean_lc,
            min_period=args.min_period,
            max_period=args.max_period,
            n_periods=args.n_periods,
        )

        periodogram_table(bls).to_csv(
            dirs["periodograms"] / f"{label}_bls_periodogram.csv",
            index=False,
        )
        folded_table(product.meta, bls).to_csv(
            dirs["folded"] / f"{label}_BLS_candidate001_folded.csv",
            index=False,
        )
        folded_binned_table(product.meta, bls, bins=args.phase_bins).to_csv(
            dirs["folded_binned"] / f"{label}_BLS_candidate001_folded_binned.csv",
            index=False,
        )
        candidates.append(
            candidate_row(
                product.meta,
                fits_path,
                bls,
                min_period=args.min_period,
                max_period=args.max_period,
            )
        )

    pd.DataFrame(summaries).to_csv(dirs["tables"] / "lightcurve_summary.csv", index=False)
    pd.DataFrame(candidates).to_csv(dirs["candidates"] / "bls_candidates.csv", index=False)

    print(f"Processed {len(lc_files)} light-curve FITS file(s).")
    print(f"Wrote quantitative products under: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

