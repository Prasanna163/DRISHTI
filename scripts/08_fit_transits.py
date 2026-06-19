"""Fit a physical transit model (trapezoid) to downloaded TCE targets and report
parameters with uncertainties.

For each target row this script: finds the LC FITS, cleans/flattens it, runs BLS to get a
seed period/epoch/duration, then fits a trapezoid transit shape and records the fitted
period/depth/duration with 1-sigma uncertainties plus fit-quality metrics. Official TCE
parameters are carried alongside so parameter accuracy can be measured directly.

Examples
--------
    python scripts/08_fit_transits.py --limit 5
    python scripts/08_fit_transits.py --targets data/drishti/targets/tce_recovery_batch_111.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detection.run_bls import run_bls_search
from fitting.transit_fit import fit_transit
from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve
from drishti_store import DOWNLOAD_LC_ROOT, RESULT_TABLE_ROOT, TARGET_ROOT

DEFAULT_TARGETS = TARGET_ROOT / "tce_recovery_batch_111.csv"
DEFAULT_FITS_DIR = DOWNLOAD_LC_ROOT
DEFAULT_OUTPUT = RESULT_TABLE_ROOT / "transit_fits.csv"

FIELDNAMES = [
    "tic_id", "sector",
    "official_period", "official_duration_hours", "official_depth_ppm",
    "fit_period_days", "fit_period_err_days",
    "fit_t0_btjd", "fit_t0_err_days",
    "fit_depth_ppm", "fit_depth_err_ppm",
    "fit_duration_hours", "fit_duration_err_hours",
    "fit_ingress_frac", "fit_ingress_frac_err",
    "depth_ratio_fit_vs_official", "duration_ratio_fit_vs_official",
    "reduced_chi2", "bic", "n_points_fit",
    "fit_status", "source_file", "message",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit trapezoid transit model with uncertainties.")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="TCE target CSV.")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR, help="Downloaded LC FITS folder.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Transit-fit result CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Fit only the first N target rows.")
    parser.add_argument("--min-period", type=float, default=0.5)
    parser.add_argument("--max-period", type=float, default=13.0)
    parser.add_argument("--n-periods", type=int, default=20000)
    parser.add_argument("--force", action="store_true", help="Refit all (default: skip already-fit ok rows).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = pd.read_csv(args.targets)
    if args.limit is not None:
        targets = targets.head(args.limit)

    already_done: set[tuple[int, int]] = set()
    existing_rows: list[dict] = []
    if not args.force and args.output.exists():
        existing_df = pd.read_csv(args.output)
        for row in existing_df[existing_df["fit_status"] == "ok"].itertuples(index=False):
            already_done.add((int(row.tic_id), int(row.sector)))
        existing_rows = existing_df.to_dict("records")
        if already_done:
            print(f"Found {len(already_done)} already-fit target(s), skipping them.", flush=True)

    rows = list(existing_rows)
    pending = [
        t for t in targets.itertuples(index=False)
        if (int(t.tic_id), int(t.sector)) not in already_done
    ]
    print(f"Fitting transit model for {len(pending)} target row(s) ({len(already_done)} skipped)...", flush=True)

    for target in tqdm(pending, desc="Transit fits", unit="target", file=sys.stdout):
        rows.append(fit_one_target(target, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output, rows)
    _print_summary(rows, args.output)
    return 0


def fit_one_target(target, args) -> dict:
    official_depth_ppm = float(getattr(target, "official_depth", math.nan))
    row = {
        "tic_id": int(target.tic_id),
        "sector": int(target.sector),
        "official_period": float(target.official_period),
        "official_duration_hours": float(target.official_duration_hours),
        "official_depth_ppm": official_depth_ppm,
        "source_file": "",
    }

    fits_path = _find_lc_fits(args.fits_dir, int(target.tic_id), int(target.sector))
    if fits_path is None:
        row.update(_empty_fit_fields())
        row["fit_status"] = "missing_lc_fits"
        row["message"] = "no LC FITS found for this TIC/sector"
        return row

    try:
        # Pass 1: detect period/epoch/duration on the standard flattened light curve.
        clean_lc = load_clean_flattened_lightcurve(fits_path)
        bls = run_bls_search(
            clean_lc,
            min_period=args.min_period,
            max_period=args.max_period,
            n_periods=args.n_periods,
        )
        # Pass 2: re-flatten with the detected transit masked, so depth is not suppressed,
        # then fit the trapezoid on that depth-preserving flux using the BLS ephemeris.
        fit_lc = load_clean_flattened_lightcurve(
            fits_path,
            mask_period=bls.best_period,
            mask_t0=bls.best_t0,
            mask_duration_days=bls.best_duration,
            mask_width_durations=1.0,
        )
        fit = fit_transit(
            bls,
            min_period=args.min_period,
            max_period=args.max_period,
            n_periods=args.n_periods,
            time_override=fit_lc.time,
            flux_override=fit_lc.flux,
        )
        row.update(fit.as_row())
        row["source_file"] = fits_path.name
        row["depth_ratio_fit_vs_official"] = _safe_ratio(row["fit_depth_ppm"], official_depth_ppm)
        row["duration_ratio_fit_vs_official"] = _safe_ratio(
            row["fit_duration_hours"], row["official_duration_hours"]
        )
    except Exception as exc:  # noqa: BLE001
        row.update(_empty_fit_fields())
        row["fit_status"] = "processing_failed"
        row["message"] = f"{type(exc).__name__}: {exc}"
        row["source_file"] = fits_path.name
    return row


def _find_lc_fits(fits_dir: Path, tic_id: int, sector: int) -> Path | None:
    pattern = f"*-s{sector:04d}-{tic_id:016d}-*_lc.fits"
    matches = sorted(fits_dir.rglob(pattern))
    return matches[0] if matches else None


def _safe_ratio(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
        return math.nan
    return a / b


def _empty_fit_fields() -> dict:
    nan = math.nan
    return {
        "fit_period_days": nan, "fit_period_err_days": nan,
        "fit_t0_btjd": nan, "fit_t0_err_days": nan,
        "fit_depth_ppm": nan, "fit_depth_err_ppm": nan,
        "fit_duration_hours": nan, "fit_duration_err_hours": nan,
        "fit_ingress_frac": nan, "fit_ingress_frac_err": nan,
        "depth_ratio_fit_vs_official": nan, "duration_ratio_fit_vs_official": nan,
        "reduced_chi2": nan, "bic": nan, "n_points_fit": 0,
        "message": "",
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def _print_summary(rows: list[dict], output_path: Path) -> None:
    ok = sum(1 for r in rows if r.get("fit_status") == "ok")
    missing = sum(1 for r in rows if r.get("fit_status") == "missing_lc_fits")
    failed = sum(1 for r in rows if r.get("fit_status") in {"failed", "processing_failed"})
    print("\n" + "=" * 55)
    print("  DRISHTI Transit Fit Summary")
    print("=" * 55)
    print(f"  Rows:               {len(rows)}")
    print(f"  Fit ok:             {ok}")
    print(f"  Missing LC FITS:    {missing}")
    print(f"  Failed:             {failed}")
    print(f"  Output: {output_path.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    raise SystemExit(main())
