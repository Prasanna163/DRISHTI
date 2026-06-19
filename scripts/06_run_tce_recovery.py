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
from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve
from drishti_store import DOWNLOAD_LC_ROOT, RESULT_TABLE_ROOT, TARGET_ROOT


DEFAULT_TARGETS = TARGET_ROOT / "tce_first_recovery_batch.csv"
DEFAULT_FITS_DIR = DOWNLOAD_LC_ROOT
DEFAULT_OUTPUT = RESULT_TABLE_ROOT / "tce_recovery_results.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BLS on downloaded official TCE targets and compare against official periods."
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="TCE target CSV.")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR, help="Downloaded LC FITS folder.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Recovery result CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N target rows.")
    parser.add_argument("--min-period", type=float, default=0.5, help="Minimum BLS period in days.")
    parser.add_argument("--max-period", type=float, default=13.0, help="Maximum BLS period in days.")
    parser.add_argument("--n-periods", type=int, default=20000, help="Number of BLS period grid points.")
    parser.add_argument("--period-tolerance-percent", type=float, default=1.0)
    parser.add_argument("--period-vetting-percent", type=float, default=0.1)
    parser.add_argument("--min-our-snr", type=float, default=7.0)
    parser.add_argument("--max-duration-ratio", type=float, default=3.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess ALL targets from scratch (default: skip already-evaluated ones).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = pd.read_csv(args.targets)
    if args.limit is not None:
        targets = targets.head(args.limit)

    # Auto-skip already-evaluated targets (unless --force)
    existing_rows: list[dict] = []
    already_done: set[tuple[int, int]] = set()
    if not args.force and args.output.exists():
        existing_df = pd.read_csv(args.output)
        evaluated_mask = existing_df["status"] == "evaluated"
        for row in existing_df[evaluated_mask].itertuples(index=False):
            already_done.add((int(row.tic_id), int(row.sector)))
        existing_rows = existing_df.to_dict("records")
        if already_done:
            print(f"Found {len(already_done)} already-evaluated target(s), skipping them.", flush=True)

    rows = list(existing_rows)
    pending = [
        target
        for target in targets.itertuples(index=False)
        if (int(target.tic_id), int(target.sector)) not in already_done
    ]
    print(f"Running BLS recovery for {len(pending)} target row(s) ({len(already_done)} skipped)...", flush=True)

    for target in tqdm(pending, desc="Recovery targets", unit="target", file=sys.stdout):
        row = base_recovery_row(target)
        fits_path = find_lc_fits(args.fits_dir, int(target.tic_id), int(target.sector))
        if fits_path is None:
            row["status"] = "missing_lc_fits"
            row["recovery_class"] = "download_failed"
            rows.append(row)
            continue

        try:
            clean_lc = load_clean_flattened_lightcurve(fits_path)
            bls = run_bls_search(
                clean_lc,
                min_period=args.min_period,
                max_period=args.max_period,
                n_periods=args.n_periods,
            )
            row.update(compare_to_official(target, bls))
            row["source_file"] = fits_path.name
            row["status"] = "evaluated"
            row["duration_ratio"] = duration_ratio(
                row["our_duration_hours"],
                row["official_duration_hours"],
            )
            row["recovery_class"] = classify_recovery(
                row,
                period_tolerance_percent=args.period_tolerance_percent,
                period_vetting_percent=args.period_vetting_percent,
                min_our_snr=args.min_our_snr,
                max_duration_ratio=args.max_duration_ratio,
            )
            row["recovered_true_false"] = row["recovery_class"] in {
                "direct_recovered",
                "alias_recovered",
            }
            row["period_recovered_true_false"] = row["recovery_class"] in {
                "direct_recovered",
                "alias_recovered",
                "period_recovered_epoch_mismatch",
                "period_recovered_bad_duration",
                "period_recovered_needs_vetting",
            }
        except Exception as exc:
            row["status"] = "failed"
            row["recovery_class"] = "processing_failed"
            row["message"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, rows)
    print_recovery_summary(rows, len(targets), args.output)
    return 0


def print_recovery_summary(rows: list[dict], total_targets: int, output_path: Path) -> None:
    """Print a detailed recovery rate summary."""
    evaluated = sum(1 for r in rows if r.get("status") == "evaluated")
    recovered = sum(1 for r in rows if r.get("recovered_true_false") is True)
    period_recovered = sum(1 for r in rows if r.get("period_recovered_true_false") is True)
    download_failed = sum(1 for r in rows if r.get("recovery_class") == "download_failed")
    processing_failed = sum(1 for r in rows if r.get("recovery_class") == "processing_failed")
    not_recovered = sum(1 for r in rows if r.get("recovery_class") == "not_recovered")

    # Class breakdown
    class_counts: dict[str, int] = {}
    for r in rows:
        cls = r.get("recovery_class", "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    pct = lambda n, d: f"{100 * n / d:.1f}%" if d > 0 else "N/A"

    print("\n" + "=" * 55)
    print("  DRISHTI TCE Recovery Summary")
    print("=" * 55)
    print(f"  Targets in list:        {total_targets}")
    print(f"  Evaluated (BLS ran):    {evaluated}")
    print(f"  Recovered (direct+alias): {recovered}  ({pct(recovered, evaluated)})")
    print(f"  Period recovered (all):   {period_recovered}  ({pct(period_recovered, evaluated)})")
    print(f"  Not recovered:            {not_recovered}")
    print(f"  Download failed:          {download_failed}")
    print(f"  Processing failed:        {processing_failed}")
    print("-" * 55)
    print("  Recovery class breakdown:")
    for cls in sorted(class_counts, key=lambda c: -class_counts[c]):
        print(f"    {cls:40s} {class_counts[cls]:>4d}")
    print("-" * 55)
    print(f"  Output: {output_path.resolve()}")
    print("=" * 55)


def base_recovery_row(target) -> dict:
    return {
        "tic_id": int(target.tic_id),
        "sector": int(target.sector),
        "official_period": float(target.official_period),
        "our_bls_period": math.nan,
        "period_error_percent": math.nan,
        "best_period_error_percent": math.nan,
        "period_match_type": "",
        "official_epoch": float(target.official_epoch),
        "our_t0": math.nan,
        "epoch_match_score": math.nan,
        "official_duration_hours": float(target.official_duration_hours),
        "our_duration_hours": math.nan,
        "duration_ratio": math.nan,
        "official_snr": float(target.official_snr),
        "our_snr": math.nan,
        "recovery_class": "pending",
        "recovered_true_false": False,
        "period_recovered_true_false": False,
        "status": "pending",
        "source_file": "",
        "message": "",
    }


def find_lc_fits(fits_dir: Path, tic_id: int, sector: int) -> Path | None:
    pattern = f"*-s{sector:04d}-{tic_id:016d}-*_lc.fits"
    matches = sorted(fits_dir.rglob(pattern))
    return matches[0] if matches else None


def compare_to_official(target, bls) -> dict:
    official_period = float(target.official_period)
    official_epoch = float(target.official_epoch)
    official_duration_days = float(target.official_duration_hours) / 24.0
    direct_error = period_error_percent(bls.best_period, official_period)
    best_error, match_type = best_period_error_with_harmonics(bls.best_period, official_period)

    return {
        "our_bls_period": bls.best_period,
        "period_error_percent": direct_error,
        "best_period_error_percent": best_error,
        "period_match_type": match_type,
        "our_t0": bls.best_t0,
        "epoch_match_score": epoch_match_score(bls.best_t0, official_epoch, official_period, official_duration_days),
        "our_duration_hours": bls.best_duration * 24.0,
        "our_snr": bls.snr,
    }


def period_error_percent(candidate_period: float, official_period: float) -> float:
    if official_period <= 0 or not math.isfinite(candidate_period):
        return math.nan
    return abs(candidate_period - official_period) / official_period * 100.0


def best_period_error_with_harmonics(candidate_period: float, official_period: float) -> tuple[float, str]:
    checks = {
        "direct": candidate_period,
        "half_period_alias": candidate_period * 2.0,
        "double_period_alias": candidate_period / 2.0,
    }
    errors = {
        label: period_error_percent(adjusted_period, official_period)
        for label, adjusted_period in checks.items()
    }
    best_label = min(errors, key=lambda label: errors[label])
    return errors[best_label], best_label


def duration_ratio(candidate_duration_hours: float, official_duration_hours: float) -> float:
    if (
        not math.isfinite(candidate_duration_hours)
        or not math.isfinite(official_duration_hours)
        or candidate_duration_hours <= 0
        or official_duration_hours <= 0
    ):
        return math.nan
    ratio = candidate_duration_hours / official_duration_hours
    return max(ratio, 1.0 / ratio)


def classify_recovery(
    row: dict,
    *,
    period_tolerance_percent: float,
    period_vetting_percent: float,
    min_our_snr: float,
    max_duration_ratio: float,
) -> str:
    best_error = float(row["best_period_error_percent"])
    direct_error = float(row["period_error_percent"])
    epoch_score = float(row["epoch_match_score"])
    our_snr = float(row["our_snr"])
    dur_ratio = float(row["duration_ratio"])
    match_type = str(row["period_match_type"])

    period_ok = math.isfinite(best_error) and best_error <= period_tolerance_percent
    direct_ok = match_type == "direct" and math.isfinite(direct_error) and direct_error <= period_tolerance_percent
    vetting_period_ok = math.isfinite(best_error) and best_error < period_vetting_percent
    epoch_ok = math.isfinite(epoch_score) and epoch_score >= 0.5
    snr_ok = math.isfinite(our_snr) and our_snr >= min_our_snr
    duration_bad = math.isfinite(dur_ratio) and dur_ratio > max_duration_ratio

    if direct_ok and epoch_ok and snr_ok and not duration_bad:
        return "direct_recovered"

    if match_type in {"half_period_alias", "double_period_alias"} and period_ok and snr_ok:
        return "alias_recovered"

    if direct_ok and duration_bad:
        return "period_recovered_bad_duration"

    if direct_ok and not epoch_ok:
        return "period_recovered_epoch_mismatch"

    if vetting_period_ok:
        return "period_recovered_needs_vetting"

    return "not_recovered"


def epoch_match_score(
    candidate_t0: float,
    official_epoch: float,
    official_period: float,
    official_duration_days: float,
) -> float:
    if official_period <= 0 or official_duration_days <= 0 or not math.isfinite(candidate_t0):
        return math.nan
    delta = abs(((candidate_t0 - official_epoch + 0.5 * official_period) % official_period) - 0.5 * official_period)
    tolerance = max(official_duration_days, 1e-6)
    return max(0.0, 1.0 - delta / tolerance)


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "tic_id",
        "sector",
        "official_period",
        "our_bls_period",
        "period_error_percent",
        "best_period_error_percent",
        "period_match_type",
        "official_epoch",
        "our_t0",
        "epoch_match_score",
        "official_duration_hours",
        "our_duration_hours",
        "duration_ratio",
        "official_snr",
        "our_snr",
        "recovery_class",
        "recovered_true_false",
        "period_recovered_true_false",
        "status",
        "source_file",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
