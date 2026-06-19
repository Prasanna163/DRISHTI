"""Run crowded-field vetting evidence on downloaded TCE targets (WS-D1, partial WS-D2).

For each target this computes:
- centroid_shift : in-transit vs out-of-transit centroid offset and its significance, using the
  official ephemeris (flags likely blends / background eclipsing binaries).
- crowding context: CROWDSAP and FLFRCSAP from the LC FITS header (fraction of aperture flux that
  is the target, and fraction of target flux captured). Low CROWDSAP => crowded aperture.

These columns form the first slice of the per-candidate evidence vector used downstream by the
classifier. All inputs already live in the standard LC FITS — no target-pixel download needed.

Examples
--------
    python scripts/09_run_vetting.py --limit 5
    python scripts/09_run_vetting.py --targets data/drishti/targets/tce_recovery_batch_111.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import pandas as pd
from astropy.io import fits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vetting.centroid_shift import measure_centroid_shift
from vetting.odd_even import measure_odd_even
from vetting.secondary_eclipse import measure_secondary_eclipse
from vetting.duration_sanity import assess_duration
from vetting.local_shape import measure_local_shape
from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve
from drishti_store import DOWNLOAD_LC_ROOT, RESULT_TABLE_ROOT, TARGET_ROOT

DEFAULT_TARGETS = TARGET_ROOT / "tce_recovery_batch_111.csv"
DEFAULT_FITS_DIR = DOWNLOAD_LC_ROOT
DEFAULT_OUTPUT = RESULT_TABLE_ROOT / "vetting_features.csv"

FIELDNAMES = [
    "tic_id", "sector",
    "official_period", "official_epoch", "official_duration_hours",
    "crowdsap", "flfrcsap",
    # centroid / blend (WS-D1)
    "centroid_status", "centroid_source",
    "centroid_shift_pixels", "centroid_shift_sigma",
    "centroid_dcol_pixels", "centroid_drow_pixels",
    "centroid_n_in_transit", "centroid_n_out_transit",
    "centroid_on_target", "blend_flag",
    # light-curve shape / eclipse evidence (WS-C)
    "lc_vetting_status",
    "oddeven_diff_sigma", "oddeven_depth_frac_diff", "depth_odd_ppm", "depth_even_ppm", "oddeven_flag",
    "primary_depth_ppm", "secondary_depth_ppm", "secondary_snr",
    "secondary_phase", "secondary_to_primary_ratio", "secondary_flag",
    "v_shape_metric", "shape_flag",
    "expected_duration_hours", "duration_sanity_ratio", "duration_flag",
    "eb_flag",
    "source_file", "centroid_message",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run crowded-field vetting (centroid shift + crowding).")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--centroid-sigma", type=float, default=3.0,
                        help="Centroid shift significance (sigma) required to flag off-target.")
    parser.add_argument("--centroid-min-shift", type=float, default=0.05,
                        help="Minimum physical centroid shift in pixels required to flag off-target.")
    parser.add_argument("--force", action="store_true", help="Recompute all (default: skip rows already ok).")
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
        for row in existing_df[existing_df["centroid_status"] == "ok"].itertuples(index=False):
            already_done.add((int(row.tic_id), int(row.sector)))
        existing_rows = existing_df.to_dict("records")
        if already_done:
            print(f"Found {len(already_done)} already-vetted target(s), skipping them.", flush=True)

    rows = list(existing_rows)
    pending = [
        t for t in targets.itertuples(index=False)
        if (int(t.tic_id), int(t.sector)) not in already_done
    ]
    print(f"Running vetting for {len(pending)} target row(s) ({len(already_done)} skipped)...", flush=True)

    for target in tqdm(pending, desc="Vetting", unit="target", file=sys.stdout):
        rows.append(vet_one_target(target, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output, rows)
    _print_summary(rows, args.output)
    return 0


def vet_one_target(target, args) -> dict:
    row = {
        "tic_id": int(target.tic_id),
        "sector": int(target.sector),
        "official_period": float(target.official_period),
        "official_epoch": float(target.official_epoch),
        "official_duration_hours": float(target.official_duration_hours),
        "crowdsap": math.nan,
        "flfrcsap": math.nan,
        "blend_flag": "",
        "source_file": "",
    }

    fits_path = _find_lc_fits(args.fits_dir, int(target.tic_id), int(target.sector))
    if fits_path is None:
        row["centroid_status"] = "missing_lc_fits"
        row["centroid_source"] = ""
        row["centroid_on_target"] = False
        row["centroid_message"] = "no LC FITS found"
        return _fill_missing(row)

    row["source_file"] = fits_path.name
    row["crowdsap"], row["flfrcsap"] = _read_crowding(fits_path)

    result = measure_centroid_shift(
        fits_path,
        period=float(target.official_period),
        epoch=float(target.official_epoch),
        duration_hours=float(target.official_duration_hours),
        significance_threshold=args.centroid_sigma,
        min_shift_pixels=args.centroid_min_shift,
    )
    row.update(result.as_row())
    row["blend_flag"] = _blend_flag(result, row["crowdsap"])

    # Duration sanity needs no light curve (physics-based on period/duration).
    dur = assess_duration(
        period_days=float(target.official_period),
        duration_hours=float(target.official_duration_hours),
    )
    row.update(dur.as_row())

    # Light-curve shape / eclipse evidence (needs the cleaned, folded light curve).
    _add_lc_shape_evidence(row, fits_path, target)
    row["eb_flag"] = _eb_flag(row)
    return _fill_missing(row)


def _add_lc_shape_evidence(row: dict, fits_path: Path, target) -> None:
    period = float(target.official_period)
    t0 = float(target.official_epoch)
    duration_days = float(target.official_duration_hours) / 24.0
    try:
        lc = load_clean_flattened_lightcurve(fits_path)
        time, flux = lc.time, lc.flux
        row.update(measure_odd_even(
            time, flux, period=period, t0=t0, duration_days=duration_days).as_row())
        row.update(measure_secondary_eclipse(
            time, flux, period=period, t0=t0, duration_days=duration_days).as_row())
        row.update(measure_local_shape(
            time, flux, period=period, t0=t0, duration_days=duration_days).as_row())
        row["lc_vetting_status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["lc_vetting_status"] = f"failed: {type(exc).__name__}"


def _eb_flag(row: dict) -> str:
    """Aggregate EB verdict from the strongest light-curve discriminators."""
    if row.get("lc_vetting_status") != "ok":
        return "unknown"
    if row.get("secondary_flag") == "eb_suspect" or row.get("oddeven_flag") == "eb_suspect":
        return "eb_suspect"
    if row.get("shape_flag") == "v_shaped":
        return "v_shaped_watch"
    return "ok"


def _blend_flag(result, crowdsap: float) -> str:
    """Coarse, transparent blend verdict combining centroid + crowding."""
    if result.centroid_status != "ok":
        return "unknown"
    if not result.centroid_on_target:
        return "likely_blend"          # significant centroid shift => off-target flux
    if math.isfinite(crowdsap) and crowdsap < 0.5:
        return "crowded_on_target"     # on-target shift, but aperture is heavily diluted
    return "on_target"


def _read_crowding(fits_path: Path) -> tuple[float, float]:
    try:
        with fits.open(fits_path) as hdul:
            hdr = hdul[1].header
            crowdsap = float(hdr.get("CROWDSAP", math.nan))
            flfrcsap = float(hdr.get("FLFRCSAP", math.nan))
            return crowdsap, flfrcsap
    except Exception:  # noqa: BLE001
        return math.nan, math.nan


def _find_lc_fits(fits_dir: Path, tic_id: int, sector: int) -> Path | None:
    pattern = f"*-s{sector:04d}-{tic_id:016d}-*_lc.fits"
    matches = sorted(fits_dir.rglob(pattern))
    return matches[0] if matches else None


def _fill_missing(row: dict) -> dict:
    for key in FIELDNAMES:
        row.setdefault(key, "")
    return row


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def _print_summary(rows: list[dict], output_path: Path) -> None:
    flags: dict[str, int] = {}
    for r in rows:
        f = r.get("blend_flag", "") or "unknown"
        flags[f] = flags.get(f, 0) + 1
    eb: dict[str, int] = {}
    for r in rows:
        f = r.get("eb_flag", "") or "unknown"
        eb[f] = eb.get(f, 0) + 1

    print("\n" + "=" * 55)
    print("  DRISHTI Vetting Summary")
    print("=" * 55)
    print(f"  Rows: {len(rows)}")
    print("  blend_flag (centroid + crowding):")
    for f in sorted(flags, key=lambda k: -flags[k]):
        print(f"    {f:22s} {flags[f]:>4d}")
    print("  eb_flag (odd/even + secondary + shape):")
    for f in sorted(eb, key=lambda k: -eb[k]):
        print(f"    {f:22s} {eb[f]:>4d}")
    print(f"  Output: {output_path.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    raise SystemExit(main())
