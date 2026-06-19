"""Classify candidates into transit / eclipsing-binary / blend / other (WS-F1).

Joins the three per-candidate evidence tables produced upstream:
  - recovery results   (detection: recovery_class, our_snr, period match)   [scripts/06]
  - transit fits       (fitted depth/duration/period + uncertainties)        [scripts/08]
  - vetting features   (centroid/crowding + odd-even/secondary/shape/duration)[scripts/09]
and applies the transparent rule-based classifier to each, writing a class + confidence + reason.

Example
-------
    python scripts/10_classify_candidates.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.classifier import classify_candidate
from drishti_store import RESULT_TABLE_ROOT

DEFAULT_RECOVERY = RESULT_TABLE_ROOT / "tce_recovery_results_111.csv"
DEFAULT_FITS = RESULT_TABLE_ROOT / "transit_fits_111.csv"
DEFAULT_VETTING = RESULT_TABLE_ROOT / "vetting_features_111.csv"
DEFAULT_OUTPUT = RESULT_TABLE_ROOT / "candidate_classifications_111.csv"

KEYS = ["tic_id", "sector"]
OUTPUT_COLS = [
    "tic_id", "sector", "recovery_class", "our_snr",
    "fit_period_days", "fit_depth_ppm", "fit_duration_hours",
    "blend_flag", "eb_flag", "oddeven_flag", "secondary_flag", "shape_flag", "duration_flag",
    "predicted_class", "class_confidence", "class_reason",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based candidate classification.")
    parser.add_argument("--recovery", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--vetting", type=Path, default=DEFAULT_VETTING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-snr", type=float, default=7.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    rec = pd.read_csv(args.recovery)
    merged = rec.copy()
    for path, label in [(args.fits, "fits"), (args.vetting, "vetting")]:
        if path.exists():
            extra = pd.read_csv(path)
            dup = [c for c in extra.columns if c in merged.columns and c not in KEYS]
            merged = merged.merge(extra.drop(columns=dup), on=KEYS, how="left")
        else:
            print(f"WARNING: {label} table not found at {path}; classifying with available evidence.")

    results = [classify_candidate(row._asdict(), min_snr=args.min_snr)
               for row in merged.itertuples(index=False)]
    merged["predicted_class"] = [r.predicted_class for r in results]
    merged["class_confidence"] = [round(r.class_confidence, 3) for r in results]
    merged["class_reason"] = [r.class_reason for r in results]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in OUTPUT_COLS if c in merged.columns]
    merged[cols].to_csv(args.output, index=False)
    _print_summary(merged, args.output)
    return 0


def _print_summary(df: pd.DataFrame, output_path: Path) -> None:
    print("\n" + "=" * 60)
    print("  DRISHTI Candidate Classification Summary")
    print("=" * 60)
    print(f"  Candidates: {len(df)}")
    counts = df["predicted_class"].value_counts()
    for cls, n in counts.items():
        conf = df.loc[df["predicted_class"] == cls, "class_confidence"]
        print(f"    {cls:18s} {n:>4d}   (median confidence {conf.median():.2f})")
    print("-" * 60)
    print("  predicted_class vs recovery_class:")
    print(pd.crosstab(df["recovery_class"], df["predicted_class"]).to_string())
    print("-" * 60)
    print(f"  Output: {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
