"""Finalize candidates: master catalog with class, calibrated confidence, and noise context (WS-G).

Combines all per-candidate evidence (recovery + transit fit + vetting), then adds:
  - rule-based class + confidence + reason (transparent baseline, all candidates)
  - ML planet probability from the trained RandomForest (applied to every candidate)
  - CDPP noise context: the star's intrinsic photometric precision at the transit duration, and a
    noise-relative significance (depth / CDPP) that explains *why* a faint signal is uncertain and
    separates genuine misses from noise-limited ones (the problem statement's "noisy light curves").
A calibration (reliability) check of the ML probabilities against the bootstrapped labels is printed.

Output: data/drishti/results/tables/master_candidates.csv

Example
-------
    python scripts/12_finalize_candidates.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.classifier import classify_candidate
from drishti_store import RESULT_TABLE_ROOT

T = RESULT_TABLE_ROOT
REF = ROOT / "data" / "Ref"
KEYS = ["tic_id", "sector"]

# CDPP columns are RMS CDPP (ppm) at these transit durations (hours).
CDPP_COLS = {
    0.5: "rrmscdpp00p5", 1.0: "rrmscdpp01p0", 1.5: "rrmscdpp01p5", 2.0: "rrmscdpp02p0",
    2.5: "rrmscdpp02p5", 3.0: "rrmscdpp03p0", 3.5: "rrmscdpp03p5", 4.5: "rrmscdpp04p5",
    5.0: "rrmscdpp05p0", 6.0: "rrmscdpp06p0", 7.5: "rrmscdpp07p5", 9.0: "rrmscdpp09p0",
    10.5: "rrmscdpp10p5", 12.5: "rrmscdpp12p5", 15.0: "rrmscdpp15p0",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the final candidate catalog with calibrated confidence.")
    p.add_argument("--recovery", type=Path, nargs="+",
                   default=[T / "tce_recovery_results_111.csv", T / "tce_recovery_results_labeled.csv"])
    p.add_argument("--fits", type=Path, nargs="+",
                   default=[T / "transit_fits_111.csv", T / "transit_fits_labeled.csv"])
    p.add_argument("--vetting", type=Path, nargs="+",
                   default=[T / "vetting_features_111.csv", T / "vetting_features_labeled.csv"])
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "raw" / "labels" / "labels.csv")
    p.add_argument("--model", type=Path, default=ROOT / "outputs" / "models" / "planet_eb_rf.joblib")
    p.add_argument("--output", type=Path, default=T / "master_candidates.csv")
    return p


def _concat(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
    if not frames:
        return pd.DataFrame(columns=KEYS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=KEYS, keep="first")


def load_cdpp() -> pd.DataFrame:
    frames = []
    for f in sorted(REF.glob("*_rms-cdpp.csv")):
        df = pd.read_csv(f, comment="#", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ticid"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("ticid", keep="first")


def cdpp_for_duration(cdpp_row, duration_hours: float) -> float:
    if cdpp_row is None or not np.isfinite(duration_hours):
        return float("nan")
    nearest = min(CDPP_COLS, key=lambda d: abs(d - duration_hours))
    col = CDPP_COLS[nearest]
    return float(cdpp_row.get(col, np.nan))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import joblib

    merged = _concat(args.recovery)
    for paths in (args.fits, args.vetting):
        extra = _concat(paths)
        if not extra.empty:
            dup = [c for c in extra.columns if c in merged.columns and c not in KEYS]
            merged = merged.merge(extra.drop(columns=dup), on=KEYS, how="left")
    labels = pd.read_csv(args.labels)[["tic_id", "class_label"]]
    merged = merged.merge(labels, on="tic_id", how="left")

    # --- Rule-based class (transparent, every candidate) ---
    res = [classify_candidate(r._asdict()) for r in merged.itertuples(index=False)]
    merged["rule_class"] = [r.predicted_class for r in res]
    merged["rule_confidence"] = [round(r.class_confidence, 3) for r in res]
    merged["rule_reason"] = [r.class_reason for r in res]

    # --- ML planet probability (trained RF) ---
    if args.model.exists():
        bundle = joblib.load(args.model)
        clf, feats = bundle["model"], bundle["features"]
        X = merged.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        merged["ml_planet_proba"] = np.round(clf.predict_proba(X)[:, 1], 3)
    else:
        merged["ml_planet_proba"] = np.nan

    # --- CDPP noise context ---
    cdpp = load_cdpp().set_index("ticid")
    cdpp_ppm, snr_vs_cdpp = [], []
    for r in merged.itertuples(index=False):
        row = cdpp.loc[int(r.tic_id)].to_dict() if int(r.tic_id) in cdpp.index else None
        c = cdpp_for_duration(row, float(getattr(r, "official_duration_hours", np.nan)))
        cdpp_ppm.append(c)
        depth = float(getattr(r, "official_depth_ppm", np.nan))
        snr_vs_cdpp.append(depth / c if (np.isfinite(depth) and np.isfinite(c) and c > 0) else np.nan)
    merged["cdpp_ppm"] = np.round(cdpp_ppm, 1)
    merged["single_transit_snr_vs_cdpp"] = np.round(snr_vs_cdpp, 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [c for c in [
        "tic_id", "sector", "recovery_class", "our_snr", "official_snr",
        "fit_period_days", "fit_depth_ppm", "fit_depth_err_ppm",
        "fit_duration_hours", "fit_duration_err_hours",
        "cdpp_ppm", "single_transit_snr_vs_cdpp",
        "rule_class", "rule_confidence", "ml_planet_proba", "class_label", "rule_reason",
    ] if c in merged.columns]
    merged[out_cols].to_csv(args.output, index=False)

    _print_summary(merged, args.output)
    _calibration_report(merged)
    return 0


def _print_summary(df: pd.DataFrame, output_path: Path) -> None:
    print("\n" + "=" * 64)
    print("  DRISHTI Master Candidate Catalog")
    print("=" * 64)
    print(f"  Candidates: {len(df)}")
    print("  rule_class:")
    for cls, n in df["rule_class"].value_counts().items():
        print(f"    {cls:18s} {n:>4d}")
    nl = df["single_transit_snr_vs_cdpp"].notna().sum()
    print(f"  CDPP noise context attached: {nl}/{len(df)} candidates")
    print(f"  Output: {output_path.resolve()}")


def _calibration_report(df: pd.DataFrame) -> None:
    sub = df[df["class_label"].isin(["planet", "eclipsing_binary"]) & df["ml_planet_proba"].notna()].copy()
    if sub.empty:
        return
    sub["is_planet"] = (sub["class_label"] == "planet").astype(int)
    print("-" * 64)
    print("  ML probability calibration (reliability) on labeled set:")
    print("    proba bin        n    actual planet frac")
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for lo, hi in bins:
        m = (sub["ml_planet_proba"] >= lo) & (sub["ml_planet_proba"] < hi)
        if m.any():
            print(f"    [{lo:.2f},{hi:.2f})    {int(m.sum()):>4d}        {sub.loc[m, 'is_planet'].mean():.2f}")
    print("=" * 64)


if __name__ == "__main__":
    raise SystemExit(main())
