"""DRISHTI — One-command end-to-end pipeline.

Just run:
    python scripts/run_all.py

That's it. It will:
  1. Migrate any legacy outputs
  2. Select targets from reference TCE CSVs
  3. Download TESS light curves
  4. Run DRISHTI cleaning + BLS transit search
  5. Compare against official periods/epochs
  6. Generate diagnostic plots
  7. Print recovery summary

Options:
    python scripts/run_all.py                  # starter 111 targets
    python scripts/run_all.py --full           # all 1538 TCE-positive targets
    python scripts/run_all.py --force          # reprocess everything from scratch
    python scripts/run_all.py --skip-download  # reuse existing FITS files
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti_store import (
    RESULT_TABLE_ROOT,
    TARGET_ROOT,
    migrate_legacy_outputs,
)

DRISHTI_SCRIPT = ROOT / "scripts" / "drishti.py"
SELECT_SCRIPT = ROOT / "scripts" / "select_tce_targets.py"

STARTER_TARGETS = TARGET_ROOT / "tce_starter_validation_targets.csv"
FULL_TARGETS = TARGET_ROOT / "tce_positive_targets.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DRISHTI end-to-end pipeline — one command, no fuss.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process ALL filtered TCE-positive targets (e.g. 1538).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process EVERY SINGLE TCE target present in reference CSVs (no threshold filters).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess everything from scratch (default: skip already-done work).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing FITS files, don't download new ones.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip diagnostic plot generation (faster).",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    print_banner()

    # Step 1: Migrate legacy outputs
    print("\n[1/6] Migrating legacy outputs...", flush=True)
    actions = migrate_legacy_outputs(dry_run=False)
    if actions:
        print(f"  Migrated {len(actions)} file(s)")
    else:
        print("  Nothing to migrate — already consolidated")

    # Step 2: Select targets
    print("\n[2/6] Selecting targets from reference TCE CSVs...", flush=True)
    select_cmd = [sys.executable, str(SELECT_SCRIPT)]
    if args.all:
        select_cmd.extend([
            "--allow-nonconverged",
            "--min-snr", "0",
            "--min-transits", "0",
            "--keep-multiple-tces",
        ])
    result = subprocess.run(
        select_cmd,
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("  ERROR: Target selection failed")
        return 1

    # Step 3: Determine batch size from actual target count
    target_file = FULL_TARGETS if (args.full or args.all) else STARTER_TARGETS
    if not target_file.exists():
        print(f"  ERROR: Target file not found: {target_file}")
        return 1

    import pandas as pd
    target_count = len(pd.read_csv(target_file))
    if args.all:
        mode_label = "ALL (every TCE on site)"
    elif args.full:
        mode_label = "FULL (all TCE-positive)"
    else:
        mode_label = "STARTER (validation set)"
    print(f"\n  Mode: {mode_label}")
    print(f"  Targets: {target_count}")

    # Step 4–6: Run the pipeline with the full batch
    print(f"\n[3/6] Downloading TESS light curves...", flush=True)
    print(f"[4/6] Running DRISHTI cleaning + BLS...", flush=True)
    print(f"[5/6] Comparing against official periods...", flush=True)
    print(f"[6/6] Generating diagnostic plots...", flush=True)
    print(f"\n  (Steps 3-6 run together via drishti.py tce-recovery)\n", flush=True)

    command = [
        sys.executable,
        str(DRISHTI_SCRIPT),
        "tce-recovery",
        "--batch-size",
        str(target_count),
        "--skip-discover",  # targets already selected above
    ]
    if args.force:
        command.append("--force")
    if args.skip_download:
        command.append("--skip-download")
    if args.skip_plots:
        command.append("--skip-plots")

    result = subprocess.run(command, cwd=ROOT)

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nTotal time: {minutes}m {seconds}s")

    if result.returncode == 0:
        print("\n✓ Pipeline complete. Check data/drishti/results/ for outputs.")
        print(f"\n✗ Pipeline exited with code {result.returncode}")
        print("  Tip: re-run the command to resume (it automatically skips completed targets). Use --force to reprocess all.")

    return result.returncode


def print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════╗
║           DRISHTI End-to-End Pipeline                ║
║  TCE Target → Download → Clean → BLS → Recovery     ║
╚══════════════════════════════════════════════════════╝""")


if __name__ == "__main__":
    raise SystemExit(main())
