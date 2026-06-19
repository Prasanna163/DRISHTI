"""DRISHTI Pipeline Runner — Simple convenience wrapper.

This is the easiest way to run the TCE recovery pipeline.
It wraps `drishti.py tce-recovery` with sensible defaults.

Examples:
    # Run first 50 targets (starter batch)
    python scripts/run_pipeline.py --batch-size 50

    # Run next 50 targets
    python scripts/run_pipeline.py --batch-size 50 --batch-offset 50

    # Run all 111 starter targets
    python scripts/run_pipeline.py --batch-size 111

    # Resume an interrupted run
    python scripts/run_pipeline.py --batch-size 111 --resume

    # Skip download (reuse existing FITS files)
    python scripts/run_pipeline.py --batch-size 111 --skip-download

    # Dry run (preview what would happen)
    python scripts/run_pipeline.py --batch-size 50 --dry-run

    # Full 1538-row expansion (after starter is validated)
    python scripts/run_pipeline.py --batch-size 1538 --skip-download
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRISHTI_SCRIPT = ROOT / "scripts" / "drishti.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DRISHTI Pipeline Runner - simple batch orchestration."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of targets to process in this run",
    )
    parser.add_argument(
        "--batch-offset",
        type=int,
        default=0,
        help="Index of target to start processing from",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess targets even if they have already been evaluated and plotted",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=True,
        help="Skip downloading FITS files (process local files only)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=True,
        help="Skip diagnostic plot generation",
    )
    parser.add_argument(
        "--skip-discover",
        action="store_true",
        default=True,
        help="Skip STScI online index discovery",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be executed without running them",
    )

    args = parser.parse_args(argv)

    command = [
        sys.executable,
        str(DRISHTI_SCRIPT),
        "tce-recovery",
        "--batch-size",
        str(args.batch_size),
        "--batch-offset",
        str(args.batch_offset),
    ]
    if args.force:
        command.append("--force")
    if args.skip_download:
        command.append("--skip-download")
    if args.skip_plots:
        command.append("--skip-plots")
    if args.skip_discover:
        command.append("--skip-discover")
    if args.dry_run:
        command.append("--dry-run")

    pretty = " ".join(command)
    print(f"Running: {pretty}\n", flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
