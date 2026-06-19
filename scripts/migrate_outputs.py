"""Migrate legacy DRISHTI outputs into the canonical data/drishti/results/ tree.

Copies CSVs and plot files from the old outputs/ directories into the
consolidated store, and renames originals to .bak so nothing is lost.

Usage:
    python scripts/migrate_outputs.py            # run migration
    python scripts/migrate_outputs.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti_store import migrate_legacy_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy DRISHTI outputs to data/drishti/results/."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without touching files.",
    )
    args = parser.parse_args(argv)

    print("DRISHTI output migration")
    print("=" * 50)
    actions = migrate_legacy_outputs(dry_run=args.dry_run)

    if not actions:
        print("Nothing to migrate — all outputs are already consolidated.")
        return 0

    for action in actions:
        print(f"  {action}")

    print(f"\n{len(actions)} action(s) {'planned' if args.dry_run else 'completed'}.")
    if args.dry_run:
        print("Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
