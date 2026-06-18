from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_access.load_fits import find_fits
from visualization.plot_fits_inspection import generate_inspection_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate first-pass inspection figures for TESS FITS files."
    )
    parser.add_argument("input", type=Path, help="A .fits file or folder containing .fits files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inspection_figures"),
        help="Output directory for generated figures.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fits_files = find_fits(args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    results = [
        generate_inspection_figures(path, args.output, dpi=args.dpi)
        for path in fits_files
    ]

    generated_count = sum(len(result.figures) for result in results)
    failed_count = sum(1 for result in results if result.error)

    print(f"Inspected {len(results)} FITS file(s).")
    print(f"Generated {generated_count} figure(s) under: {args.output.resolve()}")
    for result in results:
        if result.error:
            print(f"[SKIP] {result.fits_path.name}: {result.error}")
        elif result.warnings:
            print(f"[WARN] {result.fits_path.name}: {'; '.join(result.warnings)}")
        else:
            print(f"[OK] {result.fits_path.name}: {len(result.figures)} figure(s)")

    return 1 if failed_count == len(results) and results else 0


if __name__ == "__main__":
    raise SystemExit(main())

