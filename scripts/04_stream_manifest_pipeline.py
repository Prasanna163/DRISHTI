from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass(frozen=True)
class ManifestItem:
    index: int
    filename: str
    url: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a TESS curl manifest: download one FITS, process figures/CSVs, "
            "then delete the FITS to keep disk use low."
        )
    )
    parser.add_argument("manifest", type=Path, help="Path to tesscurl-style .sh manifest.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output root.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("outputs/download_cache"),
        help="Temporary FITS download directory. Files are deleted after processing by default.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution for generated plots.")
    parser.add_argument("--min-period", type=float, default=0.5, help="Minimum BLS period in days.")
    parser.add_argument("--max-period", type=float, default=13.0, help="Maximum BLS period in days.")
    parser.add_argument("--n-periods", type=int, default=20000, help="Number of periods in BLS grid.")
    parser.add_argument("--phase-bins", type=int, default=150, help="Number of folded phase bins.")
    parser.add_argument("--timeout", type=int, default=120, help="Network timeout in seconds.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N FITS items.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based manifest FITS item index to start at.")
    parser.add_argument("--keep-fits", action="store_true", help="Do not delete downloaded FITS files.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an item if its per-target output folder already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the manifest and print what would run without downloading anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = parse_manifest(args.manifest)
    items = [item for item in items if item.index >= args.start_index]
    if args.limit is not None:
        items = items[: args.limit]

    if args.dry_run:
        print(f"Manifest FITS items selected: {len(items)}")
        if items:
            print(f"First: #{items[0].index} {items[0].filename}")
            print(f"Last:  #{items[-1].index} {items[-1].filename}")
        return 0

    dirs = output_dirs(args.output_root)
    for directory in [args.download_dir, *dirs.values()]:
        directory.mkdir(parents=True, exist_ok=True)

    status_log = args.output_root / "tables" / "stream_manifest_status.csv"
    failures = 0
    successes = 0

    progress = tqdm(items, desc="FITS products", unit="file")
    for item in progress:
        progress.set_postfix_str(item.filename[:40])
        fits_path = args.download_dir / item.filename
        partial_path = fits_path.with_suffix(fits_path.suffix + ".part")
        item_status = "started"
        message = ""

        try:
            from data_access.load_fits import parse_tess_product_meta, validate_fits

            meta = parse_tess_product_meta(Path(item.filename))
            if args.skip_existing and meta and (dirs["figures"] / Path(item.filename).stem).exists():
                item_status = "skipped_existing"
                successes += 1
                append_status(status_log, item, item_status, message)
                continue

            download_file(item.url, partial_path, timeout=args.timeout)
            partial_path.replace(fits_path)
            validate_fits(fits_path)

            process_fits(
                fits_path,
                dirs=dirs,
                dpi=args.dpi,
                min_period=args.min_period,
                max_period=args.max_period,
                n_periods=args.n_periods,
                phase_bins=args.phase_bins,
            )
            item_status = "processed"
            successes += 1
        except Exception as exc:
            failures += 1
            item_status = "failed"
            message = f"{type(exc).__name__}: {exc}"
            tqdm.write(f"[FAIL] {item.filename}: {message}")
        finally:
            if not args.keep_fits:
                remove_if_present(fits_path)
                remove_if_present(partial_path)
            append_status(status_log, item, item_status, message)

    print(f"Processed: {successes}")
    print(f"Failed: {failures}")
    print(f"Status log: {status_log.resolve()}")
    return 1 if failures else 0


def parse_manifest(manifest_path: Path) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    pattern = re.compile(r"^curl\s+-C\s+-\s+-L\s+-o\s+(\S+)\s+(\S+)")
    for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        filename, url = match.groups()
        if filename.lower().endswith(".fits") and url.lower().endswith(".fits"):
            items.append(ManifestItem(index=len(items) + 1, filename=filename, url=url))
    return items


def output_dirs(output_root: Path) -> dict[str, Path]:
    return {
        "figures": output_root / "inspection_figures",
        "plots_periodograms": output_root / "plots" / "periodograms",
        "plots_phase": output_root / "plots" / "phase_folded",
        "cleaned": output_root / "lightcurves_cleaned",
        "tables": output_root / "tables",
        "candidates": output_root / "candidates",
        "periodograms_csv": output_root / "periodograms_csv",
        "folded": output_root / "folded",
        "folded_binned": output_root / "folded_binned",
    }


def download_file(url: str, destination: Path, *, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as handle:
            with tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
                leave=False,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bar.update(len(chunk))


def process_fits(
    fits_path: Path,
    *,
    dirs: dict[str, Path],
    dpi: int,
    min_period: float,
    max_period: float,
    n_periods: int,
    phase_bins: int,
) -> None:
    from detection.run_bls import run_bls_search
    from features.quantitative_products import (
        build_cleaned_lightcurve_table,
        candidate_row,
        folded_binned_table,
        folded_table,
        periodogram_table,
    )
    from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve
    from visualization.plot_bls_results import (
        plot_binned_phase_folded,
        plot_bls_periodogram,
        plot_phase_folded,
    )
    from visualization.plot_fits_inspection import generate_inspection_figures

    figure_result = generate_inspection_figures(fits_path, dirs["figures"], dpi=dpi)
    if figure_result.error:
        raise RuntimeError(f"figure generation failed: {figure_result.error}")

    if not fits_path.name.lower().endswith("_lc.fits"):
        return

    product = build_cleaned_lightcurve_table(fits_path)
    label = product.meta.label
    product.table.to_csv(
        dirs["cleaned"] / f"{label}_cleaned.csv.gz",
        index=False,
        compression="gzip",
    )
    append_row(dirs["tables"] / "lightcurve_summary.csv", product.summary)

    clean_lc = load_clean_flattened_lightcurve(fits_path)
    bls = run_bls_search(
        clean_lc,
        min_period=min_period,
        max_period=max_period,
        n_periods=n_periods,
    )

    periodogram_table(bls).to_csv(
        dirs["periodograms_csv"] / f"{label}_bls_periodogram.csv",
        index=False,
    )
    folded_table(product.meta, bls).to_csv(
        dirs["folded"] / f"{label}_BLS_candidate001_folded.csv",
        index=False,
    )
    folded_binned_table(product.meta, bls, bins=phase_bins).to_csv(
        dirs["folded_binned"] / f"{label}_BLS_candidate001_folded_binned.csv",
        index=False,
    )

    plot_bls_periodogram(fits_path, bls, dirs["plots_periodograms"], dpi=dpi)
    plot_phase_folded(fits_path, bls, dirs["plots_phase"], dpi=dpi)
    plot_binned_phase_folded(fits_path, bls, dirs["plots_phase"], dpi=dpi, bins=phase_bins)

    candidate = candidate_row(
        product.meta,
        fits_path,
        bls,
        min_period=min_period,
        max_period=max_period,
    )
    append_row(dirs["candidates"] / "bls_candidates.csv", candidate)


def append_status(status_log: Path, item: ManifestItem, status: str, message: str) -> None:
    append_row(
        status_log,
        {
            "manifest_index": item.index,
            "filename": item.filename,
            "url": item.url,
            "status": status,
            "message": message,
        },
    )


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
