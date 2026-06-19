from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import requests
from astroquery.mast import Catalogs, Observations
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_access.load_fits import validate_fits
from drishti_store import DOWNLOAD_ROOT, RESULT_TABLE_ROOT, TARGET_ROOT


DEFAULT_TARGETS = TARGET_ROOT / "tce_first_recovery_batch.csv"
DEFAULT_OUTPUT_DIR = DOWNLOAD_ROOT
DEFAULT_STATUS = RESULT_TABLE_ROOT / "tce_download_status.csv"

PRODUCT_SUFFIXES = {
    "lc": "_lc.fits",
    "tp": "_tp.fits",
    "dvr-pdf": "_dvr.pdf",
    "dvr-xml": "_dvr.xml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download selected TESS products for a filtered TCE target list."
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=DEFAULT_TARGETS,
        help="Target CSV produced by select_tce_targets.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root folder for downloaded products.",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS,
        help="CSV download status log.",
    )
    parser.add_argument(
        "--products",
        default="lc",
        help="Comma-separated product types: lc,tp,dvr-pdf,dvr-xml.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N target rows.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-file download timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve products without downloading files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    product_types = parse_product_types(args.products)

    targets = pd.read_csv(args.targets)
    required = {"tic_id", "sector", "source_tce_file"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"{args.targets} is missing required columns: {', '.join(sorted(missing))}")
    if args.limit is not None:
        targets = targets.head(args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)

    product_cache: dict[int, pd.DataFrame] = {}
    status_rows = []

    for row in tqdm(list(targets.itertuples(index=False)), desc="TCE targets", unit="target"):
        tic_id = int(row.tic_id)
        sector = int(row.sector)
        source_tce_file = str(row.source_tce_file)

        try:
            products = product_cache.get(tic_id)
            if products is None:
                products = query_tess_products_for_tic(tic_id)
                product_cache[tic_id] = products

            selected = select_products(
                products,
                tic_id=tic_id,
                sector=sector,
                source_tce_file=source_tce_file,
                product_types=product_types,
            )
            if selected.empty:
                status_rows.append(status_row(row, "", "", "", "missing", "No matching product found."))
                continue

            for product in selected.itertuples(index=False):
                product_type = str(product.requested_product_type)
                filename = str(product.productFilename)
                data_uri = str(product.dataURI)
                destination = args.output_dir / product_type / filename
                message = ""
                status = "planned" if args.dry_run else "downloaded"

                try:
                    if not args.dry_run:
                        if destination.exists() and destination.stat().st_size > 0:
                            status = "exists"
                        else:
                            download_mast_uri(data_uri, destination, timeout=args.timeout)
                        if product_type in {"lc", "tp"}:
                            validate_fits(destination)
                except Exception as exc:
                    status = "failed"
                    message = f"{type(exc).__name__}: {exc}"

                status_rows.append(status_row(row, product_type, filename, destination, status, message))
        except Exception as exc:
            status_rows.append(status_row(row, "", "", "", "failed", f"{type(exc).__name__}: {exc}"))

    write_status(args.status, status_rows)
    failures = sum(1 for row in status_rows if row["status"] == "failed")
    missing_count = sum(1 for row in status_rows if row["status"] == "missing")
    print(f"Targets read: {len(targets)}")
    print(f"Product rows logged: {len(status_rows)}")
    print(f"Failures: {failures}")
    print(f"Missing: {missing_count}")
    print(f"Status log: {args.status.resolve()}")
    return 1 if failures else 0


def parse_product_types(value: str) -> list[str]:
    product_types = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(product_types) - set(PRODUCT_SUFFIXES))
    if unknown:
        raise ValueError(f"Unknown product type(s): {', '.join(unknown)}")
    if not product_types:
        raise ValueError("At least one product type is required.")
    return product_types


def query_tess_products_for_tic(tic_id: int) -> pd.DataFrame:
    tic_rows = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
    if len(tic_rows) == 0:
        raise ValueError(f"TIC {tic_id} was not found in the TIC catalog.")

    ra = float(tic_rows["ra"][0])
    dec = float(tic_rows["dec"][0])
    observations = Observations.query_criteria(
        obs_collection="TESS",
        s_ra=[ra - 0.001, ra + 0.001],
        s_dec=[dec - 0.001, dec + 0.001],
    )
    if len(observations) == 0:
        return pd.DataFrame()

    products = Observations.get_product_list(observations)
    return products.to_pandas()


def select_products(
    products: pd.DataFrame,
    *,
    tic_id: int,
    sector: int,
    source_tce_file: str,
    product_types: list[str],
) -> pd.DataFrame:
    if products.empty:
        return products

    tic_tag = f"{tic_id:016d}"
    sector_tag = f"s{sector:04d}"
    source_prefix = source_tce_file.replace("_dvr-tcestats.csv", "")
    selected = []

    for product_type in product_types:
        suffix = PRODUCT_SUFFIXES[product_type]
        filenames = products["productFilename"].astype(str)
        mask = filenames.str.endswith(suffix)
        mask &= filenames.str.contains(tic_tag, regex=False)

        if product_type in {"lc", "tp"}:
            mask &= filenames.str.contains(f"-{sector_tag}-", regex=False)
        else:
            mask &= filenames.str.startswith(source_prefix)
            mask &= filenames.str.contains(f"-{sector_tag}-{sector_tag}-", regex=False)

        matches = products.loc[mask].copy()
        if matches.empty:
            continue

        matches["requested_product_type"] = product_type
        matches = matches.sort_values("productFilename", kind="mergesort")
        selected.append(matches.tail(1))

    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


def download_mast_uri(data_uri: str, destination: Path, *, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}"
    partial = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def status_row(
    target_row,
    product_type: str,
    filename: str,
    local_path: str | Path,
    status: str,
    message: str,
) -> dict:
    return {
        "tic_id": int(target_row.tic_id),
        "sector": int(target_row.sector),
        "product_type": product_type,
        "filename": filename,
        "local_path": str(local_path),
        "status": status,
        "message": message,
    }


def write_status(path: Path, rows: list[dict]) -> None:
    fieldnames = ["tic_id", "sector", "product_type", "filename", "local_path", "status", "message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
