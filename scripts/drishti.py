from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_access.load_fits import validate_fits

DATA_ROOT = ROOT / "data" / "drishti"
OUTPUT_ROOT = ROOT / "outputs" / "drishti"
RESOURCE_INDEX = DATA_ROOT / "metadata" / "bulk_resource_index.csv"
STORE_README = DATA_ROOT / "README.md"

GO_PAGE = "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_go.html"
SECTOR_PAGE = "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html"

SPLASH = r"""
██████╗ ██████╗ ██╗███████╗██╗  ██╗████████╗██╗
██╔══██╗██╔══██╗██║██╔════╝██║  ██║╚══██╔══╝██║
██║  ██║██████╔╝██║███████╗███████║   ██║   ██║
██║  ██║██╔══██╗██║╚════██║██╔══██║   ██║   ██║
██████╔╝██║  ██║██║███████║██║  ██║   ██║   ██║
╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝

Deep Recognition and Intelligent Screening of Hidden Transit Indicators
AI Exoplanet Detection & Vetting
"""


@dataclass(frozen=True)
class BulkResource:
    source_page: str
    resource_type: str
    sector: int | None
    program_id: str
    script_name: str
    url: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DRISHTI: TESS bulk discovery, target download, recovery, and vetting pipeline."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("splash", help="Print the DRISHTI splash screen.")

    init_parser = subparsers.add_parser("init-store", help="Create the DRISHTI data/output folder layout.")
    init_parser.add_argument("--dry-run", action="store_true", help="Show folders without creating them.")

    discover_parser = subparsers.add_parser(
        "discover",
        help="Scrape STScI TESS bulk pages into a local resource index.",
    )
    discover_parser.add_argument("--output", type=Path, default=RESOURCE_INDEX)
    discover_parser.add_argument("--dry-run", action="store_true", help="Print scrape summary without writing files.")
    discover_parser.add_argument("--timeout", type=int, default=60)

    pipeline_parser = subparsers.add_parser(
        "tce-recovery",
        help="Run the local TCE target selection, download, recovery, and controlled plotting pipeline.",
    )
    pipeline_parser.add_argument("--batch-size", type=int, default=50, help="Target-sector rows to process.")
    pipeline_parser.add_argument(
        "--balanced",
        action="store_true",
        help="Take an equal number of starter rows per sector where possible.",
    )
    pipeline_parser.add_argument(
        "--products",
        default="lc",
        help="Comma-separated products for download stage; default is lc.",
    )
    pipeline_parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    pipeline_parser.add_argument("--skip-discover", action="store_true", help="Do not refresh the STScI resource index.")
    pipeline_parser.add_argument("--skip-download", action="store_true", help="Reuse local products only.")
    pipeline_parser.add_argument("--skip-plots", action="store_true", help="Skip recovery plot generation.")
    pipeline_parser.add_argument("--top-direct-diagnostics", type=int, default=10)

    plan_parser = subparsers.add_parser(
        "plan-manifest",
        help="Build a local download plan from scraped sector/GO cURL scripts.",
    )
    plan_parser.add_argument("--resource-index", type=Path, default=RESOURCE_INDEX)
    plan_parser.add_argument("--resource-type", default="light_curve", help="Resource type from the index.")
    plan_parser.add_argument("--sectors", default="", help="Comma-separated sectors to include.")
    plan_parser.add_argument("--program-id", default="", help="GI program ID for guest_investigator resources.")
    plan_parser.add_argument("--target-list", type=Path, default=None, help="Optional TIC/sector target CSV filter.")
    plan_parser.add_argument("--output", type=Path, default=DATA_ROOT / "manifests" / "download_plan.csv")
    plan_parser.add_argument("--limit", type=int, default=None)
    plan_parser.add_argument("--dry-run", action="store_true")
    plan_parser.add_argument("--timeout", type=int, default=60)

    download_parser = subparsers.add_parser(
        "download-plan",
        help="Download files from a normalized DRISHTI download plan with resume/status logging.",
    )
    download_parser.add_argument("--plan", type=Path, default=DATA_ROOT / "manifests" / "download_plan.csv")
    download_parser.add_argument("--product-root", type=Path, default=DATA_ROOT / "raw")
    download_parser.add_argument("--status", type=Path, default=OUTPUT_ROOT / "tables" / "download_plan_status.csv")
    download_parser.add_argument("--limit", type=int, default=None)
    download_parser.add_argument("--timeout", type=int, default=120)
    download_parser.add_argument("--retry-failed", action="store_true")
    download_parser.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.command in {None, "splash"}:
        print_splash()
        return 0
    if args.command == "init-store":
        print_splash()
        init_store(dry_run=args.dry_run)
        return 0
    if args.command == "discover":
        print_splash()
        resources = discover_resources(timeout=args.timeout)
        print_resource_summary(resources)
        if not args.dry_run:
            write_resource_index(resources, args.output)
            init_store(dry_run=False)
            print(f"Resource index: {args.output.resolve()}")
        return 0
    if args.command == "tce-recovery":
        print_splash()
        return run_tce_recovery_pipeline(args)
    if args.command == "plan-manifest":
        print_splash()
        return build_download_plan_command(args)
    if args.command == "download-plan":
        print_splash()
        return download_plan_command(args)
    raise ValueError(f"Unknown command: {args.command}")


def ensure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def print_splash() -> None:
    print(SPLASH)


def init_store(*, dry_run: bool) -> None:
    directories = [
        DATA_ROOT / "metadata",
        DATA_ROOT / "ref",
        DATA_ROOT / "manifests" / "scripts",
        DATA_ROOT / "manifests" / "plans",
        DATA_ROOT / "raw" / "lc",
        DATA_ROOT / "raw" / "tp",
        DATA_ROOT / "raw" / "dv",
        DATA_ROOT / "processed",
        OUTPUT_ROOT / "tables",
        OUTPUT_ROOT / "plots",
        OUTPUT_ROOT / "logs",
    ]
    print("DRISHTI store layout")
    for directory in directories:
        print(f"  {directory}")
        if not dry_run:
            directory.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        STORE_README.write_text(storage_readme_text(), encoding="utf-8")
        print(f"Storage README: {STORE_README.resolve()}")


def storage_readme_text() -> str:
    return """# DRISHTI Data Store

This folder stores reproducible TESS bulk-download and recovery artifacts.

- `metadata/`: scraped STScI resource indexes and source-page metadata.
- `ref/`: official TCE/CDPP/reference CSVs.
- `manifests/scripts/`: cached STScI `.sh` cURL scripts.
- `manifests/plans/`: normalized download plans parsed from cURL scripts.
- `raw/lc/`: downloaded light-curve FITS files.
- `raw/tp/`: downloaded target-pixel FITS files.
- `raw/dv/`: downloaded DV PDF/XML/FITS products.
- `processed/`: future evidence-layer products.

Existing project outputs remain under `outputs/`; DRISHTI-specific summaries are under
`outputs/drishti/` when generated by the CLI.
"""


def discover_resources(*, timeout: int) -> list[BulkResource]:
    resources: list[BulkResource] = []
    resources.extend(discover_go_resources(timeout=timeout))
    resources.extend(discover_sector_resources(timeout=timeout))
    return sorted(resources, key=lambda r: (r.resource_type, r.sector or -1, r.program_id, r.script_name))


def discover_go_resources(*, timeout: int) -> list[BulkResource]:
    soup = fetch_soup(GO_PAGE, timeout=timeout)
    resources = []
    for link in soup.find_all("a", href=True):
        script_name = Path(link["href"]).name
        if not re.fullmatch(r"tesscurl_prop_G\d+\.sh", script_name):
            continue
        program_match = re.search(r"(G\d+)", script_name)
        resources.append(
            BulkResource(
                source_page=GO_PAGE,
                resource_type="guest_investigator",
                sector=None,
                program_id=program_match.group(1) if program_match else "",
                script_name=script_name,
                url=urljoin(GO_PAGE, link["href"]),
            )
        )
    return unique_resources(resources)


def discover_sector_resources(*, timeout: int) -> list[BulkResource]:
    soup = fetch_soup(SECTOR_PAGE, timeout=timeout)
    resources = []
    for link in soup.find_all("a", href=True):
        script_name = Path(link["href"]).name
        match = re.fullmatch(r"tesscurl_sector_(\d+)_(.+)\.sh", script_name)
        if not match:
            continue
        sector = int(match.group(1))
        token = match.group(2)
        resources.append(
            BulkResource(
                source_page=SECTOR_PAGE,
                resource_type=normalize_sector_resource_type(token),
                sector=sector,
                program_id="",
                script_name=script_name,
                url=urljoin(SECTOR_PAGE, link["href"]),
            )
        )
    return unique_resources(resources)


def normalize_sector_resource_type(token: str) -> str:
    mapping = {
        "ffic": "calibrated_ffi",
        "ffir": "uncalibrated_ffi",
        "tp": "target_pixel",
        "fast-tp": "fast_target_pixel",
        "lc": "light_curve",
        "fast-lc": "fast_light_curve",
        "dv": "data_validation",
    }
    return mapping.get(token, token.replace("-", "_"))


def unique_resources(resources: list[BulkResource]) -> list[BulkResource]:
    seen = set()
    unique = []
    for resource in resources:
        key = (resource.resource_type, resource.sector, resource.program_id, resource.script_name, resource.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resource)
    return unique


def fetch_soup(url: str, *, timeout: int) -> BeautifulSoup:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def print_resource_summary(resources: list[BulkResource]) -> None:
    rows = [
        {
            "resource_type": resource.resource_type,
            "sector": resource.sector,
            "program_id": resource.program_id,
        }
        for resource in resources
    ]
    frame = pd.DataFrame(rows)
    print(f"Discovered resources: {len(resources)}")
    if frame.empty:
        return
    print(frame.groupby("resource_type").size().sort_index().to_string())


def write_resource_index(resources: list[BulkResource], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_page", "resource_type", "sector", "program_id", "script_name", "url"],
        )
        writer.writeheader()
        for resource in resources:
            writer.writerow(
                {
                    "source_page": resource.source_page,
                    "resource_type": resource.resource_type,
                    "sector": resource.sector if resource.sector is not None else "",
                    "program_id": resource.program_id,
                    "script_name": resource.script_name,
                    "url": resource.url,
                }
            )


def run_tce_recovery_pipeline(args) -> int:
    init_store(dry_run=args.dry_run)
    if not args.skip_discover:
        discover_cmd = [sys.executable, str(Path(__file__).resolve()), "discover"]
        run_or_print(discover_cmd, dry_run=args.dry_run)

    starter = ROOT / "outputs" / "target_lists" / "tce_starter_validation_targets.csv"
    batch = ROOT / "outputs" / "target_lists" / f"tce_recovery_batch_{args.batch_size}.csv"
    recovery = ROOT / "outputs" / "tables" / f"tce_recovery_results_{args.batch_size}.csv"
    download_status = ROOT / "outputs" / "tables" / f"tce_download_status_{args.batch_size}.csv"
    plot_dir = ROOT / "outputs" / "plots" / f"tce_recovery_{args.batch_size}"

    commands = [
        [sys.executable, str(ROOT / "scripts" / "select_tce_targets.py")],
    ]

    if args.dry_run:
        print_batch_plan(starter, batch, batch_size=args.batch_size, balanced=args.balanced)
    else:
        run_or_print(commands[0], dry_run=False)
        write_batch_file(starter, batch, batch_size=args.batch_size, balanced=args.balanced)

    if not args.skip_download:
        commands.append(
            [
                sys.executable,
                str(ROOT / "scripts" / "05_download_tce_products.py"),
                "--targets",
                str(batch),
                "--products",
                args.products,
                "--status",
                str(download_status),
            ]
        )

    commands.append(
        [
            sys.executable,
            str(ROOT / "scripts" / "06_run_tce_recovery.py"),
            "--targets",
            str(batch),
            "--output",
            str(recovery),
        ]
    )

    if not args.skip_plots:
        commands.append(
            [
                sys.executable,
                str(ROOT / "scripts" / "07_plot_tce_recovery.py"),
                "--recovery",
                str(recovery),
                "--output-dir",
                str(plot_dir),
                "--diagnostics",
                "controlled",
                "--top-direct-diagnostics",
                str(args.top_direct_diagnostics),
            ]
        )

    command_start = 1 if not args.dry_run else 0
    for command in commands[command_start:]:
        run_or_print(command, dry_run=args.dry_run)

    print("\nDRISHTI pipeline artifacts")
    print(f"  batch targets: {batch}")
    print(f"  download status: {download_status}")
    print(f"  recovery table: {recovery}")
    print(f"  plots: {plot_dir}")
    return 0


def write_batch_file(starter: Path, output: Path, *, batch_size: int, balanced: bool) -> None:
    if not starter.exists():
        raise FileNotFoundError(f"Starter target list not found: {starter}")
    frame = pd.read_csv(starter)
    batch = select_batch(frame, batch_size=batch_size, balanced=balanced)
    output.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(output, index=False)
    print(f"Wrote batch target list: {output.resolve()}")
    print(batch.groupby("sector").size().to_string())


def print_batch_plan(starter: Path, output: Path, *, batch_size: int, balanced: bool) -> None:
    print("\nBatch target plan")
    print(f"  input: {starter}")
    print(f"  output: {output}")
    print(f"  rows: {batch_size}")
    print(f"  balanced: {balanced}")
    if starter.exists():
        batch = select_batch(pd.read_csv(starter), batch_size=batch_size, balanced=balanced)
        print(batch.groupby("sector").size().to_string())


def select_batch(frame: pd.DataFrame, *, batch_size: int, balanced: bool) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not balanced:
        return frame.head(batch_size).copy()

    sectors = sorted(frame["sector"].dropna().astype(int).unique())
    if not sectors:
        return frame.head(batch_size).copy()
    per_sector = max(1, batch_size // len(sectors))
    remainder = batch_size % len(sectors)
    pieces = []
    for index, sector in enumerate(sectors):
        take = per_sector + (1 if index < remainder else 0)
        pieces.append(frame[frame["sector"].astype(int) == sector].head(take))
    batch = pd.concat(pieces, ignore_index=True)
    if len(batch) < batch_size:
        used = set(zip(batch["tic_id"], batch["sector"]))
        rest = frame[~frame.apply(lambda row: (row["tic_id"], row["sector"]) in used, axis=1)]
        batch = pd.concat([batch, rest.head(batch_size - len(batch))], ignore_index=True)
    return batch.head(batch_size).copy()


def run_or_print(command: list[str], *, dry_run: bool) -> None:
    pretty = " ".join(quote_arg(part) for part in command)
    if dry_run:
        print(f"[dry-run] {pretty}")
        return
    print(f"[run] {pretty}")
    subprocess.run(command, cwd=ROOT, check=True)


def quote_arg(value: str) -> str:
    if re.search(r"\s", value):
        return f'"{value}"'
    return value


def build_download_plan_command(args) -> int:
    if not args.resource_index.exists():
        print(f"Resource index missing: {args.resource_index}")
        print("Run: python .\\scripts\\drishti.py discover")
        return 1
    index = pd.read_csv(args.resource_index)
    selected = index[index["resource_type"] == args.resource_type].copy()

    if args.sectors:
        sectors = {int(item.strip()) for item in args.sectors.split(",") if item.strip()}
        selected = selected[selected["sector"].fillna(-1).astype(int).isin(sectors)]
    if args.program_id:
        selected = selected[selected["program_id"].astype(str).str.upper() == args.program_id.upper()]

    if selected.empty:
        print("No matching resource scripts found.")
        return 1

    plan_rows = []
    for resource in selected.itertuples(index=False):
        script_text = fetch_manifest_script(str(resource.url), timeout=args.timeout)
        script_path = DATA_ROOT / "manifests" / "scripts" / str(resource.script_name)
        if not args.dry_run:
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script_text, encoding="utf-8")
        for item in parse_curl_manifest(script_text):
            item.update(
                {
                    "resource_type": resource.resource_type,
                    "sector": int(resource.sector) if pd.notna(resource.sector) else "",
                    "program_id": resource.program_id if pd.notna(resource.program_id) else "",
                    "manifest_script": resource.script_name,
                }
            )
            plan_rows.append(item)

    plan = pd.DataFrame(plan_rows)
    if args.target_list is not None:
        plan = filter_plan_to_targets(plan, pd.read_csv(args.target_list))
    if args.limit is not None:
        plan = plan.head(args.limit)

    print(f"Plan rows: {len(plan)}")
    if not plan.empty:
        print(plan.head(10).to_string(index=False))
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(args.output, index=False)
        print(f"Wrote download plan: {args.output.resolve()}")
    return 0


def fetch_manifest_script(url: str, *, timeout: int) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_curl_manifest(script_text: str) -> list[dict]:
    rows = []
    pattern = re.compile(r"^curl\s+-C\s+-\s+-L\s+-o\s+(\S+)\s+(\S+)", re.IGNORECASE)
    for line_number, line in enumerate(script_text.splitlines(), start=1):
        match = pattern.match(line.strip())
        if not match:
            continue
        filename, url = match.groups()
        meta = parse_tess_filename(filename)
        rows.append(
            {
                "line_number": line_number,
                "filename": filename,
                "url": url,
                **meta,
            }
        )
    return rows


def parse_tess_filename(filename: str) -> dict:
    product = "other"
    lower = filename.lower()
    if lower.endswith("_lc.fits"):
        product = "lc"
    elif lower.endswith("_tp.fits"):
        product = "tp"
    elif "_dvr." in lower:
        product = "dvr"
    elif "_dvt.fits" in lower:
        product = "dvt"

    match = re.search(r"-(s\d{4})-(\d{16})-", lower)
    return {
        "product": product,
        "parsed_sector": int(match.group(1)[1:]) if match else "",
        "tic_id": int(match.group(2)) if match else "",
    }


def filter_plan_to_targets(plan: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return plan
    if not {"tic_id", "sector"}.issubset(targets.columns):
        raise ValueError("Target list must contain tic_id and sector columns.")
    target_keys = set(zip(targets["tic_id"].astype(int), targets["sector"].astype(int)))
    return plan[
        plan.apply(
            lambda row: (
                int(row["tic_id"]),
                int(row["parsed_sector"] or row["sector"]),
            )
            in target_keys
            if str(row.get("tic_id", "")).strip()
            else False,
            axis=1,
        )
    ].copy()


def download_plan_command(args) -> int:
    if not args.plan.exists():
        print(f"Download plan missing: {args.plan}")
        print("Run: python .\\scripts\\drishti.py plan-manifest --resource-type light_curve --sectors 1")
        return 1

    plan = pd.read_csv(args.plan)
    if args.limit is not None:
        plan = plan.head(args.limit)
    existing_status = read_status_table(args.status)
    completed = completed_keys(existing_status, retry_failed=args.retry_failed)

    rows = []
    for item in plan.itertuples(index=False):
        key = (str(item.manifest_script), int(item.line_number), str(item.filename))
        product = str(item.product)
        destination = args.product_root / product / str(item.filename)
        status = "planned" if args.dry_run else "downloaded"
        message = ""

        if key in completed:
            status = "resume_skipped"
        elif destination.exists() and destination.stat().st_size > 0:
            status = "exists"
        elif not args.dry_run:
            try:
                download_url(str(item.url), destination, timeout=args.timeout)
                if destination.suffix.lower() == ".fits":
                    validate_fits(destination)
            except Exception as exc:
                status = "failed"
                message = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "manifest_script": item.manifest_script,
                "line_number": int(item.line_number),
                "filename": item.filename,
                "product": product,
                "tic_id": item.tic_id,
                "sector": item.parsed_sector if item.parsed_sector else item.sector,
                "url": item.url,
                "local_path": str(destination),
                "status": status,
                "message": message,
            }
        )

    if not args.dry_run:
        write_plan_status(args.status, rows)

    frame = pd.DataFrame(rows)
    print(f"Plan rows considered: {len(frame)}")
    if not frame.empty:
        print(frame["status"].value_counts().to_string())
        print(frame.head(10).to_string(index=False))
    if not args.dry_run:
        print(f"Status log: {args.status.resolve()}")
    return 1 if "failed" in set(frame.get("status", [])) else 0


def read_status_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def completed_keys(status: pd.DataFrame, *, retry_failed: bool) -> set[tuple[str, int, str]]:
    if status.empty:
        return set()
    terminal = {"downloaded", "exists", "resume_skipped"}
    if not retry_failed:
        terminal.add("failed")
    keep = status[status["status"].isin(terminal)]
    return {
        (str(row.manifest_script), int(row.line_number), str(row.filename))
        for row in keep.itertuples(index=False)
    }


def download_url(url: str, destination: Path, *, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def write_plan_status(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "manifest_script",
        "line_number",
        "filename",
        "product",
        "tic_id",
        "sector",
        "url",
        "local_path",
        "status",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
