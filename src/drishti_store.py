from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data" / "drishti"
CATALOG_ROOT = DATA_ROOT / "catalogs"
TARGET_ROOT = DATA_ROOT / "targets"
MANIFEST_ROOT = DATA_ROOT / "manifests"
MANIFEST_SCRIPT_ROOT = MANIFEST_ROOT / "scripts"
MANIFEST_PLAN_ROOT = MANIFEST_ROOT / "plans"
DOWNLOAD_ROOT = DATA_ROOT / "downloads"
DOWNLOAD_LC_ROOT = DOWNLOAD_ROOT / "lc"
DOWNLOAD_TP_ROOT = DOWNLOAD_ROOT / "tp"
DOWNLOAD_DV_ROOT = DOWNLOAD_ROOT / "dv"
RESULT_ROOT = DATA_ROOT / "results"
RESULT_TABLE_ROOT = RESULT_ROOT / "tables"
RESULT_PLOT_ROOT = RESULT_ROOT / "plots"
LOG_ROOT = DATA_ROOT / "logs"
STORE_README = DATA_ROOT / "README.md"
RESOURCE_INDEX = CATALOG_ROOT / "bulk_resource_index.csv"

# Legacy output locations (pre-consolidation)
LEGACY_OUTPUT_ROOT = ROOT / "outputs"
LEGACY_TABLE_DIRS = [
    LEGACY_OUTPUT_ROOT / "tables",
    LEGACY_OUTPUT_ROOT / "drishti" / "tables",
]
LEGACY_PLOT_DIRS = [
    LEGACY_OUTPUT_ROOT / "drishti" / "plots",
]


def store_dirs() -> list[Path]:
    return [
        CATALOG_ROOT,
        TARGET_ROOT,
        MANIFEST_SCRIPT_ROOT,
        MANIFEST_PLAN_ROOT,
        DOWNLOAD_LC_ROOT,
        DOWNLOAD_TP_ROOT,
        DOWNLOAD_DV_ROOT,
        RESULT_TABLE_ROOT,
        RESULT_PLOT_ROOT,
        LOG_ROOT,
    ]


def migrate_legacy_outputs(*, dry_run: bool = False) -> list[str]:
    """Move results from legacy outputs/ directories into data/drishti/results/.

    Returns a list of human-readable action strings describing what was (or would be) done.
    Original files are renamed to .bak so nothing is lost.
    """
    actions: list[str] = []

    table_globs = ["*.csv"]
    plot_globs = ["*.png", "*.jpg", "*.pdf"]

    for src_dir in LEGACY_TABLE_DIRS:
        actions.extend(
            _migrate_dir(src_dir, RESULT_TABLE_ROOT, table_globs, dry_run=dry_run)
        )

    for src_dir in LEGACY_PLOT_DIRS:
        actions.extend(
            _migrate_dir(src_dir, RESULT_PLOT_ROOT, plot_globs, dry_run=dry_run)
        )

    return actions


def _migrate_dir(
    src_dir: Path,
    dest_dir: Path,
    globs: list[str],
    *,
    dry_run: bool,
) -> list[str]:
    actions: list[str] = []
    if not src_dir.exists():
        return actions

    for pattern in globs:
        for src_file in sorted(src_dir.glob(pattern)):
            if src_file.name.endswith(".bak"):
                continue
            dest_file = dest_dir / src_file.name
            if dest_file.exists():
                actions.append(f"SKIP (already exists): {src_file} -> {dest_file}")
                continue

            tag = "[dry-run] " if dry_run else ""
            actions.append(f"{tag}COPY: {src_file} -> {dest_file}")
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)

            bak_file = src_file.with_suffix(src_file.suffix + ".bak")
            actions.append(f"{tag}RENAME: {src_file} -> {bak_file}")
            if not dry_run:
                src_file.rename(bak_file)

    return actions


def store_tree_text() -> str:
    return """# DRISHTI Store

This is the central DRISHTI workspace for generated artifacts.

```text
data/drishti/
  catalogs/        Scraped bulk indexes and metadata
  targets/         Selected TCE target lists and recovery batches
  manifests/
    scripts/       Cached STScI .sh files while streaming
    plans/         Normalized file-level download plans
  downloads/
    lc/            Light-curve FITS files
    tp/            Target-pixel FITS files
    dv/            DV products
  results/
    tables/        Download/recovery/status tables
    plots/         Summary plots and diagnostics
  logs/            Run logs
```

Source reference CSVs still live in `data/Ref/`.
"""
