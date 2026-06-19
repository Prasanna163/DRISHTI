from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti_store import TARGET_ROOT

DEFAULT_REF_DIR = ROOT / "data" / "Ref"
DEFAULT_OUTPUT = TARGET_ROOT / "tce_positive_targets.csv"
DEFAULT_STARTER_OUTPUT = TARGET_ROOT / "tce_starter_validation_targets.csv"
DEFAULT_FIRST_BATCH_OUTPUT = TARGET_ROOT / "tce_first_recovery_batch.csv"

FIRST_RECOVERY_TIC_IDS = [
    183979262,
    50380257,
    38846515,
    25155310,
    38937499,
    270622440,
    197760286,
    51912829,
    183985250,
    167007869,
]

OUTPUT_COLUMNS = [
    "tic_id",
    "sector",
    "official_period",
    "official_epoch",
    "official_duration_hours",
    "official_depth",
    "official_snr",
    "official_num_transits",
    "official_full_convergence",
    "source_tce_file",
]

REQUIRED_COLUMNS = {
    "ticid",
    "sectors",
    "tce_period",
    "tce_time0bt",
    "tce_duration",
    "tce_depth",
    "tce_model_snr",
    "tce_num_transits",
    "tce_full_conv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select high-confidence official TESS TCE targets for download."
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=DEFAULT_REF_DIR,
        help="Folder containing official *_dvr-tcestats.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV target list.",
    )
    parser.add_argument(
        "--starter-output",
        type=Path,
        default=DEFAULT_STARTER_OUTPUT,
        help="Output CSV for the first clean validation subset.",
    )
    parser.add_argument(
        "--first-batch-output",
        type=Path,
        default=DEFAULT_FIRST_BATCH_OUTPUT,
        help="Output CSV for the first 10-20 target recovery batch.",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=7.1,
        help="Minimum official TCE model SNR.",
    )
    parser.add_argument(
        "--min-transits",
        type=int,
        default=2,
        help="Minimum official number of transits.",
    )
    parser.add_argument(
        "--allow-nonconverged",
        action="store_true",
        help="Keep TCE rows even when the official transit model did not converge.",
    )
    parser.add_argument(
        "--keep-multiple-tces",
        action="store_true",
        help="Keep multiple qualifying TCEs for the same TIC/sector instead of one target row.",
    )
    parser.add_argument(
        "--skip-starter-output",
        action="store_true",
        help="Do not write the starter validation subset CSV.",
    )
    parser.add_argument(
        "--skip-first-batch-output",
        action="store_true",
        help="Do not write the first recovery batch CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tce_files = sorted(args.ref_dir.glob("*_dvr-tcestats.csv"))
    if not tce_files:
        raise FileNotFoundError(f"No *_dvr-tcestats.csv files found in {args.ref_dir}")

    selected = []
    for tce_file in tce_files:
        table = pd.read_csv(tce_file, comment="#")
        missing = REQUIRED_COLUMNS - set(table.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{tce_file} is missing required columns: {missing_text}")

        filtered = select_good_tces(
            table,
            min_snr=args.min_snr,
            min_transits=args.min_transits,
            require_converged=not args.allow_nonconverged,
        )
        selected.append(format_target_rows(filtered, source_tce_file=tce_file.name))

    targets = pd.concat(selected, ignore_index=True)
    if not args.keep_multiple_tces and not targets.empty:
        targets = (
            targets.sort_values(
                ["tic_id", "sector", "official_snr"],
                ascending=[True, True, False],
                kind="mergesort",
            )
            .drop_duplicates(["tic_id", "sector"], keep="first")
            .sort_values(["sector", "tic_id"], kind="mergesort")
        )

    targets = targets.loc[:, OUTPUT_COLUMNS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.output, index=False)

    print(f"Read {len(tce_files)} official TCE CSV file(s).")
    print(f"Selected {len(targets)} target row(s).")
    print(f"Wrote: {args.output.resolve()}")

    if not args.skip_starter_output:
        starter_targets = select_starter_validation_targets(targets)
        args.starter_output.parent.mkdir(parents=True, exist_ok=True)
        starter_targets.to_csv(args.starter_output, index=False)
        print(f"Selected {len(starter_targets)} starter validation row(s).")
        print(f"Wrote: {args.starter_output.resolve()}")

        if not args.skip_first_batch_output:
            first_batch = select_first_recovery_batch(starter_targets)
            args.first_batch_output.parent.mkdir(parents=True, exist_ok=True)
            first_batch.to_csv(args.first_batch_output, index=False)
            print(f"Selected {len(first_batch)} first recovery batch row(s).")
            print(f"Wrote: {args.first_batch_output.resolve()}")
    return 0


def select_good_tces(
    table: pd.DataFrame,
    *,
    min_snr: float,
    min_transits: int,
    require_converged: bool,
) -> pd.DataFrame:
    numeric_columns = [
        "ticid",
        "tce_period",
        "tce_time0bt",
        "tce_duration",
        "tce_depth",
        "tce_model_snr",
        "tce_num_transits",
    ]
    numeric = table.copy()
    for column in numeric_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    mask = (
        numeric["ticid"].notna()
        & numeric["tce_period"].gt(0)
        & numeric["tce_time0bt"].gt(0)
        & numeric["tce_duration"].gt(0)
        & numeric["tce_depth"].gt(0)
        & numeric["tce_model_snr"].ge(min_snr)
        & numeric["tce_num_transits"].ge(min_transits)
    )
    if require_converged:
        mask &= numeric["tce_full_conv"].astype(str).str.lower().eq("true")

    return numeric.loc[mask].copy()


def format_target_rows(table: pd.DataFrame, *, source_tce_file: str) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        for sector in parse_sector_list(row["sectors"]):
            rows.append(
                {
                    "tic_id": int(row["ticid"]),
                    "sector": sector,
                    "official_period": row["tce_period"],
                    "official_epoch": row["tce_time0bt"],
                    "official_duration_hours": row["tce_duration"],
                    "official_depth": row["tce_depth"],
                    "official_snr": row["tce_model_snr"],
                    "official_num_transits": int(row["tce_num_transits"]),
                    "official_full_convergence": parse_bool(row["tce_full_conv"]),
                    "source_tce_file": source_tce_file,
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def select_starter_validation_targets(targets: pd.DataFrame) -> pd.DataFrame:
    starter_mask = (
        targets["official_snr"].ge(20)
        & targets["official_depth"].between(200, 10_000, inclusive="both")
        & targets["official_period"].between(0.5, 13, inclusive="both")
        & targets["official_duration_hours"].between(0.5, 12, inclusive="both")
    )
    return targets.loc[starter_mask, OUTPUT_COLUMNS].copy()


def select_first_recovery_batch(targets: pd.DataFrame) -> pd.DataFrame:
    batch = targets.loc[targets["tic_id"].isin(FIRST_RECOVERY_TIC_IDS), OUTPUT_COLUMNS].copy()
    tic_order = {tic_id: index for index, tic_id in enumerate(FIRST_RECOVERY_TIC_IDS)}
    batch["_tic_order"] = batch["tic_id"].map(tic_order)
    batch = batch.sort_values(["_tic_order", "sector"], kind="mergesort")
    return batch.drop(columns=["_tic_order"]).reset_index(drop=True)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def parse_sector_list(value: object) -> list[int]:
    sector_text = str(value)
    sectors = [int(match) for match in re.findall(r"s0*(\d+)", sector_text, flags=re.IGNORECASE)]
    if sectors:
        return sectors

    compact_numbers = re.findall(r"\d+", sector_text)
    if compact_numbers:
        return [int(number) for number in compact_numbers]

    raise ValueError(f"Could not parse sector from value: {sector_text!r}")


if __name__ == "__main__":
    raise SystemExit(main())
