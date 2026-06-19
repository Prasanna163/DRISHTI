"""Assemble type-disposition labels by TIC-ID crossmatch (WS-E).

The problem statement promises a curated label set; until that arrives we bootstrap labels from two
public catalogs, joined to our targets by TIC ID:

  - TOI / ExoFOP dispositions  (NASA Exoplanet Archive `toi` table, column `tfopwg_disp`):
        CP/KP = confirmed/known planet, PC/APC = planet candidate (ambiguous),
        FP = false positive, FA = false alarm.
  - TESS Eclipsing Binary catalog (Prsa et al. 2022, Vizier J/ApJS/258/16): membership => EB.

Mapping to training classes (EB membership takes priority, since many TOI false positives are EBs):
    in TESS-EB            -> eclipsing_binary
    CP / KP               -> planet
    FP / FA (not EB)      -> false_positive   (heterogeneous; collapsed to "other" for training)
    PC / APC              -> planet_candidate  (ambiguous; excluded from strict training)
    otherwise             -> "" (unlabeled)

Output: data/raw/labels/labels.csv  (one row per target TIC).

Example
-------
    python scripts/get_labels.py
    python scripts/get_labels.py --targets data/drishti/targets/tce_recovery_batch_111.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti_store import TARGET_ROOT

DEFAULT_TARGETS = TARGET_ROOT / "tce_positive_targets.csv"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "labels" / "labels.csv"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crossmatch targets against TOI + TESS-EB for labels.")
    p.add_argument("--targets", type=Path, default=DEFAULT_TARGETS,
                   help="Target CSV with a tic_id column (defines which TICs to label).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def fetch_toi() -> pd.DataFrame:
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    tab = NasaExoplanetArchive.query_criteria(
        table="toi", select="toi,tid,tfopwg_disp,pl_orbper,pl_trandep,pl_trandurh")
    df = tab.to_pandas()
    df = df.rename(columns={"tid": "tic_id", "tfopwg_disp": "toi_disp"})
    df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce")
    df = df.dropna(subset=["tic_id"]).copy()
    df["tic_id"] = df["tic_id"].astype(int)
    # One disposition per TIC (prefer the most "decided": CP/KP > FP/FA > PC).
    rank = {"CP": 0, "KP": 0, "FP": 1, "FA": 1, "PC": 2, "APC": 2}
    df["rk"] = df["toi_disp"].map(lambda d: rank.get(str(d), 3))
    df = df.sort_values("rk").drop_duplicates("tic_id", keep="first")
    return df[["tic_id", "toi", "toi_disp", "pl_orbper"]]


def fetch_tess_eb() -> set[int]:
    from astroquery.vizier import Vizier
    v = Vizier(columns=["TIC", "Per", "Morph"])
    v.ROW_LIMIT = -1
    cats = v.get_catalogs("J/ApJS/258/16")
    tics: set[int] = set()
    for t in cats:
        if "TIC" in t.colnames:
            s = pd.to_numeric(pd.Series(t["TIC"]).astype(str), errors="coerce").dropna()
            tics.update(int(x) for x in s)
    return tics


def map_class(toi_disp, in_eb: bool) -> str:
    if in_eb:
        return "eclipsing_binary"
    d = str(toi_disp) if toi_disp is not None else ""
    if d in {"CP", "KP"}:
        return "planet"
    if d in {"FP", "FA"}:
        return "false_positive"
    if d in {"PC", "APC"}:
        return "planet_candidate"
    return ""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = pd.read_csv(args.targets)
    tic_ids = sorted(set(int(t) for t in targets["tic_id"].dropna().unique()))
    print(f"Targets: {len(tic_ids)} unique TIC IDs from {args.targets.name}", flush=True)

    print("Fetching TOI dispositions (NASA Exoplanet Archive)...", flush=True)
    toi = fetch_toi()
    print(f"  TOI rows: {len(toi)}", flush=True)

    print("Fetching TESS Eclipsing Binary catalog (Vizier J/ApJS/258/16)...", flush=True)
    eb_tics = fetch_tess_eb()
    print(f"  TESS-EB systems: {len(eb_tics)}", flush=True)

    rows = []
    toi_by_tic = toi.set_index("tic_id").to_dict("index")
    for tic in tic_ids:
        rec = toi_by_tic.get(tic, {})
        disp = rec.get("toi_disp")
        in_eb = tic in eb_tics
        rows.append({
            "tic_id": tic,
            "toi": rec.get("toi", ""),
            "toi_disp": disp if disp is not None else "",
            "in_tess_eb": in_eb,
            "class_label": map_class(disp, in_eb),
            "label_source": _source(disp, in_eb),
        })

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("\n" + "=" * 50)
    print("  Label assembly summary")
    print("=" * 50)
    print(f"  Targets labeled: {(out.class_label != '').sum()} / {len(out)}")
    print("  class_label distribution:")
    for cls, n in out["class_label"].replace("", "(unlabeled)").value_counts().items():
        print(f"    {cls:20s} {n:>4d}")
    print(f"  Output: {args.output.resolve()}")
    print("=" * 50)
    return 0


def _source(disp, in_eb: bool) -> str:
    parts = []
    if in_eb:
        parts.append("tess_eb")
    if disp is not None and str(disp) != "":
        parts.append("toi")
    return "+".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
