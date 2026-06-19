"""Classified-candidate diagnostic plots (WS-H1).

Produces the problem-statement visualization: the light curve together with the detected and
*classified* signal. For each selected candidate it draws:
  - left  : the cleaned light curve vs time, with the detected transit windows shaded
  - right : the phase-folded light curve (scatter + binned median) with the FITTED trapezoid model
            overlaid, so the parameter fit is visually verifiable
The title carries the predicted class (colour-coded), its confidence, the ML planet probability, and
the fitted period / depth / duration with uncertainties.

By default it plots the top-N highest-confidence candidates per class; pass --tic for a single target
or --all to plot everything.

Example
-------
    python scripts/13_plot_classified.py --per-class 6
    python scripts/13_plot_classified.py --tic 25155310
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve
from detection.run_bls import fold_lightcurve, bin_phase_curve
from fitting.transit_fit import trapezoid_flux
from drishti_store import DOWNLOAD_LC_ROOT, RESULT_TABLE_ROOT, RESULT_PLOT_ROOT

T = RESULT_TABLE_ROOT
CLASS_COLOR = {
    "planet_candidate": "#19764b", "eclipsing_binary": "#b33a3a",
    "blend": "#8a5fbf", "undetermined": "#7a7a7a",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot classified candidates with fitted-model overlay.")
    p.add_argument("--master", type=Path, default=T / "master_candidates.csv")
    p.add_argument("--recovery", type=Path, nargs="+",
                   default=[T / "tce_recovery_results_111.csv", T / "tce_recovery_results_labeled.csv"])
    p.add_argument("--fits", type=Path, nargs="+",
                   default=[T / "transit_fits_111.csv", T / "transit_fits_labeled.csv"])
    p.add_argument("--fits-dir", type=Path, default=DOWNLOAD_LC_ROOT)
    p.add_argument("--output-dir", type=Path, default=RESULT_PLOT_ROOT / "classified")
    p.add_argument("--per-class", type=int, default=6)
    p.add_argument("--tic", type=int, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    return p


def _concat(paths):
    frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
    return pd.concat(frames, ignore_index=True).drop_duplicates(["tic_id", "sector"], keep="first")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    master = pd.read_csv(args.master)
    rec = _concat(args.recovery)
    fits = _concat(args.fits)
    df = master.merge(rec[["tic_id", "sector", "our_bls_period", "our_t0", "our_duration_hours"]],
                      on=["tic_id", "sector"], how="left")
    df = df.merge(fits[["tic_id", "sector", "fit_depth_ppm", "fit_duration_hours", "fit_ingress_frac"]],
                  on=["tic_id", "sector"], how="left", suffixes=("", "_fit"))

    if args.tic is not None:
        sel = df[df.tic_id == args.tic]
    elif args.all:
        sel = df
    else:
        sel = (df.sort_values("class_confidence", ascending=False)
                 .groupby("predicted_class", group_keys=False).head(args.per_class))

    print(f"Plotting {len(sel)} candidate(s)...", flush=True)
    made = 0
    for row in sel.itertuples(index=False):
        out = _plot_one(row, args.fits_dir, args.output_dir, args.dpi)
        if out:
            made += 1
    print(f"Done. {made} plot(s) written under {args.output_dir.resolve()}")
    return 0


def _plot_one(row, fits_dir: Path, output_dir: Path, dpi: int):
    tic, sector = int(row.tic_id), int(row.sector)
    period = float(getattr(row, "our_bls_period", np.nan))
    t0 = float(getattr(row, "our_t0", np.nan))
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0)):
        return None
    fits_path = _find_lc(fits_dir, tic, sector)
    if fits_path is None:
        return None
    dur_days = float(getattr(row, "our_duration_hours", np.nan)) / 24.0
    try:
        lc = load_clean_flattened_lightcurve(
            fits_path, mask_period=period, mask_t0=t0, mask_duration_days=dur_days)
    except Exception:
        return None

    phase, fflux = fold_lightcurve(lc.time, lc.flux, period, t0)
    centers, binned = bin_phase_curve(phase, fflux, bins=120)
    cls = str(row.predicted_class)
    color = CLASS_COLOR.get(cls, "#333333")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    # Left: light curve vs time, transit windows shaded.
    ax1.scatter(lc.time, lc.flux, s=2, alpha=0.5, color="#102a83")
    n0 = int(np.floor((lc.time.min() - t0) / period))
    n1 = int(np.ceil((lc.time.max() - t0) / period))
    for n in range(n0, n1 + 1):
        c = t0 + n * period
        ax1.axvspan(c - dur_days / 2, c + dur_days / 2, color=color, alpha=0.12)
    ax1.set_xlabel("Time (BTJD)"); ax1.set_ylabel("Normalized flux")
    ax1.set_title("Light curve + detected transits")

    # Right: phase fold + fitted trapezoid model.
    ax2.scatter(phase, fflux, s=3, alpha=0.18, color="#444444")
    ax2.plot(centers, binned, color="#0b6b4f", lw=1.8, label="binned median")
    depth = float(getattr(row, "fit_depth_ppm", np.nan)) / 1e6
    fdur = float(getattr(row, "fit_duration_hours", np.nan)) / 24.0
    ingress = float(getattr(row, "fit_ingress_frac", np.nan))
    if np.isfinite(depth) and np.isfinite(fdur) and np.isfinite(ingress):
        pgrid = np.linspace(-0.5, 0.5, 2000)
        model = trapezoid_flux(pgrid * period, 0.0, depth, fdur, ingress)
        ax2.plot(pgrid, model, color="#d6312b", lw=2.0, label="fitted model")
        half = 0.6 * fdur / period
        ax2.set_xlim(-max(4 * half, 0.05), max(4 * half, 0.05))
    ax2.axhline(1.0, color="#999999", lw=0.8, ls="--")
    ax2.set_xlabel("Phase"); ax2.set_ylabel("Normalized flux")
    ax2.set_title("Phase-folded + fitted model"); ax2.legend(loc="lower right", fontsize=8)

    conf = float(getattr(row, "class_confidence", np.nan))
    mlp = float(getattr(row, "ml_planet_proba", np.nan))
    depth_ppm = float(getattr(row, "fit_depth_ppm", np.nan))
    depth_err = float(getattr(row, "fit_depth_err_ppm", np.nan)) if hasattr(row, "fit_depth_err_ppm") else np.nan
    fdur_h = float(getattr(row, "fit_duration_hours", np.nan))
    truth = f"  [label: {row.class_label}]" if isinstance(getattr(row, "class_label", None), str) else ""
    fig.suptitle(
        f"TIC {tic}  S{sector:04d}   |   class: {cls}  (conf {conf:.2f}, ML p(planet)={mlp:.2f}){truth}\n"
        f"P = {period:.4f} d   depth = {depth_ppm:.0f} ppm   duration = {fdur_h:.2f} h",
        color=color, fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    dest = output_dir / cls
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"TIC_{tic}_S{sector:04d}_classified.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _find_lc(fits_dir: Path, tic_id: int, sector: int):
    matches = sorted(fits_dir.rglob(f"*-s{sector:04d}-{tic_id:016d}-*_lc.fits"))
    return matches[0] if matches else None


if __name__ == "__main__":
    raise SystemExit(main())
