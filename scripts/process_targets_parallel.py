"""Single-pass, multi-core target processor (fast path for recovery + fit + vetting).

This replaces running scripts 06 + 08 + 09 separately. For each target it:
  - loads + cleans the light curve ONCE,
  - runs Box Least Squares ONCE (the previous flow ran BLS twice: recovery and fit each did their own),
  - derives the recovery comparison, the transit fit (with a transit-masked second detrend pass for
    accurate depth), and the full vetting evidence vector,
all inside one worker, and fans the targets out across a process pool (CPU cores).

It writes the same three CSV schemas the downstream trainer/finalizer expect:
    tce_recovery_results_<suffix>.csv, transit_fits_<suffix>.csv, vetting_features_<suffix>.csv

Note on GPU: the cost is astropy's BoxLeastSquares (CPU/NumPy, no GPU backend), so the realistic
speedup is process-level parallelism + removing the redundant second BLS, not GPU offload.

Example
-------
    python scripts/process_targets_parallel.py --targets data/drishti/targets/labeled_training_targets.csv --suffix labeled --workers 14
"""

from __future__ import annotations

# Pin BLAS/OpenMP to one thread PER worker *before* numpy is imported. With many worker processes
# each spawning its own BLAS thread pool, the threads oversubscribe the cores and crash the pool
# (BrokenProcessPool on Windows). One BLAS thread per process + N processes is both stable and faster.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing.clean_lightcurve import load_clean_flattened_lightcurve
from detection.run_bls import run_bls_search
from fitting.transit_fit import fit_transit
from vetting.centroid_shift import measure_centroid_shift
from vetting.odd_even import measure_odd_even
from vetting.secondary_eclipse import measure_secondary_eclipse
from vetting.local_shape import measure_local_shape
from vetting.duration_sanity import assess_duration
from drishti_store import DOWNLOAD_LC_ROOT, RESULT_TABLE_ROOT

from astropy.io import fits as _fits

# ---- output schemas (must match scripts 06 / 08 / 09) ----
RECOVERY_COLS = [
    "tic_id", "sector", "official_period", "our_bls_period", "period_error_percent",
    "best_period_error_percent", "period_match_type", "official_epoch", "our_t0",
    "epoch_match_score", "official_duration_hours", "our_duration_hours", "duration_ratio",
    "official_snr", "our_snr", "recovery_class", "recovered_true_false",
    "period_recovered_true_false", "status", "source_file", "message",
]
FIT_COLS = [
    "tic_id", "sector", "official_period", "official_duration_hours", "official_depth_ppm",
    "fit_period_days", "fit_period_err_days", "fit_t0_btjd", "fit_t0_err_days",
    "fit_depth_ppm", "fit_depth_err_ppm", "fit_duration_hours", "fit_duration_err_hours",
    "fit_ingress_frac", "fit_ingress_frac_err", "depth_ratio_fit_vs_official",
    "duration_ratio_fit_vs_official", "reduced_chi2", "bic", "n_points_fit",
    "fit_status", "source_file", "message",
]
VETTING_COLS = [
    "tic_id", "sector", "official_period", "official_epoch", "official_duration_hours",
    "crowdsap", "flfrcsap", "centroid_status", "centroid_source", "centroid_shift_pixels",
    "centroid_shift_sigma", "centroid_dcol_pixels", "centroid_drow_pixels",
    "centroid_n_in_transit", "centroid_n_out_transit", "centroid_on_target", "blend_flag",
    "lc_vetting_status", "oddeven_diff_sigma", "oddeven_depth_frac_diff", "depth_odd_ppm",
    "depth_even_ppm", "oddeven_flag", "primary_depth_ppm", "secondary_depth_ppm",
    "secondary_snr", "secondary_phase", "secondary_to_primary_ratio", "secondary_flag",
    "v_shape_metric", "shape_flag", "expected_duration_hours", "duration_sanity_ratio",
    "duration_flag", "eb_flag", "source_file", "centroid_message",
]


# ---- recovery helpers (mirror scripts/06) ----
def _period_error_percent(cand, off):
    if off <= 0 or not math.isfinite(cand):
        return math.nan
    return abs(cand - off) / off * 100.0


def _best_period_error_with_harmonics(cand, off):
    checks = {"direct": cand, "half_period_alias": cand * 2.0, "double_period_alias": cand / 2.0}
    errs = {k: _period_error_percent(v, off) for k, v in checks.items()}
    best = min(errs, key=lambda k: errs[k])
    return errs[best], best


def _epoch_match_score(t0, off_epoch, off_period, off_dur_days):
    if off_period <= 0 or off_dur_days <= 0 or not math.isfinite(t0):
        return math.nan
    delta = abs(((t0 - off_epoch + 0.5 * off_period) % off_period) - 0.5 * off_period)
    return max(0.0, 1.0 - delta / max(off_dur_days, 1e-6))


def _duration_ratio(cand_h, off_h):
    if not (math.isfinite(cand_h) and math.isfinite(off_h)) or cand_h <= 0 or off_h <= 0:
        return math.nan
    r = cand_h / off_h
    return max(r, 1.0 / r)


def _classify_recovery(row, ptol=1.0, pvet=0.1, min_snr=7.0, max_dur=3.0):
    best = float(row["best_period_error_percent"]); direct = float(row["period_error_percent"])
    escore = float(row["epoch_match_score"]); snr = float(row["our_snr"])
    dr = float(row["duration_ratio"]); mt = str(row["period_match_type"])
    period_ok = math.isfinite(best) and best <= ptol
    direct_ok = mt == "direct" and math.isfinite(direct) and direct <= ptol
    vet_ok = math.isfinite(best) and best < pvet
    epoch_ok = math.isfinite(escore) and escore >= 0.5
    snr_ok = math.isfinite(snr) and snr >= min_snr
    dur_bad = math.isfinite(dr) and dr > max_dur
    if direct_ok and epoch_ok and snr_ok and not dur_bad:
        return "direct_recovered"
    if mt in {"half_period_alias", "double_period_alias"} and period_ok and snr_ok:
        return "alias_recovered"
    if direct_ok and dur_bad:
        return "period_recovered_bad_duration"
    if direct_ok and not epoch_ok:
        return "period_recovered_epoch_mismatch"
    if vet_ok:
        return "period_recovered_needs_vetting"
    return "not_recovered"


# ---- vetting aggregation helpers (mirror scripts/09) ----
def _read_crowding(fits_path):
    try:
        with _fits.open(fits_path) as h:
            hdr = h[1].header
            return float(hdr.get("CROWDSAP", math.nan)), float(hdr.get("FLFRCSAP", math.nan))
    except Exception:
        return math.nan, math.nan


def _blend_flag(centroid, crowdsap):
    if centroid.centroid_status != "ok":
        return "unknown"
    if not centroid.centroid_on_target:
        return "likely_blend"
    if math.isfinite(crowdsap) and crowdsap < 0.5:
        return "crowded_on_target"
    return "on_target"


def _eb_flag(v):
    if v.get("lc_vetting_status") != "ok":
        return "unknown"
    if v.get("secondary_flag") == "eb_suspect" or v.get("oddeven_flag") == "eb_suspect":
        return "eb_suspect"
    if v.get("shape_flag") == "v_shaped":
        return "v_shaped_watch"
    return "ok"


def _find_lc(fits_dir, tic_id, sector):
    matches = sorted(Path(fits_dir).rglob(f"*-s{sector:04d}-{tic_id:016d}-*_lc.fits"))
    return matches[0] if matches else None


# ---- the worker (must be top-level & picklable) ----
def process_one(payload):
    t, cfg = payload
    tic, sector = int(t["tic_id"]), int(t["sector"])
    off_period = float(t["official_period"]); off_epoch = float(t["official_epoch"])
    off_dur_h = float(t["official_duration_hours"]); off_snr = float(t.get("official_snr", math.nan))
    off_depth = float(t.get("official_depth", math.nan))

    rec = {c: "" for c in RECOVERY_COLS}
    rec.update(tic_id=tic, sector=sector, official_period=off_period, official_epoch=off_epoch,
               official_duration_hours=off_dur_h, official_snr=off_snr,
               recovered_true_false=False, period_recovered_true_false=False)
    fit = {c: "" for c in FIT_COLS}
    fit.update(tic_id=tic, sector=sector, official_period=off_period,
               official_duration_hours=off_dur_h, official_depth_ppm=off_depth, n_points_fit=0)
    vet = {c: "" for c in VETTING_COLS}
    vet.update(tic_id=tic, sector=sector, official_period=off_period, official_epoch=off_epoch,
               official_duration_hours=off_dur_h, centroid_on_target=False)

    fp = _find_lc(cfg["fits_dir"], tic, sector)
    if fp is None:
        rec.update(status="missing_lc_fits", recovery_class="download_failed")
        fit.update(fit_status="missing_lc_fits", message="no LC FITS")
        vet.update(centroid_status="missing_lc_fits", lc_vetting_status="missing_lc_fits", blend_flag="unknown")
        return rec, fit, vet

    src = fp.name
    rec["source_file"] = fit["source_file"] = vet["source_file"] = src
    try:
        clean = load_clean_flattened_lightcurve(fp)
        bls = run_bls_search(clean, min_period=cfg["min_period"], max_period=cfg["max_period"],
                             n_periods=cfg["n_periods"])
        # ---- recovery ----
        direct_err = _period_error_percent(bls.best_period, off_period)
        best_err, mt = _best_period_error_with_harmonics(bls.best_period, off_period)
        rec.update(
            our_bls_period=bls.best_period, period_error_percent=direct_err,
            best_period_error_percent=best_err, period_match_type=mt, our_t0=bls.best_t0,
            epoch_match_score=_epoch_match_score(bls.best_t0, off_epoch, off_period, off_dur_h / 24.0),
            our_duration_hours=bls.best_duration * 24.0, our_snr=bls.snr, status="evaluated",
        )
        rec["duration_ratio"] = _duration_ratio(rec["our_duration_hours"], off_dur_h)
        rec["recovery_class"] = _classify_recovery(rec)
        rec["recovered_true_false"] = rec["recovery_class"] in {"direct_recovered", "alias_recovered"}
        rec["period_recovered_true_false"] = rec["recovery_class"] in {
            "direct_recovered", "alias_recovered", "period_recovered_epoch_mismatch",
            "period_recovered_bad_duration", "period_recovered_needs_vetting"}

        # ---- fit (transit-masked second detrend for accurate depth) ----
        fit_lc = load_clean_flattened_lightcurve(
            fp, mask_period=bls.best_period, mask_t0=bls.best_t0, mask_duration_days=bls.best_duration)
        fr = fit_transit(bls, min_period=cfg["min_period"], max_period=cfg["max_period"],
                         n_periods=cfg["n_periods"], time_override=fit_lc.time, flux_override=fit_lc.flux)
        fit.update(fr.as_row())
        fit["source_file"] = src
        if math.isfinite(fit.get("fit_depth_ppm", math.nan)) and off_depth > 0:
            fit["depth_ratio_fit_vs_official"] = fit["fit_depth_ppm"] / off_depth
        if math.isfinite(fit.get("fit_duration_hours", math.nan)) and off_dur_h > 0:
            fit["duration_ratio_fit_vs_official"] = fit["fit_duration_hours"] / off_dur_h

        # ---- vetting (reuse the cleaned light curve) ----
        cs, fl = _read_crowding(fp)
        vet["crowdsap"], vet["flfrcsap"] = cs, fl
        centroid = measure_centroid_shift(fp, period=off_period, epoch=off_epoch,
                                          duration_hours=off_dur_h, significance_threshold=3.0,
                                          min_shift_pixels=0.05)
        vet.update(centroid.as_row())
        vet["blend_flag"] = _blend_flag(centroid, cs)
        dur = assess_duration(period_days=off_period, duration_hours=off_dur_h)
        vet.update(dur.as_row())
        dd = off_dur_h / 24.0
        vet.update(measure_odd_even(clean.time, clean.flux, period=off_period, t0=off_epoch, duration_days=dd).as_row())
        vet.update(measure_secondary_eclipse(clean.time, clean.flux, period=off_period, t0=off_epoch, duration_days=dd).as_row())
        vet.update(measure_local_shape(clean.time, clean.flux, period=off_period, t0=off_epoch, duration_days=dd).as_row())
        vet["lc_vetting_status"] = "ok"
        vet["eb_flag"] = _eb_flag(vet)
        vet["source_file"] = src
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        if rec["status"] != "evaluated":
            rec.update(status="failed", recovery_class="processing_failed", message=msg)
        if not fit.get("fit_status"):
            fit.update(fit_status="processing_failed", message=msg)
        if vet.get("lc_vetting_status") != "ok":
            vet["lc_vetting_status"] = f"failed: {type(exc).__name__}"
    return rec, fit, vet


def build_parser():
    p = argparse.ArgumentParser(description="Parallel single-pass recovery+fit+vetting processor.")
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--suffix", type=str, required=True, help="Output suffix, e.g. 'labeled'.")
    p.add_argument("--fits-dir", type=Path, default=DOWNLOAD_LC_ROOT)
    p.add_argument("--workers", type=int, default=max(1, __import__("os").cpu_count() - 2))
    p.add_argument("--min-period", type=float, default=0.5)
    p.add_argument("--max-period", type=float, default=13.0)
    p.add_argument("--n-periods", type=int, default=20000)
    p.add_argument("--limit", type=int, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    targets = pd.read_csv(args.targets)
    if args.limit:
        targets = targets.head(args.limit)
    cfg = {"fits_dir": str(args.fits_dir), "min_period": args.min_period,
           "max_period": args.max_period, "n_periods": args.n_periods}
    payloads = [(row._asdict(), cfg) for row in targets.itertuples(index=False)]
    print(f"Processing {len(payloads)} targets on {args.workers} workers (single BLS per target)...", flush=True)

    recs, fitr, vetr = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, p) for p in payloads]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="target", file=sys.stdout):
            r, f, v = fut.result()
            recs.append(r); fitr.append(f); vetr.append(v)

    T = RESULT_TABLE_ROOT
    pd.DataFrame(recs)[RECOVERY_COLS].to_csv(T / f"tce_recovery_results_{args.suffix}.csv", index=False)
    pd.DataFrame(fitr)[FIT_COLS].to_csv(T / f"transit_fits_{args.suffix}.csv", index=False)
    pd.DataFrame(vetr)[VETTING_COLS].to_csv(T / f"vetting_features_{args.suffix}.csv", index=False)
    ok = sum(1 for r in recs if r["status"] == "evaluated")
    print(f"\nDone. evaluated={ok}/{len(recs)}. Wrote *_{args.suffix}.csv to {T.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
