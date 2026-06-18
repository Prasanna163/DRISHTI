from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.io import fits

from data_access.load_fits import TessProductMeta, parse_tess_product_meta
from detection.run_bls import BlsSearchResult, bin_phase_curve, fold_lightcurve
from preprocessing.clean_lightcurve import (
    flatten_window_for_count,
    known_systematic_mask,
    remove_high_scatter_regions,
)


@dataclass(frozen=True)
class CleanedTableProduct:
    meta: TessProductMeta
    source_file: str
    table: pd.DataFrame
    summary: dict


def build_cleaned_lightcurve_table(fits_path: Path) -> CleanedTableProduct:
    """Build a row-per-cadence cleaned light-curve table with masks and flat flux."""
    meta = parse_tess_product_meta(fits_path)
    if meta is None:
        raise ValueError(f"Could not parse TIC/sector from filename: {fits_path.name}")

    with fits.open(fits_path, memmap=False) as hdul:
        hdu = _find_lightcurve_hdu(hdul)
        if hdu is None:
            raise ValueError(f"No LIGHTCURVE table found in {fits_path.name}")
        data = hdu.data
        names = set(data.columns.names)

        time = np.asarray(data["TIME"], dtype=float)
        sap_flux = _column_or_nan(data, "SAP_FLUX")
        pdcsap_flux = _column_or_nan(data, "PDCSAP_FLUX")
        quality = _column_or_default(data, "QUALITY", 0).astype(np.int64)
        sap_err = _column_or_nan(data, "SAP_FLUX_ERR")
        pdcsap_err = _column_or_nan(data, "PDCSAP_FLUX_ERR")

    if "PDCSAP_FLUX" in names:
        flux_raw = pdcsap_flux
        flux_err = pdcsap_err
    else:
        flux_raw = sap_flux
        flux_err = sap_err

    quality_mask = quality == 0
    finite_mask = np.isfinite(time) & np.isfinite(flux_raw)
    norm_base = np.nanmedian(flux_raw[quality_mask & finite_mask])
    if not np.isfinite(norm_base) or norm_base == 0:
        norm_base = np.nanmedian(flux_raw[finite_mask])

    flux_norm = flux_raw / norm_base
    flux_err_norm = flux_err / norm_base
    outlier_mask = _mad_clip_mask(flux_norm, quality_mask & finite_mask, sigma=5.0)
    systematics_mask = known_systematic_mask(time, fits_path)
    preliminary_mask = quality_mask & finite_mask & outlier_mask & systematics_mask

    flux_flat = np.full(len(time), np.nan)
    final_mask = np.zeros(len(time), dtype=bool)
    if preliminary_mask.sum() > 3:
        prelim_idx = np.flatnonzero(preliminary_mask)
        window_length = flatten_window_for_count(len(prelim_idx), preferred=401)
        lc = lk.LightCurve(
            time=time[prelim_idx],
            flux=flux_norm[prelim_idx],
            flux_err=flux_err_norm[prelim_idx],
        )
        flat_lc = lc.flatten(window_length=window_length)
        flat_time = _as_float_array(flat_lc.time)
        flat_flux = _as_float_array(flat_lc.flux)

        flux_flat[prelim_idx] = flat_flux
        final_time, _ = remove_high_scatter_regions(flat_time, flat_flux)
        final_keys = set(np.round(final_time, 10))
        final_mask[prelim_idx] = np.isin(np.round(time[prelim_idx], 10), list(final_keys))

    table = pd.DataFrame(
        {
            "tic_id": meta.tic_id,
            "sector": meta.sector,
            "time_btjd": time,
            "sap_flux": sap_flux,
            "pdcsap_flux": pdcsap_flux,
            "flux_raw": flux_raw,
            "flux_norm": flux_norm,
            "flux_flat": flux_flat,
            "flux_err": flux_err,
            "quality": quality,
            "quality_mask": quality_mask,
            "outlier_mask": outlier_mask,
            "systematics_mask": systematics_mask,
            "final_mask": final_mask,
        }
    )
    summary = summarize_lightcurve(meta, fits_path, table)
    return CleanedTableProduct(meta=meta, source_file=fits_path.name, table=table, summary=summary)


def summarize_lightcurve(
    meta: TessProductMeta,
    fits_path: Path,
    table: pd.DataFrame,
) -> dict:
    final = table[table["final_mask"] & np.isfinite(table["time_btjd"]) & np.isfinite(table["flux_flat"])]
    time = final["time_btjd"].to_numpy(dtype=float)
    flux_flat = final["flux_flat"].to_numpy(dtype=float)
    gaps = np.diff(time) if len(time) > 1 else np.array([])
    large_gaps = gaps[gaps > 0.3]
    cadence = np.nanmedian(gaps[gaps > 0]) if np.any(gaps > 0) else np.nan
    span = float(np.nanmax(time) - np.nanmin(time)) if len(time) else np.nan
    expected = span / cadence + 1 if np.isfinite(span) and np.isfinite(cadence) and cadence > 0 else np.nan
    duty_cycle = len(time) / expected if np.isfinite(expected) and expected > 0 else np.nan

    return {
        "tic_id": meta.tic_id,
        "sector": meta.sector,
        "source_file": fits_path.name,
        "n_points_raw": int(len(table)),
        "n_points_quality0": int(table["quality_mask"].sum()),
        "n_points_final": int(len(final)),
        "time_start_btjd": float(np.nanmin(time)) if len(time) else np.nan,
        "time_end_btjd": float(np.nanmax(time)) if len(time) else np.nan,
        "time_span_days": span,
        "duty_cycle": duty_cycle,
        "median_flux": float(np.nanmedian(table["flux_raw"])),
        "rms_ppm": float(np.nanstd(flux_flat - 1.0) * 1_000_000.0) if len(time) else np.nan,
        "mad_ppm": float(np.nanmedian(np.abs(flux_flat - np.nanmedian(flux_flat))) * 1_000_000.0)
        if len(time)
        else np.nan,
        "scatter_1hr_ppm": estimate_hourly_scatter_ppm(time, flux_flat),
        "num_gaps": int(len(large_gaps)),
        "largest_gap_days": float(np.nanmax(large_gaps)) if len(large_gaps) else 0.0,
    }


def candidate_row(
    meta: TessProductMeta,
    fits_path: Path,
    result: BlsSearchResult,
    *,
    min_period: float,
    max_period: float,
) -> dict:
    near_boundary = (
        result.best_period <= min_period + (max_period - min_period) * 0.01
        or result.best_period >= max_period - (max_period - min_period) * 0.01
    )
    flags = candidate_flags(result, near_period_boundary=near_boundary)
    return {
        "tic_id": meta.tic_id,
        "sector": meta.sector,
        "candidate_id": "BLS_candidate001",
        "source_file": fits_path.name,
        "period_days": result.best_period,
        "t0_btjd": result.best_t0,
        "duration_days": result.best_duration,
        "duration_hours": result.best_duration * 24.0,
        "depth": result.best_depth,
        "depth_ppm": result.best_depth * 1_000_000.0,
        "bls_power": result.best_power,
        "sde": result.sde,
        "snr": result.snr,
        "period_min": min_period,
        "period_max": max_period,
        "near_period_boundary": near_boundary,
        "n_transits": result.n_transits,
        "n_in_transit_points": result.n_in_transit_points,
        "rms_ppm": result.rms_ppm,
        "mad_ppm": result.mad_ppm,
        "status": "review",
        "flags": ";".join(flags),
    }


def periodogram_table(result: BlsSearchResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_days": result.period,
            "bls_power": result.power,
            "duration_days": result.duration,
            "t0_btjd": result.transit_time,
            "depth": result.depth,
            "depth_err": result.depth_err,
        }
    )


def folded_table(
    meta: TessProductMeta,
    result: BlsSearchResult,
    *,
    candidate_id: str = "BLS_candidate001",
) -> pd.DataFrame:
    phase, folded_flux = fold_lightcurve(result.time, result.flux, result.best_period, result.best_t0)
    phase_unsorted = ((result.time - result.best_t0 + 0.5 * result.best_period) % result.best_period) / result.best_period - 0.5
    order = np.argsort(phase_unsorted)
    transit_number = np.rint((result.time - result.best_t0) / result.best_period).astype(int)
    in_transit = np.abs(phase_unsorted) <= 0.5 * result.best_duration / result.best_period
    return pd.DataFrame(
        {
            "tic_id": meta.tic_id,
            "sector": meta.sector,
            "candidate_id": candidate_id,
            "time_btjd": result.time[order],
            "phase": phase,
            "flux_flat": folded_flux,
            "flux_err": np.nanstd(result.flux),
            "in_transit": in_transit[order],
            "transit_number": transit_number[order],
        }
    )


def folded_binned_table(
    meta: TessProductMeta,
    result: BlsSearchResult,
    *,
    candidate_id: str = "BLS_candidate001",
    bins: int = 150,
) -> pd.DataFrame:
    phase, folded_flux = fold_lightcurve(result.time, result.flux, result.best_period, result.best_t0)
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers, med = bin_phase_curve(phase, folded_flux, bins=bins)
    rows = []
    for idx in range(bins):
        in_bin = (phase >= edges[idx]) & (phase < edges[idx + 1])
        values = folded_flux[in_bin]
        rows.append(
            {
                "tic_id": meta.tic_id,
                "sector": meta.sector,
                "candidate_id": candidate_id,
                "phase_bin_center": centers[idx],
                "flux_median": med[idx],
                "flux_mean": float(np.nanmean(values)) if len(values) else np.nan,
                "flux_std": float(np.nanstd(values)) if len(values) else np.nan,
                "n_points": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def estimate_hourly_scatter_ppm(time: np.ndarray, flux: np.ndarray) -> float:
    if len(time) < 2:
        return float("nan")
    bins = np.floor((time - np.nanmin(time)) * 24.0).astype(int)
    grouped = pd.DataFrame({"bin": bins, "flux": flux}).groupby("bin")["flux"].median()
    if len(grouped) < 2:
        return float("nan")
    return float(np.nanstd(grouped.to_numpy() - 1.0) * 1_000_000.0)


def candidate_flags(result: BlsSearchResult, *, near_period_boundary: bool) -> list[str]:
    flags: list[str] = []
    if not np.isfinite(result.snr) or result.snr < 7.0:
        flags.append("low_snr")
    if result.n_transits < 2:
        flags.append("too_few_transits")
    if near_period_boundary:
        flags.append("near_period_boundary")
    if np.isfinite(result.best_depth) and result.best_depth * 1_000_000.0 < 3.0 * result.rms_ppm:
        flags.append("weak_folded_signal")
    return flags


def _find_lightcurve_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        names = getattr(getattr(data, "columns", None), "names", None)
        if names and "TIME" in names and ("SAP_FLUX" in names or "PDCSAP_FLUX" in names):
            return hdu
    return None


def _column_or_nan(data, name: str) -> np.ndarray:
    if name not in data.columns.names:
        return np.full(len(data), np.nan)
    return np.asarray(data[name], dtype=float)


def _column_or_default(data, name: str, default: float) -> np.ndarray:
    if name not in data.columns.names:
        return np.full(len(data), default)
    return np.asarray(data[name])


def _mad_clip_mask(values: np.ndarray, base_mask: np.ndarray, *, sigma: float) -> np.ndarray:
    mask = np.ones(len(values), dtype=bool)
    sample = values[base_mask & np.isfinite(values)]
    if len(sample) == 0:
        return mask
    med = np.nanmedian(sample)
    mad = np.nanmedian(np.abs(sample - med))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma == 0:
        return mask
    mask[base_mask] = np.abs(values[base_mask] - med) < sigma * robust_sigma
    return mask


def _as_float_array(value) -> np.ndarray:
    return np.asarray(getattr(value, "value", value), dtype=float)

