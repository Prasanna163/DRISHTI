from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares

from preprocessing.clean_lightcurve import CleanedLightCurve


@dataclass(frozen=True)
class BlsSearchResult:
    time: np.ndarray
    flux: np.ndarray
    period: np.ndarray
    power: np.ndarray
    transit_time: np.ndarray
    duration: np.ndarray
    depth: np.ndarray
    depth_err: np.ndarray
    best_period: float
    best_t0: float
    best_duration: float
    best_depth: float
    best_depth_err: float
    best_power: float
    sde: float
    snr: float
    n_transits: int
    n_in_transit_points: int
    rms_ppm: float
    mad_ppm: float


def run_bls_search(
    lightcurve: CleanedLightCurve,
    *,
    min_period: float = 0.5,
    max_period: float = 13.0,
    n_periods: int = 20000,
    min_duration: float = 0.02,
    max_duration: float = 0.30,
    n_durations: int = 20,
    clip_sigma: float = 6.0,
) -> BlsSearchResult:
    """Run Box Least Squares on a cleaned, flattened light curve."""
    time_bls, flux_bls = prepare_bls_arrays(
        lightcurve.time,
        lightcurve.flux,
        clip_sigma=clip_sigma,
    )
    if len(time_bls) < 10:
        raise ValueError(f"Need at least 10 valid cadences for BLS; found {len(time_bls)}.")

    flux_std = np.nanstd(flux_bls)
    if not np.isfinite(flux_std) or flux_std <= 0:
        flux_std = 1.0

    flux_err = np.ones_like(flux_bls) * flux_std
    bls = BoxLeastSquares(time_bls, flux_bls, dy=flux_err)

    period_grid = np.linspace(min_period, max_period, n_periods)
    duration_grid = np.linspace(min_duration, max_duration, n_durations)
    result = bls.power(period_grid, duration_grid)

    best_idx = int(np.nanargmax(result.power))
    depth = _array_or_nan(result, "depth", len(result.period))
    depth_err = _array_or_nan(result, "depth_err", len(result.period))
    transit_time = np.asarray(result.transit_time, dtype=float)
    duration = np.asarray(result.duration, dtype=float)
    power = np.asarray(result.power, dtype=float)
    best_period = float(result.period[best_idx])
    best_t0 = float(transit_time[best_idx])
    best_duration = float(duration[best_idx])
    best_depth = float(abs(depth[best_idx])) if np.isfinite(depth[best_idx]) else float("nan")
    best_depth_err = float(depth_err[best_idx]) if np.isfinite(depth_err[best_idx]) else float("nan")
    n_transits = count_transits(time_bls, best_period, best_t0)
    n_in_transit = count_in_transit_points(time_bls, best_period, best_t0, best_duration)
    rms = float(np.nanstd(flux_bls - 1.0))
    mad = float(np.nanmedian(np.abs(flux_bls - np.nanmedian(flux_bls))))
    snr = estimate_snr(best_depth, rms, n_in_transit)

    return BlsSearchResult(
        time=time_bls,
        flux=flux_bls,
        period=np.asarray(result.period, dtype=float),
        power=power,
        transit_time=transit_time,
        duration=duration,
        depth=depth,
        depth_err=depth_err,
        best_period=best_period,
        best_t0=best_t0,
        best_duration=best_duration,
        best_depth=best_depth,
        best_depth_err=best_depth_err,
        best_power=float(power[best_idx]),
        sde=estimate_sde(power, best_idx),
        snr=snr,
        n_transits=n_transits,
        n_in_transit_points=n_in_transit,
        rms_ppm=rms * 1_000_000.0,
        mad_ppm=mad * 1_000_000.0,
    )


def prepare_bls_arrays(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    clip_sigma: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply finite filtering and a final gentle MAD clip before BLS."""
    good = np.isfinite(time) & np.isfinite(flux)
    time_bls = np.asarray(time[good], dtype=float)
    flux_bls = np.asarray(flux[good], dtype=float)

    med = np.nanmedian(flux_bls)
    mad = np.nanmedian(np.abs(flux_bls - med))
    sigma = 1.4826 * mad

    if np.isfinite(sigma) and sigma > 0:
        keep = np.abs(flux_bls - med) < clip_sigma * sigma
        time_bls = time_bls[keep]
        flux_bls = flux_bls[keep]

    return time_bls, flux_bls


def fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
) -> tuple[np.ndarray, np.ndarray]:
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def bin_phase_curve(
    phase: np.ndarray,
    flux: np.ndarray,
    *,
    bins: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned_flux = np.full(bins, np.nan)

    for idx in range(bins):
        in_bin = (phase >= edges[idx]) & (phase < edges[idx + 1])
        if np.any(in_bin):
            binned_flux[idx] = np.nanmedian(flux[in_bin])

    return centers, binned_flux


def count_transits(time: np.ndarray, period: float, t0: float) -> int:
    if len(time) == 0 or period <= 0:
        return 0
    first = int(np.ceil((np.nanmin(time) - t0) / period))
    last = int(np.floor((np.nanmax(time) - t0) / period))
    return max(0, last - first + 1)


def count_in_transit_points(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> int:
    if len(time) == 0 or period <= 0:
        return 0
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    return int(np.sum(np.abs(phase) <= 0.5 * duration / period))


def estimate_sde(power: np.ndarray, best_idx: int) -> float:
    finite = np.isfinite(power)
    if not finite.any():
        return float("nan")
    center = np.nanmedian(power[finite])
    spread = np.nanstd(power[finite])
    if not np.isfinite(spread) or spread == 0:
        return float("nan")
    return float((power[best_idx] - center) / spread)


def estimate_snr(depth: float, rms: float, n_in_transit: int) -> float:
    if not np.isfinite(depth) or not np.isfinite(rms) or rms <= 0 or n_in_transit <= 0:
        return float("nan")
    return float(depth / (rms / np.sqrt(n_in_transit)))


def _array_or_nan(result, name: str, length: int) -> np.ndarray:
    value = getattr(result, name, None)
    if value is None:
        return np.full(length, np.nan)
    return np.asarray(value, dtype=float)
