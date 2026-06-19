"""Physical transit-shape fitting with uncertainty estimation (WS-B v1).

This module fits a symmetric **trapezoid** transit model to a phase-folded light curve,
seeded from a Box Least Squares (BLS) solution. It provides the problem-statement deliverable
"estimate transit depth, period, and duration by light-curve fitting" together with formal
1-sigma uncertainties (from the fit covariance matrix).

Design notes
------------
- Period is taken from BLS (a periodogram estimates period far better than a single-epoch
  shape fit). Its uncertainty is reported as the BLS grid resolution, which is a conservative
  upper bound; a future MCMC version (batman + emcee) will refine all parameters jointly.
- The trapezoid is parameterized as (t0, depth, total_duration T14, ingress_fraction r) where
  r = T_ingress / (T14/2) in [0, 1]. r -> 0 is a box; r -> 1 is a triangle (V-shape). This
  keeps the ingress <= half-duration constraint inside simple box bounds for curve_fit.
- Uncertainties use absolute_sigma=True with per-point sigma set to the robust out-of-transit
  RMS, so the covariance is in physical flux units rather than rescaled by reduced chi-square.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import curve_fit

from detection.run_bls import BlsSearchResult, fold_lightcurve


@dataclass(frozen=True)
class TransitFitResult:
    fit_status: str                 # "ok" | "failed"
    fit_period_days: float
    fit_period_err_days: float
    fit_t0_btjd: float
    fit_t0_err_days: float
    fit_depth_ppm: float
    fit_depth_err_ppm: float
    fit_duration_hours: float
    fit_duration_err_hours: float
    fit_ingress_frac: float         # 0 = box, 1 = V-shaped (triangular)
    fit_ingress_frac_err: float
    reduced_chi2: float
    bic: float
    n_points_fit: int
    message: str

    def as_row(self) -> dict:
        return asdict(self)


def trapezoid_flux(
    t_rel: np.ndarray,
    t0: float,
    depth: float,
    total_duration: float,
    ingress_frac: float,
) -> np.ndarray:
    """Normalized trapezoidal transit (baseline 1.0, dip of `depth`).

    Parameters
    ----------
    t_rel : time relative to the folded transit centre (days).
    t0 : small centre offset (days) to refine the epoch.
    depth : fractional transit depth (e.g. 0.001 = 1000 ppm).
    total_duration : first-to-fourth contact duration T14 (days).
    ingress_frac : ingress time as a fraction of the half-duration, in [0, 1].
    """
    dt = np.abs(t_rel - t0)
    half_total = 0.5 * max(total_duration, 1e-9)
    half_full = half_total * (1.0 - float(np.clip(ingress_frac, 0.0, 1.0)))

    flux = np.ones_like(dt, dtype=float)
    # Flat transit bottom.
    flux[dt <= half_full] = 1.0 - depth
    # Linear ingress/egress ramp between flat bottom and baseline.
    ramp = (dt > half_full) & (dt < half_total)
    span = max(half_total - half_full, 1e-12)
    frac = (dt[ramp] - half_full) / span
    flux[ramp] = (1.0 - depth) + depth * frac
    return flux


def fit_transit(
    bls: BlsSearchResult,
    *,
    min_period: float,
    max_period: float,
    n_periods: int,
    window_durations: float = 4.0,
    time_override: np.ndarray | None = None,
    flux_override: np.ndarray | None = None,
) -> TransitFitResult:
    """Fit a trapezoid model to the BLS-folded light curve and return params + uncertainties.

    `window_durations` restricts the fit to +/- window_durations * BLS-duration around the
    transit (plus baseline shoulders), which stabilizes the duration/ingress estimate.

    `time_override`/`flux_override` let the caller supply depth-preserving flux (e.g. flattened
    with the transit masked) while still using the BLS ephemeris (period/t0/duration) as the
    fold and seed. This is the recommended path for accurate depth recovery.
    """
    period = float(bls.best_period)
    bls_duration = float(bls.best_duration)
    if not np.isfinite(period) or period <= 0 or not np.isfinite(bls_duration) or bls_duration <= 0:
        return _failed("invalid BLS seed (period/duration)")

    fold_time = bls.time if time_override is None else np.asarray(time_override, dtype=float)
    fold_flux = bls.flux if flux_override is None else np.asarray(flux_override, dtype=float)
    if fold_time.shape != fold_flux.shape or fold_time.size == 0:
        return _failed("override time/flux shape mismatch or empty")

    phase, flux_sorted = fold_lightcurve(fold_time, fold_flux, period, bls.best_t0)
    t_rel = phase * period  # days from folded transit centre

    # Restrict to a window around the transit (keeps baseline shoulders for an anchor).
    half_window = max(window_durations * bls_duration, 0.5 * bls_duration + 0.05)
    in_window = np.abs(t_rel) <= half_window
    t_fit = t_rel[in_window]
    f_fit = flux_sorted[in_window]
    finite = np.isfinite(t_fit) & np.isfinite(f_fit)
    t_fit, f_fit = t_fit[finite], f_fit[finite]
    n = int(t_fit.size)
    if n < 12:
        return _failed(f"too few in-window points to fit ({n})")

    # Per-point scatter from robust out-of-transit RMS.
    sigma_flux = _robust_sigma(f_fit)
    if not np.isfinite(sigma_flux) or sigma_flux <= 0:
        sigma_flux = 1.0
    sigma = np.full(n, sigma_flux, dtype=float)

    depth0 = float(bls.best_depth) if np.isfinite(bls.best_depth) and bls.best_depth > 0 else 3.0 * sigma_flux
    p0 = [0.0, depth0, bls_duration, 0.5]
    lower = [-bls_duration, 0.0, 1e-4, 0.0]
    upper = [bls_duration, 0.5, min(0.5 * period, 12.0 * bls_duration), 1.0]
    # Keep the seed strictly inside the bounds.
    p0 = [float(np.clip(v, lo + 1e-9, hi - 1e-9)) for v, lo, hi in zip(p0, lower, upper)]

    try:
        popt, pcov = curve_fit(
            trapezoid_flux, t_fit, f_fit, p0=p0,
            sigma=sigma, absolute_sigma=True,
            bounds=(lower, upper), maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001 - report any optimizer failure as a fit failure
        return _failed(f"{type(exc).__name__}: {exc}")

    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    t0_off, depth, total_dur, ingress = (float(v) for v in popt)
    t0_err, depth_err, dur_err, ingress_err = (float(v) for v in perr)

    model = trapezoid_flux(t_fit, *popt)
    resid = (f_fit - model) / sigma
    dof = max(n - len(popt), 1)
    chi2 = float(np.sum(resid ** 2))
    reduced_chi2 = chi2 / dof
    bic = chi2 + len(popt) * np.log(n)

    period_resolution = (max_period - min_period) / max(n_periods - 1, 1)

    return TransitFitResult(
        fit_status="ok",
        fit_period_days=period,
        fit_period_err_days=float(period_resolution),
        fit_t0_btjd=float(bls.best_t0 + t0_off),
        fit_t0_err_days=t0_err,
        fit_depth_ppm=depth * 1e6,
        fit_depth_err_ppm=depth_err * 1e6,
        fit_duration_hours=total_dur * 24.0,
        fit_duration_err_hours=dur_err * 24.0,
        fit_ingress_frac=ingress,
        fit_ingress_frac_err=ingress_err,
        reduced_chi2=reduced_chi2,
        bic=float(bic),
        n_points_fit=n,
        message="",
    )


def _robust_sigma(flux: np.ndarray) -> float:
    med = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - med))
    return float(1.4826 * mad)


def _failed(message: str) -> TransitFitResult:
    nan = float("nan")
    return TransitFitResult(
        fit_status="failed",
        fit_period_days=nan, fit_period_err_days=nan,
        fit_t0_btjd=nan, fit_t0_err_days=nan,
        fit_depth_ppm=nan, fit_depth_err_ppm=nan,
        fit_duration_hours=nan, fit_duration_err_hours=nan,
        fit_ingress_frac=nan, fit_ingress_frac_err=nan,
        reduced_chi2=nan, bic=nan, n_points_fit=0,
        message=message,
    )
