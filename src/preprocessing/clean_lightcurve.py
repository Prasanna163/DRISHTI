from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanedLightCurve:
    time: np.ndarray
    flux: np.ndarray
    window_length: int
    removed_quality_or_nan: int
    removed_outlier_or_systematic: int


def load_clean_flattened_lightcurve(
    fits_path: Path,
    *,
    flatten_window_length: int = 401,
    scatter_window: int = 101,
    scatter_sigma: float = 5.0,
    apply_known_systematic_masks: bool = True,
    mask_period: float | None = None,
    mask_t0: float | None = None,
    mask_duration_days: float | None = None,
    mask_width_durations: float = 1.0,
) -> CleanedLightCurve:
    """Load a TESS light curve and prepare it for transit-search inspection.

    If a transit ephemeris (``mask_period``, ``mask_t0``, ``mask_duration_days``) is supplied,
    the in-transit cadences are excluded from the Savitzky-Golay trend fit during flattening.
    This is essential for accurate *depth* recovery: a transit that occupies a non-trivial
    fraction of the flatten window is otherwise partially absorbed into the trend and its depth
    is suppressed (observed ~2-3x suppression with the default 401-cadence window). Detection
    (period/epoch/duration) is unaffected, so the standard call (no mask) is fine for BLS, and
    the masked call is used for the parameter-fitting pass.
    """
    lc = lk.read(str(fits_path))
    original_count = len(lc)

    if getattr(lc, "quality", None) is not None:
        lc = lc[lc.quality == 0]
    lc = lc.remove_nans()
    after_quality_nan = len(lc)

    lc = lc.normalize()
    lc = lc.remove_outliers(sigma=5)
    window_length = flatten_window_for_count(len(lc), preferred=flatten_window_length)
    transit_mask = _build_transit_mask(
        np.asarray(lc.time.value, dtype=float),
        period=mask_period,
        t0=mask_t0,
        duration_days=mask_duration_days,
        width_durations=mask_width_durations,
    )
    # lightkurve excludes cadences where mask is True from the trend fit.
    flat_lc = lc.flatten(window_length=window_length, mask=transit_mask)

    time = np.asarray(flat_lc.time.value, dtype=float)
    flux = np.asarray(flat_lc.flux.value, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]

    before_systematics = len(time)
    if apply_known_systematic_masks:
        good = known_systematic_mask(time, fits_path)
        time = time[good]
        flux = flux[good]

    time, flux = remove_high_scatter_regions(
        time,
        flux,
        window=scatter_window,
        sigma=scatter_sigma,
    )

    return CleanedLightCurve(
        time=time,
        flux=flux,
        window_length=window_length,
        removed_quality_or_nan=original_count - after_quality_nan,
        removed_outlier_or_systematic=before_systematics - len(time),
    )


def remove_high_scatter_regions(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    window: int = 101,
    sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove regions whose rolling MAD is much larger than the global MAD."""
    if len(time) < max(5, window):
        return time, flux

    df = pd.DataFrame({"time": time, "flux": flux})
    rolling_med = df["flux"].rolling(window, center=True).median()
    rolling_mad = (df["flux"] - rolling_med).abs().rolling(window, center=True).median()
    global_mad = np.nanmedian(np.abs(df["flux"] - np.nanmedian(df["flux"])))

    if not np.isfinite(global_mad) or global_mad == 0:
        return time, flux

    good = (rolling_mad < sigma * global_mad).fillna(True).to_numpy()
    return time[good], flux[good]


def _build_transit_mask(
    time: np.ndarray,
    *,
    period: float | None,
    t0: float | None,
    duration_days: float | None,
    width_durations: float = 1.0,
) -> np.ndarray | None:
    """Boolean mask (True = in-transit) for excluding transits from the flatten trend fit."""
    if period is None or t0 is None or duration_days is None:
        return None
    if not (period > 0 and duration_days > 0):
        return None
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    half_window_phase = 0.5 * width_durations * duration_days / period
    return np.abs(phase) <= half_window_phase


def known_systematic_mask(time: np.ndarray, fits_path: Path) -> np.ndarray:
    """Mask obvious sector-level spacecraft/systematic windows before BLS/TLS."""
    good = np.ones(len(time), dtype=bool)
    stem = fits_path.stem.lower()
    if "-s0001-" in stem:
        good &= ~((time > 1347.4) & (time < 1349.4))
    return good


def flatten_window_for_count(sample_count: int, preferred: int = 401) -> int:
    if sample_count <= 3:
        raise ValueError(f"Need more than 3 cadences to flatten a light curve; found {sample_count}.")
    window_length = min(preferred, sample_count - 1)
    if window_length % 2 == 0:
        window_length -= 1
    return max(window_length, 3)

