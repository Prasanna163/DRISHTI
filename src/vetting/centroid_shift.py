"""In-transit centroid-shift test (WS-D1): is the dip on the target star?

This directly addresses the problem statement's crowded-field / stellar-blending concern.
TESS pixels are ~21 arcsec, so flux from a nearby star can leak into the aperture and produce a
transit-like dip that is NOT on the target. During a real on-target transit the photometric
centroid stays put; if the dip comes from a neighbour, the centroid shifts towards/away from that
source while the dip is happening.

We measure the mean centroid in-transit vs out-of-transit (using a known ephemeris), after
decorrelating the centroid against the spacecraft pointing (`POS_CORR1/2`). A statistically
significant shift flags a likely blend / background eclipsing binary.

Crucially, this uses columns already present in the standard TESS LC FITS
(`MOM_CENTR1/2`, `PSF_CENTR1/2`, `POS_CORR1/2`) — no target-pixel download required.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class CentroidShiftResult:
    centroid_status: str            # "ok" | "no_centroid_data" | "insufficient_points" | "error"
    centroid_source: str            # "mom" | "psf" | ""
    centroid_shift_pixels: float
    centroid_shift_sigma: float     # significance of the shift
    centroid_dcol_pixels: float
    centroid_drow_pixels: float
    centroid_n_in_transit: int
    centroid_n_out_transit: int
    centroid_on_target: bool        # True if shift is NOT significant (consistent with on-target)
    centroid_message: str

    def as_row(self) -> dict:
        return asdict(self)


def measure_centroid_shift(
    fits_path: Path,
    *,
    period: float,
    epoch: float,
    duration_hours: float,
    significance_threshold: float = 3.0,
    min_shift_pixels: float = 0.05,
) -> CentroidShiftResult:
    """Measure the in-vs-out-of-transit centroid shift for one LC FITS file.

    Parameters
    ----------
    period, epoch, duration_hours : transit ephemeris (e.g. official TCE values, in days/BTJD/hours).
    significance_threshold : minimum sigma for the shift to be considered statistically real.
    min_shift_pixels : minimum *physical* shift magnitude to flag off-target.

    A target is flagged off-target (likely blend) only when the shift is BOTH statistically
    significant AND physically large. With TESS's ~tens-of-thousands of cadences, the standard
    error of the mean centroid is tiny, so a sub-milli-pixel shift can be many sigma yet be
    physically meaningless; requiring a magnitude floor (default 0.05 px) avoids that false alarm.
    """
    if not (np.isfinite(period) and period > 0 and np.isfinite(epoch) and np.isfinite(duration_hours) and duration_hours > 0):
        return _result("error", message="invalid ephemeris")

    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            cols = set(data.columns.names)
            time = np.asarray(data["TIME"], dtype=float)
            quality = np.asarray(data["QUALITY"], dtype=float) if "QUALITY" in cols else np.zeros_like(time)
            col, row, source = _select_centroid(data, cols)
            pos1 = np.asarray(data["POS_CORR1"], dtype=float) if "POS_CORR1" in cols else None
            pos2 = np.asarray(data["POS_CORR2"], dtype=float) if "POS_CORR2" in cols else None
    except Exception as exc:  # noqa: BLE001
        return _result("error", message=f"{type(exc).__name__}: {exc}")

    if col is None or row is None:
        return _result("no_centroid_data", message="no MOM_CENTR/PSF_CENTR columns")

    good = (
        np.isfinite(time) & np.isfinite(col) & np.isfinite(row) & (quality == 0)
    )
    time, col, row = time[good], col[good], row[good]
    if pos1 is not None and pos2 is not None:
        pos1, pos2 = pos1[good], pos2[good]
    if time.size < 50:
        return _result("insufficient_points", source=source, message=f"only {time.size} good cadences")

    # Remove pointing-driven centroid motion learned from out-of-transit baseline.
    col = _decorrelate(col, pos1, pos2)
    row = _decorrelate(row, pos1, pos2)

    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    half_dur_phase = 0.5 * (duration_hours / 24.0) / period
    in_transit = np.abs(phase) <= half_dur_phase
    # Out-of-transit baseline: exclude transit and a guard band around it.
    out_transit = np.abs(phase) >= (2.0 * half_dur_phase)

    n_in = int(np.sum(in_transit))
    n_out = int(np.sum(out_transit))
    if n_in < 5 or n_out < 20:
        return _result(
            "insufficient_points", source=source,
            message=f"in_transit={n_in}, out_transit={n_out}",
        )

    dcol, sig_col = _mean_shift(col[in_transit], col[out_transit])
    drow, sig_row = _mean_shift(row[in_transit], row[out_transit])

    shift = float(np.hypot(dcol, drow))
    denom = np.hypot(sig_col, sig_row)
    significance = float(shift / denom) if denom > 0 else float("nan")
    # Off-target requires BOTH statistical significance AND a physically meaningful magnitude.
    off_target = bool(
        np.isfinite(significance)
        and significance >= significance_threshold
        and shift >= min_shift_pixels
    )
    on_target = not off_target

    return CentroidShiftResult(
        centroid_status="ok",
        centroid_source=source,
        centroid_shift_pixels=shift,
        centroid_shift_sigma=significance,
        centroid_dcol_pixels=float(dcol),
        centroid_drow_pixels=float(drow),
        centroid_n_in_transit=n_in,
        centroid_n_out_transit=n_out,
        centroid_on_target=on_target,
        centroid_message="",
    )


def _select_centroid(data, cols: set[str]):
    """Prefer moment centroids (robust); fall back to PSF centroids."""
    if "MOM_CENTR1" in cols and "MOM_CENTR2" in cols:
        c = np.asarray(data["MOM_CENTR1"], dtype=float)
        r = np.asarray(data["MOM_CENTR2"], dtype=float)
        if np.isfinite(c).sum() > 50:
            return c, r, "mom"
    if "PSF_CENTR1" in cols and "PSF_CENTR2" in cols:
        c = np.asarray(data["PSF_CENTR1"], dtype=float)
        r = np.asarray(data["PSF_CENTR2"], dtype=float)
        if np.isfinite(c).sum() > 50:
            return c, r, "psf"
    return None, None, ""


def _decorrelate(centroid: np.ndarray, pos1: np.ndarray | None, pos2: np.ndarray | None) -> np.ndarray:
    """Subtract a linear pointing model centroid ~ a*POS_CORR1 + b*POS_CORR2 + c."""
    if pos1 is None or pos2 is None:
        return centroid - np.nanmedian(centroid)
    design = np.column_stack([pos1, pos2, np.ones_like(pos1)])
    valid = np.all(np.isfinite(design), axis=1) & np.isfinite(centroid)
    if valid.sum() < 10:
        return centroid - np.nanmedian(centroid)
    coef, *_ = np.linalg.lstsq(design[valid], centroid[valid], rcond=None)
    return centroid - design @ coef


def _mean_shift(in_vals: np.ndarray, out_vals: np.ndarray) -> tuple[float, float]:
    """Difference of means and its combined standard error."""
    mean_in = float(np.nanmean(in_vals))
    mean_out = float(np.nanmean(out_vals))
    se_in = float(np.nanstd(in_vals, ddof=1) / np.sqrt(max(in_vals.size, 1)))
    se_out = float(np.nanstd(out_vals, ddof=1) / np.sqrt(max(out_vals.size, 1)))
    return mean_in - mean_out, float(np.hypot(se_in, se_out))


def _result(status: str, *, source: str = "", message: str = "") -> CentroidShiftResult:
    nan = float("nan")
    return CentroidShiftResult(
        centroid_status=status,
        centroid_source=source,
        centroid_shift_pixels=nan,
        centroid_shift_sigma=nan,
        centroid_dcol_pixels=nan,
        centroid_drow_pixels=nan,
        centroid_n_in_transit=0,
        centroid_n_out_transit=0,
        centroid_on_target=False,
        centroid_message=message,
    )
