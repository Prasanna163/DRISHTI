"""Secondary-eclipse search (WS-C).

A significant flux dip near phase 0.5 (the occultation) indicates a self-luminous companion —
an eclipsing binary or a hot brightness source — rather than a planet transit. We scan phases
around 0.5 (covering mild eccentricity) for the deepest dip, measure its depth and significance,
and report the secondary-to-primary depth ratio (a strong EB indicator: comparable depths => EB).

Operates on a cleaned, normalized light curve (baseline ~1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class SecondaryEclipseResult:
    secondary_status: str           # "ok" | "insufficient_points" | "error"
    primary_depth_ppm: float
    secondary_depth_ppm: float
    secondary_snr: float
    secondary_phase: float
    secondary_to_primary_ratio: float
    secondary_flag: str             # "ok" | "eb_suspect" | "unknown"

    def as_row(self) -> dict:
        return asdict(self)


def measure_secondary_eclipse(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period: float,
    t0: float,
    duration_days: float,
    snr_threshold: float = 5.0,
    min_depth_ratio: float = 0.3,
) -> SecondaryEclipseResult:
    if not (period > 0 and duration_days > 0) or time.size == 0:
        return _result("error")

    # Orbital phase in [0, 1): 0 == primary transit, 0.5 == opposite side of orbit.
    phi = ((time - t0) / period) % 1.0
    half = 0.5 * duration_days / period
    if half <= 0 or half > 0.2:
        return _result("error")

    # Primary depth near phase 0 (and the 1.0 wrap).
    dist_primary = np.minimum(phi, 1.0 - phi)
    in_primary = dist_primary <= half
    # Baseline: away from both the primary and the secondary region.
    out_of_events = (dist_primary > 2.0 * half) & (np.abs(phi - 0.5) > 2.0 * half)
    if in_primary.sum() < 5 or out_of_events.sum() < 30:
        return _result("insufficient_points")

    baseline = float(np.nanmedian(flux[out_of_events]))
    scatter = float(np.nanstd(flux[out_of_events]))
    primary_depth = baseline - float(np.nanmedian(flux[in_primary]))

    # Scan candidate secondary phase centres around 0.5 (covers mild eccentricity).
    best_depth, best_phase, best_n = -np.inf, 0.5, 0
    for centre in np.linspace(0.4, 0.6, 21):
        win = np.abs(phi - centre) <= half
        n = int(win.sum())
        if n < 5:
            continue
        depth = baseline - float(np.nanmedian(flux[win]))
        if depth > best_depth:
            best_depth, best_phase, best_n = depth, centre, n

    if best_n < 5 or not np.isfinite(scatter) or scatter <= 0:
        return _result("insufficient_points")

    sec_snr = best_depth / (scatter / np.sqrt(best_n))
    ratio = (best_depth / primary_depth) if primary_depth > 0 else float("nan")
    # EB suspect requires a SIGNIFICANT and DEEP secondary (comparable to the primary).
    # A significant-but-shallow secondary (ratio below the floor) is consistent with a planetary
    # occultation / reflection and is reported as "weak_secondary" rather than EB.
    if np.isfinite(sec_snr) and sec_snr >= snr_threshold and best_depth > 0:
        flag = "eb_suspect" if (np.isfinite(ratio) and ratio >= min_depth_ratio) else "weak_secondary"
    else:
        flag = "ok"

    return SecondaryEclipseResult(
        secondary_status="ok",
        primary_depth_ppm=primary_depth * 1e6,
        secondary_depth_ppm=best_depth * 1e6,
        secondary_snr=float(sec_snr),
        secondary_phase=float(best_phase),
        secondary_to_primary_ratio=float(ratio),
        secondary_flag=flag,
    )


def _result(status: str) -> SecondaryEclipseResult:
    nan = float("nan")
    return SecondaryEclipseResult(
        secondary_status=status,
        primary_depth_ppm=nan, secondary_depth_ppm=nan, secondary_snr=nan,
        secondary_phase=nan, secondary_to_primary_ratio=nan, secondary_flag="unknown",
    )
