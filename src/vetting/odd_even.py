"""Odd/even transit-depth test (WS-C).

If alternating transits have significantly different depths, the period is likely half the true
value and the "transits" are the primary/secondary eclipses of an eclipsing binary. Comparing the
mean depth of odd-numbered vs even-numbered transits is a classic EB discriminator.

Operates on a cleaned, normalized light curve (baseline ~1.0). Depths are measured *relative* to the
out-of-transit baseline, so flatten-induced depth suppression cancels and does not bias the test.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class OddEvenResult:
    oddeven_status: str             # "ok" | "insufficient_points" | "error"
    depth_odd_ppm: float
    depth_even_ppm: float
    oddeven_diff_sigma: float
    oddeven_depth_frac_diff: float  # |odd-even| / mean depth
    oddeven_flag: str               # "ok" | "eb_suspect" | "unknown"

    def as_row(self) -> dict:
        return asdict(self)


def measure_odd_even(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period: float,
    t0: float,
    duration_days: float,
    sigma_threshold: float = 3.0,
    min_frac_diff: float = 0.2,
) -> OddEvenResult:
    if not (period > 0 and duration_days > 0) or time.size == 0:
        return _result("error")

    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    half = 0.5 * duration_days / period
    in_transit = np.abs(phase) <= half
    out_transit = (np.abs(phase) >= 1.5 * half) & (np.abs(phase) <= 5.0 * half)
    if in_transit.sum() < 10 or out_transit.sum() < 20:
        return _result("insufficient_points")

    baseline = float(np.nanmedian(flux[out_transit]))
    transit_num = np.round((time - t0) / period).astype(int)
    odd = in_transit & (transit_num % 2 != 0)
    even = in_transit & (transit_num % 2 == 0)
    # Require enough points AND at least 2 distinct transits in each parity; otherwise a single
    # noisy transit dominates one parity's median and produces a spurious odd/even difference.
    n_odd_transits = np.unique(transit_num[odd]).size
    n_even_transits = np.unique(transit_num[even]).size
    if odd.sum() < 5 or even.sum() < 5 or n_odd_transits < 2 or n_even_transits < 2:
        return _result("insufficient_points")

    depth_odd = baseline - float(np.nanmedian(flux[odd]))
    depth_even = baseline - float(np.nanmedian(flux[even]))
    se_odd = float(np.nanstd(flux[odd]) / np.sqrt(odd.sum()))
    se_even = float(np.nanstd(flux[even]) / np.sqrt(even.sum()))
    denom = float(np.hypot(se_odd, se_even))
    sigma = abs(depth_odd - depth_even) / denom if denom > 0 else float("nan")
    mean_depth = 0.5 * (depth_odd + depth_even)
    frac_diff = abs(depth_odd - depth_even) / mean_depth if mean_depth > 0 else float("nan")
    # Require BOTH statistical significance AND a physically meaningful fractional difference:
    # with many in-transit cadences a sub-percent difference can be many-sigma yet meaningless.
    eb_suspect = (
        np.isfinite(sigma) and sigma >= sigma_threshold
        and np.isfinite(frac_diff) and frac_diff >= min_frac_diff
    )

    return OddEvenResult(
        oddeven_status="ok",
        depth_odd_ppm=depth_odd * 1e6,
        depth_even_ppm=depth_even * 1e6,
        oddeven_diff_sigma=float(sigma),
        oddeven_depth_frac_diff=float(frac_diff),
        oddeven_flag="eb_suspect" if eb_suspect else "ok",
    )


def _result(status: str) -> OddEvenResult:
    nan = float("nan")
    return OddEvenResult(
        oddeven_status=status,
        depth_odd_ppm=nan, depth_even_ppm=nan,
        oddeven_diff_sigma=nan, oddeven_depth_frac_diff=nan, oddeven_flag="unknown",
    )
