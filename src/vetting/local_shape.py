"""Transit-shape (V vs U) test (WS-C).

A planet transit has a flat-ish bottom (box / U-shape); a grazing transit or an eclipsing-binary
eclipse tends to be V-shaped (the flux keeps dropping to a sharp minimum). We measure shape directly
from the folded light curve as the ratio of the inner-transit depth to the outer-transit depth:

    inner = central 0-50% of the transit half-width
    outer = 50-100% of the transit half-width
    v_shape_metric = inner_depth / outer_depth

For a flat-bottomed (U) transit the inner and outer depths are similar (metric ~ 1). For a V-shape
the centre is much deeper than the wings (metric > ~1.5). This is an independent cross-check on the
trapezoid fit's ingress fraction.

Operates on a cleaned, normalized light curve (baseline ~1.0).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class LocalShapeResult:
    shape_status: str               # "ok" | "insufficient_points" | "error"
    v_shape_metric: float
    shape_flag: str                 # "u_shaped" | "v_shaped" | "unknown"

    def as_row(self) -> dict:
        return asdict(self)


def measure_local_shape(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period: float,
    t0: float,
    duration_days: float,
    v_threshold: float = 1.5,
) -> LocalShapeResult:
    if not (period > 0 and duration_days > 0) or time.size == 0:
        return _result("error")

    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    half = 0.5 * duration_days / period
    x = np.abs(phase) / half  # 0 at centre, 1 at contact
    inner = x <= 0.5
    outer = (x > 0.5) & (x <= 1.0)
    out_transit = (np.abs(phase) >= 1.5 * half) & (np.abs(phase) <= 5.0 * half)
    if inner.sum() < 5 or outer.sum() < 5 or out_transit.sum() < 20:
        return _result("insufficient_points")

    baseline = float(np.nanmedian(flux[out_transit]))
    inner_depth = baseline - float(np.nanmedian(flux[inner]))
    outer_depth = baseline - float(np.nanmedian(flux[outer]))
    if not np.isfinite(inner_depth) or inner_depth <= 0 or outer_depth <= 0:
        return _result("insufficient_points")

    metric = inner_depth / outer_depth
    flag = "v_shaped" if metric >= v_threshold else "u_shaped"
    return LocalShapeResult(shape_status="ok", v_shape_metric=float(metric), shape_flag=flag)


def _result(status: str) -> LocalShapeResult:
    return LocalShapeResult(shape_status=status, v_shape_metric=float("nan"), shape_flag="unknown")
