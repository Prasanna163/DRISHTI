"""Transit-duration sanity check (WS-C).

For a given orbital period, a central transit across a Sun-like star has a characteristic
duration. An observed duration that is far longer than expected suggests a grazing geometry, an
eclipsing binary, an evolved/large host star, or a wrong (e.g. aliased) period; far shorter is
also suspicious. This is a physics-based plausibility flag, not a definitive classifier.

Expected duration (central, circular, small companion, Sun-like host):
    a   = P^(2/3)            [AU, solar mass via Kepler III, P in years]
    T14 = (P / pi) * (R_sun / a)
We use generous thresholds because real stellar radii vary by an order of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

_R_SUN_AU = 0.00465047
_DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class DurationSanityResult:
    duration_status: str            # "ok" | "error"
    expected_duration_hours: float
    duration_sanity_ratio: float    # observed / expected
    duration_flag: str              # "ok" | "too_long" | "too_short" | "unknown"

    def as_row(self) -> dict:
        return asdict(self)


def assess_duration(
    *,
    period_days: float,
    duration_hours: float,
    long_ratio: float = 3.0,
    short_ratio: float = 0.25,
) -> DurationSanityResult:
    if not (period_days > 0 and duration_hours > 0):
        return DurationSanityResult("error", float("nan"), float("nan"), "unknown")

    period_years = period_days / _DAYS_PER_YEAR
    a_au = period_years ** (2.0 / 3.0)
    expected_days = (period_days / np.pi) * (_R_SUN_AU / a_au)
    expected_hours = expected_days * 24.0
    ratio = duration_hours / expected_hours if expected_hours > 0 else float("nan")

    if not np.isfinite(ratio):
        flag = "unknown"
    elif ratio > long_ratio:
        flag = "too_long"
    elif ratio < short_ratio:
        flag = "too_short"
    else:
        flag = "ok"

    return DurationSanityResult(
        duration_status="ok",
        expected_duration_hours=float(expected_hours),
        duration_sanity_ratio=float(ratio),
        duration_flag=flag,
    )
