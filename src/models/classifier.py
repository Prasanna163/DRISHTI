"""Rule-based candidate classifier (WS-F1).

Combines the per-candidate evidence vector (detection + transit fit + crowded-field + eclipse/shape
vetting) into one of four problem-statement classes, with a transparent confidence and a human-readable
reason. This is deliberately a *transparent* rule cascade — every decision is explainable, which is
ideal for the report and as a baseline the ML classifier (WS-F2, once labels exist) must beat.

Classes
-------
planet_candidate   on-target, transit-like, no eclipsing-binary signature
eclipsing_binary   deep secondary eclipse and/or odd-even depth mismatch
blend              in-transit centroid offset => flux not on the target star
undetermined       no reliable independent detection (not recovered / low SNR) -> cannot classify

The cascade is priority-ordered: a confident blend or EB verdict overrides a planet call, because a
transit-like dip that fails a blend or EB test is, by definition, not a clean planet candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class ClassificationResult:
    predicted_class: str
    class_confidence: float          # 0..1, heuristic (calibrated version comes with WS-F2/WS-G)
    class_reason: str

    def as_row(self) -> dict:
        return asdict(self)


def _f(value, default=math.nan) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _clamp(x: float, lo: float = 0.05, hi: float = 0.98) -> float:
    return max(lo, min(hi, x))


def classify_candidate(features: dict, *, min_snr: float = 7.0) -> ClassificationResult:
    """Classify one candidate from its joined evidence row."""
    recovery_class = str(features.get("recovery_class", "") or "")
    snr = _f(features.get("our_snr"))
    blend_flag = str(features.get("blend_flag", "") or "")
    oddeven_flag = str(features.get("oddeven_flag", "") or "")
    secondary_flag = str(features.get("secondary_flag", "") or "")
    sec_ratio = _f(features.get("secondary_to_primary_ratio"))
    shape_flag = str(features.get("shape_flag", "") or "")
    duration_flag = str(features.get("duration_flag", "") or "")
    crowdsap = _f(features.get("crowdsap"))
    centroid_sigma = _f(features.get("centroid_shift_sigma"))

    # 0. No reliable independent detection -> cannot classify.
    if recovery_class in {"not_recovered", "download_failed", "processing_failed"} \
            or math.isnan(snr) or snr < min_snr:
        return ClassificationResult(
            "undetermined", _clamp(0.25, 0.05, 0.5),
            f"no reliable independent detection (recovery={recovery_class or 'n/a'}, snr={snr:.1f})",
        )

    # 1. Blend: in-transit centroid offset => signal not on the target star.
    if blend_flag == "likely_blend":
        conf = _clamp(0.55 + min(0.4, max(0.0, (centroid_sigma - 3.0) / 10.0)))
        return ClassificationResult(
            "blend", conf,
            f"off-target centroid shift ({centroid_sigma:.1f} sigma) => contaminating source",
        )

    # 2. Eclipsing binary: deep secondary and/or odd-even depth mismatch.
    eb_reasons = []
    conf = 0.5
    if secondary_flag == "eb_suspect":
        eb_reasons.append("deep secondary eclipse")
        conf += 0.25 + (0.1 if (math.isfinite(sec_ratio) and sec_ratio >= 0.5) else 0.0)
    if oddeven_flag == "eb_suspect":
        eb_reasons.append("odd/even depth mismatch")
        conf += 0.2
    if eb_reasons:
        if shape_flag == "v_shaped":
            conf += 0.05
            eb_reasons.append("V-shaped")
        return ClassificationResult("eclipsing_binary", _clamp(conf), "; ".join(eb_reasons))

    # 3. Planet candidate: on-target, transit-like, no EB signature.
    conf = 0.45
    reasons = ["on-target", "no significant secondary", "odd/even consistent or untested"]
    conf += 0.2 if snr >= 20 else (0.1 if snr >= 12 else 0.0)
    if shape_flag == "u_shaped":
        conf += 0.15
        reasons.append("U-shaped (flat-bottomed)")
    else:  # v_shaped or unknown shape -> grazing / EB not excluded
        conf -= 0.05
        reasons.append("V-shaped (grazing/EB not excluded)")
    if oddeven_flag == "ok":
        conf += 0.1
    if secondary_flag == "ok":
        conf += 0.1
    elif secondary_flag == "weak_secondary":
        conf -= 0.05
        reasons.append("weak secondary present")
    if duration_flag == "ok":
        conf += 0.05
    else:
        conf -= 0.1
        reasons.append(f"duration {duration_flag}")
    if recovery_class == "direct_recovered":
        conf += 0.1
    elif recovery_class == "alias_recovered":
        conf -= 0.1
        reasons.append("recovered at period alias")
    if math.isfinite(crowdsap) and crowdsap < 0.5:
        conf -= 0.1
        reasons.append(f"crowded aperture (CROWDSAP={crowdsap:.2f})")

    return ClassificationResult("planet_candidate", _clamp(conf), "; ".join(reasons))
