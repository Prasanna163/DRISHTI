# DRISHTI — Feasible Changes (mapped to the Problem Statement)

**Date:** 2026-06-18
**Companion to:** [01_repo_analysis.md](01_repo_analysis.md)

The guiding principle below: **every recommendation maps to a specific, scored line in the problem
statement.** The current repo is an excellent *detection + recovery* engine; the work that remains is
to turn it into the *detect → classify → fit → quantify confidence → report* pipeline the PS asks for.

Recovery (the current focus) should be **kept as an internal validation harness**, not abandoned —
but it is not the deliverable. The deliverable is classification + fitting + confidence on science data.

---

## 0. Re-framing the goal (read this first)

| The repo currently asks | The problem statement asks |
|---|---|
| "Did our BLS re-find a *known* TESS TCE?" | "Given a noisy light curve, **what is this dip** (transit / eclipse / blend / other), **how confident are we**, and **what are its fitted parameters ± uncertainty**?" |

So the recovery table becomes a *means* (it proves the detector works and provides labeled examples),
and three new capabilities become the *ends*: **(A) a classifier, (B) a physical transit fit with
uncertainties, (C) a calibrated confidence + a report.**

---

## 1. Prioritized roadmap

Ordered by *PS impact ÷ effort*. Phases 1–3 are the minimum to satisfy the PS; 4–6 raise the score.

### Phase 1 — Physical transit fit + real uncertainties  ⭐ highest PS impact
**Closes:** "Estimate period, duration, depth by light-curve fitting" + "how uncertainties are estimated".
**Why first:** It is the most-scored deliverable, the code slot already exists (`src/fitting/`), and it
needs no new labeled data — it runs on the LC FITS already downloaded.

Build `src/fitting/transit_fit.py`:
- Seed from the existing BLS result (period, t0, duration, depth) — already computed.
- Fit a physical transit model. Two feasible options:
  - **`batman` (Mandel–Agol)** for a true limb-darkened transit shape (most rigorous), or
  - **`scipy.optimize.curve_fit` on a trapezoid/box model** (zero new dependency; `scipy` already in
    `requirements.txt`) as a lighter first version.
- Report parameters **with uncertainties**:
  - From `curve_fit` covariance, **or**
  - From an MCMC posterior (`emcee`) for credible intervals — gives report-grade error bars.
- Output a `fit_*` block alongside the BLS block: `fit_period ± σ`, `fit_t0 ± σ`, `fit_duration ± σ`,
  `fit_depth ± σ`, plus reduced-χ² / BIC for fit quality.

**Deliverable:** `data/drishti/results/tables/transit_fits_*.csv` and a fitted-model overlay added to
the existing phase-fold panel in `07_plot_tce_recovery.py`.

---

### Phase 2 — Vetting evidence features  ⭐ enables classification
**Closes:** the discriminating features needed for "transit vs eclipse vs blend vs other" and for confidence.
**Why second:** These are deterministic, physics-based, need no training labels, and are independently
useful. The README already names these exact modules under `src/vetting/`.

Build `src/vetting/` (each returns a number + a pass/fail flag):
| Module | Discriminates | Cheap to compute? |
|---|---|---|
| `odd_even.py` | EB (alternating depths) vs planet | ✅ (LC only) |
| `secondary_eclipse.py` | EB / hot companion (occultation at phase 0.5) | ✅ (LC only) |
| `transit_snr.py` | weak vs strong → confidence | ✅ |
| `duration_sanity.py` | physically implausible duration → FP | ✅ (already partly in recovery) |
| `local_shape.py` | V-shape (EB/grazing) vs U-shape (planet) | ✅ |
| `period_alias.py` | harmonic confusion | ✅ (logic already in recovery) |

**Deliverable:** an "evidence vector" per candidate, written to a features table. This is the input
both to the classifier (Phase 3) and to the confidence score (Phase 5).

> Pixel-level vetting (`centroid_shift.py`, `difference_image.py`, `contamination_score.py`) directly
> addresses the PS's **"stellar blending / crowded fields"** language but needs the TP/DV products that
> are not yet downloaded. Treat as Phase 6 (stretch) — flag blends from catalog-neighbor + centroid
> only once `*_tp.fits` is available.

---

### Phase 3 — Classification framework  ⭐ core PS deliverable
**Closes:** "Develop a classification framework to categorize dips into transits, eclipses, blends, other"
+ "Apply the classifier on the science datasets".
**Depends on:** Phase 2 features + a labeled set.

Two feasible tracks — recommend doing **3a then 3b**:

- **3a. Rule-based / decision-tree classifier (ship this first).**
  Combine the Phase-2 evidence flags into transparent rules:
  ```text
  secondary eclipse present OR odd/even depth mismatch        → eclipsing_binary
  centroid shifts off-target / strong neighbor (when TP avail) → blend / contaminant
  V-shaped, deep, short                                        → likely EB / FP
  U-shaped, consistent odd/even, no secondary, on-target       → planet_candidate
  low SNR / few transits / inconsistent                        → other / junk
  ```
  No training data required; fully explainable (good for the report's methodology section).

- **3b. ML classifier (`scikit-learn`, already a dependency, currently unused).**
  Train `RandomForest`/`GradientBoosting` on the Phase-2 evidence vectors using the **curated label
  set the PS says will be provided** (planets / FPs / EBs). Land labels in `data/raw/labels/` (currently
  empty). Until that arrives, **bootstrap labels** from public catalogs joinable by TIC ID:
  TOI dispositions (CP/KP = planet, FP = false positive) and known-EB catalogs. Report
  precision/recall/confusion matrix (cross-validated) — directly scored under "accuracy of classification".

Build `src/models/classifier.py` + a `scripts/08_classify_candidates.py` step. Output a
`predicted_class` + `class_probability` column.

---

### Phase 4 — Run on a real science dataset (not just the answer key)
**Closes:** "Apply the classifier on the *given science datasets*" + scale ("20–30k light curves").
- The current 111 LCs are all *known* TCEs. Add a path to ingest a blind/unlabeled set of LCs (a full
  sector subset via the existing manifest streaming — the infrastructure is already there).
- Run detect → fit → vet → classify end-to-end and emit a candidate catalog with classes + confidence.
- This demonstrates the pipeline on data where the answer is *not* pre-known — which is the actual task.

---

### Phase 5 — Calibrated confidence (replace the SNR proxy)
**Closes:** "Provide the confidence level of the detected signal" properly.
- Map SDE/SNR + vetting evidence to a **calibrated** detection confidence / false-alarm probability,
  e.g. via the classifier's predicted probability (3b) or a bootstrap/false-alarm-probability estimate
  on the periodogram.
- Integrate the existing **CDPP noise CSVs** (`data/Ref/*_rms-cdpp.csv`, currently unused) to express
  confidence *relative to each star's intrinsic noise* — this is the natural way to handle the PS's
  "noisy light curves / detector response" framing, and it explains failures (noise-limited vs missed).

---

### Phase 6 — Crowded-field / blend handling (stretch, highest scientific value)
**Closes:** the PS's explicit emphasis on "crowded fields", "stellar blending", "contamination".
- Download TP (`*_tp.fits`) and DV products for candidates (downloader already supports `tp`, `dvr-*`).
- Implement `centroid_shift.py` + `difference_image.py` + catalog-neighbor `contamination_score.py`.
- This is what lets the classifier output **"blend"** with evidence rather than by inference.

---

## 2. Quick wins / hygiene (do alongside, low effort)

1. **Write the 3-page report** (`report/` is empty — it is a required deliverable). Methodology,
   assumptions, tools/libraries, and the uncertainty method from Phase 1. The existing summary plots
   are nearly report-ready. *Do not leave this to the end.*
2. **Overlay the class label + confidence on the per-target diagnostic plot** (`07_plot_tce_recovery.py`)
   — directly satisfies "visualization of the light curve along with the detected and classified signal".
3. **Add unit tests** for the recovery classifier and the epoch/period/alias math (`tests/` is empty);
   this is the highest-risk untested logic.
4. **Resolve the `150`-named-but-111-row legacy files** (already flagged in the implementation plan §6):
   archive/rename so the source of truth is unambiguous.
5. **De-hardcode the systematic mask** — move the Sector-1 `[1347.4, 1349.4]` window into a per-sector
   config so the pipeline scales beyond S1/S2.
6. **Make BLS max-period adaptive** (or add an official-period-centered diagnostic search) to recover
   longer-period `not_recovered` cases — but only after the failure-mode triage in the implementation
   plan's Phase 3, per the existing "don't tune blindly" rule.

---

## 3. Suggested new structure (fills the empty slots)

```text
src/
  fitting/
    transit_fit.py        Phase 1 — physical (batman/trapezoid) fit + uncertainties
  vetting/
    odd_even.py           Phase 2
    secondary_eclipse.py
    transit_snr.py
    duration_sanity.py
    local_shape.py
    period_alias.py
    centroid_shift.py     Phase 6 (needs TP)
    difference_image.py   Phase 6
    contamination_score.py Phase 6
  models/
    classifier.py         Phase 3 — rule-based + sklearn

scripts/
  08_fit_transits.py      Phase 1 driver
  09_run_vetting.py       Phase 2 driver
  10_classify_candidates.py  Phase 3 driver

data/raw/labels/          Phase 3 — curated planet/FP/EB labels (provided or bootstrapped)
report/                   3-page report (required)
```

---

## 4. Minimum path to "PS-complete"

If time is constrained, the smallest set that makes every PS line *answerable*:

1. **Phase 1** (fit + uncertainties) — on existing 111 LCs.
2. **Phase 2** (odd/even + secondary + shape + SNR) — on existing 111 LCs.
3. **Phase 3a** (rule-based classifier from Phase-2 evidence) — no labels needed.
4. **Quick win #1** (write the 3-page report) + **#2** (class label on plots).

That alone moves the scorecard in [01_repo_analysis.md §1](01_repo_analysis.md) from
*"detection only"* to *"detect + classify + fit + confidence + report"* — i.e. it answers every
deliverable, with the ML classifier (3b), science-scale run (4), and blend handling (6) as score-raisers.

---

## 5. What NOT to do

- **Don't chase the headline recovery %.** It is an internal metric, not a PS deliverable, and the
  implementation plan already warns against blind tuning.
- **Don't download 20–30k LCs before the classifier exists.** Scale is cheap to add later via the
  existing streaming path; build the missing science layers first on the 111 LCs you already have.
- **Don't discard the recovery harness.** It is the validation that proves the detector and supplies
  trustworthy labeled examples for training/calibration.
