# DRISHTI — What To Do Next (and how it gets us to the goal)

**Date:** 2026-06-18
**Reads with:** [01_repo_analysis.md](01_repo_analysis.md) · [02_recommendations.md](02_recommendations.md)

This is the forward plan: a sequenced set of milestones, each with **what you do**, **what it
produces**, and **why it moves us toward the destination**. It is written so you can hand it to
anyone and they know the next move without the chat history.

---

## 0. The destination (what "done" looks like)

> A pipeline that takes a **noisy TESS light curve it has never seen**, and returns:
> the **detected periodic dip**, a **class** (transit / eclipse / blend / other) with a **calibrated
> confidence**, the **fitted period, depth, and duration with uncertainties**, a **plot** showing all
> of it, and a **≤3-page report** of methodology + assumptions + uncertainty handling.

Map of that destination onto the official evaluation criteria:

| Evaluation criterion | What in our pipeline answers it |
|---|---|
| Accuracy of event detection & classification | BLS recovery (proven) + classifier (Milestone 3) |
| Accuracy of estimated parameters | Physical transit fit + uncertainties (Milestone 1) |
| Methods / approach | Rule-based + ML, documented in the report |
| Visualization & clarity | Diagnostic plots with class + confidence overlaid (Milestone 5) |

Everything below is the shortest honest path from **where we are** (a strong detector +
recovery benchmark) to **that**.

---

## 1. Where we are right now (one paragraph)

We can download TESS light curves, clean them, run BLS, and prove we re-discover known TESS signals
(80% period recovery on 111 targets). That means **the detection engine works**. What we do **not**
yet have: a physical fit with error bars, any classifier, calibrated confidence, type-labels to train
on, or the report. Sector 1+2 data is enough to *build* the first three; the *fourth* (labels) is the
real bottleneck for classifier accuracy — see [02_recommendations.md §1 Phase 3](02_recommendations.md)
and the data-strategy note in Milestone 3 below.

---

## 2. The plan, as a line of sight

```text
[NOW]  Detector + recovery benchmark works (S1+S2, 111 LCs)
   │
   ├─ M1  Physical transit fit + uncertainties      ──► answers "estimate params + uncertainty"
   │
   ├─ M2  Vetting evidence (odd/even, secondary, …)  ──► creates the features that distinguish classes
   │
   ├─ M3  Classifier (rules first, then ML on labels) ──► answers "classify transit/EB/blend/other"
   │
   ├─ M4  Confidence calibration (+ CDPP noise)       ──► answers "confidence level of detection"
   │
   ├─ M5  Visualization + the 3-page report           ──► answers "visualization" + the report deliverable
   │
   └─ M6  Run on a blind science sector (+ scale)     ──► answers "apply to the given science datasets"
[GOAL] detect → classify → fit → quantify → report, on unseen data
```

M1, M2, M5(report skeleton) can start **immediately on the 111 LCs we already have** — no new data,
no waiting. M3's ML half and M6 are the only parts that need new inputs (labels / a blind sector).

---

## 3. Milestones (do them in this order)

### M1 — Physical transit fit + uncertainties  *(start here)*
- **Do:** Create `src/fitting/transit_fit.py` + `scripts/08_fit_transits.py`. Seed from the existing
  BLS result; fit a trapezoid model with `scipy.optimize.curve_fit` (zero new deps) for v1, then add
  `batman` + `emcee` for limb-darkened shape and posterior error bars.
- **Produces:** `data/drishti/results/tables/transit_fits_*.csv` with
  `fit_period±σ, fit_t0±σ, fit_duration±σ, fit_depth±σ, reduced_χ²/BIC`.
- **Gets us closer because:** this is the single most-scored PS deliverable ("estimate parameters by
  light-curve fitting" + "how uncertainties are estimated"). After M1 we can put **real error bars**
  on every number — which recovery alone never gave us.
- **Done when:** every recovered target has fitted params with uncertainties, and the fit overlays the
  phase-folded data sensibly.

### M2 — Vetting evidence layer
- **Do:** Build `src/vetting/{odd_even, secondary_eclipse, local_shape, duration_sanity, transit_snr,
  period_alias}.py` + `scripts/09_run_vetting.py`. Each emits a number + a flag. All LC-only,
  deterministic, **no labels needed**.
- **Produces:** an "evidence vector" per candidate → `data/drishti/results/tables/vetting_features_*.csv`.
- **Gets us closer because:** these features are *the physics that separates a planet from an EB or a
  blend* (alternating depths → EB, secondary eclipse → EB/hot companion, V-shape → grazing/EB). They
  are the input to both the classifier (M3) and the confidence score (M4).
- **Done when:** each of the 111 targets has a complete evidence vector.

### M3 — Classification framework  *(the core PS deliverable)*
- **3a (do first, no data needed):** `src/models/classifier.py` rule-based — combine M2 flags into
  transparent rules (secondary eclipse / odd-even mismatch → `eclipsing_binary`; off-target centroid →
  `blend` once TP available; clean U-shape, on-target, no secondary → `planet_candidate`; low-SNR /
  inconsistent → `other`). Fully explainable → great for the report.
- **3b (when labels exist):** train `RandomForest`/`GradientBoosting` (sklearn — already a dependency,
  currently unused) on the M2 evidence vectors.
- **Data strategy (the honest bottleneck):** S1+S2 raw light curves are **not** enough to train an
  accurate generalizing classifier — too few planet positives, single-epoch systematics. The unblock is
  **type-disposition labels**, not more sectors:
  - Use the **curated label set the PS says will be provided**, OR
  - Bootstrap labels by joining on TIC ID: **TOI catalog** (CP/KP = planet, FP = false positive) and
    the **TESS Eclipsing Binary catalog** (EB). These span many sectors → balanced, diverse classes.
  - For real classifier robustness, gather labels across **3–5 sectors of different mission phases**
    (early South, later, North, extended) so the model learns transit morphology, not S1/S2 quirks.
- **Produces:** `predicted_class` + `class_probability` per candidate; for 3b, a cross-validated
  confusion matrix / precision-recall (directly scored under "accuracy of classification").
- **Gets us closer because:** this *is* the classification deliverable. 3a makes the pipeline
  PS-complete with no new data; 3b raises the accuracy score once labels land.

### M4 — Calibrated confidence
- **Do:** Replace the raw SNR/SDE proxy with a calibrated confidence — the classifier's predicted
  probability (3b) or a periodogram false-alarm-probability estimate. **Integrate the CDPP noise CSVs**
  (`data/Ref/*_rms-cdpp.csv`, currently unused) so confidence is expressed *relative to each star's
  intrinsic noise*.
- **Gets us closer because:** answers "provide the confidence level of the detected signal" properly,
  and the CDPP join lets us say *why* a faint signal is uncertain (noise-limited vs genuinely absent) —
  exactly the PS's "noisy light curves / detector response" framing.

### M5 — Visualization + the report  *(report skeleton can start at M1)*
- **Do:** (a) Extend `07_plot_tce_recovery.py` to overlay **predicted class + confidence + fitted model**
  on the per-target diagnostic. (b) Write the **≤3-page report** in `report/` (currently empty): methodology,
  assumptions, tools/libraries, uncertainty method (from M1/M4).
- **Gets us closer because:** these are two explicit deliverables ("visualization of the light curve
  along with the detected and classified signal" + the required report). Start the report early and grow
  it as milestones land — do **not** leave it to the end.

### M6 — Run on a blind science dataset + scale
- **Do:** Point the (already-built) manifest-streaming path at a sector of **unlabeled** LCs, run
  detect → fit → vet → classify → confidence end-to-end, emit a candidate catalog.
- **Gets us closer because:** answers "apply the classifier on the *given science datasets*" — the
  pipeline now works where the answer is *not* pre-known, which is the actual task. Scale (20–30k LCs)
  is cheap to add here once the science layers exist.

### M6+ (stretch, highest scientific value) — crowded-field / blend handling
- Download TP/DV products (downloader already supports them); add `centroid_shift`, `difference_image`,
  `contamination_score`. This lets the classifier output **"blend"** with pixel evidence rather than by
  inference — directly addressing the PS's emphasis on crowded fields and stellar blending.

---

## 4. Minimum path to "PS-complete" (if time is tight)

**M1 → M2 → M3a → M5.** That alone makes every problem-statement line *answerable* on the data we
already have, with no waiting on labels:
- detection ✅ (already) · parameters+uncertainty ✅ (M1) · classification ✅ (M3a) ·
  confidence 🟡→✅ (M2 SNR/evidence; full calibration at M4) · visualization ✅ (M5) · report ✅ (M5).

Then **M3b (labels) → M4 → M6 → M6+** are the score-raisers that take it from "complete" to
"highly accurate".

---

## 5. Parallel hygiene (cheap, do alongside)

From [01_repo_analysis.md §6](01_repo_analysis.md): add unit tests for the recovery/epoch/period math
(`tests/` is empty); resolve the `150`-named-but-111-row legacy files; de-hardcode the Sector-1
systematic mask into per-sector config; make BLS `max_period` adaptive (≈28% of S1+S2 TCEs are
period > 13 d and currently unreachable).

---

## 6. Immediate next action

**Start M1** (physical transit fit + uncertainties) on the 111 LCs we already have, and **open a stub
`report/` document** to grow as we go. Both need zero new data. In parallel, begin sourcing
**type-disposition labels** (curated set or TOI + TESS-EB bootstrap) so M3b is unblocked by the time
M2 is done.

If you want, I can implement **M1** now: scaffold `src/fitting/transit_fit.py` + `scripts/08_fit_transits.py`,
wire it to the existing BLS output, and add the fitted-model overlay to the diagnostic plot.
