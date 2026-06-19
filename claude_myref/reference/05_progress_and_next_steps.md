# DRISHTI — Progress Log & Next Steps

**Last updated:** 2026-06-18
**Reads with:** [04_master_plan.md](04_master_plan.md) (the authoritative plan this log executes against)

This is the running checkpoint: what has actually been built and verified, what it revealed, and
exactly what to pick up next. It is written so work can resume without the chat history.

---

## 1. Where we are in the plan

Executing **Sprint 0** and the first (zero-new-data) slice of **Sprint 1** from the master plan.
Everything below runs on the **111 LC FITS already on disk** — no new downloads were needed.

| Master-plan item | State |
|---|---|
| Sprint 0 — foundation (legacy notes, report stub) | ✅ done |
| Sprint 1 / WS-B v1 — physical transit fit + uncertainties | ✅ built + verified; full 111 run in progress |
| Sprint 1 / WS-D1 — centroid-shift + crowding vetting | ✅ built + verified; full 111 run done |
| Sprint 1 / WS-C — odd/even, secondary, V/U shape, duration sanity | ✅ built + verified; full 111 evidence vector produced |
| Sprint 2 / WS-F1 — rule-based classifier (transit/EB/blend/other) | ✅ built + run on 111 → pipeline is **PS-complete** |
| Sprint 2 — WS-H1 plot overlays, WS-D2/3/4 contamination/DV, WS-A1/2 | ⏳ next |
| Sprint 3 / WS-E — bootstrap labels (TOI + TESS-EB) | ✅ 482/1363 labeled; 67 overlap evidence |
| Sprint 3 / WS-F2 — ML classifier (planet vs EB) | ✅ full 525-set (66 planet/459 EB): ROC-AUC 0.94, bal-acc 0.80, planet P/R 0.61/0.65, EB 0.95/0.94 |
| Sprint 3 — expand labeled evidence (download+process) | ✅ 613 LCs; trainable 67→122→525 |
| Speed — parallel single-pass processor | ✅ process_targets_parallel.py (~15x: 1 BLS/target, 14 workers, 504 in 14 min) |
| Sprint 3 / WS-G — calibrated confidence + CDPP + master catalog | ✅ 615 candidates; probs sharply calibrated; CDPP on all |
| Sprint 2 / WS-H1 — classified diagnostic plots (fit overlay) | ✅ scripts/13_plot_classified.py |
| Sprint 3+ — operating-point tuning, blind sector | ⏳ later |

**Honest accuracy progression (important):** balanced 122-set gave planet P/R 0.89/0.89; the realistic
525-set (7:1 EB:planet) gave 0.61/0.65 — the balanced sample was optimistic. ROC-AUC held at 0.94 and
probabilities are sharply calibrated (>=0.75 -> 100% planets, <0.5 -> 0%), so ranking is strong and a
higher threshold yields a high-purity planet shortlist; the 0.5 default just depresses minority recall.

---

## 2. What was done this session

### 2.1 Sprint 0 — foundation
- **`data/drishti/results/tables/LEGACY_NOTES.md`** — documents the `150`-named-but-111-row files
  (`tce_recovery_results_150.csv`, `outputs/target_lists/tce_recovery_batch_150.csv`) and the `.bak`
  migration backups as **legacy**. Non-destructive; nothing deleted. States the go-forward naming
  convention (suffix by actual row count or explicit `batch_<n>_offset_<m>`).
- **`report/REPORT.md`** — the ≤3-page final-report stub, structured to the problem-statement
  deliverables (objective, methodology, assumptions, tools, uncertainty method, results, limitations).
  It is meant to grow each sprint, not be written at the end.

### 2.2 WS-B v1 — physical transit fit with uncertainties  *(core PS deliverable)*
- **`src/fitting/transit_fit.py`** — fits a symmetric **trapezoid** transit model to the phase-folded
  light curve, seeded from the BLS solution.
  - Parameters: `t0` (epoch refine), `depth`, total duration `T14`, ingress fraction `r`
    (`r→0` box, `r→1` V-shape). Parameterization keeps ingress ≤ half-duration inside simple bounds.
  - **Uncertainties** from the `scipy.optimize.curve_fit` covariance with `absolute_sigma=True`
    (per-point sigma = robust out-of-transit RMS), plus reduced-χ² and BIC for fit quality.
  - Period taken from BLS; period uncertainty reported as BLS grid resolution (conservative; to be
    refined by the planned MCMC v2 with `batman` + `emcee`).
  - Accepts `time_override`/`flux_override` so the caller can fit on depth-preserving flux (see §3.2).
- **`scripts/08_fit_transits.py`** — driver. Per target: find LC FITS → clean/flatten → BLS →
  **second masked-flatten pass** → trapezoid fit. Carries official period/duration/depth alongside the
  fitted values and emits `depth_ratio_fit_vs_official` and `duration_ratio_fit_vs_official` so
  parameter accuracy is directly measurable. Output: `data/drishti/results/tables/transit_fits.csv`
  (full run → `transit_fits_111.csv`). Resumable (skips already-fit `ok` rows unless `--force`).

### 2.5 WS-E + WS-F2 — labels + ML classifier  *(real accuracy number)*
- **`scripts/get_labels.py`** — crossmatches target TIC IDs against TOI/ExoFOP dispositions (NASA
  Exoplanet Archive `toi` table) and the TESS-EB catalog (Vizier J/ApJS/258/16) → `data/raw/labels/labels.csv`.
  Result: 482/1363 positive-target TICs labeled (375 EB, 61 planet, 12 FP, 34 candidate). EB membership
  takes priority in the class mapping (many TOI false positives are EBs).
- **`scripts/11_train_classifier.py`** — joins evidence + labels, trains a RandomForest (class-weight
  balanced) on planet vs eclipsing_binary, reports stratified-K-fold cross-validated metrics + feature
  importances, saves the model (`outputs/models/planet_eb_rf.joblib`) and CV predictions.
- Trainable overlap: 67 (56 EB, 11 planet). **CV result: ROC-AUC 0.96, balanced accuracy 0.85, accuracy
  0.94 (baseline 0.84)**; EB recall 0.98, planet recall 0.73. Top features = the engineered EB
  discriminators (secondary/primary ratio, odd/even, centroid, crowding, V-shape) — confirms the vetting
  carries real signal.
- **Concrete "need more data" finding:** 482 TICs are labeled but only 67 are trainable, because evidence
  exists for just 111 LCs. Downloading + processing LCs for the other ~415 labeled TICs is the single
  highest-leverage way to firm up classification accuracy (download-bound, not method-bound).

### 2.4 WS-F1 — rule-based classifier  *(makes the pipeline PS-complete)*
- **`src/models/classifier.py`** — transparent priority cascade over the joined evidence row →
  `planet_candidate` / `eclipsing_binary` / `blend` / `undetermined`, each with a heuristic
  `class_confidence` (0–1) and a human-readable `class_reason`. Priority: blend → EB → planet, because
  a transit-like dip failing a blend or EB test is not a clean planet.
- **`scripts/10_classify_candidates.py`** — joins recovery + transit-fit + vetting tables on
  (tic_id, sector), classifies each, writes `candidate_classifications_111.csv`.
- Full-111 result: **55 planet_candidate** (median conf 0.95), **34 eclipsing_binary** (0.75),
  **22 undetermined** (0.25), 0 blend. Internally consistent: all 22 undetermined == the not-recovered
  set; 3/4 alias-recovered → eclipsing_binary. This is the explainable baseline for the ML classifier (WS-F2).
- **The pipeline now answers every PS deliverable** on data in hand: detect → fit (params+uncertainty)
  → vet (crowded-field + eclipse) → classify → confidence, with a report draft.

### 2.3b WS-C — light-curve eclipse/shape evidence  *(EB discriminators)*
- **`src/vetting/odd_even.py`** — odd vs even transit depth; flags EB when the difference is both
  statistically significant **and** a meaningful fraction of the depth, requiring ≥2 distinct transits
  per parity (abstains as `unknown` otherwise — important for short single-sector baselines).
- **`src/vetting/secondary_eclipse.py`** — scans phase ~0.5 for an occultation; flags EB only when the
  secondary is significant **and deep** relative to the primary (`weak_secondary` if significant-but-shallow,
  consistent with a planetary occultation).
- **`src/vetting/duration_sanity.py`** — observed vs expected (Sun-like, central, circular) duration ratio.
- **`src/vetting/local_shape.py`** — inner/outer depth ratio as a V (grazing/EB) vs U (planet) metric.
- All four wired into `scripts/09_run_vetting.py`, which now emits the full per-candidate **evidence
  vector** (centroid + crowding + odd/even + secondary + shape + duration) plus an aggregate `eb_flag`.

Full-111 evidence-vector result: blend `109 on_target / 2 crowded`; EB `33 ok / 42 v_shape_watch /
36 eb_suspect` (21 odd/even, 17 secondary, 2 both). **Validation:** 3 of 4 alias-recovered targets are
flagged EB-suspect — a half-period alias is the classic EB primary/secondary pattern, so the vetting
independently recovers known astrophysics.

### 2.3 WS-D1 — crowded-field vetting (centroid + crowding)  *(PS blending concern)*
- **`src/vetting/centroid_shift.py`** — in-transit vs out-of-transit photometric centroid shift from
  `MOM_CENTR1/2` (fallback `PSF_CENTR1/2`), decorrelated against `POS_CORR1/2` pointing. Flags
  off-target (likely blend / background EB) when the shift is **both** statistically significant
  **and** physically large. Uses only columns already in the standard LC FITS — no TP download.
- **`scripts/09_run_vetting.py`** — driver. Computes the centroid shift using the official ephemeris,
  reads `CROWDSAP`/`FLFRCSAP` crowding from the header, and assigns a transparent `blend_flag`
  (`on_target` / `crowded_on_target` / `likely_blend` / `unknown`). Output:
  `data/drishti/results/tables/vetting_features_111.csv`. Resumable.

---

## 3. Findings (with evidence) — and the fixes applied

### 3.1 Centroid significance trap  → FIXED
Statistical significance **alone** flagged all targets as blends: with ~17,000 out-of-transit
cadences the standard error of the mean centroid is so small that a **0.001-pixel** shift reads as
3–7σ (and up to ~80σ), which is physically meaningless. **Fix:** require both `significance ≥ 3σ`
**and** `shift ≥ 0.05 px` (both CLI-tunable). After the fix, all 111 official TCEs correctly read
on-target (max measured shift ≈ 0.02 px). This establishes the on-target noise floor that a real
blind blend hunt will discriminate against.

### 3.2 Transit-depth suppression by flattening  → FIXED (significant accuracy win)
Fitted depths initially came out **~2–3× shallower than official**. Root cause: the Savitzky–Golay
flatten (window 401 cadences ≈ 13.4 h at 2-min cadence) fits *through* the ~3 h transit and absorbs
its depth. Confirmed empirically on 8 targets:

```text
depth ratio vs official:  normalized-only ≈ 0.9–1.0   |   flattened ≈ 0.08–0.67
```

**Fix:** transit-masked two-pass flattening — detect period/epoch/duration on the standard flatten,
then re-flatten with the in-transit cadences **excluded from the trend fit** (lightkurve `mask=`,
polarity verified empirically) and fit the trapezoid on that depth-preserving flux.

```text
depth ratio vs official after fix (cleanly recovered targets): ~0.9–1.0  (was ~0.3–0.6)
```

Implemented as `mask_period/mask_t0/mask_duration_days` args on
`load_clean_flattened_lightcurve()` in `src/preprocessing/clean_lightcurve.py` (+ `_build_transit_mask`
helper), used by the fit driver's second pass. **Detection (BLS) is left on the standard flatten** —
the suppression only affected the depth *parameter*, not period/epoch/duration recovery.

Secondary insight: the depth outliers that remain correlate with targets where BLS recovered the
wrong period/duration — so post-fix depth accuracy is a clean diagnostic of detection quality.

---

## 4. Verification status

| Check | Result |
|---|---|
| `08_fit_transits.py` smoke (8 targets) | 8/8 `ok`, reduced-χ² ≈ 0.7–1.3 |
| Depth accuracy after two-pass fix (8 targets) | ~0.9–1.0× official for cleanly-recovered targets |
| `09_run_vetting.py` full 111 | 109 on_target, 2 crowded_on_target, 0 false blend flags |
| Centroid data availability | 111/111 have `MOM_CENTR` (confirms "data in hand") |
| Full 111 transit fit | ✅ 111/111 `ok` → `transit_fits_111.csv` |

**Full-benchmark parameter accuracy (depth fix confirmed at scale):**
- Depth ratio (fit/official), all 111: median **0.92**, 16–84% [0.56, 1.08].
- Duration ratio, all 111: median **0.96**, 16–84% [0.86, 1.11].
- Direct-recovered (n=76): depth within ±25% for **84%**, duration within ±25% for **92%**, median
  formal depth uncertainty **3.0%**.
- Accuracy tracks detection quality: depth/duration ratios are clean for `direct_recovered` and
  degrade exactly for `alias`/`bad_duration`/`epoch_mismatch`/`not_recovered` — i.e. the fix holds and
  parameter error is now a diagnostic of detection, not a flattening artifact. (Full table in
  `report/REPORT.md` §6.2.)

---

## 5. Files added / modified this session

```text
ADDED
  claude_myref/reference/05_progress_and_next_steps.md   (this file)
  data/drishti/results/tables/LEGACY_NOTES.md
  report/REPORT.md
  src/fitting/transit_fit.py
  src/vetting/__init__.py
  src/vetting/centroid_shift.py
  scripts/08_fit_transits.py
  scripts/09_run_vetting.py
MODIFIED
  src/preprocessing/clean_lightcurve.py   (transit-masked flatten support)
GENERATED (git-ignored)
  data/drishti/results/tables/vetting_features_111.csv
  data/drishti/results/tables/transit_fits_111.csv   (on completion)
```

### How to run
```powershell
# Physical transit fit (+ uncertainties) over the 111-target benchmark
python scripts/08_fit_transits.py --output data/drishti/results/tables/transit_fits_111.csv --force

# Crowded-field vetting (centroid shift + crowding) over the benchmark
python scripts/09_run_vetting.py --output data/drishti/results/tables/vetting_features_111.csv --force
```

---

## 6. What to do next (in order)

### Immediate (finish Sprint 1)
1. **Quantify parameter accuracy across all 111** once `transit_fits_111.csv` is complete: depth-ratio
   and duration-ratio distributions vs official, split by recovery class; confirm the depth fix holds
   at scale. Write the numbers into `report/REPORT.md` §6 (Results).
2. **WS-C — remaining LC-only vetting evidence** (no new data, no labels):
   - `src/vetting/odd_even.py` — alternating odd/even transit depth (EB discriminator).
   - `src/vetting/secondary_eclipse.py` — search phase ~0.5 for an occultation (EB / hot companion).
   - `src/vetting/local_shape.py` — V vs U shape from the fitted `ingress_frac` (grazing/EB vs planet).
   - `src/vetting/duration_sanity.py` + `transit_snr.py` — physical-plausibility + significance flags.
   - Merge all evidence (incl. centroid + crowding) into one per-candidate **evidence vector** table.

### Then (Sprint 2 — makes the pipeline PS-complete)
3. **WS-F1 — rule-based classifier** (`src/models/classifier.py`, `scripts/10_classify_candidates.py`):
   combine the evidence vector into transparent rules → `predicted_class` ∈
   {transit / eclipsing_binary / blend / other} + a confidence proxy. No labels required.
4. **WS-D2/D3/D4** — Gaia/TIC neighbor query → full contamination score; mine `dvr-xml` DV reports
   (downloader already supports it) for official verdicts + a label source.
5. **WS-A1/A2** — make BLS `max_period` adaptive (≈28% of S1+S2 TCEs have P > 13 d, currently
   unreachable) and move the hardcoded Sector-1 systematic mask into per-sector config.
6. **WS-H1** — overlay `predicted_class` + confidence + the fitted trapezoid model on the per-target
   diagnostic plot (`07_plot_tce_recovery.py`).

### Later (Sprint 3–4 — from "complete" to "highly accurate")
7. **WS-E** — assemble type-disposition labels (curated set, or TOI/ExoFOP + TESS-EB + NEA by TIC ID).
8. **WS-F2** — train + cross-validate an ML classifier; report confusion matrix / precision-recall.
9. **WS-G** — calibrated confidence + CDPP noise integration (`data/Ref/*_rms-cdpp.csv`).
10. **WS-I / D5** — run the full pipeline on a blind science sector at scale; difference images on
    suspicious candidates (needs TP download); add diverse sectors for generalization.

### Parallel hygiene
- Add unit tests for the recovery classifier and the epoch/period/alias math (`tests/` is empty).
- Upgrade WS-B to v2: `batman` (limb-darkened) + `emcee` posteriors for fully Bayesian error bars.

---

## 7. Open items / watchouts
- **Depth-fix at scale unconfirmed** until the 111-target distribution is reviewed (§6.1).
- **Period uncertainty is grid-limited** (conservative) until the MCMC v2 lands.
- **"Blend" labels** have no clean catalog — must come from measured pixel/catalog evidence (WS-D), not
  a label table; keep this in mind when scoring classification.
- **Two-pass fit doubles runtime** (~6 s/target). Fine for 111; for a full sector, parallelize or fit
  only detected candidates.
