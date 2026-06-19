# DRISHTI — Repository Analysis

**Date:** 2026-06-18
**Reviewer:** Claude (repo analysis)
**Scope:** Full code + data audit of `e:\Prasanna\ISRO`, mapped against the problem statement
*"AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves"*.

This document records **what exists**, **how it works**, and **where it stands relative to the
stated objective**. Recommendations live in [02_recommendations.md](02_recommendations.md).

---

## 1. Executive Summary

DRISHTI is a well-engineered **TESS TCE recovery and validation pipeline**. Given the official
TESS Threshold Crossing Event (TCE) catalog as an "answer key", it downloads light curves, cleans
them, runs a Box Least Squares (BLS) transit search, and measures how well it independently
re-discovers the official period / epoch / duration. The download, storage, orchestration, and
visualization layers are mature and resumable.

**However, against the problem statement the project is roughly half-built.** The PS asks for a
pipeline that **detects, classifies (transit / eclipse / blend / other), fits physical transit
parameters with uncertainties, and reports confidence**. Today the project does the **detection**
part strongly, but:

| PS requirement | Status |
|---|---|
| Identify periodic dips in noisy light curves | ✅ **Done** (BLS, cleaning, SNR/SDE) |
| Classify dips into transit / eclipse / blend / other | ❌ **Not built** (no classifier, no labels) |
| Apply classifier to science datasets | ❌ **Not built** |
| Provide SNR / significance | 🟡 **Partial** (SNR + SDE computed; not calibrated to confidence) |
| Estimate transit depth, period, duration via *light-curve fitting* | 🟡 **Partial** (BLS box estimates only; no physical model fit, no parameter uncertainties) |
| Visualize light curve + classified signal | 🟡 **Partial** (rich diagnostics exist; no class label overlaid) |
| Confidence level of detection | 🟡 **Partial** (SNR/SDE proxy only) |
| ≤3-page report (methodology, assumptions, tools, uncertainties) | ❌ **Not written** (`report/` is empty) |

The single most important gap: **the project measures "did we recover a known signal?" but the
problem statement asks "what *kind* of signal is this, how confident are we, and what are its
fitted parameters with error bars?"** Recovery is a good internal benchmark, but it is not itself
the deliverable.

---

## 2. Repository Layout (project files only; `.venv/` excluded)

```text
config.yaml                 Path config (legacy data/ paths + canonical data/drishti/)
requirements.txt            numpy pandas matplotlib scipy astropy lightkurve scikit-learn
pyproject.toml / setup.py   Packaging
README.md                   Project overview (accurate, detailed)
docs/drishti_implementation_plan.md   Project checkpoint + phased plan (2026-06-18)

scripts/
  drishti.py                Main CLI: splash, init-store, discover, tce-recovery,
                            stream-manifest, plan-manifest, download-plan (~1089 lines)
  select_tce_targets.py     Build target lists from official TCE CSVs
  05_download_tce_products.py   Download LC/TP/DV products via MAST (astroquery)
  06_run_tce_recovery.py    Clean + BLS + recovery classification  ← core science logic
  07_plot_tce_recovery.py   Summary + per-target diagnostic plots
  01..04_*.py               Earlier standalone inspect / BLS / export / stream utilities
  run_all.py, run_pipeline.py   End-to-end orchestration wrappers
  migrate_outputs.py        Legacy outputs/ → data/drishti/ migration

src/
  drishti_store.py          Canonical storage layout + legacy migration
  data_access/load_fits.py  FITS discovery + TESS filename metadata parsing
  preprocessing/clean_lightcurve.py   Quality mask, normalize, flatten, systematic removal
  detection/run_bls.py      BLS search + SDE/SNR + folding helpers
  features/quantitative_products.py   Per-cadence tables, periodogram, folded tables, flags
  visualization/plot_bls_results.py, plot_fits_inspection.py
  fitting/                  EMPTY (only __init__.py)   ← physical transit fitting missing
  models/                   EMPTY (only __init__.py)   ← classifier / ML models missing

data/
  Ref/                      Official Sector 1 & 2 TCE-stats CSVs + CDPP CSVs (the "answer key")
  drishti/                  Canonical artifact store (targets, downloads, results, plots, logs)
  raw/labels/               EMPTY (.gitkeep only)   ← no curated classification training set
  drishti/ref/              EMPTY

outputs/                    LEGACY artifacts (periodograms, plots, *.bak tables)
report/                     EMPTY (.gitkeep only)   ← required 3-page report not started
```

---

## 3. How the Current Pipeline Works (end-to-end)

```text
data/Ref/*_dvr-tcestats.csv   (official TESS TCE catalog = answer key)
        │  select_tce_targets.py  (filter: SNR≥7.1, ≥2 transits, converged; dedup best-per-sector)
        ▼
data/drishti/targets/*.csv    (target lists; starter set is stricter: SNR≥20, depth 200–10000 ppm)
        │  05_download_tce_products.py (MAST/astroquery)  OR  manifest streaming (drishti.py)
        ▼
data/drishti/downloads/lc/*_lc.fits   (currently 111 LC FITS files)
        │  clean_lightcurve.py  (quality==0, normalize, flatten win=401, MAD 5σ clip,
        │                        rolling-scatter region removal, Sector-1 systematic mask)
        ▼
cleaned/flattened light curve
        │  run_bls.py  (BoxLeastSquares: P 0.5–13 d, 20000 periods × 20 durations 0.02–0.30 d)
        ▼
BLS result (best period/t0/duration/depth, power, SDE, SNR)
        │  06_run_tce_recovery.py  (compare to official; classify recovery)
        ▼
data/drishti/results/tables/tce_recovery_results_*.csv
        │  07_plot_tce_recovery.py
        ▼
data/drishti/results/plots/  (summary + 2×2 per-target diagnostics)
```

### 3.1 Preprocessing (`src/preprocessing/clean_lightcurve.py`)
- Loads via `lightkurve`, keeps `QUALITY == 0`, drops NaNs, normalizes, sigma-clips outliers (5σ).
- `flatten()` with adaptive odd window (default 401 cadences).
- `remove_high_scatter_regions()` — rolling-window MAD vs global MAD (window 101, 5σ).
- `known_systematic_mask()` — **hardcoded** Sector-1 exclusion window `[1347.4, 1349.4]` BTJD.
- MAD→σ conversion uses the standard 1.4826 factor.

### 3.2 Detection (`src/detection/run_bls.py`)
- `astropy.timeseries.BoxLeastSquares`. Grid: period 0.5–13 d × 20000 points, 20 durations 0.02–0.30 d.
- `estimate_sde()` = `(peak_power − median) / std(power)`.
- `estimate_snr()` = `depth / (rms / sqrt(n_in_transit))`.
- Folding + 150-bin phase binning helpers.

### 3.3 Recovery classification (`scripts/06_run_tce_recovery.py`) — core logic
Compares BLS result to the official TCE and assigns one of:

```text
direct_recovered                 period (direct) + epoch + SNR ok, duration not crazy
alias_recovered                  matched at half/double period, SNR ok
period_recovered_bad_duration    period ok but duration ratio > 3×
period_recovered_epoch_mismatch  period ok but epoch score < 0.5
period_recovered_needs_vetting   only ultra-tight (<0.1%) period match
not_recovered                    none of the above
download_failed / processing_failed
```

Key formulas / thresholds (all CLI-tunable):
- Period error % = `|P_bls − P_off| / P_off × 100`; harmonic check at ×2 and ÷2.
- Epoch match score = `max(0, 1 − Δphase/duration)`, gate `≥ 0.5`.
- Duration ratio = `max(d/d_off, d_off/d)`, "bad" if `> 3.0`.
- SNR gate `≥ 7.0`; period tolerance `1.0%`; vetting tolerance `0.1%`.

### 3.4 Current validation snapshot (111 rows; from implementation plan)
```text
direct_recovered                76
alias_recovered                  4
period_recovered_bad_duration    5
period_recovered_epoch_mismatch  4
not_recovered                   22
→ direct+alias = 80/111 (72%);  any period recovery = 89/111 (80%)
```

---

## 4. Data Inventory

| Asset | Location | State |
|---|---|---|
| Official TCE stats (S1, S2) | `data/Ref/*_dvr-tcestats.csv` | ✅ Present (the answer key) |
| CDPP noise CSVs (S1, S2) | `data/Ref/*_rms-cdpp.csv` | ✅ Present, **not yet joined** into results |
| Target lists | `data/drishti/targets/` | ✅ positive (2299), starter (143), batches 50/111 |
| Downloaded LC FITS | `data/drishti/downloads/lc/` | 🟡 **111 files** (PS suggests a full sector ≈ 20–30k) |
| TP / DV products | `data/drishti/downloads/{tp,dv}/` | ❌ Not downloaded (needed for blend/centroid work) |
| **Curated classification labels** (planets / FPs / EBs) | `data/raw/labels/` | ❌ **EMPTY** — blocks the classifier |
| Recovery results | `data/drishti/results/tables/` | ✅ multiple batches |
| Legacy outputs | `outputs/` | 🟡 present, flagged legacy; naming confusion (see §6) |

---

## 5. Strengths (keep these)

1. **Robust data engineering** — resumable downloads, `.part` files, FITS validation, status CSVs,
   manifest streaming with cache cleanup. This is the hard, tedious part and it is done well.
2. **Honest, well-structured recovery metric** — the multi-class recovery taxonomy (direct / alias /
   epoch-mismatch / bad-duration) is far more informative than a binary pass/fail.
3. **Clean module separation** — `data_access / preprocessing / detection / features / visualization`
   is a sensible skeleton; the empty `fitting/` and `models/` are clearly the intended next slots.
4. **Strong visualization** — 2×2 per-target diagnostics (LC + periodogram + dual phase-folds) and
   summary scatter/bar plots are genuinely useful and close to report-grade.
5. **Good documentation discipline** — README + implementation plan are accurate and current.

---

## 6. Issues & Risks

### 6.1 Conceptual gap (highest priority)
The pipeline answers *"can we recover a known TCE?"* not *"what is this signal and how confident
are we?"*. The PS deliverables — **classification, physical fitting, calibrated confidence** — are
the parts that are missing, and they are exactly what the evaluation criteria score.

### 6.2 No classifier / no training labels
- `src/models/` is empty; `scikit-learn` is a declared dependency but unused.
- `data/raw/labels/` is empty — there is no curated transit / EB / FP / blend label set, which the
  PS explicitly says "will be provided" and is required to train any classifier.

### 6.3 No physical transit fit
- `src/fitting/` is empty. Depth/duration come from the BLS **box** model only. The PS asks for
  parameters from **light-curve fitting** (a real transit shape) and for **uncertainty estimation** —
  neither exists. No MCMC / least-squares fit, no error bars.

### 6.4 Confidence is a proxy, not a calibration
- SNR and SDE are reported, but there is no mapping from these to a calibrated detection
  confidence / false-alarm probability, and no vetting evidence (odd/even, secondary eclipse) feeding it.

### 6.5 Scale vs. the PS
- 111 LC files vs the PS recommendation of a full high-cadence sector (~20–30k). Fine for a recovery
  benchmark, but the "apply to science datasets" deliverable wants breadth.

### 6.6 Hardcoded / fragile bits
- Systematic mask `[1347.4, 1349.4]` is Sector-1-specific and hardcoded; scaling to other sectors
  needs a per-sector approach.
- Fixed BLS upper period 13 d misses longer-period TCEs (a known `not_recovered` contributor).

### 6.7 Legacy/naming confusion (documented, not yet fixed)
- `outputs/...tce_recovery_batch_150.csv` and `tce_recovery_results_150.csv` actually contain 111
  rows. `.bak` tables and duplicate `50/111/150` artifacts make the source of truth ambiguous.

### 6.8 Tests
- `tests/` contains only `.gitkeep`. No unit tests on the recovery classifier or epoch/period math —
  the riskiest logic is untested.

---

## 7. Requirement → Code Traceability

| PS deliverable | Where (if anywhere) | Verdict |
|---|---|---|
| Identify periodic dips | `run_bls.py`, `06_run_tce_recovery.py` | ✅ |
| Classify transit/eclipse/blend/other | — | ❌ missing |
| Apply classifier to science data | — | ❌ missing |
| SNR / significance | `run_bls.py::estimate_snr/estimate_sde` | 🟡 proxy |
| Period | BLS `best_period` | ✅ (grid-limited) |
| Transit depth | BLS `best_depth` (box) | 🟡 box only |
| Transit duration | BLS `best_duration` (box) | 🟡 box only |
| Parameter *fitting* + uncertainties | `src/fitting/` (empty) | ❌ missing |
| Visualization w/ classified signal | `07_plot_tce_recovery.py`, `plot_bls_results.py` | 🟡 no class label |
| Confidence level | SNR/SDE only | 🟡 proxy |
| ≤3-page report | `report/` (empty) | ❌ missing |

---

See [02_recommendations.md](02_recommendations.md) for a prioritized, feasible plan to close these gaps.
