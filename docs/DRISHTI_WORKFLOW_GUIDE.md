# DRISHTI Workflow Guide

This guide is the practical map for using the current DRISHTI workflow. It follows the actual files in this checkout and explains what each stage proves.

## Big Picture

DRISHTI is not just a transit dip finder. The validation question is:

```text
Can we recover known official TESS TCE signals, measure them, vet them, and classify the source?
```

The workflow is a ladder:

```text
target list
  -> TESS light-curve FITS
  -> clean + BLS search
  -> official TCE recovery check
  -> transit fit with uncertainties
  -> vetting evidence
  -> rule and ML classification
  -> master catalog + plots
```

The canonical generated workspace is:

```text
data/drishti/
  targets/
  downloads/lc/
  results/tables/
  results/plots/
```

Treat similarly named files under `outputs/` as legacy or model-only artifacts unless a script explicitly writes there.

## Fastest Way To Reproduce The Current Result

Activate the environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

The current full labeled evidence set can be processed with the parallel single-pass runner:

```powershell
python .\scripts\process_targets_parallel.py `
  --targets .\data\drishti\targets\labeled_training_targets.csv `
  --suffix labeled `
  --workers 14
```

Then train the full planet-vs-EB classifier:

```powershell
python .\scripts\11_train_classifier.py `
  --recovery .\data\drishti\results\tables\tce_recovery_results_111.csv .\data\drishti\results\tables\tce_recovery_results_labeled.csv `
  --fits .\data\drishti\results\tables\transit_fits_111.csv .\data\drishti\results\tables\transit_fits_labeled.csv `
  --vetting .\data\drishti\results\tables\vetting_features_111.csv .\data\drishti\results\tables\vetting_features_labeled.csv `
  --pred-out .\data\drishti\results\tables\ml_classification_cv_full.csv
```

Build the master catalog:

```powershell
python .\scripts\12_finalize_candidates.py
```

Make classified per-target plots:

```powershell
python .\scripts\13_plot_classified.py --per-class 5
```

Make the report-level summary plots:

```powershell
python .\scripts\14_plot_summary.py
```

Current summary plots are written to:

```text
data/drishti/results/plots/summary/
  recovery_classes.png
  depth_accuracy.png
  ml_roc.png
  ml_confusion.png
  ml_reliability.png
  feature_importances.png
```

## Beginner Path: Official TCE Recovery

Start here when you want to validate the detector on known official signals.

Build target lists from the official local TCE CSVs:

```powershell
python .\scripts\select_tce_targets.py
```

This writes:

```text
data/drishti/targets/tce_starter_validation_targets.csv
data/drishti/targets/tce_positive_targets.csv
```

Run the starter validation using existing local light curves:

```powershell
python .\scripts\drishti.py tce-recovery `
  --batch-size 111 `
  --balanced `
  --products lc `
  --skip-download
```

If light curves are missing and network access is available, use the manifest downloader:

```powershell
python .\scripts\drishti.py tce-recovery `
  --batch-size 111 `
  --balanced `
  --products lc `
  --download-method manifest
```

The key output table is:

```text
data/drishti/results/tables/tce_recovery_results_111.csv
```

Important columns:

```text
official_period
our_bls_period
best_period_error_percent
period_match_type
epoch_match_score
duration_ratio
our_snr
recovery_class
```

## What Each Stage Means

### 1. Cleaning

The light curve is normalized and flattened so BLS can find periodic box-like dips. For depth fitting, DRISHTI uses a second transit-masked flattening pass so the detrending does not absorb the transit depth.

### 2. BLS Search

BLS proposes the strongest periodic transit-like signal. In this project, the BLS result is compared against the official TCE period, epoch, duration, and SNR.

### 3. Recovery Class

`direct_recovered` means the BLS signal matches the official period, phase, duration, and SNR well enough.

`alias_recovered` means the signal is likely a harmonic such as half-period or double-period. This often happens with eclipsing binaries.

`not_recovered` means the strongest detected signal did not convincingly match the official TCE.

Intermediate classes such as `period_recovered_bad_duration` and `period_recovered_epoch_mismatch` are useful because they show partial success instead of hiding everything behind pass/fail.

### 4. Transit Fit

`scripts/08_fit_transits.py` and the parallel processor fit a trapezoid transit model. The output includes:

```text
fit_depth_ppm
fit_depth_err_ppm
fit_duration_hours
fit_duration_err_hours
fit_ingress_frac
reduced_chi2
```

The summary depth plot checks whether fitted depths agree with official depths. The current full plot has median fit/official depth around 0.92, which is a healthy sanity check.

### 5. Vetting Evidence

The vetting layer asks whether the signal looks planetary or EB-like:

```text
centroid_shift_sigma
crowdsap
oddeven_diff_sigma
secondary_snr
v_shape_metric
duration_sanity_ratio
```

These are deliberately physical and inspectable. They are also the features used by the ML classifier.

### 6. Classification

There are two classification layers.

The rule-based class is transparent:

```text
rule_class
rule_confidence
rule_reason
```

The ML layer gives:

```text
ml_planet_proba
```

Read `ml_planet_proba` as a ranking and confidence signal, not as a hard truth label. In the current full 525-example cross-validation:

```text
ROC-AUC: 0.94
balanced accuracy at threshold 0.5: 0.80
planet precision/recall: 0.61 / 0.65
EB precision/recall: 0.95 / 0.94
```

The important lesson is that the model ranks well, but the default 0.5 threshold is not necessarily the best operating point for a rare planet class. High probability candidates are cleaner shortlist material.

## How To Read The Plots

`recovery_classes.png`

Shows how many known TCEs were directly recovered, alias recovered, not recovered, or partially recovered. This is the detector validation plot.

`depth_accuracy.png`

Plots fitted transit depth versus official TCE depth. Points near the dashed 1:1 line mean the fit is physically sensible. Large outliers are review targets.

`ml_roc.png`

Shows whether the ML classifier ranks planets above EBs. Current AUC is 0.94, so ranking quality is strong.

`ml_confusion.png`

Shows the hard classification result at threshold 0.5. Current matrix is:

```text
true EB      -> 431 EB, 28 planet
true planet  -> 23 EB, 43 planet
```

This is the honest realistic-prevalence result, not the earlier optimistic balanced-sample picture.

`ml_reliability.png`

The left panel checks calibration. The right panel shows score separation. Use this plot when choosing a planet probability threshold.

`feature_importances.png`

Shows what the RandomForest used most. Current top features are fitted depth, centroid shift, secondary SNR, ingress fraction, duration sanity, V-shape, BLS period, and BLS SNR.

`results/plots/classified/<class>/`

These are the per-candidate diagnostic plots. Use them for human review: cleaned light curve on the left, phase-folded signal plus fitted model on the right.

## The Files You Usually Open

For machine-readable results:

```text
data/drishti/results/tables/master_candidates.csv
data/drishti/results/tables/ml_classification_cv_full.csv
data/drishti/results/tables/tce_recovery_results_labeled.csv
data/drishti/results/tables/transit_fits_labeled.csv
data/drishti/results/tables/vetting_features_labeled.csv
```

For visual review:

```text
data/drishti/results/plots/summary/
data/drishti/results/plots/classified/
data/drishti/results/plots/tce_recovery_111/
```

For the written project state:

```text
report/REPORT.md
claude_myref/reference/05_progress_and_next_steps.md
```

## Mental Model For Using DRISHTI

Use recovery classes to ask: did we find the official signal?

Use fit parameters to ask: did we measure the signal correctly?

Use vetting evidence to ask: does the signal look like a planet, EB, or blend?

Use ML probability to ask: where should human attention go first?

Use classified plots to ask: does the table verdict survive looking at the actual light curve?
