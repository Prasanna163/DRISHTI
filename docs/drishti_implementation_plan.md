# DRISHTI Implementation Plan

Date: 2026-06-18

This document records where the DRISHTI TESS recovery pipeline currently stands, what the present results mean, and what should be done next. It is meant to be readable as a project checkpoint without needing the chat history.

## 1. Project Purpose

DRISHTI is currently being built as a TESS TCE recovery and validation pipeline.

The immediate goal is not yet to claim new planet discoveries. The immediate goal is:

```text
Can DRISHTI independently recover official TESS TCE signals from the downloaded light curves?
```

The validation workflow is:

```text
official TESS TCE CSVs
    -> selected TIC/sector targets
    -> downloaded TESS light-curve FITS files
    -> cleaned and flattened light curves
    -> BLS transit search
    -> comparison against official TCE period/epoch/duration
    -> recovery table and diagnostic plots
```

## 2. Current Data Sources

The official reference files are stored in:

```text
data/Ref/
```

There are two kinds of reference files.

### TCE Statistics CSVs

Examples:

```text
data/Ref/tess2018206190142-s0001-s0001_dvr-tcestats.csv
data/Ref/tess2018235142541-s0002-s0002_dvr-tcestats.csv
```

These files come from the official MAST/STScI TESS TCE Bulk Downloads. Each row represents one official TESS Threshold Crossing Event, meaning a transit-like signal detected by the TESS Data Validation pipeline.

Important columns include:

```text
ticid                 TESS Input Catalog target ID
sectors               TESS sector where the signal appears
tce_period            official signal period in days
tce_time0bt           official transit epoch in BTJD
tce_duration          official transit duration
tce_depth             official transit depth
tce_model_snr         official model signal-to-noise ratio
tce_num_transits      number of observed transits
tce_full_conv         whether the official model fit converged
```

In this pipeline, the TCE statistics files are the official answer key.

### CDPP CSVs

Examples:

```text
data/Ref/tess2018206190146-s0001-s0001-00366_rms-cdpp.csv
data/Ref/tess2018235142537-s0002-s0002-00372_rms-cdpp.csv
```

CDPP means Combined Differential Photometric Precision. It is a noise or photometric precision measure for a star's light curve.

These files are useful for later analysis because they help explain whether a failed recovery happened because the algorithm missed the signal or because the target light curve was intrinsically noisy.

At the moment, the active recovery pipeline mainly uses the TCE statistics CSVs. The CDPP files are available but not yet fully integrated into the recovery result analysis.

## 3. Canonical Folder Layout

The current canonical pipeline output is under:

```text
data/drishti/
```

Important folders:

```text
data/drishti/catalogs/        scraped bulk resource indexes
data/drishti/targets/         selected target lists and recovery batches
data/drishti/downloads/lc/    downloaded light-curve FITS files
data/drishti/downloads/tp/    downloaded target-pixel FITS files
data/drishti/downloads/dv/    downloaded data validation products
data/drishti/results/tables/  download and recovery tables
data/drishti/results/plots/   summary and target diagnostic plots
```

The older `outputs/target_lists/` folder still exists, but it should now be treated as legacy output. Active scripts should use `data/drishti/targets/`.

## 4. What Has Been Completed

### Reference Data

Official Sector 1 and Sector 2 TCE reference files are present in `data/Ref/`.

### Target Selection

The script:

```text
scripts/select_tce_targets.py
```

reads the official TCE CSV files and creates filtered target lists.

Current canonical target counts after the Phase 1/2 approved run:

```text
data/drishti/targets/tce_positive_targets.csv             1538 rows
data/drishti/targets/tce_starter_validation_targets.csv    111 rows
data/drishti/targets/tce_recovery_batch_50.csv              50 rows
data/drishti/targets/tce_recovery_batch_111.csv            111 rows
data/drishti/targets/tce_recovery_batch_143.csv            111 rows
```

The starter validation set is stricter than the full positive target set. It is intended to be a cleaner first benchmark.

Important correction: an earlier checkpoint treated the starter set as 143 rows. Re-running the current default selector on 2026-06-18 produced 111 starter rows. Therefore, under the current canonical defaults, 111 is the complete starter validation set.

### Downloads

The script:

```text
scripts/05_download_tce_products.py
```

downloads selected TESS products for target TIC/sector rows.

Current downloaded light-curve FITS count:

```text
111 files
```

Location:

```text
data/drishti/downloads/lc/
```

By default, the downloader only downloads `lc` products, meaning light-curve FITS files. It can also request `tp`, `dvr-pdf`, and `dvr-xml`.

### Recovery

The script:

```text
scripts/06_run_tce_recovery.py
```

has evaluated the complete current starter set of 111 target rows.

Current result table:

```text
data/drishti/results/tables/tce_recovery_results_143.csv
```

Current recovery counts:

```text
direct_recovered                    76
alias_recovered                      4
period_recovered_bad_duration        5
period_recovered_epoch_mismatch      4
not_recovered                       22
```

### Plots

The script:

```text
scripts/07_plot_tce_recovery.py
```

has generated summary and diagnostic plots for the current benchmark runs.

Important folders:

```text
data/drishti/results/plots/tce_recovery_50/
data/drishti/results/plots/tce_recovery_111/
```

## 5. What The Recovery Labels Mean

Each row in the recovery table compares the official TESS TCE signal against DRISHTI's BLS result.

The official TCE row gives:

```text
official_period
official_epoch
official_duration_hours
official_snr
```

DRISHTI computes:

```text
our_bls_period
our_t0
our_duration_hours
our_snr
```

The recovery class tells how well those match.

### direct_recovered

This is the strongest success category.

It means:

```text
our BLS period matches the official TCE period
our transit timing/epoch matches the official epoch
our SNR is strong enough
our duration is not suspiciously different
```

In plain language:

```text
DRISHTI independently found the same signal TESS reported.
```

Current count:

```text
76 / 111
```

### alias_recovered

This means DRISHTI found the same event family but at a simple period alias.

Common aliases:

```text
half-period alias
double-period alias
```

Example:

```text
official period = 4.0 days
our BLS period  = 2.0 days
```

This is still a useful recovery, but not as clean as direct recovery.

Current count:

```text
4 / 111
```

### period_recovered_bad_duration

This means the period matched, but the transit duration did not look reasonable compared with the official duration.

Example:

```text
official period   = 5.0 days
our period        = 5.01 days
official duration = 2 hours
our duration      = 10 hours
```

In plain language:

```text
DRISHTI found the rhythm of the signal, but the fitted dip shape is suspicious.
```

Current count:

```text
5 / 111
```

### period_recovered_epoch_mismatch

This means the period matched, but the transit phase or timing did not line up with the official epoch.

Example:

```text
official transits: 1325, 1328, 1331
our transits:      1326, 1329, 1332
```

The period is the same, but the dips occur at the wrong phase.

In plain language:

```text
DRISHTI found a signal with the same spacing, but not the same transit timing.
```

Current count:

```text
4 / 111
```

### not_recovered

This means the current cleaning plus BLS search did not recover the official TESS signal under the present thresholds.

This can happen because:

```text
the signal is weak
the light curve is noisy
the preprocessing removed or distorted the transit
BLS preferred a different signal
the period range or duration grid is not ideal
the official TCE itself is difficult or systematic
```

Current count:

```text
22 / 111
```

### Current Summary

Grouped interpretation:

```text
clear direct success:             76
useful alias success:              4
partial period recovery:           9
not recovered:                    22
```

So:

```text
direct + alias recovered = 80 / 111
period recovered in some form = 89 / 111
not recovered = 22 / 111
```

## 6. Important Current Caveat

There is a legacy file:

```text
outputs/target_lists/tce_recovery_batch_150.csv
```

Despite the name, this file currently contains 111 rows, not 150 rows.

There is also a similarly confusing result table:

```text
data/drishti/results/tables/tce_recovery_results_150.csv
```

It also contains 111 rows. This should be cleaned up or clearly marked as legacy to avoid future confusion.

After the approved Phase 2 run, there is also a canonical file named:

```text
data/drishti/targets/tce_recovery_batch_143.csv
data/drishti/results/tables/tce_recovery_results_143.csv
```

These names were produced because the requested batch size was 143. However, the current starter set only contains 111 rows, so both files contain 111 rows. The contents are valid, but the filename is potentially confusing. Future runs should prefer `--batch-size 111` for the complete current starter validation set unless the selector criteria are changed.

## 7. Recommended Implementation Phases

### Phase 1: Freeze The Canonical Workflow

Purpose:

```text
Remove confusion between old outputs/ paths and current data/drishti/ paths.
```

Tasks:

```text
1. Confirm all active scripts read from and write to data/drishti/.
2. Document outputs/ as legacy.
3. Rename, archive, or ignore misleading 150-labeled legacy files.
4. Keep README and this implementation plan aligned with the real workflow.
```

Success condition:

```text
A reader can tell which files are current and which are old.
No active command depends on outputs/target_lists/.
```

### Phase 2: Complete The Full Starter Validation Set

Purpose:

```text
Evaluate the complete starter validation set before scaling further.
```

Approved run status:

```text
requested batch size: 143
actual starter validation rows from current selector: 111
evaluated rows: 111
remaining starter rows: 0
```

Command that was run after approval:

```powershell
python .\scripts\drishti.py tce-recovery --batch-size 143 --products lc --skip-discover
```

Observed behavior:

```text
the selector rebuilt the canonical target lists
the current starter list contained 111 rows
the 143-named batch file contained all 111 starter rows
all 111 LC products already existed locally
all 111 rows were evaluated
new 143-named result and plot folders were produced
```

Success condition:

```text
all current starter rows are classified
download and processing failures are absent or understood
summary plots exist
```

Phase 2 result:

```text
download status:
  exists                            111

recovery status:
  direct_recovered                   76
  alias_recovered                     4
  period_recovered_bad_duration       5
  period_recovered_epoch_mismatch     4
  not_recovered                      22

download_failed                       0
processing_failed                     0
```

### Phase 3: Triage The Non-Recovered And Partial-Recovered Cases

Purpose:

```text
Understand why the pipeline misses or partially recovers some official TCEs.
```

Rows to inspect:

```text
not_recovered
period_recovered_bad_duration
period_recovered_epoch_mismatch
alias_recovered
processing_failed
download_failed
```

Questions:

```text
Was the official TCE weak?
Was the light curve noisy?
Did flattening remove the signal?
Did the BLS grid miss the correct period or duration?
Did BLS lock onto a half/double period alias?
Was the epoch wrong even when the period was right?
Was the duration physically suspicious?
```

Recommended deliverable:

```text
data/drishti/results/tables/failure_triage_starter.csv
```

Suggested columns:

```text
tic_id
sector
official_period
our_bls_period
period_error_percent
epoch_match_score
official_duration_hours
our_duration_hours
duration_ratio
official_snr
our_snr
recovery_class
likely_failure_reason
recommended_action
```

### Phase 4: Integrate CDPP Noise Context

Purpose:

```text
Separate algorithm failures from intrinsically difficult/noisy light curves.
```

Tasks:

```text
1. Parse the Sector 1 and Sector 2 CDPP files.
2. Join CDPP/noise values into the recovery table by TIC/sector.
3. Group recovery rates by noise/CDPP bins.
4. Check whether non-recovered targets are mostly high-noise cases.
```

Success condition:

```text
Recovery results include noise context.
Failures can be interpreted more scientifically.
```

### Phase 5: Improve Recovery Logic Only After Failure Analysis

Possible improvements:

```text
1. Test alternate flattening windows.
2. Adjust known systematic masks.
3. Expand or adapt the BLS period range for longer-period TCEs.
4. Add an official-period-centered diagnostic BLS search.
5. Tune duration sanity thresholds.
6. Add odd/even transit checks.
7. Add secondary eclipse checks.
```

Important rule:

```text
Do not tune blindly to improve the headline recovery rate.
Every algorithm change should map to an observed failure mode.
```

### Phase 6: Scale The Benchmark

Recommended scaling path:

```text
111 targets   complete current starter validation
500 targets   medium benchmark
1000 targets  stronger statistics
1538 targets  full current default Sector 1-2 selected benchmark
```

Recommended next benchmark after the starter set:

```text
500 balanced targets
```

This should only be run after the 111-row starter validation is understood.

## 8. Phase 1 And Phase 2 Completion Update

Approved work completed on 2026-06-18:

```text
Phase 1: canonical path documentation updated
Phase 2: current starter validation rerun
```

What changed:

```text
README.md now explicitly says data/drishti/ is canonical.
README.md now marks outputs/target_lists, outputs/tables, and outputs/plots as legacy.
Manifest examples in README now use data/drishti paths instead of outputs paths.
The implementation plan was corrected to reflect the current 111-row starter set.
```

What the run found:

```text
The current default selector produces 1538 positive rows and 111 starter rows.
The earlier 143-row assumption was stale.
The requested --batch-size 143 does not create 143 rows because only 111 starter rows exist.
All 111 requested LC files already existed locally.
The complete current starter set evaluated without download or processing failures.
```

Artifacts produced:

```text
data/drishti/targets/tce_recovery_batch_143.csv
data/drishti/results/tables/tce_download_status_143.csv
data/drishti/results/tables/tce_recovery_results_143.csv
data/drishti/results/plots/tce_recovery_143/
```

The `143` filename suffix should be interpreted as the requested batch size, not the row count. The file contains 111 rows.

## 9. Immediate Recommended Next Step

The next implementation step should be:

```text
Phase 3 only
```

That means:

```text
1. Build a failure triage table for the 22 not_recovered rows.
2. Include the 9 partial period-recovery rows.
3. Assign likely failure reasons before changing the algorithm.
```

After that, the next decision should be based on the starter failure analysis.

No next-step implementation should be started until approved.
