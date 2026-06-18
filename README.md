# DRISHTI

Deep Recognition and Intelligent Screening of Hidden Transit Indicators

DRISHTI is a TESS exoplanet detection and vetting workspace. It is built around a stricter question than "did a light curve dip?": can the pipeline recover official TESS signals and then gather enough evidence to decide where the signal likely came from.

The current system supports:

- Official TESS TCE target selection from `data/Ref/*_dvr-tcestats.csv`
- Controlled LC FITS download for selected TIC/sector rows
- Cleaning and Box Least Squares recovery against official period/epoch/duration
- Recovery classification beyond a binary pass/fail label
- Controlled diagnostic plot generation
- DRISHTI bulk-resource scraping from STScI TESS cURL manifest pages
- Resumable manifest-based download planning

## Project Direction

Ordinary light-curve projects ask whether a transit-like dip exists. DRISHTI is moving toward signal provenance:

```text
Was there a periodic transit-like signal?
Does it match an official TCE/TOI signal?
Is the event on the target star?
Could it be an eclipsing binary or nearby contaminant?
Should the candidate be kept, rejected, or escalated for review?
```

The immediate pipeline is light-curve recovery. The next major layer is evidence extraction: odd/even checks, secondary eclipse search, duration sanity, centroid shifts, difference imaging, and contamination scores.

## Repository Layout

```text
scripts/
  drishti.py                      Main DRISHTI CLI and STScI scraper/orchestrator
  select_tce_targets.py           Build TCE-positive and starter target lists
  05_download_tce_products.py     Download LC/TP/DV products by target list
  06_run_tce_recovery.py          Run cleaning + BLS and classify recovery
  07_plot_tce_recovery.py         Generate summary and target diagnostics
  01_*.py to 04_*.py              Earlier inspection/BLS/export/streaming utilities

src/
  data_access/                    FITS discovery and metadata parsing
  preprocessing/                  Light-curve cleaning and flattening
  detection/                      BLS search utilities
  features/                       Quantitative product exports
  visualization/                  Plot helpers

data/Ref/                         Local official TCE/CDPP reference CSVs
data/drishti/                     Generated DRISHTI resource metadata and raw store
outputs/                          Generated target lists, tables, plots, and diagnostics
```

Generated data and plots are intentionally ignored by git. Keep official/downloaded artifacts local unless making a deliberate data release.

## Setup

Use Python 3.11 or newer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

If `py -3.11` is not available, use the Python executable that owns your installed scientific stack.

## DRISHTI CLI

Show the splash:

```powershell
python .\scripts\drishti.py splash
```

Initialize the local storage layout:

```powershell
python .\scripts\drishti.py init-store
```

Scrape the official STScI TESS bulk pages into a local index:

```powershell
python .\scripts\drishti.py discover
```

This writes:

```text
data/drishti/metadata/bulk_resource_index.csv
```

The scraper reads the STScI Guest Investigator bulk-download page and the sector-wise FFI/TP/LC/DV bulk-download page.

## TCE Recovery Pipeline

Build target lists from official local TCE CSVs:

```powershell
python .\scripts\select_tce_targets.py
```

Important outputs:

```text
outputs/target_lists/tce_positive_targets.csv
outputs/target_lists/tce_starter_validation_targets.csv
outputs/target_lists/tce_first_recovery_batch.csv
```

Run a dry-run of the controlled 50-row workflow:

```powershell
python .\scripts\drishti.py tce-recovery --batch-size 50 --balanced --products lc --dry-run
```

Run the actual LC recovery workflow:

```powershell
python .\scripts\drishti.py tce-recovery --batch-size 50 --balanced --products lc
```

Run the same validation through the official STScI sector `.sh` manifests:

```powershell
python .\scripts\drishti.py tce-recovery `
  --batch-size 50 `
  --balanced `
  --products lc `
  --download-method manifest
```

That path streams sector manifests, downloads matching LC FITS files into `data/drishti/raw/lc/`, deletes each cached `.sh` after parsing, runs BLS recovery, and writes plots.

Primary outputs:

```text
outputs/target_lists/tce_recovery_batch_50.csv
outputs/tables/tce_download_status_50.csv
outputs/tables/tce_recovery_results_50.csv
outputs/plots/tce_recovery_50/
```

In `--download-method manifest` mode, the download status table is:

```text
outputs/drishti/tables/tce_manifest_download_status_50.csv
```

The recovery table includes:

```text
tic_id
sector
official_period
our_bls_period
period_error_percent
best_period_error_percent
period_match_type
official_epoch
our_t0
epoch_match_score
official_duration_hours
our_duration_hours
duration_ratio
official_snr
our_snr
recovery_class
recovered_true_false
period_recovered_true_false
status
source_file
message
```

Recovery classes:

```text
direct_recovered
alias_recovered
period_recovered_epoch_mismatch
period_recovered_bad_duration
period_recovered_needs_vetting
not_recovered
download_failed
processing_failed
```

## STScI Manifest Planning

There are two download paths:

```text
tce-recovery
```

Uses MAST product lookup for the selected TCE target rows. This is the easiest end-to-end validation command.

```text
stream-manifest / plan-manifest / download-plan
```

Uses the official STScI sector-wise `.sh` cURL manifests.

For the workflow where DRISHTI downloads one `.sh`, parses the matching LC URLs, downloads those products, deletes the cached `.sh`, and moves to the next script, use:

```powershell
python .\scripts\drishti.py stream-manifest `
  --resource-type light_curve `
  --sectors 1,2 `
  --target-list outputs\target_lists\tce_recovery_batch_50.csv `
  --products lc `
  --status outputs\drishti\tables\stream_lc_status.csv
```

Dry-run that same flow first:

```powershell
python .\scripts\drishti.py stream-manifest `
  --resource-type light_curve `
  --sectors 1,2 `
  --target-list outputs\target_lists\tce_recovery_batch_50.csv `
  --products lc `
  --limit 5 `
  --dry-run
```

By default, `stream-manifest` deletes each cached `.sh` after it has been parsed. Add `--keep-scripts` only when you want to inspect or debug the raw cURL manifest.

Build a normalized download plan from a sector-wise LC cURL script:

```powershell
python .\scripts\drishti.py plan-manifest `
  --resource-type light_curve `
  --sectors 1 `
  --target-list outputs\target_lists\tce_recovery_batch_50.csv `
  --output data\drishti\manifests\plans\sector1_lc_plan.csv
```

Dry-run the plan downloader:

```powershell
python .\scripts\drishti.py download-plan `
  --plan data\drishti\manifests\plans\sector1_lc_plan.csv `
  --dry-run
```

Download from the plan with resume/status logging:

```powershell
python .\scripts\drishti.py download-plan `
  --plan data\drishti\manifests\plans\sector1_lc_plan.csv `
  --status outputs\drishti\tables\sector1_lc_status.csv
```

The downloader uses `.part` files, skips existing files, records status CSVs, and validates FITS files after download.

## Current Validation Snapshot

The first controlled 50-row starter batch produced:

```text
Evaluated: 50
direct_recovered: 31
alias_recovered: 3
period_recovered_epoch_mismatch: 3
period_recovered_bad_duration: 3
not_recovered: 10

direct/alias recovered: 34 / 50 = 68.0%
period recovered or vetting-worthy: 40 / 50 = 80.0%
```

This means the base detector is alive. The next scientific work should focus on vetting evidence rather than simply adding a classifier.

## Next Stage: Evidence Layer v1

Recommended modules:

```text
src/vetting/odd_even.py
src/vetting/secondary_eclipse.py
src/vetting/transit_snr.py
src/vetting/period_alias.py
src/vetting/duration_sanity.py
src/vetting/local_shape.py
```

Future pixel-level modules:

```text
src/vetting/centroid_shift.py
src/vetting/difference_image.py
src/vetting/contamination_score.py
```

The long-term goal is a multimodal candidate evidence dataset combining light-curve, pixel, catalog-neighbor, and vetting features.

## Notes

- `outputs/`, downloaded FITS files, large TIC dumps, PDFs, and generated manifest caches are ignored by default.
- Keep dry-runs small before starting large STScI downloads.
- For LC-only recovery, `*_lc.fits` is sufficient. Pixel provenance work requires `*_tp.fits` and ideally DV XML/PDF products.
