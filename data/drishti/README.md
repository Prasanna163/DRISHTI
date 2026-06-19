# DRISHTI Store

This is the central DRISHTI workspace for generated artifacts.

```text
data/drishti/
  catalogs/        Scraped bulk indexes and metadata
  targets/         Selected TCE target lists and recovery batches
  manifests/
    scripts/       Cached STScI .sh files while streaming
    plans/         Normalized file-level download plans
  downloads/
    lc/            Light-curve FITS files
    tp/            Target-pixel FITS files
    dv/            DV products
  results/
    tables/        Download/recovery/status tables
    plots/         Summary plots and diagnostics
  logs/            Run logs
```

Source reference CSVs still live in `data/Ref/`.
