# DRISHTI — Master Execution Plan

**Date:** 2026-06-18
**Status:** Approved direction; ready to start working.
**Reads with:** [01_repo_analysis.md](01_repo_analysis.md) · [02_recommendations.md](02_recommendations.md) · [03_action_plan.md](03_action_plan.md)

This is the authoritative "how we move ahead" document. It consolidates every finding so far —
the repo audit, the data/labeling discussion, and the pixel-level (centroid/blend) analysis — into
concrete workstreams, sequenced sprints, tasks, file targets, and acceptance criteria.

---

## 1. North Star

> Given a **noisy TESS light curve it has never seen**, DRISHTI returns:
> (1) the **detected periodic dip**, (2) a **class** — transit / eclipsing binary / blend / other —
> with a **calibrated confidence**, (3) **fitted period, depth, duration with uncertainties**,
> (4) a **plot** showing all of it, and (5) a **≤3-page report**.

### Coverage matrix (every PS line must trace to a workstream)

| Problem-statement deliverable | Workstream | Evaluation criterion it scores |
|---|---|---|
| Identify periodic dips in noisy LCs | A (exists) | detection accuracy |
| Classify transit / eclipse / blend / other | C + D + F | classification accuracy |
| Apply classifier to science datasets | I | detection + classification |
| SNR / significance | A + G | detection accuracy |
| Period / depth / duration by *fitting* | B | parameter accuracy |
| How uncertainties are estimated | B + G | parameter accuracy / methods |
| Visualization w/ classified signal | H | visualization & clarity |
| Confidence level | G | methods / clarity |
| ≤3-page report | H | methods / clarity |

---

## 2. Guiding principles (non-negotiable)

1. **Build on the 111 LCs we already have.** M-fit, vetting, centroid, and the report all start with
   zero new downloads. Scale comes last.
2. **Labels are the bottleneck, not sectors.** S1+S2 is enough to *build*; classifier *accuracy* is
   unblocked by type-disposition labels (see WS-E), and by 3–5 sectors of *systematic diversity* only
   when we get there.
3. **Detection vs attribution are different.** BLS already detects dips well. The new work makes us
   accurate at *attributing and classifying* them (on-target transit vs EB vs blend).
4. **Every algorithm change maps to an observed failure mode.** No blind tuning of the headline
   recovery %. Recovery stays an internal validation harness, not the deliverable.
5. **Report grows from Sprint 0**, not at the end.

---

## 3. Current state (one line)

Detection + recovery proven (80% period recovery on 111 LCs). Missing: physical fit + uncertainties,
vetting evidence, classifier, calibrated confidence, labels, blend/pixel layer, report. The LC FITS
already contain centroid (`MOM_CENTR/PSF_CENTR`), pointing (`POS_CORR`), and crowding (`CROWDSAP`,
`FLFRCSAP`) data. TP files: 0 downloaded.

---

## 4. Workstreams

Each workstream is an independent track; §5 sequences them into sprints.

### WS-A — Detection hardening *(small, do early)*
- A1. Make BLS `max_period` adaptive (≈28% of S1+S2 TCEs have P > 13 d and are currently unreachable).
- A2. Move the hardcoded Sector-1 systematic mask `[1347.4, 1349.4]` into a **per-sector config**.
- A3. Add unit tests for the recovery classifier + epoch/period/alias math (`tests/` is empty today).
- A4. Resolve the `150`-named-but-111-row legacy files (archive/rename so source of truth is clear).

### WS-B — Physical transit fit + uncertainties  ⭐ highest single PS impact
- `src/fitting/transit_fit.py` + `scripts/08_fit_transits.py`.
- v1: trapezoid model via `scipy.optimize.curve_fit` (zero new deps), uncertainties from covariance.
- v2: `batman` (Mandel–Agol) limb-darkened shape; posterior errors via `emcee` credible intervals.
- Seed from existing BLS result. Emit `fit_period±σ, fit_t0±σ, fit_duration±σ, fit_depth±σ, redχ²/BIC`.

### WS-C — Vetting evidence (LC-only, deterministic, no labels)
- `src/vetting/{odd_even, secondary_eclipse, local_shape, duration_sanity, transit_snr, period_alias}.py`
- `scripts/09_run_vetting.py` → emits an **evidence vector** per candidate.
- Discriminators: odd/even depth mismatch & secondary eclipse → EB; V-shape → grazing/EB; duration
  sanity & SNR → quality/confidence.

### WS-D — Pixel / catalog blend layer  ⭐ targets the PS "crowded fields / blending"
- D1 (**data in hand**): `src/vetting/centroid_shift.py` — in-transit vs out-of-transit centroid from
  `MOM_CENTR/PSF_CENTR`, decorrelated by `POS_CORR`; output offset + significance. Top blend test.
- D2 (**header + catalog**): `src/vetting/contamination_score.py` — combine `CROWDSAP`/`FLFRCSAP`
  with a Gaia/TIC neighbor query (astroquery, no FITS) into a dilution/blend score.
- D3: nearby Gaia/TIC star query helper (feeds D2 + interprets D1).
- D4: mine **DV reports** (`dvr-xml`, downloader supports it) for official test verdicts + dispositions
  — doubles as a label source (WS-E) and a cross-check.
- D5 (**needs TP download**): `src/vetting/difference_image.py` — strongest localization, do for
  suspicious candidates only, after TP files are fetched.

### WS-E — Labels acquisition  *(unblocks WS-F ML)*
- E1. Use the **curated set the PS provides** as primary; land it in `data/raw/labels/` (empty today).
- E2. Bootstrap by TIC-ID join: **TOI / ExoFOP** dispositions (CP/KP=planet, FP=false positive),
  **TESS-EB catalog** (eclipsing_binary), **NASA Exoplanet Archive** (confirmed planets).
- E3. Confirm period/ephemeris match when joining (label the *same* signal, not just the same star).
- E4. For accuracy: gather labels across **3–5 sectors of different mission phases** (early South,
  later, North, extended) for morphology/systematic diversity.
- Note: **"blend" has no clean catalog** — derive it from WS-D evidence (centroid/diff-image/DV), not a
  label table.

### WS-F — Classification framework  ⭐ core deliverable
- F1 (no labels needed): `src/models/classifier.py` **rule-based** — combine WS-C + WS-D evidence into
  transparent rules (secondary/odd-even → EB; off-target centroid → blend; clean U-shape on-target →
  planet_candidate; low-SNR/inconsistent → other). Explainable → strong for the report.
- F2 (labels): train `RandomForest`/`GradientBoosting` (sklearn, currently unused) on the evidence
  vectors; report cross-validated confusion matrix + precision/recall.
- `scripts/10_classify_candidates.py` → emits `predicted_class` + `class_probability`.

### WS-G — Confidence calibration
- Replace the raw SNR/SDE proxy with calibrated confidence: classifier probability (F2) or periodogram
  false-alarm probability. **Integrate CDPP CSVs** (`data/Ref/*_rms-cdpp.csv`, currently unused) so
  confidence is expressed relative to each star's intrinsic noise (and explains noise-limited misses).

### WS-H — Visualization + report
- H1. Extend `07_plot_tce_recovery.py` to overlay **predicted class + confidence + fitted model** on the
  per-target diagnostic.
- H2. Write the **≤3-page report** in `report/` (empty today): methodology, assumptions, tools/libraries,
  uncertainty method. Start in Sprint 0, grow each sprint.

### WS-I — Science-scale blind run
- Point the existing manifest-streaming path at a sector of **unlabeled** LCs; run detect → fit → vet →
  classify → confidence end-to-end; emit a candidate catalog. Scale to 20–30k LCs here, not before.

---

## 5. Sprints (the actual order of work)

Dependencies in brackets. Sprints 1–3 use **only the 111 LCs already on disk**.

### Sprint 0 — Foundation (½–1 day)
- A4 (resolve legacy `150` files), A3-scaffold (`tests/` skeleton), H2 (open `report/REPORT.md` stub).
- Add a `requirements` bump if/when `batman`/`emcee` are adopted (defer until WS-B v2).
- **Acceptance:** unambiguous source-of-truth files; a report stub exists; CI-runnable test folder.

### Sprint 1 — Fit + first evidence  [needs nothing new]
- **B v1** (trapezoid fit + uncertainties) → `transit_fits_111.csv`.
- **C** core: `odd_even`, `secondary_eclipse`, `local_shape`, `transit_snr` → `vetting_features_111.csv`.
- **D1** centroid shift (data in hand) added to the evidence table.
- Grow report: methodology + fitting/uncertainty sections.
- **Acceptance:** every recovered target has fitted params±σ + an evidence vector incl. centroid offset.

### Sprint 2 — Blend layer + rule-based classifier  [+ catalog queries, DV-XML]
- **D2/D3** contamination score + Gaia/TIC neighbors; **D4** mine DV-XML for the 111 targets.
- **A1/A2** BLS max-period adaptive + per-sector mask config.
- **F1** rule-based classifier → `predicted_class` on all 111.
- **H1** plots overlay class + confidence + fitted model.
- **Acceptance:** pipeline is **PS-complete** — every target gets class + confidence + fitted params,
  visualized; report covers classification methodology.

### Sprint 3 — Labels + ML classifier + calibration  [needs WS-E labels]
- **E1/E2/E3** assemble labeled set (curated set or TOI+EB bootstrap).
- **F2** train + cross-validate ML classifier; report confusion matrix / precision-recall.
- **G** calibrated confidence + CDPP noise integration.
- **A3** finish unit tests.
- **Acceptance:** reported classification accuracy on a held-out labeled set; confidence is calibrated.

### Sprint 4 — Scale + crowded-field depth  [needs new downloads]
- **I** blind science-sector run end-to-end; candidate catalog.
- **D5** difference images on suspicious candidates (download TP first).
- **E4** add 3–5 diverse sectors; retrain/validate for generalization.
- **Acceptance:** pipeline runs on unseen LCs at sector scale; blend class supported by pixel evidence;
  final report finished.

---

## 6. Dependency graph

```text
WS-A ─┐
      ├──────────────► (independent, slot anywhere early)
WS-B ─┘
WS-C ─┬─► WS-F(F1 rules) ─► WS-H(H1 overlay) ─► WS-I(scale)
WS-D ─┘          ▲
WS-E ───────────►│ (F2 ML needs labels)
WS-F(F2) ─► WS-G(calibration)
WS-H(H2 report) grows every sprint
```

Critical path to **PS-complete**: B v1 → C → D1 → F1 → H1 (all on existing data, no label wait).
Critical path to **highly accurate**: + E → F2 → G → I → D5.

---

## 7. Data needs summary

| Need | When | Source | Have it? |
|---|---|---|---|
| 111 LC FITS | now | already downloaded | ✅ |
| Centroid/crowding columns | Sprint 1–2 | inside the LC FITS | ✅ |
| Gaia/TIC neighbor catalog | Sprint 2 | astroquery (Vizier/MAST/Gaia) | query |
| DV reports (`dvr-xml`) | Sprint 2 | MAST (downloader supports) | download |
| Type-disposition labels | Sprint 3 | curated set / TOI+ExoFOP + TESS-EB + NEA | acquire |
| TP files | Sprint 4 | MAST (downloader supports `tp`) | download |
| 3–5 diverse sectors | Sprint 4 | STScI bulk / manifest streaming | download |
| Blind science sector | Sprint 4 | manifest streaming (built) | download |

---

## 8. Definition of done

- **PS-complete (end of Sprint 2):** every PS line is *answerable* on the 111 LCs — detection, class,
  confidence, fitted params±σ, visualization, report draft.
- **Highly accurate (end of Sprint 4):** reported classification accuracy on held-out labels, calibrated
  confidence, pixel-evidence blend handling, and a clean run on an unseen science sector at scale; final
  ≤3-page report submitted.

---

## 9. First action

Start **Sprint 0 + the first item of Sprint 1**:
1. Resolve the legacy `150` files and open `report/REPORT.md`.
2. Implement **WS-B v1** (`src/fitting/transit_fit.py` + `scripts/08_fit_transits.py`) and
   **WS-D1** (`src/vetting/centroid_shift.py`) on the 111 LCs — both need zero new data.

Say the word and we begin with these.
