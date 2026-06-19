"""DRISHTI End-to-End Orchestrator

Usage:
    python scripts/run_end_to_end.py <ref_dir>
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRISHTI_SCRIPT = ROOT / "scripts" / "drishti.py"
SELECT_SCRIPT = ROOT / "scripts" / "select_tce_targets.py"
FIT_SCRIPT = ROOT / "scripts" / "08_fit_transits.py"
VET_SCRIPT = ROOT / "scripts" / "09_run_vetting.py"
CLASSIFY_SCRIPT = ROOT / "scripts" / "10_classify_candidates.py"
PLOT_SCRIPT = ROOT / "scripts" / "13_plot_classified.py"

def run_cmd(cmd, step_name):
    print(f"\n[{step_name}] Running: {' '.join(str(x) for x in cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        print(f"ERROR: {step_name} failed with code {res.returncode}")
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_dir", type=Path, nargs="?", default=ROOT / "data" / "Ref")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    ref_dir = args.ref_dir.resolve()
    print(f"Starting DRISHTI pipeline using reference folder: {ref_dir}")

    # 1. Target Selection
    target_csv = ROOT / "data" / "drishti" / "targets" / "tce_positive_targets.csv"
    cmd = [sys.executable, str(SELECT_SCRIPT), "--ref-dir", str(ref_dir)]
    run_cmd(cmd, "Target Selection")

    if not target_csv.exists():
        print(f"ERROR: Target selection did not produce {target_csv}")
        sys.exit(1)
        
    df = pd.read_csv(target_csv)
    n_targets = len(df)
    print(f"\nFound {n_targets} targets.")
    if n_targets == 0:
        print("No targets to process.")
        sys.exit(0)

    # 2. TCE Recovery (Download, Clean, BLS)
    cmd = [
        sys.executable, str(DRISHTI_SCRIPT), "tce-recovery", 
        "--batch-size", str(n_targets), "--skip-discover"
    ]
    if args.force:
        cmd.append("--force")
    run_cmd(cmd, "TCE Recovery (Download + BLS)")

    # Paths derived from the pipeline batch logic
    batch_csv = ROOT / "data" / "drishti" / "targets" / f"tce_recovery_batch_{n_targets}.csv"
    recovery_results = ROOT / "data" / "drishti" / "results" / "tables" / f"tce_recovery_results_{n_targets}.csv"
    
    if not batch_csv.exists() or not recovery_results.exists():
        print(f"ERROR: Recovery step did not produce expected files: {batch_csv.name} or {recovery_results.name}")
        sys.exit(1)

    # 3. Fit Transits
    fits_csv = ROOT / "data" / "drishti" / "results" / "tables" / f"transit_fits_{n_targets}.csv"
    cmd = [
        sys.executable, str(FIT_SCRIPT), 
        "--targets", str(batch_csv),
        "--output", str(fits_csv)
    ]
    if args.force:
        cmd.append("--force")
    run_cmd(cmd, "Physical Transit Fit")

    # 4. Vetting
    vetting_csv = ROOT / "data" / "drishti" / "results" / "tables" / f"vetting_features_{n_targets}.csv"
    cmd = [
        sys.executable, str(VET_SCRIPT), 
        "--targets", str(batch_csv),
        "--output", str(vetting_csv)
    ]
    if args.force:
        cmd.append("--force")
    run_cmd(cmd, "Crowded-field & Shape Vetting")

    # 5. Classification
    classifications_csv = ROOT / "data" / "drishti" / "results" / "tables" / f"candidate_classifications_{n_targets}.csv"
    cmd = [
        sys.executable, str(CLASSIFY_SCRIPT), 
        "--recovery", str(recovery_results),
        "--fits", str(fits_csv),
        "--vetting", str(vetting_csv),
        "--output", str(classifications_csv)
    ]
    run_cmd(cmd, "Rule-based Classification")

    # 6. Plotting
    plot_dir = ROOT / "data" / "drishti" / "results" / "plots" / f"classified_{n_targets}"
    cmd = [
        sys.executable, str(PLOT_SCRIPT),
        "--master", str(classifications_csv),
        "--recovery", str(recovery_results),
        "--fits", str(fits_csv),
        "--output-dir", str(plot_dir),
        "--all"
    ]
    run_cmd(cmd, "Diagnostic Plotting")

    elapsed = time.time() - t0
    m, s = divmod(elapsed, 60)
    print(f"\n✓ DRISHTI pipeline finished successfully in {int(m)}m {int(s)}s!")
    print(f"Results are available under: {plot_dir}")

if __name__ == '__main__':
    main()
