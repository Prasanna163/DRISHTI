"""Train + cross-validate an ML classifier on labeled candidates (WS-F2).

Builds a feature matrix from the per-candidate evidence vector (detection + transit fit + vetting),
joins the bootstrapped disposition labels (scripts/get_labels.py), and trains a RandomForest to
separate planets from eclipsing binaries. Because labels overlap only the targets we have evidence
for, the trainable set is small and imbalanced; we therefore report *cross-validated* metrics
(stratified K-fold out-of-fold predictions) with explicit caveats rather than a single train/test split.

Outputs:
  - cross-validated confusion matrix + per-class precision/recall/F1 + ROC-AUC + balanced accuracy
  - feature importances
  - a model fit on all labeled data, saved for reuse

Example
-------
    python scripts/11_train_classifier.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti_store import RESULT_TABLE_ROOT

DEFAULT_RECOVERY = RESULT_TABLE_ROOT / "tce_recovery_results_111.csv"
DEFAULT_FITS = RESULT_TABLE_ROOT / "transit_fits_111.csv"
DEFAULT_VETTING = RESULT_TABLE_ROOT / "vetting_features_111.csv"
DEFAULT_LABELS = ROOT / "data" / "raw" / "labels" / "labels.csv"
DEFAULT_MODEL = ROOT / "outputs" / "models" / "planet_eb_rf.joblib"
DEFAULT_PRED = RESULT_TABLE_ROOT / "ml_classification_cv.csv"

KEYS = ["tic_id", "sector"]
FEATURES = [
    "our_snr", "our_bls_period", "fit_depth_ppm", "fit_duration_hours",
    "fit_ingress_frac", "reduced_chi2",
    "oddeven_diff_sigma", "oddeven_depth_frac_diff",
    "secondary_snr", "secondary_to_primary_ratio",
    "v_shape_metric", "duration_sanity_ratio",
    "crowdsap", "centroid_shift_sigma",
]
POSITIVE = "planet"   # vs eclipsing_binary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/validate RandomForest planet-vs-EB classifier.")
    p.add_argument("--recovery", type=Path, nargs="+", default=[DEFAULT_RECOVERY],
                   help="One or more recovery-result CSVs (concatenated).")
    p.add_argument("--fits", type=Path, nargs="+", default=[DEFAULT_FITS],
                   help="One or more transit-fit CSVs (concatenated).")
    p.add_argument("--vetting", type=Path, nargs="+", default=[DEFAULT_VETTING],
                   help="One or more vetting-feature CSVs (concatenated).")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--pred-out", type=Path, default=DEFAULT_PRED)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score,
    )
    import joblib

    def load_concat(paths: list[Path]) -> pd.DataFrame:
        frames = [pd.read_csv(p) for p in paths if p.exists()]
        if not frames:
            return pd.DataFrame(columns=KEYS)
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates(subset=KEYS, keep="first")

    merged = load_concat(args.recovery)
    for paths in (args.fits, args.vetting):
        extra = load_concat(paths)
        if not extra.empty:
            dup = [c for c in extra.columns if c in merged.columns and c not in KEYS]
            merged = merged.merge(extra.drop(columns=dup), on=KEYS, how="left")
    labels = pd.read_csv(args.labels)[["tic_id", "class_label"]]
    merged = merged.merge(labels, on="tic_id", how="left")

    # Keep the two confident, populated classes.
    data = merged[merged["class_label"].isin(["planet", "eclipsing_binary"])].copy()
    feats = [f for f in FEATURES if f in data.columns]
    X = data[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = (data["class_label"] == POSITIVE).astype(int).to_numpy()

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"Trainable set: {len(y)}  (planet={n_pos}, eclipsing_binary={n_neg})  features={len(feats)}")
    if n_pos < 5 or n_neg < 5:
        print("Too few examples in one class for meaningful cross-validation. Aborting.")
        return 1

    clf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", min_samples_leaf=2,
        random_state=args.seed, n_jobs=-1,
    )
    folds = min(args.folds, n_pos)  # cannot have more folds than positives
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print(f"  Cross-validated planet-vs-EB classifier ({folds}-fold)")
    print("=" * 60)
    print(f"  Majority-class baseline accuracy: {max(n_pos, n_neg) / len(y):.2f}")
    print(f"  Balanced accuracy:                {balanced_accuracy_score(y, pred):.2f}")
    try:
        print(f"  ROC-AUC:                          {roc_auc_score(y, proba):.2f}")
    except ValueError:
        pass
    print("\n  Confusion matrix (rows=true, cols=pred) [EB, planet]:")
    print("   ", confusion_matrix(y, pred).tolist())
    print("\n" + classification_report(y, pred, target_names=["eclipsing_binary", "planet"], digits=2))

    # Feature importances from a model fit on all labeled data.
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    print("  Top feature importances:")
    for name, val in imp.head(8).items():
        print(f"    {name:28s} {val:.3f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": feats, "positive_class": POSITIVE}, args.model_out)

    data = data.assign(ml_planet_proba=np.round(proba, 3),
                       ml_pred=np.where(pred == 1, "planet", "eclipsing_binary"))
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    data[["tic_id", "sector", "class_label", "ml_pred", "ml_planet_proba"]].to_csv(args.pred_out, index=False)
    print(f"\n  Model: {args.model_out.resolve()}")
    print(f"  CV predictions: {args.pred_out.resolve()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
