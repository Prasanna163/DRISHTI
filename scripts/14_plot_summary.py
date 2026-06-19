"""Summary result plots for the DRISHTI pipeline (WS-H2 / report figures).

Generates the key evaluation figures into data/drishti/results/plots/summary/:
  1. recovery_classes.png        - recovery-class distribution (detection benchmark)
  2. depth_accuracy.png          - fitted vs official transit depth (parameter accuracy)
  3. ml_roc.png                  - ROC curve of the planet-vs-EB classifier (CV)
  4. ml_confusion.png            - confusion matrix at threshold 0.5 (CV)
  5. ml_reliability.png          - probability reliability + histogram (calibration)
  6. feature_importances.png     - what the classifier keys on

Example
-------
    python scripts/14_plot_summary.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti_store import RESULT_TABLE_ROOT, RESULT_PLOT_ROOT

T = RESULT_TABLE_ROOT
OUT = RESULT_PLOT_ROOT / "summary"
GREEN, RED, BLUE, GRAY = "#19764b", "#b33a3a", "#2f6fba", "#7a7a7a"


def _concat(names):
    frames = [pd.read_csv(T / n) for n in names if (T / n).exists()]
    return pd.concat(frames, ignore_index=True).drop_duplicates(["tic_id", "sector"], keep="first")


def plot_recovery_classes():
    rec = _concat(["tce_recovery_results_111.csv", "tce_recovery_results_labeled.csv"])
    counts = rec["recovery_class"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [GREEN if "direct" in c else BLUE if "alias" in c else RED if "not_" in c or "failed" in c
              else "#c77c1a" for c in counts.index]
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + 1, i, str(v), va="center", fontsize=9)
    ax.set_xlabel("number of targets")
    ax.set_title(f"Detection: BLS recovery class ({len(rec)} targets)")
    fig.tight_layout(); _save(fig, "recovery_classes.png")


def plot_depth_accuracy():
    fits = _concat(["transit_fits_111.csv", "transit_fits_labeled.csv"])
    d = fits.dropna(subset=["official_depth_ppm", "fit_depth_ppm"])
    d = d[(d.official_depth_ppm > 0) & (d.fit_depth_ppm > 0)]
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(d.official_depth_ppm, d.fit_depth_ppm, s=14, alpha=0.5, color=GREEN, edgecolor="none")
    lim = [d.official_depth_ppm.min() * 0.7, d.official_depth_ppm.max() * 1.3]
    ax.plot(lim, lim, "k--", lw=1, label="1:1 (perfect)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    med = (d.fit_depth_ppm / d.official_depth_ppm).median()
    ax.set_xlabel("official depth (ppm)"); ax.set_ylabel("fitted depth (ppm)")
    ax.set_title(f"Parameter accuracy: depth\n(median fit/official = {med:.2f})")
    ax.legend(); fig.tight_layout(); _save(fig, "depth_accuracy.png")


def _cv():
    p = T / "ml_classification_cv_full.csv"
    if not p.exists():
        p = T / "ml_classification_cv_expanded.csv"
    df = pd.read_csv(p)
    df["y"] = (df["class_label"] == "planet").astype(int)
    return df


def plot_roc():
    from sklearn.metrics import roc_curve, roc_auc_score
    df = _cv()
    fpr, tpr, _ = roc_curve(df.y, df.ml_planet_proba)
    auc = roc_auc_score(df.y, df.ml_planet_proba)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, color=GREEN, lw=2.2, label=f"RandomForest (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate (planet)")
    ax.set_title("Classifier ROC (cross-validated)"); ax.legend(loc="lower right")
    fig.tight_layout(); _save(fig, "ml_roc.png")


def plot_confusion():
    from sklearn.metrics import confusion_matrix
    df = _cv()
    pred = (df.ml_planet_proba >= 0.5).astype(int)
    cm = confusion_matrix(df.y, pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(cm, cmap="Greens")
    labels = ["eclipsing_binary", "planet"]
    ax.set_xticks([0, 1], labels=labels); ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=15, fontweight="bold")
    ax.set_title("Confusion matrix (CV, threshold 0.5)")
    fig.tight_layout(); _save(fig, "ml_confusion.png")


def plot_reliability():
    df = _cv()
    bins = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(df.ml_planet_proba, bins) - 1, 0, len(bins) - 2)
    centers, fracs, ns = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.any():
            centers.append((bins[b] + bins[b + 1]) / 2)
            fracs.append(df.y[m].mean()); ns.append(int(m.sum()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax1.plot(centers, fracs, "o-", color=GREEN, lw=2, label="model")
    ax1.set_xlabel("predicted planet probability"); ax1.set_ylabel("actual planet fraction")
    ax1.set_title("Reliability (calibration)"); ax1.legend(); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax2.hist(df.ml_planet_proba[df.y == 0], bins=20, alpha=0.6, color=RED, label="eclipsing_binary")
    ax2.hist(df.ml_planet_proba[df.y == 1], bins=20, alpha=0.6, color=GREEN, label="planet")
    ax2.axvline(0.5, color="k", ls="--", lw=1)
    ax2.set_xlabel("predicted planet probability"); ax2.set_ylabel("count")
    ax2.set_title("Score distribution by true class"); ax2.legend()
    fig.tight_layout(); _save(fig, "ml_reliability.png")


def plot_importances():
    import joblib
    p = ROOT / "outputs" / "models" / "planet_eb_rf.joblib"
    if not p.exists():
        return
    b = joblib.load(p)
    imp = pd.Series(b["model"].feature_importances_, index=b["features"]).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("importance"); ax.set_title("What the classifier keys on (feature importance)")
    fig.tight_layout(); _save(fig, "feature_importances.png")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=140); plt.close(fig)
    print("wrote", OUT / name)


def main():
    plot_recovery_classes()
    plot_depth_accuracy()
    plot_roc()
    plot_confusion()
    plot_reliability()
    plot_importances()
    print(f"\nAll summary plots in {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
