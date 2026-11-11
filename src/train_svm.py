"""
train_svm.py – Phase 2: Train an SVM classifier on hand-crafted features.

Pipeline:
  1. Load features from cache (or extract if missing).
  2. StandardScaler  ->  optional PCA (keep 95 % variance).
  3. GridSearchCV on a small subsample to find best (C, gamma).
  4. Refit the best SVM on the FULL training set.
  5. Evaluate on val and test; save metrics to models/svm_metrics.json.
  6. Save the full pipeline (scaler + optional PCA + SVM) to models/svm.joblib.

Usage:
    python -m src.train_svm                     # full run
    python -m src.train_svm --cap 400           # 400 images/class (fast test)
    python -m src.train_svm --no-pca            # skip PCA
    python -m src.train_svm --force-features    # re-extract features
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SVM_MODEL_PATH   = MODELS_DIR / "svm.joblib"
SVM_METRICS_PATH = MODELS_DIR / "svm_metrics.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameter search grid
# ---------------------------------------------------------------------------
# Searched on a stratified subsample (GRID_SUBSAMPLE images/class) for speed.
GRID_SUBSAMPLE = 150   # images per class used in grid search

PARAM_GRID = {
    "svm__estimator__C":     [0.1, 1.0, 10.0, 100.0],
    "svm__estimator__gamma": ["scale", "auto", 0.001, 0.0001],
}


# ---------------------------------------------------------------------------
# 1. Load / extract features
# ---------------------------------------------------------------------------

def _get_features(
    split: str,
    df,
    class_to_idx: Dict[str, int],
    cap_per_class: Optional[int],
    force_features: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    from src.features import build_feature_matrix
    return build_feature_matrix(
        df, split, class_to_idx,
        force=force_features,
        cap_per_class=cap_per_class,
    )


# ---------------------------------------------------------------------------
# 2. GridSearch subsample helper
# ---------------------------------------------------------------------------

def _subsample(X: np.ndarray, y: np.ndarray,
               n_per_class: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return at most n_per_class examples per class."""
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        chosen  = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# 3. Build sklearn pipeline
# ---------------------------------------------------------------------------

def build_pipeline(use_pca: bool = True, pca_variance: float = 0.95) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, whiten=True, random_state=42)))
    # CalibratedClassifierCV replaces the deprecated SVC(probability=True).
    # ensemble=False uses a single Platt-scaling pass (same behaviour, no warning).
    base_svc = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    steps.append(("svm", CalibratedClassifierCV(base_svc, ensemble=False)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------

def train(
    cap_per_class:  Optional[int] = None,
    use_pca:        bool = True,
    force_features: bool = False,
) -> None:
    from src.data import load_splits, get_label_map, SPLITS_CSV
    from src.evaluate import compute_metrics

    # -- Load data -----------------------------------------------------------
    df              = load_splits(SPLITS_CSV)
    class_to_idx, idx_to_class = get_label_map(df)
    n_classes       = len(class_to_idx)

    log.info("Loading / extracting train features …")
    X_train, y_train = _get_features("train", df, class_to_idx, cap_per_class, force_features)
    log.info("Loading / extracting val   features …")
    X_val,   y_val   = _get_features("val",   df, class_to_idx, cap_per_class, force_features)
    log.info("Loading / extracting test  features …")
    X_test,  y_test  = _get_features("test",  df, class_to_idx, cap_per_class, force_features)

    log.info(
        "Shapes – train: %s  val: %s  test: %s  classes: %d",
        X_train.shape, X_val.shape, X_test.shape, n_classes,
    )

    # -- GridSearchCV on a subsample ----------------------------------------
    X_sub, y_sub = _subsample(X_train, y_train, n_per_class=GRID_SUBSAMPLE)
    log.info(
        "Running GridSearchCV on %d-image subsample (param_grid size=%d) …",
        len(X_sub), len(PARAM_GRID["svm__estimator__C"]) * len(PARAM_GRID["svm__estimator__gamma"]),
    )

    base_pipe = build_pipeline(use_pca=use_pca)
    gs = GridSearchCV(
        base_pipe,
        PARAM_GRID,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=False,   # we'll refit on full data with the best params
    )
    t0 = time.perf_counter()
    gs.fit(X_sub, y_sub)
    grid_time = time.perf_counter() - t0

    best_params = gs.best_params_
    log.info("Best params: %s  (CV macro-F1=%.4f, search took %.1fs)",
             best_params, gs.best_score_, grid_time)

    # -- Refit on FULL training set -----------------------------------------
    final_pipe = build_pipeline(use_pca=use_pca)
    final_pipe.set_params(**best_params)

    log.info("Fitting final SVM on %d training examples …", len(X_train))
    t0 = time.perf_counter()
    final_pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    log.info("Training done in %.1f s", train_time)

    # -- Validation metrics -------------------------------------------------
    log.info("Evaluating on validation set …")
    val_metrics = compute_metrics(final_pipe, X_val, y_val, idx_to_class, split="val")

    # -- Test metrics -------------------------------------------------------
    log.info("Evaluating on test set …")
    t_inf = time.perf_counter()
    test_metrics = compute_metrics(final_pipe, X_test, y_test, idx_to_class, split="test")
    inference_time_total = time.perf_counter() - t_inf
    inference_time_per_image = inference_time_total / max(len(X_test), 1)

    # -- Save model ---------------------------------------------------------
    joblib.dump(final_pipe, SVM_MODEL_PATH)
    model_size_mb = SVM_MODEL_PATH.stat().st_size / (1024 ** 2)
    log.info("Model saved to %s  (%.1f MB)", SVM_MODEL_PATH, model_size_mb)

    # -- Save metrics -------------------------------------------------------
    n_components = None
    if use_pca and "pca" in final_pipe.named_steps:
        pca_step = final_pipe.named_steps["pca"]
        if hasattr(pca_step, "n_components_"):
            n_components = int(pca_step.n_components_)

    metrics_out = {
        "model": "SVM (RBF)",
        "feature_pipeline": {
            "hog":    True,
            "hsv":    True,
            "lbp":    True,
            "scaler": "StandardScaler",
            "pca":    use_pca,
            "pca_components": n_components,
        },
        "best_params":           best_params,
        "grid_search_time_s":    round(grid_time, 2),
        "train_time_s":          round(train_time, 2),
        "inference_time_per_image_ms": round(inference_time_per_image * 1000, 3),
        "model_size_mb":         round(model_size_mb, 2),
        "cap_per_class":         cap_per_class,
        "val":                   val_metrics,
        "test":                  test_metrics,
    }

    with open(SVM_METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    log.info("Metrics saved to %s", SVM_METRICS_PATH)

    # -- Console summary ----------------------------------------------------
    print("\n" + "=" * 55)
    print("SVM Results")
    print("=" * 55)
    print(f"  Val  accuracy : {val_metrics['accuracy']:.4f}")
    print(f"  Val  macro-F1 : {val_metrics['macro_f1']:.4f}")
    print(f"  Test accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test macro-F1 : {test_metrics['macro_f1']:.4f}")
    print(f"  Inference     : {inference_time_per_image*1000:.2f} ms/image")
    print(f"  Model size    : {model_size_mb:.1f} MB")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SVM on PlantVillage features")
    p.add_argument("--cap", type=int, default=None, metavar="N",
                   help="Cap images per class for speed (e.g. 400)")
    p.add_argument("--no-pca", dest="use_pca", action="store_false",
                   help="Disable PCA dimensionality reduction")
    p.add_argument("--force-features", action="store_true",
                   help="Re-extract features even if cache exists")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        cap_per_class=args.cap,
        use_pca=args.use_pca,
        force_features=args.force_features,
    )

