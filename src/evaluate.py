"""
evaluate.py – Shared evaluation utilities (Phase 2 & 3).

Provides:
  compute_metrics(model, X_or_loader, y, idx_to_class, split)
      -> dict of accuracy, macro/weighted F1, precision, recall,
         confusion matrix, per-class report.

  plot_confusion_matrix(cm, labels, title, out_path)
  plot_training_curves(history, model_name, out_path)

Both SVM (sklearn pipeline) and DL models (PyTorch nn.Module + DataLoader)
are accepted.  The function dispatches on input type automatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe in scripts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARTS_DIR   = PROJECT_ROOT / "data" / "charts"
MODELS_DIR   = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# 1. Core metric computation
# ---------------------------------------------------------------------------

def _predict_sklearn(model: Any, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run sklearn pipeline / estimator and return (predictions, confidences)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        confs = np.max(proba, axis=1)
        return preds, confs
    preds = model.predict(X)
    return preds, None


def _predict_torch(model: Any, loader: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a PyTorch model over a DataLoader.
    Returns (y_true, y_pred, y_conf) as numpy arrays.
    """
    import torch
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []
    all_confs = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_confs)


def compute_metrics(
    model: Any,
    X_or_loader: Any,
    y: Optional[np.ndarray],
    idx_to_class: Dict[int, str],
    split: str = "test",
    paths: Optional[List[str]] = None,
    model_name: Optional[str] = None,
) -> Dict:
    """
    Compute classification metrics for either an sklearn model or a PyTorch model.

    Parameters
    ----------
    model         : sklearn Pipeline / estimator  OR  torch.nn.Module
    X_or_loader   : numpy feature matrix (sklearn) OR DataLoader (torch)
    y             : ground-truth labels (sklearn); pass None for torch
                    (labels are taken from the DataLoader)
    idx_to_class  : integer -> class-name mapping
    split         : string label added to the returned dict
    paths         : list of image paths corresponding to samples (optional)
    model_name    : name of the model (optional, used for saving charts)

    Returns
    -------
    dict with keys: split, accuracy, macro_f1, weighted_f1, macro_precision,
                    macro_recall, confusion_matrix, per_class_report,
                    sample_predictions (optional)
    """
    # --- Dispatch ----------------------------------------------------------
    try:
        import torch
        is_torch = isinstance(model, torch.nn.Module)
    except ImportError:
        is_torch = False

    sample_paths = paths
    if is_torch:
        y_true, y_pred, y_conf = _predict_torch(model, X_or_loader)
        if sample_paths is None and hasattr(X_or_loader, "dataset") and hasattr(X_or_loader.dataset, "paths"):
            sample_paths = X_or_loader.dataset.paths
    else:
        y_true = y
        y_pred, y_conf = _predict_sklearn(model, X_or_loader)

    # --- Metrics -----------------------------------------------------------
    labels_present = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    label_names    = [idx_to_class.get(i, str(i)) for i in labels_present]

    acc       = float(accuracy_score(y_true, y_pred))
    macro_f1  = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    wt_f1     = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_p   = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    macro_r   = float(recall_score(y_true, y_pred,    average="macro",    zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels_present).tolist()

    per_class = classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    log.info(
        "[%s] acc=%.4f  macro-F1=%.4f  weighted-F1=%.4f",
        split, acc, macro_f1, wt_f1,
    )

    res: Dict[str, Any] = {
        "split":            split,
        "accuracy":         acc,
        "macro_f1":         macro_f1,
        "weighted_f1":      wt_f1,
        "macro_precision":  macro_p,
        "macro_recall":     macro_r,
        "confusion_matrix": cm,
        "class_names":      label_names,
        "per_class_report": per_class,
    }

    # Extract qualitative sample predictions if paths are available
    if sample_paths is not None and len(sample_paths) == len(y_true):
        correct, incorrect = collect_sample_predictions(
            y_true, y_pred, y_conf, sample_paths, idx_to_class, num_samples=4
        )
        res["sample_predictions"] = {
            "correct": correct,
            "incorrect": incorrect,
        }
        if model_name:
            plot_sample_predictions(model_name, correct, incorrect)

    return res


# ---------------------------------------------------------------------------
# 2. Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    labels: List[str],
    title: str = "Confusion Matrix",
    out_path: Optional[Path] = None,
    figsize_per_class: float = 0.45,
    normalize: bool = True,
) -> Path:
    """
    Plot and optionally save a (optionally normalised) confusion matrix heatmap.

    Returns the saved file path (or the default path if out_path is None).
    """
    cm = np.array(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1          # avoid division by zero
        cm_plot = cm / row_sums
        fmt     = ".2f"
        cb_label = "Recall (row-normalised)"
    else:
        cm_plot  = cm
        fmt      = "d"
        cb_label = "Count"

    n = len(labels)
    fig_size = max(10, n * figsize_per_class)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_plot,
        annot=(n <= 50),    # skip per-cell annotations for large matrices
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.3 if n <= 50 else 0,
        cbar_kws={"label": cb_label},
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=max(4, 8 - n // 10))
    plt.yticks(rotation=0,  fontsize=max(4, 8 - n // 10))
    plt.tight_layout()

    if out_path is None:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_title = title.lower().replace(" ", "_").replace("/", "_")
        out_path = CHARTS_DIR / f"{safe_title}.png"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3. Training-curve plot (Phase 3)
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Plot train/val loss and accuracy curves.

    Parameters
    ----------
    history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    model_name : used in the title and default filename
    out_path   : where to save (PNG)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   marker="o", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} – Loss"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "train_acc" in history:
        axes[1].plot(epochs, history["train_acc"], label="Train", marker="o", markersize=3)
        axes[1].plot(epochs, history["val_acc"],   label="Val",   marker="o", markersize=3)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{model_name} – Accuracy"); axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path is None:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_")
        out_path  = CHARTS_DIR / f"{safe_name}_training_curves.png"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Training curves saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 4. Model comparison bar chart (Phase 3)
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_dict: Dict[str, Dict],
    out_path: Optional[Path] = None,
) -> Path:
    """
    Bar chart comparing accuracy and macro-F1 across multiple models.

    Parameters
    ----------
    metrics_dict : {model_name: metrics_dict_from_compute_metrics, ...}
    out_path     : save path
    """
    names    = list(metrics_dict.keys())
    accs     = [metrics_dict[n]["accuracy"]  for n in names]
    f1s      = [metrics_dict[n]["macro_f1"]  for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2.2), 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy",  color="#4c72b0")
    bars2 = ax.bar(x + width / 2, f1s,  width, label="Macro-F1",  color="#dd8452")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison – Test Set", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if out_path is None:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CHARTS_DIR / "model_comparison.png"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Model comparison chart saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 5. Qualitative predictions plot (Phase 3)
# ---------------------------------------------------------------------------

def plot_sample_predictions(
    model_name: str,
    samples_correct: List[Dict[str, Any]],
    samples_incorrect: List[Dict[str, Any]],
    out_path: Optional[Path] = None,
) -> Path:
    """
    Plot sample correct and misclassified predictions side-by-side.

    Each item in samples_correct/incorrect is a dict:
        {"path": str, "true_class": str, "pred_class": str, "confidence": float}
    """
    from PIL import Image

    n_corr = len(samples_correct)
    n_inc = len(samples_incorrect)
    n_cols = max(1, max(n_corr, n_inc))
    n_rows = 2 if n_inc > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.5), squeeze=False)

    # Row 0: Correct predictions
    for j in range(n_cols):
        ax = axes[0, j]
        if j < n_corr:
            item = samples_correct[j]
            try:
                img = Image.open(item["path"]).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Image Error", ha="center", va="center")
            true_short = item["true_class"].replace("___", "\n").replace("_", " ")
            conf = item.get("confidence", 1.0)
            ax.set_title(f"CORRECT ({conf:.1%})\n{true_short}", color="green", fontsize=8, fontweight="bold")
        ax.axis("off")

    # Row 1: Incorrect predictions (if any)
    if n_rows > 1:
        for j in range(n_cols):
            ax = axes[1, j]
            if j < n_inc:
                item = samples_incorrect[j]
                try:
                    img = Image.open(item["path"]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Image Error", ha="center", va="center")
                true_short = item["true_class"].replace("___", "\n").replace("_", " ")
                pred_short = item["pred_class"].replace("___", "\n").replace("_", " ")
                conf = item.get("confidence", 0.0)
                ax.set_title(f"MISCLASSIFIED ({conf:.1%})\nTrue: {true_short}\nPred: {pred_short}",
                             color="darkred", fontsize=8, fontweight="bold")
            ax.axis("off")

    fig.suptitle(f"{model_name} – Sample Test Predictions", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    if out_path is None:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_")
        out_path = CHARTS_DIR / f"{safe_name}_sample_predictions.png"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Sample predictions chart saved to %s", out_path)
    return out_path


def collect_sample_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: Optional[np.ndarray],
    paths: List[str],
    idx_to_class: Dict[int, str],
    num_samples: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract up to num_samples correct and misclassified predictions."""
    correct_indices = np.where(y_true == y_pred)[0]
    incorrect_indices = np.where(y_true != y_pred)[0]

    np.random.seed(42)
    if len(correct_indices) > num_samples:
        correct_indices = np.random.choice(correct_indices, size=num_samples, replace=False)
    if len(incorrect_indices) > num_samples:
        incorrect_indices = np.random.choice(incorrect_indices, size=num_samples, replace=False)

    def _make_item(idx: int) -> Dict[str, Any]:
        return {
            "path": str(paths[idx]),
            "true_class": idx_to_class[int(y_true[idx])],
            "pred_class": idx_to_class[int(y_pred[idx])],
            "confidence": round(float(y_conf[idx]), 4) if y_conf is not None else 1.0,
        }

    correct_items = [_make_item(i) for i in correct_indices]
    incorrect_items = [_make_item(i) for i in incorrect_indices]
    return correct_items, incorrect_items


# ---------------------------------------------------------------------------
# 6. Unified evaluation runner
# ---------------------------------------------------------------------------

def evaluate_all() -> None:
    """
    Evaluate all available models (SVM, SmallCNN, ResNet18) on the identical
    test split, generate all charts, and write models/metrics.json.
    """
    from src.data import load_splits, get_label_map, SPLITS_CSV

    df = load_splits(SPLITS_CSV)
    class_to_idx, idx_to_class = get_label_map(df)
    test_df = df[df["split"] == "test"].copy()
    test_df = test_df[~test_df["corrupted"]].reset_index(drop=True)
    test_paths = test_df["path"].tolist()

    all_models_summary: Dict[str, Dict] = {}

    # 1. SVM
    svm_path = MODELS_DIR / "svm.joblib"
    svm_json_path = MODELS_DIR / "svm_metrics.json"
    if svm_path.exists():
        import joblib
        log.info("Evaluating SVM on test split …")
        pipe = joblib.load(svm_path)
        x_test_path = PROJECT_ROOT / "data" / "features" / "test_X.npy"
        y_test_path = PROJECT_ROOT / "data" / "features" / "test_y.npy"

        if x_test_path.exists() and y_test_path.exists():
            X_test = np.load(x_test_path)
            y_test = np.load(y_test_path)
            svm_test = compute_metrics(
                pipe, X_test, y_test, idx_to_class, split="test",
                paths=test_paths, model_name="SVM"
            )
            plot_confusion_matrix(
                svm_test["confusion_matrix"],
                svm_test["class_names"],
                title="SVM – Test Confusion Matrix",
            )
            if svm_json_path.exists():
                with open(svm_json_path) as f:
                    svm_meta = json.load(f)
            else:
                svm_meta = {"model": "SVM (RBF)"}
            svm_meta["test"] = svm_test
            with open(svm_json_path, "w") as f:
                json.dump(svm_meta, f, indent=2)
            all_models_summary["SVM"] = svm_meta

    # 2. SmallCNN
    cnn_path = MODELS_DIR / "cnn.pt"
    cnn_json_path = MODELS_DIR / "cnn_metrics.json"
    if cnn_path.exists():
        import torch
        from src.models import SmallCNN
        from src.data import make_loaders
        log.info("Evaluating SmallCNN on test split …")
        _, _, test_loader, _, _ = make_loaders(image_size=128, batch_size=32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn = SmallCNN(num_classes=len(class_to_idx))
        cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        cnn.to(device).eval()

        cnn_test = compute_metrics(
            cnn, test_loader, None, idx_to_class, split="test",
            model_name="SmallCNN"
        )
        plot_confusion_matrix(
            cnn_test["confusion_matrix"],
            cnn_test["class_names"],
            title="SmallCNN – Test Confusion Matrix",
        )
        if cnn_json_path.exists():
            with open(cnn_json_path) as f:
                cnn_meta = json.load(f)
        else:
            cnn_meta = {"model": "SmallCNN"}
        cnn_meta["test"] = cnn_test
        with open(cnn_json_path, "w") as f:
            json.dump(cnn_meta, f, indent=2)
        all_models_summary["SmallCNN"] = cnn_meta

    # 3. ResNet18
    rn_path = MODELS_DIR / "resnet18.pt"
    rn_json_path = MODELS_DIR / "resnet18_metrics.json"
    if rn_path.exists():
        import torch
        from src.models import build_resnet18
        from src.data import make_loaders
        log.info("Evaluating ResNet18 on test split …")
        _, _, test_loader, _, _ = make_loaders(image_size=160, batch_size=32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rn = build_resnet18(num_classes=len(class_to_idx), pretrained=False)
        rn.load_state_dict(torch.load(rn_path, map_location=device, weights_only=True))
        rn.to(device).eval()

        rn_test = compute_metrics(
            rn, test_loader, None, idx_to_class, split="test",
            model_name="ResNet18"
        )
        plot_confusion_matrix(
            rn_test["confusion_matrix"],
            rn_test["class_names"],
            title="ResNet18 – Test Confusion Matrix",
        )
        if rn_json_path.exists():
            with open(rn_json_path) as f:
                rn_meta = json.load(f)
        else:
            rn_meta = {"model": "ResNet18"}
        rn_meta["test"] = rn_test
        with open(rn_json_path, "w") as f:
            json.dump(rn_meta, f, indent=2)
        all_models_summary["ResNet18"] = rn_meta

    # 4. Save combined metrics.json
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_models_summary, f, indent=2)
    log.info("Saved combined metrics to %s", metrics_path)

    # 5. Generate comparison chart if 2 or more models are present
    if len(all_models_summary) >= 2:
        flat = {}
        for name, m in all_models_summary.items():
            test_m = m.get("test", {})
            flat[name] = {
                "accuracy": test_m.get("accuracy", 0.0),
                "macro_f1": test_m.get("macro_f1", 0.0),
            }
        plot_model_comparison(flat)


if __name__ == "__main__":
    evaluate_all()



