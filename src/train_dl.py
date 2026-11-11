"""
train_dl.py – Phase 3: Train SmallCNN and ResNet18 on PlantVillage.

Training recipe (identical for both models unless noted):
  • Loss        : CrossEntropyLoss (with optional class weights)
  • Optimiser   : Adam
  • LR          : 1e-3 (CNN + ResNet head);  1e-4 (ResNet fine-tune stage 2)
  • Batch size  : 32
  • Scheduler   : ReduceLROnPlateau (patience=2, factor=0.5)
  • Early stop  : patience 3 on val loss
  • Best ckpt   : saved by lowest val loss
  • Seed        : 42 (fixed everywhere)

ResNet18 two-stage fine-tuning:
  Stage 1 (~3 epochs) : backbone frozen, only head trained at LR=1e-3
  Stage 2 (~5 epochs) : layer4 + head unfrozen, LR=1e-4

Usage:
    python -m src.train_dl --model cnn      [--cap 400] [--epochs 15]
    python -m src.train_dl --model resnet18 [--cap 400] [--epochs 10]
    python -m src.train_dl --model all      [--cap 400]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
BATCH_SIZE    = 32
LR_HEAD       = 1e-3   # CNN full training + ResNet head (stage 1)
LR_FINETUNE   = 1e-4   # ResNet stage 2 (layer4 + head)
EPOCHS_CNN    = 15
EPOCHS_RN_S1  = 3      # ResNet stage 1 (head only)
EPOCHS_RN_S2  = 7      # ResNet stage 2 (layer4 + head)

SCHEDULER_PATIENCE = 2
EARLY_STOP_PATIENCE = 3
CNN_IMAGE_SIZE   = 128
RESNET_IMAGE_SIZE = 160   # smaller than 224 to save CPU time


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 2. Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as e:
            log.warning("CUDA is detected, but device execution failed (%s). Falling back to CPU.", e)
            return torch.device("cpu")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 3. Class weights helper
# ---------------------------------------------------------------------------

def compute_class_weights(
    loader: DataLoader, num_classes: int, device: torch.device
) -> Optional[torch.Tensor]:
    """
    Compute inverse-frequency class weights from a DataLoader.
    Returns None if the imbalance ratio is below 3× (weights not needed).
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in loader:
        for lbl in labels:
            counts[lbl.item()] += 1

    if counts.min() == 0:
        return None

    imbalance_ratio = counts.max() / counts.min()
    if imbalance_ratio < 3.0:
        log.info("Class imbalance ratio %.1f× < 3 – not using class weights.", imbalance_ratio)
        return None

    log.info("Class imbalance ratio %.1f× – using inverse-frequency weights.", imbalance_ratio)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes   # normalise
    return weights.to(device)


# ---------------------------------------------------------------------------
# 4. One epoch of training
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = correct = seen = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(labels)
        correct    += (outputs.argmax(1) == labels).sum().item()
        seen       += len(labels)

    return total_loss / seen, correct / seen


# ---------------------------------------------------------------------------
# 5. One epoch of evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = correct = seen = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * len(labels)
        correct    += (outputs.argmax(1) == labels).sum().item()
        seen       += len(labels)

    return total_loss / seen, correct / seen


# ---------------------------------------------------------------------------
# 6. Training loop with early stopping + best-ckpt save
# ---------------------------------------------------------------------------

def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    n_epochs: int,
    ckpt_path: Path,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    stage_label: str = "",
) -> Dict[str, List[float]]:
    """
    Full training loop.  Returns history dict with train/val loss & accuracy.
    Saves the best checkpoint (by val loss) to ckpt_path.
    """
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_loss = float("inf")
    patience_ctr  = 0
    label = f"[{stage_label}] " if stage_label else ""

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, optimiser, device)
        vl_loss, vl_acc = _eval_epoch(model, val_loader, criterion, device)

        scheduler.step(vl_loss)
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        log.info(
            "%sEpoch %2d/%d  tr_loss=%.4f  tr_acc=%.4f  "
            "vl_loss=%.4f  vl_acc=%.4f  lr=%.2e  (%.1fs)",
            label, epoch, n_epochs,
            tr_loss, tr_acc, vl_loss, vl_acc,
            optimiser.param_groups[0]["lr"], elapsed,
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), ckpt_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                log.info("%sEarly stopping at epoch %d.", label, epoch)
                break

    # Restore best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return history


# ---------------------------------------------------------------------------
# 7. Train SmallCNN
# ---------------------------------------------------------------------------

def train_cnn(
    cap_per_class: Optional[int] = None,
    n_epochs: int = EPOCHS_CNN,
) -> None:
    from src.data      import make_loaders, load_splits, get_label_map, SPLITS_CSV
    from src.models    import SmallCNN, count_parameters
    from src.evaluate  import compute_metrics, plot_confusion_matrix, plot_training_curves

    set_seed()
    device = get_device()
    log.info("Device: %s", device)

    df = load_splits(SPLITS_CSV)
    class_to_idx, idx_to_class = get_label_map(df)
    num_classes = len(class_to_idx)

    train_loader, val_loader, test_loader, _, _ = make_loaders(
        image_size=CNN_IMAGE_SIZE, batch_size=BATCH_SIZE,
        num_workers=0, cap_per_class=cap_per_class,
    )

    model = SmallCNN(num_classes=num_classes).to(device)
    params = count_parameters(model)
    log.info("SmallCNN  total params: %s  trainable: %s",
             f"{params['total']:,}", f"{params['trainable']:,}")

    class_weights = compute_class_weights(train_loader, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = Adam(model.parameters(), lr=LR_HEAD)
    scheduler = ReduceLROnPlateau(optimiser, mode="min",
                                  patience=SCHEDULER_PATIENCE, factor=0.5)

    ckpt_path = MODELS_DIR / "cnn_best.pt"
    t_start   = time.perf_counter()

    history = training_loop(
        model, train_loader, val_loader, criterion, optimiser,
        scheduler, device, n_epochs, ckpt_path, stage_label="CNN",
    )
    train_time = time.perf_counter() - t_start

    # Test evaluation
    log.info("Evaluating SmallCNN on test set …")
    t_inf  = time.perf_counter()
    test_metrics = compute_metrics(model, test_loader, None, idx_to_class, split="test", model_name="SmallCNN")
    inf_time_per = (time.perf_counter() - t_inf) / max(len(test_loader.dataset), 1)

    val_metrics = compute_metrics(model, val_loader, None, idx_to_class, split="val")

    # Save final weights (same as best ckpt)
    final_path = MODELS_DIR / "cnn.pt"
    torch.save(model.state_dict(), final_path)
    model_size_mb = final_path.stat().st_size / (1024 ** 2)

    # Charts
    plot_training_curves(history, model_name="SmallCNN")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        test_metrics["class_names"],
        title="SmallCNN – Test Confusion Matrix",
    )

    # Persist metrics
    out = {
        "model": "SmallCNN",
        "architecture": "4×(Conv/BN/ReLU/MaxPool) + GAP + Dropout + Linear",
        "input_size": CNN_IMAGE_SIZE,
        "params": params,
        "train_time_s": round(train_time, 1),
        "inference_time_per_image_ms": round(inf_time_per * 1000, 3),
        "model_size_mb": round(model_size_mb, 2),
        "cap_per_class": cap_per_class,
        "history": history,
        "val":  val_metrics,
        "test": test_metrics,
    }
    metrics_path = MODELS_DIR / "cnn_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("CNN metrics saved to %s", metrics_path)

    _print_summary("SmallCNN", val_metrics, test_metrics,
                   inf_time_per, model_size_mb, train_time)


# ---------------------------------------------------------------------------
# 8. Train ResNet18 (two-stage fine-tuning)
# ---------------------------------------------------------------------------

def train_resnet18(
    cap_per_class: Optional[int] = None,
    epochs_stage1: int = EPOCHS_RN_S1,
    epochs_stage2: int = EPOCHS_RN_S2,
) -> None:
    from src.data      import make_loaders, load_splits, get_label_map, SPLITS_CSV
    from src.models    import build_resnet18, freeze_backbone, unfreeze_layer4, count_parameters
    from src.evaluate  import compute_metrics, plot_confusion_matrix, plot_training_curves

    set_seed()
    device = get_device()
    log.info("Device: %s", device)

    df = load_splits(SPLITS_CSV)
    class_to_idx, idx_to_class = get_label_map(df)
    num_classes = len(class_to_idx)

    train_loader, val_loader, test_loader, _, _ = make_loaders(
        image_size=RESNET_IMAGE_SIZE, batch_size=BATCH_SIZE,
        num_workers=0, cap_per_class=cap_per_class,
    )

    model = build_resnet18(num_classes=num_classes, pretrained=True).to(device)
    class_weights = compute_class_weights(train_loader, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    ckpt_path = MODELS_DIR / "resnet18_best.pt"
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    t_start = time.perf_counter()

    # ---- Stage 1: head only -----------------------------------------------
    freeze_backbone(model)
    params_s1 = count_parameters(model)
    log.info("ResNet18 Stage 1 – trainable: %s / %s",
             f"{params_s1['trainable']:,}", f"{params_s1['total']:,}")

    optimiser = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scheduler = ReduceLROnPlateau(optimiser, mode="min",
                                  patience=SCHEDULER_PATIENCE, factor=0.5)

    h1 = training_loop(
        model, train_loader, val_loader, criterion, optimiser,
        scheduler, device, epochs_stage1, ckpt_path,
        early_stop_patience=EARLY_STOP_PATIENCE, stage_label="ResNet18-S1",
    )
    for k in history:
        history[k].extend(h1[k])

    # ---- Stage 2: layer4 + head -------------------------------------------
    unfreeze_layer4(model)
    params_s2 = count_parameters(model)
    log.info("ResNet18 Stage 2 – trainable: %s / %s",
             f"{params_s2['trainable']:,}", f"{params_s2['total']:,}")

    optimiser = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE)
    scheduler = ReduceLROnPlateau(optimiser, mode="min",
                                  patience=SCHEDULER_PATIENCE, factor=0.5)

    h2 = training_loop(
        model, train_loader, val_loader, criterion, optimiser,
        scheduler, device, epochs_stage2, ckpt_path,
        early_stop_patience=EARLY_STOP_PATIENCE, stage_label="ResNet18-S2",
    )
    for k in history:
        history[k].extend(h2[k])

    train_time = time.perf_counter() - t_start

    # Test evaluation
    log.info("Evaluating ResNet18 on test set …")
    t_inf = time.perf_counter()
    test_metrics = compute_metrics(model, test_loader, None, idx_to_class, split="test", model_name="ResNet18")
    inf_time_per = (time.perf_counter() - t_inf) / max(len(test_loader.dataset), 1)

    val_metrics = compute_metrics(model, val_loader, None, idx_to_class, split="val")

    # Save final weights
    final_path = MODELS_DIR / "resnet18.pt"
    torch.save(model.state_dict(), final_path)
    model_size_mb = final_path.stat().st_size / (1024 ** 2)

    # Charts
    plot_training_curves(history, model_name="ResNet18")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        test_metrics["class_names"],
        title="ResNet18 – Test Confusion Matrix",
    )

    # Persist metrics
    out = {
        "model": "ResNet18",
        "architecture": "ImageNet-pretrained ResNet18, head replaced, 2-stage fine-tune",
        "input_size": RESNET_IMAGE_SIZE,
        "params_stage1": params_s1,
        "params_stage2": params_s2,
        "train_time_s": round(train_time, 1),
        "inference_time_per_image_ms": round(inf_time_per * 1000, 3),
        "model_size_mb": round(model_size_mb, 2),
        "cap_per_class": cap_per_class,
        "history": history,
        "val":  val_metrics,
        "test": test_metrics,
    }
    metrics_path = MODELS_DIR / "resnet18_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("ResNet18 metrics saved to %s", metrics_path)

    _print_summary("ResNet18", val_metrics, test_metrics,
                   inf_time_per, model_size_mb, train_time)


# ---------------------------------------------------------------------------
# 9. Combined metrics.json (for the Streamlit app)
# ---------------------------------------------------------------------------

def merge_metrics() -> None:
    """
    Merge svm_metrics.json + cnn_metrics.json + resnet18_metrics.json
    into a single models/metrics.json that the UI reads.
    """
    combined = {}
    for name, fname in [
        ("SVM",      "svm_metrics.json"),
        ("SmallCNN", "cnn_metrics.json"),
        ("ResNet18", "resnet18_metrics.json"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            with open(p) as f:
                combined[name] = json.load(f)
        else:
            log.warning("Missing %s – skipping in metrics.json", fname)

    out_path = MODELS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    log.info("Combined metrics saved to %s", out_path)


# ---------------------------------------------------------------------------
# 10. Comparison chart
# ---------------------------------------------------------------------------

def make_comparison_chart() -> None:
    """Read metrics.json and produce a model comparison bar chart."""
    from src.evaluate import plot_model_comparison

    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        log.warning("metrics.json not found – skipping comparison chart.")
        return

    with open(metrics_path) as f:
        all_metrics = json.load(f)

    # Flatten: each entry needs accuracy + macro_f1 at the top level
    flat = {}
    for model_name, m in all_metrics.items():
        test_m = m.get("test", m)
        flat[model_name] = {
            "accuracy": test_m.get("accuracy", 0.0),
            "macro_f1": test_m.get("macro_f1", 0.0),
        }

    if flat:
        plot_model_comparison(flat)
        log.info("Comparison chart saved.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_summary(name, val_metrics, test_metrics, inf_time_per, model_size_mb, train_time):
    print("\n" + "=" * 55)
    print(f"{name} Results")
    print("=" * 55)
    print(f"  Val  accuracy : {val_metrics['accuracy']:.4f}")
    print(f"  Val  macro-F1 : {val_metrics['macro_f1']:.4f}")
    print(f"  Test accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test macro-F1 : {test_metrics['macro_f1']:.4f}")
    print(f"  Train time    : {train_time/60:.1f} min")
    print(f"  Inference     : {inf_time_per*1000:.2f} ms/image")
    print(f"  Model size    : {model_size_mb:.1f} MB")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DL models on PlantVillage")
    p.add_argument("--model", choices=["cnn", "resnet18", "all"], default="all",
                   help="Which model to train (default: all)")
    p.add_argument("--cap", type=int, default=None, metavar="N",
                   help="Cap images per class for speed (e.g. 400)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epoch count for CNN (or stage1+stage2 for ResNet18)")
    p.add_argument("--merge", action="store_true",
                   help="Just merge existing metric files into metrics.json and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.merge:
        merge_metrics()
        make_comparison_chart()
    else:
        if args.model in ("cnn", "all"):
            train_cnn(
                cap_per_class=args.cap,
                n_epochs=args.epochs or EPOCHS_CNN,
            )

        if args.model in ("resnet18", "all"):
            train_resnet18(
                cap_per_class=args.cap,
                epochs_stage1=max(1, (args.epochs or (EPOCHS_RN_S1 + EPOCHS_RN_S2)) // 3),
                epochs_stage2=(args.epochs or (EPOCHS_RN_S1 + EPOCHS_RN_S2)) - max(1, (args.epochs or (EPOCHS_RN_S1 + EPOCHS_RN_S2)) // 3),
            )

        merge_metrics()
        make_comparison_chart()

