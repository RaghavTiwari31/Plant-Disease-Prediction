"""
predict.py – Phase 3: Single-image inference for all three models.

Loads any saved model (SVM, SmallCNN, ResNet18) and classifies one image,
returning the predicted class, confidence, and top-3 probabilities.

Usage (CLI):
    python -m src.predict --image path/to/leaf.jpg --model cnn
    python -m src.predict --image path/to/leaf.jpg --model resnet18
    python -m src.predict --image path/to/leaf.jpg --model svm

Usage (Python API):
    from src.predict import load_model, predict_image
    model_bundle = load_model("cnn")               # or "resnet18" / "svm"
    result = predict_image(model_bundle, "leaf.jpg")
    # result = {"class": "...", "confidence": 0.97, "top3": [...]}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"
SPLITS_CSV   = DATA_DIR / "splits.csv"

# Model file map
MODEL_FILES = {
    "cnn":      MODELS_DIR / "cnn.pt",
    "resnet18": MODELS_DIR / "resnet18.pt",
    "svm":      MODELS_DIR / "svm.joblib",
}


# ---------------------------------------------------------------------------
# 1. Label map (loaded from splits.csv so it matches training exactly)
# ---------------------------------------------------------------------------

def _get_label_map() -> Tuple[Dict[str, int], Dict[int, str]]:
    from src.data import load_splits, get_label_map
    df = load_splits(SPLITS_CSV)
    return get_label_map(df)


# ---------------------------------------------------------------------------
# 2. Model bundle dataclass
# ---------------------------------------------------------------------------

class ModelBundle:
    """
    Thin wrapper holding a loaded model + metadata needed for inference.

    Attributes
    ----------
    model_name   : "cnn" | "resnet18" | "svm"
    model        : the underlying model object (nn.Module or sklearn Pipeline)
    idx_to_class : int -> class-name string
    image_size   : expected spatial resolution (for DL models)
    device       : torch.device (for DL models)
    """

    def __init__(
        self,
        model_name: str,
        model: Any,
        idx_to_class: Dict[int, str],
        image_size: int,
        device: Any = None,
    ) -> None:
        self.model_name   = model_name
        self.model        = model
        self.idx_to_class = idx_to_class
        self.image_size   = image_size
        self.device       = device


# ---------------------------------------------------------------------------
# 3. load_model
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> ModelBundle:
    """
    Load a saved model and return a ModelBundle ready for inference.

    Parameters
    ----------
    model_name : "cnn" | "resnet18" | "svm"

    Raises
    ------
    FileNotFoundError  if the model file is missing.
    ValueError         if model_name is not recognised.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_FILES.keys())}"
        )

    model_path = MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Train the model first with: python -m src.train_dl --model {model_name}"
        )

    _, idx_to_class = _get_label_map()
    num_classes = len(idx_to_class)

    # ---- DL models ---------------------------------------------------------
    if model_name in ("cnn", "resnet18"):
        import torch
        from src.train_dl import CNN_IMAGE_SIZE, RESNET_IMAGE_SIZE, get_device
        from src.models   import SmallCNN, build_resnet18

        device = get_device()

        if model_name == "cnn":
            net        = SmallCNN(num_classes=num_classes)
            image_size = CNN_IMAGE_SIZE
        else:
            net        = build_resnet18(num_classes=num_classes, pretrained=False)
            image_size = RESNET_IMAGE_SIZE

        net.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        net.to(device).eval()
        return ModelBundle(model_name, net, idx_to_class, image_size, device)

    # ---- SVM ---------------------------------------------------------------
    else:
        import joblib
        from src.features import IMG_SIZE
        pipeline = joblib.load(model_path)
        return ModelBundle("svm", pipeline, idx_to_class, IMG_SIZE, device=None)


# ---------------------------------------------------------------------------
# 4. Preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_for_dl(
    img: Image.Image,
    image_size: int,
) -> "torch.Tensor":
    """Return a (1, 3, H, W) float tensor normalised with ImageNet stats."""
    from src.data import get_transforms
    transform = get_transforms("val", image_size=image_size)
    return transform(img.convert("RGB")).unsqueeze(0)


def _preprocess_for_svm(img: Image.Image) -> np.ndarray:
    """Return a (1, D) feature vector ready for the SVM pipeline."""
    from src.features import extract_features
    arr = np.array(img.convert("RGB"))
    return extract_features(arr).reshape(1, -1)


# ---------------------------------------------------------------------------
# 5. predict_image
# ---------------------------------------------------------------------------

def predict_image(
    bundle: ModelBundle,
    image_path: str | Path,
    top_k: int = 3,
) -> Dict:
    """
    Classify a single image with the given ModelBundle.

    Parameters
    ----------
    bundle      : loaded ModelBundle (from load_model)
    image_path  : path to the image file
    top_k       : number of top predictions to return

    Returns
    -------
    dict with keys:
        class        – predicted class name (str)
        confidence   – probability / decision score (float, 0–1 for DL/SVM)
        top3         – list of {"class": str, "confidence": float} dicts
        inference_ms – inference time in milliseconds
        model        – model name
    """
    image_path = Path(image_path)

    # Validate image
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)   # re-open after verify
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

    t0 = time.perf_counter()

    # ---- DL inference ------------------------------------------------------
    if bundle.model_name in ("cnn", "resnet18"):
        import torch
        tensor = _preprocess_for_dl(img, bundle.image_size).to(bundle.device)
        with torch.no_grad():
            logits = bundle.model(tensor)              # (1, C)
            probs  = torch.softmax(logits, dim=1)[0]   # (C,)

        top_k_actual = min(top_k, len(probs))
        top_values, top_indices = torch.topk(probs, top_k_actual)
        top_values  = top_values.cpu().numpy().tolist()
        top_indices = top_indices.cpu().numpy().tolist()

        pred_idx   = top_indices[0]
        confidence = top_values[0]
        top_list   = [
            {"class": bundle.idx_to_class[i], "confidence": round(float(v), 4)}
            for i, v in zip(top_indices, top_values)
        ]

    # ---- SVM inference -----------------------------------------------------
    else:
        feat = _preprocess_for_svm(img)   # (1, D)
        proba = bundle.model.predict_proba(feat)[0]   # (C,)

        top_indices = np.argsort(proba)[::-1][:top_k]
        pred_idx    = top_indices[0]
        confidence  = float(proba[pred_idx])
        top_list    = [
            {"class": bundle.idx_to_class[i], "confidence": round(float(proba[i]), 4)}
            for i in top_indices
        ]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "class":        bundle.idx_to_class[pred_idx],
        "confidence":   round(confidence, 4),
        "top3":         top_list,
        "inference_ms": round(elapsed_ms, 2),
        "model":        bundle.model_name,
    }


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify a single leaf image with a trained model."
    )
    p.add_argument("--image", required=True, help="Path to the image file")
    p.add_argument(
        "--model", choices=["cnn", "resnet18", "svm"], default="cnn",
        help="Which model to use (default: cnn)",
    )
    p.add_argument("--top", type=int, default=3, help="Number of top predictions to show")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"Loading model: {args.model} …")
    bundle = load_model(args.model)

    print(f"Classifying: {args.image} …")
    result = predict_image(bundle, args.image, top_k=args.top)

    print(f"\nPrediction : {result['class']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"Time       : {result['inference_ms']:.1f} ms")
    print(f"\nTop-{args.top}:")
    for i, entry in enumerate(result["top3"], 1):
        print(f"  {i}. {entry['class']:40s}  {entry['confidence']:.2%}")

