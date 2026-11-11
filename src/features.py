"""
features.py – Phase 2: Hand-crafted feature extraction for PlantVillage.

Feature groups (all computed on a 128x128 resize of the image):
  1. HOG  – Histogram of Oriented Gradients on the grayscale image.
             Captures edge/gradient patterns (shape, texture).
  2. HSV  – Per-channel histogram + mean + std on the HSV image.
             Captures lesion colour and chlorosis.
  3. LBP  – Local Binary Pattern histogram on the grayscale image.
             Captures fine-grained micro-texture.

Pipeline:
  extract_features(img_rgb)  ->  raw feature vector (numpy 1-D)
  build_feature_matrix(df, split)  ->  (X, y) with disk caching (.npy)

Usage:
    python -m src.features          # extract & cache all splits
    from src.features import build_feature_matrix
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR    = PROJECT_ROOT / "data" / "features"

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
# Hyper-parameters (kept here so train_svm.py can read them for the report)
# ---------------------------------------------------------------------------
IMG_SIZE = 128          # resize target (pixels)

# HOG
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (8, 8)
HOG_CELLS_PER_BLOCK  = (2, 2)

# HSV histogram
HSV_BINS_H = 36        # hue:  36 bins  (10° per bin)
HSV_BINS_S = 32        # saturation
HSV_BINS_V = 32        # value

# LBP
LBP_P      = 24        # number of circularly-symmetric neighbour set points
LBP_R      = 3.0       # radius of circle
LBP_METHOD = "uniform"
LBP_BINS   = LBP_P + 2  # uniform LBP has P+2 distinct patterns


# ---------------------------------------------------------------------------
# 1. Single-image feature extractor
# ---------------------------------------------------------------------------

def _extract_hog(gray_uint8: np.ndarray) -> np.ndarray:
    """HOG feature vector from a uint8 grayscale image."""
    feat = hog(
        gray_uint8,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
        channel_axis=None,  # already grayscale
    )
    return feat.astype(np.float32)


def _extract_hsv(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    HSV histogram (H, S, V independently) + per-channel mean & std.
    Returns a 1-D float32 vector of length HSV_BINS_H + HSV_BINS_S + HSV_BINS_V + 6.
    """
    # OpenCV expects BGR
    bgr   = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Normalise channels to [0, 1]
    h = hsv[:, :, 0] / 180.0
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    hist_h, _ = np.histogram(h, bins=HSV_BINS_H, range=(0.0, 1.0), density=True)
    hist_s, _ = np.histogram(s, bins=HSV_BINS_S, range=(0.0, 1.0), density=True)
    hist_v, _ = np.histogram(v, bins=HSV_BINS_V, range=(0.0, 1.0), density=True)

    # Bin counts already sum to ~1 when density=True; keep as float32
    stats = np.array(
        [h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std()],
        dtype=np.float32,
    )

    return np.concatenate(
        [hist_h.astype(np.float32),
         hist_s.astype(np.float32),
         hist_v.astype(np.float32),
         stats]
    )


def _extract_lbp(gray_uint8: np.ndarray) -> np.ndarray:
    """
    Uniform LBP histogram (normalised) from a uint8 grayscale image.
    Returns float32 vector of length LBP_P + 2.
    """
    lbp = local_binary_pattern(
        gray_uint8.astype(np.uint8),
        P=LBP_P,
        R=LBP_R,
        method=LBP_METHOD,
    )
    hist, _ = np.histogram(
        lbp,
        bins=LBP_BINS,
        range=(0, LBP_BINS),
        density=True,
    )
    return hist.astype(np.float32)


def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract the full concatenated feature vector from an RGB image (H x W x 3,
    uint8 or float).  The image is resized to IMG_SIZE x IMG_SIZE internally.

    Feature layout:
        [HOG (8100-d)] | [HSV hist+stats (100-d)] | [LBP hist (26-d)]
        Total: 8226-d  (with default hyper-parameters)

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image array.  Accepted dtypes: uint8 or float32/64 in [0, 1].

    Returns
    -------
    np.ndarray, shape (D,), dtype float32
    """
    # Ensure uint8 RGB
    if img_rgb.dtype != np.uint8:
        img_rgb = (np.clip(img_rgb, 0.0, 1.0) * 255).astype(np.uint8)

    # Resize
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Grayscale for HOG / LBP
    gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)

    hog_feat = _extract_hog(gray)
    hsv_feat = _extract_hsv(img_rgb)
    lbp_feat = _extract_lbp(gray)

    return np.concatenate([hog_feat, hsv_feat, lbp_feat])


# ---------------------------------------------------------------------------
# 2. Batch extraction with disk caching
# ---------------------------------------------------------------------------

def _cache_path(split: str, suffix: str = "") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{split}{suffix}.npy"


def build_feature_matrix(
    df: pd.DataFrame,
    split: str,
    class_to_idx: Dict[str, int],
    force: bool = False,
    cap_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for all images in *split* and return (X, y).
    Results are cached to `data/features/<split>_X.npy` and `<split>_y.npy`.

    Parameters
    ----------
    df            : full splits DataFrame (from load_splits).
    split         : 'train', 'val', or 'test'.
    class_to_idx  : mapping from class name to integer label.
    force         : if True, ignore cache and recompute.
    cap_per_class : if set, randomly sample at most this many images per class
                    before extraction (useful for fast CPU experiments).

    Returns
    -------
    X : np.ndarray, shape (N, D)
    y : np.ndarray, shape (N,), dtype int
    """
    from src.data import RANDOM_SEED

    x_path = _cache_path(split, "_X")
    y_path = _cache_path(split, "_y")

    if not force and x_path.exists() and y_path.exists():
        log.info("Loading cached features for '%s' from %s", split, CACHE_DIR)
        return np.load(x_path), np.load(y_path)

    rows = df[df["split"] == split].copy()
    rows = rows[~rows["corrupted"]].reset_index(drop=True)

    if cap_per_class is not None:
        sampled_indices = []
        for _, grp in rows.groupby("class_name"):
            n = min(len(grp), cap_per_class)
            sampled_indices.extend(grp.sample(n=n, random_state=RANDOM_SEED).index.tolist())
        rows = rows.loc[sampled_indices].reset_index(drop=True)

    log.info("Extracting features for '%s': %d images …", split, len(rows))

    X_list = []
    y_list = []
    for i, row in enumerate(rows.itertuples(), 1):
        try:
            img = np.array(Image.open(row.path).convert("RGB"))
            feat = extract_features(img)
            X_list.append(feat)
            y_list.append(class_to_idx[row.class_name])
        except Exception as exc:
            log.warning("  Skipping %s: %s", row.path, exc)
        if i % 500 == 0:
            log.info("  processed %d / %d", i, len(rows))

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.save(x_path, X)
    np.save(y_path, y)
    log.info("Cached to %s  (X: %s, y: %s)", CACHE_DIR, X.shape, y.shape)

    return X, y


# ---------------------------------------------------------------------------
# 3. Feature-dimension summary helper
# ---------------------------------------------------------------------------

def feature_dim_report() -> None:
    """Print a breakdown of the feature vector dimensions."""
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    gray  = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    hog_len = len(_extract_hog(gray))
    hsv_len = len(_extract_hsv(dummy))
    lbp_len = len(_extract_lbp(gray))
    total   = hog_len + hsv_len + lbp_len

    print("Feature vector breakdown:")
    print(f"  HOG  (shape+texture) : {hog_len:5d}")
    print(f"  HSV  (colour)        : {hsv_len:5d}")
    print(f"  LBP  (micro-texture) : {lbp_len:5d}")
    print(f"  TOTAL                : {total:5d}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data import load_splits, get_label_map, SPLITS_CSV

    feature_dim_report()
    print()

    df             = load_splits(SPLITS_CSV)
    class_to_idx, _ = get_label_map(df)

    for split in ("train", "val", "test"):
        X, y = build_feature_matrix(df, split, class_to_idx, force=False)
        print(f"{split:5s}  X: {X.shape}  y: {y.shape}  classes: {len(np.unique(y))}")

