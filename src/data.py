"""
data.py – Phase 1: Dataset utilities for PlantVillage.

Responsibilities:
  - Scan the dataset, count images per class, detect corrupted files, check
    resolutions.
  - Compute perceptual hashes (imagehash) for near-duplicate detection.
  - Cluster near-identical images into groups and perform a leakage-safe
    stratified split (GroupShuffleSplit) → 70 / 15 / 15.
  - Save the split manifest to data/splits.csv so every model uses identical splits.
  - Provide PyTorch Dataset and DataLoader factories with correct transforms.

Expected dataset layout:
    data/raw/color/<ClassName>/<image_file>

Usage:
    python -m src.data          # runs the full scan + split pipeline
    from src.data import make_loaders, load_splits
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "color"
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_CSV = DATA_DIR / "splits.csv"

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
# Constants
# ---------------------------------------------------------------------------
HASH_SIZE = 8          # pHash grid dimension (8 → 64-bit hash)
HASH_THRESHOLD = 10    # Hamming distance ≤ this → near-duplicate
RANDOM_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test_frac  = 1 − TRAIN_FRAC − VAL_FRAC = 0.15

# ImageNet statistics used for normalisation (applied to all splits)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 1. Dataset scanning
# ---------------------------------------------------------------------------

def scan_dataset(root: Path = DATA_RAW) -> pd.DataFrame:
    """
    Walk *root* and return a DataFrame with one row per image.

    Columns: path, class_name, width, height, corrupted, file_size_kb
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Please place the PlantVillage color dataset at:\n"
            f"  {root}\n"
            "Expected layout: <root>/<ClassName>/<image>.jpg"
        )

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in {root}")

    log.info("Scanning %d class folders in %s ...", len(class_dirs), root)

    records: List[dict] = []
    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        image_files = [
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ]
        for img_path in image_files:
            corrupted = False
            w = h = None
            try:
                with Image.open(img_path) as img:
                    img.verify()   # lightweight corruption check
                # Re-open to get size (verify() closes the file handle)
                with Image.open(img_path) as img:
                    w, h = img.size
            except (UnidentifiedImageError, Exception):
                corrupted = True

            records.append(
                {
                    "path":         str(img_path),
                    "class_name":   cls_name,
                    "width":        w,
                    "height":       h,
                    "corrupted":    corrupted,
                    "file_size_kb": round(img_path.stat().st_size / 1024, 1),
                }
            )

    df = pd.DataFrame(records)
    log.info(
        "Scan complete: %d images, %d classes, %d corrupted.",
        len(df), df["class_name"].nunique(), df["corrupted"].sum(),
    )
    return df


# ---------------------------------------------------------------------------
# 2. Perceptual hashing & group assignment
# ---------------------------------------------------------------------------

def compute_phashes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'phash' column to *df* (string representation of the perceptual hash).
    Skips corrupted images (phash = None).
    """
    hashes: List[Optional[str]] = []
    total = len(df)
    for i, row in enumerate(df.itertuples(), 1):
        if row.corrupted:
            hashes.append(None)
            continue
        try:
            with Image.open(row.path) as img:
                h = imagehash.phash(img, hash_size=HASH_SIZE)
            hashes.append(str(h))
        except Exception:
            hashes.append(None)
        if i % 1000 == 0:
            log.info("  hashed %d / %d", i, total)

    df = df.copy()
    df["phash"] = hashes
    return df


def assign_hash_groups(df: pd.DataFrame, threshold: int = HASH_THRESHOLD) -> pd.DataFrame:
    """
    Cluster images whose perceptual hashes are within *threshold* Hamming
    distance into the same group.  Images in the same group must stay in the
    same split to prevent data leakage.

    Uses a union-find structure with a sliding-window comparison for speed.

    Returns df with a new integer column 'group_id'.
    """
    df = df.copy()
    df["group_id"] = -1

    valid_mask = df["phash"].notna()
    valid_idx  = df.index[valid_mask].tolist()

    # --- Union-Find ---
    parent: Dict[int, int] = {i: i for i in range(len(df))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    # Convert hex strings to imagehash objects
    hash_objects: Dict[int, imagehash.ImageHash] = {}
    for df_idx in valid_idx:
        try:
            hash_objects[df_idx] = imagehash.hex_to_hash(df.at[df_idx, "phash"])
        except Exception:
            pass

    # Sliding-window Hamming comparison
    WINDOW = 200
    keys = list(hash_objects.keys())
    for i, idx_i in enumerate(keys):
        window_start = max(0, i - WINDOW)
        for j in range(window_start, i):
            idx_j = keys[j]
            if (hash_objects[idx_i] - hash_objects[idx_j]) <= threshold:
                union(idx_i, idx_j)

    # Map union-find roots → contiguous group IDs
    next_group = 0
    root_to_group: Dict[int, int] = {}

    for df_idx in df.index:
        if not valid_mask[df_idx] or df_idx not in hash_objects:
            df.at[df_idx, "group_id"] = next_group
            next_group += 1
        else:
            root = find(df_idx)
            if root not in root_to_group:
                root_to_group[root] = next_group
                next_group += 1
            df.at[df_idx, "group_id"] = root_to_group[root]

    n_groups = df["group_id"].nunique()
    log.info(
        "Hash grouping done: %d images -> %d groups (avg %.1f images/group).",
        len(df), n_groups, len(df) / max(n_groups, 1),
    )
    return df


# ---------------------------------------------------------------------------
# 3. Leakage-safe split
# ---------------------------------------------------------------------------

def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each row a 'split' label ('train', 'val', 'test') such that:
      - No hash group spans more than one split.
      - Class proportions are approximately preserved (stratified by class).
      - Approximate ratio: 70 / 15 / 15.

    Uses two rounds of GroupShuffleSplit:
      Round 1: carve out test  (~15 %)
      Round 2: split remaining into train / val  (~82 % / 18 % -> 70 / 15 overall)
    """
    good = df[~df["corrupted"]].copy()
    bad  = df[df["corrupted"]].copy()
    bad["split"] = "excluded"

    X      = good.index.to_numpy().reshape(-1, 1)
    y      = good["class_name"].to_numpy()
    groups = good["group_id"].to_numpy()

    # Round 1: separate test set (~15 %)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=RANDOM_SEED)
    trainval_idx, test_idx = next(gss_test.split(X, y, groups))

    # Round 2: separate val set from train+val
    val_frac_of_trainval = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    X_tv = X[trainval_idx]
    y_tv = y[trainval_idx]
    g_tv = groups[trainval_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=RANDOM_SEED)
    train_idx_local, val_idx_local = next(gss_val.split(X_tv, y_tv, g_tv))

    train_df_idx = good.index[trainval_idx[train_idx_local]]
    val_df_idx   = good.index[trainval_idx[val_idx_local]]
    test_df_idx  = good.index[test_idx]

    good = good.copy()
    good["split"] = "train"
    good.loc[val_df_idx,   "split"] = "val"
    good.loc[test_df_idx,  "split"] = "test"

    result = pd.concat([good, bad], ignore_index=True)

    # Verification: no group should appear in more than one real split
    splits_per_group = (
        result[result["split"].isin(["train", "val", "test"])]
        .groupby("group_id")["split"]
        .nunique()
    )
    leaking_groups = (splits_per_group > 1).sum()
    if leaking_groups > 0:
        log.warning("  %d hash groups appear in more than one split!", leaking_groups)
    else:
        log.info("  No hash group appears in more than one split.")

    counts = result.groupby("split").size().rename("images")
    log.info("Split sizes:\n%s", counts.to_string())
    return result


# ---------------------------------------------------------------------------
# 4. Build & save splits.csv
# ---------------------------------------------------------------------------

def build_splits(root: Path = DATA_RAW, out_csv: Path = SPLITS_CSV) -> pd.DataFrame:
    """
    Full pipeline: scan -> hash -> group -> split -> save CSV.
    Returns the final DataFrame.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    log.info("=== Phase 1: Building dataset splits ===")
    df = scan_dataset(root)
    df = compute_phashes(df)
    df = assign_hash_groups(df)
    df = make_splits(df)

    df.to_csv(out_csv, index=False)
    log.info("Splits saved to %s", out_csv)
    return df


def load_splits(csv_path: Path = SPLITS_CSV) -> pd.DataFrame:
    """Load a previously saved splits.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"splits.csv not found at {csv_path}.\n"
            "Run `python -m src.data` to generate it."
        )
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# 5. Label map helpers
# ---------------------------------------------------------------------------

def get_label_map(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Returns (class_to_idx, idx_to_class) sorted alphabetically.
    Works on the full splits DataFrame or any subset.
    """
    classes = sorted(df["class_name"].dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


# ---------------------------------------------------------------------------
# 6. Transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str, image_size: int = 128) -> transforms.Compose:
    """
    Return the torchvision transform pipeline for a given split.

    Train:
        Resize -> RandomHorizontalFlip -> RandomVerticalFlip ->
        RandomRotation(15) -> ColorJitter(brightness, contrast only) ->
        ToTensor -> Normalize

    Val / Test:
        Resize -> ToTensor -> Normalize

    NOTE: Hue / saturation jitter is intentionally omitted because colour is a
    disease signal in PlantVillage.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.0,  # keep hue/saturation intact
                    hue=0.0,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:  # val / test
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )


# ---------------------------------------------------------------------------
# 7. PyTorch Dataset
# ---------------------------------------------------------------------------

class PlantVillageDataset(Dataset):
    """
    PyTorch Dataset that reads images from the splits DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Rows from splits.csv filtered to a single split (or the full CSV —
        only rows with matching split label are used).
    class_to_idx : dict
        Mapping from class name to integer label.
    transform : callable, optional
        torchvision transform to apply to each PIL image.
    target_split : str, optional
        If given, only rows where split == target_split are included.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        target_split: Optional[str] = None,
    ) -> None:
        mask = ~df["corrupted"]
        if target_split is not None:
            mask = mask & (df["split"] == target_split)
        else:
            mask = mask & df["split"].isin(["train", "val", "test"])
        self.records = df[mask].reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

        # Pre-extract lists to eliminate pandas .iloc overhead in training loop
        self.paths = self.records["path"].tolist()
        self.labels = [self.class_to_idx[c] for c in self.records["class_name"].tolist()]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# ---------------------------------------------------------------------------
# 8. DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(
    splits_csv:    Path = SPLITS_CSV,
    image_size:    int  = 128,
    batch_size:    int  = 32,
    num_workers:   int  = 0,
    cap_per_class: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    Build train / val / test DataLoaders from splits.csv.

    Parameters
    ----------
    splits_csv    : path to the saved splits CSV.
    image_size    : spatial size for resizing (default 128 for CNN/SVM speed).
    batch_size    : images per batch.
    num_workers   : DataLoader worker processes (keep 0 on Windows to avoid
                    multiprocessing issues with CUDA).
    cap_per_class : if set, randomly sample at most this many images per class
                    per split (useful for fast CPU experiments, e.g. 300-500).

    Returns
    -------
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class
    """
    df = load_splits(splits_csv)
    class_to_idx, idx_to_class = get_label_map(df)

    def _make_loader(split_name: str, shuffle: bool) -> DataLoader:
        split_df = df[df["split"] == split_name].copy()
        if split_name == "train" and cap_per_class is not None:
            sampled_indices = []
            for _, grp in split_df.groupby("class_name"):
                n = min(len(grp), cap_per_class)
                sampled_indices.extend(grp.sample(n=n, random_state=RANDOM_SEED).index.tolist())
            split_df = split_df.loc[sampled_indices].reset_index(drop=True)
        ds = PlantVillageDataset(
            df=split_df,
            class_to_idx=class_to_idx,
            transform=get_transforms(split_name, image_size=image_size),
            target_split=split_name,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = _make_loader("train", shuffle=True)
    val_loader   = _make_loader("val",   shuffle=False)
    test_loader  = _make_loader("test",  shuffle=False)

    log.info(
        "DataLoaders ready -- train: %d, val: %d, test: %d batches (bs=%d).",
        len(train_loader), len(val_loader), len(test_loader), batch_size,
    )
    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_splits()
    print("\n=== Dataset summary ===")
    totals = df.groupby("split").size().rename("images")
    print(totals.to_string())
    print(f"\nFull splits CSV -> {SPLITS_CSV}")

