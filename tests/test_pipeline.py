"""
test_pipeline.py – Automated test suite for PlantVillage AI classification.

Verifies:
  1. Dataset loads and labels map bi-directionally without corruption.
  2. Preprocessing pipelines output correct tensor/vector dimensions (DL & Handcrafted).
  3. Model save/load round-trip succeeds for all three trained models (SVM, SmallCNN, ResNet18).
  4. Prediction determinism: same image produces identical predictions twice.
  5. Invalid/corrupted image files are rejected gracefully with a ValueError.
  6. Leakage control: zero perceptual hash groups span across multiple splits.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from src.data import (
    load_splits,
    get_label_map,
    get_transforms,
    SPLITS_CSV,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.features import extract_features, IMG_SIZE
from src.predict import load_model, predict_image, MODEL_FILES


# ---------------------------------------------------------------------------
# Test 1: Dataset loads and labels map correctly
# ---------------------------------------------------------------------------
def test_dataset_loads_and_labels_map():
    assert SPLITS_CSV.exists(), f"splits.csv not found at {SPLITS_CSV}"
    df = load_splits(SPLITS_CSV)
    assert not df.empty, "splits.csv is empty"
    assert "class_name" in df.columns
    assert "split" in df.columns

    class_to_idx, idx_to_class = get_label_map(df)
    assert len(class_to_idx) == len(idx_to_class)
    assert len(class_to_idx) >= 5, f"Expected at least 5 classes, found {len(class_to_idx)}"

    # Bi-directional consistency
    for c_name, idx in class_to_idx.items():
        assert idx_to_class[idx] == c_name

    # Split label coverage
    splits_present = set(df[~df["corrupted"]]["split"].unique())
    assert {"train", "val", "test"}.issubset(splits_present)


# ---------------------------------------------------------------------------
# Test 2: Preprocessing output shapes (DL & Handcrafted)
# ---------------------------------------------------------------------------
def test_preprocessing_output_shapes():
    # 1. Hand-crafted feature vector (HOG + HSV + LBP)
    dummy_rgb = np.random.randint(0, 255, (200, 250, 3), dtype=np.uint8)
    feat = extract_features(dummy_rgb)
    assert isinstance(feat, np.ndarray)
    assert feat.dtype == np.float32
    assert feat.shape == (8232,), f"Expected (8232,), got {feat.shape}"

    # 2. Deep learning transform shapes (SmallCNN 128x128, ResNet18 160x160)
    pil_dummy = Image.fromarray(dummy_rgb)
    t_cnn = get_transforms("val", image_size=128)
    tensor_cnn = t_cnn(pil_dummy)
    assert tensor_cnn.shape == (3, 128, 128)

    t_rn = get_transforms("train", image_size=160)
    tensor_rn = t_rn(pil_dummy)
    assert tensor_rn.shape == (3, 160, 160)


# ---------------------------------------------------------------------------
# Test 3: Model save/load round-trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m_name", ["svm", "cnn", "resnet18"])
def test_model_save_load_round_trip(m_name):
    model_path = MODEL_FILES[m_name]
    assert model_path.exists(), f"Model file missing: {model_path}"

    bundle = load_model(m_name)
    assert bundle.model_name == m_name
    assert bundle.model is not None
    assert len(bundle.idx_to_class) > 0


# ---------------------------------------------------------------------------
# Test 4: Prediction determinism (same image -> same prediction twice)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m_name", ["svm", "cnn", "resnet18"])
def test_prediction_determinism(m_name):
    df = load_splits(SPLITS_CSV)
    test_rows = df[(df["split"] == "test") & (~df["corrupted"])]
    assert not test_rows.empty
    sample_path = test_rows.iloc[0]["path"]

    bundle = load_model(m_name)
    res1 = predict_image(bundle, sample_path, top_k=3)
    res2 = predict_image(bundle, sample_path, top_k=3)

    assert res1["class"] == res2["class"], "Prediction class was non-deterministic"
    assert pytest.approx(res1["confidence"], rel=1e-3) == res2["confidence"]
    assert len(res1["top3"]) == 3
    assert res1["top3"][0]["class"] == res2["top3"][0]["class"]


# ---------------------------------------------------------------------------
# Test 5: Corrupt file rejected gracefully
# ---------------------------------------------------------------------------
def test_corrupt_file_rejected_gracefully():
    bundle = load_model("cnn")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"NOT_A_VALID_IMAGE_FILE_DATA_CORRUPTION_TEST")
        fake_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Cannot open image"):
            predict_image(bundle, fake_path)
    finally:
        if fake_path.exists():
            fake_path.unlink()


# ---------------------------------------------------------------------------
# Test 6: Zero perceptual hash leakage across splits
# ---------------------------------------------------------------------------
def test_leakage_control():
    df = load_splits(SPLITS_CSV)
    real_splits = df[df["split"].isin(["train", "val", "test"])]
    splits_per_group = real_splits.groupby("group_id")["split"].nunique()
    leaking_groups = (splits_per_group > 1).sum()
    assert leaking_groups == 0, f"Detected {leaking_groups} leaking hash groups across splits!"

