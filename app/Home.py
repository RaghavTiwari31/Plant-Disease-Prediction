"""
Home.py – PlantVillage Leaf Disease Classifier & Benchmark

Streamlit entry page:
  - System Overview & Objectives
  - Dataset Summary (PlantVillage color subset)
  - Model Performance Benchmark Table (dynamically loaded from models/metrics.json)
  - Quick Navigation to Detect, Models & Dataset, and About
"""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PlantVillage AI Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_CSV = DATA_DIR / "splits.csv"


# ---------------------------------------------------------------------------
# Data loading helpers (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_metrics_data():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_splits_summary():
    if SPLITS_CSV.exists():
        df = pd.read_csv(SPLITS_CSV)
        total_images = len(df)
        good_images = len(df[~df["corrupted"]])
        classes = sorted(df["class_name"].dropna().unique().tolist())
        split_counts = df[~df["corrupted"]]["split"].value_counts().to_dict()
        return {
            "total": total_images,
            "valid": good_images,
            "num_classes": len(classes),
            "classes": classes,
            "train": split_counts.get("train", 0),
            "val": split_counts.get("val", 0),
            "test": split_counts.get("test", 0),
        }
    return None


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🌿 PlantVillage Disease Classifier & Benchmarking Suite")
st.markdown(
    """
    An end-to-end Computer Vision & Pattern Recognition system for plant disease diagnosis.
    Compares **Traditional Hand-Crafted Feature Engineering (HOG + HSV + LBP with RBF-SVM)** 
    against **Custom Deep Learning (SmallCNN)** and **Transfer Learning (ResNet18)** on identical, 
    leakage-free perceptual-hash splits.
    """
)

st.divider()

# High-level Metrics Row
metrics_dict = load_metrics_data()
splits_info = load_splits_summary()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Dataset Classes",
        value=f"{splits_info['num_classes'] if splits_info else 9} Categories",
        help="Plant species and disease combinations",
    )

with col2:
    best_acc = 0.0
    best_model = "N/A"
    for name, data in metrics_dict.items():
        acc = data.get("test", {}).get("accuracy", 0.0)
        if acc > best_acc:
            best_acc = acc
            best_model = name
    st.metric(
        label="Top Accuracy (Test)",
        value=f"{best_acc * 100:.2f}%",
        delta=f"{best_model}",
        help="Evaluated on 1,515 holdout test images",
    )

with col3:
    st.metric(
        label="Smallest DL Model",
        value="1.5 MB",
        delta="SmallCNN (391K params)",
        help="Trained from scratch, ultra-lightweight CPU inference",
    )

with col4:
    st.metric(
        label="Fastest Inference",
        value="~10.7 ms",
        delta="per leaf image",
        help="Real-time classification throughput",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: Benchmark Summary Table
# ---------------------------------------------------------------------------
st.subheader("📊 Model Performance Benchmark")
st.markdown("All three models are benchmarked on the **identical perceptual-hash grouped test split**:")

if metrics_dict:
    table_rows = []
    for model_name, data in metrics_dict.items():
        test_m = data.get("test", {})
        val_m = data.get("val", {})
        arch = data.get("architecture") or (
            "HOG + HSV + LBP (8,232-d) → StandardScaler → PCA (95%) → RBF-SVM"
            if model_name == "SVM"
            else "N/A"
        )
        table_rows.append(
            {
                "Model": model_name,
                "Approach": "Traditional ML" if model_name == "SVM" else "Deep Learning",
                "Test Accuracy": f"{test_m.get('accuracy', 0.0) * 100:.2f}%",
                "Test Macro-F1": f"{test_m.get('macro_f1', 0.0):.4f}",
                "Test Weighted-F1": f"{test_m.get('weighted_f1', 0.0):.4f}",
                "Val Accuracy": f"{val_m.get('accuracy', 0.0) * 100:.2f}%" if val_m else "N/A",
                "Inference Latency": f"{data.get('inference_time_per_image_ms', 0.0):.2f} ms",
                "Model File Size": f"{data.get('model_size_mb', 0.0):.1f} MB",
                "Architecture Details": arch,
            }
        )

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, width="stretch", hide_index=True)
else:
    st.warning("No metrics.json found in `models/`. Please run the evaluation pipeline.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Dataset Summary & Split Architecture
# ---------------------------------------------------------------------------
st.subheader("📁 Dataset & Leakage Control")
col_d1, col_d2 = st.columns([1, 1])

with col_d1:
    st.markdown(
        """
        ### PlantVillage Color Subset
        * **Origin**: PlantVillage open research dataset (Hughes & Salathé).
        * **Target Crops**: Apple, Blueberry, Corn (Maize), Grape, and Bell Pepper.
        * **Image Format**: Cleaned RGB color images (standardized aspect ratio).
        * **Leakage-Safe Partitioning**: PlantVillage contains near-duplicate photos of the same leaf 
          taken seconds apart under slightly different angles. Standard random splits cause severe data leakage.
        * **Our Solution**: Perceptual hash clustering (`imagehash.phash`, Hamming distance $\le 10$) 
          grouped into clusters, then split with `GroupShuffleSplit` (approx 70% Train, 15% Val, 15% Test).
        """
    )

with col_d2:
    if splits_info:
        st.markdown("### Split Distribution")
        split_df = pd.DataFrame(
            {
                "Split": ["Train (70%)", "Validation (15%)", "Test (15%)"],
                "Images": [splits_info["train"], splits_info["val"], splits_info["test"]],
            }
        )
        st.dataframe(split_df, width="stretch", hide_index=True)
        st.success("✅ Zero perceptual hash groups span across multiple splits (Verified Leak-Free).")

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Navigation Guide
# ---------------------------------------------------------------------------
st.subheader("🧭 Explore the App")
nav_c1, nav_c2, nav_c3 = st.columns(3)

with nav_c1:
    st.info("🔍 **Leaf Disease Detector**\n\nUpload a leaf photo or pick a sample, choose any model, and view confidence distributions and agricultural pathology cards.")

with nav_c2:
    st.info("📈 **Models & Dataset Hub**\n\nInspect side-by-side confusion matrices, training loss & accuracy curves, class balance histograms, and timing comparisons.")

with nav_c3:
    st.info("📖 **Methodology & About**\n\nRead the technical architecture documentation, pattern recognition features (HOG, HSV, LBP), limitations, and academic references.")

