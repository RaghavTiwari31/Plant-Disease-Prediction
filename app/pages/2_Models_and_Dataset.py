"""
2_Models_and_Dataset.py – Comprehensive Evaluation & Dataset Exploration Dashboard

Features:
  - Side-by-side performance comparison: Accuracy, Macro-F1, Latency, Size
  - Interactive tabs for Confusion Matrices (SVM, SmallCNN, ResNet18)
  - Training & validation learning curves
  - Qualitative sample predictions (Correct vs Misclassified)
  - PlantVillage Dataset distribution, per-split breakdown, and class sample gallery
"""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Models & Dataset Hub", page_icon="📈", layout="wide")

CHARTS_DIR = PROJECT_ROOT / "data" / "charts"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_CSV = DATA_DIR / "splits.csv"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_splits_df():
    if SPLITS_CSV.exists():
        return pd.read_csv(SPLITS_CSV)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("📈 Models & Dataset Evaluation Hub")
st.markdown(
    """
    Comprehensive experimental evaluation comparing **Traditional Feature Engineering (HOG + Color + LBP with RBF-SVM)**, 
    **Custom Lightweight CNN (SmallCNN)**, and **ImageNet Fine-Tuned ResNet18** on the PlantVillage dataset.
    """
)

tab_overview, tab_matrices, tab_curves, tab_samples, tab_data = st.tabs(
    [
        "🏆 Model Comparison",
        "🎯 Confusion Matrices",
        "📉 Training Curves",
        "🔍 Qualitative Error Analysis",
        "📁 Dataset & Distributions",
    ]
)

metrics_dict = load_metrics()
splits_df = load_splits_df()

# ---------------------------------------------------------------------------
# Tab 1: Model Comparison
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Model Performance Summary")

    if metrics_dict:
        rows = []
        for name, data in metrics_dict.items():
            test_m = data.get("test", {})
            val_m = data.get("val", {})
            train_time = data.get("train_time_s", 0)
            train_str = f"{train_time / 60:.1f} min" if train_time > 60 else f"{train_time:.1f} s"

            rows.append(
                {
                    "Model": name,
                    "Type": "Hand-crafted Features" if name == "SVM" else "Deep Learning",
                    "Test Accuracy": f"{test_m.get('accuracy', 0.0) * 100:.2f}%",
                    "Test Macro-F1": f"{test_m.get('macro_f1', 0.0):.4f}",
                    "Test Weighted-F1": f"{test_m.get('weighted_f1', 0.0):.4f}",
                    "Macro Precision": f"{test_m.get('macro_precision', 0.0):.4f}",
                    "Macro Recall": f"{test_m.get('macro_recall', 0.0):.4f}",
                    "Latency / Image": f"{data.get('inference_time_per_image_ms', 0):.2f} ms",
                    "Disk Size": f"{data.get('model_size_mb', 0):.1f} MB",
                    "Train Time": train_str,
                }
            )

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        col_c1, col_c2 = st.columns([1.2, 1])
        with col_c1:
            comp_chart = CHARTS_DIR / "model_comparison.png"
            if comp_chart.exists():
                st.image(str(comp_chart), caption="Test Accuracy and Macro-F1 Comparison", width="stretch")
            else:
                st.info("Comparison chart will appear here once generated.")

        with col_c2:
            st.markdown(
                """
                ### Key Findings & Analysis
                * **ResNet18 Dominance**: Transfer learning with ImageNet weights achieved an exceptional **99.74% test accuracy** and **0.9967 macro-F1** through hierarchical multi-scale feature reuse.
                * **SmallCNN Efficiency**: The 4-block custom CNN reached **96.17% test accuracy** from scratch with only **391K parameters** (1.5 MB model file), making it ideal for edge devices and mobile deployment.
                * **SVM Baseline**: The traditional pattern recognition pipeline (HOG shape gradients + HSV color distributions + LBP texture) achieved **87.46% test accuracy**, demonstrating strong handcrafted discriminatory power without deep neural networks.
                """
            )
    else:
        st.warning("No metrics data available.")


# ---------------------------------------------------------------------------
# Tab 2: Confusion Matrices
# ---------------------------------------------------------------------------
with tab_matrices:
    st.subheader("Normalized Test Confusion Matrices")
    st.markdown("Rows denote True agricultural classes; columns denote Model Predictions.")

    m_tab_rn, m_tab_cnn, m_tab_svm = st.tabs(["ResNet18 (99.7%)", "SmallCNN (96.2%)", "SVM Baseline (87.5%)"])

    with m_tab_rn:
        rn_cm = CHARTS_DIR / "resnet18_–_test_confusion_matrix.png"
        if rn_cm.exists():
            st.image(str(rn_cm), caption="ResNet18 Row-Normalized Confusion Matrix (1,515 test images)", width="stretch")
        else:
            st.info("ResNet18 confusion matrix image not found.")

    with m_tab_cnn:
        cnn_cm = CHARTS_DIR / "smallcnn_–_test_confusion_matrix.png"
        if cnn_cm.exists():
            st.image(str(cnn_cm), caption="SmallCNN Row-Normalized Confusion Matrix (1,515 test images)", width="stretch")
        else:
            st.info("SmallCNN confusion matrix image not found.")

    with m_tab_svm:
        svm_cm = CHARTS_DIR / "svm_–_test_confusion_matrix.png"
        if svm_cm.exists():
            st.image(str(svm_cm), caption="SVM Row-Normalized Confusion Matrix (1,515 test images)", width="stretch")
        else:
            st.info("SVM confusion matrix image not found.")


# ---------------------------------------------------------------------------
# Tab 3: Training Curves
# ---------------------------------------------------------------------------
with tab_curves:
    st.subheader("Training & Validation Curves")
    st.markdown("Optimization progress across training epochs showing CrossEntropy loss and classification accuracy.")

    curve_c1, curve_c2 = st.columns(2)

    with curve_c1:
        cnn_curve = CHARTS_DIR / "smallcnn_training_curves.png"
        if cnn_curve.exists():
            st.image(str(cnn_curve), caption="SmallCNN: Loss and Accuracy Learning Curves", width="stretch")
            st.caption("Custom CNN trained from scratch (Adam, lr=1e-3, ReduceLROnPlateau).")

    with curve_c2:
        rn_curve = CHARTS_DIR / "resnet18_training_curves.png"
        if rn_curve.exists():
            st.image(str(rn_curve), caption="ResNet18: Two-Stage Fine-Tuning Curves", width="stretch")
            st.caption("Stage 1 (Head only, lr=1e-3) followed by Stage 2 (Layer4 + Head, lr=1e-4).")


# ---------------------------------------------------------------------------
# Tab 4: Qualitative Error Analysis
# ---------------------------------------------------------------------------
with tab_samples:
    st.subheader("Qualitative Analysis: Correct vs Misclassified Predictions")
    st.markdown("Sample images from the holdout test set with predicted class labels and confidence scores.")

    sample_sub_rn, sample_sub_cnn, sample_sub_svm = st.tabs(["ResNet18 Samples", "SmallCNN Samples", "SVM Samples"])

    with sample_sub_rn:
        rn_samp = CHARTS_DIR / "resnet18_sample_predictions.png"
        if rn_samp.exists():
            st.image(str(rn_samp), caption="ResNet18 Test Sample Predictions", width="stretch")

    with sample_sub_cnn:
        cnn_samp = CHARTS_DIR / "smallcnn_sample_predictions.png"
        if cnn_samp.exists():
            st.image(str(cnn_samp), caption="SmallCNN Test Sample Predictions", width="stretch")

    with sample_sub_svm:
        svm_samp = CHARTS_DIR / "svm_sample_predictions.png"
        if svm_samp.exists():
            st.image(str(svm_samp), caption="SVM Test Sample Predictions", width="stretch")


# ---------------------------------------------------------------------------
# Tab 5: Dataset & Distributions
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("PlantVillage Dataset Architecture & Statistics")

    if not splits_df.empty:
        col_st1, col_st2 = st.columns([1, 1.2])

        with col_st1:
            st.markdown("#### Split Breakdown by Class")
            clean_df = splits_df[~splits_df["corrupted"]]
            breakdown = (
                clean_df.groupby(["class_name", "split"])
                .size()
                .unstack(fill_value=0)[["train", "val", "test"]]
            )
            breakdown["total"] = breakdown.sum(axis=1)
            st.dataframe(breakdown, width="stretch")

            st.markdown(
                f"""
                * **Total Images**: {len(clean_df):,}
                * **Train Set**: {len(clean_df[clean_df['split']=='train']):,} images (70%)
                * **Validation Set**: {len(clean_df[clean_df['split']=='val']):,} images (15%)
                * **Test Set**: {len(clean_df[clean_df['split']=='test']):,} images (15%)
                * **Corrupted Files Rejected**: {splits_df['corrupted'].sum()}
                """
            )

        with col_st2:
            st.markdown("#### Class Distribution Chart")
            class_chart = CHARTS_DIR / "class_distribution.png"
            if class_chart.exists():
                st.image(str(class_chart), caption="Class Distribution across Splits", width="stretch")
            else:
                # Dynamic horizontal bar chart
                counts = clean_df["class_name"].value_counts()
                st.bar_chart(counts)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Sample Grid: One Image per Target Class")
        sample_grid = CHARTS_DIR / "sample_grid.png"
        if sample_grid.exists():
            st.image(str(sample_grid), caption="PlantVillage Sample Grid (One Image per Category)", width="stretch")
    else:
        st.info("Splits dataset not loaded.")

