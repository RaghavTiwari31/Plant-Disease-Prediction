"""
1_Detect.py – Interactive Leaf Disease Detector

Features:
  - File uploader (with corrupt file / non-image safety check)
  - Pre-loaded test sample selector (handy for live lab demo)
  - Model selection: SVM, SmallCNN, ResNet18, or Compare All 3 Models
  - Single-image inference via src.predict (cached via @st.cache_resource)
  - Predicted class, confidence gauge, and top-3 bar chart
  - Feature extraction visualization: HOG gradient map & HSV channel decomposition
  - Comprehensive disease information card (symptoms, pathogen, prevention, educational disclaimer)
"""

import json
from pathlib import Path
import sys
from typing import Dict, Optional

# Ensure project root is in sys.path BEFORE src imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from skimage.color import rgb2gray
from skimage.feature import hog
import streamlit as st

from src.predict import load_model, predict_image

# ---------------------------------------------------------------------------
# Paths & Settings
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Detect – Plant Disease AI", page_icon="🔍", layout="wide")

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_CSV = DATA_DIR / "splits.csv"
DISEASE_INFO_PATH = PROJECT_ROOT / "app" / "disease_info.json"


# ---------------------------------------------------------------------------
# Cached Resources & Data
# ---------------------------------------------------------------------------
@st.cache_resource
def get_cached_model(model_name: str):
    """Load model once and cache in memory for ultra-fast inference."""
    return load_model(model_name)


@st.cache_data
def get_disease_info() -> Dict:
    if DISEASE_INFO_PATH.exists():
        with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_sample_test_images() -> pd.DataFrame:
    """Load up to 2 sample images per class from the test split for demo."""
    if SPLITS_CSV.exists():
        df = pd.read_csv(SPLITS_CSV)
        test_df = df[(df["split"] == "test") & (~df["corrupted"])].copy()
        samples = test_df.groupby("class_name").head(2).reset_index(drop=True)
        return samples[["path", "class_name"]]
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🔍 Leaf Disease Diagnosis")
st.markdown("Upload a leaf image or pick a test sample to classify disease status and view treatment guidelines.")

disease_db = get_disease_info()
samples_df = get_sample_test_images()

# Top control row
col_ctrl1, col_ctrl2 = st.columns([1, 1])

with col_ctrl1:
    input_mode = st.radio(
        "Choose Image Input Method:",
        ["📁 Upload New Image", "🧪 Pick from Pre-loaded Test Samples"],
        horizontal=True,
    )

with col_ctrl2:
    model_choice = st.selectbox(
        "Select Classification Model:",
        [
            "ResNet18 (Transfer Learning - 99.7% Test Acc)",
            "SmallCNN (Custom CNN - 96.2% Test Acc)",
            "SVM (HOG + Color + LBP - 87.5% Test Acc)",
            "⚡ Compare All 3 Models Side-by-Side",
        ],
        index=0,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Image Acquisition
# ---------------------------------------------------------------------------
image_to_process: Optional[Image.Image] = None
image_label_hint: str = "Uploaded Image"
selected_file_path: Optional[str] = None

col_img, col_pred = st.columns([1, 1.2])

with col_img:
    st.subheader("📷 Leaf Image")

    if input_mode == "📁 Upload New Image":
        uploaded_file = st.file_uploader(
            "Upload leaf photograph (JPG, PNG, JPEG):",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Images of a single leaf on a clean background yield optimal results.",
        )
        if uploaded_file is not None:
            try:
                # Security and validity check
                img_candidate = Image.open(uploaded_file)
                img_candidate.verify()
                # Re-open after verify
                uploaded_file.seek(0)
                image_to_process = Image.open(uploaded_file).convert("RGB")
                image_label_hint = uploaded_file.name
                st.image(image_to_process, caption=f"Uploaded: {image_label_hint}", width="stretch")
            except (UnidentifiedImageError, Exception) as exc:
                st.error(f"❌ Invalid or corrupted image file: {exc}")
                image_to_process = None
    else:
        if not samples_df.empty:
            sample_options = [
                f"{row['class_name']} — {Path(row['path']).name}"
                for _, row in samples_df.iterrows()
            ]
            selected_option = st.selectbox("Select a verified test sample:", sample_options)
            chosen_row = samples_df.iloc[sample_options.index(selected_option)]
            selected_file_path = chosen_row["path"]
            image_label_hint = f"Ground Truth: {chosen_row['class_name']}"

            try:
                image_to_process = Image.open(selected_file_path).convert("RGB")
                st.image(image_to_process, caption=image_label_hint, width="stretch")
            except Exception as e:
                st.error(f"Failed to load sample image: {e}")
        else:
            st.warning("No test samples found in splits.csv.")

    # Lab Demo Feature: Feature Extraction Visualization
    if image_to_process is not None:
        with st.expander("🔬 Lab Demo: View Hand-Crafted Feature Representations (HOG & Color)"):
            arr = np.array(image_to_process.resize((128, 128)))
            gray = rgb2gray(arr)
            _, hog_image = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
                channel_axis=None,
            )

            fig_feat, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
            ax1.imshow(gray, cmap="gray")
            ax1.set_title("Grayscale (128×128)", fontsize=9)
            ax1.axis("off")

            ax2.imshow(hog_image, cmap="magma")
            ax2.set_title("HOG Gradient Map", fontsize=9)
            ax2.axis("off")

            st.pyplot(fig_feat, width="stretch")
            plt.close(fig_feat)


# ---------------------------------------------------------------------------
# Prediction Execution
# ---------------------------------------------------------------------------
with col_pred:
    st.subheader("🎯 Diagnosis & Predictions")

    if image_to_process is None:
        st.info("👈 Please upload a leaf image or pick a test sample to run the diagnosis.")
    else:
        # Save temp image if uploaded
        temp_img_path = PROJECT_ROOT / "data" / "temp_inference.jpg"
        temp_img_path.parent.mkdir(parents=True, exist_ok=True)
        image_to_process.save(temp_img_path)

        # Decide which model(s) to run
        if "Compare All 3" in model_choice:
            st.markdown("### Model Comparison on This Image")

            comp_cols = st.columns(3)
            results_dict = {}

            for idx, m_key in enumerate(["resnet18", "cnn", "svm"]):
                with comp_cols[idx]:
                    bundle = get_cached_model(m_key)
                    res = predict_image(bundle, temp_img_path, top_k=3)
                    results_dict[m_key] = res

                    m_title = (
                        "ResNet18" if m_key == "resnet18"
                        else "SmallCNN" if m_key == "cnn"
                        else "SVM (HOG)"
                    )
                    st.markdown(f"**{m_title}**")
                    is_healthy = "healthy" in res["class"].lower()
                    badge_color = "🟢" if is_healthy else "🔴"
                    clean_name = res["class"].replace("___", " - ").replace("_", " ")

                    st.markdown(f"{badge_color} **{clean_name}**")
                    st.metric("Confidence", f"{res['confidence'] * 100:.1f}%")
                    st.caption(f"Latency: {res['inference_ms']:.1f} ms")

            # Primary display uses ResNet18
            primary_result = results_dict["resnet18"]

        else:
            m_key = "resnet18" if "ResNet18" in model_choice else "cnn" if "SmallCNN" in model_choice else "svm"
            bundle = get_cached_model(m_key)
            primary_result = predict_image(bundle, temp_img_path, top_k=3)

            pred_class = primary_result["class"]
            conf = primary_result["confidence"]
            clean_name = pred_class.replace("___", " - ").replace("_", " ")

            is_healthy = "healthy" in pred_class.lower()

            if is_healthy:
                st.success(f"### 🟢 Diagnosis: {clean_name}")
            else:
                st.error(f"### 🔴 Diagnosis: {clean_name}")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Model Confidence", f"{conf * 100:.2f}%")
            with m2:
                st.metric("Inference Time", f"{primary_result['inference_ms']:.1f} ms")

            # Top-3 Probabilities Bar Chart
            st.markdown("#### Top-3 Class Probability Distribution")
            top3_data = pd.DataFrame(primary_result["top3"])
            top3_data["clean_label"] = top3_data["class"].apply(
                lambda x: x.replace("___", " - ").replace("_", " ")
            )
            top3_data["percentage"] = top3_data["confidence"] * 100

            fig_bar, ax = plt.subplots(figsize=(6, 2.3))
            bars = ax.barh(
                top3_data["clean_label"],
                top3_data["percentage"],
                color="#4c72b0" if not is_healthy else "#55a868",
            )
            ax.set_xlim(0, 105)
            ax.set_xlabel("Confidence (%)", fontsize=8)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            for bar in bars:
                ax.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}%",
                    va="center",
                    fontsize=8,
                )
            plt.tight_layout()
            st.pyplot(fig_bar, width="stretch")
            plt.close(fig_bar)

        # -----------------------------------------------------------------------
        # Disease Pathology Card
        # -----------------------------------------------------------------------
        pred_key = primary_result["class"]
        info = disease_db.get(pred_key, {})

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 Agricultural Pathology & Management Card")

        if info:
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.markdown(f"**Target Crop**: {info.get('plant', 'N/A')}")
                st.markdown(f"**Pathogen**: *{info.get('pathogen', 'N/A')}*")
                st.markdown(f"**Severity**: {info.get('severity', 'N/A')}")

            with c_p2:
                st.markdown(f"**Condition Status**: {info.get('status', 'N/A')}")
                st.markdown(f"**Disease Category**: {info.get('disease_name', 'N/A')}")

            st.markdown("#### Primary Symptoms:")
            for s in info.get("symptoms", []):
                st.markdown(f"- {s}")

            st.markdown("#### Recommended Prevention & Treatment:")
            for p in info.get("prevention_and_treatment", []):
                st.markdown(f"- {p}")

            st.warning(
                f"⚠️ **Educational Disclaimer**: {info.get('disclaimer', 'This application is for educational purposes.')}"
            )
        else:
            st.info(f"Class: {pred_class}. General sanitation and plant scouting recommended.")

