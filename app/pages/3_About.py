"""
3_About.py – Technical Methodology, System Architecture, & Citations

Sections:
  1. Project Overview & Research Objectives
  2. Pattern Recognition & Feature Engineering Methodology (HOG, HSV, LBP)
  3. Deep Learning Architectures (SmallCNN & ResNet18 Fine-Tuning)
  4. Data Integrity & Leakage Prevention (Perceptual Hash Clustering)
  5. Dataset Attribution & Licenses
  6. Real-World Limitations & Future Work
  7. Academic References
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="About – PlantVillage AI Suite", page_icon="📖", layout="wide")

st.title("📖 Technical Methodology & Architecture")
st.markdown(
    """
    This documentation details the computer vision pipeline, mathematical formulations, 
    neural network architectures, and evaluation methodology implemented for the PlantVillage Leaf Disease benchmark.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Section 1: Traditional Pattern Recognition & Feature Extraction
# ---------------------------------------------------------------------------
st.subheader("1. Hand-Crafted Feature Engineering Pipeline (SVM)")

st.markdown(
    """
    To establish a strong traditional machine learning baseline, each image is resized to $128 \\times 128$ 
    pixels and processed through three complementary feature extractors that capture shape, color, and texture:
    """
)

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("#### 📐 HOG (Shape & Edges)")
    st.markdown(
        """
        * **Method**: Histogram of Oriented Gradients (Dalal & Triggs, 2005) computed on grayscale imagery.
        * **Parameters**: 9 orientation bins, $8 \\times 8$ pixels per cell, $2 \\times 2$ cells per block with $L_2$-Hys normalization.
        * **Output**: **8,100 dimensions**.
        * **What it captures**: Lesion contours, vein structure, and leaf boundary gradients invariant to illumination shifts.
        """
    )

with col_f2:
    st.markdown("#### 🎨 HSV Color Distributions")
    st.markdown(
        """
        * **Method**: Cylindrical Hue-Saturation-Value color decomposition.
        * **Parameters**: Independent 1D histograms (Hue: 36 bins, Saturation: 32 bins, Value: 32 bins) + per-channel mean and standard deviation (6 values).
        * **Output**: **106 dimensions**.
        * **What it captures**: Diagnostic chlorosis (yellowing), necrosis (browning), rust pustules, and fungal darkening.
        """
    )

with col_f3:
    st.markdown("#### 🧵 Uniform LBP (Texture)")
    st.markdown(
        """
        * **Method**: Uniform Local Binary Patterns (Ojala et al., 2002).
        * **Parameters**: Circular radius $R = 3.0$, $P = 24$ sampling points, with uniform patterns ($U \\le 2$).
        * **Output**: **26 dimensions** ($P + 2$).
        * **What it captures**: Micro-surface roughness, powdery mold textures, and crusty scab tissue.
        """
    )

st.markdown(
    """
    **Dimensionality Reduction & Classification**:
    1. Concatenation into a unified feature vector: $8,100 + 106 + 26 = \\mathbf{8,232}$ dimensions.
    2. Zero-mean, unit-variance standardization via `StandardScaler`.
    3. Principal Component Analysis (`PCA`) retaining **95% of explained variance** (reduces 8,232 features to ~1,859 orthogonal components).
    4. Support Vector Classifier with **Radial Basis Function (RBF) Kernel**:
       $$K(x, x') = \\exp(-\\gamma \\|x - x'\\|^2)$$
       Calibrated via `CalibratedClassifierCV` for reliable probabilistic confidence output.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Deep Learning Architectures
# ---------------------------------------------------------------------------
st.subheader("2. Deep Learning Architectures")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    st.markdown("### SmallCNN (Custom Architecture from Scratch)")
    st.markdown(
        """
        Designed specifically for ultra-fast, lightweight CPU and edge deployment:
        * **Input Resolution**: $128 \\times 128 \\times 3$ RGB.
        * **Block 1**: Conv2d(3 → 32, $3 \\times 3$) + BatchNorm2d + ReLU + MaxPool2d($2 \\times 2$) → Output: $64 \\times 64 \\times 32$
        * **Block 2**: Conv2d(32 → 64, $3 \\times 3$) + BatchNorm2d + ReLU + MaxPool2d($2 \\times 2$) → Output: $32 \\times 32 \\times 64$
        * **Block 3**: Conv2d(64 → 128, $3 \\times 3$) + BatchNorm2d + ReLU + MaxPool2d($2 \\times 2$) → Output: $16 \\times 16 \\times 128$
        * **Block 4**: Conv2d(128 → 256, $3 \\times 3$) + BatchNorm2d + ReLU + MaxPool2d($2 \\times 2$) → Output: $8 \\times 8 \\times 256$
        * **Head**: `AdaptiveAvgPool2d(1)` (Global Average Pooling) → `Dropout(p=0.4)` → `Linear(256, num_classes)`.
        * **Complexity**: **391,209 parameters** | Model size: **1.5 MB** | Inference: **~10.7 ms/image**.
        """
    )

with col_dl2:
    st.markdown("### ResNet18 (Two-Stage Transfer Learning)")
    st.markdown(
        """
        Leverages deep representations pre-trained on ImageNet-1K (He et al., 2016):
        * **Input Resolution**: $160 \\times 160 \\times 3$ RGB (optimized for rapid CPU convergence).
        * **Residual Connections**: Skip connections solve vanishing gradients by learning residual mappings $\\mathcal{F}(x) = \\mathcal{H}(x) - x$.
        * **Stage 1 (Warmup Head)**:
          * Entire convolutional backbone frozen (`requires_grad = False`).
          * Replaced classifier head: `Linear(512, num_classes)`.
          * Trained for 2-3 epochs at learning rate $\\eta = 1 \\times 10^{-3}$ using Adam.
        * **Stage 2 (Fine-Tuning)**:
          * Unfreeze `layer4` (final residual block) + classifier head (8.4M trainable parameters).
          * Fine-tune with low learning rate $\\eta = 1 \\times 10^{-4}$ to refine high-level domain-specific visual features.
        * **Complexity**: **11.2M parameters** | Model size: **42.7 MB** | Accuracy: **99.74%**.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Leakage Control & Perceptual Hashing
# ---------------------------------------------------------------------------
st.subheader("3. Perceptual Hashing & Leakage Prevention")
st.markdown(
    """
    Standard random train/test splits on datasets like PlantVillage cause **silent data leakage**. 
    Photographers captured multiple near-identical frames of the same infected leaf from slightly varying camera angles. 
    A random split places copies of the same leaf in both train and test, yielding inflated test scores that fail in the field.

    **Our Leakage-Safe Strategy**:
    1. Compute a 64-bit perceptual hash (`imagehash.phash`) for all non-corrupted images.
    2. Cluster images within Hamming distance threshold $\\le 10$ into identical **hash groups** using Disjoint-Set Union (Union-Find).
    3. Partition the dataset across hash groups using two successive rounds of `GroupShuffleSplit`:
       * Round 1: Separate 15% holdout test group.
       * Round 2: Split remaining groups into 70% Train and 15% Validation.
    4. **Verification**: Rigorously proved that zero hash clusters span across more than one split partition (`data/splits.csv`).
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Section 4: Limitations & Future Scope
# ---------------------------------------------------------------------------
st.subheader("4. Limitations & Real-World Considerations")
col_l1, col_l2 = st.columns(2)

with col_l1:
    st.markdown("#### Known Limitations")
    st.markdown(
        """
        * **Laboratory Condition Bias**: PlantVillage images are captured on uniform, neutral backgrounds under controlled lighting. 
          Real agricultural field images have cluttered soil, weeds, shadows, and multiple overlapping leaves (domain shift).
        * **Single-Leaf Assumption**: The model expects a centered photograph of a single leaf rather than whole-crop field canopy.
        * **Co-infection**: Diseases appearing simultaneously (e.g. bacterial spot alongside fungal rust) may compete for top probability.
        """
    )

with col_l2:
    st.markdown("#### Future Work")
    st.markdown(
        """
        * **PlantDoc Field Domain Adaptation**: Benchmark and adapt models on in-the-wild datasets (PlantDoc) to measure and mitigate domain shift.
        * **Object Detection (YOLOv8/v11)**: Transition from full-image classification to bounding box detection and segmentation of individual disease lesions on entire plants.
        * **Offline Edge Deployment**: Quantize SmallCNN to INT8 (ONNX / TFLite) for sub-5ms offline deployment on farmer smartphones without cellular data connectivity.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 5: References
# ---------------------------------------------------------------------------
st.subheader("5. Key References")
st.markdown(
    """
    1. **Hughes, D., & Salathé, M.** (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv preprint arXiv:1511.08060.
    2. **Dalal, N., & Triggs, B.** (2005). *Histograms of oriented gradients for human detection.* IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR).
    3. **Ojala, T., Pietikäinen, M., & Mäenpää, T.** (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns.* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
    4. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep residual learning for image recognition.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
    5. **Singh, D., et al.** (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection in the Wild.* ACM CoNEXT.
    """
)

