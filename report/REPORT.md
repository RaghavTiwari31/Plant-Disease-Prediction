# Plant Disease Diagnosis via Computer Vision: Comparative Study of Hand-Crafted Feature Engineering, Custom Lightweight CNNs, and Deep Transfer Learning

**Course**: Pattern Recognition & Computer Vision  
**Date**: November 2026  
**Artifact Directory**: `report/REPORT.md`  

---

## Abstract

Automated detection and classification of agricultural foliar diseases is critical for timely crop protection, global food security, and scalable precision farming. In this study, we present a rigorous, end-to-end comparative benchmark evaluating three distinct paradigms on the PlantVillage dataset across nine target crop-disease categories:
1. **Traditional Feature Engineering**: Histogram of Oriented Gradients (HOG, 8,100-d), Cylindrical HSV Color Histograms & Moments (106-d), and Uniform Local Binary Patterns (LBP, 26-d) combined with PCA and an RBF Support Vector Machine (SVM).
2. **Custom Lightweight Deep Learning**: A 4-block Convolutional Neural Network (**SmallCNN**, 391K parameters, 1.5 MB) trained entirely from scratch.
3. **Deep Transfer Learning**: An ImageNet-pretrained **ResNet18** model adapted via a two-stage fine-tuning schedule.

Crucially, we address the pervasive issue of **silent data leakage** in plant pathology datasets by developing an automated perceptual-hash (`imagehash.phash`) clustering pipeline with Disjoint-Set Union (Union-Find) and `GroupShuffleSplit`, guaranteeing that near-duplicate leaf photographs cannot cross-contaminate training and test splits. On the identical, verified holdout test set (1,515 images), **ResNet18** achieved an outstanding **99.74% accuracy** (0.9967 macro-F1), **SmallCNN** attained **96.17% accuracy** (0.9630 macro-F1) at just 1.5 MB disk size, and the **SVM baseline** achieved **87.46% accuracy** (0.8665 macro-F1). We integrate these models into an offline, interactive Streamlit diagnostic application featuring pathology decision cards, top-$k$ confidence distributions, feature visualizations, and a verified test suite.

---

## 1. Introduction and Objectives

Plant diseases cause substantial yield losses in staple and cash crops globally, threatening subsistence farmers and large-scale agricultural economies alike. Early visual diagnosis has historically required trained agronomists or plant pathologists, a resource unavailable in many remote farming regions.

The primary objectives of this project are:
1. **Establish a Leak-Free Benchmark**: Prevent inflated test performance caused by near-duplicate frames through perceptual-hash grouping.
2. **Contrast Machine Learning Paradigms**: Directly compare classical computer vision feature extraction against deep neural networks on the exact same test images.
3. **Analyze Edge Feasibility**: Evaluate models not only on accuracy and F1-score, but also on model size, parameter counts, and inference latency on standard CPU hardware.
4. **Deploy an Interactive Tool**: Deliver a production-ready, fully offline multi-page Streamlit application with diagnostic pathology cards.

---

## 2. Dataset and Leakage Prevention

### 2.1 PlantVillage Dataset Overview
We utilized the color subset of the benchmark PlantVillage dataset (Hughes & Salathé, 2015), comprising high-resolution photographs of individual leaves captured on neutral, uniform backgrounds. Our study investigates nine critical target classes across five major crops:

| Target Class | Crop | Status | Pathogen / Condition |
|---|---|:---:|---|
| `Apple___Apple_scab` | Apple | Diseased | *Venturia inaequalis* (Fungus) |
| `Apple___healthy` | Apple | Healthy | None |
| `Blueberry___healthy` | Blueberry | Healthy | None |
| `Corn_(maize)___Common_rust_` | Corn | Diseased | *Puccinia sorghi* (Fungus) |
| `Corn_(maize)___healthy` | Corn | Healthy | None |
| `Grape___Black_rot` | Grape | Diseased | *Guignardia bidwellii* (Fungus) |
| `Grape___healthy` | Grape | Healthy | None |
| `Pepper,_bell___Bacterial_spot` | Bell Pepper | Diseased | *Xanthomonas campestris* (Bacterium) |
| `Pepper,_bell___healthy` | Bell Pepper | Healthy | None |

### 2.2 The Silent Data Leakage Problem
In agricultural computer vision datasets, field collectors often capture consecutive photos of the same physical leaf from slightly differing angles or varying zoom levels. If images are split randomly into train/val/test partitions, these **near-duplicate leaf pairs** end up in both training and test sets. This creates severe optimistic bias—the model memorizes specific leaf contours rather than generalizable disease pathology.

### 2.3 Perceptual Hash Clustering Pipeline
To eliminate this flaw, we implemented an algorithmic leakage control pipeline in `src/data.py`:
1. **Pillow Verification**: Every file is opened and checked for header corruption via `Image.verify()`.
2. **Perceptual Hashing**: We compute a 64-bit perceptual hash (`imagehash.phash`, $8 \times 8$ frequency DCT grid) for each image. Perceptual hashes are invariant to minor rotations, slight exposure changes, and JPEG compression artifacts.
3. **Union-Find Clustering**: Images with a Hamming distance $\le 10$ are clustered into identical `group_id` sets using a sliding-window Union-Find algorithm.
4. **Grouped Stratified Splitting**: Two rounds of `GroupShuffleSplit` assign entire hash clusters to:
   - **Train**: ~70% (7,072 images)
   - **Validation**: ~15% (1,622 images)
   - **Test**: ~15% (1,515 images)
5. **Leakage Verification**: An automated assertion verifies that **zero** hash groups appear in more than one partition:
   $$\forall g \in \mathcal{G}, \quad |\{\text{split}(x) \mid x \in g\}| = 1$$

---

## 3. Preprocessing and Feature Extraction

### 3.1 Hand-Crafted Feature Vector (SVM Pipeline)
For the traditional pattern recognition pipeline in `src/features.py`, every image is resized to $128 \times 128$ pixels. We extract three complementary descriptors:

1. **Histogram of Oriented Gradients (HOG)**:
   - Input: Grayscale image ($128 \times 128$).
   - Cell size: $8 \times 8$ pixels (16 cells horizontal $\times$ 16 cells vertical).
   - Block size: $2 \times 2$ cells (with 50% block overlap).
   - Orientation bins: 9 unsigned gradient directions ($0^\circ$ to $180^\circ$).
   - Feature Dimension: $15 \times 15 \text{ blocks} \times 4 \text{ cells/block} \times 9 \text{ bins} = \mathbf{8,100}$.
   - **Role**: Encodes edge orientations, leaf vein geometry, and lesion borders.

2. **Cylindrical HSV Color Histograms & Moments**:
   - Input: RGB image converted to Hue-Saturation-Value.
   - Histograms: Hue (36 bins), Saturation (32 bins), Value (32 bins) normalized independently ($36 + 32 + 32 = 100$ bins).
   - First and second order color moments: Mean and standard deviation per channel ($\mu_H, \sigma_H, \mu_S, \sigma_S, \mu_V, \sigma_V = 6$ values).
   - Feature Dimension: $100 + 6 = \mathbf{106}$.
   - **Role**: Color is a definitive agricultural diagnostic signal distinguishing chlorotic yellow halos, necrotic brown lesions, and rust pustules.

3. **Uniform Local Binary Patterns (LBP)**:
   - Input: Grayscale image.
   - Neighborhood: $P = 24$ points on a circular radius $R = 3.0$.
   - Uniform patterns ($U \le 2$ bitwise transitions) normalized into a histogram.
   - Feature Dimension: $P + 2 = \mathbf{26}$.
   - **Role**: Captures fine-grained leaf surface micro-textures (e.g., powdery mildew vs. rough scab tissue).

**Total Feature Dimension**: $8,100 + 106 + 26 = \mathbf{8,232}$ features.

### 3.2 Standardization & Principal Component Analysis (PCA)
Features are centered and scaled to unit variance using `StandardScaler`. Because $8,232$ dimensions would cause high computational overhead for kernel SVM fitting, we apply `PCA` retaining **95% of total variance**, reducing the feature space to **1,859 orthogonal components**.

---

## 4. Model Architectures and Training Protocols

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Machine Learning Pipeline                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [Leaf Image] ──┬──> [HOG + HSV + LBP] ──> [Scaler+PCA] ──> [RBF-SVM]  │
│                 │                                                      │
│                 ├──> [Resize 128x128] ──> [4-Block SmallCNN]           │
│                 │                                                      │
│                 └──> [Resize 160x160] ──> [Pretrained ResNet18]        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Traditional SVM Baseline
- **Kernel**: Radial Basis Function (RBF), $K(x, x') = \exp(-\gamma \|x - x'\|^2)$.
- **Hyperparameter Search**: 3-fold stratified `GridSearchCV` over $C \in \{0.1, 1.0, 10.0, 100.0\}$ and $\gamma \in \{\text{'scale'}, \text{'auto'}, 10^{-3}, 10^{-4}\}$ on a 150-image/class subsample.
- **Optimal Parameters Found**: $C = 100.0$, $\gamma = 0.0001$.
- **Probability Calibration**: Calibrated via `CalibratedClassifierCV(ensemble=False)` using Platt scaling to output well-calibrated class probabilities.

### 4.2 Custom SmallCNN (Trained from Scratch)
Built to evaluate what a compact, purpose-built architecture can achieve without transfer learning:
- **Architecture**:
  - Block 1: `Conv2d(3, 32, k=3, p=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)`
  - Block 2: `Conv2d(32, 64, k=3, p=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)`
  - Block 3: `Conv2d(64, 128, k=3, p=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)`
  - Block 4: `Conv2d(128, 256, k=3, p=1)` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d(2, 2)`
  - Head: `AdaptiveAvgPool2d(1)` $\to$ `Flatten` $\to$ `Dropout(p=0.4)` $\to$ `Linear(256, 9)`
- **Parameters**: 391,209 total (all trainable).
- **Optimization**: Adam ($\text{lr}=10^{-3}$), Cross-Entropy Loss, `ReduceLROnPlateau(patience=2, factor=0.5)`, Early Stopping (patience=3).

### 4.3 ResNet18 Two-Stage Transfer Learning
- **Backbone**: ImageNet-1K pretrained convolutional feature extractor.
- **Input Resolution**: $160 \times 160$ pixels.
- **Stage 1 (Head Warmup)**:
  - Conv backbone frozen (`requires_grad = False`).
  - Classifier head replaced with `Linear(512, 9)`.
  - Trained for 2 epochs at $\text{lr}=10^{-3}$.
- **Stage 2 (Fine-Tuning)**:
  - `layer4` (the final residual block, 8.4M parameters) unfrozen alongside the head.
  - Fine-tuned for 5 epochs at a reduced learning rate $\text{lr}=10^{-4}$ with Adam.

---

## 5. Experimental Results and Discussion

### 5.1 Comparative Benchmark Results
All models were evaluated on the **identical 1,515 holdout test images**:

| Model | Test Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | Latency (ms/img) | Disk Size |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RBF-SVM (HOG+HSV+LBP)** | 87.46% | 0.8665 | 0.8749 | 0.8687 | 0.8664 | 10.67 ms | 158.5 MB |
| **SmallCNN (Scratch)** | 96.17% | 0.9630 | 0.9619 | 0.9639 | 0.9626 | 10.70 ms | **1.5 MB** |
| **ResNet18 (Fine-Tuned)** | **99.74%** | **0.9967** | **0.9974** | **0.9969** | **0.9965** | 11.64 ms | 42.7 MB |

### 5.2 Key Analytical Observations
1. **The Power of Transfer Learning**: ResNet18 misclassified only **4 out of 1,515 test images**, achieving an astounding **99.74% accuracy**. Pretrained ImageNet filters (edge detectors, textures, high-level shapes) transferred almost seamlessly to plant leaves.
2. **SmallCNN as an Edge Champion**: SmallCNN reached **96.17% accuracy** with zero pretraining. At only **1.5 MB** on disk, it requires 28$\times$ less storage than ResNet18 and 105$\times$ less memory than the SVM support-vector matrix, while delivering the same ~10 ms latency on CPU.
3. **Effectiveness of Handcrafted Features**: The SVM achieved **87.46% accuracy** without neural representations. Color and texture histograms successfully segregated distinct pathologies (e.g. Corn Rust pustules from healthy Corn), but struggled with subtle intra-crop variances (e.g., distinguishing early Apple Scab lesions from minor leaf blemishes).

---

## 6. Software Architecture and Streamlit Application

The interactive demonstration application is organized as an enterprise multi-page Streamlit suite:
- **`app/Home.py`**: Executive summary, dataset metrics, and live dynamic benchmark table.
- **`app/pages/1_Detect.py`**: Leaf upload, interactive sample picker, single-model vs. 3-model side-by-side comparison, HOG gradient visualizer, and comprehensive pathology cards with actionable prevention tips.
- **`app/pages/2_Models_and_Dataset.py`**: Model comparison bar charts, normalized confusion matrices, training loss/accuracy curves, qualitative error analysis, and dataset breakdown.
- **`app/pages/3_About.py`**: Technical methodology, math formulas, data leakage explanation, limitations, and citations.
- **`app/disease_info.json`**: Agricultural pathology knowledge base containing pathogen taxonomy, visual symptoms, chemical/organic treatments, and educational disclaimers.

### 6.1 Automated Test Suite
A comprehensive `pytest` test suite (`tests/test_pipeline.py`) validates system reliability across 10 unit and integration tests:
- `test_dataset_loads_and_labels_map`: Verifies bidirectional label mapping and integrity.
- `test_preprocessing_output_shapes`: Confirms 8,232-d feature vector and PyTorch tensor dimensions.
- `test_model_save_load_round_trip`: Confirms all three models load correctly into memory.
- `test_prediction_determinism`: Asserts identical predictions on repeated inference calls.
- `test_corrupt_file_rejected_gracefully`: Confirms corrupt files trigger safe `ValueError` handling.
- `test_leakage_control`: Rigorously verifies zero leaking hash groups across split boundaries.

---

## 7. 5-Minute Lab Demonstration Walkthrough

For academic evaluation or laboratory presentations, the recommended sequence is:
1. **Minute 1: Introduction & Dataset (Home Page)**
   - Open `streamlit run app/Home.py`.
   - Show the benchmark summary table comparing SVM (87.5%), SmallCNN (96.2%), and ResNet18 (99.7%).
   - Explain the perceptual hash clustering that prevents data leakage.
2. **Minute 2: Handcrafted Feature Extraction Visualizer (Detect Page)**
   - Navigate to **Detect**.
   - Pick a sample leaf (e.g., `Corn_(maize)___Common_rust_`).
   - Expand the **Lab Demo: View Hand-Crafted Feature Representations** section to show the HOG gradient map and explain how edge orientations capture lesion pustules.
3. **Minute 3: Live Inference & Model Comparison (Detect Page)**
   - Select **⚡ Compare All 3 Models Side-by-Side**.
   - Click to predict and demonstrate all three models executing inference in real-time.
   - Show the agricultural pathology card: symptoms, pathogen (*Puccinia sorghi*), and preventative fungicide recommendations.
4. **Minute 4: Confusion Matrices & Learning Curves (Models & Dataset Page)**
   - Navigate to **Models & Dataset Hub**.
   - Review the normalized confusion matrix heatmaps showing near-zero off-diagonal confusion for ResNet18.
   - Inspect the training curves showing fast convergence within 7 epochs.
5. **Minute 5: Test Suite Verification & Wrap-up (Terminal & About Page)**
   - Run `pytest tests/ -v` in the terminal to show all 10 unit tests passing.
   - Conclude with the trade-off analysis: ResNet18 for cloud diagnostic precision vs. SmallCNN (1.5 MB) for offline edge deployment.

---

## 8. Limitations and Future Work

### Limitations
1. **Controlled Background Bias**: PlantVillage images are photographed against uniform neutral paper. Real field imagery features soil clutter, weeds, and varying sun angles.
2. **Single Leaf Focus**: The current models expect a centered, single leaf photograph rather than whole-canopy imagery.

### Future Work
1. **Domain Adaptation with PlantDoc**: Fine-tune models on in-the-wild datasets (PlantDoc) to address outdoor background shifts.
2. **YOLO Lesion Detection**: Transition from image-level classification to bounding-box lesion detection (YOLOv11) to localize multiple simultaneous co-infections.
3. **Mobile Edge Quantization**: Quantize SmallCNN to INT8 (ONNX/TFLite) for instant, sub-5ms diagnosis on low-cost smartphones without cellular internet.

---

## 9. References

1. **Hughes, D., & Salathé, M.** (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
2. **Dalal, N., & Triggs, B.** (2005). *Histograms of oriented gradients for human detection.* IEEE CVPR.
3. **Ojala, T., Pietikäinen, M., & Mäenpää, T.** (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns.* IEEE TPAMI.
4. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep residual learning for image recognition.* IEEE CVPR.
5. **Singh, D., et al.** (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection in the Wild.* ACM CoNEXT.

