# PlantVillage Disease Diagnosis & AI Benchmarking System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

An end-to-end Pattern Recognition and Computer Vision (PRCV) pipeline benchmarking **Classical Handcrafted Feature Extraction (HOG + HSV + LBP + RBF-SVM)** against **Modern Deep Convolutional Architectures (Custom SmallCNN & Pretrained ResNet18)** on the PlantVillage dataset across 9 distinct crop health classes.

---

## 📌 Key Highlights

- **Leakage-Free Perceptual Splitting**: Uses perceptual hashing (`imagehash.phash`) with Hamming distance threshold $\le 6$ and `GroupShuffleSplit` (70% Train / 15% Val / 15% Test) to guarantee that burst-shot and near-duplicate leaf captures never cross partition boundaries.
- **Multimodal Classical Features**: 8,232-dimensional hybrid descriptors combining:
  - **HOG**: Histogram of Oriented Gradients ($9 \times 9 \times 100 = 8,100$-d)
  - **HSV Color Histograms**: 3D spatial color distributions ($8 \times 8 \times 8 + 3 \times 16 + 6 = 106$-d)
  - **LBP**: Local Binary Patterns texture descriptors (26-d)
  - Dimensionality reduction via **PCA** (preserving 95% cumulative variance) + **Calibrated RBF-SVM**.
- **Deep Learning Architectures**:
  - **Custom SmallCNN**: 4-stage convolutional network with batch normalization, spatial dropout, and global average pooling (391,337 parameters, 1.5 MB).
  - **ResNet18 Transfer Learning**: Pretrained on ImageNet with fine-tuned residual blocks (11.18M parameters).
- **Interactive Multi-Page Streamlit App**:
  - Single-image & batch leaf diagnostic interface with confidence distributions.
  - Comprehensive model comparison dashboard with interactive confusion matrices and ROC curves.
  - Curated agronomic disease guide with biological pathogen details, visual symptoms, and treatment guidelines.
- **Full Automated Pytest Suite**: 10 unit/integration tests verifying feature dimensions, perceptual splitting, model inference, and metric logging.

---

## 📊 Benchmark Results

Evaluated on the unseen independent test partition (1,438 test images):

| Model | Parameters | Checkpoint Size | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calibrated RBF-SVM** | — | 158.5 MB | **78.44%** | 0.7818 | 0.7766 | 0.7781 |
| **Custom SmallCNN** | 391,337 | 1.5 MB | **93.88%** | 0.9412 | 0.9392 | 0.9399 |
| **ResNet18 (Transfer)** | 11,181,129 | 42.7 MB | **99.17%** | **0.9914** | **0.9912** | **0.9913** |

---

## 🗂️ Project Structure

```text
PRCV_proj/
├── .gitignore                      # Git exclusion rules for large datasets/weights
├── README.md                       # Project overview and reproduction guide
├── requirements.txt                # Python package dependencies
├── plan.txt                        # Engineering phase specifications
├── generate_report.py              # Word report generator script
├── Plant_Disease_Detection_Report.docx  # 5-Page formatted Word technical report
├── Plant_Disease_Detection_Report.pdf   # Compiled PDF report
│
├── app/                            # Streamlit multi-page web application
│   ├── Home.py                     # Landing page and system overview
│   ├── disease_info.json           # Agronomic reference database (9 classes)
│   └── pages/
│       ├── 1_Detect.py             # Diagnostic prediction interface
│       ├── 2_Models_and_Dataset.py # Benchmarking, confusion matrices & curves
│       └── 3_About.py              # Architecture, methodology & metrics
│
├── src/                            # Core pipeline implementation
│   ├── data.py                     # Ingestion, pHash grouping, and PyTorch dataset
│   ├── features.py                 # Handcrafted feature extraction (HOG, HSV, LBP)
│   ├── train_svm.py                # PCA reduction & GridSearch RBF-SVM training
│   ├── models.py                   # PyTorch SmallCNN & ResNet18 definitions
│   ├── train_dl.py                 # Training loops, LR scheduling & checkpoints
│   ├── evaluate.py                 # Metric computation and chart generation
│   └── predict.py                  # Unified inference across all 3 models
│
├── tests/                          # Automated unit and integration tests
│   └── test_pipeline.py            # 10 pytest test cases
│
├── data/
│   ├── splits.csv                  # Reproducible split manifest (hashes & groups)
│   └── charts/                     # Generated evaluation plots and figures
│
├── models/
│   ├── metrics.json                # Unified benchmark metrics summary
│   ├── svm_metrics.json            # SVM classification report and confusion matrix
│   ├── cnn_metrics.json            # SmallCNN training curves and test metrics
│   └── resnet18_metrics.json       # ResNet18 training curves and test metrics
│
└── report/
    ├── REPORT.md                   # Complete academic markdown report
    └── Plant_Disease_Detection_Report.docx
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/PRCV_proj.git
cd PRCV_proj
```

### 2. Environment Setup
Create and activate a virtual environment:
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Streamlit Web Application

Launch the interactive dashboard locally:
```bash
streamlit run app/Home.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the test suite with `pytest`:
```bash
pytest tests/ -v
```

---

## 🏋️ Training & Evaluation Pipelines

### Reproduce Data Splits
```bash
python -c "from src.data import create_leakage_free_splits; create_leakage_free_splits('data/raw', 'data/splits.csv')"
```

### Classical SVM Pipeline
```bash
# Extract HOG + HSV + LBP feature matrices and train calibrated RBF-SVM
python src/train_svm.py
```

### Deep Learning Pipeline
```bash
# Train SmallCNN from scratch and fine-tune ResNet18
python src/train_dl.py
```

### Generate Comprehensive Evaluation Charts
```bash
python -c "from src.evaluate import evaluate_all; evaluate_all()"
```

---

## 📄 License
This project is open-source under the MIT License.

