"""
generate_report.py – Generates the calibrated, publication-quality 5-page Word (.docx)
project report matching the format, density, and structure of Hand_Sign_Detection_Report.pdf.
"""

from pathlib import Path
import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent
CHARTS_DIR = PROJECT_ROOT / "data" / "charts"
OUTPUT_DOCX_REPORT = PROJECT_ROOT / "report" / "Plant_Disease_Detection_Report.docx"
OUTPUT_DOCX_ROOT = PROJECT_ROOT / "Plant_Disease_Detection_Report.docx"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
COLOR_PRIMARY = RGBColor(27, 54, 93)     # Deep Navy (#1B365D)
COLOR_SECONDARY = RGBColor(74, 96, 122)  # Slate Blue (#4A607A)
COLOR_BODY = RGBColor(34, 34, 34)        # Charcoal (#222222)
COLOR_MUTED = RGBColor(100, 116, 139)    # Muted Slate (#64748B)
COLOR_GREEN = RGBColor(22, 101, 52)      # Forest Green (#166534)

HEX_PRIMARY = "1B365D"
HEX_ALT_ROW = "F8FAFC"
HEX_BORDER = "CBD5E1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=50, bottom=50, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def set_table_borders(table, color_hex=HEX_BORDER):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="{HEX_PRIMARY}"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
        f'  <w:insideV w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f"</w:tblBorders>"
    )
    tblPr.append(borders)


def format_paragraph(p, space_before=0, space_after=3, line_spacing=1.12):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing


def add_section_heading(doc, text, space_before=5, space_after=1):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=space_before, space_after=space_after)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return p


def add_bullet_item(doc, bold_prefix, text, space_after=2):
    p = doc.add_paragraph(style="List Bullet")
    format_paragraph(p, space_before=0, space_after=space_after, line_spacing=1.10)
    run_bold = p.add_run(bold_prefix)
    run_bold.font.name = "Calibri"
    run_bold.font.size = Pt(9.0)
    run_bold.font.bold = True
    run_bold.font.color.rgb = COLOR_PRIMARY

    run_text = p.add_run(text)
    run_text.font.name = "Calibri"
    run_text.font.size = Pt(9.0)
    run_text.font.color.rgb = COLOR_BODY
    return p


def add_figure(doc, img_path, caption_text, width_inches=6.2, space_before=2, space_after=2):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_img, space_before=space_before, space_after=1)
    if Path(img_path).exists():
        run_img = p_img.add_run()
        run_img.add_picture(str(img_path), width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_paragraph(p_cap, space_before=0, space_after=space_after)
    run_cap = p_cap.add_run(caption_text)
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(8.0)
    run_cap.font.italic = True
    run_cap.font.color.rgb = COLOR_MUTED


# ---------------------------------------------------------------------------
# Document Builder
# ---------------------------------------------------------------------------
def build_calibrated_report():
    doc = docx.Document()

    # Exact page margins (0.6 in top/bottom, 0.7 in left/right)
    for section in doc.sections:
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)

    # =========================================================================
    # PAGE 1: Header, Aim, Objective, Theory (Fits exactly on Page 1)
    # =========================================================================

    # Document Header
    p_hdr = doc.add_paragraph()
    format_paragraph(p_hdr, space_before=0, space_after=1)
    r_hdr = p_hdr.add_run("PRCV PROJECT (COMPUTER VISION BASED)")
    r_hdr.font.name = "Calibri"
    r_hdr.font.size = Pt(9.5)
    r_hdr.font.bold = True
    r_hdr.font.color.rgb = COLOR_SECONDARY

    # Title
    p_title = doc.add_paragraph()
    format_paragraph(p_title, space_before=0, space_after=1)
    r_title = p_title.add_run("Plant Foliar Disease Diagnosis & AI Benchmarking")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(16.5)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY

    # Subtitle
    p_sub = doc.add_paragraph()
    format_paragraph(p_sub, space_before=0, space_after=3)
    r_sub = p_sub.add_run(
        "Comparative Evaluation of Handcrafted Features (HOG + HSV + LBP), "
        "Custom Lightweight CNN, and ResNet18 Transfer Learning on Leakage-Safe Splits"
    )
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(9.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = COLOR_SECONDARY

    # AIM
    add_section_heading(doc, "Aim:", space_before=3, space_after=1)
    p_aim = doc.add_paragraph()
    format_paragraph(p_aim, space_before=0, space_after=3)
    r_aim = p_aim.add_run(
        "Automated multi-class diagnosis of agricultural crop diseases from foliar photographs using "
        "classical pattern recognition (HOG, HSV color, LBP texture with RBF-SVM), custom lightweight deep learning "
        "(SmallCNN from scratch), and deep transfer learning (ResNet18) evaluated on a verified, leak-free test partition."
    )
    r_aim.font.name = "Calibri"
    r_aim.font.size = Pt(9.0)
    r_aim.font.color.rgb = COLOR_BODY

    # OBJECTIVE
    add_section_heading(doc, "Objective:", space_before=3, space_after=1)
    p_obj = doc.add_paragraph()
    format_paragraph(p_obj, space_before=0, space_after=3)
    r_obj = p_obj.add_run(
        "To accurately detect and classify nine target plant-health conditions across five economically vital crops "
        "(Apple, Blueberry, Corn, Grape, and Bell Pepper) without cloud dependency; to eliminate optimistic test bias "
        "by preventing data leakage through perceptual hash clustering; to benchmark classification accuracy against "
        "inference latency, model parameter counts, and disk footprints on CPU hardware; and to deploy a real-time, "
        "offline multi-page Streamlit application providing visual confidence rankings and actionable agronomic disease cards."
    )
    r_obj.font.name = "Calibri"
    r_obj.font.size = Pt(9.0)
    r_obj.font.color.rgb = COLOR_BODY

    # THEORY
    add_section_heading(doc, "Theory:", space_before=3, space_after=1)

    add_bullet_item(
        doc,
        "Dataset Characteristics: ",
        "The PlantVillage color dataset contains 10,209 validated leaf photographs across 9 target categories (healthy and diseased states for Apple, Blueberry, Corn, Grape, and Pepper). All images were inspected via Pillow: zero files were corrupted, with standardized square aspect ratios.",
        space_after=2,
    )
    add_bullet_item(
        doc,
        "Data Leakage & Perceptual Hashing: ",
        "PlantVillage contains burst captures of the exact same leaf photographed seconds apart under slight angle variations. Standard random splitting assigns near-duplicate pairs to both training and test partitions, artificially inflating test scores. To guarantee strict leakage control, we compute 64-bit perceptual hashes (imagehash.phash) for every image, cluster images with Hamming distance ≤ 10 via Disjoint-Set Union (Union-Find), and partition groups using GroupShuffleSplit (approx. 70% Train, 15% Validation, 15% Test). Verification confirms zero hash clusters span across split boundaries.",
        space_after=2,
    )
    add_bullet_item(
        doc,
        "Handcrafted Feature Representation (8,232-d): ",
        "Each image (resized to 128×128) yields three complementary feature vectors: (1) HOG on grayscale imagery (9 orientation bins, 8×8 cells, 2×2 blocks, 50% overlap = 8,100 dimensions) capturing edge contours and lesion margins; (2) Cylindrical HSV Color Histograms & Moments (36 Hue bins, 32 Saturation bins, 32 Value bins + channel mean and standard deviation = 106 dimensions) capturing diagnostic chlorosis, necrosis, and fungal discoloration; (3) Uniform LBP (P=24, R=3.0, U≤2 = 26 dimensions) capturing surface roughness and pustule micro-textures. The concatenated 8,232-d vector is standardized via StandardScaler and reduced via PCA (retaining 95% variance to 1,859 orthogonal components).",
        space_after=2,
    )
    add_bullet_item(
        doc,
        "Models Evaluated: ",
        "1. RBF-Kernel SVM: Radial Basis Function kernel K(x, x') = exp(-γ||x - x'||²) with Platt scaling via CalibratedClassifierCV(ensemble=False) for calibrated probability outputs.\n"
        "2. SmallCNN (from scratch): 4-block Conv2d/BatchNorm2d/ReLU/MaxPool2d (32→64→128→256 filters) + Global Average Pooling + Dropout(0.4) + Linear head (391,209 parameters, 1.5 MB disk size).\n"
        "3. ResNet18 (Transfer Learning): ImageNet-pretrained backbone with replaced classification head, trained via a two-stage fine-tuning protocol (Stage 1: head warmup; Stage 2: layer4 + head refinement).",
        space_after=2,
    )
    add_bullet_item(
        doc,
        "Evaluation Metrics: ",
        "Macro-F1, Weighted-F1, Precision, Recall, overall Accuracy, row-normalized Confusion Matrix, inference latency per image (ms), model disk footprint (MB), and CPU utilization.",
        space_after=2,
    )

    # PAGE BREAK 1 -> PAGE 2
    doc.add_page_break()

    # =========================================================================
    # PAGE 2: Methodology (1-10), Fig 1, Fig 2, Real-Time Application
    # =========================================================================

    add_section_heading(doc, "Methodology:", space_before=1, space_after=1)
    p_meth = doc.add_paragraph()
    format_paragraph(p_meth, space_before=0, space_after=3)
    p_meth.add_run(
        "1. Virtual Environment Setup: Configured isolated Python 3.11 environment with PyTorch, Torchvision, Scikit-learn, Scikit-image, OpenCV, Pillow, Pandas, and Streamlit.\n"
        "2. Data Integrity Ingestion: Walked class folders, performed Pillow verify() checks, and tabulated image counts and dimensions.\n"
        "3. Perceptual Hash Clustering: Computed 64-bit pHash vectors, built sliding-window Hamming distance clusters (threshold ≤ 10), and assigned disjoint group IDs.\n"
        "4. Leakage-Free Stratified Splitting: Executed two rounds of GroupShuffleSplit to yield 7,072 Train (70%), 1,622 Validation (15%), and 1,515 Test (15%) images.\n"
        "5. Handcrafted Feature Extraction & Caching: Extracted HOG, HSV, and LBP descriptors, saved precomputed matrices (train_X.npy, val_X.npy, test_X.npy) to bypass repeated disk I/O.\n"
        "6. Data Augmentation: Applied random horizontal flip (p=0.5), vertical flip (p=0.2), random rotation (±15°), and gentle brightness/contrast jitter (0.2). Hue jitter was intentionally omitted as color is a diagnostic disease indicator.\n"
        "7. Traditional SVM Training: Executed 3-fold GridSearchCV over C ∈ {0.1, 1, 10, 100} and γ ∈ {scale, auto, 1e-3, 1e-4} on a 150-image/class stratified subsample; refit best estimator (C=100.0, γ=0.0001) on the full training set.\n"
        "8. Deep Learning Training Schedule: Optimized using Adam, Cross-Entropy Loss, ReduceLROnPlateau scheduler (patience=2, factor=0.5), and Early Stopping (patience=3). SmallCNN was trained for 8 epochs (lr=1e-3); ResNet18 was fine-tuned across 2 stages (Stage 1: head only at lr=1e-3 for 2 epochs; Stage 2: layer4 + head at lr=1e-4 for 5 epochs).\n"
        "9. Standardized Holdout Evaluation: Tested all three models on the identical 1,515 test images, recording accuracy, macro-F1, confusion matrices, and inference latency.\n"
        "10. Production Web Application: Built a multi-page Streamlit suite (Home, Detect, Models & Dataset, About) featuring real-time inference, model comparisons, and agronomic pathology cards."
    )
    p_meth.runs[0].font.name = "Calibri"
    p_meth.runs[0].font.size = Pt(8.8)
    p_meth.runs[0].font.color.rgb = COLOR_BODY

    # Figure 1: Pipeline
    add_figure(
        doc,
        CHARTS_DIR / "fig1_pipeline.png",
        "Fig. 1: End-to-end Machine Learning Pipeline: Perceptual Hash clustering for leakage prevention, dual feature branches (Handcrafted HOG+HSV+LBP vs. Deep Learning), and offline Streamlit deployment.",
        width_inches=6.6,
        space_before=2,
        space_after=2,
    )

    # Figure 2: Sample Leaves
    add_figure(
        doc,
        CHARTS_DIR / "fig2_sample_leaves.png",
        "Fig. 2: Representative foliar samples from the nine target PlantVillage categories (healthy and diseased states for Apple, Blueberry, Corn, Grape, and Bell Pepper).",
        width_inches=6.6,
        space_before=2,
        space_after=3,
    )

    # System Architecture & Streamlit Interface
    add_section_heading(doc, "Real-Time Application & System Architecture:", space_before=2, space_after=1)
    p_app = doc.add_paragraph()
    format_paragraph(p_app, space_before=0, space_after=1)
    p_app.add_run(
        "The diagnostic application is structured as a modular Streamlit multi-page dashboard running entirely offline on local hardware without cloud API dependencies:\n"
        "• Model Caching & Memory Efficiency: Models are instantiated once via @st.cache_resource, enabling instantaneous switching between SVM, SmallCNN, and ResNet18 without reloading overhead.\n"
        "• Unified Inference Pipeline: Reuses the identical validation transforms and normalization parameters through src.predict.predict_image(), returning predicted class, confidence, top-3 distribution, and execution latency.\n"
        "• Handcrafted Feature Visualizer: Features a real-time HOG gradient map generator allowing users and students to inspect the edge orientation structures extracted from leaves during the live lab demo.\n"
        "• Multi-Model Comparative Mode: Executes all three models side-by-side on a single uploaded leaf, comparing prediction consensus and confidence margins.\n"
        "• Agricultural Pathology Database: Integrates app/disease_info.json containing pathogen taxonomy, visual symptomatology, chemical and cultural preventative recommendations, and educational disclaimers."
    )
    p_app.runs[0].font.name = "Calibri"
    p_app.runs[0].font.size = Pt(8.8)
    p_app.runs[0].font.color.rgb = COLOR_BODY

    # PAGE BREAK 2 -> PAGE 3
    doc.add_page_break()

    # =========================================================================
    # PAGE 3: Fig 3 (Interface), Table 1, Table 2 (Fits exactly on Page 3)
    # =========================================================================

    add_section_heading(doc, "Diagnostic Application Interface:", space_before=1, space_after=1)
    add_figure(
        doc,
        CHARTS_DIR / "fig3_interface.png",
        "Fig. 3: Interactive Streamlit Diagnostic Interface: Real-time leaf upload, model selector (Single or 3-way comparison), confidence gauge, top-3 class distribution, and agricultural pathology advice card.",
        width_inches=5.6,
        space_before=1,
        space_after=4,
    )

    add_section_heading(doc, "Output & Benchmark Results:", space_before=2, space_after=1)
    p_tbl_intro = doc.add_paragraph()
    format_paragraph(p_tbl_intro, space_before=0, space_after=2)
    p_tbl_intro.add_run(
        "Table 1: Comprehensive benchmark results evaluated on the identical 1,515 holdout test images across all three models. "
        "Metrics reflect uncorrupted leaf images partitioned via perceptual hash clustering."
    )
    p_tbl_intro.runs[0].font.name = "Calibri"
    p_tbl_intro.runs[0].font.size = Pt(8.5)
    p_tbl_intro.runs[0].font.italic = True
    p_tbl_intro.runs[0].font.color.rgb = COLOR_SECONDARY

    # TABLE 1: Model Comparison
    table1 = doc.add_table(rows=4, cols=9)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table1)

    headers1 = [
        "Model",
        "Paradigm",
        "Test Acc",
        "Macro-F1",
        "Weighted-F1",
        "Precision",
        "Recall",
        "Latency",
        "Disk Size",
    ]
    t1_data = [
        ["RBF-SVM", "HOG+HSV+LBP (PCA)", "87.46%", "0.8665", "0.8749", "0.8687", "0.8664", "10.67 ms", "158.5 MB"],
        ["SmallCNN", "Custom CNN (Scratch)", "96.17%", "0.9630", "0.9619", "0.9639", "0.9626", "10.70 ms", "1.5 MB"],
        ["ResNet18", "Two-Stage Transfer", "99.74%", "0.9967", "0.9974", "0.9969", "0.9965", "11.64 ms", "42.7 MB"],
    ]

    for col_idx, h_text in enumerate(headers1):
        cell = table1.cell(0, col_idx)
        set_cell_shading(cell, HEX_PRIMARY)
        set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
        p = cell.paragraphs[0]
        format_paragraph(p, space_before=0, space_after=0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.name = "Calibri"
        r.font.size = Pt(8.0)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, row_data in enumerate(t1_data, start=1):
        bg_hex = HEX_ALT_ROW if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(row_data):
            cell = table1.cell(row_idx, col_idx)
            set_cell_shading(cell, bg_hex)
            set_cell_margins(cell, top=45, bottom=45, left=60, right=60)
            p = cell.paragraphs[0]
            format_paragraph(p, space_before=0, space_after=0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx >= 2 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = "Calibri"
            r.font.size = Pt(8.0)
            r.font.color.rgb = COLOR_BODY
            if col_idx == 2 and row_idx == 3:
                r.font.bold = True
                r.font.color.rgb = COLOR_GREEN

    col_widths1 = [Inches(1.0), Inches(1.5), Inches(0.7), Inches(0.75), Inches(0.8), Inches(0.7), Inches(0.7), Inches(0.75), Inches(0.7)]
    for row in table1.rows:
        for idx, width in enumerate(col_widths1):
            row.cells[idx].width = width

    # TABLE 2: Computational Runtime
    p_tbl2_intro = doc.add_paragraph()
    format_paragraph(p_tbl2_intro, space_before=5, space_after=2)
    p_tbl2_intro.add_run("Table 2: Computational complexity, parameter counts, training times, and resource profiles on local CPU hardware.")
    p_tbl2_intro.runs[0].font.name = "Calibri"
    p_tbl2_intro.runs[0].font.size = Pt(8.5)
    p_tbl2_intro.runs[0].font.italic = True
    p_tbl2_intro.runs[0].font.color.rgb = COLOR_SECONDARY

    table2 = doc.add_table(rows=4, cols=6)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table2)

    headers2 = ["Model", "Input Size", "Total Parameters", "Trainable Params", "Training Time", "Inference Latency"]
    t2_data = [
        ["RBF-SVM", "128×128 (8,232-d)", "1,859 PCA components", "4,218 Support Vectors", "5.9 min (GridSearch + Fit)", "10.67 ms / image"],
        ["SmallCNN", "128×128×3 RGB", "391,209 parameters", "391,209 (100%)", "7.9 min (8 epochs)", "10.70 ms / image"],
        ["ResNet18", "160×160×3 RGB", "11,181,129 parameters", "8,398,345 (Stage 2)", "8.4 min (2 + 5 epochs)", "11.64 ms / image"],
    ]

    for col_idx, h_text in enumerate(headers2):
        cell = table2.cell(0, col_idx)
        set_cell_shading(cell, HEX_PRIMARY)
        set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
        p = cell.paragraphs[0]
        format_paragraph(p, space_before=0, space_after=0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.name = "Calibri"
        r.font.size = Pt(8.0)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, row_data in enumerate(t2_data, start=1):
        bg_hex = HEX_ALT_ROW if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(row_data):
            cell = table2.cell(row_idx, col_idx)
            set_cell_shading(cell, bg_hex)
            set_cell_margins(cell, top=45, bottom=45, left=60, right=60)
            p = cell.paragraphs[0]
            format_paragraph(p, space_before=0, space_after=0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx >= 4 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = "Calibri"
            r.font.size = Pt(8.0)
            r.font.color.rgb = COLOR_BODY

    col_widths2 = [Inches(1.1), Inches(1.3), Inches(1.4), Inches(1.4), Inches(1.4), Inches(1.2)]
    for row in table2.rows:
        for idx, width in enumerate(col_widths2):
            row.cells[idx].width = width

    # PAGE BREAK 3 -> PAGE 4
    doc.add_page_break()

    # =========================================================================
    # PAGE 4: Fig 4, Fig 5, Results Narrative (Fits exactly on Page 4)
    # =========================================================================

    add_section_heading(doc, "Diagnostic Analysis & Learning Curves:", space_before=1, space_after=1)

    # Figure 4: Model Comparison & Confusion Matrix
    add_figure(
        doc,
        CHARTS_DIR / "fig4_5_combined.png",
        "Fig. 4: Quantitative evaluation on the 1,515 test images: (Left) Accuracy and Macro-F1 comparison; (Right) Row-normalized confusion matrices for ResNet18, SmallCNN, and SVM showing classification precision across all 9 classes.",
        width_inches=6.3,
        space_before=1,
        space_after=3,
    )

    # Figure 5: Training Curves
    add_figure(
        doc,
        CHARTS_DIR / "fig6_curves_combined.png",
        "Fig. 5: Optimization dynamics across training epochs: (Left) SmallCNN trained from scratch converging steadily by epoch 8; (Right) ResNet18 two-stage fine-tuning achieving rapid validation loss reduction below 0.02.",
        width_inches=6.3,
        space_before=1,
        space_after=3,
    )

    # Detailed Discussion & Results Narrative
    add_section_heading(doc, "Results & Discussion:", space_before=2, space_after=1)
    p_disc = doc.add_paragraph()
    format_paragraph(p_disc, space_before=0, space_after=2)
    p_disc.add_run(
        "1. Transfer Learning Dominance (ResNet18): ResNet18 demonstrated decisive performance, misclassifying only 4 out of 1,515 test images to attain 99.74% accuracy and 0.9967 macro-F1. The two-stage protocol proved essential: warming up the classification head on frozen convolutional features prevented gradient explosion, while unfreezing layer4 (8.4M parameters) allowed the network to specialize high-level filters to microscopic leaf lesion structures.\n"
        "2. Extreme Parameter Efficiency (SmallCNN): SmallCNN achieved 96.17% accuracy from scratch with just 391,209 parameters. At 1.5 MB on disk, it requires 28× less storage than ResNet18 (42.7 MB) and 105× less memory than the SVM support vector matrix (158.5 MB), while delivering identical ~10.7 ms CPU inference latency. This proves that a compact 4-block CNN is exceptionally viable for mobile edge devices without cellular connectivity.\n"
        "3. Handcrafted Feature Viability & Limitations (SVM): The traditional baseline reached a respectable 87.46% accuracy. Color histograms and LBP texture provided sharp separation between distinct diseases across crop species (e.g., Corn Common Rust vs. Healthy Corn). However, the handcrafted pipeline struggled with subtle intra-crop variances (e.g., distinguishing early Apple Scab lesions from minor leaf abrasions), where localized edge gradients are overwhelmed by the broader leaf silhouette.\n"
        "4. Training Convergence & Optimization Dynamics: As illustrated in Fig. 5, SmallCNN exhibited smooth monotonic loss reduction, with validation accuracy rising from 78.1% (Epoch 1) to 97.2% (Epoch 8). ResNet18 converged substantially faster: Stage 1 achieved 95.7% accuracy in 2 epochs, and Stage 2 immediately dropped validation loss to 0.0157, reaching 99.63% validation accuracy."
    )
    p_disc.runs[0].font.name = "Calibri"
    p_disc.runs[0].font.size = Pt(8.8)
    p_disc.runs[0].font.color.rgb = COLOR_BODY

    # PAGE BREAK 4 -> PAGE 5
    doc.add_page_break()

    # =========================================================================
    # PAGE 5: Fig 6, Fig 7, Limitations, Outcomes, References (Fits exactly on Page 5)
    # =========================================================================

    add_section_heading(doc, "Qualitative Error Analysis & Class Distributions:", space_before=1, space_after=1)

    # Figure 6: Qualitative Samples (Compact width 5.2 in)
    add_figure(
        doc,
        CHARTS_DIR / "smallcnn_sample_predictions.png",
        "Fig. 6: Qualitative test predictions: Top row shows high-confidence correct classifications across diverse crops; Bottom row highlights edge-case misclassifications with true vs. predicted labels.",
        width_inches=5.2,
        space_before=1,
        space_after=2,
    )

    # Figure 7: Class Distribution (Compact width 4.8 in)
    add_figure(
        doc,
        CHARTS_DIR / "fig6_class_distribution.png",
        "Fig. 7: Class balance and stratified split distribution across Train (70%), Validation (15%), and Test (15%) partitions with zero perceptual hash leakage.",
        width_inches=4.8,
        space_before=1,
        space_after=2,
    )

    # Limitations
    add_section_heading(doc, "Limitations:", space_before=2, space_after=1)
    p_lim = doc.add_paragraph()
    format_paragraph(p_lim, space_before=0, space_after=2)
    p_lim.add_run(
        "• Laboratory Background Bias: PlantVillage specimens are captured against uniform grey/white sheets under artificial lighting. Field foliage exhibits cluttered soil backgrounds, weed interference, and intense sunlight glare, representing an unmeasured domain shift.\n"
        "• Single Leaf Constraint: The system assumes a centered photograph of a single isolated leaf. In agricultural production, whole-canopy imagery containing overlapping foliage is the standard operational input.\n"
        "• Co-Infection Blindness: The multiclass softmax formulation assumes mutually exclusive single labels, whereas crops frequently suffer concurrent bacterial and fungal co-infections."
    )
    p_lim.runs[0].font.name = "Calibri"
    p_lim.runs[0].font.size = Pt(8.5)
    p_lim.runs[0].font.color.rgb = COLOR_BODY

    # Learning Outcomes
    add_section_heading(doc, "Learning Outcomes:", space_before=2, space_after=1)
    p_lo = doc.add_paragraph()
    format_paragraph(p_lo, space_before=0, space_after=2)
    p_lo.add_run(
        "• Mastered end-to-end computer vision workflows: from raw dataset verification and hash clustering to model deployment.\n"
        "• Engineered multi-modal handcrafted feature descriptors (HOG, cylindrical HSV, uniform LBP) and applied dimensionality reduction via PCA.\n"
        "• Implemented and fine-tuned custom convolutional networks and transfer learning architectures on local hardware.\n"
        "• Identified and resolved silent data leakage via perceptual hashing (pHash) and Disjoint-Set Union (Union-Find) clustering.\n"
        "• Built an offline, real-time diagnostic Streamlit application with pathology decision cards and automated pytest test suites."
    )
    p_lo.runs[0].font.name = "Calibri"
    p_lo.runs[0].font.size = Pt(8.5)
    p_lo.runs[0].font.color.rgb = COLOR_BODY

    # References
    add_section_heading(doc, "References:", space_before=2, space_after=1)
    p_ref = doc.add_paragraph()
    format_paragraph(p_ref, space_before=0, space_after=0)
    p_ref.add_run(
        "1. Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable mobile diagnostics. arXiv:1511.08060.\n"
        "2. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. IEEE CVPR, 886-893.\n"
        "3. Ojala, T., et al. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI, 24(7), 971-987.\n"
        "4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. IEEE CVPR, 770-778.\n"
        "5. Singh, D., et al. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection in the Wild. ACM CoNEXT, 249-256."
    )
    p_ref.runs[0].font.name = "Calibri"
    p_ref.runs[0].font.size = Pt(8.0)
    p_ref.runs[0].font.color.rgb = COLOR_MUTED

    # Save to both paths
    OUTPUT_DOCX_REPORT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT_DOCX_REPORT))
    doc.save(str(OUTPUT_DOCX_ROOT))
    print(f"Report successfully generated and saved to:")
    print(f"  -> {OUTPUT_DOCX_REPORT}")
    print(f"  -> {OUTPUT_DOCX_ROOT}")


if __name__ == "__main__":
    build_calibrated_report()
