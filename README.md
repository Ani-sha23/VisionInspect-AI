<p align="center">
  
  # **VisionInspect-AI**
<img width="508" height="290" alt="image" src="https://github.com/user-attachments/assets/6d3205d0-7958-4262-aa35-a62b5f34b2f2" />


<p align="center">

### 🏭 Automated Industrial Surface Defect Inspection Framework

**From raw steel-surface images → defect localization → severity estimation → quality decision → auditable reports**

<p>

<p align="center">

[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Industrial%20Inspection-0A66C2?style=for-the-badge)](https://github.com/Ani-sha23/VisionInspect-AI)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTest](https://img.shields.io/badge/Tests-PyTest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge)](LICENSE)

</p>

<p align="center">

**A modular, reproducible and configurable computer-vision pipeline for industrial steel-surface inspection.**

</p>

---

##  Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [System Workflow](#-system-workflow)
- [Architecture](#-architecture)
- [Model Design](#-model-design)
- [Dataset](#-dataset)
- [Dataset Statistics](#-dataset-statistics)
- [Data Preparation](#-data-preparation)
- [Image Preprocessing](#-image-preprocessing)
- [Defect Detection](#-defect-detection)
- [Defect Localization](#-defect-localization)
- [Severity Engine](#-severity-engine)
- [Quality Decision Engine](#-quality-decision-engine)
- [Evaluation](#-evaluation)
- [Visual Outputs](#-visual-outputs)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [CLI Reference](#-cli-reference)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Reproducibility](#-reproducibility)
- [Engineering Design Decisions](#-engineering-design-decisions)
- [Limitations](#-limitations)
- [Future Roadmap](#-future-roadmap)
- [Academic Information](#-academic-information)
- [Author](#-author)
- [License](#-license)

---

#  Overview

**VisionInspect-AI** is a modular computer-vision framework designed for automated inspection of **cold-rolled steel surfaces**.

The system transforms a raw industrial image into structured inspection evidence by combining:

-  **Multi-label U-Net segmentation**
-  **Classical computer-vision baseline detection**
-  **Pixel-level defect localization**
-  **Connected-component instance extraction**
-  **Configurable 0–100 severity scoring**
-  **Percentile-based severity calibration**
-  **ACCEPT / REVIEW / REJECT quality decisions**
-  **Segmentation evaluation**
-  **Inspection visualization**
-  **JSON / CSV / Markdown reporting**
-  **Automated testing**
-  **Unified command-line interface**

The project separates **data preparation, preprocessing, detection, severity logic, evaluation and reporting** into independent components so that each stage can be tested and reproduced.

---

#  Problem Statement

Industrial steel surfaces may contain visual defects such as cracks, scratches, inclusions and other surface irregularities.

Manual inspection can be:

- Time-consuming
- Difficult to scale
- Subjective
- Dependent on operator experience
- Challenging for high-volume manufacturing environments

A computer-vision-based inspection system can assist by automatically identifying suspicious regions and converting visual information into structured inspection results.

### VisionInspect-AI addresses this problem through a complete pipeline:

**Image → Preprocessing → Segmentation → Localization → Severity → Decision → Report**

---

#  Objectives

The main objectives of VisionInspect-AI are:

1. **Detect surface defects automatically**
2. **Localize defects at pixel level**
3. **Support multiple defect categories**
4. **Extract spatial defect instances**
5. **Estimate an interpretable severity score**
6. **Convert severity into configurable quality decisions**
7. **Evaluate segmentation performance quantitatively**
8. **Generate inspection-ready visual and structured reports**
9. **Maintain reproducibility through configuration-driven experiments**
10. **Provide both deep-learning and classical-CV inspection paths**

---
# Key Features

<img width="472" height="278" alt="image" src="https://github.com/user-attachments/assets/22043519-3f7c-4ccb-874c-9c04ecd5d663" />


---

#  System Workflow

```text
                    ┌──────────────────────┐
                    │    Steel Image      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Ingestion &          │
                    │ Validation           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    │ CLAHE • Resize •     │
                    │ Normalization        │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │        Detection Layer         │
              │                                │
              │   ┌────────────┐ ┌──────────┐ │
              │   │  U-Net     │ │ Classical│ │
              │   │  Backend   │ │    CV    │ │
              │   └─────┬──────┘ └────┬─────┘ │
              └─────────┼──────────────┼───────┘
                        │              │
                        └──────┬───────┘
                               ▼
                    ┌──────────────────────┐
                    │ Defect Masks /       │
                    │ Spatial Regions      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Instance Extraction  │
                    │ Connected Components │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Severity Engine      │
                    │ Score: 0 – 100       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Quality Decision     │
                    │ ACCEPT / REVIEW /    │
                    │ REJECT               │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Reports & Visuals    │
                    │ JSON / CSV / MD / PNG│
                    └──────────────────────┘

 Architecture

VisionInspect-AI follows a modular architecture:

┌───────────────────────────────────────────────────────────┐
│                     VisionInspect-AI                      │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  DATA LAYER                                               │
│  ├── Dataset Loading                                       │
│  ├── RLE Decoding                                          │
│  ├── Schema Validation                                     │
│  └── Leakage-Safe Splitting                                │
│                                                           │
│  PREPROCESSING                                             │
│  ├── CLAHE                                                 │
│  ├── Resize                                                │
│  └── Normalization                                         │
│                                                           │
│  DETECTION                                                 │
│  ├── Multi-Label U-Net                                     │
│  └── Classical CV Baseline                                 │
│                                                           │
│  POST-PROCESSING                                           │
│  ├── Thresholding                                          │
│  ├── Connected Components                                  │
│  └── Defect Instances                                      │
│                                                           │
│  INSPECTION LOGIC                                          │
│  ├── Severity Engine                                       │
│  ├── Severity Calibration                                  │
│  └── Quality Decision Engine                               │
│                                                           │
│  EVALUATION                                                │
│  ├── Dice                                                  │
│  ├── IoU                                                    │
│  ├── Precision                                             │
│  ├── Recall                                                │
│  └── F1                                                     │
│                                                           │
│  REPORTING                                                 │
│  ├── JSON                                                  │
│  ├── CSV                                                   │
│  ├── Markdown                                              │
│  └── PNG Visualization                                     │
│                                                           │
└───────────────────────────────────────────────────────────┘
 Model Design
Multi-Label U-Net

The deep-learning component uses a compact U-Net architecture for pixel-level segmentation.

                         INPUT IMAGE
                              │
                              ▼
                    ┌─────────────────┐
                    │     ENCODER     │
                    │ Feature Extract │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   BOTTLENECK    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     DECODER     │
                    │ Feature Recover │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
           Channel 1      Channel 2      Channel 3
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                         Channel 4
                             │
                             ▼
                 DEFECT PROBABILITY MAPS

The model produces four independent output channels, with each channel representing a defect category.

Why Sigmoid?

The defect channels are treated independently.

Therefore, a sigmoid-based multi-label formulation is used instead of softmax.

Input Image
     │
     ▼
   U-Net
     │
     ▼
4 Independent Channels
     │
     ▼
Sigmoid Probability Maps
     │
     ▼
Thresholded Defect Masks

This allows different defect categories to be represented independently.

 Training Objective

The training objective combines Binary Cross-Entropy and Dice Loss:

L_total = L_BCE + L_Dice

This combines:

Pixel-wise classification supervision
Segmentation overlap optimization
Training Configuration

The project supports:

AdamW optimizer
Cosine learning-rate scheduling
Validation Dice monitoring
Early stopping
Configurable training parameters
🛡️ Leakage-Safe Dataset Splitting

A major engineering consideration is data leakage.

The dataset may contain multiple annotation records associated with the same source image.

A naive row-level split could therefore place information from the same image into both training and validation/test sets.

VisionInspect-AI instead performs splitting based on ImageId.

              Annotation Records
                       │
                       ▼
                  Group by
                   ImageId
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          TRAIN       VAL       TEST
             │         │         │
             └─────────┼─────────┘
                       │
              No ImageId overlap

This provides a cleaner separation between training and held-out evaluation subsets.

📊 Dataset

VisionInspect-AI is designed around the:

Severstal: Steel Defect Detection dataset

The dataset contains high-aspect-ratio steel-surface images with run-length encoded segmentation annotations.

Dataset characteristics
High-resolution / high-aspect-ratio steel images
RLE-based segmentation annotations
Multiple annotation records
Multiple defect configurations
Four defect categories
Significant class imbalance
📈 Dataset Statistics

The current dataset inspection produced the following measurements:

Statistic	Value
🖼️ Images on Disk	12,568
📝 Annotation Rows	7,095
🔴 Images With Defects	6,666
🟢 Images Without Defects	5,902
🏷️ Defect Classes	4
📐 Sampled Image Shape	256 × 1600 × 3
🚂 Training Images	8,800
🔎 Validation Images	1,884
🧪 Test Images	1,884
Annotation Distribution
Defect Class	Annotation Rows
Class 1	897
Class 2	247
Class 3	5,150
Class 4	801

Important: These values describe the inspected dataset. They are not model-performance metrics.

📦 Data Preparation

The project automatically creates leakage-safe train/validation/test splits.

visioninspect prepare-data

Current generated split:

TRAIN : 8,800 images
VAL   : 1,884 images
TEST  : 1,884 images

Generated files:

data/
└── processed/
    ├── train.csv
    ├── val.csv
    └── test.csv
🎨 Image Preprocessing

The preprocessing pipeline supports:

1. CLAHE

Contrast Limited Adaptive Histogram Equalization is used to enhance local image contrast.

2. Resizing

Images are transformed according to the configured model input dimensions.

3. Intensity Normalization

Pixel intensities are normalized before being passed to the model.

Pipeline
RAW IMAGE
    │
    ▼
CLAHE
    │
    ▼
RESIZE
    │
    ▼
NORMALIZATION
    │
    ▼
MODEL INPUT

All preprocessing parameters are controlled through:

configs/config.yaml
🔍 Defect Detection

VisionInspect-AI provides two detection paths.

🧠 1. U-Net Backend

The U-Net backend performs learned multi-label segmentation.

Image
  ↓
U-Net
  ↓
4 Sigmoid Channels
  ↓
Probability Maps
  ↓
Thresholding
  ↓
Defect Masks
🔧 2. Classical Computer-Vision Baseline

A classical CV detector is also provided.

The baseline is useful for:

CPU smoke testing
Pipeline verification
Baseline comparison
Debugging
Environments where deep-learning inference is unavailable

The baseline is class-agnostic and is intended as a lightweight comparison/fallback path rather than a replacement for learned multi-class segmentation.

🎯 Defect Localization

Once segmentation masks are generated, the system converts pixel-level predictions into spatial regions.

The pipeline performs:

Predicted Mask
      │
      ▼
Thresholding
      │
      ▼
Connected Components
      │
      ▼
Defect Instances
      │
      ├── Location
      ├── Area
      └── Class / Channel

This creates an interpretable bridge between:

Pixel-level segmentation → Object-level inspection information

📊 Severity Engine

VisionInspect-AI includes a configurable severity engine that produces a 0–100 severity score.

Conceptually:

Defect Coverage
       +
Class-Specific Factors
       +
Severity Multipliers
       │
       ▼
 Raw Severity
       │
       ▼
0 ─────────────────── 100

Defect coverage is based on:

                  Defect Pixels
Coverage = ─────────────────────────
                 Total Pixels
Important

The severity score is an engineered operational score.

It should not be interpreted as a direct physical measurement of material damage.

⚖️ Severity Calibration

Raw severity scores can be calibrated using percentile boundaries derived from the training distribution.

Example configuration:

              P50          P80          P95
               │            │            │
               ▼            ▼            ▼

            ACCEPT        REVIEW       REJECT

The configured percentile relationship is expected to remain:

P50 < P80 < P95

Before calibration, severity-based decisions can be identified as:

UNCalibrated
🏭 Quality Decision Engine

The inspection pipeline converts severity and configured defect rules into three operational outcomes.

Decision	Meaning
🟢 ACCEPT	Inspection satisfies configured acceptance conditions
🟡 REVIEW	Inspection requires additional verification
🔴 REJECT	Inspection satisfies configured rejection / critical-defect conditions
Important Engineering Note

These decision rules are project-specific configurable rules.

They should not be treated as universal manufacturing standards without domain validation.

📈 Evaluation

VisionInspect-AI supports quantitative segmentation evaluation.

Metrics
Dice Coefficient
Intersection over Union (IoU)
Precision
Recall
F1 Score
Per-class measurements where applicable
Evaluation Principle

Model-performance numbers are reported only after an actual experiment has been executed.

Test Dice      : TO BE FILLED
Test IoU       : TO BE FILLED
Precision      : TO BE FILLED
Recall         : TO BE FILLED
F1             : TO BE FILLED

No fabricated model-performance numbers are included in this repository.

After training and evaluation, these values should be replaced with the actual held-out test results.

🖼️ Visual Outputs

The project is designed to generate inspection-friendly visualizations.

Original Image
      │
      ▼
Defect Mask
      │
      ▼
Spatial Regions
      │
      ▼
Class / Instance Information
      │
      ▼
Severity
      │
      ▼
Quality Decision

Typical generated artifacts include:

outputs/
├── results.json
├── results.csv
├── report.md
└── figures/
    └── *.png
🧰 Technology Stack
Category	Technology
Programming Language	Python
Deep Learning	PyTorch
Segmentation	U-Net
Image Processing	OpenCV
Augmentation	Albumentations
Data Processing	NumPy, Pandas
Machine Learning Utilities	scikit-learn
Visualization	Matplotlib
Configuration	YAML / PyYAML
Testing	PyTest
CLI	Typer
Version Control	Git / GitHub
📁 Project Structure
VisionInspect-AI/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── train_images/
│   │
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── docs/
│   ├── adr/
│   │   └── ADR-001.md
│   │
│   ├── results/
│   │   ├── README.md
│   │   └── dataset_stats.md
│   │
│   ├── REPORT_SKELETON.md
│   ├── TRAINING.md
│   └── architecture.md
│
├── models/
│
├── outputs/
│   └── figures/
│
├── scripts/
│   └── make_demo_images.py
│
├── src/
│   └── visioninspect/
│       ├── __init__.py
│       ├── __main__.py
│       ├── baseline_detector.py
│       ├── calibration.py
│       ├── classifier.py
│       ├── cli.py
│       ├── config_loader.py
│       ├── dataset.py
│       ├── dataset_inspection.py
│       ├── detector.py
│       ├── evaluate.py
│       ├── exceptions.py
│       ├── logger.py
│       ├── metrics.py
│       ├── model.py
│       ├── pipeline.py
│       ├── preprocessing.py
│       ├── quality_engine.py
│       ├── report_generator.py
│       ├── rle.py
│       ├── schemas.py
│       ├── severity.py
│       ├── train.py
│       └── visualization.py
│
├── tests/
│   ├── conftest.py
│   ├── test_dataset_and_metrics.py
│   ├── test_detector.py
│   ├── test_pipeline_and_config.py
│   ├── test_preprocessing.py
│   ├── test_rle.py
│   └── test_severity_quality.py
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── README.md
└── SUBMISSION_CHECKLIST.md
⚙️ Installation
1. Clone the Repository
git clone https://github.com/Ani-sha23/VisionInspect-AI.git
cd VisionInspect-AI
2. Create a Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Install the Project
pip install -e .
⚡ Quick Start
Step 1 — Inspect Dataset
visioninspect inspect-data

This generates:

docs/results/dataset_stats.md

and a class-distribution visualization.

Step 2 — Prepare Dataset
visioninspect prepare-data

This generates:

data/processed/
├── train.csv
├── val.csv
└── test.csv
Step 3 — Calibrate Severity
visioninspect calibrate --apply
Step 4 — Train the U-Net
visioninspect train --epochs 12

Training requires an appropriate PyTorch environment. GPU acceleration is recommended for practical deep-learning experimentation.

Step 5 — Run Inspection
visioninspect run --input data/raw/train_images --limit 20
Step 6 — Evaluate
visioninspect evaluate --split data/processed/test.csv
💻 CLI Reference

The project exposes a unified command-line interface:

visioninspect
Available Commands
Command	Purpose
inspect-data	Analyze the raw dataset
prepare-data	Create leakage-safe train/val/test splits
calibrate	Derive/apply severity calibration
train	Train the U-Net
run	Inspect an image or directory
evaluate	Evaluate a split against ground-truth masks
General Options
--help
--config CONFIG
--log-level LOG_LEVEL
--version
⚙️ Configuration

Project behavior is centralized in:

configs/config.yaml

Configuration includes:

Preprocessing parameters
Model parameters
Severity weights
Severity multipliers
Decision thresholds
Percentile calibration
Critical-defect rules
Experiment settings

Centralizing configuration helps:

Reduce hard-coded assumptions
Improve reproducibility
Make experiments easier to compare
Keep operational rules separate from implementation code
🧪 Testing

Run the complete test suite with:

pytest -q

The project includes tests for:

RLE encoding / decoding
Dataset schema handling
Preprocessing
Detector behavior
Severity calculations
Calibration
Pipeline behavior
Configuration validation
Report generation
Leakage-related logic
🔬 Reproducibility

Before reporting final model results, verify:

☐ Dataset is correctly placed
☐ Annotation CSV is valid
☐ Images are readable
☐ Dataset statistics have been generated
☐ ImageId-based split has been created
☐ Training configuration is recorded
☐ Model checkpoint is saved
☐ Validation metrics are recorded
☐ Test set remains untouched during model selection
☐ Evaluation results are generated
☐ Visual predictions are inspected
☐ Final report values match actual outputs
🧩 Engineering Design Decisions
1. Multi-Label U-Net

A single U-Net predicts four independent segmentation channels.

Benefit

A consistent segmentation pipeline can support multiple defect categories.

Trade-off

Different classes may have different visual characteristics and frequencies, requiring suitable model capacity and thresholding.

2. ImageId-Based Splitting

The dataset is split using ImageId groups rather than independent annotation rows.

Reason

Multiple annotation records can belong to the same source image.

Goal

Prevent information from the same source image from appearing across different evaluation subsets.

3. Deep Learning + Classical Baseline

The system contains both:

Deep Learning
     +
Classical Computer Vision
Reason

The classical baseline provides:

Lightweight smoke testing
Pipeline verification
Comparison
CPU-friendly execution
4. Configurable Severity Logic

Severity thresholds and weights are configuration-driven.

Reason

Severity definitions can vary depending on the application.

Keeping them outside the model implementation makes the system easier to adapt and validate.

⚠️ Limitations

VisionInspect-AI is an academic/research-oriented computer-vision framework and has several limitations.

1. Engineered Severity Score

The 0–100 severity score is an engineered operational score.

It is not a direct physical measurement of material damage.

2. Classical Baseline

The classical CV baseline is class-agnostic.

It is intended for:

Baseline comparison
Smoke testing
Pipeline validation
Fallback processing
3. Compact Model

The current U-Net is intentionally lightweight.

More advanced pretrained architectures may provide stronger performance and should be evaluated experimentally.

4. Demo Images

Synthetic/demo images are intended for smoke testing.

They must not be used to claim real-world industrial model performance.

5. Operational Thresholds

ACCEPT / REVIEW / REJECT thresholds are configurable project assumptions.

Real manufacturing deployment would require validation by appropriate domain experts.

6. Experiment-Dependent Results

Model performance depends on:

Training configuration
Data split
Preprocessing
Hardware
Hyperparameters
Randomness
Model architecture

Therefore, final performance values must come from actual held-out evaluation.

🛣️ Future Roadmap
Current
✅ Dataset inspection
✅ Leakage-safe splitting
✅ Multi-label U-Net architecture
✅ Classical CV baseline
✅ Image preprocessing
✅ Defect localization
✅ Connected-component extraction
✅ Severity engine
✅ Severity calibration
✅ Quality decision engine
✅ Evaluation framework
✅ Automated reporting
✅ Automated tests
Future Enhancements
⬜ Stronger pretrained encoders
⬜ Per-class threshold optimization
⬜ Test-time augmentation
⬜ Improved class imbalance handling
⬜ Real-time production-line video
⬜ FastAPI inference service
⬜ Dockerized deployment
⬜ Factory / MES integration
⬜ Model monitoring
⬜ Production-grade inference optimization

Future roadmap items represent planned enhancements and are not claims about the current implementation.

📚 Documentation

Project documentation is organized under:

docs/
├── adr/
├── results/
├── REPORT_SKELETON.md
├── TRAINING.md
└── architecture.md

Important documentation includes:

📐 Architecture documentation
🧩 Architecture Decision Records
📊 Dataset statistics
🧠 Training guidance
📈 Evaluation results
📋 Submission checklist
📊 Current Project Status
Component	Status
Repository Setup	✅ Complete
Dataset Integration	✅ Complete
Dataset Inspection	✅ Complete
Dataset Statistics	✅ Generated
Class Distribution	✅ Generated
Leakage-Safe Split	✅ Complete
Train / Val / Test CSVs	✅ Generated
Preprocessing Pipeline	✅ Implemented
U-Net Architecture	✅ Implemented
Classical CV Baseline	✅ Implemented
Severity Engine	✅ Implemented
Calibration	✅ Implemented
Quality Decision Engine	✅ Implemented
Evaluation Framework	✅ Implemented
Reporting Framework	✅ Implemented
Automated Tests	✅ Implemented
GPU Training Experiment	🔄 Pending
Final Test Metrics	🔄 Pending
Final Model Checkpoint	🔄 Pending
Final Visual Results	🔄 Pending
🏆 Project Highlights
Why this project is more than a simple image-classification model

VisionInspect-AI does not stop at:

Image → Class

Instead, it builds a complete inspection workflow:

                    ┌─────────────┐
                    │    IMAGE    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ PREPROCESS  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  SEGMENT    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ LOCALIZE    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  SEVERITY   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  DECISION   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   REPORT    │
                    └─────────────┘

This makes the project suitable for demonstrating concepts from:

Computer Vision
Image Segmentation
Deep Learning
Image Processing
Machine Learning Evaluation
Software Engineering
Data Leakage Prevention
Explainable Inspection Logic
Reproducible ML Pipelines
🎓 Academic Information

Project: VisionInspect-AI

Course: Computer Vision

Project Type: Computer Vision – Evaluated Project

Student: Anisha Garg

Registration Number: 24BAI1037

Program: B.Tech Computer Science Engineering

Specialization: Artificial Intelligence & Machine Learning

👩‍💻 Author
Anisha Garg

B.Tech CSE — Artificial Intelligence & Machine Learning

VisionInspect-AI was developed as an academic Computer Vision project focused on combining:

Deep Learning
      +
Image Processing
      +
Segmentation
      +
Evaluation
      +
Inspection Logic
      +
Reproducible Software Engineering
📜 License

This project is released under the MIT License.

See the LICENSE file for details.

⭐ Final Project Summary

VisionInspect-AI is a modular industrial computer-vision inspection framework that transforms steel-surface images into structured inspection evidence by detecting and localizing defects, estimating configurable severity, applying quality-decision rules, evaluating model behavior, and generating reproducible inspection reports.
