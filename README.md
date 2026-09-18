# **VisionInspect-AI**
<img width="508" height="290" alt="image" src="https://github.com/user-attachments/assets/6d3205d0-7958-4262-aa35-a62b5f34b2f2" />


<p align="center">
  <img src="https://img.shields.io/badge/Computer%20Vision-Industrial%20Inspection-0A66C2?style=for-the-badge" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Tests-PyTest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="PyTest">
  <img src="https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge" alt="MIT License">
</p>

<h3 align="center">Automated Industrial Surface Defect Inspection Framework</h3>

<p align="center">
  <b>From raw steel-surface images to localized defects, severity estimates, quality decisions, and auditable reports.</b>
</p>

<p align="center">
  Built for computer-vision experimentation and reproducible industrial inspection workflows.
</p>

📌 Project at a Glance

VisionInspect-AI is a modular computer-vision framework for inspecting cold-rolled steel surfaces for visual defects.

The system combines:

🧠 Multi-label U-Net segmentation

👁️ Classical computer-vision baseline detection

🎯 Pixel-level defect localization

🏷️ Four-channel defect representation

🔢 Connected-component instance extraction

📊 Configurable 0–100 severity scoring

⚖️ Percentile-based severity calibration

🟢 ACCEPT / REVIEW / REJECT decision logic

📈 Segmentation and classification-oriented evaluation

🖼️ Visual inspection overlays

📄 JSON / CSV / Markdown reporting

🧪 Automated tests

💻 Unified command-line interface

Core idea: transform an industrial surface image into an interpretable inspection result while keeping data preparation, inference, severity logic, evaluation, and reporting as separate, testable components.

🧭 End-to-End Workflow

┌─────────────────────┐
│   Steel Image       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Ingestion &         │
│ Validation          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Preprocessing       │
│ CLAHE • Resize •    │
│ Normalization       │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────┐
│        Detection Layer       │
│                              │
│   ┌──────────┐ ┌──────────┐  │
│   │  U-Net   │ │ Baseline │  │
│   │ Backend  │ │   CV     │  │
│   └────┬─────┘ └────┬─────┘  │
└────────┼─────────────┼────────┘
         │             │
         └──────┬──────┘
                ▼
┌─────────────────────┐
│ Defect Masks /      │
│ Spatial Regions     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Classification &    │
│ Instance Extraction │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Severity Scoring    │
│       0 ── 100      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Quality Decision    │
│ ACCEPT / REVIEW /   │
│ REJECT              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Reports & Visuals   │
│ JSON / CSV / MD /   │
│ PNG                 │
└─────────────────────┘

✨ Key Capabilities

Capability

Description

🔍 Dataset inspection

Profiles annotations, image counts, shapes, class distribution and defect presence

🛡️ Leakage-safe splitting

Groups samples by ImageId before train/validation/test splitting

🎨 Preprocessing

CLAHE, resizing and intensity normalization

🧠 Segmentation

Compact multi-label U-Net with four sigmoid output channels

🔧 Baseline

Classical CV detector for CPU-friendly smoke testing and comparison

🎯 Localization

Converts predicted masks into spatial defect regions

🔢 Instance extraction

Uses connected components to identify disconnected defect regions

📊 Severity

Converts defect coverage and configured class factors into a 0–100 score

⚖️ Calibration

Derives severity bands from training-distribution percentiles

🏭 Quality logic

Maps severity and configured critical-defect rules to an operational decision

📈 Evaluation

Supports Dice, IoU, precision, recall and F1-style measurements

🖼️ Visualization

Produces inspection overlays and audit-friendly figures

📄 Reporting

Generates structured JSON, CSV and Markdown outputs

🧪 Testing

Includes unit/integration tests for core pipeline behavior

💻 CLI

Provides one command surface for the major workflows

🧠 Model Design

Multi-Label U-Net

VisionInspect-AI uses a single compact U-Net with four independent output channels.

                    Input Image
                         │
                         ▼
                  ┌─────────────┐
                  │    U-Net    │
                  │  Encoder +  │
                  │   Decoder   │
                  └──────┬──────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Channel 1      Channel 2      Channel 3
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                      Channel 4
                         │
                         ▼
                Defect probability maps

Each output channel represents an independently predicted defect mask.

Why sigmoid?

The four channels are treated independently, so a sigmoid output is appropriate for multi-label mask prediction. This differs from a softmax formulation, which would impose a mutually exclusive class assumption at each pixel.

Training objective

The project combines Binary Cross-Entropy and Dice loss:

L_total = L_BCE + L_Dice

This provides a combination of pixel-wise supervision and overlap-oriented optimization.

Optimization strategy

The training configuration uses:

AdamW

Cosine learning-rate scheduling

Validation Dice monitoring

Early stopping

Configurable training parameters

🛡️ Leakage-Safe Dataset Splitting

Dataset leakage is an important concern when multiple annotation records can correspond to the same source image.

A row-level random split can allow information from the same ImageId to appear in different subsets.

VisionInspect-AI therefore uses an ImageId-grouped split:

                 Annotation Records
                         │
                         ▼
                    Group by
                     ImageId
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
         TRAIN          VAL          TEST
            │            │            │
            └────── No ImageId overlap ──────┘

This is intended to provide a cleaner separation between training and held-out evaluation data.

📊 Dataset

The project is designed around the Severstal: Steel Defect Detection dataset.

Measured dataset snapshot

Statistic

Value

Images on disk

12,568

Annotation rows

7,095

Images with defects

6,666

Images without defects

5,902

Defect classes

4

Sampled image shape

256 × 1600 × 3

Training images

8,800

Validation images

1,884

Test images

1,884

Annotation distribution

Defect class

Annotation rows

Class 1

897

Class 2

247

Class 3

5,150

Class 4

801

These are dataset-inspection measurements, not model-performance results.

Dataset characteristics

The dataset contains:

High-aspect-ratio steel images

Run-length encoded (RLE) segmentation annotations

Multiple annotation records

Images containing different defect configurations

Significant class imbalance

🔬 Inspection Pipeline

1. Ingestion

Loads images from a configured local input directory.

2. Validation

Checks image-related assumptions before processing.

3. Preprocessing

The preprocessing stage includes:

CLAHE enhancement

Resizing

Intensity normalization

Configurable preprocessing parameters

4. Detection

Two backends are available:

🧠 U-Net backend

Used for learned multi-label segmentation.

Image
  ↓
U-Net
  ↓
4 sigmoid channels
  ↓
Thresholded masks
  ↓
Defect regions

🔧 Classical CV baseline

Provides an alternative anomaly-localization path based on classical computer vision.

It is useful for:

CPU smoke testing

Pipeline verification

Baseline comparison

Environments where deep-learning inference is unavailable

The baseline is class-agnostic.

🎯 Defect Localization & Instance Extraction

After segmentation, predicted masks can be converted into spatial regions.

The pipeline uses connected components to identify disconnected defect regions.

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
      └── Class/channel

This provides a bridge between pixel-level segmentation and higher-level inspection information.

📊 Severity Engine

VisionInspect-AI converts defect information into a configurable 0–100 severity score.

Conceptually:

Defect coverage
      +
Class-specific factors
      +
Severity multipliers
      │
      ▼
Raw severity
      │
      ▼
0 ───────────────────────── 100

Defect coverage is based on the relationship between defect pixels and total image pixels:

                 Defect Pixels
Coverage = ─────────────────────────
              Total Image Pixels

The resulting score is an operational engineering score, not a direct physical measurement of material damage.

⚖️ Severity Calibration

Raw severity can be calibrated using percentile boundaries derived from the training distribution.

Conceptually:

             P50          P80          P95
              │            │            │
              ▼            ▼            ▼

       ACCEPT       REVIEW        REJECT

The configured percentile ordering is expected to remain:

P50 < P80 < P95

Before calibration, severity-based decisions can be marked as UNCALIBRATED.

🏭 Quality Decision Engine

The decision engine translates inspection information into three configured operational outcomes:

Decision

Interpretation

🟢 ACCEPT

Inspection falls within the configured acceptance conditions

🟡 REVIEW

Inspection falls into a review condition or requires human verification

🔴 REJECT

Inspection meets configured rejection/critical-defect conditions

These decisions are controlled by configuration and should be treated as project-specific operational rules rather than universal industrial standards.

📈 Evaluation

The project supports quantitative evaluation of segmentation and inspection behavior.

Metrics

Dice coefficient

Intersection over Union (IoU)

Precision

Recall

F1

Per-class measurements where applicable

Results policy

Model-performance numbers should be reported only after an actual experiment has been executed.

Test Dice       : [TO BE FILLED]
Test IoU        : [TO BE FILLED]
Precision       : [TO BE FILLED]
Recall          : [TO BE FILLED]
F1              : [TO BE FILLED]

No performance values are fabricated in this README.

🖼️ Visual Outputs

The reporting pipeline is designed to produce audit-friendly outputs such as:

Original Image
      │
      ▼
Defect Mask
      │
      ▼
Spatial Region
      │
      ▼
Class / Instance Information
      │
      ▼
Severity
      │
      ▼
Quality Decision

Typical output artifacts include:

outputs/
├── results.json
├── results.csv
├── report.md
└── figures/
    └── *.png

🧰 Technology Stack

Area

Technologies

Language

Python

Deep Learning

PyTorch, Torchvision

Segmentation

U-Net, segmentation-models-pytorch

Image Processing

OpenCV

Augmentation

Albumentations

Data Processing

NumPy, Pandas

ML Utilities

scikit-learn

Visualization

Matplotlib

Configuration

YAML / PyYAML

Testing

PyTest

CLI

Typer

Version Control

Git / GitHub

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
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── docs/
│   ├── adr/
│   └── results/
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
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── README.md
└── SUBMISSION_CHECKLIST.md

💻 Installation

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd VisionInspect-AI

2. Create a virtual environment

Windows

python -m venv .venv
.venv\Scripts\activate

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

3. Install core dependencies

pip install -r requirements.txt

4. Install the project in editable mode

pip install -e .

Deep-learning training requires an appropriate PyTorch installation for the target hardware. For GPU training, install the PyTorch build matching the available CUDA environment.

⚡ Quick Start

Inspect the dataset

visioninspect inspect-data

This writes dataset statistics and a class-distribution figure.

Prepare train/validation/test splits

visioninspect prepare-data

Expected structure:

train → 8,800 images
val   → 1,884 images
test  → 1,884 images

Calibrate severity

visioninspect calibrate --apply

Train the U-Net

visioninspect train --epochs 12

Training requires PyTorch and is realistically intended for a GPU-enabled environment for practical experimentation.

Run inspection

visioninspect run \
  --input data/raw/train_images \
  --limit 20

Evaluate

visioninspect evaluate \
  --split data/processed/test.csv

🧪 Testing

Run the test suite with:

pytest -q

The project includes tests covering areas such as:

RLE encoding/decoding

Dataset schema handling

Preprocessing

Detector behavior

Severity calculations

Calibration

Pipeline behavior

Configuration validation

Report generation

Leakage-related logic

🖥️ CLI Reference

visioninspect <command>

Command

Purpose

inspect-data

Analyze the raw dataset

prepare-data

Build leakage-safe train/validation/test splits

calibrate

Derive/apply severity calibration

train

Train the U-Net

run

Inspect an image or directory

evaluate

Evaluate a split against ground-truth masks

General options:

--help
--config CONFIG
--log-level LOG_LEVEL
--version

⚙️ Configuration

Project behavior is centralized through:

configs/config.yaml

Configuration covers project-specific settings such as:

preprocessing

model parameters

severity weights

severity multipliers

decision thresholds

percentile calibration

critical-defect rules

Centralizing these parameters helps keep experiments reproducible and avoids hard-coding operational assumptions throughout the codebase.

🧩 Engineering Design Decisions

1. One multi-label U-Net

A single model predicts four independent segmentation channels.

Benefit: one consistent segmentation pipeline can support multiple defect categories.

Trade-off: model capacity and thresholding must be sufficient for classes with different visual characteristics and frequencies.

2. ImageId-based splitting

The split is grouped by ImageId instead of treating annotation rows as independent samples.

Reason: multiple annotation records may belong to the same image.

Goal: prevent the same source image from contributing information to multiple evaluation subsets.

3. Deep-learning + classical baseline

The project keeps a classical CV backend alongside the learned model.

Reason: the baseline provides a lightweight path for smoke tests, pipeline verification and comparison.

4. Configurable severity logic

Severity thresholds and weights are configuration-driven.

Reason: operational definitions of severity can vary by application and should not be hidden inside model code.

🧱 Limitations

The current framework has several important limitations:

Severity is an engineered score. It should not be interpreted as a direct physical measurement of material damage.

The classical baseline is class-agnostic. It is intended as a baseline/fallback path rather than a replacement for learned multi-class segmentation.

The compact U-Net is intentionally lightweight. Stronger pretrained encoders may provide a useful future comparison.

Demo/synthetic images are for smoke testing. They should not be used to claim model performance on real industrial data.

Operational thresholds are configurable assumptions. They require domain validation before being used in a real manufacturing decision process.

Model results are experiment-dependent. Final performance must be reported from actual held-out evaluation.

🛣️ Future Roadmap

CURRENT
   │
   ├── Multi-label U-Net
   ├── Classical CV baseline
   ├── Leakage-safe splitting
   ├── Severity engine
   ├── Calibration
   ├── Quality decision engine
   └── Automated evaluation/reporting
   │
   ▼
FUTURE
   │
   ├── Stronger pretrained encoders
   ├── Per-class threshold optimization
   ├── Test-time augmentation
   ├── Real-time production-line video
   ├── FastAPI service
   ├── Dockerized deployment
   └── Factory/MES integration

Future items are planned enhancements, not claims about the current implementation.

📚 Documentation

Project documentation is organized under:

docs/
├── adr/
├── results/
├── REPORT_SKELETON.md
└── TRAINING.md

Useful project artifacts include:

Architecture documentation

Architecture decision records

Dataset statistics

Training guidance

Evaluation results

Submission checklist

🔁 Reproducibility Checklist

Before reporting an experiment, verify:

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

🎓 Academic Project Information

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

VisionInspect-AI was developed as an academic Computer Vision project focused on combining deep learning, image processing, evaluation, and explainable inspection logic into a reproducible software pipeline.

📜 License

This project is released under the MIT License.

See LICENSE for details.

⭐ Project Summary

VisionInspect-AI turns steel-surface images into structured inspection evidence — detecting defects, localizing regions, estimating severity, applying configurable quality rules, and producing reproducible reports.
