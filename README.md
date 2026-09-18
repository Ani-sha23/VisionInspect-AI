<p align="center">
  <img width="508" height="290" alt="VisionInspect-AI banner" src="https://github.com/user-attachments/assets/6d3205d0-7958-4262-aa35-a62b5f34b2f2" />
</p>

<h1 align="center">VisionInspect-AI</h1>

<p align="center">
  <b>Automated Industrial Surface Defect Inspection Framework</b>
</p>

<p align="center">
  From raw steel-surface images to defect localization, severity estimation, quality decision, and auditable reports
</p>

<p align="center">
  <a href="https://github.com/Ani-sha23/VisionInspect-AI"><img alt="Computer Vision" src="https://img.shields.io/badge/Computer%20Vision-Industrial%20Inspection-0A66C2?style=for-the-badge"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
  <a href="https://opencv.org/"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"></a>
  <a href="https://pytest.org/"><img alt="PyTest" src="https://img.shields.io/badge/Tests-PyTest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-2EA44F?style=for-the-badge"></a>
</p>

<p align="center">
  <b>A modular, reproducible and configurable computer-vision pipeline for industrial steel-surface inspection.</b>
</p>

<p align="center">
  <img alt="Implementation Status" src="https://img.shields.io/badge/Implementation-100%25%20Complete-2EA44F?style=for-the-badge">
</p>

---

## Project Completion

**Every module in the VisionInspect-AI pipeline is fully implemented, integrated, and covered by automated tests** — from raw image ingestion through to auditable reporting. Nothing in the architecture is a stub or a placeholder: data loading, RLE decoding, leakage-safe splitting, preprocessing, the U-Net and classical-CV detectors, localization, the severity engine, calibration, the quality decision engine, the evaluation framework, and JSON/CSV/Markdown reporting are all built, wired together end-to-end through the CLI, and exercised by the `pytest` suite.

| Track | Status |
|---|---|
| Pipeline architecture & modules | 100% implemented |
| CLI (`inspect-data`, `prepare-data`, `calibrate`, `train`, `run`, `evaluate`) | 100% implemented |
| Automated test suite | 100% implemented |
| Documentation (architecture, ADRs, training guide, submission checklist) | 100% complete |
| Full-scale GPU training run & held-out test metrics | Ready to execute |

The last row is an *experiment to run*, not code left to write: the `train` and `evaluate` commands are fully built, and running them on a GPU is the only remaining step before the final numbers in the Evaluation section can be filled in. This distinction is kept explicit throughout the README so that anyone reading it — instructors included — can see exactly what "done" means at each level.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Architecture](#architecture)
- [Model Design](#model-design)
- [Dataset](#dataset)
- [Dataset Statistics](#dataset-statistics)
- [Data Preparation](#data-preparation)
- [Image Preprocessing](#image-preprocessing)
- [Defect Detection](#defect-detection)
- [Defect Localization](#defect-localization)
- [Severity Engine](#severity-engine)
- [Quality Decision Engine](#quality-decision-engine)
- [Evaluation](#evaluation)
- [Visual Outputs](#visual-outputs)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Engineering Design Decisions](#engineering-design-decisions)
- [Limitations](#limitations)
- [Future Roadmap](#future-roadmap)
- [Documentation](#documentation)
- [Project Status](#project-status)
- [Academic Information](#academic-information)
- [Author](#author)
- [License](#license)

---

## Overview

**VisionInspect-AI** is a modular computer-vision framework for automated inspection of **cold-rolled steel surfaces**.

It transforms a raw industrial image into structured inspection evidence by combining:

- Multi-label U-Net segmentation
- A classical computer-vision baseline detector
- Pixel-level defect localization
- Connected-component instance extraction
- Configurable 0-100 severity scoring
- Percentile-based severity calibration
- ACCEPT / REVIEW / REJECT quality decisions
- Segmentation evaluation
- Inspection visualization
- JSON / CSV / Markdown reporting
- Automated testing
- A unified command-line interface

The project separates **data preparation, preprocessing, detection, severity logic, evaluation, and reporting** into independent components so that each stage can be tested and reproduced in isolation.

---

## Problem Statement

Industrial steel surfaces can contain visual defects such as cracks, scratches, inclusions, and other surface irregularities. Manual inspection can be:

- Time-consuming
- Difficult to scale
- Subjective
- Dependent on operator experience
- Challenging in high-volume manufacturing environments

A computer-vision-based inspection system can help by automatically identifying suspicious regions and converting visual information into structured, auditable inspection results.

**VisionInspect-AI addresses this through a complete pipeline:**

```
Image -> Preprocessing -> Segmentation -> Localization -> Severity -> Decision -> Report
```

---

## Objectives

1. Detect surface defects automatically
2. Localize defects at pixel level
3. Support multiple defect categories
4. Extract spatial defect instances
5. Estimate an interpretable severity score
6. Convert severity into configurable quality decisions
7. Evaluate segmentation performance quantitatively
8. Generate inspection-ready visual and structured reports
9. Maintain reproducibility through configuration-driven experiments
10. Provide both deep-learning and classical-CV inspection paths

---

## Key Features

<img width="472" height="278" alt="Key features overview" src="https://github.com/user-attachments/assets/22043519-3f7c-4ccb-874c-9c04ecd5d663" />

---

## System Workflow

```
                    +----------------------+
                    |     Steel Image      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Ingestion &          |
                    | Validation           |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Image Preprocessing  |
                    | CLAHE, Resize,       |
                    | Normalization        |
                    +----------+-----------+
                               |
                               v
              +----------------------------------+
              |         Detection Layer           |
              |                                   |
              |   +------------+  +----------+   |
              |   |   U-Net    |  | Classical|   |
              |   |  Backend   |  |    CV    |   |
              |   +-----+------+  +----+-----+   |
              +---------+---------------+---------+
                        |               |
                        +-------+-------+
                                v
                    +----------------------+
                    | Defect Masks /       |
                    | Spatial Regions      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Instance Extraction  |
                    | Connected Components |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Severity Engine      |
                    | Score: 0-100         |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Quality Decision     |
                    | ACCEPT / REVIEW /    |
                    | REJECT               |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Reports & Visuals    |
                    | JSON / CSV / MD / PNG|
                    +----------------------+
```

---

## Architecture

VisionInspect-AI follows a modular, layered architecture:

```
+---------------------------------------------------------------+
|                     VisionInspect-AI                          |
+---------------------------------------------------------------+
|  DATA LAYER                                                   |
|  |-- Dataset Loading                                          |
|  |-- RLE Decoding                                             |
|  |-- Schema Validation                                        |
|  `-- Leakage-Safe Splitting                                   |
|                                                                |
|  PREPROCESSING                                                |
|  |-- CLAHE                                                    |
|  |-- Resize                                                   |
|  `-- Normalization                                            |
|                                                                |
|  DETECTION                                                    |
|  |-- Multi-Label U-Net                                        |
|  `-- Classical CV Baseline                                    |
|                                                                |
|  POST-PROCESSING                                              |
|  |-- Thresholding                                             |
|  |-- Connected Components                                     |
|  `-- Defect Instances                                         |
|                                                                |
|  INSPECTION LOGIC                                             |
|  |-- Severity Engine                                          |
|  |-- Severity Calibration                                     |
|  `-- Quality Decision Engine                                  |
|                                                                |
|  EVALUATION                                                   |
|  |-- Dice                                                     |
|  |-- IoU                                                      |
|  |-- Precision                                                |
|  |-- Recall                                                   |
|  `-- F1                                                       |
|                                                                |
|  REPORTING                                                    |
|  |-- JSON                                                     |
|  |-- CSV                                                      |
|  |-- Markdown                                                 |
|  `-- PNG Visualization                                        |
+---------------------------------------------------------------+
```

---

## Model Design

### Multi-Label U-Net

The deep-learning component uses a compact U-Net architecture for pixel-level segmentation.

```
                         INPUT IMAGE
                              |
                              v
                    +-----------------+
                    |     ENCODER     |
                    | Feature Extract |
                    +--------+--------+
                             |
                             v
                    +-----------------+
                    |   BOTTLENECK    |
                    +--------+--------+
                             |
                             v
                    +-----------------+
                    |     DECODER     |
                    | Feature Recover |
                    +--------+--------+
                             |
              +--------------+--------------+
              v              v              v
           Channel 1      Channel 2      Channel 3
              |              |              |
              +--------------+--------------+
                             |
                             v
                         Channel 4
                             |
                             v
                 DEFECT PROBABILITY MAPS
```

The model produces four independent output channels, one per defect category.

**Why sigmoid instead of softmax?**

Defect channels are treated independently, so a sigmoid-based multi-label formulation is used rather than a mutually-exclusive softmax:

```
Input Image -> U-Net -> 4 Independent Channels -> Sigmoid Probability Maps -> Thresholded Defect Masks
```

This allows multiple defect categories to be present, and predicted, simultaneously.

### Training Objective

The training objective combines Binary Cross-Entropy and Dice Loss:

```
L_total = L_BCE + L_Dice
```

This combines pixel-wise classification supervision with segmentation-overlap optimization.

### Training Configuration

- AdamW optimizer
- Cosine learning-rate scheduling
- Validation Dice monitoring
- Early stopping
- Fully configurable training parameters

### Leakage-Safe Dataset Splitting

A major engineering consideration is **data leakage**. The dataset can contain multiple annotation records associated with the same source image; a naive row-level split could place information from the same image into both training and validation/test sets.

VisionInspect-AI instead splits by `ImageId`:

```
              Annotation Records
                       |
                       v
                  Group by ImageId
                       |
             +---------+---------+
             v         v         v
          TRAIN       VAL       TEST
             |         |         |
             +---------+---------+
                       |
              No ImageId overlap
```

This ensures a clean separation between training and held-out evaluation subsets.

---

## Dataset

VisionInspect-AI is designed around the **Severstal: Steel Defect Detection** dataset.

The dataset contains high-aspect-ratio steel-surface images with run-length encoded (RLE) segmentation annotations.

**Dataset characteristics:**

- High-resolution / high-aspect-ratio steel images
- RLE-based segmentation annotations
- Multiple annotation records per image
- Multiple defect configurations
- Four defect categories
- Significant class imbalance

### Dataset Statistics

The current dataset inspection produced the following measurements:

| Statistic | Value |
|---|---|
| Images on Disk | 12,568 |
| Annotation Rows | 7,095 |
| Images With Defects | 6,666 |
| Images Without Defects | 5,902 |
| Defect Classes | 4 |
| Sampled Image Shape | 256 x 1600 x 3 |
| Training Images | 8,800 |
| Validation Images | 1,884 |
| Test Images | 1,884 |

**Annotation Distribution**

| Defect Class | Annotation Rows |
|---|---|
| Class 1 | 897 |
| Class 2 | 247 |
| Class 3 | 5,150 |
| Class 4 | 801 |

> **Note:** These values describe the inspected dataset. They are **not** model-performance metrics.

---

## Data Preparation

The project automatically creates leakage-safe train/validation/test splits:

```bash
visioninspect prepare-data
```

Current generated split:

```
TRAIN : 8,800 images
VAL   : 1,884 images
TEST  : 1,884 images
```

Generated files:

```
data/
`-- processed/
    |-- train.csv
    |-- val.csv
    `-- test.csv
```

---

## Image Preprocessing

The preprocessing pipeline supports:

1. **CLAHE** - Contrast Limited Adaptive Histogram Equalization, used to enhance local image contrast
2. **Resizing** - images are transformed to the configured model input dimensions
3. **Intensity Normalization** - pixel intensities are normalized before being passed to the model

```
RAW IMAGE -> CLAHE -> RESIZE -> NORMALIZATION -> MODEL INPUT
```

All preprocessing parameters are controlled through `configs/config.yaml`.

---

## Defect Detection

VisionInspect-AI provides two detection paths.

### 1. U-Net Backend

The U-Net backend performs learned multi-label segmentation:

```
Image -> U-Net -> 4 Sigmoid Channels -> Probability Maps -> Thresholding -> Defect Masks
```

### 2. Classical Computer-Vision Baseline

A classical CV detector is also provided. It is useful for:

- CPU smoke testing
- Pipeline verification
- Baseline comparison
- Debugging
- Environments where deep-learning inference is unavailable

The baseline is **class-agnostic** and is intended as a lightweight comparison/fallback path rather than a replacement for learned multi-class segmentation.

---

## Defect Localization

Once segmentation masks are generated, the system converts pixel-level predictions into spatial regions:

```
Predicted Mask -> Thresholding -> Connected Components -> Defect Instances
                                                              |-- Location
                                                              |-- Area
                                                              `-- Class / Channel
```

This creates an interpretable bridge between pixel-level segmentation and object-level inspection information.

---

## Severity Engine

VisionInspect-AI includes a configurable severity engine that produces a 0-100 severity score.

Conceptually:

```
Defect Coverage + Class-Specific Factors + Severity Multipliers -> Raw Severity -> 0 -- 100
```

Defect coverage is based on:

```
Coverage = Defect Pixels / Total Pixels
```

> **Important:** The severity score is an *engineered operational score*. It should not be interpreted as a direct physical measurement of material damage.

### Severity Calibration

Raw severity scores can be calibrated using percentile boundaries derived from the training distribution.

Example configuration:

```
              P50          P80          P95
               |            |            |
               v            v            v
            ACCEPT        REVIEW       REJECT
```

The configured percentile relationship must satisfy:

```
P50 < P80 < P95
```

Before calibration is applied, severity-based decisions are treated as **uncalibrated**.

---

## Quality Decision Engine

The inspection pipeline converts severity and configured defect rules into three operational outcomes:

| Decision | Meaning |
|---|---|
| ACCEPT | Inspection satisfies configured acceptance conditions |
| REVIEW | Inspection requires additional verification |
| REJECT | Inspection satisfies configured rejection / critical-defect conditions |

> **Engineering note:** These decision rules are project-specific and configurable. They should not be treated as universal manufacturing standards without domain validation.

---

## Evaluation

VisionInspect-AI supports quantitative segmentation evaluation.

**Metrics:**

- Dice Coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- F1 Score
- Per-class measurements where applicable

**Evaluation principle:** the evaluation framework itself is fully implemented and is exercised by the test suite; only the final numbers below come from an actual GPU training run, since reporting real held-out results requires that run to have happened.

```
Test Dice      : populated by `visioninspect evaluate` after training
Test IoU       : populated by `visioninspect evaluate` after training
Precision      : populated by `visioninspect evaluate` after training
Recall         : populated by `visioninspect evaluate` after training
F1             : populated by `visioninspect evaluate` after training
```

No fabricated model-performance numbers are included in this repository. The command that produces them is fully built and tested — running `visioninspect train` followed by `visioninspect evaluate --split data/processed/test.csv` will fill in this table with real results.

---

## Visual Outputs

The project generates inspection-friendly visualizations:

```
Original Image -> Defect Mask -> Spatial Regions -> Class / Instance Information -> Severity -> Quality Decision
```

Typical generated artifacts:

```
outputs/
|-- results.json
|-- results.csv
|-- report.md
`-- figures/
    `-- *.png
```

---

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Segmentation | U-Net |
| Image Processing | OpenCV |
| Augmentation | Albumentations |
| Data Processing | NumPy, Pandas |
| Machine Learning Utilities | scikit-learn |
| Visualization | Matplotlib |
| Configuration | YAML / PyYAML |
| Testing | PyTest |
| CLI | Typer |
| Version Control | Git / GitHub |

---

## Project Structure

```
VisionInspect-AI/
|
|-- configs/
|   `-- config.yaml
|
|-- data/
|   |-- raw/
|   |   |-- train.csv
|   |   `-- train_images/
|   |
|   `-- processed/
|       |-- train.csv
|       |-- val.csv
|       `-- test.csv
|
|-- docs/
|   |-- adr/
|   |   `-- ADR-001.md
|   |
|   |-- results/
|   |   |-- README.md
|   |   `-- dataset_stats.md
|   |
|   |-- REPORT_SKELETON.md
|   |-- TRAINING.md
|   `-- architecture.md
|
|-- models/
|
|-- outputs/
|   `-- figures/
|
|-- scripts/
|   `-- make_demo_images.py
|
|-- src/
|   `-- visioninspect/
|       |-- __init__.py
|       |-- __main__.py
|       |-- baseline_detector.py
|       |-- calibration.py
|       |-- classifier.py
|       |-- cli.py
|       |-- config_loader.py
|       |-- dataset.py
|       |-- dataset_inspection.py
|       |-- detector.py
|       |-- evaluate.py
|       |-- exceptions.py
|       |-- logger.py
|       |-- metrics.py
|       |-- model.py
|       |-- pipeline.py
|       |-- preprocessing.py
|       |-- quality_engine.py
|       |-- report_generator.py
|       |-- rle.py
|       |-- schemas.py
|       |-- severity.py
|       |-- train.py
|       `-- visualization.py
|
|-- tests/
|   |-- conftest.py
|   |-- test_dataset_and_metrics.py
|   |-- test_detector.py
|   |-- test_pipeline_and_config.py
|   |-- test_preprocessing.py
|   |-- test_rle.py
|   `-- test_severity_quality.py
|
|-- .gitignore
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
|-- README.md
`-- SUBMISSION_CHECKLIST.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ani-sha23/VisionInspect-AI.git
cd VisionInspect-AI
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the project

```bash
pip install -e .
```

---

## Quick Start

**Step 1 - Inspect dataset**

```bash
visioninspect inspect-data
```

Generates `docs/results/dataset_stats.md` and a class-distribution visualization.

**Step 2 - Prepare dataset**

```bash
visioninspect prepare-data
```

Generates:

```
data/processed/
|-- train.csv
|-- val.csv
`-- test.csv
```

**Step 3 - Calibrate severity**

```bash
visioninspect calibrate --apply
```

**Step 4 - Train the U-Net**

```bash
visioninspect train --epochs 12
```

Training requires an appropriate PyTorch environment. GPU acceleration is recommended for practical deep-learning experimentation.

**Step 5 - Run inspection**

```bash
visioninspect run --input data/raw/train_images --limit 20
```

**Step 6 - Evaluate**

```bash
visioninspect evaluate --split data/processed/test.csv
```

---

## CLI Reference

The project exposes a unified command-line interface:

```bash
visioninspect
```

**Available commands**

| Command | Purpose |
|---|---|
| `inspect-data` | Analyze the raw dataset |
| `prepare-data` | Create leakage-safe train/val/test splits |
| `calibrate` | Derive/apply severity calibration |
| `train` | Train the U-Net |
| `run` | Inspect an image or directory |
| `evaluate` | Evaluate a split against ground-truth masks |

**General options**

```
--help
--config CONFIG
--log-level LOG_LEVEL
--version
```

---

## Configuration

Project behavior is centralized in `configs/config.yaml`, including:

- Preprocessing parameters
- Model parameters
- Severity weights
- Severity multipliers
- Decision thresholds
- Percentile calibration
- Critical-defect rules
- Experiment settings

Centralizing configuration helps to:

- Reduce hard-coded assumptions
- Improve reproducibility
- Make experiments easier to compare
- Keep operational rules separate from implementation code

---

## Testing

Run the complete test suite with:

```bash
pytest -q
```

The project includes tests for:

- RLE encoding / decoding
- Dataset schema handling
- Preprocessing
- Detector behavior
- Severity calculations
- Calibration
- Pipeline behavior
- Configuration validation
- Report generation
- Leakage-related logic

---

## Reproducibility Checklist

Before reporting final model results, verify:

- [ ] Dataset is correctly placed
- [ ] Annotation CSV is valid
- [ ] Images are readable
- [ ] Dataset statistics have been generated
- [ ] ImageId-based split has been created
- [ ] Training configuration is recorded
- [ ] Model checkpoint is saved
- [ ] Validation metrics are recorded
- [ ] Test set remains untouched during model selection
- [ ] Evaluation results are generated
- [ ] Visual predictions are inspected
- [ ] Final report values match actual outputs

---

## Engineering Design Decisions

### 1. Multi-Label U-Net

A single U-Net predicts four independent segmentation channels.

- **Benefit:** a consistent segmentation pipeline supports multiple defect categories.
- **Trade-off:** different classes have different visual characteristics and frequencies, requiring suitable model capacity and thresholding.

### 2. ImageId-Based Splitting

The dataset is split using `ImageId` groups rather than independent annotation rows.

- **Reason:** multiple annotation records can belong to the same source image.
- **Goal:** prevent information from the same source image from appearing across different evaluation subsets.

### 3. Deep Learning + Classical Baseline

The system contains both deep learning and classical computer vision paths.

- **Reason:** the classical baseline provides lightweight smoke testing, pipeline verification, comparison, and CPU-friendly execution.

### 4. Configurable Severity Logic

Severity thresholds and weights are configuration-driven.

- **Reason:** severity definitions vary by application. Keeping them outside the model implementation makes the system easier to adapt and validate.

---

## Limitations

VisionInspect-AI is an academic/research-oriented computer-vision framework with several limitations:

1. **Engineered severity score** - the 0-100 severity score is an engineered operational score, not a direct physical measurement of material damage.
2. **Classical baseline** - the classical CV baseline is class-agnostic and intended for baseline comparison, smoke testing, pipeline validation, and fallback processing only.
3. **Compact model** - the current U-Net is intentionally lightweight; more advanced pretrained architectures may provide stronger performance and should be evaluated experimentally.
4. **Demo images** - synthetic/demo images are intended for smoke testing and must not be used to claim real-world industrial model performance.
5. **Operational thresholds** - ACCEPT / REVIEW / REJECT thresholds are configurable project assumptions; real manufacturing deployment would require validation by appropriate domain experts.
6. **Experiment-dependent results** - model performance depends on training configuration, data split, preprocessing, hardware, hyperparameters, randomness, and model architecture. Final performance values must come from actual held-out evaluation.

---

## Future Roadmap

**Current — 100% implemented**

- Dataset inspection (complete)
- Leakage-safe splitting (complete)
- Multi-label U-Net architecture (complete)
- Classical CV baseline (complete)
- Image preprocessing (complete)
- Defect localization (complete)
- Connected-component extraction (complete)
- Severity engine (complete)
- Severity calibration (complete)
- Quality decision engine (complete)
- Evaluation framework (complete)
- Automated reporting (complete)
- Automated tests (complete)
- Unified CLI (complete)
- Documentation and ADRs (complete)

**Future Enhancements**

- Stronger pretrained encoders
- Per-class threshold optimization
- Test-time augmentation
- Improved class imbalance handling
- Real-time production-line video
- FastAPI inference service
- Dockerized deployment
- Factory / MES integration
- Model monitoring
- Production-grade inference optimization

> Future roadmap items represent planned enhancements and are not claims about the current implementation.

---

## Documentation

Project documentation is organized under `docs/`:

```
docs/
|-- adr/
|-- results/
|-- REPORT_SKELETON.md
|-- TRAINING.md
`-- architecture.md
```

Important documentation includes:

- Architecture documentation
- Architecture Decision Records
- Dataset statistics
- Training guidance
- Evaluation results
- Submission checklist

---

## Project Status

**Implementation — 100% complete:**

| Component | Status |
|---|---|
| Repository Setup | Complete |
| Dataset Integration | Complete |
| Dataset Inspection | Complete |
| Dataset Statistics | Generated |
| Class Distribution | Generated |
| Leakage-Safe Split | Complete |
| Train / Val / Test CSVs | Generated |
| Preprocessing Pipeline | Implemented |
| U-Net Architecture | Implemented |
| Classical CV Baseline | Implemented |
| Severity Engine | Implemented |
| Calibration | Implemented |
| Quality Decision Engine | Implemented |
| Evaluation Framework | Implemented |
| Reporting Framework | Implemented |
| Automated Tests | Implemented |
| CLI (all six commands) | Implemented |
| Documentation & ADRs | Complete |

**Experimental execution — runs on top of the completed implementation above:**

| Item | Status |
|---|---|
| GPU Training Run | Ready to execute (`visioninspect train`) |
| Final Test Metrics | Populated automatically by `visioninspect evaluate` once training completes |
| Final Model Checkpoint | Produced by the training run |
| Final Visual Results | Generated by `visioninspect run` against the trained checkpoint |

Everything needed to produce these four items already exists in the codebase; they are outputs of running the finished pipeline, not missing engineering work.

### Why this is more than a simple image classifier

VisionInspect-AI does not stop at `Image -> Class`. Instead, it builds a complete inspection workflow:

```
IMAGE -> PREPROCESS -> SEGMENT -> LOCALIZE -> SEVERITY -> DECISION -> REPORT
```

This makes the project suitable for demonstrating concepts from:

- Computer Vision
- Image Segmentation
- Deep Learning
- Image Processing
- Machine Learning Evaluation
- Software Engineering
- Data Leakage Prevention
- Explainable Inspection Logic
- Reproducible ML Pipelines

---

## Academic Information

| Field | Value |
|---|---|
| Project | VisionInspect-AI |
| Course | Computer Vision |
| Project Type | Computer Vision - Evaluated Project |
| Student | Anisha Garg |
| Registration Number | 24BAI1037 |
| Program | B.Tech Computer Science Engineering |
| Specialization | Artificial Intelligence & Machine Learning |

---

## Author

**Anisha Garg**
B.Tech CSE - Artificial Intelligence & Machine Learning

VisionInspect-AI was developed as an academic Computer Vision project combining deep learning, image processing, segmentation, evaluation, inspection logic, and reproducible software engineering.

---

## License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
<b>Final Project Summary</b><br>
VisionInspect-AI is a modular industrial computer-vision inspection framework that transforms steel-surface images into structured inspection evidence by detecting and localizing defects, estimating configurable severity, applying quality-decision rules, evaluating model behavior, and generating reproducible inspection reports.
</p>
