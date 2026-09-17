
````markdown
# 🔍 VisionInspect-AI
### Automated Industrial Surface Defect Inspection Framework

<p align="center">
  <b>AI-powered visual quality inspection for cold-rolled steel surfaces</b>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Optional-ee4c2c?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

## 🚀 Overview

**VisionInspect-AI** is an automated computer-vision quality-control framework designed to detect and analyze surface defects in **cold-rolled steel sheets**.

The system processes high-aspect-ratio steel images, identifies defective regions, classifies defects into **four defect categories**, calculates a **0–100 severity score**, and generates an operational quality decision:

| Decision | Meaning |
|----------|---------|
| 🟢 **ACCEPT** | Surface quality is within acceptable limits |
| 🟡 **REVIEW** | Human/operator verification is recommended |
| 🔴 **REJECT** | Defect severity or critical defect requires rejection |

The system is built around the **Severstal: Steel Defect Detection** dataset and combines deep-learning segmentation with rule-based industrial decision logic.

---

# ✨ Key Features

- 🧠 **Multi-label U-Net segmentation**
- 🎯 **Pixel-level defect localization**
- 🏷️ **Four-class defect classification**
- 🔢 **Connected-component defect counting**
- 📊 **0–100 severity scoring**
- ⚙️ **Configurable quality decision engine**
- 🟢🟡🔴 **ACCEPT / REVIEW / REJECT decisions**
- 📈 **Dice, IoU, Precision, Recall and F1 metrics**
- 🛡️ **Leakage-safe dataset splitting**
- 🔄 **Deep-learning + classical CV backends**
- 🖼️ **Visual defect overlays**
- 📄 **JSON / CSV / Markdown reports**
- 🧪 **Automated testing with PyTest**
- 💻 **Command-line interface**
- ⚡ **CPU baseline execution**
- 🔧 **Configurable severity calibration**

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Steel Surface      │
                    │       Image          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     INGESTION        │
                    │     pipeline.py      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      VALIDATION      │
                    │   preprocessing.py   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    PREPROCESSING     │
                    │ CLAHE + Resize + Norm│
                    └──────────┬───────────┘
                               │
                               ▼
             ┌─────────────────────────────────┐
             │          DETECTION               │
             │                                 │
             │   ┌────────────┐  ┌──────────┐ │
             │   │ U-Net      │  │ Baseline │ │
             │   │ Backend    │  │ CV       │ │
             │   └────────────┘  └──────────┘ │
             └────────────────┬────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │    CLASSIFICATION    │
                    │   classifier.py      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   SEVERITY SCORING   │
                    │     0 ─────── 100    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   QUALITY ENGINE      │
                    │                      │
                    │ 🟢 ACCEPT             │
                    │ 🟡 REVIEW             │
                    │ 🔴 REJECT             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      REPORTING       │
                    │ JSON / CSV / MD / PNG│
                    └──────────────────────┘
````

---

# 🔬 How It Works

VisionInspect-AI uses an **8-stage inspection pipeline**.

### 1️⃣ Ingestion

Loads steel-surface images from local directories or manufacturing feeds.

### 2️⃣ Validation

Checks:

* Image dimensions
* Color channels
* Bit depth
* File integrity

### 3️⃣ Preprocessing

Applies:

* CLAHE
* Dynamic resizing
* Intensity normalization

### 4️⃣ Detection

Detects defective regions using either:

**U-Net backend**

```text
Image
  ↓
Multi-label U-Net
  ↓
4 sigmoid probability channels
  ↓
Binary defect masks
```

or the **baseline computer-vision backend**.

### 5️⃣ Classification

The system derives defect-class presence directly from active segmentation channels.

### 6️⃣ Severity Scoring

Calculates a continuous score:

```text
0 ─────────────────────────────── 100
│                                  │
Pristine                       Critical
```

### 7️⃣ Decision Engine

Converts the prediction into an operational decision:

```text
             Severity
                │
                ▼
       ┌─────────────────┐
       │ Quality Engine  │
       └────────┬────────┘
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
   🟢 ACCEPT  🟡 REVIEW  🔴 REJECT
```

### 8️⃣ Reporting

Generates:

```text
results.json
results.csv
report.md
PNG visual overlays
```

---

# 🧠 Multi-Label U-Net

Unlike a traditional architecture that separates classification, detection and segmentation, VisionInspect-AI uses a **single multi-label U-Net**.

The network produces four independent sigmoid output channels:

```text
                 Input Image
                      │
                      ▼
                ┌───────────┐
                │  U-Net    │
                └─────┬─────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     Class 1       Class 2       Class 3
        │             │             │
        └─────────────┼─────────────┘
                      │
                   Class 4
```

These segmentation channels are used for:

### 🎯 Localization

Connected components are converted into spatial bounding boxes.

### 🏷️ Classification

Active segmentation channels identify the defect classes.

### 🔢 Instance Counting

Disconnected components are counted as individual defect regions.

---

# ⚙️ Detection Backends

VisionInspect-AI supports two detection engines.

| Feature               | 🧠 U-Net | 🔧 Baseline |
| --------------------- | -------- | ----------- |
| Deep Learning         | ✅        | ❌           |
| PyTorch               | ✅        | ❌           |
| OpenCV                | Optional | ✅           |
| Multi-class Detection | ✅        | ❌           |
| Class-specific Masks  | ✅        | ❌           |
| Bounding Boxes        | ✅        | ✅           |
| Confidence Scores     | ✅        | ❌           |
| Quantitative Metrics  | ✅        | Limited     |
| CPU Smoke Testing     | ⚠️       | ✅           |
| Production Inference  | ✅        | Fallback    |

### 🧠 U-Net Backend

Used for quantitative deep-learning evaluation and production-style inference.

### 🔧 Baseline Backend

Uses classical computer vision and adaptive thresholding to locate anomalous regions.

It is useful for:

* CPU verification
* CI testing
* Smoke testing
* Fallback environments

Baseline detections remain **class-agnostic**.

---

# 📊 Severity Scoring

The severity engine converts predicted defect coverage into a score between:

```text
0.0 → 100.0
```

Where:

```text
0.0  = Pristine surface
100.0 = Critical defect density
```

For each defect class, the system calculates surface coverage:

```text
             Defect Pixels
Coverage = ─────────────────
            Total Pixels
```

The final score combines:

* Defect coverage
* Class-specific weights
* Severity multipliers

Conceptually:

```text
Severity
   │
   ├── Defect Area
   ├── Class Weight
   └── Severity Multiplier
              │
              ▼
       Final Score 0–100
```

---

# 📈 Percentile Calibration

Raw severity scores are calibrated using historical training distributions.

The calibration system uses percentile boundaries:

```text
             P50          P80          P95
              │            │            │
              ▼            ▼            ▼

🟢 ACCEPT ────┼── 🟡 REVIEW ────────────┼── 🔴 REJECT
```

Calibration can be applied with:

```bash
visioninspect calibrate --apply
```

> ⚠️ Before calibration, the system marks severity decisions as `UNCALIBRATED`.

---

# 🏭 Quality Decision Engine

The quality engine maps severity and defect information into operational decisions.

### 🟢 ACCEPT

Used when:

* No critical defect is detected
* Severity is below the configured acceptance boundary

### 🟡 REVIEW

Used when:

* Severity lies within the review range
* Low-confidence segmentation requires human verification

### 🔴 REJECT

Used when:

* Severity exceeds the configured rejection boundary
* A configured critical defect is detected

---

# 🛡️ Leakage-Safe Dataset Splitting

A major concern in steel-defect datasets is **spatial data leakage**.

Adjacent image crops may contain parts of the same physical defect.

A random split could therefore produce:

```text
Same Physical Defect
        │
        ├── Training Set ❌
        │
        └── Validation Set ❌
```

This can artificially inflate evaluation metrics.

VisionInspect-AI instead groups data by unique `ImageId`:

```text
                 ImageId
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Training Split       Validation Split
          │                   │
          └────── NO OVERLAP ─┘
```

This preserves evaluation integrity.

---

# 🖥️ CLI Commands

VisionInspect-AI provides a unified CLI:

```bash
visioninspect <command>
```

## 📋 Available Commands

| Command        | Purpose                       |
| -------------- | ----------------------------- |
| `inspect-data` | Analyze raw dataset           |
| `prepare-data` | Create leakage-safe splits    |
| `calibrate`    | Calibrate severity thresholds |
| `train`        | Train U-Net                   |
| `run`          | Run inspection                |
| `evaluate`     | Evaluate test performance     |

---

# 🧪 CLI Examples

### Inspect Dataset

```bash
visioninspect inspect-data
```

### Prepare Dataset

```bash
visioninspect prepare-data
```

### Calibrate Severity

```bash
visioninspect calibrate --apply
```

### Train Model

```bash
visioninspect train --epochs 12
```

### Run Inspection

```bash
visioninspect run \
  --input data/raw/train_images \
  --limit 20
```

### Evaluate Model

```bash
visioninspect evaluate \
  --split data/processed/test.csv
```

---

# 📁 Project Structure

```text
VisionInspect-AI/
│
├── 📁 configs/
│   └── config.yaml
│
├── 📁 src/
│   └── 📁 visioninspect/
│       ├── cli.py
│       ├── config_loader.py
│       ├── preprocessing.py
│       ├── rle.py
│       ├── dataset.py
│       ├── dataset_inspection.py
│       ├── model.py
│       ├── train.py
│       ├── detector.py
│       ├── baseline_detector.py
│       ├── classifier.py
│       ├── severity.py
│       ├── quality_engine.py
│       ├── visualization.py
│       ├── report_generator.py
│       ├── metrics.py
│       ├── calibration.py
│       ├── evaluate.py
│       └── pipeline.py
│
├── 📁 scripts/
│   └── make_demo_images.py
│
├── 📁 tests/
│
├── 📁 docs/
│
├── 📁 outputs/
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 pyproject.toml
```

---

# 💻 Installation

## Requirements

| Component  | Requirement             |
| ---------- | ----------------------- |
| 🐍 Python  | 3.9+                    |
| 💻 OS      | Windows / Linux / macOS |
| 🔥 PyTorch | Optional                |
| 👁️ OpenCV | Required for baseline   |

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/VisionInspect-AI.git

cd VisionInspect-AI
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Install the project:

```bash
pip install -e .
```

---

# ⚡ Quick Start

You can test the complete pipeline without a GPU or external dataset download.

### Generate Demo Images

```bash
python scripts/make_demo_images.py \
  --out samples \
  --count 6
```

### Run Baseline Inspection

```bash
visioninspect run \
  --input samples \
  --backend baseline
```

Expected outputs:

```text
outputs/
│
├── results.json
├── results.csv
├── report.md
└── figures/
    ├── image_01.png
    ├── image_02.png
    └── ...
```

---

# 🧠 Production Dataset Workflow

For training with the Severstal dataset:

## Step 1 — Install PyTorch

```bash
pip install torch \
  --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 2 — Download Dataset

```bash
kaggle competitions download \
  -c severstal-steel-defect-detection \
  -p data/raw
```

Extract:

```bash
unzip -q \
  data/raw/severstal-steel-defect-detection.zip \
  -d data/raw
```

---

## Step 3 — Inspect Dataset

```bash
visioninspect inspect-data
```

---

## Step 4 — Prepare Splits

```bash
visioninspect prepare-data
```

---

## Step 5 — Calibrate Severity

```bash
visioninspect calibrate --apply
```

---

## Step 6 — Train U-Net

```bash
visioninspect train --epochs 12
```

---

## Step 7 — Run Inspection

```bash
visioninspect run \
  --input data/raw/train_images \
  --limit 20
```

---

## Step 8 — Evaluate

```bash
visioninspect evaluate \
  --split data/processed/test.csv
```

---

# 📊 Evaluation Metrics

The system supports:

```text
                    Evaluation
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
       Dice             IoU          Precision
        │                │                │
        └────────────────┼────────────────┘
                         │
                   ┌─────▼─────┐
                   │  Recall   │
                   └─────┬─────┘
                         │
                         ▼
                         F1
```

---

# 🧪 Testing

VisionInspect-AI includes automated unit and integration tests.

Run:

```bash
pytest -q
```

The test suite validates:

* RLE encoding/decoding
* Dataset schema compatibility
* Severity monotonicity
* Percentile calibration
* Leakage prevention
* End-to-end pipeline execution
* Report generation

---

# 🖼️ Visual Audit Outputs

The reporting system can generate annotated images containing:

```text
Original Image
      │
      ▼
Defect Mask
      │
      ▼
Bounding Box
      │
      ▼
Class Label
      │
      ▼
Severity / Decision
```

Example conceptual output:

```text
┌──────────────────────────────────────┐
│        STEEL SURFACE IMAGE           │
│                                      │
│      ┌──────────────┐                │
│      │   DEFECT     │                │
│      │   REGION     │                │
│      └──────────────┘                │
│                                      │
│  Class: Defect 2                     │
│  Severity: 67.4                      │
│  Decision: REVIEW                    │
└──────────────────────────────────────┘
```

---

# 🔧 Configuration

System behavior is controlled through:

```text
configs/config.yaml
```

Configuration includes:

* Severity weights
* Severity multipliers
* Thresholds
* Percentile bands
* Preprocessing parameters
* Critical defect classes
* Model configuration

The configuration loader validates important constraints before execution.

For example:

```text
P50 < P80 < P95
```

must always remain strictly ordered.

---

# ⚠️ Limitations

### 1. Baseline Classification

The classical CV backend can localize anomalies but does not determine specific defect classes.

### 2. Synthetic Images

Demo images are intended for smoke testing only.

They should **not** be used to make production performance claims.

### 3. Model Capacity

The included U-Net is compact and trained from scratch.

Industrial deployments may benefit from stronger pretrained backbones.

### 4. Severity Assumptions

Severity weights are configurable operational assumptions and do not directly represent physical material measurements.

---

# 🛣️ Future Roadmap

```text
Current
   │
   ├── Multi-label U-Net
   ├── Baseline CV
   ├── Severity Engine
   └── Quality Decision Engine
   │
   ▼
Future
   │
   ├── 🎥 Real-time production video
   ├── 🌐 FastAPI / Docker deployment
   ├── 🧠 Pretrained U-Net backbones
   ├── 🔄 Test-Time Augmentation
   └── 🎯 Per-class threshold calibration
```

### Planned Improvements

* Real-time production-line video processing
* Factory MES integration
* FastAPI microservice
* Docker deployment
* ImageNet-pretrained encoders
* Test-Time Augmentation
* Per-class probability threshold calibration

---

# 🧰 Technology Stack

```text
┌────────────────────────────────────┐
│          VisionInspect-AI           │
├────────────────────────────────────┤
│ 🐍 Python                          │
│ 🔥 PyTorch                         │
│ 👁️ OpenCV                          │
│ 🔢 NumPy                           │
│ 📊 Pandas                          │
│ 🧠 U-Net                           │
│ 📈 Scikit-learn / Metrics          │
│ 🧪 PyTest                           │
│ ⚙️ YAML Configuration              │
│ 💻 CLI                             │
└────────────────────────────────────┘
```

---

# 📌 Project Highlights

| Capability              | Status |
| ----------------------- | ------ |
| Image Validation        | ✅      |
| CLAHE Preprocessing     | ✅      |
| U-Net Segmentation      | ✅      |
| Multi-label Detection   | ✅      |
| Baseline CV Detector    | ✅      |
| Defect Classification   | ✅      |
| Defect Counting         | ✅      |
| Severity Scoring        | ✅      |
| Percentile Calibration  | ✅      |
| Quality Decision Engine | ✅      |
| Visual Overlays         | ✅      |
| JSON Reports            | ✅      |
| CSV Reports             | ✅      |
| Markdown Reports        | ✅      |
| Leakage-safe Splitting  | ✅      |
| Automated Tests         | ✅      |
| Real-time Video         | 🚧     |
| FastAPI Deployment      | 🚧     |
| Docker Deployment       | 🚧     |

---

# 🎯 Project Goal

> **Transform raw steel-surface images into an explainable industrial quality decision using computer vision, deep learning, severity scoring, and rule-based decision logic.**

```text
IMAGE
  ↓
VALIDATE
  ↓
PREPROCESS
  ↓
DETECT
  ↓
CLASSIFY
  ↓
MEASURE SEVERITY
  ↓
QUALITY DECISION
  ↓
REPORT
```

---

# 👩‍💻 Author

**Anisha Garg**

B.Tech Computer Science Engineering
Artificial Intelligence & Machine Learning

---

# ⭐ Acknowledgement

This project uses the **Severstal: Steel Defect Detection** benchmark dataset for industrial surface-defect inspection research and development.

---

<p align="center">

### 🔍 VisionInspect-AI

**See the defect. Measure the severity. Make the decision.**

⭐ If this project helped you, consider starring the repository.

</p>
```

This version keeps the original project's technical structure—including the **8-stage pipeline, dual detection backends, severity calibration, leakage-safe splitting, CLI workflow, testing, limitations, and roadmap**—but turns it into a much more visual GitHub presentation.   

**One important correction before you paste it:** replace `<your-username>` in the clone command with your actual GitHub username/repository URL.
