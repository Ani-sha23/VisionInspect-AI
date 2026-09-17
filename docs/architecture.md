# Architecture

## 1. System overview

```mermaid
flowchart LR
    A[Image file or directory] --> B[Validation]
    B --> C[Preprocessing<br/>CLAHE, resize, normalise]
    C --> D{Detection backend}
    D -->|weights present| E[U-Net<br/>4-channel sigmoid]
    D -->|no weights| F[Classical baseline<br/>adaptive threshold]
    E --> G[Connected components<br/>masks to instances]
    F --> G
    G --> H[Classification<br/>classes present]
    G --> I[Severity scoring<br/>area + count + class weight]
    H --> J[Quality decision<br/>ACCEPT / REVIEW / REJECT]
    I --> J
    J --> K[Reports<br/>JSON, CSV, Markdown]
    J --> L[Visualisation<br/>overlay + banner]
```

## 2. Module responsibilities

| Module | Responsibility | Input | Output |
|---|---|---|---|
| `config_loader` | Load and validate `config.yaml` | YAML path | `Config` |
| `preprocessing` | Reject bad images, enhance contrast, normalise | file path | BGR image, CHW tensor |
| `detector` | Pick a backend, produce defect instances | image, tensor | `List[Defect]` |
| `baseline_detector` | Training-free localisation | BGR image | `List[Defect]` (class 0) |
| `model` / `train` | U-Net definition and training loop | tensors, masks | checkpoint |
| `classifier` | Image-level labels from instances | `List[Defect]` | class ids, areas |
| `severity` | Weighted 0–100 score and band | defects, image size | `SeverityResult` |
| `quality_engine` | Decision plus written reasons | severity, defects | `QualityDecision` |
| `visualization` | Overlay, boxes, decision banner | image, result | PNG |
| `report_generator` | Machine- and human-readable reports | results | JSON, CSV, MD |
| `evaluate` / `metrics` | Held-out scoring | split CSV | Dice, IoU, P/R/F1 |
| `pipeline` | Orchestration and per-image error isolation | paths | `List[InspectionResult]` |

Stages communicate only through the dataclasses in `schemas.py`, so a backend can
be replaced without touching anything downstream.

## 3. Inference sequence

```mermaid
sequenceDiagram
    participant U as User (CLI)
    participant P as InspectionPipeline
    participant D as DefectDetector
    participant S as Severity + Quality
    participant R as Reporters

    U->>P: visioninspect run --input samples
    P->>P: load + validate + preprocess
    P->>D: detect(image, tensor)
    D-->>P: List[Defect]
    P->>S: compute_severity / decide
    S-->>P: score, label, decision, reasons
    P->>R: write JSON, CSV, Markdown, overlay
    R-->>U: paths + decision summary
```

## 4. Data flow during training

```mermaid
flowchart TD
    A[train.csv] --> B[Normalise schema<br/>ImageId, ClassId, EncodedPixels]
    B --> C[Group by ImageId]
    C --> D[Stratify on class signature]
    D --> E[Split on unique image ids]
    E --> F[train.csv / val.csv / test.csv]
    F --> G[SteelDataset<br/>RLE to 4-channel mask]
    G --> H[U-Net<br/>BCE + Dice loss]
    H --> I{Val Dice improved?}
    I -->|yes| J[Save checkpoint]
    I -->|no x3| K[Early stop]
```

Splitting happens on unique `ImageId`, never on annotation rows. An image with
two defect classes occupies two rows in `train.csv`; a row-level split would put
one row in train and the other in test, leaking the same pixels across the
boundary and inflating every reported metric.

## 5. Severity model

```
score = 100 × class_factor × (w_area × area_term + w_count × count_term)

area_term    = min(1, defect_area / image_area ÷ area_scale)
count_term   = min(1, defect_count ÷ count_saturation)
class_factor = max class weight present ÷ largest configured weight
```

Three properties matter and are enforced by tests: the score rises with defect
area, it rises with class weight at equal area, and it never exceeds 100.

Band edges (MINOR / MODERATE / SEVERE / CRITICAL) come from percentiles of the
score distribution over defective training images, computed by `calibrate`. Until
that runs, `severity.calibrated` stays `false` and every output is stamped
`UNCALIBRATED`.

## 6. Decision logic

```mermaid
flowchart TD
    A[Severity result] --> B{Any defect?}
    B -->|no| C[ACCEPT]
    B -->|yes| D{score >= reject_min?}
    D -->|yes| E[REJECT]
    D -->|no| F{score > accept_max?}
    F -->|yes| G[REVIEW]
    F -->|no| H{Critical class present?}
    H -->|covers > limit| E
    H -->|present| G
    H -->|no| I{Unclassified regions?}
    I -->|yes| G
    I -->|no| C
```

Every branch appends a human-readable reason. A REJECT is never returned without
the trigger being stated in the report.

## 7. Error handling

A typed exception hierarchy (`VisionInspectError` → `ConfigError`, `DataError`,
`ImageValidationError`, `ModelError`, `PipelineError`) lets the CLI turn expected
failures into a one-line message and exit code 2, while unexpected failures keep
their traceback. Within a batch, one unreadable image is logged, recorded in the
report's "skipped inputs" table, and the run continues.
