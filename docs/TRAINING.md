# Training the U-Net

Training is the only step that really wants a GPU. Everything else in this project
runs comfortably on a laptop CPU.

## Local (GPU or patient CPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or the CPU wheel
visioninspect prepare-data
visioninspect train --epochs 12
```

Smoke-test the loop before committing to a long run:

```bash
visioninspect train --epochs 1 --limit 40
```

The best checkpoint by validation Dice is written to `models/unet_best.pt`
(configurable via `model.weights`). After that, `--backend auto` picks the U-Net
automatically.

## Kaggle or Colab

PyTorch is preinstalled on both, and the Severstal data is already mounted on
Kaggle at `/kaggle/input/severstal-steel-defect-detection`.

```python
!git clone https://github.com/<your-username>/VisionInspect-AI.git
%cd VisionInspect-AI
!pip install -q -r requirements.txt

import yaml
cfg = yaml.safe_load(open("configs/config.yaml"))
cfg["data"]["train_csv"] = "/kaggle/input/severstal-steel-defect-detection/train.csv"
cfg["data"]["images_dir"] = "/kaggle/input/severstal-steel-defect-detection/train_images"
cfg["model"]["device"] = "cuda"
yaml.safe_dump(cfg, open("configs/kaggle.yaml", "w"), sort_keys=False)

!PYTHONPATH=src python -m visioninspect --config configs/kaggle.yaml prepare-data
!PYTHONPATH=src python -m visioninspect --config configs/kaggle.yaml train --epochs 12
```

Download `models/unet_best.pt` from the session output and drop it into `models/`
locally. Checkpoints are git-ignored — do not commit them.

## Training setup

| Choice | Value | Why |
|---|---|---|
| Loss | `BCEWithLogits` + soft Dice | BCE alone under-segments on heavily imbalanced masks; Dice pushes on overlap directly |
| Optimiser | AdamW | Decoupled weight decay, stable on segmentation |
| Schedule | Cosine annealing | No tuning needed for a short run |
| Augmentation | Horizontal and vertical flips | Defect orientation carries no semantic meaning on rolled steel |
| Early stopping | 3 epochs without val-Dice improvement | Protects a limited GPU quota |
| Selection metric | Validation Dice, not loss | Dice is what the evaluation reports |

## After training

```bash
visioninspect calibrate --apply            # severity bands from the real distribution
visioninspect evaluate --split data/processed/test.csv
```

`evaluate` writes `docs/results/evaluation.md` and `.json`. Those files are the
only legitimate source of accuracy numbers for the report. If the training run
did not finish, say exactly that in the report rather than quoting a number from
anywhere else.
