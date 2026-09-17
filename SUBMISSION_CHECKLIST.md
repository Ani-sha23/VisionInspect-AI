# Submission checklist

## Repository
- [ ] Visibility set to **Public** (private repositories are rejected)
- [ ] Submitted URL is the root form: `https://github.com/<username>/<repo-name>` — no `/tree/`, no `/blob/`
- [ ] `README.md` present at the repository root
- [ ] No dataset files, no `.pt`/`.pth` checkpoints committed (`.gitignore` covers both)

## Runs cleanly from a terminal
```bash
git clone <your-repo-url> && cd VisionInspect-AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/make_demo_images.py --out samples --count 6
visioninspect run --input samples --backend baseline
pytest -q
```
- [ ] Every line above works on a clean clone with no manual fixes
- [ ] No GUI or notebook is required for any step

## Before results go in the report
- [ ] `visioninspect inspect-data` has been run → `docs/results/dataset_stats.md`
- [ ] `visioninspect prepare-data` has been run → `data/processed/*.csv`
- [ ] `visioninspect calibrate --apply` has been run → `severity.calibrated: true`
- [ ] `visioninspect evaluate` has been run → `docs/results/evaluation.md`
- [ ] Every number in the report traces back to one of those files

## Report
- [ ] Follows `docs/REPORT_SKELETON.md`
- [ ] All **[WRITE THIS YOURSELF]** sections are in your own voice
- [ ] No unmeasured accuracy figures anywhere
- [ ] Submitted through the platform alongside the repository URL
