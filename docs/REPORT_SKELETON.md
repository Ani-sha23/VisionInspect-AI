# Project report — skeleton

This is a **structure to fill in, not a report to submit**. Sections marked
**[WRITE THIS YOURSELF]** must be in your own words: they are the sections an
evaluator reads to judge whether you understand your own system, and your
submission is screened for AI-generated prose. Sections marked **[PASTE MEASURED
OUTPUT]** should be filled from files the code actually produced — never from
memory, never from a plausible-looking guess.

---

## 1. Title page
Project title, your name, registration number, course, date.

## 2. Abstract (150–200 words)
**[WRITE THIS YOURSELF]** — what the system does, on what data, with what result.
Write this last, once you know your real numbers.

## 3. Introduction
- Problem: manual visual inspection of steel surfaces is slow, subjective, and does not scale.
- Why computer vision fits: defects are visually distinctive but hard to specify in rules.
- Scope of this project and what it deliberately leaves out.
- **[WRITE THIS YOURSELF]** — one paragraph on why you picked this problem.

## 4. Literature / background (½–1 page)
Surface-defect inspection approaches, U-Net for segmentation, the Severstal
competition. Cite properly. Keep it short; this is not a survey paper.

## 5. Dataset
**[PASTE MEASURED OUTPUT]** from `docs/results/dataset_stats.md`:
image counts, annotation rows, per-class distribution, images carrying multiple
classes, image dimensions, and the RLE decode verification.
Include `outputs/figures/class_distribution.png`.

Discuss the class imbalance you actually measured and what it implies for training.

## 6. System architecture
Use the diagrams in `docs/architecture.md` (Mermaid renders on GitHub; export to
PNG for a PDF report). Cover the stage pipeline, module responsibilities, and the
dataclass contracts between stages.

## 7. Design decisions
Summarise `docs/adr/ADR-001.md` (one model serving three tasks) with the
alternatives you rejected and why.
**[WRITE THIS YOURSELF]** — also explain the leakage-safe split: why grouping by
`ImageId` matters, and what would have gone wrong with a row-level split.

## 8. Implementation
- Preprocessing: CLAHE and why plain histogram equalisation was not used.
- Model: compact U-Net, four sigmoid channels, why sigmoid and not softmax.
- Training: BCE + Dice, AdamW, cosine schedule, early stopping on validation Dice.
- Masks to instances: connected components, `min_component_area`.
- Severity formula and the calibration procedure.
- Decision engine and its reason strings.
- Error handling: the typed exception hierarchy and per-image isolation in batches.

## 9. Results
**[PASTE MEASURED OUTPUT]** from `docs/results/evaluation.md`:
per-class Dice and IoU, image-level precision/recall/F1, macro F1, class-agnostic
Dice. Include training curves from the `train` command's history output.

Add qualitative examples: two or three annotated images from `outputs/figures/`,
including at least one failure case.

If a run did not finish, state that plainly. An honest "not measured" costs you
far less than a fabricated number that does not match your own repository.

## 10. Analysis
**[WRITE THIS YOURSELF]** — which classes the model handles worst and your
hypothesis as to why; whether errors are false positives or false negatives and
what that means on a real production line; how sensitive results are to
`mask_threshold`.

## 11. Challenges faced
**[WRITE THIS YOURSELF]** — the real ones, with specifics. What broke, what you
misdiagnosed first, how you found the actual cause. Vague difficulty claims read
as filler; concrete bugs read as experience.

## 12. Learnings
**[WRITE THIS YOURSELF]** — what you would do differently, and what transferred
beyond this project.

## 13. Limitations
Take these from the README's Limitations section and expand them in your own words.

## 14. Future enhancements
Video-stream inspection, REST API for line integration, pretrained encoder,
test-time augmentation, per-class thresholds.

## 15. How to run
Point to the README rather than duplicating it; include the quick-start block.

## 16. References
Dataset, U-Net paper, OpenCV and PyTorch docs, anything else you actually used.

---

### Before you submit

- Every number in the report traces to a file in `docs/results/` or `outputs/`.
- Repository is **public**, URL is the root form `https://github.com/<user>/<repo>`.
- `README.md` sits at the repository root.
- A clean clone runs the quick-start block without any manual fixes.
