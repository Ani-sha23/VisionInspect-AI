# Dataset statistics (measured)

- Annotation CSV: `data/raw/train.csv`
- Images directory: `data/raw/train_images`
- Annotation rows (non-empty masks): **7095**
- Images on disk: **12568**
- Images with at least one defect: **6666**
- Images with no annotated defect: **5902**

## Annotation rows per class

| Class | Rows |
|---|---|
| 1 | 897 |
| 2 | 247 |
| 3 | 5150 |
| 4 | 801 |

## Defect classes per image

| Classes on one image | Images |
|---|---|
| 1 | 6239 |
| 2 | 425 |
| 3 | 2 |

## Most common class combinations

| Combination | Images |
|---|---|
| 3 | 4759 |
| 1 | 769 |
| 4 | 516 |
| 3-4 | 284 |
| 2 | 195 |
| 1-3 | 91 |
| 1-2 | 35 |
| 2-3 | 14 |
| 1-2-3 | 2 |
| 2-4 | 1 |

## Image shapes (sample of 50)

| Shape (HxWxC) | Count |
|---|---|
| 256x1600x3 | 50 |

## RLE decode verification

Decoded class 1 of `0002cc93b.jpg` into a 256x1600 mask with 4396 positive pixels (1.0732% of the image).

A plausible, non-zero coverage confirms the column-major, 1-based RLE convention.
