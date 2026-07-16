# Dataset Card

## 1. Object Selection

| Field | Detail |
|---|---|
| **Object chosen** | Book |
| **Approximate real-world dimensions** | Width: 152 mm , Height: 145 mm  |

### Justification for choice

A book was selected as the measurement target for the following reasons:

- **Rigid and non-deformable** — unlike fabric, food, or soft objects, a book maintains a fixed shape across every photo, so measurements taken on day 1 remain valid on day 2.
- **Clear, flat geometry** — a book has well-defined straight edges and a flat front/back face, which makes both polygon labelling and pixel-to-mm width/height measurement unambiguous, since "width and height" are well-defined for a rectangular object.
- **Easy to measure precisely** — real-world dimensions can be measured accurately and repeatably with a ruler or calipers, giving a reliable ground truth for Step 3 accuracy validation.
- **Availability and consistency** — readily available, easy to photograph repeatedly under varied conditions without wear, damage, or change in appearance over the course of the project.
- **Clean edges for labelling** — high contrast between the book's boundary and typical backgrounds (table, floor, shelf) makes polygon annotation in CVAT fast and accurate.

## 2. Data Collection Strategy

| Field | Detail |
|---|---|
| **Total images collected** | 74 |
| **Camera** | Same phone camera used for calibration, fixed focal length / no digital zoom |
| **Capture conditions varied** | Background (table, floor, shelf, plain wall, cluttered desk), lighting (daylight, lamp, mixed, shadowed), angle (top-down, tilted, side-angled), distance (close, medium, far), orientation (portrait/landscape, cover up/down, propped/flat), position within frame |
| **Distortion handling** | Raw images were labelled directly (see Section 5 below). A parallel undistorted reference copy of the full raw set is also provided — see `dataset/samples/undistorted/` — generated via `undistort_reference.py` using the Step 1 camera calibration parameters. |

## 3. Labelling

| Field | Detail |
|---|---|
| **Tool used** | CVAT (cvat.ai) |
| **Annotation type** | Polygon (instance segmentation mask) |
| **Label classes** | 1 class: `book` |
| **Export format** | COCO 1.0 (JSON annotations + images) |
| **Annotator** | Single annotator, self-labelled |

## 4. Class Distribution

Single-class dataset — every image contains exactly one instance of the `book` class. No class imbalance applies since there is only one object category.

| Class | Instance count |
|---|---|
| `book` | 74 |

## 5. Dataset Split

| Split | Percentage | Image count |
|---|---|---|
| Train | 70% | ___ |
| Validation | 20% | ___ |
| Test | 10% | ___ |

Split performed using `dataset/split_dataset.py` with a fixed random seed for reproducibility.

## 6. Design Decision: Raw vs. Undistorted Images for Labelling

Section 4.1 of the assessment specification does not explicitly require undistortion before the Data Labelling phase — only Section 4.3 (Step 3, Measurement) explicitly mandates that measurement images be undistorted using the Step 1 calibration parameters. Based on this, images were labelled directly from raw camera captures.

Section 4.1's "Expected Outcomes" summary does separately state "all images undistorted and verified" as an outcome of Step 1. To address this without invalidating existing polygon annotations (which are aligned to raw image pixel coordinates), a parallel undistorted copy of the full raw dataset was generated and is provided in `dataset/undistorted_reference/` and uploaded to Google Drive. This satisfies both the literal task-table requirement (labelling on raw images) and the stated Step 1 outcome (an undistorted, verified image set exists) without requiring re-labelling.

## 7. File Locations

| Content | Location |
|---|---|
| Raw book images (70+) | Google Drive — see link in `README.md` |
| Undistorted reference set | Google Drive — see link in `README.md` |
| CVAT COCO export (annotations + images) | Google Drive — see link in `README.md` |
| Train/val/test split script | `dataset/split_dataset.py` |
| Undistortion script | `undistort_reference.py` |