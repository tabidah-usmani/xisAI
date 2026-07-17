# XIS Technical Assessment — Object Segmentation & Metric Measurement Pipeline

An end-to-end computer vision pipeline that calibrates a camera, trains a custom segmentation model on a self-collected dataset, and computes real-world width/height (in millimetres) of the target object from a single photo.

**Pipeline:** Camera Calibration → Custom Dataset Collection & Labelling → U-Net Segmentation Training → Calibrated Inference → Pixel-to-MM Measurement



---

## Project Overview

This project implements a full industrial-style measurement workflow:

1. **Camera Calibration** — Intrinsic calibration using a checkerboard pattern to remove lens distortion (`calibration/`)
2. **Dataset Collection & Labelling** — Custom images of `book` collected and labelled (`dataset/`)
3. **Segmentation Model Training** — A custom U-Net (not YOLO/Roboflow, per assessment requirements) trained from scratch to segment the object (`models/`)
4. **Calibrated Inference** — A script that undistorts a raw image and runs the trained model to produce an annotated segmentation mask (`inference/`)
5. **Pixel-to-MM Measurement** — Converts the predicted mask's pixel dimensions into real-world millimetres using a ruler-based reference calibration, validated against physical ruler/calliper measurements (`measurement/`)

### Key capabilities
- From-scratch U-Net segmentation (13.4M parameters)
- Full camera calibration pipeline with documented reprojection error
- End-to-end inference: raw image → undistortion → mask → annotated output
- Metric measurement with documented accuracy (MAE/MPE) against ground-truth physical measurements

---

## Repository Structure

```
xisAI/
├── calibration/          # Camera calibration scripts, checkerboard samples, saved intrinsics
├── dataset/              # Dataset splitting/undistortion scripts + small representative samples
├── models/               # U-Net architecture, training script, metrics, training outputs/logs
├── inference/            # Single-image inference script (undistort → segment → annotate)
├── measurement/          # Pixel-to-mm measurement pipeline + accuracy validation
├── docs/                 # Full technical documentation (see below)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

> Note: full-resolution datasets, the complete calibration image set, and trained model weights are **not** stored in this repository (see [Large Files](#large-files) below) — only small representative samples are tracked in Git, per the assessment's file-hosting requirements.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/CALIBRATION_REPORT.md`](docs/CALIBRATION_REPORT.md) | Calibration method, intrinsic matrix, distortion coefficients, reprojection error |
| [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) | Object chosen, collection strategy, labelling tool, class distribution, splits |
| [`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md) | Architecture, hyperparameters, training/validation/test metrics, loss curves |
| [`docs/MEASUREMENT_REPORT.md`](docs/MEASUREMENT_REPORT.md) | Pixel-to-mm methodology, calibration dependency, accuracy report (MAE/MPE) |
| [`SETUP.md`](SETUP.md) | Full installation, environment, and run instructions |

---

## Object of Choice

**Object:** Book
**Approximate real-world dimensions:** 152 mm (width) × 145 mm (height)
**Why this object:** Rigid and non-deformable (consistent measurements across sessions), clear flat geometry with well-defined straight edges (unambiguous width/height), easy to measure precisely and repeatably with a ruler, readily available for repeated photography under varied conditions, and high edge contrast against typical backgrounds for clean CVAT polygon labelling. Full reasoning in [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md).

**Dataset:** 74 total images collected, single class (`book`), labelled via CVAT (polygon, COCO 1.0 export). Split 70/20/10 → 51 train / 14 val / 9 test.

---

## Camera Calibration Summary

- **Method:** OpenCV checkerboard intrinsic calibration (`cv2.findChessboardCorners` + `cornerSubPix` + `cv2.calibrateCamera`)
- **Calibration target:** 8×10 square checkerboard (7×9 inner corners), 15.0mm square size, displayed on-screen (calib.io)
- **Calibration images:** 21, all with successful corner detection (100%)
- **Reprojection error:** **0.4819 px** (below the spec's 0.5px acceptable threshold) — improved from an initial 0.6766px attempt after increasing frame coverage and angle diversity of the checkerboard; see `docs/CALIBRATION_REPORT.md` Section 3 for the full before/after comparison
- **Camera:** iPhone 12, fixed focal length, no digital zoom
- **Saved parameters:** `calibration/calibration_data.yaml` (also mirrored as `.pkl`)

---

## Model Summary

- **Architecture:** Custom U-Net, 13,395,329 parameters (see `docs/TRAINING_REPORT.md` for full architecture breakdown)
- **Why not YOLO/Roboflow:** Excluded per assessment requirements (Section 4.2); U-Net was chosen for its encoder-decoder design purpose-built for dense pixel-wise segmentation, which the downstream mm-measurement step depends on.
- **Training hardware:** CPU (no CUDA GPU used for this run)
- **Final test metrics:** IoU 0.8556, F1 0.9220, mAP@0.5 0.8889 (see `docs/TRAINING_REPORT.md` for full metrics table and the mAP-definition note)

---

## Measurement Accuracy Summary

- **Reference method:** Manually-clicked ruler reference points (chosen over automatic ruler-edge detection for robustness — see `docs/MEASUREMENT_REPORT.md` for the full design-decision rationale)
- **Validation:** 12 instances measured against physical ruler ground truth (same book, 152×145mm, photographed under varied angles/positions)

| | MAE | MPE |
|---|---|---|
| Width | 2.70 mm | 1.78% |
| Height | 6.27 mm | 4.33% |
| Overall | 4.48 mm | 3.05% |

Height error is consistently larger than width error across all 12 instances, with height over-predicted in 12/12 cases (zero exceptions) — a one-directional pattern too consistent to be random noise. This rules out simple ruler/object perspective mismatch as the sole cause (which would inflate both axes together); the leading hypothesis in `docs/MEASUREMENT_REPORT.md` Section 6 is mask-boundary bleed specific to the shorter mask axis (`minAreaRect` reports the shorter side as height), most likely from shadow inclusion or resizing artefacts at the 256×256 inference resolution. Width error, by contrast, is roughly symmetric (7 over- / 5 under-predictions) and tighter (std. dev. 2.28mm vs 3.53mm), consistent with more evenly-distributed sources like click imprecision and residual depth mismatch.

---

## Large Files

Per the assessment's file-hosting requirement (Section 2.2), the following are hosted in a single Google Drive folder — **[XIS_Assessment](https://drive.google.com/drive/folders/1x88m0pdnI6hol6zB0WD0Ln-TCRS-q7zN?usp=sharing)** — and are **not** committed to this repository:

| Drive subfolder | Contents | Place at (local path) |
|---|---|---|
| `book (raw)` | Raw dataset images (74) | `dataset/raw/Object1/` |
| `instances_default.json` (root file) | COCO annotation export from CVAT | `dataset/raw/annotations/instances_default.json` |
| `book (undistorted)` | Undistorted dataset images | `dataset/undistorted_reference/` |
| `book (mask)` | Mask reference/backup (not required — masks are auto-generated from `instances_default.json`'s polygons) | — |
| `calibrated checkerboard` | Raw checkerboard calibration photos (21) | `calibration/samples/checkerboard/` |
| `detected checkerboard` | Corner-detection visualizations | `calibration/samples/detected/` |
| `model weights` | Trained `best_model.pth` checkpoint | `models/checkpoints/best_model.pth` |
| `book with ruler` | Photos used for Step 3 accuracy validation | `measurement/validation_images/` |
| `ground truth` | Physical ruler/calliper ground-truth records for accuracy validation | reference alongside `measurement/outputs/measurement_log.csv` |

After downloading `book (raw)`, `instances_default.json`, and `book (undistorted)` into the paths above, run `python dataset/split_dataset.py` — this auto-generates `dataset/split/{train,val,test}/{images,images_undistorted,masks,annotations.json}` (masks are built from the COCO polygons, not downloaded). Full details in [`SETUP.md`](SETUP.md).



---

## Assumptions & Limitations

- Segmentation model trained on a small dataset (~49 training images); some overfitting observed after ~epoch 30 (see `docs/TRAINING_REPORT.md` Section 6/8)
- No data augmentation applied in the reported training run
- mAP is computed as a segmentation-adapted per-image IoU-threshold proxy, not COCO-style detection mAP (see `docs/TRAINING_REPORT.md` Section 4)
- Pixel-to-mm accuracy depends on the ruler and object being coplanar and equidistant from the camera; violations of this assumption were observed to cause measurable error during development (see `docs/MEASUREMENT_REPORT.md`)
- Single object class, single instance per image — not evaluated on multi-instance or occluded scenes