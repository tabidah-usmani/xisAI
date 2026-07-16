# Training Report

## 1. Model Architecture

**Architecture:** Custom U-Net (implemented from scratch in `models/unet.py`)

- **Encoder:** 4 downsampling stages (`DoubleConv` + `MaxPool2d`), channel progression 3 → 64 → 128 → 256 → 512 → 512 (bottleneck). Note: the source code computes the bottleneck's output channels as `1024 // factor` where `factor = 2` (since `bilinear=True`), which evaluates to 512 *before* the layer is constructed — the bottleneck feature map itself has 512 channels at runtime, not 1024; nothing is ever computed at 1024 channels and then reduced.
- **Decoder:** 4 upsampling stages using bilinear interpolation (`nn.Upsample`, which changes spatial resolution only, not channel count) + `DoubleConv`, with skip connections concatenating encoder feature maps at each corresponding resolution. Channel progression: 512 → 256 → 128 → 64 → 64.
- **Output:** 1×1 convolution producing a single-channel logit map (binary segmentation)
- **Total parameters:** 13,395,329

**Why U-Net over YOLO/Roboflow models:** The assessment explicitly excludes Roboflow's and Ultralytics' YOLO architectures. U-Net was selected because:
- It is a well-established, encoder-decoder architecture purpose-built for dense pixel-wise segmentation (as opposed to YOLO's bounding-box detection design), which matches this task's need for precise mask boundaries for downstream mm-measurement.
- Skip connections preserve fine spatial detail lost during downsampling, which is important given the object's boundary is later used directly for pixel-to-mm conversion.
- Its parameter count (~13M) is tractable to train from scratch on a small (~50–70 image) custom dataset without requiring pretrained backbones.

## 2. Dataset & Split

| Split | Images |
|---|---|
| Train | 51 (49 with matched masks used for training) |
| Val | 14 (13 with matched masks used for validation) |
| Test | 9 (8 with matched masks used for testing) |

Split ratio: 70% / 20% / 10% (train/val/test), as required by the assessment spec. Images were resized to 256×256 and normalized to [0, 1]. All images used for training/validation/testing are the **undistorted** versions (`images_undistorted/`), produced via the Step 1 camera calibration pipeline (`cv2.undistort()`) — not the raw distorted captures.

## 3. Hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau (mode='min', patience=10) |
| Loss function | BCEWithLogitsLoss |
| Image size | 256 × 256 |
| Augmentation strategy | **None applied in this run** (see Limitations) |

## 4. Metrics Logged

Per the assessment requirements, the following were tracked for both train and validation sets every epoch, and reported once on the held-out test set:

- Loss (BCE)
- IoU (Intersection over Union)
- Precision, Recall, F1 (pixel-level)
- mAP@0.5 and mAP@0.5:0.95

### Note on mAP definition

Conventional (COCO-style) mAP is defined for multi-instance object detection: it requires confidence-ranked candidate boxes per image, computing precision-recall curves across confidence thresholds, then averaging across classes. This pipeline is single-class, single-instance-per-image, plain binary segmentation with no confidence-ranked candidate proposals — a direct COCO-style mAP computation does not apply.

Instead, mAP here is computed as a **segmentation-adapted proxy**: for each image, the predicted mask's IoU against ground truth is computed. At each IoU threshold *t* ∈ {0.50, 0.55, ..., 0.95}, an image is counted as a "positive" if its IoU ≥ *t*. mAP@0.5 is the fraction of images clearing the 0.5 threshold; mAP@0.5:0.95 is the mean of this fraction across all ten thresholds. This is a legitimate and commonly used adaptation for single-instance segmentation evaluation, but it is **not numerically equivalent** to detection-style mAP and should not be directly compared to YOLO/Roboflow mAP figures reported elsewhere.

## 5. Training Curves

Loss and IoU curves (train vs. validation) are available via TensorBoard:

```
tensorboard --logdir models/logs
```

*(Insert exported PNG screenshots of `Loss/train` vs `Loss/val` and `Iou/train` vs `Iou/val` here before final submission.)*

**Summary of trend:**
- Training loss decreased smoothly and consistently from 0.5634 (epoch 1) to 0.1314 (epoch 50).
- Validation loss decreased from 0.6948 (epoch 1) to a best of 0.2126 (epoch 50), with some oscillation between epochs 30–48 (ranging 0.21–0.28) rather than a fully monotonic decline.
- Validation IoU rose from 0.30 (epoch 1) to a stable plateau around 0.82–0.86 from roughly epoch 30 onward.

## 6. Final Metrics

### Validation (best checkpoint, epoch 50)

| Metric | Value |
|---|---|
| Loss | 0.2126 |
| IoU | 0.8621 |
| Precision | 0.8829 |
| Recall | 0.9746 |
| F1 | 0.9258 |
| mAP@0.5 | 0.9286 |
| mAP@0.5:0.95 | 0.8500 |

### Test set (held out, best checkpoint)

| Metric | Value |
|---|---|
| Loss | 0.2203 |
| IoU | 0.8556 |
| Precision | 0.8691 |
| Recall | 0.9820 |
| F1 | 0.9220 |
| mAP@0.5 | 0.8889 |
| mAP@0.5:0.95 | 0.8556 |

Test-set prediction visualizations (input / ground truth / prediction triptychs) are saved at `models/outputs/test_predictions/`.

## 7. Analysis

- **Recall consistently exceeds precision** (test: 0.98 vs 0.87), indicating the model rarely misses true object pixels but has a mild tendency to over-predict foreground area (some false positives at the mask boundary). This is a common pattern for BCE-trained segmentation models on small datasets and could be improved with a combined BCE + Dice loss, which more directly penalizes boundary imprecision.
- **Mild overfitting from ~epoch 30 onward:** training IoU continued climbing toward 0.95–0.96 while validation IoU plateaued around 0.82–0.86, and validation loss stopped decreasing monotonically. This gap is expected given the small training set (~49 images) and is not severe — the model did not collapse or diverge — but indicates further gains would likely come from either more training data or augmentation, rather than more epochs.
- **Test metrics closely track validation metrics** (IoU 0.856 vs 0.862, F1 0.922 vs 0.926), suggesting the validation set was a reasonably representative proxy for test performance despite its small size (13 images).

## 8. Limitations

- **No data augmentation was applied in this training run** (no flips, rotation, or color jitter). Given the small dataset size and the mild overfitting observed above, augmentation is a clear candidate for improving generalization and is recommended as a follow-up experiment.
- **Small validation and test sets** (13 and 8 images respectively) mean per-epoch metrics, particularly mAP and F1, are subject to noticeable batch-to-batch noise. Reported numbers should be read as indicative rather than statistically precise.
- **Single object class, single instance per image** — the model was not evaluated on multi-instance scenes or occluded/overlapping objects, since the task and dataset were scoped to one object per image.
- mAP figures use the segmentation-adapted definition described in Section 4 and are not directly comparable to detection-benchmark mAP values.

## 9. Inference Pipeline

The trained model (`models/checkpoints/best_model.pth`) is deployed via a standalone script, `inference/inference.py`, which implements the full conceptual flow specified in Section 5.2 of the assessment: raw image → undistortion → segmentation → annotated output.

**Pipeline steps:**
1. Load the camera intrinsic matrix and distortion coefficients from `calibration/calibration_data.pkl` (OpenCV `FileStorage` format, produced in Step 1).
2. Load a **raw** (distorted) input image and apply `cv2.undistort()` using the calibrated parameters, cropping to the valid ROI.
3. Resize the undistorted image to 256×256, normalize, and run it through the trained U-Net.
4. Resize the predicted probability mask back to the original undistorted image's resolution.
5. Overlay the binary mask (thresholded at 0.5) on the undistorted image with a contour outline, and report a confidence score (mean predicted probability within the mask region).

**Usage:**
```
python inference/inference.py --image path/to/raw_image.jpg
```

### Validation: undistortion must run on raw images only

During testing, the script was run twice on the same underlying photo (`IMG_6285`) to confirm the undistortion step behaves correctly:

| Input | Mask pixel count | Confidence |
|---|---|---|
| `dataset/split/test/images_undistorted/IMG_6285.jpg` (already undistorted) | 3,185,950 | 0.9441 |
| `dataset/split/test/images/IMG_6285.jpg` (raw, correct usage) | 3,368,894 | 0.9437 |

Feeding an already-undistorted image back through the pipeline applies the distortion-correction transform a second time, subtly warping the image geometry and shifting the resulting mask area by roughly **6%** relative to the correct single-pass result. This is a concrete illustration of why `inference.py` — and the downstream pixel-to-mm measurement pipeline in Step 3 — must always be run on **raw, never-before-undistorted images**, and why raw/distorted images cannot be used directly for metric measurement without calibration correction (see Section 4.3 of the assessment and `MEASUREMENT_REPORT.md`).

## 10. Reproducibility

Training is fully reproducible via `models/train.py`, which fixes all hyperparameters at the top of the script (`BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `IMAGE_SIZE`). The best checkpoint is saved at `models/checkpoints/best_model.pth`, containing the model state dict, optimizer state dict, and full validation metrics dict at the epoch it was saved.

```
python models/train.py
```