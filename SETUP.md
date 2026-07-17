# Setup Guide

## 1. Prerequisites

- Python `3.10` 
- pip
- (Optional but recommended) a virtual environment tool: `venv` or `conda`
- A CUDA-capable GPU is optional — this project's reported training run was executed entirely on **CPU** (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')` in `models/train.py`), so a GPU is not required to reproduce the reported results, though training will be faster with one available.

## 2. Clone the repository

```bash
git clone https://github.com/tabidah-usmani/xisAI.git
cd xisAI
```

## 3. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Download large files (not included in this repository)

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


**After placing `book (raw)`, `instances_default.json`, and `book (undistorted)` at the paths above, generate the train/val/test split:**
```bash
python dataset/split_dataset.py
```
This automatically creates `dataset/split/{train,val,test}/images/`, `images_undistorted/`, `masks/`, and `annotations.json` — **you do not need to manually create these folders or download pre-split data.** Masks are generated on the fly from the COCO polygon annotations, using a fixed random seed (42) for reproducible splits.

The drive link can be found over here : `https://drive.google.com/drive/folders/1x88m0pdnI6hol6zB0WD0Ln-TCRS-q7zN?usp=sharing`
The repository ships with small representative samples only (4 images per category) so the code can be inspected without downloading the full dataset.


## 6. Verify the calibration file

Camera intrinsics are already computed and saved at `calibration/calibration_data.yaml` (also available as `calibration_data.pkl`). No re-calibration is required unless you are using a different camera — see `docs/CALIBRATION_REPORT.md` for the method used.

## 7. Running the pipeline

### 7.1 Train the segmentation model
```bash
python models/train.py
```
- Reads dataset from `dataset/split/{train,val,test}/images_undistorted` and `.../masks`
- Saves the best checkpoint to `models/checkpoints/best_model.pth`
- Logs all metrics (loss, IoU, precision, recall, F1, mAP@0.5, mAP@0.5:0.95) to `models/logs/` (view with `tensorboard --logdir models/logs`)
- Saves test-set prediction visualizations to `models/outputs/test_predictions/`

### 7.2 Run inference on a new raw image
```bash
python inference/inference.py --image path/to/raw_image.jpg
```
Optional flags:
- `--calib` — path to calibration file (default: `calibration/calibration_data.yaml`)
- `--checkpoint` — path to model checkpoint (default: `models/checkpoints/best_model.pth`)
- `--output` — path to save the annotated result (default: `inference/outputs/<name>_annotated.png`)

**Important:** always pass a **raw** (never-before-undistorted) image. Passing an already-undistorted image will apply the undistortion transform twice and distort the result — see `docs/TRAINING_REPORT.md` Section 9 for a measured example of this exact failure mode.

### 7.3 Run the full pixel-to-mm measurement pipeline
```bash
python measurement/measurement.py --image path/to/raw_image.jpg --ruler-mm 100
```
- `--ruler-mm` must equal the **exact** real-world distance (in mm) between the two ruler tick marks you are about to click when the image window opens.
- A window will open showing the undistorted image — click two points on a ruler visible in the frame, then press any key to confirm.
- Requires a physical ruler or tape measure visible in-frame, coplanar with and adjacent to the object being measured (see `docs/MEASUREMENT_REPORT.md` for why this matters).
- Outputs the annotated image with bounding box, ruler reference line, and computed width/height/confidence to `measurement/outputs/`.

### 7.4 Run accuracy validation
```bash
# Take a single measurement and log it against known ground truth
python measurement/measurement.py --image path/to/image.jpg --ruler-mm 100 \
    --gt-width-mm 152 --gt-height-mm 145

# After logging several instances, aggregate them into an accuracy report
python measurement/accuracy_report.py
```
Ground-truth-logged runs accumulate in `measurement/outputs/measurement_log.csv`. `accuracy_report.py` reads that log and writes `measurement/outputs/accuracy_summary.md`, containing the full per-image error table plus MAE/MPE for width, height, and overall.

## 8. Troubleshooting

- **`ValueError: num_samples should be a positive integer`** — the dataset path is empty or pointing at the wrong folder; confirm `dataset/split/{train,val,test}/images_undistorted` exists and contains images with matching `_mask.png` files in the corresponding `masks/` folder.
- **Calibration file not found** — confirm `calibration/calibration_data.yaml` exists at the path passed via `--calib`.
- **Double/over-corrected images in inference or measurement output** — you likely passed an already-undistorted image as input; always use raw camera captures for `inference.py` and `measurement.py`.
