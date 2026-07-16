# measurement/measurement.py
"""
End-to-end pixel-to-mm measurement pipeline.

Pipeline:
    1. Load camera calibration (intrinsic matrix + distortion coefficients).
    2. Load a raw input image and undistort it (cv2.undistort()).
    3. Establish a pixels-per-mm ratio using a ruler visible in the frame:
       the user clicks two points on the ruler corresponding to a known
       real-world distance (e.g. the 0cm and 10cm marks), and provides that
       known distance in mm.
    4. Run the trained segmentation model on the undistorted image to get
       the object's mask.
    5. Extract the object's pixel width/height from the mask using a
       minimum-area rotated bounding box (robust to object rotation).
    6. Convert pixel width/height to millimetres using the pixels-per-mm ratio.
    7. Save an annotated image showing the mask, bounding box, ruler
       reference points, and the computed width/height in mm.
    8. If ground-truth dimensions are provided, log the result (predicted
       vs. ground truth, absolute error, percentage error) to a CSV file
       for downstream MAE/MPE accuracy analysis (see accuracy_report.py).

Design decision: a manually-clicked ruler reference (rather than automatic
ruler-edge detection via contours) was chosen for robustness. Automatic
detection of a thin ruler is fragile against background clutter and
lighting variation on a small, from-scratch dataset; manual point selection
takes seconds per image and removes this failure mode entirely, at the cost
of requiring one manual step per measurement.

Usage:
    python measurement/measurement.py --image path/to/raw_image.jpg --ruler-mm 100

    # With ground truth logging (for MAE/MPE accuracy analysis):
    python measurement/measurement.py --image path/to/raw_image.jpg --ruler-mm 100 \
        --gt-width-mm 150 --gt-height-mm 145
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference'))
from unet import create_unet  # noqa: E402
from inference import load_calibration, undistort_image, run_inference  # noqa: E402

# ---------- DEFAULT CONFIGURATION ----------
DEFAULT_CALIB_PATH = 'calibration/calibration_data.pkl'
DEFAULT_CHECKPOINT_PATH = 'models/checkpoints/best_model.pth'
DEFAULT_OUTPUT_DIR = 'measurement/outputs'
LOG_FILENAME = 'measurement_log.csv'
IMAGE_SIZE = (256, 256)
MASK_THRESHOLD = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------------------------------------


class RulerPointSelector:
    """
    Interactive click-based tool to select two reference points on a ruler
    visible in the image. Displays the image in a resizable window; the
    user clicks two points, then presses any key to confirm.
    """

    def __init__(self, image_bgr, window_name="Click two ruler reference points, then press any key"):
        self.image_bgr = image_bgr
        self.window_name = window_name
        self.points = []

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 2:
            self.points.append((x, y))

    def select(self):
        display_img = self.image_bgr.copy()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            frame = display_img.copy()
            for pt in self.points:
                cv2.circle(frame, pt, 8, (0, 0, 255), -1)
            if len(self.points) == 2:
                cv2.line(frame, self.points[0], self.points[1], (0, 0, 255), 2)

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(20)

            if len(self.points) == 2 and key != -1:
                break
            if key == 27:  # ESC to cancel
                self.points = []
                break

        cv2.destroyWindow(self.window_name)

        if len(self.points) != 2:
            raise RuntimeError(
                "Ruler reference selection cancelled or incomplete. "
                "Two points are required."
            )

        return self.points[0], self.points[1]


def compute_pixels_per_mm(point_a, point_b, known_distance_mm):
    """Compute the pixels-per-mm ratio from two clicked points and a known real-world distance."""
    pixel_distance = np.sqrt(
        (point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2
    )
    if pixel_distance == 0:
        raise ValueError("The two reference points cannot be identical.")

    return pixel_distance / known_distance_mm


def extract_object_dimensions_px(binary_mask):
    """
    Extract the object's pixel width and height using a minimum-area
    rotated bounding box around the largest contour in the mask. This is
    robust to the object being rotated relative to the image axes, unlike
    a plain axis-aligned bounding box.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)  # ((cx, cy), (w, h), angle)
    (_, _), (w_px, h_px), _ = rect

    # Normalize so width_px is always the longer side, height_px the shorter
    width_px = max(w_px, h_px)
    height_px = min(w_px, h_px)

    box_points = cv2.boxPoints(rect).astype(np.int32)

    return width_px, height_px, box_points


def log_measurement(image_name, width_mm, height_mm, confidence,
                     gt_width_mm=None, gt_height_mm=None):
    """
    Appends a row to measurement/outputs/measurement_log.csv.

    If ground-truth dimensions are provided, also computes and logs the
    absolute error (mm) and percentage error for width and height, which
    accuracy_report.py later aggregates into MAE / MPE.
    """
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(DEFAULT_OUTPUT_DIR, LOG_FILENAME)
    file_exists = os.path.isfile(log_path)

    width_abs_err = abs(width_mm - gt_width_mm) if gt_width_mm else ""
    height_abs_err = abs(height_mm - gt_height_mm) if gt_height_mm else ""
    width_pct_err = (width_abs_err / gt_width_mm * 100) if gt_width_mm else ""
    height_pct_err = (height_abs_err / gt_height_mm * 100) if gt_height_mm else ""

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "image", "gt_width_mm", "gt_height_mm",
                "pred_width_mm", "pred_height_mm",
                "width_abs_error_mm", "height_abs_error_mm",
                "width_pct_error", "height_pct_error", "confidence"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            image_name,
            gt_width_mm if gt_width_mm else "",
            gt_height_mm if gt_height_mm else "",
            round(width_mm, 2),
            round(height_mm, 2),
            round(width_abs_err, 2) if gt_width_mm else "",
            round(height_abs_err, 2) if gt_height_mm else "",
            round(width_pct_err, 2) if gt_width_mm else "",
            round(height_pct_err, 2) if gt_height_mm else "",
            round(confidence, 4)
        ])

    return log_path


def main():
    parser = argparse.ArgumentParser(
        description="Measure real-world width/height (mm) of the segmented object."
    )
    parser.add_argument('--image', required=True, help="Path to the raw input image.")
    parser.add_argument('--calib', default=DEFAULT_CALIB_PATH,
                         help=f"Path to calibration file (default: {DEFAULT_CALIB_PATH})")
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT_PATH,
                         help=f"Path to trained model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})")
    parser.add_argument('--ruler-mm', type=float, required=True,
                         help="Known real-world distance in mm between the two ruler points you will click "
                              "(e.g. 100 for a 0cm-to-10cm span).")
    parser.add_argument('--gt-width-mm', type=float, default=None,
                         help="Ground truth object width in mm, for accuracy logging (optional).")
    parser.add_argument('--gt-height-mm', type=float, default=None,
                         help="Ground truth object height in mm, for accuracy logging (optional).")
    parser.add_argument('--output', default=None,
                         help=f"Path to save annotated output (default: {DEFAULT_OUTPUT_DIR}/<input_name>_measured.png)")
    args = parser.parse_args()

    print("=" * 60)
    print("📏 MEASUREMENT PIPELINE")
    print("=" * 60)

    # 1. Load calibration
    print(f"\n📐 Loading calibration from: {args.calib}")
    camera_matrix, dist_coeffs = load_calibration(args.calib)

    # 2. Load and undistort image
    print(f"\n📷 Loading image: {args.image}")
    raw_image = cv2.imread(args.image)
    if raw_image is None:
        raise FileNotFoundError(f"Could not read image at '{args.image}'.")

    print("   Undistorting (measurement MUST be done on undistorted images -")
    print("   see TRAINING_REPORT.md Section 9 for why this matters)...")
    undistorted_bgr = undistort_image(raw_image, camera_matrix, dist_coeffs)
    undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)

    # 3. Ruler reference point selection
    print(f"\n📍 Click two points on the ruler corresponding to {args.ruler_mm} mm.")
    print("   A window will open. Click two points, then press any key to confirm.")
    selector = RulerPointSelector(undistorted_bgr)
    point_a, point_b = selector.select()

    pixels_per_mm = compute_pixels_per_mm(point_a, point_b, args.ruler_mm)
    print(f"   Reference points: {point_a}, {point_b}")
    print(f"   Pixels-per-mm ratio: {pixels_per_mm:.4f}")

    # 4. Run segmentation model
    print(f"\n🏗️ Loading model from: {args.checkpoint}")
    model = create_unet().to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("   Running segmentation inference...")
    prob_mask = run_inference(model, undistorted_rgb, DEVICE, image_size=IMAGE_SIZE)
    binary_mask = (prob_mask >= MASK_THRESHOLD).astype(np.uint8)

    # 5. Extract object dimensions in pixels
    width_px, height_px, box_points = extract_object_dimensions_px(binary_mask)
    if width_px is None:
        raise RuntimeError("No object detected in the segmentation mask. "
                            "Check that the object is visible and the model is loading correctly.")

    # 6. Convert to mm
    width_mm = width_px / pixels_per_mm
    height_mm = height_px / pixels_per_mm

    confidence = float(prob_mask[binary_mask == 1].mean()) if binary_mask.sum() > 0 else 0.0

    print(f"\n📏 Measured dimensions:")
    print(f"   Width:  {width_mm:.2f} mm  ({width_px:.1f} px)")
    print(f"   Height: {height_mm:.2f} mm  ({height_px:.1f} px)")
    print(f"   Confidence: {confidence:.4f}")

    if args.gt_width_mm or args.gt_height_mm:
        if args.gt_width_mm:
            w_err = abs(width_mm - args.gt_width_mm)
            w_pct = w_err / args.gt_width_mm * 100
            print(f"   Width error:  {w_err:.2f} mm  ({w_pct:.2f}%)  [GT: {args.gt_width_mm} mm]")
        if args.gt_height_mm:
            h_err = abs(height_mm - args.gt_height_mm)
            h_pct = h_err / args.gt_height_mm * 100
            print(f"   Height error: {h_err:.2f} mm  ({h_pct:.2f}%)  [GT: {args.gt_height_mm} mm]")

    # 7. Annotate and save
    annotated = undistorted_bgr.copy()
    cv2.drawContours(annotated, [box_points], 0, (0, 255, 0), 2)
    cv2.circle(annotated, point_a, 8, (0, 0, 255), -1)
    cv2.circle(annotated, point_b, 8, (0, 0, 255), -1)
    cv2.line(annotated, point_a, point_b, (0, 0, 255), 2)

    label = f"W: {width_mm:.1f}mm  H: {height_mm:.1f}mm  conf: {confidence:.2f}"
    cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f'{base_name}_measured.png')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, annotated)

    print(f"\n✅ Annotated result saved to: {output_path}")

    # 8. Log result (for MAE/MPE aggregation via accuracy_report.py)
    log_path = log_measurement(
        image_name=os.path.basename(args.image),
        width_mm=width_mm,
        height_mm=height_mm,
        confidence=confidence,
        gt_width_mm=args.gt_width_mm,
        gt_height_mm=args.gt_height_mm,
    )
    print(f"📝 Logged to: {log_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()