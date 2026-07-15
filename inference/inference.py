# inference/inference.py
"""
End-to-end inference pipeline for the segmentation model.

Pipeline:
    1. Load camera calibration parameters (intrinsic matrix + distortion coefficients)
       from an OpenCV FileStorage (.yml/.yaml) file produced in Step 1.
    2. Load a raw input image and undistort it using cv2.undistort().
    3. Run the trained U-Net (best_model.pth) on the undistorted image.
    4. Overlay the predicted mask on the undistorted image and save the
       annotated result.

Usage:
    python inference/inference.py --image path/to/raw_image.jpg
    python inference/inference.py --image path/to/raw_image.jpg --output path/to/out.png
    python inference/inference.py --image path/to/raw_image.jpg --calib calibration/calibration_params.yml
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

# Allow importing the model definition from models/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from unet import create_unet  # noqa: E402

# ---------- DEFAULT CONFIGURATION ----------
DEFAULT_CALIB_PATH = 'calibration/calibration_data.pkl'
DEFAULT_CHECKPOINT_PATH = 'models/checkpoints/best_model.pth'
DEFAULT_OUTPUT_DIR = 'inference/outputs'
IMAGE_SIZE = (256, 256)          # must match training IMAGE_SIZE in models/train.py
MASK_THRESHOLD = 0.5
OVERLAY_COLOR = (0, 255, 0)      # BGR green
OVERLAY_ALPHA = 0.45
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------------------------------------------


def load_calibration(calib_path):
    """
    Load camera matrix and distortion coefficients from:
    - Pickle file (.pkl) - RECOMMENDED
    - OpenCV FileStorage YAML (.yaml/.yml)
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"Calibration file not found at '{calib_path}'."
        )
    
    # Try pickle first (recommended)
    if calib_path.endswith('.pkl'):
        import pickle
        with open(calib_path, 'rb') as f:
            data = pickle.load(f)
        
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['distortion_coefficients']
        return camera_matrix, dist_coeffs
    
    # Try OpenCV YAML format
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    
    if not fs.isOpened():
        raise ValueError(f"Could not open calibration file: {calib_path}")
    
    camera_matrix_node = fs.getNode('camera_matrix')
    dist_coeffs_node = fs.getNode('dist_coeffs')
    
    if dist_coeffs_node.empty():
        dist_coeffs_node = fs.getNode('distortion_coefficients')
    
    if camera_matrix_node.empty() or dist_coeffs_node.empty():
        fs.release()
        raise KeyError(
            f"Could not find 'camera_matrix' and/or 'dist_coeffs' "
            f"in '{calib_path}'."
        )
    
    camera_matrix = camera_matrix_node.mat()
    dist_coeffs = dist_coeffs_node.mat()
    fs.release()
    
    return camera_matrix, dist_coeffs

def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistort a raw image using the calibrated intrinsic parameters."""
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0
    )
    undistorted = cv2.undistort(
        image, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    # Crop to the valid ROI to remove black border regions introduced by undistortion
    x, y, roi_w, roi_h = roi
    if roi_w > 0 and roi_h > 0:
        undistorted = undistorted[y:y + roi_h, x:x + roi_w]

    return undistorted


def load_model(checkpoint_path, device):
    """Load the trained U-Net from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{checkpoint_path}'. "
            f"Run models/train.py first, or pass the correct path with --checkpoint."
        )

    model = create_unet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def run_inference(model, image_rgb, device, image_size=IMAGE_SIZE):
    """
    Run the model on an RGB image (as loaded/undistorted).
    Returns the predicted mask resized back to the original image's dimensions,
    as a float32 array in [0, 1].
    """
    original_h, original_w = image_rgb.shape[:2]

    resized = cv2.resize(image_rgb, image_size)
    normalized = resized.astype(np.float32) / 255.0

    tensor = torch.tensor(normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Resize the predicted mask back to the original (undistorted) image size
    prob_mask_full = cv2.resize(prob_mask, (original_w, original_h))

    return prob_mask_full


def overlay_mask(image_bgr, prob_mask, threshold=MASK_THRESHOLD,
                  color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA):
    """Overlay a binary mask (from prob_mask >= threshold) on the BGR image."""
    binary_mask = (prob_mask >= threshold).astype(np.uint8)

    overlay = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr)
    color_layer[:] = color

    mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    blended = np.where(
        mask_3ch == 1,
        cv2.addWeighted(image_bgr, 1 - alpha, color_layer, alpha, 0),
        image_bgr
    )

    # Draw contour outline for clarity
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, color, 2)

    # Confidence score: mean predicted probability within the predicted mask region
    if binary_mask.sum() > 0:
        confidence = float(prob_mask[binary_mask == 1].mean())
    else:
        confidence = 0.0

    return blended, binary_mask, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Run undistortion + segmentation inference on a single image."
    )
    parser.add_argument('--image', required=True, help="Path to the raw input image.")
    parser.add_argument('--calib', default=DEFAULT_CALIB_PATH,
                         help=f"Path to calibration .yml file (default: {DEFAULT_CALIB_PATH})")
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT_PATH,
                         help=f"Path to trained model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})")
    parser.add_argument('--output', default=None,
                         help=f"Path to save annotated output (default: {DEFAULT_OUTPUT_DIR}/<input_name>_annotated.png)")
    args = parser.parse_args()

    print("=" * 60)
    print("🔍 INFERENCE PIPELINE")
    print("=" * 60)

    # 1. Load calibration
    print(f"\n📐 Loading calibration from: {args.calib}")
    camera_matrix, dist_coeffs = load_calibration(args.calib)
    print("   Camera matrix:\n", camera_matrix)
    print("   Distortion coefficients:\n", dist_coeffs)

    # 2. Load and undistort image
    print(f"\n📷 Loading image: {args.image}")
    raw_image = cv2.imread(args.image)
    if raw_image is None:
        raise FileNotFoundError(f"Could not read image at '{args.image}'.")

    print("   Undistorting...")
    undistorted_bgr = undistort_image(raw_image, camera_matrix, dist_coeffs)
    undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)

    # 3. Load model and run inference
    print(f"\n🏗️ Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, DEVICE)

    print("   Running segmentation inference...")
    prob_mask = run_inference(model, undistorted_rgb, DEVICE)

    # 4. Build annotated output
    annotated_bgr, binary_mask, confidence = overlay_mask(undistorted_bgr, prob_mask)

    # 5. Save
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f'{base_name}_annotated.png')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, annotated_bgr)

    print(f"\n✅ Done.")
    print(f"   Confidence score (mean prob. within mask): {confidence:.4f}")
    print(f"   Mask pixel count: {int(binary_mask.sum())}")
    print(f"   Annotated image saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()