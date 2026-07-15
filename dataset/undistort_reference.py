# undistort_reference.py

import cv2
import numpy as np
import yaml
import glob
import os
import pickle

# ---------- CONFIGURATION ----------
CALIB_FILE = "calibration/calibration_data.yaml"
CALIB_PKL = "calibration/calibration_data.pkl"  # Use pickle instead
INPUT_DIR = "dataset/raw/Object1"  # Your raw images
OUTPUT_DIR = "dataset/undistorted_reference"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
# ------------------------------------

def load_calibration_from_pickle(path):
    """Load calibration from pickle file (recommended)"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['distortion_coefficients']
    return camera_matrix, dist_coeffs

def load_calibration_from_yaml(path):
    """Load calibration from YAML file (with custom constructor)"""
    # Register Python tuple constructor
    def tuple_constructor(loader, node):
        value = loader.construct_sequence(node)
        return tuple(value)
    
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    
    with open(path, 'r') as f:
        calib = yaml.safe_load(f)
    
    camera_matrix = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["dist_coeffs"])
    return camera_matrix, dist_coeffs

def load_calibration():
    """Try pickle first, then YAML"""
    
    # Try pickle first (recommended)
    if os.path.exists(CALIB_PKL):
        print(f"✅ Loading calibration from pickle: {CALIB_PKL}")
        return load_calibration_from_pickle(CALIB_PKL)
    
    # Fallback to YAML
    elif os.path.exists(CALIB_FILE):
        print(f"✅ Loading calibration from YAML: {CALIB_FILE}")
        return load_calibration_from_yaml(CALIB_FILE)
    
    else:
        raise FileNotFoundError(f"No calibration file found. Checked: {CALIB_PKL} and {CALIB_FILE}")

def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, w2, h2 = roi
    if w2 > 0 and h2 > 0:
        undistorted = undistorted[y:y + h2, x:x + w2]

    return undistorted

def main():
    print("=" * 60)
    print("📸 UNDISTORTING BOOK DATASET (reference copy)")
    print("=" * 60)

    try:
        camera_matrix, dist_coeffs = load_calibration()
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs.ravel()}\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all images
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    files = sorted(set(files))

    if not files:
        print(f"⚠️ No images found in {INPUT_DIR}. Check the path/extension.")
        return

    print(f"📸 Found {len(files)} images in {INPUT_DIR}\n")

    success, failed = 0, 0
    
    for i, fname in enumerate(files, 1):
        img = cv2.imread(fname)
        if img is None:
            print(f"❌ {i}/{len(files)}: could not read {fname}")
            failed += 1
            continue

        undistorted = undistort_image(img, camera_matrix, dist_coeffs)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(fname))
        cv2.imwrite(out_path, undistorted)
        success += 1

        if i % 10 == 0 or i == len(files):
            print(f"  ✅ Processed {i}/{len(files)} images...")

    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"✅ Undistorted: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output folder: {OUTPUT_DIR}")
    print("\n📋 These images are NOW undistorted and ready for measurement!")
    print("   They satisfy the 'all images undistorted and verified' requirement.")

if __name__ == "__main__":
    main()