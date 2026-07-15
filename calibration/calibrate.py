# calibration/calibrate_camera.py

import cv2
import numpy as np
import glob
import os
import yaml
from datetime import datetime

# ---------- CONFIGURATION ----------
# YOUR BOARD: 8x10 checkerboard from calib.io
# Inner corners: (8-1) x (10-1) = 7 x 9
CHECKERBOARD = (7, 9)           # Inner corners (columns, rows)
SQUARE_SIZE_MM = 15.0           # Your board has 15mm squares

# Adjust paths to your folder structure
IMAGES_PATH = "calibration/calibrated/*.jpg"  # Renamed images
OUTPUT_FILE = "calibration/calibration_data.yaml"
# ------------------------------------

def calibrate_camera():
    """Main calibration function"""
    
    print("=" * 60)
    print("CAMERA CALIBRATION")
    print("=" * 60)
    print(f"Checkerboard size: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE_MM} mm")
    print(f"Looking for images in: {IMAGES_PATH}")
    print("=" * 60)
    
    # Prepare object points (3D points in real world)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get all images
    images = glob.glob(IMAGES_PATH)
    
    # Also try .JPG (uppercase)
    if len(images) == 0:
        images = glob.glob(IMAGES_PATH.replace('.jpg', '.JPG'))
    
    # Also check the original folder if no images found
    if len(images) == 0:
        print("No images found in 'calibration/calibrated/'")
        print("Checking 'calibration/checkerboard/'...")
        images = glob.glob("calibration/checkerboard/*.JPG") + glob.glob("calibration/checkerboard/*.jpg")
    
    print(f"\nFound {len(images)} images total")
    
    if len(images) == 0:
        print("\n❌ No images found!")
        print("Please place your checkerboard images in:")
        print("  - calibration/calibrated/ (renamed images)")
        print("  - OR calibration/checkerboard/ (original images)")
        return None
    
    # Sort images for consistent processing
    images.sort()
    
    # Show first few images
    print("\n📋 Images found:")
    for img in images[:5]:
        print(f"  - {os.path.basename(img)}")
    if len(images) > 5:
        print(f"  ... and {len(images) - 5} more")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    img_shape = None
    used_images = []
    failed_images = []
    
    # Create directory for visualization
    os.makedirs("calibration/detected", exist_ok=True)
    
    print("\n" + "=" * 60)
    print("PROCESSING IMAGES")
    print("=" * 60)
    
    for idx, fname in enumerate(images, 1):
        print(f"Processing {idx}/{len(images)}: {os.path.basename(fname)}", end=" ")
        
        img = cv2.imread(fname)
        if img is None:
            print("❌ Cannot read image")
            failed_images.append(fname)
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        # Try to find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            used_images.append(fname)
            
            # Visualize detection
            vis = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
            cv2.imwrite(f"calibration/detected/detected_{os.path.basename(fname)}", vis)
            
            print("✅ Found!")
        else:
            print("❌ Not found")
            failed_images.append(fname)
    
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"✅ Corners detected: {len(used_images)}/{len(images)} images")
    print(f"❌ Failed: {len(failed_images)} images")
    
    if len(failed_images) > 0:
        print("\n⚠️ Failed images (check these):")
        for fname in failed_images[:5]:
            print(f"  - {os.path.basename(fname)}")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more")
    
    if len(used_images) < 10:
        print("\n⚠️ WARNING: Too few valid images for reliable calibration.")
        print("   Aim for 20+ images with varied angles and positions.")
        print("   Current: {} images".format(len(used_images)))
        return None
    
    if len(used_images) < 20:
        print(f"\n⚠️ You have {len(used_images)} images. For best results, use 20+.")
    
    # Run calibration
    print("\n🔧 Running calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    # Compute reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    
    print("\n📷 Camera Matrix (Intrinsic Parameters):")
    print(camera_matrix)
    
    print("\n📐 Distortion Coefficients:")
    print(dist_coeffs.ravel())
    
    print(f"\n📊 Mean Reprojection Error: {mean_error:.4f} pixels")
    
    # Evaluate calibration quality
    if mean_error < 0.3:
        print("🎉 EXCELLENT calibration! (< 0.3 px)")
    elif mean_error < 0.5:
        print("✅ GOOD calibration! (< 0.5 px)")
    elif mean_error < 1.0:
        print("⚠️ ACCEPTABLE but could be improved (< 1.0 px)")
    else:
        print("❌ HIGH ERROR (> 1.0 px). Please re-calibrate with better images.")
    
    # Save results to YAML
    calib_data = {
        "calibration_date": datetime.now().isoformat(),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "reprojection_error": float(mean_error),
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "square_size_mm": SQUARE_SIZE_MM,
        "checkerboard_size": CHECKERBOARD,
        "num_images_used": len(used_images),
        "images_used": used_images,
    }
    
    with open(OUTPUT_FILE, "w") as f:
        yaml.dump(calib_data, f, default_flow_style=False)
    
    print(f"\n💾 Saved calibration parameters to: {OUTPUT_FILE}")
    
    # Also save as pickle for easier loading in Python
    import pickle
    with open("calibration/calibration_data.pkl", "wb") as f:
        pickle.dump({
            'camera_matrix': camera_matrix,
            'distortion_coefficients': dist_coeffs,
            'reprojection_error': mean_error,
            'image_shape': img_shape,
            'square_size_mm': SQUARE_SIZE_MM,
            'checkerboard_size': CHECKERBOARD,
            'num_images': len(used_images)
        }, f)
    
    print(f"💾 Also saved as: calibration/calibration_data.pkl")
    
    return camera_matrix, dist_coeffs, mean_error

def test_undistortion():
    """Test undistortion on sample images"""
    
    print("\n" + "=" * 60)
    print("TESTING UNDISTORTION")
    print("=" * 60)
    
    # Load calibration data
    try:
        import pickle
        with open("calibration/calibration_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        mtx = data['camera_matrix']
        dist = data['distortion_coefficients']
        
        # Find a test image
        test_images = glob.glob("calibration/calibrated/*.jpg") + glob.glob("calibration/calibrated/*.JPG")
        
        if len(test_images) == 0:
            print("No test images found in calibration/calibrated/")
            return
        
        # Use first image as test
        img_path = test_images[0]
        img = cv2.imread(img_path)
        
        if img is None:
            print("Cannot read test image")
            return
        
        # Undistort
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
        
        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Save comparison
        os.makedirs("calibration/output", exist_ok=True)
        cv2.imwrite("calibration/output/original.jpg", img)
        cv2.imwrite("calibration/output/undistorted.jpg", dst)
        
        print("✅ Undistortion test complete!")
        print("📁 Saved to: calibration/output/")
        print("   - original.jpg")
        print("   - undistorted.jpg")
        
        # Display comparison (if running interactively)
        print("\n🔍 Check the output images to verify distortion removal.")
        print("   Look at straight lines near the edges of the image.")
        
    except Exception as e:
        print(f"⚠️ Could not test undistortion: {e}")

if __name__ == "__main__":
    # Run calibration
    result = calibrate_camera()
    
    if result is not None:
        # Test undistortion
        test_undistortion()
        
        print("\n" + "=" * 60)
        print("✅ CALIBRATION COMPLETE")
        print("=" * 60)
        print("\n📋 Summary:")
        print(f"   - Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} inner corners")
        print(f"   - Square size: {SQUARE_SIZE_MM} mm")
        print(f"   - Images used: {len(glob.glob('calibration/calibrated/*.jpg'))}")
        print(f"   - Reprojection error: {result[2]:.4f} px")
        print("\n📁 Output files:")
        print("   - calibration/calibration_data.yaml")
        print("   - calibration/calibration_data.pkl")
        print("   - calibration/detected/*.jpg (visualizations)")
        print("   - calibration/output/ (undistortion test)")
    else:
        print("\n❌ Calibration failed. Please check:")
        print("   1. You have images in: calibration/calibrated/")
        print("   2. Images show the full checkerboard clearly")
        print("   3. Checkerboard size matches: (7, 9) inner corners")
        print("   4. Try running: python calibration/rename_images.py first")