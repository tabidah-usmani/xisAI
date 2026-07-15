# Camera Calibration Report

## 1. Objective

This report documents the intrinsic camera calibration performed to determine the camera matrix and distortion coefficients used to undistort images throughout this pipeline, in accordance with Section 4.1 (Step 1) and Section 5.1 of the assessment specification.

## 2. Calibration Target

| Parameter | Value |
|---|---|
| Target type | Checkerboard (displayed on monitor screen) |
| Board size | 8×10 squares |
| Inner corners used | 7×9 |
| Square size | 15.0 mm |
| Source | www.calib.io |

## 3. Image Acquisition

| Parameter | Value |
|---|---|
| Number of images captured | 21 |
| Number of images with successful corner detection | 21 / 21 (100%) |
| Camera | iPhone 12 |
| Camera settings | Fixed focal length, no digital zoom, consistent across all shots |

Images were captured at varied tilt angles, distances, and frame positions to ensure sufficient geometric diversity for accurate distortion estimation, per the guidance in Section 5.1 of the spec (minimum 20 images, varied angles/distances/positions).

**Note on iteration:** An initial calibration attempt using 24 images produced a mean reprojection error of 0.6766 px. Reviewing the detected-corner visualizations showed that several images had the checkerboard occupying too small a portion of the frame, along with limited angle diversity (most shots shared a similar tilt direction). A second set of 21 images was captured with the board filling more of the frame and a wider range of tilt/rotation angles, which reduced the reprojection error to 0.4819 px. This is reported for transparency and to demonstrate the diagnostic process behind the final result.

## 4. Calibration Method

- **Library:** OpenCV (`cv2.findChessboardCorners`, `cv2.cornerSubPix`, `cv2.calibrateCamera`)
- **Corner refinement:** Sub-pixel corner refinement applied via `cornerSubPix` (window size 11×11, termination criteria: 30 iterations or 0.001 epsilon)
- **World coordinates:** Object points scaled by the true square size (15.0 mm) so that calibration output is in real-world units

## 5. Calibration Results

### 5.1 Camera Matrix (Intrinsic Parameters)

```
[[3176.10485      0.0     1910.45124]
 [   0.0       3163.72263  1145.08773]
 [   0.0          0.0        1.0    ]]
```

Where:
- `fx = 3176.10`, `fy = 3163.72` — focal lengths in pixels
- `cx = 1910.45`, `cy = 1145.09` — principal point (optical center) in pixels

### 5.2 Distortion Coefficients

```
[k1, k2, p1, p2, k3] = [-0.00310, -0.05950, -0.00531, 0.00067, 0.03742]
```

Where `k1, k2, k3` are radial distortion coefficients and `p1, p2` are tangential distortion coefficients. The relatively small magnitude of these coefficients is consistent with a modern phone camera lens, which typically exhibits mild-to-moderate distortion rather than strong fisheye-style distortion.

### 5.3 Reprojection Error

| Metric | Value |
|---|---|
| Mean reprojection error | **0.4819 px** |
| Spec threshold (acceptable) | < 0.5 px |
| Spec threshold (excellent) | < 0.3 px |
| Result | ✅ Acceptable (below 0.5 px threshold) |

## 6. Undistortion Verification

Undistortion was tested by applying `cv2.undistort()` (using `cv2.getOptimalNewCameraMatrix` with alpha=1, followed by cropping to the valid ROI) to a sample calibration image. The output was visually compared against the original:

- `calibration/output/original.jpg`
- `calibration/output/undistorted.jpg`

Straight edges near the periphery of the frame (checkerboard borders) appear marginally straighter in the undistorted output, consistent with correction of mild radial distortion.


## 7. Files

| File | Description |
|---|---|
| `calibration/calibrate.py` | Calibration script |
| `calibration/calibration_data.yaml` | Saved camera matrix, distortion coefficients, and metadata |
| `calibration/calibration_data.pkl` | Same data, pickle format |
| `calibration/detected/` | Sample corner-detection visualizations |
| `calibration/output/` | Undistortion test (before/after) |

Raw calibration images (21 photos) are hosted on Google Drive — see link in main `README.md`.