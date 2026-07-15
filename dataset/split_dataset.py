# dataset/split_dataset.py

import json
import os
import random
import shutil
import cv2
import numpy as np

# ---------- CONFIGURATION ----------
# YOUR ACTUAL PATHS:
COCO_JSON = "dataset/raw/annotations/instances_default.json"  # Merged annotations
IMAGES_DIR = "dataset/raw/Object1"                            # Raw images
UNDISTORTED_DIR = "dataset/undistorted_reference"             # Undistorted images
OUTPUT_DIR = "dataset/split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42
# ------------------------------------

def load_coco(path):
    with open(path, "r") as f:
        return json.load(f)

def save_coco(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def split_coco(coco, train_ratio, val_ratio, test_ratio, seed):
    images = coco["images"]
    random.seed(seed)
    shuffled = images[:]
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_imgs = shuffled[:n_train]
    val_imgs = shuffled[n_train:n_train + n_val]
    test_imgs = shuffled[n_train + n_val:]
    
    return train_imgs, val_imgs, test_imgs

def build_subset(coco, image_subset):
    image_ids = {img["id"] for img in image_subset}
    annotations = [ann for ann in coco["annotations"] if ann["image_id"] in image_ids]
    
    return {
        "images": image_subset,
        "annotations": annotations,
        "categories": coco["categories"],
    }

def copy_files(image_list, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    missing = 0
    
    for img in image_list:
        fname = img["file_name"]
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            missing += 1
    
    return copied, missing

def create_masks(image_list, coco, output_dir):
    """Create binary masks from COCO annotations"""
    os.makedirs(output_dir, exist_ok=True)
    created = 0
    missing = 0
    
    # Group annotations by image ID
    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    for img in image_list:
        img_id = img["id"]
        fname = img["file_name"]
        name, ext = os.path.splitext(fname)
        mask_name = f"{name}_mask.png"
        mask_path = os.path.join(output_dir, mask_name)
        
        # Find corresponding image to get size
        src_path = os.path.join(IMAGES_DIR, fname)
        if not os.path.exists(src_path):
            missing += 1
            continue
        
        img_data = cv2.imread(src_path)
        if img_data is None:
            missing += 1
            continue
        
        h, w = img_data.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                seg = ann["segmentation"]
                for polygon in seg:
                    points = np.array(polygon).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], 255)
            created += 1
        
        cv2.imwrite(mask_path, mask)
    
    return created, missing

def main():
    print("=" * 60)
    print("📊 SPLITTING DATASET (70 / 20 / 10)")
    print("=" * 60)
    
    # Check files
    if not os.path.exists(COCO_JSON):
        print(f"❌ Annotation file not found: {COCO_JSON}")
        return
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return
    
    # Load COCO
    coco = load_coco(COCO_JSON)
    total_images = len(coco["images"])
    total_annotations = len(coco["annotations"])
    
    print(f"\n📸 Loaded:")
    print(f"   Images: {total_images}")
    print(f"   Annotations: {total_annotations}")
    print(f"   Categories: {[c['name'] for c in coco['categories']]}")
    
    # Split
    train_imgs, val_imgs, test_imgs = split_coco(
        coco, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }
    
    print("\n📂 Creating splits...")
    
    for split_name, image_subset in splits.items():
        print(f"\n📁 {split_name.upper()} ({len(image_subset)} images)")
        
        subset_coco = build_subset(coco, image_subset)
        
        # Save annotations
        json_out = os.path.join(OUTPUT_DIR, split_name, "annotations.json")
        save_coco(subset_coco, json_out)
        print(f"   ✅ Annotations saved")
        
        # Copy raw images
        images_out = os.path.join(OUTPUT_DIR, split_name, "images")
        copied, missing = copy_files(image_subset, IMAGES_DIR, images_out)
        print(f"   📸 Images: {copied} copied, {missing} missing")
        
        # Copy undistorted images
        if os.path.exists(UNDISTORTED_DIR):
            undistorted_out = os.path.join(OUTPUT_DIR, split_name, "images_undistorted")
            copied_u, missing_u = copy_files(image_subset, UNDISTORTED_DIR, undistorted_out)
            print(f"   🔄 Undistorted: {copied_u} copied, {missing_u} missing")
        
        # Create masks
        masks_out = os.path.join(OUTPUT_DIR, split_name, "masks")
        created, missing_m = create_masks(image_subset, subset_coco, masks_out)
        print(f"   🎭 Masks: {created} created, {missing_m} missing")
    
    print("\n" + "=" * 60)
    print("✅ SPLIT COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_imgs)} ({len(train_imgs)/total_images*100:.1f}%)")
    print(f"   Val:   {len(val_imgs)} ({len(val_imgs)/total_images*100:.1f}%)")
    print(f"   Test:  {len(test_imgs)} ({len(test_imgs)/total_images*100:.1f}%)")
    print(f"\n📁 Output: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()