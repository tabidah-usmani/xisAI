# calibration/rename_images.py

import os
import shutil
import glob
from datetime import datetime

def rename_checkerboard_images():
    """
    Rename checkerboard images with a consistent naming convention
    Handles various formats: IMG_*.JPG, *.jpg, *.png
    """
    
    # YOUR FOLDER STRUCTURE
    source_folder = 'calibration/checkerboard'  # Where your images are
    dest_folder = 'calibration/calibrated'       # Where renamed images go
    
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Get all image files from source
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_folder, ext)))
    
    # Also check for files with "IMG" in name (your current files)
    image_files.extend(glob.glob(os.path.join(source_folder, 'IMG_*')))
    
    # Remove duplicates
    image_files = list(set(image_files))
    
    print("=" * 60)
    print("CHECKERBOARD IMAGE RENAMER")
    print("=" * 60)
    print(f"Source folder: {source_folder}")
    print(f"Destination folder: {dest_folder}")
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("\n❌ No images found!")
        print(f"Please place your checkerboard images in: {source_folder}")
        print("Supported formats: JPG, JPEG, PNG")
        return
    
    # Show files found
    print("\n📋 Files found:")
    for img in sorted(image_files)[:10]:  # Show first 10
        print(f"  - {os.path.basename(img)}")
    if len(image_files) > 10:
        print(f"  ... and {len(image_files) - 10} more")
    
    # Ask for confirmation
    print("\n" + "=" * 60)
    print("RENAMING SCHEME:")
    print("=" * 60)
    print("  IMG_6345.JPG → checkerboard_001.JPG")
    print("  IMG_6347.JPG → checkerboard_002.JPG")
    print("  IMG_6348.JPG → checkerboard_003.JPG")
    print(f"\nAll images will be renamed to: checkerboard_XXX.jpg")
    
    response = input("\nProceed with rename? (y/n): ").strip().lower()
    
    if response != 'y':
        print("❌ Rename cancelled.")
        return
    
    # Rename all images
    renamed_count = 0
    skipped_count = 0
    
    # Sort images by modification time for consistent ordering
    image_files.sort(key=os.path.getmtime)
    
    for idx, img_path in enumerate(image_files, 1):
        # Get file extension
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        # Create new name
        new_name = f"checkerboard_{idx:03d}{ext}"
        new_path = os.path.join(dest_folder, new_name)
        
        # Handle duplicate names
        counter = 1
        while os.path.exists(new_path):
            new_name = f"checkerboard_{idx:03d}_{counter}{ext}"
            new_path = os.path.join(dest_folder, new_name)
            counter += 1
        
        try:
            # Copy instead of move (keeps originals)
            shutil.copy2(img_path, new_path)
            print(f"✅ Copied: {filename:30} → {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"❌ Error copying {filename}: {e}")
            skipped_count += 1
    
    print("\n" + "=" * 60)
    print("RENAME COMPLETE")
    print("=" * 60)
    print(f"✅ Successfully renamed: {renamed_count} files")
    print(f"⚠️  Skipped: {skipped_count} files")
    print(f"📁 Files saved to: {dest_folder}")
    
    # Show final list
    print("\n📋 Final file list:")
    final_files = sorted(glob.glob(os.path.join(dest_folder, '*.JPG')) + 
                        glob.glob(os.path.join(dest_folder, '*.jpg')) +
                        glob.glob(os.path.join(dest_folder, '*.png')))
    
    for f in final_files[:10]:
        print(f"  - {os.path.basename(f)}")
    
    if len(final_files) > 10:
        print(f"  ... and {len(final_files) - 10} more")
    
    return final_files

def verify_files():
    """Verify the renamed files are valid images"""
    
    folder = 'calibration/calibrated'
    image_files = glob.glob(os.path.join(folder, '*.JPG')) + \
                  glob.glob(os.path.join(folder, '*.jpg')) + \
                  glob.glob(os.path.join(folder, '*.png'))
    
    if len(image_files) == 0:
        print(f"❌ No files found in {folder}")
        return
    
    print("\n" + "=" * 60)
    print("VERIFYING RENAMED FILES")
    print("=" * 60)
    
    import cv2
    
    valid = 0
    invalid = 0
    
    for img_path in sorted(image_files):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ {filename}: Cannot read (corrupt?)")
            invalid += 1
        else:
            h, w = img.shape[:2]
            size = os.path.getsize(img_path) / 1024
            print(f"✅ {filename}: {w}x{h}, {size:.1f} KB")
            valid += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Valid images: {valid}")
    print(f"❌ Invalid images: {invalid}")

if __name__ == "__main__":
    print("""
    📸 CHECKERBOARD IMAGE RENAMER
    ================================
    
    This script will:
    1. Copy your IMG_*.JPG files from 'calibration/checkerboard/'
    2. Rename them to 'checkerboard_001.JPG', 'checkerboard_002.JPG', etc.
    3. Save them to 'calibration/calibrated/'
    
    Your original files will NOT be deleted (they're copied).
    
    """)
    
    # Run the renamer
    renamed_files = rename_checkerboard_images()
    
    if renamed_files and len(renamed_files) > 0:
        # Verify the files
        verify_files()
        
        print("\n" + "=" * 60)
        print("✅ NEXT STEPS")
        print("=" * 60)
        print("1. Verify images in: calibration/calibrated/")
        print("2. Use these images for calibration")
        print("3. Run: python calibration/calibrate_camera.py")