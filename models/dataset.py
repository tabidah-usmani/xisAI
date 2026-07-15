# models/dataset.py

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob

class SegmentationDataset(Dataset):
    """Dataset for segmentation task - ONLY loads images that have masks"""
    
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get all images
        all_images = set()
        for ext in ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG'):
            all_images.update(glob.glob(f'{image_dir}/{ext}'))
        all_images = sorted(all_images)
        
        # Get all mask names (without _mask.png)
        mask_files = glob.glob(f'{mask_dir}/*.png')
        mask_names = set()
        for m in mask_files:
            name = os.path.basename(m)
            # Remove _mask.png
            name_no_ext = name.replace('_mask.png', '')
            mask_names.add(name_no_ext)
        
        # ONLY keep images that have a corresponding mask
        self.images = []
        for img_path in all_images:
            name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(name)[0]
            
            if name_no_ext in mask_names:
                self.images.append(img_path)
        
        print(f"📸 Found {len(all_images)} total images in {image_dir}")
        print(f"📸 Loaded {len(self.images)} images with masks (skipped {len(all_images) - len(self.images)} without masks)")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(name)[0]
        
        # Find corresponding mask
        mask_path = os.path.join(self.mask_dir, f'{name_no_ext}_mask.png')
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Could not load: {img_path}")
            # Return a dummy image
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Could not load mask: {mask_path}")
            mask = np.zeros(self.image_size, dtype=np.uint8)
        
        mask = cv2.resize(mask, self.image_size)
        mask = mask.astype(np.float32) / 255.0
        
        # Convert to tensor
        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)
        
        return image, mask, name