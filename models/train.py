# models/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from unet import create_unet
from dataset import SegmentationDataset
from metrics import (
    calculate_iou,
    calculate_precision_recall_f1,
    calculate_per_image_iou,
    calculate_map,
)

# ---------- CONFIGURATION ----------
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DIR = 'dataset/split/train'
VAL_DIR = 'dataset/split/val'
TEST_DIR = 'dataset/split/test'

CHECKPOINT_DIR = 'models/checkpoints'
LOG_DIR = 'models/logs'
OUTPUT_DIR = 'models/outputs'
# ------------------------------------


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch, logging all required metrics."""
    model.train()

    total_loss = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    all_ious = []

    for images, masks, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = torch.sigmoid(outputs)

        iou = calculate_iou(pred, masks)
        precision, recall, f1 = calculate_precision_recall_f1(pred, masks)
        per_image_ious = calculate_per_image_iou(pred, masks)

        total_iou += iou
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        all_ious.extend(per_image_ious)

    n = len(dataloader)
    map_50, map_50_95 = calculate_map(all_ious)

    return {
        'loss': total_loss / n,
        'iou': total_iou / n,
        'precision': total_precision / n,
        'recall': total_recall / n,
        'f1': total_f1 / n,
        'map_50': map_50,
        'map_50_95': map_50_95,
    }


def validate(model, dataloader, criterion, device):
    """Validate the model, logging all required metrics."""
    model.eval()

    total_loss = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    all_ious = []

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

            pred = torch.sigmoid(outputs)

            iou = calculate_iou(pred, masks)
            precision, recall, f1 = calculate_precision_recall_f1(pred, masks)
            per_image_ious = calculate_per_image_iou(pred, masks)

            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            all_ious.extend(per_image_ious)

    n = len(dataloader)
    map_50, map_50_95 = calculate_map(all_ious)

    return {
        'loss': total_loss / n,
        'iou': total_iou / n,
        'precision': total_precision / n,
        'recall': total_recall / n,
        'f1': total_f1 / n,
        'map_50': map_50,
        'map_50_95': map_50_95,
    }


def save_prediction_samples(model, dataloader, device, epoch, output_dir):
    """Save sample predictions (used during training, sampled from val set)."""
    model.eval()

    images, masks, names = next(iter(dataloader))
    images = images[:4].to(device)
    masks = masks[:4].cpu().numpy()

    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs).cpu().numpy()

    images = images.cpu().numpy()

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for i in range(4):
        img = images[i].transpose(1, 2, 0)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masks[i][0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(preds[i][0], cmap='gray')
        axes[i, 2].set_title(f'Prediction (Epoch {epoch})')
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/epoch_{epoch:03d}.png')
    plt.close()


def save_test_set_visualizations(model, test_loader, device, output_dir):
    """
    Visualize predictions on the held-out TEST set using the best saved model.
    Required explicitly by the assessment spec (Step 2: 'Visualise predictions
    on a held-out test set') - distinct from the val-set sampling used during
    training for monitoring purposes.
    """
    model.eval()

    test_output_dir = f'{output_dir}/test_predictions'
    os.makedirs(test_output_dir, exist_ok=True)

    num_batches = len(test_loader)

    for batch_idx, (images, masks, names) in enumerate(test_loader):
        images_gpu = images.to(device)

        with torch.no_grad():
            outputs = model(images_gpu)
            preds = torch.sigmoid(outputs).cpu().numpy()

        images_np = images.numpy()
        masks_np = masks.numpy()

        batch_size = images_np.shape[0]
        fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

        # When batch_size == 1, axes is 1D; normalize to 2D indexing
        if batch_size == 1:
            axes = axes.reshape(1, -1)

        for i in range(batch_size):
            img = images_np[i].transpose(1, 2, 0)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Input: {names[i]}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(masks_np[i][0], cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(preds[i][0], cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{test_output_dir}/test_batch_{batch_idx:03d}.png')
        plt.close()

    print(f"   ✅ Saved {num_batches} test-set visualization batch(es) to {test_output_dir}/")


def main():
    print("=" * 60)
    print("🚀 TRAINING SEGMENTATION MODEL")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print("=" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n📂 Loading datasets...")
    train_dataset = SegmentationDataset(
        f'{TRAIN_DIR}/images_undistorted',
        f'{TRAIN_DIR}/masks',
        image_size=IMAGE_SIZE
    )
    val_dataset = SegmentationDataset(
        f'{VAL_DIR}/images_undistorted',
        f'{VAL_DIR}/masks',
        image_size=IMAGE_SIZE
    )
    test_dataset = SegmentationDataset(
        f'{TEST_DIR}/images_undistorted',
        f'{TEST_DIR}/masks',
        image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val: {len(val_dataset)} images")
    print(f"   Test: {len(test_dataset)} images")

    print("\n🏗️ Creating model...")
    model = create_unet().to(DEVICE)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)

    writer = SummaryWriter(LOG_DIR)

    print("\n" + "=" * 60)
    print("🎯 Starting Training...")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"\n📊 Epoch {epoch}/{EPOCHS}")
        print("-" * 40)

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_metrics['loss'])

        # Log every metric to TensorBoard
        for key, value in train_metrics.items():
            writer.add_scalar(f'{key.capitalize()}/train', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'{key.capitalize()}/val', value, epoch)

        print(f"   Train | Loss: {train_metrics['loss']:.4f} | IoU: {train_metrics['iou']:.4f} "
              f"| Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f} "
              f"| F1: {train_metrics['f1']:.4f} | mAP@0.5: {train_metrics['map_50']:.4f} "
              f"| mAP@0.5:0.95: {train_metrics['map_50_95']:.4f}")

        print(f"   Val   | Loss: {val_metrics['loss']:.4f} | IoU: {val_metrics['iou']:.4f} "
              f"| Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} "
              f"| F1: {val_metrics['f1']:.4f} | mAP@0.5: {val_metrics['map_50']:.4f} "
              f"| mAP@0.5:0.95: {val_metrics['map_50_95']:.4f}")

        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")

        if epoch % 10 == 0 or epoch == 1:
            save_prediction_samples(model, val_loader, DEVICE, epoch, OUTPUT_DIR)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, f'{CHECKPOINT_DIR}/best_model.pth')
            print(f"   ✅ Best model saved! (Val Loss: {val_metrics['loss']:.4f})")

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, f'{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth')

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)

    print("\n📊 Testing best model...")
    checkpoint = torch.load(f'{CHECKPOINT_DIR}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(model, test_loader, criterion, DEVICE)

    print(f"   Test Loss: {test_metrics['loss']:.4f}")
    print(f"   Test IoU: {test_metrics['iou']:.4f}")
    print(f"   Test Precision: {test_metrics['precision']:.4f}")
    print(f"   Test Recall: {test_metrics['recall']:.4f}")
    print(f"   Test F1: {test_metrics['f1']:.4f}")
    print(f"   Test mAP@0.5: {test_metrics['map_50']:.4f}")
    print(f"   Test mAP@0.5:0.95: {test_metrics['map_50_95']:.4f}")

    # Visualize predictions on the held-out TEST set (spec requires this explicitly)
    print("\n🖼️ Saving test-set prediction visualizations...")
    save_test_set_visualizations(model, test_loader, DEVICE, OUTPUT_DIR)

    # Save test results
    with open(f'{OUTPUT_DIR}/test_results.txt', 'w') as f:
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Test IoU: {test_metrics['iou']:.4f}\n")
        f.write(f"Test Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"Test F1: {test_metrics['f1']:.4f}\n")
        f.write(f"Test mAP@0.5: {test_metrics['map_50']:.4f}\n")
        f.write(f"Test mAP@0.5:0.95: {test_metrics['map_50_95']:.4f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")

    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
    print(f"📁 Test visualizations: {OUTPUT_DIR}/test_predictions/")
    print(f"📁 Model saved to: {CHECKPOINT_DIR}/best_model.pth")
    print(f"📁 TensorBoard logs: {LOG_DIR}")
    print("\n📋 Next: Run 'tensorboard --logdir models/logs' to view training curves")


if __name__ == "__main__":
    main()