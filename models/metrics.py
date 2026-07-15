# models/metrics.py

import torch

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU for a single batch (binary masks)."""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    if union == 0:
        return 1.0

    return (intersection / union).item()


def calculate_precision_recall_f1(pred, target, threshold=0.5, eps=1e-7):
    """Pixel-level precision, recall, F1 for binary segmentation."""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def calculate_per_image_iou(pred, target, threshold=0.5):
    """
    Returns a list of IoU values, one per image in the batch.
    Needed for mAP, which is computed per-image, not pooled across the batch.
    """
    ious = []
    batch_size = pred.shape[0]

    for i in range(batch_size):
        p = pred[i]
        t = target[i]

        p_bin = (p > threshold).float()
        t_bin = (t > threshold).float()

        intersection = (p_bin * t_bin).sum()
        union = p_bin.sum() + t_bin.sum() - intersection

        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())

    return ious


def calculate_map(all_ious, iou_thresholds=None):
    """
    Segmentation-style mAP: at each IoU threshold, an image counts as a
    'true positive' if its mask IoU >= threshold. Average precision at
    each threshold = fraction of images meeting that threshold.
    mAP@0.5:0.95 = mean of that fraction across thresholds 0.5:0.05:0.95.
    mAP@0.5 = fraction of images with IoU >= 0.5.

    all_ious: flat list of per-image IoU values collected across an epoch.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95

    if len(all_ious) == 0:
        return 0.0, 0.0

    all_ious_t = torch.tensor(all_ious)

    ap_per_threshold = []
    for t in iou_thresholds:
        ap_t = (all_ious_t >= t).float().mean().item()
        ap_per_threshold.append(ap_t)

    map_50 = ap_per_threshold[0]          # threshold == 0.5
    map_50_95 = sum(ap_per_threshold) / len(ap_per_threshold)

    return map_50, map_50_95