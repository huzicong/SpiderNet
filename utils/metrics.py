import torch
import numpy as np
from torch import nn


###### METRICS #######
def compute_errors(pred, depth_gt, mask):
    """Compute error for depth as required for paper (RMSE, REL, etc)
    Args:
        gt (torch.Tensor): Ground truth depth (metric). Shape: [B, 1, H, W], dtype: float32
        pred (torch.Tensor): Predicted depth (metric). Shape: [B, 1, H, W], dtype: float32
        mask (torch.Tensor): Mask of pixels to consider while calculating error.
                              Pixels not in mask are ignored and do not contribute to error.
                              Shape: [B, 1, H, W], dtype: bool

    Returns:
        dict: Various measures of error metrics
    """
    pred = pred.squeeze(1)
    depth_gt = depth_gt.squeeze(1)
    mask = mask.squeeze(1)

    # prevent dividing 0 error
    pred = torch.clamp(pred, min=1e-8, max=1e8)

    depth_gt[torch.isnan(depth_gt)] = 0
    depth_gt[torch.isinf(depth_gt)] = 0

    mask = (mask > 0)
    mask_valid_region = (depth_gt > 0)
    mask = torch.logical_and(mask_valid_region, mask)

    depth_gt = depth_gt[mask]
    pred = pred[mask]
    thresh = torch.max(depth_gt / pred, pred / depth_gt)

    a1 = (thresh < 1.05).float().mean()
    a2 = (thresh < 1.10).float().mean()
    a3 = (thresh < 1.25).float().mean()

    rmse = ((depth_gt - pred) ** 2).mean().sqrt().item()
    abs_rel = ((depth_gt - pred).abs() / depth_gt).mean().item()
    mae = (depth_gt - pred).abs().mean().item()

    return rmse, abs_rel, mae, a1.item() * 100, a2.item() * 100, a3.item() * 100


def compute_iou(pred, gt):
    match_it = torch.sum(torch.logical_and(pred == 1, gt == 1))
    match_un = torch.sum(torch.logical_or(pred == 1, gt == 1))
    total_iou = match_it / match_un

    return total_iou / gt.shape[0]



