import torch
import torch.nn as nn

def mae_loss(pred, target, mask):
    """
    Args:
        pred: reconstructed patches (B, N, D)
        target: original patch embeddings (B, N, D)
        mask: binary mask (B, N), 1 = masked, 0 = visible
    Returns:
        MSE loss averaged over masked patches
    """
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # per patch
    loss = (loss * mask).sum() / mask.sum()
    return loss
