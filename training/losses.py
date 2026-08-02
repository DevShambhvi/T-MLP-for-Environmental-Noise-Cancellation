import torch.nn.functional as F

def mask_mse(pred, target):
    """MSE between predicted and ideal ratio masks (primary loss)."""
    return F.mse_loss(pred, target)

def mask_l1(pred, target):
    """L1 loss — more robust to outliers in the mask."""
    return F.l1_loss(pred, target)
