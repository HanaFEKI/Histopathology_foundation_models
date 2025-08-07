import torch
import numpy as np

def random_masking(x, mask_ratio):
    """
    This function simulates the masking process used in Masked Autoencoders (MAE).
    It shuffles patch tokens randomly, keeps a fraction (1 - mask_ratio), and masks
    the rest. It returns the masked input, the binary mask indicating masked positions,
    and the indices needed to restore the original sequence order during decoding.
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
