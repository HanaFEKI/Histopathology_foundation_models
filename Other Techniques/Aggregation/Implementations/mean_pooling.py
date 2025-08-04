import torch
import torch.nn as nn

class MeanPooling(nn.Module):
    """
    Simple average pooling over patch embeddings.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): [batch_size, num_patches, embedding_dim]
        Returns:
            Tensor: [batch_size, embedding_dim]
        """
        return x.mean(dim=1)
