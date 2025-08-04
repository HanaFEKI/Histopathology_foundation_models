import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    Attention-based MIL pooling (Ilse et al., 2018).
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): [batch_size, num_patches, embedding_dim]
        Returns:
            Tensor: [batch_size, embedding_dim]
        """
        A = self.w(torch.tanh(self.V(x)))  # [B, N, 1]
        A = torch.softmax(A, dim=1)        # Normalize attention weights
        z = torch.sum(A * x, dim=1)        # Weighted sum
        return z
