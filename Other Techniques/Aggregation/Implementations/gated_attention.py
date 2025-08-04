import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionPooling(nn.Module):
    """
    Gated Attention-based MIL pooling.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): [batch_size, num_patches, embedding_dim]
        Returns:
            Tensor: [batch_size, embedding_dim]
        """
        tanh_out = torch.tanh(self.V(x))
        sigm_out = torch.sigmoid(self.U(x))
        gated = tanh_out * sigm_out
        A = self.w(gated)
        A = torch.softmax(A, dim=1)
        z = torch.sum(A * x, dim=1)
        return z
