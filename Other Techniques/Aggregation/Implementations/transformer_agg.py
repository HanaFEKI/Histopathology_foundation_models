import torch
import torch.nn as nn

class TransformerAggregation(nn.Module):
    """
    Transformer-based aggregation of patch embeddings.
    Each patch attends to others to form a context-aware global representation.
    """
    def __init__(self, input_dim, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): [batch_size, num_patches, embedding_dim]
        Returns:
            Tensor: [batch_size, embedding_dim]
        """
        x = self.transformer(x)  # [B, N, D]
        z = x.mean(dim=1)        # Global token = mean of all context-aware patches
        return z
