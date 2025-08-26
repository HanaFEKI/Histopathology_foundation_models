import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) module for a linear layer.
    W_adapted = W + alpha * (A @ B)
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Low-rank matrices
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        # Original weight (frozen during fine-tuning)
        self.W = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

    def forward(self, x):
        # Compute LoRA adaptation
        lora_update = self.alpha * (self.A @ self.B)
        W_eff = self.W + lora_update
        return x @ W_eff.T

# Example usage
if __name__ == "__main__":
    x = torch.randn(8, 128)  # batch of 8, input dim 128
    layer = LoRALayer(128, 64, rank=8, alpha=0.5)
    y = layer(x)
    print(y.shape)  # should be [8, 64]
