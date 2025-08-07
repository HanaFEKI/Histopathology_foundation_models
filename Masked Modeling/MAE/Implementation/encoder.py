import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock

class MAEEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.blocks = nn.Sequential(
            *[EncoderBlock(embed_dim, num_heads, mlp_dim=embed_dim * 4) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.blocks(x)
        return self.norm(x)
