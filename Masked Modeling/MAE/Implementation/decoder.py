import torch.nn as nn

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_dim=512, patch_dim=768, depth=8, num_heads=16):
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim)

        self.blocks = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)
