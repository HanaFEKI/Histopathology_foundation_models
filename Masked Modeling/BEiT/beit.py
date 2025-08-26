import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# --- Patch Embedding ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)
        return x

# --- Transformer Encoder Block ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
    
    def forward(self, x):
        # Self-attention
        x_attn, _ = self.attn(x, x, x)
        x = x + x_attn
        # Feed-forward
        x = x + self.mlp(self.norm2(x))
        return x

# --- BEiT Encoder ---
class BEiT(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768, img_size=224, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.encoder_layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.patch_embed(x)
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        # Transformer blocks
        x = x.transpose(0, 1)  # [N+1, B, D] for nn.MultiheadAttention
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.transpose(0, 1)
        x = self.norm(x)
        return x[:, 0]  # Return cls token
