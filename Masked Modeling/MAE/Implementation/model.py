import torch
import torch.nn as nn
from mae.patchify import PatchEmbed
from mae.encoder import MAEEncoder
from mae.decoder import MAEDecoder
from mae.utils import random_masking

class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, decoder_dim=512, mask_ratio=0.75,
                 encoder_depth=12, decoder_depth=8, num_heads=12):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.encoder = MAEEncoder(embed_dim, encoder_depth, num_heads)
        self.decoder = MAEDecoder(embed_dim, decoder_dim, embed_dim, decoder_depth, num_heads)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # Step 1: Embed patches
        x = self.patch_embed(x) + self.pos_embed  # (B, N, D)

        # Step 2: Random masking
        x_masked, mask, ids_restore = random_masking(x, self.mask_ratio)

        # Step 3: Encode visible patches
        encoded = self.encoder(x_masked)

        # Step 4: Prepare full sequence with masked tokens
        B, N, D = x.shape
        decoder_tokens = torch.zeros(B, N, D, device=x.device)
        decoder_tokens.scatter_(1, ids_restore.unsqueeze(-1), encoded)

        # Step 5: Decode and reconstruct
        reconstructed = self.decoder(decoder_tokens)

        return reconstructed, x, mask
