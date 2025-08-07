import torch
import torch.nn as nn
from mae.patchify import PatchEmbed
from mae.encoder import MAEEncoder
from mae.decoder import MAEDecoder
from mae.utils import random_masking

class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, decoder_dim=512, mask_ratio=0.75):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.mask_ratio = mask_ratio

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.encoder = MAEEncoder(embed_dim)
        self.decoder = MAEDecoder(embed_dim, decoder_dim)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x_masked, mask, ids_restore = random_masking(x, self.mask_ratio)
        encoded = self.encoder(x_masked)

        # create tokens for masked patches
        B, L, D = x.shape
        masked_tokens = torch.zeros(B, L, D, device=x.device)
        masked_tokens.scatter_(1, ids_restore.unsqueeze(-1), encoded)

        decoded = self.decoder(masked_tokens)
        return decoded, x, mask
