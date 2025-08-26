import torch
import torch.nn as nn
import math

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        # x: queries, y: keys/values
        y = x if y is None else y
        B, N, D = x.shape
        qkv = self.qkv(torch.cat([x, y, y], dim=0) if x is not y else x)
        # split q,k,v manually if needed
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # simple for self-attention

        # reshape for multi-heads
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(self.dropout(out))

# Feed Forward Block
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ff(self.norm2(x))
        return x

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
        x = x + self.cross_attn(self.norm2(x), y=enc_out, mask=memory_mask)
        x = x + self.ff(self.norm3(x))
        return x

# Full Transformer Encoder-Decoder
  class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, ff_dim=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, embed_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_decoder_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, tgt_vocab)

    def encode(self, src, src_mask=None):
        x = self.src_embed(src) + self.pos_embed[:, :src.size(1), :]
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.tgt_embed(tgt) + self.pos_embed[:, :tgt.size(1), :]
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, memory_mask)
        return self.out_proj(out)

# Example usage
if __name__ == "__main__":
    src_vocab, tgt_vocab = 1000, 1000
    B, N_src, N_tgt = 2, 10, 12
    src = torch.randint(0, src_vocab, (B, N_src))
    tgt = torch.randint(0, tgt_vocab, (B, N_tgt))

    model = Transformer(src_vocab, tgt_vocab, embed_dim=128, num_heads=8, num_encoder_layers=3, num_decoder_layers=3, ff_dim=512)
    out = model(src, tgt)
    print("Transformer output shape:", out.shape)  # [B, N_tgt, tgt_vocab]
