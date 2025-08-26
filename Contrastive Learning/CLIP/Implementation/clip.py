import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# Tokenizer Class
class ToyTokenizer:
    """Minimal whitespace + BPE-like tokenizer (replace with real BPE for real use)"""
    def __init__(self):
        self.stoi = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
        self.itos = list(self.stoi.keys())

    def build_vocab(self, texts, min_freq=1):
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())
        for tok, c in counter.items():
            if c >= min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text, add_bos=True, add_eos=True):
        toks = text.lower().split()
        ids = [self.stoi["<bos>"]] if add_bos else []
        ids += [self.stoi.get(t, self.stoi["<unk>"]) for t in toks]
        if add_eos: ids.append(self.stoi["<eos>"])
        return ids

    @property
    def pad_id(self): return self.stoi["<pad>"]
    def __len__(self): return len(self.itos)


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=77):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, width=512, ctx_length=77, heads=8, layers=6):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_encoding = PositionalEncoding(width, max_len=ctx_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, batch_first=True, dim_feedforward=width*4, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln_final = nn.LayerNorm(width)
        self.text_proj = nn.Linear(width, width, bias=False)

    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        key_padding_mask = ~attention_mask.bool()
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.ln_final(x)
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        pooled = (x*attention_mask.unsqueeze(-1)).sum(dim=1)/lengths
        return self.text_proj(pooled)

# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(2048, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        feats = self.backbone(x).flatten(1)
        proj = self.proj(feats)
        return self.norm(proj)

# CLIP Model
class CLIPModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, width=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

    def forward(self, images, input_ids, attention_mask):
        img_emb = F.normalize(self.image_encoder(images), dim=-1)
        txt_emb = F.normalize(self.text_encoder(input_ids, attention_mask), dim=-1)
        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        logits_per_image = logit_scale * img_emb @ txt_emb.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, img_emb, txt_emb


# Trainer class
class CLIPTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def contrastive_loss(self, logits_i, logits_t):
        B = logits_i.size(0)
        targets = torch.arange(B, device=logits_i.device)
        loss_i = F.cross_entropy(logits_i, targets)
        loss_t = F.cross_entropy(logits_t, targets)
        return (loss_i + loss_t)/2

    def train_step(self, images, input_ids, attention_mask):
        self.model.train()
        logits_i, logits_t, _, _ = self.model(images, input_ids, attention_mask)
        loss = self.contrastive_loss(logits_i, logits_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), self.model.logit_scale.exp().item()


# Zero-shot classifier
class CLIPZeroShot:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def classify(self, images, classnames, prompt_templates):
        device = next(self.model.parameters()).device
        text_embeds = []
        for cname in classnames:
            prompts = [tmpl.format(cname) for tmpl in prompt_templates]
            input_ids = []
            for p in prompts:
                input_ids.append(self.tokenizer.encode(p))
            max_len = max(len(x) for x in input_ids)
            ids_tensor = torch.full((len(prompts), max_len), self.tokenizer.pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(ids_tensor)
            for i, seq in enumerate(input_ids):
                L = len(seq)
                ids_tensor[i, :L] = torch.tensor(seq, device=device)
                attention_mask[i, :L] = 1
            _, _, _, txt_emb = self.model(images[:1]*0, ids_tensor, attention_mask)
            txt_emb = F.normalize(txt_emb, dim=-1)
            text_embeds.append(txt_emb.mean(dim=0))
        text_matrix = torch.stack(text_embeds, dim=1)
        _, _, img_emb, _ = self.model(images, ids_tensor[:1], attention_mask[:1])
        img_emb = F.normalize(img_emb, dim=-1)
        logits = img_emb @ text_matrix
        probs = logits.softmax(dim=-1)
        return probs
