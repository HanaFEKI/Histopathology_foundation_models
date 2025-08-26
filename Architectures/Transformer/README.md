# 🔹 Transformer Encoder-Decoder Implementation

This repository provides a **PyTorch implementation of a full Transformer model** with encoder-decoder architecture, inspired by the original *“Attention is All You Need”* paper. It can be used for sequence-to-sequence tasks, including text translation, masked image modeling, or reconstruction tasks in domains like histopathology.

---

## 🔹 Overview

The Transformer architecture replaces recurrent and convolutional networks with **attention mechanisms**, allowing:

- **Parallel computation** over sequences  
- **Long-range dependency modeling**  
- **Flexible encoder-decoder structure** for seq2seq tasks  

This implementation includes:

1. **Multi-Head Self-Attention**
2. **Feed-Forward Network**
3. **Layer Normalization + Residual connections**
4. **Stackable Encoder and Decoder layers**
5. **Learnable positional embeddings**
6. **Output projection for target vocabulary or reconstruction**

---

## 🔹 Architecture Details

### 1. Multi-Head Attention
- Splits embeddings into multiple heads  
- Computes **scaled dot-product attention** for each head  
- Concatenates outputs and applies a linear projection  

### 2. Feed-Forward Network (FFN)
- Two linear layers with **GELU activation**  
- Provides non-linear transformation of features  

### 3. Encoder Layer
- **Self-attention → Add & Norm → FFN → Add & Norm**  
- Captures relationships between tokens in the **input sequence**

### 4. Decoder Layer
- **Masked self-attention** for target sequence  
- **Cross-attention** with encoder outputs  
- FFN + residual connections  

### 5. Positional Embeddings
- Learnable positional encodings added to token embeddings  
- Enables the model to capture **sequence order**  

### 6. Full Transformer
- Stacks multiple encoder and decoder layers  
- Returns logits over target vocabulary for each position  
- Can handle **variable-length sequences**  

---

## 🔹 Input/Output

- **Encoder input:** `src` sequence `[batch_size, src_seq_len]`  
- **Decoder input:** `tgt` sequence `[batch_size, tgt_seq_len]`  
- **Output:** `[batch_size, tgt_seq_len, tgt_vocab]`  

Supports optional masks:

- **src_mask:** ignore padding in source sequence  
- **tgt_mask:** prevent attending to future tokens (autoregressive)  
- **memory_mask:** mask encoder outputs during cross-attention  

---

## 🔹 Example Usage

```python
import torch
from transformer_full import Transformer

# Example sequences
src_vocab, tgt_vocab = 1000, 1000
src = torch.randint(0, src_vocab, (2, 10))  # batch_size=2, seq_len=10
tgt = torch.randint(0, tgt_vocab, (2, 12))  # batch_size=2, seq_len=12

# Initialize Transformer
model = Transformer(src_vocab, tgt_vocab, embed_dim=128, num_heads=8, num_encoder_layers=3, num_decoder_layers=3, ff_dim=512)

# Forward pass
out = model(src, tgt)
print(out.shape)  # [2, 12, tgt_vocab]
