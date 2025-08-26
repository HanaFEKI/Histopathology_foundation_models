# 🔹 BEiT: Bidirectional Encoder representation for Image Transformers

BEiT is a **Vision Transformer** variant that uses **masked image modeling (MIM)** for pretraining.  
It treats an image as a sequence of patches, predicts masked patches in a self-supervised manner, and learns rich image representations.

---

## 1. Pipeline Overview

1. **Input Image**  
   - An image
 ``` math
X \in \mathbb{R}^{H \times W \times C}
```
2. **Patch Embedding**  
   - Split image into non-overlapping patches of size P×P  
   - Flatten each patch and project to embedding vector  
   ```math
   x_p = Flatten(X_{patch}) W_e + b_e
   
3. Add Class Token & Positional Embedding
  - cls token aggregates global info
  - Positional embeddings encode patch location
``` math
X_{input} = [x_{cls}, x_1, ..., x_N] + E_{pos}
```
4. Transformer Encoder
  - Stack of N encoder layers
  - Each layer: Multi-head self-attention + MLP + residual connections

5. Masked Patch Prediction (for pretraining)
  - Randomly mask patches
  - Reconstruct masked patches from context

## 2. Mathematical Intuition

- **Attention between patches**:
``` math
Attention(Q, K, V) = softmax(Q K^T / sqrt(D)) V
```

- **Class token aggregates information**:
``` math
z_{cls} = Attention(x_{cls}, [x_1, ..., x_N], [x_1, ..., x_N])
```

- **Masked patch prediction**:
```math
y_{masked} = Decoder(z_{masked})
```
