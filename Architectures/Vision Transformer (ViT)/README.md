# 🔹 Vision Transformer (ViT) Architecture

We have already tackled the **encoder-decoder Transformer** for sequence modeling.  
The **Vision Transformer (ViT)** builds on the same principles of attention but adapts it for **images** instead of text sequences. The key idea is to treat an image as a sequence of patches.

---

## 1. Image as a Sequence of Patches

An input image \(X \in \mathbb{R}^{H \times W \times C}\) is split into \(N\) non-overlapping patches of size \(P \times P\):

```math
N = \frac{H \cdot W}{P^2}
```
This produces a sequence:
```math
X_seq = [x_1, x_2, ..., x_N]
```

- W_e ∈ R^(P^2 * C) × D is the patch embedding matrix  
- D is the embedding dimension

---

## 2. Adding a Class Token & Positional Embeddings

Similar to positional embeddings in text Transformers, ViT adds positional information:
```math
X_input = [x_cls, x_1, x_2, ..., x_N] + E_pos
```

- x_cls is a learnable class token used for image-level classification  
- E_pos ∈ R^(N+1) × D encodes patch positions



