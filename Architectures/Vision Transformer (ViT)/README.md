# 🔹 Vision Transformer (ViT) Architecture

We have already tackled the **encoder-decoder Transformer** for sequence modeling.  
The **Vision Transformer (ViT)** builds on the same principles of attention but adapts it for **images** instead of text sequences. The key idea is to treat an image as a sequence of patches.

---

## 1. Image as a Sequence of Patches

An input image 
```math 
X \in \mathbb{R}^{H \times W \times C}
```
is split into ```N``` non-overlapping patches of size ```P``` x ```P```:

```math
N = \frac{H \cdot W}{P^2}
```
This produces a sequence:
```math
X_{seq} = [x_1, x_2, ..., x_N]
```

- W_e ∈ R^(P^2 * C) × D is the patch embedding matrix  
- D is the embedding dimension

## 2. Adding a Class Token & Positional Embeddings

Similar to positional embeddings in text Transformers, ViT adds positional information:
```math
X_{input} = [x_{cls}, x_1, x_2, ..., x_N] + E_{pos}
```

- x_cls is a learnable class token used for image-level classification  
- E_pos ∈ R^(N+1) × D encodes patch positions

## 3. Transformer Encoder (Same as Before)

The sequence of patches is fed into a stack of Transformer encoder layers, exactly like the encoder we already explained:

``` math
z_0 = X_{input}
```

- Multi-head self-attention captures relationships between patches  
- Feed-forward networks add non-linear transformations  
- LayerNorm and residual connections are identical to the original Transformer  

The final class embedding `z_cls` is used for downstream classification:
```math
y_{pred} = MLPHead(z_{cls})
```

## 4. Key Differences from Text Transformer

| Aspect             | Text Transformer       | Vision Transformer (ViT)         |
|-------------------|----------------------|---------------------------------|
| Input              | Token embeddings      | Flattened image patches          |
| Sequence length    | Number of tokens      | Number of patches + 1 (class token) |
| Positional embeddings | Learnable or sinusoidal | Learnable per patch            |
| Decoder            | Optional for seq2seq  | Typically only encoder (classification) |
| Output             | Token predictions     | Class label or patch embeddings |


## 5. Mathematical Intuition

- **Patch attention** is equivalent to token attention in NLP:
``` math
Attention(Q, K, V) = softmax(Q K^T / sqrt(D)) V
```

- **Class token** aggregates global information from all patches:
``` math
z_{cls} = Attention(x_{cls}, [x_1, ..., x_N], [x_1, ..., x_N])
```

- The final prediction is a simple linear projection of the class token:
``` math
y_{pred} = W_o z_{cls} + b_o
```








