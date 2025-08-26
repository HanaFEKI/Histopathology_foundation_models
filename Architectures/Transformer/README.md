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

The core of the Transformer is the **Scaled Dot-Product Attention** mechanism:

```math 
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
```

-  ```Q``` = Queries  
-  ```K``` = Keys  
-  ```V``` = Values  
-  ```d_k```  = dimensionality of keys  

In **multi-head attention**, we project the inputs into  ```h``` different subspaces, apply attention in each, and then concatenate:

```math 
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
```

where each head is:

```math
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```


### 2. Feed-Forward Network (FFN)

Each encoder and decoder block has a **position-wise feed-forward network** applied independently to each token:

```math
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
```

Often with **GELU activation** instead of ReLU for smoother gradients.


### 3. Encoder Layer

Each encoder layer consists of:

1. **Multi-head self-attention** (tokens attend to each other in the input)  
2. **Add & Norm** (residual connection + layer normalization)  
3. **Feed-forward network**  
4. **Add & Norm**  

Mathematically:

```math
z' = \text{LayerNorm}(x + \text{MultiHeadSelfAttn}(x))
```

```math
z = \text{LayerNorm}(z' + \text{FFN}(z'))
```


### 4. Decoder Layer

The decoder has an additional **masked self-attention** step that prevents attending to future positions (autoregressive):

1. **Masked self-attention** over target sequence ```y``` 
  ```math
   z'_t = \text{LayerNorm}(y + \text{MaskedMultiHeadAttn}(y))
   ```

2. **Cross-attention** with encoder outputs  
   ```math
   z''_t = \text{LayerNorm}(z'_t + \text{MultiHeadAttn}(z'_t, \text{EncoderOutput}, \text{EncoderOutput}))
   ```

3. **Feed-forward network**  
   ```math
   z_t = \text{LayerNorm}(z''_t + \text{FFN}(z''_t))
  ``


### 5. Positional Embeddings

Since the Transformer has **no recurrence or convolution**, positional embeddings provide sequence order information. A common method:

\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]
\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

Alternatively, learnable positional embeddings are added to token embeddings.

### 6. Full Transformer

- The **encoder** stacks \( N \) layers, producing contextualized token embeddings  
- The **decoder** also stacks \( N \) layers, attending both to past tokens (causal mask) and encoder outputs  

Final step:

\[
\hat{y} = \text{softmax}(W_o z_t)
\]

where \( W_o \) projects decoder outputs to vocabulary space.

### 🔹 Key Intuitions

- **Attention ≈ dynamic weighted averaging**: each token looks at others to decide its representation.  
- **Multi-head**: allows capturing different types of relationships (syntax, semantics, spatial patterns).  
- **Encoder-decoder design**: encoder learns representations, decoder generates sequences step by step.  
- **Positional encoding**: injects order since the model has no inherent sense of sequence.  

---
## 🔹 Applications

1. **Sequence-to-Sequence NLP tasks**
   - Machine translation  
   - Text summarization  
   - Question answering  

2. **Masked Image Modeling (MIM)**
   - Encoder processes visible image patches  
   - Decoder reconstructs masked patches  

3. **Histopathology & Medical Imaging**
   - Learn tissue representations from unlabeled slides  
   - Reconstruct missing or corrupted regions  

4. **Time Series / Multi-modal Data**
   - Model complex dependencies in sequences  
   - Cross-attention between modalities  


