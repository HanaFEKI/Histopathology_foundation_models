# Aggregation Techniques in Computational Pathology

Aggregation is a fundamental component in computational pathology models, particularly in **Multiple Instance Learning (MIL)** settings. It refers to the method used to combine patch-level or token-level features into a single global slide-level representation. The choice of aggregation can significantly impact model performance, especially for whole slide images (WSIs), which are extremely large and often processed in smaller patches.

## 📌 Why Aggregation Matters

In digital pathology, we usually do **not** have labels for individual patches — only for the **entire slide** (e.g., tumor vs. normal). Therefore, aggregation functions are essential to convert a set of patch embeddings (instances) into a **single vector** suitable for downstream tasks like classification or report generation.

Aggregation helps:
- Reduce computational burden
- Preserve global context from localized features
- Enable end-to-end learning from weak labels (WSI-level)

## 🧠 Categories of Aggregation Techniques

### 1. 🔘 **Mean Pooling (Average MIL)**
Simple and effective baseline where the global representation is the **average** of all patch embeddings.

- **Pros:** No learnable parameters, fast
- **Cons:** Treats all patches equally — even irrelevant ones

**Example Models:**
- CLAM (with average pooling)
- Some variants of ViT pretraining on WSIs

### 2. 🔺 **Max Pooling**
Selects the most **informative patch** (highest activation) across the slide.

- **Pros:** Focuses on the most relevant region
- **Cons:** Sensitive to noise and may ignore relevant contextual information

**Example Models:**
- Early MIL models like DeepMIL (Ilse et al.)

### 3. 💡 **Attention-based MIL**
Introduced in the seminal work of [Ilse et al., 2018](https://arxiv.org/abs/1802.04712), attention pooling **learns to weight** each patch based on its importance for the task.

- **Pros:** Learns which patches matter
- **Cons:** Slightly more compute; may overfit on small datasets

**Key Formula:**
> \( z = \sum_{i=1}^{n} \alpha_i h_i \quad \text{where } \alpha_i = \frac{\exp(w^T \tanh(Vh_i^T))}{\sum_j \exp(w^T \tanh(Vh_j^T))} \)

**Example Models:**
- CLAM (Lu et al., 2021)
- TransMIL (Shao et al., 2021)
- iBOT (in pathology pretraining)


### 4. 🧠 **Gated Attention (Gated MIL)**
A refinement of attention-based MIL that combines both **tanh** and **sigmoid** nonlinearities for better gating.

> \( \alpha_i = \frac{\exp(w^T (\tanh(Vh_i^T) \odot \sigma(Uh_i^T)))}{\sum_j \exp(w^T (\tanh(Vh_j^T) \odot \sigma(Uh_j^T)))} \)

- **Pros:** Better patch selection control
- **Cons:** Slightly more complex to train

**Example Models:**
- CLAM w/ gated attention variant
- ABCMIL


### 5. 🧭 **Transformer-based Aggregation**
Uses self-attention (from transformer encoders) across patch embeddings. Supports richer interaction between patches.

- **Pros:** Captures relationships between patches
- **Cons:** Memory-intensive

**Example Models:**
- TransMIL
- UNI / UNIv2 (BatsResearch)
- RudolfV (ViT-based pretraining)


### 6. ⚖️ **Learnable Pooling (NetVLAD, DeepSets, etc.)**
Advanced aggregation mechanisms designed for more structured or unordered inputs.

- **Pros:** Adaptable; can learn global structure
- **Cons:** Less interpretable, harder to tune

**Example Models:**
- PathFormer variants
- NetVLAD-based WSIs encoders in retrieval


## 🧪 Summary Table

| Aggregation Method     | Learnable? | Robustness | Interpretability | Common in Pathology? | Example Models                |
|------------------------|------------|------------|------------------|----------------------|-------------------------------|
| Mean Pooling           | ❌         | Medium     | ❌               | ✅                   | CLAM (baseline), ViT          |
| Max Pooling            | ❌         | Low        | ❌               | ✅                   | DeepMIL                       |
| Attention-based MIL    | ✅         | High       | ✅               | ✅✅✅                | CLAM, iBOT, TransMIL          |
| Gated Attention MIL    | ✅         | High       | ✅               | ✅✅                 | CLAM (variant), ABCMIL        |
| Transformer Attention  | ✅         | High       | ✅✅              | ✅✅✅                | TransMIL, RudolfV, UNI        |
| NetVLAD / DeepSets     | ✅         | High       | ❌               | 🔁 Occasionally      | PathFormer, retrieval models  |

## 📚 Sources

- Ilse et al., "Attention-based Deep Multiple Instance Learning" – [arXiv:1802.04712](https://arxiv.org/abs/1802.04712)
- Lu et al., "CLAM: Weakly-Supervised Classification Using Attention-based Multiple Instance Learning" – [Code](https://github.com/mahmoodlab/CLAM)
- Shao et al., "TransMIL: Transformer based MIL for WSI Classification" – [arXiv:2106.00908](https://arxiv.org/abs/2106.00908)
- Lu et al., "Data-efficient and weakly supervised computational pathology on whole-slide images" – [Nature Biomedical Engineering](https://www.nature.com/articles/s41551-021-00814-0)
- Diao et al., "Human-Centric Whole Slide Image Pretraining" – [arXiv:2403.10870](https://arxiv.org/abs/2403.10870)

## 🧠 Implementation Examples

You can find implementations or scripts for several of these aggregation types in:
``` bash
./explanations/aggregation/mean_pooling.py
./explanations/aggregation/attention_pooling.py
./explanations/aggregation/gated_attention.py
./explanations/aggregation/transformer_agg.py
```

> 🛠️ Feel free to extend this by plugging in your models and observing which aggregation style best suits your pathology tasks!
