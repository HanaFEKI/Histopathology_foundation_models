# Aggregation Techniques in Computational Pathology

Aggregation is a fundamental component in computational pathology models, particularly in **Multiple Instance Learning (MIL)** settings. It refers to the method used to combine patch-level or token-level features into a single global slide-level representation. The choice of aggregation can significantly impact model performance, especially for whole slide images (WSIs), which are extremely large and often processed in smaller patches.

## 📌 Why Aggregation Matters

In digital pathology, we usually do **not** have labels for individual patches — only for the **entire slide** (e.g., tumor vs. normal). Therefore, aggregation functions are essential to convert a set of patch embeddings (instances) into a **single vector** suitable for downstream tasks like classification or report generation.

Aggregation helps:
- Reduce computational burden
- Preserve global context from localized features
- Enable end-to-end learning from weak labels (WSI-level)

## 🧠 Aggregation Methods

| **Method** | **Learnable** | **Description** | **Pros** | **Cons** | **Example Models** |
|------------|---------------|-----------------|----------|----------|---------------------|
| **Mean Pooling (Avg MIL)** | ❌ | Computes the **average** of all patch embeddings to represent the whole slide. Simple and effective. | ✅ Fast and parameter-free<br>✅ Easy to implement | ❌ Treats all patches equally<br>❌ May dilute critical features | ViT (baseline), CLAM (avg variant) |
| **Max Pooling** | ❌ | Selects the **most activated patch** (highest value) to represent the slide. Assumes one key patch determines the label. | ✅ Highlights relevant regions<br>✅ No training needed | ❌ Ignores all other patches<br>❌ Highly sensitive to noise | DeepMIL |
| **Attention-based MIL** | ✅ | From [Ilse et al., 2018](https://arxiv.org/abs/1802.04712). Learns attention weights to focus on informative patches. Aggregates a weighted sum of patches.| ✅ Learns patch importance<br>✅ Interpretable<br>✅ Works well in WSI settings | ❌ May overfit on small data<br>❌ Adds parameters | CLAM, TransMIL, iBOT |
| **Gated Attention MIL** | ✅ | A variant of attention pooling that uses both **tanh** and **sigmoid** nonlinearities to refine attention signals. | ✅ More expressive<br>✅ Better patch discrimination | ❌ Requires more compute<br>❌ Slightly complex | CLAM (gated), ABCMIL |
| **Transformer Aggregation** | ✅ | Uses **transformer self-attention** to model relationships between patches and produce a global token or aggregate feature. | ✅ Captures rich interactions<br>✅ Scales well with data<br>✅ Powerful for WSI classification | ❌ Memory intensive<br>❌ Longer training time | TransMIL, UNI, UNIv2, RudolfV |
| **Learnable Pooling (NetVLAD, DeepSets, etc.)** | ✅ | Advanced aggregation techniques for unordered inputs. NetVLAD uses soft assignments to learned cluster centers. DeepSets use permutation-invariant architectures. | ✅ Models global structure<br>✅ Flexible | ❌ Less interpretable<br>❌ Heavier to tune | PathFormer, retrieval models |



### 🔷 1. Attention-Based MIL

From [Ilse et al., 2018](https://arxiv.org/abs/1802.04712), this method learns **attention weights** to determine the importance of each instance (e.g., patch).

#### 📘 Formula

```math
z = \sum_{i=1}^{n} \alpha_i h_i \quad \text{where } 
\alpha_i = \frac{\exp(w^T \tanh(V h_i^T))}{\sum_{j=1}^{n} \exp(w^T \tanh(V h_j^T))}
```

#### 🔍 Explanation of Terms

| Variable   | Meaning                                         |
|------------|-------------------------------------------------|
| `h_i`      | Feature vector of instance `i`                  |
| `z`        | Aggregated global representation                |
| `V`        | Weight matrix applied to each instance (`tanh`) |
| `w`        | Weight vector to score each transformed patch   |
| `α_i`      | Attention weight for instance `i` (softmax)     |
| `tanh`     | Non-linearity improving expressiveness          |


### 🔶 2. Gated Attention MIL

A variant also from [Ilse et al., 2018](https://arxiv.org/abs/1802.04712), using both `tanh` and `sigmoid` activations to introduce a gating mechanism.

#### 📘 Formula

```math
\alpha_i = \frac{\exp(w^T ( \tanh(V h_i^T) \odot \sigma(U h_i^T) ))}{\sum_{j=1}^{n} \exp(w^T ( \tanh(V h_j^T) \odot \sigma(U h_j^T) ))}
```

#### 🔍 Explanation of Terms

| Variable       | Meaning                                                   |
|----------------|-----------------------------------------------------------|
| `U`            | Weight matrix for the sigmoid gate                        |
| `σ(U h_i^T)`   | Sigmoid gate (importance controller)                      |
| `⊙`            | Element-wise multiplication (gating between two branches) |
| Rest           | Same as above                                             |

### ⚖️ Comparison

| Feature                | Attention MIL                        | Gated Attention MIL                    |
|------------------------|--------------------------------------|----------------------------------------|
| Learnable Parameters   | `V`, `w`                             | `V`, `U`, `w`                          |
| Activation Functions   | `tanh`                               | `tanh` + `sigmoid`                     |
| Gating                 | ❌                                   | ✅                                     |
| Expressiveness         | Medium                               | High                                  |
| Complexity             | Low                                  | Moderate                              |
| Usage Examples         | CLAM, TransMIL, iBOT                 | CLAM-Gated, ABCMIL                    |


## 📂 Implementations
You can find implementations or scripts for several of these aggregation types in:

| File                      | Description                                |
|---------------------------|--------------------------------------------|
| `mean_pooling.py`         | Implements average (mean) pooling          |
| `attention_pooling.py`    | Implements standard attention-based MIL    |
| `gated_attention.py`      | Implements gated attention-based MIL       |
| `transformer_agg.py`      | Aggregation using Transformer encoder      |

---

## 📂 References

- Ilse et al., 2018 — [Attention-based Deep MIL](https://arxiv.org/abs/1802.04712)
- Lu et al., 2021 — CLAM ([code](https://github.com/mahmoodlab/CLAM))
- Shao et al., 2021 — TransMIL ([paper](https://arxiv.org/abs/2106.00908))
- UNI/UNIv2 — [Virchow Repo](https://github.com/BatsResearch/Virchow)
- ABCMIL, iBOT — see relevant GitHub implementations


> 🛠️ Feel free to extend this by plugging in your models and observing which aggregation style best suits your pathology tasks!
