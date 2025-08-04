# Aggregation Techniques in Computational Pathology

Aggregation is a fundamental component in computational pathology models, particularly in **Multiple Instance Learning (MIL)** settings. It refers to the method used to combine patch-level or token-level features into a single global slide-level representation. The choice of aggregation can significantly impact model performance, especially for whole slide images (WSIs), which are extremely large and often processed in smaller patches.

## 📌 Why Aggregation Matters

In digital pathology, we usually do **not** have labels for individual patches — only for the **entire slide** (e.g., tumor vs. normal). Therefore, aggregation functions are essential to convert a set of patch embeddings (instances) into a **single vector** suitable for downstream tasks like classification or report generation.

Aggregation helps:
- Reduce computational burden
- Preserve global context from localized features
- Enable end-to-end learning from weak labels (WSI-level)

## 🧠 Aggregation Techniques

| **Method** | **Learnable** | **Description** | **Pros** | **Cons** | **Example Models** |
|------------|---------------|-----------------|----------|----------|---------------------|
| **Mean Pooling (Avg MIL)** | ❌ | Computes the **average** of all patch embeddings to represent the whole slide. Simple and effective. | ✅ Fast and parameter-free<br>✅ Easy to implement | ❌ Treats all patches equally<br>❌ May dilute critical features | ViT (baseline), CLAM (avg variant) |
| **Max Pooling** | ❌ | Selects the **most activated patch** (highest value) to represent the slide. Assumes one key patch determines the label. | ✅ Highlights relevant regions<br>✅ No training needed | ❌ Ignores all other patches<br>❌ Highly sensitive to noise | DeepMIL |
| **Attention-based MIL** | ✅ | From [Ilse et al., 2018](https://arxiv.org/abs/1802.04712). Learns attention weights to focus on informative patches. Aggregates a weighted sum of patches. <br><br>**Formula:**<br>\( z = \sum_{i=1}^{n} \alpha_i h_i \), where:<br>\( \alpha_i = \frac{\exp(w^T \tanh(V h_i^T))}{\sum_j \exp(w^T \tanh(V h_j^T))} \) | ✅ Learns patch importance<br>✅ Interpretable<br>✅ Works well in WSI settings | ❌ May overfit on small data<br>❌ Adds parameters | CLAM, TransMIL, iBOT |
| **Gated Attention MIL** | ✅ | A variant of attention pooling that uses both **tanh** and **sigmoid** nonlinearities to refine attention signals. | ✅ More expressive<br>✅ Better patch discrimination | ❌ Requires more compute<br>❌ Slightly complex | CLAM (gated), ABCMIL |
| **Transformer Aggregation** | ✅ | Uses **transformer self-attention** to model relationships between patches and produce a global token or aggregate feature. | ✅ Captures rich interactions<br>✅ Scales well with data<br>✅ Powerful for WSI classification | ❌ Memory intensive<br>❌ Longer training time | TransMIL, UNI, UNIv2, RudolfV |
| **Learnable Pooling (NetVLAD, DeepSets, etc.)** | ✅ | Advanced aggregation techniques for unordered inputs. NetVLAD uses soft assignments to learned cluster centers. DeepSets use permutation-invariant architectures. | ✅ Models global structure<br>✅ Flexible | ❌ Less interpretable<br>❌ Heavier to tune | PathFormer, retrieval models |

## 🔍 Detailed Formula Explanation (Attention-based MIL)

From Ilse et al., 2018:

``` math
z = \sum_{i=1}^{n} \alpha_i h_i \quad \text{where } \alpha_i = \frac{\exp(w^T \tanh(Vh_i^T))}{\sum_j \exp(w^T \tanh(Vh_j^T))}
```

| Variable | Meaning |
|--------|---------|
| \( h_i \in \mathbb{R}^d \) | Embedding of patch \(i\) |
| \( z \in \mathbb{R}^d \) | Aggregated slide-level embedding |
| \( V \in \mathbb{R}^{l \times d} \) | Weight matrix for attention |
| \( w \in \mathbb{R}^l \) | Weight vector to score each patch |
| \( \tanh \) | Activation function (nonlinearity) |
| \( \alpha_i \) | Importance score of patch \(i\), normalized with softmax |


## 📂 References

- Ilse et al., 2018 — [Attention-based Deep MIL](https://arxiv.org/abs/1802.04712)
- Lu et al., 2021 — CLAM ([code](https://github.com/mahmoodlab/CLAM))
- Shao et al., 2021 — TransMIL ([paper](https://arxiv.org/abs/2106.00908))
- UNI/UNIv2 — [Virchow Repo](https://github.com/BatsResearch/Virchow)
- ABCMIL, iBOT — see relevant GitHub implementations

## Implementation

You can find implementations or scripts for several of these aggregation types in:
``` bash
./explanations/aggregation/mean_pooling.py
./explanations/aggregation/attention_pooling.py
./explanations/aggregation/gated_attention.py
./explanations/aggregation/transformer_agg.py
```

> 🛠️ Feel free to extend this by plugging in your models and observing which aggregation style best suits your pathology tasks!
