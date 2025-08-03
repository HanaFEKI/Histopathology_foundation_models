# 🧬 Histopathology Foundation Models

**Inspired by:**  
> Bilal, M., Aadam, M., Raza, M., Altherwy, Y., Alsuhaibani, A., Abduljabbar, A., Almarshad, F., Golding, P., & Rajpoot, N. (2025). *Foundation Models in Computational Pathology: A Review of Challenges, Opportunities, and Impact*. arXiv:2502.08333. https://arxiv.org/abs/2502.08333

---

## 📣 Introduction

Welcome! This repository is inspired by the excellent review by Bilal et al. (2025) on foundation models in computational pathology. It aims to be your **one-stop resource** for understanding and implementing foundation models in histopathology.

Whether you're a **student**, **researcher**, or **practitioner**, this repo will help you:

- ✅ Understand the **core concepts** behind foundation models for medical imaging.  
- 🔍 Access **modular PyTorch implementations** of key models.  
- 📓 Explore **real-life examples and notebooks** that demonstrate practical usage.  
- 🌍 Discover **beyond-histopathology applications** in broader medical vision tasks.

> 🔖 *"I wish I had this repo when I started my internship. It would’ve saved me hours of searching, testing, and debugging. So I created what I wish I had."* — **Hana FEKI**

---

## 🧠 Foundation Models in Histopathology: DINO-based & SSL Advances

This section is inspired by the comprehensive review by [Bilal et al., 2025](https://arxiv.org/abs/2502.08333).  
It summarizes recent **self-supervised learning (SSL)** and **DINO-based** models for whole slide image (WSI) analysis in computational pathology.

| Model            | Architecture       | Parameters | WSI Tiles | Training Algorithm                          | Paper / Link | Implemented | Repo Link | Official GitHub |
|------------------|--------------------|------------|-----------|---------------------------------------------|---------------|-------------|-----------|------------------|
| Virchow          | ViT-H              | 632M       | 1.5M      | DINOv2 (SSL)                                 | [arXiv:2403.10870](https://arxiv.org/abs/2403.10870) | ✅ Yes | [models/virchow](./models/virchow) | [BatsResearch/Virchow](https://github.com/BatsResearch/Virchow) |
| Virchow2         | ViT-H              | 632M       | 3.1M      | DINOv2 (SSL)                                 | [arXiv:2403.10870](https://arxiv.org/abs/2403.10870) | ✅ Yes | [models/virchow2](./models/virchow2) | [BatsResearch/Virchow](https://github.com/BatsResearch/Virchow) |
| Virchow2G        | ViT-G              | 1.9B       | 3.1M      | DINOv2 (SSL)                                 | [arXiv:2403.10870](https://arxiv.org/abs/2403.10870) | ❌ No | — | [BatsResearch/Virchow](https://github.com/BatsResearch/Virchow) |
| OmniScreen       | Virchow2           | 632M       | 48K       | Weakly-Supervised (on Virchow2 embeddings)  | [arXiv:2403.10870](https://arxiv.org/abs/2403.10870) | ❌ No | — | [BatsResearch/Virchow](https://github.com/BatsResearch/Virchow) |
| H-Optimus-0      | ViT-G              | 1.1B       | >500K     | DINOv2 (SSL)                                 | — | ❌ No | — | — |
| Kaiko-ai         | ViT-L              | 303M       | 29K       | DINOv2 (SSL)                                 | — | ❌ No | — | — |
| UNI              | ViT-L              | 307M       | 100K      | DINOv2 (SSL)                                 | — | ❌ No | — | — |
| BROW             | ViT-B              | 86M        | 11K       | DINO (SSL)                                   | — | ❌ No | — | — |
| Phikon           | ViT-B              | 86M        | 6K        | iBOT (Masked Image Modeling)                | [arXiv:2311.11023](https://arxiv.org/abs/2311.11023) | ✅ Yes | [models/phikon](./models/phikon) | — |
| HIPT             | ViT-HIPT           | 10M        | 11K       | DINO (SSL)                                   | [arXiv:2206.02680](https://arxiv.org/abs/2206.02680) | ❌ No | — | [mahmoodlab/HIPT](https://github.com/mahmoodlab/HIPT) |
| CTransPath       | Swin Transformer   | 28M        | 32K       | MoCoV3 (SRCL)                                | [arXiv:2209.05578](https://arxiv.org/abs/2209.05578) | ✅ Yes | [models/ctranspath](./models/ctranspath) | [Bin-Chen-Lab/CTransPath](https://github.com/Bin-Chen-Lab/CTransPath) |
| Phikon-v2        | ViT-L              | 307M       | 58K       | DINOv2 (SSL)                                 | [arXiv:2311.11023](https://arxiv.org/abs/2311.11023) | ❌ No | — | — |
| TissueConcepts   | Swin Transformer   | -          | 7K        | Supervised multi-task learning              | — | ❌ No | — | — |
| PLUTO            | FlexiVit-S         | 22M        | 158K      | DINOv2 + MAE + Fourier-loss                 | [arXiv:2403.00827](https://arxiv.org/abs/2403.00827) | ✅ Yes | [models/pluto](./models/pluto) | — |
| Hibou-B          | ViT-B              | 86M        | 1.1M      | DINOv2 (SSL)                                 | [arXiv:2406.06589](https://arxiv.org/abs/2406.06589) | ❌ No | — | — |
| Hibou-L          | ViT-L              | 307M       | 1.1M      | DINOv2 (SSL)                                 | [arXiv:2406.06589](https://arxiv.org/abs/2406.06589) | ❌ No | — | — |
| Madeleine        | CONCH              | 86M        | 23K       | Multiheaded attention-based MIL             | — | ❌ No | — | — |
| PathoDuet        | ViT-B              | 86M        | 11K       | MoCoV3 extension                             | [arXiv:2403.09677](https://arxiv.org/abs/2403.09677) | ✅ Yes | [models/pathoduet](./models/pathoduet) | — |
| RudolfV          | ViT-L              | 307M       | 103K      | Semi-supervised with DINOv2 (SSL)           | [arXiv:2403.01821](https://arxiv.org/abs/2403.01821) | ❌ No | — | — |
| REMEDIS          | ResNet-152         | 232M       | 29K       | SimCLR (contrastive learning)               | [arXiv:2212.08677](https://arxiv.org/abs/2212.08677) | ❌ No | — | [boschresearch/remedis](https://github.com/boschresearch/remedis) |
| BEPH             | BEiTv2             | 86M        | 11K       | BEiTv2 (SSL)                                 | — | ❌ No | — | — |
| COBRA            | Mamba-2            | 15M        | 3,048     | Self-supervised contrastive learning        | [arXiv:2405.20233](https://arxiv.org/abs/2405.20233) | ❌ No | — | [BatsResearch/COBRA](https://github.com/BatsResearch/COBRA) |



## 📁 Repo Structure

