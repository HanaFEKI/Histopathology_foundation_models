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

## DINO-based & SSL Advances

This section is inspired by the comprehensive review by [Bilal et al., 2025](https://arxiv.org/abs/2502.08333).  
It summarizes recent **self-supervised learning (SSL)** and **DINO-based** models for whole slide image (WSI) analysis in computational pathology.

| Model            | Paper / Link                                                                 | Official Link                                | My Implementation & Explanation            | Characteristics |
|------------------|------------------------------------------------------------------------------|-----------------------------------------------|---------------------------------------------|------------------|
| Virchow          | [arXiv:2309.07778](https://arxiv.org/pdf/2309.07778)                         | 🤗 [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) | [models/virchow](./models/virchow)         | —                |
| Virchow2         | [arXiv:2403.10870](https://arxiv.org/abs/2403.10870)                         | 🐙 [BatsResearch/Virchow](https://github.com/BatsResearch/Virchow) | [models/virchow2](./models/virchow2)       | —                |
| Virchow2G        | [arXiv:2403.10870](https://arxiv.org/abs/2403.10870)                         | 🐙 [BatsResearch/Virchow](https://github.com/BatsResearch/Virchow) | —                                           | —                |
| OmniScreen       | [arXiv:2403.10870](https://arxiv.org/abs/2403.10870)                         | 🐙 [BatsResearch/Virchow](https://github.com/BatsResearch/Virchow) | —                                           | —                |
| H-Optimus-0      | —                                                                            | —                                             | —                                           | —                |
| Kaiko-ai         | —                                                                            | —                                             | —                                           | —                |
| UNI              | —                                                                            | —                                             | —                                           | —                |
| BROW             | —                                                                            | —                                             | —                                           | —                |
| Phikon           | [arXiv:2311.11023](https://arxiv.org/abs/2311.11023)                         | —                                             | [models/phikon](./models/phikon)           | —                |
| HIPT             | [arXiv:2206.02680](https://arxiv.org/abs/2206.02680)                         | 🐙 [mahmoodlab/HIPT](https://github.com/mahmoodlab/HIPT) | —                                           | —                |
| CTransPath       | [arXiv:2209.05578](https://arxiv.org/abs/2209.05578)                         | 🐙 [Bin-Chen-Lab/CTransPath](https://github.com/Bin-Chen-Lab/CTransPath) | [models/ctranspath](./models/ctranspath) | —                |
| Phikon-v2        | [arXiv:2311.11023](https://arxiv.org/abs/2311.11023)                         | —                                             | —                                           | —                |
| TissueConcepts   | —                                                                            | —                                             | —                                           | —                |
| PLUTO            | [arXiv:2403.00827](https://arxiv.org/abs/2403.00827)                         | —                                             | [models/pluto](./models/pluto)             | —                |
| Hibou-B          | [arXiv:2406.06589](https://arxiv.org/abs/2406.06589)                         | —                                             | —                                           | —                |
| Hibou-L          | [arXiv:2406.06589](https://arxiv.org/abs/2406.06589)                         | —                                             | —                                           | —                |
| Madeleine        | —                                                                            | —                                             | —                                           | —                |
| PathoDuet        | [arXiv:2403.09677](https://arxiv.org/abs/2403.09677)                         | —                                             | [models/pathoduet](./models/pathoduet)     | —                |
| RudolfV          | [arXiv:2403.01821](https://arxiv.org/abs/2403.01821)                         | —                                             | —                                           | —                |
| REMEDIS          | [arXiv:2212.08677](https://arxiv.org/abs/2212.08677)                         | 🐙 [boschresearch/remedis](https://github.com/boschresearch/remedis) | —                                     | —                |
| BEPH             | —                                                                            | —                                             | —                                           | —                |
| COBRA            | [arXiv:2405.20233](https://arxiv.org/abs/2405.20233)                         | 🐙 [BatsResearch/COBRA](https://github.com/BatsResearch/COBRA) | —                                       | —                |



## 📁 Repo Structure

