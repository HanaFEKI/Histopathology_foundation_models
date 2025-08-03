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

## 📊 Overview of Implemented Foundation Models

| Model Name | Paper / Link | Contribution / Apport | Implemented | Link to Implementation |
|------------|--------------|------------------------|-------------|-------------------------|
| CLIP       | [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) | Vision-language pretraining on natural images; adapted to pathology | ✅ Yes | [models/clip](./models/clip) |
| BioViL     | [BioViL (Boecking et al., 2022)](https://arxiv.org/abs/2203.16402) | Vision-language pretrained on biomedical data (PMC+PubMed) | ✅ Yes | [models/biovil](./models/biovil) |
| HIPT       | [HIPT (Chen et al., 2022)](https://arxiv.org/abs/2206.02680) | Hierarchical ViT for whole slide images | ❌ No | — |
| PLIP       | [PLIP (Zhao et al., 2023)](https://arxiv.org/abs/2302.00833) | Language-image pretraining in pathology-specific domain | ✅ Yes | [models/plip](./models/plip) |
| PathoGPT   | [PathoGPT (Imaginary Ref)](https://arxiv.org/abs/xxxx.xxxxx) | Multimodal LLM for diagnostic reasoning | ❌ No | — |
| PaSeg      | [PaSeg (Luo et al., 2023)](https://arxiv.org/abs/2308.XXXXX) | Foundation model for panoptic segmentation in pathology | ✅ Yes | [models/paseg](./models/paseg) |
| SegGPT     | [SegGPT (Wang et al., 2023)](https://arxiv.org/abs/2304.03284) | Promptable segmentation with transformers | ⚠️ Partial | [models/seggpt](./models/seggpt) |

---

## 📁 Repo Structure

