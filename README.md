# 🧬 Histopathology Foundation Models

This repository is inspired by the excellent review by [Bilal et al. (2025)](https://arxiv.org/abs/2502.08333) on foundation models in computational pathology. It aims to be your **one-stop resource** for understanding and implementing foundation models in histopathology.

Whether you're a **student**, **researcher**, or **practitioner**, this repo will help you:

- ✅ Understand the **core concepts** behind foundation models for medical imaging.  
- 🔍 Access **modular PyTorch implementations** of key models.  
- 📓 Explore **real-life examples** that demonstrate practical usage.  
- 🌍 Discover **beyond-histopathology applications** in broader medical vision tasks.

> 🔖 *"I wish I had this repo when I started my internship. It would’ve saved me hours of searching, testing, and debugging. So I created what I wish I had."* — **Hana FEKI**
---
## 🔍 What's a Foundation Model?
A foundation model is a large-scale model trained on massive and diverse datasets (often multimodal, such as images, text, or both) in a self-supervised or weakly supervised manner. These models learn general-purpose, high-dimensional representations that capture broad knowledge about the data domain.
Once trained, the frozen encoder can be adapted to a variety of downstream tasks (e.g., classification, segmentation, captioning) by attaching lightweight modules such as MLPs (Multi-Layer Perceptrons) or task-specific heads — often with little or no additional fine-tuning.

## Required Knowledge

In this table, techniques are grouped by research category and ordered by their importance in recent research.

| Category                  | Technique                | Paper / Link                                                        | Description                                                                                           | Explanation & Implementation                   |
|---------------------------|--------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| **[Architectures](Architectures/)**         | Transformer              | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)       | Transformer architecture                                                                               | ⌛[Implementation](./Architectures/Transformer/transformer.py)      |
|                           | ViT                      | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)   | Vision Transformer architecture for image classification                                              |⌛[Implementation](./Architectures/ViT/vit.py)                      |
|                           | Swin Transformer | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | - | ⌛[Explanation](./Architectures/Swin%20Transformer/swin.py) |
| **[Learning Techniques](Learning%20Techniques/)** | Self-Supervised Learning     | [A Survey on Self-supervised Learning: Algorithms, Applications, and Future Trends](https://arxiv.org/abs/2301.05712) | Learning useful representations from unlabeled data without manual annotations                      | ✅ [Explanation](/Learning%20Techniques/Self-Supervised%20Learning/README.md)                         |
|                           | Semi-Supervised Learning | [A Survey on Deep Semi-supervised Learning](https://arxiv.org/abs/2103.00550)                        | Learning from small labeled and large unlabeled datasets                                              | ✅ [Explanation](./Learning%20Techniques/Semi-Supervised%20Learning/README.md)   |
|                           | Weakly Supervised Learning | [A Critical Look at Weakly Supervised Learning](https://arxiv.org/abs/2305.17442)                  | Learning from weak/noisy labels instead of fully annotated data                                      | ✅ [Explanation](./Learning%20Techniques/Weakly%20Supervised%20Learning/README.md) |
|                           | Unsupervised Learning    | [Semi-Supervised and Unsupervised Deep Visual Learning: A Survey](https://arxiv.org/abs/2208.11296) | Learning patterns from unlabeled data                                                                 | ✅ [Explanation](./Learning%20Techniques/Unsupervised%20Learning/README.md)      |
|                           | Multiple Instance Learning (MIL) | [MIL: A Review of Recent Advances and Applications in Deep Learning](https://arxiv.org/abs/2305.17849) | Learning from bags of instances where only bag labels are available                                  | ✅[Explanation](./Learning%20Techniques/Multiple%20Instance%20Learning/README.md) 
| **[Multimodal Learning](Multimodal%20Learning/)**   | Multimodal Transformers  | [Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488)                    | Models that jointly learn from multiple modalities (e.g. image, text, audio)                         | ⌛[Explanation](./Multimodal%20Learning/README.md)           |
| **Contrastive Learning**  | SimCLR                   | [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) | Contrastive SSL framework using augmented views and a simple architecture                             | ✅[Explanation & implementation](./Contrastive%20Learning/SimCLR/README.md)               |
|                           | MoCo                     | [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) | Momentum contrast for building dynamic dictionaries                                                   | ✅[Explanation & implementation](./Contrastive%20Learning/MoCO/README.md)                    |
|                           | CLIP                     | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | Contrastive learning with image-text pairs for multi-modal learning                                  | ⌛[Explanation](./Contrastive%20Learning/CLIP/README.md)                    ||
| **[Masked Modeling](Masked%20Modeling/)**       | MAE                      | [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | Masked Autoencoder predicting masked image patches                                                     | ⌛[Implementation](./Masked%20Modeling/MAE/mae.py)                      |
|                           | MIM                      | [Masked Image Modeling: A Survey](https://arxiv.org/abs/2408.06687) | Predict masked parts of images during training                                                        | ⌛[Implementation](./Masked%20Modeling/MIM/mim.py)                    |
|                                                 | BEiT                     | [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)     | Adapts BERT-style masked language modeling to images using discrete visual tokens                     | ⌛[Implementation](../../Masked%20Modeling/BEiT/beit.py)                    |
| **[Self-Distillation](Self-Distillation/)**     | DINO                     | [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)        | Self-distillation without labels, training student to match teacher                                  | [dino](./implementations/dino)                    |
|                           | DinoV2                   | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)       | Improved DINO with stronger recipes                                                                   | [dinov2](./implementations/dinov2)                |
| **[Fine-Tuning Strategies](Fine-Tuning%20Strategies/)**|  LoRA                 | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)                 | Efficient fine-tuning via injecting trainable low-rank matrices into frozen weights                  | [lora](./explanations/lora)                       |
| **[Other Techniques](Other%20Techniques/)**   | Aggregation              | [An Aggregation of Aggregation Methods in Computational Pathology](https://arxiv.org/abs/2211.01256)  | Techniques to aggregate patch or token embeddings to form global representations                     | ✅[Explanation & implementation](Other%20Techniques/Aggregation/)        |
|                           | Linear Probing       | [Do Better ImageNet Models Transfer Better?](https://arxiv.org/abs/1912.11370)                         | Training only the classification head on top of a frozen encoder to evaluate representation quality  | [linear_probing](./explanations/linear_probing)   |
|                           | Mixture of Experts       | [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | Activating only a few expert sub-networks per input for scalability and specialization               | [moe](./explanations/moe)                         |


  
---
## 🧠 Foundation Models in Computational Pathology

A curated list of foundational and large-scale models for computational pathology. 
These tables categorize each model by its learning paradigm and highlights core innovations and downstream capabilities.


### Self-Supervised Learning (SSL)

| Model     | Paper / Link                                         | Key Innovation                                                      | Slides    | Report Gen. | Vision+Lang | Captioning |
| --------- | ---------------------------------------------------- | ------------------------------------------------------------------- | --------- | ----------- | ----------- | ---------- |
| Virchow   | [arXiv:2309.07778](https://arxiv.org/pdf/2309.07778) | Global & local crops + morphology-preserving ECT augmentations      | 1,488,550 | ✅           | ✅           | ❌          |
| UNI       | —                                                    | MIM + self-distillation with Sinkhorn & KoLeo regularization        | 100,000   | ✅           | ✅           | ✅          |
| Phikon    | [arxiv:2311.11023](https://arxiv.org/abs/2311.11023) | iBOT-based masked self-distillation; robust to visual perturbations | 6,093     | ✅           | ✅           | ✅          |
| Univ2     | [arXiv:2404.xxxxx](https://arxiv.org/abs/2404.xxxxx) | [Add key innovation here]                                           | [Slides]  | [Report]    | [Vision+Lang] | [Captioning] |
| ProvGigapath | [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx) | [Add key innovation here]                                           | [Slides]  | [Report]    | [Vision+Lang] | [Captioning] |


### Masked Image Modeling (MIM)

| Model     | Paper / Link                                         | Key Innovation                                                                | Slides  | Report Gen. | Vision+Lang | Captioning |
| --------- | ---------------------------------------------------- | ----------------------------------------------------------------------------- | ------- | ----------- | ----------- | ---------- |
| Phikon-v2 | [arXiv:2311.11023](https://arxiv.org/abs/2311.11023) | ViT-L scaled; trained on 460M tiles; robust ensemble for biomarker prediction | 58,359  | ✅           | ✅           | ❌          |
| PLUTO     | [arXiv:2403.00827](https://arxiv.org/abs/2403.00827) | Multi-scale MIM with Fourier loss for out-of-distribution robustness          | 158,852 | ✅           | ✅           | ✅          |


### Hybrid / Expert-Inspired Learning

| Model       | Paper / Link                                         | Key Innovation                                                                | Slides  | Report Gen. | Vision+Lang | Captioning |
| ----------- | ---------------------------------------------------- | ----------------------------------------------------------------------------- | ------- | ----------- | ----------- | ---------- |
| RudolfV     | [arXiv:2403.01821](https://arxiv.org/abs/2403.01821) | Trained with stain-specific augmentations and pathologist guidance            | 133,998 | ✅           | ✅           | ✅          |
| H-Optimus-0 | —                                                    | ViT-G/14 with 40 transformer blocks for efficient high-dimensional processing | 500,000 | ✅           | ✅           | ❌          |
| H-Optimus-1 | —                                                    | Variant of H-Optimus-0                                                        | 500,000 | ✅           | ✅           | ❌          |


### Multimodal / Multitask Architectures (Hybrid)

| Model    | Paper / Link                                         | Key Innovation                                                          | Slides    | Report Gen. | Vision+Lang | Captioning |
| -------- | ---------------------------------------------------- | ----------------------------------------------------------------------- | --------- | ----------- | ----------- | ---------- |
| Virchow2 | [arXiv:2403.10870](https://arxiv.org/abs/2403.10870) | Trained on 3.1M WSIs; diverse data and pathology-inspired augmentations | 3,134,922 | ✅           | ✅           | ❌          |
---

More categories and models will be added progressively as we parse the landscape of generative pathology models and language-vision integrations.
If you use or extend this repo, please cite the source papers and link back to this project. 🙏
