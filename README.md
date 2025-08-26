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

| Category                  | Technique                | Paper                                                         | Description                                                                                           | Explanation & Implementation                   |
|---------------------------|--------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| **[Architectures](Architectures/)**         | Transformer              | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)       | Transformer architecture                                                                               | ✅[Explanation & implementation](./Architectures/Transformer/)      |
|                           | ViT                      | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)   | Vision Transformer architecture for image classification                                              |✅[Explanation & implementation](./Architectures/Vision%20Transformer%20(ViT)/)                      |
|                           | Swin Transformer | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | A hierarchical vision transformer that computes self-attention within local shifted windows, enabling scalable and efficient modeling of high-resolution images. | ✅[Explanation](./Architectures/Swin%20Transformer/README.md) |
| **[Learning Techniques](Learning%20Techniques/)** | Self-Supervised Learning     | [A Survey on Self-supervised Learning: Algorithms, Applications, and Future Trends](https://arxiv.org/abs/2301.05712) | Learning useful representations from unlabeled data without manual annotations                      | ✅ [Explanation](/Learning%20Techniques/Self-Supervised%20Learning/README.md)                         |
|                           | Semi-Supervised Learning | [A Survey on Deep Semi-supervised Learning](https://arxiv.org/abs/2103.00550)                        | Learning from small labeled and large unlabeled datasets                                              | ✅ [Explanation](./Learning%20Techniques/Semi-Supervised%20Learning/README.md)   |
|                           | Weakly Supervised Learning | [A Critical Look at Weakly Supervised Learning](https://arxiv.org/abs/2305.17442)                  | Learning from weak/noisy labels instead of fully annotated data                                      | ✅ [Explanation](./Learning%20Techniques/Weakly%20Supervised%20Learning/README.md) |
|                           | Unsupervised Learning    | [Semi-Supervised and Unsupervised Deep Visual Learning: A Survey](https://arxiv.org/abs/2208.11296) | Learning patterns from unlabeled data                                                                 | ✅ [Explanation](./Learning%20Techniques/Unsupervised%20Learning/README.md)      |
|                           | Multiple Instance Learning (MIL) | [MIL: A Review of Recent Advances and Applications in Deep Learning](https://arxiv.org/abs/2305.17849) | Learning from bags of instances where only bag labels are available                                  | ✅[Explanation](./Learning%20Techniques/Multiple%20Instance%20Learning/README.md) 
| **[Multimodal Learning](Multimodal%20Learning/)**   | Multimodal Transformers  | [Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488)                    | Models that jointly learn from multiple modalities (e.g. image, text, audio)                         | ✅[Explanation](./Multi-Modal%20Learning/README.md)           |
| **[Contrastive Learning](Contrastive%20Learning/)**  | SimCLR                   | [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) | Contrastive SSL framework using augmented views and a simple architecture                             | ✅[Explanation & implementation](./Contrastive%20Learning/SimCLR/)               |
|                           | MoCo                     | [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) | Momentum contrast for building dynamic dictionaries                                                   | ✅[Explanation & implementation](./Contrastive%20Learning/MoCO/)                    |
|                           | CLIP                     | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | Contrastive learning with image-text pairs for multi-modal learning                                  | ✅[Explanation & implementation](./Contrastive%20Learning/CLIP/)                    ||
| **[Masked Modeling](Masked%20Modeling/)**       | MIM                      | [Masked Image Modeling: A Survey](https://arxiv.org/abs/2408.06687) | Predict masked parts of images during training                                                        | ✅[Explanation](./Masked%20Modeling/MIM/README.md)  |
|             | MAE                      | [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | Masked Autoencoder predicting masked image patches                                                     | ✅[Explanation & implementation](./Masked%20Modeling/MAE/)               |               
|                                                 | BEiT                     | [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)     | Adapts BERT-style masked language modeling to images using discrete visual tokens                     | ✅[Explanation & implementation](../../Masked%20Modeling/BEiT/)                    |
| **[Self-Distillation](Self-Distillation/)**     | DINO                     | [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)        | Self-distillation without labels, training student to match teacher                                  | ✅[Explanation](./Self-Distillation/Dino/README.md)                    |
|                           | DinoV2                   | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)       | Improved DINO with stronger recipes                                                                   | ✅[Explanation](./Self-Distillation/DinoV2/README.md)               |
| **[Fine-Tuning Strategies](Fine-Tuning%20Strategies/)**|  LoRA                 | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)                 | Efficient fine-tuning via injecting trainable low-rank matrices into frozen weights                  | ✅[Explanation & implementation](./Fine-Tuning%20Strategies/LoRA/README.md)                       |
|                           | Linear Probing       | [Do Better ImageNet Models Transfer Better?](https://arxiv.org/abs/1912.11370)                         | Training only the classification head on top of a frozen encoder to evaluate representation quality  | ✅[Explanation & implementation](./Fine-Tuning%20Strategies/Linear%20Probing/)   |
| **[Other Techniques](Other%20Techniques/)**   | Aggregation              | [An Aggregation of Aggregation Methods in Computational Pathology](https://arxiv.org/abs/2211.01256)  | Techniques to aggregate patch or token embeddings to form global representations                     | ✅[Explanation & implementation](Other%20Techniques/Aggregation/)        |
|                           | Mixture of Experts       | [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | Activating only a few expert sub-networks per input for scalability and specialization               | [moe](./explanations/moe)                         |


  
---
## Foundation Models in Computational Pathology

A curated list of foundational and large-scale models for computational pathology. 
These tables categorize each model by its learning paradigm and highlights core innovations and downstream capabilities.

### 1. Self-Supervised DINOv2-based Models (Image-only)

| Model           | Backbone       | Params  | WSIs      | Tiles| Link | Key Innovation | Common Use Case |
|-----------------|----------------|---------|----------|----------------|-------------|----------------|----------------|
| [Virchow ViT-H](https://arxiv.org/abs/2309.07778) | ViT-H | 632M | 1.5M | 2B | [Hugging Face](https://huggingface.co/paige-ai/Virchow) | Student-teacher paradigm with global and local cropping; extended-context translation (ECT) augmentation preserves cellular morphology | WSI-level representation, multi-task downstream tasks |
| [Virchow2](https://arxiv.org/abs/2408.00738) | ViT-H | 632M | 3.1M | 1.9B |  [Hugging Face](https://huggingface.co/paige-ai/Virchow2) | Scaled dataset size and increased diversity, trained on 3.1M WSIs, uses domain-inspired augmentation | Tile-level representation for large pathology datasets |
| [Virchow2G](https://arxiv.org/html/2408.00738v1) | ViT-G | 1.9B | 3.1M | 1.9B |  [Hugging Face](https://huggingface.co/paige-ai/Virchow2) | Scaled both data and model size, mixed magnification training; enhanced generalization across datasets | Multi-task WSI analysis, high generality and multipurpose use |
| [Phikon-v2](https://arxiv.org/abs/2409.09173) | ViT-L | 307M | 58K | 456M | [Hugging Face](https://huggingface.co/owkin/phikon-v2) | Scaled ViT-L architecture; trained on 460M pathology tiles; robust ensembling for biomarker prediction | Genomic biomarker prediction, WSI-level classification |
| [RudolfV](https://arxiv.org/abs/2401.04079) | ViT-L | 307M | 103K | 750M | - | Trained on 134k slides across 58 tissue types and 129 staining methods; integrates pathologist expertise, uses stain-specific augmentations | Tumor microenvironment profiling, biomarker evaluation, slide-level tasks |
| [TissueConcepts](https://arxiv.org/abs/2409.03519) | Swin Transformer | - | 7K | 1.7B | - | Joint encoder utilizes transformer and convolution architectures trained with multi-task learning for classification, segmentation, and detection tasks | Multi-task classification, detection, segmentation |
| [Kaiko-ai](https://arxiv.org/abs/2404.15217) | ViT-L | 303M | 29K | 50M | [Hugging Face](https://huggingface.co/kaiko-ai/midnight) | Modified DINO recipes; trained on multi-magnification TCGA WSIs; reduced GPU and batch size requirements because of Dynamic Patch Extraction | Cellular and tissue-level representation tasks |
| [UNI](https://arxiv.org/abs/2308.15474) | ViT-L | 307M | 100K | 100M | [Hugging Face](https://huggingface.co/MahmoodLab/UNI) | Combines self-distillation and masked image (resolution-agnostic) modeling; incorporates Sinkhorn-Knopp centering and KoLeo regularization for robustness. | General-purpose WSI embedding, multi-task pathology evaluation |
| [Hibou-L](https://arxiv.org/abs/2406.05074) | ViT-L | 307M | 1.1M | 512M | [Hugging Face](https://huggingface.co/histai/hibou-L) | Trained on over 1 million WSIs with RandStainNA augmentation for WSI-specific optimization. | Cell-level segmentation, classification, patch-level tasks |



### 2. Supervised and Weakly-Supervised Models

| Model       | Backbone  | Params | WSIs  | Tiles | Learning Method | Link | Key Innovation | Common Use Case | 
|------------|-----------|--------|-------|-------|----------------|----------------|----------------|----------------------|
| [OmniScreen](https://arxiv.org/abs/2408.00738) | Virchow2 | 632M | 48K | - | Weakly-Supervised (on Virchow2 embeddings) | Leverages pretrained Virchow2 embeddings, scalable weak supervision | Predicting genomic biomarkers from WSIs, multi-cancer analysis | [Hugging Face](https://huggingface.co/owkin/omniscreen) |
| [PathoDuet](https://arxiv.org/abs/XXXX.XXXX) | ViT-B | 86M | 11K | 13M | Multi-headed attention-based MIL | Uses multi-instance learning with attention to aggregate tile-level information | Slide-level classification, WSI-level phenotype prediction | - |
| [REMEDIS](https://arxiv.org/abs/XXXX.XXXX) | ResNet-152 | 232M | 29K | - | SimCLR (contrastive learning) | Self-supervised contrastive learning for slide and tile embeddings | Multi-domain histopathology tasks, cross-stain generalization | - |



### 3. Masked Image Modeling and Other SSL Methods

| Model           | Backbone       | Params  | WSIs      | Embedding Size | SSL Method |
|-----------------|----------------|---------|----------|----------------|------------|
| BepH            | BEiTv2         | 86M     | 11K      | 11M            | BEiTv2 SSL |
| iBOT            | -              | 43M     | -        | -              | Masked Image Modeling |
| MoCoV3          | -              | 15M     | -        | -              | SRCL       |
| PLUTO           | FlexiVit-S     | 22M     | 158K     | 195M           | DINOv2 + MAE + Fourier-loss |


### 4. Other Notable Models

| Model           | Backbone       | Params  | WSIs      | Embedding Size | Learning Method |
|-----------------|----------------|---------|----------|----------------|----------------|
| BROW            | ViT-B          | 86M     | 11K      | 180M           | -              |
| Phikon          | ViT-B          | 86M     | 10M      | 104M           | DINO SSL       |
| HIPT            | ViT-HIPT       | 10M     | 11K      | 104M           | DINO SSL       |
| Madeleine       | -              | -       | -        | -              | -              |
| CONCH           | 86M            | -       | 23K      | 48M            | -              |
| COBRA           | Mamba-2        | 15M     | 3,048    | -              | Self-supervised contrastive learning |

More categories and models will be added progressively as we parse the landscape of generative pathology models and language-vision integrations.
If you use or extend this repo, please cite the source papers and link back to this project. 🙏
