# 🧠 Self-Supervised Learning (SSL)

Self-Supervised Learning (SSL) is a paradigm where models learn meaningful representations **without requiring manual labels**. It creates pretext tasks using the data itself, enabling generalizable encoders useful for downstream tasks like classification, detection, or segmentation.

---

## 📚 Categories of SSL

This README provides an **introductory overview** of Self-Supervised Learning.

🔍 The detailed techniques are explored in the respective folders of this repository:

- `Contrastive Learning/` – Contrastive Learning methods like SimCLR, MoCo, CLIP
- `Masked Modeling/` – Masked modeling methods like MAE, BEiT
- `Self-Distillation/` – Self-Distillation methods like DINO, BYOL


### 1. 🔄 Contrastive Learning

Learns by pulling together positive pairs (e.g., augmented views of the same image) and pushing apart negative pairs.

#### 🔑 Key Methods:

| Method | Paper | Core Idea |
|--------|-------|-----------|
| **SimCLR** | [SimCLR](https://arxiv.org/abs/2002.05709) | Contrastive loss + augmentations |
| **MoCo** | [Momentum Contrast](https://arxiv.org/abs/1911.05722) | Momentum encoder + dynamic queue |
| **CLIP** | [CLIP](https://arxiv.org/abs/2103.00020) | Contrastive loss between image-text pairs |

➡️ See: [`Contrastive Learning/`](Contrastive%20Learning/)


### 2. 🎭 Masked Modeling

Predict missing or masked parts of the input.

#### 🔑 Key Methods:

| Method | Paper | Core Idea |
| --- | --- | --- |
| **MAE** | [Masked Autoencoders](https://arxiv.org/abs/2111.06377) | Mask image patches and reconstruct |
| **BEiT** | [BEiT](https://arxiv.org/abs/2106.08254) | Predict visual tokens using transformer |
| **MIM** | [Masked Image Modeling: A Survey](https://arxiv.org/abs/2408.06687) | Generic framework for masked modeling |

➡️ See: [`Masked Modeling/`](Masked%20Modeling/)

---

### 3. 🧑‍🏫 Self-Distillation

Learns by training a model (student) to imitate a slowly updated teacher.

#### 🔑 Key Methods:

| Method | Paper | Core Idea |
| --- | --- | --- |
| **DINO** | [DINO](https://arxiv.org/abs/2104.14294) | Teacher-student architecture without labels |
| **DINOv2** | [DINOv2](https://arxiv.org/abs/2304.07193) | Stronger recipe and large-scale training |
| **BYOL** | [BYOL](https://arxiv.org/abs/2006.07733) | Learns without negative samples |

➡️ See: [`Self-Distillation/`](Self-Distillation/)

## 🧩 Other SSL Techniques

| Task | Description |
|------|-------------|
| Rotation Prediction | Predict 0°, 90°, 180°, 270° rotation |
| Jigsaw Puzzle | Predict correct permutation of shuffled patches |
| Colorization | Predict RGB values from grayscale input |
| Inpainting | Fill in missing regions of an image |

---

## 🏗️ Typical SSL Pipeline

### 🔧 Pretraining Phase

- **Input**: Unlabeled data
- **Model**: ResNet / ViT / CNN Encoder
- **SSL Task**: Contrastive, Masked, Self-distillation
- **Output**: Pretrained encoder (feature extractor)

### 🎯 Downstream Fine-Tuning

- Add task-specific head (e.g. MLP)
- Train on small labeled dataset
- Fine-tune or freeze encoder

---

## 🔬 SSL in Pathology

Labels in digital pathology are:

- 💸 Expensive (require expert annotations)
- ⚠️ Noisy / subjective
- 🧩 Sparse (for rare classes)

SSL enables building **general-purpose encoders for WSI (Whole Slide Images)** using unlabeled data.

### 🧪 Example Models:

| Model | SSL Method | Use Case |
|--------|------------|----------|
| **iBOT** | Self-Distillation | Patch representation learning |
| **CLAM-SSL** | Patch contrastive pretraining | Slide-level classification |
| **UNI / PLUTO / Virchow** | Masked modeling + distillation | Pretraining on WSI datasets |
| **DINOv2-WSI** | Vision transformer pretraining | Computational pathology foundation model |


