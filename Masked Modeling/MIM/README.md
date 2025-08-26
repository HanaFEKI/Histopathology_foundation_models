# 🧬 Masked Image Modeling (MIM)

**Masked Image Modeling (MIM)** is a self-supervised learning technique for images inspired by **Masked Language Modeling (MLM)** in NLP (e.g., BERT).  
The core idea is simple:

1. **Mask patches** of an input image (hide part of the data).  
2. **Train a model** to reconstruct the masked patches from the visible context.  

Unlike standard supervised learning, **MIM does not require labels**, which is particularly useful for domains like histopathology where annotations are costly and scarce.

---

## 🔹 How MIM Works

1. **Patch Tokenization**  
   - Divide an image into **non-overlapping patches** (e.g., 16×16 pixels).  
   - Randomly mask a subset of patches (e.g., 40–75%).  

2. **Encoder**  
   - Feed the visible patches into a **Vision Transformer (ViT)** or CNN-based encoder.  
   - The encoder learns **contextual representations** of images.  

3. **Decoder / Reconstruction Head**  
   - A lightweight decoder predicts the **pixel values or latent features** of the masked patches.  
   - Loss: typically **mean squared error (MSE)** or **cross-entropy** for discrete tokens.  

4. **Self-Supervised Pretraining**  
   - The model learns **high-level semantic and structural features** without labels.  
   - After pretraining, the encoder can be **fine-tuned** on downstream tasks (classification, segmentation, detection).

---

## 🔹 Why MIM is Important for Histopathology

Histopathology images (WSIs, biopsy patches) present unique challenges:

- **Gigapixel size** → cannot process the full image at once.  
- **Scarce annotations** → labeling requires expert pathologists.  
- **Complex textures and structures** → cellular and tissue-level patterns vary widely.

**MIM addresses these issues:**

1. **Label-free pretraining**  
   - Learn meaningful representations from **unlabeled tissue images**, which are abundant.  

2. **Improved feature extraction**  
   - Captures **cellular patterns, gland structures, and tumor morphology** that are crucial for diagnosis.  

3. **Data efficiency**  
   - Fine-tuning on small labeled datasets is possible because the encoder already learned rich representations.  

4. **Transferability**  
   - Pretrained MIM encoders can generalize across **different organs, stains, and cancer types**.


## 🔹 Common Approaches

1. **MAE (Masked Autoencoders)**  
   - Mask a high percentage of patches (e.g., 75%).  
   - Use a lightweight decoder to reconstruct the missing patches.  

2. **SimMIM**  
   - Similar to MAE but reconstructs pixels using the **same encoder output** without a separate decoder.  

3. **BEiT (Bidirectional Encoder representation for Image Transformers)**  
   - Converts image patches into discrete visual tokens and predicts masked tokens (like BERT).

---

## 🔹 Applications in Histopathology

- **Cancer subtype classification** : Pretrain on unlabeled slides → fine-tune on limited labeled patches.  

- **Tissue segmentation** : Masked reconstruction helps capture **cell boundaries and gland structures**.  

- **Survival prediction / outcome modeling** : MIM representations capture **subtle histological patterns** correlated with prognosis.  

- **Cross-modality learning** : Combine histopathology with **radiology or genomic data** for multimodal prediction.
