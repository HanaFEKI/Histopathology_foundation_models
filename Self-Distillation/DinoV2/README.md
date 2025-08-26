# 🔹 DINOv2: Improved Self-Distillation with No Labels

DINOv2 is an evolution of the original DINO framework for **self-supervised learning** of vision models.  
It improves representation quality by addressing stability and scalability issues and supports **larger backbones and datasets**.


## 1. Key Improvements over DINO

| Feature                 | DINO                     | DINOv2                                      |
|-------------------------|--------------------------|---------------------------------------------|
| Backbone                | ViT or CNN               | Larger ViT or hybrid architectures         |
| Training Stability       | Sensitive to augmentations| Improved training with careful normalization and loss scaling |
| Output Embeddings        | Projection head only     | Adds **residual MLP heads** for better representations |
| Multi-crop strategy      | Fixed views              | More diverse cropping and augmentation strategies |
| Pretraining Datasets     | Moderate-scale ImageNet  | Large, diverse datasets for better generalization |


## 2. Pipeline Overview

1. **Input Images**
   - Multiple augmented views (`x_student`, `x_teacher`)
2. **Backbone Network**
   - Larger, possibly hierarchical ViT or hybrid CNN-ViT
3. **Projection Head**
   - Residual MLP with normalization
   ```math
   z = normalize(MLP(f(x)))

4. **Teacher-Student Framework**
     - Student learns from teacher outputs
     - Teacher updated via EMA
    ```math
    theta_{teacher} = m * theta_{teacher} + (1-m) * theta_{student}
    ```
