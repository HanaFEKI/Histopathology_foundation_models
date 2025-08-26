# 🔹 DINOv2: Improved Self-Distillation with No Labels

DINOv2 is an evolution of the original DINO framework for **self-supervised learning** of vision models.  
It improves representation quality by addressing stability and scalability issues and supports **larger backbones and datasets**.


## Key Improvements over DINO

| Feature                 | DINO                     | DINOv2                                      |
|-------------------------|--------------------------|---------------------------------------------|
| Backbone                | ViT or CNN               | Larger ViT or hybrid architectures         |
| Training Stability       | Sensitive to augmentations| Improved training with careful normalization and loss scaling |
| Output Embeddings        | Projection head only     | Adds **residual MLP heads** for better representations |
| Multi-crop strategy      | Fixed views              | More diverse cropping and augmentation strategies |
| Pretraining Datasets     | Moderate-scale ImageNet  | Large, diverse datasets for better generalization |


## Key Advantages
- Higher-quality representations than DINO
- Stable training with larger backbones
- Works well for few-shot and transfer learning tasks
- Supports dense tasks like segmentation and detection
