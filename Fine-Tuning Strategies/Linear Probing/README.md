# 🔹 Linear Probing

Linear probing is a **parameter-efficient evaluation method** for pre-trained representations.  
It trains only a **linear classifier** on top of a frozen backbone to measure the quality of learned features.


## 1. Pipeline Overview

1. **Pre-trained Backbone**
   - e.g., ViT, BEiT, ResNet
   - Backbone weights are **frozen**
2. **Feature Extraction**
   - Compute embeddings for input images:
   ```math
   features = backbone(x)
  
3. **Linear Classifier**
- Train a single linear layer on top of features:
```math 
logits = Linear(features)
```


4. **Training**
- Only update the linear classifier weights
- Backbone remains unchanged


5. **Evaluation**
- Accuracy or downstream metrics show representation quality


## 2. Mathematical Formulation

- Backbone output: `f(x) ∈ R^D`  
- Linear classifier: `y = W f(x) + b`  
- Only `W, b` are updated; backbone parameters are frozen.

---

## 3. Key Advantages

- **Parameter-efficient**: Only train a single layer  
- **Fast training**: Much fewer parameters than full fine-tuning  
- **Benchmarking representations**: Measures the quality of embeddings learned in pretraining

---

## 4. Use Cases

- Evaluate features from **self-supervised models** (SimCLR, MoCo, DINO, BEiT)  
- Quick assessment of **pretrained vision or language models**  
- Downstream tasks in **histopathology, medical imaging, or NLP**

