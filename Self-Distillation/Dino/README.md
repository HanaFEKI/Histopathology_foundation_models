# 🔹 DINO: Self-Distillation with No Labels

DINO is a **self-supervised learning framework** for vision models.  
It trains a **student network** to match a **teacher network**’s outputs, without labels.

---

## 1. Pipeline Overview

1. **Input Image**
   - Apply multiple augmentations (views) to generate `x_student` and `x_teacher`

2. **Backbone Network**
   - Student and teacher share the same backbone (e.g., ViT, ResNet)

3. **Projection Head**
   - MLP projects features to a normalized embedding space:
   ```math
   z = MLP(f(x))
   ```
   ``` math
   z = normalize(z)

4. **Teacher-Student Training**
  - Student output: `z_student`
  - Teacher output: `z_teacher` (EMA (Exponential Moving Average) of student)
  - Loss: Cross-entropy between softmax distributions of teacher and student

5. **EMA Update**
Teacher parameters updated using:
``` math
theta_{teacher} = m * theta_{teacher} + (1-m) * theta_{student}
```
where `m = momentum coefficient (e.g., 0.996)`

## 2. Mathematical Formulation

- **Student embedding**: `z_s = f_s(x_s)`  
- **Teacher embedding**: `z_t = f_t(x_t)`  
- **Softmax probabilities**: `p_s = softmax(z_s / T)`
`p_t = softmax(z_t / T)`


- **DINO loss**: Cross-entropy between `p_t` and `p_s`  
- **Teacher update**:
``` math
theta_t = m * theta_t + (1 - m) * theta_s
```


---

## 3. Key Advantages

- **Self-supervised**: No labeled data required  
- **Works with vision transformers and CNNs**  
- **Teacher-student distillation** improves stability and representation quality  
- **Multi-view learning** allows learning invariance to augmentations

## 4. Use Cases

- Pretraining **Vision Transformers** for classification, segmentation, or detection  
- Learning **domain-specific features** in histopathology or medical imaging  
- Can be combined with **masked image modeling** or other SSL methods





