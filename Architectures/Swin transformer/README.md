# 🔹 Swin Transformer

The **Swin Transformer** is a hierarchical Vision Transformer designed for computer vision tasks.  
Unlike standard Vision Transformers (ViT), it introduces **shifted windows** to reduce computation and enable local-global feature representation.

---

## 1. Key Concepts

1. **Hierarchical Representation**
   - Images are represented at multiple scales
   - Patch merging layers reduce spatial resolution while increasing feature dimension

2. **Window-based Self-Attention**
   - Self-attention is computed **within local windows**
   - Reduces complexity from O(N^2) to O(M^2), where M is window size

3. **Shifted Windows**
   - Alternating layers **shift the window partitioning**
   - Allows **cross-window connections** and captures global context

---

## 2. Pipeline Overview

1. **Input Image**
   - e.g., X ∈ ℝ^(H × W × C)

2. **Patch Partition**
   - Image is split into non-overlapping patches (like ViT)
   - Each patch is projected to an embedding
```math
x_p = Flatten(X_{patch}) W_e + b_e
```

3. **Hierarchical Feature Extraction**
   - Stacked **Swin Transformer blocks** process features at multiple scales
   - Each block: Window-based Multi-Head Self-Attention + MLP + LayerNorm + Residual

4. **Shifted Windows**
   - Alternate layers shift window positions
   - Enables cross-window information flow

5. **Patch Merging**
   - Reduce spatial resolution and increase channel dimension
   - Forms hierarchical feature maps

6. **Classification / Output Head**
   - Global average pooling of final features
   - Linear layer for classification or task-specific head

---

## 3. Mathematical Intuition

- **Window-based attention**:
``` math
Attention(Q, K, V) = softmax(Q K^T / sqrt(D)) V
```
- **Shifted window**:
  - Shift windows by half the window size in alternating layers
  - Ensures overlapping context between windows

- **Patch merging**:
  - Concatenate neighboring patches along channel dimension
  - Reduce spatial dimensions: H × W → H/2 × W/2

---

## 4. Key Advantages

- **Linear computational complexity** with respect to image size  
- **Hierarchical feature maps** suitable for dense prediction tasks  
- **Flexible window-based attention** balances local and global modeling  
- Works well for **classification, detection, and segmentation**  

---

## 5. Use Cases

- **Image classification**: Large-scale datasets like ImageNet  
- **Object detection**: Use in detectors like Mask R-CNN  
- **Semantic segmentation**: High-resolution medical images or histopathology slides  
- **Self-supervised pretraining**: Can be combined with masked image modeling (MIM)

