
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

