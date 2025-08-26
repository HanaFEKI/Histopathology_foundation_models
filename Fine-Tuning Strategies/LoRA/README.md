# 🔹 LoRA: Low-Rank Adaptation

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method for large pre-trained models.  
Instead of updating all model weights, LoRA injects **trainable low-rank matrices** into existing layers.

---

## 1. Pipeline Overview

1. **Start with Pre-trained Model**
   - e.g., Transformer, ViT, BERT, or GPT model
2. **Inject LoRA Layers**
   - For a linear layer W, introduce low-rank matrices A and B
   ```math
   W_{adapted} = W + alpha * (A @ B)

3. **Freeze Original Weights**
Only A and B are updated during fine-tuning

4. **Forward Pass**
Input x passes through W_adapted instead of W

5. **Backpropagation**
Gradients only flow through A and B, reducing memory and compute
