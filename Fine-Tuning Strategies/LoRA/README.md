# 🔹 LoRA: Low-Rank Adaptation

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method for large pre-trained models.  
Instead of updating all model weights, LoRA injects **trainable low-rank matrices** into existing layers.

---

## 1. Pipeline Overview

1. **Start with Pre-trained Model** : e.g., Transformer, ViT, BERT, or GPT model
2. **Inject LoRA Layers** : 
For a linear layer W, introduce low-rank matrices A and B
   ```math
   W_{adapted} = W + alpha * (A @ B)

3. **Freeze Original Weights** : 
Only A and B are updated during fine-tuning

4. **Forward Pass** : 
Input x passes through W_adapted instead of W

5. **Backpropagation** : 
Gradients only flow through A and B, reducing memory and compute


## 2. Mathematical Formulation

- **Original linear layer:**
``` math
y = x W^T
```

- **With LoRA adaptation:**
```math
W_{adapted} = W + alpha * (A @ B)
```
```math
y = x W_{adapted}^T = x (W + alpha * A @ B)^T
```
- A ∈ R^(out_features × r), B ∈ R^(r × in_features) with r ≪ min(out_features, in_features)

---

## 3. Key Advantages

- **Parameter efficiency**: Only low-rank matrices are trainable  
- **Memory efficiency**: Reduces GPU memory usage  
- **Compatibility**: Works with pre-trained models without modifying their weights  
- **Fast fine-tuning**: Suitable for large language models or vision transformers

---

## 4. Use Cases

- Fine-tuning **large language models** (GPT, BERT) on small datasets  
- Adapting **Vision Transformers** or BEiT for domain-specific tasks (e.g., histopathology)  
- Multi-task learning with minimal parameter updates

