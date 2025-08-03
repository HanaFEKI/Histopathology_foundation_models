# 🧠 Semi-Supervised Learning (SSL)

## Overview

Semi-supervised learning (SSL) lies between supervised and unsupervised learning. It leverages a **small amount of labeled data** and a **large amount of unlabeled data** to improve model performance. SSL is especially valuable in scenarios where labeling is expensive or time-consuming, such as in medical imaging, NLP, or autonomous driving.

---

## 🔍 Motivation

Labeling data is expensive and labor-intensive. In contrast, unlabeled data is abundant and cheap. SSL techniques aim to **make better use of the unlabeled data** by propagating label information or enforcing consistency between predictions under perturbations.

---

## ✅ Common Use Cases

- Medical diagnosis with limited expert annotations
- Text classification from few labeled documents
- Object recognition with sparse human labels
- Speech and audio classification with minimal transcriptions

---

## 📌 Popular SSL Techniques

### 🧪 1. **Pseudo-Labeling**
- Use the model to label the unlabeled data, then train on those pseudo-labels.
- SOTA: [FixMatch (2020)](https://arxiv.org/abs/2001.07685)

### 🧊 2. **Consistency Regularization**
- Encourage model to produce consistent predictions when inputs are perturbed.
- SOTA: [Mean Teacher](https://arxiv.org/abs/1703.01780), [UDA (Unsupervised Data Augmentation)](https://arxiv.org/abs/1904.12848)

### 📊 3. **Graph-Based Methods**
- Represent data as a graph and propagate labels through connected nodes.
- SOTA: [Label Propagation](https://papers.nips.cc/paper_files/paper/2002/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf)

### 🌀 4. **Generative Models**
- Use generative models to model the joint distribution of inputs and labels.
- SOTA: [Semi-Supervised GANs (SGAN)](https://arxiv.org/abs/1606.01583)

---

## 📁 Repository Structure

```
ssl/
├── data/
├── models/
├── scripts/
├── notebooks/
└── README.md
```

---

## 🧪 References

- [FixMatch](https://arxiv.org/abs/2001.07685)
- [MixMatch](https://arxiv.org/abs/1905.02249)
- [Mean Teacher](https://arxiv.org/abs/1703.01780)

---

