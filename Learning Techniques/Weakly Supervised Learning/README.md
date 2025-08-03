# 🧠 Weakly Supervised Learning (WSL)

## Overview

Weakly supervised learning is a family of techniques that learn from **incomplete**, **inaccurate**, or **inexact** supervision. This allows training powerful models without requiring expensive, high-quality labels.

---

## 🔍 Motivation

Labeling at fine-grained levels (e.g., pixel-wise annotations, bounding boxes, or sentence-level sentiment) can be prohibitively expensive. WSL allows us to train models using **coarser, noisier, or partially available labels**.

---

## ✅ Common Use Cases

- Learning from noisy web labels or metadata
- Object detection using image-level tags (no bounding boxes)
- Disease detection from clinical notes without labeled images
- Sentiment analysis from review ratings

---

## 📌 Types of Weak Supervision

| Type              | Description                                       | Example                                |
|-------------------|---------------------------------------------------|----------------------------------------|
| Incomplete Labels | Only a subset of data is labeled                  | Some images have no class labels       |
| Inaccurate Labels | Labels contain noise or are incorrect             | Crowd-sourced annotations              |
| Inexact Labels    | Coarse-grained labels for fine-grained tasks      | Image-level tags for object detection  |

---

## 🚀 SOTA Techniques and Frameworks

### 🪶 1. **Learning with Noisy Labels**
- Correct or mitigate noisy labels via modeling or confidence.
- SOTA: [Co-Teaching](https://arxiv.org/abs/1804.06872), [MentorNet](https://arxiv.org/abs/1805.04770)

### 🧭 2. **Multiple Instance Learning (MIL)**
- Learning from bags of instances labeled at the bag level.
- SOTA: Used in histopathology slide classification

### 🏷 3. **Label Propagation & Bootstrapping**
- Combine noisy labels with confident predictions for training.
- SOTA: [Self-Learning with Noisy Labels](https://arxiv.org/abs/2006.07836)

### 📦 4. **Programmatic Weak Labeling**
- Use heuristic rules, patterns, or knowledge bases to assign noisy labels.
- SOTA: [Snorkel](https://snorkel.org/)

---

## 📁 Repository Structure

```
wsl/
├── data/
├── models/
├── heuristics/
├── notebooks/
└── README.md
```

---

## 🧪 References

- [Snorkel: Weak Supervision](https://arxiv.org/abs/1711.10160)
- [Co-Teaching](https://arxiv.org/abs/1804.06872)
- [MentorNet](https://arxiv.org/abs/1805.04770)

---

