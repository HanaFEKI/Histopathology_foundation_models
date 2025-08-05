# Multiple Instance Learning (MIL)

Multiple Instance Learning (MIL) is a specialized form of supervised learning designed to handle **weakly labeled data**, where labels are provided at a **bag level** rather than on individual instances within the bag.

In MIL, the data is organized into **bags**, each containing many instances (e.g., image patches). The label is associated only with the entire bag — not with each instance — making MIL especially useful when precise annotations are unavailable or costly.

---

## Why MIL?

- **Weak or coarse labeling:**  
  In computational pathology, obtaining detailed pixel- or patch-level annotations is difficult and expensive. Often, only slide-level labels (e.g., cancer present or not) are available.

- **Instance ambiguity:**  
  Not all instances (patches) in a positive bag are necessarily positive, but at least one instance should be.

- **Scalability:**  
  MIL enables training models on whole-slide images (WSIs) by treating them as bags of smaller patches, avoiding exhaustive manual labeling.

## Core Concepts

- **Bags and instances:**  
  Each sample is a bag (e.g., a WSI) composed of many instances (image patches or regions).

- **Bag label:**  
  Indicates the presence or absence of a condition (e.g., tumor) somewhere in the bag, but instance labels are unknown.

- **MIL assumption:**  
  A bag is positive if it contains at least one positive instance; otherwise, it is negative.

- **Model workflow:**  
  Models typically extract features from instances, then aggregate instance-level information to predict the bag label (check the [aggregation folder](../../Other%20Techniques/Aggregation/) for detailed techniques).

## Importance in Computational Pathology

- **Annotation bottleneck reduction:**  
  MIL significantly reduces the need for costly pixel-level annotations by leveraging slide-level labels.

- **Whole-slide image analysis:**  
  MIL frameworks allow efficient modeling of gigapixel WSIs by dividing them into manageable patches while respecting their weakly supervised nature.

- **Improved diagnostic tools:**  
  MIL models help detect and localize disease areas implicitly, improving accuracy and interpretability without requiring dense labels.

- **Flexibility:**  
  MIL frameworks can incorporate different feature extractors, instance encoders, and pooling/aggregation strategies (check the [aggregation folder](../../Other%20Techniques/Aggregation/) for details).

---

## References and Further Reading

- **Recent MIL survey:**  [Ilse et al., Attention-based Deep Multiple Instance Learning (2018)](https://arxiv.org/abs/1802.04712) 
- **MIL in computational pathology:**  [Campanella et al., Clinical-grade computational pathology using weakly supervised deep learning on whole slide images (2019)](https://www.nature.com/articles/s41591-019-0508-1)
