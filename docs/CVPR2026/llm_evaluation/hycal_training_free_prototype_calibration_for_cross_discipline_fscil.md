---
title: >-
  [Paper Note] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning
description: >-
  [CVPR 2026][LLM Evaluation][Continual Learning] This paper identifies a "Domain Gravity" bias in heterogeneous-domain continual learning—whereby data-rich or low-entropy domains exert disproportionate influence in a shar…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Continual Learning"
  - "Few-Shot Class-Incremental Learning"
  - "Cross-Domain Adaptation"
  - "Prototype Calibration"
  - "Domain Gravity"
date: 2026-05-08
content_hash: 9938f66a1aa70250
---

# HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning

**Conference**: CVPR 2026
**arXiv**: [2604.15678](https://arxiv.org/abs/2604.15678)  
**Code**: N/A  
**Area**: LLM Evaluation
**Keywords**: Continual Learning, Few-Shot Class-Incremental Learning, Cross-Domain Adaptation, Prototype Calibration, Domain Gravity

## TL;DR
This paper identifies a "Domain Gravity" bias in heterogeneous-domain continual learning—whereby data-rich or low-entropy domains exert disproportionate influence in a shared embedding space—and proposes HyCal, a training-free method that calibrates prototypes by fusing cosine similarity and Mahalanobis distance, achieving robust classification in cross-discipline imbalanced few-shot class-incremental learning.

## Background & Motivation

1. **Background**: Pre-trained vision-language models (e.g., CLIP) have demonstrated strong performance in continual learning. Few-Shot Class-Incremental Learning (FSCIL) simulates real-world scenarios by limiting per-class samples, and has recently been extended to cross-domain settings that leverage the zero-shot capabilities of VLMs to preserve knowledge across domains.
2. **Limitations of Prior Work**: Existing cross-domain FSCIL methods still assume fixed few-shot configurations and balanced data distributions. In practice, heterogeneous domains differ substantially in visual entropy, feature geometry, and data availability. Projection- or kernel-based approaches (e.g., RanPAC) enrich feature representations but exacerbate drift toward data-rich domains. Covariance-based methods produce unstable covariance estimates under few-shot heterogeneous domains.
3. **Key Challenge**: Data imbalance across heterogeneous domains induces "Domain Gravity"—over-represented or low-entropy domains exert disproportionate influence in the shared embedding space, causing prototype drift for weakly represented domains and blurring decision boundaries. Existing methods implicitly assume homogeneous feature distributions and cannot counteract this asymmetric representational force.
4. **Goal**: (1) Define the Cross-Discipline Variable Few-Shot Class-Incremental Learning (XD-VSCIL) benchmark; (2) Propose a training-free prototype calibration method to mitigate Domain Gravity.
5. **Key Insight**: Cosine similarity and Mahalanobis distance capture complementary and statistically independent geometric information in high-dimensional space—directional alignment and covariance-aware magnitude, respectively.
6. **Core Idea**: Dynamically fuse cosine similarity (global directional stability) with Mahalanobis distance (domain-specific covariance correction) without modifying the backbone network, enabling robust prototype matching.

## Method

### Overall Architecture
Using a frozen CLIP model, class prototypes (mean embeddings and regularized precision matrices) are constructed from few-shot images for each incremental task. At inference, HyCal simultaneously computes cosine similarity and Mahalanobis distance between a test sample and each prototype, fusing the two scores via adaptive weights for classification. The entire pipeline requires no training or parameter updates.

### Key Designs

1. **Domain Gravity Concept**:
    - **Function**: Provides a structured explanation for representational bias in heterogeneous-domain continual learning.
    - **Mechanism**: Each domain generates a "representational potential" based on its visual consistency and data density. Low-entropy or data-rich domains exert disproportionate influence in the shared embedding space. Domain Gravity has two sources: (1) pre-training bias—CLIP inherits distributional bias from large-scale corpora, with frequent domains dominating embedding geometry; (2) incremental accumulation—as incremental tasks arrive, prototypes and embeddings drift toward visually consistent domains. t-SNE visualizations directly demonstrate prototype drift of under-represented domains in methods such as RanPAC.
    - **Design Motivation**: Elevates the observation that "domain imbalance causes performance degradation" from a phenomenological description to a structural explanation, providing a clear objective for method design.

2. **Cosine–Mahalanobis Fusion (Core of HyCal)**:
    - **Function**: Leverages complementary geometric information for robust prototype matching.
    - **Mechanism**: The paper theoretically proves that, under an isotropic Gaussian assumption, the directional vector $U$ and magnitude $R$ are statistically independent ($R \perp U$); therefore, cosine similarity (dependent on $U$) and Mahalanobis distance (dependent on $R$ and the covariance) capture non-overlapping information, yielding $H(C, M) = H(C) + H(M)$. Mutual information analysis further proves that the combination provides strictly more discriminative information: $I(L; C, M) \geq \max\{I(L; C), I(L; M)\}$. The final prediction is obtained by fusing the two scores with adaptive weights:

$$c_{\text{pred}} = \arg\max_c \left[ w_c \cdot d_{\text{maha}} + (1 - w_c) \cdot s_{\text{cos}} \right]$$

where the weight $w_c = \sigma\!\left((K_c^t - \alpha)/\beta\right)$ adapts to the per-class sample count.
   - **Design Motivation**: A single distance metric is unstable under heterogeneous domains—cosine similarity ignores domain-specific variance, while Mahalanobis distance yields unreliable covariance estimates under few-shot conditions. Fusing the two exploits the strengths of each to compensate for the weaknesses of the other.

3. **XD-VSCIL Benchmark and CDE Evaluation Metric**:
    - **Function**: Provides standardized evaluation reflecting real-world heterogeneity and imbalance.
    - **Mechanism**: Eight cross-discipline datasets are selected (Aircraft, ArtBench, DTD, EuroSAT, Galaxy, MNIST, OrganMNIST, OxfordFlowers) as sequential tasks. The number of classes and per-class samples are allowed to vary across tasks. The Cross-Discipline Efficiency (CDE) metric is proposed, combining adaptability $S_{\text{adapt}}$ and final accuracy $S_{\text{last}}$ via harmonic mean, weighted by $w^t \propto 1/\sqrt{K^t}$ to reward data efficiency.
    - **Design Motivation**: Existing FSCIL benchmarks assume fixed few-shot settings and homogeneous domains, which fails to reflect real-world scenarios. The variable sample counts and heterogeneous domains in XD-VSCIL more closely approximate practical conditions.

### Loss & Training
HyCal is entirely training-free and involves no loss functions or backpropagation. Only the mean embedding, regularized precision matrix, and sample count for each class need to be stored. Covariance regularization uses $\Sigma_c^{reg} = (1-\lambda)\Sigma_c + \lambda\gamma I$ to ensure stability under few-shot conditions.

## Key Experimental Results

### Main Results

**High-Scale Domain Imbalance (8 Domains)**:

| Method | Average Accuracy | Final Accuracy | Std. Dev. |
|---|---|---|---|
| Primal-RAIL | 53.49% | 59.86% | 22.04 |
| RanPAC | 49.98% | 61.13% | 21.57 |
| KLDA | 41.06% | 61.43% | 24.61 |
| **HyCal** | **54.48%** | **63.50%** | **19.50** |

**Preliminary Domain Imbalance Analysis (2 Domains)**:

| Setting | HyCal | RanPAC | Inter-Domain Gap |
|---|---|---|---|
| General (10-shot) | 65.26% | 63.57% | **0.45** vs 2.06 |
| Balanced (20/5-shot) | 64.98% | 60.77% | 8.80 vs 11.49 |
| Imbalanced (5/10-shot) | 62.84% | 59.23% | 4.94 vs 6.83 |

### Ablation Study

| Configuration | Final Accuracy | Notes |
|---|---|---|
| HyCal (Cosine + Mahalanobis) | 63.50% | Full method |
| Cosine similarity only | ~61% | Missing covariance information |
| Mahalanobis distance only | ~60% | Unstable under few-shot |
| FeCAM (covariance method) | 5.69% | Catastrophic collapse on heterogeneous domains |

### Key Findings
- **HyCal achieves the lowest standard deviation (19.50 vs. 21–24)**: The method performs more consistently across domains, effectively mitigating the performance asymmetry induced by Domain Gravity.
- **FeCAM collapses entirely under heterogeneous domains (5.69%)**: Pure covariance-based methods are not viable under heterogeneous few-shot settings.
- **Inter-domain gap minimization**: Under the general 10-shot setting, HyCal's inter-domain gap is only 0.45% (vs. 2.06% for RanPAC), directly validating the fusion strategy's effectiveness in mitigating Domain Gravity.

## Highlights & Insights
- The **"Domain Gravity" concept** is the most valuable contribution of this paper: it attributes performance degradation in heterogeneous-domain continual learning to a structural bias rather than mere data scarcity, providing a clear analytical framework for subsequent research in this area.
- The **information-theoretic proofs** (Theorems 1 & 2) provide a solid theoretical foundation for cosine–Mahalanobis fusion, particularly the independence proof and mutual information inequality, elevating the method beyond empirical trial-and-error.
- The **training-free design** offers strong practical utility: no additional parameters, no backpropagation, and no backbone modification are required, enabling direct integration into existing CLIP-based continual learning pipelines.

## Limitations & Future Work
- The theoretical analysis of complementarity relies on an isotropic Gaussian assumption, which diverges from the highly anisotropic distributions observed in actual VLM embeddings.
- The hyperparameters $\alpha$ and $\beta$ in the sigmoid function for the fusion weight $w_c$ require manual tuning.
- Validation is limited to CLIP; the method has not been extended to other VLMs (e.g., SigLIP, EVA-CLIP).
- The sequential order of the 8 domains may affect results; order sensitivity is not thoroughly explored.
- Future work could investigate adaptive covariance regularization strength and meta-learning-based fusion weights.

## Related Work & Insights
- **vs. RanPAC**: RanPAC uses random projections to enrich prototype representations, but suffers severe prototype drift under heterogeneous domains (as directly demonstrated by t-SNE visualizations). HyCal's dual-distance fusion does not modify the feature space; instead, it calibrates at inference time.
- **vs. Primal-RAIL**: Primal-RAIL employs a parametric approach to adapt to new domains, but exhibits large performance fluctuations under imbalance. HyCal's training-free nature makes it inherently more robust to variations in sample count.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The Domain Gravity concept is insightful; the cosine–Mahalanobis fusion is simple yet theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple imbalance settings and baseline comparisons are provided, though the number of domains (8) is relatively limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and theoretical derivations are rigorous, though certain sections are overly verbose.
- **Value**: ⭐⭐⭐⭐ — The XD-VSCIL benchmark and Domain Gravity concept offer long-term value to the continual learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)
- [\[CVPR 2026\] Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](../../NeurIPS2025/llm_evaluation/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [\[CVPR 2026\] Free-Grained Hierarchical Visual Recognition](free-grained_hierarchical_visual_recognition.md)

</div>

<!-- RELATED:END -->
