---
title: >-
  [Paper Note] Understanding the Emergence of Multimodal Representation Alignment
description: >-
  [ICML 2025][Object Detection][Multimodal Alignment] This work systematically investigates the emergence mechanism of multimodal representation alignment. It reveals that the occurrence of implicit alignment and its relationship with performance depend on the ratio of redundant to unique information in the data and modal heterogeneity, challenging the common assumption of "larger models $\rightarrow$ better alignment $\rightarrow$ better performance."
tags:
  - "ICML 2025"
  - "Object Detection"
  - "Multimodal Alignment"
  - "Representation Learning"
  - "Implicit Alignment"
  - "Platonic Representation Hypothesis"
  - "CKA"
  - "Redundancy"
  - "Uniqueness"
  - "Heterogeneity"
date: 2026-05-08
content_hash: eee29c99691b4d26
---

# Understanding the Emergence of Multimodal Representation Alignment

**Conference**: ICML 2025  
**arXiv**: [2502.16282](https://arxiv.org/abs/2502.16282)  
**Area**: Object Detection  
**Keywords**: Multimodal Alignment, Representation Learning, Implicit Alignment, Platonic Representation Hypothesis, CKA, Redundancy, Uniqueness, Heterogeneity

## TL;DR

This work systematically investigates the emergence mechanism of multimodal representation alignment. It reveals that the occurrence of implicit alignment and its relationship with performance depend on the ratio of redundant to unique information in the data and modal heterogeneity, challenging the common assumption of "larger models $\rightarrow$ better alignment $\rightarrow$ better performance."

## Background & Motivation

The Platonic Representation Hypothesis (Huh et al., 2024) proposes a compelling view: as model scale increases, independently trained unimodal models (e.g., vision and language models) will naturally converge toward alignment. However, this raises two key questions:

**When and why does implicit alignment emerge?** If models always align automatically, why do explicit alignment methods (such as contrastive learning in CLIP) still remain effective?

**Does alignment reliably predict performance?** Does better alignment equate to better performance on downstream tasks?

The authors argue that existing research overlooks the crucial impact of data characteristics on the emergence of alignment, specifically:
- **Interaction Dimension**: How much task-relevant information is shared between the two modalities (redundancy $R$ vs. uniqueness $U$)
- **Heterogeneity Dimension**: How different the two modalities are structurally (e.g., text vs. image)

## Method

### Overall Architecture

This study systematically varies data characteristics along two orthogonal dimensions:
- **Interaction** (y-axis): From high redundancy (both modalities share the same information) to high uniqueness (each modality contains exclusive information)
- **Heterogeneity** (x-axis): From high similarity (e.g., two languages) to high difference (e.g., text and video)

### Synthetic Data Construction

Data for two modalities are constructed as: $x_1 = [x_r, x_{u1}]$ and $x_2 = [x_r, x_{u2}]$

where $x_r$ represents shared redundant information, and $x_{u1}$, $x_{u2}$ represent the unique information of each modality. Labels are determined by a non-linear function of subset features:

$$y = \psi_Y(x_r \odot M_R, x_{u1} \odot M_{U1}, x_{u2} \odot M_{U2})$$

By adjusting the ratio of $n_R$ (number of redundant features) to $n_U$ (number of unique features), the interaction properties of the data are controlled. Heterogeneity is introduced via an MLP transformation $\phi(\cdot)$: $x_2 = \phi([x_r, x_{u2}])$, where a larger number of MLP layers $D_\phi$ indicates higher heterogeneity.

### Alignment Metrics

- **CKA (Centered Kernel Alignment)** is used for synthetic data:

$$\text{CKA}(Z_1, Z_2) = \frac{\text{HSIC}(Z_1 Z_1^T, Z_2 Z_2^T)}{\sqrt{\text{HSIC}(Z_1 Z_1^T, Z_1 Z_1^T) \cdot \text{HSIC}(Z_2 Z_2^T, Z_2 Z_2^T)}}$$

- **Mutual KNN** is used for large-scale vision-language models (following Huh et al., 2024):

$$\text{ALIGN}_{\text{MKNN}}(Z_1, Z_2) = \sum_i \sum_j \mathbf{1}[Z_{1,j} \in knn(Z_{1,i}) \wedge Z_{2,j} \in knn(Z_{2,i}) \wedge i \neq j]$$

### Experimental Setup

- **Synthetic Experiments**: Train MLP encoders (depth 1-10), varying uniqueness $U \in \{0,...,8\}$ and transformation depth $D_\phi$.
- **Vision-Language Experiments**: Evaluate using the DINOv2 vision model and multiple LLMs (BLOOM, OpenLLaMA, LLaMA) on the Wikipedia Caption dataset.
- **MultiBench Experiments**: Validate on real-world multimodal datasets such as MOSEI, MOSI, URFUNNY, MUStARD, and AVMNIST.

## Key Experimental Results

### RQ1: When Does Alignment Emerge?

| Uniqueness $U$ | Max Achievable Alignment | Alignment Trend |
|-----------|------------|---------|
| 0 (Pure Redundancy) | High | Monotonically increases with model capacity |
| 1-3 | Medium | Increases but has an upper bound |
| 4-6 | Relatively Low | Weak increase |
| 7-8 | Low | Almost no increase |

**Key Formula**: Under high redundancy, alignment follows $(D_{Enc} - D_\phi) \propto \text{Alignment}$ (the difference between encoder depth and transformation depth is positively correlated with alignment); however, this relationship disappears as uniqueness increases.

### RQ2: Correlation Between Alignment and Performance

| Uniqueness $U$ | Alignment-Performance Pearson $r$ | Depth-Performance Pearson $r$ | Interpretation |
|-----------|---------------------|---------------------|------|
| 0 | ~1.0 (Strong positive correlation) | ~1.0 | Alignment reliably predicts performance |
| 1-3 | 0.5-0.8 | ~0.9 | Alignment is partially effective |
| >3 | ~0 (Even negative correlation) | ~0.8 | **Alignment fails, but model capacity remains effective** |

### RQ3: Alignment-Performance Correlation on Real MultiBench Data

| Dataset | Vision-Audio | Vision-Text | Audio-Text |
|-------|----------|----------|---------|
| MOSEI (Sentiment) | -0.193 / -0.154 | -0.154 / -0.351 | -0.158 / -0.366 |
| MOSI (Emotion) | -0.135 / 0.249 | 0.092 / -0.336 | 0.291 / -0.374 |
| URFUNNY (Humor) | -0.384 / -0.369 | -0.327 / 0.347 | -0.380 / 0.074 |
| MUStARD (Sarcasm) | 0.404 / 0.180 | 0.530 / 0.014 | 0.139 / 0.458 |
| **AVMNIST (Digits)** | **0.944 / 0.974** | - | - |

**Key Findings**:
- In AVMNIST (high redundancy: image digits + audio digits $\rightarrow$ classification), the alignment-performance correlation is as high as 0.97.
- In sentiment analysis tasks (high uniqueness), alignment and performance are frequently **negatively correlated**.
- The alignment-performance relationship also differs across different modal pairs within the same dataset (e.g., vision vs. text in MUStARD).

## Highlights & Insights

1. **Refinement of the Platonic Hypothesis**: The hypothesis holds under pure redundancy settings, but as uniqueness increases, alignment no longer emerges, let alone predicts performance.
2. **Scenarios Where Alignment is Harmful**: When modalities provide unique information (e.g., tone of voice in sentiment analysis), forcing alignment actually hurts performance.
3. **Model Capacity vs. Alignment**: Model capacity is always positively correlated with performance, but is only positively correlated with alignment under high redundancy—indicating that the source of performance gains is not alignment itself.
4. **Practical Guidance**: Helps practitioners determine when to use contrastive learning for multimodal alignment and when to avoid it.
5. **Intrinsic Property of Datasets**: The alignment-performance correlation is an inherent characteristic of the dataset, rather than a consequence of model selection.

## Limitations & Future Work

1. **Simplified Synthetic Data**: Using MLPs and binary features creates a large gap compared to real-world data distributions.
2. **Difficulty in Quantifying Uniqueness**: Accurately measuring redundant and unique information in real-world datasets heavily relies on human annotation or heuristic methods.
3. **Uniqueness Introduced via Noise in Vision-Language Experiments**: Deleting characters/pixels is not a genuine "unique info" manipulation.
4. **Lack of Theoretical Analysis**: The study provides only empirical findings without offering theoretical conditions for the emergence of alignment.
5. **Unexplored Scenarios with More Than Two Modalities**: Experiments are restricted to bi-modality; interactions in tri-modal contexts and beyond are more complex.

## Related Work & Insights

- **Direct Dialogue with the Platonic Representation Hypothesis**: This work provides a conditional version—which holds only under high redundancy conditions.
- **Connection to Contrastive Learning**: Theoretically, contrastive learning captures redundant information (Tian et al., 2020), which is empirically supported by this work.
- **Insights**:
    - When designing multimodal fusion architectures, the distribution of redundant/unique information in the dataset should be evaluated first.
    - For tasks dominated by uniqueness (e.g., sentiment analysis), architectures that preserve modality-specific information should be designed instead of blindly pursuing alignment.
    - This work can inspire new metrics to characterize which multimodal learning strategy is best suited for a given dataset.

## Rating

- **Novelty**: ★★★★☆ — Systemally studies the conditions for the emergence of alignment, offering a strong reflection on common assumptions.
- **Practicality**: ★★★★☆ — Provides clear practical guidance: when to align and when not to.
- **Experimental Thoroughness**: ★★★★★ — Three-level experiments comprising synthetic data, large-scale VLMs, and MultiBench, featuring 22 figures and 3 tables.
- **Writing Quality**: ★★★★★ — Clear research questions, a complete logical chain, and elegant figures and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Self-Organizing Visual Prototypes for Non-Parametric Representation Learning](self-organizing_visual_prototypes_for_non-parametric_representation_learning.md)
- [\[ICML 2025\] FG-CLIP: Fine-Grained Visual and Textual Alignment](fg-clip_fine-grained_visual_and_textual_alignment.md)
- [\[CVPR 2025\] T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting](../../CVPR2025/object_detection/t2icount_enhancing_cross-modal_understanding_for_zero-shot_counting.md)
- [\[NeurIPS 2025\] Multimodal Generative Flows for LHC Jets](../../NeurIPS2025/object_detection/multimodal_generative_flows_for_lhc_jets.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](../../NeurIPS2025/object_detection/detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)

</div>

<!-- RELATED:END -->
