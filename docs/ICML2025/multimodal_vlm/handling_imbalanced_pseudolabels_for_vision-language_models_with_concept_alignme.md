---
title: >-
  [Paper Note] Handling Imbalanced Pseudolabels for Vision-Language Models with Concept Alignment and Confusion-Aware Calibrated Margin
description: >-
  [ICML2025][Multimodal VLM][VLM Pseudolabels] Proposes the CAP framework, which addresses the class imbalance problem of VLMs generating pseudolabels through **concept alignment** (detecting and fixing concept mismatch) and **confusion-aware calibrated margin** (alleviating concept confusion), achieving a 6.29% relative improvement over SOTA models across six datasets and three paradigms.
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "VLM Pseudolabels"
  - "CLIP Fine-tuning"
  - "Imbalanced Pseudolabels"
  - "Concept Alignment"
  - "Calibrated Margin"
  - "Unsupervised/Semi-supervised Learning"
date: 2026-05-08
content_hash: 8759a38e8f3fc653
---

# Handling Imbalanced Pseudolabels for Vision-Language Models with Concept Alignment and Confusion-Aware Calibrated Margin

**Conference**: ICML2025  
**arXiv**: [2505.02056](https://arxiv.org/abs/2505.02056)  
**Code**: [GitHub](https://anonymous.4open.science/r/CAP-C642/)  
**Area**: Multimodal VLM  
**Keywords**: VLM Pseudolabels, CLIP Fine-tuning, Imbalanced Pseudolabels, Concept Alignment, Calibrated Margin, Unsupervised/Semi-supervised Learning

## TL;DR

Proposes the CAP framework, which addresses the class imbalance problem of VLMs generating pseudolabels through **concept alignment** (detecting and fixing concept mismatch) and **confusion-aware calibrated margin** (alleviating concept confusion), achieving a 6.29% relative improvement over SOTA models across six datasets and three paradigms.

## Background & Motivation

Utilizing the zero-shot capability of VLMs (such as CLIP) to generate pseudolabels for downstream unlabeled data for fine-tuning has recently become a research hotspot. The core challenge is that **VLMs exhibit preference bias toward different categories**, which leads to a severely imbalanced pseudolabel distribution and consequently triggers confirmation bias.

Existing methods such as UPL and FPL adopt a forced balance by selecting top-k high-confidence samples per class, GRIP increments the $k$ value round-by-round, and CPL assigns a candidate pseudolabel set for each sample—yet these are all **ex-post remedies** that fail to deeply analyze the root causes of the imbalance.

This work provides the first in-depth analysis of the two root causes of imbalance:

**Concept Mismatch**: The text features of class names are severely misaligned with image features, making it nearly impossible to correctly predict that class (e.g., approximately 5% of classes in RESISC45 suffer from this issue).

**Concept Confusion**: Text features of similar categories fail to capture the most discriminative visual concepts, causing predictions to skew toward a certain class (affecting about 30% of classes).

The authors visualized the clustering distribution of the five classes with the lowest accuracy on RESISC45—revealing that although image features cluster well, the zero-shot prediction accuracy of CLIP is extremely low, confirming the existence of a semantic gap.

## Method

### Overall Architecture: CAP

CAP (Concept-Adaptive Pseudolabeling) consists of three steps:

1. **Concept Alignment** (§3.1): Detects concept mismatch categories and enhances text descriptions using LLMs.
2. **Confusion-Aware Calibrated Margin** (§3.2): Constructs a margin matrix based on inter-class similarity and prediction tendencies.
3. **Dual-Adapter Fine-Tuning** (§3.3): Learns from high-quality pseudolabels and dynamic pseudolabels, respectively.

### 3.1 Concept Alignment

**Mismatch Detection Algorithm**: An iterative clustering strategy gradually removes well-matched classes, leaving the remaining ones as mismatch classes.

- Performs K-Means clustering (number of clusters = number of classes) on image features $\mathcal{I}$.
- Computes the similarity matrix $\mathbf{S}^{\mathcal{TC}}$ between text features and cluster centers, applying softmax to obtain the probability matrix.
- Finds the (text feature, cluster center) pair $(i^*, j^*)$ with the highest confidence and removes it.
- Iterates until the number of remaining classes is below a threshold $t$. The remaining classes are the candidate mismatch classes.
- Takes the intersection with the classes having the fewest predicted samples: $\mathcal{Y}_{\text{MM}} = \mathcal{Y}_{\text{final}} \cap \mathcal{Y}_{\text{low-}t}$.

**LLM Text Enhancement**: For mismatch classes, an LLM is queried to generate $n$ enhanced descriptions. The description with the highest similarity to the cluster center is selected to replace the original class name template, and pseudolabels are assigned based on top-k cosine similarity.

### 3.2 Confusion-Aware Calibrated Margin

Core Idea: Incorporates adaptive margins into the cross-entropy loss, encouraging the model to make more discriminative predictions between confusable classes.

**Calibrated Margin Loss**:

$$\mathcal{L}_m(y, \mathbf{z}) = -\log \frac{e^{z_y}}{e^{z_y} + \sum_{c \neq y} e^{z_c + \mathbf{M}_{yc}}}$$

**Construction of Margin Matrix $\mathbf{M}$**:

1. **Inter-class Similarity Matrix** $\mathbf{S}$: Takes the maximum of visual prototype similarity and textual prototype similarity.
   $$\mathbf{S}_{ij} = \max(\text{sim}(\bar{\mathbf{v}}_i, \bar{\mathbf{v}}_j), \text{sim}(\mathbf{w}_i, \mathbf{w}_j))$$

2. **Class Prediction Tendency** $\delta_c$: Counts the number of samples $\sigma(c)$ predicted as class $c$ with confidence exceeding the threshold $\tau$.
   $$\delta_c = 1 - \frac{\sigma(c)}{\max_j \sigma(j)}$$

3. **Class Margin Scaling** $m_c = m \times \Delta \times \delta_c$, where $\Delta = \max_c(\delta_c)$.

4. **Final Margin Matrix**: $\mathbf{M} = \mathbf{S} \odot \mathbf{m}$ (Hadamard product).

Key Design: $\mathbf{M}$ is updated once per epoch to progressively mitigate confusion. For classes with low prediction tendencies (large $\delta_c$) and high similarity to other classes, a larger margin penalty is applied.

### 3.3 Dual-Adapter Fine-Tuning Framework

Based on MaPLe prompt tuning, two independent visual adapters are deployed:

- **Main Adapter** ($\phi^m$): Learns solely from the high-precision pseudolabels $\mathcal{D}_{\text{PL}}$ obtained during the concept alignment phase, while simultaneously generating pseudolabels for unlabeled data.
- **Pseudo Adapter** ($\phi^p$): Learns solely from the unlabeled data $\mathcal{D}_{\text{UL}}$ with dynamic pseudolabels (FixMatch-style, filtered by threshold $\tau$).
- An adapter $\psi^a$ is also deployed on the text branch.
- **All adapters are disabled during inference.**

Total Loss: $\mathcal{L} = \mathcal{L}_{\text{PL}} + \mathcal{L}_{\text{UL}}$ (Unsupervised). SSL/TRZSL additionally adds $\mathcal{L}_{\text{L}}$.

## Key Experimental Results

The approach is compared across 3 paradigms (SSL / UL / TRZSL) on 6 datasets (Flowers102, RESISC45, DTD, EuroSAT, CUB, FGVCAircraft):

| Method | Flowers102 (UL) | RESISC45 (UL) | DTD (UL) | EuroSAT (UL) | CUB (UL) |
|------|---------|---------|------|---------|------|
| Zero-shot CLIP | 63.40 | 54.46 | 43.45 | 30.54 | 51.57 |
| FPL | 65.67 | 68.13 | 44.96 | 48.96 | 53.04 |
| GRIP | 69.84 | 74.11 | 46.09 | 57.21 | 51.42 |
| CPL | 72.90 | 80.98 | 51.91 | 67.26 | — |
| **CAP (Ours)** | **76.80** | **83.32** | **55.29** | — | — |

- Achieves a 3.9 percentage point improvement over CPL on Flowers102 UL, and a 2.3 percentage point improvement on RESISC45 UL.
- Achieves SOTA performance across all three paradigms (SSL/UL/TRZSL); obtaining an overall relative improvement of **6.29%** over CPL.
- Particularly shows significant improvement on classes suffering from severe concept mismatch.

## Highlights & Insights

1. **In-depth Problem Analysis**: For the first time, pseudolabel imbalance is attributed to two manifestations of the semantic gap: concept mismatch and concept confusion, supported by quantitative statistics (5% mismatch classes, 30% confusion classes).
2. **Iterative Clustering Detection**: Automatically discovers mismatch categories in an unsupervised manner without requiring any annotation.
3. **Ingenious Margin Matrix Design**: Jointly utilizes inter-class similarity and prediction tendencies to adaptively adjust the decision boundaries between different class pairs.
4. **Dual-Adapter Noise Isolation**: The main adapter maintains high precision without being contaminated by dynamic pseudolabel noise, providing a simple yet effective architecture.
5. **Comprehensive Paradigm Coverage**: Unifies three learning paradigms (UL / SSL / TRZSL) into a single framework.

## Limitations & Future Work

1. **LLM Dependency**: Concept alignment requires calling an LLM to generate text descriptions, increasing the pipeline complexity and cost.
2. **Hyperparameter Sensitivity**: Parameters such as threshold $t$ (for mismatch detection), $\tau$ (for confidence filtering), and the margin scale $m$ require tuning.
3. **Limited to Classification Tasks**: The framework design centers around image classification, and its applicability to tasks like object detection or segmentation has not been explored.
4. **Limited Dataset Scale**: The 6 datasets evaluated are of medium size; validation on large-scale scenarios such as ImageNet has not been performed.
5. **Adapters Disabled during Inference**: Adapters are enabled during training but disabled at inference. This training-inference discrepancy may limit the performance ceiling.

## Rating
- Novelty: ⭐⭐⭐⭐ — The problem analysis (mismatch vs. confusion) is novel; the combined scheme of iterative clustering detection and calibrated margin is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 6 datasets × 3 paradigms with complete ablation studies, though it lacks large-scale validation.
- Writing Quality: ⭐⭐⭐⭐ — Features clear motivational diagrams and complete mathematical derivations, resulting in good overall readability.
- Value: ⭐⭐⭐⭐ — VLM pseudolabel imbalance is a real practical pain point; the proposed method is transferable to other VLM fine-tuning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] CoCoA-Mix: Confusion-and-Confidence-Aware Mixture Model for Context Optimization](cocoa-mix_confusion-and-confidence-aware_mixture_model_for_context_optimization.md)
- [\[CVPR 2026\] CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](../../CVPR2026/multimodal_vlm/capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)
- [\[ICLR 2026\] Unified Vision-Language Modeling via Concept Space Alignment](../../ICLR2026/multimodal_vlm/unified_vision-language_modeling_via_concept_space_alignment.md)
- [\[ICML 2025\] Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models](kernel-based_unsupervised_embedding_alignment_for_enhanced_visual_representation.md)
- [\[CVPR 2026\] Quota-Calibrated Fine-Grained Alignment with Context-Aware Marginals for Text-based Person Retrieval](../../CVPR2026/multimodal_vlm/quota-calibrated_fine-grained_alignment_with_context-aware_marginals_for_text-ba.md)

</div>

<!-- RELATED:END -->
