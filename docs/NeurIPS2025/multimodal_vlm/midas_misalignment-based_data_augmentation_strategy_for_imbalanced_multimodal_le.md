---
title: >-
  [Paper Note] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][modality imbalance] This work is the first to propose using cross-modal misaligned samples as supervised training signals—rather than treating them as noise or interference—to alleviate mod…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "modality imbalance"
  - "data augmentation"
  - "misaligned samples"
  - "weak-modality weighting"
  - "hard-sample weighting"
date: 2026-05-08
content_hash: e2716c1d942342eb
---

# MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.25831](https://arxiv.org/abs/2509.25831)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / Multimodal Learning / Data Augmentation
**Keywords**: modality imbalance, data augmentation, misaligned samples, weak-modality weighting, hard-sample weighting

## TL;DR
This work is the first to propose using cross-modal misaligned samples as supervised training signals—rather than treating them as noise or interference—to alleviate modality imbalance in multimodal learning. The proposed MIDAS data augmentation framework combines three complementary mechanisms: confidence-based labeling of misaligned samples, weak-modality weighting, and hard-sample weighting. MIDAS substantially outperforms existing methods across four multimodal classification benchmarks.

## Background & Motivation

**Background**: Multimodal learning is widely applied in vision-language modeling, medical diagnosis, autonomous driving, and other domains. However, **modality imbalance** remains a persistent challenge—models tend to rely on the more informative dominant modality, neglect weaker modalities, and may even perform worse than unimodal counterparts.

**Limitations of Prior Work**:
- **Optimization-based methods** (OGM, AGM): Suppress the dominant modality by adjusting gradients or weights, but introduce additional computational overhead.
- **Data/feature-based methods** (AMCo, SMV): Balance modalities by masking dominant-modality features or resampling, but fail to fully exploit available information.
- **Contrastive/unsupervised methods** (LFM, MCR): Use misaligned data as negative samples or for mutual information estimation, but lack direct supervision signals.

**Core Insight**: Misaligned samples (e.g., a cat image paired with dog-related text) contain rich modality-specific information and can expose a model's over-reliance on the dominant modality. On misaligned inputs, a standard model achieves only 6.3% accuracy (vs. 65.5% on aligned inputs) and consistently predicts the class corresponding to the dominant modality with high confidence.

**Goal**: Transform misaligned samples from noise into supervised training signals, compelling the model to learn balanced utilization of all modalities from contradictory inputs.

## Method

### Overall Architecture
MIDAS trains jointly on aligned and misaligned samples, comprising three core components: confidence-based labeling, weak-modality weighting, and hard-sample weighting.

### 1. Generating Misaligned Samples
Two samples $(x_i, y_i)$ and $(x_j, y_j)$ with different labels ($y_i \neq y_j$) are randomly selected from the same mini-batch, and one modality is swapped:
$$\tilde{x}_i = (\tilde{x}_i^1, \tilde{x}_i^2) = (x_i^1, x_j^2)$$
For example: a cat image paired with dog text. Symmetrically, $(x_j^1, x_i^2)$ is also generated.

### 2. Unimodal Confidence-based Labeling
Hard labels from either class cannot be naively assigned to misaligned samples. Pre-trained unimodal classifiers are used to assess each modality's confidence in its original class.

Normalized confidence scores:
$$\tilde{c}_i^1 = \frac{(p_i^1)_{y_i}}{(p_i^1)_{y_i} + (p_j^2)_{y_j}}, \quad \tilde{c}_i^2 = \frac{(p_j^2)_{y_j}}{(p_i^1)_{y_i} + (p_j^2)_{y_j}}$$

The soft label is a weighted average:
$$\tilde{y}_i = \tilde{c}_i^1 \mathbf{y}_i + \tilde{c}_i^2 \mathbf{y}_j$$

For example, if visual confidence is 0.9 and textual confidence is 0.3, then $\tilde{c}^1=0.75$ and $\tilde{c}^2=0.25$.

### 3. Weak-Modality Weighting
Confidence-based labeling still tends to favor the stronger modality. To address this, the loss weight of the least confident modality is dynamically increased.

The modality with the lowest average confidence across the batch is identified:
$$\hat{m} = \arg\min_{m \in \{1,2\}} \mathbb{E}_{(\tilde{x}_i, \tilde{y}_i) \sim \tilde{B}}[\tilde{c}_i^m]$$

The discrepancy between the weak modality's contribution in the target label and the multimodal model's actual prediction is measured:
$$\Delta_{\alpha} = \text{sign}\left(\mathbb{E}[\tilde{c}_i^{\hat{m}}] - \mathbb{E}[(\tilde{c}_i)_{\tilde{y}_i^{\hat{m}}}]\right)$$

Update rule:
$$\alpha_{\hat{m}}^{(t+1)} = \max(1, \alpha_{\hat{m}}^{(t)} + \eta \cdot \Delta_{\alpha})$$

When the model underestimates the weak modality's contribution, $\alpha$ increases, amplifying the corresponding loss weight.

### 4. Hard-Sample Weighting
Not all misaligned samples are equally informative—when the swapped features are more similar to the original features, the semantic conflict is more subtle and thus more valuable for training.

$$\tilde{s}_i = \frac{(f_i^2)^\top f_j^2}{\|f_i^2\|_2 \|f_j^2\|_2}$$

### 5. Total Loss
$$\mathcal{L}_{mis}(\tilde{x}_i, \tilde{y}_i; \alpha^{(t)}, \tilde{s}_i) = \left(1 + \frac{\tilde{s}_i + 1}{2}\right) \cdot \left[-\sum_{c=1}^{C}(\alpha_1^{(t)}\tilde{c}_i^1(\mathbf{y}_i)_c + \alpha_2^{(t)}\tilde{c}_i^2(\mathbf{y}_j)_c)\log((\tilde{p}_i)_c)\right]$$

$$\mathcal{L}_{total} = \frac{1}{|B|}\sum_{(x_i, y_i) \in B}\left[\mathcal{L}_{align}(x_i, y_i) + \mathcal{L}_{uni}(x_i, y_i) + \lambda \mathcal{L}_{mis}(\tilde{x}_i, \tilde{y}_i; \alpha^{(t)}, \tilde{s}_i)\right]$$

A warm-up phase pre-trains the encoders and unimodal classifiers before the main training begins. Computational complexity remains $O(N)$.

## Key Experimental Results

### Main Results (4 Datasets)

| Method | K-S Acc | K-S F1 | CREMA-D Acc | CREMA-D F1 | UCF-101 Acc | Food-101 Acc |
|--------|---------|--------|-------------|------------|-------------|--------------|
| Joint | 63.92 | 55.54 | 60.28 | 58.60 | 90.07 | 91.35 |
| SMV | 65.76 | 57.59 | 67.94 | 66.83 | **95.24** | 91.64 |
| OPM | 67.35 | 59.29 | 63.97 | 62.71 | 91.73 | 92.40 |
| AMCo | 67.04 | 58.41 | 69.91 | 68.85 | 93.77 | 92.00 |
| MCR | 71.75 | 64.23 | 70.91 | 70.19 | 91.84 | 90.58 |
| **MIDAS** | **74.88** | **67.18** | **74.99** | **73.82** | 95.20 | **93.46** |

- Kinetics-Sounds: +3.13%p over the best baseline (MCR)
- CREMA-D: +4.08%p over the best baseline

### Ablation Study

| W | WM | HS | K-S Acc | CREMA-D Acc | UCF-101 Acc | Food-101 Acc |
|---|----|----|---------|-------------|-------------|--------------|
| ✗ | ✗ | ✗ | 71.70 | 72.32 | 94.16 | 93.39 |
| ✓ | ✓ | ✓ | **74.88** | **74.99** | **95.20** | 93.46 |

Each component yields limited gains individually; their combination produces clear synergistic effects.

### Comparison with Data Augmentation Methods

| Method | CREMA-D Acc | Food-101 Acc |
|--------|-------------|--------------|
| Mixup | 61.84 | 91.36 |
| PowMix | 63.66 | 89.59 |
| LeMDA | 58.13 | 91.19 |
| **MIDAS** | **74.99** | **93.11** |

### Three-Modality Experiment (CMU-MOSI)
MIDAS achieves 74.00% Acc / 73.64 F1 vs. Joint's 71.13% Acc / 70.86 F1, demonstrating scalability to three-modality settings.

## Highlights & Insights
- ⭐⭐⭐⭐ **Novel Perspective**: Reframing misaligned samples from noise into valuable supervised signals represents an impressive conceptual shift.
- ⭐⭐⭐⭐ **Complete Mechanism**: Confidence-based labeling, weak-modality weighting, and hard-sample weighting form a complementary and self-consistent framework.
- ⭐⭐⭐⭐ **Theoretical Clarity**: The design motivation, mathematical derivation, and ablation validation for each component are thorough and well-grounded.
- ⭐⭐⭐ **Strong Generalizability**: The modality-agnostic design has been validated on audio-video, image-text, RGB-optical flow, and three-modality settings.

## Limitations & Future Work
1. Validation is limited to classification tasks; applicability to generation, retrieval, and other tasks requires further exploration.
2. When the information gap between modalities is extreme, the labeling quality of misaligned samples may degrade.
3. The warm-up phase requires separate training of unimodal classifiers, increasing the complexity of the overall training pipeline.
4. The random pairing strategy is simple and efficient but may be suboptimal; semantics-aware intelligent pairing strategies warrant further investigation.

## Rating
⭐⭐⭐⭐ A highly inspiring work in which the core idea of "misalignment as a resource" elegantly unifies three technical components. The experimental coverage is comprehensive and the ablation analysis is thorough. The paper offers a novel data-driven solution to the modality imbalance problem in multimodal learning.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)
- [\[NeurIPS 2025\] Learning Shared Representations from Unpaired Data](learning_shared_representations_from_unpaired_data.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)
- [\[AAAI 2026\] Panda: Test-Time Adaptation with Negative Data Augmentation](../../AAAI2026/multimodal_vlm/panda_test-time_adaptation_with_negative_data_augmentation.md)

</div>

<!-- RELATED:END -->
