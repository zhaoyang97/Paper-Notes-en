---
title: >-
  [Paper Note] Exploring Structural Degradation in Dense Representations for Self-supervised Learning
description: >-
  [NeurIPS 2025][Segmentation][Self-supervised Learning] This paper identifies and systematically investigates the *Self-supervised Dense Degradation* (SDD) phenomenon — where longer training improves classification yet hurts dense task performance — and proposes the DSE metric along with DSE-guided model selection and regularization strategies, achieving an average mIoU improvement of 3.0%.
tags:
  - NeurIPS 2025
  - Segmentation
  - Self-supervised Learning
  - Dense Representations
  - Performance Degradation
  - Model Selection
  - Regularization
date: 2026-05-08
content_hash: d34bf5096e30d4a4
---

# Exploring Structural Degradation in Dense Representations for Self-supervised Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.17299](https://arxiv.org/abs/2510.17299)
**Code**: [GitHub](https://github.com/EldercatSAM/SSL-Degradation)
**Area**: Image Segmentation
**Keywords**: Self-supervised Learning, Dense Representations, Performance Degradation, Model Selection, Regularization

## TL;DR

This paper identifies and systematically investigates the *Self-supervised Dense Degradation* (SDD) phenomenon — where longer training improves classification yet hurts dense task performance — and proposes the DSE metric along with DSE-guided model selection and regularization strategies, achieving an average mIoU improvement of 3.0%.

## Background & Motivation

Self-supervised learning (SSL) has achieved remarkable success in image-level representation learning, but improvements in dense (patch/pixel-level) representation learning remain limited. This paper uncovers a counterintuitive phenomenon:

**Self-supervised Dense Degradation (SDD)**: Although training loss converges and classification performance steadily improves, performance on dense tasks such as semantic segmentation *degrades* in the later stages of training.

**Universality**: SDD appears consistently across 16 state-of-the-art SSL methods, spanning contrastive learning (MoCo v3, DenseCL), non-contrastive learning (BYOL, SimSiam, DINO), clustering-based methods (SwAV), and masked modeling approaches (MAE, I-JEPA).

**Not Overfitting**: SDD persists even when training and evaluation are conducted on the same dataset (COCO) — e.g., DINO exhibits a 4.0% mIoU drop.

**Existing Metrics Fail**: Metrics such as α-REQ, RankMe, and Lidar are primarily designed for image-level tasks and exhibit *negative* correlation with dense task performance.

## Method

### Overall Architecture

Grounded in error rate decomposition theory, the paper proposes the **Dense representation Structure Estimator (DSE)**, which combines a class separability measure and an effective dimensionality measure:

$$\text{DSE} = M_{inter} - M_{intra} + \lambda \cdot M_{dim}$$

### Key Designs

**Theoretical Foundation**:
- **Theorem 2** (Class-related measure): Proves that when the intra-class radius (estimated via the trace of the normalized representation matrix) is smaller than the inter-class distance, a simple nearest-neighbor classifier suffices for correct classification.
- **Corollary 5** (Dimensionality effect): Proves that the downstream error rate decays exponentially with representation dimension $d$: $\text{Err} \leq \delta + 2K\exp(-\tilde{C}_\delta \cdot d)$.

**Class Separability Measure**:
- Pseudo-labels are generated via k-means clustering.
- Intra-class radius: $M_{intra} = \frac{1}{k}\sum_{j=1}^{k} \frac{\sum_{i=1}^{\min(\tilde{N}_j, d)} \sigma_i(\tilde{Z}_c^j)}{(\tilde{N}_j - 1)}$
- Inter-class distance: $M_{inter} = \frac{1}{k}\sum_{j=1}^{k} \frac{1}{N_j}\sum_{z \in \tilde{Z}_j} \min_{i \neq j} \|z - \tilde{\mu}_i\|^2$

**Effective Dimensionality Measure**:
- Randomly samples $B'$ independent dense representations and computes the effective rank: $M_{dim} = \text{Erank}(\bar{Z}) = \exp(-\sum_i p_i \log p_i)$

**Adaptive Scaling**: $\lambda = \text{Std}(M_{inter} - M_{intra}) / \text{Std}(M_{dim})$

### Loss & Training

**DSE-Guided Model Selection** (offline):
1. Compute DSE for all checkpoints.
2. Identify local maxima of DSE as candidates.
3. Select the top-3 checkpoints with the highest DSE values.

**DSE Regularization** (online):
$$\mathcal{L} = \mathcal{L}_{original} - \beta \cdot \text{DSE}$$
where $\lambda = 1$, $\beta = 0.001$, with training resumed from the checkpoint of best initial performance for 10 epochs.

## Key Experimental Results

### Main Results

**SDD phenomenon across 16 SSL methods** (Best vs. Last mIoU gap on COCO-Stuff / PASCAL VOC / ADE20k / Cityscapes):

| Method | COCO Diff | VOC Diff | ADE20k Diff | Cityscapes Diff |
|--------|-----------|----------|-------------|-----------------|
| MoCo v3 | -22.0 | -45.2 | -14.4 | -11.5 |
| DINO | -4.4 | -11.3 | -4.2 | -0.1 |
| iBOT | -2.5 | -3.0 | -3.7 | -3.2 |
| I-JEPA | -5.6 | -7.6 | -4.5 | -3.9 |
| BYOL | -6.4 | -6.7 | -7.9 | -7.5 |
| MAE | -0.4 | -1.3 | -0.7 | -2.1 |

**DSE model selection results** (+MS denotes improvement after model selection):

| Method | COCO mIoU | VOC mIoU |
|--------|-----------|----------|
| MoCo v3 | 15.1 → **30.9** (+15.8) | 5.9 → **42.0** (+36.1) |
| BYOL | 30.7 → **37.1** (+6.4) | 45.4 → **51.1** (+5.7) |
| I-JEPA | 34.0 → **39.6** (+5.6) | 52.6 → **59.3** (+6.7) |
| EsViT | 33.4 → **41.6** (+8.2) | 54.3 → **59.8** (+5.5) |

### Ablation Study

**DSE vs. other metrics** (average Kendall's τ):

| Metric | COCO | VOC | ADE20k | City | Avg. |
|--------|------|-----|--------|------|------|
| α-ReQ | -0.07 | -0.05 | -0.05 | 0.09 | -0.02 |
| RankMe | -0.10 | -0.09 | -0.14 | 0.00 | -0.08 |
| Lidar | -0.37 | -0.36 | -0.26 | -0.21 | -0.30 |
| RankMe† (dense-adapted) | 0.25 | 0.26 | 0.22 | 0.23 | 0.24 |
| **DSE (Ours)** | **0.58** | **0.60** | **0.56** | **0.49** | **0.57** |

**DSE component ablation** (average Kendall's τ):

| Class Separability | Eff. Dimensionality | COCO | VOC | ADE20k | City | Avg. |
|--------------------|---------------------|------|-----|--------|------|------|
| ✓ | ✗ | 0.45 | 0.42 | 0.33 | 0.37 | 0.39 |
| ✗ | ✓ | 0.25 | 0.26 | 0.22 | 0.23 | 0.24 |
| ✓ | ✓ | **0.58** | **0.60** | **0.56** | **0.49** | **0.57** |

**Efficiency comparison**:

| Method | Avg. Improvement | Compute Cost (GPU·h) |
|--------|-----------------|----------------------|
| Loss-based | -1.0 | 0.0 |
| Supervised | +3.6 | 2.43 |
| DSE (Ours) | **+3.0** | **0.025** (~97× speedup) |

### Key Findings

1. **Degradation causes vary by method**: In MoCo v3, SDD stems from dimensional collapse; in DINO, it stems from declining class separability.
2. **DSE regularization reverses degradation**: On iBOT and I-JEPA, adding DSE regularization halts the performance decline.
3. **DSE generalizes to image-level tasks**: On ImageNet k-NN evaluation, the average Kendall's τ reaches 0.86, outperforming RankMe (0.79).
4. **Minimal data suffices**: Only 2,048 images (~0.16% of training data) are needed to reliably compute DSE.

## Highlights & Insights

1. **Breadth of the discovered phenomenon**: SDD is systematically validated across 16 methods × 4 datasets × multiple evaluation protocols, constituting a community-level finding of significant importance.
2. **Elegant unification of theory and practice**: Starting from error rate decomposition, the paper derives class separability and dimensionality as the two governing factors, then designs a directly optimizable DSE metric.
3. **High practical utility**: DSE-guided model selection requires only 0.025 GPU·h while yielding an average mIoU gain of 3.0%.
4. **Insight from Proposition 1**: Reveals the fundamental issue that instance-level distance measures using k-means pseudo-labels always predict 100% accuracy, motivating the design of a class-level radius measure instead.

## Limitations & Future Work

1. Theoretical analysis is primarily scoped to the linear probing setting, without fully accounting for distribution shift in transfer learning.
2. DSE exhibits relatively weaker predictive power on regression tasks such as depth estimation (lower Kendall's τ).
3. Online DSE regularization requires method-specific adaptation (e.g., the manner in which dense representations are extracted from the student model).
4. The specific mechanisms underlying dimensional collapse or separability degradation in individual methods remain underexplored.

## Related Work & Insights

- **DINOv3** (concurrent work) investigates degradation from the perspective of the iBOT/DINOv2 family and addresses it via Gram matrix distillation, complementing the present work.
- **RankMe** is essentially a dense-adapted version of the proposed $M_{dim}$, but fails to capture degradation in class separability.
- The findings offer a new perspective for SSL training strategy design: an appropriate trade-off between class separability and dimensional collapse should be pursued.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Identifies an important phenomenon insufficiently recognized by the community.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Systematic validation across 16 methods × 4 datasets.
- Value: ⭐⭐⭐⭐⭐ — A near-zero-cost model selection strategy with consistent gains.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from phenomenon to theory to method is exceptionally clear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-supervised Synthetic Pretraining for Inference of Stellar Mass Embedded in Dense Gas](self-supervised_synthetic_pretraining_for_inference_of_stellar_mass_embedded_in_.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](../../ICCV2025/segmentation/joint_self-supervised_video_alignment_and_action_segmentation.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)
- [\[NeurIPS 2025\] Vision Transformers with Self-Distilled Registers](vision_transformers_with_self-distilled_registers.md)
- [\[CVPR 2026\] Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning](../../CVPR2026/segmentation/follow_the_saliency_supervised_saliency_for_retrieval-augmented_dense_video_capt.md)

</div>

<!-- RELATED:END -->
