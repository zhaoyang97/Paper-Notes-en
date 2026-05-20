---
title: >-
  [Paper Note] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation
description: >-
  [NeurIPS 2025][Segmentation][Unsupervised Domain Adaptation] This paper proposes DiDA, which formalizes image degradation operations as the forward process of diffusion models to construct a continuous intermediate domai…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Unsupervised Domain Adaptation"
  - "Semantic Segmentation"
  - "Diffusion Process"
  - "Image Degradation"
  - "Domain Bridging"
date: 2026-05-08
content_hash: f8240969e183c2f9
---

# Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation

**Conference**: NeurIPS 2025
**arXiv**: [2412.10339](https://arxiv.org/abs/2412.10339)  
**Code**: [Available](https://github.com/Woof6/DiDA)  
**Area**: Image Segmentation
**Keywords**: Unsupervised Domain Adaptation, Semantic Segmentation, Diffusion Process, Image Degradation, Domain Bridging

## TL;DR

This paper proposes DiDA, which formalizes image degradation operations as the forward process of diffusion models to construct a continuous intermediate domain between the source and target domains. Combined with a semantic shift compensation mechanism, DiDA serves as a plug-and-play module that consistently improves existing UDA semantic segmentation methods.

## Background & Motivation

Semantic segmentation models suffer from severe performance degradation when deployed across domains. While self-training (ST) has become the dominant paradigm in UDA (e.g., DAFormer, HRDA, MIC series), these methods **overlook the explicit modeling of domain-shared feature extraction**.

From a causal representation learning perspective, the observed feature $x = \Phi(c, e)$, where $c$ denotes causal features determining class identity (e.g., shape) and $e$ denotes domain-specific features (e.g., texture). Since $e_S \neq e_T$, we have $x_S \neq x_T$, which hinders the learning of domain-invariant features.

The **core insight** is drawn from the forward process of diffusion models: progressively adding noise removes attributes in order of granularity — **fine-grained domain-specific attributes (texture) are lost first, while coarse-grained domain-invariant attributes (shape) are lost later**. This implies that the overlapping region of intermediate domain distributions created by degradation can serve as a **prior for the domain-shared distribution**.

However, directly using degradation as domain bridging poses two major challenges: (1) stable feature representations must be maintained across a wide range of degradation levels; and (2) degradation inevitably damages domain-invariant features, leading to the **semantic shift** problem.

## Method

### Overall Architecture

DiDA is integrated into the standard self-training (ST) UDA pipeline and consists of two core modules: (1) degradation-based intermediate domain construction, which creates a continuous intermediate domain via the diffusion forward process; and (2) semantic shift compensation, which uses a diffusion encoder to disentangle and compensate for semantic information loss caused by degradation. At inference time, only the backbone segmentation network $f_\theta = h \circ g$ is used, with no additional computational overhead.

### Key Designs

1. **Degradation-based Intermediate Domain Construction**: The intermediate states $X_1, X_2, \ldots, X_T$ produced by the diffusion forward process $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ are treated as intermediate domains. As the timestep increases, the overlapping area across different domain distributions gradually expands, eliminating domain-specific attributes. Based on a theoretical proposition (monotonic relationship between attribute loss and timestep), the degradation operation constructs a continuous bridge from the source/target domain to a shared domain.

2. **Semantic Shift Compensation**: A trainable diffusion encoder $g'$, conditioned on a time embedding module, is introduced to extract semantic shift information from the degraded image $x_t$:
    $\hat{z}_{(t,i)} = z'_{(t,i)} (MLP_s^i \circ \text{Embed}(t) + 1) + MLP_b^i \circ \text{Embed}(t)$
   Features are fused via residual connections $g + g'$ at multiple levels, supervised by a reconstruction loss $\mathcal{L}^R = \|f_\theta(x_t, t) - \epsilon\|_2^2$. The design motivation is that time embeddings enable the network to precisely disentangle the semantic loss corresponding to different degradation levels, thereby enabling targeted compensation.

3. **Degraded Image Consistency (DIC) Loss**:
    $\mathcal{L}^D = \sum_{i}^{N_S} \mathcal{L}_{ce}(\bar{f}_\theta(x_{i,t}^S, t), y_i^S) + \sum_{i}^{N_T} \mathcal{L}_{ce}(\bar{f}_\theta(x_{i,t}^T, t), p_i^T, q^T)$
   where $\bar{f}_\theta = h \circ (g + g')$, enforcing prediction consistency between degraded and original images.

### Loss & Training

The total training loss is a weighted sum of four terms:
$$\mathcal{L} = \mathcal{L}^S + \mathcal{L}^T + \lambda_D \mathcal{L}^D + \lambda_R \mathcal{L}^R$$

- $\mathcal{L}^S$: source domain supervised loss
- $\mathcal{L}^T$: target domain pseudo-label self-training loss
- $\lambda_D = 0.5$; $\lambda_R$ is adjusted per architecture (DAFormer: 5, DeepLabV2: 1)
- Noise schedule: $T=100$, sigmoid schedule
- At inference, $g'$ and $h'$ are fully removed, incurring zero additional overhead

## Key Experimental Results

### Main Results

**Consistent Gains Across Methods, Architectures, and Benchmarks (mIoU)**

| Method | GTA→CS (CNN) | GTA→CS (Trans) | SYN→CS (CNN) | SYN→CS (Trans) | CS→ACDC (Trans) |
|--------|-------------|----------------|-------------|----------------|-----------------|
| DAFormer | 56.0 | 68.3 | 54.7 | 60.9 | 55.4 |
| +DiDA | **58.3** (+2.3) | **70.3** (+2.0) | **57.6** (+2.9) | **63.1** (+2.2) | **59.1** (+3.7) |
| HRDA | 63.0 | 73.8 | 61.2 | 65.8 | 68.0 |
| +DiDA | **64.3** (+1.3) | **75.4** (+1.6) | **62.6** (+1.4) | **67.8** (+2.0) | **70.7** (+2.7) |
| MIC | 64.2 | 75.5 | 62.4 | 67.3 | 69.8 |
| +DiDA | **65.0** (+0.8) | **76.8** (+1.3) | **63.5** (+1.1) | **68.6** (+1.3) | **72.1** (+2.3) |

### Ablation Study

**GTA→CS (Transformer), based on DAFormer**

| $\mathcal{L}^D$ | $\mathcal{L}^R$ | $g_{time}$ | $g'$ | $h'$ | mIoU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| - | - | - | - | - | 68.3 |
| ✓ | - | - | - | - | 66.5 |
| ✓ | - | ✓ | - | - | 69.5 |
| ✓ | ✓ | ✓ | - | - | 69.4 |
| ✓ | ✓ | ✓ | - | ✓ | 69.9 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **70.3** |

### Key Findings

- **Plug-and-play effectiveness**: DiDA consistently improves performance across all 3 UDA methods × 2 architectures × 5 settings
- **Largest gains in weather adaptation**: Improvements reach +3.7 mIoU on CS→ACDC, indicating that degradation bridging is particularly effective when domain gaps are large
- **Semantic shift compensation is critical**: Applying the DIC loss without time embeddings actually decreases performance by 1.8 mIoU (66.5 vs. 68.3); introducing time embeddings recovers and surpasses the baseline
- **Strong extensibility**: The framework is compatible with arbitrary degradation operations such as blur and inpainting, all yielding improvements

## Highlights & Insights

- **Elegant theoretical motivation**: The paper formalizes the intuition that "degradation = domain bridging" by grounding it in a theoretical proposition about attribute loss in diffusion models
- **Zero inference overhead**: The degradation encoder and reconstruction head are used only during training and incur no deployment cost
- **Strong generality**: Compatible with both CNN and Transformer architectures, multiple UDA baselines, and various degradation operations
- The domain bridging perspective offers a new direction compared to conventional adversarial training and style transfer approaches

## Limitations & Future Work

- The choice of degradation level $T=100$ and noise schedule is empirically determined; the theoretically optimal degradation strategy remains unclear
- The diffusion encoder $g'$ shares the same architecture as the backbone encoder $g$, doubling the parameter count during training (though removed at inference)
- Gains over the strong MIC baseline are relatively modest (+0.8–1.3), possibly approaching a performance ceiling
- The combination with non-self-training UDA methods (e.g., adversarial training) remains unexplored

## Related Work & Insights

- **Relationship with MIC/HRDA**: DiDA serves as a plug-and-play complement rather than a replacement, and is orthogonal to consistency regularization approaches
- **Diffusion models in segmentation**: Unlike methods that leverage diffusion for data generation or internal feature extraction, DiDA directly integrates the diffusion strategy into the UDA training pipeline
- **Broader inspiration**: The idea of degradation as domain bridging is generalizable to other cross-domain tasks such as detection and classification

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The degradation-as-domain-bridging perspective is novel, with a natural connection to diffusion theory)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple methods, architectures, benchmarks, and detailed ablations)
- Writing Quality: ⭐⭐⭐⭐ (Motivation is clearly articulated; theory and practice are well integrated)
- Value: ⭐⭐⭐⭐⭐ (A general plug-and-play UDA enhancement strategy with high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](../../ICCV2025/segmentation/exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [\[NeurIPS 2025\] Exploring Structural Degradation in Dense Representations for Self-supervised Learning](exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)
- [\[ICCV 2025\] Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation](../../ICCV2025/segmentation/exploring_probabilistic_modeling_beyond_domain_generalization_for_semantic_segme.md)

</div>

<!-- RELATED:END -->
