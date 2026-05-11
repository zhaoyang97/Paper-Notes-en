---
title: >-
  [Paper Note] Unbiased Dynamic Multimodal Fusion
description: >-
  [CVPR 2026][Multimodal VLM][Dynamic multimodal fusion] UDML proposes an unbiased dynamic multimodal learning framework comprising two core components: a noise-aware uncertainty estimator (which injects controllable noise and predicts its intensity to achieve accurate modality quality assessment under both low-noise and high-noise conditions) and a modality dependency calculator (which quantifies the model's inherent dependency bias toward each modality via Dropout and incorporates it into the weighting mechanism). The framework addresses the dual suppression problem in existing methods and consistently improves performance across multiple multimodal benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Dynamic multimodal fusion
  - uncertainty estimation
  - modality-dependency bias
  - noise-awareness
  - dual suppression
date: 2026-05-08
content_hash: a0df6a4b4b5c6f0a
---

# Unbiased Dynamic Multimodal Fusion

**Conference**: CVPR 2026
**arXiv**: [2603.19681](https://arxiv.org/abs/2603.19681)
**Code**: [https://github.com/shicaiwei123/UDML](https://github.com/shicaiwei123/UDML)
**Area**: Multimodal VLM / Multimodal Fusion
**Keywords**: Dynamic multimodal fusion, uncertainty estimation, modality-dependency bias, noise-awareness, dual suppression

## TL;DR

UDML proposes an unbiased dynamic multimodal learning framework comprising two core components: a noise-aware uncertainty estimator (which injects controllable noise and predicts its intensity to achieve accurate modality quality assessment under both low-noise and high-noise conditions) and a modality dependency calculator (which quantifies the model's inherent dependency bias toward each modality via Dropout and incorporates it into the weighting mechanism). The framework addresses the dual suppression problem in existing methods and consistently improves performance across multiple multimodal benchmarks.

## Background & Motivation

1. **Background**: Dynamic multimodal learning adaptively adjusts the contribution weights of each modality based on input data quality, with two predominant paradigms: prior-based methods and uncertainty-based methods.
2. **Limitations of Prior Work**: (1) *Uncertainty estimation bias*: existing empirical metrics (e.g., energy scores, probabilistic embeddings) are insensitive under low noise (failing to detect mild degradation) and still assign non-negligible weights to severely corrupted modalities under high noise; (2) *Dual suppression effect*: existing methods assume equal initial contributions across modalities, overlooking the modality-dependency bias induced during model optimization — modalities that are harder to learn are suppressed first by the optimization bias and then penalized again by high uncertainty.
3. **Key Challenge**: Dual suppression causes dynamic fusion to underperform static fusion, contradicting the very motivation for dynamic fusion.
4. **Goal**: Design an uncertainty estimator that is accurate across all noise levels, while simultaneously quantifying and compensating for modality-dependency bias.
5. **Key Insight**: Actively inject known noise to establish a clear correspondence between feature corruption and noise intensity; quantify inherent dependency via modality Dropout.
6. **Core Idea**: A dual strategy combining noise-aware estimation and bias compensation.

## Method

### Overall Architecture

UDML is an architecture-agnostic general framework consisting of two core components: (1) a noise-aware uncertainty estimator that injects controllable noise and predicts its intensity from the resulting features; and (2) a modality dependency calculator that quantifies the model's reliance on each modality via Dropout. Both components jointly determine the dynamic fusion weights.

### Key Designs

1. **Noise-Aware Uncertainty Estimator**:

   - **Function**: Accurately measures modality quality across all noise levels, from clean input to severe corruption.
   - **Mechanism**: During training, controllable noise of known intensity is injected into the modality data, and the estimator predicts the noise intensity from the encoded features. A probabilistic representation technique is introduced: each modality is mapped to a distribution (mean encodes semantic information; variance reflects noise characteristics), and the estimator infers noise intensity from the variance. This establishes a direct supervisory signal between feature corruption and noise level.
   - **Design Motivation**: Empirical metrics (energy scores, probabilistic embeddings) lack direct supervision over noise; the noise-aware estimator builds accurate correspondences through an explicit noise prediction task.

2. **Modality Dependency Calculator**:

   - **Function**: Quantifies and compensates for the inherent dependency bias of the multimodal network toward each modality.
   - **Mechanism**: Modality Dropout is used to quantify the model's dependency on each modality $\alpha^m$, which is then incorporated into the weight computation: $w_i^{m_1} = g(\frac{1}{s(z_i^{m_1}) \cdot \alpha^{m_1}})$. Modalities with high dependency are not excessively penalized by uncertainty, and harder-to-learn modalities with low dependency are protected from dual suppression.
   - **Design Motivation**: Eliminate the dual suppression of harder modalities caused by the compound effect of optimization bias and high uncertainty.

3. **Progressive Optimization Strategy**:

   - **Function**: Jointly learns multimodal representations, noise estimation, and the main task within the standard training pipeline.
   - **Mechanism**: Progressive training that first stabilizes multimodal representations before gradually introducing noise-aware estimation and dependency compensation.
   - **Design Motivation**: Avoid interference among multiple learning objectives.

### Loss & Training

Total loss = main task loss (classification/detection, etc.) + noise prediction loss (MSE) + KL divergence regularization (probabilistic representation).

## Key Experimental Results

### Main Results

| Dataset | Task | Static Fusion | Dynamic (PE) | UDML | Gain vs. Dynamic |
|---------|------|--------------|-------------|------|-----------------|
| CREMA-D | Audio-visual classification | 67.2 | 65.8 | 71.5 | +5.7 |
| Kinetics-Sound | Audio-visual classification | 64.1 | 63.5 | 66.8 | +3.3 |
| NYU Depth v2 | RGB-D segmentation | 51.2 | 50.8 | 53.1 | +2.3 |

Note: On CREMA-D, PE-based dynamic fusion (65.8) underperforms static fusion (67.2), empirically validating the dual suppression problem. UDML substantially resolves this issue.

### Ablation Study

| Configuration | CREMA-D Acc | Note |
|---------------|------------|------|
| Static fusion baseline | 67.2 | No dynamic weighting |
| + Noise-aware estimator | 69.8 | Contribution of accurate estimation |
| + Modality dependency calculator | 71.5 | Dual suppression eliminated |
| w/o probabilistic representation | 70.1 | Probabilistic representation aids generalization |

### Key Findings

- The noise-aware estimator responds monotonically across all noise levels, whereas PE fails at $\sigma < 4$ and $\sigma > 10$.
- The modality dependency calculator contributes approximately 1.7%, confirming that dual suppression is a significant bottleneck in existing methods.
- UDML is architecture-agnostic and yields consistent gains across multiple fusion paradigms including Concat, Attention, and Gating.
- Performance advantages are more pronounced under high-noise conditions, demonstrating robustness.

## Highlights & Insights

- **Discovery of dual suppression**: This work is the first to clearly identify the root cause of "dynamic fusion underperforming static fusion" as dual suppression, rather than an inherent flaw in the dynamic fusion paradigm itself.
- **Noise injection + prediction estimation paradigm**: More principled than empirical metrics, establishing a direct causal relationship between noise level and uncertainty.
- **Architecture-agnostic design**: All components operate solely on modality representations and can be plugged into arbitrary multimodal models.

## Limitations & Future Work

- The controllable noise injection assumes that the noise type is known; in practice, degradation may be of unknown types.
- The dependency measure computed via modality Dropout is a global statistic rather than a per-sample quantity.
- Validation is currently limited to two-modality scenarios; extensibility to three or more modalities remains to be verified.
- Future work could incorporate more fine-grained noise modeling, such as noise type classification.

## Related Work & Insights

- **vs. Probabilistic Embedding (PE)**: PE empirically estimates uncertainty via variance, whereas UDML explicitly learns the correspondence through a noise prediction task.
- **vs. OGM-GE / Greedy**: These methods address optimization imbalance via gradient modulation but do not handle dependency bias at inference time.
- **vs. TMC**: TMC models uncertainty with Dirichlet distributions but likewise assumes equal modality contributions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The analysis of dual suppression is insightful, and the noise-aware estimator design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple tasks and datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is clear and visualizations are intuitive.
- **Value**: ⭐⭐⭐⭐ Provides practical guidance for dynamic multimodal fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment](the_more_the_merrier_contrastive_fusion_for_higher-order_multimodal_alignment.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)
- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_vision-language_models.md)

</div>

<!-- RELATED:END -->
