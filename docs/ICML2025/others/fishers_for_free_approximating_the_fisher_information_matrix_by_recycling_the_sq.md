---
title: >-
  [Paper Note] Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator
description: >-
  [ICML 2025 Spotlight][Fisher Information Matrix] This paper systematically analyzes the theoretical connection between the squared gradient accumulator (Squisher) of the Adam optimizer and the diagonal of the Fisher Information Matrix. It demonstrates that Squisher can serve as a zero-cost approximation of the Fisher diagonal, performing comparably to Fisher across five major applications, including model merging, continual learning, and sparsification.
tags:
  - "ICML 2025 Spotlight"
  - "Fisher Information Matrix"
  - "Adam Optimizer"
  - "Squared Gradient Accumulator"
  - "Model Merging"
  - "Parameter Sensitivity"
date: 2026-05-08
content_hash: 2fc11cef9bafaea5
---

# Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2507.18807](https://arxiv.org/abs/2507.18807)  
**Code**: Yes  
**Area**: Other  
**Keywords**: Fisher Information Matrix, Adam Optimizer, Squared Gradient Accumulator, Model Merging, Parameter Sensitivity

## TL;DR
This paper systematically analyzes the theoretical connection between the squared gradient accumulator (Squisher) of the Adam optimizer and the diagonal of the Fisher Information Matrix. It demonstrates that Squisher can serve as a zero-cost approximation of the Fisher diagonal, performing comparably to Fisher across five major applications, including model merging, continual learning, and sparsification.

## Background & Motivation
**Background**: The diagonal of the Fisher Information Matrix is widely used to measure parameter sensitivity, but its computation requires squaring and summing per-sample gradients, which is computationally expensive.

**Limitations of Prior Work**: Computing the Fisher diagonal requires extra computation, access to training data, and is non-trivial to implement efficiently, which hinders the widespread adoption of Fisher-based methods (such as Fisher Merging).

**Key Challenge**: The Adam optimizer already maintains an exponential moving average of squared gradients during training. This resembles the Fisher diagonal on the surface but differs in details.

**Key Insight**: Rigorous analysis of the relationship between the two, identifying differences, and quantifying their impact.

**Core Idea**: The squared gradient accumulator $v^{(t)}$ in Adam can be directly "recycled" as an approximation of the Fisher diagonal (termed Squisher) with zero additional computational cost.

## Method

### Overall Architecture
Theoretical analysis of the exact relationship between the Fisher diagonal and the squared gradient accumulator $\rightarrow$ Experimental validation of whether Squisher can replace Fisher across six application scenarios.

### Key Designs

1. **Standard vs Joint Fisher**:

    - Standard Fisher: $\text{diag}(F_{\text{std}}^{\text{emp}}) = \sum_n g_n^2$ (square then sum)
    - Joint Fisher: $\text{diag}(F_{\text{joint}}^{\text{emp}}) = (\sum_n g_n)^2$ (sum then square)
    - Squisher is closer to Joint Fisher rather than Standard Fisher.

2. **Three sources of approximation in Squisher**:

    - Sum-then-square vs square-then-sum ($\sum g_n^2$ vs $(\sum g_n)^2$), introducing batch-level differences.
    - Temporal weighting introduced by EMA, which non-uniformly weights historical gradients.
    - The $1/N$ scaling factor when using mean reduction loss.
    - Key observation: Many applications are invariant to the scaling of Fisher (e.g., Fisher Merging), so the scaling difference has no impact.

3. **Experimental validation**:

    - Covers five major applications: model merging, continual learning, sparse training, task similarity, model sparsification, and sparse fine-tuning.
    - Each application is compared against using the true Fisher and a no-Fisher baseline.

## Key Experimental Results

### Main Results

| Application | Fisher | Squisher | No-Fisher Baseline |
|------|--------|----------|-------------|
| Fisher Merging (8 Tasks) | 73.2% | 73.2% | 70.8% (Mean Merging) |
| Continual Learning (EWC) | 82.1% | 81.8% | 79.5% (No Regularization) |
| Sparse Training | 88.3% | 88.1% | 86.7% (Random) |

### Ablation Study

| Configuration | Model Merging Performance | Description |
|------|-------------|------|
| Standard Fisher | 73.2% | Standard computation |
| Squisher (Direct recycling) | 73.2% | Zero extra cost |
| Uniform weight merging | 70.8% | Completely ignoring parameter importance |

### Key Findings
- Squisher performs comparably to Fisher in all six settings, and consistently outperforms the no-Fisher baselines.
- The formula for Fisher Merging is scale-invariant to Fisher, so the scaling differences of Squisher have absolutely no impact.
- The EMA coefficient $\beta_2=0.999$ allows Squisher to contain approximately 10,000 steps of gradient history.

## Highlights & Insights
- Highly practical: Reduces the computational cost of Fisher-based methods to zero (by directly reading from the Adam state).
- Rigorous theoretical analysis: Establishes a precise mathematical connection between Squisher and Fisher via the Joint Fisher.
- Lin et al. (2024) proved that Joint Fisher = Standard Fisher, providing a theoretical foundation for this paper.
- Potentially promotes the wider adoption of methods like Fisher Merging (e.g., direct integration into mergekit).

## Limitations & Future Work
- Squisher is an approximation of the empirical Fisher, not the standard Fisher.
- The EMA introduces temporal bias, which may affect scenarios where parameters change significantly during training.
- Only applicable to Adam-family optimizers.

## Rating
- Novelty: ⭐⭐⭐⭐ Although the connection is intuitively natural, the rigorous analysis is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across six major applications.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theory and elegant diagrams.
- Value: ⭐⭐⭐⭐⭐ Extremely strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information](../../ECCV2024/others/fisherrf_active_view_selection_and_mapping_with_radiance_fields_using_fisher_inf.md)
- [\[ICML 2025\] Residual Matrix Transformers: Scaling the Size of the Residual Stream](residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [\[ICML 2025\] Gradient Aligned Regression via Pairwise Losses](gradient_aligned_regression_via_pairwise_losses.md)
- [\[CVPR 2025\] Event Ellipsometer: Event-based Mueller-Matrix Video Imaging](../../CVPR2025/others/event_ellipsometer_event-based_mueller-matrix_video_imaging.md)
- [\[ACL 2025\] The Harmonic Structure of Information Contours](../../ACL2025/others/the_harmonic_structure_of_information_contours.md)

</div>

<!-- RELATED:END -->
