---
title: >-
  [Paper Note] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation
description: >-
  [CVPR 2025][Model Compression][Post-Training Quantization] Proposes FIMA-Q, which replaces the traditional diagonal approximation with a diagonal-plus-low-rank (DPLR) Fisher Information Matrix approximation to capture the impact of quantization errors on the output distribution more accurately, significantly outperforming existing methods in ultra-low-bit (3-bit) ViT quantization (ViT-B 77.63% vs QDrop 74.75%).
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Post-Training Quantization"
  - "Vision Transformer"
  - "Fisher Information Matrix"
  - "Diagonal Plus Low-Rank Approximation"
  - "Low-Bit Quantization"
date: 2026-05-08
content_hash: 5d3d01327fcda2b9
---

# FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation

**Conference**: CVPR 2025  
**arXiv**: [2506.11543](https://arxiv.org/abs/2506.11543)  
**Code**: [https://github.com/ShiheWang/FIMA-Q](https://github.com/ShiheWang/FIMA-Q)  
**Area**: Model Compression / Quantization  
**Keywords**: Post-Training Quantization, Vision Transformer, Fisher Information Matrix, Diagonal Plus Low-Rank Approximation, Low-Bit Quantization

## TL;DR

Proposes FIMA-Q, which replaces the traditional diagonal approximation with a diagonal-plus-low-rank (DPLR) Fisher Information Matrix approximation to capture the impact of quantization errors on the output distribution more accurately, significantly outperforming existing methods in ultra-low-bit (3-bit) ViT quantization (ViT-B 77.63% vs QDrop 74.75%).

## Background & Motivation

**Background**: Post-Training Quantization (PTQ) is a practical approach for compressing Vision Transformers. The dominant method, BRECQ, uses a diagonal approximation of the Fisher Information Matrix (FIM) to measure the impact of quantization error, performing block-by-block reconstruction to minimize the KL divergence.

**Limitations of Prior Work**: The diagonal FIM approximation in BRECQ ignores correlations between output dimensions. Quantization errors across different output channels can amplify or cancel each other out, which the diagonal approximation fails to capture. At ultra-low bitwidths (e.g., 3-bit), the large quantization errors amplify this approximation error, leading to severe performance degradation.

**Key Challenge**: The computational cost of exact FIM calculation ($O(d^2)$) is prohibitive, while the diagonal approximation ($O(d)$) discards critical cross-dimensional information.

**Key Insight**: Approximating the FIM using a diagonal-plus-low-rank (DPLR) decomposition $F \approx D + UU^\top$, where the diagonal terms preserve the independent importance of each dimension, while the low-rank terms capture cross-dimensional correlations.

**Core Idea**: Replacing the pure diagonal approximation of FIM with the DPLR approximation to capture cross-dimensional correlations in low-bit ViT quantization, thereby significantly improving accuracy.

## Method

### Key Designs

1. **Correcting the Hessian Approximation Error in BRECQ**:

    - Function: Identifies and corrects theoretical flaws in existing methods
    - Mechanism: Reveals that the Hessian diagonal matrix in BRECQ is actually linearly proportional to the KL gradient (rather than squared), indicating that its diagonal approximation is mathematically inconsistent. FIMA-Q reverts to the correct definition of FIM for re-derivation.
    - Design Motivation: Theoretical correction leads to practical improvements—the corrected loss function more accurately reflects the real impact of quantization errors.

2. **Diagonal Plus Low-Rank (DPLR) Approximation**:

    - Function: Captures cross-dimensional correlations with $O(d \cdot r)$ complexity
    - Mechanism: The FIM is decomposed into $F \approx D + UU^\top$, where $D$ is the diagonal term and $U \in \mathbb{R}^{d \times r}$ is the low-rank factor. The loss function is defined as $\mathcal{L}_{DPLR} = \alpha \mathcal{L}_{rank-k} + (1-\alpha) \mathcal{L}_{diag}$, where the low-rank term is extracted from the outer product of quantization errors.
    - Design Motivation: Pure diagonal approximation loses correlations, while pure low-rank approximation loses individual dimension importance; DPLR combines the strengths of both.

### Loss & Training

$\mathcal{L}_{DPLR} = \alpha \mathcal{L}_{rank-k} + (1-\alpha) \mathcal{L}_{diag}$, with a progressive growth strategy for rank $r$. It uses a uniform quantizer, 1024 calibration images, and the Adam optimizer for block-by-block reconstruction.

## Key Experimental Results

### Main Results

ImageNet Top-1 Accuracy (W3A3 = Weight 3-bit / Activation 3-bit):

| Model | FIMA-Q | QDrop | RepQ-ViT |
|------|--------|-------|----------|
| ViT-B | **77.63** | 74.75 | 70.10 |
| DeiT-B | **76.54** | 72.97 | 75.11 |
| Swin-B | **78.82** | 76.57 | 72.36 |

### Key Findings
- **Most Significant Gains at 3-bit**: Lower bitwidths lead to larger quantization errors, making cross-dimensional correlation modeling more critical.
- **Generality**: Consistent improvements across all three architectures: ViT, DeiT, and Swin.

## Highlights & Insights
- **Theoretical Correction Powers Practical Breakthroughs**: Revealing the mathematical flaw in BRECQ is not just of academic interest; it directly yields a 2-3% accuracy improvement.
- **DPLR is Generalizable**: This approximation framework can be applied to any scenario that requires FIM (e.g., Elastic Weight Consolidation (EWC) regularization in continual learning).

## Limitations & Future Work
- Low-rank calculation increases the computational overhead during the reconstruction phase.
- Performance is dependent on the quality of calibration data.
- Only validated on vision tasks; application to NLP and speech domains remains to be explored.

## Rating
- Novelty: ⭐⭐⭐⭐ Strong theoretical contribution (correcting prior errors + DPLR approximation)
- Experimental Thoroughness: ⭐⭐⭐⭐ Three ViT architectures with multiple bitwidth configurations
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations
- Value: ⭐⭐⭐⭐ New SOTA for 3-bit ViT quantization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients for Zero-Shot NAS on Vision Transformers](l_swag_zero_shot_nas_vision_transformers.md)
- [\[ECCV 2024\] AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](../../ECCV2024/model_compression/adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)

</div>

<!-- RELATED:END -->
