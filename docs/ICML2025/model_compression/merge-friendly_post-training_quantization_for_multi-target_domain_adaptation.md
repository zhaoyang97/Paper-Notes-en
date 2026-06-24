---
title: >-
  [Paper Note] Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
description: >-
  [ICML 2025][Model Compression][post-training quantization] This paper presents the first systematic analysis of how discretization noise introduced by quantization degrades model merging performance. It proposes HDRQ (Hessian and Distance Regularizing Quantization), which uses Hessian regularization to flatten the loss landscape, distance regularization to align weights across quantized models, and noise-sampling rounding to resolve rounding ambiguity. This allows quantized m…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "post-training quantization"
  - "model merging"
  - "multi-target domain adaptation"
  - "error barrier"
  - "Hessian regularization"
date: 2026-05-08
content_hash: a5e5a5179e8fa91e
---

# Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

**Conference**: ICML 2025  
**Authors**: Juncheol Shin, Minsang Seok, Seonggon Kim, Eunhyeok Park  
**arXiv**: [2505.23651](https://arxiv.org/abs/2505.23651)  
**Code**: Not released  
**Area**: Model Compression, Domain Adaptation  
**Keywords**: post-training quantization, model merging, multi-target domain adaptation, error barrier, Hessian regularization  

## TL;DR

This paper presents the first systematic analysis of how discretization noise introduced by quantization degrades model merging performance. It proposes HDRQ (Hessian and Distance Regularizing Quantization), which uses Hessian regularization to flatten the loss landscape, distance regularization to align weights across quantized models, and noise-sampling rounding to resolve rounding ambiguity. This allows quantized models to achieve merging performance close to full-precision equivalents in multi-target domain adaptation.

## Background & Motivation

**Background**: Model merging consolidates multiple models fine-tuned on different target domains into a single unified model via simple weight averaging, enabling training-free multi-target domain adaptation (MTDA). Concurrently, quantization is essential for edge deployment, reducing memory footprint and computational overhead by lowering weight precision.

**Limitations of Prior Work**: While quantization and model merging have both been widely studied individually, their interaction has been completely overlooked. The discretization effect introduced by quantization disrupts weight continuity. Consequently, the weight interpolation path between multiple quantized models traverses high-altitude regions of the loss landscape (high error barriers), leading to severe degradation in merged performance—sometimes dropping below that of individual quantized models.

**Key Challenge**: Traditional PTQ methods only optimize individual model reconstruction errors without considering the compatibility of the quantized models during merging. Quantization deviates model weights from their original positions and increases the weight distance between different quantized models. These dual factors lead to a sharp rise in the error barrier along the interpolation path.

**Key Insight**: Starting from error barrier theory, this work explicitly introduces quantization noise into the error barrier analytical framework. It derives two key factors that dictate merging quality: (1) the curvature of the loss landscape (Hessian sensitivity) and (2) the weight distance between the quantized models.

**Core Idea**: Integrate merging-oriented regularization constraints into the post-training quantization process, making quantized models "naturally" compatible for merging.

## Method

### Overall Architecture

HDRQ is a modified PTQ pipeline: it takes a pretrained source model and small amounts of calibration data from multiple target domains as inputs, and outputs multiple "merging-friendly" quantized models. The core modification lies in the optimization objective: adding a Hessian regularization term and a distance regularization term to the standard reconstruction loss, and replacing traditional deterministic rounding with noise-sampling rounding.

### Key Designs

1. **Hessian Regularization (Sensitivity Control)**:

    - **Function**: Flattens the loss landscape around the quantized models to mitigate the impact of weight perturbations on the output.
    - **Mechanism**: The height of the error barrier is proportional to the weight offset and the eigenvalues of the local Hessian matrix. By imposing a stronger regularization penalty on parameters with a larger Hessian trace, the quantizer is forced to maintain higher precision at highly sensitive locations. In practice, the Fisher Information Matrix is used to approximate the Hessian diagonal to reduce computational complexity.
    - **Design Motivation**: A flat loss landscape ensures that sharp loss spikes do not occur along the weight interpolation path, which is a prerequisite for successful merging.

2. **Distance Regularization (Weight Alignment)**:

    - **Function**: Constraints the weight distance between the quantized models and the pretrained source model.
    - **Mechanism**: It minimizes the L2 distance between each quantized model and the pretrained source model. Since all target domain models are fine-tuned from the same source model, maintaining proximity to the source model indirectly keeps the quantized models close to each other.
    - **Design Motivation**: Model merging is essentially linear interpolation in the weight space. A shorter interpolation path and closer endpoints reduce the probability of traversing high-loss regions.

3. **Noise-Sampling Rounding (Resolving Rounding Ambiguity)**:

    - **Function**: Resolves the instability of traditional round-to-nearest at decision boundaries.
    - **Mechanism**: Quantization noise is modeled as additive noise. By introducing controlled Gaussian noise to simulate quantization granularity, the regularization constraints can be optimized via backpropagation. This avoids approximation errors from the Straight-Through Estimator (STE) and exhibits higher stability under the low-data calibration regime of PTQ.
    - **Design Motivation**: Deterministic rounding is highly unstable near quantization boundaries; minute weight changes can flip the rounding direction, introducing unpredictable errors during merging.

### Loss & Training

The total optimization objective is a weighted sum of the reconstruction loss, Hessian regularization, and distance regularization terms: $\mathcal{L} = \mathcal{L}_{recon} + \lambda_H \cdot \mathcal{L}_{Hessian} + \lambda_D \cdot \mathcal{L}_{dist}$, where $\mathcal{L}_{recon}$ is the standard layer-wise output reconstruction loss, $\mathcal{L}_{Hessian}$ is the sensitivity penalty weighted by the Hessian diagonal, and $\mathcal{L}_{dist}$ represents the L2 distance from the quantized weights to the source model weights.

## Key Experimental Results

### Main Results

| Method | Bit-width | Single Model mIoU | 2-Domain Merged mIoU | 3-Domain Merged mIoU |
|------|------|-----------|-------------|-------------|
| Full Precision (Upper Bound) | 32-bit | 64.23 | 65.41 | 66.12 |
| Traditional PTQ (BRECQ) | 4-bit | 63.15 | 60.82 | 59.54 |
| **HDRQ (Ours)** | 4-bit | 63.58 | **65.03** | **65.87** |
| Improvement | - | +0.43 | **+4.21** | **+6.33** |

### Ablation Study

| Configuration | Max Barrier Height | Interpolation Loss Variance | Merged mIoU |
|------|-----------|-----------|----------|
| PTQ without Regularization | 8.34 | 2.81 | 60.82 |
| + Hessian Regularization | 5.21 | 1.45 | 63.17 |
| + Distance Regularization | 4.87 | 1.22 | 63.62 |
| + Both + Noise Rounding (Full HDRQ) | **3.12** | **0.93** | **65.03** |

### Key Findings

- Traditional PTQ results in a merged mIoU that is even lower than that of single models (60.82 < 63.15), indicating that quantization severely disrupts the merging capability. In contrast, HDRQ restores the merging gains almost completely (65.03 vs. 65.41 in full precision).
- The error barrier height decreases from 8.34 to 3.12 (-62.6%), directly validating the effectiveness of the theoretical analysis.
- The improvement in the 4-bit scenario (+4.21) is significantly larger than in the 8-bit scenario (approx. +1.9), demonstrating that HDRQ yields greater value when quantization is more aggressive.
- The computational overhead is approximately 1.5 times that of standard PTQ (since the Hessian can be approximated using Fisher Information), which is substantially lower than Quantization-Aware Training (QAT).

## Highlights & Insights

- Defines and addresses the "quantization-merging compatibility" problem for the first time, targeting a central bottleneck of co-existing quantization and multi-tasking in actual deployments.
- Theory-driven design: derives the specific formulations of the regularization terms from the error barrier analysis rather than adding empirical heuristics.
- A PTQ-level approach (requiring no full-dataset training) that can be seamlessly integrated into any existing PTQ pipeline, offering strong reproducibility and utility.
- Achieves an improvement of over 4 mIoU in 4-bit merging with only a 1.5x increase in computational overhead, which is highly significant for semantic segmentation tasks.

## Limitations & Future Work

- The accuracy of the Hessian diagonal approximation is unverified on ultra-large models (100B+ params), and its computational feasibility remains questionable.
- Distance regularization assumes that all target domain models originate from the same pretrained source model, making it unsuitable for heterogeneous initialization scenarios.
- The hyperparameters $\lambda_H$ and $\lambda_D$ must be manually tuned, lacking an adaptive adjustment strategy.
- The experiments are primarily conducted on semantic segmentation tasks, leaving NLP or generative model merging scenarios unexplored.
- The merging strategy is restricted to simple weight averaging; advanced merging methods such as Task Arithmetic or TIES-MERGING are not investigated.

## Related Work & Insights

- **PTQ Evolution**: From layer-wise reconstruction (Nagel 2020) to block-wise reconstruction (Li 2021), and subsequently to merging-aware reconstruction in this study, the optimization objectives of PTQ are becoming increasingly global.
- **Model Merging**: Li et al. 2024 demonstrated that homogeneously fine-tuned models can achieve MTDA via weight averaging but overlooked quantization; this work fills this critical research gap.
- **Insights**: Similar "downstream-task-aware quantization" concepts can be extended to safety and alignment constraints (e.g., preserving fairness or alignment characteristics in quantized models).

## Rating

| Metric | Score | Rationale |
|------|------|------|
| Novelty | ⭐⭐⭐⭐⭐ | First to identify and resolve the quantization-merging compatibility issue. |
| Technical Depth | ⭐⭐⭐⭐ | Solid theoretical analysis of error barriers; the formulation of regularization is theoretically grounded. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Thorough ablations on potential bit-widths and domain numbers, though limited in task diversity. |
| Writing Quality | ⭐⭐⭐⭐ | Clear formulation of theory and comprehensive motivation. |
| Utility | ⭐⭐⭐⭐⭐ | PTQ-level complexity combined with significant merging improvements, ready for direct deployment. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](../../CVPR2025/model_compression/q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[ICLR 2026\] Post-Training Quantization for Video Matting](../../ICLR2026/model_compression/post-training_quantization_for_video_matting.md)
- [\[ICLR 2026\] Training Dynamics Impact Post-Training Quantization Robustness](../../ICLR2026/model_compression/training_dynamics_impact_post-training_quantization_robustness.md)

</div>

<!-- RELATED:END -->
