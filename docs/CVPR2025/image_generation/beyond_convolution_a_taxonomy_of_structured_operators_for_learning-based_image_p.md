---
title: >-
  [Paper Note] Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing
description: >-
  [CVPR 2025][Image Generation][Convolution alternative operators] This paper systematically organizes alternative/extended operators of convolution in learning-based image processing into five major families (decomposition-based, adaptive weighted, basis-adaptive, integral/kernel-based, and attention-based), and provides a comparative analysis across multiple dimensions including linearity, locality, equivariance, computational cost, and task suitability.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Convolution alternative operators"
  - "SVD denoising"
  - "adaptive-weight convolution"
  - "F-transform"
  - "attention mechanisms"
  - "taxonomy"
date: 2026-05-08
content_hash: 3349e18c02f3a722
---

# Beyond Convolution: A Taxonomy of Structured Operators for Learning-Based Image Processing

**Conference**: CVPR 2025  
**arXiv**: [2603.12067](https://arxiv.org/abs/2603.12067)  
**Code**: To be confirmed  
**Area**: Image Processing / Survey  
**Keywords**: Convolution alternative operators, SVD denoising, adaptive-weight convolution, F-transform, attention mechanisms, taxonomy

## TL;DR

This paper systematically organizes alternative/extended operators of convolution in learning-based image processing into five major families (decomposition-based, adaptive weighted, basis-adaptive, integral/kernel-based, and attention-based), and provides a comparative analysis across multiple dimensions including linearity, locality, equivariance, computational cost, and task suitability.

## Background & Motivation

**Background**: The convolution operator is the cornerstone of CNNs, but it is fundamentally a fixed, linear, local averaging operation.

**Limitations of Prior Work**: Four structural limitations of convolution: linearity (does not model non-linear interactions), translation equivariance (insensitive to position-dependent statistics), fixed locality (restricted receptive field), and uniform weighting (does not distinguish between edges and noise).

**Key Challenge**: A large number of works that replace convolution are scattered across multiple communities including signal processing, numerical linear algebra, fuzzy mathematics, and deep learning, lacking a unified perspective.

**Goal**: Provide a systematic taxonomic framework to help practitioners understand when, why, and which operators to use to replace convolution.

**Key Insight**: Taking the four structural properties of standard convolution as reference points.

**Core Idea**: Operator selection is not merely an implementation detail, but a fundamental modeling decision that encodes prior knowledge of signals and tasks.

## Method

### Overall Architecture

Taking standard discrete convolution as the reference point, alternative operators are classified into five major families based on relaxed structural properties. For each family, a formal definition, structural property analysis, and task suitability evaluation are provided.

### Key Designs

1. **Decomposition-Based Operators**

    - **Function**: Use SVD/HOSVD to perform matrix/tensor decomposition on local patches, and use thresholding to separate structure from noise
    - **Mechanism**: Perform SVD on the patch, use a network to predict the optimal threshold to truncate singular values
    - **Relaxed Properties**: Uniform weighting (changed to spectrum-based differentiation) + introduction of non-linearity
    - **Suitability**: Image denoising, compression; works well in conjunction with block-matching (BM3D/WNNM)

2. **Adaptive Weighted Operators**

    - **Function**: Modulate convolution kernel weights element-wise using a density function
    - **Mechanism**: Outer-loop DIRECT-L global derivative-free optimization to find the optimal density function + inner-loop SGD to train kernel weights
    - **Representatives**: Density function convolution, Dynamic Conv (multi-kernel aggregation), Deformable Conv (learning sampling offsets)
    - **Performance**: Denoising PSNR +6-7%, Classification Acc +7%, GPU overhead only ~7%

3. **Basis-Adaptive Operators**

    - **Function**: Treat analysis/synthesis bases as learnable objects
    - **Mechanism**: Project using F-transform onto optimizable fuzzy partition functions
    - **Extensions**: Learnable wavelet transforms, shearlets, K-SVD dictionary learning
    - **Suitability**: Scenarios with strong physical priors, such as medical imaging

4. **Integral and Kernel Operators**

    - **Function**: Generalize convolutions to position-dependent and non-linear kernels
    - **Representatives**: FNO (Fourier Neural Operator), Non-Local Means
    - **Relaxed Properties**: Locality + translation equivariance

5. **Attention-Based Operators**

    - **Function**: Completely abandon the locality assumption, allowing global dependencies
    - **Relaxed Properties**: All four properties
    - **Cost**: $O(N^2 d)$ complexity

### Loss & Training

As a survey paper, this work does not involve a single training strategy. Density function optimization adopts a bi-level optimization (outer global optimization + inner SGD).

## Key Experimental Results

### Main Results

| Operator Family | Relaxed Properties | Typical Tasks | Performance Highlights | Computational Overhead |
|---------|---------|---------|---------|---------|
| Decomposition-Based | Uniform weighting + Linearity | Denoising | WNNM/BM3D level | High (SVD) |
| Adaptive Weighted | Uniform weighting | Denoising / Classification | PSNR +6-7%, Acc +7% | +7% GPU |
| Basis-Adaptive | Equivariance + Uniform weighting | Medical Imaging | Physical prior modeling | Medium |
| Integral / Kernel | Locality + Equivariance | PDE solving | Global dependency | High |
| Attention-Based | All four | General | ViT level | $O(N^2)$ |

### Ablation Study

As a survey paper, the key comparisons of various operators are drawn from independent evaluations in the cited literature.

### Key Findings

- Decomposition-based operators are naturally effective in denoising; SVD provides a physical explanation for signal/noise separation.
- Density function convolution has the highest cost-performance ratio: no increase in parameter count, +7% GPU overhead.
- F-transform is valuable in medical imaging, where fuzzy partition functions can encode domain priors.
- Different tasks require different operators; there is no one-size-fits-all replacement.

## Highlights & Insights

- Unifies works scattered across multiple communities under a systematic taxonomic framework.
- Emphasizes the perspective that "operator selection is a modeling decision".
- Provides formal definitions, structural property analyses, and task suitability discussions for each family.

## Limitations & Future Work

- Being a pure survey, it lacks original experimental validation.
- Does not cover the latest state-space models such as Mamba/SSMs.
- Lacks a practical workflow guide for operator selection during deployment.

## Related Work & Insights

- The bi-level structure of density function optimization is an interesting paradigm.
- The idea of predicting SVD thresholds using a network can be generalized to other decomposition methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The survey framework holds systematic value
- **Experimental Thoroughness**: ⭐⭐⭐ Pure survey without original experiments
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure and rigorous formal definitions
- **Value**: ⭐⭐⭐⭐⭐ Bridges cognitive gaps across different communities

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales](data-free_group-wise_fully_quantized_winograd_convolution_via_learnable_scales.md)
- [\[CVPR 2025\] AutoPresent: Designing Structured Visuals from Scratch](autopresent_designing_structured_visuals_from_scratch.md)
- [\[ICLR 2026\] Structured Flow Autoencoders: Learning Structured Probabilistic Representations with Flow Matching](../../ICLR2026/image_generation/structured_flow_autoencoders_learning_structured_probabilistic_representations_w.md)
- [\[NeurIPS 2025\] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models](../../NeurIPS2025/image_generation/fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)
- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)

</div>

<!-- RELATED:END -->
