---
title: >-
  [Paper Note] BiGain: Unified Token Compression for Joint Generation and Classification
description: >-
  [CVPR 2026][Image Generation][Token Compression] BiGain proposes a frequency-aware token compression framework. Through Laplacian-gated token merging (preserving high-frequency details) and interpolate-extrapolate KV downsampling (preserving query precision), it is the first to simultaneously optimize generation quality and classification accuracy in diffusion model inference acceleration.
tags:
  - CVPR 2026
  - Image Generation
  - Token Compression
  - Frequency-Aware
  - Diffusion Model Classification
  - Laplacian Filtering
  - KV Downsampling
date: 2026-05-08
content_hash: 9ee172bd468ae35c
---

# BiGain: Unified Token Compression for Joint Generation and Classification

**Conference**: CVPR 2026  
**arXiv**: [2603.12240](https://arxiv.org/abs/2603.12240)  
**Code**: [https://github.com/Greenoso/BiGain](https://github.com/Greenoso/BiGain)  
**Area**: Diffusion Models / Inference Acceleration  
**Keywords**: Token Compression, Frequency-Aware, Diffusion Model Classification, Laplacian Filtering, KV Downsampling

## TL;DR

BiGain proposes a frequency-aware token compression framework. Through Laplacian-gated token merging (preserving high-frequency details) and interpolate-extrapolate KV downsampling (preserving query precision), it is the first to simultaneously optimize generation quality and classification accuracy in diffusion model inference acceleration.

## Background & Motivation

**Background**: Inference acceleration for diffusion models primarily relies on training-free methods like token merging (ToMe) and token downsampling (ToDo), with evaluation metrics focused almost exclusively on generation quality (FID).

**Limitations of Prior Work**: The same diffusion models are increasingly reused for classification (via per-class denoising likelihood scoring), but existing compression operations that have minimal impact on generation severely damage classification. Experiments show that token merging hardly affects FID but causes classification accuracy to plummet—because the "redundant" tokens prioritized for removal are precisely the edge/texture details that classification depends on.

**Key Challenge**: Generation tasks rely on low/medium-frequency semantics (global structure), while classification tasks rely on high-frequency details (edges/textures). Traditional compression only optimizes the former while ignoring the latter.

**Goal**: Redefine token compression as a multi-objective optimization problem: simultaneously maintaining generative fidelity and discriminative utility.

**Key Insight**: Frequency separation—decoupling high-frequency details from low/medium-frequency content through frequency-aware representations to achieve "balanced spectral retention" compression.

**Core Idea**: Use a Laplacian filter to distinguish between high-frequency and low-frequency tokens, merging low-frequency tokens while preserving high-frequency ones, while maintaining full-resolution Queries during KV downsampling to preserve attention precision.

## Method

### Overall Architecture

BiGain consists of two training-free, plug-and-play operators that can be used individually or in combination. L-GTM guides merging decisions via frequency-aware gating during the token merging stage; IE-KVD balances the spectrum by controlling the KV downsampling method during attention calculation. Both are designed based on the "balanced spectral retention" principle and are applicable to DiT and U-Net architectures.

### Key Designs

1. **Laplacian-Gated Token Merging (L-GTM)**:

    - **Function**: Guide token merging with Laplacian frequency scores, preserving high-frequency tokens and merging low-frequency ones.
    - **Mechanism**: The hidden state $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ is convolved with a Laplacian kernel $\mathbf{L} = [[0,1,0],[1,-4,1],[0,1,0]]$ to obtain frequency scores $\mathbf{F} = \text{Reduce}_c(|\mathbf{X} * \mathbf{L}|)$. For each grid, the token with the lowest frequency score acts as a destination (low-frequency anchor), while others are sources. The top $r\%$ of source-destination pairs are merged based on similarity.
    - **Design Motivation**: Standard ToMe does not distinguish frequency characteristics, easily merging away edge/texture tokens and harming classification. L-GTM quantifies "high-frequenciness" via Laplacian response; low-frequency (smooth region) tokens are merged, while high-frequency (edge/texture) tokens are preserved.

2. **Interpolate-Extrapolate KV-Downsampling (IE-KVD)**:

    - **Function**: Downsample Key/Value while maintaining full Query resolution, balancing the spectrum through controllable interpolation/extrapolation factors.
    - **Mechanism**: $\mathcal{D}_{\alpha,s}(\mathbf{Z})[i] = \alpha \cdot \mathbf{Z}[\text{nearest}(i)] + (1-\alpha) \cdot \frac{1}{|\mathcal{N}_s(i)|}\sum_j \mathbf{Z}[j]$, where $\alpha$ controls the balance between nearest (preserving high-frequency) and average (preserving low-frequency). Extrapolation occurs when $\alpha > 1$, magnifying high frequencies; interpolation occurs when $\alpha < 1$, smoothing high frequencies. Keeping Query at full resolution ensures attention precision.
    - **Design Motivation**: ToDo directly uses average pooling for KV downsampling, losing high-frequency information. Preserving full Q resolution keeps the receptive field of each output token unchanged, which is critical for per-token scoring in classification.

3. **Compatibility with Diffusion Classifier**:

    - **Function**: Ensure the compression method is compatible with diffusion-based classification decision rules.
    - **Mechanism**: Both operators are timestep-local and deterministic, not relying on cross-timestep caching. All classes receive the same $(t_s, \epsilon_s)$ and the same compression schedule, keeping the paired-difference estimator valid.
    - **Design Motivation**: Cache-based acceleration methods (e.g., cross-timestep feature reuse) are incompatible with diffusion classifiers because classification requires independent scoring for each class.

### Loss & Training

A training-free method requiring no training. It is used plug-and-play directly on pre-trained Stable Diffusion 2.0 and DiT-XL/2.

## Key Experimental Results

### Main Results (SD-2.0, Pets Dataset, under similar FLOPs reduction)

| Method | Acceleration Type | FLOPs Reduction | Classification Acc@1 | vs Baseline |
|------|---------|-----------|-----------|------------|
| Baseline (No Accel) | — | — | 81.03% | — |
| ToMe | Token Merging | 10% | 72.96% | ↓8.07 |
| SiTo | Token Merging | 7% | 68.84% | ↓12.19 |
| **BiGain_TM (Ours)** | Token Merging | 10% | **78.38%** | ↓2.65 |
| ToDo | Token Downsampling | 14.2% | 79.15% | ↓1.88 |
| **BiGain_TD (Ours)** | Token Downsampling | 14.2% | **79.90%** | ↓1.13 |

### Ablation Study (ImageNet-1K, SD-2.0, 70% Token Merging Rate)

| Configuration | Acc@1 (%) | FID | Note |
|------|-----------|-----|------|
| ToMe (baseline) | 37.40 | 18.38 | Merging without frequency awareness |
| + Laplacian gating | 41.90 | 18.04 | Class. +7.15%, FID improved by 0.34 |
| ToDo (baseline) | 67.78 | 15.93 | KV average downsampling |
| + IE-KVD (ours) | 72.88 | 15.46 | Class. +5.10%, FID also improved |

### Key Findings
- Frequency awareness is key: Removing Laplacian gating leads to substantial degradation in classification accuracy, confirming that high-frequency preservation is vital for classification.
- Generation and classification can be a win-win: BiGain improves classification while slightly improving FID (0.34/1.85% on ImageNet-1K), as preserving edges/textures also helps generate details.
- Preserving full Query resolution is central: Downsampling Q along with KV destroys attention precision, leading to losses in both classification and generation.

## Highlights & Insights
- **Valuable Problem Discovery**: The observation that token compression has asymmetric effects on classification and generation is significant, pointing out the gap where "looking good does not equal accurate classification."
- **Simple yet Powerful Design Principles**: Balanced spectral retention is a reusable design rule.
- **Completely Training-Free**: High practicality as it requires no retraining.

## Limitations & Future Work
- The Laplacian kernel is a fixed $3 \times 3$, which may not suit features at all scales.
- Significant degradation occurs in both classification and generation at extreme compression rates (>80%).
- Discriminative ability was only validated on classification tasks, not extended to detection/segmentation.
- The $\alpha$ parameter for IE-KVD needs to be adjusted by task (using different values for generation and classification).

## Related Work & Insights
- **vs ToMe/ToMeSD**: Merges tokens directly based on embedding similarity without distinguishing frequency characteristics, causing significant harm to classification.
- **vs ToDo**: Downsamples KV using average pooling, losing high frequencies; BiGain preserves them via controllable inter-extrapolation.
- **vs Diff-Pruning/DiP-GO**: These are model pruning methods that change the model structure, whereas BiGain is a training-free token-level operation.

## Rating
- Novelty: ⭐⭐⭐⭐ First to define diffusion model compression as a joint generation+classification objective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple models (DiT/UNet) × multiple datasets × multiple tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation with a unique frequency analysis perspective.
- Value: ⭐⭐⭐⭐ Directly provides guidance for deploying dual-purpose diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unified Vector Floorplan Generation via Markup Representation](unified_vector_floorplan_generation_via_markup_representation.md)
- [\[CVPR 2026\] Mixture of States: Routing Token-Level Dynamics for Multimodal Generation](mixture_of_states_routing_token-level_dynamics_for_multimodal_generation.md)
- [\[CVPR 2026\] CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation](cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)
- [\[CVPR 2026\] CoD: A Diffusion Foundation Model for Image Compression](cod_a_diffusion_foundation_model_for_image_compression.md)
- [\[CVPR 2026\] Flash-Unified: Training-Free and Task-Aware Acceleration for Native Unified Models](flash-unified_a_training-free_and_task-aware_acceleration_framework_for_native_u.md)

</div>

<!-- RELATED:END -->
