---
title: >-
  [Paper Note] BiGain: Unified Token Compression for Joint Generation and Classification
description: >-
  [CVPR 2025][Image Generation][Token Compression] BiGain, for the first time, reformulates token compression in diffusion models as a dual-objective optimization problem for both generation and classification. It proposes two frequency-aware operators: Laplacian-Gated Token Merging (L-GTM) and Interpolation-Extrapolation KV Downsampling (IE-KVD). BiGain significantly improves classification accuracy while maintaining generation quality (Acc +7.15%…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Token Compression"
  - "Diffusion Classifier"
  - "Frequency-Aware"
  - "Token Merging"
  - "KV Downsampling"
date: 2026-05-08
content_hash: 14445cefcc174e71
---

# BiGain: Unified Token Compression for Joint Generation and Classification

**Conference**: CVPR 2025  
**arXiv**: [2603.12240](https://arxiv.org/abs/2603.12240)  
**Code**: [https://github.com/Greenoso/BiGain](https://github.com/Greenoso/BiGain)  
**Area**: Image Generation / Diffusion Model Acceleration  
**Keywords**: Token Compression, Diffusion Classifier, Frequency-Aware, Token Merging, KV Downsampling

## TL;DR

BiGain, for the first time, reformulates token compression in diffusion models as a dual-objective optimization problem for both generation and classification. It proposes two frequency-aware operators: Laplacian-Gated Token Merging (L-GTM) and Interpolation-Extrapolation KV Downsampling (IE-KVD). BiGain significantly improves classification accuracy while maintaining generation quality (Acc +7.15%, FID -0.34 under a 70% merging ratio on ImageNet-1K).

## Background & Motivation

### Background

**Background**: Diffusion models are not only dominant in generation but are also repurposed as classifiers (via class-wise denoising likelihood evaluation), where a single backbone supports both generative and discriminative tasks.

**Limitations of Prior Work**: Existing token compression methods (such as ToMe and ToDo) only optimize for generation quality (FID) while ignoring discriminative capability. Compression that has almost no effect on generation can severely degrade classification.

**Key Challenge**: Generation relies on low-to-medium frequency global semantics, whereas classification depends on high-frequency details. Naive compression often removes structures critical for classification.

**Goal**: Design training-free, plug-and-play token compression that preserves both generation and classification.

**Key Insight**: Frequency separation—mapping intermediate features to frequency-aware representations to decouple high-frequency elements from low/medium-frequency ones.

**Core Idea**: Balancing spectrum preservation serves as a reliable design principle for token compression.

## Method

### Overall Architecture

Two complementary training-free operators are proposed: L-GTM for token merging and IE-KVD for key/value (K/V) downsampling in attention blocks. They are compatible with both DiT and U-Net.

### Key Designs

1. **Laplacian-Gated Token Merging (L-GTM)**

    - **Function**: Prioritizes merging frequency-smooth tokens while preserving high-frequency tokens.
    - **Mechanism**: Applies a spatial Laplacian filter to latent states to obtain position-wise frequency scores. Tokens with the lowest scores serve as destinations, while others act as sources for bipartite matching and merging.
    - **Design Motivation**: Standard ToMe does not distinguish frequency bands, leading to high-frequency tokens being merged prematurely, which degrades classification.

2. **Interpolation-Extrapolation KV Downsampling (IE-KVD)**

    - **Function**: Performs spatial downsampling on K/V features while keeping Q at full resolution.
    - **Mechanism**: Conducts controllable interpolation/extrapolation between nearest-neighbor and average pooling. Specifically, nearest-neighbor is favored for classification to preserve high frequencies, while a linear transition from low-to-high frequency is used for generation.
    - **Design Motivation**: Keeping full-resolution Q maintains fine-grained receptive fields of tokens, which is crucial for Monte Carlo estimation accuracy in diffusion classifiers.

3. **Diffusion Classifier Compatibility**

    - **Function**: Ensures the estimation validity of the diffusion classifier is not compromised.
    - **Mechanism**: Since both operators are timestep-local and deterministic, all classes share the identical compression schedule.

### Loss & Training

Training-free and requires no optimization. It is directly inserted into attention layers during inference. A multi-stage pruning acceleration schedule is adopted for classification.

## Key Experimental Results

### Main Results

| Method | FLOPs Reduction | Acc@1 (Pets) | vs Baseline |
|------|-----------|-------------|------------|
| Baseline | - | 81.03 | - |
| ToMe | 10% | 72.96 | -8.07 |
| BiGain_TM | 10% | 78.38 | -2.65 |
| ToDo | 14.2% | 79.15 | -1.88 |
| BiGain_TD | 14.2% | 79.90 | -1.13 |

### Ablation Study

- Removing Laplacian gating results in significant drop in classification accuracy.
- IE-KVD with $\alpha = 0.9$ yields the best classification accuracy, while a linear transition of $\alpha$ from $0.8$ to $1.2$ performs best during generation.
- Optimal $\alpha$ values differ between DiT and U-Net, requiring architecture-specific adaptation.

### Key Findings

- Baseline compression degrades classification accuracy much earlier and more severely than generation quality.
- Retaining full-resolution Q is critical to preserving classification accuracy.
- Frequency balancing is consistently effective across two architectures and four datasets.
- Under a 70% merging ratio on ImageNet-1K: yields Acc +7.15% and FID -0.34.

## Highlights & Insights

- The first systematic study to analyze the impact of token compression on the diagnostic/classification capabilities of diffusion models.
- The frequency separation insight is simple yet powerful: "looking good" is not equivalent to "classifying well".
- Both operators are training-free and plug-and-play.
- Achieves Pareto improvement instead of a simple trade-off.

## Limitations & Future Work

- Optimal parameters remain dependent on specific architectures and datasets.
- The Laplacian filter is a handcrafted frequency proxy.
- Combining this method with step-reduction techniques has not been explored yet.

## Related Work & Insights

- ToMe (Bolya et al. 2023) first introduced training-free token merging; BiGain incorporates frequency awareness on top of it.
- ToDo proposed token downsampling instead of merging; BiGain's IE-KVD introduces controllable interpolation/extrapolation on top of this concept.
- Diffusion classifiers (Li et al., Chen et al.) demonstrated the discriminative ability of diffusion backbones; this work is the first to focus on the impact of compression on classification.
- The frequency separation philosophy can be generalized to other token compression scenarios, such as video generation and 3D.
- Diff-Pruning and DiP-GO accelerate diffusion via model pruning, which is complementary to the token-level compression in this work.
- Adaptive Block Merging (ABM) variants can be leveraged to further accelerate high-resolution stages.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to formulate dual-objective token compression for both generation and classification.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across four datasets and two architectures.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and coherent derivations.
- **Value**: ⭐⭐⭐⭐ Direct guidance for deploying dual-use diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MExD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification](mexd_an_expert-infused_diffusion_model_for_whole-slide_image_classification.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] DreamOmni: Unified Image Generation and Editing](dreamomni_unified_image_generation_and_editing.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] Random Conditioning for Diffusion Model Compression with Distillation](random_conditioning_for_diffusion_model_compression_with_distillation.md)

</div>

<!-- RELATED:END -->
