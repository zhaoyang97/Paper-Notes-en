---
title: >-
  [Paper Note] Nested Diffusion Models Using Hierarchical Latent Priors
description: >-
  [CVPR 2025][Image Generation][Nested Diffusion] This paper proposes Nested Diffusion Models, which sequentially generate latent variables at different semantic levels using a series of coarse-to-fine diffusion models, conditioning each stage on the outputs of coarser stages. On ImageNet 256×256, with only a 25% increase in computational cost, it reduces the unconditional FID from 45.19 to 11.05 and the conditional FID to 3.97.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Nested Diffusion"
  - "Hierarchical Latent Variables"
  - "Semantic Priors"
  - "Information Compression"
  - "Non-Markovian Generation"
date: 2026-05-08
content_hash: d3e1606ee23bb392
---

# Nested Diffusion Models Using Hierarchical Latent Priors

**Conference**: CVPR 2025  
**arXiv**: [2412.05984](https://arxiv.org/abs/2412.05984)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Nested Diffusion, Hierarchical Latent Variables, Semantic Priors, Information Compression, Non-Markovian Generation

## TL;DR

This paper proposes Nested Diffusion Models, which sequentially generate latent variables at different semantic levels using a series of coarse-to-fine diffusion models, conditioning each stage on the outputs of coarser stages. On ImageNet 256×256, with only a 25% increase in computational cost, it reduces the unconditional FID from 45.19 to 11.05 and the conditional FID to 3.97.

## Background & Motivation

1. **Background**: Diffusion models (such as DiT) have achieved SOTA performance in image generation, but the quality of unconditional generation is far inferior to class-conditional generation (FID 45.19 vs 13.75).
2. **Limitations of Prior Work**: Unconditional generation lacks semantic guidance, requiring the diffusion process to "invent" all semantic information from pure noise—which is extremely difficult.
3. **Key Challenge**: Scaling up models (such as DiT-XL with 118 GFlops) yields diminishing returns while drastically increasing computational costs; a more efficient solution is needed to introduce semantic priors.
4. **Goal**: To provide coarse-to-fine semantic guidance for diffusion models through hierarchical latent variables, significantly improving generation quality with low additional overhead.
5. **Key Insight**: Features from different levels and scales of pretrained vision encoders (e.g., MoCo-v3/CLIP) naturally contain semantic information at various granularities.
6. **Core Idea**: $L$-level nesting—the coarsest level generates global semantics from noise, while each finer level generates more detailed features conditioned on all coarser levels, and the final level generates pixels.

## Method

### Overall Architecture

Pretrained encoder extracts multi-scale features $\rightarrow$ SVD dimensionality reduction $\rightarrow$ Gaussian noise injection to control information capacity $\rightarrow$ $L$-level diffusion models: $z_L$ (coarsest) $\rightarrow$ $z_{L-1}$ $\rightarrow$ ... $\rightarrow$ $z_1 = x$ (image). The denoiser at each level is conditioned on the outputs of all coarser levels (non-Markovian).

### Key Designs

1. **Hierarchical Latent Variable Construction**

    - **Function**: Extracting features of different semantic granularities from images to serve as training targets.
    - **Mechanism**: Pretrained vision encoders extract features at different patch scales $\rightarrow$ SVD dimensionality reduction is applied to prevent information over-completeness $\rightarrow$ Gaussian noise injection $\tilde{z}_l \sim \mathcal{N}(z_l, \sigma_l^2 I)$ controls the KL divergence (information capacity).
    - **Design Motivation**: Noise injection is crucial—when $\sigma=0$, the system collapses into an autoencoder (rendering the method ineffective); when $\sigma=1$, the information capacity is maximized, but this increases the learning difficulty. Ablations confirm that $\sigma^2=1.0$ is optimal.

2. **Non-Markovian Conditioning**

    - **Function**: Enabling each diffusion model level to utilize information from all coarser levels.
    - **Mechanism**: The denoiser at the $l$-th level is conditioned on the complete set of $z_{>l} = \{z_{l+1}, ..., z_L\}$, rather than only the preceding level.
    - **Design Motivation**: A Markov chain tends to lose coarse-level information (which decays after passing through multiple levels); the non-Markovian approach ensures each level directly accesses global semantics.

3. **Hierarchical CFG Decay**

    - **Function**: Balancing the guidance scale of different levels during inference.
    - **Mechanism**: CFG weights decrease from coarse to fine, e.g., $\{w_i\} = [0.5, 0.4, 0.3, 0.2, 0.1]$—coarser levels provide stronger guidance, while finer levels gain more freedom.
    - **Design Motivation**: Coarse levels determine global semantics (requiring strong guidance), whereas fine levels govern detailed diversity (where excessively strong guidance leads to a loss of diversity).

### Loss & Training

$\mathcal{L} = \sum_{l=1}^{L-1} \mathbb{E}[||\epsilon_l - D_{\theta_l}(\alpha^{(t)} z_l + \beta^{(t)} \epsilon_l, \tilde{z}_{>l}, t)||^2] + \mathbb{E}[||\epsilon_L - D_{\theta_L}(...)||^2]$. U-ViT-Base architecture, ImageNet 200 epochs.

## Key Experimental Results

### Main Results

| Method | GFlops | Unconditional FID↓ | Conditional FID↓ |
|------|--------|-----------|----------|
| DiT-L/2 | 80.0 | - | 23.3 |
| DiT-XL/2+REPA | 118.6 | - | 12.3 |
| Baseline (L=1) | 27.0 | 45.19 | 13.75 |
| **Nested L=5** | **34.0** | **11.05** | **3.97** |

### Ablation Study

| Levels | Unconditional FID | Conditional FID | Description |
|------|-----------|---------|------|
| L=1 | 45.19 | 13.75 | Baseline |
| L=2 | 20.66 | 5.31 | Coarse level brings significant improvement |
| L=3 | 19.00 | 4.69 | Diminishing returns |
| L=5 | **11.05** | **3.97** | Optimal |

### Key Findings

- Unconditional L=5 (FID 11.05) outperforms the conditional baseline (FID 13.75)—hierarchical priors are more effective than class labels.
- With only a 25% increase in computational cost (27 $\rightarrow$ 34 GFlops), the FID is reduced by 75%.
- When noise injection is set to $\sigma^2=0$, the FID surges to 19.04—demonstrating that information compression is key to the success of the method.

## Highlights & Insights

- **Unconditional Outperforming Conditional**: Hierarchical semantic priors provide richer guidance information than class labels.
- **75% FID Reduction with 25% Overhead**: An extremely high efficiency-to-quality ratio.
- **Theoretical Foundation of Information Compression**: Controlling the amount of information per level via KL divergence has a solid information-theoretic basis.

## Limitations & Future Work

- Hyperparameters $\{\sigma_l\}$ require level-by-level tuning. Although mitigated by greedy search, this still increases parameter tuning complexity.
- Only explored up to L=5; the returns/costs of deeper hierarchies remain unexplored.
- Performance depends on the feature quality of the pretrained vision encoder.

## Related Work & Insights

- **vs DiT-XL+REPA**: DiT-XL+REPA requires 118.6 GFlops to reach an FID of 12.3, whereas the nested model achieves 3.97 with only 34 GFlops—making it 3.5 times more efficient.
- **vs Cascaded Diffusion (Imagen)**: Cascaded models increase resolution in the pixel space, while nested models increase granularity in the semantic space—representing hierarchical architectures across different dimensions.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of nested diffusion with hierarchical semantic priors is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation and ablations on ImageNet, but lacks testing on more diverse datasets.
- Writing Quality: ⭐⭐⭐⭐ Thorough and solid theoretical analysis.
- Value: ⭐⭐⭐⭐⭐ Significantly improves the quality of unconditional generation, with broad potential impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Hierarchical Flow Diffusion for Efficient Frame Interpolation](hierarchical_flow_diffusion_for_efficient_frame_interpolation.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[ICML 2025\] Learning Single Index Models with Diffusion Priors](../../ICML2025/image_generation/learning_single_index_models_with_diffusion_priors.md)
- [\[CVPR 2025\] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [\[CVPR 2025\] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting](luminet_latent_intrinsics_meets_diffusion_models_for_indoor_scene_relighting.md)

</div>

<!-- RELATED:END -->
