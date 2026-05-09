---
title: >-
  [Paper Note] Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering
description: >-
  [ICCV 2025][Image Generation][Inverse Rendering] This paper presents Ouroboros, a unified framework comprising two single-step diffusion models (for inverse rendering RGB→X and forward rendering X→RGB respectively) that are jointly trained with cycle-consistency to enforce bidirectional rendering coherence. The method achieves state-of-the-art performance across multiple datasets while running 50× faster than multi-step diffusion baselines, and can be applied to video decomposition in a training-free manner.
tags:
  - ICCV 2025
  - Image Generation
  - Inverse Rendering
  - Forward Rendering
  - Cycle Consistency
  - Single-step Diffusion
  - Intrinsic Image Decomposition
date: 2026-05-08
content_hash: 1cb9b0d4e9c0d198
---

# Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering

**Conference**: ICCV 2025
**arXiv**: [2508.14461](https://arxiv.org/abs/2508.14461)
**Code**: [https://siwensun.github.io/ouroboros-project/](https://siwensun.github.io/ouroboros-project/)
**Area**: Diffusion Models / 3D Vision
**Keywords**: Inverse Rendering, Forward Rendering, Cycle Consistency, Single-step Diffusion, Intrinsic Image Decomposition

## TL;DR

This paper presents Ouroboros, a unified framework comprising two single-step diffusion models (for inverse rendering RGB→X and forward rendering X→RGB respectively) that are jointly trained with cycle-consistency to enforce bidirectional rendering coherence. The method achieves state-of-the-art performance across multiple datasets while running 50× faster than multi-step diffusion baselines, and can be applied to video decomposition in a training-free manner.

## Background & Motivation

**State of the Field**: Inverse rendering (estimating intrinsic properties such as geometry, material, and lighting from images) and forward rendering (synthesizing images from intrinsic properties) are fundamental problems in computer graphics and vision. Diffusion models have made recent advances in both directions: RGB↔X first proposed a unified diffusion framework supporting bidirectional rendering, and DiffusionRenderer extended this to the video domain.

**Limitations of Prior Work**:
- **Computational inefficiency**: Existing diffusion-based methods require multi-step denoising (e.g., 50-step DDIM), resulting in slow inference.
- **Lack of cycle consistency**: Inverse and forward rendering models are trained independently; when applied sequentially (image → decomposition → reconstruction), the reconstructed output is often inconsistent with the original input.
- **Domain limitation**: RGB↔X is trained exclusively on indoor datasets and generalizes poorly to outdoor scenes.
- **Scarce training data**: Large-scale datasets with complete RGB-X pairs are limited, and the available intrinsic property channels vary across datasets.

**Root Cause**: How can diffusion-based rendering be substantially accelerated without sacrificing quality, while simultaneously ensuring cycle consistency between the two rendering directions?

**Paper Goals**: (a) Accelerate multi-step diffusion rendering to single-step inference; (b) establish cycle consistency between inverse and forward rendering; (c) extend coverage to both indoor and outdoor multi-domain scenes; (d) enable training-free video decomposition.

**Starting Point**: Inspired by E2E (End-to-End Fine-tuning), the method fixes the timestep at $t = T$ for single-step prediction fine-tuning. The key innovation is extending single-step fine-tuning from perceptual tasks (e.g., geometry estimation) to synthesis tasks (forward rendering), and introducing CycleGAN-style cycle-consistency training to allow inverse and forward rendering to mutually reinforce each other.

**Core Idea**: Fine-tune RGB↔X into two single-step diffusion models (achieving 50× speedup), jointly train them with a cycle-consistency loss to mutually regularize bidirectional rendering, and leverage the cyclic structure to incorporate unannotated real-world images for cross-domain generalization.

## Method

### Overall Architecture

Ouroboros consists of two complementary single-step diffusion models: (1) an inverse rendering model (RGB→X) that predicts five intrinsic property maps from an RGB image—surface normals $\mathbf{n}$, albedo $\mathbf{a}$, roughness $\mathbf{r}$, metallicity $\mathbf{m}$, and diffuse irradiance $\mathbf{E}$; and (2) a forward rendering model (X→RGB) that synthesizes an RGB image from intrinsic property maps. Both models are fine-tuned from the pretrained RGB↔X weights and mutually reinforced through cyclic training.

### Key Designs

1. **Single-step Finetuning**:

    - **Function**: Fine-tunes the multi-step diffusion model for single-step inference, achieving 50× speedup.
    - **Mechanism**: The timestep is fixed at $t = T$, forcing the model to generate the target from the maximum-noise state in one step. Unlike E2E, multi-resolution noise initialization is used instead of zero noise. The UNet's v-prediction output is converted to a predicted latent via $\hat{\mathbf{z}}_0 = \sqrt{\bar\alpha_T} \mathbf{z}_T - \sqrt{1-\bar\alpha_T} \hat{\mathbf{v}}_\theta$, which is then decoded and compared against the ground truth.
    - **Design Motivation**: Non-deterministic noise initialization (rather than zero noise) is particularly suitable for intrinsic image decomposition, where the solution is inherently ambiguous—e.g., the division between albedo and irradiance has no unique answer.

2. **Task-specific Loss Functions**:

    - **Function**: Designing specialized losses tailored to different types of intrinsic attributes.
    - Surface normals: angular loss $\mathcal{L}_\mathbf{n} = \frac{1}{N}\sum_i \arccos \frac{\mathbf{n}_i \cdot \hat{\mathbf{n}}_i}{||\mathbf{n}_i|| \cdot ||\hat{\mathbf{n}}_i||}$
    - Irradiance: affine-invariant loss $\mathcal{L}_\mathbf{E} = |\mathbf{E} - \mathbf{S}\hat{\mathbf{E}} - \mathbf{T}|_F^2$ (fitting per-channel scale and shift parameters via least squares)
    - Other attributes (albedo, roughness, metallicity, RGB): MSE loss
    - **Design Motivation**: Surface normals require angular rather than numerical consistency; the decomposition of irradiance and albedo is subject to scale ambiguity, which the affine-invariant loss explicitly removes.

3. **Cycle-consistency Training**:

    - **Function**: Enforces consistency in the RGB→X→RGB' and X→RGB→X' cycles.
    - **Mechanism**: Given a pair $(X, I)$, the two models first generate $(\hat{I}, \hat{X})$ respectively; these generated outputs are then used as inputs for a second round of inference to obtain $(\tilde{X}, \tilde{I})$. The cycle loss is minimized: $\mathcal{L}_{cycle} = |\mathbf{X} - \tilde{\mathbf{X}}|^2 + |\mathbf{I} - \tilde{\mathbf{I}}|^2$
    - **Design Motivation**: (a) Improves consistency of bidirectional rendering; (b) more importantly, the cyclic structure enables the incorporation of unannotated real-world images (MSCOCO, Flickr30k) for self-supervised training, reducing dependence on scarce synthetic data.

4. **Training-free Video Inference**:

    - **Function**: Extends the image-level single-step model to video decomposition without requiring video training data.
    - **Mechanism**: 2D convolution kernels of size $3 \times 3$ are replaced with pseudo-3D kernels of size $1 \times 3 \times 3$, and multi-frame patches are flattened for joint attention over the spatial-temporal dimension. A sliding window processes long videos, with overlapping regions initialized as a weighted mixture of the previous window's prediction and fresh noise: $\mathbf{z}_{init} = \gamma \cdot \mathbf{z}_{prev} + (1-\gamma) \cdot \epsilon$ (with $\gamma=0.1$).
    - **Design Motivation**: Training a native video diffusion model is prohibitively expensive; the proposed lightweight architectural extension exploits spatiotemporal locality to achieve temporal consistency.

### Loss & Training

Training proceeds in two stages: (1) **Initial single-step fine-tuning**—the model is fine-tuned on three datasets: Hypersim and InteriorVerse (indoor) and MatrixCity (outdoor), sampling 17K images from each, with channel dropout applied to handle inconsistent intrinsic property channels across datasets; (2) **Cycle training**—the two initially fine-tuned models are jointly trained, with an additional 20K unannotated images each from MSCOCO and Flickr30k introduced for self-supervised learning. The total loss is the sum of task-specific losses and the cycle-consistency loss.

## Key Experimental Results

### Main Results (Inverse Rendering)

**Albedo Prediction**:

| Method | Hypersim PSNR↑ | MatrixCity PSNR↑ | InteriorVerse PSNR↑ |
|------|---------------|------------------|---------------------|
| RGB↔X | 18.67 | 12.61 | 16.17 |
| Careaga & Aksoy | 12.01 | 17.30 | 15.51 |
| Kocsis et al. | 12.40 | 15.66 | 14.62 |
| **Ouroboros** | **18.98** | **25.38** | **22.07** |

**Normal Prediction**:

| Method | Hypersim Mean↓ | MatrixCity Mean↓ | InteriorVerse Mean↓ |
|------|---------------|------------------|---------------------|
| RGB↔X | 17.21 | 23.82 | 12.10 |
| StableNormal | 16.65 | 18.18 | 10.73 |
| E2E | 16.30 | 13.91 | 15.87 |
| **Ouroboros** | **11.98** | **18.12** | **9.58** |

**Forward Rendering**:

| Method | Hypersim PSNR↑ | MatrixCity PSNR↑ | InteriorVerse PSNR↑ |
|------|---------------|------------------|---------------------|
| RGB↔X | 16.37 | 9.24 | 13.70 |
| **Ouroboros** | **18.09** | **21.57** | **15.79** |

### Ablation Study

| Configuration | Effect |
|------|------|
| Without cycle training | Irradiance estimation lacks detail; reconstructed colors are inaccurate |
| With cycle training | Irradiance is sharper; reconstructed colors are more faithful |
| Cycle training without wild data | Insufficient understanding of high-rise lighting and surface materials |
| Cycle training with wild data | More realistic irradiance estimation; more continuous metallicity prediction |
| Cycle training without e2e loss | Discontinuous metallicity and irradiance predictions |
| Cycle training with e2e loss | Better physical property continuity and material understanding |

Roughness and Metallicity (MatrixCity, PSNR↑):

| Method | Roughness PSNR | Metallicity PSNR |
|------|---------------|-----------------|
| RGB↔X | 23.82 | 6.83 |
| **Ouroboros** | **24.04** | **26.32** |

### Key Findings

- **Most significant gains on outdoor scenes**: On MatrixCity, albedo PSNR improves from 12.61 to 25.38 (+12.77) and forward rendering PSNR improves from 9.24 to 21.57 (+12.33), demonstrating that cycle training combined with multi-domain data substantially enhances outdoor generalization.
- **Dramatic improvement in metallicity estimation**: PSNR improves from 6.83 to 26.32, indicating a severe deficiency in the original RGB↔X model's understanding of metallic materials.
- **Value of in-the-wild data**: Incorporating unannotated real images through cyclic self-supervision significantly improves understanding of real-world lighting and materials—a unique advantage of cycle-consistency training.
- **50× speedup without notable quality degradation**: Single-step inference outperforms multi-step RGB↔X on most metrics, confirming that E2E fine-tuning is effective for rendering tasks.
- **Cross-domain generalization of irradiance**: The model is trained for irradiance estimation only on Hypersim (indoor), yet cycle training enables successful generalization to outdoor scenes.

## Highlights & Insights

- **Cycle consistency as a dual win**: Beyond improving consistency itself, the cyclic structure opens a pathway to leveraging unannotated real-world data. In the rendering domain where annotated synthetic data is scarce, this strategy of "introducing real data through cyclic structure" is particularly elegant.
- **Viability of single-step diffusion rendering**: The paper demonstrates that E2E fine-tuning is applicable not only to perceptual tasks such as depth and normal estimation, but also to synthesis tasks such as forward rendering, paving the way for diffusion models in real-time rendering applications.
- **Design rationale for non-deterministic single-step prediction**: Using multi-resolution noise rather than zero noise acknowledges the inherent ambiguity of intrinsic decomposition, which is a more principled choice than imposing a unique solution.

## Limitations & Future Work

- **Training data quality is the primary bottleneck**: The authors explicitly note that publicly available datasets (InteriorVerse, Hypersim) provide unreliable intrinsic property maps and lack accurate lighting information.
- No quantitative comparison with DiffusionRenderer (a video diffusion approach) on video decomposition is provided.
- The pseudo-3D video inference scheme may suffer from error accumulation over long videos.
- Cycle training increases training complexity, requiring simultaneous updates to both models.
- Concrete inference speed comparisons are absent; only the 50× speedup claim is stated without reporting actual runtimes.

## Related Work & Insights

- **vs. RGB↔X**: The direct predecessor of Ouroboros. Building upon it, the proposed method achieves three major improvements—single-step acceleration, cycle consistency, and multi-domain training—and comprehensively outperforms it on nearly all metrics.
- **vs. DiffusionRenderer**: DiffusionRenderer employs a video diffusion model for video decomposition, whereas Ouroboros achieves temporally consistent video decomposition through a training-free pseudo-3D extension, making it considerably more lightweight.
- **vs. CycleGAN**: The source of the cycle-consistency idea. Applying it to conditional diffusion models in the context of rendering is a novel attempt, and using the cyclic structure to incorporate unannotated data constitutes an additional contribution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of single-step diffusion rendering and cycle consistency is a valuable contribution, though the individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage spans multiple indoor and outdoor datasets across diverse intrinsic attributes with thorough ablations, but a comparison with DiffusionRenderer is missing.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is clearly structured, though some descriptions are redundant and the mathematical derivations could be more concise.
- **Value**: ⭐⭐⭐⭐ The 50× speedup combined with quality improvements is practically significant, and the work broadens the prospects of cycle-consistency training in the rendering domain.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](../../CVPR2026/image_generation/renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[ICCV 2025\] Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences](cycle_consistency_as_reward_learning_image-text_alignment_without_human_preferen.md)
- [\[CVPR 2026\] Cycle-Consistent Tuning for Layered Image Decomposition](../../CVPR2026/image_generation/cycle-consistent_tuning_for_layered_image_decomposition.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [\[ICCV 2025\] Learning Few-Step Diffusion Models by Trajectory Distribution Matching](learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)

<!-- RELATED:END -->
