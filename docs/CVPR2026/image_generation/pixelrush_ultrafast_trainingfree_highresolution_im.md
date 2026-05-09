---
title: >-
  [Paper Note] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion
description: >-
  [CVPR 2026][Image Generation][Training-free high-resolution generation] PixelRush is the first method to bring training-free high-resolution image generation into practical deployment. By truncating the reverse diffusion process via partial DDIM inversion to skip redundant low-frequency reconstruction steps, it enables few-step diffusion models to function within a patch-based refinement pipeline. Combined with Gaussian filter blending and noise injection to eliminate artifacts, the method generates 2K images in 4 seconds and 4K images in 20 seconds—10–35× faster than the state of the art while achieving superior FID.
tags:
  - CVPR 2026
  - Image Generation
  - Training-free high-resolution generation
  - patch-based inference
  - partial inversion
  - few-step diffusion
  - Gaussian blending
date: 2026-05-08
content_hash: a4bb4ecc4184e0b5
---

# PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion

**Conference**: CVPR 2026
**arXiv**: [2602.12769](https://arxiv.org/abs/2602.12769)
**Code**: N/A
**Area**: Image Generation / Diffusion Model Acceleration
**Keywords**: Training-free high-resolution generation, patch-based inference, partial inversion, few-step diffusion, Gaussian blending

## TL;DR

PixelRush is the first method to bring training-free high-resolution image generation into practical deployment. By truncating the reverse diffusion process via partial DDIM inversion to skip redundant low-frequency reconstruction steps, it enables few-step diffusion models to function within a patch-based refinement pipeline. Combined with Gaussian filter blending and noise injection to eliminate artifacts, the method generates 2K images in 4 seconds and 4K images in 20 seconds—10–35× faster than the state of the art while achieving superior FID.

## Background & Motivation

**Background**: Pretrained diffusion models (e.g., SDXL) generate high-quality images at their native training resolution, but produce severe artifacts when applied at higher resolutions. Training-free high-resolution methods fall into two categories: direct inference approaches (ScaleCrafter, FreeScale, etc., which modify the frequency domain) and patch-based approaches (DemoFusion, etc., which process image patches), both of which rely on a full 50-step reverse diffusion process.

**Limitations of Prior Work**: **Speed is the critical bottleneck**—generating a single 4K image takes 5–10 minutes, and 8K generation requires over an hour. The acceleration offered by CutDiffusion through reducing patch count is marginal, while LSNR reduces steps from 50 to 30 via a trained plug-in module but remains a multi-step paradigm. This renders high-resolution generation entirely impractical.

**Key Challenge**: Existing methods perturb a coarsely upsampled image to full Gaussian noise and then execute a complete reverse process. However, the reverse process in high-resolution refinement also follows a frequency-hierarchical reconstruction pattern—low-frequency global structure forms in the early steps and high-frequency details emerge in the later steps. Since the coarse image already contains the complete low-frequency structure, **reconstructing it from full noise is computationally redundant**.

**Goal**: (1) Eliminate the redundant steps in the reverse process dedicated to reconstructing low-frequency structure; (2) enable few-step models to operate effectively within a patch-based pipeline; (3) overcome the boundary artifacts and over-smoothing introduced by few-step models.

**Key Insight**: The authors observe that the denoising process for high-resolution refinement likewise follows frequency-hierarchical reconstruction; therefore, denoising from an intermediate timestep is sufficient. This naturally complements the large-update characteristic of few-step diffusion models—a short truncated trajectory combined with large denoising steps is sufficient to synthesize all high-frequency details.

**Core Idea**: Truncate the full-noise reverse diffusion into partial inversion followed by one-step refinement, achieving for the first time a successful integration of few-step diffusion models with patch-based high-resolution generation.

## Method

### Overall Architecture

A two-stage pipeline: (1) **Base stage**—SDXL generates a base image at the native 1024 resolution; (2) **Cascaded upsampling**—resolution is doubled at each stage (4× in pixel count) through the cycle of "pixel-space bicubic upsampling → VAE encoding to obtain a coarse latent → PixelRush refinement → VAE decoding," repeated until the target resolution is reached. Pixel-space upsampling avoids the artifacts introduced by direct interpolation in latent space.

### Key Designs

1. **Partial Inversion**:

    - *Function*: Skip the redundant early steps of the reverse diffusion process that reconstruct low-frequency structure, retaining only the later steps responsible for high-frequency detail synthesis.
    - *Mechanism*: The coarse latent is perturbed only to an intermediate timestep $K=249$ (rather than $T=999$) via DDIM inversion, preserving structural information as a strong prior. Experiments show that initiating denoising from $t=259$ reduces computation by 75% while actually improving FID (52.90 vs. 54.70), since the unnecessary low-frequency reconstruction steps are eliminated.
    - *Design Motivation*: The coarse upsampled image already contains the complete low-frequency structure, so full-noise reconstruction is unnecessary. The truncated trajectory naturally accommodates the large-step updates of few-step models.

2. **Gaussian Filter Patch Blending**:

    - *Function*: Eliminate the checkerboard artifacts at patch boundaries introduced by few-step models.
    - *Mechanism*: A Gaussian blur kernel is applied to the binary blending mask over overlapping regions, producing a continuously smooth weight map in which weights are highest at patch centers and decay smoothly toward edges. This is essentially image feathering applied in latent space.
    - *Design Motivation*: Standard mean blending (as in MultiDiffusion) is effective under multi-step denoising, where many small updates gradually reconcile boundary discrepancies. However, the large updates of few-step models produce irreconcilable boundary differences—naive averaging merely blurs rather than eliminates them.

3. **Noise Injection**:

    - *Function*: Counter the inherent over-smoothing of few-step models and restore high-frequency detail.
    - *Mechanism*: During the reverse step, the predicted noise is interpolated with random noise via spherical linear interpolation: $\epsilon' = \text{slerp}(\epsilon_\theta, \epsilon_{rand}, \lambda)$ with $\lambda=0.95$. The injected stochasticity flattens the data distribution $p_\gamma(\mathbf{x})$, promoting the synthesis of high-frequency components.
    - *Design Motivation*: Multi-step models synthesize fine details through many small incremental updates; the large jumps of few-step models cannot adequately recover high-frequency content. Noise injection is only beneficial for few-step models—applying it to multi-step models degrades quality due to accumulated errors.

### Loss & Training

The method is entirely training-free. It uses pretrained SDXL (for base generation) and SDXL-Turbo (for few-step refinement) without any fine-tuning or additional training.

## Key Experimental Results

### Main Results

| Method | 2K FID↓ | 2K IS↑ | 2K Time | 4K FID↓ | 4K IS↑ | 4K Time |
|--------|---------|--------|---------|---------|--------|---------|
| SDXL-DI | 73.34 | 10.93 | 28s | 153.53 | 7.32 | 247s |
| FouriScale | 72.65 | 12.31 | 87s | 98.97 | 8.54 | 680s |
| DemoFusion | 68.46 | 13.15 | 75s | 74.75 | 12.57 | 507s |
| FreeScale | 52.87 | 13.56 | 53s | 58.28 | 13.35 | 323s |
| **PixelRush** | **50.13** | **14.32** | **4s** | **54.67** | **13.75** | **20s** |

### Ablation Study

| Configuration | Steps | FID↓ | IS↑ | Time | Notes |
|---------------|-------|------|-----|------|-------|
| Baseline (A) | 50 | 54.70 | 13.92 | 67s | DDIM inversion + full 50-step denoising |
| + Partial Inversion | 15 | 52.90 | 13.89 | 18s | 3.7× speedup; quality improves |
| + Few-step model | 1 | 57.23 | 13.65 | 4s | Extremely fast but artifacts + over-smoothing |
| + Gaussian blending | 1 | 56.16 | 13.77 | 4s | Checkerboard artifacts eliminated |
| + Noise injection | 1 | 50.13 | 14.32 | 4s | Over-smoothing eliminated; best overall |

Ablation on inversion depth ($K$):

| Configuration | K | FID↓ | Time |
|---------------|---|------|------|
| Baseline | 50 steps | 54.70 | 67s |
| K=249 | 1 step | **50.13** | **4s** |
| K=499 | 2 steps | 66.24 | 7s |
| K=749 | 3 steps | 72.34 | 10s |
| K=999 | 4 steps | 79.45 | 13s |

### Key Findings

- The four components form a progressive chain in which each addresses the problem introduced by the previous step: partial inversion enables few-step feasibility → few-step models introduce artifacts and smoothing → Gaussian blending fixes artifacts → noise injection fixes smoothing.
- Shallow inversion ($K=249$) is optimal; larger $K$ consistently worsens FID, confirming that deep DDIM inversion is incompatible with few-step models.
- 2K generation takes only 4 seconds (vs. 75 seconds for DemoFusion, a 17× speedup); 4K takes only 20 seconds (vs. 323 seconds for FreeScale, a 16× speedup).
- The SDXL + SDXL-Turbo combination performs best, though substituting SD-Turbo or Pixart-δ as the refinement model remains competitive.

## Highlights & Insights

- **A remarkably clean logical chain**: the coarse image already contains low-frequency structure → full-noise reverse diffusion is redundant → partial inversion truncates it → this naturally accommodates few-step models → few-step models introduce side effects → each is systematically remedied. The design rationale for all four components forms a fully closed loop. This analyze–truncate–patch methodology is transferable to accelerating other multi-step processes.
- **Feathering migrated from image processing to latent space**: Gaussian filter blending is a minimal yet effective design. Mean blending suffices in the multi-step regime but fails in the few-step regime; redistributing weights via a Gaussian-blurred mask resolves the issue, demonstrating that classical image processing techniques retain value in novel settings.
- **A milestone for training-free generation**: For the first time, training-free high-resolution image generation reaches practical speed (4K in 20 seconds), transforming high-resolution synthesis from an offline batch task into an interactive one.

## Limitations & Future Work

- Quality is bounded by that of SDXL-Turbo; limitations inherent to the distilled model propagate to the final output.
- Frame-by-frame application to video provides no temporal consistency guarantee.
- Compatibility with Transformer-based diffusion architectures (e.g., DiT, FLUX) has not been verified.
- The noise injection coefficient $\lambda=0.95$ is fixed; adaptive tuning may be necessary for content of varying complexity.
- 8K experiments are presented qualitatively only, without quantitative comparisons.

## Related Work & Insights

- **vs. DemoFusion**: Relies on full-noise perturbation and 50-step denoising, resulting in slow generation and object repetition artifacts. PixelRush uses DDIM inversion to preserve structure and performs one-step refinement, achieving both higher speed and better quality.
- **vs. FreeScale**: Frequency-domain operations (dilation rate modification and feature fusion) frequently introduce unnatural textures. PixelRush operates purely in the spatial domain, yielding more natural outputs.
- **vs. CutDiffusion**: Achieves only marginal acceleration by reducing the number of patches. PixelRush fundamentally restructures the reverse process, achieving order-of-magnitude speedups.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of partial inversion and few-step models constitutes a novel paradigm shift; individual components are simple but their composition is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Dual-resolution evaluation, FID/IS metrics, comprehensive ablations covering component contributions, model combinations, and inversion depth.
- Writing Quality: ⭐⭐⭐⭐⭐ — Narrative is fluid, guiding readers step by step through the necessity of each component with clear figures.
- Value: ⭐⭐⭐⭐⭐ — A 10–35× speedup that genuinely resolves the practicality problem, enabling 4K generation in 20 seconds for the first time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)
- [\[CVPR 2026\] ChordEdit: One-Step Low-Energy Transport for Image Editing](chordedit_one-step_low-energy_transport_for_image_editing.md)
- [\[CVPR 2026\] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)

</div>

<!-- RELATED:END -->
