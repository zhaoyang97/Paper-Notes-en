---
title: >-
  [Paper Note] BokehDiff: Neural Lens Blur with One-Step Diffusion
description: >-
  [ICCV 2025][3D Vision][Bokeh rendering] BokehDiff proposes a one-step inference bokeh rendering method built upon a pretrained diffusion model. It incorporates energy conservation, circle-of-confusion constraints, and self-occlusion effects via a Physics-Inspired Self-Attention (PISA) module, combined with synthetic foreground data for training, achieving significant improvements over conventional methods at depth-discontinuous regions.
tags:
  - ICCV 2025
  - 3D Vision
  - Bokeh rendering
  - diffusion models
  - one-step inference
  - physics-inspired attention
  - data synthesis
date: 2026-05-08
content_hash: 527f6ba0c9ddd83c
---

# BokehDiff: Neural Lens Blur with One-Step Diffusion

**Conference**: ICCV 2025  
**arXiv**: [2507.18060](https://arxiv.org/abs/2507.18060)  
**Code**: [https://github.com/FreeButUselessSoul/bokehdiff](https://github.com/FreeButUselessSoul/bokehdiff)  
**Area**: 3D Vision  
**Keywords**: Bokeh rendering, diffusion models, one-step inference, physics-inspired attention, data synthesis

## TL;DR
BokehDiff proposes a one-step inference bokeh rendering method built upon a pretrained diffusion model. It incorporates energy conservation, circle-of-confusion constraints, and self-occlusion effects via a Physics-Inspired Self-Attention (PISA) module, combined with synthetic foreground data for training, achieving significant improvements over conventional methods at depth-discontinuous regions.

## Background & Motivation
Bokeh is the out-of-focus blur produced by large-aperture lenses, widely used in portrait photography to highlight subjects. Due to the high cost of large-aperture optics, computational bokeh rendering has become an active research area.

**Limitations of Prior Work**:
1. Traditional neural rendering methods (DeepLens, BokehMe, etc.) rely heavily on depth estimation accuracy, producing visible artifacts at depth-discontinuous regions such as hair and fur edges.
2. Layered methods (MPIB, Dr.Bokeh) require decomposing the scene into layers and inpainting each layer, which tends to fail on complex scenes.
3. Although diffusion models possess strong generative priors, their iterative denoising process alters image content and is slow; moreover, their self-attention mechanism ignores the 3D physical properties of bokeh rendering.

**Key Insight**: The all-in-focus image is treated as a combination of the bokeh image and "noise to be removed." Without adding any noise, the transformation is achieved via a single forward pass, while a physically constrained attention module is designed alongside.

## Method

### Overall Architecture
Built on the pretrained SDXL model, the all-in-focus image is fed directly as input (without added noise) at a fixed timestep $T=499$. The U-Net LoRA weights and encoder $\mathcal{E}$ are fine-tuned while the decoder $\mathcal{D}$ is frozen. PISA modules are inserted in the downsampling layers to impose physical constraints. The final bokeh image is generated in a single forward pass.

### Key Designs

1. **One-Step Diffusion Inference**:

    - Function: Models bokeh rendering as a one-step transformation from an all-in-focus image to a bokeh image.
    - Mechanism: Leverages the diffusion model denoising formula $\hat{z}_0 = \frac{z_t - \beta_t \cdot \epsilon_\theta(z_t; c_{\text{txt}})}{\alpha_t}$, treating the latent of the all-in-focus image as $z_t$ and learning to convert it to the latent of the bokeh image. The fixed timestep $T=499$ serves as a balance point; only LoRA (rank=8) is fine-tuned.
    - Design Motivation: Diffusion models perform best at specific timesteps, and the one-step scheme avoids the accumulated errors and content changes introduced by iterative denoising. Not adding noise ensures preservation of the original structure.

2. **PISA (Physics-Inspired Self-Attention) Module**:

    - Function: Replaces standard self-attention to enforce the physical laws of bokeh rendering.
    - Mechanism encompasses three physical constraints:
        - **Energy Conservation Normalization**: Replaces key-dimension softmax with query-dimension normalization $A_{qk}^{(Q)} = \frac{\exp(A_{qk})}{\sum_s \exp(A_{sk})}$, ensuring the total contribution of each light source sums to one.
        - **Circle-of-Confusion Spatial Constraint**: Computes the circle-of-confusion radius based on focal-plane disparity differences $r_c(k) = |d_f - d_k| \cdot A$, limiting the influence range of each pixel via a differentiable soft-edge mask $C_{qk}$.
        - **Self-Occlusion Mask**: Computes sampled point positions $\tilde{P}$ via collinearity relations to determine whether the path from a light source to the target pixel is occluded, producing a visibility mask $M_{\text{vis}}$.
    - Design Motivation: The global receptive field of standard self-attention and its tendency to suppress less important pixels contradict the physical model of bokeh rendering, which requires energy-conserving weighted aggregation of contributions from neighboring pixels.

3. **Data Synthesis Pipeline**:

    - Function: Synthesizes high-quality paired training data.
    - Mechanism: Uses a pretrained T2I model to generate realistic foregrounds with transparency (rather than segmenting from photographs), combined with real backgrounds captured at small aperture. Layers at varying depths and angles are randomly placed and rendered with ray-traced bokeh effects.
    - Design Motivation: Addresses the alignment issues in real paired data and the quality bottleneck of conventional synthetic data (segmented foregrounds / 3D engines), balancing data scale and quality.

### Loss & Training
Losses are computed in pixel space (rather than latent space) and consist of four terms:
- MSE loss $\mathcal{L}_{\text{MSE}}$: basic reconstruction.
- Perceptual loss $\mathcal{L}_{\text{VGG}}$: LPIPS distance.
- Multi-scale edge loss $\mathcal{L}_{\text{edge}}$: uses an extended Sobel operator to focus on edge changes between in-focus and out-of-focus regions.
- Adversarial loss $\mathcal{L}_{\text{adv}}$: discriminator with a pretrained ConvNext backbone.

Total loss: $\mathcal{L} = \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{VGG}} \mathcal{L}_{\text{VGG}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$, where $\lambda_{\text{MSE}}=1, \lambda_{\text{VGG}}=5, \lambda_{\text{adv}}=0.5, \lambda_{\text{edge}}=1$.

Training requires approximately 12 hours on a single NVIDIA L40s GPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | BokehDiff | BokehMe | MPIB | Dr.Bokeh | DeepLens |
|--------|------|-----------|---------|------|----------|----------|
| EBB Val294 (real) | PSNR↑ | **24.652** | 24.014 | 23.334 | 23.479 | 22.703 |
| EBB Val294 | SSIM↑ | **0.8357** | 0.8134 | 0.7920 | 0.8221 | 0.7623 |
| EBB Val294 | DISTS↓ | **0.1155** | 0.1460 | 0.1581 | 0.1225 | 0.1483 |
| EBB Val294 | LPIPS↓ | **0.3737** | 0.3921 | 0.4031 | 0.3771 | 0.4191 |
| BLB Level 5 (synthetic) | LPIPS↓ | **0.0888** | 0.1404 | 0.2561 | 0.4539 | 0.2976 |
| User Study | Accuracy↑ | **4.42** | 3.81 | 1.83 | 3.41 | 1.55 |
| User Study | Realism↑ | **4.37** | 3.93 | 1.89 | 3.38 | 1.68 |
| User Study | Preference↑ | **4.56** | 4.03 | 2.04 | 3.64 | 1.96 |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Note |
|------|-------|-------|--------|------|
| w/o $\mathcal{L}_{\text{adv}}$ | 24.623 | 0.8322 | 0.3768 | Adversarial loss contributes marginally |
| w/o $\mathcal{L}_{\text{VGG}}$ | 24.285 | 0.8196 | 0.4218 | Perceptual loss is highly important |
| w/o $\mathcal{L}_{\text{edge}}$ | 24.628 | 0.8346 | 0.3785 | Edge loss contributes noticeably |
| w/o CoC | 22.217 | 0.6881 | 0.4280 | CoC constraint is critical |
| w/o SoftmaxQ | 24.468 | 0.8325 | 0.3800 | Energy conservation normalization is effective |
| w/o occlusion | 24.399 | 0.8291 | 0.3808 | Self-occlusion contributes |
| $T=249$ | 24.646 | 0.8335 | 0.3781 | Timestep selection is robust |
| $T=749$ | 24.481 | 0.8319 | 0.3838 | Extreme timestep degrades performance |
| **Full model** | **24.652** | **0.8357** | **0.3737** | All components synergize optimally |

### Key Findings
- The circle-of-confusion (CoC) constraint is the most critical component of the PISA module; removing it causes a 2.4 dB drop in PSNR and a 0.15 drop in SSIM.
- In a user study with 50 photography enthusiasts, BokehDiff substantially outperforms all baselines across accuracy, realism, and preference.
- BokehDiff is robust to depth estimation errors, exhibiting minimal performance degradation under disparity map erosion/dilation experiments.
- The method supports arbitrary focal plane adjustment, enabling the generation of complete focal stacks.

## Highlights & Insights
- The one-step diffusion inference paradigm is innovative: no noise is added, no iterative denoising is performed, and the difference in image quality is reinterpreted as "noise."
- The PISA module elegantly integrates optical imaging physics (energy conservation, circle of confusion, occlusion) into the self-attention mechanism.
- The data synthesis pipeline leverages a T2I model to generate foregrounds with transparency, resolving the long-standing tension between data quality and scale.
- One-step inference opens the possibility of real-time rendering.

## Limitations & Future Work
- The VAE decoder still introduces inevitable changes to subtle structures.
- The current method is based on SDXL; replacing it with a backbone featuring less information compression (e.g., an improved VAE) may yield further gains.
- Errors in depth estimation itself remain an upstream bottleneck, though BokehDiff demonstrates considerable robustness to this.
- Support for non-circular aperture shapes (e.g., starburst effects) is not discussed.

## Related Work & Insights
- Inherits the conceptual lineage of one-step diffusion (e.g., SinSR) and "noise-free denoising."
- The physically constrained attention mechanism is generalizable to other image processing tasks that require physical priors.
- Insight: The prior of pretrained diffusion models can be adapted to specific physical transformation tasks via lightweight fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐ Recasts bokeh rendering as a one-step diffusion problem; the physical constraint design of PISA is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on both real and synthetic datasets; user study is of substantial scale.
- Writing Quality: ⭐⭐⭐⭐ Physical motivation is clearly articulated; mathematical derivations are complete.
- Value: ⭐⭐⭐⭐ Provides a new diffusion-based paradigm for computational photography with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] From One to More: Contextual Part Latents for 3D Generation](from_one_to_more_contextual_part_latents_for_3d_generation.md)
- [\[CVPR 2026\] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](../../CVPR2026/3d_vision/ada3drift_adaptive_trainingtime_drifting_for_onest.md)
- [\[ICCV 2025\] From Image to Video: An Empirical Study of Diffusion Representations](from_image_to_video_an_empirical_study_of_diffusion_representations.md)
- [\[ICCV 2025\] Textured 3D Regenerative Morphing with 3D Diffusion Prior](textured_3d_regenerative_morphing_with_3d_diffusion_prior.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
