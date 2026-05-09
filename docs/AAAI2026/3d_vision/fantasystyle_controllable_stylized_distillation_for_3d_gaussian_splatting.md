---
title: >-
  [Paper Note] FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3DGS style transfer] This paper presents FantasyStyle, the first 3DGS style transfer framework built entirely on diffusion model distillation. It introduces a Multi-View Frequency Consistency (MVFC) mechanism that suppresses low-frequency components to reduce cross-view conflicts, and designs Controllable Stylized Distillation (CSD) with negative guidance to eliminate content leakage from style images. The method surpasses existing VGG-based and diffusion-based approaches in both stylization quality and content preservation.
tags:
  - AAAI 2026
  - 3D Vision
  - 3DGS style transfer
  - diffusion model distillation
  - multi-view consistency
  - frequency analysis
  - negative guidance
date: 2026-05-08
content_hash: 3db9da865ddd9c18
---

# FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting

**Conference**: AAAI 2026
**arXiv**: [2508.08136](https://arxiv.org/abs/2508.08136)
**Code**: [https://github.com/yangyt46/FantasyStyle](https://github.com/yangyt46/FantasyStyle)
**Area**: 3D Vision / Style Transfer
**Keywords**: 3DGS style transfer, diffusion model distillation, multi-view consistency, frequency analysis, negative guidance

## TL;DR
This paper presents FantasyStyle, the first 3DGS style transfer framework built entirely on diffusion model distillation. It introduces a Multi-View Frequency Consistency (MVFC) mechanism that suppresses low-frequency components to reduce cross-view conflicts, and designs Controllable Stylized Distillation (CSD) with negative guidance to eliminate content leakage from style images. The method surpasses existing VGG-based and diffusion-based approaches in both stylization quality and content preservation.

## Background & Motivation
With the growing demand for artistic 3D content in VR/AR, 3D style transfer has become an active research area. 3D Gaussian Splatting (3DGS) has emerged as a prominent 3D representation owing to its fast rendering and high visual quality. However, existing 3DGS style transfer methods still face two core challenges:

**Multi-view inconsistency**: When 2D diffusion priors are used to guide 3D stylization, stylized results from different viewpoints frequently exhibit style conflicts (inconsistent colors and brushstrokes), leading to gradient cancellation during optimization and ultimately resulting in blurriness and geometric distortion.

**Content leakage and over-stylization**: Mainstream methods rely on VGG feature extraction, yet VGG struggles to effectively decouple style from content. This causes content information from the style image (e.g., specific object shapes) to be incorrectly transferred to the target scene, while excessive low-level texture matching produces over-stylization that obscures structural details.

This work is the **first 3DGS style transfer framework built entirely on diffusion model distillation** (without any VGG features). The core approach addresses the above issues from two dimensions: the frequency domain and the guidance mechanism.

## Method

### Overall Architecture
FantasyStyle adopts a dual-path architecture based on DDS (Delta Denoising Score): a Source Image path and a Rendered Image path. MVFC is applied to the rendered image path to enhance multi-view consistency. Style features are injected via IP-Adapter to obtain 2D stylization priors. Negative guidance suppresses content leakage. The final optimization updates only the color parameters of 3D Gaussians, keeping geometry fixed.

### Key Designs
1. **Multi-View Frequency Consistency (MVFC)**:

    - Function: Applies 3D frequency-domain filtering to multi-view latents after DDIM noise addition, improving cross-view consistency.
    - Mechanism: A 3D FFT decomposes multi-view noisy latents into low- and high-frequency components. The key observation is that low-frequency components primarily reflect view-dependent local details with poor cross-view consistency, while high-frequency components more stably capture texture features with better cross-view consistency. Therefore, all high-frequency components are retained, while low-frequency components are selectively attenuated (controlled by coefficient $\gamma$), and cross-view shared low-frequency Gaussian noise is introduced to explicitly enhance consistency.
    - Design Motivation: Inspired by FreeU and FreeInit's findings on the critical role of frequency components in image/video generation. Frequency-domain manipulation can effectively reduce cross-view style conflicts without degrading texture quality.

2. **Controllable Stylized Distillation (CSD)**:

    - Function: Designs a novel distillation loss to optimize 3D scenes using 2D stylization priors.
    - Mechanism: The failure of SDS and DDS is first analyzed — their reconstruction term $\delta_{z_t}^{recon}$ causes over-smoothing and loss of critical brushstroke details. CSD removes the reconstruction term entirely, retaining only the CFG guidance term. Additionally, the null-text condition in standard CFG is replaced by the content features of the style image as negative guidance, ensuring that the resulting stylization prior is free of content information.
    - Design Motivation: In style transfer, only color parameters need to be modified without preserving geometry or identity; the reconstruction term thus becomes a limiting factor. Negative guidance actively excludes content information in the style image that should not be transferred.

3. **IP-Adapter + ControlNet Integration**:

    - Function: Injects style information while maintaining structural consistency.
    - Mechanism: IP-Adapter-Instruct is used to separately extract style features $\text{IP}(I_r)^s$ and content features $\text{IP}(I_r)^c$ from the style image. Style features serve as positive guidance; content features serve as negative guidance. ControlNet provides structural guidance to compensate for geometric information lost during 2D prior generation.
    - Design Motivation: Text prompts alone are insufficient to precisely describe the visual characteristics of style images; IP-Adapter provides a more direct and effective means of style injection.

### Loss & Training
CSD gradient formula:

$$\nabla_\theta \mathcal{L}_{CSD} = \mathbb{E}_{t,\epsilon}[\omega(t)(\Phi^{tgt} - \Phi^{src})\frac{\partial z_t^{tgt}}{\partial \theta}]$$

where $\Phi^{tgt} = \beta(\epsilon_\phi(z_t^{tgt}, t, [\mathcal{P}, \text{IP}(I_r)^s]) - \epsilon_\phi(z_t^{tgt}, t, \text{IP}(I_r)^c))$

$\Phi^{src} = \beta(\epsilon_\phi(z_t^{src}, t, \mathcal{P}) - \epsilon_\phi(z_t^{src}, t, \varnothing))$

SDXL is used as the diffusion backbone with CFG scale $\beta=7.5$ and MVFC parameter $\gamma=0.9$. Timesteps are randomly sampled from a discrete set to simulate the DDIM denoising process. All experiments are conducted on 2×NVIDIA L20 (48GB) GPUs.

## Key Experimental Results

### Main Results

| Method | ArtFID↓ | FID_style↓ | FID_content↓ | Short LPIPS↓ | Long LPIPS↓ |
|------|---------|-----------|-------------|-------------|-------------|
| StyleGaussian | 45.31 | 398.17 | 331.53 | 0.290 | 0.542 |
| SGSST | 44.70 | 370.03 | 314.09 | 0.295 | 0.569 |
| **FantasyStyle** | **43.52** | **347.61** | **261.71** | **0.285** | **0.529** |

FantasyStyle achieves the best or second-best results on all key metrics, with FID_content reduced by approximately 50 relative to the second-best method.

### Ablation Study

| Ablation | Short LPIPS↓ | Long LPIPS↓ |
|--------|-------------|-------------|
| w/o MVFC | 0.253 | 0.587 |
| **Full Method** | **0.250** | **0.574** |

| Optimization Strategy | Visual Effect |
|---------|---------|
| SDS | Color transfer succeeds but brushstroke textures are lost; over-smoothing |
| DDS | Similar to SDS; brushstroke details lost |
| **CSD** | **Brushstroke features preserved; best stylization quality** |

The improvement from MVFC is more pronounced in long-range consistency, consistent with its design objective of enhancing multi-view consistency. Removing the reconstruction term renders the CFG scale less sensitive, reducing hyperparameter tuning complexity.

### Key Findings
- **Low-frequency components are the root cause of cross-view inconsistency**: Moderate attenuation of low frequencies slightly reduces local detail but substantially improves multi-view consistency, whereas removing high-frequency components severely degrades texture.
- **The reconstruction term in SDS/DDS is harmful for style transfer**: Since style transfer modifies only color without identity preservation, the reconstruction term causes over-smoothing and slows optimization.
- **Fundamental limitations of VGG-based methods**: VGG focuses excessively on the appearance of style images rather than extracting transferable abstract style representations, leading to content leakage. Diffusion models extract higher-level style semantics.
- **Method extensibility**: FantasyStyle can flexibly integrate other 2D style transfer methods (as shown in Figure 7), with improvements in 2D stylization quality directly translating to gains in 3D visual quality.

## Highlights & Insights
- The first 3DGS style transfer framework built entirely on diffusion model distillation, bridging the gap between 2D and 3D diffusion-based style transfer.
- Frequency-domain analysis reveals the underlying cause of multi-view inconsistency; the MVFC design is concise and elegant.
- The combination of removing the reconstruction term and applying negative guidance in CSD elegantly resolves both content leakage and over-smoothing.
- The architecture is highly extensible and can serve as a general bridge for adapting 2D style transfer methods to 3D scenes.

## Limitations & Future Work
- The SDXL-based optimization pipeline is time-consuming, though this can be mitigated with smaller models, lower resolutions, or adjusted learning rates.
- Only two baseline methods (StyleGaussian, SGSST) are included in quantitative comparisons; broader comparisons are desirable.
- Validation on dynamic scenes (e.g., 4D Gaussians) has not been conducted.
- The selection of parameter $\gamma$ in MVFC lacks an adaptive mechanism.
- Stylization quality depends substantially on the feature extraction capability of IP-Adapter.

## Related Work & Insights
- This work stands in sharp contrast to VGG-based methods: while diffusion models have largely replaced VGG in 2D style transfer, the 3DGS community still relies heavily on VGG. This paper advances the transition toward diffusion-based paradigms in 3DGS style transfer.
- The CSD design (removing the reconstruction term + negative guidance) is generalizable to other 3D editing tasks.
- The idea of controlling multi-view consistency via frequency-domain manipulation may also benefit other tasks such as 3D generation and NeRF/3DGS editing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (first purely diffusion-distillation-based 3DGS style transfer + frequency-domain consistency control)
- Experimental Thoroughness: ⭐⭐⭐⭐ (thorough ablations but limited number of baselines)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear mathematical derivations, well-motivated analysis)
- Value: ⭐⭐⭐⭐ (pioneering contribution to the 3DGS style transfer field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)
- [\[ICLR 2026\] Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer](../../ICLR2026/3d_vision/color3d_controllable_and_consistent_3d_colorization_with_personalized_colorizer.md)
- [\[CVPR 2026\] PoseMaster: A Unified 3D Native Framework for Stylized Pose Generation](../../CVPR2026/3d_vision/posemaster_a_unified_3d_native_framework_for_stylized_pose_generation.md)

</div>

<!-- RELATED:END -->
