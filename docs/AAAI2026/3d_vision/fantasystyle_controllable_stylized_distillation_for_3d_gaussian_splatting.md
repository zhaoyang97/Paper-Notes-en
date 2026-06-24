---
title: >-
  [Paper Note] FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3DGS Style Transfer] This paper proposes FantasyStyle, the first 3DGS style transfer framework entirely based on diffusion model distillation. It utilizes a Multi-view Frequency Consistency (MVFC) mechanism to suppress low-frequency components and reduce inconsistencies between perspectives, and designs Controllable Stylized Distillation (CSD) to introduce negative guidance, eliminating content leakage from style images. It outperforms existing VGG and…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3DGS Style Transfer"
  - "Diffusion Model Distillation"
  - "Multi-view Consistency"
  - "Frequency Analysis"
  - "Negative Guidance"
date: 2026-05-08
content_hash: f9add36722c3011a
---

# FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting

**Conference**: AAAI 2026  
**arXiv**: [2508.08136](https://arxiv.org/abs/2508.08136)  
**Code**: [https://github.com/yangyt46/FantasyStyle](https://github.com/yangyt46/FantasyStyle)  
**Area**: 3D Vision / Style Transfer  
**Keywords**: 3DGS Style Transfer, Diffusion Model Distillation, Multi-view Consistency, Frequency Analysis, Negative Guidance

## TL;DR
This paper proposes FantasyStyle, the first 3DGS style transfer framework entirely based on diffusion model distillation. It utilizes a Multi-view Frequency Consistency (MVFC) mechanism to suppress low-frequency components and reduce inconsistencies between perspectives, and designs Controllable Stylized Distillation (CSD) to introduce negative guidance, eliminating content leakage from style images. It outperforms existing VGG and diffusion-based methods in both stylization quality and content preservation.

## Background & Motivation
With the growing demand for artistic 3D content in VR/AR, 3D style transfer has become a research hotspot. 3D Gaussian Splatting (3DGS) has emerged as a promising 3D representation due to its fast rendering and high visual quality. However, existing 3DGS style transfer methods still face two core challenges:

**Multi-view Inconsistency**: When driving 3D stylization with 2D diffusion priors, stylized results from different perspectives often experience style conflicts (inconsistencies in color and brushstrokes). This leads to mutual cancellation during optimization, ultimately resulting in blurriness and geometric distortion.

**Content Leakage and Over-Stylization**: Mainstream methods rely on VGG feature extraction, but VGG struggles to effectively disentangle style and content. This causes the content information of the style image (such as specific object shapes) to be incorrectly transferred to the target scene, while over-matching of low-level textures produces over-stylization, concealing structural details.

This paper presents the **first 3DGS style transfer framework purely based on diffusion model distillation** (without using any VGG features). The core Key Insight is to address the aforementioned issues from two dimensions: the frequency domain and the guidance mechanism.

## Method

### Overall Architecture
FantasyStyle is based on a dual-path architecture of DDS (Delta Denoising Score): the Source Image path and the Rendered Image path. An MVFC mechanism is introduced to the rendered image path to enhance multi-view consistency. Style features are injected via IP-Adapter to obtain 2D stylization priors, and negative guidance is used to suppress content leakage. Finally, the color parameters of 3D Gaussians are optimized via CSD (while keeping the geometry fixed).

### Key Designs
1. **Multi-View Frequency Consistency (MVFC)**:

    - **Function**: Performs 3D frequency-domain filtering on multi-view latents after DDIM noise addition to improve consistency across views.
    - **Mechanism**: Uses 3D FFT to decompose multi-view noise latents into low-frequency and high-frequency components. A key observation is that low-frequency components mainly reflect view-dependent local details, exhibiting poor cross-view consistency, whereas high-frequency components more stably capture texture features, exhibiting good cross-view consistency. Therefore, all high-frequency components are retained, low-frequency components are selectively attenuated (controlled by a coefficient $\gamma$), and a cross-view shared low-frequency Gaussian noise is introduced to explicitly enhance consistency.
    - **Design Motivation**: Inspired by the findings of FreeU and FreeInit regarding the key role of frequency components in image/video generation. Operations in the frequency domain can effectively reduce style conflicts between views without compromising textures.

2. **Controllable Stylized Distillation (CSD)**:

    - **Function**: Designs a new distillation loss function to optimize 3D scenes using 2D stylization priors.
    - **Mechanism**: First analyzes why SDS and DDS fail—their reconstruction terms $\delta_{z_t}^{recon}$ cause over-smoothed outputs and loss of key brushstroke details. CSD addresses this by directly removing the reconstruction term and keeping only the CFG guidance term. Additionally, the empty-text condition in standard CFG is replaced with content features of the style image as negative guidance, ensuring the generated stylized prior is free of content information.
    - **Design Motivation**: In style transfer tasks, only color parameters need to be modified without involving geometry or identity preservation, making the reconstruction term a limiting factor instead; negative guidance actively excludes unwanted content information from the style image.

3. **IP-Adapter + ControlNet Integration**:

    - **Function**: Injects style information while maintaining structural consistency.
    - **Mechanism**: Uses IP-Adapter-Instruct to extract style features $\text{IP}(I_r)^s$ and content features $\text{IP}(I_r)^c$ from the style image; style features are used for positive guidance, and content features are used for negative guidance. ControlNet guides structural information to compensate for the loss of geometric details when generating 2D priors.
    - **Design Motivation**: Pure text prompts are insufficient to describe the visual features of style images accurately, and IP-Adapter provides a more direct and effective way of injecting style.

### Loss & Training
CSD gradient formula:

$$\nabla_\theta \mathcal{L}_{CSD} = \mathbb{E}_{t,\epsilon}[\omega(t)(\Phi^{tgt} - \Phi^{src})\frac{\partial z_t^{tgt}}{\partial \theta}]$$

where $\Phi^{tgt} = \beta(\epsilon_\phi(z_t^{tgt}, t, [\mathcal{P}, \text{IP}(I_r)^s]) - \epsilon_\phi(z_t^{tgt}, t, \text{IP}(I_r)^c))$

$\Phi^{src} = \beta(\epsilon_\phi(z_t^{src}, t, \mathcal{P}) - \epsilon_\phi(z_t^{src}, t, \varnothing))$

SDXL is utilized as the backbone diffusion model, with a CFG scale of $\beta=7.5$ and an MVFC parameter of $\gamma=0.9$. Random sampling is performed from a discrete set of timesteps (simulating the DDIM denoising process). All experiments are conducted on 2×NVIDIA L20 (48GB) GPUs.

## Key Experimental Results

### Main Results

| Method | ArtFID↓ | FID_style↓ | FID_content↓ | Short LPIPS↓ | Long LPIPS↓ |
|------|---------|-----------|-------------|-------------|-------------|
| StyleGaussian | 45.31 | 398.17 | 331.53 | 0.290 | 0.542 |
| SGSST | 44.70 | 370.03 | 314.09 | 0.295 | 0.569 |
| **FantasyStyle** | **43.52** | **347.61** | **261.71** | **0.285** | **0.529** |

FantasyStyle achieves the best or second-best results across all key metrics, with FID_content reduced by approximately 50 compared to the second-best method.

### Ablation Study

| Ablation Item | Short LPIPS↓ | Long LPIPS↓ |
|--------|-------------|-------------|
| w/o MVFC | 0.253 | 0.587 |
| **Full Method** | **0.250** | **0.574** |

| Optimization Strategy | Visual Effect |
|---------|---------|
| SDS | Color transfer is successful but brushstroke textures are lost, leading to over-smoothing |
| DDS | Similar to SDS, losing brushstroke details |
| **CSD** | **Preserves brushstroke features, achieving the best stylization quality** |

The improvement from MVFC is more significant in long-term consistency (specifically designed for multi-view consistency). Removing the reconstruction term makes the CFG scale insensitive, reducing the complexity of hyperparameter tuning.

### Key Findings
- **Low-frequency components are the main culprit of view inconsistency**: Moderately attenuating low frequencies only slightly reduces local details but significantly improves multi-view consistency, whereas removing high-frequency components severely damages textures.
- **The reconstruction term in SDS/DDS is harmful in style transfer**: Because style transfer only modifies color and does not involve identity preservation, the reconstruction term causes over-smoothing and slows down optimization.
- **Fundamental limitation of VGG methods**: VGG over-focuses on the appearance of the style image rather than extracting a transferable abstract style representation, leading to content leakage. Diffusion models can extract higher-level style semantics.
- **Method scalability**: FantasyStyle can flexibly integrate other 2D style transfer methods (as shown in Figure 7). The improvement in 2D stylization quality directly translates into an enhancement of 3D visual quality.

## Highlights & Insights
- The first 3DGS style transfer framework entirely based on diffusion model distillation, bridging the gap in 2D-to-3D diffusion style transfer.
- Frequency domain analysis reveals the underlying cause of multi-view inconsistency, with a simple and elegant MVFC design.
- The combination of CSD removing the reconstruction term plus negative guidance cleverly solves both content leakage and over-smoothing issues.
- The architectural design shows excellent scalability, serving as a general bridge for extending 2D style transfer to 3D scenes.

## Limitations & Future Work
- The SDXL-based optimization process is time-consuming, but this can be mitigated by smaller models, lower resolutions, or adjusting the learning rate.
- There are only two baseline methods for quantitative comparison (StyleGaussian and SGSST); more comparisons could be added.
- It has not been verified on dynamic scenes (e.g., 4D Gaussians).
- The selection of the $\gamma$ parameter in MVFC lacks an adaptive mechanism.
- Stylization quality heavily relies on the feature extraction capability of the IP-Adapter.

## Related Work & Insights
- Sharp contrast with VGG methods: While VGG has been replaced by diffusion methods in 2D style transfer, it is still widely used in the 3DGS domain. Ours promotes the transition of 3DGS style transfer toward the diffusion paradigm.
- The design of CSD (removing reconstruction terms + negative guidance) can be generalized to other 3D editing tasks.
- The idea of controlling multi-view consistency in the frequency domain may also be valuable for other tasks such as 3D generation and NeRF/3DGS editing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First pure-diffusion-distilled 3DGS style transfer + frequency-domain consistency control)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Thorough ablations but limited number of baselines)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear mathematical derivations, well-analyzed motivations)
- Value: ⭐⭐⭐⭐ (Of pioneering significance in the field of 3DGS style transfer)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[CVPR 2025\] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](../../CVPR2025/3d_vision/dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)
- [\[CVPR 2026\] EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images](../../CVPR2026/3d_vision/ecosplat_efficiency-controllable_feed-forward_3d_gaussian_splatting_from_multi-v.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
