---
title: >-
  [Paper Note] NeAR: Coupled Neural Asset–Renderer Stack
description: >-
  [CVPR 2026][3D Vision][Neural rendering] NeAR proposes jointly designing neural asset creation and neural rendering as a coupled stack. By introducing illumination-homogenized structured 3D latents (LH-SLAT) to remove baked lighting from input images, and employing an illumination-aware neural decoder for real-time synthesis of relightable 3D Gaussian fields, NeAR surpasses existing methods across four tasks: forward rendering, reconstruction, relighting, and novel-view relighting.
tags:
  - CVPR 2026
  - 3D Vision
  - Neural rendering
  - illumination homogenization
  - 3D Gaussian splatting
  - relighting
  - coupled asset–renderer design
date: 2026-05-08
content_hash: 728ebe18d37177f9
---

# NeAR: Coupled Neural Asset–Renderer Stack

**Conference**: CVPR 2026
**arXiv**: [2511.18600](https://arxiv.org/abs/2511.18600)
**Code**: [https://near-project.github.io/](https://near-project.github.io/) (project page)
**Area**: 3D Vision / Diffusion Models
**Keywords**: Neural rendering, illumination homogenization, 3D Gaussian splatting, relighting, coupled asset–renderer design

## TL;DR
NeAR proposes jointly designing neural asset creation and neural rendering as a coupled stack. By introducing illumination-homogenized structured 3D latents (LH-SLAT) to remove baked lighting from input images, and employing an illumination-aware neural decoder for real-time synthesis of relightable 3D Gaussian fields, NeAR surpasses existing methods across four tasks: forward rendering, reconstruction, relighting, and novel-view relighting.

## Background & Motivation

1. **Background**: Neural graphics currently follows two independent tracks—neural asset creation (synthesizing 3D assets via generative models) and neural rendering (mapping assets to images). These are typically developed in isolation: asset generation assumes a fixed renderer, while renderers are trained for static asset distributions.

2. **Limitations of Prior Work**: (a) PBR decomposition-based methods (e.g., Hunyuan3D) are prone to material misclassification (e.g., identifying wood as metal); decomposition errors are amplified through nonlinear rendering pipelines, leading to baked shadows and illumination inconsistencies. (b) Diffusion-based 2D relighting methods (e.g., DiLightNet, IC-Light) lack 3D consistency and incur high computational cost. (c) Existing SLAT representations blindly encode appearance including shadows and specular highlights, making them unsuitable for relighting.

3. **Key Challenge**: Shadows, highlights, and inter-reflections are inherently entangled with geometry and material in 3D assets. Explicit PBR inverse decomposition is fragile in real-world scenarios, while fully black-box neural generation lacks controllability.

4. **Goal**: How to achieve high-quality, controllable single-image relightable 3D generation without relying on unstable PBR inversion?

5. **Key Insight**: The authors propose a *homogenize-then-synthesize* strategy—co-designing the asset representation and rendering process so that they form a robust "contract" through a shared illumination-homogenized latent space.

6. **Core Idea**: Jointly design illumination-homogenized 3D latent representations and an illumination-aware neural renderer, forming a coupled asset–renderer stack that replaces the conventional paradigm of decoupled asset generation and independent rendering.

## Method

### Overall Architecture
NeAR proceeds in two stages. **Stage 1** fine-tunes a rectified-flow model with LoRA to lift a single input image under arbitrary illumination into an illumination-homogenized SLAT (LH-SLAT), removing baked shadows and unstable highlights. **Stage 2** uses a feed-forward decoder to synthesize a relightable 3D Gaussian splatting field conditioned on target illumination and viewpoint from the LH-SLAT. The entire pipeline requires no per-object optimization and supports real-time inference.

### Key Designs

1. **Illumination-Homogenized Structured 3D Latent (LH-SLAT)**:
    - **Function**: Maps inputs under arbitrary illumination to a canonical illumination space, yielding an illumination-invariant asset representation.
    - **Mechanism**: Defines homogeneous illumination $E_h$ as white uniform ambient light. A pretrained SLAT flow model $f_s$ first generates a shaded SLAT $Z_s$ from the input image; a LoRA-adapted model $f_\theta$ then guides $Z_s$ toward an illumination-homogenized $Z_{\text{lh}} = f_\theta(Z_s, I_{\text{in}})$ in sparse voxel space. Ground-truth LH-SLATs are obtained by rendering 3D assets under homogeneous illumination, with inputs rendered under random illumination. For highly reflective materials, an optional Base Color SLAT $Z_{\text{bc}}$ is extracted and concatenated.
    - **Design Motivation**: Direct PBR decomposition is an ill-posed problem. Learning illumination removal in the latent space via flow models avoids the instability of explicit material decomposition. LH-SLAT retains essential geometry–material–illumination interaction information, providing a stable foundation for downstream rendering.

2. **Illumination Tokenizer**:
    - **Function**: Encodes HDR environment maps into compact illumination condition tokens.
    - **Mechanism**: Decomposes the environment map into an LDR tone-mapped image $\mathbf{E}_{\text{ldr}}$, a normalized log-intensity map $\mathbf{E}_{\text{log}}$, and a camera-space direction encoding $\mathbf{E}_{\text{dir}}$. ConvNeXt extracts visual features; NeRF positional encoding processes $\mathbf{E}_{\text{dir}}$; spatial cross-attention fuses directional information with visual features; self-attention then produces illumination condition tokens $C_L \in \mathbb{R}^{4096 \times 768}$.
    - **Design Motivation**: Compared to compressing the entire environment map with a VAE, explicitly embedding directional information makes illumination direction editable when switching viewpoints.

3. **Intrinsic-Aware Decoder (IAD) + Lighting-Aware Decoder (LAD)**:
    - **Function**: IAD decodes view-independent, illumination-invariant intrinsic features; LAD injects illumination and viewpoint conditions to produce final illumination-dependent features.
    - **Mechanism**: **IAD** processes LH-SLAT with Transformer and shifted window attention, augmented by 16 register tokens that capture global context via global cross-attention, outputting intrinsic features $\boldsymbol{h}$. **LAD** first computes view embeddings (ray distance + ray direction NeRF positional encoding) and adds them to $\boldsymbol{h}$ to obtain $\boldsymbol{h}^v$; stacked cross-attention blocks then inject illumination conditions $C_L$ to produce illumination-aware features $\boldsymbol{h}^e$.
    - **Design Motivation**: Separating decoding into illumination-independent and illumination-dependent stages ensures stable intrinsic attributes while enabling flexible illumination control. Replacing spherical harmonics with explicit viewpoint injection better models view-dependent specular highlights.

4. **Neural 3D Gaussian Splatting**:
    - **Function**: Regresses 3DGS parameters from intrinsic and illumination-aware features.
    - **Mechanism**: Intrinsic features $\boldsymbol{h}$ are decoded into illumination-independent parameters including position offsets, base color, roughness, metalness, scale, rotation, and opacity; illumination features $\boldsymbol{h}^e$ are decoded into 48-dimensional color features, illumination scale, and shadow. A shallow MLP combines normal positional encoding with color features to predict radiance values; differentiable rasterization yields HDR predicted images.
    - **Design Motivation**: Partitioning Gaussian parameters into intrinsic and illumination-related groups achieves disentangled material–illumination representation.

### Loss & Training

**Stage 1**: Conditional flow matching loss $\mathcal{L}_{\text{stage1}} = \mathbb{E}\|\boldsymbol{v}_\theta(\boldsymbol{z}, Z_s, I_{\text{in}}, t) - (\boldsymbol{\epsilon} - \boldsymbol{z}_0)\|_2^2$.

**Stage 2**: HDR reconstruction loss (log-space L1 + LPIPS + D-SSIM) $\mathcal{L}_{\text{hdr}}$ + PBR material auxiliary supervision $\mathcal{L}_{\text{pbr}}$ + shadow supervision $\mathcal{L}_{\text{shadow}}$.

Training data: 87K 3D assets with PBR textures + 2K 4K-resolution HDR environment maps, rendered with Blender EEVEE Next.

## Key Experimental Results

### Main Results (PSNR↑ comparison across four tasks)

| Task | Method | ADT | DTC | Objaverse | Glossy Syn. |
|------|------|-----|-----|-----------|-------------|
| G-buffer forward rendering | DiffusionRenderer | 24.41 | 27.16 | 27.09 | 25.46 |
| | **NeAR** | **29.15** | **31.59** | **32.23** | **30.47** |
| Random illumination reconstruction | DiLightNet | 21.11 | 23.53 | 25.65 | 24.09 |
| | **NeAR** | **22.89** | **24.68** | **26.53** | **25.32** |
| Unknown illumination relighting | DiffusionRenderer | 21.91 | 22.99 | 23.75 | 22.13 |
| | **NeAR** | **21.95** | **23.47** | **24.38** | **22.61** |
| Novel-view relighting | Hunyuan3D-2.1 | 22.30 | 24.89 | 25.47 | 22.26 |
| | **NeAR** | **22.87** | **25.53** | **25.97** | **22.94** |

### Ablation Study

| Input SLAT Type | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------------|-------|-------|--------|
| Shaded SLAT | 28.95 | 0.9281 | 0.0813 |
| Base Color SLAT | 30.38 | 0.9541 | 0.0564 |
| LH-SLAT | 32.02 | 0.9631 | 0.0494 |
| **LH + Base Color** | **32.54** | **0.9649** | **0.0442** |

| LAD Layers (IAD=12) | PSNR | FPS |
|-------------------|------|-----|
| 1 layer | 31.56 | 48 |
| 3 layers | 32.35 | 38 |
| **6 layers** | **32.54** | **30** |
| 9 layers | 32.56 | 23 |

### Key Findings
- LH-SLAT surpasses Shaded SLAT by over 3 dB PSNR, confirming the necessity of illumination homogenization.
- Combining LH-SLAT with Base Color SLAT yields the best results; Base Color SLAT supplements information for highly reflective materials.
- LAD with 6 layers provides the optimal performance–speed trade-off (PSNR 32.54, 30 FPS).
- Injecting viewpoint information before baking illumination (architectures c+d+g in Figure 9) significantly outperforms the reverse order.
- HY3D-2.1 incorrectly classifies wood as metal (erroneous metalness maps), whereas NeAR's LH-SLAT correctly recovers material properties.

## Highlights & Insights
- **Coupled asset–renderer design paradigm**: Rather than treating asset generation and rendering as independent components, NeAR establishes a "contract" between them through a shared latent space—a paradigm with broad implications for the design of neural graphics stacks.
- **Illumination homogenization strategy**: By learning illumination removal in latent space rather than performing explicit PBR decomposition, the method elegantly circumvents the ill-posedness of inverse rendering.
- **Texture style transfer application**: LH-SLAT accepts stylized image inputs, enabling semantically consistent style transfer combined with photorealistic relighting, demonstrating the flexibility of the representation.

## Limitations & Future Work
- The method still struggles with transparent objects (e.g., helmets); the neural renderer partially mitigates but does not fully resolve this issue.
- Training requires multi-illumination rendering of large quantities of 3D assets, resulting in high data preparation costs.
- Evaluation is limited to static objects; dynamic scenes and human bodies remain unexplored.
- The real-time performance of 30 FPS may still be insufficient for certain applications.

## Related Work & Insights
- **vs. Trellis**: Trellis uses SLAT but blindly encodes illumination; NeAR proposes LH-SLAT to explicitly remove illumination, yielding a more stable representation suited for relighting.
- **vs. DiffusionRenderer**: DiffusionRenderer performs 2D rendering from G-buffers and lacks 3D structural information; NeAR's 3D Gaussian field achieves greater accuracy in shadow and specular detail.
- **vs. HY3D-2.1**: HY3D's decoupled PBR decomposition leads to material estimation errors (e.g., incorrect metalness), whereas NeAR avoids the fragility of explicit decomposition.

## Rating
- Novelty: ⭐⭐⭐⭐ The coupled asset–renderer design concept is original, and the LH-SLAT representation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four sub-tasks, four datasets, multi-method comparisons, ablation studies, and application demonstrations are all comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with thorough comparative analysis.
- Value: ⭐⭐⭐⭐ Offers important inspiration for design paradigms in neural rendering and 3D generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow](easy3e_feed-forward_3d_asset_editing_via_rectified_voxel_flow.md)
- [\[CVPR 2026\] Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting](indoor_asset_detection_in_large_scale_360_drone-captured_imagery_via_3d_gaussian.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](neural_gabor_splatting.md)
- [\[CVPR 2026\] NTK-Guided Implicit Neural Teaching](ntk-guided_implicit_neural_teaching.md)
- [\[CVPR 2026\] Unblur-SLAM: Dense Neural SLAM for Blurry Inputs](unblur-slam_dense_neural_slam_for_blurry_inputs.md)

</div>

<!-- RELATED:END -->
