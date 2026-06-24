---
title: >-
  [Paper Note] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting
description: >-
  [CVPR 2025][Image Generation][relighting] Proposes LumiNet, which injects the latent intrinsic features (a 128-dimensional albedo-like representation) of the source image and the latent extrinsic lighting code (16-dimensional) of the target image into a modified ControlNet. This enables image-only indoor scene-level light transfer, capturing complex effects such as specular highlights, shadows, and indirect illumination.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "relighting"
  - "latent intrinsics"
  - "diffusion model"
  - "ControlNet"
  - "StyleGAN"
  - "indoor scene"
  - "light transfer"
  - "cross-attention"
date: 2026-05-08
content_hash: 5c7ec2c73d9091ef
---

# LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting

**Conference**: CVPR 2025  
**arXiv**: [2412.00177](https://arxiv.org/abs/2412.00177)  
**Code**: [https://luminet-relight.github.io](https://luminet-relight.github.io)  
**Area**: Image Generation  
**Keywords**: relighting, latent intrinsics, diffusion model, ControlNet, StyleGAN, indoor scene, light transfer, cross-attention

## TL;DR

Proposes LumiNet, which injects the latent intrinsic features (a 128-dimensional albedo-like representation) of the source image and the latent extrinsic lighting code (16-dimensional) of the target image into a modified ControlNet. This enables image-only indoor scene-level light transfer, capturing complex effects such as specular highlights, shadows, and indirect illumination.

## Background & Motivation

**Background**: Image-level relighting has critical applications in film production, architectural visualization, and mixed reality. Diffusion models combined with ControlNet have demonstrated powerful conditional generation capabilities.

**Limitations of Prior Work**: (1) **Traditional inverse rendering**: Requires precise 3D reconstruction and material decomposition, which is computationally expensive and prone to error accumulation; (2) **StyLitGAN**: Can manipulate lighting within the StyleGAN latent space but fails to generalize to real-world images; (3) **Latent Intrinsics (Zhang et al.)**: Can decompose latent intrinsic/extrinsic representations but cannot generalize to complex, arbitrary scenes; (4) **IC-Light**: Excels at portrait relighting but struggles to handle full scenes; (5) **RGB↔X**: Requires G-buffer inputs (such as normals, depth, etc.), depending on explicit geometric information.

**Key Challenge**: Indoor scene light transfer requires understanding light source positions, material interactions, and indirect illumination — information that is highly scene-specific, making it difficult for traditional explicit modeling to cover all cases.

**Key Insight**: Operating in the latent space — leveraging a pre-trained latent intrinsic decomposition model to extract geometry/albedo-invariant features and lighting codes, thereby formulating light transfer as a conditional image generation problem.

## Method

### Overall Architecture

Three parts:
1. **Data Preparation**: Variational-StyLitGAN generates synthetic training pairs, which are complemented by real-world datasets.
2. **Latent ControlNet**: Processes the latent intrinsic features of the source image to preserve geometry and material.
3. **Lighting Adaptor**: Injects the target lighting code via cross-attention.

### Key Designs

**1. Variational-StyLitGAN Data Strategy**
- **Function**: Addresses the scarcity of training data — pairs of real-world scenes under different illumination are extremely difficult to acquire.
- **Mechanism**: Uses a ConvNeXt variational encoder to map real LSUN bedroom images into the StyLitGAN latent space, generating 7 lighting variations per scene on a frozen pre-trained generator:
  $$\mathcal{L} = \underbrace{\text{MSE}(\mathbf{x}, \hat{\mathbf{x}}) + \text{LPIPS}(\mathbf{x}, \hat{\mathbf{x}})}_{\mathcal{L}_{rec}} + \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| \mathcal{N}(0,I))}_{\mathcal{L}_{KL}}$$
- **CLIP Filtering**: Screens ~1K high-quality samples using keywords such as "photo-realistic", "good lighting", and "illumination".
- **Design Motivation**: Random sampling in the original StyLitGAN causes mode collapse (where outputs are nearly identical every 10–20 iterations). Variational mapping introduces real-world image diversity to overcome this issue, reducing the FID from 47.99 to 35.81.

**2. Latent Intrinsic ControlNet (Latent Intrinsic Control)**
- **Function**: Redesigns ControlNet to operate in the latent space rather than the image space, processing representations from **two different images**.
- **Mechanism**:
    - Extracts latent intrinsic features $\mathcal{A}_o \in \mathbb{R}^{H \times W \times 128}$ (albedo-like, light-invariant) from the source image.
    - Extracts the lighting extrinsic code $\mathcal{I}_{L_t} \in \mathbb{R}^{16}$ from the target image.
    - Broadcasts $\mathcal{I}_{L_t}$ to the spatial dimensions and concatenates it with $\mathcal{A}_o$ to form a $\mathbb{R}^{H \times W \times 144}$ tensor, which is then projected via convolution to serve as the ControlNet input.
- **vs. Traditional ControlNet**: Traditional ControlNet processes condition maps from a single scene; LumiNet processes latent representations of two different scenes — preserving the source geometry/albedo while transferring the target lighting.

**3. Lighting Adaptor Network (Lighting Adaptor)**
- **Function**: Injects the low-dimensional target lighting code into the cross-attention layers of the pre-trained diffusion model.
- **Mechanism**: An MLP (3072 $\rightarrow$ 4096 $\rightarrow$ 4096 $\rightarrow$ 4096 $\rightarrow$ 3072) maps the lighting code to $\mathcal{I}_{E_t} \in \mathbb{R}^{3 \times 1024}$, substituting the text embedding for input into cross-attention. No text prompts are used during training.
- **Design Motivation**: Cross-attention provides a control pathway for the global behavior of the diffusion model, through which the lighting code affects second-order effects (e.g., tabletop glare/reflections, indirect illumination).

### Loss & Training

$$\mathcal{L}_\text{Lumi} = \|\epsilon - \theta(\epsilon(S^{L_t})_t, t, \{\mathcal{A}_o, \mathcal{I}_{L_t'}\}, \mathcal{I}_{E_t}, \epsilon(S^{L_o}))\|_2^2$$

Standard latent diffusion denoising loss. Only the ControlNet and cross-attention layers are trained, while the remaining parameters are frozen.

### Inference Enhancement

- **Bypass Decoder**: Replaces the default VAE decoder and is fine-tuned to better preserve geometric details.
- **Nearest-Neighbor Seed Selection**: Selects the optimal seed from multi-seed generation based on the target lighting code.
- **Flow-Based Clean Up**: Employs rectified-flow inversion ($\eta=0.99$) to remove artifacts.

## Key Experimental Results

### Main Results — Quantitative Evaluation on MIIW

| Method | Label Requirement | RMSE↓ | SSIM↑ |
|---|---|---|---|
| SA-AE | Light | 0.232 | 0.559 |
| Latent-Intrinsic | - | 0.222 | 0.571 |
| RGB↔X (same scene) | G-Buffer | 0.340 | 0.350 |
| **LumiNet** | **-** | **0.240** | **0.527** |

As a general-purpose model (trained on diverse data), LumiNet achieves performance close to that of the task-specific MIIW models.
  
### User Study — Real-world Evaluation

| Method | Normal-Consistent AE↓ | Image Quality↓ | Lighting Quality↓ | Prompt Alignment↓ |
|---|---|---|---|---|
| RGB↔X | 3.14 | 2.21 | 2.88 | 2.70 |
| IC-Light-v2 | 3.42 | 3.06 | 2.57 | 2.74 |
| Latent-Intrinsic | 3.61 | 2.24 | 2.52 | 2.40 |
| **LumiNet** | **2.74** | **1.71** | **1.30** | **1.40** |

Ranked first across all metrics in a user study with 31 participants. The median angular error for normal consistency is <3°, showing optimal geometry preservation.

### Ablation Study

- **W/o Variational-StyLitGAN data**: Fails to learn scene-level lighting effects (e.g., turning lamps on/off).
- **W/o latent intrinsic conditions**: Degenerates into standard ControlNet, only altering the average color rather than the lighting.
- **W/o adapter + cross-attention fine-tuning**: Loses second-order effects (e.g., glossy reflections on table surfaces).
- **W/o Flow Inversion**: Produces reasonable relighting but with artifacts.
- **Var-StyLitGAN Ablation**: FID 47.98 $\rightarrow$ 37.07 $\rightarrow$ 35.81 (adding variational mapping + CLIP filtering).

### Key Findings

1. **Latent vs. Pixel space conditional control**: Latent intrinsic features are more robust than explicit albedo/normals.
2. **Cross-scene transfer despite training only on same-scene pairs**: The abstraction of the latent space enables the model to learn scene-agnostic lighting manipulation capabilities.
3. **Synthetic data is key to scene-level effects**: Without the Var-StyLitGAN data, the model fails to learn light source controls like lamp switches.
4. **Cross-attention is required for second-order lighting effects**: Spatial conditioning in ControlNet is insufficient; the lighting code must influence indirect illumination through a global attention pathway.

## Highlights & Insights

1. **First scene-level image relighting diffusion method**: Goes beyond the limitations of portrait or object-level relighting to directly handle complete indoor scenes.
2. **Clever utilization of latent intrinsic representations**: Instead of performing explicit intrinsic decomposition (albedo/roughness/normal), Ours leverages the latent decomposition of a pre-trained model, avoiding error accumulation.
3. **Modified ControlNet with dual-image inputs**: While standard ControlNet handles same-scene conditions, LumiNet handles cross-scene combinations of "source geometry + target lighting" for the first time.
4. **From same-scene training to cross-scene generalization**: Generalization ability stems from the abstraction of the latent space — in the pre-trained intrinsic space, lighting and content are already explicitly decoupled.

## Limitations & Future Work

1. Fails to recognize light sources that are extremely small or facing away from the camera (failure cases in Figure 7).
2. Cannot transfer extreme color temperature variations (e.g., colored lights in a karaoke room).
3. Does not control light intensity — output brightness may be inconsistent with the target.
4. Inference requires multiple seeds + nearest neighbor selection + Flow Inversion post-processing, making the pipeline relatively heavy.
5. Dependent on the quality ceiling of the pre-trained Latent Intrinsic model.
6. Temporal consistency under video scenarios has not been validated.

## Related Work & Insights

- **StyLitGAN (Bhattad et al.)**: Manipulates lighting within the StyleGAN latent space $\rightarrow$ limited to synthetic images; LumiNet employs it as a data engine rather than an inference tool.
- **Latent Intrinsics (Zhang et al.)**: Proves the existence of intrinsic/extrinsic decomposition in latent space $\rightarrow$ LumiNet injects this decomposition into a diffusion model to achieve end-to-end relighting.
- **IC-Light**: SOTA in portrait relighting $\rightarrow$ struggles to handle complex multi-light interactions in full scenes.
- **Insights**: The combination paradigm of "latent intrinsic representation + diffusion model" can be extended to other scene editing tasks (e.g., material transfer, style transfer). The key is finding a suitable latent decomposition as the conditioning signal.

## Rating

⭐⭐⭐⭐ — A new breakthrough in scene-level relighting. The combination of data strategy, architectural design, and inference enhancement forms a complete solution, leading comprehensively in user studies. However, the inference pipeline is somewhat heavy, and performance on extreme scenes remains insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[CVPR 2025\] RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing](roompainter_view-integrated_diffusion_for_consistent_indoor_scene_texturing.md)
- [\[CVPR 2025\] Comprehensive Relighting: Generalizable and Consistent Monocular Human Relighting and Harmonization](comprehensive_relighting_generalizable_and_consistent_monocular_human_relighting.md)
- [\[ECCV 2024\] EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion](../../ECCV2024/image_generation/echoscene_indoor_scene_generation_via_information_echo_over_scene_graph_diffusio.md)
- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)

</div>

<!-- RELATED:END -->
