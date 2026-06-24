---
title: >-
  [Paper Note] Uni-Renderer: Unifying Rendering and Inverse Rendering via Dual Stream Diffusion
description: >-
  [CVPR 2025][Image Generation][Rendering Equation] Uni-Renderer proposes a unified framework based on a dual-stream diffusion model, which formulates rendering (from intrinsic attributes to RGB images) and inverse rendering (from RGB images to intrinsic attributes) as two conditional generation tasks. By utilizing cycle-consistency constraints, it mitigates the inherent ambiguity in inverse rendering, achieving outcomes superior to existing methods in material decomposition an…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Rendering Equation"
  - "Inverse Rendering"
  - "Dual Stream Diffusion"
  - "Cycle Consistency"
  - "Material Editing"
date: 2026-05-08
content_hash: b11dbf6f12e9c8ff
---

# Uni-Renderer: Unifying Rendering and Inverse Rendering via Dual Stream Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2412.15050](https://arxiv.org/abs/2412.15050)  
**Code**: To be open-sourced (promised in the paper)  
**Area**: Diffusion Models / Rendering  
**Keywords**: Rendering Equation, Inverse Rendering, Dual Stream Diffusion, Cycle Consistency, Material Editing

## TL;DR

Uni-Renderer proposes a unified framework based on a dual-stream diffusion model, which formulates rendering (from intrinsic attributes to RGB images) and inverse rendering (from RGB images to intrinsic attributes) as two conditional generation tasks. By utilizing cycle-consistency constraints, it mitigates the inherent ambiguity in inverse rendering, achieving outcomes superior to existing methods in material decomposition and rendering editing.

## Background & Motivation

**Background**: Physically-based rendering (PBR) simulates the interaction between light and materials using the rendering equation to generate realistic 2D images. Traditional methods rely on Monte Carlo light transport simulation and path tracing to solve the rendering equation, but recursive tracing is extremely computationally expensive. Inverse rendering, on the other hand, infers intrinsic attributes such as geometry, materials, and lighting from an image, which is a highly ill-posed problem. Recently, diffusion models have demonstrated great potential in material estimation and rendering.

**Limitations of Prior Work**: (1) Traditional rendering methods are computationally expensive and difficult to run in real-time. (2) Inverse rendering suffers from inherent ambiguity, where a single image can correspond to multiple combinations of geometry, materials, and lighting, meaning the mapping from images to intrinsic attributes is not one-to-one. (3) Existing diffusion-based methods (e.g., RGB2X) treat rendering and inverse rendering as two independent tasks and train two separate models, failing to leverage the complementary relationship between the two tasks.

**Key Challenge**: The ambiguity in inverse rendering fundamentally stems from a lack of constraints. Given an image, there are infinite material-lighting combinations that can produce the same appearance. If the rendering process can be used to validate the results of inverse rendering (i.e., re-rendering the decomposed attributes should reconstruct the original image), the search space can be dramatically reduced.

**Goal**: To build a unified framework that simultaneously handles rendering and inverse rendering such that the two tasks mutually benefit each other, where rendering provides consistency constraints for inverse rendering, and inverse rendering provides conditional inputs for rendering.

**Key Insight**: Inspired by UniDiffuser, this study models two conditional distributions (attributes $\rightarrow$ RGB and RGB $\rightarrow$ attributes) simultaneously using two independent timestep schedules, rather than modeling all joint and marginal distributions as in UniDiffuser. This targeted multi-task learning reduces task conflict.

**Core Idea**: To use a dual-stream diffusion model to simultaneously learn the forward and backward conditional distributions of the rendering equation, explicitly reducing ambiguity through a cycle-consistency constraint (inverse rendering $\rightarrow$ rendering $\rightarrow$ comparison with the original image).

## Method

### Overall Architecture

The inputs and outputs of Uni-Renderer depend on the task mode. In the rendering mode, the inputs are intrinsic attributes (metallic, roughness, albedo, normal, specular lighting, diffuse lighting) and the output is the RGB image. In the inverse rendering model, the input is the RGB image and the outputs are all the intrinsic attributes. The core of the framework is a dual-stream diffusion network: the upper branch processes attributes while the lower branch processes RGB, and information is exchanged via cross-conditioning. During training, a timestep selector alternates between the two tasks: one branch has a timestep of 0 (acting as the condition), and the other branch selects a timestep $t \in [0, T]$ (requiring denoising).

### Key Designs

1. **Dual Stream Diffusion Architecture**:

    - **Function**: To simultaneously support both rendering and inverse rendering conditional generation tasks within a single model.
    - **Mechanism**: The model consists of two pre-trained diffusion model branches—one processing RGB images and the other processing PBR attribute maps. Information is cross-conditioned between the two branches via a "dual-stream module." Timestep selection follows $(t_{\text{attributes}}, t_{\text{RGB}}) = (0, \tilde{t})$ (rendering mode) or $(\tilde{t}, 0)$ (inverse rendering mode), where $\tilde{t}$ takes $t$ with probability $p$, and $T$ otherwise. Unlike UniDiffuser, which models all joint and marginal distributions, Uni-Renderer only models two conditional distributions $q(\mathbf{x}_0 | \mathbf{y}_0)$ and $q(\mathbf{y}_0 | \mathbf{x}_0)$, mitigating task conflict.
    - **Design Motivation**: From a multi-task learning perspective, as network capacity is limited, fitting too many distributions simultaneously can lead to task competition and suboptimal performance. Restricting the modeling to two mutually beneficial conditional distributions is sufficient to cover the rendering and inverse rendering requirements while avoiding unnecessary task conflicts. The dual-stream architecture allows the two branches to share intermediate representation features for mutually beneficial learning.

2. **Cycle-Consistent Constraint**:

    - **Function**: To reduce inverse rendering ambiguity through a cycle validation process of "inverse rendering $\rightarrow$ rendering $\rightarrow$ comparison."
    - **Mechanism**: During training, an additional forward rendering is performed on the predicted inverse rendering results: the PBR attributes predicted by the model, denoted as $\hat{\mathbf{C}}$, are re-rendered to generate $\hat{\mathbf{x}}_{\text{rgb}}$, which is then compared with the original RGB image. The loss function is formulated as $\mathcal{L} = \mathbb{E}[||\mathbf{x}_0 - \hat{\mathbf{x}}_0(\hat{\mathbf{x}}_{\text{rgb}}, t, \mathbf{C})||^2]$. This is essentially a self-supervised constraint—if the decomposed attributes are correct, the re-rendered image should reconstruct the original image.
    - **Design Motivation**: The ambiguity of inverse rendering stems from the fact that the mapping from RGB to attributes is not injective. The cycle-consistency constraint forces the model's predicted properties to be "reconstructible" under the rendering equation, greatly narrowing down the solution space. This constraint is naturally implemented in the unified framework—since the same model can perform both rendering and inverse rendering, the cycle process requires no external models.

3. **Latent Preparation**:

    - **Function**: To encode different types and scales of intrinsic attributes into a unified latent space of the diffusion model.
    - **Mechanism**: Independent pre-trained VAEs are used to separately encode albedo $\mathbf{a}$, surface normal $\mathbf{n}$, and environmental lighting $\mathbf{s}, \mathbf{d}$. For the scalar attributes metallic $m$ and roughness $r$, they are first expanded into grayscale maps and grouped with a binary mask $\mathbb{m}$ to form a three-channel group $[m, r, \mathbb{m}]$, which is then encoded using an RGB VAE. All encoded latent features are concatenated along the channel dimension before being fed into the model.
    - **Design Motivation**: Metallic and roughness are scalars and cannot be directly encoded by VAEs. Formatting them as images along with a mask to utilize the existing RGB VAE is an elegant engineering solution. Isolating VAEs for distinct attributes avoids information confusion among different properties.

### Loss & Training

- Diffusion training is conducted using the $\mathbf{x}_0$-prediction formulation.
- The training dataset utilizes 200K 3D assets from Objaverse. For each asset, 121 rendering pairs are generated by varying the metallic (0 to 1, step 0.1) and roughness (0 to 1, step 0.1) attributes, randomly selecting from 20K environment maps in LHQ-1024.
- The rendering resolution is $1024 \times 1024$, and the camera position is fixed at the front of the object.
- 100 unseen objects are held out during training for testing.

## Key Experimental Results

### Main Results

Rendering performance comparison (metallic/roughness editing, PSNR↑ / LPIPS↓):

| Method | Metallic PSNR↑ | Metallic LPIPS↓ | Roughness PSNR↑ | Roughness LPIPS↓ |
|------|---------------|----------------|----------------|-----------------|
| InstructPix2Pix* | 24.25 | 0.1032 | 24.43 | 0.1056 |
| Subias et al. | 28.09 | 0.0954 | 28.13 | 0.0817 |
| **Ours** | **30.72** | **0.0763** | **31.68** | **0.0695** |

Inverse rendering performance comparison (albedo / metallic / roughness / normal):

| Method | Albedo PSNR↑ | Albedo LPIPS↓ | Metallic MSE↓ | Roughness MSE↓ | Normal cos↑ |
|------|-------------|--------------|--------------|----------------|------------|
| IntrinsicAnything | 22.67 | 0.0633 | - | - | - |
| RGB2X | 18.15 | 0.0851 | - | - | 0.871 |
| GaussianShader | 16.55 | 0.0906 | 0.3421 | 0.3714 | 0.908 |
| Intrinsic Image Diff | 21.83 | 0.0632 | 0.1920 | 0.1315 | - |
| **Ours** | **23.20** | **0.0532** | **0.1182** | **0.1037** | **0.928** |

### Ablation Study

| Configuration | Albedo PSNR↑ | Metallic MSE↓ | Roughness MSE↓ | Relighting PSNR↑ |
|------|-------------|--------------|----------------|-----------------|
| **Full model** | **23.20** | **0.1182** | **0.1037** | **30.84** |
| w/o unified (separate models) | 18.62 | 0.1632 | 0.1391 | 26.95 |
| w/o cycle constrain | 21.20 | 0.1391 | 0.1304 | 28.12 |

### Key Findings

- Unified framework vs. separate models: Albedo PSNR improves from 18.62 to 23.20 (+4.58), and relighting PSNR improves from 26.95 to 30.84 (+3.89), proving that joint training allows the two tasks to mutually enhance each other.
- The cycle-consistency constraint contributes an additional improvement of approximately 2.0 dB in albedo PSNR (inverse rendering) and 2.72 dB in relighting PSNR, validating the effectiveness of evaluating decomposition results through a rendering loop.
- In terms of rendering quality, the model exceeds InstructPix2Pix (even when fine-tuned on the same dataset) by 6+ dB PSNR, as physically-grounded attribute conditioning provides more precise control than pure text prompts.
- The model also exhibits effective inverse rendering capabilities on real-world images (e.g., metal phone stands, water bottles), despite being trained purely on synthetic data.

## Highlights & Insights

- **%Modeling the rendering equation as a conditional generation problem**: Using a data-driven diffusion model to approximate the rendering equation instead of physical simulation avoids the high computational cost of recursive ray tracing. This idea opens up a new direction of "using generative models to replace or assist physical simulation."
- **Clever implementation of cycle-consistency constraint**: In the unified framework, the exact same model performs both rendering and inverse rendering. The cycle verification process requires no external model or extra training overhead, achieving a highly natural implementation.
- **Trade-offs in multi-task learning**: Unlike UniDiffuser, which models all joint and marginal distributions, Uni-Renderer selectively models only two complementary conditional distributions, thereby reducing task conflict. This "less is more" design philosophy is worth emulating in other multi-task generative scenarios.

## Limitations & Future Work

- The training dataset consists entirely of synthetic data (Objaverse), leading to a synthetic-to-real domain gap. Although promising results are demonstrated on some real-world images, complex real-world scenes (e.g., multi-object scenes, complex lighting) may yield suboptimal performance.
- The camera position is fixed at the front of the object, and multi-view rendering or inverse rendering is not supported.
- Each object only has 121 pairs (11×11 metallic-roughness combinations), which limits the fine-grained level of material variations.
- Environmental lighting is represented by 2D environment maps, rendering it incapable of handling complex illumination phenomena like local light sources and shadow casting in 3D space.
- Currently, the framework only supports single-object scenes and cannot handle layered rendering of multi-object scenes.
- Future work needs to incorporate more real-world datasets to bridge the domain gap.

## Related Work & Insights

- **vs RGB2X**: RGB2X also employs diffusion models for forward and inverse rendering but implements them as two separate models, failing to exploit their mutual synergies. Uni-Renderer achieves substantially better inverse rendering outcomes (albedo PSNR: 23.20 vs 18.15) through its unified framework and cycle-consistency constraints.
- **vs MaterialGAN/SIC**: These approaches utilize GANs or encoder-decoder architectures, primarily handling planar surface materials and still requiring external renderers to integrate predicted attributes; Uni-Renderer generates end-to-end directly.
- **vs UniDiffuser**: While serving as the technical inspiration, Uni-Renderer selectively models only two conditional distributions rather than all distributions, aligning better with the specific demands of rendering/inverse rendering.
- **vs NvDiffRec/GaussianShader**: These are optimization-based methods requiring multi-view inputs and per-object optimization, whereas Uni-Renderer is feed-forward and achieves single-image inverse rendering.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of unifying rendering and inverse rendering into a single diffusion framework is refreshing, and the design of the cycle-consistency constraint is clever. However, the technical foundation of dual-stream diffusion largely derives from UniDiffuser.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative and qualitative comparisons on both rendering and inverse rendering directions are comprehensive, and the ablation study is properly designed, though validations on real-world scenes are somewhat sparse.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the figure visualizations are outstanding, although some implementation details (e.g., the specific interaction mechanism of the dual-stream module) could be more thoroughly elaborated.
- Value: ⭐⭐⭐⭐ Provides an effective unified paradigm for data-driven rendering/inverse rendering, holding potential value for downstream applications such as game production and architectural visualization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)
- [\[ICCV 2025\] Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering](../../ICCV2025/image_generation/ouroboros_single-step_diffusion_models_for_cycle-consistent_forward_and_inverse_.md)
- [\[CVPR 2025\] PICD: Versatile Perceptual Image Compression with Diffusion Rendering](picd_versatile_perceptual_image_compression_with_diffusion_rendering.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2026\] Towards Photorealistic and Efficient Bokeh Rendering via Diffusion Framework](../../CVPR2026/image_generation/towards_photorealistic_and_efficient_bokeh_rendering_via_diffusion_framework.md)

</div>

<!-- RELATED:END -->
