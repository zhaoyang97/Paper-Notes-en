---
title: >-
  [Paper Note] SphereDiff: Tuning-free Omnidirectional Panoramic Image and Video Generation via Spherical Latent Representation
description: >-
  [AAAI 2026][Video Generation][Panoramic Generation] This paper proposes SphereDiff, which defines a spherical latent representation (uniformly distributed via Fibonacci Lattice) to replace conventional equirectangular pr…
tags:
  - "AAAI 2026"
  - "Video Generation"
  - "Panoramic Generation"
  - "Spherical Latent Space"
  - "MultiDiffusion"
  - "Diffusion Models"
  - "VR/AR"
date: 2026-05-08
content_hash: 68d39d16a5a9c11a
---

# SphereDiff: Tuning-free Omnidirectional Panoramic Image and Video Generation via Spherical Latent Representation

**Conference**: AAAI 2026
**arXiv**: [2504.14396](https://arxiv.org/abs/2504.14396)  
**Code**: [https://github.com/pmh9960/SphereDiff](https://github.com/pmh9960/SphereDiff)  
**Area**: Video Generation
**Keywords**: Panoramic Generation, Spherical Latent Space, MultiDiffusion, Diffusion Models, VR/AR

## TL;DR
This paper proposes SphereDiff, which defines a spherical latent representation (uniformly distributed via Fibonacci Lattice) to replace conventional equirectangular projection (ERP), combined with a dynamic sampling algorithm and distortion-aware weighted averaging. Without any fine-tuning, SphereDiff leverages pretrained diffusion models such as SANA and LTX Video to generate seamless, low-distortion 360° panoramic images and videos.

## Background & Motivation

**Background**: AR/VR applications demand high-quality 360° panoramic content. Panoramas are typically represented using equirectangular projection (ERP), which maps the spherical surface to a 2D rectangle. Existing methods fall into two categories: (1) fine-tuning diffusion models on ERP datasets (e.g., PanFusion, 360DVD), which are constrained by limited data and suffer from severe distortion near the poles; (2) tuning-free methods based on MultiDiffusion (e.g., DynamicScaler), which still operate in the ERP latent space and produce discontinuities at the poles.

**Limitations of Prior Work**: The fundamental issue with ERP is its non-uniform distribution—latent variable density near the poles is far greater than near the equator, causing severe distortion and artifacts at high latitudes. Fine-tuning approaches are limited by the scarcity of text–ERP paired data and cannot fully adapt; tuning-free methods, despite leveraging MultiDiffusion, introduce discontinuous seams during ERP–perspective projection conversion due to interpolation or sampling issues.

**Key Challenge**: Standard diffusion models are trained in perspective space, whereas 360° panorama generation requires operating on a sphere. ERP as an intermediate representation introduces inherent distribution shift and polar distortion that is difficult to resolve fundamentally, whether through fine-tuning or tuning-free approaches.

**Goal**: To design a genuinely tuning-free framework that operates in spherical space, fundamentally eliminating ERP distortion while leveraging state-of-the-art pretrained diffusion models to generate seamless panoramas.

**Key Insight**: Latent variables are uniformly sampled on the sphere using a Fibonacci Lattice, ensuring that each viewing direction encompasses approximately equal numbers of latent variables. Spherical MultiDiffusion is then extended to denoise these uniformly distributed spherical latent variables.

**Core Idea**: Define a spherical latent space (each latent variable paired with spherical coordinates), use dynamic sampling to discretize continuous spherical projected latents onto a 2D grid for compatibility with standard diffusion models, and apply distortion-aware weighted averaging to mitigate residual distortion from the spherical-to-perspective projection.

## Method

### Overall Architecture
A total of 2,600 noise latent variables uniformly distributed on the sphere (Fibonacci Lattice) are initialized. At each denoising step: (1) spherical latents are projected into perspective space for 89 uniformly distributed viewing directions; (2) the projected results are discretized onto an $H \times W$ grid via dynamic sampling; (3) each view is denoised using a pretrained diffusion model (SANA/LTX Video); (4) all views are aggregated back into the spherical latent space via distortion-aware weighted averaging. After iterative denoising, a VAE decoder is applied to each view to produce the final panorama.

### Key Designs

1. **Spherical Latent Representation**:

    - Function: Uniformly distributes latent variables on the sphere, fundamentally eliminating the non-uniform distribution problem of ERP.
    - Mechanism: Spherical latent variables are defined as $\mathbf{s}_i = (\mathbf{d}_i, \mathbf{f}_i)$, where $\mathbf{d}_i \in \mathbb{S}^2$ denotes spherical coordinates and $\mathbf{f}_i \in \mathbb{R}^C$ is a feature vector. A Fibonacci Lattice generates $N=2600$ near-uniformly distributed points. The spherical-to-perspective projection function $\mathcal{T}_{\mathbb{S}^2 \to \mathbb{P}^2}$ maps spherical coordinates to a 2D plane given the viewing direction $\mathbf{v}$ and focal length $f$.
    - Design Motivation: ERP concentrates an excessive number of latent variables near the poles, causing abnormal denoising in those regions. The Fibonacci Lattice ensures that each viewing direction encounters approximately equal numbers of latent variables, enabling truly uniform omnidirectional processing.

2. **Dynamic Latent Sampling**:

    - Function: Discretizes continuously distributed spherical projected latents onto a standard 2D grid, enabling direct use by standard diffusion models.
    - Mechanism: Nearest-neighbor sampling causes the same latent variable to be selected repeatedly (altering its distribution) and leads to under-sampling where some variables are never selected. The dynamic sampling algorithm addresses this by: (1) employing a queue mechanism that removes selected latents from the queue to prevent repetition; (2) dynamically adjusting the field-of-view $H \times W$ to allow flexible window sizes; (3) sorting by distance from the image center and prioritizing central-region latents (which carry the most reliable information), while discarding unselected variables at the periphery.
    - Design Motivation: Under-sampling disrupts information exchange between adjacent views—if a latent variable is not denoised in the current view, the next view may receive stale information, causing discontinuities. Dynamic sampling ensures that all latents within the field of view are processed.

3. **Distortion-Aware Weighted Averaging**:

    - Function: Mitigates residual distortion from the spherical-to-perspective projection when fusing multi-view denoising results in MultiDiffusion.
    - Mechanism: For each view's perspective image space, an exponential weight is defined as $W_{jk}^i = \exp(-\|\mathbf{u}_{jk}\| / \tau)$, where $\|\mathbf{u}_{jk}\|$ is the distance to the image center. Pixels farther from the center receive lower weights, as projection distortion increases with distance. The MultiDiffusion formulation becomes $\Psi(\mathbf{S}_t | \mathbf{z}) = \sum_{i=1}^n \mathbf{W}_{\mathcal{S}}^i \otimes F_i^{-1}(\Phi(\mathbf{I}_t^i | \mathbf{y}_i))$.
    - Design Motivation: Even though the spherical representation substantially reduces distortion, perspective projection still introduces slight distortion at the edges. The weighted averaging allows each position to rely more heavily on the view with minimal distortion, further improving seamlessness.

### Loss & Training
The method is entirely training-free. SANA is used for image generation and LTX Video for video generation. Experiments use 89 viewing directions, an 80° FoV, and 60% overlap. Generation takes approximately 30 seconds per image sample and 20 minutes per video sample on an A100-40GB GPU. Text prompts describe the scene separately for upper, middle, and lower regions.

## Key Experimental Results

### Main Results

| Method | Distortion↑ | Continuity↑ | Image Quality↑ | Aesthetics↑ | CLIP-Score↑ |
|--------|-------------|-------------|----------------|-------------|-------------|
| SphereDiff (Image) | **3.238** | **4.892** | **4.496** | **4.685** | **28.65** |
| DynamicScaler (Image) | 2.854 | 3.985 | 4.496 | 4.577 | 26.63 |
| PanFusion | 1.965 | 3.696 | 2.819 | 3.450 | 25.70 |
| SphereDiff (Video) | **2.579** | **4.496** | 3.050 | 3.593 | **27.52** |
| DynamicScaler (Video) | 1.971 | 2.971 | 2.711 | 3.236 | 26.89 |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Nearest-neighbor sampling | Inter-view information breaks down; visible stitching artifacts in overlapping regions |
| Dynamic sampling | Improved inter-view information exchange; more coherent images |
| Dynamic sampling + distortion-aware weighting | Best overall performance; fully seamless results |
| Nearest-neighbor + distortion-aware weighting | Some improvement, but adjacent regions remain incoherent |

### Key Findings
- SphereDiff achieves substantial improvements over all baselines on distortion and continuity metrics, consistent with both GPT-4o evaluation scores and user study results.
- In the user study, SphereDiff achieves the highest preference rate on distortion and end-to-end continuity (38.1% vs. 20.24% for DynamicScaler).
- Both dynamic sampling and distortion-aware weighting are necessary components; removing either one significantly degrades quality.
- Video generation quality is slightly lower than image generation, primarily limited by the underlying video model (LTX Video).

## Highlights & Insights
- The conceptual shift from "patching ERP distortion" to "abandoning ERP and operating directly on the sphere" is fundamentally principled. The use of Fibonacci Lattice is both elegant and concise.
- The "queue + center-priority" strategy in dynamic sampling cleverly resolves information loss arising from discretizing continuous coordinates, and is transferable to other scenarios requiring irregular sampling.
- As a training-free method, SphereDiff directly benefits from rapid advances in diffusion models—upgrading the underlying model immediately translates to improved output quality.

## Limitations & Future Work
- Each view is denoised independently, lacking global context—different views may produce stylistically inconsistent content.
- Inference is slow (approximately 20 minutes per video sample), limiting interactive applications.
- Inter-view consistency relies primarily on MultiDiffusion's weighted averaging, with no explicit global consistency constraint.
- Future work could incorporate globally context-aware refinement methods to further improve consistency.

## Related Work & Insights
- **vs. PanFusion/Text2Light**: These methods fine-tune on ERP data and fail near the poles. SphereDiff fundamentally avoids ERP distortion.
- **vs. DynamicScaler**: Also a tuning-free method but still operates in the ERP latent space, producing blurry artifacts at the poles. SphereDiff's spherical latent space resolves this issue completely.
- **vs. CubeDiff**: Uses cubemap representation to reduce polar distortion, but discontinuities persist at face boundaries. SphereDiff's continuous spherical representation is more natural.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Replacing ERP with a spherical latent representation is a fundamentally innovative contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ GPT-4o evaluation, user study, and ablation are comprehensive, though quantitative metrics rely on subjective scores.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear, with rich algorithmic illustrations.
- Value: ⭐⭐⭐⭐⭐ Has direct application value for AR/VR panoramic content generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning](../../CVPR2026/video_generation/from_static_to_dynamic_exploring_self-supervised_image-to-video_representation_t.md)
- [\[ICCV 2025\] Long Context Tuning for Video Generation](../../ICCV2025/video_generation/long_context_tuning_for_video_generation.md)
- [\[ICML 2026\] LuVe: Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts](../../ICML2026/video_generation/luve_latent-cascaded_ultra-high-resolution_video_generation_with_dual_frequency_.md)
- [\[ACL 2026\] Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity](../../ACL2026/video_generation/accelerating_training_of_autoregressive_video_generation_models_via_local_optimi.md)
- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](../../CVPR2026/video_generation/drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)

</div>

<!-- RELATED:END -->
