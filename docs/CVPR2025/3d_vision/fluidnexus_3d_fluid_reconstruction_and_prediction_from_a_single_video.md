---
title: >-
  [Paper Note] FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video
description: >-
  [CVPR 2025][3D Vision][Fluid Reconstruction] FluidNexus is proposed to reconstruct 3D fluid appearance and velocity fields and predict future states from a single video for the first time. By synthesizing multi-view reference videos using a video generation model and bridging differentiable simulation and rendering via a coupled physical-visual particle representation, this method significantly outperforms existing multi-view approaches in novel view synthesis and future pred…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Fluid Reconstruction"
  - "Physical Simulation"
  - "Video Generation"
  - "3D Gaussian Splatting"
  - "Differentiable Rendering"
date: 2026-05-08
content_hash: ae98f98b220dad63
---

# FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video

**Conference**: CVPR 2025  
**arXiv**: [2503.04720](https://arxiv.org/abs/2503.04720)  
**Code**: [https://yuegao.me/FluidNexus](https://yuegao.me/FluidNexus)  
**Area**: 3D Vision  
**Keywords**: Fluid Reconstruction, Physical Simulation, Video Generation, 3D Gaussian Splatting, Differentiable Rendering

## TL;DR
FluidNexus is proposed to reconstruct 3D fluid appearance and velocity fields and predict future states from a single video for the first time. By synthesizing multi-view reference videos using a video generation model and bridging differentiable simulation and rendering via a coupled physical-visual particle representation, this method significantly outperforms existing multi-view approaches in novel view synthesis and future prediction.

## Background & Motivation

1. **Background**: Video-based 3D fluid reconstruction has made progress in recent years (e.g., PINF and HyFluid combine neural rendering with physical priors), but all of these methods require synchronized multi-view videos as input.
2. **Limitations of Prior Work**: Synchronized multi-view videos are often unavailable in real-world scenarios—such as industrial monitoring and outdoor observations, which typically feature only a single camera. Furthermore, existing methods either neglect physical modeling and thus cannot predict future states (e.g., dynamic 3D Gaussians) or require fully known, complete fluid physical properties (e.g., CFD simulation).
3. **Key Challenge**: Reconstructing 3D fluids from a single-view video is highly ill-posed, as a single observed frame can correspond to infinitely many 3D fluid states. Meanwhile, complex fluid dynamics, scattering effects, and vortex details make prediction post-reconstruction extremely challenging.
4. **Goal**: (1) How to synthesize multi-view consistent fluid videos from a single view? (2) How to reconstruct physically plausible 4D fluids from potentially inconsistent synthesized videos? (3) How to predict future fluid motion and support interaction based on the reconstruction results?
5. **Key Insight**: Leveraging the fluid dynamics generation capabilities of video diffusion models to supplement missing viewpoint information, and using physically-constrained particle representations to extract physically consistent 3D fluid motion from synthesized multi-view videos.
6. **Core Idea**: Synthesize multi-view references using a video generation model and employ a coupled physical-visual dual-particle representation to achieve an end-to-end pipeline from a single video to 3D fluid reconstruction and prediction.

## Method

### Overall Architecture
FluidNexus consists of two main components: (1) **Novel-View Video Synthesizer**—which first synthesizes novel views for each frame independently using a frame-by-frame novel-view diffusion model (Zero123) and then refines temporal consistency using a video diffusion model (CogVideoX-5b) to generate $C$ novel-view videos; (2) **Physical-Visual Particle Representation**—which couples physical particles (parameterizing the velocity and density fields) with visual particles (representing appearance via 3D Gaussian Splatting), reconstructing 4D fluids from multi-view videos through differentiable simulation and rendering, and supporting future prediction and interactive simulation.

### Key Designs

1. **Novel-View Video Synthesizer**:

    - **Function**: Synthesizing spatially and temporally consistent multi-view fluid videos from a single-view video.
    - **Mechanism**: A two-step process: first, an image diffusion model conditioned on camera transformations, $\hat{I}_t^c = g(I_t^0, \pi_c)$, is used to independently synthesize novel views for each frame, yielding a coarse video that is spatially consistent but temporally inconsistent; then, a video diffusion model is applied for SDEdit-style refinement, adding moderate noise to the coarse video followed by partial denoising $\mathcal{V}^c = v(\hat{\mathcal{V}}^c | \lambda_{\text{SDEdit}})$, where $\lambda_{\text{SDEdit}}$ controls the balance between content preservation and temporal consistency. For long videos, an unconditional refiner is trained to handle the initial segment, and a conditional refiner (conditioned on previous frames) is used to recursively extend subsequent segments.
    - **Design Motivation**: Frame-by-frame synthesis ensures spatial geometric consistency but lacks temporal coherence of fluid dynamics; video diffusion refinement leverages its learned fluid motion prior to restore temporal consistency. The two steps complement each other.

2. **Coupled Physical-Visual Dual-Particle Representation**:

    - **Function**: Bridging differentiable physical simulation and differentiable rendering to achieve a unified representation that serves both reconstruction and prediction.
    - **Mechanism**: **Physical particles** define the fluid velocity field $\mathbf{V}_t(\mathbf{x}) = \sum_j \mathbf{u}_{t,j} K(\mathbf{x} - \mathbf{p}_{t,j}) / \sum_j K(\mathbf{x} - \mathbf{p}_{t,j})$ and density field, which are constrained physically via Position-Based Fluid (PBF) differentiable simulation. **Visual particles** represent the fluid appearance using 3D Gaussian Splatting properties (position, color, scale, opacity, rotation) and are advected by the velocity field: $\mathbf{x}_t = \text{Adv}(\mathbf{V}_t, \mathbf{x}_{t-1})$. These two representations are separated because the velocity field is defined in the full 3D space, whereas the appearance is only defined on the visible regions of the fluid.
    - **Design Motivation**: Pure physical simulation is accurate under known initial conditions but quickly diverges in the presence of reconstruction errors; pure rendering representation cannot predict future states. The dual-particle coupling allows physical particles to provide dynamics constraints and predictive capabilities, while visual particles handle appearance reconstruction and rendering.

3. **Generative Fluid Simulation**:

    - **Function**: Combining physical simulation and video generation to achieve high-quality future prediction.
    - **Mechanism**: During the prediction phase $(t > T)$, a PBF simulation is first executed to obtain coarse physical particle trajectories and visual particle positions, from which coarse multi-view prediction videos are rendered. Then, a video diffusion model is used to refine these coarse videos (using $\lambda_{\text{SDEdit}}=0.75$, which is stronger than during reconstruction). The refined videos then serve as new reference inputs to run the reconstruction algorithm again to solve for physical and appearance parameters of future frames.
    - **Design Motivation**: Pure physical simulation tends to produce over-simplified motion and deviate from real dynamics due to the accumulation of reconstruction errors and the lack of real physical attributes such as temperature and viscosity. The generative capability of the video diffusion model can supplement the fluid details (vortices, scattering, etc.) missing in physical simulation, and generative simulation combines the advantages of both.

### Loss & Training
- **Physical Loss**: $\mathcal{L}_{\text{physics}} = \lambda_{\text{sim}} ||\mathbf{p}_t - \mathbf{p}_t^{\text{sim}}||_2^2 + \mathcal{L}_{\text{incomp}}$, where the incompressibility loss is composed of density constraints in current frames, density constraints in next frames, and minimum distance constraints between visual particles.
- **Visual Loss**: $\mathcal{L}_{\text{visual}} = \sum_{c=0}^{C}(\mathcal{L}_1(I_t^c, I_t'^c) + \mathcal{L}_{\text{SSIM}}(I_t^c, I_t'^c))$, comparing the rendered images with the reference videos.
- **Regularization Loss**: Encouraging temporal consistency of appearance attributes.
- **Two-stage Optimization**: First fix appearance to optimize physical particles (dynamics), then fix physics to optimize appearance attributes.
- **Loss Weights**: $\lambda_{\text{sim}}=0.1$, $\lambda_{\text{next}}=0.1$, $\lambda_{\text{v-incomp}}=0.1$.

## Key Experimental Results

### Main Results (ScalarFlow Dataset)

| Method | Input | NVS PSNR↑ | NVS SSIM↑ | Pred PSNR↑ | ∇·V↓ |
|------|------|-----------|-----------|------------|------|
| PINF | Multi-view | 22.68 | 0.7597 | 20.48 | 0.0297 |
| HyFluid | Multi-view | 22.23 | 0.7645 | 26.84 | 0.0619 |
| STG | Multi-view | 19.85 | 0.7063 | 21.79 | 0.0973 |
| **FluidNexus** | **Single-view** | **32.45** | **0.9544** | **28.51** | **0.0126** |

### Ablation Study (FluidNexus-Smoke Dataset)

| Method | NVS PSNR↑ | NVS LPIPS↓ | Pred PSNR↑ | ∇·V↓ |
|------|-----------|------------|------------|------|
| PINF | 22.40 | 0.5089 | 26.48 | 0.0451 |
| HyFluid | 22.64 | 0.4764 | 21.14 | 0.0573 |
| **FluidNexus** | **30.62** | **0.1707** | **27.79** | **0.0246** |

### Key Findings
- **Single-view >> Multi-view Baselines**: With only a single-view input, FluidNexus substantially outperforms PINF, HyFluid, and STG, which use multi-view inputs, across all metrics (e.g., PSNR 32.45 vs 22.68 on ScalarFlow, an absolute gain of nearly 10 dB). This is attributed to the high-quality reference videos synthesized by the video generation model and the effectiveness of the coupled physical-visual representation.
- **Optimal Incompressibility**: FluidNexus achieves the lowest velocity field divergence (∇·V) among all methods (0.0126 vs 0.0297), demonstrating that the PBF simulation constraints on physical particles successfully ensure the physical plausibility of the fluid.
- **Supporting Interactive Simulation**: FluidNexus can not only predict future frames but also simulate wind-fluid and object-fluid interactions, a capability completely absent in other methods.
- **Balance in Video Refinement**: Using an SDEdit strength of 0.5 (preserving more content) during reconstruction and 0.75 (generating more details) during prediction; this adaptive adjustment is critical.

## Highlights & Insights
- **Video Generation Model as a Physical Prior**: This is a profound insight—having been trained on massive amounts of fluid videos, video diffusion models implicitly learn fluid dynamics priors, which can be utilized to compensate for the limitations of explicit physical simulations. This "generative simulation" paradigm could be widely applicable to other physical systems.
- **Clever Dual-Particle Decoupled Design**: The velocity field (full space) and appearance (visible regions only) inherently have different spatial distributions. Representing them with two types of particles coupled via advection is a highly natural design.
- **From Single View to Multi-Task**: Starting from a single video to simultaneously achieve novel view synthesis, future prediction, and interactive simulation demonstrates the powerful capability of blending video generation, physical simulation, and neural rendering.

## Limitations & Future Work
- **Dependency on Video Diffusion Model Quality**: The multi-view consistency and temporal coherence of synthesized videos are limited by the diffusion model's capabilities, which might result in inconsistencies in complex fluid scenes.
- **High Computational Overhead**: Training video diffusion models and optimizing physical and appearance parameters frame-by-frame leads to a long end-to-end time.
- **Limited to Smoke/Gas Fluids**: The efficacy has not been validated on other fluid types like liquids (water flows, waves).
- **Constant Temperature and Viscosity Assumption**: The PBF simulation does not model properties such as temperature and viscosity, which restricts its capability to model complex fluid behaviors.
- **Future Work**: Incorporate more physical attributes (temperature field, buoyancy) into the particle representation; replace the Zero123 + SDEdit pipeline with stronger 3D-consistent video generation models; and extend to liquid fluid reconstruction and prediction.

## Related Work & Insights
- **vs PINF / HyFluid**: Both require synchronized multi-view videos and do not support interactive simulation. FluidNexus uses single-view videos alongside video generation as "virtual multi-views," which not only relaxes input requirements but also boosts performance—suggesting that the priors provided by generative models can be more valuable than additional real views.
- **vs SpaceTimeGaussians**: STG is a general dynamic scene reconstruction method that does not consider physical constraints and thus cannot predict the future. FluidNexus's physical particles endow the representation with physical reasoning capabilities.
- **vs Sora and Other Video Generation**: Video generation models can simulate fluids, but they are 2D and uncontrollable. FluidNexus leverages their generative capacity but elevates it to a 3D, controllable fluid reconstruction through physical constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First 3D fluid reconstruction + prediction + interaction from a single video, creating a paradigm innovation by blending video generation and physical simulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset (including two new datasets) and multi-task evaluation, though a detailed ablation study for each individual component is somewhat lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-defined problem, though some technical details are deferred to the supplementary materials.
- Value: ⭐⭐⭐⭐⭐ Pioneering work that integrates the fields of video generation, physical simulation, and neural rendering, with broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](../../ICCV2025/3d_vision/shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[CVPR 2025\] Coherent 3D Portrait Video Reconstruction via Triplane Fusion](coherent_3d_portrait_video_reconstruction_via_triplane_fusion.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)

</div>

<!-- RELATED:END -->
