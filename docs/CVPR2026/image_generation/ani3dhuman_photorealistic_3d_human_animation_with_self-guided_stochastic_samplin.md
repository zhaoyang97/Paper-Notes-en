---
title: >-
  [Paper Note] Ani3DHuman: Photorealistic 3D Human Animation with Self-guided Stochastic Sampling
description: >-
  [CVPR2026][Image Generation][3D Human Animation] The Ani3DHuman framework is proposed, combining kinematics-driven mesh animation with video diffusion priors. Through Self-guided Stochastic Sampling…
tags:
  - "CVPR2026"
  - "Image Generation"
  - "3D Human Animation"
  - "Video Diffusion Priors"
  - "Stochastic Sampling"
  - "Non-rigid Motion"
  - "3D Gaussian Splatting"
date: 2026-05-08
content_hash: 35cdddead8fd8fce
---

# Ani3DHuman: Photorealistic 3D Human Animation with Self-guided Stochastic Sampling

**Conference**: CVPR2026  
**arXiv**: [2602.19089](https://arxiv.org/abs/2602.19089)  
**Code**: [qiisun/ani3dhuman](https://github.com/qiisun/ani3dhuman)  
**Area**: Image Generation  
**Keywords**: 3D Human Animation, Video Diffusion Priors, Stochastic Sampling, Non-rigid Motion, 3D Gaussian Splatting

## TL;DR

The Ani3DHuman framework is proposed, combining kinematics-driven mesh animation with video diffusion priors. Through Self-guided Stochastic Sampling, it restores low-quality rigid body renderings into high-fidelity videos, achieving realistic modeling of non-rigid clothing dynamics.

## Background & Motivation

**Limitations of Prior Work**: 
- **Kinematic Methods**: Methods based on skeleton/SMPL provide precise control over rigid motion but fail to model non-rigid dynamics like clothing folds and swaying, resulting in a lack of realism.
- **High Cost of Physical Simulation**: Physical methods (e.g., PhysAvatar) can simulate clothing-human interaction but require separate clothing mesh modeling and extensive physical parameter tuning, incurring extremely high computational and preprocessing costs.
- **Data Bottleneck for Multi-view Diffusion**: Methods based on multi-view video diffusion (SV4D 2.0, CharacterShot) are limited by the scarcity of 4D training data, with generation quality far inferior to 2D video models.
- **Identity Loss in Pose-driven Methods**: Methods like PERSONA reconstruct 3D animations directly from pose-driven 2D videos, where each video segment generates different appearance hallucinations, leading to severe identity inconsistency.
- **Quality Defects of SDS Distillation**: Score Distillation Sampling (SDS) methods (e.g., Disco4D) suffer from typical optimization artifacts such as over-saturation and over-smoothing, resulting in poor visual effects.

**Key Challenge**: 
- **Sampling Challenges for OOD Renderings**: Initial mesh renderings deviate significantly from the training distribution of diffusion models (out-of-distribution). Standard deterministic ODE samplers cannot correct trajectory biases, which represents the core technical bottleneck.

## Method

### Overall Architecture

Ani3DHuman consists of three stages:

1.  **Layered Motion Representation**: Decomposes human motion into mesh rigid motion + residual non-rigid motion fields.
2.  **Self-guided Stochastic Sampling**: Restores coarse videos from rigid renderings into high-quality, identity-preserving videos.
3.  **Progressive 4D Optimization**: Uses the restored high-fidelity videos as supervision signals to optimize the residual motion fields.

The input is a single reference image (reconstructed into 3DGS via LHM) and an SMPL pose sequence; the output is photo-realistic 3D human animation from arbitrary views.

### Key Designs: Layered Motion

-   **Mesh Rigid Motion**: Drives 3D Gaussians via SMPL bone parameters, establishing a bijective correspondence between Gaussians and SMPL-X canonical mesh vertices to apply skeletal transformations for rigid animation.
-   **Residual Non-rigid Motion Field**: Uses HexPlane-parameterized implicit functions to query features in canonical space, decoded by an MLP to predict Gaussian attribute offsets (position, rotation, etc.) to capture non-rigid deformations like clothing dynamics.
-   **Novelty**: The layered design allows the rigid part to benefit from strong motion priors while the non-rigid part focuses on learning subtle dynamics, offering higher stability than single-layer motion fields.

### Key Designs: Self-guided Stochastic Sampling

Core Problem: The initial rendering $\bm{y}$ is severely OOD, and deterministic Flow-ODE sampling accumulates errors along incorrect trajectories.

**Stochastic Sampling (Quality Preservation)**:

-   A reverse SDE for Flow Matching is proposed, injecting stochastic noise into the posterior noise prediction $\hat{\bm{x}}_{1|t}$ at each step: $\hat{\bm{x}}_{1|t} \leftarrow \sqrt{\gamma(t)}\epsilon + \sqrt{1-\gamma(t)}\hat{\bm{x}}_{1|t}$.
-   The stochastic term actively pushes samples back toward the marginal distribution $p_t(\bm{x})$, correcting OOD biases, where $\gamma(t) = \sigma_t$.

**Self-guidance (Identity Preservation)**:

-   High noise injection repairs OOD but destroys identity information. Borrowing from DPS posterior sampling:
-   A masked L2 guidance is applied to the posterior mean at each step: $\hat{\bm{x}}_{0|t} \leftarrow \hat{\bm{x}}_{0|t} - \lambda \nabla_{x_t}\|\mathcal{M} \odot (\bm{y} - \hat{\bm{x}}_{0|t})\|^2$.
-   The mask $\mathcal{M}$ (obtained via SAM2) covers regions requiring identity preservation like the face and hands. The gradient has a closed-form solution, making it computationally efficient.

**Personalized Diffusion Prior**: Finetuned based on Wan2.1-1.3B, adding a reference image branch and 2D pose sequence conditions specifically for human animation scenarios.

### Key Designs: Diagonal View-Time Sampling

-   Problem: Inconsistency between multiple generation trajectories leads to reconstruction blurring.
-   Mechanism: Simultaneously varies camera perspective and time steps (diagonal sampling) to cover spatio-temporal information with a minimum number of trajectories, minimizing inconsistency exposure.
-   Paired with progressive data updates: New trajectories based on the current model state are generated and added to the training set every 5k iterations.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{L1}} + \lambda_1 \mathcal{L}_{\text{LPIPS}} + \lambda_2 \mathcal{L}_{\text{dssim}} + \lambda_3 \mathcal{L}_{\text{mask}} + \lambda_4 \mathcal{L}_{\text{reg}}$$

where $\mathcal{L}_{\text{reg}}$ is the depth difference regularization for rigid regions to maintain geometric invariance.

## Key Experimental Results

### Main Results (ActorsHQ Dataset, 10 cases)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP-I↑ | FID↓ | FVD↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Disco4D | 12.05 | 0.559 | 0.502 | 0.644 | 613.9 | 622.1 |
| SV4D 2.0 | 15.25 | 0.771 | 0.377 | 0.764 | 364.9 | 478.7 |
| PERSONA | 17.01 | 0.822 | 0.260 | 0.878 | 199.1 | 367.0 |
| LHM | 19.51 | 0.838 | 0.217 | 0.901 | 124.1 | 339.9 |
| **Ours** | **20.08** | 0.831 | **0.213** | **0.916** | **105.3** | **295.2** |

-   Compared to LHM (second best), FID decreased by 18.8, FVD by 44.7, and CLIP-I improved by 0.015.
-   Reconstruction metrics (PSNR/LPIPS) are the best overall, while video quality (FID/FVD) remains significantly leading.

### User Study (New motion, no GT)

| Method | Identity Preservation | Frame Quality | Motion Realism | Non-rigid Physics | Overall Preference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Disco4D | 0% | 1.5% | 4.4% | 0% | 0% |
| SV4D 2.0 | 0% | 2.9% | 16.2% | 4.3% | 5.9% |
| PERSONA | 20.6% | 30.9% | 14.7% | 19.1% | 22.1% |
| LHM | 39.2% | 29.4% | 29.4% | 14.7% | 17.6% |
| **Ours** | **40.1%** | **35.3%** | **35.3%** | **61.8%** | **54.4%** |

Non-rigid physical plausibility leads with an absolute 61.8%, with an overall preference of 54.4%.

### Ablation Study Key Findings

1.  **Stochastic vs. Deterministic ODE**: Quality drops significantly without the stochastic term, verifying that stochastic sampling's ability to correct OOD inputs is irreplaceable.
2.  **Self-guidance vs. No Guidance**: Without self-guidance, video quality is high but identity is completely lost, proving the guidance mechanism is critical for fidelity.
3.  **Personalized vs. General Prior**: Replacing with a general diffusion model leads to slight degradation and artifacts.
4.  **Layered vs. Single-layer Motion**: Single-layer motion fields fail to model fine transformations (e.g., hands); the layered design shows significant improvement.
5.  **Diagonal vs. Fixed/Independent Sampling**: Baseline methods produce obvious floaters and spikes; diagonal sampling results in clearer reconstruction.

## Highlights & Insights

-   Elegant combination of kinematic animation controllability and the realistic non-rigid generation capabilities of video diffusion models, without relying on multi-view 4D data.
-   The proposed Self-guided Stochastic Sampling algorithm is tailored for OOD restoration scenarios, achieving a balance between quality and fidelity with clear theoretical analysis.
-   Diagonal View-Time Sampling solves multi-trajectory inconsistency with a minimum number of trajectories, offering a simple and effective solution.
-   Comprehensive experimental design: Quantitative metrics + User study + Multi-dimensional ablation + Comparison with 6 sampling methods.

## Limitations & Future Work

-   Sampling time for the video diffusion prior is long, limiting practical efficiency; the authors suggest introducing few-step generation techniques for acceleration.
-   Preservation region masks rely on SAM2; mask quality affects self-guidance results.
-   Current finetuning is limited to Wan2.1-1.3B; the model scale may constrain generation diversity.
-   The residual motion field based on HexPlane's ability to model extreme non-rigid deformations (e.g., large swaying of loose long skirts) remains to be verified.
-   Progressive training requires 25k iterations with dataset updates every 5k, resulting in high total training costs.

## Related Work & Insights

-   **Kinematic Animation**: SMPL/LBS series $\rightarrow$ LHM uses 3DGS for single-image reconstruction and mesh animation but lacks non-rigid parts.
-   **Physical Simulation**: PhysAvatar (C-IPC solver) can simulate clothing at the cost of complex modeling and high computation.
-   **SDS Distillation**: MAV3D $\rightarrow$ DG4D/Disco4D, limited by over-saturation/over-smoothing.
-   **Multi-view Video Reconstruction**: SV4D 2.0/CharacterShot, limited by 4D data scarcity.
-   **Pose Video Reconstruction**: PERSONA uses 2D video diffusion but suffers from severe identity loss.
-   **Video Edit Sampling**: SDEdit/FlowEdit/MCS/HFS-SDEdit are all based on deterministic ODEs and perform poorly on OOD renderings.

## Rating

-   Novelty: ⭐⭐⭐⭐ — Self-guided stochastic sampling is proposed for the first time within the Flow Matching framework, combining stochastic bias correction with posterior guidance for OOD restoration.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comparison with 6 sampling methods + 5 ablation groups + User study + 5 quantitative metrics, providing comprehensive coverage.
-   Writing Quality: ⭐⭐⭐⭐ — Clear structure, with a complete logic from problem motivation to method and experiment.
-   Value: ⭐⭐⭐⭐ — Provides a general paradigm for diffusion restoration under OOD conditions, significantly advancing SOTA in non-rigid human animation.

## Related Papers

- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[CVPR 2026\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_textguided_multihuman_3d_moti.md)
- [\[ICLR 2026\] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](../../ICLR2026/image_generation/stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)
- [\[CVPR 2026\] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[CVPR 2026\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_textguided_multihuman_3d_moti.md)
- [\[CVPR 2026\] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[ICLR 2026\] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](../../ICLR2026/image_generation/stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)

</div>

<!-- RELATED:END -->
