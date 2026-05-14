---
title: >-
  [Paper Note] MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion
description: >-
  [ICLR 2026][Image Generation][Multi-view customized generation] This paper introduces a new task termed multi-view customization and proposes the MVCustom framework…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Multi-view customized generation"
  - "camera pose control"
  - "feature field rendering"
  - "video diffusion"
  - "geometric consistency"
date: 2026-05-08
content_hash: 077164ddc39a2815
---

# MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion

**Conference**: ICLR 2026
**arXiv**: [2510.13702](https://arxiv.org/abs/2510.13702)
**Code**: [Project Page](https://minjung-s.github.io/mvcustom/)
**Area**: Diffusion Models / Personalized Generation
**Keywords**: Multi-view customized generation, camera pose control, feature field rendering, video diffusion, geometric consistency

## TL;DR

This paper introduces a new task termed multi-view customization and proposes the MVCustom framework, which leverages a video diffusion backbone with dense spatio-temporal attention for holistic frame consistency. At inference time, two novel techniques are introduced—depth-aware feature rendering and consistency-aware latent completion—achieving for the first time the simultaneous satisfaction of camera pose control, subject identity preservation, and cross-view geometric consistency.

## Background & Motivation

**Background**: Controllable image generation has two key dimensions—camera control (multi-view generation) and customization (preserving subject identity from reference images). Each dimension has been extensively studied, yet methods that **jointly** address both remain virtually absent.

**Limitations of Prior Work**:
- Traditional customization methods (DreamBooth, Custom Diffusion) do not support camera pose control.
- Multi-view generation methods (CameraCtrl, SEVA) do not support personalized customization.
- Customization methods with viewpoint control (CustomDiffusion360, CustomNet) focus solely on the subject, neglecting cross-view consistency of the background.
- Directly applying customization methods (e.g., DreamBooth-LoRA) to multi-view generation backbones leads to loss of subject identity and degraded camera control.

**Key Challenge**: Multi-view generation relies on large-scale data to learn 3D geometry, whereas customization scenarios provide only a handful of reference images—a fundamental tension between data scarcity and the demand for geometric consistency.

**Goal**: Define and address the "multi-view customization" task: (i) generate images matching specified camera poses; (ii) preserve subject identity from reference images; (iii) maintain cross-view consistency for both subject and background.

**Key Insight**: Decouple the training and inference stages—use limited data during training to learn subject identity and geometry, and apply explicit geometric constraints (depth rendering) at inference to enforce consistency.

**Core Idea**: Leverage a video diffusion backbone to learn temporal consistency, employ feature field modeling for geometry, and use depth-guided rendering at inference to ensure cross-view geometric consistency.

## Method

### Overall Architecture

MVCustom consists of two stages:
- **Training Stage**: A video diffusion backbone based on AnimateDiff with dense spatio-temporal attention (replacing the original 1D temporal attention), a pose-conditioned Transformer block (incorporating FeatureNeRF), and textual inversion to learn the subject embedding.
- **Inference Stage**: Depth-aware feature rendering explicitly enforces geometric consistency across views; consistency-aware latent completion fills newly visible regions.

### Key Designs

1. **Pose-Conditioned Transformer Block (FeatureNeRF)**:

    - **Function**: Injects camera pose information into the diffusion model and learns the geometric structure of the subject.
    - **Mechanism**: A dual-branch architecture is designed—the main branch generates target-view feature maps, while the multi-view branch aggregates reference-view features via FeatureNeRF. FeatureNeRF exploits epipolar geometry and volume rendering to synthesize pose-aligned feature maps $\bm{X}_y$ from reference image features $\{(\bm{X}_i, \pi_i)\}$.
    - **Design Motivation**: Enables the diffusion model to learn 3D geometric information from a small number of reference images.

2. **Dense Spatio-Temporal Attention**:

    - **Function**: Replaces AnimateDiff's 1D temporal attention to enable information exchange across frames and spatial positions.
    - **Mechanism**: The original 1D temporal attention only interacts between frames at identical spatial positions, failing to model spatial displacements caused by viewpoint changes. Dense 3D spatio-temporal attention allows cross-frame interaction at arbitrary spatial positions. A progressively expanding spatial attention scope strategy is adopted to maintain training stability and preserve pre-trained knowledge.
    - **Design Motivation**: Ablation studies confirm that when performing feature replacement, 1D temporal attention fails to propagate spatial flow correctly; dense spatio-temporal attention is essential.

3. **Depth-Aware Feature Rendering**:

    - **Function**: Explicitly enforces cross-view geometric consistency at inference time.
    - **Mechanism**: An anchor frame is selected, its depth is estimated via ZoeDepth, and an anchor feature grid $\mathcal{M}_a = (\bm{P}_a, \bm{F}_a, \mathcal{T}_a)$ is constructed. A differentiable mesh renderer projects the anchor frame features onto other camera poses. During the first 35 DDIM sampling steps, visible regions are replaced with rendered features: $\hat{\bm{F}}_n = \bm{M}_n^a \odot \bm{F}_n^a + (1-\bm{M}_n^a) \odot \bm{F}_n$.
    - **Design Motivation**: Data scarcity during training precludes implicit learning of geometric consistency as in large-scale multi-view methods; explicit geometric constraints are therefore necessary.

4. **Consistency-Aware Latent Completion**:

    - **Function**: Generates plausible content for newly visible (disoccluded) regions arising from viewpoint changes.
    - **Mechanism**: During denoising, $x_0$ is predicted from the intermediate latent $x_t$, then re-noised to timestep $t$ to obtain perturbed latents $x_t'$. Newly visible regions in the original latents are replaced with their perturbed counterparts. This process iterates from timestep $T$ down to an early timestep $\tau$ near $T$.
    - **Design Motivation**: Feature rendering can only handle regions visible in the anchor frame; newly visible regions require the generative model's prior knowledge for coherent completion.

### Loss & Training

Standard denoising loss combined with the FeatureNeRF loss. The video backbone is trained on a subset of WebVid10M (430K samples); CO3Dv2 is used for customization experiments (3 concepts each for car, chair, and motorcycle categories).

## Key Experimental Results

### Main Results

| Method | Camera Pose Accuracy↑ | Multi-View Consistency↓ | Identity Preservation↓ | Text Alignment↑ |
|---|---|---|---|---|
| Custom Img + Img-MV gen | 0.675 | 0.214 | 0.504 | 0.676 |
| Txt-MV gen with DB | 0.283 | **0.116** | 0.557 | 0.723 |
| CustomDiffusion360 | 0.000 | 0.190 | **0.417** | **0.806** |
| **MVCustom (ours)** | **0.735** | 0.121 | 0.448 | 0.744 |

MVCustom is the only method achieving high scores simultaneously on camera pose accuracy and multi-view consistency.

### Ablation Study

| Configuration | Outcome |
|---|---|
| Customization fine-tuning only (no DFR/LCC) | Background remains static across viewpoints |
| + Depth-Aware Feature Rendering (DFR) | Background shifts correctly with camera motion, but disoccluded regions exhibit repeated content |
| + Consistency-Aware Latent Completion (LCC) | Disoccluded regions completed naturally; full geometric consistency achieved |
| 1D temporal attention + feature replacement | Spatial flow propagation fails |
| Dense spatio-temporal attention + feature replacement | Spatial consistency propagated correctly |

### Key Findings

- COLMAP reconstruction fails entirely for CustomDiffusion360 (pose accuracy = 0), demonstrating that focusing solely on the subject while ignoring background consistency is infeasible.
- Depth-aware feature rendering and latent completion are complementary: the former ensures geometric consistency in visible regions, while the latter handles generation in invisible regions.
- Dense spatio-temporal attention is a prerequisite for the feature replacement strategy to function correctly.
- MVCustom incurs notable computational overhead (130.92s, 19.29GB), primarily due to the depth estimator and feature replacement operations.

## Highlights & Insights

- **Clear and systematic task formulation**: Table 1 systematically analyzes the capability gaps of existing methods across all dimensions of multi-view customization, defining an important and unaddressed task.
- **Elegant training-inference decoupling**: Subject representation is learned from limited data at training time, while explicit geometric constraints compensate for data insufficiency at inference—a strategy generalizable to other data-scarce generative settings.
- **Transfer from video diffusion to multi-view generation**: Exploiting the temporal consistency of video models to achieve multi-view consistency represents an effective cross-task transfer.

## Limitations & Future Work

- The framework cannot alter the intrinsic pose of the subject through text (e.g., changing from "sitting" to "standing"), as FeatureNeRF learns a fixed canonical pose.
- Computational overhead is substantially higher than competing methods (130.92s vs. 27–97s; 19.29GB vs. 5–7GB).
- Evaluation is conducted on only 3 categories from CO3Dv2, leaving generalization insufficiently validated.
- Depth estimation quality directly impacts rendering results and may lack robustness for complex scenes.

## Related Work & Insights

- **vs. CustomDiffusion360**: Both target viewpoint-controllable customization, but CD360 neglects background consistency, causing COLMAP failure; MVCustom addresses this via a video backbone combined with inference-time geometric constraints.
- **vs. SEVA (Img-MV gen)**: Supports multi-view generation from a single image but lacks subject identity information, suffering severe degradation at views far from the input.
- **vs. CameraCtrl + DB**: Directly fine-tuning a multi-view model with DreamBooth paradoxically degrades camera control capability.
- The depth-aware feature rendering paradigm is transferable to other generation tasks requiring geometric consistency (e.g., video editing, 3D-aware inpainting).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The task formulation is original, and the inference-time geometric constraint strategy is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐ Limited to 3 categories; large-scale validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear and method description is thorough.
- **Value**: ⭐⭐⭐⭐ Opens a new direction in multi-view customized generation with promising applications in 3D content creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](../../NeurIPS2025/image_generation/a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[ICCV 2025\] LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering](../../ICCV2025/image_generation/larender_training-free_occlusion_control_in_image_generation_via_latent_renderin.md)

</div>

<!-- RELATED:END -->
