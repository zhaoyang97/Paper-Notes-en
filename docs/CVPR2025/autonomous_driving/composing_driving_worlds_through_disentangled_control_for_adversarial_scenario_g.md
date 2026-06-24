---
title: >-
  [Paper Note] CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation
description: >-
  [CVPR 2025][Autonomous Driving][Driving World Models] CompoSIA proposes a compositional driving video generation framework based on Flow Matching DiT. By disentangling the injection of three types of control signals—structure (3D bboxes), identity (a single reference image), and ego-motion (camera trajectories)—it achieves fine-grained independent control and compositional editing for systematically synthesizing adversarial driving scenarios…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Driving World Models"
  - "Disentangled Control"
  - "Adversarial Scenario Generation"
  - "Identity Injection"
  - "Flow Matching"
date: 2026-05-08
content_hash: 7140f8475019d5d2
---

# CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation

**Conference**: CVPR 2025  
**arXiv**: [2603.12864](https://arxiv.org/abs/2603.12864)  
**Code**: [GitHub](https://github.com/Yifever20002/CompoSIA)  
**Area**: Autonomous Driving / Video Generation / Scenario Simulation  
**Keywords**: Driving World Models, Disentangled Control, Adversarial Scenario Generation, Identity Injection, Flow Matching

## TL;DR
CompoSIA proposes a compositional driving video generation framework based on Flow Matching DiT. By disentangling the injection of three types of control signals—structure (3D bboxes), identity (a single reference image), and ego-motion (camera trajectories)—it achieves fine-grained independent control and compositional editing for systematically synthesizing adversarial driving scenarios, resulting in a 17% improvement in FVD and a 173% increase in collision rate.

## Background & Motivation
**Background**: End-to-end autonomous driving relies on large-scale video data to learn unified representations. However, safety-critical long-tail scenarios in datasets such as nuScenes/Waymo are severely lacking. Controllable generative models are key tools to synthesize these scenarios.

**Limitations of Prior Work**: Existing driving world models (MagicDrive-V2, DriveEditor, Vista) either only control a subset of scene elements or couple the control signals—injecting multiple conditions through shared paths, which degrades editing quality and prevents independent manipulation of structure, identity, and ego-motion.

**Key Challenge**: Adversarial scenarios often stem from the "abnormal combinations of common traffic elements" (e.g., sudden truck lane change + emergency braking), requiring independent, fine-grained control over scene structure, element identity, and ego-motion—which cannot be achieved by coupled generators.

**Goal**: How to achieve disentangled control of structure, identity, and action in a unified framework, allowing them to be independently manipulated and freely composed to systematically synthesize adversarial driving scenarios?

**Key Insight**: Different types of control signals should be injected at different levels of the diffusion process: structure is injected into the data stream, identity at the noise level, and ego-motion via AdaLN and PRoPE attention.

**Core Idea**: Three independent conditional branches (structure-zero-initialized projection, identity-noise-level replacement, motion-hierarchical dual-branch) achieve disentangled control, enabling compositional adversarial scenario synthesis.

## Method

### Overall Architecture
Based on the Flow Matching DiT backbone of Wan2.1-T2V-1.3B, three types of control signals are injected through independent pathways. Structural conditions are added to the latent tokens via zero-initialized projections; identity conditions replace the latent in corresponding regions during the high-noise phase; ego-motion conditions are injected through AdaLN local modulation and PRoPE global attention. The training modality allocation ratios are 0.6:0.3:0.1 for [motion], [structure+identity+motion], and unconditional.

### Key Designs

1. **Structure Condition (Spatio-temporal Layout)**

    - **Function**: Controls the positions and trajectories of scene elements via a sequence of 3D bboxes.
    - **Mechanism**: 3D bboxes $\to$ projected onto the 2D image plane $\to$ VAE encoded $\to$ lightweight convolutional adapter $\to$ added to latent tokens via zero-initialized projection.
    - **Design Motivation**: 3D bboxes are in the world coordinate system while latents are in the 2D image domain, requiring projection alignment beforehand. Zero-initialization ensures the pretrained weights are not disrupted in the early stages of training.

2. **Identity Condition (Noise-level Injection)**

    - **Function**: Controls the appearance identity of scene elements using a single reference image.
    - **Mechanism**: A select-and-repaint strategy is used to construct training pairs $\to$ the reference image is cropped and padded to the bbox sequence to form $\bm{r}_f$ $\to$ during the high-noise phase ($t > T_\text{id}$), the corresponding region of the latent is replaced with the identity condition latent $\to$ the model learns to recover a geometrically consistent sequence from noise.
    - **Design Motivation**: Modeling identity control as an inpainting problem instead of using attention mechanisms avoids the issue where overly strong identity constraints suppress motion expressiveness. Replacing only in the high-noise phase avoids disrupting the generation trajectory in low-noise phases.

3. **Action Condition (Hierarchical Dual-branch)**

    - **Function**: Controls ego-motion via frame-level continuous camera trajectories.
    - **Mechanism**:
        - *Local Branch*: Extracts relative transformations $(\Delta x, \Delta y, \Delta\text{yaw})$ of adjacent frames $\to$ sinusoidal frequency-encoding $\to$ zero-initialized projection $\to$ AdaLN gating injection (shift/scale/gate).
        - *Global Branch*: PRoPE projects positional encodings into low-dimensional subspace attention $\to$ zero convolutions add it back to the primary attention.
    - **Design Motivation**: The local branch accelerates convergence in early training, while the global branch improves overall motion control accuracy and stability.

### Loss & Training
$v$-prediction loss (standard loss for Flow Matching). The first frame's background region remains clean latent to anchor the scene identity, while foreground regions remain editable.
Local noise perturbation is applied to the first frame during training to enhance inpainting capabilities.
Initialized based on Wan2.1-T2V-1.3B, with a learning rate of $2 \times 10^{-4}$ for the motion projector and $1 \times 10^{-5}$ for other components.
A mixed-resolution training strategy (33×256×512 and 33×480×960) is used to accelerate condition modeling convergence.
Data sampling rate is 10 Hz, trained on 16 NVIDIA A100 80GB GPUs.

## Key Experimental Results

### Main Results

| Task | Method | FVD↓ | VBench Score↑ |
|------|------|------|--------------|
| Multi-view Scene Layout-following | MagicDrive-V2 | 152.80 | 77.23% |
| | **CompoSIA** | **133.66** | **81.05%** |
| Identity Control | DriveEditor | 179.57 | 79.13% |
| | LoRA-Edit | 161.32 | 79.83% |
| | **CompoSIA** | **149.15** | 80.30% |
| Ego-motion Control | Vista | 171.49 | 75.35% |
| | ReCamMaster | 190.52 | 74.29% |
| | **CompoSIA** | **137.21** | **80.79%** |

### Downstream Stress Test

| Editing Modality | 3s Collision Rate Increase |
|---------|-------------|
| Average across Editing Modalities | **+173%** |

### Key Findings
- Identity editing FVD is 17% lower than DriveEditor (149.15 vs 179.57), showing stronger generalization across element poses.
- Ego-motion control reduces rotation error by 30% and translation error by 47%.
- Synthesized adversarial scenarios can expose hidden fault modes of planners—increasing the collision rate by 173%.
- Disentangled design maintains high quality for both independent editing and compositional editing.

## Highlights & Insights
- **Hierarchical Design Principle for Control Signal Injection**: Injecting different factors at distinct stages of the diffusion process is a design principle that transcends specific implementations and can be transferred to other conditional generation tasks.
- **Inpainting Paradigm for Identity Injection**: Modeling identity control as region replacement and inpainting at the noise level rather than as attention conditions elegantly avoids the identity-motion trade-off.
- **Systematic Approach to Adversarial Scenario Synthesis**: Rather than random synthesis, it systematically constructs "abnormal combinations of common elements" through control strategies, functioning more like a controllable stress-testing tool.
- **PRoPE + AdaLN Dual-branch Ego-motion Control**: Fusing local residual signals with global camera trajectories, where the former accelerates convergence and the latter improves precision.

## Limitations & Future Work
- Trained on nuScenes + 100 hours of in-house data, scenario diversity remains limited.
- Control is limited to elements with existing 3D bbox annotations; unannotated entities (e.g., temporary construction zones) cannot be controlled.
- Synthetic video resolution (256×512 or 480×960) may be insufficient for certain downstream tasks.
- The actual improvement in downstream planner training from the realism of generated images has not been analyzed.

## Related Work & Insights
- **vs MagicDrive-V2**: MagicDrive-V2 injects multiple conditions through a shared path, leading to coupling and quality degradation; CompoSIA avoids conflicts by disentangling injection paths.
- **vs DriveEditor**: DriveEditor only supports structure and identity editing without ego-motion control, and identity editing is limited to pose-aligned images.
- **vs Vista**: Vista supports ego-motion and scene-identity control but lacks element-level structure and identity control.
- **vs GAIA-2**: GAIA-2 supports structure + scene-identity + motion control but lacks element-level identity control; CompoSIA further disentangles element identity.
- **vs TTM**: TTM performs identity editing at the noise level but uses a training-free strategy, making it difficult to achieve identity control under precise motion; CompoSIA addresses this via a trained inpainting paradigm.
- **Insights for Autonomous Driving Safety Testing**: CompoSIA can serve as a controllable stress-testing tool, systematically constructing adversarial scenarios to expose planner vulnerabilities—offering more targeted evaluation than random data augmentation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The hierarchical injection design disentangling three control signals + the noise-level inpainting paradigm for identity is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional comparisons + downstream stress testing, but lacks user studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured paper; the logic chain from motivation to method and experiments is complete, accompanied by rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Provides a controllable simulation tool as a new paradigm for autonomous driving safety testing and data augmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Steerable Adversarial Scenario Generation through Test-Time Preference Alignment (SAGE)](../../ICLR2026/autonomous_driving/steerable_adversarial_scenario_generation_through_test-time_preference_alignment.md)
- [\[CVPR 2025\] Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](scenario_dreamer_vectorized_latent_diffusion_for_generating_driving_simulation_e.md)
- [\[CVPR 2025\] UniScene: Unified Occupancy-centric Driving Scene Generation](uniscene_unified_occupancy-centric_driving_scene_generation.md)
- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](../../ICCV2025/autonomous_driving/dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](generative_gaussian_splatting_for_unbounded_3d_city_generation.md)

</div>

<!-- RELATED:END -->
