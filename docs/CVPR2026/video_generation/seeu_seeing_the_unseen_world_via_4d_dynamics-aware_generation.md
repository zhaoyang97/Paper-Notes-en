---
title: >-
  [Paper Note] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation
description: >-
  [CVPR 2026][Video Generation][4D dynamic modeling] SeeU is a 2D→4D→2D learning framework that reconstructs a 4D world representation from sparse monocular 2D frames, learns continuous and physically consistent 4D dynamics on a low-rank representation (B-spline parameterization + physical constraints), and reprojects the 4D world back to 2D, completing unseen regions with a spatiotemporally context-aware video generator—enabling generation of unseen visual content across time and space.
tags:
  - CVPR 2026
  - Video Generation
  - 4D dynamic modeling
  - continuous dynamics
  - spatiotemporal generation
  - B-spline
  - physical consistency
date: 2026-05-08
content_hash: 233c928507903659
---

# SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation

**Conference**: CVPR 2026
**arXiv**: [2512.03350](https://arxiv.org/abs/2512.03350)
**Code**: [https://yuyuanspace.com/SeeU/](https://yuyuanspace.com/SeeU/) (data and code publicly available)
**Area**: Video Generation
**Keywords**: 4D dynamic modeling, continuous dynamics, spatiotemporal generation, B-spline, physical consistency

## TL;DR
SeeU is a 2D→4D→2D learning framework that reconstructs a 4D world representation from sparse monocular 2D frames, learns continuous and physically consistent 4D dynamics on a low-rank representation (B-spline parameterization + physical constraints), and reprojects the 4D world back to 2D, completing unseen regions with a spatiotemporally context-aware video generator—enabling generation of unseen visual content across time and space.

## Background & Motivation

1. **Background**: Tasks such as video generation, frame interpolation, and frame prediction predominantly model dynamics end-to-end in 2D pixel or latent spaces. Large-scale video diffusion models (e.g., Sora, Wan) perform well on in-distribution scenes. World model approaches learn dynamics in low-dimensional latent spaces for efficiency.
2. **Limitations of Prior Work**: Modeling dynamics directly on 2D frames entails three fundamental limitations: (a) images and videos are discrete 2D projections of a 4D world (3D space + time), so learning directly in 2D discards important 3D structure and temporal correlations; (b) observations conflate camera motion and scene dynamics, with continuously varying camera poses adding complexity and irregularity to apparent motion; (c) in complex out-of-distribution scenes (occlusions, non-rigid deformations, etc.), 2D models lacking 3D or physical supervision often fail to capture true geometric and physical dynamics.
3. **Key Challenge**: Real-world motion is typically simple and structured in 4D space—governed by biological/mechanical constraints, classical mechanics, symmetry, etc.—but becomes complex and ill-posed when projected to 2D. Modeling dynamics in 4D naturally exploits these physical priors, yet existing methods either remain in 2D or lack continuous dynamics modeling in 4D reconstruction.
4. **Goal**: (1) How to reconstruct a 4D dynamic scene from sparse monocular frames? (2) How to learn continuous and physically consistent 4D dynamics? (3) How to generate 2D content at arbitrary times and viewpoints from the 4D world?
5. **Key Insight**: Three advantages of modeling continuous dynamics in native 4D space—3D awareness (explicit 3D representations handle occlusion and viewpoint changes), physical consistency (motion is simpler in 4D, enabling physical priors as constraints), and motion disentanglement (camera, foreground, and background can be explicitly separated in a unified 4D coordinate system).
6. **Core Idea**: Through a 2D→4D→2D information flow—first lifting 2D observations to a 4D world representation, then learning continuous physically consistent dynamics via B-splines in 4D, and finally projecting back to 2D with context-aware completion—the framework generates unseen content at arbitrary times and viewpoints.

## Method

### Overall Architecture
SeeU is a three-stage pipeline:
- **Stage 1 (2D→4D)**: Reconstructs a dynamic 4D scene from sparse monocular frames, yielding a set of 3D Gaussian primitives with per-frame transformations and camera poses.
- **Stage 2 (Discrete 4D→Continuous 4D)**: Learns a continuous temporal dynamics function on a low-rank motion basis using B-spline parameterization, with physical regularization to ensure smoothness and physical consistency.
- **Stage 3 (4D→2D)**: Evolves the 4D world to arbitrary times and viewpoints, renders 2D scaffold frames (which may be incomplete), and completes missing regions with a spatiotemporally context-aware video generator.

### Key Designs

1. **Dynamic Scene Reconstruction (2D→4D)**:

    - **Function**: Constructs a unified 4D representation from sparse monocular frames.
    - **Mechanism**: Built on the Shape-of-Motion framework. The scene is represented by a set of canonical 3D Gaussians $\{g_0^i\}_{i=1}^N$ (with position $\mu_0^i$, orientation $R_0^i$, scale $s^i$, opacity $o^i$, color $c^i$). Each Gaussian evolves from the canonical frame to frame $t$ via a per-frame rigid transformation $T_{0 \to t} \in SE(3)$: $\mu_t^i = R_{0 \to t} \mu_0^i + t_{0 \to t}$. Preprocessing employs MegaSaM for camera parameter and depth estimation, Track-Anything for foreground segmentation, and TAPIR for 2D point track extraction.
    - **Design Motivation**: Shape-of-Motion is adopted for its compatibility with casually captured inputs with weak parallax and its ability to explicitly separate static and dynamic regions.

2. **Continuous 4D Dynamics Model (C4DD)**:

    - **Function**: Fits discrete per-frame motion bases and camera poses to continuous temporal functions, supporting interpolation and extrapolation at arbitrary times.
    - **Mechanism**: Two challenges are addressed. **Efficiency**: The large number of foreground Gaussians (~80K) precludes learning independent trajectories for each. A low-rank motion parameterization is adopted: $P_t^i = P_0^i + B(t) w_i$, where $B(t) \in \mathbb{R}^{m \times K}$ is a globally shared motion basis ($K \ll N$) and $w_i$ are time-invariant per-Gaussian coefficients. **Physical consistency**: Motion basis trajectories in $SE(3)$ exhibit simple, smooth temporal trends (even when raw video motion appears complex), so cubic B-splines are used for parameterization: $\hat{B}_t = \sum_{j=1}^M N_{j,d}(t) q_j$. Camera and motion bases are jointly optimized with a data loss $\mathcal{L}_{data}$ to fit discrete observations, and a physical loss $\mathcal{L}_{phys}$ penalizes translational and rotational acceleration of motion bases and camera trajectories, with higher weights in extrapolation regions.
    - **Design Motivation**: B-splines possess a natural smoothness inductive bias over MLPs (ablations show MLP variants produce noisy, non-smooth trajectories). The number of control points $M$ governs the capacity–smoothness trade-off. The physical loss prevents non-physical abrupt changes.

3. **Spatiotemporal Context-aware Video Generation (4D→2D)**:

    - **Function**: Projects the 4D world into potentially incomplete 2D scaffold frames and completes unseen regions with a generative model.
    - **Mechanism**: The continuous dynamics learned by C4DD are used to evolve the scene to arbitrary timestamps and camera poses, rendering 2D projections as a video "skeleton." Three types of regions in the skeleton require completion: (1) never-observed regions (novel viewpoints/occluded areas), (2) low-confidence projected Gaussian regions, and (3) projection artifacts at depth discontinuities. The VACE video generation model is employed, conditioned on three contextual priors: structured text prompts generated by a VLM (global semantics + inpainting instructions), projected frames (geometric and photometric references), and per-frame inpainting masks (marking uncertain regions).
    - **Design Motivation**: Pure reconstruction methods produce holes or artifacts in unseen regions; a generative model leveraging spatiotemporal context is required to complete fine details. The three priors provide a complete guidance chain from semantics and structure to spatial localization.

### Loss & Training
- Stage 1: 80K foreground + 80K background Gaussians, 10 motion bases, 4,000 iterations; typical runtime ~1 hour for 10 frames at 960×540.
- Stage 2: Cubic B-splines (degree=3), 8 control points, $\lambda_{phy} = 1 \times 10^{-4}$, lr=1e-5, batch=64, 1,000 epochs in ~10 minutes.
- Stage 3: VACE fine-tuned on a distribution of multi-semantic masks, ~2 hours.
- All stages run on a single A100 80GB GPU.

## Key Experimental Results

### Main Results
Temporal unseen generation (SeeU45 dataset):

| Method | Past PSNR↑ | Interp PSNR↑ | Future PSNR↑ | Past LPIPS↓ | Interp LPIPS↓ | Future LPIPS↓ |
|--------|-----------|-------------|-------------|------------|--------------|--------------|
| SoM | 15.55 | 16.37 | 15.43 | 0.388 | 0.356 | 0.389 |
| InterpAny | - | 20.54 | - | - | 0.242 | - |
| VACE | 17.14 | 18.16 | 17.71 | 0.367 | 0.359 | 0.354 |
| **SeeU** | **20.47** | **21.07** | **20.54** | **0.248** | **0.227** | **0.243** |

Spatial unseen generation (EE↓ lower is better, EIR↑ higher is better):

| Method | Dolly Out EE↓ | EIR↑ | CLIP-V↑ |
|--------|-------------|------|---------|
| ReCamMaster | 0.238 | 0.674 | 0.937 |
| **SeeU** | **0.200** | **0.785** | **0.969** |

### Ablation Study

| Configuration | PSNR↑ | LPIPS↓ | EE↓ | CLIP-V↑ |
|---------------|-------|--------|-----|---------|
| C4DD w/ MLP | 17.54 | 0.427 | 0.313 | 0.739 |
| w/o physics loss | 19.36 | 0.274 | 0.224 | 0.920 |
| 5 frames input | 18.36 | 0.305 | 0.285 | 0.928 |
| 10 frames input | 20.16 | 0.251 | 0.204 | 0.955 |
| 15 frames input | 20.39 | 0.241 | 0.200 | 0.958 |
| **20 frames input** | **21.08** | **0.239** | **0.197** | **0.960** |

### Key Findings
- **B-spline >> MLP**: The MLP variant reduces PSNR by 3.5 points and increases LPIPS by 0.19, demonstrating that the smooth inductive bias of B-splines is critical for continuous dynamics modeling—MLPs can fit the general trend but produce noisy trajectories.
- **Physical loss is important**: Removing $\mathcal{L}_{phys}$ causes significant degradation in inter-frame consistency, especially in extrapolation regions.
- **Robust to sparse inputs**: Reducing from 20 to 5 frames causes only ~2.7 PSNR drop, indicating C4DD maintains reasonable temporal continuity under extremely sparse observations.
- **Temporal prediction error grows approximately linearly**: Extrapolation accuracy degrades roughly linearly with temporal distance, consistent with physical intuition.
- SeeU comprehensively outperforms task-specific models across all three temporal sub-tasks (past inference, dynamic interpolation, and future prediction).

## Highlights & Insights
- **2D→4D→2D information flow paradigm**: Rather than performing end-to-end learning directly in 2D, the framework first lifts observations to 4D for world understanding and then returns to 2D for generation. This "understanding-first" paradigm stands in sharp contrast to purely data-driven generation. A transferable insight: for any generation task governed by physical laws, modeling in the physical space before projecting to the observation space may be preferable.
- **Low-rank motion parameterization + B-splines**: Two levels of simplification—first compressing the motion of 80K Gaussians into 10 basis functions via low-rank decomposition, then continuously parameterizing the discrete bases with B-splines. This hierarchical strategy for simplifying complex dynamics is both elegant and efficient.
- **Triple prior injection for unseen region completion**: The combination of text semantics, projected structure, and spatial masks provides the video generator with a complete guidance chain from "what should be generated" to "where to generate it."

## Limitations & Future Work
- Performance is bounded by the quality of underlying modules (tracking, camera estimation, 4D reconstruction)—small or textureless foreground objects can cause failures.
- The current approach focuses on scenes with prominent, smooth, and temporally stable foreground motion; highly non-rigid or abruptly changing motion is not well supported.
- Stage 1 4D reconstruction (~1 hour) is the efficiency bottleneck, precluding real-time applications.
- The SeeU45 dataset contains only 45 scenes; while diverse, its scale is limited.
- Extrapolation accuracy degrades linearly with time, and physical consistency in long-range prediction remains to be improved.

## Related Work & Insights
- **vs. Shape-of-Motion (SoM)**: Stage 1 of SeeU builds on SoM, but SoM provides only discrete per-frame reconstruction with linear interpolation/extrapolation for temporal transfer, yielding poor results (PSNR 15.5). SeeU adds continuous dynamics modeling and context-aware completion on top of SoM, improving PSNR to 20.5.
- **vs. VACE**: VACE is a purely 2D video inpainting method without 3D awareness. SeeU provides VACE with 4D-projected scaffold frames and precise masks, relieving VACE of the need to infer geometric structure.
- **vs. ReCamMaster**: A camera-controllable video generation model that lacks explicit 3D reconstruction. SeeU comprehensively surpasses it in geometric consistency (EE/EIR) and scene coherence (CLIP-V).
- **vs. physics-aware video generation**: Prior methods either generate first and then simulate physics, simulate first and then generate, or guide generation with distilled physical priors. SeeU directly infers deterministic dynamics from multi-frame observations and uses them as a physical skeleton for generation, representing a new paradigm.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The 2D→4D→2D information flow paradigm is conceptually innovative; learning continuous dynamics in native 4D space is a genuinely novel direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers both temporal and spatial dimensions with thorough ablations, though the dataset scale is limited (45 scenes).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is rigorously articulated (Section 2 provides a dedicated analysis of why dynamics should be modeled in 4D); figures and tables are clear.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for physically consistent video generation and world models, though practical applicability is currently constrained by efficiency and scene coverage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Seeing the Unseen: Zooming in the Dark with Event Cameras](../../AAAI2026/video_generation/seeing_the_unseen_zooming_in_the_dark_with_event_cameras.md)
- [\[CVPR 2026\] NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos](neoverse_enhancing_4d_world_model_with_in-the-wild_monocular_videos.md)
- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](../../ICLR2026/video_generation/geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[CVPR 2026\] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)
- [\[CVPR 2026\] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)

</div>

<!-- RELATED:END -->
