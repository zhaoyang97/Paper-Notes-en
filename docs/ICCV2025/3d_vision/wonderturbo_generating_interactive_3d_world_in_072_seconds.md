---
title: >-
  [Paper Note] WonderTurbo: Generating Interactive 3D World in 0.72 Seconds
description: >-
  [ICCV 2025][3D Vision][Real-time 3D scene generation] WonderTurbo proposes the first real-time interactive 3D scene generation framework. Through the coordinated acceleration of three modules — StepSplat (feed-forward 3D…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Real-time 3D scene generation"
  - "Gaussian splatting"
  - "depth completion"
  - "diffusion distillation"
  - "interactive generation"
date: 2026-05-08
content_hash: 6c96c4376e462942
---

# WonderTurbo: Generating Interactive 3D World in 0.72 Seconds

**Conference**: ICCV 2025
**arXiv**: [2504.02261](https://arxiv.org/abs/2504.02261)  
**Code**: [https://github.com/GigaAI-research/WonderTurbo](https://github.com/GigaAI-research/WonderTurbo)  
**Area**: 3D Vision
**Keywords**: Real-time 3D scene generation, Gaussian splatting, depth completion, diffusion distillation, interactive generation

## TL;DR

WonderTurbo proposes the first real-time interactive 3D scene generation framework. Through the coordinated acceleration of three modules — StepSplat (feed-forward 3DGS), QuickDepth (lightweight depth completion), and FastPaint (2-step diffusion inpainting) — it compresses single-step scene extension time from 10+ seconds to 0.72 seconds, achieving a 15× speedup while maintaining generation quality comparable to WonderWorld.

## Background & Motivation

**Background**: Interactive 3D scene generation from a single image (online 3D scene generation) is a core technology for immersive virtual experiences. Existing methods fall into two categories: offline methods such as LucidDreamer and Text2Room first generate multi-view images and then optimize a 3D representation, while online methods such as WonderJourney and WonderWorld support incremental, user-driven scene creation.

**Limitations of Prior Work**: Even the fastest existing online method, WonderWorld, requires approximately 10 seconds to generate a new viewpoint — far from satisfying real-time interaction requirements. The efficiency bottleneck stems from two aspects: (1) geometry modeling relies on per-scene iterative optimization of 3DGS, requiring hundreds of iterations; and (2) appearance modeling relies on diffusion-based image inpainting, requiring tens of inference steps.

**Key Challenge**: There exists a severe conflict between the quality requirements of 3D scene generation and the demand for real-time response — high-quality geometry and appearance modeling both require substantial computation, while interactive scenes demand sub-second latency.

**Goal**: The paper aims to design a complete framework capable of finishing one scene extension within 1 second without significant degradation in geometry or appearance quality. This is decomposed into three sub-problems: how to accelerate geometry modeling, how to provide consistent depth priors, and how to accelerate appearance inpainting.

**Key Insight**: The authors observe that feed-forward 3DGS methods (e.g., MVSplat, PixelSplat) can directly infer Gaussian parameters without iterative optimization, but do not support interactive scenarios where viewpoints are added incrementally. Meanwhile, diffusion model distillation techniques can compress inference steps to an extremely small number. Combining these two directions can simultaneously address the efficiency bottlenecks in both geometry and appearance.

**Core Idea**: Feed-forward 3DGS with feature memory enables incremental geometry modeling; lightweight depth completion provides consistent depth priors; 2-step distilled diffusion enables instant appearance inpainting. The combination of these three components achieves real-time 3D interaction in 0.72 seconds.

## Method

### Overall Architecture

The WonderTurbo pipeline operates as follows: after the user moves the camera to a new position, the system first renders the current 3D scene to obtain image $I_{render}^i$ and depth map $D_{render}^i$. FastPaint then takes the rendered image and user text as input to generate appearance for new regions $I_{target}^i$ (0.22s); QuickDepth takes the rendered depth and new appearance image as input to complete the depth of new regions $D_{target}^i$ (0.24s); finally, StepSplat converts the new image and depth into a local Gaussian representation and incrementally merges it into the global 3D scene (0.26s). The total time for one scene extension step is 0.72 seconds.

### Key Designs

1. **StepSplat — Incremental Feed-Forward 3DGS**:

    - **Function**: Converts the image and depth of a new viewpoint into a 3D Gaussian representation and merges it into the global scene within 0.26 seconds.
    - **Mechanism**: StepSplat uses RepVGG as the backbone to extract matching features $F_m^i$ and image features $F_e^i$. A Feature Memory is introduced to store matching features and poses from historical viewpoints. For each new viewpoint, the $N_v$ nearest neighboring viewpoints are selected based on pose distance $d(P_n, P_i) = \|P_n - P_i\|_2$. Using the depth map provided by QuickDepth, $N_d$ depth candidate values are uniformly sampled within the depth range $R = \{d \mid (1-a) \cdot D_{target}^i \leq d \leq (1+a) \cdot D_{target}^i\}$. Neighboring features are warped to the current viewpoint via plane-sweep stereo, and normalized dot-product correlations are computed to construct a cost volume $S^i$. The final depth prediction is obtained via softmax-weighted averaging: $\hat{d} = \text{softmax}(S^i) \cdot d$, and depth values are back-projected as Gaussian centers. The incremental fusion strategy removes conflicting Gaussians using a depth consistency constraint $|d_{local} - d_j^g| < \delta \cdot d_{local}$, and merges valid local Gaussians into the global representation.
    - **Design Motivation**: Traditional 3DGS requires hundreds of iterative optimization steps, whereas feed-forward methods directly infer Gaussian parameters but do not support incrementally added viewpoints. StepSplat extends the feed-forward paradigm to interactive scenes by maintaining a feature memory and adaptively constructing cost volumes. The depth-guided cost volume ensures geometric accuracy, while incremental fusion reduces floating-point artifacts caused by redundant Gaussians.

2. **QuickDepth — Lightweight Depth Completion**:

    - **Function**: Completes a full depth map for newly generated regions within 0.24 seconds, providing a consistent depth prior.
    - **Mechanism**: Initialized from a lightweight depth estimation model (Depth Anything), the network takes the RGB image of the target frame, an incomplete depth map, and a binary mask as input to predict a complete depth map. Training data is constructed via geometric projection between adjacent frames — the depth map $D_{j-1}$ from the previous frame is projected into the current frame's coordinate system using relative pose $T_{j-1 \to j}$, yielding an incomplete depth $D'_{j-1 \to j}$ and a validity mask $M_{j-1 \to j}$. Training is supervised with an $L_1$ loss.
    - **Design Motivation**: Existing depth completion methods primarily target sparse depth completion (e.g., from LiDAR) and struggle with the large regions lacking depth information characteristic of 3D scene generation. WonderWorld's guided depth diffusion approach requires more than 3 seconds. QuickDepth is specifically trained on mask patterns from interactive 3D generation, yielding strong generalization.

3. **FastPaint — 2-Step Diffusion Inpainting**:

    - **Function**: Completes image inpainting within 0.22 seconds to generate appearance for new regions.
    - **Mechanism**: Knowledge distillation is applied to a pretrained Stable Diffusion Inpainting model, combined with ODE trajectory preservation and reconstruction strategies, compressing inference from tens of steps to only 2 steps. The model is further fine-tuned on a dataset that simulates the mask patterns encountered in interactive 3D generation, aligning the inpainting region distribution with that of actual 3D scene generation.
    - **Design Motivation**: The original diffusion inpainting model requires approximately 20–50 inference steps, and the mask distribution used during fine-tuning differs from that of 3D generation scenes, leading to quality degradation or the need for additional VLM verification. FastPaint simultaneously addresses speed and quality through distillation and targeted fine-tuning.

### Training Data Construction

The authors leverage multiple 3D scene generation methods (WonderJourney, WonderWorld, Text2Room, LucidDreamer, etc.) to construct a training dataset comprising over 6 million frames, covering indoor environments (32%), urban landscapes (28%), natural terrain (25%), and stylized artistic scenes (15%). Training data is generated by simulating interactive trajectories (rotational, linear, and mixed), with data quality verified using a VLM.

## Key Experimental Results

### Main Results

| Method | Type | Geometry (s) | Appearance (s) | Total (s) |
|--------|------|-------------|---------------|----------|
| LucidDreamer | Offline | 35.38 | 8.32 | 43.70 |
| Text2Room | Offline | 34.23 | 7.32 | 41.55 |
| Pano2Room | Offline | 27.91 | 1.47 | 29.38 |
| DreamScene360 | Offline | 44.29 | 1.45 | 45.74 |
| WonderJourney | Online | 78.12 | 1.45 | 79.57 |
| WonderWorld | Online | 6.62 | 4.43 | 11.05 |
| **WonderTurbo** | **Online** | **0.50** | **0.22** | **0.72** |

| Method | CS↑ | CC↑ | CIQA↑ | Q-Align↑ | CA↑ |
|--------|-----|-----|-------|----------|-----|
| LucidDreamer | 27.72 | 0.9213 | 0.6023 | 3.5439 | 6.8231 |
| Text2Room | 24.50 | 0.9035 | 0.4910 | 2.6732 | 6.5324 |
| WonderJourney | 27.63 | 0.9652 | 0.4753 | 3.5272 | 7.0134 |
| WonderWorld | 28.14 | 0.9654 | 0.6764 | 3.7823 | 7.2121 |
| **WonderTurbo** | **28.65** | **0.9732** | **0.6812** | 3.7253 | **7.3243** |

### Ablation Study

| Configuration | CS↑ | CC↑ | CIQA↑ | Q-Align↑ | CA↑ |
|--------------|-----|-----|-------|----------|-----|
| w/ FreeSplat | 27.65 | 0.9542 | 0.6460 | 3.1543 | 6.6235 |
| w/ DepthSplat | 27.32 | 0.9675 | 0.6620 | 3.2145 | 6.7432 |
| w/o depth guided | 27.72 | 0.9532 | 0.6359 | 3.4361 | 7.1734 |
| w/o incremental fusion | 27.87 | 0.9654 | 0.6459 | 3.5431 | 7.2734 |
| w/o FastPaint | 27.82 | 0.9683 | 0.6574 | 3.7146 | 7.2136 |
| **WonderTurbo (full)** | **28.65** | **0.9732** | **0.6812** | **3.7253** | **7.3243** |

### Key Findings

- StepSplat demonstrates significant advantages over FreeSplat and DepthSplat, particularly on Q-Align (+0.57/+0.51) and CLIP aesthetic (+0.70/+0.58), indicating that the depth-guided cost volume is critical for geometric accuracy.
- Removing the depth-guided cost volume causes the most severe performance drop (CS −0.93, CC −0.020), as the cost volume search range becomes too large without a depth prior, leading to geometric inaccuracies.
- In the user study, WonderTurbo achieves a win rate of 69.43% against WonderWorld and exceeds 94% against all other methods, demonstrating that perceptual quality is largely preserved despite a 15× speedup.
- FastPaint's contribution is most evident in CS (+0.83) and CIQA (+0.124), indicating that targeted fine-tuning improves semantic consistency between inpainted regions and text prompts.

## Highlights & Insights

- **The combination of feed-forward paradigm and feature memory** is particularly elegant: by retaining matching features from historical viewpoints rather than raw images, the approach exploits the speed advantage of feed-forward inference while achieving multi-view information fusion akin to iterative optimization. This design is transferable to any scenario requiring incremental 3D reconstruction.
- **The design of the depth-guided cost volume**: using QuickDepth's depth prediction as a prior to constrain the depth search range of the cost volume is an elegant application of the classic coarse-to-fine idea from MVS to interactive generation. Searching only within ±a of the depth prediction substantially reduces computation while improving accuracy.
- **The training data construction strategy is noteworthy**: leveraging the complementary strengths of multiple existing 3D generation methods to build training data, with VLM-based quality verification, exemplifies a practical bootstrap strategy for data-scarce settings.

## Limitations & Future Work

- The codebase has not been fully open-sourced; the GitHub repository serves as a placeholder, making reproduction and verification difficult.
- Although 0.72 seconds approaches real-time, it remains far from true 30 fps interactive performance (requiring ≤33 ms), and is still insufficient for high-frame-rate scenarios such as VR/AR.
- Training data relies on the outputs of other 3D generation methods, and data quality is bounded by the capability ceiling of those methods.
- While the user study reports high win rates, quantitative evaluation of geometric accuracy (e.g., point cloud precision, mesh quality) is absent.
- FastPaint, being distillation-based, may underperform full-step diffusion models on complex textures and fine-grained details.

## Related Work & Insights

- **vs. WonderWorld**: WonderWorld uses FLAGS combined with diffusion-guided depth estimation, still requiring 10 seconds. WonderTurbo replaces iterative optimization with feed-forward StepSplat and QuickDepth, achieving a 15× speedup with comparable quality.
- **vs. MVSplat/DepthSplat**: These feed-forward 3DGS methods are designed for fixed dual-view inputs and do not support interactive scenarios with incrementally added viewpoints. StepSplat extends the feed-forward paradigm through feature memory and incremental fusion.
- **vs. Hyper-SD and other distillation methods**: FastPaint adopts a similar ODE trajectory distillation approach but is specifically adapted for the inpainting task and the mask distribution characteristic of 3D generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combined innovation across three modules is solid, though the technical contribution of each individual module is relatively incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative comparisons, ablation studies, and user studies are all present, but quantitative geometric accuracy evaluation is missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with a clear pipeline description and rigorous mathematical derivations.
- **Value**: ⭐⭐⭐⭐⭐ — Real-time interactive 3D generation addresses a strong practical need; the engineering value of a 15× speedup is substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bolt3D: Generating 3D Scenes in Seconds](bolt3d_generating_3d_scenes_in_seconds.md)
- [\[ICML 2026\] PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](../../ICML2026/3d_vision/physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)
- [\[ICCV 2025\] A Recipe for Generating 3D Worlds from a Single Image](a_recipe_for_generating_3d_worlds_from_a_single_image.md)
- [\[ICCV 2025\] Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting](text2vdm_text_to_vector_displacement_maps_for_expressive_and_interactive_3d_scul.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)

</div>

<!-- RELATED:END -->
