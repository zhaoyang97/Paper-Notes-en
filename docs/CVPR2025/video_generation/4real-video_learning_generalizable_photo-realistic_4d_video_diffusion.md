---
title: >-
  [Paper Note] 4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion
description: >-
  [CVPR 2025][Video Generation][4D Video Generation] This work proposes 4Real-Video, a 4D video generation framework based on a two-stream architecture. By splitting video tokens into parallel time and view streams and introducing hard/soft synchronization layers to harmonize information between them, it generates high-quality $8 \times 8$ spatio-temporal video grids in approximately 1 minute, outperforming existing methods in visual quality and multi-view consistency.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "4D Video Generation"
  - "Two-Stream Architecture"
  - "Synchronizing Layers"
  - "Video Diffusion"
  - "Multi-View Consistency"
date: 2026-05-08
content_hash: de50dae06a7ff8d8
---

# 4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2412.04462](https://arxiv.org/abs/2412.04462)  
**Code**: [https://snap-research.github.io/4Real-Video/](https://snap-research.github.io/4Real-Video/) (No standalone repository)  
**Area**: Video Generation / 4D Generation  
**Keywords**: 4D Video Generation, Two-Stream Architecture, Synchronizing Layers, Video Diffusion, Multi-View Consistency

## TL;DR
This work proposes 4Real-Video, a 4D video generation framework based on a two-stream architecture. By splitting video tokens into parallel time and view streams and introducing hard/soft synchronization layers to harmonize information between them, it generates high-quality $8 \times 8$ spatio-temporal video grids in approximately 1 minute, outperforming existing methods in visual quality and multi-view consistency.

## Background & Motivation

With the rise of video diffusion models (such as Sora), 4D video generation—generating a grid of video frames that vary along both temporal and viewpoint axes simultaneously—has emerged as an important extension. This capability is crucial for dynamic scene creation, immersive experiences, and image-based rendering.

**Ours: Definition of 4D Video**: A grid of frames where rows share timestamps and columns share viewpoints. Distinct from "camera-aware videos" (which generate only a single video path with camera control), a 4D video grid provides a complete space-time experience and is more amenable to dynamic reconstruction.

**Limitations of Prior Work**:

**Optimization-based methods** (e.g., 4Dfy, Dream-in-4D): These distill pretrained models using SDS, which takes hours and tends to produce object-centric/non-photorealistic outputs.

**Direct 4D training methods** (e.g., SV4D, Diffusion4D): Trained on limited 4D datasets (such as Objaverse animations), their generalization capability is constrained by the training data distribution.

**Sequential interleaving methods** (e.g., CVD): Alternating view attention and temporal attention updates, these fail to fully capture mutual dependencies; the outputs of view updates are out-of-distribution for temporal updates, leading to visual artifacts.

**Key Challenge**: ① High-quality 4D data is extremely scarce; ② sequential interleaving architectures break the prior distribution of pretrained video models, leading to quality degradation; ③ existing methods are either too slow (SDS taking several hours) or yield poor quality.

**Key Insight**: Design a parallel two-stream architecture to reuse pretrained video model weights, coordinating temporal and view consistency through synchronization layers to avoid distribution shifts. Core Idea: **Model 4D generation as the joint optimization of two video tasks, utilizing variable splitting and synchronization to maintain consistency between the two streams.**

## Method

### Overall Architecture
- **Input**: The first row of frames (fixed-view video) + the first column of frames (freeze-time video)
- **Output**: Complete $V \times T$ frame grid (default is 8×8)
- **Two Stages**: ① Train a base video model that supports both freeze-time and dynamic modes; ② train the two-stream 4D video model on top of it.

### Key Designs

1. **Two-Stream Architecture**:

    - Function: Split the video frame grid tokens into two independent streams processed in parallel—the view stream processes rows (freeze-time video) and the temporal stream processes columns (fixed-view video).
    - Mechanism: Copy the frame grid tokens into two copies, $\mathbf{x}_l^{\text{v}}$ and $\mathbf{x}_l^{\text{t}}$. In each layer, the view stream performs $T$ parallel row updates using the pretrained DiT, while the temporal stream performs $V$ parallel column updates:
    $\mathbf{y}_l^{\text{v}} = \mathbf{x}_l^{\text{v}} + \varphi_l^{\text{v}}(\mathbf{x}_l^{\text{v}}; \mathbf{c}^{\text{v}}); \quad \mathbf{y}_l^{\text{t}} = \mathbf{x}_l^{\text{t}} + \varphi_l^{\text{t}}(\mathbf{x}_l^{\text{t}}; \mathbf{c}^{\text{t}})$
   The two streams calculate independently and then exchange information through synchronization layers.
    - Design Motivation: Compared to sequential interleaving, the parallel two-stream design prevents the output of one stream from becoming an out-of-distribution input for the other. The pretrained DiT layers are not fine-tuned; only the parameters of the newly added synchronization layers are trained, preventing degradation of the pretrained video model's generation quality.

2. **Synchronization Layer—Hard Synchronization**:

    - Function: Merge the tokens from the two streams strictly after each DiT layer.
    - Mechanism: Merge the two streams via learnable weighting to satisfy the constraint $\mathbf{x}^{\text{v}} = \mathbf{x}^{\text{t}}$:
    $\mathbf{x}_{l+1} = \mathbf{W}_l^{\text{v}} \mathbf{y}_l^{\text{v}} + \mathbf{W}_l^{\text{t}} \mathbf{y}_l^{\text{t}}$
   The weights are initialized to $\frac{1}{2}\mathbf{I}$ and modulated by the diffusion timestep $\sigma$ to adapt to different denoising stages.
    - Design Motivation: Analogy to projected gradient descent in optimization—projecting the two variables onto the equality constraint manifold at each step. However, experiments reveal that this produces stretching artifacts under large view changes because the merged tokens deviate from the base model's distribution.

3. **Synchronization Layer—Soft Synchronization**:

    - Function: Keep the tokens of both streams independent while encouraging them to converge towards consistency via soft updates.
    - Mechanism: Predict asymmetric token increments using a timestep-modulated linear layer:
    $(\Delta\mathbf{y}_l^{\text{v}}, \Delta\mathbf{y}_l^{\text{t}}) = \text{Mod\_Linear}(\mathbf{y}_l^{\text{v}}, \mathbf{y}_l^{\text{t}}; \sigma)$
    $\mathbf{x}_{l+1}^{\text{v}} = \mathbf{y}_l^{\text{v}} + \Delta\mathbf{y}_l^{\text{v}}, \quad \mathbf{x}_{l+1}^{\text{t}} = \mathbf{y}_l^{\text{t}} + \Delta\mathbf{y}_l^{\text{t}}$
    - Design Motivation: Analogy to optimization algorithms like ADMM that do not strictly satisfy constraints at each step but gradually converge. This provides the model with more flexibility, allowing different layers to adaptively adjust synchronization strength. Experiments show that synchronization strength automatically increases in deeper layers, while shallower layers allow some degree of divergence between the two streams.

4. **Base Video Model Training**:

    - Function: Train a base video model that supports both freeze-time and dynamic generation modes.
    - Mechanism: Classify training data into dynamic videos and static-scene videos, using different context embeddings to control the generation mode. A random masking training strategy is employed, enabling the model to predict unseen frames from an arbitrary subset of frames, thereby supporting autoregressive extension.
    - Design Motivation: Provide a high-quality single-dimension video generation foundation for the 4D model; masking training enables the model to flexibly accept different conditional inputs.

### Loss & Training
- **Base model**: Pixel-space diffusion (non-latent), progressive resolution training (36×64 → 72×128), trained for 12 days on 24×A100 GPUs.
- **4D model**: Velocity matching loss (rectified flow), two-stage training—first trained on pseudo-4D videos derived from 2D affine transformations for 20k steps, then fine-tuned on Animated Objaverse for 3k steps.
- During fine-tuning, **only the parameters of the synchronization layers are updated**, while the pretrained DiT layers remain frozen.
- A diffusion upsampler is used to increase the resolution from 72×128 to 288×512.

## Key Experimental Results

### Main Results

| Method | FID ↓ | CLIP ↑ | FVD ↓ | Visual Quality ↑ | Temporal Consist. ↑ |
|------|-------|--------|-------|-------------------|---------------------|
| SV4D | 204.81 | 19.46 | 1053.10 | 2.26/2.02 | 2.03/1.68 |
| MotionCtrl | 87.10 | 20.20 | 1556.36 | 2.36/2.30 | 2.38/2.25 |
| Sequential | 96.64 | 28.16 | 1662.54 | 2.30/2.28 | 2.21/2.15 |
| Hard Sync | 79.92 | 28.16 | 972.87 | 2.42/2.40 | 2.40/2.33 |
| **Soft Sync** | **78.36** | **28.22** | **906.16** | **2.43/2.42** | **2.41/2.36** |

### Ablation Study

| Configuration | τ=2.0 ↑ | τ=2.5 ↑ | τ=3.0 ↑ |
|------|---------|---------|---------|
| Sequential | 33.5 | 24.6 | 16.6 |
| Soft w/o Obj | 39.1 | 31.4 | 24.0 |
| Hard Sync | 39.3 | 31.5 | 23.8 |
| **Soft Sync** | **41.0** | **33.4** | **25.7** |

### Key Findings
- **Soft Sync > Hard Sync > Sequential**: Soft synchronization outperforms both hard synchronization and sequential interleaving across all metrics.
- Even without using any 4D data (Soft w/o Obj), training solely on pseudo-data generated from 2D transformations still yields competitive results.
- In the user study, the proposed method substantially outperforms optimization-based methods (4Dfy, Dream-in-4D, AYG, 4Real) across all 7 evaluation dimensions.
- Generation takes approximately 1 minute (8×8 @ 288×512), compared to SDS-based methods which require several hours.
- Fine-tuning on Objaverse requires only 3k steps; however, over-tuning degrades performance on real-world scenes.

## Highlights & Insights
- **Architecture Design from an Optimization Theory Perspective**: The DiT layers are analogized to iterative solvers in implicit optimization, framing 4D generation as a variable splitting and constrained optimization problem where hard and soft synchronizations correspond to projected gradient descent and ADMM, respectively. This offers an elegant theoretical intuition.
- Training only the synchronization layers (with very few parameters) while completely freezing the pretrained DiT weights maximally preserves the generalization capability of the base model.
- Pixel-space diffusion models converge faster and yield more coherent motion at smaller model scales compared to latent-based alternatives.
- Eliminates the need for explicit camera pose conditioning, automatically inferring viewpoints from the conditioning videos.

## Limitations & Future Work
- The base model features only 600M parameters, which limits the upper bound of visual quality and resolution.
- Does not support 360-degree view generation.
- Exhibits limited robustness in freeze-time video generation for highly dynamic elements (such as running horses or fire).
- Post-processing steps (e.g., 3DGS reconstruction) are required to obtain explicit 3D representations.
- Training on pseudo-4D data (2D affine transformations) can cause foreground objects to appear flat under large viewpoint variations.

## Related Work & Insights
- 4Real (prior work) generated freeze-time and dynamic videos via video models followed by optimization-based reconstruction; ours transitions this process into a feed-forward generation pipeline.
- CVD proposed fine-tuning video models on pseudo-paired data to generate structurally consistent video pairs, though its sequential interleaving architecture possesses inherent limitations.
- SV4D trains 4D models directly on Objaverse, which fails to generalize to real-world scenes.
- Core Insight: **Decomposing the 4D generation task into the joint optimization of two 1D video problems and maintaining consistency through synchronization layers—rather than shared tokens—presents an elegant paradigm for multi-axis generation.**

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The two-stream and synchronization-layer architecture is highly novel, backed by a profound optimization-theory analogy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive metrics (6 types of evaluation) and clear ablation studies, though limited by the lack of true 4D ground truth.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly defined problems, with a highly logical derivation of the architectural design motivated by optimization theory.
- Value: ⭐⭐⭐⭐ The first highly efficient 4D video generation method capable of generalizing to real scenes, holding significant potential for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)
- [\[CVPR 2026\] VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control](../../CVPR2026/video_generation/versecrafter_dynamic_realistic_video_world_model_with_4d_geometric_control.md)
- [\[CVPR 2025\] Learning from Streaming Video with Orthogonal Gradients](learning_from_streaming_video_with_orthogonal_gradients.md)
- [\[CVPR 2025\] Towards Precise Scaling Laws for Video Diffusion Transformers](towards_precise_scaling_laws_for_video_diffusion_transformers.md)
- [\[CVPR 2026\] WorldReel: 4D Video Generation with Consistent Geometry and Motion Modeling](../../CVPR2026/video_generation/worldreel_4d_video_generation_with_consistent_geometry_and_motion_modeling.md)

</div>

<!-- RELATED:END -->
