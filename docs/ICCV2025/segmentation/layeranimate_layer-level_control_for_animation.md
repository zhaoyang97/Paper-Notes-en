---
title: >-
  [Paper Note] LayerAnimate: Layer-level Control for Animation
description: >-
  [ICCV 2025][Segmentation][Animation Video Generation] This paper proposes LayerAnimate, a framework that integrates the layer-separation paradigm of traditional animation production with video diffusion models to enable…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Animation Video Generation"
  - "Layer Control"
  - "Video Diffusion Models"
  - "Data Curation"
  - "ControlNet"
date: 2026-05-08
content_hash: df8b76c92159f48d
---

# LayerAnimate: Layer-level Control for Animation

**Conference**: ICCV 2025
**arXiv**: [2501.08295](https://arxiv.org/abs/2501.08295)  
**Code**: [https://layeranimate.github.io](https://layeranimate.github.io)  
**Area**: Image Segmentation
**Keywords**: Animation Video Generation, Layer Control, Video Diffusion Models, Data Curation, ControlNet

## TL;DR

This paper proposes LayerAnimate, a framework that integrates the layer-separation paradigm of traditional animation production with video diffusion models to enable fine-grained layer-level control (motion scores, trajectories, sketches). An automated data curation pipeline is designed to address the scarcity of layered animation data. The framework comprehensively outperforms existing methods across six video generation tasks.

## Background & Motivation

Traditional animation production decomposes visual elements into discrete layers, each processed through sketching, refinement, coloring, and in-between interpolation. However, existing animation generation methods suffer from two core issues:

**Lack of layer-level control**: Existing methods treat animation as a data domain distinct from live-action video and support only frame-level control, disregarding the fundamental concept of "layers" in animation. Frame-level control leads to unpredictable deformations in regions without explicit control signals.

**Scarcity of layered data**: Professional animation assets are difficult to obtain due to commercial sensitivity, and the 2D nature of animation precludes the use of geometry-based methods such as depth estimation, making reliable frame-to-layer decomposition infeasible.

## Method

### Overall Architecture

LayerAnimate consists of two major components: (1) a layer data curation pipeline that automatically extracts layer information from existing animations; and (2) a video diffusion framework with layer-level control supporting flexible combinations of three control modalities—motion scores, trajectories, and sketches. Built upon the pretrained ToonCrafter UNet, the framework introduces a layer encoder, a control encoder, and a ControlNet branch to handle layer-level features, fusing them into the denoising UNet via cross-attention.

### Key Designs

1. **Automated Element Segmentation**: Keyframes are sampled at uniform 4-frame intervals. The first keyframe is segmented by SAM to obtain atomic element masks $\mathcal{M}_0$, which are propagated to all frames via SAM2 to initialize masklets. Newly appearing elements in subsequent frames are detected through iterative refinement: $\Delta\mathcal{M}_i = \text{SAM}(K_i) \setminus \mathcal{T}_{t_i}^{i-1}$. Mask prompts are updated and re-propagated, ensuring consistent extraction of dynamically appearing elements.

2. **Motion-based Hierarchical Merging (MHM)**: This component resolves the over-segmentation problem of SAM2. Optical flow is estimated using Unimatch, and a motion score (average flow magnitude, direction-agnostic) is computed for each masklet. Treating masklets as nodes, hierarchical clustering constructs a dendrogram based on motion scores, merging bottom-up those with similar scores until the layer count falls below a maximum capacity $N$ (default 4) and the motion score difference exceeds a threshold $\eta_s$ (default 1.0).

3. **Frame Decomposition & Motion-based Assignment**: The reference image is decomposed into layer regions $\mathbf{R} \in \mathbb{R}^{N \times 3 \times H \times W}$ using layer masks. For non-reference frames, static layers (motion score below threshold $\eta=0.1$) are copied from the reference frame, while dynamic layers are filled with zero images. This extends layer information from a single frame $\mathbf{M} \in \mathbb{R}^{N \times 1 \times H \times W}$ to the temporal dimension $\bar{\mathbf{M}} \in \mathbb{R}^{N \times F \times 1 \times H \times W}$.

4. **Three Layer-level Control Modalities**:

    - **Motion Score**: A scalar field normalized to $[0,1]$, concatenated with the layer mask after spatial and temporal alignment. Suitable for elements such as fire or particles that are difficult to describe via trajectories.
    - **Trajectory**: Point trajectories are tracked on a $60 \times 60$ grid using CoTracker3, filtered by masklet constraints to remove cross-layer tracks (retaining those with over 80% overlap), and converted to a three-channel map (Gaussian heatmap + normalized offsets). The heatmap resolves the zero-value ambiguity between static and uncontrolled regions in offset maps.
    - **Sketch**: Dense structural priors supporting partial sketches—only specific layers are provided while regions from other layers are randomly removed.

5. **Layer Feature Fusion**: Layer regions are encoded by the VAE encoder and concatenated with resized masks, then processed by the layer encoder $\varepsilon_l$. Control signals are encoded by the control encoder $\varepsilon_c$ (sketches use VAE + trainable convolutions). The encoded features are passed into ControlNet for independent per-layer processing. The resulting layer features $\mathbb{R}^{N \times F \times c \times h \times w}$ are fused into the UNet via cross-attention, where frame-level features serve as queries and layer features as keys/values. A validity mask ensures that only valid layers participate in attention.

### Loss & Training

Standard diffusion denoising objective: $\min \mathbb{E}_{\mathbf{z}_0, t, \epsilon \sim \mathcal{N}(0, \mathbf{I})} [\|\epsilon - \epsilon_\theta(\mathbf{z}_t; c, \bar{\mathbf{R}}, \bar{\mathbf{M}}, \mathbf{L}_c)\|_2^2]$

Training strategy:
- Random control selection: for each retained layer, motion score is selected with 20% probability, trajectory with 40%, and sketch with 40% (only one modality per sample).
- A 10% dropout probability is applied to layer masks to simulate incomplete user annotations.
- AdamW optimizer, lr=2e-5, 32× A100 GPUs, total batch size=96, 30,000 steps.
- Training resolution $320 \times 512$, 16 frames.

## Key Experimental Results

### Main Results (Table)

Comparison across six tasks on a 665K animation clips evaluation set:

| Task | Method | FVD↓ | FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|------|------|------|--------|-------|-------|
| I2V | DynamiCrafter | 114.80 | 14.36 | 0.354 | 14.89 | 0.554 |
| I2V | **LayerAnimate** | **87.96** | **14.66** | 0.370 | **15.45** | **0.556** |
| I2V+Trajectory | Tora | 190.61 | 22.03 | 0.376 | 15.32 | 0.525 |
| I2V+Trajectory | **LayerAnimate** | **72.04** | **12.55** | **0.281** | **17.46** | **0.634** |
| I2V+Sketch | LVCD | 29.85 | 7.01 | **0.076** | **26.22** | **0.862** |
| I2V+Sketch | **LayerAnimate** | **26.64** | **5.92** | 0.075 | 25.71 | 0.858 |
| Interpolation | ToonCrafter | 74.63 | 9.97 | 0.244 | 19.92 | 0.668 |
| Interpolation | **LayerAnimate** | **59.64** | **8.38** | **0.216** | **20.07** | **0.696** |
| Interpolation+Sketch | ToonCrafter | 66.26 | 8.40 | 0.128 | 23.28 | 0.794 |
| Interpolation+Sketch | **LayerAnimate** | **15.63** | **3.23** | **0.044** | **29.84** | **0.908** |

### Ablation Study (Table)

Ablation over layer capacity, motion scores, and trajectory representations:

| Setting | FVD↓ | FID↓ | PSNR↑ | SSIM↑ |
|------|------|------|-------|-------|
| N=1 (no layers) | 87.88 | 14.63 | 15.05 | 0.546 |
| N=2 | 81.93 | 14.15 | 15.39 | 0.560 |
| **N=4** | **81.36** | **13.84** | **15.81** | **0.574** |
| I2V (no motion info) | 87.96 | 14.66 | 15.45 | 0.556 |
| I2V + motion assignment | 87.12 | 14.44 | 15.64 | 0.565 |
| **I2V + motion assignment + score** | **81.36** | **13.84** | **15.81** | **0.574** |
| Trajectory (offset only) | 87.83 | 12.74 | 16.94 | 0.612 |
| Trajectory (heatmap only) | 80.57 | 12.65 | 17.57 | 0.635 |
| **Trajectory (hybrid representation)** | **72.04** | **12.55** | 17.46 | **0.634** |

### Key Findings

- Increasing layer capacity $N$ from 1 to 4 yields consistent performance gains (FVD from 87.88 to 81.36), validating the superiority of the layer-level design.
- Motion scores provide finer-grained motion information than binary motion states, improving PSNR from 15.64 to 15.81.
- The hybrid trajectory representation (heatmap + offsets) outperforms either component alone; the heatmap resolves the zero-value ambiguity.
- In a user study with 20 participants, LayerAnimate was voted best across all six tasks.
- The interpolation + sketch task shows the most dramatic improvement, with FVD dropping from 66.26 to 15.63 and PSNR rising from 23.28 to 29.84.

## Highlights & Insights

- **Introducing the traditional animation concept of "layers" into AI generation** is a natural and valuable innovation. Layer separation enables independent control over different elements, substantially improving controllability.
- **The data curation pipeline is elegantly designed**: iterative SAM+SAM2 segmentation addresses newly appearing elements, while motion-based hierarchical merging resolves over-segmentation—forming a complete and reusable engineering solution.
- **Composite control** (applying different control modalities to different layers) is a unique capability that is not achievable within conventional frame-level control frameworks.
- The framework supports interactive layer mask creation via SAM point-click interaction, lowering the barrier to use.

## Limitations & Future Work

- The maximum layer capacity is fixed at 4, which may be insufficient for complex animation scenes.
- The model is trained and evaluated solely on the animation domain; generalization to layered editing of live-action video has not been validated.
- Motion scores are scalar quantities and cannot distinguish directional motion (e.g., leftward vs. rightward), potentially requiring richer motion representations.
- The framework depends on ToonCrafter pretrained weights, constraining model capacity and generation quality to the base model.

## Related Work & Insights

- Compared to AniDoc (character-only) and LVCD (frame-level sketch control), LayerAnimate is more general and supports multimodal control.
- The data curation pipeline can supply training data for other tasks requiring layer annotations, such as animation segmentation and animation editing.
- The concept of layer-level control is generalizable to live-action video editing (e.g., independent foreground/background control).
- Future work may explore layer control frameworks based on DiT architectures to support higher resolutions and longer videos.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Layer-level control represents a paradigm innovation in animation generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across six tasks, with ablation studies and user studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed pipeline descriptions.
- **Value**: ⭐⭐⭐⭐⭐ Strong practical applicability; opens a new direction for AI-assisted animation production.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](object-level_correlation_for_few-shot_segmentation.md)
- [\[NeurIPS 2025\] PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding](../../NeurIPS2025/segmentation/partonomy_large_multimodal_models_with_part-level_visual_understanding.md)
- [\[CVPR 2026\] SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons](../../CVPR2026/segmentation/semlayer_semantic-aware_generative_segmentation_and_layer_construction_for_abstr.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](../../CVPR2026/segmentation/task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)

</div>

<!-- RELATED:END -->
