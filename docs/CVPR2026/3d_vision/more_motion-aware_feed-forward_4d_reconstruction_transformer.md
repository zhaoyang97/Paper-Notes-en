---
title: >-
  [Paper Note] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer
description: >-
  [CVPR 2026][3D Vision][4D reconstruction] This paper presents MoRe, a feed-forward motion-aware 4D reconstruction Transformer that decouples dynamic motion from static structure during training via an attention enforcement strategy, and achieves efficient streaming inference through grouped causal attention, attaining state-of-the-art performance in camera pose estimation and depth prediction on dynamic scenes.
tags:
  - CVPR 2026
  - 3D Vision
  - 4D reconstruction
  - dynamic scenes
  - attention enforcement
  - streaming inference
  - motion disentanglement
date: 2026-05-08
content_hash: bf4e5f5cc7272f2c
---

# MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer

**Conference**: CVPR 2026
**arXiv**: [2603.05078](https://arxiv.org/abs/2603.05078)
**Code**: [Project Page](https://hellexf.github.io/MoRe/)
**Area**: 3D Vision
**Keywords**: 4D reconstruction, dynamic scenes, attention enforcement, streaming inference, motion disentanglement

## TL;DR

This paper presents MoRe, a feed-forward motion-aware 4D reconstruction Transformer that decouples dynamic motion from static structure during training via an attention enforcement strategy, and achieves efficient streaming inference through grouped causal attention, attaining state-of-the-art performance in camera pose estimation and depth prediction on dynamic scenes.

## Background & Motivation

Reconstructing temporally evolving 3D structure from monocular video (4D reconstruction) is a core requirement for AR, robotics, and digital twin applications. Existing approaches face the following challenges:

- **Feed-forward models with static-scene assumptions** (DUSt3R, VGGT, Fast3R): These methods directly regress point maps and poses at high speed, but in the presence of dynamic objects, features used for camera estimation are severely contaminated—attention disperses onto moving objects—leading to significant degradation in pose accuracy.
- **Optimization-based pipelines** (MonST3R, CasualSAM): These integrate optical flow, segmentation, and depth estimation modules, offering greater robustness to dynamic scenes, but their multi-stage structure incurs heavy computational overhead, making them unsuitable for real-time or long-sequence processing.
- **Streaming reconstruction methods** (CUT3R, StreamVGGT): These adopt LLM-style causal attention for online processing, but standard causal attention disrupts spatial consistency among intra-frame tokens, and errors accumulate over time.

The core challenge is to design a fast, generalizable framework that simultaneously guarantees pose and depth accuracy under dynamic scenes and streaming inputs.

## Method

### Overall Architecture

MoRe builds upon a strong static reconstruction backbone (the VGGT architecture) and jointly estimates depth $\{D_t\}$, camera parameters $\{g_t\}$, point maps $\{P_t\}$, and motion masks $\{M_t\}$ from a monocular video frame sequence $\{I_t\}_{t=1}^T$. Motion-awareness is injected during training via an attention enforcement strategy, with no reliance on additional motion priors or segmentation inputs at inference time. The model supports both a full-attention mode (optimal quality) and a streaming causal attention mode (online processing).

### Key Designs

1. **Motion-aligned Attention**: The core contribution. During training, ground-truth motion masks explicitly guide the attention distribution of camera tokens. The motion mask is partitioned into patches of size $s \times s$, and a static score for each image token is computed as:
    $$a_i = 1 - \frac{1}{s^2} \sum_{(u,v) \in m_i} m_i(u,v)$$
   where $a_i \in [0,1]$, with higher values indicating more static regions. The attention weights $\alpha_i$ of camera tokens are supervised to align with $a_i$, training the model to focus on static regions and suppress dynamic objects. A key advantage is that this incurs **zero test-time overhead**—GT masks are used only during training, and the model internalizes motion disentanglement capability at inference. The design motivation stems from direct observation of VGGT's attention maps: in dynamic scenes, camera token attention disperses uniformly across both moving and static regions, confusing pose estimation.

2. **Grouped Causal Attention (GCA)**: Standard causal attention is reformulated into a frame-aware variant that allows bidirectional attention among image tokens within the same frame (preserving spatial consistency), while permitting only forward attention across frames (maintaining temporal causality). During streaming inference, the first frame initializes the KV cache, and subsequent frames are processed incrementally:
    $$F_t = \text{Attn}(\mathbf{Q}_t, [\mathbf{K}_{1:t-1}, \mathbf{K}_t], [\mathbf{V}_{1:t-1}, \mathbf{V}_t])$$
   The design motivation is that standard LLM causal attention treats image tokens as a flat sequence, destroying intra-frame spatial correspondence.

3. **BA-like Token Aggregation**: A lightweight global optimization step applied after streaming causal inference. All camera queries $\mathbf{Q}_t^{\text{cam}}$ and KV features are cached; upon completing the sequence, each camera token re-attends to features from all frames:
    $$\mathbf{C}_t^{\text{opt}} = \text{Attn}(\mathbf{Q}_t^{\text{cam}}, [\mathbf{K}_{1:T}], [\mathbf{V}_{1:T}])$$
   This is analogous to Bundle Adjustment for global consistency, but requires only one additional attention computation. During training, camera tokens are duplicated at the end of the sequence to supervise both paths (streaming + global) in parallel, ensuring consistency between the two.

### Loss & Training

- **Depth/point maps**: Confidence-weighted regression loss $\mathcal{L}_{\text{conf}} = \sum_i (\hat{c}_i \|\hat{y}_i - y_i\|_2^2 - \lambda \log(\hat{c}_i))$
- **Motion masks**: Standard BCE loss $\mathcal{L}_{\text{motion}}$
- **Attention alignment**: $\mathcal{L}_{\text{attn}} = \frac{1}{M} \sum_i \max(0, a_i - C) \cdot \alpha_i$, penalizing attention weights on dynamic regions only
- **Camera pose**: Relative transformation supervision $\mathcal{L}_{\text{cam}}$, with gradient truncation applied to early streaming tokens and full gradients retained for the duplicated global tokens
- **Training data**: A mixture of 12 datasets (Dynamic Replica, PointOdyssey, Spring, KITTI, ScanNet, Co3Dv2, etc.), covering indoor/outdoor and dynamic/static scenarios

## Key Experimental Results

### Main Results

**Camera Pose Estimation (Dynamic Scenes)**:

| Method | Type | Sintel ATE↓ | Bonn ATE↓ | TUM-dyn ATE↓ | ScanNet ATE↓ |
|--------|------|-------------|-----------|--------------|--------------|
| VGGT | Full Attn. | 0.1715 | 0.0141 | 0.0109 | 0.0347 |
| MoRe (FA) | Full Attn. | **0.0877** | **0.0138** | 0.0115 | 0.0375 |
| CUT3R | Streaming | 0.2163 | 0.0420 | 0.0438 | 0.0929 |
| Stream3R | Streaming | 0.2144 | 0.0235 | 0.0240 | 0.0521 |
| MoRe | Streaming | **0.1474** | **0.0211** | **0.0260** | **0.0605** |

**Video Depth Estimation**:

| Method | Type | Sintel AbsRel↓ | Bonn AbsRel↓ | KITTI AbsRel↓ |
|--------|------|----------------|--------------|---------------|
| VGGT | Full Attn. | 0.387 | 0.055 | 0.073 |
| MoRe (FA) | Full Attn. | **0.335** | **0.055** | **0.066** |
| Stream3R | Streaming | 0.397 | 0.070 | 0.079 |
| MoRe | Streaming | **0.254** | **0.068** | **0.072** |

### Ablation Study

| Configuration | Sintel ATE↓ | Sintel RPE_trans↓ | TUM ATE↓ | Note |
|---------------|-------------|-------------------|----------|------|
| w/o Attention Enforcement | 0.163 | 0.092 | 0.028 | Remove motion-aligned attention |
| w/o BA-like Optimization | 0.155 | 0.085 | 0.027 | Remove global token aggregation |
| Full MoRe | **0.147** | **0.082** | **0.026** | Complete method |

| Configuration | Sintel AbsRel↓ | Bonn AbsRel↓ | KITTI AbsRel↓ | Note |
|---------------|----------------|--------------|---------------|------|
| w/o GCA | 0.277 | 0.070 | 0.079 | Standard causal attention |
| w/ GCA | **0.254** | **0.068** | **0.072** | Grouped causal attention |

### Key Findings

- The attention enforcement strategy yields the most significant improvement on Sintel (dense dynamic objects): ATE drops from 0.163 to 0.147, validating the effectiveness of motion disentanglement.
- Grouped causal attention consistently improves depth estimation across all benchmarks, demonstrating that intra-frame spatial consistency is critical for geometric reasoning.
- In full-attention mode, MoRe reduces VGGT's Sintel ATE from 0.1715 to 0.0877 (−49%), a breakthrough improvement.
- In streaming mode, MoRe consistently outperforms all comparable methods (CUT3R, StreamVGGT, Wint3R, Stream3R) while supporting incremental processing.
- Zero-shot generalization: none of the dynamic evaluation datasets appear in the training set.

## Highlights & Insights

1. The **attention enforcement** idea is highly elegant: it leaves the inference architecture unchanged and teaches the model "where to look" purely through attention supervision during training, achieving motion disentanglement at zero inference cost.
2. The motivation is grounded in direct observation of VGGT's attention maps (Figure 3), providing a clear and compelling visualization of the problem definition.
3. Grouped causal attention is a minimal yet effective redesign—preserving temporal causality while restoring intra-frame spatial consistency through the smallest possible modification to LLM-style causal attention for image tokens.
4. BA-like token aggregation achieves global consistency with only a single additional attention computation, far more efficient than classical Bundle Adjustment.

## Limitations & Future Work

1. Training requires GT motion masks, limiting the scale and diversity of available training data (annotated dynamic datasets with segmentation labels are needed).
2. In streaming mode, ATE on ScanNet (static scenes) is 0.0605, higher than the full-attention mode's 0.0375, indicating that causal constraints still incur loss on long static sequences.
3. The BA-like optimization requires the full sequence to be processed before execution, and thus does not constitute strictly real-time streaming.
4. The accuracy of the predicted motion masks themselves is not reported, nor are downstream task evaluations (e.g., dynamic object segmentation/removal) conducted.
5. Self-supervised motion mask generation could be explored to eliminate dependence on GT annotations.

## Related Work & Insights

- **VGGT**: The direct baseline that MoRe improves upon; attention enforcement addresses its attention confusion in dynamic scenes.
- **CUT3R**: Introduces persistent Transformer hidden states for online reconstruction, but its attention design does not distinguish between intra- and inter-frame token interactions.
- **MonST3R/CasualSAM**: Optimization-based pipelines are more robust to dynamic scenes but slow; MoRe achieves both speed and robustness through a feed-forward design.
- **Inspiration**: Attention enforcement, as a general strategy for "teaching models what to attend to during training," is transferable to other vision tasks requiring motion/static disentanglement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The attention enforcement strategy is novel and highly effective; grouped causal attention is an elegant design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four benchmarks, 10+ baselines, full-attention and streaming modes, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Attention map visualizations provide clear motivation; formulations are rigorous; experimental organization is systematic.
- **Value**: ⭐⭐⭐⭐ — Provides a practical feed-forward solution for 4D reconstruction; the attention enforcement strategy has broad transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)
- [\[CVPR 2026\] Speed3R: Sparse Feed-forward 3D Reconstruction Models](speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [\[CVPR 2026\] PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)
- [\[CVPR 2026\] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](physgm_large_physical_gaussian_4d_synthesis.md)

</div>

<!-- RELATED:END -->
