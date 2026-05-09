---
title: >-
  [Paper Note] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation
description: >-
  [ICCV 2025][Video Generation][motion transfer] This paper proposes MotionShot, a training-free motion transfer framework that achieves high-fidelity motion transfer between arbitrary reference–target object pairs with significant appearance and structural differences, via a two-level motion alignment strategy combining high-level semantic alignment and low-level morphological alignment.
tags:
  - ICCV 2025
  - Video Generation
  - motion transfer
  - text-to-video
  - training-free
  - TPS warping
  - temporal attention guidance
date: 2026-05-08
content_hash: 85cfab2ec4caa1fd
---

# MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation

**Conference**: ICCV 2025
**arXiv**: [2507.16310](https://arxiv.org/abs/2507.16310)
**Code**: [Project Page](https://motionshot.github.io/)
**Area**: Video Generation
**Keywords**: motion transfer, text-to-video, training-free, TPS warping, temporal attention guidance

## TL;DR

This paper proposes MotionShot, a training-free motion transfer framework that achieves high-fidelity motion transfer between arbitrary reference–target object pairs with significant appearance and structural differences, via a two-level motion alignment strategy combining high-level semantic alignment and low-level morphological alignment.

## Background & Motivation

**Challenges in Motion Transfer**:
- Most existing methods can only handle reference–target pairs with similar appearance (e.g., human→human, animal→same species).
- When reference and target objects exhibit significant appearance/structural differences (e.g., anime character→Winnie the Pooh), motion transfer quality degrades sharply.

**Limitations of Prior Work**:

**Keypoint sequence methods**: Require predefined keypoints for each object category and cannot generalize to arbitrary objects.

**Spatiotemporal feature methods**: Motion and appearance are entangled in latent representations, causing reference appearance leakage.

**Depth/edge/optical flow conditioning**: Do not account for region-level semantic correspondence or pixel-level structural correspondence, and fail on object pairs with large differences.

**Attention-based methods**: Motion and structure are tightly coupled; when the target and reference differ significantly, the transferred motion becomes incompatible.

**Core Problem**: How to accurately transfer the motion patterns of a reference object while preserving the appearance of the target object?

## Method

### Overall Architecture

MotionShot builds upon the AnimateDiff video generation framework and consists of three main stages:
1. **Semantic Motion Alignment**: Establishes high-level semantic correspondences between reference and target objects.
2. **Morphological Motion Alignment**: Achieves low-level structural mapping via TPS transformation.
3. **Attention-guided Generation**: Guides video generation using the warped reference frames.

### Key Designs

1. **Semantic Motion Alignment**:

    - **Pseudo-target generation**: A ControlNet-segmentation model takes a degraded segmentation map of the reference object (coarse initial pose cue) and a text prompt as input to generate a pseudo-target object whose initial pose is close to that of the reference. The ControlNet conditioning weight is set to 0.6 so that the text prompt remains dominant.
    - **Structure-aware keypoint sampling**: $m=30$ keypoints are sampled on the reference object, comprising uniform contour sampling (interval $d=200$) and Poisson disk interior sampling, ensuring that keypoints are spread across all regions of the object.
    - **Semantic feature matching**: Features from Stable Diffusion (low-level spatial information) and DINOv2 (high-level semantic information) are fused, and similarity is computed via $L_2$ distance: $\text{Sim}(i,j) = -\|f_\text{tar}^s(i) - f_\text{ref}^s(j)\|_2$
    - Design motivation: SD features provide fine spatial details but are error-prone in ambiguous regions; DINO captures high-level semantics but may miss fine details; fusion is complementary.

2. **Morphological Motion Alignment**:

    - **Target keypoint sequence construction**: CoTracker3 tracks reference keypoints, and motion is transferred to the target space via global motion (elliptical rotation and translation) and local motion (polar-coordinate relative offsets).
    - Global motion: $K_\text{tar}^t = \mathcal{S}(\mathcal{R}(K_\text{tar}^0, \Delta\Theta^t), \Delta O^t)$
    - Local motion: Keypoint offsets are decomposed into radial scaling and polar angle offsets.
    - **TPS shape warping**: A Thin Plate Spline (TPS) transformation warps reference frames into the target shape:
    $$\mathcal{T}^t(p) = A^t\begin{bmatrix}p\\1\end{bmatrix} + \sum_{i=1}^m w^{t,i}\mathcal{U}(\|\mathbf{K}_\text{tar}^{t,i}-p\|^2)$$
    - Design motivation: Point-level guidance lacks continuity and disrupts temporal attention; TPS warping provides a continuous shape mapping.

3. **Attention-guided Video Generation**:

    - A single-step noise–denoise pass is applied to the warped reference frames to extract temporal attention maps $A_\text{ref}^\tau$.
    - Top-$k$ ($k=1$) sparse control masks are selected to reduce noise.
    - An energy function is defined: $g = \|M^\tau \cdot (A_\text{ref}^\tau - A_\text{gen}^t)\|_2^2$
    - Score-based guidance steers diffusion sampling: $\hat{\epsilon}_\theta = \epsilon_\theta(z_t, \text{text}, t) - \lambda\nabla_{z_t}g$
    - A DDIM sampler is used; guidance is applied during the first 180 of 300 steps.
    - Design motivation: Since reference frames have already been warped into the target shape, motion information in the temporal attention is naturally aligned with the target structure.

### Loss & Training

MotionShot is a **fully training-free** framework:
- No additional training data or fine-tuning is required.
- It builds on pretrained AnimateDiff, ControlNet, Stable Diffusion, DINOv2, and CoTracker3.
- All alignment is achieved through feature matching and geometric transformation.

## Key Experimental Results

### Main Results (Tables)

**Quantitative Comparison (CLIP Scores + User Study)**:

| Method | Text Align↑ | Temporal Consist↑ | Motion Preserv↑ | Appear Diversity↑ | User-Text↑ | User-Temporal↑ |
|------|------------|-------------------|-----------------|-------------------|-----------|---------------|
| VideoComposer | 26.54 | 95.95 | 3.00 | 2.72 | 2.79 | 2.82 |
| Gen-1 | 22.79 | 97.67 | 2.87 | 2.71 | 2.75 | 2.87 |
| VMC | 26.77 | 97.72 | 2.80 | 2.78 | 2.78 | 2.87 |
| Tune-A-Video | 26.60 | 95.99 | 2.86 | 2.78 | 2.88 | 2.86 |
| Control-A-Video | 24.87 | 95.54 | 2.94 | 2.66 | 2.40 | 2.92 |
| MotionClone | 26.41 | 97.48 | 2.90 | 2.50 | 2.80 | 2.82 |
| **MotionShot** | **26.95** | **97.81** | **4.95** | **4.95** | **4.94** | **4.90** |

- In the user study, MotionShot achieves near-perfect scores (on a 5-point scale) across all four dimensions, far surpassing all baselines.

### Ablation Study (Tables)

**Ablation on Number of Keypoints**:

| Keypoints $m$ | Contour | Interior | Effect |
|-------------|--------|--------|------|
| 10 | 8 | 2 | TPS warping fails; cannot match target shape |
| **30** | **24** | **6** | **Reasonable warping; best overall performance** |
| 60 | 48 | 12 | Overfitting; unnatural warping results |

**Comparison of Semantic Feature Matching Methods**:

| Method | Effect |
|------|---------|
| X-Pose keypoint detector | Only predicts 17 points with uneven distribution; appearance mismatch |
| SD features only | Fine spatial detail but error-prone in ambiguous regions (e.g., tail) |
| DINO features only | Good high-level semantics but misses fine details (e.g., legs) |
| **SD + DINO fusion** | **Balances fine-grained and high-level accuracy; best performance** |

**Comparison of Shape Retargeting Methods**:

| Method | Issue |
|------|------|
| No warping of original sequence | Motion–shape mismatch; generated object deforms |
| Simple scaling | Consistent size but topological distortion (e.g., leg misalignment) |
| **Keypoint-based TPS warping** | **Best motion accuracy and structural consistency** |

### Key Findings

- Motion preservation and appearance diversity scores in the user study approach the maximum (4.95/5), far exceeding the second-best method (~3.0/5).
- The fully training-free approach substantially outperforms methods that require training in motion transfer quality.
- SD+DINO feature fusion is critical for semantic matching; using either feature alone is insufficient.
- TPS warping is a key step; point-level guidance or simple scaling both lead to shape artifacts.
- 30 keypoints is the optimal count: too few yields insufficient warping, too many leads to overfitting.

## Highlights & Insights

1. **Two-level alignment framework**: The first work to explicitly model both high-level semantic alignment and low-level morphological alignment, addressing the core challenge of motion transfer between arbitrary object pairs.
2. **Fully training-free**: No additional training or fine-tuning is needed; powerful capabilities are achieved by composing existing pretrained models.
3. **Pseudo-target generation strategy**: Degraded segmentation maps guide the generation of a pseudo-target with a consistent initial pose, elegantly resolving the initial pose mismatch problem.
4. **Structure-aware keypoint sampling**: The combination of uniform contour sampling and Poisson disk interior sampling ensures even keypoint coverage across all regions of the object.
5. **Motion decomposition**: Decomposing motion into global rotation+translation and local polar-coordinate offsets enables fine-grained transfer of complex motions.

## Limitations & Future Work

- The method fails when the reference and target objects share no semantic similarity (e.g., airplane→flower).
- Semantic correspondence relies on the feature quality of SD and DINOv2, and may be unreliable for objects far from the pretraining distribution.
- Being built on the AnimateDiff framework, video quality and length are constrained by the underlying model.
- The 300-step sampling with guidance applied for the first 180 steps leaves room for improving inference efficiency.
- Only single-object motion transfer is supported; motion transfer involving multi-object interactions remains unexplored.

## Related Work & Insights

- The two-level alignment paradigm (semantic + morphological) is generalizable to other tasks requiring cross-domain correspondence (e.g., style transfer, animation).
- TPS warping, as a continuous spatial mapping tool, is better suited for attention-guided generation than discrete point-level guidance.
- Training-free methods can address complex visual generation problems through the judicious combination of pretrained models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The two-level motion alignment framework is pioneering and resolves a longstanding challenge of arbitrary object pair motion transfer.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative evaluation, user study, and detailed ablations are provided, though additional automated motion fidelity metrics would strengthen the evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is clearly described with rich illustrations and well-motivated design choices.
- **Value**: ⭐⭐⭐⭐⭐ Near-perfect user study scores demonstrate strong practical value, and the training-free design lowers the barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)
- [\[ICCV 2025\] DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization](dualreal_adaptive_joint_training_for_lossless_identity-motion_fusion_in_video_cu.md)
- [\[NeurIPS 2025\] Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](../../NeurIPS2025/video_generation/foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)

</div>

<!-- RELATED:END -->
