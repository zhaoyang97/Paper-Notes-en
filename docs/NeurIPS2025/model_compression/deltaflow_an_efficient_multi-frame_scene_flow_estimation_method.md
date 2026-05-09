---
title: >-
  [Paper Note] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method
description: >-
  [NeurIPS 2025][Model Compression][scene flow] This paper proposes DeltaFlow (ΔFlow), which extracts motion cues via inter-frame voxel differences (Δ scheme) to enable multi-frame scene flow estimation with feature sizes that remain constant regardless of the number of input frames. The method achieves state-of-the-art performance on Argoverse 2, Waymo, and nuScenes while running 2× faster than the second-best multi-frame approach.
tags:
  - NeurIPS 2025
  - Model Compression
  - scene flow
  - multi-frame
  - delta scheme
  - autonomous driving
  - computational efficiency
date: 2026-05-08
content_hash: f2e8dbf25aee8761
---

# DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method

**Conference**: NeurIPS 2025
**arXiv**: [2508.17054](https://arxiv.org/abs/2508.17054)
**Code**: [https://github.com/Kin-Zhang/DeltaFlow](https://github.com/Kin-Zhang/DeltaFlow)
**Area**: Model Compression
**Keywords**: scene flow, multi-frame, delta scheme, autonomous driving, computational efficiency

## TL;DR
This paper proposes DeltaFlow (ΔFlow), which extracts motion cues via inter-frame voxel differences (Δ scheme) to enable multi-frame scene flow estimation with feature sizes that remain constant regardless of the number of input frames. The method achieves state-of-the-art performance on Argoverse 2, Waymo, and nuScenes while running 2× faster than the second-best multi-frame approach.

## Background & Motivation

**State of the Field**: Scene flow estimation predicts the 3D motion of each point between consecutive point cloud frames. Mainstream methods voxelize point clouds and fuse temporal information either by concatenating multi-frame features along the channel dimension or by constructing 4D spatiotemporal voxels.

**Limitations of Prior Work**: (1) Concatenation-based methods cause the channel dimension to grow linearly with the number of frames, increasing memory and computational overhead; (2) 4D methods introduce an additional temporal dimension, leading to similarly linear input growth; (3) class imbalance (pedestrians/cyclists are far fewer than vehicles) and intra-instance motion inconsistency further limit performance.

**Root Cause**: While leveraging more historical frames improves accuracy, the computational cost of existing approaches scales linearly or super-linearly with frame count, making extension to long temporal sequences infeasible. Feature concatenation and stacking both "accumulate" temporal information, whereas scene flow is fundamentally concerned with "change."

**Paper Goals**: (1) Efficiently exploit multi-frame information without increasing computational cost; (2) Address class imbalance and intra-instance motion consistency.

**Starting Point**: Scene flow estimation is inherently about identifying "what changes." Computing inter-frame differences of voxel features (Δ scheme) naturally focuses on changing regions, and the resulting difference features maintain a constant size regardless of the number of frames.

**Core Idea**: Replace feature concatenation and 4D stacking with weighted inter-frame differences to encode temporal motion cues, supplemented by a category-balanced loss and an instance consistency loss to improve estimation quality for dynamic objects.

## Method

### Overall Architecture
The input consists of $N+1$ point cloud frames $\{\mathcal{P}_{t-N}, ..., \mathcal{P}_{t-1}, \mathcal{P}_t\}$ with ego-motion compensation applied. Point features are extracted via PointPillars and voxelized into sparse voxel features $\mathscr{D}$. The Δ scheme then computes difference features $\mathscr{D}_{\text{delta}}$, which are fed into a MinkowskiNet 3D backbone followed by a DeFlow decoder to estimate residual scene flow $\Delta\hat{\mathcal{F}}$. The final scene flow equals ego-motion plus residual flow.

### Key Designs

1. **Temporal Δ Scheme**:

    - Function: Extracts motion signals from multi-frame voxel features; the output feature size remains constant regardless of the number of frames.
    - Core formula: $\mathscr{D}_{\text{delta}} = \sum_{n=1}^{N} \lambda^{n-1}(\mathscr{D}_t - \mathscr{D}_{t-n}) / N$
    - Here $\lambda \in (0,1]$ is a temporal decay factor that assigns higher weights to more recent frames.
    - Design Motivation: (1) Inter-frame differences focus on the "changing parts" of the scene (moving objects), naturally filtering static backgrounds in alignment with the core objective of scene flow estimation; (2) The difference result always maintains a $V \times C$ size, so backbone parameters and computation are entirely unaffected by frame count — inference time remains nearly constant from 2 to 15 frames; (3) Weighted summation accumulates the "motion trajectory" of moving objects, analogous to how humans perceive motion direction from motion blur.

2. **Sparse Voxel Representation**:

    - Function: Aggregates point features into non-empty voxels, processing only voxels that contain data.
    - Formula: $\mathscr{D}[v_i] = \frac{\sum_{p \in \mathcal{P}^{v_i}} \mathbf{f}_p}{|\mathcal{P}^{v_i}|}$
    - Design Motivation: Preserves full 3D structure (unlike BEV, which compresses height information) while significantly reducing memory consumption through sparse storage.

3. **Category-Balanced Loss**:

    - Function: Groups objects by category and velocity into bins, assigning different weights to different categories to address the severe underrepresentation of small classes such as pedestrians and cyclists.
    - Formula: $\mathcal{L}_C = \sum_{c \in \mathcal{C}} w_c \sum_{b \in \mathcal{B}} \gamma_b \frac{1}{|\mathcal{P}_{c,b}|} \sum_{p \in \mathcal{P}_{c,b}} \|\Delta\hat{\mathcal{F}}(p) - \Delta\mathcal{F}_{\text{gt}}(p)\|_2$
    - Design Motivation: The prior DeFlow loss only distinguishes static from dynamic objects without differentiating object categories, causing safety-critical small targets (pedestrians) to be dominated by the abundance of vehicle points.

4. **Instance Consistency Loss**:

    - Function: Enforces that all points belonging to the same rigid-body instance share a consistent scene flow.
    - Formula: $\mathcal{L}_I = \frac{1}{|\mathcal{I}'|} \sum_{I \in \mathcal{I}'} \omega_{c_I} \hat{e}_I \exp(\hat{e}_I)$, where $\hat{e}_I$ is the mean intra-instance error and $\exp(\hat{e}_I)$ further penalizes high-error instances.
    - Design Motivation: Scene flow labels are point-wise, so the model may predict different flow directions for different points on the same vehicle. Instance-level consistency constraints ensure the physical plausibility of rigid-body motion.

### Loss & Training
Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{deflow}} + \mathcal{L}_C + \mathcal{L}_I$

## Key Experimental Results

### Main Results (Argoverse 2 Test Set, Public Leaderboard)

| Method | Frames | Runtime/Seq | Mean Bucket-Norm ↓ | Mean EPE (cm) ↓ |
|--------|--------|-------------|-------------------|----------------|
| **ΔFlow (Ours)** | **5** | **8s** | **0.113** | **2.11** |
| Flow4D | 5 | 15s | 0.145 | 2.24 |
| EulerFlow | all | 24h | 0.130 | 4.23 |
| SSF | 2 | 5.2s | 0.181 | 2.73 |
| DeFlow | 2 | 7.2s | 0.276 | 3.43 |

ΔFlow reduces Bucket-Normalized EPE by 22% relative to Flow4D while running 2× faster.

| Dataset | Method | Mean EPE ↓ |
|---------|--------|-----------|
| Waymo | ΔFlow | **1.64** |
| Waymo | Flow4D | 2.03 |
| nuScenes | ΔFlow | Best (−39%) |
| nuScenes | Flow4D | Baseline |

### Ablation Study (Runtime vs. Frame Count)

| Frames | 2 | 5 | 10 | 15 |
|--------|---|---|----|----|
| ΔFlow time | 7.6s | 8s | ~8.5s | ~9s |
| Flow4D time | 12.8s | 15s | — | — |

ΔFlow's runtime remains nearly constant from 2 to 15 frames (a core advantage), whereas Flow4D scales linearly with frame count.

### Key Findings
- The Δ scheme with 2 frames matches the 5-frame performance of Flow4D, demonstrating that the difference representation exploits temporal information more efficiently than concatenation or stacking.
- The category-balanced loss yields substantial gains for pedestrians and VRUs (PED bucket-norm drops from 0.216 to 0.149).
- The model generalizes well across datasets: training on Argoverse 2 and transferring directly to nuScenes remains competitive.
- The instance consistency loss produces smoother flow predictions within the same object, reducing physically implausible estimates.

## Highlights & Insights
- **The Δ scheme is remarkably simple**: it amounts to weighted averaging of inter-frame differences, yet substantially outperforms complex 4D convolution approaches, embodying the philosophy that "choosing the right representation matters more than adding computation."
- **Constant computational cost** is the defining advantage: backbone parameters and FLOPs are entirely independent of frame count, enabling the use of arbitrarily long historical sequences — a property of great value for real-time autonomous driving.
- The motion-blur analogy is intuitive: accumulating differences resembles long-exposure photography producing motion blur, allowing the model to infer motion direction and speed from the "blurred traces."

## Limitations & Future Work
- The Δ scheme assumes voxel alignment — errors in ego-motion compensation may introduce noise into the difference features.
- The category weights $w_c$ and velocity coefficients $\gamma_b$ are set manually; adaptive schemes remain unexplored.
- Only MinkowskiNet is used as the 3D backbone; compatibility with other sparse convolution architectures (e.g., SpConv, TorchSparse) has not been verified.
- The method's ability to handle non-rigid motion (e.g., pedestrian limb articulation) is not analyzed in detail.

## Related Work & Insights
- **vs. Flow4D**: Flow4D constructs 4D spatiotemporal voxels and processes them with 3D spatial + 1D temporal convolutions, with computation scaling linearly with frame count. ΔFlow compresses temporal information into a constant-size representation via differencing, yielding a clear efficiency advantage.
- **vs. DeFlow**: DeFlow uses only 2 frames with channel concatenation. ΔFlow extends this with multi-frame differencing and richer loss functions.
- **vs. EulerFlow**: EulerFlow is an offline method (24h per sequence) that achieves high accuracy but is entirely unsuitable for real-time deployment. ΔFlow strikes a balance between accuracy and efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ The Δ scheme is simple yet highly effective; the category-balanced and instance consistency losses also provide practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three large-scale driving datasets, public leaderboard validation, detailed efficiency analysis, and cross-domain generalization experiments.
- Writing Quality: ⭐⭐⭐⭐ Figures are clear (particularly the comparison of three multi-frame strategies), and the method description is concise and accessible.
- Value: ⭐⭐⭐⭐⭐ Addresses the core scalability problem of multi-frame scene flow estimation; open-sourced code and pretrained weights offer significant value to the autonomous driving community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning](../../ICCV2025/model_compression/multi-object_sketch_animation_by_scene_decomposition_and_motion_planning.md)
- [\[NeurIPS 2025\] KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning](ktae_a_model-free_algorithm_to_key-tokens_advantage_estimation_in_mathematical_r.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [\[CVPR 2026\] DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation](../../CVPR2026/model_compression/dage_dual-stream_architecture_for_efficient_and_fine-grained_geometry_estimation.md)
- [\[NeurIPS 2025\] Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving](loquetier_a_virtualized_multi-lora_framework_for_unified_llm_fine-tuning_and_ser.md)

<!-- RELATED:END -->
