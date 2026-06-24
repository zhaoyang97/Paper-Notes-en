---
title: >-
  [Paper Note] Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations
description: >-
  [CVPR 2025][3D Vision][Neural Radiance Fields] This paper proposes Flow-NeRF, which is the first to integrate scene geometry, camera poses, and dense optical flow as a unified joint optimization target within a pose-free NeRF framework. Through shared point sampling, a pose-conditioned bijective mapping, and a feature message passing mechanism, it significantly outperforms prior methods in novel view synthesis and depth estimation, while defining and achieving novel view opti…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Neural Radiance Fields"
  - "Pose-free NeRF"
  - "Optical Flow"
  - "Joint Optimization"
  - "Novel View Flow"
date: 2026-05-08
content_hash: b870c6e8e0afb1fb
---

# Flow-NeRF: Joint Learning of Geometry, Poses, and Dense Flow within Unified Neural Representations

**Conference**: CVPR 2025  
**arXiv**: [2503.10464](https://arxiv.org/abs/2503.10464)  
**Code**: [https://zhengxunzhi.github.io/flownerf/](https://zhengxunzhi.github.io/flownerf/)  
**Area**: 3D Vision  
**Keywords**: Neural Radiance Fields, Pose-free NeRF, Optical Flow, Joint Optimization, Novel View Flow

## TL;DR
This paper proposes Flow-NeRF, which is the first to integrate scene geometry, camera poses, and dense optical flow as a unified joint optimization target within a pose-free NeRF framework. Through shared point sampling, a pose-conditioned bijective mapping, and a feature message passing mechanism, it significantly outperforms prior methods in novel view synthesis and depth estimation, while defining and achieving novel view optical flow estimation for the first time.

## Background & Motivation

1. **Background**: NeRF typically requires camera poses provided by SfM pipelines like COLMAP. Pose-free NeRFs (such as BARF, Nope-NeRF) can jointly optimize poses and scene geometry, but suffer from poor reconstruction quality due to the lack of cross-frame consistency constraints.
2. **Limitations of Prior Work**: Some methods leverage optical flow supervision to constrain pose learning (e.g., LocalRF), but they only treat optical flow as a regularization term for pose optimization without exploring its potential to benefit novel view synthesis and scene geometry. Furthermore, existing methods rely on pre-trained optical flow models (e.g., RAFT), which limits their performance upper bound.
3. **Key Challenge**: Optical flow provides rich cross-view dense correspondence information, but current frameworks fail to incorporate it as a first-class optimization objective within a unified representation, leading to underutilized optical flow information.
4. **Goal**: (1) How to jointly learn optical flow in a pose-free NeRF? (2) How to leverage optical flow learning to conversely improve geometric reconstruction? (3) How to achieve dense correspondence estimation between novel views (novel view flow)?
5. **Key Insight**: The authors' key observation is that the geometry field and the optical flow field should share the underlying scene representation, as they mathematically model the exact same physical 3D scene. By conditioning optical flow prediction on poses as frame identifiers, it becomes possible to infer novel view optical flow beyond the training views.
6. **Core Idea**: Simultaneously learn geometry, poses, and dense optical flow using a unified neural scene representation, and enhance geometric reconstruction with optical flow learning via feature message passing from canonical space to world space.

## Method

### Overall Architecture
The input is a sequence of pose-free images. The framework consists of a geometry branch and an optical flow branch. In each iteration, two frames $I_i$ and $I_j$ are selected, and shared point sampling is utilized to ensure both branches process the exact same physical scene points. The geometry branch learns appearance and depth using volume rendering of standard NeRF, while the optical flow branch learns frame-to-frame 2D-2D correspondences via a bijective mapping (Real-NVP). The two branches are coupled through a feature message passing mechanism, where canonical space features enhance the world space representation.

### Key Designs

1. **Shared Points Sampling**:

    - **Function**: Ensures that both the geometry branch and the optical flow branch operate on the same set of physical scene points.
    - **Mechanism**: In each iteration, $N=1024$ 2D pixel points are randomly sampled from frame $i$ and shared between the two branches. The geometry branch back-projects them into the world space $\mathbf{p}_i = z_i\mathbf{r}$ ($\mathbf{r} = TK^{-1}[u,v]$) using the known intrinsic $K$ and learned pose $T$. Conversely, the optical flow branch back-projects them to the camera space $K^{-1}[u,v]d_i$. The correspondences are maintained via a fixed ratio $z_i = \alpha d_i$ ($\alpha=0.2$).
    - **Design Motivation**: Utilizing camera space projection (rather than simple orthogonal projection) preserves the perspective relationships of the scene, which experiments prove is crucial for depth estimation.

2. **Pose-Conditioned Bijective Mapping**:

    - **Function**: Learns pixel-level 2D-2D correspondences between frames and supports novel view flow inference.
    - **Mechanism**: A bijective mapping $\epsilon$ parameterized by a normalizing flow (Real-NVP) is used to map 3D points $\mathbf{O}_i$ in the camera space of frame $i$ to points $\mathbf{r}$ in the canonical 3D space, and then map them back to points $\mathbf{O}_j$ in the camera space of frame $j$. The key innovation is using the 6-DoF pose vector $[r_1,r_2,r_3,t_1,t_2,t_3]$ as the frame identifier instead of time indices. Poses update dynamically during optimization and carry physical geometric information, allowing generalization to unseen novel views beyond the training views.
    - **Design Motivation**: Time-conditioning can only infer optical flow within training views (since time does not carry the physical information of camera motion), whereas pose-conditioning allows the model to query corresponding optical flows with arbitrary poses, achieving unprecedented novel view optical flow estimation.

3. **Feature Message Passing**:

    - **Function**: Passes canonical space features from the optical flow branch to the geometry branch to enhance scene geometry reconstruction.
    - **Mechanism**: A 3-layer, 256-dimensional Gabornet is used to extract 128-dimensional features from 3D points in the canonical space. These features are directly concatenated into the intermediate layer after the skip connection in the geometry branch MLP $F_{\theta1}$. The two branches are optimized using different loss functions (photometric consistency vs. optical flow supervision), thereby learning complementary feature representations. The 2D correspondence information provided by the optical flow branch constrains and guides more accurate geometry estimation.
    - **Design Motivation**: This is based on the insight that while canonical space features and world space features come from different branches and are optimized with different losses, they represent the same physical scene and are thus complementary. Injecting motion consistency knowledge from the canonical space into the geometry branch significantly improves novel view synthesis and depth prediction.

### Loss & Training
- **RGB Rendering Loss**: $L_{rgb} = \frac{1}{N}\sum ||\hat{\mathbf{C}}(\mathbf{p}) - \mathbf{C}(\mathbf{p})||_1$, the standard photometric consistency loss.
- **Optical Flow Loss**: $L_{flow} = \frac{1}{N}\sum ||\hat{\mathbf{p}}_j - \mathbf{p}_j||_1$, representing the L1 distance between the predicted 2D points and the pseudo-optical flow provided by RAFT.
- Additionally, we include a 3D point cloud loss and a 2D photometric warping loss, with the two branches optimized jointly in an end-to-end manner.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Tanks & Temples (Mean) | BARF | 23.42 | 0.61 | 0.54 |
| Tanks & Temples (Mean) | Nope-NeRF | 26.34 | 0.74 | 0.39 |
| Tanks & Temples (Mean) | **Flow-NeRF** | **28.73** | **0.82** | **0.29** |
| ScanNet (Mean) | BARF | 31.41 | 0.82 | 0.39 |
| ScanNet (Mean) | Nope-NeRF | 31.86 | 0.83 | 0.38 |
| ScanNet (Mean) | **Flow-NeRF** | **32.55** | **0.85** | **0.34** |

### Ablation Study (Long-range Flow EPE on Sintel Dataset)

| Method | Frame Interval = 1 | Frame Interval = 8 | Frame Interval = 16 |
|------|---------|---------|----------|
| RAFT (Pre-trained + Fine-tuned) | **0.455** | 1.567 | 2.089 |
| Flow-NeRF | 0.689 | **1.318** | **1.683** |

### Key Findings
- **Feature Message Passing contributes the most**: Passing features from the canonical space to the world space results in an average PSNR improvement of over 2 dB on Tanks & Temples and 0.8 dB on ScanNet, with comprehensive improvements in depth metrics.
- **Long-range optical flow is significantly better than RAFT**: Although the model is only trained with forward optical flow between consecutive frames, it achieves an EPE of 1.683 at a frame interval of 16 (compared to RAFT's 2.089), and it can infer both forward and backward long-range optical flow.
- **Pose-conditioning is key to novel view flow**: Time-conditioning can only infer within training views, whereas pose-conditioning enables true novel view optical flow estimation capabilities.

## Highlights & Insights
- **Upgrading optical flow from a regularization term to a first-class optimization target**: Prior works only used optical flow to constrain poses. This study demonstrates that treating it as an explicit target can, in turn, enhance geometry reconstruction—reflecting the deep value of "multi-task joint learning" in 3D scene understanding.
- **Definition and implementation of novel view flow**: This work defines the new task of "novel view flow" for the first time—inferring dense correspondences between any unseen views, opening up a new direction for holistic scene modeling.
- **Implicit distillation from canonical space to world space**: Knowledge transfer from optical flow learning to geometric learning is achieved via feature message passing. This mechanism is simple yet highly effective and can be transferred to any multi-branch neural field architecture.

## Limitations & Future Work
- **Limited to static scenes**: It is assumed that all optical flow is caused by camera motion, meaning it cannot handle dynamic objects.
- **Reliance on RAFT pseudo-optical flow**: The quality of supervision during training is constrained by the performance upper bound of RAFT, especially in texture-sparse areas.
- **Computational overhead**: The dual-branch architecture increases training time and memory requirements.
- **Potential improvements**: Extending the method to dynamic scenes (by introducing object motion modeling); replacing RAFT pseudo-labels with self-supervised approaches; applying the message passing mechanism to more efficient scene representations like 3D Gaussian Splatting.

## Related Work & Insights
- **vs BARF**: BARF resolves joint pose-geometry optimization using coarse-to-fine positional encoding but lacks cross-frame correspondence constraints, performing poorly in complex scenes. Flow-NeRF provides strong cross-frame constraints via optical flow, gaining over 5 dB in PSNR.
- **vs Nope-NeRF**: Nope-NeRF regularizes poses using monocular depth priors but depends heavily on the quality of pre-trained depth models. Flow-NeRF obtains stronger geometric constraints by jointly learning optical flow.
- **vs Omnimotion**: Omnimotion uses time-conditioning to learn correspondence fields but can only query within training views. Flow-NeRF's pose-conditioned design enables novel view generalization.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Integrates optical flow as a joint optimization target of NeRF for the first time and defines the novel view flow task.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across multiple datasets and tasks (NVS, depth, pose, optical flow) with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-defined motivation, and complete technical details.
- **Value**: ⭐⭐⭐⭐ The unified framework concept is highly inspiring, and novel view flow could foster new downstream applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation](floxels_fast_unsupervised_voxel_based_scene_flow_estimation.md)
- [\[CVPR 2025\] Zero-Shot Monocular Scene Flow Estimation in the Wild](zero-shot_monocular_scene_flow_estimation_in_the_wild.md)
- [\[CVPR 2026\] UniPixie: Unified and Probabilistic 3D Physics Learning via Flow Matching](../../CVPR2026/3d_vision/unipixie_unified_and_probabilistic_3d_physics_learning_via_flow_matching.md)
- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [\[CVPR 2025\] End-to-End Implicit Neural Representations for Classification](end-to-end_implicit_neural_representations_for_classification.md)

</div>

<!-- RELATED:END -->
