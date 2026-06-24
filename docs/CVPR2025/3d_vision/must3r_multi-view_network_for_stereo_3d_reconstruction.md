---
title: >-
  [Paper Note] MUSt3R: Multi-view Network for Stereo 3D Reconstruction
description: >-
  [CVPR 2025][3D Vision][3D Reconstruction] This paper proposes MUSt3R, extending DUSt3R from a pairwise to a multi-view architecture. By symmetrizing the decoder (halving the parameters) and introducing a multi-layer memory mechanism, it achieves high-frame-rate 3D reconstruction of an arbitrary number of images in a unified coordinate system. The same network can handle offline SfM and online Visual Odometry (VO) scenarios simultaneously, achieving an ATE of only 5.5 cm in un…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Reconstruction"
  - "Multi-view"
  - "Visual Odometry"
  - "DUSt3R"
  - "Memory Mechanism"
  - "Uncalibrated"
date: 2026-05-08
content_hash: a561f82d4c160281
---

# MUSt3R: Multi-view Network for Stereo 3D Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2503.01661](https://arxiv.org/abs/2503.01661)  
**Code**: [https://github.com/naver/must3r](https://github.com/naver/must3r)  
**Area**: 3D Vision/3D Reconstruction  
**Keywords**: 3D Reconstruction, Multi-view, Visual Odometry, DUSt3R, Memory Mechanism, Uncalibrated

## TL;DR

This paper proposes MUSt3R, extending DUSt3R from a pairwise to a multi-view architecture. By symmetrizing the decoder (halving the parameters) and introducing a multi-layer memory mechanism, it achieves high-frame-rate 3D reconstruction of an arbitrary number of images in a unified coordinate system. The same network can handle offline SfM and online Visual Odometry (VO) scenarios simultaneously, achieving an ATE of only 5.5 cm in uncalibrated VO on TUM-RGBD.

## Background & Motivation

**Background**: DUSt3R pioneered the paradigm of directly regressing 3D pointmaps from image pairs, which was further improved in matching capability by MASt3R. However, both fundamentally operate in a pairwise manner—processing multiple images requires $O(N^2)$ pairwise matches and global alignment, with computational complexity scaling quadratically with the number of images.

**Limitations of Prior Work**: (1) DUSt3R's pairwise predictions require $O(N^2)$ forward passes on $N$ images, followed by an expensive backend global alignment; (2) global alignment itself is a non-convex optimization problem that is unstable and slow for larger image sets; (3) it cannot process video streams in real-time, as each frame must be paired with all previous frames; (4) concurrent work Spann3R, although introducing a memory mechanism, retains the pairwise architecture and requires two forward passes per image.

**Key Challenge**: Since each pair of pointmaps in DUSt3R resides in its own local coordinate system, coordinate system alignment represents the fundamental bottleneck of pairwise methods. If the model could directly predict the 3D structure of all views in a common coordinate system, the backend alignment step would be completely bypassed.

**Goal**: (1) Eliminate the quadratic complexity and global alignment steps; (2) enable the same network to support both offline reconstruction and online VO; (3) achieve high-frame-rate inference under uncalibrated conditions.

**Key Insight**: Symmetrize DUSt3R's asymmetric dual-decoder into a single shared-weight decoder, identify the reference view using a learnable offset embedding $\mathbf{B}$ to naturally extend to $N$-view cross-attention, and introduce a multi-layer memory cache to enable causal/non-causal inference.

**Core Idea**: Symmetric DUSt3R + dual pointmap prediction heads (global + local) + multi-layer KV-cache-like memory + 3D feedback injection, achieving unified SfM and SLAM in a single network.

## Method

### Overall Architecture

Input images are encoded into features through a shared ViT encoder. The first frame initializes the memory. The decoder tokens of new frames perform cross-attention with the cached tokens of all layers in the memory (without updating old frames) to output dual pointmaps: $\mathbf{X}_{i,1}$ (global coordinate system) and $\mathbf{X}_{i,i}$ (local coordinate system). By performing Procrustes analysis on $\mathbf{X}_{i,i}$ and $\mathbf{X}_{i,1}$, relative poses and intrinsic parameters can be efficiently recovered. The memory updates based on the scene discovery rate (using a KD-Tree).

### Key Designs

1. **Symmetric Siamese Decoder**:
    - **Function**: Eliminates the redundant dual decoders in DUSt3R, halving the parameters while naturally extending to $N$ views.
    - **Core Modification**: Replaces the two decoders with different weights (Dec₁ and Dec₂) with a single shared-weight Dec. The reference frame is identified by adding a learnable offset embedding $\mathbf{B}$ to the decoder input of the non-reference views: $\mathbf{D}_2^0 = \text{Lin}(\mathbf{E}_2) + \mathbf{B}$.
    - **$N$-View Extension**: In each layer $l$, the tokens of the $i$-th view perform cross-attention with the tokens of **all other views**: $\mathbf{D}_i^l = \text{Dec}^l(\mathbf{D}_i^{l-1}, \mathbf{M}_{n,-i}^{l-1})$.
    - RoPE in cross-attention is also removed, as experiments show it to be unnecessary.

2. **Multi-layer Memory Mechanism (Causal KV-Cache)**:
    - **Function**: Reduces the complexity of $N$-view cross-attention to incremental processing.
    - **Core Operation**: The memory stores the decoder tokens $\mathbf{M}_n^l$ at **each layer** for all processed frames. A new frame only requires a single forward pass; its tokens are appended to the memory after performing cross-attention with the memory.
    - **Difference from Spann3R**: Spann3R uses an extra encoder + memory attention to augment features, requiring two forward passes per frame. In contrast, MUSt3R's memory is a standard KV-cache for cross-attention, requiring no extra parameters.
    - **Support for rendering mode**: Re-predicting a frame's pointmap without updating the memory is supported (used to break causality).

3. **3D Feedback Injection (Global 3D Feedback)**:
    - **Function**: Backpropagates global 3D information from the final layer to the earlier layers in the memory.
    - **Core Operation**: For frames already in memory, the final layer $\mathbf{D}_i^{L-1}$ is added to the memory tokens of all shallower layers after passing through a LayerNorm + 2-layer MLP: $\bar{\mathbf{D}}_i^l = \mathbf{D}_i^l + \text{Inj}^{3D}(\mathbf{D}_i^{L-1})$.
    - **Design Motivation**: Layer 0 of the memory contains only encoder features, lacking 3D global information. Injecting feedback enables shallow memory layers to perceive the global 3D structure.
    - This injection is only applied to existing frames, as the final-layer information of new frames does not exist yet.

### Loss & Training

$$\mathcal{L} = \sum_{i=1}^{n+N} \ell_{regr}(i,1) + \ell_{regr}(i,i)$$

The regression loss is computed in log space to better handle distant points: $f: x \rightarrow \frac{x}{\|x\|} \log(1+\|x\|)$. It is calculated separately for the global pointmap $\mathbf{X}_{i,1}$ and the local pointmap $\mathbf{X}_{i,i}$. Training consists of two stages: first, pairwise pre-training of the symmetric DUSt3R, followed by freezing the encoder and training the multi-view version using 10-view sequences.

## Key Experimental Results

### Main Results: TUM-RGBD Visual Odometry (Table 1, ATE RMSE [cm])

| Method | Type | fr1_desk | fr1_room | fr3_long | Mean↓ |
|------|------|----------|----------|----------|------|
| DROID-VO | Calibrated Dense | 5.2 | 33.4 | 7.3 | 11.4 |
| COMO | Calibrated Dense | 4.9 | 27.0 | 10.5 | 10.8 |
| Spann3R | Uncalibrated Dense | 16.1 | 84.8 | 193.9 | 47.9 |
| MUSt3R-C (Causal) | Uncalibrated Dense | 5.1 | 13.4 | 5.9 | 7.1 |
| **MUSt3R** | Uncalibrated Dense | **4.0** | **9.9** | **4.3** | **5.5** |

### Focal Length Estimation (Table 3, TUM-RGBD Vertical FoV Error, Degrees)

| Method | Mean↓ | Median↓ |
|------|-------|--------|
| Spann3R | 12.06 | 12.16 |
| **MUSt3R** | **4.32** | **4.32** |

### Key Findings

- MUSt3R's uncalibrated VO (5.5cm ATE) outperforms most **calibrated** methods (DROID-VO: 11.4cm, COMO: 10.8cm).
- It significantly outperforms Spann3R, which is also based on DUSt3R (47.9cm -> 5.5cm mean ATE), while reducing the focal length estimation error by 3 times.
- The symmetric decoder halves the parameter count while maintaining or even improving performance.
- The rendering mode (recalculating past frames using future frames) further enhances accuracy in offline scenarios.
- It runs at an online frame rate of 11.1 FPS (at 512 resolution), showing strong practical utility.

## Highlights & Insights

1. **"Symmetrization is Simplification"**: Merging DUSt3R's two independent decoders into a shared-weight version and using a single learnable bias to distinguish the reference frame significantly simplifies the model while naturally extending to multiple views.
2. **Memory as KV Cache**: Analogous to KV Cache inference in causal language models, multi-view 3D reconstruction requires only a single forward pass per frame, elegantly unifying SfM and VO.
3. **Dual Pointmap Design**: Predicting both global ($\mathbf{X}_{i,1}$) and local ($\mathbf{X}_{i,i}$) pointmaps allows direct pose recovery via Procrustes analysis, which is faster than PnP and independent of focal length.
4. **3D Feedback Injection**: Enhancing shallow memory layers with final-layer information addresses the lack of global 3D awareness in the early stages of memory, presenting a unique and effective design.

## Limitations & Future Work

1. Linear memory growth—although heuristic selection strategies (discovery rate threshold) are used, extremely long sequences may still cause out-of-memory issues.
2. Training requires freezing the encoder; only the decoder and memory modules are trained during the multi-view phase.
3. Keyframe selection in offline scenarios depends heavily on the quality of ASMK image retrieval.
4. Scale estimation still exhibits large errors on certain sequences (fr1_rpy: 86.3%).

## Related Work & Insights

- **DUSt3R** [Leroy et al.]: Direct predecessor of MUSt3R, pioneering pairwise pointmap regression.
- **MASt3R** [Same group]: Enhances matching capabilities of DUSt3R; compared with MASt3R-SfM in offline scenarios.
- **Spann3R** [Concurrent]: Also introduces memory into the DUSt3R framework, but retains a pairwise architecture with an extra encoder. MUSt3R is simpler and more efficient.
- **DROID-SLAM**: Representative of calibrated dense VO; outperformed by MUSt3R under uncalibrated settings.

## Rating

- ⭐ Novelty: 8/10 — The combined design of symmetrization, multi-layer memory, and dual pointmaps is elegant and highly effective.
- ⭐ Experimental Thoroughness: 9/10 — Comprehensively validated across four downstream tasks: VO, pose estimation, reconstruction, and depth.
- ⭐ Practical Value: 9/10 — High engineering value owing to its uncalibrated nature, high frame rate, and unification of SfM/VO.
- ⭐ Overall: 8.5/10 — An important evolution of the DUSt3R family that successfully pushes dense 3D reconstruction toward real-time uncalibrated applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[CVPR 2025\] Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass](fast3r_towards_3d_reconstruction_of_1000_images_in_one_forward_pass.md)
- [\[CVPR 2025\] MVBoost: Boost 3D Reconstruction with Multi-View Refinement](mvboost_boost_3d_reconstruction_with_multi-view_refinement.md)
- [\[ICLR 2026\] Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator](../../ICLR2026/3d_vision/text-to-3d_by_stitching_a_multi-view_reconstruction_network_to_a_video_generator.md)
- [\[ICCV 2025\] Thermal Polarimetric Multi-view Stereo](../../ICCV2025/3d_vision/thermal_polarimetric_multi-view_stereo.md)

</div>

<!-- RELATED:END -->
