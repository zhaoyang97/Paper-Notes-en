---
title: >-
  [Paper Note] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction
description: >-
  [ICLR2026][3D Vision][Dynamic Gaussian Splatting] This paper proposes USplat4D, an uncertainty-aware dynamic Gaussian splatting framework that estimates per-Gaussian time-varying uncertainty scores and constructs uncerta…
tags:
  - "ICLR2026"
  - "3D Vision"
  - "Dynamic Gaussian Splatting"
  - "uncertainty estimation"
  - "4D Reconstruction"
  - "Monocular"
  - "novel view synthesis"
date: 2026-05-08
content_hash: 9821fb0d0dadaf4c
---

# Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction

**Conference**: ICLR2026  
**arXiv**: [2510.12768](https://arxiv.org/abs/2510.12768)  
**Code**: [tamu-visual-ai/usplat4d](https://tamu-visual-ai.github.io/usplat4d/)  
**Area**: 3D Vision  
**Keywords**: Dynamic Gaussian Splatting, uncertainty estimation, 4D Reconstruction, Monocular, novel view synthesis

## TL;DR

This paper proposes USplat4D, an uncertainty-aware dynamic Gaussian splatting framework that estimates per-Gaussian time-varying uncertainty scores and constructs uncertainty-guided spatiotemporal graphs to propagate reliable motion cues, substantially improving monocular 4D reconstruction quality in occluded regions and under extreme novel viewpoints.

## Background & Motivation

Reconstructing dynamic 3D scenes from monocular video is a fundamental problem for AR, robotics, and human motion analysis, yet it remains highly challenging due to occlusions and extreme viewpoint changes.

- **Common limitations of existing dynamic Gaussian splatting methods**: Regardless of whether canonical fields, deformation bases, or direct 4D modeling is employed, existing methods optimize all Gaussian primitives **uniformly**, relying on 2D supervision signals such as depth, optical flow, and photometric consistency. This uniform treatment overlooks a critical fact: some Gaussians are repeatedly observed and well-constrained, while others are observed only sparsely and remain poorly constrained.
- **Consequences**: Motion drift occurs under occlusion, and synthesis quality degrades severely at extreme novel viewpoints. For example, a rotating backpack always has part of its surface self-occluded at any given moment, yet humans can infer the appearance and motion of occluded regions through memory and temporal continuity.
- **Core insight**: When observations are incomplete, reconstruction should be **anchored to high-confidence cues** and propagated to uncertain regions in a structured manner. High-confidence Gaussians should be prioritized and used to guide the optimization of unreliable ones.

## Method

### Overall Architecture: USplat4D

USplat4D is a model-agnostic, uncertainty-aware framework that can be integrated into any dynamic Gaussian splatting method that estimates per-Gaussian motion. The overall pipeline consists of three steps:

1. **Dynamic uncertainty estimation** (Section 4.1): Estimate time-varying uncertainty scores for each Gaussian at every frame.
2. **Uncertainty-encoded graph construction** (Section 4.2): Partition Gaussians into keynode and non-keynode sets based on uncertainty and build a spatiotemporal graph.
3. **Uncertainty-aware optimization** (Section 4.3): Propagate reliable motion cues to uncertain regions via the graph structure.

### Key Designs 1: Dynamic Uncertainty Estimation

**Per-Gaussian scalar uncertainty**

Given the photometric reconstruction loss $\mathcal{L}_{2,t} = \sum_{h \in \Omega} \|\bar{C}_t^h - C_t^h\|_2^2$, differentiating with respect to the color parameter $c_i$ and assuming a local minimum yields a closed-form variance estimate:

$$\sigma_{i,t}^2 = \left(\sum_{h \in \Omega_{i,t}} (T_{i,t}^h \alpha_i)^2 \right)^{-1}$$

where $T_{i,t}^h$ is the transmittance of Gaussian $i$ at pixel $h$ and $\alpha_i$ is the opacity.

To handle unconverged pixels, a per-pixel convergence indicator $\mathbb{1}_t(h)$ is introduced (set to 1 when the color error is below threshold $\eta_c$). The final scalar uncertainty is:

$$u_{i,t} = \mathbb{1}_{i,t} \cdot \sigma_{i,t}^2 + (1 - \mathbb{1}_{i,t}) \cdot \phi$$

Intuitively, well-observed Gaussians obtain low $u_{i,t}$ (high reliability), while unreliable Gaussians obtain high $u_{i,t}$.

**From scalar to depth-aware uncertainty**

In monocular settings, uncertainty along the depth direction is far greater than along the image-plane directions. The scalar uncertainty implicitly assumes isotropy, which leads to overconfidence along the camera axis and causes geometric distortion. To address this, image-space errors are propagated into 3D and represented by an anisotropic uncertainty matrix:

$$\mathbf{U}_{i,t} = \mathbf{R}_{wc} \cdot \text{diag}(r_x u_{i,t}, r_y u_{i,t}, r_z u_{i,t}) \cdot \mathbf{R}_{wc}^\mathsf{T}$$

where $\mathbf{R}_{wc}$ is the camera-to-world rotation matrix and $r_x, r_y, r_z$ are axis-aligned scaling factors (with the depth-axis factor $r_z$ typically larger), converting 2D uncertainty into 3D uncertainty that accounts for camera pose and depth sensitivity.

### Key Designs 2: Uncertainty-Encoded Graph Construction

**Node partitioning**: Gaussians are partitioned into a small set of keynodes $\mathcal{V}_k$ (low uncertainty, providing motion anchors) and a large set of non-keynodes $\mathcal{V}_n$ (inheriting motion from neighboring keynodes).

**Keynode selection (two-stage strategy)**:

1. **3D voxel grid sampling**: At each frame, the scene is divided into a 3D voxel grid; voxels containing only high-uncertainty Gaussians are discarded, and one low-uncertainty Gaussian is randomly selected per remaining voxel to ensure uniform spatial coverage.
2. **Significant-period threshold filtering**: The "significant period" of each candidate Gaussian (the number of frames with uncertainty below a threshold) is computed; only candidates with a significant period of ≥5 frames are retained, ensuring sufficient temporal support.

The keynode-to-non-keynode ratio is maintained at approximately 1:49 (top 2% most confident Gaussians are selected); ablation experiments show stable performance across the 0.5%–4% range.

**Edge construction**:

- **Between keynodes**: Uncertainty-Aware kNN (UA-kNN) is applied at each node's most reliable frame $\hat{t} = \arg\min_t \{u_{i,t}\}$, using Mahalanobis distance to select neighbors, ensuring connections between spatially close and reliable nodes.
- **Non-keynodes**: Each non-keynode is associated with the nearest keynode across the entire sequence and inherits that keynode's neighbor structure.

### Key Designs 3: Uncertainty-Aware Optimization

**Keynode loss**: Encourages keynodes to remain near their pre-trained positions, weighted by the inverse uncertainty matrix $\mathbf{U}_{w,t,i}^{-1}$ to ensure motion corrections primarily occur along reliable directions:

$$\mathcal{L}^{\text{key}} = \sum_t \sum_{i \in \mathcal{V}_k} \|\mathbf{p}_{i,t} - \mathbf{p}_{i,t}^o\|_{\mathbf{U}_{w,t,i}^{-1}} + \mathcal{L}^{\text{motion,key}}$$

**Non-keynode loss**: Motion is interpolated from neighboring keynodes via Dual Quaternion Blending (DQB), while non-keynodes are constrained to remain close to both pre-trained states and interpolated trajectories:

$$\mathcal{L}^{\text{non-key}} = \sum_t \sum_{i \in \mathcal{V}_n} \|\mathbf{p}_{i,t} - \mathbf{p}_{i,t}^o\|_{\mathbf{U}_{w,i}^{-1}} + \sum_t \sum_{i \in \mathcal{V}_n} \|\mathbf{p}_{i,t} - \mathbf{p}_{i,t}^{\text{DQB}}\|_{\mathbf{U}_{w,i}^{-1}} + \mathcal{L}^{\text{motion,non-key}}$$

**Total loss**: $\mathcal{L}^{\text{total}} = \mathcal{L}^{\text{rgb}} + \mathcal{L}^{\text{key}} + \mathcal{L}^{\text{non-key}}$

### Loss & Training

- USplat4D employs **two-stage training**: a baseline model (e.g., SoM or MoSca) first pre-trains the dynamic Gaussian field, followed by refinement using USplat4D's uncertainty-aware optimization.
- The framework is model-agnostic and can be plugged into any baseline method that estimates per-Gaussian motion.
- Motion regularization includes isometry, rigidity, relative rotation, velocity, and acceleration constraints.

## Key Experimental Results

### Main Results: Quantitative Results on DyCheck

| Setting | Method | mPSNR↑ | mSSIM↑ | mLPIPS↓ |
|------|------|--------|--------|---------|
| 5 scenes, 1× | SC-GS | 14.13 | 0.477 | 0.49 |
| 5 scenes, 1× | Deformable 3DGS | 11.92 | 0.490 | 0.66 |
| 5 scenes, 1× | 4DGS | 13.42 | 0.490 | 0.56 |
| 5 scenes, 1× | MoDec-GS | 15.01 | 0.493 | 0.44 |
| 5 scenes, 1× | MoBlender | 16.79 | 0.650 | 0.37 |
| 5 scenes, 1× | SoM | 16.72 | 0.630 | 0.45 |
| 5 scenes, 1× | **USplat4D** | **16.85** | **0.650** | **0.38** |
| 7 scenes, 2× | Dynamic Gaussians | 7.29 | – | 0.69 |
| 7 scenes, 2× | 4DGS | 13.64 | – | 0.43 |
| 7 scenes, 2× | Gaussian Marbles | 16.72 | – | 0.41 |
| 7 scenes, 2× | MoSca | 19.32 | 0.706 | 0.26 |
| 7 scenes, 2× | **USplat4D** | **19.63** | **0.716** | **0.25** |

### Main Results: Extreme Novel View Synthesis on Objaverse

| Method | Viewpoint Range | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|----------|-------|-------|--------|
| SoM | (0°, 60°] | 16.09 | 0.860 | 0.31 |
| USplat4D (SoM) | (0°, 60°] | **16.63** | **0.866** | **0.27** |
| SoM | (60°, 120°] | 15.58 | 0.854 | 0.32 |
| USplat4D (SoM) | (60°, 120°] | **16.57** | **0.868** | **0.27** |
| SoM | (120°, 180°] | 16.45 | 0.858 | 0.31 |
| USplat4D (SoM) | (120°, 180°] | **17.03** | **0.872** | **0.26** |
| MoSca | (0°, 60°] | 16.18 | 0.881 | 0.24 |
| USplat4D (MoSca) | (0°, 60°] | **16.22** | **0.885** | **0.22** |
| MoSca | (120°, 180°] | 15.89 | 0.876 | 0.25 |
| USplat4D (MoSca) | (120°, 180°] | **16.31** | **0.886** | **0.21** |

Gains are most pronounced at extreme viewpoints (120°–180°), with PSNR improving by +0.58 dB and LPIPS by 0.05 over the SoM baseline.

### Ablation Study

| Ablation Setting | PSNR↑ | SSIM↑ | LPIPS↓ |
|----------|-------|-------|--------|
| USplat4D (full model) | **19.63** | **0.716** | **0.25** |
| (a) w/o uncertainty-guided keynode selection | 18.86 | 0.688 | 0.28 |
| (b) w/o UA-kNN | 19.50 | 0.711 | 0.26 |
| (c) w/o uncertainty-weighted loss | 19.08 | 0.681 | 0.25 |
| (d) w/o 3D voxelization | 19.50 | 0.712 | 0.25 |

- **(a) Removing uncertainty-guided keynode selection** has the largest impact: PSNR drops by 0.77 dB, confirming that uncertainty is critical for anchor selection.
- **(c) Removing uncertainty weighting in the loss**: SSIM drops by 0.035, as unreliable Gaussians are updated with the same strength as reliable ones, causing drift.

## Highlights & Insights

1. **Concise and powerful core idea**: Elevating uncertainty from an auxiliary signal to the framework's centerpiece, the "high-confidence anchoring + structured propagation" paradigm addresses occlusion and extreme viewpoint challenges with strong intuitive interpretability.
2. **Model-agnostic plug-and-play design**: USplat4D integrates seamlessly with different baselines such as SoM and MoSca, consistently delivering gains and demonstrating strong generality.
3. **Depth-aware anisotropic uncertainty**: Extending scalar uncertainty to a 3D anisotropic matrix that accounts for camera pose effectively mitigates overconfidence along the depth direction inherent in monocular reconstruction.
4. **Natural scene segmentation from the graph**: The weight matrix of the keynode graph, when reordered, approximates a block-diagonal structure, naturally supporting multi-object motion segmentation without additional supervision.
5. **Triple role of uncertainty**: Uncertainty operates in a unified manner across three levels — weighting keynode position regularization, guiding non-keynode motion interpolation, and balancing the total loss.

## Limitations & Future Work

1. **Dependence on pre-trained baseline quality**: USplat4D refines upon a pre-trained model; if the baseline initialization is poor (e.g., severe initial motion errors), refinement effectiveness is limited.
2. **Computational overhead and errors from visual foundation models**: The framework still incurs the computational cost and inherent errors of the underlying visual foundation models (depth estimation, optical flow, etc.).
3. **Limited gains at near-input viewpoints**: On validation views close to the input viewpoints, USplat4D shows only marginal improvement over strong baselines such as MoBlender and SoM (+0.13 dB PSNR); advantages are primarily observed at extreme viewpoints.
4. **Hyperparameter sensitivity**: The keynode ratio (2%), significant-period threshold (5 frames), and color convergence threshold $\eta_c$ require scene-specific tuning.
5. **Insufficient analysis of textureless and fast-motion scenes**: In regions with sparse texture and scenes with extremely fast motion, the uncertainty estimation itself may fail.

## Related Work & Insights

| Direction | Representative Methods | Difference from USplat4D |
|------|----------|-------------------|
| Dynamic Gaussian splatting (motion bases) | SoM, MoSca, Marbles, 4D-Rotor | Use low-rank motion bases to regularize deformation but treat all Gaussians uniformly, causing motion drift under occlusion |
| Dynamic Gaussian splatting (canonical field) | Deformable 3DGS, SC-GS | Model motion via canonical space; similarly lack uncertainty awareness |
| Uncertainty in scene reconstruction | SE-GS, Kim et al. (2024) | SE-GS applies self-ensemble uncertainty to static scenes; Kim et al. use uncertainty as an auxiliary signal to smooth motion or reweight gradients, but do not integrate it into structured graph propagation |
| Graph-based motion modeling | MoSca (lifting graph), SC-GS (local kNN) | Construct graphs with fixed distance metrics, without considering node reliability |

## Rating

| Dimension | Score (1–5) | Notes |
|------|-----------|------|
| Novelty | 4 | Elevates uncertainty from an auxiliary signal to the core of a unified graph construction and optimization framework; conceptually innovative |
| Technical Depth | 4 | Derivation from scalar to anisotropic uncertainty is rigorous; graph construction and optimization are well-designed |
| Experimental Thoroughness | 4 | Covers three datasets (DyCheck, DAVIS, Objaverse) with comprehensive ablations; quantitative analysis is primarily on validation views |
| Writing Quality | 4 | Motivation is clear, formula derivations are explicit, and figures are informative |
| Value | 4 | Model-agnostic plug-and-play design is highly practical and can directly enhance existing methods |
| Overall | 4.0 | High-quality methodological contribution; introduces structured uncertainty modeling to monocular 4D reconstruction with significant gains at extreme viewpoints |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[ICLR 2026\] Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos](mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)
- [\[CVPR 2026\] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](../../CVPR2026/3d_vision/aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](../../AAAI2026/3d_vision/sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)

</div>

<!-- RELATED:END -->
