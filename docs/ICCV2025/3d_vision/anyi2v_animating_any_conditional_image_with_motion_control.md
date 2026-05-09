---
title: >-
  [Paper Note] AnyI2V: Animating Any Conditional Image with Motion Control
description: >-
  [ICCV 2025][3D Vision][training-free] This paper proposes AnyI2V, a training-free framework that accepts arbitrary-modality images (mesh, point cloud, depth map, skeleton, etc.) as first-frame conditions and combines user-defined trajectories for motion-controlled video generation, outperforming existing training-free methods and competing with trained methods on FID/FVD/ObjMC metrics.
tags:
  - ICCV 2025
  - 3D Vision
  - training-free
  - Image-to-Video
  - Motion Control
  - diffusion model
  - multimodal conditioning
date: 2026-05-08
content_hash: 4dd071edba4e0982
---

# AnyI2V: Animating Any Conditional Image with Motion Control

**Conference**: ICCV 2025
**arXiv**: [2507.02857](https://arxiv.org/abs/2507.02857)
**Code**: [https://henghuiding.com/AnyI2V/](https://henghuiding.com/AnyI2V/) (project page available)
**Area**: 3D Vision
**Keywords**: training-free, Image-to-Video, Motion Control, diffusion model, multimodal conditioning

## TL;DR
This paper proposes AnyI2V, a training-free framework that accepts arbitrary-modality images (mesh, point cloud, depth map, skeleton, etc.) as first-frame conditions and combines user-defined trajectories for motion-controlled video generation, outperforming existing training-free methods and competing with trained methods on FID/FVD/ObjMC metrics.

## Background & Motivation
- **T2V methods** (e.g., AnimateDiff) rely on text prompts and lack precise spatial layout control.
- **I2V methods** (e.g., DragNUWA, DragAnything) require real RGB images as the first frame, limiting content editability.
- **ControlNet-based methods** can incorporate image conditions but require large amounts of training data (especially paired data), do not support modalities such as mesh or point clouds, and require retraining when switching backbones.
- Existing motion control methods either lack spatial layout control (T2V-based) or motion controllability (ControlNet-based), and most require training.

**Core Limitation**: No unified, training-free framework simultaneously supports (1) arbitrary-modality first-frame spatial conditioning and (2) user-defined trajectory-based motion control.

## Core Problem
How can a video diffusion model accept arbitrary-modality conditional images and generate motion along user-specified trajectories without training additional modules? The key challenges are:
1. How to inject features from diverse modality images without introducing appearance bias.
2. How to achieve cross-frame temporal alignment for motion control.
3. How to handle precise motion control for irregularly shaped objects.

## Method

### Overall Architecture
The framework is built on a 3D U-Net video diffusion model (AnimateDiff by default) and consists of three steps:
1. DDIM inversion of the conditional image to extract features (1000 steps, extracted at $t_\alpha = 201$).
2. Injection of debiased features into the first frame of the generation process (structure-preserved feature injection).
3. Cross-frame alignment and trajectory control via latent optimization (zero-shot trajectory control + semantic masks).

### Key Designs
1. **Structure-Preserved Feature Injection**:

    - Empirical analysis reveals: residual hidden states provide the best structural control but leak appearance information; queries provide high-level semantic and entity-aware representations; attention maps exhibit poor temporal consistency.
    - **Patch-wise AdaIN debiasing** is applied to residual hidden states: features are divided into non-overlapping patches ($p=4$), and AdaIN is applied at the patch level to remove appearance leakage while preserving structural information.
    - Debiased residual hidden states and queries are injected as first-frame guidance.
    - K/V of the first frame are copied to subsequent frames ($K_{2:f}=K_1,\ V_{2:f}=V_1$) to ensure content consistency.

2. **Zero-Shot Trajectory Control**:

    - PCA-based analysis reveals that queries exhibit strong temporal consistency and entity-awareness, making them the most suitable features for cross-frame alignment.
    - Users define bounding boxes to specify target regions and trajectories.
    - The top $M=64$ principal components of queries are extracted via PCA, and subsequent frames are aligned to the first frame in this feature space.
    - Optimization objective:
    $$z_t^* = \arg\min_{z_t} \sum_{i=1}^{n}\sum_{j=2}^{f} \mathcal{L}(F_j[\mathcal{B}_j^i], \text{SG}(F_1[\mathcal{B}_1^i]))$$
    - where $F_j = \text{PCA}(\text{Query}_j, M)$ and SG denotes stop-gradient.

3. **Semantic Mask Generation**:

    - Addresses the imprecision of bounding boxes for irregularly shaped objects.
    - Salient points $P$ are selected within the first-frame bounding box, and cosine similarity with features across frames is computed.
    - K-Means binary clustering on the similarity map generates adaptive semantic masks.
    - The final loss is a mask-constrained MSE:
    $$\mathcal{L}_j^i = \|M_1^i \odot M_j^i \odot (F_j[\mathcal{B}_j^i] - \text{SG}(F_1[\mathcal{B}_1^i]))\|_2^2$$
    - Dynamic masks allow natural object deformation, offering greater flexibility than static masks.

### Loss & Training
- **Fully training-free**: no additional modules require training.
- DDIM sampling uses 25 steps with latent optimization every 5 steps ($t' \geq 20$) at a learning rate of 0.01.
- Inversion takes approximately 8 seconds; generation takes approximately 35 seconds (A800 GPU, half precision).
- Optimization targets Query 1.1 from `up_blocks.1` and Query 2.0 from `up_blocks.2` (multi-resolution optimization).

## Key Experimental Results

| Dataset | Metric | AnyI2V | DragAnything | FreeTraj | ObjCtrl-2.5D | Baseline |
|--------|------|--------|-------------|----------|-------------|----------|
| VIPSeg+Web | FID↓ | **104.53** | 95.83 | 128.78 | 111.82 | 141.95 |
| VIPSeg+Web | FVD↓ | **569.89** | 556.09 | 672.87 | 605.96 | 970.26 |
| VIPSeg+Web | ObjMC↓ | **16.39** | 13.60 | 24.00 | 23.12 | 38.26 |

Note: DragAnything and similar methods require training (achieving a better FID of 95.83), but AnyI2V **ranks first among all training-free methods** and remains highly competitive with trained methods. In particular, AnyI2V achieves an ObjMC of 16.39, far surpassing the best competing training-free method (23.12) and approaching the trained method DragAnything (13.60).

### Ablation Study

| Configuration | FID↓ | FVD↓ | ObjMC↓ |
|------|------|------|--------|
| w/o K&V consistency | 108.18 | 587.69 | 16.81 |
| w/o PCA Reduction | 105.95 | 585.04 | 17.14 |
| w/ Static Mask | 105.44 | 598.15 | 16.92 |
| w/o Semantic Mask | 105.78 | 579.88 | 17.62 |
| opt. Residual Hidden (instead of Query) | 129.40 | 647.52 | 36.23 |
| **Full (AnyI2V)** | **104.53** | **569.89** | **16.39** |

- Optimizing residual hidden states instead of queries causes ObjMC to surge to 36.23, validating the critical role of queries as alignment targets.
- PCA dimensionality $M=64$ is optimal; both smaller and larger values degrade performance.
- Dynamic semantic masks outperform static masks, particularly on FVD (temporal consistency).
- Multi-resolution optimization (Query 1.1 & 2.0) significantly outperforms single-resolution optimization.

## Highlights & Insights
- **Arbitrary modality input**: The first framework to support mesh, point clouds, and other modalities that ControlNet cannot handle as conditional images.
- **Mixed modality input**: Supports simultaneous use of depth maps (background) and sketches (foreground) as combined conditions.
- **Fully training-free**: Backbone switching requires no retraining; generalizability has been verified on AnimateDiff, LaVie, and VideoCrafter2.
- **Patch-wise AdaIN debiasing**: A simple and effective solution to the appearance leakage problem in residual hidden states.
- **PCA-based feature analysis**: PCA-reduced visualizations provide deep insight into the temporal properties of different features, identifying queries as the optimal alignment target.
- Supports LoRA and text-based editing, offering rich editing flexibility.

## Limitations & Future Work
- Motion control accuracy degrades for large-scale motions with long trajectories.
- Occlusion and ambiguous spatial relationship scenarios are not handled well.
- First-frame control precision is lower than ControlNet, as feature injection only operates in early denoising steps.
- Generation speed (approximately 43 seconds per video) still has room for optimization.
- Future work could incorporate lightweight fine-tuning to further improve adaptability.

## Related Work & Insights
- **vs. DragAnything/DragNUWA**: These methods require training and only accept RGB images; AnyI2V is training-free and supports arbitrary modalities. DragAnything achieves marginally better quantitative results (FID 95.83 vs. 104.53), but AnyI2V offers far greater flexibility.
- **vs. ControlNet-based methods**: ControlNet requires separate training for each modality, cannot handle mesh or point clouds, and does not support motion control. AnyI2V handles all modalities uniformly with built-in trajectory control.
- **vs. ObjCtrl-2.5D/FreeTraj/TrailBlazer**: Among training-free methods, AnyI2V leads comprehensively across all metrics (ObjMC 16.39 vs. 23.12) while supporting a richer set of conditional inputs.

## Potential Research Directions
- Extending AnyI2V's training-free conditional control paradigm to DiT-based architectures (e.g., Open-Sora, CogVideoX) may require re-analyzing which features within DiT are suitable for injection and alignment.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The training-free unified framework and PCA-based feature analysis are novel, though individual techniques (AdaIN debiasing, latent optimization) are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are comprehensive (5 variants + hyperparameter analysis), but lack user studies and quantitative evaluation across more backbone networks.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is clearly structured, feature analysis visualizations are intuitive, and motivation is thoroughly articulated.
- **Value**: ⭐⭐⭐⭐ The combination of training-free operation and arbitrary modality support offers strong practical value, though applicability in the DiT era remains to be examined.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)
- [\[ICCV 2025\] Find Any Part in 3D](find_any_part_in_3d.md)
- [\[ICCV 2025\] CMT: A Cascade MAR with Topology Predictor for Multimodal Conditional CAD Generation](cmt_a_cascade_mar_with_topology_predictor_for_multimodal_conditional_cad_generat.md)
- [\[ICCV 2025\] Easi3R: Estimating Disentangled Motion from DUSt3R Without Training](easi3r_estimating_disentangled_motion_from_dust3r_without_training.md)
- [\[ICCV 2025\] Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)

<!-- RELATED:END -->
