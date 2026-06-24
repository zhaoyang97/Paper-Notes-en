---
title: >-
  [Paper Note] RayZer: A Self-supervised Large View Synthesis Model
description: >-
  [ICCV 2025][3D Vision][Self-supervised Learning] This paper proposes RayZer, a self-supervised multi-view 3D vision model that requires no 3D supervision (no camera poses, no scene geometry annotations). By decoupling images into camera parameters and scene representations, RayZer performs 3D-aware image autoencoding and achieves performance on novel view synthesis that matches or surpasses oracle methods relying on pose annotations.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Self-supervised Learning"
  - "Novel View Synthesis"
  - "Camera Pose Estimation"
  - "Plücker Rays"
  - "Transformer"
date: 2026-05-08
content_hash: be2b70047c72f20d
---

# RayZer: A Self-supervised Large View Synthesis Model

**Conference**: ICCV 2025
**arXiv**: [2505.00702](https://arxiv.org/abs/2505.00702)  
**Code**: [Project Page](https://hwjiang1510.github.io/RayZer/)  
**Area**: 3D Vision
**Keywords**: Self-supervised Learning, Novel View Synthesis, Camera Pose Estimation, Plücker Rays, Transformer

## TL;DR

This paper proposes RayZer, a self-supervised multi-view 3D vision model that requires no 3D supervision (no camera poses, no scene geometry annotations). By decoupling images into camera parameters and scene representations, RayZer performs 3D-aware image autoencoding and achieves performance on novel view synthesis that matches or surpasses oracle methods relying on pose annotations.

## Background & Motivation

Self-supervised learning has driven the rise of foundation models in LLMs, VLMs, and visual generation, yet **3D vision models remain heavily dependent on ground-truth 3D geometry and camera pose annotations**. Such annotations are typically obtained via time-consuming optimization pipelines (e.g., COLMAP) and are not always accurate. This dependency limits the scalability and effectiveness of 3D model learning.

The key question is: **How far can a 3D vision model go without any 3D supervision?**

Specific limitations of existing approaches include:
- Oracle methods such as GS-LRM and LVSM require ground-truth poses at both training and inference time.
- Methods like MegaSynth and Stereo4D scale training with synthetic data, but data curation remains laborious.
- COLMAP-derived pose annotations are inherently noisy, which caps the performance of models that depend on them.
- Self-supervised methods such as RUST employ implicit pose representations, making pose–scene disentanglement difficult and precluding explicit 3D awareness.

The core insight of RayZer is to **use the model's own predicted camera poses to render target views and provide photometric supervision**, rather than relying on ground-truth poses. This reformulates self-supervised training as a 3D-aware image autoencoding problem.

## Method

### Overall Architecture

RayZer takes pose-free, calibration-free multi-view images $\mathcal{I} = \{I_i \in \mathbb{R}^{H \times W \times 3} | i = 1, ..., K\}$ as input and sequentially predicts: (1) camera parameters (intrinsics + poses) → (2) Plücker ray maps → (3) a latent scene representation → (4) rendered novel views. The entire pipeline is a pure Transformer architecture with 24 layers (8 for camera estimation, 8 for scene encoding, 8 for rendering decoding).

A key information-flow control for self-supervised training: the input images are split into two non-overlapping subsets $\mathcal{I}_\mathcal{A}$ (for scene reconstruction) and $\mathcal{I}_\mathcal{B}$ (for supervision), preventing trivial solutions.

### Key Designs

1. **Camera Estimator**:

    - Employs learnable camera tokens $\mathbf{p} \in \mathbb{R}^{K \times d}$ (one per view), concatenated with image tokens $\mathbf{f} \in \mathbb{R}^{Khw \times d}$ and fed into full self-attention Transformer layers.
    - One reference view is chosen (identity rotation + zero translation); remaining views predict relative poses.
    - Rotation is parameterized with a 6D continuous representation and predicted via MLP: $p_i = \text{MLP}_{pose}([\mathbf{p}_i^*, \mathbf{p}_c^*])$.
    - Intrinsics are parameterized with a single focal length value: $\text{focal} = \text{MLP}_{focal}(\mathbf{p}_c^*)$.
    - **Design Motivation**: A low-dimensional, geometrically well-defined SE(3) parameterization facilitates information disentanglement; the pose-first paradigm (predicting poses before reconstructing the scene) provides better mutual regularization.

2. **Plücker Ray Bridging**: The predicted SE(3) poses and intrinsics are converted into pixel-aligned Plücker ray maps $\mathcal{R} \in \mathbb{R}^{K \times H \times W \times 6}$. This is the **only 3D prior** in RayZer, jointly encoding:

    - The alignment between 2D pixels and rays.
    - 3D ray geometry (direction and origin).
    - The physical relationship among camera, pixels, and scene.

   Ray maps are tokenized via a linear layer and fused with image tokens: $\mathbf{x}_\mathcal{A} = \text{MLP}_{fuse}([\mathbf{f}_\mathcal{A}, \mathbf{r}_\mathcal{A}])$.
    - **Key Detail**: The original image tokens $\mathbf{f}$ are used instead of the camera estimator outputs $\mathbf{f}^*$, preventing information leakage from $\mathcal{I}_\mathcal{B}$.

3. **Latent Set Scene Representation and Full-Transformer Rendering**:

    - The scene is represented as a set of learnable tokens $\mathbf{z} \in \mathbb{R}^{L \times d}$ without explicit 3D structure; 3D properties are acquired entirely through learning.
    - Scene reconstruction: $\{\mathbf{z}^*, \mathbf{x}_\mathcal{A}^*\} = \mathcal{E}_{scene}(\{\mathbf{z}, \mathbf{x}_\mathcal{A}\})$.
    - Rendering: given Plücker ray tokens $\mathbf{r}$ for the target camera, the rendering decoder generates images: $\hat{I} = \text{MLP}_{rgb}(\mathbf{r}^*)$.
    - This is analogous to the classical rendering formulation $v = R(\text{scene}, \text{ray})$, except that the "rendering equation" is a parameterized learnable model.

### Loss & Training

Pure photometric self-supervised loss:

$$\mathcal{L} = \frac{1}{K_\mathcal{B}} \sum_{\hat{I} \in \hat{\mathcal{I}}_\mathcal{B}} (\text{MSE}(I, \hat{I}) + \lambda \cdot \text{Percep}(I, \hat{I}))$$

where $\lambda = 0.2$ is the perceptual loss weight. Training uses a learning rate of $4 \times 10^{-4}$ with a cosine scheduler, 50,000 iterations, batch size 256, resolution $256 \times 256$, and patch size 16. The two subsets $\mathcal{I}_\mathcal{A}$ and $\mathcal{I}_\mathcal{B}$ are randomly sampled during training.

## Key Experimental Results

### Main Results

| Dataset | Method | Training Supervision | Requires GT Poses | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|---------|---------|-------|-------|--------|
| DL3DV | GS-LRM | 2D+Camera | Yes | 23.49 | 0.712 | 0.252 |
| DL3DV | LVSM | 2D+Camera | Yes | 23.69 | 0.723 | 0.242 |
| DL3DV | **RayZer** | **2D only** | **No** | **24.36** | **0.757** | **0.209** |
| RealEstate | GS-LRM | 2D+Camera | Yes | 24.25 | 0.770 | 0.227 |
| RealEstate | LVSM | 2D+Camera | Yes | 27.00 | 0.851 | 0.157 |
| RealEstate | **RayZer** | **2D only** | **No** | **27.48** | **0.861** | **0.146** |
| Objaverse | LVSM | 2D+GT | Yes | **32.34** | **0.950** | **0.050** |
| Objaverse | RayZer | 2D only | No | 31.52 | 0.945 | 0.052 |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Note |
|------|-------|-------|--------|------|
| RayZer (full) | **24.36** | **0.757** | **0.209** | pose-first + Plücker + latent |
| 3DGS representation | — | — | failed | Explicit 3D representation fails to converge |
| w/o Plücker rays, w/ SE(3) | 22.73 | 0.687 | 0.249 | Lacks pixel-level conditioning |
| w/o explicit poses, w/ implicit features | 23.13 | 0.700 | 0.251 | Information leakage + non-interpolable |
| Scene-first paradigm | 13.31 | 0.338 | 0.732 | Scene reconstructed before pose estimation; collapses |

### Key Findings

- **RayZer surpasses oracle methods on DL3DV and RealEstate**: Poses on these datasets are derived from COLMAP and are inherently noisy. Self-supervised learning can discover a pose space more favorable to view synthesis than COLMAP.
- **Slight gap behind LVSM on Objaverse (perfect GT poses)**: The gap is small (32.34 vs. 31.52), confirming that oracle methods have an advantage when GT poses are perfect.
- **The pose-first paradigm is critical**: The scene-first paradigm yields only 13.31 dB PSNR, a complete collapse.
- **Plücker rays are the essential 3D prior**: They outperform direct SE(3) pose encoding by 1.6 dB PSNR.
- Self-supervised poses support geometric interpolation, enabling novel view synthesis along continuous camera trajectories.

## Highlights & Insights

- **Self-supervised 3D models can match or exceed supervised counterparts**: This counterintuitive result indicates that COLMAP pose noise is a bottleneck for supervised methods.
- **Minimal 3D prior**: The only 3D prior is the Plücker ray structure; everything else is learned from data.
- **Elegant information-flow control**: Using original image tokens rather than camera estimator outputs prevents information leakage; separating the two subsets ensures non-trivial solutions.
- **Latent representation outperforms explicit 3D**: 3DGS representation fails entirely to converge under self-supervised training, validating the necessity of latent representations in this setting.

## Limitations & Future Work

- The learned pose space does not fully correspond to real-world pose space, limiting direct application to pose estimation tasks.
- Training on consecutive video frames outperforms training on unordered image sets, indicating residual sensitivity to input ordering.
- Training and evaluation are conducted only at $256$ resolution; performance on high-resolution scenes remains to be verified.
- Each dataset is trained separately; zero-shot generalization across datasets has not been validated.
- The static scene assumption limits applicability to dynamic scenes.

## Related Work & Insights

- Three key distinctions from RUST: (1) pose-first vs. scene-first, (2) explicit SE(3) vs. implicit pose, (3) pure self-attention vs. cross-attention.
- The latent set scene representation of LVSM is a key inspiration; RayZer demonstrates that this representation can be learned effectively without pose supervision.
- Training on video data (abundantly available online) offers greater scaling potential than relying on unordered image collections.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Achieves oracle-level performance with zero 3D supervision; a significant milestone in self-supervised 3D learning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons across three datasets (DL3DV / RealEstate / Objaverse) with thorough ablation and pose analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous logic, clear motivation, and well-justified design choices.
- Value: ⭐⭐⭐⭐⭐ Demonstrates the feasibility of 3D vision models without supervised learning, establishing a new paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input](../../CVPR2025/3d_vision/spatialdreamer_self-supervised_stereo_video_synthesis_from_monocular_input.md)
- [\[ICCV 2025\] S3E: Self-Supervised State Estimation for Radar-Inertial System](s3e_self-supervised_state_estimation_for_radar-inertial_system.md)
- [\[CVPR 2026\] From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis](../../CVPR2026/3d_vision/from_none_to_all_self-supervised_3d_reconstruction_via_novel_view_synthesis.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[CVPR 2026\] WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments](../../CVPR2026/3d_vision/wildrayzer_self-supervised_large_view_synthesis_in_dynamic_environments.md)

</div>

<!-- RELATED:END -->
