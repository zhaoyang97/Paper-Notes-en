---
title: >-
  [Paper Note] Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot
description: >-
  [ECCV 2024][3D Vision][Human Mesh Recovery] Multi-HMR is the first single-shot multi-person whole-body (including hands and facial expressions) 3D human mesh recovery method. It employs a ViT backbone and a Human Perception Head (HPH) with cross-attention, combined with a new synthetic dataset named CUFFS to address the difficulty of hand pose learning, achieving state-of-the-art (SOTA) performance on both multi-person and whole-body benchmarks.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Human Mesh Recovery"
  - "Multi-Person Whole-Body Pose Estimation"
  - "Single-Shot Detection"
  - "SMPL-X"
  - "Synthetic Dataset"
date: 2026-05-08
content_hash: 7eb144eb66814181
---

# Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot

**Conference**: ECCV 2024  
**arXiv**: [2402.14654](https://arxiv.org/abs/2402.14654)  
**Code**: [https://github.com/naver/multi-hmr](https://github.com/naver/multi-hmr)  
**Area**: 3D Vision  
**Keywords**: Human Mesh Recovery, Multi-Person Whole-Body Pose Estimation, Single-Shot Detection, SMPL-X, Synthetic Dataset

## TL;DR

Multi-HMR is the first single-shot multi-person whole-body (including hands and facial expressions) 3D human mesh recovery method. It employs a ViT backbone and a Human Perception Head (HPH) with cross-attention, combined with a new synthetic dataset named CUFFS to address the difficulty of hand pose learning, achieving state-of-the-art (SOTA) performance on both multi-person and whole-body benchmarks.

## Background & Motivation

**Background**: Human Mesh Recovery (HMR) has advanced significantly over the past years. Single-person HMR methods (e.g., HMR, HMR2.0) regress SMPL parameters from cropped images; whole-body methods (e.g., PIXIE, Hand4Whole) process the body, hands, and face separately via multi-stage cropping; multi-person methods (e.g., ROMP, BEV, PSVT) achieve single-shot multi-person detection but are limited to body-only estimation. However, no existing method simultaneously satisfies all four requirements: whole-body prediction, multi-person processing, camera-space localization, and camera-intrinsic adaptation.

**Limitations of Prior Work**: (1) The combination of multi-person and whole-body estimation is exceptionally challenging; hands and faces often have very low resolutions in natural images, making it difficult for single-shot methods to directly learn fine-grained hand pose from the global image. (2) Existing whole-body methods rely on multi-stage cropping pipelines (detecting the person first, then cropping the hand/face regions), which increases computational overhead and prevents end-to-end learning. (3) Current multi-person methods do not support whole-body prediction (facial expressions + hand poses). (4) Most methods assume a fixed camera model and cannot adapt to different camera intrinsics.

**Key Challenge**: While single-shot methods are efficient and end-to-end trainable, regressing fine-grained hand and face parameters from low-resolution global features is extremely difficult; multi-stage cropping methods capture fine details but introduce error propagation and efficiency issues from the detection pipeline.

**Goal**: (1) How to simultaneously achieve multi-person detection and whole-body (including hands and face) parameter regression within a single-shot framework? (2) How to address the issues of insufficient and invisible hand details in the training data?

**Key Insight**: The authors adopt a simple Transformer-first design, utilizing a ViT backbone where a cross-attention head allows each detected human token to query global image features for predicting whole-body parameters. Concurrently, a synthetic dataset named CUFFS, specifically containing close-ups with clearly visible hands of full-body subjects, is introduced to enhance training.

**Core Idea**: Implementing single-shot multi-person whole-body mesh recovery using a ViT and a cross-attention head, complemented by a close-up synthetic hand dataset to alleviate the scarcity of hand training data.

## Method

### Overall Architecture

Inputting an RGB image, the ViT backbone extracts patch-level token embeddings $\mathbf{E} \in \mathbb{R}^{H/P \times W/P \times D}$. First, a CenterNet-like paradigm is used to predict human center heatmaps for person detection. Second, the detected center tokens serve as queries for the Human Perception Head (HPH), which aggregates global image features via cross-attention and regresses the SMPL-X parameters (body pose $\boldsymbol{\theta}$, shape $\boldsymbol{\beta}$, facial expression $\boldsymbol{\alpha}$) and depth $t_z$. Optionally, camera ray directions are encoded for camera-aware prediction.

### Key Designs

1. **Human Perception Head (HPH)**:

    - **Function**: Efficiently regressing whole-body SMPL-X parameters for each detected human from the global image features.
    - **Mechanism**: For $N$ detected persons, $N$ query vectors are initialized as $\mathbf{q}_n = (\mathbf{E}_{i,j} \oplus \bar{\mathbf{x}}) + \mathbf{p}_{ij}$, concatenating the token embedding with the mean body parameters. These are processed through $L=2$ layers of cross-attention blocks: $\mathbf{Q}^l = \text{MLP}_l(\text{SA}_l(\text{CA}_l[\mathbf{Q}^{l-1}, \mathbf{E}]))$, where CA dynamically focuses each person's query on all image tokens, and SA enables interaction among queries of different individuals.
    - **Design Motivation**: Compared to traditional per-token independent regressors, cross-attention allows the prediction head to access global contextual information (such as cues from other body parts), and self-attention between different persons helps handle occlusions and depth ordering. Experiments show that HPH converges faster and performs better than iterative regressors.

2. **CUFFS Synthetic Dataset (Close-Up Frames of Full-Body Subjects)**:

    - **Function**: Providing close-up full-body training data with diverse and clearly visible hand poses.
    - **Mechanism**: Utilizing HumGen3D in Blender to render 60k synthetic images. Humans are placed around 2.5 meters away from the camera to ensure hand visibility. Body poses are sampled from BEDLAM/AGORA/UBody, and rich hand poses from the InterHand dataset are introduced for hand-replacement augmentation. Precise ground-truth annotations are obtained via mesh correspondence from SMPL-X to HumGen3D.
    - **Design Motivation**: In existing synthetic datasets (such as BEDLAM and AGORA), persons are usually far from the camera with hands occupying only a few pixels, and the hand poses lack diversity. Single-shot methods do not rely on hand-cropping pipelines, hence they require training data that inherently contains clear hand details.

3. **Optional Camera Embedding**:

    - **Function**: Improving 3D spatial localization accuracy when camera intrinsics are available.
    - **Mechanism**: For each patch center $(u_i, v_j)$, the ray direction is computed as $\mathbf{r}_{i,j} = \mathbf{K}^{-1}[u_i, v_j, 1]^T$. The first two components are Fourier-encoded and concatenated to the patch token embedding from the backbone: $\mathbf{E} \doteq \mathbf{E} \oplus \mathbf{E}_K$.
    - **Design Motivation**: Camera intrinsics directly affect the 3D-to-2D projection relationship. Simple linear embeddings can degrade performance, whereas Fourier encoding of ray directions paired with focal length normalization yields consistent performance improvements.

### Loss & Training

The total loss is formulated as $\mathcal{L} = \mathcal{L}_\text{det} + \mathcal{L}_\text{params} + \lambda(\mathcal{L}_\text{mesh} + \mathcal{L}_\text{reproj})$. Binary cross-entropy is used for detection; L1 loss is applied to the SMPL-X parameters, offsets, and depth; the mesh loss directly supervises the 3D vertex coordinates; the 2D reprojection loss provides additional geometric constraints. Depth prediction is performed in log-space using normalized nearness parametrization to ensure robustness to focal length variations. The ViT backbone is initialized with DINOv2, trained with a batch size of 8, a learning rate of 5e-5, and for 400k iterations.

## Key Experimental Results

### Main Results

Multi-person body-only benchmarks (MPJPE on 3DPW / PCK3D on MuPoTs):

| Method | 3DPW PA-MPJPE↓ | MuPoTs PCK3D↑ | CMU MPJPE↓ |
|------|----------------|---------------|------------|
| ROMP | 47.3 | 69.9 | 108.1 |
| BEV | 46.9 | 70.2 | 105.3 |
| PSVT | 45.7 | - | 97.7 |
| **Multi-HMR-448** | 43.8 | **80.6** | - |
| **Multi-HMR** | **41.7** | **85.0** | **82.8** |

Whole-body benchmark EHF (PVE-All / PVE-Hands):

| Method | PVE-All↓ | PA-PVE-All↓ | PVE-Hands↓ |
|------|----------|-------------|------------|
| Hand4Whole | 76.8 | 50.3 | 39.8 |
| OSX | 70.8 | 48.7 | 53.7 |
| **Multi-HMR** | **44.2** | **32.7** | **36.4** |

### Ablation Study

| Configuration | MuPoTs PCK3D↑ | 3DPW MPJPE↓ | EHF PVE↓ | Notes |
|------|---------------|-------------|----------|------|
| HRNet + Reg | 65.8 | 83.2 | 143.1 | CNN backbone + iterative regression |
| ViT-S + Reg | 70.1 | 80.2 | 90.6 | ViT backbone + iterative regression |
| ViT-S + HPH | 70.9 | 80.1 | 80.1 | ViT backbone + HPH |
| ViT-B + HPH | 76.3 | 73.5 | 55.3 | Larger backbone |
| +CUFFS | 76.0 | 72.9 | 49.8 | Significant hand improvement |

### Key Findings

- The ViT backbone significantly outperforms HRNet with comparable parameters (EHF PVE drops from 143.1 to 90.6). The global attention of ViT is crucial for whole-body prediction.
- HPH converges faster and performs better than iterative regressors, with self-attention among queries also positively contributing.
- The CUFFS dataset mainly improves hand metrics (EHF-H drops from 47.4 to 40.5), while having minimal effect on other metrics.
- DINOv2 pretraining consistently outperforms other pretraining schemes and yields much faster convergence.
- Even ViT-S at $448\times 448$ resolution remains competitive (achieving real-time performance at 30 fps).

## Highlights & Insights

- **Four-in-one Unified Framework**: Achieving multi-person, whole-body, camera-space, and camera-aware capabilities simultaneously within a single model for the first time.
- **Impacting Performance of Synthetic Data**: Pure synthetic training outperforms training mixed with real data, which challenges the intuition that "real data is always necessary."
- **ViT + Single-Shot = Winner**: The standard ViT architecture directly benefits from advancements in self-supervised pretraining.
- **Simple and Effective HPH Design**: Adding a simple two-layer cross-attention mechanism significantly outperforms complex multi-stage pipelines.

## Limitations & Future Work

- Patch-level detection limits performance in dense crowd scenes (where two person centers within the same patch would collide).
- Detection is difficult when the head is occluded (since the head is used as the primary root joint).
- Hand and facial estimation for distant persons still has room for improvement.
- The kinematic tree joint rotation representation of SMPL-X can lead to error accumulation in end effectors.
- Multi-query per patch schemes can be explored to handle crowded scenarios.

## Related Work & Insights

- **ROMP/BEV**: Single-shot multi-person methods using HRNet backbones for body-only recovery.
- **OSX**: Single-person whole-body method using a ViT but relying on keypoint-guided high-resolution feature resampling.
- **BEDLAM**: Demonstrated that purely synthetic data can be used to train SOTA models.
- **Related Work & Insights**: A simple architecture (ViT + cross-attention) combined with appropriate synthetic data can outperform complex multi-stage pipelines.

## Rating

- Novelty: ⭐⭐⭐⭐ The first four-in-one unified framework; the CUFFS dataset is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6+ evaluation benchmarks with comprehensive ablations (architecture, data, losses, cameras, and resolutions).
- Writing Quality: ⭐⭐⭐⭐ Structure is clear with detailed experimental setups.
- Value: ⭐⭐⭐⭐ Highly practical, open-source code, and promotes the democratization of multi-person whole-body HMR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Divide and Fuse: Body Part Mesh Recovery from Partially Visible Human Images](divide_and_fuse_body_part_mesh_recovery_from_partially_visible_human_images.md)
- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](../../AAAI2026/3d_vision/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](global-to-pixel_regression_for_human_mesh_recovery.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)

</div>

<!-- RELATED:END -->
