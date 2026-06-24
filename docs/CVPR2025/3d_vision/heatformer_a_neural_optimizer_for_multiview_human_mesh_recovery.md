---
title: >-
  [Paper Note] HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery
description: >-
  [CVPR 2025][3D Vision][Human Mesh Recovery] Proposes HeatFormer, a Transformer-based neural optimizer that formulates SMPL parameter estimation as a heatmap generation and alignment problem to iteratively optimize and recover human shape and pose from multi-view images. It achieves state-of-the-art accuracy of 29.5mm MPJPE on Human3.6M, demonstrating strong robustness to the number of views, camera configurations, and occlusions.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human Mesh Recovery"
  - "Multi-view"
  - "Neural Optimizer"
  - "Heatmap"
  - "Transformer"
date: 2026-05-08
content_hash: eb0fa60d6f5f59e0
---

# HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery

**Conference**: CVPR 2025  
**arXiv**: [2412.04456](https://arxiv.org/abs/2412.04456)  
**Code**: [https://vision.ist.i.kyoto-u.ac.jp/research/heatformer/](https://vision.ist.i.kyoto-u.ac.jp/research/heatformer/) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Human Mesh Recovery, Multi-view, Neural Optimizer, Heatmap, Transformer

## TL;DR

Proposes HeatFormer, a Transformer-based neural optimizer that formulates SMPL parameter estimation as a heatmap generation and alignment problem to iteratively optimize and recover human shape and pose from multi-view images. It achieves state-of-the-art accuracy of 29.5mm MPJPE on Human3.6M, demonstrating strong robustness to the number of views, camera configurations, and occlusions.

## Background & Motivation

**Background**: Most mainstream methods for Human Mesh Recovery (HMR) are based on monocular images, directly predicting SMPL parameters via feed-forward regression networks. Some multi-view methods integrate multi-view information through feature aggregation or voxelization, but essentially remain single-pass inference approaches.

**Limitations of Prior Work**: (1) Monocular methods are limited by depth ambiguity and occlusions—in real-world scenes, humans are often occluded by objects like tables, chairs, and sofas, or experience self-occlusion, where a single view cannot recover the occluded parts. (2) Existing multi-view methods mostly adopt feed-forward architectures, utilizing multi-view information only once at the input stage, and cannot repeatedly refer to visual evidence for correction during the estimation process. (3) Most methods rely on fixed camera configurations, restricting actual deployment flexibility.

**Key Challenge**: How to design a method that fully exploits complementary multi-view information (especially for resolving occlusions across views) while remaining independent of specific camera numbers and configurations? Feed-forward methods output results by only "looking once" at the input, lacking self-correction capabilities; traditional optimization methods can iterate and correct but require precise 2D-3D correspondences and are difficult to train end-to-end.

**Goal**: To design a neural optimizer capable of iteratively utilizing multi-view image feedback during inference to progressively refine SMPL estimates, while remaining agnostic to the number of views and camera configurations.

**Key Insight**: Reformulate SMPL parameter estimation as a heatmap generation and alignment problem. Heatmaps provide stable spatial gradients (more suitable for backpropagation than keypoint coordinates), enabling end-to-end neural optimization. By using a Transformer encoder-decoder architecture, multiple forward passes of the decoder are unrolled into an iterative optimization process.

**Core Idea**: Use repetitive forward inference of the Transformer decoder as an unrolled iterative optimization, aligning current SMPL estimates with multi-view observations via heatmap cross-attention at each step.

## Method

### Overall Architecture

The input consists of multi-view images, and the output is the corresponding SMPL shape parameters $\beta \in \mathbb{R}^{10}$ and pose parameters $\theta \in \mathbb{R}^{24 \times 3}$ for each view. The pipeline is divided into: (1) extracting image features for each view using a ViT and extracting multi-joint heatmaps using AdaFuse; (2) integrating the multi-joint heatmaps of each view into a single feature using HeatEncoder, then concatenating it with the image features; (3) the Decoder receives heatmaps generated from the current SMPL estimation as queries, performs cross-attention with the encoder output, and outputs parameter updates; (4) updating the SMPL parameters, repeating step 3 for a total of 3 times.

### Key Designs

1. **HeatEncoder**:

    - **Function**: Integrates $k$ joint heatmaps from each view into a unified feature representation.
    - **Mechanism**: For the heatmap set $P \in \mathbb{R}^{k \times H \times W}$ of each view, it is first split into patches according to spatial locations. Each patch is flattened into a token, which is accompanied by joint order encoding and spatial position encoding. All tokens, along with a CLS token, are input into a self-attention module, where the CLS token aggregates spatial information of all joints via self-attention to output an integrated heatmap feature. After performing this operation independently for each view, the integrated heatmap is concatenated with the ViT image features spatially to form the final token sequence.
    - **Design Motivation**: Directly using keypoint coordinates as features is unfavorable for gradient propagation (discontinuous gradients). Heatmaps, as 2D probability distributions of keypoint locations, provide smooth spatial gradients and are naturally suited for end-to-end optimization. Integrating across joints via self-attention enables learning the spatial coordination relationships among joints.

2. **Decoder as a Neural Optimizer**:

    - **Function**: Achieves iterative optimization of SMPL parameters through repeated forward inference.
    - **Mechanism**: In each iteration, the current SMPL parameters are instantiated into a mesh $\rightarrow$ joints are extracted $\rightarrow$ projected onto each view as 2D heatmaps $\rightarrow$ encoded by HeatEncoder $\rightarrow$ used as queries for the Decoder. The Decoder aligns the queries (current estimated heatmaps) with the keys/values (input image heatmaps + features) via cross-attention. The output tensor is average-pooled and passed through an MLP to obtain the corrections (residuals) for the SMPL parameters. The corrections are added to the current estimate, and new heatmaps are generated for the next iteration. Experiments show convergence within 3–4 iterations.
    - **Design Motivation**: Traditional optimization requires explicit gradient computation, which is slow and prone to local optima. "Unrolling" the optimization process as repetitive forward inference of a Transformer inherits the self-correction capability of optimization while maintaining the end-to-end trainability and inference speed of neural networks.

3. **View-Agnostic Estimation and Flexibility**:

    - **Function**: Makes the model completely agnostic to the number of cameras and their configurations.
    - **Mechanism**: Outputs view-dependent SMPL parameters (rather than a single 3D estimate), which can be selected based on the view closest to the downstream application, or by choosing the optimal estimate based on 2D reprojection quality. Since the number of tokens in a Transformer is variable, the same model can be trained and evaluated on different numbers of views. When camera calibration is available, extrinsic parameters are utilized to optimize the heatmaps, and when unavailable, the system degrades to a weak-perspective camera model.
    - **Design Motivation**: Fixed camera configurations severely limit practical applications—different rooms and installation conditions lead to varying setups. The view-agnostic design makes the model "plug-and-play".

### Loss & Training

The total loss is the sum of six terms: $\mathcal{L} = \lambda_{3D}\mathcal{L}_{3D} + \lambda_{2D}\mathcal{L}_{2D} + \mathcal{L}_{smpl} + \mathcal{L}_{hm} + \lambda_v\mathcal{L}_v + \lambda_{adv}\mathcal{L}_{adv}$, where $\mathcal{L}_{3D}$ and $\mathcal{L}_{2D}$ are the 3D/2D joint MSE losses, $\mathcal{L}_{smpl}$ is the SMPL parameter supervision, $\mathcal{L}_{hm}$ is the step-weighted heatmap loss (with weights increasing with iterations, e.g., $[0.001, 0.003, 0.005]$), $\mathcal{L}_v$ is the vertex loss (to prevent body distortion), and $\mathcal{L}_{adv}$ is the adversarial prior loss. First, the HeatEncoder is pre-trained on a single view, and then the entire model is trained while freezing HeatEncoder and ViT.

## Key Experimental Results

### Main Results

| Method | Type | Human3.6M MPJPE ↓ | Human3.6M PA-MPJPE ↓ | MPI-INF-3DHP MPJPE ↓ | MPI-INF-3DHP PCK ↑ |
|------|------|-------------------|----------------------|----------------------|-------------------|
| HMR2.0 | Monocular | 44.8 | 33.6 | - | - |
| PaFF | Multi-view | 33.0 | 26.9 | 48.4 | 98.6 |
| U-HMR | Multi-view | 31.0 | 22.8 | - | - |
| **HeatFormer (iter3)** | Multi-view | **30.7** | 23.3 | **39.8** | **99.5** |
| **HeatFormer (iter4)** | Multi-view | **29.5** | **22.4** | 40.6 | **99.5** |

### Ablation Study

| Configuration | Human3.6M MPJPE ↓ | PA-MPJPE ↓ | MPVPE ↓ |
|------|-------------------|-----------|---------|
| iter1 | 34.9 | 26.2 | 41.9 |
| iter2 | 31.2 | 23.6 | 37.5 |
| iter3 | 30.7 | 23.3 | 37.0 |

**Cross-Dataset Generalization (BEHAVE Occlusion Dataset):**

| Method | Protocol 1 MPJPE ↓ | Protocol 2 MPJPE ↓ |
|------|-------------------|-------------------|
| HMR2.0a | 72.2 | 48.1 |
| HMR2.0a+scoreHMR | 72.9 | 49.0 |
| **HeatFormer (iter3)** | **51.1** | **34.2** |

### Key Findings

- The MPJPE decreases from 34.9 to 30.7 as the number of iterations increases from 1 to 3, indicating that the iterative refinement of the neural optimizer is highly effective and converges rapidly.
- On the BEHAVE occlusion dataset, HeatFormer significantly outperforms the monocular method HMR2.0 (despite the latter having more pre-training data), proving that multi-view utilization is essential for occluded scenarios.
- On cross-dataset evaluation (trained only on Human3.6M and tested on MPI-INF-3DHP), HeatFormer outperforms U-HMR by a large margin (56.0 vs 73.2 MPJPE), suggesting that U-HMR overfits to specific datasets while the neural optimization formulation brings better generalization.

## Highlights & Insights

- **Heatmaps as Intermediate Representations for Optimization**: Traditional methods use keypoint coordinates for 2D-3D alignment, which suffer from discontinuous gradients. Replacing them with heatmaps provides smooth spatial gradients, enabling end-to-end neural optimization. This "back-to-basics" approach is highly ingenious.
- **Unrolled Optimization as Repeated Inference**: Equating multiple forward passes of the Transformer decoder to iterative optimization steps, where each step refers back to the original input (via cross-attention), is an elegant way to integrate optimization into a feed-forward network. This concept can be transferred to other tasks requiring iterative alignment.
- **View-Agnostic Design**: Outputting view-dependent estimates rather than a global estimate leverages the flexibility of Transformers regarding token sequence length, achieving true plug-and-play deployment.

## Limitations & Future Work

- Temporal information is not utilized, failing to fully exploit the constraints of consecutive frames in video surveillance scenarios.
- Relies on AdaFuse for multi-view-aware 2D joint detection as a preprocessing step, the accuracy of which affects the subsequent optimization.
- Training data mainly comes from Human3.6M and MPI-INF-3DHP, offering limited scene diversity.
- Requires 3–4 decoder forward passes during inference, which demands higher computational cost compared to single-pass feed-forward methods.

## Related Work & Insights

- **vs SPIN**: SPIN also combines regression and optimization, but its optimization phase uses traditional SMPLify and is limited to monocular input. HeatFormer internalizes optimization into neural network inference and natively supports multi-view.
- **vs PyMAF**: PyMAF also performs iterative alignment feedback within the network, but is based on feature-pixel alignment rather than heatmaps and does not support multi-view.
- **vs U-HMR**: U-HMR uses a Transformer to aggregate multi-view features but employs a feed-forward architecture, showing clear overfitting in cross-dataset evaluations.

## Rating

- Novelty: ⭐⭐⭐⭐ The first neural optimizer for HMR, with a novel heatmap alignment formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, with evaluations on four datasets, occlusion analysis, cross-dataset generalization, and analysis of the number of views.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation derivation and methodology description.
- Value: ⭐⭐⭐⭐ Direct value for multi-camera applications with fixed setups, such as eldercare and security monitoring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](prompthmr_promptable_human_mesh_recovery.md)
- [\[CVPR 2025\] MEGA: Masked Generative Autoencoder for Human Mesh Recovery](mega_masked_generative_autoencoder_for_human_mesh_recovery.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](../../ICCV2025/3d_vision/ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](../../ICCV2025/3d_vision/fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](../../ECCV2024/3d_vision/global-to-pixel_regression_for_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
