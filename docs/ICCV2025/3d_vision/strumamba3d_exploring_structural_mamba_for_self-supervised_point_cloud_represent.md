---
title: >-
  [Paper Note] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning
description: >-
  [ICCV 2025][3D Vision][Point cloud representation learning] StruMamba3D is proposed to maintain 3D point adjacency relationships by endowing SSM hidden states with spatial positional attributes (spatial states)…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Point cloud representation learning"
  - "state space models"
  - "Mamba"
  - "self-supervised learning"
  - "structural modeling"
date: 2026-05-08
content_hash: 801fa0720ef7c537
---

# StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning

**Conference**: ICCV 2025
**arXiv**: [2506.21541](https://arxiv.org/abs/2506.21541)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: Point cloud representation learning, state space models, Mamba, self-supervised learning, structural modeling

## TL;DR

StruMamba3D is proposed to maintain 3D point adjacency relationships by endowing SSM hidden states with spatial positional attributes (spatial states), and introduces a sequence-length-adaptive strategy to address the sequence length discrepancy between pre-training and downstream tasks. The method achieves 92.75% accuracy on the hardest ScanObjectNN split and 95.1% on ModelNet40, both representing single-modality SOTA.

## Background & Motivation

### Problem Definition

Point cloud representation learning aims to extract geometric and semantic features from unordered, sparse point clouds to support downstream tasks such as classification, segmentation, and detection. Self-supervised pre-training methods learn general representations from large amounts of unlabeled data, significantly improving downstream performance.

### Limitations of Prior Work

Existing Mamba-based point cloud methods (PointMamba, Mamba3D, PCM) suffer from two fundamental problems:

**Destruction of 3D point adjacency**: When serializing 3D points into a 1D sequence, spatially adjacent points are not necessarily adjacent in the sequence. Unlike text and images, point clouds **inherently lack contextual information** and rely on local structure as the basic feature unit. The spatial distortion introduced by serialization hinders Mamba's ability to model fine-grained structural information.

**Insufficient long-sequence memory in downstream tasks**: Mamba-based methods use masked point modeling (MPM) for pre-training with short input sequences, while downstream tasks use complete sequences (which are longer). Mamba's selection mechanism learns relatively high-frequency state updates during pre-training; when sequence length increases, frequent updates make it difficult for the model to maintain long-range memory, degrading long-range semantic modeling.

### Root Cause

**Core Problem**: Can SSM hidden states be leveraged to model spatial relationships among 3D points? If positional attributes are assigned to hidden states so that they serve as proxies for local structure, spatial dependencies between points can be maintained during SSM processing. Furthermore, if the model can adjust state update frequency according to sequence length, long-range memory can be preserved over longer inputs.

## Method

### Overall Architecture

Given a raw point cloud $P_{raw} \in \mathbb{R}^{N_0 \times 3}$, input points $P_x$ are obtained via FPS and KNN, and token embeddings $F_x \in \mathbb{R}^{N \times D}$ are extracted using PointNet. Spatial states $P_h \in \mathbb{R}^{M \times 3}$ ($M=16$) are simultaneously initialized. Input tokens and spatial states are processed through multiple structural SSM blocks, with spatial states propagated across blocks in a cascaded manner. Pre-training employs an MPM task combined with a spatial state consistency loss.

### Key Designs

#### 1. **Spatial State Initialization and State-Level Update**

- **Function**: Endows SSM hidden states with 3D positional attributes, enabling them to represent local structural regions of the point cloud.
- **Mechanism**:

  **Spatial state initialization**: The raw point cloud is partitioned into $M$ groups $\{\mathcal{G}_m\}_{m=1}^M$ via FPS and KNN, and centroids are computed as state positions:
  $$P_h^m = \frac{1}{|\mathcal{G}_m|} \sum_{P_i \in \mathcal{G}_m} P_i$$
  The positions are then encoded into state features via a linear mapping: $F_h = \phi_h(P_h) \in \mathbb{R}^{M \times D}$.

  **State-level update**: The SSM state update equation is modified to explicitly model the spatial relationship between input points and spatial states. The relative offset $\triangle P_i^m = P_x^i - P_h^m$ is first computed, and offset information is then incorporated into SSM parameters:
  $$(\mathbf{B}_i^m, \mathbf{C}_i^m) = \phi(x_i) + \text{MLP}(\triangle P_i^m)$$
  This allows spatial states to selectively update features using points within the same region, while input points can retrieve local structural information from the states.

- **Design Motivation**: The original Mamba hidden states contain no geometric information and cannot model local structure in point clouds. By assigning positional attributes, each state is made responsible for a specific spatial region, and spatial relationships are incorporated into SSM parameter generation, realizing "structure-aware" state selection and propagation.

#### 2. **Structural SSM Block**

- **Function**: A complete structural SSM processing unit comprising bidirectional scanning and lightweight convolution.
- **Mechanism**:

  **Bidirectional structural SSM**: Forward and backward structural SSMs share spatial state $\hat{F}_h$ as the initial state, and their outputs are fused via a linear layer:
  $$F'_x, F'_h = \phi_o(\text{SSM}_f(\hat{F}_x, \hat{F}_h) + \text{SSM}_b(\hat{F}_x, \hat{F}_h))$$

  **Lightweight convolution module**: Addresses the lack of direct interaction among spatial states. For each spatial state $P_h^m$, its $k$ nearest neighbors $\mathcal{N}(m)$ are found, attention weights are generated from relative positions, and neighbor features are aggregated:
  $$w_h^{mj} = \text{softmax}(\phi_w(\triangle P_h^{mj}, P_h^m))$$
  $$\hat{F}_h^m = \phi_c(\sum_{j \in \mathcal{N}(m)} w_h^{mj} F_h^{mj})$$
  Lightweight convolution is also applied to input points, replacing the causal 1D convolution in standard Mamba.

- **Design Motivation**: Unidirectional scanning cannot achieve bidirectional information exchange in a single forward pass; the bidirectional mechanism compensates for this. Spatial states are isolated in standard SSM, and lightweight convolution expands their receptive field to capture global semantics.

#### 3. **Sequence-Length-Adaptive Strategy**

- **Function**: Addresses the sequence length discrepancy between pre-training (short sequences) and downstream tasks (long sequences).
- **Mechanism**:

  **Adaptive state update mechanism**: A learnable parameter $\tau$ is introduced to regulate total sampling time, maintaining a consistent total sampling time $\Delta_{all} = \tau$ across different sequence lengths:
  $$\Delta_i = \frac{\tau \times \Delta_i}{\sum_{i=1}^N \Delta_i}$$

  **Spatial state consistency loss**: A teacher–student framework is adopted. The teacher model updates spatial states using complete tokens (output $F_h'^f$), while the student model uses only visible tokens (output $F_h'^v$), with consistency enforced as:
  $$\mathcal{L}_{ssc} = \text{Smooth L1}(F_h'^v, F_h'^f)$$
  The total pre-training loss is: $\mathcal{L}_{total} = \mathcal{L}_{cd} + \lambda \times \mathcal{L}_{ssc}$

- **Design Motivation**: Since $\Delta$ controls state update frequency, longer sequences require lower update frequency to maintain long-range memory. The state consistency loss further ensures that the model can infer complete structure even when only a subset of points is observed.

### Loss & Training

- **Pre-training loss**: $\mathcal{L}_{total} = \mathcal{L}_{cd} + 2 \times \mathcal{L}_{ssc}$
- Pre-trained on ShapeNet: 52,472 3D models across 55 categories
- 1,024 points per point cloud, partitioned into 64 patches (32 points each), masking ratio 0.6
- 12 structural SSM blocks, feature dimension 384, number of spatial states $M=16$
- Teacher model updated via EMA

## Key Experimental Results

### Main Results

**ScanObjectNN Classification** (real-world dataset, hardest split PB-T50-RS):

| Method | Backbone | Params (M) | OBJ-BG | OBJ-ONLY | PB-T50-RS |
|--------|----------|------------|--------|----------|-----------|
| PointMAE† | Transformer | 22.1 | 92.77 | 91.22 | 89.04 |
| PointGPT-S† | Transformer | 29.2 | 93.39 | 92.43 | 89.17 |
| PointMamba† | Mamba | 12.3 | 94.32 | 92.60 | 89.31 |
| Mamba3D† | Mamba | 16.9 | 93.12 | 92.08 | 92.05 |
| **StruMamba3D†** | **Structural SSM** | **15.8** | **95.18** | **93.63** | **92.75** |

**ModelNet40 Classification**:

| Method | w/o Voting | w/ Voting |
|--------|-----------|----------|
| Mamba3D† | 94.7 | 95.1 |
| **StruMamba3D†** | **95.1** | **95.4** |

**ShapeNetPart Part Segmentation** (single-scale model):

| Method | mIoU_c | mIoU_i |
|--------|--------|--------|
| PointMamba | 84.4 | 86.2 |
| Mamba3D | 83.6 | 85.6 |
| **StruMamba3D** | **85.0** | **86.7** |

### Ablation Study

**Per-module contribution**:

| Structural SSM Block | Length-Adaptive Strategy | ScanObjectNN | ModelNet40 | ShapeNetPart (mIoU_c) |
|---------------------|------------------------|-------------|-----------|----------------------|
| ✗ | ✗ | 87.23 | 91.86 | 81.56 |
| ✓ | ✗ | 92.09 | 94.45 | 84.49 |
| ✓ | ✓ | **92.75** | **95.06** | **84.96** |

**Internal components of the structural SSM**:

| Method | ScanNN | MN40 | SNPart |
|--------|--------|------|--------|
| Baseline (standard Mamba) | 88.24 | 92.50 | 82.08 |
| + Structural SSM | 91.78 | 93.92 | 84.15 |
| + Spatial state lightweight conv. | 92.22 | 94.65 | 84.62 |
| + Input point lightweight conv. | 92.40 | 94.81 | 84.77 |
| + Bidirectional scanning | **92.75** | **95.06** | **84.96** |

**Ablation on SSM parameter sources**:

| $\phi(x)$ | Spatial State | $\text{MLP}(\triangle P)$ | ScanNN | MN40 |
|-----------|--------------|--------------------------|--------|------|
| ✓ | ✗ | ✗ | 90.94 | 93.84 |
| ✓ | ✓ | ✗ | 91.33 | 94.12 |
| ✓ | ✓ | ✓ | **92.75** | **95.06** |
| ✗ | ✓ | ✓ | 91.12 | 94.25 |

### Key Findings

1. **Structural SSM block contributes the most**: Compared to the standard Mamba baseline, it yields a 4.86% gain on ScanObjectNN, demonstrating the effectiveness of modeling structural information through hidden states.
2. **Spatial relationships and input features are both indispensable**: Using spatial relationships alone (91.12%) or input features alone (90.94%) is inferior to their combination (92.75%).
3. **Two components of the length-adaptive strategy are complementary**: Using the adaptive update mechanism or the consistency loss in isolation yields only marginal improvement; combining them produces a significant effect.
4. **Outperforms cross-modal methods**: StruMamba3D surpasses most methods that leverage cross-modal (image + text) information using single-modality information only.
5. **Mamba3D fails on part segmentation**: Without a serialization strategy, Mamba3D even underperforms PointMAE (83.6 vs. better mIoU_c), validating the critical importance of structural information for fine-grained tasks.

## Highlights & Insights

1. **Original core insight**: The SSM hidden state is recast from a "black-box memory" to a "spatial proxy" by assigning positional attributes to model 3D structure — this represents the first attempt to address point cloud structural modeling from a state-space perspective.
2. **Resolves the fundamental tension in Mamba-based point cloud methods**: Serialization and 3D adjacency preservation are mutually exclusive; spatial states provide a viable third path.
3. **Insightful length-adaptive strategy**: The method identifies the impact of the pre-training–downstream sequence length discrepancy on Mamba's selection mechanism, a problem overlooked in prior work.
4. **Elegant lightweight convolution design**: Inspired by graph convolution, it enables interaction among spatial states while preserving the linear complexity of SSM and enhancing global perception.

## Limitations & Future Work

1. **Fixed number of spatial states**: $M=16$ is set manually and may not be suitable for all scenarios; adaptively determining the number of states is a promising direction.
2. **Only single-scale architecture validated**: Multi-scale designs (e.g., PointM2AE) may further improve performance on fine-grained tasks such as part segmentation.
3. **Larger-scale models not explored**: The current parameter count is 15.8M; the effect of scaling to larger models remains unknown.
4. **Spatial state positions are fixed**: State positions are determined by preprocessing and are not updated across layers — learnable positional updates could offer greater flexibility.

## Related Work & Insights

- **Relation to PointMamba**: PointMamba uses space-filling curves for serialization but still disrupts adjacency; StruMamba3D circumvents this issue through spatial states.
- **Comparison with Mamba3D**: Mamba3D employs bidirectional scanning and local norm pooling but performs poorly on segmentation tasks (83.6 mIoU_c), whereas StruMamba3D (85.0) achieves a substantial improvement.
- The graph convolution idea from DGCNN is incorporated into the lightweight convolution module design.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to address point cloud structural modeling from an SSM state-space perspective; the spatial state concept is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four downstream tasks with comprehensive ablations; every design choice is empirically validated.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem analysis is clear, though the dense mathematical notation increases reading difficulty in certain sections.
- **Value**: ⭐⭐⭐⭐ — Provides an important structural modeling paradigm for the application of Mamba in the 3D domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] S3E: Self-Supervised State Estimation for Radar-Inertial System](s3e_self-supervised_state_estimation_for_radar-inertial_system.md)
- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)

</div>

<!-- RELATED:END -->
