---
title: >-
  [Paper Note] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling
description: >-
  [ICCV 2025][3D Vision][Point cloud video] This paper proposes UST-SSM, which extends selective state space models to point cloud video analysis via three core modules — Spatio-Temporal Selective Scanning (STSS), Spatio-Temporal Structure Aggregation (STSA), and Temporal Interaction Sampling (TIS) — achieving linear complexity while surpassing Transformer-based methods.
tags:
  - ICCV 2025
  - 3D Vision
  - Point cloud video
  - state space models
  - spatio-temporal modeling
  - action recognition
  - Mamba
date: 2026-05-08
content_hash: 189bd2a2178b496e
---

# UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling

**Conference**: ICCV 2025
**arXiv**: [2508.14604](https://arxiv.org/abs/2508.14604)
**Code**: [GitHub](https://github.com/wangzy01/UST-SSM)
**Area**: 3D Vision
**Keywords**: Point cloud video, state space models, spatio-temporal modeling, action recognition, Mamba

## TL;DR

This paper proposes UST-SSM, which extends selective state space models to point cloud video analysis via three core modules — Spatio-Temporal Selective Scanning (STSS), Spatio-Temporal Structure Aggregation (STSA), and Temporal Interaction Sampling (TIS) — achieving linear complexity while surpassing Transformer-based methods.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Point cloud video modeling faces three compounding challenges:

**Spatio-temporal disorder**: Point cloud videos lack consistent spatio-temporal ordering, conflicting with the unidirectional scanning paradigm of SSMs.

**Loss of local geometry**: Serialization strategies disrupt geometric relationships among neighboring points.

**Limited temporal interaction**: Conventional sampling strategies produce fragmented temporal contexts.

Shortcomings of existing approaches:

### State of the Field

**State of the Field**: Mamba4D employs sequential temporal scanning, restricting each query point to accessing information from only the immediately preceding frame.

### Root Cause

**Root Cause**: SSMs suffer from attenuation in long-range token interactions.

### Starting Point

**Starting Point**: Points that are semantically similar but spatially and temporally distant are placed far apart in the sequence.

## Method

### Temporal Interaction Sampling (TIS)

In conventional single-stride sampling (stride=2), anchor frames interact only with their direct neighbors. TIS extends the temporal receptive field via two-step sampling:

Step 1 (stride=1):
$$\mathbf{Feat}_{T_i}' = \mathcal{F}(\mathcal{S}(T_{i-1}, T_i, T_{i+1}))$$

Step 2 (stride=2):
$$\mathbf{Feat}_{T_{2i}} = \bar{\mathcal{F}}(\bar{\mathcal{S}}(\mathbf{Feat}_{T_{2i-1}}', \mathbf{Feat}_{T_{2i}}, \mathbf{Feat}_{T_{2i+1}}'))$$

Non-anchor frames are reused multiple times, allowing each anchor frame to access information from all preceding frames.

### Spatio-Temporal Selective Scanning (STSS)

Unlike sequential temporal scanning, STSS clusters points by semantic similarity:

1. A lightweight Prompt network generates a classification matrix $\mathcal{P} \in \mathbb{R}^{N \times K}$.
2. Points are clustered by semantic category: $\mathbf{X}_j = \{x_i | \arg\max(p_i) = j\}$.
3. Intra-cluster Hilbert sorting preserves local geometry: $\bar{\mathbf{X}}_j = \text{HilbertSort}(\mathbf{X}_j)$.
4. Inter-cluster ordering follows temporal sequence to maintain motion continuity.

### Spatio-Temporal Structure Aggregation (STSA)

STSA recovers local geometric relationships disrupted by serialization via 4D KNN:

**4D neighborhood construction**: A temporal embedding $\mathbf{E}^t$ is incorporated:
$$K = \underset{\mathbf{X}_K \in X}{\text{argmin}}(|\mathbf{X}_C - \mathbf{X}_K| + |\mathbf{E}_C^t - \mathbf{E}_K^t|)$$

**Feature propagation**: Neighbor features are normalized and concatenated with absolute features:
$$\mathbf{F}_K' = \frac{\mathbf{F}_K - \mathbf{F}_C}{|\mathbf{F}_K - \mathbf{F}_C|_2 + \epsilon} \oplus \mathbf{F}_C$$

**Adaptive pooling**: Exponential weighted pooling replaces standard convolution:
$$\mathbf{F}_C' = \text{MLP}\left(\sum_{i=1}^K \frac{e^{\mathbf{F}_K'^i}}{\sum_j e^{\mathbf{F}_K'^j}} \cdot \mathbf{F}_K'^i\right)$$

## Key Experimental Results

### MSR-Action3D

### Main Results

| Backbone | Method | 24-frame Acc (%) | 36-frame Acc (%) |
|----------|--------|-----------------|-----------------|
| CNN | MeteorNet | 88.50 | - |
| CNN | Kinet | 93.27 | - |
| Transformer | PST-Transformer | 93.73 | 91.15 |
| Transformer | LeaF | 93.84 | - |
| SSM | MAMBA4D | 92.68 | 93.23 |
| SSM | **UST-SSM** | **94.77** | **95.12** |

UST-SSM achieves state-of-the-art results across all frame-count settings, with performance improving consistently as frame count increases (36-frame best: 95.12%).

### NTU RGB+D

### Ablation Study

| Method | Cross-Subject | Cross-View |
|--------|--------------|------------|
| P4Transformer | 90.2 | 96.4 |
| PST-Transformer | 91.0 | 96.4 |
| **UST-SSM** | SOTA | SOTA |

### Efficiency Comparison

With linear complexity, UST-SSM outperforms P4Transformer and PST-Transformer (quadratic complexity) in runtime, GPU memory consumption, and accuracy.

## Highlights & Insights

1. **STSS addresses the root contradiction**: Semantic clustering brings spatiotemporally distant but semantically similar points closer in sequence, mitigating long-range attenuation in unidirectional modeling.
2. **Non-anchor frame reuse in TIS**: Elegantly expands the temporal receptive field without increasing sampling complexity.
3. **STSA complements SSM**: The former handles local interactions while the latter captures global sequential dependencies.
4. **Hilbert sorting preserves geometry**: Maintains local spatial coherence even after semantic clustering.

## Limitations & Future Work

- The Prompt network introduces additional computational overhead.
- The number of clusters $K$ is a hyperparameter that may require tuning across datasets.
- Temporal embeddings in 4D KNN require additional learning.
- Validation on large-scale autonomous driving point cloud datasets has not been conducted.

## Related Work & Insights

- P4Transformer, PST-Transformer: Transformer-based point cloud video methods.
- Mamba4D: The first work to apply SSMs to point cloud video.
- PointMamba, PCM: SSMs applied to 3D point clouds (spatial only).

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The STSS semantic-clustering scan strategy is novel)
- **Technical Depth**: ⭐⭐⭐⭐⭐ (Each of the three modules addresses a distinct core problem)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Three datasets with efficiency analysis)
- **Value**: ⭐⭐⭐⭐ (Linear complexity is well-suited for long-horizon modeling)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Global-Aware Monocular Semantic Scene Completion with State Space Models](global-aware_monocular_semantic_scene_completion_with_state_space_models.md)
- [\[ICCV 2025\] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction](meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)
- [\[CVPR 2026\] STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding](../../CVPR2026/3d_vision/sts_mixer_4d_point_cloud.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)

<!-- RELATED:END -->
