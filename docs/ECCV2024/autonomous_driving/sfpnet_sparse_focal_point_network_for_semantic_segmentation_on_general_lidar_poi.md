---
title: >-
  [Paper Note] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds
description: >-
  [ECCV 2024][Autonomous Driving][LiDAR Semantic Segmentation] SFPNet proposes Sparse Focal Modulation (SFPM) to replace window-attention. By avoiding inductive bias designs targeted at specific LiDAR types through multi-level context extraction and gated adaptive aggregation, it achieves leading or competitive performance on mechanical spinning, solid-state, and hybrid solid-state LiDAR datasets. It also releases S.MID, the first hybrid solid-state LiDAR semantic segmentation…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "LiDAR Semantic Segmentation"
  - "Sparse Focal Modulation"
  - "General LiDAR"
  - "Inductive Bias"
  - "Cross-LiDAR Generalization"
date: 2026-05-08
content_hash: 63cf3f212946873c
---

# SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds

**Conference**: ECCV 2024  
**arXiv**: [2407.11569](https://arxiv.org/abs/2407.11569)  
**Code**: [https://github.com/Cavendish518/SFPNet](https://github.com/Cavendish518/SFPNet)  
**Area**: Autonomous Driving  
**Keywords**: LiDAR Semantic Segmentation, Sparse Focal Modulation, General LiDAR, Inductive Bias, Cross-LiDAR Generalization

## TL;DR

SFPNet proposes Sparse Focal Modulation (SFPM) to replace window-attention. By avoiding inductive bias designs targeted at specific LiDAR types through multi-level context extraction and gated adaptive aggregation, it achieves leading or competitive performance on mechanical spinning, solid-state, and hybrid solid-state LiDAR datasets. It also releases S.MID, the first hybrid solid-state LiDAR semantic segmentation dataset.

## Background & Motivation

LiDAR semantic segmentation methods usually design inductive biases for specific types of LiDAR. For example, Cylinder3D designs cylindrical partitions for mechanical spinning LiDAR, and SphereFormer designs radial window attention. These methods perform excellently on corresponding LiDAR data, but the core problem is that diverse LiDAR technologies (mechanical spinning, solid-state, hybrid solid-state) co-exist in the market. Their point cloud distributions vary significantly, and inductive biases tailored to a specific distribution generalize poorly to other LiDAR types.

According to the No Free Lunch theorem, specialized designs inevitably sacrifice generality. The core idea of this paper is to replace window-attention with sparse focal modulation, leveraging adaptive multi-level context aggregation to adapt to point cloud distributions of different LiDAR types without introducing specific spatial partitioning or window shape assumptions.

## Method

### Overall Architecture

SFPNet takes sparse voxels and point attributes as input, using submanifold sparse convolution as the backbone (UNet architecture), and replaces the window-attention module with the SFPM module. The overall pipeline remains simple: MLP projection -> multi-level context extraction -> gated aggregation -> channel-wise query -> output feature.

### Key Designs

1. **Sparse Focal Modulation (SFPM)**: Borrowing from the Focal Modulation paradigm, the operation is decomposed into three steps: first aggregating multi-level context $\kappa_{focal}$, and then performing channel-wise information query $\xi_{focal}$. The formula is $y_i = \xi_{focal}(x_i, \kappa_{focal}(i, X))$. The critical difference from window-attention is that attention calculates the query-key interactions before aggregation, whereas SFPM aggregates context before the query. This design possesses both the translation invariance of submanifold sparse convolution and the long-range dependency modeling capability of attention.

2. **Multi-Level Context Extraction**: For the input feature sequence, $L$ submanifold sparse convolution layers are sequentially applied, with the kernel size increasing as $k^l = k^{l-1} + 2$, and the effective receptive field being $RF^l = 1 + \sum_{i=1}^{l}(k^l - 1)$, expanding layer-by-layer from local to global. Finally, global average pooling is used to obtain the global context. The formula is:
   $S^l = LN(GeLU(SubMconv_{3d}^l(S^{l-1})))$

3. **Gated Adaptive Aggregation**: The importance of different levels of context varies per point. A gate mechanism $G = MLP(X)$ computes spatial-aware weights for $L+1$ channels, which are then aggregated via weighted summation: $S^{out} = \sum_{l=1}^{L+1} G^l \odot S^l$, followed by 1×1×1 SubMconv for cross-channel aggregation. The gating mechanism allows the model to adaptively learn the context levels required for difficult tokens, avoiding the introduction of excessive invalid information for simple tokens. The core advantage is that it does not require designing special window shapes or partitioning strategies for specific point cloud distributions.

4. **Channel-Level Information Query**: Completed through query projection $q(x_i) = MLP(X)$ and element-wise multiplication: $y_i = q(x_i) \odot h(\sum_{l=1}^{L+1} g_i^l \cdot s_i^l)$. The lightweight element-wise multiplication preserves channel-level information without computing an attention matrix.

### Loss & Training

Uses standard semantic segmentation cross-entropy loss and Lovász-Softmax loss. AdamW optimizer, lr=0.0008, polynomial learning rate schedule. 2×RTX 3090 GPUs (4 GPUs for SemanticKITTI), batch size 8, trained for 70 epochs. No special data augmentation, distillation, or post-processing techniques are used, focusing on demonstrating the representation capability of the network design itself.

### S.MID Dataset

The first semantic segmentation dataset based on hybrid solid-state LiDAR (Livox Mid-360), collected from substation industrial scenes. It contains 38,904 frames of data, and 25 annotated categories are merged into 14 categories. The training set has 13,101 frames, the validation set has 5,000 frames, and the test set has 20,803 frames, from different substations.

## Key Experimental Results

### Main Results

Comparison of backbone-level methods across four LiDAR types:

| Method | nuScenes (val) | SemanticKITTI (test) | PandaSet (val) | S.MID (val) |
|------|----------------|---------------------|----------------|-------------|
| Cylinder3D | 76.1 | 68.9 | 55.0 | 68.8 |
| SphereFormer | 79.5 | 74.8 | 63.5 | 67.8 |
| **SFPNet** | **80.1** | 70.3 | **64.0** | **71.9** |

nuScenes val detailed comparison (vs all LiDAR methods, published before March 2024):

| Method | Modality | mIoU |
|------|------|------|
| SphereFormer | L | 79.5 |
| 2DPASS | L(C) | 79.4 |
| 2D3DNet | L+C | 79.0 |
| **SFPNet** | **L** | **80.1** |

PandaSet (solid-state LiDAR) comparison:

| Method | mIoU | Description |
|------|------|------|
| Cylinder3D | 55.0 | Cylindrical partition is not applicable |
| SphereFormer | 63.5 | Radial window has some adaptability |
| **SFPNet** | **64.0** | Adapts well even without special biases |

S.MID (hybrid solid-state LiDAR) comparison:

| Method | mIoU | vs baseline SSCN |
|------|------|------------------|
| SSCN baseline | 67.6 | - |
| Cylinder3D | 68.8 | +1.2 |
| SphereFormer | 67.8 | +0.2 |
| **SFPNet** | **71.9** | **+4.3** |

### Ablation Study

| Configuration | nuScenes val mIoU | Description |
|------|-------------------|------|
| SSCN baseline | 76.1 | Without SFPM |
| + Single-layer SubMconv | - | Only local context |
| + Multi-level context (L=3) | - | Hierarchical receptive fields |
| + Gated aggregation | - | Adaptive weights |
| **SFPNet (Full)** | **80.1** | All modules |

The comparison of SFPM vs window-attention on different LiDAR types is the most convincing experiment. On S.MID, methods with specialized biases (Cylinder3D, SphereFormer) only improve by 0.2-1.2% over the baseline, whereas SFPNet achieves a 4.3% improvement, proving that a general design offers significant advantages under distribution shifts.

### Key Findings

- **Limitations of Specific Inductive Biases**: Cylinder3D's cylindrical partitioning is almost ineffective on non-mechanical spinning LiDAR (only +1.2% on S.MID, and only 55.0% on PandaSet); SphereFormer's radial window only gains +0.2% on S.MID.
- **Generality Advantage**: SFPNet achieves competitive or state-of-the-art results across four different LiDAR datasets without requiring specialized spatial partitioning.
- **SFPM Combines Both Advantages**: It possesses both the translation invariance of SubMconv (requiring no positional encoding) and the long-range dependency learning capability of attention.
- On nuScenes, the single-modal method (80.1%) surpasses multi-modal methods like 2D3DNet (79.0%, L+C).

## Highlights & Insights

- **No Free Lunch Perspective on LiDAR Segmentation**: Systematically examines existing methods from the lens of inductive bias, and the classification framework in Table 1 is very clear.
- **Convincing Experimental Design**: Concurrent verification on three LiDAR types: mechanical spinning (nuScenes, SemanticKITTI), solid-state (PandaSet), and hybrid solid-state (S.MID), serving as the strongest argument for generalization.
- **Contribution of a New Dataset**: S.MID fills the dataset gap for industrial scenes and hybrid solid-state LiDAR.
- **Extension of Focal Modulation to 3D Sparse Data**: Successfully ports the 2D visual focal modulation paradigm to 3D sparse voxels, demonstrating the feasibility of paradigm transfer.

## Limitations & Future Work

- On the SemanticKITTI test, the mIoU is only 70.3, falling behind SphereFormer (74.8), illustrating that abandoning specialized biases on mechanical spinning LiDAR indeed comes with a cost.
- Stacking multi-level SubMconv may increase computational overhead, especially as kernel sizes increase.
- No comparison has been made with the latest multi-frame/temporal methods.
- The S.MID dataset currently only supports single-frame segmentation tasks, with scenes limited to substations.
- Detailed comparison of inference latency and model parameters is missing.

## Related Work & Insights

- **Focal Modulation Networks (Yang et al.)**: Direct inspiration for SFPM, extending 2D focal modulation to 3D sparse voxels.
- **Cylinder3D**: Representative of cylindrical partitions, and the main baseline for comparison.
- **SphereFormer**: Representative of radial window attention, possessing advantages in long-range point perception.
- **Insight**: Methods pursuing generality perform more robustly when data distribution changes, which is particularly critical for practical application scenarios involving diverse sensor deployments.

## Rating

- Novelty: ⭐⭐⭐⭐ — SFPM is a natural extension of focal modulation to 3D sparse data; the idea is novel but not revolutionary.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive validation across four LiDAR types, contribution of a new dataset, and rigorous experimental design.
- Writing Quality: ⭐⭐⭐⭐ — The analytical framework for inductive bias is clear, though the method description is slightly mathematical.
- Value: ⭐⭐⭐⭐ — The practical demand for general LiDAR segmentation is clear, and the new dataset is of lasting value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)
- [\[ECCV 2024\] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](../../CVPR2025/autonomous_driving/reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[ECCV 2024\] RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion](roofdiffusion_constructing_roofs_from_severely_corrupted_point_data_via_diffusio.md)
- [\[ECCV 2024\] RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation](rapid-seg_range-aware_pointwise_distance_distribution_networks_for_3d_lidar_segm.md)

</div>

<!-- RELATED:END -->
