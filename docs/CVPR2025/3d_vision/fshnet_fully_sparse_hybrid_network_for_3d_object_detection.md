---
title: >-
  [Paper Note] FSHNet: Fully Sparse Hybrid Network for 3D Object Detection
description: >-
  [CVPR 2025][3D Vision][3D object detection] FSHNet proposes a fully sparse hybrid network that establishes global-range sparse voxel interactions using SlotFormer (slot partitioning + linear attention). Together with dynamic sparse label assignment and a sparse upsampling module, it outperforms existing sparse and dense detectors on three major benchmarks: Waymo, nuScenes, and Argoverse2.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D object detection"
  - "sparse detector"
  - "linear attention"
  - "dynamic label assignment"
  - "sparse upsampling"
date: 2026-05-08
content_hash: f3fb511c7ced63f4
---

# FSHNet: Fully Sparse Hybrid Network for 3D Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2506.03714](https://arxiv.org/abs/2506.03714)  
**Code**: [https://github.com/Say2L/FSHNet](https://github.com/Say2L/FSHNet)  
**Area**: 3D Vision  
**Keywords**: 3D object detection, sparse detector, linear attention, dynamic label assignment, sparse upsampling  

## TL;DR
FSHNet proposes a fully sparse hybrid network that establishes global-range sparse voxel interactions using SlotFormer (slot partitioning + linear attention). Together with dynamic sparse label assignment and a sparse upsampling module, it outperforms existing sparse and dense detectors on three major benchmarks: Waymo, nuScenes, and Argoverse2.

## Background & Motivation

**Background**: LiDAR-based 3D object detection is divided into dense detectors (converting sparse features to 2D dense BEV, then using 2D detection heads) and sparse detectors (processing only non-empty voxels throughout). Dense detectors waste massive computational resources processing empty voxels, which is particularly severe in long-range detection. Sparse detectors (FSD, VoxelNeXt, SAFDNet) are more efficient but still suffer from a performance gap.

**Limitations of Prior Work**: Sparse detectors face two core challenges: (1) **Difficulty in long-range interactions**: Sparse convolutions only operate on non-empty voxels. If there are no non-empty voxels between two distant voxels, they cannot interact at all, unlike dense detectors which can propagate information through intermediate empty voxels; (2) **Missing center features**: The centers of objects are often empty (lacking point cloud returns), especially for large objects, making it difficult for the network to optimize effectively, as many methods rely on center features as object proxies.

**Key Challenge**: Simply increasing kernel sizes expands the receptive field but incurs heavy computational overhead. Transformers enable long-range interactions, but directly processing raw point clouds involves too many points. None of the existing sparse detectors offer a standardized solution that simultaneously addresses weak feature extraction capabilities and optimization difficulties.

**Goal**: (1) To equip sparse detectors with efficient global interaction capabilities; (2) To provide more high-quality positive samples for optimizing the detection network; (3) To recover fine-grained information lost during downsampling.

**Key Insight**: Instead of using Transformers to process raw point clouds (which is too slow), sparse convolutions are first used to efficiently extract and downsample features, and then linear attention is applied to a small number of downsampled voxels to establish global interactions. "Slots" are used instead of traditional window partitioning to achieve a larger receptive field.

**Core Idea**: Sparse convolution for efficient downsampling + SlotFormer linear attention for global interaction + dynamic sparse label assignment + sparse upsampling constitute a complete, high-performance sparse detector where the four components complement each other.

## Method

### Overall Architecture
The input raw point cloud is converted into sparse voxels via VFE. A sparse convolution encoder progressively extracts and downsamples features. SlotFormer establishes global interactions on the downsampled voxels, and a sparse upsampling module restores fine-grained representations. Finally, a dynamic sparse detection head generates predictions. There are two variants: FSHNet_light (VoxelNeXt encoder + 4-layer SlotFormer) and FSHNet_base (SAFDNet encoder + 8-layer SlotFormer).

### Key Designs

1. **SlotFormer (Slot Partitioning + Linear Attention)**:

    - **Function**: Establish long-range interactions among sparse voxels at a global scene level.
    - **Mechanism**: Partition the BEV scene into several "slots" along the X or Y axis (each slot covers the entire scene length along one axis, with a width of $w$ along the other). Inside each slot, linear attention is applied to all non-empty voxels: first calculate $Q=\phi(fW_q)$, $K=\phi(fW_k)$, $V=fW_v$ (where $\phi$ is ReLU), then aggregate the K-V product $kv_j = \sum k_i^T \cdot v_i$ within the same slot, and finally each voxel queries $v_i' = q_i \cdot kv_{d_i} / (q_i \sum k_j^T)$. Alternating X/Y slot directions across different layers forms global interactions.
    - **Design Motivation**: Traditional window partitioning is limited by the window size, whereas slot partitioning allows the receptive field to extend infinitely in one direction. Linear attention reduces complexity from $O(N^2)$ to $O(N)$ and naturally handles variable-length inputs (since voxel counts across different slots vary significantly).

2. **Dynamic Sparse Label Assignment (DSLA)**:

    - **Function**: Dynamically select multiple high-quality positive samples for each GT box to improve network optimization.
    - **Mechanism**: For each GT box $\mathbf{b}^t$, find the $n$ nearest voxels to its center as the candidate set $\mathcal{V}_b$. Calculate the selection cost for each candidate as $c_i = \ell_{cls}(Pred(\nu_i), \mathbf{b}^t) + \lambda \ell_{reg}(Pred(\nu_i), \mathbf{b}^t)$, then determine the number of positive samples $k = \max(\lfloor\sum IoU\rfloor, 1)$ based on the sum of IoUs of candidate voxels. Finally, select the top-k voxels with the minimum cost as positive samples.
    - **Design Motivation**: Existing sparse detectors only assign the single voxel closest to the center as the positive sample, excluding a large number of high-quality candidates and leading to insufficient optimization. Selecting multiple positive samples dynamically with self-adaptive counts based on detection quality resolves this issue.

3. **Sparse Upsampling (SU)**:

    - **Function**: Restore fine-grained information lost during downsampling, improving small object detection.
    - **Mechanism**: First double the voxel coordinates $(x_i, y_i) \rightarrow (2x_i, 2y_i)$ (equivalent to halving the voxel size), and then use a 3×3 stride-1 sparse convolution to diffuse voxel features to neighboring positions $\mathcal{V}^{up} = SpConv(\mathcal{V}')$. This restores spatial resolution to 1/4 of the original and leverages neighborhood features to replenish details.
    - **Design Motivation**: Although multi-step downsampling reduces computational cost, it loses small object information. Sparse upsampling using convolutional diffusion both restores resolution and mitigates missing center features (by diffusing features to positions that were previously empty).

### Loss & Training
- Classification loss: Focal Loss, with a positive sample weight of 1, candidate negative sample weight as the IoU of prediction and GT, and other negative sample weights as 0.
- Regression loss: Rotation-weighted IoU Loss.
- Adam optimizer, learning rate of 0.003, end-to-end training from scratch.
- Train for 12 epochs on Waymo/AV2, 36 epochs on nuScenes (without CBGS).
- 2×A100 GPU, total batch size 16.

## Key Experimental Results

### Main Results

**Waymo Open Validation Set (LEVEL 2 mAP/mAPH)**:

| Method | Type | mAP/mAPH | Vehicle | Pedestrian | Cyclist |
|------|------|----------|---------|------------|---------|
| **FSHNet_base** | Sparse | **77.1/74.9** | 82.2/81.7 | 85.9/80.8 | 80.5/79.4 |
| SAFDNet | Sparse | 75.7/73.9 | 80.6/80.1 | 84.7/80.4 | 80.0/79.0 |
| ScatterFormer | Dense | 75.7/73.8 | 81.0/80.5 | 84.5/79.9 | 79.9/78.9 |
| HEDNet | Dense | 75.3/73.4 | 81.1/80.6 | 84.4/80.0 | 78.7/77.7 |
| DSVT | Dense | 74.0/72.1 | 79.7/79.3 | 83.7/78.9 | 77.5/76.5 |

**nuScenes Validation Set**:

| Method | Type | NDS | mAP |
|------|------|-----|-----|
| **FSHNet_base** | Sparse | **71.7** | **68.1** |
| DSVT | Dense | 71.1 | 66.4 |
| SAFDNet | Sparse | 71.0 | 66.3 |

**Argoverse2 Validation Set (Long-range detection 200m)**:

| Method | Type | mAP |
|------|------|-----|
| **FSHNet_base** | Sparse | **40.2** |
| SAFDNet | Sparse | 38.7 |
| HEDNet | Dense | 37.1 |

### Ablation Study

| SlotFormer | DSLA | SU | Vehicle | Pedestrian | Cyclist |
|:---:|:---:|:---:|---------|------------|---------|
| | | | 69.1/68.7 | 75.3/69.5 | 75.0/73.9 |
| ✓ | | | 70.3/69.9 | 75.9/70.5 | 76.2/75.1 |
| | ✓ | | 69.9/69.5 | 75.5/69.7 | 75.7/74.5 |
| | | ✓ | 69.3/68.8 | 76.6/71.0 | 75.2/74.1 |
| ✓ | ✓ | | 70.5/70.1 | 77.1/71.6 | 75.6/74.5 |
| ✓ | ✓ | ✓ | **72.5/72.0** | **77.9/72.6** | **77.2/76.1** |

**Voxel Partitioning Method Comparison**:

| Partitioning Method | Vehicle | Pedestrian | Cyclist |
|---------|---------|------------|---------|
| Slot + linear attn | **72.5/72.0** | **77.9/72.6** | **77.2/76.1** |
| Window + linear attn | 72.2/71.7 | 77.8/72.4 | 77.1/76.1 |
| Window + Set + linear attn | 72.0/71.5 | 77.7/72.3 | 77.1/76.0 |
| Window + Set + self-attn | 71.8/71.3 | 77.3/72.0 | 76.8/75.7 |

### Key Findings
- SlotFormer yields the largest improvement for large objects (Vehicle, Cyclist) (+1.2/+1.2 AP), verifying the importance of global interactions for large object detection.
- Sparse upsampling contributes the most to small objects (Pedestrian) (+1.3 APH), reflecting the value of restoring fine-grained information.
- DSLA consistently improves all categories, showing that more high-quality positive samples are crucial for sparse detector optimization.
- Slot partitioning outperforms window partitioning, and linear attention is superior to self-attention (more efficient and effective).
- The three components are fully complementary — their joint usage yields improvements far greater than the sum of their individual gains.

## Highlights & Insights
- **Slot Partitioning Design**: Partitions the scene into infinitely long strips along one axis, combining with linear attention to achieve global interaction with $O(N)$ complexity — simpler than window partitioning but with a larger receptive field. Alternating X/Y directions covers the entire scene.
- **Dual Role of Sparse Upsampling**: It not only recovers fine-grained resolution to improve small object detection, but its convolutional diffusion also generates new voxels at previously empty locations, indirectly mitigating the missing center feature issue.
- **Dynamic Positive Sample Counts**: Adaptively determines the $k$ value based on the sum of candidate voxel IoUs — good predictions receive more positive samples to form positive feedback, while poor predictions conservatively assign 1.

## Limitations & Future Work
- Only validated under a single-frame setting; the efficacy of multi-frame temporal fusion has not been explored.
- The slot width $w$ in SlotFormer is a manually set hyperparameter ($w=12$), and different scenes might require different settings.
- The impact of representational capacity lost by linear attention compared to standard attention in sparse scenes on final detection accuracy has not been analyzed in depth.
- Only tested in the LiDAR-only setting; combination with camera-fusion methods remains to be explored.

## Related Work & Insights
- **vs SAFDNet**: SAFDNet uses adaptive feature diffusion to alleviate missing center features, but only diffuses within a local range. FSHNet's SlotFormer provides global interaction capabilities, and sparse upsampling can also diffuse and generate new voxels.
- **vs DSVT/FlatFormer**: They use window + sorting partitioning strategies to handle raw voxels, which are limited by window sizes. FSHNet first uses sparse convolutions for downsampling to reduce voxel counts, then applies slot partitioning for more efficient global interaction.
- **vs DCDet**: DCDet's dynamic cross label assignment is designed for dense detectors. FSHNet's DSLA is tailored for loose sparse voxels, dynamically selecting from the nearest candidate voxels.

## Rating
- Novelty: ⭐⭐⭐⭐ The slot partitioning idea in SlotFormer is novel, and its combination with sparse convolution is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three major datasets + validation/test sets + detailed ablation studies + various partition comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed method descriptions.
- Value: ⭐⭐⭐⭐ Provides a new standardized solution for sparse 3D detection, with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[CVPR 2025\] SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts](sp3d_boosting_sparsely-supervised_3d_object_detection_via_accurate_cross-modal_s.md)
- [\[CVPR 2025\] FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection](faster_focal_token_acquiring-and-scaling_transformer_for_long-term_3d_objection_.md)
- [\[CVPR 2025\] MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection](monoplace3d_learning_3d-aware_object_placement_for_3d_monocular_detection.md)
- [\[ECCV 2024\] Interactive 3D Object Detection with Prompts](../../ECCV2024/3d_vision/interactive_3d_object_detection_with_prompts.md)

</div>

<!-- RELATED:END -->
