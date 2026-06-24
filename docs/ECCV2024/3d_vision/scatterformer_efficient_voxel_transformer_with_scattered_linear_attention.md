---
title: >-
  [Paper Note] ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention
description: >-
  [ECCV 2024][3D Vision][3D Object Detection] ScatterFormer is proposed, which is the first voxel Transformer to directly apply linear attention on variable-length voxel sequences across windows. By implementing a Scattered Linear Attention (SLA) module and a chunk-wise matrix multiplication algorithm, it achieves sub-millisecond latency. Paired with a Cross-Window Interaction (CWI) module to replace window shifting, it achieves state-of-the-art accuracy on Waymo and nuScenes w…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Object Detection"
  - "Voxel Transformer"
  - "Linear Attention"
  - "Point Cloud Processing"
  - "LiDAR Perception"
date: 2026-05-08
content_hash: a609e8a04ea42e91
---

# ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention

**Conference**: ECCV 2024  
**arXiv**: [2401.00912](https://arxiv.org/abs/2401.00912)  
**Code**: [https://github.com/skyhehe123/ScatterFormer](https://github.com/skyhehe123/ScatterFormer)  
**Area**: 3D Vision  
**Keywords**: 3D Object Detection, Voxel Transformer, Linear Attention, Point Cloud Processing, LiDAR Perception

## TL;DR

ScatterFormer is proposed, which is the first voxel Transformer to directly apply linear attention on variable-length voxel sequences across windows. By implementing a Scattered Linear Attention (SLA) module and a chunk-wise matrix multiplication algorithm, it achieves sub-millisecond latency. Paired with a Cross-Window Interaction (CWI) module to replace window shifting, it achieves state-of-the-art accuracy on Waymo and nuScenes while maintaining a detection speed of 23 FPS.

## Background & Motivation

**Background**: Window-based Transformers (such as SST, DSVT) perform outstandingly in large-scale point cloud 3D object detection, obtaining context-aware feature representations by localizing attention computation. However, the sparsity of point clouds leads to highly variable numbers of voxels in each window (the variable-length sequence problem).

**Limitations of Prior Work**: Existing methods (SST, DSVT, FlatFormer) adopt group-based strategies to solve the parallel computation problem of variable-length sequences, sorting and padding voxels into fixed-length sequences. This introduces significant computational and memory overhead. In DSVT, sorting and resorting operations account for ~24% of the total backbone latency.

**Key Challenge**: The unequal number of voxels per window causes the attention matrix to occupy irregular memory space, preventing direct parallel computation. Meanwhile, the additional overhead introduced by existing solutions (padding + sorting) offsets the performance gains brought by attention.

**Goal**: Design a method to efficiently compute attention in parallel on variable-length voxel sequences without resorting and padding.

**Key Insight**: Leverage the "recurrence form" nature of linear attention—the hidden state matrix $S \in \mathbb{R}^{d \times d}$ is independent of sequence length, allowing parallel attention across different windows to be completed in shared memory via chunk-wise computation.

**Core Idea**: Flatten voxels of all windows into a single sequence, and utilize the KV pre-computation property of linear attention to perform chunk-wise parallel calculation, avoiding the overhead of padding and sorting.

## Method

### Overall Architecture

The overall architecture of ScatterFormer is as follows: the input point cloud is voxelized and converted into high-dimensional embeddings through a VFE layer, then processed by Conditional Position Encoding (CPE). The encoded features are fed into a backbone consisting of 6 ScatterFormer Blocks. Each Block contains an SLA module, a CWI module, and an FFN, supplemented by Batch Normalization and skip connections. After passing through 3 Blocks, features are downsampled via sparse convolution, then converted to pillar features to generate compact BEV features for bounding box prediction.

### Key Designs

1. **Linear Attention Basics**: Standard self-attention has a complexity of $O(N^2)$. Linear attention approximates softmax through kernel functions:

$$\phi(Q) \cdot \phi(K)^\mathsf{T} \approx \text{softmax}(QK^\mathsf{T})$$

By changing the computation order from $(\phi(Q) \cdot \phi(K)^\mathsf{T}) \cdot V$ to $\phi(Q) \cdot (\phi(K)^\mathsf{T} \cdot V)$, the complexity is reduced to $O(N)$. A key property is that the hidden state matrix $S \in \mathbb{R}^{d \times d}$ is independent of sequence length and can be converted into a recurrence form:

$$o_t = q_t S_t; \quad S_t = S_{t-1} + k_t^\mathsf{T} v_t$$

This implies that the sequence can be divided into non-overlapping chunks for computation.

2. **Scattered Linear Attention (SLA)**: The core innovative module. Voxels of the entire scene are sorted by window coordinates and then flattened into a single matrix. Voxels within the same window form sub-matrices in contiguous memory. After obtaining $Q, K, V$ via shared projection layers, linear attention is computed separately for each window's sub-matrix:

$$\text{SLA}(Q,K,V) = \text{Concat}[\text{LA}(Q^j, K^j, V^j)]_{j=1:M}$$

$$\text{LA}: O^j = \frac{\phi(Q) \sum_{i=1}^{m^j} \phi(K_i)^\top V_i}{\phi(Q) \sum_{i=1}^{m^j} \phi(K_i)^\top}$$

where $M$ is the number of non-empty windows, and $m^j$ is the number of voxels in the $j$-th window.

   **Hardware-Efficient Implementation (Chunk-wise Algorithm)**: Optimization utilizing GPU memory hierarchy:
    - Split the flattened $Q, K, V$ into fixed-length chunks.
    - Load chunks from slow HBM to fast SRAM (shared memory).
    - Assign a single thread to each window, iterating through all corresponding K/V chunks to accumulate matrix products into the hidden state matrix $S$.
    - Assign independent threads to each query chunk $q_i \in Q^j$ to multiply with the corresponding hidden state matrix to obtain the output.
    - Implemented using Triton, avoiding large output matrices, with less than 1ms latency.

   Compared to a naive scatter-based implementation, the chunk-wise method has significant advantages in speed and memory usage.

3. **Cross-Window Interaction (CWI)**: A cross-window interaction module that replaces window shifting. Window shifting requires re-calculating window coordinates and resorting voxels, which accounts for ~24% of latency in DSVT. The CWI module employs an Inception-style multi-branch design:

    - Input features are divided into 4 parts along the channel dimension.
    - Branch 1: $(S_h+1) \times 1 \times 1$ depthwise convolution (long 1D kernel along the height direction).
    - Branch 2: $1 \times (S_w+1) \times 1$ depthwise convolution (long 1D kernel along the width direction).
    - Branch 3: $3 \times 3 \times 3$ depthwise convolution (enhancing locality).
    - Branch 4: Identity mapping.

$$X' = \text{Concat}(X_k, X_w, X_h, X^{3c:4c})$$

   The axially decomposed large kernel design allows voxel features to be efficiently mixed across different windows, achieving a better accuracy-latency trade-off than window shifting.

### Loss & Training

- The detection head and loss function are identical to DSVT: heatmap estimation + bounding box regression + IoU loss confidence calibration.
- The detection head on nuScenes adopts the TransFusion architecture.
- Waymo: voxel size (0.32m, 0.32m, 0.1875m), window (12,12), 24 epochs, lr=0.006.
- nuScenes: voxel size (0.3m, 0.3m, 8m), window (20,20), 20 epochs, lr=0.004.
- Ground-truth sampling data augmentation is disabled in the last 4 epochs.
- 8 RTX A6000 GPUs, batch size 32.

## Key Experimental Results

### Main Results — Waymo Open Dataset (Val Set)

| Method | Frames | ALL L2 mAPH↑ | Vehicle L2↑ | Pedestrian L2↑ | Cyclist L2↑ |
|------|------|-------------|------------|---------------|------------|
| DSVT-1f | 1 | 72.1 | 71.0 | 71.5 | 73.7 |
| HEDNet-1f | 1 | 73.4 | 72.7 | 72.6 | 74.9 |
| **ScatterFormer** | **1** | **73.8** | 72.7 | 72.6 | **76.1** |
| DSVT-4f | 4 | 75.6 | 73.6 | 75.9 | 77.3 |
| **ScatterFormer-4f** | **4** | **76.7** | **74.7** | **76.6** | **78.1** |

### Main Results — nuScenes (Val Set)

| Method | NDS↑ | mAP↑ |
|------|------|------|
| DSVT | 71.1 | 66.4 |
| HEDNet | 71.4 | 66.7 |
| **ScatterFormer** | **72.4** | **68.3** |

### Ablation Study

| Configuration | Vehicle APH/L2 | Ped APH/L2 | Cyclist APH/L2 | Description |
|------|---------------|-----------|---------------|------|
| baseline (Full) | 71.1 | 70.9 | 73.1 | All components |
| w/o SLA | 68.0 (-3.1) | 67.0 (-3.9) | 69.9 (-3.2) | SLA is the most critical component |
| w/o CWI | 69.2 (-1.9) | 69.9 (-1.0) | 71.1 (-2.0) | CWI contributes to all classes |
| CWI→SW | 69.5 (-1.6) | 70.4 (-0.5) | 72.4 (-0.7) | CWI outperforms window shifting |
| w/o CPE | 69.8 (-1.3) | 67.2 (-3.7) | 70.5 (-2.6) | CPE is crucial for Pedestrian/Cyclist |

### Window Size Ablation

| Window Size | Area Size | Vehicle | Pedestrian | Cyclist |
|---------|---------|---------|-----------|---------|
| 10 | 3.20m | 70.2 | 69.2 | 71.0 |
| **12** | **3.84m** | **71.1** | **70.9** | **73.1** |
| 14 | 4.48m | 71.3 | 69.9 | 72.5 |
| 16 | 5.12m | 71.0 | 69.2 | 72.3 |

### Key Findings

- ScatterFormer single-frame achieves a 1.7% improvement in Waymo L2 mAPH over DSVT and 0.4% over HEDNet.
- On nuScenes, mAP outperforms HEDNet by 1.6% (68.3 vs 66.7) and NDS by 1.0%.
- Detection speed reaches 23 FPS, significantly outperforming other Transformer detectors and approaching sparse convolutional detectors.
- SLA module latency is < 1ms, with the chunk-wise implementation performing significantly better in speed and memory than the scatter-based implementation.
- The model is insensitive to window size, with 12×12 being the optimal choice.
- CWI yields better performance than Shifted Window with lower latency.
- The model is not highly sensitive to different linear attention variants (Efficient Attn, Gated Linear Attn, Focused Linear Attn, XCA).

## Highlights & Insights

- **Pioneering Nature**: The first method to directly apply attention to variable-length voxel sequences, completely avoiding sorting and padding.
- **I/O-Aware Algorithm**: Design of chunk-wise matrix multiplication utilizing GPU memory hierarchy (HBM→SRAM) is an excellent example of hardware-algorithm co-design.
- **CWI Replacing Window Shifting**: Sorting/resorting accounts for 24% of latency in DSVT; CWI thoroughly eliminates this overhead with simple axially decomposed large-kernel convolutions.
- **Linear Complexity without Sacrificing Accuracy**: Proves that linear attention is not only feasible in 3D detection but also superior to softmax attention.
- **Significant Improvement for Cyclist**: ScatterFormer-4f improves by 0.8 APH (L2) on Cyclist compared to DSVT-4f, indicating a special advantage in detecting sparse, small objects.

## Limitations & Future Work

- Reliance on custom operators (Triton kernel), which have not been implemented as TensorRT plugins, requiring additional engineering effort for in-vehicle deployment.
- Performance of linear attention degrades with excessively large windows, likely because the "focusing capability" of linear attention is diluted as the token count increases.
- The current depthwise convolution requires support from a modified Spconv library, increasing the barrier to entry.
- Deeper integration with temporal modeling (multi-frame fusion) can be explored.
- Adaptive mechanisms such as gating could be introduced to enhance the expressiveness of linear attention.
- Failed to completely surpass multi-frame methods on Waymo test set Pedestrian and Cyclist categories.

## Related Work & Insights

- **DSVT**: Alternating axis sorting + fixed-length grouping. This is the most direct baseline; ScatterFormer comprehensively outperforms it in accuracy and speed.
- **SST**: Pioneered window-based Transformers for point cloud detection, but has low mixed serial-parallel computation efficiency.
- **FlatFormer**: Flattens voxels into fixed-length sequences, but still incurs sorting overhead.
- **FlashAttention**: I/O-aware attention computation inspired the chunk-wise implementation of SLA.
- **Inspiration**: The recurrence form of linear attention $S_t = S_{t-1} + k_t^\top v_t$ is naturally suited to solving variable-length sequence problems. This insight can be extended to other sparse data processing with Transformers.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to combine the recurrence property of linear attention with variable-length voxel sequences; SLA + CWI designs are elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on two datasets + detailed ablation study + multiple linear attention comparison + runtime analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation, detailed algorithm description, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Pushes the frontier of voxel Transformers in both accuracy and efficiency, carrying major significance for real-time 3D detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[CVPR 2026\] VIAFormer: Voxel-Image Alignment Transformer for High-Fidelity Voxel Refinement](../../CVPR2026/3d_vision/viaformer_voxel-image_alignment_transformer_for_high-fidelity_voxel_refinement.md)
- [\[ECCV 2024\] FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance](falip_visual_prompt_as_foveal_attention_boosts_clip_zero-shot_performance.md)
- [\[ECCV 2024\] Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction](analysis-by-synthesis_transformer_for_single-view_3d_reconstruction.md)
- [\[ECCV 2024\] LaRa: Efficient Large-Baseline Radiance Fields](lara_efficient_large-baseline_radiance_fields.md)

</div>

<!-- RELATED:END -->
