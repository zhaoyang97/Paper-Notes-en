---
title: >-
  [Paper Note] Random Wins All: Rethinking Grouping Strategies for Vision Tokens
description: >-
  [CVPR 2026][3D Vision][Vision Transformer] This paper proposes a minimalist random grouping strategy to replace various elaborately designed token grouping methods in Vision Transformers. The approach achieves near-universal improvements over all baselines across image classification, object detection, semantic segmentation, point cloud segmentation, and VLMs, and provides a four-dimensional explanation for its success: positional information, per-head feature diversity, global receptive field, and fixed grouping patterns.
tags:
  - CVPR 2026
  - 3D Vision
  - Vision Transformer
  - Token Grouping
  - Random Grouping
  - Attention Mechanism
  - Efficiency Optimization
date: 2026-05-08
content_hash: a1d7899437009752
---

# Random Wins All: Rethinking Grouping Strategies for Vision Tokens

**Conference**: CVPR 2026
**arXiv**: [2603.00486](https://arxiv.org/abs/2603.00486)
**Authors**: Qihang Fan, Yuang Ai, Huaibo Huang, Ran He (Institute of Automation, Chinese Academy of Sciences)
**Code**: [GitHub](https://github.com/qhfan/random)
**Area**: 3D Vision
**Keywords**: Vision Transformer, Token Grouping, Random Grouping, Attention Mechanism, Efficiency Optimization

## TL;DR

This paper proposes a minimalist random grouping strategy to replace various elaborately designed token grouping methods in Vision Transformers. The approach achieves near-universal improvements over all baselines across image classification, object detection, semantic segmentation, point cloud segmentation, and VLMs, and provides a four-dimensional explanation for its success: positional information, per-head feature diversity, global receptive field, and fixed grouping patterns.

## Background & Motivation

- **State of the Field**: The self-attention mechanism in Transformers incurs $O(n^2)$ quadratic complexity, and vision token grouping is the dominant approach to reducing this cost. Grouping strategies have grown increasingly sophisticated, from simple window partitioning (Swin Transformer) to semantically aware tree-structured grouping (Quadtree) and bi-level routing grouping (BiFormer), yet inference efficiency has continuously declined.
- **Limitations of Prior Work**: Are these elaborately designed grouping methods truly necessary? Complex clustering and routing operations severely hinder deployment efficiency, and it remains unclear whether performance gains originate from the grouping strategy itself.
- **Root Cause**: A minimalist random grouping strategy—simply applying a random permutation to tokens followed by equal partitioning—almost universally outperforms existing complex grouping methods across all tasks and baselines, while also achieving faster inference.
- **Paper Goals**: The paper proposes the Random Grouping Strategy and conducts an in-depth analysis of why such a simple method succeeds, summarizing four key design principles for grouping strategies.

## Method

### Overall Architecture

The core mechanism of the random grouping strategy is extremely straightforward: a fixed random tensor is generated to shuffle the token order, and the tokens are then equally partitioned into groups for intra-group self-attention or pooling. The strategy serves as a drop-in replacement for the token grouping modules in various baselines including Swin, CSwin, Quadtree, BiFormer, PVTv2, and Focal, and can be extended to multi-modal tasks such as point cloud processing and VLMs.

### Key Designs

#### Design 1: Random Tensor Generation and Sort-Based Grouping

- **Function**: Randomly shuffles input tokens and partitions them into equal groups.
- **Mechanism**: Given input $X \in \mathbb{R}^{h \times w \times d}$, a random tensor $P \in \mathbb{R}^{h \times w}$ is generated with a one-to-one spatial correspondence to $X$. Tokens are reordered by sorting $P$ in descending order to obtain $X_p$, which is then evenly split into groups. Once generated, $P$ is stored and fixed, so all subsequent images share the same token ordering.
- **Design Motivation**: Random shuffling completely breaks local bias, ensuring that tokens within each group originate from globally distributed positions across the image, thereby naturally acquiring a global receptive field. Fixing $P$ ensures a consistent grouping pattern during training, enabling the model to learn stable feature representations.

#### Design 2: Per-Head Independent Random Grouping

- **Function**: Assigns a distinct random grouping pattern to each head in multi-head attention.
- **Mechanism**: The shape of $P$ is extended from $h \times w$ to $n \times h \times w$ (where $n$ is the number of attention heads), so each head uses an independent random tensor for grouping.
- **Design Motivation**: Using different random groupings per head encourages greater diversity in the features learned by each head. Ablation experiments show that sharing a single $P$ across all heads leads to a significant performance drop (e.g., Random-Swin-T drops from 82.7% to 80.5%), confirming the critical role of per-head feature diversity.

#### Design 3: Nearest-Neighbor Interpolation for High-Resolution Adaptation

- **Function**: Adapts the fixed-resolution random tensor to different scales in downstream tasks.
- **Mechanism**: Since $P$ is fixed at shape $h \times w$, nearest-neighbor interpolation is used to resize $P$ to the target resolution when applied to high-resolution scenarios such as object detection (800×1333) or semantic segmentation (512×512).
- **Design Motivation**: Nearest-neighbor interpolation preserves the discrete structural properties of the random permutation, avoiding the smoothing effects that methods such as bilinear interpolation may introduce, and ensures sharp grouping boundaries.

#### Design 4: Unified Adaptation for Three Backbone Categories

- **Function**: Provides a unified grouping replacement scheme for Plain, Partition-based, and Pooling-based backbones.
- **Mechanism**:
    - **Plain backbones** (e.g., DeiT): Random grouping is applied directly; intra-group self-attention reduces global $O(n^2)$ complexity to $O((n/g)^2)$.
    - **Partition-based backbones** (e.g., Swin, CSwin, BiFormer): Original window/routing grouping is replaced by random grouping.
    - **Pooling-based backbones** (e.g., PVTv2, Focal): Spatial grouping prior to token pooling is replaced by random grouping.
- **Design Motivation**: Demonstrates that random grouping is a general-purpose strategy that is architecture-agnostic and can uniformly replace diverse grouping methods.

## Key Experimental Results

### Main Results: ImageNet-1K Image Classification

| Model | Params (M) | FLOPs (G) | Throughput (img/s) | Top-1 Acc (%) |
|-------|-----------|-----------|-------------------|---------------|
| DeiT-T | 6 | 1.3 | 6433 | 72.2 |
| **Random-DeiT-T** | 6 | 1.1 | 6682 | **73.1 (+0.9)** |
| DeiT-S | 22 | 4.6 | 3122 | 79.8 |
| **Random-DeiT-S** | 22 | 4.3 | 3313 | **80.9 (+1.1)** |
| DeiT-B | 87 | 17.6 | 1226 | 81.8 |
| **Random-DeiT-B** | 87 | 17.0 | 1348 | **82.5 (+0.7)** |
| Swin-T | 28 | 4.5 | 1738 | 81.3 |
| **Random-Swin-T** | 28 | 4.5 | 1866 | **82.7 (+1.4)** |
| Swin-S | 50 | 8.7 | 1186 | 83.0 |
| **Random-Swin-S** | 50 | 8.7 | 1248 | **83.9 (+0.9)** |
| Swin-B | 88 | 15.4 | 864 | 83.5 |
| **Random-Swin-B** | 88 | 15.4 | 902 | **84.4 (+0.9)** |
| Quadtree-b2 | 24 | 4.5 | 467 | 82.7 |
| **Random-Quadtree-b2** | 21 | 4.3 | 1926 | **83.4 (+0.7)** |
| BiFormer-B | 57 | 9.8 | 544 | 84.3 |
| **Random-BiFormer-B** | 57 | 9.6 | 667 | **85.1 (+0.8)** |
| PVTv2-B2 | 25 | 4.0 | 1663 | 82.0 |
| **Random-PVTv2-B2** | 21 | 4.2 | 1678 | **82.7 (+0.7)** |
| Focal-B | 90 | 16.0 | 248 | 83.8 |
| **Random-Focal-B** | 88 | 15.5 | 887 | **84.5 (+0.7)** |

### Main Results: COCO Object Detection and Instance Segmentation

| Backbone | Mask R-CNN AP^b | AP^m | RetinaNet AP^b |
|----------|----------------|------|----------------|
| Swin-T | 43.7 | 39.8 | 41.7 |
| **Random-Swin-T** | **46.0 (+2.3)** | **41.9 (+2.1)** | **44.3 (+2.6)** |
| Swin-S | 45.7 | 41.1 | 44.5 |
| **Random-Swin-S** | **48.0 (+2.3)** | **43.2 (+2.1)** | **46.6 (+2.1)** |
| Swin-B | 46.9 | 42.3 | 45.0 |
| **Random-Swin-B** | **49.1 (+2.2)** | **44.6 (+2.3)** | **47.4 (+2.4)** |
| PVTv2-B2 | 45.3 | 41.2 | 44.6 |
| **Random-PVTv2-B2** | **47.1 (+1.8)** | **42.4 (+1.2)** | **46.0 (+1.4)** |

### Main Results: ADE20K Semantic Segmentation

| Model | UperNet 160K mIoU (%) |
|-------|-----------------------|
| Swin-T | 44.5 |
| **Random-Swin-T** | **46.8 (+2.3)** |
| Swin-S | 47.6 |
| **Random-Swin-S** | **48.9 (+1.3)** |
| CSwin-B | 51.1 |
| **Random-CSwin-B** | **52.2 (+1.1)** |
| BiFormer-B | 51.0 |
| **Random-BiFormer-B** | **52.0 (+1.0)** |

### Ablation Study: Four Key Factors

| Ablation | Model | Acc (%) | Change |
|----------|-------|---------|--------|
| **Positional Information** | Random-Swin-T | 82.7 | - |
|　Remove PE | Random-Swin-T w/o PE | 79.3 | **-3.4** |
|　Reference: Swin-T w/o PE | - | 80.1 | -1.6 |
| **Per-Head Feature Diversity** | Random-Swin-T (multi-P) | 82.7 | - |
|　All heads share single P | Random-Swin-T (single-P) | 80.5 | **-2.2** |
| **Global Receptive Field** | Random-Swin-T (global) | 82.7 | - |
|　Restricted to local regions | Random-Swin-T (regional) | 81.5 | **-1.2** |
| **Fixed Grouping Pattern** | Random-Swin-T (fixed P) | 82.7 | - |
|　Different P per image | Fully Random Swin-T | 76.4 | **-6.3** |

### Ablation Study: Roadmap from Fully Random to Random-Swin

| Configuration | Throughput (img/s) | Acc (%) |
|--------------|-------------------|---------|
| Fully Random (single P) | 1922 | 71.2 |
| + Fixed grouping pattern | 1922 | 77.6 (+6.4) |
| + Per-head independent P | 1917 | 80.1 (+2.5) |
| + CPE positional encoding | 1866 | 82.7 (+2.6) |
| Swin-T (reference) | 1738 | 81.3 |

### Key Findings

1. **Random grouping comprehensively outperforms carefully designed methods**: Across 3 backbone categories × 6+ architectures × 5 tasks, random grouping almost universally surpasses the original grouping strategies while achieving faster inference.
2. **Larger gains on downstream tasks**: On COCO detection, Random-Swin-T yields gains of up to +2.3 AP^b and +2.6 RetinaNet AP^b, far exceeding the +1.4% improvement on classification.
3. **Significant speed advantages**: Random-Quadtree-b2 achieves 1926 img/s vs. 467 img/s for the original Quadtree-b2—a 4.1× speedup—alongside a +0.7% accuracy gain.
4. **Fixed pattern is the most critical factor**: Fully Random (different P per image) causes a catastrophic -6.3% drop, demonstrating that models require a consistent grouping pattern to learn stable feature representations.
5. **Multi-modal generalization**: On Point Transformer v3, latency is reduced by 23% (88ms→68ms) while mIoU improves by +0.2; all benchmarks improve when applied to LLaVA-1.5/1.6.

## Highlights & Insights

- **Counter-intuitive yet profound finding**: Elaborately designed grouping strategies are outperformed by random grouping, overturning the default assumption that "more complex is better." The core contribution lies not only in the method itself but in a deeper understanding of the grouping strategy design space.
- **Practical guidance from the four-factor analysis**: The four factors—positional information, per-head diversity, global receptive field, and fixed patterns—provide clear design principles for future Transformer efficiency research.
- **Exceptional engineering simplicity**: No complex clustering, routing, or tree structures are required; only sorting and splitting, making the method deployment-friendly and well-suited for industrial applications.
- **Elegant roadmap experiment design**: The progressive ablation in Tab. 10 elegantly demonstrates the journey from fully random grouping (71.2%) to incrementally incorporating each key factor, ultimately surpassing Swin-T (82.7% vs. 81.3%).

## Limitations & Future Work

- Random grouping yields smaller gains on architectures that already possess a global receptive field (e.g., CSwin-T gains only +0.4%), suggesting that its advantages stem primarily from enhancing global receptive fields and per-head diversity.
- The shape of the fixed random tensor is tied to the training resolution; transferring to higher resolutions requires interpolation, which may degrade the quality of the random permutation.
- The paper does not discuss random seed sensitivity or whether different random initializations introduce performance variance.

## Related Work & Insights

- **Swin Transformer** [ICCV 2021]: The canonical window-based grouping method, universally outperformed by random grouping, demonstrating that local windows are not the optimal choice.
- **BiFormer** [CVPR 2023]: Bi-level routing grouping with high complexity, surpassed by a simple random approach, revealing the problem of over-engineering.
- **Point Transformer v3** [CVPR 2024]: Grouping strategies in point cloud scenarios can also be replaced by random grouping, demonstrating cross-modal universality.

## Rating

| Dimension | Score (1–10) | Notes |
|-----------|-------------|-------|
| Novelty | 9 | A minimalist yet counter-intuitive finding that challenges the entire token grouping paradigm. |
| Technical Depth | 8 | The four-factor analysis is thorough and systematic; ablation design is elegant. |
| Experimental Thoroughness | 9 | Comprehensive validation across 6 backbone types, 5 tasks, and 3 modalities. |
| Writing Quality | 8 | Logical and clear, progressively building from observations to explanations. |
| Value | 9 | Plug-and-play, deployment-friendly, with extremely high engineering value. |
| **Overall** | **8.6** | A paradigm-shifting work that challenges complex designs with extreme simplicity. |

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](../../ICCV2025/3d_vision/3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [\[CVPR 2026\] Pano360: Perspective to Panoramic Vision with Geometric Consistency](pano360_perspective_to_panoramic_vision_with_geome.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](ava_bench_atomic_visual_ability_vfm.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)

<!-- RELATED:END -->
