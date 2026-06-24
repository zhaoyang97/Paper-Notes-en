---
title: >-
  [Paper Note] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This paper proposes VersatileGaussian, which equips 3D Gaussians with shared multi-task features and designs a Task Correlation Attention (TCA) module to enable cross-task information flow, achieving SOTA accuracy for multi-task label prediction on ScanNet and Replica datasets while maintaining a real-time rendering speed of 35 FPS.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Multi-Task Learning"
  - "Neural Rendering"
  - "Real-Time Rendering"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: 63a80ea154d86304
---

# VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting

**Conference**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03032.pdf)
**Code**: No public code  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Multi-Task Learning, Neural Rendering, Real-Time Rendering, Semantic Segmentation

## TL;DR

This paper proposes VersatileGaussian, which equips 3D Gaussians with shared multi-task features and designs a Task Correlation Attention (TCA) module to enable cross-task information flow, achieving SOTA accuracy for multi-task label prediction on ScanNet and Replica datasets while maintaining a real-time rendering speed of 35 FPS.

## Background & Motivation

**Background**: Acquiring multi-task (MT) labels (such as semantic segmentation, depth estimation, surface normals) in 3D scenes is crucial for applications like autonomous driving, AR/VR, and robotic navigation. Traditional methods adopt an analysis-by-synthesis strategy, which renders RGB images from novel views first and then uses 2D models to predict labels. NeRF-based methods (such as Semantic-NeRF) encode multi-task information within implicit representations, but suffer from extremely slow rendering speeds.

**Limitations of Prior Work**: (1) The rendering speed of NeRF pipelines (< 1 FPS) cannot meet real-time application requirements. (2) The continuity of multi-task fields in implicit representations causes artifacts like blurry boundaries during rendering—labels between different semantic regions tend to seep into each other within continuous implicit fields. (3) Although 3D Gaussian Splatting (3DGS) achieves real-time rendering, simply appending multi-task attributes to each Gaussian degrades rendering quality due to the lack of info interaction across tasks.

**Key Challenge**: Each Gaussian in 3DGS is optimized independently, lacking information flow between different task attributes. However, multi-task learning inherently exhibits correlation (e.g., semantic labels are strongly correlated with depth). Ignoring cross-task correlation leads to degraded multi-task prediction quality and can even negatively affect the original RGB rendering quality.

**Goal**: (1) How to effectively encode multi-task information within the 3DGS framework? (2) How to achieve cross-task information interaction to improve the prediction quality of each task? (3) How to maintain the real-time rendering speed of 3DGS?

**Key Insight**: The authors observe a complementary relationship between different tasks—for instance, boundary information from semantic segmentation helps improve depth estimation accuracy, and vice versa. By equipping Gaussians with shared task features instead of independent task attributes, and then using an attention mechanism to exchange information across tasks during rendering, cooperative effects between tasks can be obtained without significantly increasing computational costs.

**Core Idea**: Replace the individual multi-task attributes of 3DGS with a shared feature + task correlation attention decoding architecture to enable cross-task information flow, improving multi-task prediction accuracy while maintaining real-time rendering.

## Method

### Overall Architecture

On top of standard 3DGS, VersatileGaussian adds a shared multi-task feature vector to each Gaussian. During rendering, these features are projected onto a 2D feature map through the 3DGS rasterization pipeline, and then decoded into prediction results for each task (semantic labels, depth, surface normals, etc.) via a lightweight Task Correlation Attention (TCA) module. The input consists of multi-view RGB images and their corresponding multi-task labels, and the output is a 3D Gaussian scene capable of rendering novel-view RGB images and multi-task labels in real time.

### Key Designs

1. **Shared MT Feature Gaussians**:

    - **Function**: Replaces storing independent attributes for each task with compact shared features, reducing the parameter footprint and laying the foundation for cross-task interaction.
    - **Mechanism**: In addition to standard attributes (position $\mu$, covariance $\Sigma$, opacity $\alpha$, spherical harmonics $c$), each 3D Gaussian carries an extra $d$-dimensional shared feature vector $f \in \mathbb{R}^d$. This feature is not tailored to any specific task but encodes generic multi-task information of that position. During rendering, the features are projected into 2D via alpha blending: $F(p) = \sum_{i \in \mathcal{N}} f_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$
    - **Design Motivation**: Storing attributes for $K$ tasks independently requires $K \times d_k$ dimensions, with complete information isolation between tasks. Sharing a $d$-dimensional feature ($d << K \times d_k$) not only reduces storage overhead but also naturally sets up a pathway for cross-task information flow.

2. **Feature Map Rasterizer**:

    - **Function**: Efficiently projects the shared features of 3D Gaussians into a 2D feature map.
    - **Mechanism**: Extends the standard 3DGS CUDA rasterization pipeline to render feature vectors in parallel while rendering RGB colors. Leveraging the pre-existing sorting and tile-based rendering architecture of 3DGS, the additional computational overhead of feature map rendering is minimal. The final feature of each pixel is an alpha-weighted blend of the Gaussian features overlapping that pixel.
    - **Design Motivation**: Directly reuses the highly efficient rasterization architecture of 3DGS, avoiding the introduction of additional rendering branches and preserving real-time performance.

3. **Task Correlation Attention (TCA)**:

    - **Function**: Decodes predictions for each task from the shared feature map, while exploiting cross-task correlations to enhance the quality of each task.
    - **Mechanism**: TCA takes the rendered 2D feature map $F$ as input and achieves information exchange between tasks via multi-head cross-attention. Specifically, a query vector $q_k$ is learned for each task $k$, while the shared feature map provides the keys and values. The queries of different tasks participate in the attention computation simultaneously, aggregating useful information from other tasks through a soft-weighting mechanism: $O_k = \text{softmax}(q_k K^T / \sqrt{d}) V + \text{FFN}(F)$. Finally, predictions are output through task-specific linear heads.
    - **Design Motivation**: Complementary relationships exist between multi-tasks—for instance, semantic boundaries aid depth discontinuity detection, and surface normals aid geometry perception. TCA adaptively discovers and utilizes these cross-task correlations via an attention mechanism, eliminating the need for manual design.

### Loss & Training

The total loss is a weighted sum of individual task losses: $L = L_{rgb} + \sum_k \lambda_k L_k$. $L_{rgb}$ is the standard L1 + SSIM loss; semantic segmentation uses cross-entropy loss; depth estimation uses L1 loss; surface normals use cosine similarity loss. Training can be supervised by ground truth labels, or weakly supervised using pseudo-labels predicted by pre-trained 2D models (such as SAM, DPT).

## Key Experimental Results

### Main Results

| Dataset | Method | mIoU (Semantic) ↑ | RMSE (Depth) ↓ | Angular Error (Normal) ↓ | FPS ↑ |
|--------|------|--------------|--------------|----------------------|-------|
| ScanNet | VersatileGaussian | **68.2** | **0.089** | **12.3°** | **35** |
| ScanNet | Semantic-NeRF | 63.5 | 0.102 | 14.7° | 0.3 |
| ScanNet | 3DGS + Independent | 64.1 | 0.098 | 14.1° | 32 |
| Replica | VersatileGaussian | **82.5** | **0.031** | **6.8°** | **35** |
| Replica | Semantic-NeRF | 78.3 | 0.038 | 8.2° | 0.3 |

### Ablation Study

| Configuration | mIoU ↑ | PSNR (RGB) ↑ | Description |
|------|--------|-------------|------|
| VersatileGaussian (Full) | **68.2** | **31.4** | Full model |
| w/o TCA (Direct linear decoding) | 64.8 | 30.9 | TCA contributes ~3.4 mIoU |
| w/o Shared features (Independent attributes) | 64.1 | 30.5 | Shared features contribute ~4.1 mIoU |
| TCA replaced with MLP | 66.3 | 31.0 | Attention outperforms MLP by 1.9 mIoU |
| Feature dimension d=16 | 66.5 | 31.1 | Small dimension slightly limits performance |
| Feature dimension d=64 | 68.0 | 31.3 | Performance close to d=32 |

### Key Findings
- The TCA module not only improves multi-task prediction quality but also retroactively enhances RGB rendering quality (+0.5 dB PSNR), showing that multi-task information indeed assists geometric understanding.
- Even when using pseudo-labels from 2D pre-trained models (without GT labels), VersatileGaussian still achieves reasonable multi-task predictions, demonstrating its flexibility in integrating with existing foundation models.
- The shared feature dimension $d=32$ is the most cost-effective choice, as further increasing the dimension yields diminishing returns.

## Highlights & Insights
- **Cross-task information flow boosting single-task quality** is the core insight: Multi-tasking is not merely simple parameter sharing, but mutual knowledge assistance among tasks. The TCA module achieves this at an extremely low computational cost, and this design can be transferred to any multi-task 3D representation method.
- **Reusing the 3DGS rasterization architecture** to render feature maps is highly efficient, incurring almost zero FPS overhead. This approach provides a paradigm for extending 3DGS to more downstream tasks—simply by increasing feature channels without modifying the core rendering pipeline.
- Utilizing pseudo-labels from 2D foundation models as weak supervision when GT labels are missing greatly expands the application scenarios of this method.

## Limitations & Future Work
- The TCA module operates on 2D feature maps without leveraging the neighborhood relationships between Gaussians in 3D space, potentially missing out on 3D geometric priors.
- It has only been validated on indoor scenes (ScanNet, Replica) and has not been tested on large-scale outdoor scenes (e.g., autonomous driving scenarios).
- The shared feature dimension is constant across all tasks, but different tasks might require varying amounts of information; adaptive dimension allocation could be considered.
- Integration with the latest 3DGS variants (such as Mip-Splatting, 2D Gaussian Splatting) has not been explored, which could potentially further improve rendering quality.

## Related Work & Insights
- **vs Semantic-NeRF**: Semantic-NeRF encodes semantics in implicit fields, resulting in slow rendering and continuity artifacts. VersatileGaussian resolves these two issues using explicit Gaussians + shared features, boosting speed by 100x.
- **vs 3DGS + Independent**: Simply appending multi-task attributes to each Gaussian independently lacks cross-task interaction. VersatileGaussian enables information flow between tasks via TCA, bringing improvements across all tasks.
- **vs Feature 3DGS**: Feature 3DGS also adds features to Gaussians but lacks a cross-task interaction mechanism. The TCA module of VersatileGaussian is the key differentiating design.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of introducing cross-task attention to 3DGS multi-task rendering is novel, but the overall framework is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation studies and thorough validation across multiple datasets and tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and easy-to-understand method descriptions.
- Value: ⭐⭐⭐⭐ Provides an effective solution for multi-task extension of 3DGS, with real-time performance giving it strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)
- [\[ECCV 2024\] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [\[ECCV 2024\] A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)
- [\[ECCV 2024\] CityGaussian: Real-Time High-Quality Large-Scale Scene Rendering with Gaussians](citygaussian_real-time_high-quality_large-scale_scene_rendering_with_gaussians.md)
- [\[ECCV 2024\] Hyperion: A Fast, Versatile Symbolic Gaussian Belief Propagation Framework for Continuous-Time SLAM](hyperion_-_a_fast_versatile_symbolic_gaussian_belief_propagation_framework_for_c.md)

</div>

<!-- RELATED:END -->
