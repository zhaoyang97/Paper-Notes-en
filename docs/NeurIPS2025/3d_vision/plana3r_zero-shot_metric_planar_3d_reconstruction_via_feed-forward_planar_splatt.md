---
title: >-
  [Paper Note] Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting
description: >-
  [NeurIPS 2025][3D Vision][Planar 3D reconstruction] This paper proposes Plana3R, a feed-forward framework that requires neither camera poses nor planar annotations, predicting sparse 3D planar primitives and metric-scale relative poses from unpaired two-view images for zero-shot metric planar 3D reconstruction of indoor scenes.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Planar 3D reconstruction
  - metric reconstruction
  - planar splatting
  - indoor scenes
  - feed-forward model
date: 2026-05-08
content_hash: 8cdb34d9f677d0ec
---

# Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2510.18714](https://arxiv.org/abs/2510.18714)
**Code**: [Project Page](https://lck666666.github.io/plana3r/)
**Area**: 3D Vision
**Keywords**: Planar 3D reconstruction, metric reconstruction, planar splatting, indoor scenes, feed-forward model

## TL;DR

This paper proposes Plana3R, a feed-forward framework that requires neither camera poses nor planar annotations, predicting sparse 3D planar primitives and metric-scale relative poses from unpaired two-view images for zero-shot metric planar 3D reconstruction of indoor scenes.

## Background & Motivation

Indoor environments are the primary spaces of human activity, and creating their digital twins is critical for numerous applications. Indoor scenes are inherently rich in planar structures (floors, walls, tabletops, etc.), making planar primitives a compact and well-suited representation for indoor 3D reconstruction.

**Two key limitations of existing methods**:

**Annotation dependency**: Feed-forward planar reconstruction methods (e.g., SparsePlanes, NOPE-SAC) require precise planar masks and 3D planar annotations for training. Such dense annotations are scarce and costly to produce, severely limiting the available data volume and model generalizability.

**Pose dependency**: Per-scene optimization methods (e.g., PlanarSplatting) require accurately registered dense multi-view images, making them inapplicable in sparse, pose-free settings.

**Core insight of this paper**: Indoor environments follow human-scale size distributions, and the planar 3D representation inherently possesses the potential to predict metric 3D geometry. By leveraging the differentiable planar rendering technique from PlanarSplatting, a Transformer-based feed-forward model can be trained using only depth maps and normal maps as supervision—far more accessible than planar annotations—to directly predict sparse planar primitives and metric poses.

## Method

### Overall Architecture

The input consists of two pose-free images $I^1, I^2$ of the same scene along with their intrinsics $\mathbf{K}^1, \mathbf{K}^2$. The network $\mathcal{F}$ outputs a set of 3D planar primitives (depth $d_\pi$, radius $\mathbf{r}_\pi$, quaternion $\mathbf{q}_\pi$) and a 6-DoF metric relative pose $P_{\text{rel}}$. The CUDA-based differentiable renderer from PlanarSplatting renders the planar primitives into depth maps and normal maps, which are compared against ground truth to enable gradient backpropagation.

### Key Designs

1. **ViT-based encoder-decoder architecture**: A Siamese ViT encoder extracts features $F^i \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times D_{\text{enc}}}$, followed by a Transformer decoder with cross-attention to generate low-resolution embeddings $G_{\text{low}}^i$. A pose head regresses the relative pose from concatenated two-view features. Both the encoder and decoder are initialized with pretrained weights from DUSt3R.

2. **Hierarchical Primitive Prediction Architecture (HPPA)**: Low-resolution ($\frac{H}{16} \times \frac{W}{16}$) planar primitives are predicted from $G_{\text{low}}$ via three regression heads. A deconvolution network upsamples $G_{\text{low}}$ to $G_{\text{high}}$, from which the same regression heads predict high-resolution ($\frac{H}{8} \times \frac{W}{8}$) primitives. The key question is which regions should use low-resolution versus high-resolution primitives. A simple heuristic is adopted: the gradient magnitude of the low-resolution normal map $\mathbf{N}_{\text{low}}^{\text{patch}}$ is computed, and regions where the gradient exceeds a threshold $g_{\text{th}}=0.5$ switch to high-resolution primitives. Regions with large normal variation require more small planar primitives for accurate fitting, while regions with small variation can be represented by fewer large planes.

3. **Supervision without planar annotations**: The CUDA differentiable renderer from PlanarSplatting renders planar primitives into full-resolution depth and normal maps, which are directly compared against ground-truth depth and normal maps. Normal map ground truth is generated as pseudo-labels using Metric3Dv2. This enables training on large-scale two-view datasets with only depth and normal annotations, without any planar-level labels.

4. **Plane merging**: Predicted planar primitives are merged into semantically coherent large planes via thresholding on normal and distance similarity, enabling plane-level instance segmentation as a naturally emergent capability without additional training.

### Loss & Training

**Three loss components**:

- **Patch loss** (warm-up stage): Supervises depth and normals at patch resolution:
$$\mathcal{L}_*^{\text{patch}} = \alpha_1\|1 - (\mathbf{N}_*^{\text{patch}})^\top\mathbf{N}_*^{\text{r.gt}}\|_1 + \alpha_1\|\mathbf{N}_*^{\text{patch}} - \mathbf{N}_*^{\text{r.gt}}\|_1 + \alpha_2\|\mathbf{D}_*^{\text{patch}} - \mathbf{D}_*^{\text{r.gt}}\|_1$$

- **Render loss**: Supervises at full resolution via differentiable rendering:
$$\mathcal{L}_*^{\text{render}} = \beta_1\|1 - (\mathbf{N}_*^{\text{render}})^\top\mathbf{N}^{\text{gt}}\|_1 + \beta_1\|\mathbf{N}_*^{\text{render}} - \mathbf{N}^{\text{gt}}\|_1 + \beta_2\|\mathbf{D}_*^{\text{render}} - \mathbf{D}^{\text{gt}}\|_1$$

- **Pose loss**:
$$\mathcal{L}^{\text{pose}} = \gamma_1\|\mathbf{t}^{\text{gt}} - \mathbf{t}\|_1 + \gamma_2\|\mathbf{q}^{\text{gt}} - \frac{\mathbf{q}}{\|\mathbf{q}\|}\|_1 + \gamma_3(1 - \frac{\mathbf{t} \cdot \mathbf{t}^{\text{gt}}}{\|\mathbf{t}\|\|\mathbf{t}^{\text{gt}}\|})$$

**Training configuration**: The model is trained on approximately 4 million image pairs from 4 datasets. Training begins with a 1-epoch warm-up using only patch and pose losses, followed by 10 epochs with all three losses. Input resolution is $512 \times 384$. Total training cost is 256 GPU-days (H20 GPUs).

## Key Experimental Results

### Main Results (Two-view Reconstruction and Pose Estimation)

| Method | ScanNetV2 Chamfer↓ | ScanNetV2 F-score↑ | ScanNetV2 Trans Med (m)↓ | ScanNetV2 Rot Med (°)↓ |
|--------|-------------------|-------------------|--------------------------|------------------------|
| SparsePlanes | - | - | 0.56 | 15.46 |
| NOPE-SAC | 0.26 | 61.86 | 0.41 | 8.27 |
| MASt3R | 0.21 | 74.92 | 0.11 | 2.17 |
| **Plana3R** | **0.11** | **92.52** | **0.07** | **2.01** |

On Matterport3D (zero-shot, unseen during training), Plana3R achieves an F-score of 56.63, surpassing NOPE-SAC (54.96), which was trained on that dataset.

### Ablation Study (Monocular Depth Estimation on NYUv2)

| Method | Rel↓ | RMSE↓ | δ₁↑ |
|--------|------|-------|-----|
| PlaneRecTR | 0.157 | 0.547 | 74.2 |
| MASt3R | 0.152 | 0.51 | 83.0 |
| **Plana3R** | **0.132** | **0.463** | **86.4** |

Plana3R achieves zero-shot metric depth estimation on NYUv2, a dataset unseen during training, outperforming MASt3R.

### Key Findings

- Sparse planar primitive representations are more compact and efficient than dense point clouds in structured indoor environments while maintaining high accuracy.
- The planar segmentation capability that emerges naturally after plane merging outperforms PlaneRecTR on the Replica dataset, despite the latter requiring planar annotation supervision.
- The hierarchical primitive prediction adaptively selects resolution via gradient thresholding, enabling flexible adjustment of primitive count between 768 and 3072.
- Multi-view reconstruction can be extended to 8 or more input frames through pairwise inference.

## Highlights & Insights

- Replacing explicit planar annotation supervision with differentiable planar rendering is an elegant design choice that substantially reduces data requirements.
- The heuristic for hierarchical primitive prediction (normal gradient thresholding) is simple yet effective, avoiding additional learning overhead.
- Direct metric scale prediction benefits from the human-scale prior of indoor scenes, which is an insightful observation.
- Planar segmentation emerges naturally as a byproduct of the compact representation, reflecting the principle that good representations inherently carry upstream capabilities.

## Limitations & Future Work

- The current design supports only pairwise two-view inference; multi-view scenarios require multiple forward passes.
- The planar representation has limited modeling capacity for non-planar regions (curved surfaces, complex objects).
- The method is sensitive to the quality of normal pseudo-labels generated by Metric3Dv2.
- The gradient threshold $g_{\text{th}}$ requires manual tuning and is not learned adaptively.

## Related Work & Insights

- **DUSt3R / MASt3R**: Foundation models for feed-forward two-view 3D reconstruction, but relying on dense point cloud representations.
- **PlanarSplatting**: The core differentiable planar rendering component this work builds upon, providing CUDA-accelerated planar primitive rendering.
- **SparsePlanes / NOPE-SAC**: Prior two-view planar reconstruction methods that require planar annotations.
- Insight: Leveraging domain-specific structural priors (e.g., indoor planarity) to select more compact representations enables surpassing dense methods with far fewer primitives.

## Rating

- **Novelty**: ⭐⭐⭐⭐ A synergistic combination of annotation-free planar reconstruction, hierarchical primitive prediction, and metric scale estimation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 4 datasets and 5 tasks; zero-shot generalization is impressive.
- **Writing Quality**: ⭐⭐⭐⭐ Modular descriptions are clear and equations are well-formatted.
- **Value**: ⭐⭐⭐⭐ Provides a more compact and semantically meaningful alternative for indoor 3D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](planargs_high-fidelity_indoor_3d_gaussian_splatting_guided_by_vision-language_pl.md)
- [\[NeurIPS 2025\] Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting](learning_efficient_fuse-and-refine_for_feed-forward_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation](fin3r_fine-tuning_feed-forward_3d_reconstruction_models_via_monocular_knowledge_.md)
- [\[NeurIPS 2025\] ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](zpressor_bottleneck-aware_compression_for_scalable_feed-forward_3dgs.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)

</div>

<!-- RELATED:END -->
