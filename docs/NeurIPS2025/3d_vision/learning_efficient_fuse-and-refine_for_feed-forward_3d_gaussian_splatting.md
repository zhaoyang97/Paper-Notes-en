---
title: >-
  [Paper Note] Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D Vision][3DGS] This paper proposes a Fuse-and-Refine module that aggregates pixel-aligned Gaussian primitives into a coarse-to-fine voxel hierarchy via a hybrid Splat-Voxel representation. A sparse voxel…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3DGS"
  - "Splat-Voxel"
  - "feed-forward reconstruction"
  - "streaming dynamic scenes"
  - "sparse-view"
date: 2026-05-08
content_hash: 60f92fcc3b65be7a
---

# Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2503.14698](https://arxiv.org/abs/2503.14698)  
**Code**: [GitHub](https://19reborn.github.io/SplatVoxel/)  
**Area**: 3D Vision / Feed-Forward 3DGS
**Keywords**: 3DGS, Splat-Voxel, feed-forward reconstruction, streaming dynamic scenes, sparse-view

## TL;DR

This paper proposes a Fuse-and-Refine module that aggregates pixel-aligned Gaussian primitives into a coarse-to-fine voxel hierarchy via a hybrid Splat-Voxel representation. A sparse voxel Transformer fuses approximately 200K primitives within 15 ms, yielding ~2 dB PSNR improvement. The model is trained exclusively on static scenes yet generalizes zero-shot to streaming dynamic scene reconstruction.

## Background & Motivation

Feed-forward 3DGS methods (e.g., PixelSplat, GS-LRM) predict Gaussian primitives directly from sparse views through large-scale training, enabling efficient scene reconstruction. However, three core issues persist:

**Redundancy**: Primitives are pixel-aligned with input images, producing massive redundancy when input views overlap.

**Positional constraint**: Primitives are confined to input rays and cannot be freely placed in 3D space, limiting reconstruction resolution.

**Difficulty in dynamic extension**: Pixel-aligned designs cannot naturally handle inter-frame redundancy merging or new-content infilling across temporal frames.

Existing heuristic primitive merging methods (e.g., hierarchical Gaussians) typically degrade reconstruction quality and require additional optimization. The core idea of this paper is to **fuse and refine primitives in a canonical 3D space through learning, using voxels as a structured intermediate representation**.

## Method

### Overall Architecture

Input: pixel-aligned Gaussian primitives from a feed-forward 3DGS model (e.g., GS-LRM) → Splat-to-Voxel conversion → coarse-to-fine voxel hierarchy construction → sparse voxel Transformer → output refined Gaussian primitives.

### Key Designs

1. **Splat-to-Voxel Conversion**:

    - **Splat deposition**: For a dense high-resolution voxel grid, each Gaussian primitive is deposited into its 8 nearest voxels using a distance kernel, with weights computed via a normalized kernel function.
    - **Splat fusion**: Each voxel accumulates deposited primitive features and attributes via opacity-weighted summation, enabling physically meaningful feature aggregation.
    - This process transforms unordered point-cloud-style primitives into a structured voxel representation.

2. **Coarse-to-Fine Voxel Hierarchy**:

    - High-resolution voxels are downsampled by a factor of $[d,h,w]$ to generate low-resolution coarse voxels.
    - Coarse voxel features are produced from the corresponding fine voxel features via a shallow MLP.
    - **Voxel sparsification**: Voxels are ranked by weight; only the top 20% of coarse voxels are retained, reducing the number of Transformer input tokens to approximately 10K.
    - This is the key to efficiency: global attention is performed at the coarse level (fewer tokens), while local refinement at the fine level preserves detail.

3. **Sparse Voxel Transformer**:

    - Coarse voxel features are treated as a 1D token sequence and fed into a Transformer for global context modeling.
    - The processed features are propagated back to the fine voxel grid, where a shallow MLP combines them with the initial fine voxel attributes to produce refined Gaussian primitives.
    - The entire Fuse-and-Refine module processes approximately 200K primitives in only 15 ms.

4. **Zero-Shot Streaming Dynamic Scene Reconstruction**:

    - **3D deformation**: A pretrained 2D point-tracking model is used to obtain cross-frame correspondences, from which 3D motion is recovered via triangulation.
    - An Embedded Deformation Graph propagates the motion of sparse anchor points to the entire scene.
    - **Error-aware fusion**: Deformed primitives from historical keyframes are combined with new primitives from the current frame in the Splat-Voxel representation; adaptive weights based on rendering error filter out deformation artifacts.
    - Only keyframe primitives are maintained; old keyframes are replaced with smooth transitions.

### Loss & Training

- Photometric loss: MSE + λ·LPIPS, where λ = 0.5 for the multi-view Transformer and λ = 4.0 for the voxel Transformer.
- Two-stage training: the multi-view Transformer is trained for 200K iterations, followed by joint fine-tuning of the voxel Transformer for 100K iterations.
- Training data: DL3DV large-scale scene dataset, batch size 128, 300K iterations total.
- Training is conducted on static scenes only; no additional training is required for dynamic scene inference.

## Key Experimental Results

### Main Results

| Dataset | Metric | SplatVoxel | Prev. SOTA (GS-LRM) | Gain |
|--------|------|-----------|-------------------|------|
| RealEstate10K | PSNR↑ | **28.47** | 28.10 | +0.37 |
| RealEstate10K | SSIM↑ | **0.907** | 0.892 | +0.015 |
| RealEstate10K | LPIPS↓ | **0.078** | 0.114 | −31.6% |
| DL3DV | PSNR↑ | **30.61** | 28.59 | +2.02 |
| DL3DV | SSIM↑ | **0.935** | 0.925 | +0.010 |
| Neural3DVideo (streaming) | PSNR↑ | **27.41** | 21.80 | +5.61 |
| Neural3DVideo (streaming) | Flicker↓ | **2.916** | 5.714 | −49% |

### Ablation Study

| Configuration | PSNR↑ | Time (ms) | Notes |
|------|-------|---------|------|
| GS-LRM baseline | 28.59 | 52.8 | 24-layer multi-view Transformer |
| + non-learned fusion | 12.57 | 33.8 | Heuristic merging causes severe degradation |
| + w/o coarse-to-fine | 20.62 | 48.6 | Full-resolution Transformer infeasible |
| + w/o sparsification | 29.69 | 72.0 | Excessive computation |
| + 3D CNN instead of Transformer | 29.44 | 82.1 | Global attention outperforms local convolution |
| + w/o Splat features | 29.40 | 49.2 | Feature information is important |
| + Full method | **30.61** | **52.5** | All designs complement each other |

### Key Findings

- Non-learned heuristic fusion causes catastrophic degradation (PSNR drops from 28.59 to 12.57), confirming the necessity of learned fusion.
- The coarse-to-fine hierarchy achieves the best quality–efficiency trade-off; without it, PSNR falls to only 20.62.
- PSNR improves substantially on streaming dynamic scenes (+5.61 dB) with a significant reduction in flickering, demonstrating the effectiveness of historical information fusion.
- Training on 4 views generalizes to 2/8/16 views, with larger gains at 16 views.

## Highlights & Insights

- **Splat-Voxel hybrid representation**: Combines the flexibility of point clouds with the structure of voxels, making the representation well-suited for both aggregation and Transformer processing.
- **Coarse-to-fine + sparsification**: Performing global attention at the coarse level and local refinement at the fine level is an elegant balance between efficiency and quality.
- **Static training, zero-shot dynamic generalization**: The core Fuse-and-Refine capability is general-purpose and transfers to dynamic scenes without requiring dynamic training data.
- **15 ms inference**: Fusing and refining 200K primitives in 15 ms enables interactive-rate (15 fps) streaming reconstruction.

## Limitations & Future Work

- Streaming reconstruction depends on the quality of the 2D point-tracking model (e.g., TAPIR); tracking failures degrade deformation accuracy.
- Voxel resolution and sparsification ratio are hyperparameters that require tuning for different scenes.
- Validation is currently limited to sparse-view (2–4 views) settings; performance in dense-view scenarios remains unexplored.

## Related Work & Insights

- **vs. GS-LRM**: GS-LRM predicts pixel-aligned primitives directly from a multi-view Transformer without global optimization in 3D space; SplatVoxel adds 3D fusion and refinement on top of this pipeline.
- **vs. 4DGS**: 4DGS requires per-scene optimization for dynamic scenes, whereas SplatVoxel generalizes zero-shot and operates much faster.
- **vs. EvolvSplat / S-Cube**: These methods also employ voxel representations for 3DGS processing but do not extend to dynamic scenes and require scene-specific training.

## Rating

- Novelty: ⭐⭐⭐⭐ The Splat-Voxel hybrid design and coarse-to-fine hierarchy are elegant; zero-shot dynamic generalization is a notable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on both static and dynamic scenes, detailed ablations, and generalization experiments across different view counts.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, figures are high quality, and paper structure is well organized.
- Value: ⭐⭐⭐⭐⭐ Provides a general solution for primitive fusion in feed-forward 3DGS; the streaming reconstruction application has broad practical potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting](plana3r_zero-shot_metric_planar_3d_reconstruction_via_feed-forward_planar_splatt.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](../../CVPR2026/3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](../../CVPR2026/3d_vision/off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation](fin3r_fine-tuning_feed-forward_3d_reconstruction_models_via_monocular_knowledge_.md)
- [\[NeurIPS 2025\] ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](zpressor_bottleneck-aware_compression_for_scalable_feed-forward_3dgs.md)

</div>

<!-- RELATED:END -->
