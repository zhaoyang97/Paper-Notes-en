---
title: >-
  [Paper Note] Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting
description: >-
  [ICLR 2026][3D Vision][3D style transfer] Stylos proposes a single-forward 3D style transfer framework that achieves zero-shot 3D stylization from uncalibrated inputs via a dual-path design with a shared Transformer back…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "3D style transfer"
  - "Gaussian splatting"
  - "cross-view consistency"
  - "voxel style loss"
  - "feed-forward model"
date: 2026-05-08
content_hash: 80dd732feb01158c
---

# Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting

**Conference**: ICLR 2026
**arXiv**: [2509.26455](https://arxiv.org/abs/2509.26455)  
**Code**: [https://github.com/HanzhouLiu/Stylos](https://github.com/HanzhouLiu/Stylos)  
**Area**: 3D Vision
**Keywords**: 3D style transfer, Gaussian splatting, cross-view consistency, voxel style loss, feed-forward model

## TL;DR

Stylos proposes a single-forward 3D style transfer framework that achieves zero-shot 3D stylization from uncalibrated inputs via a dual-path design with a shared Transformer backbone (geometry self-attention + style cross-attention) and a voxel-level 3D style loss, supporting scalability from single-view to hundreds of views.

## Background & Motivation

3D style transfer aims to transfer a reference style while preserving scene geometry and cross-view consistency. Existing methods suffer from three levels of limitations:

**NeRF/3DGS methods require per-scene optimization**: StyleRF, StyleGaussian, and similar approaches, though more efficient than NeRF, still require per-scene fitting and cannot achieve truly real-time 3D stylization.

**Weak generalization**: Existing methods are confined to scene-specific training and cannot generalize to unseen categories, scenes, or styles.

**2D style losses lack 3D consistency**: Classical Gram matrix or AdaIN (channel statistics matching) operates at the image level and cannot explicitly enforce multi-view structural consistency.

The closest related work, Styl3R (Wang et al., 2025b), proposes a feed-forward framework but is designed only for 2–8 input views and does not specifically target strong multi-view consistency.

## Method

### Overall Architecture

Stylos uses VGGT as the geometry backbone and introduces a Style Aggregator branch that fuses content and style features via CrossBlocks. Geometric attributes (depth, pose) are derived solely from the backbone, while style influences only the color spherical harmonic coefficients, enabling disentanglement of geometry and style.

### Key Designs

1. **CrossBlock Style–Content Fusion Module**

   A cross-attention operation is inserted between the self-attention and MLP of a standard Transformer block: content tokens serve as Query, and style tokens serve as Key/Value. Three topological strategies are proposed:
   - **Frame CrossBlock**: Each view independently interacts with the style, preserving view-specific structure.
   - **Global CrossBlock**: All views are concatenated into a global sequence; self-attention ensures multi-view geometric consistency while cross-attention broadcasts style information globally.
   - **Hybrid CrossBlock**: Frame CrossBlock followed by Global CrossBlock.

   Experiments show that **Global CrossBlock performs best** (PSNR improvement of 0.79 dB on the Pizza scene), as global self-attention guarantees cross-view consistency while cross-attention globally broadcasts style.

2. **Voxel-level 3D Style Loss**

   Multi-view rendered features are fused into a voxel grid $\mathcal{G}_b^l$ via differentiable back-projection, and style statistics are then matched against the reference style in voxel space:

   $$\mathcal{L}_{\text{sty}}^{3D} = \frac{1}{B} \sum_{b=1}^B \sum_{l=1}^5 \alpha_l \left(\|\mu(\mathcal{G}_b^l) - \mu(\mathcal{S}_b^l)\|_2^2 + \|\sigma(\mathcal{G}_b^l) - \sigma(\mathcal{S}_b^l)\|_2^2\right)$$

   Compared to image-level style losses (per-frame matching without cross-view consistency guarantees) and scene-level losses (2D feature concatenation, still operating in 2D space), the voxel-level loss directly encodes geometry in 3D space and enforces cross-view style consistency.

3. **Prediction Head Design**

    - Geometry head: DPT regression head outputting position, scale, rotation, and opacity.
    - Style head: Color head predicting spherical harmonic coefficients.
    - Auxiliary heads: VGGT camera head estimating intrinsic/extrinsic parameters; depth head predicting scene geometry.

### Loss & Training

**Stage 1 — Geometry Pre-training**: Initialized with VGGT weights; end-to-end geometry learning. A randomly selected input view with color jitter is used as the style reference (to prevent identity mapping). Loss: $\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{distill}} \mathcal{L}_{\text{distill}}$

**Stage 2 — Stylization Fine-tuning**: Geometry modules are frozen; only the Style Aggregator and color head are updated. Loss:
$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{style}} \mathcal{L}_{\text{style}}^{3D} + \lambda_{\text{cnt}} \mathcal{L}_{\text{content}} + \lambda_{\text{clip}} \mathcal{L}_{\text{clip}} + \lambda_{\text{tv}} \mathcal{L}_{\text{TV}}$$

## Key Experimental Results

### Main Results

| Dataset/Scene | Metric | Stylos | StyleGaussian | Styl3R | Notes |
|---|---|---|---|---|---|
| T&T Short LPIPS↓ | Consistency | **0.033–0.047** | 0.031–0.038 | — | Competitive |
| T&T Long LPIPS↓ | Consistency | **0.153** | 0.157 | — | Better long-range consistency |
| CO3D ArtScore↑ | Artistic quality | **9.15** | — | — | Highest with voxel loss |
| CO3D Recon. PSNR↑ | Reconstruction | 21.68 | — | — | Global CrossBlock |

### Ablation Study

| Configuration | Short RMSE↓ | ArtScore↑ | Notes |
|---|---|---|---|
| Image-level style loss | 0.038 | 4.78 | Baseline |
| Scene-level style loss | 0.036 | 9.12 | +4.34 ArtScore |
| 3D Voxel-level loss | **0.034** | **9.15** | Best in 3D |

### Key Findings

- Global CrossBlock outperforms Frame and Hybrid variants across all tested categories.
- Voxel-level 3D style loss surpasses 2D style losses on both consistency and artistic quality.
- Quality remains stable for up to 32 views per batch; edge artifacts emerge beyond 64 views (training uses at most 24 views).
- Image-level loss occasionally fails to transfer style entirely (e.g., the donut scene).

## Highlights & Insights

1. **Geometry–style disentanglement**: Backbone features drive only geometry; CrossBlocks affect only color — conceptually clean and modular.
2. **Progression from 2D to 3D style loss**: The work systematically advances from image-level → scene-level → voxel-level, providing a clear ablation trajectory.
3. **Strong scalability**: The framework naturally supports 1 to hundreds of views by adjusting batch size alone.
4. **Strong geometric foundation via VGGT**: Leveraging a pre-trained 3D foundation model ensures high-quality geometry.

## Limitations & Future Work

- Quality degrades beyond 32 views; larger training batches may be needed to address this.
- Only static scenes are evaluated; dynamic scene stylization remains a future direction.
- Style reference is limited to a single image; multi-reference style control could offer richer expressiveness.
- The effect of voxelization resolution on style quality warrants further analysis.

## Related Work & Insights

- VGGT (Wang et al., 2025a) and AnySplat (Jiang et al., 2025) provide a strong foundation for pose-free 3D reconstruction.
- The feature-level style/content loss from ArtFlow (An et al., 2021) is effectively extended to the 3D voxel space.
- The voxel-level statistics matching paradigm may generalize to other tasks requiring 3D consistency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The voxel-level 3D style loss and CrossBlock design are innovative, though the overall framework is a composition of mature components.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset evaluation with systematic ablations; baseline comparisons could be more comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with complete derivations; some descriptions could be more concise.
- **Value**: ⭐⭐⭐⭐ The first truly scalable single-forward 3D stylization method with clear practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](../../CVPR2026/3d_vision/brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](../../CVPR2026/3d_vision/instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[CVPR 2026\] Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](../../CVPR2026/3d_vision/reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)
- [\[CVPR 2026\] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](../../CVPR2026/3d_vision/coherent_humanscene_reconstruction_from_multiperso.md)
- [\[ICLR 2026\] Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator](text-to-3d_by_stitching_a_multi-view_reconstruction_network_to_a_video_generator.md)

</div>

<!-- RELATED:END -->
