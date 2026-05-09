---
title: >-
  [Paper Note] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes a feed-forward 3DGS decoder based on keypoint detection, liberating Gaussian primitives from the pixel grid by placing them adaptively at sub-pixel precision. Combined with an adaptive density mechanism and confidence-based pruning, the method surpasses state-of-the-art feed-forward approaches in novel view synthesis using only 1/7 of the primitives required by pixel-aligned methods.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - feed-forward reconstruction
  - keypoint detection
  - adaptive density
  - pose-free reconstruction
date: 2026-05-08
content_hash: d8ee4a9a2065cf69
---

# Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2512.15508](https://arxiv.org/abs/2512.15508)
**Code**: [Project Page](https://arthurmoreau.github.io/OffTheGrid/)
**Area**: 3D Vision / 3D Gaussian Splatting
**Keywords**: 3D Gaussian Splatting, feed-forward reconstruction, keypoint detection, adaptive density, pose-free reconstruction

## TL;DR

This paper proposes a feed-forward 3DGS decoder based on keypoint detection, liberating Gaussian primitives from the pixel grid by placing them adaptively at sub-pixel precision. Combined with an adaptive density mechanism and confidence-based pruning, the method surpasses state-of-the-art feed-forward approaches in novel view synthesis using only 1/7 of the primitives required by pixel-aligned methods.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the dominant method for efficient 3D scene representation. Classical 3DGS requires SfM initialization followed by per-scene optimization (taking tens of minutes to hours), whereas recent feed-forward methods (e.g., PixelSplat, AnySplat) directly predict Gaussian primitives in a single forward pass, reducing reconstruction time to seconds.

**Limitations of Prior Work**: Existing feed-forward methods almost universally adopt pixel-aligned or voxel-aligned primitive placement strategies, where each input pixel corresponds to one Gaussian primitive and primitive positions are rigidly locked to a regular grid. This introduces two problems: (1) the number of primitives equals the number of input pixels, restricting these methods to low resolutions (typically 256×256); and (2) regular grids cannot adaptively allocate more primitives to high-frequency detail regions or reduce redundancy in flat regions, resulting in simultaneous losses in quality and efficiency.

**Key Challenge**: Optimization-based 3DGS dynamically adjusts primitive distributions through densification and pruning, but feed-forward methods lack this capability. The pixel-aligned design inherently limits the model's expressive power over primitive placement, as it cannot learn an "optimal placement" strategy.

**Goal**: How to achieve adaptive, off-grid primitive placement in feed-forward 3DGS while remaining end-to-end trainable.

**Key Insight**: Inspired by keypoint detection, the authors recast Gaussian primitive placement as a 2D detection problem—extracting continuous coordinates from image patches via convolutional heatmaps, enabling primitives to be localized at sub-pixel precision.

**Core Idea**: Replace pixel-grid alignment with the DSNT soft-argmax formulation from keypoint detection, allowing feed-forward 3DGS models to learn adaptive placement of Gaussian primitives at sub-pixel accuracy.

## Method

### Overall Architecture

The pipeline comprises three components: (1) VGGT as the 3D reconstruction backbone, predicting depth maps and camera parameters from $N$ uncalibrated images; (2) the proposed 3D Gaussian decoder, which detects and describes Gaussian primitives from VGGT features and predicted geometry; and (3) end-to-end training via photometric loss computed by rendering the input images, requiring no 3D annotations.

$N$ input images → VGGT feature extraction + depth/camera prediction → U-Net decoder for detection/description features → heatmap-based 2D primitive localization → bilinear interpolation of depth/color/descriptors → back-projection to 3D Gaussian centers → MLP prediction of remaining parameters → multi-view aggregation + rendering.

### Key Designs

1. **Sub-pixel Primitive Detection (DSNT)**:

    - **Function**: Precisely localize Gaussian primitive centers in continuous 2D space, transcending the pixel grid constraint.
    - **Mechanism**: The detection features from the U-Net decoder are reshaped into $14\times14$ patches, each with $P$ channels (where $P$ is the number of primitives per patch). A spatial softmax is applied to each channel to produce a heatmap, and the soft-argmax (DSNT) computes the expected coordinate $x = \sum_{i,j} c_x(i,j)\, h(i,j)$, yielding continuous floating-point coordinates. This allows primitive centers to lie between pixels, accurately representing geometric structures that span pixel boundaries.
    - **Design Motivation**: Pixel-aligned strategies anchor primitives to integer coordinates, precluding optimal scene representation. DSNT is differentiable and supports end-to-end training; its effectiveness has already been demonstrated in human pose estimation.

2. **Adaptive Density Mechanism (Shannon Entropy)**:

    - **Function**: Dynamically allocate different numbers of primitives to image patches based on content complexity.
    - **Mechanism**: The Shannon entropy $H = -\sum_k p_k \log_2(p_k + \epsilon)$ of the grayscale histogram is computed for each $14\times14$ patch. Patches are ranked by entropy: the lowest 55% are assigned 16 primitives (low density), the middle 35% receive 32 (medium density), and the top 15% receive 64 (high density). Independent detection and description convolutional heads are learned for each density level.
    - **Design Motivation**: High-detail regions (high entropy) require more primitives for accurate representation, while uniform regions (low entropy) need only a few. Even the 64 primitives assigned to high-density patches are far fewer than the 196 pixels per patch, maintaining efficiency throughout.

3. **Confidence-Based Multi-View Aggregation**:

    - **Function**: Intelligently select among primitives predicted from multiple views to avoid redundancy-induced blurring.
    - **Mechanism**: A confidence value $c \in [0,1]$ is predicted for each Gaussian, and the final opacity is set to $\alpha \cdot c$. The model implicitly learns to reduce a primitive's confidence when the corresponding region is better observed from another viewpoint. At test time, primitives with $\alpha c < 0.1$ are pruned, further improving efficiency.
    - **Design Motivation**: Naively aggregating primitives from all views causes the same content to be represented multiple times, producing blurring. Through confidence learning, the model acquires an implicit deduplication capability.

### Loss & Training

Training employs four complementary loss terms: (1) **Photometric loss**: L1 + SSIM + LPIPS, rendering only the input views (no held-out target views required); (2) **Geometric consistency loss**: L1 loss between predicted and rendered depth + normal consistency loss, ensuring Gaussian orientations align with local surface geometry; (3) **Teacher geometry loss**: constrains fine-tuned depth and camera parameters to remain close to original VGGT predictions, preventing training collapse; (4) **Regularization loss**: opacity regularization $L_{op} = \sum \sin(\alpha \cdot c)$, encouraging opacities toward 0 or 1 to avoid semi-transparency artifacts.

Training uses a single GPU (140 GB VRAM), processing up to 24 images per iteration, with no 3D annotations required.

## Key Experimental Results

### Main Results

| Dataset | Metric | Off The Grid | AnySplat | DA3-Giant |
|---------|--------|-------------|----------|-----------|
| Average (6 datasets) | PSNR↑ | **21.21** | 17.71 | 18.83 |
| Average | SSIM↑ | **0.647** | 0.508 | 0.543 |
| Average | LPIPS↓ | **0.353** | 0.394 | 0.383 |
| DL3DV | PSNR↑ | **20.48** | 17.31 | 18.46 |
| Tanks & Temples | PSNR↑ | **19.37** | 16.28 | 16.94 |

Compression ratio: the proposed method uses only **0.143 primitives/pixel** (1/7 of the input pixels), compared to 1.0 for pixel-aligned methods and 0.814 for AnySplat.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------------|-------|-------|--------|
| Ours (Off-The-Grid detection) | **19.22** | **0.604** | **0.333** |
| Pixel-aligned baseline | 18.90 | 0.590 | 0.382 |
| SplatterImage-style 3D offset | 18.87 | 0.586 | 0.389 |

Geometry evaluation (average):

| Model | AbsRel↓ | AUC@30↑ | FoV error↓ |
|-------|---------|---------|------------|
| Off The Grid | 0.143 | 0.928 | **0.96** |
| DA3-Giant | **0.134** | **0.934** | 3.66 |
| AnySplat | 0.159 | 0.833 | 3.225 |

### Key Findings
- Off-The-Grid improves PSNR by +0.3 dB and reduces LPIPS by 13% over pixel-aligned placement, yielding sharper renderings with fewer artifacts.
- SplatterImage-style 3D offset prediction not only fails to improve results but introduces isolated point artifacts, demonstrating that accurate primitive placement requires more sophisticated techniques.
- The adaptive density mechanism achieves an 86% compression ratio, with the model outperforming baselines while using 7× fewer primitives.
- Fine-tuned VGGT achieves the best intrinsic parameter estimation, which is critical for accurate back-projection.

## Highlights & Insights
- **Transfer of keypoint detection thinking to 3DGS**: Treating Gaussian primitives—which have no physical existence—as "detectable keypoints" and elegantly achieving sub-pixel precision localization via heatmaps and softmax is a clever analogical transfer.
- **Shannon entropy for adaptive density**: Patch complexity can be assessed without any learning, representing a simple yet effective engineering design choice.
- **Multi-view reasoning through confidence learning**: The model automatically learns to reason "this region is better observed from another viewpoint, so I reduce the confidence from this view"—an elegant solution that requires no explicit multi-view aggregation.
- Photometric supervision renders only input images rather than held-out views; teacher geometry loss prevents geometric collapse, simplifying the training pipeline.

## Limitations & Future Work
- Geometry of extremely fine structures may be lost due to limited expressiveness when the number of primitives is reduced.
- Alternative multi-view aggregation strategies have not been explored; the current confidence-based pruning is relatively simple.
- The hyperparameters for adaptive density (the 55%/35%/15% partition ratios) are set manually and could potentially be learned as an optimal allocation strategy.
- Training requires a single GPU with 140 GB VRAM, imposing high hardware demands.

## Related Work & Insights
- **vs. AnySplat**: Both fine-tune VGGT, but AnySplat uses voxel-aligned Gaussians while this work uses detection-based sub-pixel Gaussians. The proposed method significantly outperforms AnySplat across all datasets while using fewer primitives.
- **vs. PixelSplat**: The pioneering pixel-aligned work, limited to 256×256 low resolution. This paper fundamentally addresses the limitations of pixel alignment.
- **vs. DA3-Giant**: DA3 achieves stronger depth and pose estimation but suffers from inaccurate intrinsic estimation that leads to blurry renderings. This work improves intrinsic estimation through photometric fine-tuning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Applying keypoint detection concepts to 3DGS primitive placement is a novel and natural idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six evaluation datasets, multiple view-count settings, geometry evaluation, and ablation studies provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, method descriptions are detailed, and figures are well designed.
- **Value**: ⭐⭐⭐⭐ Represents an important paradigm shift for feed-forward 3DGS, from grid alignment to adaptive detection.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2026\] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](physgm_large_physical_gaussian_model_for_feed-forward_4d_synthesis.md)
- [\[CVPR 2026\] Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)

<!-- RELATED:END -->
