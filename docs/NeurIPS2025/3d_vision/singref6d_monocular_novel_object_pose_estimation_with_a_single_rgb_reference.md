---
title: >-
  [Paper Note] SingRef6D: Monocular Novel Object Pose Estimation with a Single RGB Reference
description: >-
  [NeurIPS 2025][3D Vision][6D pose estimation] SingRef6D is a lightweight 6D pose estimation pipeline requiring only a single RGB reference image. It fine-tunes Depth-Anything v2 via a token-scaler mechanism for robust depth prediction and introduces depth-aware matching to enhance LoFTR's spatial reasoning, substantially outperforming existing methods on transparent and reflective objects.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 6D pose estimation
  - monocular depth estimation
  - single reference image
  - Depth-Anything
  - depth-aware matching
date: 2026-05-08
content_hash: c2ad15c634de0f2f
---

# SingRef6D: Monocular Novel Object Pose Estimation with a Single RGB Reference

**Conference**: NeurIPS 2025
**arXiv**: [2509.21927](https://arxiv.org/abs/2509.21927)
**Code**: [https://plusgrey.github.io/singref6d/](https://plusgrey.github.io/singref6d/)
**Area**: 3D Vision / 6D Pose Estimation
**Keywords**: 6D pose estimation, monocular depth estimation, single reference image, Depth-Anything, depth-aware matching

## TL;DR

SingRef6D is a lightweight 6D pose estimation pipeline requiring only a single RGB reference image. It fine-tunes Depth-Anything v2 via a token-scaler mechanism for robust depth prediction and introduces depth-aware matching to enhance LoFTR's spatial reasoning, substantially outperforming existing methods on transparent and reflective objects.

## Background & Motivation

6D pose estimation is a core task in robotics, industrial automation, and augmented reality. Current methods face several practical limitations:

**CAD model dependency**: Acquiring CAD models for novel objects is costly, requiring specialized scanning equipment and manual refinement.

**Depth sensor limitations**: Sensors fail on transparent and highly reflective objects at rates exceeding 85% (ClearPose dataset).

**High overhead of multi-view methods**: Multi-view matching requires large template libraries, and neural field construction is computationally intensive and restricted to per-instance training.

**Lack of geometric information in RGB methods**: Matching performance degrades under low-light and texture-less conditions.

**Root Cause**: How can robust 6D pose estimation be achieved for challenging surface conditions (transparent, reflective) under the constraint of minimal reference input (a single RGB image)?

This work draws inspiration from the human visual system—humans efficiently estimate object pose without CAD models or binocular vision, relying instead on cognitive depth perception and shape understanding. SingRef6D emulates this mechanism by learning depth priors to implicitly expand the reference space.

## Method

### Overall Architecture

SingRef6D consists of three stages:
1. **Robust depth prediction**: Fine-tunes Depth-Anything v2 with a token-scaler mechanism to predict accurate metric depth from a single RGB image.
2. **Depth-aware matching**: Fuses RGB and depth into a unified latent space to extend LoFTR's matching capability.
3. **Pose solving**: Refines correspondences using PointDSC and computes the relative 6D pose via depth-projected point clouds.

### Key Designs

1. **Token-Scaler Fine-tuning Mechanism**: Introduces a ControlNet-style structure on the hierarchical features of DPAv2, applying dynamic scaling and modulation across feature levels:
    $F_l' = \mathcal{F}_l(F_l, Scaler(F_{l+1}'))$
    - Low/mid-level features ($F_1, F_2$): Enhanced with efficient attention layers to improve global perception and suppress noise.
    - High/global-level features ($F_3, F_4$): Processed with an InceptConv network to emphasize local features and enrich high-level feature maps.
    - **Design Motivation**: Mimics the hierarchical spatial perception of the human visual system; the DPAv2 backbone is frozen and only the lightweight token scaler is trained.

2. **Multi-level Loss Function**: Combines global and local losses: $\mathcal{L}_{depth} = \mathcal{L}_{local} + \mathcal{L}_{global}$

    - **Scale Alignment Loss**: Enforces object-level scale alignment with a robustness term against outliers:
    $\mathcal{L}_{scale} = \frac{1}{M}\sum_i \frac{(\hat{d}_i - d_i)^2}{1 + \eta|\hat{d}_i - d_i|}$
    - **Edge-emphasize Loss**: Weights depth gradient errors by RGB gradients to improve boundary reconstruction:
    $\mathcal{L}_{edge} = \frac{1}{M}\sum_i e^{-\sigma\|\nabla I_i\|} \cdot \|\nabla\hat{d}_i - \nabla d_i\|_2^2$
    - **Normal Consistency Loss**: Enforces surface normal consistency to maintain geometric coherence.
    - **Global Loss**: SSI + BerHu + regularization.

3. **Depth-Aware Matching Module**: Freezes LoFTR parameters and fuses RGB and depth features within its latent space. The depth map provides spatial priors, enabling robust matching under low-texture and challenging lighting conditions. PointDSC then refines the correspondences, and the 6D pose is solved via $T_q^{-1} = T_r^{-1} T_{q\to r}$.

### Loss & Training

During training, the parameters of DPAv2 and LoFTR are frozen; only the token scaler is trained, significantly reducing training cost. The depth loss integrates global scale calibration (SSI + BerHu) with local geometric accuracy (Scale + Edge + Normal).

## Key Experimental Results

### Main Results (Depth Estimation)

| Dataset | Metric | Ours | DPAv2(FT) | UniDepth(FT) | Gain |
|--------|------|------|-----------|-------------|------|
| Toyota-Light | $\delta_{1.05}$↑ | **80.09** | 14.64 | 11.80 | +65.45 |
| REAL275 | $\delta_{1.05}$↑ | **44.28** | 29.87 | 33.81 | +14.41 vs DPAv2 |
| ClearPose | $\delta_{1.05}$↑ | **54.30** | 31.23 | 12.73 | +23.07 vs DPAv2 |

### Main Results (6D Pose Estimation)

| Dataset | Matcher | Depth | AR↑ | vs Oryon | Gain |
|--------|--------|------|------|-----------|------|
| REAL275 | Ours | Ours | **28.7** | 20.4 | +8.3 |
| Toyota-Light | Ours | Ours | **31.7** | 24.1 | +7.6 |
| ClearPose | Ours | Ours | **19.4** | 17.1 | +2.3 |
| Avg. (3 datasets) | - | - | - | - | +6.1 |

### Ablation Study

| Configuration | $\delta_{1.05}$↑ | Abs.Rel.↓ | RMSE↓ | Note |
|------|---------|-----------|-------|------|
| w/o local loss | 31.16 | 0.279 | 0.281 | Global loss only |
| +Edge+Norm | 40.23 | 0.139 | 0.162 | +Edge+Normal |
| +Scale+Edge | 40.41 | 0.124 | 0.140 | +Scale+Edge |
| Full (all) | **44.28** | **0.082** | **0.107** | All three losses contribute |

**Efficiency comparison**: The proposed method requires only 11.6M parameters / 13.9 GFLOPs / 0.74 GB VRAM, compared to Oryon's 264.3M / 120.1G / 5.90 GB—approximately 8× more computationally efficient.

### Key Findings

- Depth prediction accuracy on transparent objects (ClearPose) improves from 31.23% to 54.30%, a gain of 23%.
- Depth quality directly impacts pose accuracy: using oracle depth yields AR up to 56.8, indicating remaining headroom for depth improvement.
- Using only 50% of training data matches UniDepth's full-data performance.

## Highlights & Insights

- **Minimal-reference design philosophy**: No CAD models, multi-view images, neural fields, or diffusion-based generation—only a single RGB image is required.
- The token-scaler fine-tuning strategy preserves DPAv2's pretrained knowledge while achieving accurate metric depth, by freezing the backbone and training only a lightweight module.
- The three-level local loss (Scale + Edge + Normal) systematically addresses the primary sources of geometric error in depth prediction, with clear and well-motivated design.

## Limitations & Future Work

- A significant gap remains between oracle depth and predicted depth (AR: 56.8 vs. 28.7), making depth prediction the primary bottleneck.
- On Toyota-Light, using predicted depth underperforms Oryon + DPAv2, suggesting room for improvement in robustness to lighting variation.
- LoFTR parameters are frozen without fine-tuning; joint training may further improve matching quality.
- Evaluation under the full BOP Challenge standard protocol has not been conducted.

## Related Work & Insights

- **vs. FoundationPose**: FoundationPose requires training a neural field and multi-view templates with high computational cost; SingRef6D requires no view synthesis at all.
- **vs. NOPE**: NOPE trains a U-Net to synthesize 342 novel views and is limited to single-object textured scenes; SingRef6D supports multi-object complex scenes.
- **vs. Zero123-6D**: Zero123-6D uses diffusion models for novel view synthesis and NeRF for 3D reconstruction; SingRef6D avoids all such expensive steps.
- **vs. Oryon**: Oryon relies on CLIP text embeddings and sensor depth, failing on transparent objects; SingRef6D addresses this by learning depth priors.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The single-RGB-reference formulation is well-motivated and practical; the token-scaler fine-tuning and depth-aware matching are cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets covering diverse challenges, with detailed evaluation of both depth and pose, and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation and Table 1 clearly articulate the differences from prior work.
- **Value**: ⭐⭐⭐⭐ Strong practical value in low-resource settings, particularly for pose estimation of transparent and reflective objects.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes](../../ICCV2025/3d_vision/unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)
- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[ICCV 2025\] BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation](../../ICCV2025/3d_vision/boxdreamer_dreaming_box_corners_for_generalizable_object_pose_estimation.md)
- [\[ICCV 2025\] Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras](../../ICCV2025/3d_vision/single-scanline_relative_pose_estimation_for_rolling_shutter_cameras.md)
- [\[NeurIPS 2025\] Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)

<!-- RELATED:END -->
