---
title: >-
  [Paper Note] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D Vision][3DGS] AsymGS leverages a key observation—that reconstruction artifacts caused by in-the-wild training data are stochastic in nature—and proposes an asymmetric dual 3DGS framework that suppresses…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3DGS"
  - "in-the-wild scene reconstruction"
  - "dual-model consistency"
  - "transient distractors"
  - "EMA proxy"
date: 2026-05-08
content_hash: f8173920cc08f4c5
---

# Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2506.03538](https://arxiv.org/abs/2506.03538)  
**Code**: [GitHub](https://steveli88.github.io/AsymGS)  
**Area**: 3D Vision / Neural Rendering
**Keywords**: 3DGS, in-the-wild scene reconstruction, dual-model consistency, transient distractors, EMA proxy

## TL;DR

AsymGS leverages a key observation—that reconstruction artifacts caused by in-the-wild training data are stochastic in nature—and proposes an asymmetric dual 3DGS framework that suppresses artifacts via complementary masking strategies and consistency constraints. A Dynamic EMA Proxy is introduced for efficient training, achieving significant improvements over existing methods on multiple in-the-wild benchmarks.

## Background & Motivation

3D Gaussian Splatting (3DGS) achieves high-quality scene reconstruction under ideal conditions, but real-world in-the-wild images commonly contain illumination variations and transient distractors (pedestrians, vehicles, etc.), which introduce noisy supervision signals and degrade reconstruction quality.

Core limitations of existing methods:
- Per-image appearance embeddings are only weakly supervised via photometric loss, lacking direct constraints
- Outlier filtering relies on hand-crafted rules with limited stability and generalizability
- Single-model approaches cannot distinguish true scene structure from artifacts induced by data noise

**Core Insight**: Artifacts caused by in-the-wild data are **stochastic**—when the same scene is trained multiple times with different training orders, the true structure remains consistent while artifacts vary across runs. This implies that cross-model consistency can be exploited to filter artifacts.

## Method

### Overall Architecture

AsymGS maintains two 3DGS models trained in parallel, encouraging convergence on reliable scene structures through consistency constraints, while employing an asymmetric masking strategy to prevent confirmation bias.

### Key Designs

1. **Dual 3DGS with Mutual Consistency Constraint**:

    - Two Gaussian sets $\mathbb{G}_1$ and $\mathbb{G}_2$ are maintained and sampled from different view lists at each training iteration.
    - Mutual consistency loss: the renderings of both models from the same viewpoint (using view-independent rendering, without appearance transformation) are required to remain consistent.
    - Key point: the consistency constraint is applied on view-independent renderings $\hat{\mathbf{I}}$ rather than view-dependent renderings $\tilde{\mathbf{I}}$, as the former reflects the intrinsic scene appearance.
    - A progressive strategy is adopted: models are first warmed up independently before consistency constraints are introduced.

2. **Asymmetric Masking Strategy**:

    - **Multi-Cue Adaptive Mask ($\mathbf{M}_h$)**: A hard mask integrating multiple cues to identify transient regions:
        - SAM segments semantic regions
        - COLMAP multi-view stereo detects static content (regions with sufficient matches are considered static)
        - Pixel-level residuals (L1 reconstruction error) and feature-level residuals (DINOv2 cosine distance) jointly identify distractors
    - **Self-Supervised Soft Mask ($\mathbf{M}_s$)**: A learnable soft mask with continuous values in $[0, 1]$
        - Self-supervised via cosine similarity of DINOv2 features
        - Initialized to all ones and progressively refined during training; more sensitive to ambiguous regions
    - The two models use $\mathbf{M}_h$ and $\mathbf{M}_s$ respectively, introducing complementary inductive biases to prevent convergence to the same errors.

3. **Dynamic EMA Proxy**:

    - Replaces the second 3DGS model with a dynamic EMA copy, substantially reducing computational overhead.
    - **Dynamic update mechanism**: handles clone, split, and prune operations inherent to 3DGS training:
        - Clone: EMA attributes are cloned accordingly
        - Prune: corresponding EMA entries are removed synchronously
        - Split: position and variance are reinitialized; remaining attributes are inherited
    - **Alternating mask strategy**: under the single-model setting, $\mathbf{M}_h$ and $\mathbf{M}_s$ are applied alternately to maintain training diversity.

### Loss & Training

Total loss (GS-GS variant):
$$\mathcal{L} = \mathcal{L}_{r1}^{\mathbf{M}_h} + \mathcal{L}_{r2}^{\mathbf{M}_s} + \lambda_m(\mathcal{L}_{m1} + \mathcal{L}_{m2}) + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}$$

- Reconstruction loss: DSSIM + L1, with distractor regions filtered via masks
- Mutual consistency loss: L1 only (DSSIM degrades performance)
- Mask loss: L1 loss on DINOv2 feature cosine similarity
- Appearance modeling: following WildGaussian, per-Gaussian and per-view embeddings with an MLP predict affine transformation parameters

## Key Experimental Results

### Main Results

| Dataset | Metric | AsymGS (GS-GS) | AsymGS (EMA-GS) | HybridGS (Prev. SOTA) | Training Time |
|--------|------|-----------------|-----------------|---------------------|----------|
| NeRF On-the-go (High Occ.) | PSNR↑ | **24.34** | 24.12 | 23.05 | GS-GS: 0.28h |
| NeRF On-the-go (Med. Occ.) | PSNR↑ | **24.56** | 24.32 | 23.51 | EMA-GS: 0.18h |
| NeRF On-the-go (Low Occ.) | PSNR↑ | **21.91** | 21.77 | 21.42 | - |
| RobustNeRF (Yoda) | PSNR↑ | **37.18** | - | 35.32 | 0.31h |
| RobustNeRF (Crab) | PSNR↑ | **36.18** | - | 35.17 | - |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Note |
|------|-------|-------|--------|------|
| Single 3DGS (baseline) | ~19.0 | ~0.65 | ~0.34 | No processing |
| + Appearance embedding (WildGaussian) | 23.03 | 0.771 | 0.172 | Appearance modeling effective |
| + Dual model + consistency | Gain | Gain | Reduction | Mutual consistency suppresses artifacts |
| + Asymmetric masking | **24.34** | **0.825** | **0.150** | Complementary masks further improve results |

### Key Findings

- The asymmetric masking strategy is critical for preventing confirmation bias; symmetric designs (both models using identical masks) show noticeably degraded performance.
- EMA-GS achieves performance close to GS-GS with only ~50% additional training time over the single-model baseline, saving approximately one-third of the training time compared to GS-GS.
- Introducing consistency constraints too early impairs convergence; a progressive strategy is necessary.
- Applying consistency constraints on view-independent renderings is more effective than on view-dependent renderings.

## Highlights & Insights

- **Observation-driven design**: The insight that artifacts are stochastic is both concise and profound, directly motivating the dual-model consistency framework.
- **Asymmetric training prevents confirmation bias**: The hard mask is deterministic but potentially overconfident, while the soft mask is flexible but less precise; their complementary use is an elegant design choice.
- **Dynamic EMA adapted for 3DGS**: Standard EMA assumes a fixed number of parameters, whereas 3DGS involves clone/split/prune operations; the dynamic EMA mechanism is a practical engineering innovation.
- **Efficiency**: EMA-GS reduces training time on PhotoTourism from the previous SOTA's 2.9h to a comparable level while achieving higher quality.

## Limitations & Future Work

- The dual-model variant (GS-GS) still incurs considerable training overhead, though EMA-GS substantially mitigates this.
- The multi-cue hard mask depends on external tools such as SAM and COLMAP.
- Validation on dynamic scenes (e.g., consecutive frames in autonomous driving) has not been conducted.

## Related Work & Insights

- **vs. WildGaussian**: WildGaussian employs per-Gaussian appearance embeddings to handle illumination changes but does not address transient artifacts; AsymGS builds upon this by adding dual-model consistency.
- **vs. HybridGS**: HybridGS uses 3DGS + 2DGS to separately model static and dynamic content, but the dual-representation design is complex; AsymGS addresses the problem more principally through statistical consistency.
- **vs. SpotlessSplats**: SpotlessSplats learns masks based on thresholded residuals; AsymGS's multi-cue adaptive mask is more robust.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Both the stochasticity observation and the asymmetric dual-model design are highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three datasets with detailed ablations and training efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow, progressing coherently from observation to method to implementation.
- Value: ⭐⭐⭐⭐ Provides an effective and efficient solution for in-the-wild 3DGS reconstruction; EMA-GS offers strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](../../ICCV2025/3d_vision/robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](../../ICCV2025/3d_vision/longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICCV 2025\] Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts](../../ICCV2025/3d_vision/learning_robust_stereo_matching_in_the_wild_with_selective_mixture-of-experts.md)
- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)

</div>

<!-- RELATED:END -->
