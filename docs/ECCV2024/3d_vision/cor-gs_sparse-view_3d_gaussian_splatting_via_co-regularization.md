---
title: >-
  [Paper Note] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] Identifying that the disagreement in Gaussian locations and rendering results between two co-trained 3DGS radiance fields is negatively correlated with reconstruction quality, this paper proposes CoR-GS to suppress inaccurate reconstructions through co-pruning and pseudo-view co-regularization, achieving state-of-the-art sparse-view novel view synthesis.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Sparse-View"
  - "Novel View Synthesis"
  - "Co-Regularization"
  - "Point Cloud Pruning"
date: 2026-05-08
content_hash: a3a5aea127317602
---

# CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization

**Conference**: ECCV 2024  
**arXiv**: [2405.12110](https://arxiv.org/abs/2405.12110)  
**Code**: Yes ([https://jiaw-z.github.io/CoR-GS](https://jiaw-z.github.io/CoR-GS))  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Sparse-View, Novel View Synthesis, Co-Regularization, Point Cloud Pruning

## TL;DR

Identifying that the disagreement in Gaussian locations and rendering results between two co-trained 3DGS radiance fields is negatively correlated with reconstruction quality, this paper proposes CoR-GS to suppress inaccurate reconstructions through co-pruning and pseudo-view co-regularization, achieving state-of-the-art sparse-view novel view synthesis.

## Background & Motivation

3D Gaussian Splatting (3DGS) achieves high-quality real-time novel view synthesis by representing scenes with a set of 3D Gaussians. However, it suffers from overfitting under **sparse training views**, leading to degraded novel view rendering quality.

Existing methods (such as FSGS) mainly rely on external depth priors from pre-trained depth estimators for regularization, but external depth supervision may introduce additional noise. This paper proposes a novel perspective of **co-regularization**:

Core Discovery: When training two 3DGS radiance fields to represent the same scene, randomness in the implementation of densification causes them to exhibit disagreement in the following aspects:
- **Point disagreement**: differing Gaussian positions
- **Rendering disagreement**: differing rendered pixels

Key Insight: **These two types of disagreement are negatively correlated with reconstruction accuracy**—regions with larger disagreement are reconstructed less accurately. Thus, inaccurate reconstructed regions can be identified without ground-truth data.

## Method

### Overall Architecture

CoR-GS simultaneously trains two 3DGS radiance fields $\Theta^1$ and $\Theta^2$, performing co-regularization during training:

1. **Co-pruning**: Identifies and prunes Gaussians at inaccurate positions based on point disagreement.
2. **Pseudo-view Co-regularization**: Suppresses inconsistent rendering results on pseudo-views based on rendering disagreement.

After training, only one radiance field is retained for inference.

### Key Designs

#### 1. Point Disagreement Metric

Consider the Gaussian centers of the two radiance fields as two point clouds, and measure their difference using the following metrics:
- **Fitness**: Computes the overlapping region under a maximum distance of $\tau{=}5$.
- **RMSE**: Computes the average distance of matching points.

Experimental observation: Both types of disagreement grow significantly during densification because the density control process is blind to the scene geometry when creating new Gaussians.

#### 2. Negative Correlation Between Disagreement and Reconstruction Quality

By progressively masking out regions with the highest disagreement, the reconstruction quality (PSNR, SSIM) of the remaining regions consistently improves. This confirms that disagreement can serve as an unsupervised proxy metric for reconstruction quality.

#### 3. Co-Pruning

Perform co-pruning based on point disagreement:

- Establish point-matching relationships between the two radiance fields using KNN: $f(\theta_i^1) = \text{KNN}(\theta_i^1, \Theta^2)$
- Compute the unmatched mask $M$ given the maximum allowable distance $\tau{=}5$.
- Gaussians in one radiance field without matching neighbors in the other are considered **outliers in inaccurate positions** and are pruned.
- Co-pruning is executed once every 5 iterations of the optimization/densification alternation.

Effect: Reduces scattered Gaussians far from the reconstructed scene, making the representation more compact.

#### 4. Pseudo-view Co-regularization

Improving generalization by suppressing rendering disagreement on unseen views:

- Sample pseudo-views by interpolating between the two nearest training views: $P' = (t + \epsilon, q)$
- Render images $I'^1, I'^2$ from the two radiance fields on the pseudo-views.
- Compute their color difference as a regularization term:

$$\mathcal{R}_{pcolor} = (1-\lambda)\mathcal{L}_1(I'^1, I'^2) + \lambda\mathcal{L}_{D\text{-}SSIM}(I'^1, I'^2)$$

- Final training loss: $\mathcal{L} = \mathcal{L}_{color} + \lambda_p \mathcal{R}_{pcolor}$, where $\lambda_p{=}1.0$.

### Loss & Training

**Training view loss** (standard 3DGS):

$$\mathcal{L}_{color} = (1-\lambda)\mathcal{L}_1(I^1, I^*) + \lambda\mathcal{L}_{D\text{-}SSIM}(I^1, I^*)$$

$\lambda{=}0.2$, consistent with the original 3DGS.

**Total loss**: Training view GT supervision + pseudo-view co-regularization.

**Initialization**: Uses stereo-fused point clouds from sparse views (similar to FSGS) instead of COLMAP sparse points.

**Training settings**: Trained for 10K steps on LLFF/DTU/Blender, and 30K steps on Mip-NeRF360.

## Key Experimental Results

### Main Results (LLFF Dataset, 3/6/9 Views)

| Method | 3-view PSNR↑ | 3-view SSIM↑ | 3-view LPIPS↓ | 6-view PSNR↑ | 9-view PSNR↑ |
|---|---|---|---|---|---|
| FreeNeRF | 19.63 | 0.612 | 0.308 | 23.73 | 25.13 |
| 3DGS | 19.22 | 0.649 | 0.229 | 23.80 | 25.44 |
| FSGS | 20.43 | 0.682 | 0.248 | 24.09 | 25.31 |
| **CoR-GS** | **20.45** | **0.712** | **0.196** | **24.49** | **26.06** |

### DTU Dataset (3/6/9 Views)

| Method | 3-view PSNR↑ | 3-view SSIM↑ | 3-view LPIPS↓ |
|---|---|---|---|
| FreeNeRF | 19.92 | 0.787 | 0.182 |
| 3DGS | 17.65 | 0.816 | 0.146 |
| FSGS | - | - | - |
| **CoR-GS** | **19.21** | **0.853** | **0.119** |

### Efficiency Comparison (LLFF 3-view, RTX 3090 Ti)

| Method | Gaussian Count | FPS | PSNR↑ | Training Time |
|---|---|---|---|---|
| FreeNeRF | - | 0.09 | 19.63 | 2.3h |
| 3DGS | 1.16×10⁵ | 318 | 19.22 | 2.5min |
| **CoR-GS** | **7.85×10⁴** | **349** | **20.45** | 6min |

### Ablation Study

| Co-Pruning | Pseudo-view Co-reg | LLFF PSNR↑ | LLFF SSIM↑ | LLFF LPIPS↓ | DTU PSNR↑ |
|---|---|---|---|---|---|
| ✗ | ✗ | 19.22 | 0.649 | 0.229 | 17.65 |
| ✓ | ✗ | 19.62 | 0.673 | 0.217 | 18.59 |
| ✗ | ✓ | 20.26 | 0.706 | 0.198 | 18.56 |
| **✓** | **✓** | **20.45** | **0.712** | **0.196** | **19.21** |

### Key Findings

1. The two forms of regularization are complementary: co-pruning eliminates outlier Gaussians far from the scene, while pseudo-view co-regularization corrects Gaussians that have plausible positions but produce inaccurate renderings.
2. CoR-GS reduces the number of Gaussians by 33% (1.16e5 $\rightarrow$ 7.85e4), which inversely accelerates inference speed (349 FPS vs 318 FPS).
3. At 9 views, FSGS performance drops compared to vanilla 3DGS due to noise introduced by depth priors, whereas CoR-GS achieves consistent improvements across all view counts.
4. It remains effective in 360° scenes (Mip-NeRF360), improving 12-view PSNR from 18.52 (3DGS) to 19.52.

## Highlights & Insights

1. **A Novel Regularization Perspective**: Exploits the random discrepancies during the co-training of two models as an unsupervised quality metric, which is elegant and free from external priors.
2. **In-depth Analysis of "Randomness"**: Identifies that the random sampling in density control is the source of incorrect geometry in sparse-view 3DGS, providing highly insightful observations.
3. **Compact Representation**: Not only improves quality but also cuts down the number of Gaussians by 1/3, enabling faster inference.
4. **Strong Generalizability**: Demonstrates effectiveness across diverse benchmarks including LLFF (forward-facing scenes), Mip-NeRF360 (360° scenes), DTU (objects), and Blender (synthetic).
5. Shares similar intuition with co-training/mutual teaching in machine learning, but represents the first application in 3DGS.

## Limitations & Future Work

1. Training two radiance fields doubles the training time (2.5 mins $\rightarrow$ 6 mins), hindering real-time deployment.
2. Sampling pseudo-views solely by interpolating between the two nearest training views may fail to cover all critical unobserved regions.
3. The distance threshold $\tau$ for co-pruning is fixed, which might not generalize to all scene scales.
4. It is not integrated with depth-prior methods, despite their potential complementarity.

## Related Work & Insights

- **FSGS**: Relies on external depth priors, where depth noise can negatively affect geometry; CoR-GS requires no external supervision.
- **Co-training Philosophy** (Blum & Mitchell 1998): Two learners correct each other, which is introduced to 3DGS in this work.
- **Prediction Agreement in Semi-Supervised Learning**: Leverages the consistency of predictions from two networks for pseudo-labeling or noise filtering, which this work analogizes to 3D reconstruction.
- Insight: **The randomness of model training itself can serve as a useful signal.**

## Rating

| Dimension | Score (1-10) |
|---|---|
| Novelty | 8 |
| Technical Depth | 7 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Value | 8 |
| **Overall Score** | **8.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)
- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [\[ECCV 2024\] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](../../CVPR2026/3d_vision/dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](../../CVPR2025/3d_vision/s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
