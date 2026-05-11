---
title: >-
  [Paper Note] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction
description: >-
  [ICCV 2025][3D Vision][3DGS] This paper proposes a robust and efficient 3DGS reconstruction framework for city-scale scenes. Through a visibility-based partitioning strategy, controllable LOD generation…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3DGS"
  - "urban scene reconstruction"
  - "LOD strategy"
  - "appearance transformation"
  - "partitioned training"
  - "real-time rendering"
date: 2026-05-08
content_hash: d6f68787c60c57c2
---

# Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2507.23006](https://arxiv.org/abs/2507.23006)
**Code**: [https://yzslab.github.io/REUrbanGS](https://yzslab.github.io/REUrbanGS)
**Area**: 3D Vision
**Keywords**: 3DGS, urban scene reconstruction, LOD strategy, appearance transformation, partitioned training, real-time rendering

## TL;DR
This paper proposes a robust and efficient 3DGS reconstruction framework for city-scale scenes. Through a visibility-based partitioning strategy, controllable LOD generation, a fine-grained appearance transformation module, and multiple regularization techniques, the framework achieves high-quality reconstruction and real-time rendering on urban data with large appearance variations and transient objects.

## Background & Motivation

### Core Problem
City-scale scene reconstruction is critical for autonomous driving, urban planning, and digital twins. 3DGS has become a mainstream choice due to its explicit representation and real-time rendering capability, but scaling it to urban scenes poses three major challenges:

### Challenge Analysis

**Scalability Bottleneck**: Larger scenes require more Gaussians. Reconstructing the MipNeRF360 Bicycle scene requires 6M+ Gaussians; exceeding 11M on a 24GB GPU causes OOM. The problem is far more severe for urban scenes.

**Appearance Inconsistency**: Urban data is collected across time (seasonal/weather/lighting variations), causing significant appearance differences for the same object across images. 3DGS tends to create redundant Gaussians to account for per-viewpoint appearance differences, producing floating artifacts.

**Transient Object Interference**: Pedestrians, vehicles, and other transient objects are unavoidable and further introduce artifacts.

### Limitations of Prior Work
- **VastGaussian**: Uses a CNN for color transformation to handle appearance differences, but image-level transformation is unstable and inflexible; does not address real-time rendering.
- **CityGaussian/Hierarchical-3DGS**: Does not constrain resources during training; relies on post-processing compression with extensive fine-tuning, leading to severe quality degradation at high compression ratios for large scenes.
- **Taming3DGS**: Controls the densification strategy but is limited to small scenes.
- **Grendel-GS**: A multi-GPU solution where hardware requirements scale linearly with scene size, which is impractical.

## Method

### Overall Architecture
The framework enhances the full pipeline from preprocessing to training to rendering on top of vanilla 3DGS:
1. **Scene partitioning + visibility-based image selection** (preprocessing efficiency)
2. **Partition-prioritized densification** (training efficiency)
3. **Controllable LOD generation + dynamic selection** (rendering efficiency)
4. **Appearance transformation + regularization** (reconstruction quality)

### Key Designs

#### 1. Scene Partitioning and Visibility-Based Image Selection
After horizontal partitioning, **point-based visibility** is computed for images outside the partition:
- SfM generates a 3D point cloud and 2D feature point associations.
- 3D points are projected onto out-of-partition image planes, and the convex hull area $V_i$ is computed.
- Feature points within the partition are extracted to compute the convex hull area $V_{ij}$.
- Visibility $= V_{ij}/V_i$; only high-visibility images participate in training.
- Feature points are naturally occlusion-aware, avoiding redundant image selection.

**Partition Rebalancing**: Central partitions contain more images than edge partitions. Partitions with too few images are merged with their smallest neighbor; those with too many are subdivided. This process iterates until the distribution is uniform.

#### 2. Partition-Prioritized Densification
Regions outside the partition do not require excessive resource allocation, but naively raising the densification threshold causes Gaussians inside the partition to expand outward to compensate. A distance-dependent threshold is proposed:
$$\tau_i = \hat{\tau}_{min} \left(\frac{\min(d_i, \hat{d}_{max})}{\hat{d}_{max}} \cdot (\eta - 1) + 1\right)$$
where $d_i$ is the distance from the $i$-th Gaussian to the partition boundary. The in-partition threshold is $\hat{\tau}_{min}$, linearly increasing to $\hat{\tau}_{max} = \hat{\tau}_{min} \cdot \eta$ at distance. Densification occurs only when the mean gradient $\bar{\Delta}_{G_i} > \tau_i$.

#### 3. Controllable LOD Generation (Bottom-Up)
Vanilla 3DGS lacks resource constraints. The paper extends Taming3DGS's controllable densification strategy and defines multi-level LOD parameters:
- Budgets $B_1 < B_2 < \cdots < B_l$
- Densification intervals $T_1 > T_2 > \cdots > T_l$
- Image downsampling factors $D_1 < D_2 < \cdots < D_l = 1$

Lower levels are trained with smaller budgets, longer intervals, and lower resolutions, without attending to high-frequency details. After each level completes, a checkpoint is saved and parameters are switched to train the next level — fully end-to-end, with no post-processing compression required.

**Dynamic LOD Selection at Rendering**: LOD levels are selected based on partition-camera distance; higher levels are used for nearby regions and lower levels for distant ones, with invisible partitions culled. Tile-based culling from StopThePop further accelerates rendering.

#### 4. Appearance Transformation Module (Fine-Grained)
Separate embeddings $\ell^{(\mathcal{I})}$ and $\ell^{(\mathcal{G})}$ are assigned to each image and each 3D Gaussian, respectively. A lightweight MLP predicts per-Gaussian color offsets $\Delta c$ and opacity offsets $\Delta o$:

**Similarity Regularization** (neighboring Gaussians should have similar appearance transformations):
$$\mathcal{L}_{sim} = \frac{1}{|M|\binom{k}{2}} \sum_{i \in M} \sum_{j,l \in knn_{i;k}} w_{i,j} \left(1 - \frac{\ell_i^{(\mathcal{G})} \cdot \ell_j^{(\mathcal{G})}}{\|\ell_i^{(\mathcal{G})}\| \|\ell_j^{(\mathcal{G})}\|}\right)$$
where $w_{i,j} = \exp(-\lambda_w \|\mu_i - \mu_j\|)$ is a distance decay factor.

**Opacity Offset Regularization** (most appearance transformations do not involve transparency changes):
$$\mathcal{L}_{\Delta o} = \frac{1}{N} \sum_{i=1}^N \Delta o_i$$

#### 5. Scale Regularization (Suppressing Abnormal Gaussians)
**Maximum Scale Constraint** (prevents Gaussians from growing to unreasonable sizes):
$$\mathcal{L}_{ms} = \frac{\sum_i \mathbb{1}\{S_i > s_{max}\} \cdot S_i}{\sum_i \mathbb{1}\{S_i > s_{max}\} + \delta}$$

**Aspect Ratio Constraint** (prevents highly anisotropic shapes):
$$r_i = \frac{\max(S_i)}{\text{median}(S_i)}, \quad \mathcal{L}_r = \frac{\sum_i \mathbb{1}\{r_i > r_{max}\} \cdot r_i}{\sum_i \mathbb{1}\{r_i > r_{max}\} + \delta}$$

#### 6. Depth Regularization + Anti-Aliasing
Pseudo-depth is predicted using Depth Anything V2 and aligned to metric scale via SfM point clouds. Hard and soft depth regularization are applied alternately. Anti-aliasing is adopted from Mip-Splatting, and detail enhancement from AbsGS.

### Total Loss
$$\mathcal{L}' = \mathcal{L}_{3DGS} + 0.2\mathcal{L}_{sim} + 0.05\mathcal{L}_{\Delta o} + \lambda_d \mathcal{L}_d + 0.05(\mathcal{L}_{ms} + \mathcal{L}_r)$$
$\lambda_d$ decays exponentially from 0.5 to 0.01.

## Experiments

### Main Results: Three City-Scale Scenes

| Method | Rubble SSIM/PSNR | JNU-ZH SSIM/PSNR | BigCity SSIM/PSNR |
|------|------|------|------|
| Switch-NeRF | 0.544/23.05 | 0.574/21.96 | 0.469/20.39 |
| CityGaussian (no LOD) | 0.813/25.77 | 0.776/22.57 | 0.825/24.57 |
| 3DGS | 0.796/25.72 | 0.763/22.02 | 0.830/24.52 |
| **Ours (no LOD)** | **0.826/27.29** | **0.822/25.85** | **0.847/26.62** |
| CityGaussian (LOD) | 0.785/24.90 | 0.770/22.33 | 0.712/22.24 |
| Hierarchical-3DGS (LOD) | 0.741/23.38 | 0.760/21.12 | 0.775/23.17 |
| **Ours (LOD)** | **0.814/27.03** | **0.816/25.71** | **0.838/26.41** |

Key finding: In LOD mode, the proposed method still outperforms other methods' non-LOD results, while achieving significantly higher FPS of 63–100.

### LOD Budget Ablation

| Budget B (×100) | SSIM | PSNR | #G(M) | FPS |
|------|------|------|------|------|
| (1024,2048,4096) | 0.771 | 26.13 | 1.61 | 126.9 |
| (4096,8192,16384) | 0.814 | 27.03 | 3.60 | 99.7 |
| (8192,16384,32768) | 0.816 | 27.11 | 3.80 | 96.4 |

Quality stops improving beyond a certain budget threshold, indicating that the intrinsic complexity of the scene determines the upper bound on the required number of Gaussians.

### Component Ablation

| Configuration | Rubble SSIM/PSNR | JNU-ZH SSIM/PSNR |
|------|------|------|
| w/o visibility-based image selection | 0.803/26.95 | 0.809/25.14 |
| w/o appearance module | 0.771/25.17 | 0.780/22.57 |
| **Full method** | **0.826/27.29** | **0.822/25.85** |

The appearance module yields the most significant improvement on the JNU-ZH dataset, which is collected across different times (PSNR +3.28 dB).

## Highlights & Insights
1. **Full-pipeline system design**: A complete technical stack spanning partitioning → densification → LOD → appearance → regularization, where each component collaboratively addresses urban scene reconstruction.
2. **Bottom-up LOD** outperforms train-then-compress: lower levels are trained on low-frequency information, and higher levels incrementally refine from lower ones, yielding higher quality while avoiding compression loss.
3. **Gaussian-level appearance transformation** is more fine-grained and flexible than image-level: after reconstruction, appearance editing can be performed by modifying image embeddings without affecting rendering speed.
4. **Visibility-based image selection**: Feature points are naturally occlusion-aware, making this approach more principled than distance- or frustum-based selection strategies.

## Limitations & Future Work
1. Multiple hyperparameters across components (number of LOD levels, budgets, partition thresholds, etc.) may require scene-specific tuning.
2. The pipeline depends on SfM preprocessing and Depth Anything V2 predictions, making it relatively lengthy.
3. Transient object removal relies on an open-vocabulary detector, which may miss atypical dynamic objects.
4. Evaluation is limited to aerial scenes; street-level scenes (e.g., autonomous driving perspectives) are not validated.

## Related Work & Insights
- **Large-scale reconstruction**: Block-NeRF (divide-and-conquer NeRF) → VastGaussian (divide-and-conquer 3DGS) → CityGaussian/Hierarchical-3DGS (LOD-3DGS)
- **Appearance modeling**: NeRF-W (image embedding) → SWAG → this work (Gaussian-level embedding + dual-stream MLP)
- **Resource control**: Taming3DGS (controllable densification) → this work (LOD extension)

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Systematic integration rather than a single breakthrough, though the LOD strategy and appearance module designs are innovative.
- **Technical Depth**: ⭐⭐⭐⭐ — Each component is well-motivated and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-scene evaluation with ablations, though comparisons with the latest methods (e.g., Grendel-GS) are missing.
- **Practical Value**: ⭐⭐⭐⭐⭐ — Directly addresses engineering challenges in city-scale reconstruction and real-time rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[ICCV 2025\] StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions](stealthattack_robust_3d_gaussian_splatting_poisoning_via_density-guided_illusion.md)
- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)

</div>

<!-- RELATED:END -->
