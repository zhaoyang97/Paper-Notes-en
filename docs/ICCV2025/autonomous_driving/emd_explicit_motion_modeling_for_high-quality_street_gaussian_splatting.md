---
title: >-
  [Paper Note] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting
description: >-
  [ICCV 2025][Autonomous Driving][3D Gaussian Splatting] This paper proposes the Explicit Motion Decomposition (EMD) module, which models the motion characteristics of each Gaussian primitive via learnable motion embedding…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D Gaussian Splatting"
  - "Dynamic Scene Reconstruction"
  - "Motion Modeling"
  - "Self-Supervised Learning"
  - "Street View Simulation"
date: 2026-05-08
content_hash: d3a575e80e8867ef
---

# EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2411.15582](https://arxiv.org/abs/2411.15582)  
**Code**: [qingpowuwu.github.io/emd](https://qingpowuwu.github.io/emd)  
**Area**: Autonomous Driving
**Keywords**: 3D Gaussian Splatting, Dynamic Scene Reconstruction, Motion Modeling, Self-Supervised Learning, Street View Simulation

## TL;DR

This paper proposes the Explicit Motion Decomposition (EMD) module, which models the motion characteristics of each Gaussian primitive via learnable motion embeddings and a dual-scale deformation framework. As a plug-and-play module, EMD integrates seamlessly into both self-supervised and supervised street-view Gaussian splatting methods, achieving state-of-the-art performance under the self-supervised setting on the Waymo and KITTI datasets.

## Background & Motivation

### Problem Definition

Novel view synthesis of dynamic street scenes is a core technology for closed-loop autonomous driving simulation. Methods based on 3DGS/4DGS address street scene reconstruction by decomposing the scene into static and dynamic components; however, existing approaches fail to effectively model the heterogeneous motion patterns of dynamic objects.

### Limitations of Prior Work

**Supervised methods** (StreetGaussian, OmniRe):
- Use 3D bounding box supervision to classify scene elements as either "static" or "dynamic"
- Ignore the **continuous spectrum** of motion (e.g., pedestrian motion is far slower than vehicular motion)
- Although bounding box optimization partially mitigates dynamic reconstruction errors, motion patterns are not fundamentally modeled

**Self-supervised methods** (S3Gaussian, DeSiRe-GS):
- Optimize a holistic 4D street-scene representation via intrinsic motion cues
- Overlook inter-object differences in motion speed (e.g., pedestrians vs. vehicles)
- Lack effective motion modeling mechanisms, leading to blurry reconstruction of dynamic objects

### Core Motivation

Different object categories in street scenes exhibit fundamentally distinct motion patterns: vehicles undergo fast global motion, while pedestrians undergo slow local motion. An **explicit motion modeling mechanism** is needed to distinguish and handle these multi-scale motion patterns. The key insight is that assigning each Gaussian a learnable motion embedding and processing fast global motion and slow local motion separately through a dual-scale deformation network can significantly improve dynamic scene reconstruction quality.

## Method

### Overall Architecture

EMD is a plug-and-play module consisting of two core components:
1. **Motion-aware Feature Encoding**: Encodes spatial, temporal, and Gaussian-specific information into a unified motion-aware feature representation.
2. **Dual-scale Deformation Framework**: Hierarchically handles fast global motion and slow local deformation.

Given a set of static 3D Gaussian primitives $\mathbb{G} = \{(\mu_k, \mathbf{s}_k, \mathbf{q}_k, \alpha_k, \mathbf{c}_k)\}_{k=1}^K$ and a timestamp $t$, the goal is to learn a deformation field $\mathcal{D}$ that maps each Gaussian's parameters from a canonical state to the deformed state at time $t$.

### Key Designs

#### 1. **Motion-aware Feature Encoding**

- **Function**: Fuses the spatial position, temporal information, and individual motion characteristics of each Gaussian primitive into a comprehensive feature representation.
- **Mechanism**:

  The aggregated feature is formed by concatenating three components:
  $$\mathbf{F}_{aggr}(\mu, t) = [\mathbf{F}_{pos}(\mu), \mathbf{F}_{temp}(t), \mathbf{F}_{gauss}]$$

  **Spatial encoding** $\mathbf{F}_{pos}$: Multi-frequency positional encoding with $P=10$ frequency bands:
  $$\mathbf{F}_{pos}(\mu) = [\mu, \{\sin(2^i\pi\mu), \cos(2^i\pi\mu)\}_{i=0}^{P-1}]$$

  **Adaptive temporal embedding** $\mathbf{F}_{temp}$: Realized via a learnable embedding matrix $\mathbf{W} \in \mathbb{R}^{N_{max} \times D}$ and progressive temporal sampling:
  $$\mathbf{F}_{temp}(t) = \text{Interp}(\mathbf{W}, t, N(i))$$
  where $N(i)$ progressively grows from $N_{min}=30$ to $N_{max}=150$, controlled by training iteration $i$:
  $$N(i) = N_{min} + (N_{max} - N_{min}) \cdot \min(i, T) / T$$
  $D=4$ is the temporal embedding dimension, and $T=25000$ controls the duration of progressive sampling.

  **Gaussian embedding** $\mathbf{F}_{gauss}$: Each Gaussian $k$ is assigned a learnable latent variable $\mathbf{z}_k \in \mathbb{R}^M$ ($M=32$) encoding its individual motion characteristics.

- **Design Motivation**:
    - Multi-frequency spatial encoding captures multi-level information from fine geometry to global structure.
    - Progressive temporal sampling learns temporal dynamics from coarse to fine, preventing overfitting to high-frequency temporal changes in early training.
    - Gaussian embeddings enable instance-level motion representation — Gaussians belonging to the same moving object should learn similar embeddings.

#### 2. **Dual-scale Deformation Framework**

- **Function**: Decomposes deformation into two levels — coarse-scale (fast global motion) and fine-scale (slow local deformation).
- **Mechanism**:

  $$\mathcal{D}(\mu, t) = \mathcal{D}_{coarse}(\mathbf{F}_{aggr}(\mu, t)) + \mathcal{D}_{fine}(\mathbf{F}_{aggr}(\mu + \Delta\mu_{coarse}, t))$$

  The final deformation parameters combine predictions from both scales:
  $$\mu_t = \mu + \Delta\mu_{coarse} + \Delta\mu_{fine}$$
  $$\mathbf{s}_t = \mathbf{s} + \Delta\mathbf{s}_{coarse} + \Delta\mathbf{s}_{fine}$$
  $$\mathbf{q}_t = \mathbf{q} \otimes \Delta\mathbf{q}_{coarse} \otimes \Delta\mathbf{q}_{fine}$$

  Crucially, the fine-scale deformation network re-encodes spatial features using the **coarse-displaced position** $\mu + \Delta\mu_{coarse}$ as input.

- **Design Motivation**:
    - $\mathcal{D}_{coarse}$ focuses on large-scale motion such as vehicle translation, providing the primary deformation direction.
    - $\mathcal{D}_{fine}$ captures local details such as articulated motion on top of the coarse deformation.
    - The coarse-to-fine cascaded design allows the two networks to specialize, avoiding the difficulty of learning both large displacements and fine deformations within a single network.

#### 3. **Integration with Existing Frameworks**

- **Function**: Seamlessly integrates EMD into both self-supervised and supervised methods.
- **Mechanism**:

  **Self-supervised integration** (S3Gaussian, DeSiRe-GS):
  - A learnable embedding $\mathbf{z}_k$ is added to each Gaussian.
  - The original decoder is replaced with the dual-scale framework.
  - The original self-supervised deformation setup is retained.

  **Supervised integration** (StreetGaussian, OmniRe):
  - Dual-scale decomposition is introduced into bounding box pose optimization: $R'_t = \Delta R_t^f \cdot \Delta R_t^c \cdot R_t$, $T'_t = T_t + (\Delta T_t^c + \Delta T_t^f)$
  - Dual-scale refinement is also applied to OmniRe's non-rigid SMPL model: $\theta'_t = \Delta\theta_t^f \cdot \Delta\theta_t^c \cdot \theta_t$

- **Design Motivation**: The plug-and-play design enables EMD to augment any existing street-view Gaussian method without redesigning the entire pipeline.

### Loss & Training

- The same reconstruction losses as the respective baselines are used (photometric loss, etc.).
- An additional **local smoothness regularization on Gaussian embeddings** encourages spatially adjacent Gaussians to share similar embeddings:
  $$\mathcal{L}_{\mathbf{z}_k} = \frac{1}{d|\mathcal{U}|}\sum_{i \in \mathcal{U}}\sum_{j \in \text{KNN}_{i;d}} (e^{-\lambda_w\|\mu_j - \mu_i\|_2} \|\mathbf{z}_{k_i} - \mathbf{z}_{k_j}\|_2)$$
  where $\lambda_w = 2000$ and $d=20$ (number of KNN neighbors).
- Regularization on coarse/fine deformation magnitudes constrains deformation values close to zero, preventing excessively large deformations.

## Key Experimental Results

### Main Results

Self-supervised setting — comparison with S3Gaussian (Waymo-D32, scene reconstruction):

| Method | Full PSNR↑ | Full SSIM↑ | Full LPIPS↓ | Vehicle PSNR↑ | Vehicle SSIM↑ |
|--------|-----------|-----------|------------|--------------|--------------|
| EmerNeRF | 28.16 | 0.806 | 0.228 | 24.32 | 0.682 |
| 3DGS | 28.47 | 0.876 | 0.136 | 23.26 | 0.716 |
| S3Gaussian | 30.69 | 0.900 | 0.121 | 26.23 | 0.804 |
| **S3Gaussian+EMD** | **32.50** | **0.933** | **0.082** | **29.04** | **0.879** |

Self-supervised setting — comparison with DeSiRe-GS (Waymo, scene reconstruction):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS |
|--------|-------|-------|--------|-----|
| PVG | 32.46 | 0.910 | 0.229 | 50 |
| DeSiRe-GS | 33.61 | 0.919 | 0.204 | 36 |
| **DeSiRe-GS+EMD** | **34.15** | **0.948** | **0.130** | 32 |

Supervised setting — comparison with OmniRe (Waymo, 3 front cameras, novel view synthesis):

| Method | Full PSNR↑ | Human PSNR↑ | Vehicle PSNR↑ |
|--------|-----------|------------|--------------|
| OmniRe | 32.57 | 24.36 | 27.57 |
| **OmniRe+EMD** | **33.89** | **25.97** | **27.82** |

### Ablation Study

Contribution of each component (Waymo-D32, S3Gaussian baseline):

| Configuration | Full PSNR↑ | Vehicle PSNR↑ | Note |
|---------------|-----------|--------------|------|
| **Full model** | **32.50** | **29.04** | — |
| w/o Gaussian embedding | 32.21 (−0.29) | 28.80 | Individual motion characteristics not captured |
| w/o temporal embedding | 32.23 (−0.27) | 28.08 | Insufficient temporal dynamics modeling |
| w/o coarse-scale deformation | 29.40 (−3.10) | 24.54 | Unable to handle large displacements |
| w/o fine-scale deformation | 32.45 (−0.05) | 28.80 | Local detail lost |

Novel trajectory synthesis (FID↓, Waymo):

| Method | 0.5 m offset | 1.0 m offset | 1.5 m offset |
|--------|-------------|-------------|-------------|
| S3Gaussian | 83.48 | 110.11 | 134.38 |
| **S3Gaussian+EMD** | **45.11** | **70.26** | **90.20** |

### Key Findings

1. **EMD yields the greatest improvement on vehicle regions**: Vehicle PSNR increases from 26.23 to 29.04 (+2.81 dB), demonstrating that explicit motion modeling directly improves dynamic object reconstruction.
2. **Coarse-scale deformation is the most critical component**: Removing it causes PSNR to drop sharply from 32.50 to 29.40 (−3.10), confirming that large-scale motion modeling is central.
3. **Substantial improvement in novel trajectory synthesis**: FID drops from 83.48 to 45.11 at a 0.5 m offset, indicating that EMD's motion modeling improves simulation fidelity for lane-change scenarios.
4. **Demonstrated generality as a plug-and-play module**: Consistent improvements are observed across all four baselines: S3Gaussian, DeSiRe-GS, StreetGaussian, and OmniRe.
5. **Effective for human body modeling**: OmniRe+EMD improves Human PSNR from 24.36 to 25.97, confirming the dual-scale framework's applicability to non-rigid motion.
6. **Acceptable inference speed trade-off**: DeSiRe-GS+EMD runs at 32 FPS vs. the original 36 FPS, incurring only an 11% speed reduction.

## Highlights & Insights

1. **Motion continuum insight**: Object motion in street scenes is not a simple static/dynamic binary but exists along a continuous spectrum from stationary to high-speed.
2. **Elegance of the dual-scale design**: The coarse-to-fine cascade naturally corresponds to large displacements (vehicles) and small deformations (pedestrians, wheel rotation).
3. **Gaussian embedding + smoothness regularization**: Enables Gaussians belonging to the same object to automatically cluster into similar motion patterns without explicit object segmentation.
4. **Progressive temporal sampling**: Avoids interference from high-frequency temporal signals during early training, serving as a simple yet effective curriculum learning strategy.
5. **Practical plug-and-play engineering**: Existing methods can be enhanced by simply adding motion embeddings and replacing the decoder.

## Limitations & Future Work

1. **Increased computational cost**: The dual-scale deformation network and embedding smoothness regularization increase training time.
2. **No explicit modeling of inter-object relationships**: Each Gaussian models its motion independently, without considering motion correlations between objects.
3. **Fixed hyperparameters**: Progressive sampling parameters such as $N_{min}$, $N_{max}$, and $T$ are not adapted to individual scenes.
4. **Focused solely on appearance reconstruction**: Motion modeling is not applied to downstream tasks such as object detection or tracking.
5. **Limited multi-camera evaluation**: While Waymo supports 5 cameras, experiments are conducted primarily under 1–3 camera settings.

## Related Work & Insights

- **vs. S3Gaussian**: S3Gaussian optimizes the scene holistically without distinguishing motion patterns; EMD addresses this gap through explicit motion embeddings.
- **vs. StreetGaussian**: StreetGaussian handles dynamic reconstruction errors via bounding box optimization; EMD further refines motion modeling on this foundation.
- **Gaussian embedding concept** is analogous to latent codes in NeRF, but applied to motion modeling rather than appearance alone.
- The dual-scale deformation paradigm may inspire other multi-scale motion scenarios, such as indoor robotics or sports scene reconstruction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The explicit motion modeling concept is novel, though the technical components (dual-scale deformation, embeddings) are individually well-established.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four baselines, two datasets, both self-supervised and supervised settings, and novel trajectory synthesis evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, with rich figures and convincing visualizations.
- **Value**: ⭐⭐⭐⭐ — The plug-and-play design is highly practical, providing a standardized motion modeling component for street-scene reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DAMap: Distance-aware MapNet for High Quality HD Map Construction](damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad_gs_object_aware_bspline_gaussian_splatting_self_supervised_autonomous_driving.md)
- [\[ICCV 2025\] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)
- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)

</div>

<!-- RELATED:END -->
