---
title: >-
  [Paper Note] Towards Realistic Example-Based Modeling via 3D Gaussian Stitching
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Stitching] Proposes the first realistic example-based modeling method based on 3D Gaussian representation. It achieves seamless stitching and harmonious appearance fusion of multiple 3D Gaussian fields through sample-based cloning (S-phase) and cluster-based tuning (T-phase), supporting interactive real-time editing.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Stitching"
  - "Example-based Modeling"
  - "Appearance Harmonization"
  - "Interactive Editing"
  - "Sample-based Cloning"
date: 2026-05-08
content_hash: f971e98239ed58f6
---

# Towards Realistic Example-Based Modeling via 3D Gaussian Stitching

**Conference**: CVPR 2025  
**arXiv**: [2408.15708](https://arxiv.org/abs/2408.15708)  
**Code**: None  
**Area**: 3D Vision / Neural Rendering  
**Keywords**: 3D Gaussian Stitching, Example-based Modeling, Appearance Harmonization, Interactive Editing, Sample-based Cloning

## TL;DR

Proposes the first realistic example-based modeling method based on 3D Gaussian representation. It achieves seamless stitching and harmonious appearance fusion of multiple 3D Gaussian fields through sample-based cloning (S-phase) and cluster-based tuning (T-phase), supporting interactive real-time editing.

## Background & Motivation

Example-based modeling is a classic methodology in computer graphics—creating new models by selecting and combining parts from different models. With the development of NeRF/3DGS technologies, realistic example-based modeling directly from real-world scenes has become possible. However, existing methods face the following challenges:

1. **Inharmonious Appearance**: Simply placing multiple neural fields together without addressing appearance fusion leads to inconsistent tones and textures at the junctions.
2. **Limitations of SeamlessNeRF**: The only prior work attempting to solve seamless stitching uses a gradient propagation-based strategy combined with a mesh representation, which suffers from three issues: the inability to perform interactive editing, prominent artifacts in real-world scenes, and the inability to propagate complex structural features.
3. **Discrete Challenges of 3DGS**: 3D Gaussians are discrete, irregularly distributed point representations, which do not support gradient propagation like NeRF's implicit grids, thus requiring an entirely new fusion strategy.

The core insight of this work is to leverage the point cloud characteristics of 3DGS to achieve fine-grained editing and real-time previewing, while replacing gradient propagation with a sampling-based strategy to enable seamless feature propagation.

## Method

### Overall Architecture

The pipeline consists of four steps: (1) using a customized GUI to segment and perform rigid transformations on multiple pre-trained 3DGS scenes to obtain semantically reasonable combinations; (2) KNN analysis to identify boundary points in the intersecting regions of the source and target fields; (3) S-phase (sample-based cloning): explicitly propagating boundary color and structural features to all points in the target field through neighborhood sampling; (4) T-phase (cluster-based tuning): performing global tone harmonization using a color palette extracted from the source field. All steps can be operated interactively and previewed in real-time within the GUI.

### Key Designs

1. **KNN Boundary Condition Identification**:
    - Function: Automatically identifies boundary points in the target field located in the intersection area after combination.
    - Mechanism: For each Gaussian point $a$ in the target field $\mathcal{T}$, its K-nearest neighbors $\{b_i\}_K$ in the source field $\mathcal{S}$ are searched. When the average distance $\frac{1}{K}\sum|b_i - a| < \beta$ and the opacity $o(a) > \tau$, it is marked as a boundary point. The target features of the boundary points are determined by the mean of the SH coefficients of the K-nearest neighbors from the source field, optimized via the $\mathcal{L}_{feature}$ loss.
    - Design Motivation: The boundary condition is the initial state of harmonization, and precise boundary extraction directly determines the quality of subsequent optimization. $\tau=0.95$ filters out low-opacity noise points, and $\beta=0.05 \times L$ (where $L$ is the size of the combined object) ensures a reasonable boundary width.

2. **Sample-Based Cloning (S-phase)**:
    - Function: Seamlessly propagates boundary color and structural features to all non-boundary points in the target field.
    - Mechanism: For each non-boundary point $a$, its position is perturbed using the mapping function $\phi(x) = x + \sin(\gamma \cdot \delta x)$ (where $\gamma=10$ and $\delta x$ is the distance to the nearest boundary point), and its K-nearest neighbors in the boundary point set are searched as "driving points". The view-dependent SH colors of the driving points serve as the optimization target (color loss) for $a$. Simultaneously, 2D gradients are pre-computed using the Sobel operator within the local space of the target field to apply a gradient preservation loss, retaining the original texture content.
    - Design Motivation: Naive Laplacian methods (using neighborhood differences as regularization) have gradients that are too weak to initiate propagation. Explicit sampling provides sufficient driving force via the color loss. Sinusoidal perturbation introduces randomness to make texture propagation more natural, and high $\gamma$ is suitable for high-frequency structures.

3. **Cluster-Based Tuning (T-phase)**:
    - Function: Globally harmonizes the tone, brightness, and saturation of the combined object.
    - Mechanism: A color palette (cluster centers $c_i$ + weights $w_i$) is extracted from multi-view renderings of the source field via streaming clustering. Starting with 3 bins, it is progressively expanded, with centers updated by the average of new samples, and centers with insufficient votes for 20 iterations are expired. For each rendered pixel in the target field, the nearest palette center is matched, and a weighted L2 loss is applied. This is only performed for high-opacity ($\alpha > 0.95$) pixels.
    - Design Motivation: S-phase ensures local consistency but may result in global tone imbalances (such as shifts in brightness/hue). T-phase achieves global color alignment through palette matching.

### Loss & Training

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{feature} + \mathcal{L}_{color} + \lambda_1\mathcal{L}_{grad} + \lambda_2\mathcal{L}_{tune}$

$\lambda_1 = \lambda_2 = 2$. The T-phase loss is introduced after the S-phase has run for a certain duration (joint optimization, while the S-phase loss is maintained throughout). Camera centers are uniformly sampled on a sphere centered at the origin of the combined object.

## Key Experimental Results

### Main Results

| Method | Average VQA Score↑ |
|------|------------|
| SeamlessNeRF | 0.753 |
| **Ours** | **0.784** |

The experiments utilize 21 combinations involving 39 part models: 17 BlendedMVS + 4 Mip360 + 16 SeamlessNeRF + 2 custom-built models.

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Without color loss (S-phase) | Propagation fails to start | The gradient of the feature loss is too weak |
| S-phase only | Seamless boundaries but global tone imbalance | Locally harmonious, globally discordant |
| T-phase only | Lack of structural propagation | Color matching alone is insufficient |
| S+T (Full) | Optimal | Two stages are complementary |
| High-weight gradient loss | Retains more content but impedes harmonization | Balance is required |
| Without $\phi$ perturbation | Mechanical texture propagation | Randomness improves realism |
| 2D Sobel vs 3D gradient | 2D is more effective | Focusing on the surface is more reasonable |
| Random vs Strategic view sampling | Strategic sampling better preserves view-dependent effects | Correctly propagates view-dependent colors |

### Key Findings

- In the S-phase, the color loss is indispensable; the propagation cannot start with the feature loss alone.
- The 2D screen-space Sobel gradient loss is more effective than the 3D-space gradient (focusing on the visible surface).
- The sinusoidal perturbation function $\phi$ makes texture propagation more natural, avoiding mechanical replication.
- SeamlessNeRF fails in all real-world scenes, proving the limitations of gradient propagation methods in complex scenes.
- 2D style transfer methods also fail to achieve seamless stitching.

## Highlights & Insights

- **First realistic part-combination method for 3DGS**: Fills the gap in 3DGS example-based modeling.
- **Sample-based cloning** elegantly solves the challenge of gradient propagation being inapplicable to the discrete and irregular distribution of 3DGS.
- **S+T two-stage design** decomposes the problem into local seamlessness (S) and global harmonization (T).
- **Interactive GUI** is highly practical, supporting real-time preview of the entire process from segmentation, transformation, and boundary identification to optimization.
- **Correct propagation of view dependency**: Spherical sampling + SH color sampling ensure consistency of effects like reflections.

## Limitations & Future Work

- Only supports rigid-body transformations and lacks support for non-rigid deformations (e.g., ARAP), limiting creative flexibility.
- Does not consider illumination consistency; the quality of combinations may degrade under strong lighting conditions.
- Lacks standardized quantitative evaluation metrics and ground truth (GT), meaning the VQA evaluation has limitations.
- Future work can integrate with deformation methods and illumination estimation methods.

## Related Work & Insights

- **vs SeamlessNeRF**: SeamlessNeRF fails in real-world scenes due to its gradient propagation and mesh representation; this work completely overcomes these issues with a sampling strategy and point-cloud representation.
- **vs Neural Imposter**: Neural Imposter only places objects without fusion, whereas this work achieves harmonious and seamless stitching.
- **vs SNeRF Style Transfer**: 2D style transfer fails to achieve 3D-consistent seamless effects.

## Rating

- Novelty: ⭐⭐⭐⭐ First 3DGS example-based modeling method, with sample-based cloning as the core innovation.
- Experimental Thoroughness: ⭐⭐⭐ Detailed ablations, but quantitative evaluation is limited by the lack of a benchmark.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and intuitive ablation visualization.
- Value: ⭐⭐⭐⭐ Opens a new direction for 3DGS editing, with a highly practical interactive design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](vid2sim_realistic_and_interactive_simulation_from_video_for_urban_navigation.md)
- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [\[ICCV 2025\] From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos](../../ICCV2025/3d_vision/from_gallery_to_wrist_realistic_3d_bracelet_insertion_in_videos.md)
- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)

</div>

<!-- RELATED:END -->
