---
title: >-
  [Paper Note] Seeing A 3D World in A Grain of Sand
description: >-
  [CVPR 2025][3D Vision][Miniature Scene Reconstruction] A catadioptric imaging system based on eight pairs of planar mirrors is designed to capture 360° surrounding multi-view images of miniature scenes in a single snapshot, combining visual hull depth constraints to improve sparse-view 3DGS reconstruction quality.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Miniature Scene Reconstruction"
  - "Catadioptric Imaging"
  - "3D Gaussian Splatting"
  - "Sparse Views"
  - "Visual Hull Depth Constraint"
date: 2026-05-08
content_hash: bc62e708a7c9a4b6
---

# Seeing A 3D World in A Grain of Sand

**Conference**: CVPR 2025  
**arXiv**: [2503.00260](https://arxiv.org/abs/2503.00260)  
**Code**: [Project Homepage](https://miniature-3dgs.github.io/)  
**Area**: 3D Vision  
**Keywords**: Miniature Scene Reconstruction, Catadioptric Imaging, 3D Gaussian Splatting, Sparse Views, Visual Hull Depth Constraint

## TL;DR

A catadioptric imaging system based on eight pairs of planar mirrors is designed to capture 360° surrounding multi-view images of miniature scenes in a single snapshot, combining visual hull depth constraints to improve sparse-view 3DGS reconstruction quality.

## Background & Motivation

3D reconstruction of miniature scenes (object sizes ranging from millimeters to centimeters) is widely demanded in daily life, such as in the digital preservation of toys, ornaments, and antiques. However, miniature scene reconstruction faces unique challenges: micro lenses are required for magnification but suffer from extremely shallow depth-of-field; scarce texture on objects makes traditional photometric reconstruction difficult; and self-calibration methods like COLMAP easily fail on textureless scenes.

Most existing 3DGS methods require dense view inputs to achieve high-quality rendering. Although sparse-view 3DGS methods exist (e.g., FSGS, SparseGS, DNGaussian), they mainly rely on monocular depth prediction, which suffers from insufficient accuracy in miniature scenes.

The core motivation of this work is to design an optical hardware system that captures synchronized 360° surrounding multi-view images in a single shot. Concurrently, precise pre-calibrated camera parameters and visual-hull-based depth constraints are utilized to achieve high-quality 3DGS reconstruction of miniature scenes, avoiding the dependency on self-calibration and dense views.

## Method

### Overall Architecture

The system consists of three core components: (1) Catadioptric lens design—eight pairs of planar mirrors are arranged on the surfaces of two nested octagonal pyramids to realize single-snapshot 360° multi-view acquisition; (2) Ray geometry analysis and mirror parameter optimization—closed-form formulas are derived to optimize the mirror configuration based on scene dimensions; (3) 3DGS reconstruction with visual hull depth constraints—foreground silhouettes are utilized to extract the visual hull and generate depth maps for regularization.

### Key Design 1: Multi-view Imaging of Catadioptric Lenses

- **Function**: Achieving simultaneous acquisition of eight surrounding perspective images with a single camera through optical path folding.
- **Mechanism**: Each pair of mirrors ($M_1$ and $M_2$) guides light from below the scene to the camera above via two reflections. The tilt angle $\alpha_1$ of $M_1$ and tilt angle $\alpha_2$ of $M_2$ jointly determine the system field of view $\text{FoV} = 4\Delta\alpha = 4(\alpha_2 - \alpha_1)$. The bottom width of the valid observation volume is $l = h_1/(\tan\alpha_1 \cdot \cos 2\Delta\alpha)$.
- **Design Motivation**: The lack of textures in miniature scenes makes SfM self-calibration unreliable, whereas optical pre-calibration provides high-precision camera parameters (with a re-projection error of only 0.77 pixels). Meanwhile, it avoids the mutual reflection problem of kaleidoscope systems, simplifying ray geometry analysis and calibration.

### Key Design 2: Optimal Mirror Configuration Given Scene Dimensions

- **Function**: Automatically calculating the optimal mirror angle difference based on the scene bounding box $W \times L \times H$.
- **Mechanism**: Deriving the closed-form formula $\Delta\alpha = \frac{1}{2}(\arcsin(\frac{w_{\max}}{\sqrt{L^2+H^2}}) - \arctan(\frac{L}{H}))$ to maximize the field of view while ensuring the valid observation volume completely encloses the scene.
- **Design Motivation**: A larger FoV implies more tilted viewpoints for virtual cameras, providing more thorough coverage of side surfaces, but the height of the volume decreases. A balance must be struck between coverage completeness and viewpoint diversity.

### Key Design 3: Visual Hull Constrained Weighted Depth Loss

- **Function**: Providing geometric regularization for sparse-view 3DGS to suppress artifacts in unobserved regions.
- **Mechanism**: Utilizing foreground masks and camera parameters to generate the visual hull depth map $\mathbf{D}_{\text{VH}}$, and designing an asymmetrically weighted $L_1$ depth loss: $\mathcal{L}_{\text{depth}} = \frac{2}{1+e^{\Delta d_i}} |\mathbf{D}_{\text{render}} - \mathbf{D}_{\text{VH}}|$.
- **Design Motivation**: The visual hull is the convex hull of the actual geometry. Points outside the hull ($\Delta d_i > 0$) should undergo heavier penalties, whereas internal points might be correct even with differing depths (due to concave regions). Therefore, a sigmoid logistic function is adopted for asymmetric weighting.

### Loss & Training

The total loss is $\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_{\text{D-SSIM}} + \lambda_3 \mathcal{L}_{\text{depth}}$, where $\lambda_1=0.8, \lambda_2=0.2, \lambda_3=0.5$. The color loss comprises $L_1$ and D-SSIM items, and the depth loss is based on the visual hull constraint.

## Key Experimental Results

### Main Results: Quantitative Comparison on Synthetic Data

| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Hierarchical 3DGS | 0.9750 | 26.83 | 0.0298 |
| FSGS | 0.7844 | 18.93 | 0.1100 |
| DNGaussian | 0.9128 | 21.40 | 0.1296 |
| SparseGS | 0.9756 | 31.84 | 0.0367 |
| **Ours** | **0.9783** | **32.48** | **0.0265** |

### Ablation Study: Comparison of Mirror Configurations

| Design | $\alpha_1$ | $\alpha_2$ | $\Delta\alpha$ |
|------|-----------|-----------|----------------|
| Design (a) | 75° | 85° | 10° |
| Design (b) | 60° | 85° | 25° |

A larger $\Delta\alpha$ provides better side coverage (e.g., the face of the figurine becomes visible), verifying the correctness of the theoretical derivation.

### Key Findings

- COLMAP failed on all miniature scenes; pre-calibrated camera parameters are crucial for miniature scene reconstruction.
- Visual-hull-based depth constraints are more effective for miniature scenes than monocular depth prediction.
- The overall reconstruction time is approximately 2 minutes (8 reference views at $800 \times 800$ resolution, NVIDIA 4090).

## Highlights & Insights

1. **Hardware-Algorithm Co-design**: Organically combining optical system design with the 3DGS algorithm, using hardware to guarantee high-precision calibration parameters, thereby avoiding the limitations of software self-calibration on textureless scenes.
2. **Asymmetric Weighting of Visual Hull Depth**: Designing an asymmetric loss utilizing the convex hull property of the visual hull, reflecting a profound understanding of geometric priors.
3. **Single Snapshot Scalable to Dynamic Scenes**: All viewpoints are optically synchronized in time, unlocking potentials for dynamic miniature scene reconstruction.

## Limitations & Future Work

- Currently, there are only 8 views with limited angular resolution. Complex scenes (especially those with fine structures) may still remain incompletely reconstructed.
- Physical hardware is required, and the generalization is constrained by the lens design.
- Future work: Introducing temporal consistency constraints to enable smooth reconstruction of dynamic miniature scenes.

## Related Work & Insights

- Catadioptric imaging systems have a long research history. The innovation of this work lies in avoiding mutual reflections and deriving closed-form optimization formulas.
- Sparse-view 3DGS is an active research area. The concept of visual hull depth constraints presented in this paper can be extended to other scenarios with silhouette masks.

## Rating

⭐⭐⭐⭐ — The co-design of hardware and algorithm is novel and practical, resolving the specific but practically-demanded task of miniature scene reconstruction. The visual hull depth constraint is ingeniously designed, though the reliance on hardware restricts the generalizability of the method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation](../../ICCV2025/3d_vision/seeing_and_seeing_through_the_glass_real_and_synthetic_data_for_multi-layer_dept.md)
- [\[CVPR 2025\] MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans](metascenes_towards_automated_replica_creation_for_real-world_3d_scans.md)
- [\[CVPR 2025\] Open-World Amodal Appearance Completion](open-world_amodal_appearance_completion.md)
- [\[CVPR 2025\] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](open-vocabulary_functional_3d_scene_graphs_for_real-world_indoor_spaces.md)
- [\[CVPR 2025\] PhysGen3D: Crafting a Miniature Interactive World from a Single Image](physgen3d_crafting_a_miniature_interactive_world_from_a_single_image.md)

</div>

<!-- RELATED:END -->
