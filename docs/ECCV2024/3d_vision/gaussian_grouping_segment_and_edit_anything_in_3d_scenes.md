---
title: >-
  [Paper Note] Gaussian Grouping: Segment and Edit Anything in 3D Scenes
description: >-
  [ECCV2024][3D Vision][3D Gaussian Splatting] Learns a 16-dimensional Identity Encoding for each Gaussian in 3D Gaussian Splatting to achieve instance-level grouping, utilizing SAM + DEVA video tracking to generate multi-view consistent 2D pseudo-labels for supervision. It achieves 69-77% mIoU on LERF-Mask open-vocabulary segmentation (outperforming LERF by over 2x) and outperforms Panoptic Lifting by 4.9% mIoU in panoptic segmentation while being 14x faster…
tags:
  - "ECCV2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "scene segmentation"
  - "3D editing"
  - "SAM"
  - "instance grouping"
date: 2026-05-08
content_hash: e591b269d0a13579
---

# Gaussian Grouping: Segment and Edit Anything in 3D Scenes

**Conference**: ECCV2024  
**arXiv**: [2312.00732](https://arxiv.org/abs/2312.00732)  
**Code**: [https://github.com/lkeab/gaussian-grouping](https://github.com/lkeab/gaussian-grouping)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, scene segmentation, 3D editing, SAM, instance grouping

## TL;DR
Learns a 16-dimensional Identity Encoding for each Gaussian in 3D Gaussian Splatting to achieve instance-level grouping, utilizing SAM + DEVA video tracking to generate multi-view consistent 2D pseudo-labels for supervision. It achieves 69-77% mIoU on LERF-Mask open-vocabulary segmentation (outperforming LERF by over 2x) and outperforms Panoptic Lifting by 4.9% mIoU in panoptic segmentation while being 14x faster, while simultaneously supporting various editing operations such as 3D object removal, inpainting, colorization, and style transfer.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has emerged as a powerful alternative to NeRF due to its extremely fast rendering speed and high-quality reconstruction. However, 3DGS represents scenes as an unstructured collection of Gaussians and lacks semantic understanding—it does not know which Gaussians belong to the same object.

**Limitations of Prior Work**: (a) Methods like SA3D require segmenting objects one by one, which can take over 35 minutes for a complex scene; (b) LERF distills CLIP features on NeRF for open-vocabulary segmentation but suffers from low accuracy (mIoU of only 30-50%); (c) Panoptic Lifting performs panoptic segmentation but is based on NeRF, yielding only ~10 FPS; (d) There is a lack of a unified segmentation and editing framework based on 3DGS.

**Key Challenge**: The number of Gaussians in 3DGS is huge (hundreds of thousands), yet they lack semantic grouping—how to efficiently assign an instance label to each Gaussian? Directly learning an independent embedding for each Gaussian is difficult to train, requiring cross-view consistent 2D supervisory signals.

**Goal**: (1) Group 3DGS Gaussians into semantic instances; (2) enable various 3D editing tasks based on this grouping; (3) maintain the real-time rendering advantage of 3DGS.

**Key Insight**: Leveraging the "everything" mode of SAM to generate complete segmentations for each frame, and then using the video tracker DEVA to associate instance IDs across views to obtain multi-view consistent pseudo-labels. Each 3D Gaussian learns a 16-dimensional Identity Encoding, which is splatted to 2D and supervised via cross-entropy loss.

**Core Idea**: Adding a compact identity encoding to each 3DGS Gaussian + a joint 2D/3D grouping loss = zero-cost instance-level 3D scene understanding + arbitrary editing.

## Method

### Overall Architecture
(1) SAM "everything" mode generates complete segmentation masks for each frame $\to$ (2) DEVA zero-shot video tracker associates mask IDs across frames (60x faster than linear assignment) $\to$ (3) each 3D Gaussian is augmented with a 16-dimensional Identity Encoding and jointly trained with position, color, and opacity $\to$ (4) 2D rendered identity features + a linear classification layer + cross-entropy loss ($\mathcal{L}_{2d}$) + 3D KNN regularization ($\mathcal{L}_{3d}$) $\to$ (5) after training, Gaussians are grouped by their identity encodings to support various editing tasks.

### Key Designs

1. **16-dimensional Identity Encoding**:

    - **Function**: Learns a 16-dimensional differentiable vector for each 3D Gaussian to represent its instance assignment.
    - **Mechanism**: Splats the 3D identity encoding to 2D via standard 3DGS alpha-blending: $E_{id} = \sum_{i} e_i \alpha'_i \prod_{j<i}(1 - \alpha'_j)$. Spherical harmonics are not used (SH degree=0) because instance labels should be view-independent.
    - **Design Motivation**: A 16-dimensional vector is compact enough (reducing rendering speed from ~200 FPS only to ~170 FPS), and experiments show that 32 dimensions yield no further improvements.

2. **SAM + DEVA Cross-View Pseudo-Labels**:

    - **Function**: SAM "everything" mode generates unassociated masks frame-by-frame $\to$ DEVA video tracker propagates instance IDs across frames.
    - **Mechanism**: DEVA is a zero-shot video segmentation tracker capable of associating the same instance across different views to output unified ID numbers.
    - **Design Motivation**: It is **60x faster** than traditional cost-matrix linear assignment (the Panoptic Lifting approach) (1 minute vs. 1 hour) and achieves better tracking quality.

3. **Joint 2D + 3D Grouping Loss**:

    - $\mathcal{L}_{2d}$: The rendered 2D identity features are mapped to K classes (K = total number of masks in the scene) via a linear layer and supervised by standard cross-entropy loss.
    - $\mathcal{L}_{3d}$: For the sampled 3D Gaussians, the identity encodings of their $K=5$ nearest neighbors are encouraged to be consistent with their own (using KL divergence).
    - Crucial role of $\mathcal{L}_{3d}$: 2D supervision cannot cover occluded Gaussians or those inside objects—3D regularization supplements supervision via spatial proximity assumptions. Ablation shows that with $K=5$, the removal accuracy increases from 41.2% to 67.5%.

4. **Local Gaussian Editing**:

    - Removal: Directly delete the target Gaussians.
    - Inpainting: Delete the target Gaussians and inpaint via LaMa 2D inpainting $\to$ add new Gaussians + fine-tune.
    - Colorization/Style Transfer: Freeze non-target Gaussians and only fine-tune the color SH/position attributes of the target Gaussians.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{rec} + \lambda_{2d}\mathcal{L}_{2d} + \lambda_{3d}\mathcal{L}_{3d}$, where $\lambda_{2d}=1.0$ and $\lambda_{3d}=2.0$. Configured with 30K iterations on a single A100 GPU. The learning rate for Identity Encoding is 2.5e-3, and for the linear layer is 5e-4.

## Key Experimental Results

### Main Results

**Open-Vocabulary Segmentation (LERF-Mask Dataset)**:

| Method | figurines mIoU | ramen mIoU | teatime mIoU |
|------|--------------|-----------|-------------|
| LERF | 33.5 | 28.3 | 49.7 |
| SA3D | 24.9 | 7.4 | 42.5 |
| LangSplat | 52.8 | 50.4 | 69.5 |
| **Gaussian Grouping** | **69.7** | **77.0** | **71.7** |

**Panoptic Segmentation**:

| Method | Replica mIoU | Replica PQ | Replica FPS | ScanNet mIoU | ScanNet FPS |
|------|-------------|-----------|------------|-------------|-----------|
| Panoptic Lifting | 66.22 | 64.34 | ~10 | 67.01 | ~10 |
| **Gaussian Grouping** | **71.15** | **66.52** | **~140** | **68.70** | **~150** |

Practically lossless rendering quality: PSNR 28.43 vs. original 3DGS 28.69 (only a 0.26 dB drop).

### Ablation Study

| K (Number of Nearest Neighbors in 3D Regularization) | Removal Accuracy |
|---|---|
| K=0 (No 3D loss) | — |
| K=1 | 41.2% |
| K=5 | 67.5% |
| K=10 | 76.6% |

### Key Findings
- **3D Regularization is Crucial**: $K=1$ achieves only a 41.2% removal accuracy, whereas $K=5$ increases it to 67.5%, as 2D supervision alone cannot cover occluded Gaussians.
- **DEVA Tracking $\gg$ Linear Assignment**: 60x faster (1 min vs. 1 hr) with superior mask quality.
- **Editing Quality Outperforms Specialized Methods**: 3D inpainting CLIP similarity reaches 0.153 vs. 0.126 (+21%) for SPIn-NeRF, and style transfer achieves 0.178 vs. 0.171 for Instruct-NeRF2NeRF.
- **Overwhelming Speed Advantage**: Segmentation takes 9 minutes vs. 35 minutes for SA3D; rendering speeds reach 140-150 FPS vs. 10 FPS for Panoptic Lifting.

## Highlights & Insights
- **Sparsely Elegant yet Powerful Design**: By simply adding a 16-dimensional vector to each Gaussian combined with a cross-entropy loss, 3DGS is upgraded from pure reconstruction to a unified platform for segmentation and editing. Its architectural simplicity is impressive, introducing almost no additional complexity.
- **Ingenious Use of SAM + Video Tracking**: Bypasses the difficult problem of "generating multi-view consistent annotations for 3D scenes" by leveraging the strong zero-shot segmentation capabilities of SAM and the temporal consistency of video trackers to automatically generate high-quality pseudo-labels. This pipeline can be reused in any 3D task requiring multi-view consistent segmentation.
- **3D Regularization for Occlusion Handling**: The inherent defect of 2D loss is the inability to supervise invisible Gaussians. The KNN consistency assumption perfectly compensates for this.

## Limitations & Future Work
- **Pseudo-label Quality Bottleneck**: The final segmentation quality is limited by the 2D segmentation/tracking quality of SAM + DEVA, which may be insufficient for fine boundaries and heavily occluded scenes.
- **Fixed Number of Classes $K$**: During training, $K$ is fixed to the total number of masks in the scene; the appearance of new objects requires retraining.
- **Editing Requires Extra Fine-tuning**: Editing operations such as inpainting and style transfer require an extra 20 minutes to 1 hour of fine-tuning, falling short of real-time performance.
- **Future Directions**: (1) Replace discrete ID encodings with open-vocabulary features (e.g., CLIP) to enable zero-shot category querying; (2) introduce support for dynamic scenes; (3) incorporate depth consistency constraints to further improve 3D regularization.

## Related Work & Insights
- **vs SA3D**: SA3D requires interactive segmentation on an object-by-object basis, needing multiple operations for a single scene. Gaussian Grouping segments all instances at once, achieving 4x higher efficiency.
- **vs LERF/LangSplat**: These methods distill CLIP features for open-vocabulary segmentation, but the low spatial resolution of CLIP features results in low mIoU. Gaussian Grouping uses instance-level ID encodings, yielding much sharper segmentation boundaries.
- **vs Panoptic Lifting**: Also performs panoptic segmentation but is based on NeRF, making it 14x slower. Gaussian Grouping inherits the rendering speed advantages of 3DGS.

## Rating
- Novelty: ⭐⭐⭐⭐ First to realize a complete instance grouping and editing pipeline on 3DGS, with a simple and elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of open-vocabulary segmentation, panoptic segmentation, 5 editing tasks, rendering quality, speed, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear pipeline diagrams, rich experimental results, and thorough quantitative + qualitative evaluations.
- Value: ⭐⭐⭐⭐⭐ Bridges the gap between 3DGS reconstruction and downstream understanding/editing; the open-source code deeply impacts the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAMosaic3D: Modular Scene Assembly for Real-Time 3D Segment Anything](../../CVPR2026/3d_vision/samosaic3d_modular_scene_assembly_for_real-time_3d_segment_anything.md)
- [\[CVPR 2025\] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](../../CVPR2025/3d_vision/isegman_interactive_segment-and-manipulate_3d_gaussians.md)
- [\[CVPR 2026\] Space-Time Forecasting of Dynamic Scenes with Motion-aware Gaussian Grouping](../../CVPR2026/3d_vision/space-time_forecasting_of_dynamic_scenes_with_motion-aware_gaussian_grouping.md)
- [\[CVPR 2026\] RoSAMDepth: Robust Self-supervised Depth Estimation Leveraging Segment Anything Model](../../CVPR2026/3d_vision/rosamdepth_robust_self-supervised_depth_estimation_leveraging_segment_anything_m.md)
- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](../../ICCV2025/3d_vision/3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)

</div>

<!-- RELATED:END -->
