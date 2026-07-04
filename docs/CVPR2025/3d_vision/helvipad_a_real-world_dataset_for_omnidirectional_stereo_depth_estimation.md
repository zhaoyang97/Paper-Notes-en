---
title: >-
  [Paper Note] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation
description: >-
  [CVPR 2025][3D Vision][Omnidirectional depth estimation] This paper proposes Helvipad—the first real-world dataset for omnidirectional stereo depth estimation (40K frames, top-bottom dual 360° cameras + LiDAR). It also introduces two lightweight adaptation strategies, polar angle input and circular padding, to improve stereo matching models for handling equirectangular projection images, with the proposed 360-IGEV-Stereo achieving state-of-the-art performance across all metri…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Omnidirectional depth estimation"
  - "Stereo matching"
  - "Equirectangular projection"
  - "360° camera"
  - "Dataset"
date: 2026-05-08
content_hash: e574491237513cc9
---

# Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation

**Conference**: CVPR 2025  
**arXiv**: [2411.18335](https://arxiv.org/abs/2411.18335)  
**Code**: [https://vita-epfl.github.io/Helvipad/](https://vita-epfl.github.io/Helvipad/) (Project Page)  
**Area**: Autonomous Driving  
**Keywords**: Omnidirectional depth estimation, Stereo matching, Equirectangular projection, 360° camera, Dataset

## TL;DR

This paper proposes Helvipad—the first real-world dataset for omnidirectional stereo depth estimation (40K frames, top-bottom dual 360° cameras + LiDAR). It also introduces two lightweight adaptation strategies, polar angle input and circular padding, to improve stereo matching models for handling equirectangular projection images, with the proposed 360-IGEV-Stereo achieving state-of-the-art performance across all metrics.

## Background & Motivation

**Background**: Stereo depth estimation has made significant progress in recent years, with deep learning methods (e.g., IGEV-Stereo) performing exceptionally well on standard datasets. However, these methods and datasets primarily target standard perspective cameras with a limited field of view (FoV). 360° omnidirectional cameras provide complete FoV coverage, which is highly beneficial for scenarios such as robot navigation and indoor surveillance.

**Limitations of Prior Work**: (1) Absence of real-world omnidirectional stereo depth datasets—existing omnidirectional datasets are either synthetic (e.g., MP3D, SF3D), lack pixel-level depth annotations (e.g., JRDB), or only cover a single type of environment. (2) Equirectangular projection (ERP) introduces severe spherical distortion, meaning traditional stereo matching models designed for rectangular images perform poorly when applied directly. (3) Existing omnidirectional stereo methods (e.g., 360SD-Net) are based on older architectures and fail to leverage recent advancements.

**Key Challenge**: Omnidirectional imaging possesses massive advantages in applications requiring complete spatial awareness, but the lack of suitable real-world data and algorithmic adaptation hinders the development of deep learning methods in this field. The challenge lies in constructing a high-quality dataset and adapting state-of-the-art models.

**Goal**: (1) To construct a real-world omnidirectional stereo dataset containing diverse indoor and outdoor scenes under various lighting conditions; (2) To solve the precise mapping of LiDAR point clouds to 360° images and the densification of sparse annotations; (3) To adapt SOTA stereo matching models to handle spherical geometry.

**Key Insight**: Build an acquisition platform using a top-bottom dual 360° camera setup (to avoid occlusion) and a LiDAR. Address the sparse annotation issue via depth completion, and enable current SOTA models to understand omnidirectional geometry through two lightweight adaptation strategies: polar angle input and circular padding.

**Core Idea**: Construct the first real-world omnidirectional stereo depth dataset, and adapt IGEV-Stereo into an omnidirectional version using two minimal modifications: polar angle input and circular padding.

## Method

### Overall Architecture

The system consists of two parts: (1) Dataset construction—capturing 29 video sequences on a university campus using a custom acquisition setup (dual Ricoh Theta V cameras arranged vertically with a 19.1 cm baseline, paired with an Ouster OS1-64 LiDAR), projecting 3D point clouds onto equirectangular images via LiDAR-camera calibration to obtain depth labels, and increasing the annotated pixel ratio from 12% to 61% via depth completion. (2) Model adaptation—introducing polar angle map input and circular padding on top of IGEV-Stereo to form 360-IGEV-Stereo.

### Key Designs

1. **Precise LiDAR-to-360° Image Mapping**:

    - **Function**: Project LiDAR 3D point clouds onto equirectangular images accurately to generate depth labels.
    - **Mechanism**: At the beginning of each acquisition session, a 19×19 checkerboard calibration board is used to obtain the correspondence between LiDAR points and image pixels. A preliminary rotation and translation align the LiDAR coordinate system to the camera center, followed by converting the 3D points to spherical coordinates $(r, \theta, \phi)$ and projecting them onto the equirectangular plane as $(x^{eq}, y^{eq}) = (\frac{\phi + \pi}{2\pi}W, \frac{\theta}{\pi}H)$. Minimizing the projection error via BFGS optimization yields an average alignment error of only 1.7 pixels. The spherical disparity is defined as $d = \arctan(\frac{\sin(\theta_b)}{r_{bottom}/B_{camera} - \cos(\theta_b)})$.
    - **Design Motivation**: The spherical projection of omnidirectional images is fundamentally different from standard perspective projection, requiring spherical coordinate transformations for correct mapping. Precise calibration is the foundation of dataset quality.

2. **Depth Completion Pipeline**:

    - **Function**: Densify sparse LiDAR depth annotations, increasing the ratio of annotated pixels from 12% to 61%.
    - **Mechanism**: A three-step pipeline: (1) **Temporal aggregation**: merging the point cloud of the current frame with those from 4 preceding and 4 succeeding frames, where the error is negligible due to high frame rate and low moving speed. (2) **Spherical interpolation**: estimating the depth of a query point on the spherical grid via inverse distance-weighted average of its $k$-nearest neighbors $r_q = \sum w_i r_i$ (where weights are the reciprocal of spherical coordinate distance). (3) **Filtering**: using relative weighted variance as an uncertainty metric $\sigma^2_{r_q} = \sum w_i (\frac{r_q - r_i}{r_q})^2$ to remove high-uncertainty points and regions lacking close neighbors (e.g., the sky).
    - **Design Motivation**: LiDAR point clouds are inherently sparse, and using them directly as training labels results in insufficient supervision. Temporal aggregation followed by interpolation and filtering provides an automated densification pipeline without human annotations while guaranteeing label quality.

3. **360-IGEV-Stereo (Panorama-Adapted Stereo Matching Model)**:

    - **Function**: Adapt IGEV-Stereo to process equirectangular projection images.
    - **Mechanism**: Two adaptations are introduced. **Polar angle input**: in a top-bottom setup, images are vertically distorted to construct the cost volume, with the distortion level varying according to the polar angle $\theta$. The polar angle map is added as an extra input channel, processed by a shared polar angle encoder (strided convolutional layers) and concatenated with the feature network bottleneck at the lowest resolution (1/32), as well as context features at 1/4 resolution. **Circular padding**: leveraging the horizontal continuity of 360° images, the left boundary of the image is padded with 64 columns of pixels from the right side, and vice versa during inference. This enables the network to utilize cross-boundary context. The cost volume is constructed via vertical warping. Photometric data augmentation is additionally incorporated during training to cope with lighting variations in the dataset.
    - **Design Motivation**: The distortion of equirectangular projection varies with the polar angle. Failing to feed this prior to the model leads to biased depth estimation. Circular padding eliminates discontinuities at the left and right boundaries of 360° images, reducing LRCE from 1.18m to 0.36m.

### Loss & Training

The original training strategy of IGEV-Stereo is adopted, utilizing the maximum possible batch size on a single NVIDIA A100 with early stopping. The training set contains 20 sequences (29,407 frames) and the test set has 6 sequences (10,146 frames), ensuring consistent ratios of indoor, outdoor, and night scenes without overlapping environments.

## Key Experimental Results

### Main Results

| Method | Stereo Setup | Disparity MAE (°) ↓ | Depth MAE (m) ↓ | Depth RMSE (m) ↓ | Depth MARE ↓ | LRCE (m) ↓ |
|------|---------|-------------|-------------|--------------|-----------|-----------|
| PSMNet | Standard | 0.286 | 2.509 | 5.673 | 0.176 | 1.809 |
| 360SD-Net | Omnidirectional | 0.224 | 2.122 | 5.077 | 0.152 | 0.904 |
| IGEV-Stereo | Standard | 0.225 | 1.860 | 4.474 | 0.146 | 1.203 |
| **360-IGEV-Stereo** | Omnidirectional | **0.188** | **1.720** | **4.297** | **0.130** | **0.388** |

### Ablation Study

| Configuration | Depth MAE ↓ | Depth RMSE ↓ | Depth MARE ↓ | LRCE ↓ |
|------|----------|-----------|-----------|--------|
| 360-IGEV-Stereo (full) | 1.720 | 4.297 | 0.130 | 0.388 |
| w/o circular padding | 1.726 | 4.314 | 0.130 | **1.153** |
| w/o photometric augmentation | **1.845** | **4.466** | **0.135** | 0.347 |

**Enhancement effect of depth completion** (taking Depth MARE as an example):

| Method | Original Labels | Enhanced Labels |
|------|---------|---------|
| 360SD-Net | 0.17 | 0.15 |
| PSMNet | ~0.19 | ~0.18 |
| IGEV-Stereo | ~0.15 | ~0.146 |

### Key Findings

- Circular padding contributes most significantly to LRCE (1.153m $\rightarrow$ 0.388m), effectively eliminating depth jumps caused by discontinuities at the 360° image boundaries.
- Photometric augmentation contributes more to overall depth accuracy (MARE: 0.135 $\rightarrow$ 0.130), indicating that lighting variation is one of the primary challenges in this dataset.
- Omnidirectional adapted methods (such as 360-IGEV-Stereo and 360SD-Net) exhibit better cross-scene generalization, demonstrating smaller performance drops in night outdoor scenes after being trained on indoor data.
- Performance improves for all methods after enhancing training data with depth completion, indicating that annotation density is a primary bottleneck for training quality.
- Modern stereo methods (such as IGEV-Stereo) outperform specialized omnidirectional methods (such as 360SD-Net) even without adaptation, demonstrating that architectural progress is more crucial than domain adaptation, though combining both yields the best results.

## Highlights & Insights

- **Minimalistic Adaptation Strategy**: General stereo models are adapted to omnidirectional scenarios with only two lightweight modifications—polar angle input and circular padding. This adds negligible computational overhead (inference time of 0.24s vs 0.25s), and this "minimally intrusive adaptation" scheme can be generalized to other omnidirectional vision tasks.
- **Automated Depth Completion Pipeline**: The three-step completion pipeline (temporal aggregation + spherical interpolation + uncertainty filtering) is fully automated while ensuring annotation quality. The significant increase in annotation density from 12% to 61% can be directly applied to other sparse depth datasets.
- **Systemic Benchmark Construction**: The work not only provides the dataset but also systematically compares standard and omnidirectional methods, presents cross-scene generalization analyses, and evaluates the impact of depth completion, establishing a comprehensive baseline for future research.

## Limitations & Future Work

- The 19.1 cm baseline of the top-bottom camera configuration is relatively short, limiting depth accuracy for long-range objects.
- The LiDAR vertical field of view is 42.4°, resulting in a lack of depth labels in the upper and lower edge regions of the images.
- Temporal aggregation in depth completion can introduce tracking errors for fast-moving objects (e.g., cars).
- Data collection is restricted to a university campus environment, offering less scene diversity than large-scale driving datasets.
- Monocular omnidirectional depth estimation baselines are not explored, as the study focuses solely on stereo matching.

## Related Work & Insights

- **vs KITTI/DrivingStereo**: Standard stereo driving datasets with limited FOV, only covering the front view; Helvipad provides full 360° coverage, which is more suitable for robotic applications requiring omnidirectional perception.
- **vs 360SD-Net**: The only omnidirectional stereo method with a top-bottom setup, but based on the older PSMNet architecture; 360-IGEV-Stereo outperforms it across all metrics by utilizing a more advanced iterative optimization architecture and polar angle encoding.
- **vs JRDB**: Also a real-world 360° dataset but lacks pixel-level depth annotations; Helvipad provides high-quality pixel-level depth labels through LiDAR projection and depth completion.

## Rating

- Novelty: ⭐⭐⭐ The dataset and adaptation strategies represent an incremental contribution but fill an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely systematic, including multi-method comparison, cross-scene analysis, ablation studies, and depth completion validation.
- Writing Quality: ⭐⭐⭐⭐ The data acquisition and calibration pipelines are clearly and thoroughly described.
- Value: ⭐⭐⭐⭐ Fills a void in real-world omnidirectional stereo datasets, laying the groundwork for future research by establishing strong baselines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2025\] Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses](efficient_depth_estimation_for_unstable_stereo_camera_systems_on_ar_glasses.md)
- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[CVPR 2025\] Pixel-Aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision](pixel-aligned_rgb-nir_stereo_imaging_and_dataset_for_robot_vision.md)
- [\[CVPR 2025\] MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans](metascenes_towards_automated_replica_creation_for_real-world_3d_scans.md)

</div>

<!-- RELATED:END -->
