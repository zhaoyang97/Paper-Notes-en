---
title: >-
  [Paper Note] Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos
description: >-
  [CVPR 2025][3D Vision][4D Reconstruction] Stereo4D proposes an automated pipeline to mine high-quality 4D reconstruction data from internet stereo fisheye videos (VR180). It generates over 100K video clips containing pseudo-metric 3D point clouds and long-range trajectories in the world coordinate system, and trains a DynaDUSt3R model to achieve the capability of predicting 3D structure and motion directly from image pairs.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "4D Reconstruction"
  - "Stereo Video"
  - "3D Motion Estimation"
  - "Dynamic Point Cloud"
  - "Dataset"
date: 2026-05-08
content_hash: d9d38353e44316e0
---

# Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos

**Conference**: CVPR 2025  
**arXiv**: [2412.09621](https://arxiv.org/abs/2412.09621)  
**Code**: [https://stereo4d.github.io](https://stereo4d.github.io)  
**Area**: 3D Vision / Dynamic Scene Reconstruction  
**Keywords**: 4D Reconstruction, Stereo Video, 3D Motion Estimation, Dynamic Point Cloud, Dataset

## TL;DR

Stereo4D proposes an automated pipeline to mine high-quality 4D reconstruction data from internet stereo fisheye videos (VR180). It generates over 100K video clips containing pseudo-metric 3D point clouds and long-range trajectories in the world coordinate system, and trains a DynaDUSt3R model to achieve the capability of predicting 3D structure and motion directly from image pairs.

## Background & Motivation

**Background**: Static 3D reconstruction (e.g., DUSt3R, Depth Anything) has made significant progress through large-scale training data. However, dynamic 3D scene understanding—simultaneously predicting geometry and motion—remains a core unsolved challenge.

**Limitations of Prior Work**: The key bottleneck in learning 3D motion estimation lies in the lack of large-scale, real-world training data. Synthetic datasets (e.g., PointOdyssey) struggle to capture real-world content distributions and motion patterns. Motion capture and multi-view camera arrays are accurate but difficult to scale, offering limited scene diversity. Existing real-world datasets (e.g., KITTI, Waymo) are confined to specific scenarios like autonomous driving.

**Key Challenge**: While large-scale data-driven learning paradigms have proven effective in language, image generation, and static 3D domains, the dynamic 3D domain lacks a corresponding large-scale real-world data source. The domain gap of synthetic data hinders models from generalizing to real-world motion.

**Goal**: Find a scalable source of real-world 3D motion data and design a pipeline to extract high-quality 4D reconstructions from it.

**Key Insight**: The authors identify online VR180 stereo fisheye videos as an underutilized data source—these videos feature a wide field of view, standardized stereo baselines, and rich everyday scene content.

**Core Idea**: Integrate the outputs of camera pose estimation, stereo depth estimation, and temporal tracking methods. Through carefully designed filtering and optimization steps, extract pseudo-metric 3D point clouds and their long-range trajectories in the world coordinate system from VR180 videos.

## Method

### Overall Architecture

The system consists of two main parts: (1) A data generation pipeline that extracts camera poses, stereo depth maps, and 2D tracking trajectories from VR180 videos, fuses them into 3D point clouds and motion trajectories, and generates a high-quality 4D dataset through filtering and optimization; (2) The DynaDUSt3R model, which adds a motion prediction head on top of DUSt3R to predict 3D structure and 3D scene flow given two images.

### Key Designs

1. **4D Data Processing Pipeline**:

    - Function: Converts raw VR180 stereo videos into dynamic point clouds with long-range 3D motion trajectories.
    - Mechanism: First, ORB-SLAM2 is used to segment the video into trackable shots. Then, a COLMAP-like incremental SfM is run to estimate camera poses and optimize the stereo rig calibration parameters ($\mathbf{c}_r, \mathbf{R}_r$). Next, RAFT is employed to estimate disparity maps for the rectified stereo pairs of each frame, and BootsTAP is used to extract dense 2D long-range tracking trajectories. These 2D trajectories are back-projected into 3D motion trajectories using camera poses and disparity maps. Finally, quality control steps such as semantic filtering (discarding drifting trajectories on static categories like walls/roads) and cross-fade detection are applied.
    - Design Motivation: Desktop stereo depth estimation exhibits frame-by-frame jitter, and independent 2D tracking can drift. By fusing and optimizing multiple signals, these issues can compensate for each other to yield high-quality results.

2. **3D Track Optimization**:

    - Function: Eliminates high-frequency jitter in 3D trajectories caused by frame-by-frame independent depth estimation.
    - Mechanism: Solves a scalar offset $\delta_i$ along the camera ray direction for each trajectory point, such that $\mathbf{p}'_i = \mathbf{p}_i + \delta_i \mathbf{r}_i$. The optimization objective consists of three terms: a static loss $\mathcal{L}_{\text{static}}$ (encouraging points to remain stationary in the world coordinate system), a dynamic loss $\mathcal{L}_{\text{dynamic}}$ (minimizing acceleration along the ray via a discrete Laplacian operator to smooth the motion), and a regularization loss $\mathcal{L}_{\text{reg}}$ (constraining the offset in disparity space). The two losses are combined using weights from a sigmoid function $\sigma(m)$ based on the motion magnitude.
    - Design Motivation: Stereo depth is estimated independently for each frame, and directly back-projected 3D points contain high-frequency noise. This optimization eliminates jitter while maintaining the physical plausibility of the motion trajectories.

3. **DynaDUSt3R Motion prediction head**:

    - Function: Adds a parallel motion head to the DUSt3R architecture to predict 3D scene flow from two images.
    - Mechanism: Given two images $\mathbf{I}_0$ and $\mathbf{I}_1$ and a query time $t_q \in [0,1]$, a shared ViT encoder and cross-attention decoder extract global features $G^0, G^1$. The point map head predicts 3D point maps $\mathbf{P}^v$ for each frame (identical to DUSt3R), while the newly added motion head predicts 3D displacement maps $\mathbf{M}^{v \to t_q}$ from each frame to the target time $t_q$. The time $t_q$ is injected into the motion features via positional encoding. The training loss consists of a confidence-weighted scale-invariant 3D regression loss $\mathcal{L}_{\text{point}}$ and a motion loss $\mathcal{L}_{\text{motion}}$.
    - Design Motivation: Predicting to an intermediate time point (rather than just end-to-end) allows the model to learn complete motion trajectories and enables the utilization of incomplete ground-truth tracks as supervision.

### Loss & Training

The training loss is a confidence-weighted, scale-invariant 3D regression loss, where predicted and ground-truth point maps are normalized before computing the Euclidean distance. Initialized from DUSt3R weights, the motion head is initialized using the point map head weights. With a batch size of 64 and a learning rate of 2.5e-5, the model is trained for 49K steps using the Adam optimizer (weight decay 0.95). The training data consists of randomly sampled video frame pairs with a maximum interval of 60 frames.

## Key Experimental Results

### Main Results — 3D Motion Prediction

| Training Data | Stereo4D EPE3D ↓ | Stereo4D $\delta_{3D}^{0.05}$ ↑ | Stereo4D $\delta_{3D}^{0.10}$ ↑ | ADT EPE3D ↓ | ADT $\delta_{3D}^{0.05}$ ↑ | ADT $\delta_{3D}^{0.10}$ ↑ |
|---------|-----|-----|-----|-----|-----|-----|
| PointOdyssey (Synthetic) | 0.619 | 11.6 | 20.3 | 0.313 | 8.6 | 18.0 |
| Stereo4D (Ours) | **0.111** | **65.1** | **75.2** | **0.123** | **52.0** | **65.2** |

### Depth Estimation (Bonn Dataset)

| Method | Abs Rel ↓ | RMSE ↓ | $\delta_1$ ↑ |
|------|----------|--------|------|
| DUSt3R | 0.078 | 0.205 | 0.942 |
| MonST3R | 0.066 | 0.182 | 0.952 |
| DynaDUSt3R | **0.059** | **0.168** | **0.965** |

### Key Findings

- DynaDUSt3R trained on real-world data (Stereo4D) **completely outperforms** the model trained on synthetic data (PointOdyssey) in 3D motion prediction, with EPE3D dropping from 0.619 to 0.111 (an 82% reduction). This confirms the critical value of real-world data for learning 3D motion priors.
- Even when evaluated on the ADT test set from a completely different source, the model trained on Stereo4D demonstrates stronger generalization capability.
- DynaDUSt3R outperforms DUSt3R and MonST3R in depth estimation on dynamic scenes, indicating that motion modeling can conversely enhance the accuracy of geometry estimation.

## Highlights & Insights

- **The clever choice of data source** is the biggest highlight: VR180 videos naturally provide stereo baselines, wide fields of view, and abundant everyday scenes, solving the core bottleneck of scaling real-world 3D motion data acquisition.
- **The "data + simple model" paradigm is validated once again**: The model modifications in DynaDUSt3R are very lightweight (only adding a motion head), yet the data quality brings huge performance gains.
- **3D Track Optimization** employs motion-magnitude adaptive weights for static/dynamic losses, which is an exquisite design choice.

## Limitations & Future Work

- The content distribution of VR180 videos is biased (leaning heavily towards tourism, outdoor activities, etc.), with insufficient coverage of long-tail scenarios like fine indoor manipulation.
- The precision of the pseudo-metric annotations is constrained by the quality of the stereo calibration, and the depth estimation for distant objects is noisy.
- DynaDUSt3R currently only supports two-frame inputs; extending it to multi-frame inputs could bring stronger motion reasoning capabilities.
- The dataset relies on a cascade of multiple existing methods (SfM + RAFT + BootsTAP), meaning errors from each stage can accumulate.

## Related Work & Insights

- **vs PointOdyssey**: Although synthetic data is easy to obtain, it exhibits noticeable domain gaps. This work quantitatively demonstrates the necessity of real-world data (reducing EPE by more than 5 times).
- **vs DUSt3R/MASt3R**: DUSt3R only handles static scenes; DynaDUSt3R extends capability to dynamic scenes with minimal modifications.
- **vs KITTI/Waymo**: While these datasets are limited to driving scenarios, Stereo4D possesses significantly greater content diversity.

## Rating

- Novelty: ⭐⭐⭐⭐ The identification of the data source and the design of the complete pipeline are pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ The comparison between synthetic and real-world data is convincing, and cross-dataset generalization testing is thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ The pipeline description is clear, and visualizations are rich.
- Value: ⭐⭐⭐⭐⭐ The contribution of over 100K real-world dynamic 3D data clips is of major significance to the entire field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)
- [\[CVPR 2025\] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)
- [\[CVPR 2025\] You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale](you_see_it_you_got_it_learning_3d_creation_on_pose-free_videos_at_scale.md)
- [\[CVPR 2025\] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)

</div>

<!-- RELATED:END -->
