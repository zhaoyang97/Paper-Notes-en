---
title: >-
  [Paper Note] FunREC: Reconstructing Functional 3D Scenes from Egocentric Interaction Videos
description: >-
  [CVPR 2026][3D Vision][functional 3D reconstruction] This paper presents FunREC, a training-free optimization-based method that reconstructs functional articulated 3D digital twin scenes directly from egocentric RGB-D in…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "functional 3D reconstruction"
  - "egocentric video"
  - "articulated object reconstruction"
  - "digital twin"
  - "motion estimation"
date: 2026-05-08
content_hash: 29ed9eddfedba838
---

# FunREC: Reconstructing Functional 3D Scenes from Egocentric Interaction Videos

**Conference**: CVPR 2026
**arXiv**: [2604.05621](https://arxiv.org/abs/2604.05621)
**Code**: [https://functionalscenes.github.io/](https://functionalscenes.github.io/)
**Area**: 3D Vision
**Keywords**: functional 3D reconstruction, egocentric video, articulated object reconstruction, digital twin, motion estimation

## TL;DR
This paper presents FunREC, a training-free optimization-based method that reconstructs functional articulated 3D digital twin scenes directly from egocentric RGB-D interaction videos. It automatically discovers articulated parts, estimates kinematic parameters, tracks 3D motion, and reconstructs both static and dynamic geometry. FunREC substantially outperforms prior methods across all benchmarks (part segmentation mIoU improves by 50+, joint angle error reduced by 5–10×) and supports simulation export and robotic interaction.

## Background & Motivation

1. **Background**: 3D scene reconstruction has advanced considerably, yet existing large-scale RGB-D datasets (ScanNet, ARKitScenes, etc.) capture only a single static state of the environment and cannot represent functional interactions such as opening doors or sliding drawers. Digital twins require not only geometric capture but also an understanding of how objects move and articulate.

2. **Limitations of Prior Work**: (1) MultiScan requires two scans of the same room (open/closed states) and manual alignment annotation, making it highly inefficient. (2) SceneFun3D/Articulate3D annotates functional information on static LiDAR scans but cannot directly observe kinematic properties. (3) Digital cousins methods retrieve CAD proxy models as substitutes, maintaining only a loose association with actual geometry. (4) Object-level articulation reconstruction methods generally assume controlled environments, fixed cameras, or known CAD models, and are unable to handle scene-level reconstruction in the wild.

3. **Key Challenge**: Human interactions naturally reveal which parts move, around which joints, and what internal volumes are exposed — yet these rich signals remain unexploited. Existing methods either rely on multiple scans with manual annotation or on CAD retrieval as a weak proxy.

4. **Goal**: Can a complete, interactive, physically-simulation-compatible functional 3D digital twin be automatically reconstructed from a single ordinary egocentric interaction video?

5. **Key Insight**: Human interaction provides the most direct and informative functional supervision signal. When people manipulate the environment, egocentric observation naturally reveals articulation information. FunREC leverages the semantic and motion priors of visual foundation models to complete the entire pipeline without any training.

6. **Core Idea**: FunREC segments egocentric interaction videos into clips, uses the semantic and motion priors of foundation models to discover articulated parts and track their motion, and jointly optimizes part poses and joint parameters to reconstruct functional 3D digital twins directly from video.

## Method

### Overall Architecture
FunREC is a training-free optimization pipeline. The input is an egocentric RGB-D interaction video. The pipeline proceeds as follows: (1) segment the video into static and dynamic clips; (2) for each dynamic clip, estimate camera poses, compute sparse 3D trajectories, and perform articulation-aware motion clustering to discover moving parts; (3) perform pixel-level segmentation and joint optimization of pose and joint parameters for discovered moving parts; (4) reconstruct TSDF volumes separately for the static scene and moving parts; (5) globally align all clips to produce a unified functional digital twin.

### Key Designs

1. **Clip Construction and Dynamic/Static Classification**

    - **Function**: Decomposes the long video into independently processable clips and automatically identifies the temporal intervals in which interactions occur, along with joint types.
    - **Mechanism**: A vision-language model (VLM) automatically segments the video into interaction clips (dynamic) and non-interaction clips (static), while predicting the joint type of each interaction (revolute or prismatic).
    - **Design Motivation**: Clip-level processing reduces problem complexity, and the VLM's semantic understanding automates steps that would otherwise require manual annotation.

2. **Articulation-Aware Motion Clustering**

    - **Function**: Discovers moving parts from sparse 3D point trajectories and separates them from the static background.
    - **Mechanism**: TAPIP3D is first used to obtain sparse 3D point trajectories. Points with motion magnitude below threshold $\epsilon_s$ are filtered out as static. For each moving point, an independent joint hypothesis (line or arc) is fitted, and only points with fitting error below $\epsilon_f$ are retained. HDBSCAN then clusters points by similarity of joint parameters (axis direction, pivot point, motion pattern). The cluster whose consistency score $s_\gamma$ with the interaction object mask provided by VISOR is highest is selected as the manipulated moving part.
    - **Design Motivation**: Discovering moving parts directly at the geometric level is more robust than relying on segmentation models. HDBSCAN requires no preset cluster count, making it suitable for an unknown number of moving parts.

3. **Pixel-Aligned Part Segmentation**

    - **Function**: Obtains dense pixel-level moving part masks from sparse motion trajectories.
    - **Mechanism**: SAM's automatic mask generator is applied to selected keyframes for over-segmentation. Moving and static points are projected onto each keyframe, and the ratio $\gamma_r = n_r^m / (n_r^m + n_r^s + \epsilon)$ of moving to total points is computed per segmented region. Regions exceeding threshold $\eta_m$ are labeled as moving parts. Keyframe masks are used as prompts for SAM2's video propagation module to generate temporally consistent dense segmentation sequences.
    - **Design Motivation**: Direct projection of sparse trajectories can suffer from noise and occlusion. By combining with SAM's over-segmentation and adopting region-level voting instead of point-level projection, segmentation robustness is greatly improved.

4. **Joint Optimization of Part Pose and Joint Parameters**

    - **Function**: Recovers globally consistent part pose sequences and joint parameters from noisy 3D trajectories.
    - **Mechanism**: 3D-3D correspondences are constructed for each frame pair, and SuperRANSAC is used to estimate relative transformations. A pose graph optimization objective $\mathcal{L}(T^m, L^m, \phi^m)$ is formulated, incorporating adjacent-frame constraints, loop-closure constraints (with learnable confidence weights $l_{ij}^m$), and joint parameter constraints. Ceres Solver performs nonlinear optimization with manifold optimization to enforce constraints on joint parameters (e.g., rotation axis on the unit sphere, angle on the unit circle).
    - **Design Motivation**: Frame-by-frame estimation alone leads to cumulative drift. Joint optimization of the pose graph and joint parameters ensures temporal consistency and physical plausibility.

### Loss & Training
- FunREC is a training-free optimization method. The core optimization objective is the pose graph energy:
$$\mathcal{L} = \sum_i f(T_i^m, T_{i+1}^m, T_{i \to i+1}^m) + \sum_{i,j} l_{ij}^m f(T_i^m, T_j^m, T_{i \to j}^m) + \mu \sum_{i,j}(\sqrt{l_{ij}^m} - 1)^2$$
- The static background and moving parts are reconstructed separately using TSDF volumes.
- Global clip alignment uses geometric correspondences extracted by PREDATOR.

## Key Experimental Results

### Main Results — Articulated Motion Estimation

| Method | OmniFun4D Axis Error (°) | Position Error (m) | State Error (°/m) | Failure Rate (%) |
|---|---|---|---|---|
| MonST3R (CoTr3) | 46.8/58.9 | 1.20 | 45.3/0.18 | 11.7 |
| BundleSDF (GT mask) | 38.2/55.9 | 0.95 | 23.4/0.20 | 55.0 |
| **FunREC** | **5.3/5.4** | **0.03** | **5.0/0.02** | **1.7** |

FunREC achieves an axis direction error of only 5.3°, more than 30° lower than BundleSDF, with position error reduced by an order of magnitude.

### 6D Part Pose and Reconstruction Quality

| Method | OmniFun4D ADD-S (%) | CD (cm) | HOI4D ADD-S (%) | CD (cm) |
|---|---|---|---|---|
| MonST3R (GT depth+CoTr3) | 37.12 | 13.9 | 54.83 | 1.3 |
| SpatialTrackerV2 (GT depth) | 29.71 | 9.88 | 60.98 | 0.8 |
| BundleSDF (GT mask) | 22.84 | 17.1 | 53.12 | 1.4 |
| **FunREC** | **78.96** | **3.2** | **79.43** | **0.7** |

ADD-S accuracy more than doubles and Chamfer Distance is substantially reduced.

### Moving Part Segmentation

| Method | OmniFun4D mIoU | HOI4D mIoU | RealFun4D mIoU |
|---|---|---|---|
| MonST3R | 23.6 | 26.8 | 23.7 |
| SpatialTrackerV2 (SAM2) | 6.2 | 5.8 | 13.4 |
| **FunREC** | **77.9** | **76.4** | **74.8** |

mIoU improves by more than 50 percentage points.

### Key Findings
- FunREC leads all baselines by large margins across all three datasets (synthetic/controlled/real) and all four evaluation tasks.
- Baseline methods exhibit high failure rates (BundleSDF: 55% on OmniFun4D), whereas FunREC achieves near-zero failure rate.
- Even when baselines are provided with GT depth and GT masks, FunREC still outperforms them substantially.
- Joint optimization of pose and joint parameters is the key driver of performance improvement; naively fitting joint parameters from a 3D tracker with RANSAC is far insufficient.

## Highlights & Insights
- **"Interaction as supervision" paradigm**: Multiple scans and manual annotation are unnecessary — human interaction behavior itself constitutes the best supervision signal for functional understanding. This principle can transfer to other tasks requiring object-level functional understanding.
- **Training-free system design**: The pipeline is entirely composed of existing foundation models (VLM, TAPIP3D, SAM2, RoMA, PREDATOR) without training any new model, demonstrating the strong potential of foundation model composition.
- **Sparse-to-dense segmentation strategy**: The three-step strategy of sparse 3D trajectories → region-level voting → SAM2 video propagation elegantly addresses the challenge of obtaining accurate dense segmentation from noisy sparse signals.
- **New dataset contributions**: RealFun4D (351 real interaction videos from 60 apartments across 4 countries) and OmniFun4D (127 simulated interaction sequences) fill a critical data gap in functional scene understanding.

## Limitations & Future Work
- Requires RGB-D input (depth sensor), which limits practical deployment scenarios.
- Handles only single articulated part interactions at a time; complex scenarios with multiple simultaneously moving parts cannot be handled.
- Erroneous joint type classification by the VLM propagates failures throughout the downstream pipeline.
- Parts with very small motion (below threshold $\epsilon_s$) or those entirely occluded by hands may go undetected.
- 3D point trackers can be inaccurate under occlusion; although the pipeline includes filtering mechanisms, heavy occlusion remains a challenge.

## Related Work & Insights
- **vs. MultiScan**: MultiScan requires two scans of the same room plus manual alignment, whereas FunREC automates the entire process from a single interaction video, offering substantially greater practical utility.
- **vs. BundleSDF**: BundleSDF requires GT masks and a fixed camera, and assumes the object has been pre-scanned. FunREC requires no prior information and still outperforms by a large margin under more challenging settings.
- **vs. ArtGS**: ArtGS requires multi-view captures of two static states (open and closed), while FunREC processes continuous video, which is more natural and practical.
- **vs. 4D reconstruction methods (MonST3R)**: These methods lack understanding of articulation semantics and cannot distinguish between revolute and prismatic joints, let alone estimate joint parameters.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First method to reconstruct scene-level functional digital twins directly from egocentric interaction video.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets (including two newly introduced), four evaluation tasks, multiple baselines, with consistently large performance gaps.
- Writing Quality: ⭐⭐⭐⭐ Method is clearly described and experimental results are convincing.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to embodied intelligence and robotic scene understanding; application demonstrations (URDF export, robotic interaction) validate practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)
- [\[CVPR 2026\] DeepShapeMatchingKit: Accelerated Functional Map Solver and Shape Matching Pipelines Revisited](deepshapematchingkit_accelerated_functional_map_solver.md)
- [\[CVPR 2026\] Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery](reparameterized_tensor_ring_functional_decomposition_for_multi-dimensional_data_.md)

</div>

<!-- RELATED:END -->
