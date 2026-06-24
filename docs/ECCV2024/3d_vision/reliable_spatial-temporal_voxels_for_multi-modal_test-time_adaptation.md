---
title: >-
  [Paper Note] Reliable Spatial-Temporal Voxels For Multi-Modal Test-Time Adaptation
description: >-
  [ECCV 2024][3D Vision][Multi-Modal Test-Time Adaptation] This paper proposes Latte (ReLiable Spatial-temporal Voxels), a multi-modal test-time adaptation method that constructs spatial-temporal voxels (ST voxels) via sliding window frame aggregation and computes spatial-temporal entropy (ST entropy) to evaluate prediction reliability, thereby enabling adaptive cross-modal learning and achieving SOTA performance on three MM-TTA benchmarks.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Multi-Modal Test-Time Adaptation"
  - "3D Semantic Segmentation"
  - "Spatial-Temporal Voxels"
  - "Cross-Modal Learning"
  - "Online Adaptation"
date: 2026-05-08
content_hash: 2b7b4a8b57f681be
---

# Reliable Spatial-Temporal Voxels For Multi-Modal Test-Time Adaptation

**Conference**: ECCV 2024  
**arXiv**: [2403.06461](https://arxiv.org/abs/2403.06461)  
**Code**: [https://sites.google.com/view/eccv24-latte](https://sites.google.com/view/eccv24-latte) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Multi-Modal Test-Time Adaptation, 3D Semantic Segmentation, Spatial-Temporal Voxels, Cross-Modal Learning, Online Adaptation

## TL;DR

This paper proposes Latte (ReLiable Spatial-temporal Voxels), a multi-modal test-time adaptation method that constructs spatial-temporal voxels (ST voxels) via sliding window frame aggregation and computes spatial-temporal entropy (ST entropy) to evaluate prediction reliability, thereby enabling adaptive cross-modal learning and achieving SOTA performance on three MM-TTA benchmarks.

## Background & Motivation

**Background**: 3D semantic segmentation is a fundamental task for autonomous driving and robot navigation, where multi-modal sensors (camera + LiDAR) are widely adopted. Multi-modal test-time adaptation (MM-TTA) aims to adapt models online to an unlabeled target domain during the test phase without accessing source domain data. Prior MMTTA methods perform adaptation at the single-frame level through cross-modal pseudo-labels and prediction consistency.

**Limitations of Prior Work**: Existing MM-TTA methods (such as MMTTA) rely on cross-modal information per frame for adaptation, but ignore an important fact: predictions of geometric neighborhoods in consecutive frames are highly correlated. Due to domain shift, single-frame predictions are usually unstable—the same object may be predicted as different categories in consecutive frames (e.g., a car is correctly identified in one frame but misclassified in adjacent frames). This temporal inconsistency leads to: (1) unreliable predictions being incorrectly treated as "reliable" and propagated to another modality, causing error accumulation; (2) averaging multi-augmented frames can alleviate this, but the computational cost increases linearly with the number of augmented frames.

**Key Challenge**: MM-TTA requires reliable supervisory signals to update models online, but single-frame predictions are precisely unreliable under domain shifts. How to obtain more stable reliability estimates without introducing significant computational overhead is the key challenge.

**Goal**: (1) How to leverage spatial-temporal associations between consecutive frames to obtain more reliable prediction estimates? (2) How to efficiently evaluate the reliability of each modality in different spatial regions? (3) How to perform adaptive cross-modal learning based on reliability estimates?

**Key Insight**: The authors observe that 3D space can be partitioned into voxels, and points from different frames within the same voxel can be viewed as different observations of the same semantic object. Reliable predictions should be consistent and certain within a spatial-temporal neighborhood. By aggregating predictions of adjacent frames within the same voxel and evaluating their consistency, the credibility of each modality and region can be determined more reliably than using single-frame predictions.

**Core Idea**: Aggregating consecutive frame predictions through sliding-window voxelization and measuring prediction reliability with spatial-temporal entropy to achieve adaptive cross-modal test-time adaptation.

## Method

### Overall Architecture

The pipeline of Latte: (1) for each modality (2D/3D), a student-teacher architecture is used to generate frame-by-frame predictions; (2) points from consecutive frames are aggregated through a sliding window and voxelized to construct ST voxels; (3) the spatial-temporal entropy (ST entropy) of teacher predictions within each ST voxel is calculated, where low entropy indicates reliability, and unreliable voxels with high entropy are filtered; (4) based on ST entropy-guided cross-modal weighting, the student learns from teacher predictions of the more reliable modality.

### Key Designs

1. **Slide Window Aggregation & Voxelization**:

    - **Function**: Establishes spatial-temporal correspondences between consecutive frames in an efficient manner.
    - **Mechanism**: Given the current frame $i$ as a query, frames within the time window $\{j : |j-i| \leq w_t\}$ are aligned to the same coordinate system through pose transformation $\mathbf{T}_{j \to i}$, merged, and then voxelized with a voxel size of $\mathbf{s}$. Points within the same voxel from different frames are treated as spatial-temporal correspondences. The sliding window ($w_t=3$) allows each frame to have overlapping evaluation intervals, capturing local consistency better than "all-frame merging" or "non-overlapping blocking".
    - **Design Motivation**: All-frame merging fails to highlight inconsistencies within short time windows, and frame-to-frame corresponding points are too sparse to be representative. The sliding window takes a compromise, ensuring both sufficient correspondence points and a focus on local consistency. Online pose estimation is provided by KISS-ICP with manageable computational overhead.

2. **ST Voxels & ST Entropy**:

    - **Function**: Quantifies the prediction reliability of each modality in each spatial region.
    - **Mechanism**: An ST voxel contains queries (current frame student prediction $\mathbf{p}_q^m$) and references (multi-frame teacher predictions $\mathbf{p}_r^m$ within the window). The Shannon entropy of the average class probability of reference predictions serves as the ST entropy: $E_{i,k}^m = -\sum_c \bar{p}_{r,c}^m \log \bar{p}_{r,c}^m$. High ST entropy indicates that reference predictions are inconsistent or uncertain within the spatial-temporal neighborhood—these voxels are filtered out by an $\alpha$-quantile ($\alpha=0.9$). Low ST entropy indicates that multi-frame teacher predictions in this region are consistent and confident, and thus should be trusted.
    - **Design Motivation**: Compared with single-frame point-level entropy, ST entropy integrates information from spatial-temporal neighborhoods across multiple frames, providing a more robust reliability metric. Experiments demonstrate that ST entropy outperforms point-level entropy by 2.0% mIoU on average across three benchmarks.

3. **ST Voxel-Assisted Adaptive Cross-Modal Learning**:

    - **Function**: Adaptively performs cross-modal knowledge transfer according to modality reliability.
    - **Mechanism**: At the voxel level, cross-modal weighting is computed: $w_v^{2D} = \frac{\exp(E_{i,k}^{2D})}{\exp(E_{i,k}^{2D}) + \exp(E_{i,k}^{3D})}$ (modalities with higher entropy receive higher weights, meaning that modality, acting as the student in the KL divergence loss, is guided). The cross-modal consistency loss is: $\mathcal{L}_{xM} = w_v^{2D} D_{KL}(\bar{p}_q^{3D} \| \bar{p}_r^{2D}) + w_v^{3D} D_{KL}(\bar{p}_q^{2D} \| \bar{p}_r^{3D})$. At the point level, the ST entropy is also propagated to each point to generate weighted cross-modal pseudo-labels: $\mathbf{p}^{xM} = w_p^{2D} \mathbf{p}^{2D} + w_p^{3D} \mathbf{p}^{3D}$.
    - **Design Motivation**: In different scenarios, the reliability of 2D and 3D modalities can be entirely different (e.g., LiDAR is more reliable in outdoor scenes, while cameras are more reliable in low-texture regions). Adaptive weighting allows the reliable modality to guide the unreliable one, avoiding noise propagation.

### Loss & Training

Total loss: $\mathcal{L} = \sum_t \mathcal{F}(\mathbf{p}_t^m, \mathbf{y}_t^{xM}) + \frac{\lambda_{xM}}{B} \sum_t \sum_k \mathcal{L}_{t,k}^{xM}$, where $\mathcal{F}$ is the cross-entropy and $\mathbf{y}_t^{xM}$ is the cross-modal pseudo-label. The teacher is updated via EMA: $\tilde{\theta}_t^m = \lambda_s \tilde{\theta}_{t-1}^m + (1-\lambda_s) \theta_t^m$ with $\lambda_s = 0.99$. Only BN layer parameters are updated. The one-pass protocol is strictly followed: evaluate before updating. $\lambda_{xM} = 0.3$.

## Key Experimental Results

### Main Results

| Method | MM | U-to-S (xM) | A-to-S (xM) | S-to-S (xM) | Avg |
|------|-----|-------------|-------------|-------------|-----|
| Source only | ✗ | 43.9 | 44.3 | 38.2 | 42.1 |
| TENT | ✗ | 41.1 | 49.1 | 37.5 | 42.6 |
| SAR | ✗ | 43.9 | 50.3 | 32.9 | 42.4 |
| xMUDA+PL | ✓ | 43.0 | 50.9 | 36.3 | 43.4 |
| MMTTA | ✓ | 45.4 | 53.7 | 35.5 | 44.9 |
| **Latte** | ✓ | **46.0** | **54.3** | **41.6** | **47.3** |

### Ablation Study

| No. | Configuration | U-to-S (xM) | A-to-S (xM) | S-to-S (xM) |
|-----|------|-------------|-------------|-------------|
| 0 | Source only | 43.9 | 44.3 | 38.2 |
| 7 | w/o Quantile Filtering | 45.4 | 53.2 | 41.7 |
| 8 | Point-level Entropy instead of ST Entropy | 45.1 | 52.7 | 38.2 |
| 9 | Full Latte | **46.0** | **54.3** | **41.6** |

### Key Findings

- **The S-to-S benchmark is the most convincing**: This is a challenging synth-to-real scenario, where all prior methods (including MMTTA) perform worse than the Source only baseline. Only Latte achieves a positive adaptation of +3.4%. This demonstrates the unique value of spatial-temporal consistency under extreme domain shifts.
- ST entropy yields an average improvement of 2.0%+ mIoU over point-level entropy, validating that multi-frame aggregation is more reliable than single-frame evaluation.
- Quantile filtering ($\alpha=0.9$) improves results by 0.6-1.1% on U-to-S and A-to-S, with minor differences on S-to-S.
- A voxel size of 0.2m is optimal across all benchmarks—too large leads to blurred semantic boundaries, while too small results in insufficient corresponding points.
- Small windows ($w_t=3$) perform better than large windows, indicating that local temporal consistency is more informative than global consistency.
- Almost all multi-modal methods outperform single-modality TTA, showing the critical value of cross-modal information for TTA.

## Highlights & Insights

- **Ingenious Design of Spatial-Temporal Voxels**: Leverages the natural spatial-temporal structure of 3D point clouds to evaluate prediction reliability—predictions of the same voxel at different times should be consistent. This idea of "measuring reliability with consistency" can be transferred to other temporal-aware tasks (e.g., object tracking, SLAM).
- **Sliding Window vs. Global Aggregation**: Global aggregation is undesirable, and frame-to-frame association is too sparse, so the sliding window serves as a compromise. This design choice offers valuable insights for temporal modeling—more information is not always better; finding the appropriate temporal granularity is key.
- **Adaptive Cross-Modal Weighting**: Dynamically adjusting the direction of cross-modal learning at the voxel level based on reliability is more fine-grained than a globally fixed modality fusion strategy. This has direct reference value for scenarios requiring multi-sensor fusion, such as autonomous driving.

## Limitations & Future Work

- Latte struggles to correct predictions that make temporally consistent errors; if a region is incorrectly predicted across all frames, the ST entropy remains low, and the model will mistakenly trust it.
- It relies on SLAM algorithms to provide pose estimation; inaccurate poses will degrade the quality of voxelization.
- Updating only BN layer parameters limits adaptation capacity; exploring strategies with more adaptable parameters is a potential direction.
- Voxel size and window size require manual tuning; although the optimal values are consistent across the three benchmarks (0.2m, $w_t=3$), they might vary on new datasets.
- Other 3D perception tasks, such as object detection, are not considered.

## Related Work & Insights

- **vs MMTTA (Shin et al., CVPR 2022)**: MMTTA adapts via single-frame cross-modal pseudo-label refinement, but fails under challenging scenarios like S-to-S due to unstable single-frame predictions. Latte obtains more stable reliability estimates through spatial-temporal aggregation.
- **vs CoTTA (Wang et al., CVPR 2022)**: CoTTA mitigates temporal instability using augmentation invariance, but its computational cost scales linearly with the number of augmentations. Latte leverages naturally occurring inter-frame consistency, which is computationally more efficient.
- **vs GIPSO (Saltori et al., ECCV 2022)**: GIPSO merges frames globally for propagation but cannot highlight local inconsistencies. Latte's sliding window evaluates local consistency more effectively.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce spatial-temporal correspondence into MM-TTA; the designs of ST voxels and ST entropy are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks, various baselines, detailed ablations (component effectiveness, aggregation mechanisms, parameter sensitivity, qualitative analysis).
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rigorous method description, and complete mathematical formulations.
- Value: ⭐⭐⭐⭐ Direct practical value for online adaptation in autonomous driving, with prominent advantages under challenging scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation](cloudfixer_test-time_adaptation_for_3d_point_clouds_via_diffusion-guided_geometr.md)
- [\[ICCV 2025\] 3D Test-time Adaptation via Graph Spectral Driven Point Shift](../../ICCV2025/3d_vision/3d_testtime_adaptation_via_graph_spectral_driven_point_shift.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](../../NeurIPS2025/3d_vision/pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)
- [\[ICLR 2026\] SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling](../../ICLR2026/3d_vision/spacecontrol_introducing_test-time_spatial_control_to_3d_generative_modeling.md)
- [\[CVPR 2026\] Anatomical Domain Shifts: Test-time Heterogeneous Adaptation for 3D Human Pose Prediction](../../CVPR2026/3d_vision/anatomical_domain_shifts_test-time_heterogeneous_adaptation_for_3d_human_pose_pr.md)

</div>

<!-- RELATED:END -->
