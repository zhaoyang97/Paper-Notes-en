---
title: >-
  [Paper Note] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation
description: >-
  [ECCV 2024][Human Understanding][Multi-person pose estimation] By leveraging the multi-view static camera infrastructure deployed in the 2022 FIFA World Cup stadiums, this work constructs WorldPose, the first large-scale multi-person global 3D pose estimation dataset. It contains approximately 2.5 million 3D poses and over 120 km of global trajectories, revealing the severe challenges that existing global pose estimation methods face in multi-person scenarios.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Multi-person pose estimation"
  - "global trajectory"
  - "SMPL"
  - "dataset"
  - "sports analysis"
date: 2026-05-08
content_hash: 9dff9e93a0f63eb8
---

# WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation

**Conference**: ECCV 2024  
**arXiv**: [2501.02771](https://arxiv.org/abs/2501.02771)  
**Code**: [https://eth-ait.github.io/WorldPoseDataset/](https://eth-ait.github.io/WorldPoseDataset/) (dataset page)  
**Area**: Human Understanding  
**Keywords**: Multi-person pose estimation, global trajectory, SMPL, dataset, sports analysis

## TL;DR
By leveraging the multi-view static camera infrastructure deployed in the 2022 FIFA World Cup stadiums, this work constructs WorldPose, the first large-scale multi-person global 3D pose estimation dataset. It contains approximately 2.5 million 3D poses and over 120 km of global trajectories, revealing the severe challenges that existing global pose estimation methods face in multi-person scenarios.

## Background & Motivation
Multi-person 3D human pose estimation is of great value in sports science, social interaction analysis, and crowd behavior research. However, existing datasets have significant limitations: most of them only focus on single-person or few-person scenarios, either confined to laboratory environments (e.g., CMU Panoptic with a maximum of 8 people) or only providing relative camera coordinates rather than global coordinates (e.g., 3DPW). This prevents researchers from evaluating the real-world performance of methods in large-scale outdoor multi-person scenarios.

Key Challenge: Coordinated and dynamic activities of a large number of people in the real world typically occur in vast outdoor areas and involve moving cameras, but existing datasets cannot capture this complexity. Marker-based methods are infeasible in large areas, and wearable sensors suffer from severe drift issues.

Key Insight: The authors leverage the multi-view static cameras (16-18 1080p cameras covering a 105x68 meter pitch) deployed for the VAR system in the FIFA World Cup, building a high-precision global 3D pose dataset through a carefully designed data acquisition pipeline. Core Idea: **Borrow the professional camera infrastructure of elite sports events, build upon classical optical methods, and obtain multi-person global pose data of unprecedented scale and accuracy through multi-stage calibration and optimization.**

## Method

### Overall Architecture
The entire data acquisition pipeline is divided into three key steps: (1) static camera calibration $\rightarrow$ (2) 3D human pose and SMPL parameter estimation $\rightarrow$ (3) mobile broadcast camera calibration. The input consists of 16-18 channels of HD static camera video from the stadium, and the output is the SMPL pose parameters $\Omega = (\theta_r, \theta_b, t, \beta)$ of all players and the broadcast camera parameters $\Lambda_b$.

### Key Designs
1. **Three-stage static camera calibration**: 

    - Stage 1: Approximate the pitch as a plane, and use the official FIFA 3D LiDAR measurement data to obtain the 2D homography $H_c$ between world coordinates and image coordinates.
    - Stage 2: Based on the homography initialization, jointly solve for camera intrinsic parameters, extrinsic parameters, and distortion coefficients through non-linear optimization, taking into account the actual pitch undulations (about 20cm of drainage camber).
    - Stage 3: Extract the field line edges in the image to construct a distance map $D$, and project sampled 3D template field line points to minimize the distance to the nearest edge pixels, achieving sub-pixel accuracy.
    - Design Motivation: Due to the massive scale of the stadium, calibration using only corner points is not accurate enough; thus, dense field line information must be incorporated for photometric optimization.

2. **3D human pose estimation and tracking**: 

    - Use ByteTrack to detect player bounding boxes and ViTPose to estimate 2D keypoints. Since low resolution degrades performance, the detection and keypoint models are fine-tuned.
    - Use pitch projection to filter out false detections in the spectator area.
    - Perform cross-camera 3D tracking using an affinity function based on point-to-ray distance: $A(P_i^{t-1}, p_{j,c}^t) = -\text{PointToRayDist}(P_i, \Pi^{-1}(p_{j,c}))$
    - Bundle Adjustment is used to further optimize camera parameters and 3D keypoints.
    - Design Motivation: Players have low resolution in images, move rapidly, and suffer from frequent occlusions; therefore, domain-specific knowledge must be used to constrain tracking.

3. **SMPL parameter fitting and broadcast camera calibration**: 

    - Direct estimation of SMPL shape parameters $\beta$ through bone length matching.
    - Joint optimization of the data term $E_{data}$, smoothness term $E_{smooth}$, and shape regularization term $E_{shape}$.
    - Broadcast camera calibration: Semi-automatic initialization using commercial software, followed by joint optimization with 3D player poses and 2D field line markings.
    - The matching function combines IoU similarity and bone cosine similarity: $sim = sim_{IoU} \cdot sim_{bone}$
    - Design Motivation: Broadcast cameras undergo translation, rotation, and zooming; calibration based solely on field lines is inaccurate, so player poses are required as additional constraints.

### Loss & Training
The final loss for SMPL fitting is a weighted combination:
$$E_{refine} = \lambda_1 E_{data} + \lambda_2 E_{smooth} + \lambda_3 E_{shape}$$

Broadcast camera calibration loss:
$$E_{calib} = \lambda_4 E_{field} + \lambda_5 E_{player}$$
where the Geman-McClure robust function $\rho$ is used to handle outliers.

## Key Experimental Results

### Main Results
Pipeline accuracy validation in comparison with the Vicon system:

| Configuration | G-MPJPE (mm) ↓ | PA-MPJPE (mm) ↓ |
|------|----------------|-----------------|
| Base | 83.5 | 70.8 |
| + BA (S and P) | 86.2 | 70.7 |
| + BA (S only) | 548.4 | 75.4 |
| + BA + SMPL | **80.0** | **66.3** |

Performance of SOTA methods on WorldPose:

| Method | G-MPJPE (mm) ↓ | PA-MPJPE (mm) ↓ | Per-Meter Drift (cm/m) ↓ |
|------|----------------|-----------------|--------------------------|
| HybrIK | N/A | 78.8 | N/A |
| 4DHuman | N/A | 116.5 | N/A |
| GLAMR | 18,888.9 | 85.2 | 53.3 |
| SLAHMR | 8,334.1 | 163.9 | 17.6 |
| SLAHMR w/ GT Cameras | 5,837.2 | 199.6 | 10.7 |
| GLAMR (per-person) | 3,749.7 | 85.2 | 8.3 |
| SLAHMR (per-person) | 4,699.5 | 163.9 | 8.9 |

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| Field lines only BA | G-MPJPE surges to 548.4mm, overfitting field line markings |
| Field lines + Keypoints BA | Avoids overfitting, but G-MPJPE increases slightly |
| Full pipeline (BA+SMPL) | Best result, 8cm global error |
| SLAHMR w/o HuMoR | Performance actually improves because ground plane estimation is unreliable |
| SLAHMR with GT cameras | G-MPJPE and drift are halved, indicating that SLAM failure is the main cause |

### Key Findings
- The WorldPose pipeline achieves an 8cm global joint error against the Vicon reference, achieving unprecedented accuracy in a 7000 $m^2$ pitch.
- GLAMR and SLAHMR perform extremely poorly in global trajectory estimation (with G-MPJPE reaching several meters or even tens of meters), but their PA-MPJPE is relatively reasonable.
- DROID-SLAM performs poorly in virtually textureless pitches and highly dynamic crowd backgrounds, which is the main source of global error.
- Existing methods face significant difficulties in determining relative positioning between players, even when assuming a shared ground plane.
- The optimized PA-MPJPE of SLAHMR and GLAMR is surprisingly worse than their initialization methods.

## Highlights & Insights
- **Unique Data Source**: Cleverly utilizes the professional infrastructure of the FIFA World Cup, a resource advantage that general research teams cannot replicate.
- **Filling a Key Gap**: Secures a unique position in the 3D pose dataset landscape—simultaneously featuring multi-person (16 people/frame), global trajectories, large areas, moving cameras, and high-quality SMPL fitting.
- **In-depth Benchmark Analysis**: Beyond merely providing a dataset, it reveals the fundamental limitations of existing methods through detailed experiments.
- **Per-person vs. Global Error Comparison**: Provides insightful analysis showing that relative position estimation is the core bottleneck.
- **Staggering Dataset Scale**: 2.5 million 3D poses and 120 km of walking distance, far exceeding all existing datasets.

## Limitations & Future Work
- Heavily relies on 2D detection quality and static camera layouts, requiring considerable human intervention to correct false detections and tracking errors.
- Only contains male tournament data, leading to a lack of gender representation.
- The data acquisition pipeline is highly expensive, making it difficult to replicate at scale in other scenarios.
- Does not provide a new methodology solution—the primary contribution is the dataset and benchmark, with limited methodological innovation.
- Future extension possible to other sports events or more complex outdoor scenarios.

## Related Work & Insights
- Complements CMU Panoptic (indoor, few people) and BEDLAM (synthetic, uncoordinated actions).
- Validates the performance of global pose estimation (GLAMR, SLAHMR) in real-world large-scale scenarios, providing important references for improvements in this direction.
- Suggests that future research should focus on: (1) SLAM improvements in low-texture environments, (2) joint estimation of multi-person relative positioning, and (3) robust estimation with moving + zooming cameras.
- Directly accelerates AI applications in the field of sports analytics.

## Rating
- Novelty: ⭐⭐⭐⭐ (Dataset construction method is solid but lacks methodological breakthroughs)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Vicon validation + comprehensive SOTA evaluation + ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, in-depth analysis)
- Value: ⭐⭐⭐⭐⭐ (Fills an important dataset gap, of great significance to field development)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)
- [\[ECCV 2024\] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [\[ECCV 2024\] UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)
- [\[ECCV 2024\] 3D Hand Pose Estimation in Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)

</div>

<!-- RELATED:END -->
