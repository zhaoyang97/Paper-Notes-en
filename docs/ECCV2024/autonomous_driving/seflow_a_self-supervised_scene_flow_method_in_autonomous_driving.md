---
title: >-
  [Paper Note] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving
description: >-
  [ECCV 2024][Autonomous Driving][Scene Flow] SeFlow proposes integrating traditional ray-casting-based dynamic point classification into a self-supervised scene flow learning pipeline. By utilizing tailored dynamic/static loss functions and a cluster-based object-level motion consistency constraint, it achieves state-of-the-art (SOTA) self-supervised scene flow performance on Argoverse 2 and Waymo at real-time speeds (48ms/frame), even outperforming some supervised methods.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Scene Flow"
  - "Self-Supervised Learning"
  - "Point Cloud"
  - "Dynamic Awareness"
date: 2026-05-08
content_hash: c47780be182dc73a
---

# SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving

**Conference**: ECCV 2024  
**arXiv**: [2407.01702](https://arxiv.org/abs/2407.01702)  
**Code**: [https://github.com/KTH-RPL/SeFlow](https://github.com/KTH-RPL/SeFlow)  
**Area**: Autonomous Driving  
**Keywords**: Scene Flow, Self-Supervised Learning, Point Cloud, Dynamic Awareness, Autonomous Driving

## TL;DR
SeFlow proposes integrating traditional ray-casting-based dynamic point classification into a self-supervised scene flow learning pipeline. By utilizing tailored dynamic/static loss functions and a cluster-based object-level motion consistency constraint, it achieves state-of-the-art (SOTA) self-supervised scene flow performance on Argoverse 2 and Waymo at real-time speeds (48ms/frame), even outperforming some supervised methods.

## Background & Motivation

**Background**: Scene flow estimation aims to predict the 3D motion vector of each point in consecutive LiDAR scans. Currently, top-performing methods mostly depend on annotated data (e.g., FastFlow3D, DeFlow), which is extremely expensive to obtain. Self-supervised methods (e.g., NSFP, ZeroFlow) do not require annotations but still exhibit a performance gap.

**Limitations of Prior Work**: Self-supervised scene flow methods face two major challenges: (1) **Imbalance in point distribution** — approximately 86% of points belong to the static background, leading models to predict zero flow (conservative estimation); (2) **Neglect of object-level motion constraints** — Chamfer distance-based loss functions suffer from nearest-neighbor mismatching, resulting in inconsistent flow predictions within the same rigid object (e.g., the flow in the middle of a large truck is incorrectly predicted as zero).

**Key Challenge**: The nearest neighbor assumption of the Chamfer distance loss does not hold for dynamic objects. When an object translates, points near the middle of its surface lie very close to their nearest neighbors in the next frame (due to overlapping areas), causing the flow to be severely underestimated.

**Goal**  
   - Settle the data imbalance between static and dynamic points within a self-supervised framework.  
   - Correct the faulty correspondences of Chamfer distance on dynamic objects.  
   - Maintain real-time inference speed ($\ge 10\text{Hz}$).

**Key Insight**: Draw inspiration from dynamic-aware mapping methods in the SLAM field (e.g., DUFOMap) to classify point clouds into dynamic and static categories using ray-casting, and then design specific loss functions for each category.

**Core Idea**: Leverage traditional ray-casting dynamic classification to design class-specific losses and cluster consistency constraints for self-supervised scene flow, addressing data imbalance and mismatching issues.

## Method

### Overall Architecture
Input: Two consecutive point cloud frames $\mathcal{P}_t, \mathcal{P}_{t+1}$ and the ego-motion $\mathbf{T}_{t,t+1}$. Ground points are first removed, and the total flow is decomposed as $\hat{\mathcal{F}} = \mathcal{F}_{ego} + \Delta\hat{\mathcal{F}}$, where the network only predicts the residual flow $\Delta\hat{\mathcal{F}}$ after removing ego-motion. Concurrently, DUFOMap is employed to categorize $\mathcal{P}_t$ into dynamic/static points, followed by HDBSCAN clustering on the dynamic points. Finally, four complementary loss functions are utilized for self-supervised training. During inference, dynamic classification and clustering are not required.

### Key Designs

1. **Dynamic Point Classification (DUFOMap)**:

    - **Function**: Classifies the point cloud into a dynamic set $\mathcal{P}_{t,d}$ and a static set $\mathcal{P}_{t,s}$ during the training phase.
    - **Mechanism**: Utilizes the ray-casting principle — if a spatial region is observed to be empty at one timestamp but has points at another, those points must be dynamic. DUFOMap operates on CPUs at sensor frame rates with negligible computational overhead.
    - **Design Motivation**: Compared to the ground truth labels of supervised methods, this classification is obtained "almost for free" and is sufficiently accurate. Decoupling it from the learning pipeline (used during training, omitted during inference) preserves the flexibility of the method.

2. **Class-Specific Loss Functions (3 New Losses)**:

    - **Function**: Formulates distinct training targets for static and dynamic points.
    - **Mechanism**:
        - **Dynamic Chamfer Distance** ($\mathcal{L}_{dcham}$): Calculates the Chamfer distance only among dynamic points, preventing it from being overwhelmed by the zero flow of massive static points.
        - **Static Flow Loss** ($\mathcal{L}_{static}$): Forces the network output $\Delta\hat{\mathcal{F}}$ for static points to be zero: $\mathcal{L}_{static} = \frac{1}{|\mathcal{P}_{t,s}|}\sum_{p \in \mathcal{P}_{t,s}} \|\Delta\hat{\mathcal{F}}(p)\|_2^2$
        - **Dynamic Cluster Flow** ($\mathcal{L}_{dcls}$): Clusters dynamic points into object candidates using HDBSCAN, identifies the point with the largest nearest-neighbor distance in each cluster as the motion upper bound $\tilde{f}_{c_i}$, and constrains all points in the cluster to move consistently toward this upper bound: $\mathcal{L}_{c_i} = \sum_{p_j \in \mathcal{P}_{c_i}} \|\hat{f}_{p_j} - \tilde{f}_{c_i}\|_2^2$
    - **Design Motivation**: $\mathcal{L}_{dcham}$ addresses data imbalance, $\mathcal{L}_{static}$ eliminates matching noise on static points, and $\mathcal{L}_{dcls}$ corrects the systematic underestimation of Chamfer distance in the middle of object surfaces (core innovation).

3. **DeFlow + GRU Backbone**:

    - **Function**: Efficiently processes large-scale point clouds (80K-177K points per frame) and predicts point-wise flow.
    - **Mechanism**: Employs voxelized encoding coupled with a GRU iterative refinement decoder. The GRU module takes voxel features as hidden states and selectively updates them at each iteration based on point features. The optimized voxel features after multiple iterations are concatenated with the original point features to yield the final point-wise features.
    - **Design Motivation**: Compared to the FastFlow3D backbone, this design improves inference efficiency without compromising accuracy under coarse resolution settings.

### Loss & Training
- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{cham} + \mathcal{L}_{static} + \mathcal{L}_{dcham} + \mathcal{L}_{dcls}$
- The four loss terms do not require extra hyperparameter weights for balancing (all weighted equally with weight = 1).
- Dynamic classification and clustering are only used during training; inference only requires forward propagation through the network.

## Key Experimental Results

### Main Results: Argoverse 2 Test Set (EPE ↓)

| Method | Type | Inference Time | EPE 3-way | EPE FD | EPE FS | EPE BS |
|------|------|----------|-----------|--------|--------|--------|
| FastFlow3D | Supervised | 34ms | 0.0782 | 0.2072 | 0.0253 | 0.0020 |
| DeFlow | Supervised | 48ms | 0.0534 | 0.1340 | 0.0232 | 0.0029 |
| NSFP | Self-supervised | 32s | 0.0685 | 0.1503 | 0.0302 | 0.0248 |
| ZeroFlow | Self-supervised | 34ms | 0.0814 | 0.2109 | 0.0254 | 0.0080 |
| **SeFlow** | **Self-supervised** | **48ms** | **0.0628** | **0.1525** | **0.0321** | **0.0038** |

### Ablation Study: Contributions of Loss Terms (Argoverse 2 Val)

| $\mathcal{L}_{cham}$ | $\mathcal{L}_{dcham}$ | $\mathcal{L}_{static}$ | $\mathcal{L}_{dcls}$ | EPE 3-way | EPE FD | EPE FS | EPE BS |
|---|---|---|---|---|---|---|---|
| ✓ | | | | 0.0962 | 0.203 | 0.052 | 0.033 |
| ✓ | ✓ | | | 0.0916 | 0.181 | 0.059 | 0.035 |
| ✓ | ✓ | ✓ | | 0.0779 | 0.220 | 0.012 | 0.002 |
| ✓ | ✓ | ✓ | ✓ | **0.0643** | **0.160** | **0.029** | **0.004** |

### Ablation Study: Training Data Volume

| Data Volume | EPE 3-way | EPE FD |
|--------|-----------|--------|
| SeFlow 10% | 0.094 | 0.234 |
| SeFlow 20% | 0.078 | 0.197 |
| SeFlow 50% | 0.066 | 0.167 |
| ZeroFlow 100% | 0.088 | 0.231 |
| ZeroFlow 200% | 0.076 | 0.198 |

### Key Findings
- $\mathcal{L}_{dcls}$ is the most critical loss term: its inclusion reduces FD EPE from 0.220 to 0.160 (-27%), proving that the cluster consistency constraint effectively fixes the systematic underestimation of Chamfer distance.
- $\mathcal{L}_{static}$ has a significant effect on static points (FS: -80%, BS: -94%), although it slightly degrades foreground dynamic accuracy, which is compensated for by $\mathcal{L}_{dcls}$.
- SeFlow exceeds ZeroFlow trained on 100% of the dataset using only 20% of the training data, indicating a 5x increase in data efficiency.
- SeFlow surpasses the supervised FastFlow3D method and is the only self-supervised method capable of running in real-time.
- It is capable of detecting flow missed by GT annotations (such as a pushed shopping cart).

## Highlights & Insights
- **Elegant combination of traditional methods and deep learning**: Utilizing zero-cost ray-casting classification as a prior for self-supervised signals without altering the inference pipeline is a paradigm of "enhancing with traditional methods during training, while keeping pure neural networks during inference", which can be generalized to other self-supervised tasks.
- **Bound constraints instead of variance constraints**: Rather than directly constraining the mean or variance of flow within a cluster (since the mean itself might be incorrect), using the maximum distance point as a motion upper bound cleverly bypasses the systematic underestimation of Chamfer distance.
- **Impressive data efficiency**: Achieving performance superior to ZeroFlow with 100% data using only 20% data demonstrates that a solid inductive bias is more critical than scaling up data.

## Limitations & Future Work
- Estimating scene flow for distant and sparse objects remains challenging (as both voxelization and clustering tend to neglect them).
- Dynamic classification in DUFOMap labels "historically moved objects" as dynamic, whereas scene flow is concerned with "whether they are moving in the current frame", leading to a definition mismatch.
- The hyperparameters of the clustering algorithm HDBSCAN might affect performance across different scenarios, and its robustness warrants further validation.
- Exploring multi-modal inputs (camera + point cloud) could further improve the scene flow estimation of distant objects.

## Related Work & Insights
- **vs ZeroFlow**: ZeroFlow employs NSFP as a teacher to generate pseudo-labels (requiring 3.6 GPU months). In contrast, SeFlow circumvents the need for a teacher entirely, directly utilizing physical priors (ray-casting) to retrieve self-supervision signals, which is more efficient and accurate (EPE 0.063 vs. 0.081).
- **vs DeFlow**: DeFlow is a supervised method (demanding GT flow annotations); its EPE of 0.053 is close to SeFlow's 0.063. When considering annotation costs, SeFlow offers higher value.
- **vs NSFP**: NSFP requires optimizing an MLP for each frame (30s/frame), rendering it unsuitable for real-time applications. SeFlow runs in just 48ms during inference after training, which is 600+ times faster.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The concept of introducing traditional dynamic classification into self-supervised scene flow is simple yet effective, and the cluster upper-bound constraint is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive, featuring dual-dataset validation, detailed ablation studies, data efficiency analysis, and qualitative visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem formulation and intuitive diagrams (particularly Fig. 3, which clarifies why Chamfer distance fails).
- **Value**: ⭐⭐⭐⭐ Real-time SOTA self-supervised scene flow method with open-source code, highlighting high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow](../../CVPR2025/autonomous_driving/voteflow_enforcing_local_rigidity_in_self-supervised_scene_flow.md)
- [\[ECCV 2024\] Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection](equivariant_spatio-temporal_self-supervision_for_lidar_object_detection.md)
- [\[ECCV 2024\] Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [\[ECCV 2024\] Risk-Aware Self-Consistent Imitation Learning for Trajectory Planning in Autonomous Driving](risk-aware_self-consistent_imitation_learning_for_trajectory_planning_in_autonom.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](../../ICCV2025/autonomous_driving/ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)

</div>

<!-- RELATED:END -->
