---
title: >-
  [Paper Note] UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues
description: >-
  [ECCV 2024][Human Understanding][Multi-view 3D Pose Estimation] Proposes UPose3D, an uncertainty-aware multi-view 3D human pose estimation method. By modeling 2D keypoint uncertainty with Normalizing Flow, utilizing a scalable cross-view point cloud projection fusion strategy, and employing a Pose Compiler module trained on synthetic data, it achieves state-of-the-art performance in Out-of-Distribution (OoD) scenarios without requiring 3D annotations…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Multi-view 3D Pose Estimation"
  - "Uncertainty Modeling"
  - "Cross-view Fusion"
  - "Temporal Modeling"
  - "Synthetic Data"
date: 2026-05-08
content_hash: 8af6e2681fd4ef9e
---

# UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues

**Conference**: ECCV 2024  
**arXiv**: [2404.14634](https://arxiv.org/abs/2404.14634)  
**Code**: None  
**Area**: Human Understanding  
**Keywords**: Multi-view 3D Pose Estimation, Uncertainty Modeling, Cross-view Fusion, Temporal Modeling, Synthetic Data

## TL;DR
Proposes UPose3D, an uncertainty-aware multi-view 3D human pose estimation method. By modeling 2D keypoint uncertainty with Normalizing Flow, utilizing a scalable cross-view point cloud projection fusion strategy, and employing a Pose Compiler module trained on synthetic data, it achieves state-of-the-art performance in Out-of-Distribution (OoD) scenarios without requiring 3D annotations, while remaining competitive with 3D-supervised methods in In-Distribution (InD) scenarios.

## Background & Motivation
Multi-view 3D human pose estimation is crucial in markerless motion capture, especially in gaming and film production where sub-centimeter accuracy is required. Traditional methods first predict 2D keypoints independently for each view and then obtain 3D coordinates through triangulation. However, they rely heavily on the accuracy of individual 2D predictions, performing poorly under occlusions and complex body interactions. Outlier mitigation techniques like RANSAC offer only limited robustness.

The **Key Challenge** consists of three aspects: (1) **Poor Scalability**: Existing cross-view fusion methods (e.g., Epipolar Transformers) suffer from computational complexity that scales rapidly with the number of cameras; (2) **Dependency on 3D Annotations**: End-to-end methods (e.g., MTF-Transformer) require extensive 3D annotated training data, which is scarce in outdoor and in-the-wild scenarios; (3) **Weak Generalizability**: Models trained on fixed indoor setups struggle to generalize to unseen camera configurations and environments.

The **Core Idea** of this work is: **to leverage 2D keypoint uncertainty from two sources (image-level RLE + post-Pose Compiler refinement) for enhanced robustness, avoid reliance on 3D annotations by using synthetic training data, and achieve scalable fusion for arbitrary numbers of cameras through epipolar geometry projection**.

## Method

### Overall Architecture
UPose3D consists of four stages: (1) An RLE-based 2D pose estimator extracts keypoints $\hat{\mu}$ and uncertainties $\hat{\sigma}$ from each view; (2) Cross-view projection: uses epipolar geometry to project keypoints from all views onto the reference view to form a 2D point cloud; (3) The Pose Compiler module refines the keypoint locations $\hat{\mu}'$ and uncertainties $\hat{\sigma}'$ using cross-view and temporal cues; (4) An iterative Maximum Likelihood Estimation (MLE) optimization is performed using predictions from both stages to obtain the final 3D pose.

### Key Designs
1. **RLE-Based Uncertainty Modeling**: 

    - Models the keypoint distribution $P_{\Theta}(x|I)$ using a Residual Log-likelihood Estimation (RLE) head.
    - RLE learns the probability distribution of keypoints via a normalizing flow, simultaneously outputting location predictions $\hat{\mu}$ and uncertainties $\hat{\sigma}$.
    - Compared to heatmap-based methods, RLE is more computationally efficient and differentiable, allowing direct usage in the subsequent MLE optimization.
    - **Design Motivation**: Uncertainty provides a confidence measure for each prediction, allowing the system to automatically discount low-confidence predictions and enhance robustness against occlusions and noise.

2. **Scalable Cross-View Point Cloud Fusion**: 

    - For each keypoint $k$, the fundamental matrix $F_{ij}$ is used to project the prediction of view $j$ onto the epipolar line of reference view $i$, finding the nearest point on the epipolar line as the projection result.
    - This process is repeated across all views to form a 2D point cloud $C_{i,k,t}$.
    - An encoder based on Point Cloud Transformer is used to process the point cloud, retaining residual connections for coordinate information.
    - Crucial advantage: regardless of the number of cameras, each reference view only needs to process a single point cloud, making the computational complexity scale linearly rather than quadratically with the number of views.
    - **Design Motivation**: Epipolar Transformers require feature fusion for every pair of views, which incurs a prohibitive computational cost when scaling to more cameras.

3. **Pose Compiler Module**: 

    - Point cloud encoder: a 4-layer multi-head self-attention network with a hidden dimension of 64 to extract cross-view features.
    - Spatio-temporal encoder: 4 layers of criss-cross Transformer blocks that perform attention along the temporal and joint dimensions respectively, approximating full spatio-temporal dependencies while saving memory.
    - The output is passed through an RLE head to yield the refined positions $\hat{\mu}'$ and uncertainties $\hat{\sigma}'$.
    - Training data: online synthesis of multi-view training data from the AMASS motion capture dataset (randomly placing cameras, projecting 3D keypoints, and adding noise augmentation).
    - **Design Motivation**: Criss-cross attention is more memory-efficient than full attention, and the synthetic training data strategy decouples the model from the constraints of real-world multi-view 3D datasets.

### Loss & Training
Maximum Likelihood Estimation (MLE) loss function:
$$\mathcal{L}_{mle} = -\log \prod_{i \in V} P_\Theta(u_i | \mathcal{I})|_{u_i = \hat{\mu}_i} - \log \prod_{i \in V} P_{\Theta'}(u_i | \mathcal{C})|_{u_i = \hat{\mu}'_i}$$

where $u_i$ is the projection of the 3D variable $U$ on each view. $U$ is initialized via DLT and iteratively optimized using L-BFGS. This approach allows training without 3D annotations by indirectly aligning 3D poses through maximizing projection likelihood.

Training strategy:
- The 2D pose estimator is fine-tuned on Human3.6m + MPII.
- The Pose Compiler is trained on AMASS synthetic data using the AdamW optimizer with a learning rate of 4e-5, warm-up, and cosine annealing, taking approximately 6 hours on an NVIDIA 2080 RTX.
- Online data synthesis: randomly placing up to 8 cameras, with motion data augmented by random rotation and mirroring.

## Key Experimental Results

### Main Results
Comparison of InD results on the Human3.6m dataset (mm):

| Method | Supervision Type | Backbone | Frames | MPJPE↓ | PA-MPJPE↓ |
|------|---------|----------|--------|--------|-----------|
| Learnable Triangulation | 3D | ResNet152 | 1 | 20.7 | 17.0 |
| AdaFuse | 3D | ResNet152 | 1 | 19.5 | - |
| MTF-Transformer | 3D | CPN | 27 | 25.8 | - |
| DLT | 2D | CPN | 1 | 30.5 | 27.6 |
| **UPose3D** | **2D** | **CPN** | **1** | **26.9** | **24.1** |
| **UPose3D** | **2D** | **CPN** | **27** | **26.4** | **23.4** |

Comparison of OoD results on the RICH dataset (mm):

| Method | Training Source | MPJPE↓ | PA-MPJPE↓ |
|------|--------|--------|-----------|
| AdaFuse | (H3.6m+MPII, H3.6m) | 524.0 | 85.8 |
| UPose3D (T=27) | (H3.6m+MPII, H3.6m) | 51.8 | 43.6 |
| HRNet-W48+DLT | (COCO, N/A) | 66.0 | 55.1 |
| **UPose3D (T=27)** | **(COCO, AMASS)** | **34.7** | **32.0** |

### Ablation Study
Ablation study on the Human3.6m dataset (T=27):

| Configuration | MPJPE↓ | PA-MPJPE↓ | Note |
|------|--------|-----------|------|
| UPose3D (Full) | 26.42 | 23.42 | - |
| w/o compiler | 37.14 | 33.90 | +10.7mm, compiler is crucial |
| w/o image branch | 69.90 | 50.97 | using compiler only, lacks raw predictions |
| w/o compiler uncertainty | 26.42 | 23.58 | removing compiler uncertainty during inference |
| w/o image uncertainty | 27.61 | 24.88 | +1.2mm, image uncertainty is important |
| w/o uncertainty (All) | 48.11 | 41.20 | +21.7mm, degenerates to simple triangulation |

Comparison of computational efficiency:

| Method | Params(M)↓ | FLOPs-4cam(G)↓ | FLOPs-10cam(G)↓ |
|------|-----------|----------------|-----------------|
| Learnable Triangulation | 80.6 | 716.1 | 1326.9 |
| Epipolar Transformers | 68.1 | 406.5 | 1016.2 |
| AdaFuse | 69.7 | 595.0 | 1487.6 |
| **UPose3D** | **65.4** | **208.7** | **517.7** |

### Key Findings
- **Uncertainty modeling is key**: Completely removing uncertainty causes MPJPE to surge by 21.7mm, indicating that uncertainty is crucial for filtering out low-quality predictions.
- **Stunning OoD performance**: On the RICH dataset, AdaFuse's MPJPE reaches 524mm, whereas UPose3D achieves a mere 34.7mm—representing a more than 10-fold reduction.
- AdaFuse fails in OoD scenarios because it requires predictions from all views to fall within reasonable ranges; a single noisy view can lead to massive triangulation errors.
- **Excellent scalability**: With 10 cameras, UPose3D's FLOPs are only 35-50% of other methods, and its inference time barely increases with the number of cameras.
- The choice of 2D pose estimator is critical: CPN significantly outperforms ResNet152, showing that backbone quality is fundamental.
- The temporal window is helpful but yields limited gain: 27 frames only reduces MPJPE by 0.5mm compared to 1 frame.

## Highlights & Insights
- **Dual utilization of uncertainty**: Simultaneously obtaining uncertainty from both the image branch and the Pose Compiler branch naturally weights the credibility of each prediction in MLE optimization, offering an elegant framework design.
- **Synthetic data training strategy**: Completely avoids reliance on real-world 3D annotated data by synthetically generating multi-view training data online from AMASS mocap data, enabling cross-dataset generalization.
- **Epipolar geometry as point clouds**: Formulating the cross-view problem as a point cloud processing task is a highly creative abstraction.
- **A counter-intuitive finding**: Simple DLT triangulation coupled with a strong 2D estimator (CPN) can outperform some complex methods (e.g., AdaFuse + noisy views), highlighting the importance of the baseline.
- The computational efficiency advantage becomes increasingly prominent as the number of cameras grows.

## Limitations & Future Work
- The MLE optimization stage uses L-BFGS, resulting in fluctuating computational costs and rendering it unsuitable for real-time applications.
- The diversity of synthetic training data remains constrained by the action types present in the AMASS dataset.
- Multi-person scenarios are not addressed; the method is limited to single-person multi-view settings.
- Although efficient, criss-cross attention relatively simplifies full spatio-temporal modeling.
- Future work could explore: (1) using deep networks to estimate the Hessian matrix for faster MLE; (2) incorporating additional modalities such as scene depth and trajectories.

## Related Work & Insights
- Keypoint uncertainty modeling via RLE (Residual Log-likelihood Estimation) provides valuable insights for subsequent 3D reconstruction tasks.
- The cross-view point cloud projection fusion strategy can be generalized to any multi-view geometry problem.
- The paradigm of training on synthetic data and testing on real-world data is validated as highly effective in multi-view pose estimation.
- The comparison with AdaFuse highlights an important lesson: methods performing exceptionally well in InD settings can utterly collapse in OoD scenarios, which underscores the vital importance of robustness evaluation.
- Enjoys direct application value for the markerless motion capture industry.

## Rating
- Novelty: ⭐⭐⭐⭐ (Cross-view point cloud fusion and dual-source uncertainty modeling show good novelty)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Evaluated in both InD/OoD settings with thorough ablations and comprehensive scalability analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, but a higher volume of notations poses a slightly high entry barrier)
- Value: ⭐⭐⭐⭐⭐ (Solves practical challenges: free of 3D annotations, robust to OoD, and highly scalable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [\[ICLR 2026\] From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper](../../ICLR2026/human_understanding/from_sparse_to_dense_spatio-temporal_fusion_for_multi-view_3d_human_pose_estimat.md)
- [\[ECCV 2024\] ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation](aden_adaptive_density_representations_for_sparseview_camera.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
