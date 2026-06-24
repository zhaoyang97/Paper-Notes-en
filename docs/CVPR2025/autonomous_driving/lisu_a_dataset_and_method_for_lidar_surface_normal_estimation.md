---
title: >-
  [Paper Note] LiSu: A Dataset and Method for LiDAR Surface Normal Estimation
description: >-
  [CVPR 2025][Autonomous Driving][Point Cloud Surface Normal Estimation] This paper proposes LiSu, the first large-scale synthetic LiDAR point cloud surface normal dataset, and designs a spatiotemporal regularization method to enhance normal estimation accuracy, effectively suppressing pseudo-label noise during self-training and achieving robust synthetic-to-real domain adaptation.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Point Cloud Surface Normal Estimation"
  - "LiDAR"
  - "Synthetic Dataset"
  - "Spatiotemporal Regularization"
  - "Domain Adaptation"
date: 2026-05-08
content_hash: 785232e7599df6e5
---

# LiSu: A Dataset and Method for LiDAR Surface Normal Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.08601](https://arxiv.org/abs/2503.08601)  
**Code**: None  
**Area**: Autonomous Driving / 3D Vision  
**Keywords**: Point Cloud Surface Normal Estimation, LiDAR, Synthetic Dataset, Spatiotemporal Regularization, Domain Adaptation

## TL;DR
This paper proposes LiSu, the first large-scale synthetic LiDAR point cloud surface normal dataset, and designs a spatiotemporal regularization method to enhance normal estimation accuracy, effectively suppressing pseudo-label noise during self-training and achieving robust synthetic-to-real domain adaptation.

## Background & Motivation

**Background**: Surface normals are fundamental geometric features in 3D scene analysis and are widely used in scene understanding, surface reconstruction, and autonomous driving perception. However, the task of estimating surface normals from LiDAR point clouds has been largely overlooked, lagging significantly behind normal estimation from RGB images.

**Limitations of Prior Work**: The issue is twofold. First, there is a lack of large-scale annotated LiDAR normal datasets, making supervised training difficult for existing methods. Second, the inherent sparsity and noise of LiDAR data make it challenging for traditional methods (e.g., PCA fitting to local planes) to robustly estimate normals in a reasonable timeframe, especially in long-range areas and object boundaries.

**Key Challenge**: Ground-truth surface normal annotations for real-world LiDAR data are virtually impossible to obtain at scale (requiring precise 3D model alignment), while traditional geometric methods exhibit limited performance on sparse and noisy point clouds, resulting in a dual dilemma of "data scarcity and inadequate methods."

**Goal**: (1) Construct the first large-scale synthetic LiDAR normal dataset; (2) design a normal estimation method that exploits the spatiotemporal characteristics of driving data; (3) achieve effective synthetic-to-real domain adaptation.

**Key Insight**: The authors notice that traffic simulation engines can obtain precise ground-truth surface normals for each point. Meanwhile, strong spatiotemporal correlation exists between consecutive frames in autonomous driving scenarios—the normals of adjacent points should be spatially consistent, and the normals of the same surface should be temporally smooth across consecutive frames.

**Core Idea**: Utilize traffic simulation to generate synthetic datasets with ground-truth surface normal annotations, and enhance pseudo-label quality in self-training via spatiotemporal regularization constraints to achieve robust synthetic-to-real domain adaptation.

## Method

### Overall Architecture
The overall method consists of two core components: dataset construction and the normal estimation method. On the data side, a large-scale synthetic LiDAR point cloud dataset, LiSu, with precise normal annotations is rendered via traffic simulation engines (e.g., CARLA/LGSVL). On the method side, given input LiDAR point clouds, a neural network predicts the surface normal for each point. During training, two regularization terms—spatial consistency and temporal smoothness—are introduced to improve accuracy. In the domain adaptation phase, a self-training strategy is adopted. A model pre-trained on synthetic data generates pseudo-labels for real data, and regularization constraints are applied to mitigate the effect of pseudo-label noise.

### Key Designs

1. **LiSu Synthetic Dataset**:

    - **Function**: Provides the first large-scale LiDAR point cloud dataset with surface normal annotations.
    - **Mechanism**: Use traffic simulation engines to generate diverse driving scenes, where the simulator can directly acquire the 3D surface normal ground truth corresponding to each LiDAR point. The dataset covers various weather conditions, lighting, and traffic scenarios, including rich road geometries (planes, curved surfaces, object boundaries, etc.).
    - **Design Motivation**: Annotating surface normals for real LiDAR data is extremely costly and lacks precision guarantees. Synthetic data can provide pixel-perfect annotations at zero cost, which is crucial for breaking through the data bottleneck.

2. **Spatial Consistency Regularizer**:

    - **Function**: Constrains the normal predictions of adjacent points to maintain spatial coherence.
    - **Mechanism**: For each query point, among its K-nearest neighbors, if two points lie on the same surface (determined by distance and normal angle thresholds), a consistency loss is applied to their predicted normals to encourage adjacent points on the same surface to have similar normal directions. The loss is formulated as the angular difference between the predicted normals of adjacent points.
    - **Design Motivation**: LiDAR data is sparse and noisy, making single-point predictions susceptible to local noise fluctuations. Spatial regularization smooths predictions by leveraging priors of local geometric structures.

3. **Temporal Smoothness Regularizer**:

    - **Function**: Constrains the normal predictions of the same surface points to maintain temporal consistency across consecutive frames.
    - **Mechanism**: Exploiting the temporal nature of autonomous driving data, points from the current frame are projected onto adjacent frames using the vehicle's ego-motion to find corresponding nearest neighbors. A consistency constraint is then applied to their normal predictions, requiring the corresponding normals to remain consistent after rigid body transformation.
    - **Design Motivation**: The observed normal of the same surface in a static scene should remain the same across different timestamps (after coordinate transformation). This temporal prior effectively constrains predictions and reduces inter-frame jitter.

### Loss & Training
The total loss consists of three parts: supervised loss (normal angular error), spatial regularization loss, and temporal regularization loss. During self-training, a teacher model is first trained on synthetic data to generate pseudo-labels for real data. Then, a student model is trained jointly using these pseudo-labels and regularization terms. The weights of the regularization terms are adjusted via the validation set and gradually increased during self-training to suppress the propagation of pseudo-label noise.

## Key Experimental Results

### Main Results

| Method | LiSu-Val MAE↓ | LiSu-Val Acc@11.25°↑ | Real Data MAE↓ |
|------|--------------|---------------------|-------------|
| PCA (Traditional) | 18.7° | 52.3% | 22.4° |
| DeepFit | 15.2° | 61.8% | 19.6° |
| AdaFit | 14.1° | 64.5% | 18.3° |
| Ours (Synthetic) | 11.3° | 72.1% | 16.8° |
| Ours (+Self-Training+Reg.) | **10.2°** | **76.4%** | **13.5°** |

### Ablation Study

| Configuration | MAE↓ | Acc@11.25°↑ | Note |
|------|------|------------|------|
| Full model | 10.2° | 76.4% | Full model |
| w/o Spatial Regularization | 11.8° | 71.2% | Removing spatial constraint drops performance by 5.2% |
| w/o Temporal Regularization | 11.5° | 72.6% | Removing temporal constraint drops performance by 3.8% |
| w/o both Regularizations | 12.6° | 68.3% | Significant degradation with self-training alone |
| Trained only on Synthetic | 11.3° | 72.1% | No domain adaptation |

### Key Findings
- The two regularization terms complement each other significantly during self-training; joint use yields much better results than using either individually, indicating that spatial and temporal priors constrain prediction quality from different perspectives.
- The benefit of spatiotemporal regularization is far more pronounced in the self-training phase than in the supervised training phase, suggesting that their main value lies in suppressing pseudo-label noise.
- In long-range areas (>30m), where the point cloud is sparser, the contribution of spatial regularization is particularly prominent.
- Downstream neural surface reconstruction tasks demonstrate that the improvement in normal quality directly enhances 3D reconstruction accuracy.

## Highlights & Insights
- **Synthetic data + self-training paradigm** is highly practical: In tasks where annotation costs are extremely high, using simulators to obtain GT annotations and then transferring to real data via domain adaptation is a reusable general strategy. The clever part is that the regularization terms directly counteract pseudo-label noise, which is the biggest bottleneck in self-training.
- **The design of spatiotemporal regularization** fully exploits the inherent characteristics of autonomous driving data (consecutive frames, ego-motion information), encoding domain knowledge into training constraints, which is more efficient and targeted than general data augmentation strategies.

## Limitations & Future Work
- The LiSu dataset is entirely based on simulation. Although it covers diverse scenarios, a distribution gap with the real world still exists, especially since the sensor noise model may not be perfectly accurate.
- Temporal regularization relies on accurate ego-motion estimation, and its performance may degrade when SLAM localization errors are large.
- The current method focuses on normal estimation for static scenes, and does not specifically handle normal changes for dynamic objects (pedestrians, vehicles).
- Future work could explore joint optimization of normal estimation and semantic segmentation, leveraging semantic priors to further improve normal estimation accuracy in boundary regions.

## Related Work & Insights
- **vs DeepFit/AdaFit**: These methods directly fit local surfaces on 3D point clouds to estimate normals, whereas this paper learns normals in a data-driven manner. Combined with spatiotemporal regularization, it is more robust on sparse LiDAR data.
- **vs DSINE (RGB normal)**: DSINE achieves SOTA normal estimation on RGB images. This paper transfers a similar concept to the LiDAR modality, but faces entirely different sparsity challenges.
- The synthetic-to-real domain adaptation + self-training pipeline can be transferred to other LiDAR perception tasks (e.g., semantic segmentation, panoptic segmentation), and the design of spatiotemporal regularization is particularly worth referencing.

## Rating
- Novelty: ⭐⭐⭐⭐ The first large-scale LiDAR normal dataset + spatiotemporal regularization is a meaningful contribution, though the overall technical approach is relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete synthetic and real experiments, validated with ablation studies and downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured and the problem definition is clear.
- Value: ⭐⭐⭐⭐ The LiSu dataset is of long-term value to the community, and the method is directly applicable to 3D perception in autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate](climbingcap_multi-modal_dataset_and_method_for_rock_climbing_in_world_.md)
- [\[CVPR 2025\] Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method](towards_satellite_image_road_graph_extraction_a_global-scale_dataset_and_a_novel.md)
- [\[CVPR 2025\] A Dataset for Semantic Segmentation in the Presence of Unknowns](a_dataset_for_semantic_segmentation_in_the_presence_of_unknowns.md)
- [\[CVPR 2025\] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting](toward_real-world_bev_perception_depth_uncertainty_estimation_via_gaussian_splat.md)
- [\[CVPR 2025\] TacoDepth: Towards Efficient Radar-Camera Depth Estimation with One-Stage Fusion](tacodepth_towards_efficient_radar-camera_depth_estimation_with_one-stage_fusion.md)

</div>

<!-- RELATED:END -->
