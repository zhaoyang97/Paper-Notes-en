---
title: >-
  [Paper Note] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud
description: >-
  [ICCV 2025][3D Vision][CAD point cloud] This paper proposes CstNet, the first constraint-aware feature learning method for parametric point clouds. CAD constraints are encoded as point-level MAD-Adj-PT triplet representations, and a two-stage network (constraint acquisition + constraint feature learning) achieves state-of-the-art results on the newly constructed Param20K dataset, with classification accuracy improved by +3.49% and rotation robustness improved by +26.17%.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "CAD point cloud"
  - "constraint representation"
  - "parametric shape"
  - "point cloud classification"
  - "rotation robustness"
date: 2026-05-08
content_hash: 2b44cb5967566257
---

# CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud

**Conference**: ICCV 2025
**arXiv**: [2411.07747](https://arxiv.org/abs/2411.07747)  
**Code**: [https://cstnetwork.github.io/](https://cstnetwork.github.io/)  
**Area**: 3D Vision / Point Cloud Analysis
**Keywords**: CAD point cloud, constraint representation, parametric shape, point cloud classification, rotation robustness

## TL;DR

This paper proposes CstNet, the first constraint-aware feature learning method for parametric point clouds. CAD constraints are encoded as point-level MAD-Adj-PT triplet representations, and a two-stage network (constraint acquisition + constraint feature learning) achieves state-of-the-art results on the newly constructed Param20K dataset, with classification accuracy improved by +3.49% and rotation robustness improved by +26.17%.

## Background & Motivation

Parametric point clouds are sampled from CAD models and are widely used in industrial manufacturing. Existing deep learning methods tailored for CAD data suffer from three core issues:

**Focus on geometry while ignoring constraints**: Most methods (e.g., ParSeNet, HPNet) focus on the regularity of primitives (planes, cylinders, etc.) but overlook the **constraint relationships** between primitives (coaxiality, parallelism, distance, etc.). This prevents them from distinguishing CAD parts that appear similar but serve different functions — for example, eccentric wheels and flywheels look alike but have entirely different constraints (eccentric vs. coaxial).

**Reliance on specialized labels**: Many methods require additional annotations such as primitive types, parameters, and surface normals, whereas real-world point clouds typically only provide coordinate information.

**Challenges in constraint learning**: Constraints describe relationships between pairs of primitives; how to encode these into representations suitable for deep learning remains an open problem.

## Method

### Overall Architecture

CstNet consists of two stages:
- **Stage 1 (Constraint Acquisition)**: Extracts or predicts MAD-Adj-PT constraint representations from BRep data (CstBRep) or raw point clouds (CstPnt).
- **Stage 2 (Constraint Feature Learning)**: Fuses geometric features and constraint features via attention mechanisms to perform downstream classification.

### Key Designs

1. **MAD-Adj-PT Constraint Representation (Deep Learning-Friendly Constraint Encoding)**:

    - Traditional CAD constraints are converted into three point-level components:
        - **MAD (Main Axis Direction)**: The principal axis direction of the primitive to which a point belongs (plane → normal vector; cylinder/cone → rotation axis).
        - **Adj (Adjacency)**: Whether a point lies near an edge (one-hot encoded), indicating connectivity between primitives.
        - **PT (Primitive Type)**: The type of primitive a point belongs to (plane/cylinder/cone, one-hot encoded).
    - The complete representation per point is $(x, y, z, \text{MAD}, \text{Adj}, \text{PT})$.
    - **Design Motivation**: Transforming pairwise primitive relationships into relationships between each primitive and a shared reference significantly reduces data volume; the MAD+Adj combination can encode most common constraints such as parallelism, perpendicularity, and distance.

2. **CstPnt Module (Predicting Constraints from Raw Point Clouds)**:

    - Simulates the four-step manual constraint extraction pipeline: search for neighboring points → identify points on the same primitive → fit the shape → compute MAD/Adj/PT.
    - Proposes **SurfaceKNN**: neighborhood search along the shape surface (rather than conventional spherical KNN), better suited to the characteristics of CAD shapes.
    - Applies Q-K-V attention to assign different weights to neighboring points: $\mathbf{f}'_i = \sum_{f_j \in \mathcal{N}_i} \rho(\text{MLP}(\text{Q}(\mathbf{f}_i) - \text{K}(\mathbf{f}_j))) \odot \text{V}(\mathbf{f}_i)$
    - Uses only **local features**, enabling generalization to unseen datasets after pretraining on the ABC dataset.
    - **Design Motivation**: Removes dependency on BRep data; the limited variety of primitive types in CAD shapes (plane + cylinder + cone covers ~94%) allows local patterns to transfer across datasets.

3. **Stage 2 Constraint Feature Learning Network**:

    - **Triple MLP**: Concatenates xyz with MAD, Adj, and PT separately to produce Axis Feature, Adjacency Feature, and Primitive Feature.
    - **C-MLP**: Concatenates xyz with the full MAD-Adj-PT to produce an initial Constraint Feature.
    - **Quartic SSA**: Four parallel branches process the four feature types independently, each applying FPS + SurfaceKNN + point-level attention.
    - **Fea Attention**: Feature-level attention that adaptively reweights different features — points near edges attend more to Adjacency Feature, while other points may weight Primitive Feature more heavily.
    - **Design Motivation**: Different constraint components carry different importance for different regions of a point cloud, necessitating adaptive weighting.

### Loss & Training

- Negative Log Likelihood Loss is used.
- Adam optimizer, initial lr = 0.0001, StepLR (step = 20, gamma = 0.7), weight decay = 0.0001.
- Training for 200 epochs with batch size = 16.
- Stage 1 is pretrained on 25 trunks (~250K BRep files) of the ABC dataset; weights are frozen before being applied in Stage 2.

## Key Experimental Results

### Main Results

**Param20K Classification Results**:

| Method | Acc(%) | Acc*(%) | F1 | mAP(%) |
|---|---|---|---|---|
| PointNet | 81.30 | 83.21 | 82.06 | 85.18 |
| PointNet++ | 83.70 | 86.37 | 85.30 | 87.94 |
| DGCNN | 85.40 | 87.28 | 86.43 | 89.17 |
| PTMamba | 86.45 | 87.28 | 86.63 | 91.34 |
| **CstNet** | **89.94** | **91.06** | **90.34** | **92.72** |

Rotation robustness: CstNet exhibits the smallest accuracy drop when the training set is unchanged and the test set is rotated along +Z (outperforming the previous SOTA by 26.17%).

**Constraint Prediction Accuracy (ABC → Param20K Generalization)**:

| Method | MAD(MSE)↓ | Adj(%)↑ | PT(%)↑ |
|---|---|---|---|
| ParSeNet | 0.2247 | 81.42 | 59.75 |
| HPNet | 0.2570 | 78.60 | 57.66 |
| **CstNet** | **0.1390** | **87.95** | **86.52** |

### Ablation Study

**Contribution of Each Constraint Component (SurfaceKNN)**:

| MAD | Adj | PT | Acc(%) | mAP(%) |
|---|---|---|---|---|
| ✓ | ✓ | ✓ | **89.94** | **92.72** |
| ✗ | ✓ | ✓ | 86.32 | 90.59 |
| ✓ | ✗ | ✓ | 88.14 | 91.35 |
| ✓ | ✓ | ✗ | 88.68 | 91.22 |
| ✗ | ✗ | ✗ | 83.99 | 86.15 |

**Stage 2 Backbone Comparison (Using Predicted Constraints)**:

| Backbone | Acc(%) | mAP(%) |
|---|---|---|
| PointNet | 83.83 | 87.27 |
| PointNet++ | 85.52 | 90.79 |
| DGCNN | 86.81 | 91.35 |
| **CstNet Stage 2** | **89.94** | **92.72** |

### Key Findings

- Among the three constraint components, MAD contributes the most (removing it reduces accuracy by 3.62%).
- SurfaceKNN outperforms standard KNN in most cases, primarily by enhancing the effectiveness of Primitive Feature.
- CstPnt exhibits strong rotation robustness (dashed curves are unaffected by rotation) because the ABC training set contains local features from diverse orientations.
- Using predicted constraints, while inferior to ground-truth constraint labels, still significantly outperforms the constraint-free baseline (89.94% vs. 83.99%).
- Validation experiments on prisms vs. rectangular boxes intuitively demonstrate that geometric methods fail to distinguish the two when angles approach 90°, whereas the constraint-aware method maintains high accuracy throughout the 50°–87° range.

## Highlights & Insights

- **First constraint-aware CAD point cloud analysis framework**, filling a gap in the field.
- The MAD-Adj-PT representation is elegantly designed: complex pairwise primitive relationships are simplified into point-level attributes, substantially reducing computational complexity.
- Emphasis on practicality: CstPnt requires only raw point cloud input, generalizes across datasets, and does not depend on BRep data.
- The newly constructed Param20K dataset fills the gap in benchmarks for parametric CAD classification.
- The 26.17% improvement in rotation robustness is a particularly notable result.

## Limitations & Future Work

- Only three primitive types are considered (plane/cylinder/cone); constraint representations for free-form surfaces require further investigation.
- The Param20K dataset is limited in scale (75 categories, ~20K instances), and industrial scenarios may be considerably more complex.
- Computational overhead analysis of SurfaceKNN is absent.
- Other downstream CAD tasks such as segmentation and assembly have not been explored.
- The robustness of CstPnt on very small patches or noisy point clouds has not been validated.

## Related Work & Insights

- ParSeNet and HPNet are representative methods for CAD point cloud analysis but focus solely on primitive fitting.
- BRep-based methods such as BRepNet operate on topological data but require specialized data formats.
- The core insight of this paper: **function determines constraints, and constraints determine shape** — parts that appear similar but serve different functions should be distinguished from a constraint perspective.
- The MAD-Adj-PT framework is extensible to tasks such as CAD retrieval and assembly prediction.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to introduce constraints into point cloud deep learning; representation design is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-task evaluation and comprehensive ablations; dataset scale and number of baselines could be expanded.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear; validation experiments proceed in a well-structured manner.
- **Value**: ⭐⭐⭐⭐ — Strong prospects for industrial applications; opens a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Parametric Point Cloud Completion for Polygonal Surface Reconstruction](../../CVPR2025/3d_vision/parametric_point_cloud_completion_for_polygonal_surface_reconstruction.md)
- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[ICCV 2025\] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)

</div>

<!-- RELATED:END -->
