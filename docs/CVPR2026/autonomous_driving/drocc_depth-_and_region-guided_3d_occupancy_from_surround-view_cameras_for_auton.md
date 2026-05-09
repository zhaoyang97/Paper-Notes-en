---
title: >-
  [Paper Note] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving
description: >-
  [CVPR 2026][Autonomous Driving][3D occupancy prediction] Dr.Occ proposes a unified 3D occupancy prediction framework with depth guidance and region guidance. It employs D2-VFormer to leverage high-quality depth priors from MoGe-2 for accurate 2D→3D geometric mapping, and R/R2-EFormer to adaptively assign region-specific experts inspired by MoE/MoR for handling spatial semantic anisotropy, achieving a +7.43% mIoU improvement over the BEVDet4D baseline.
tags:
  - CVPR 2026
  - Autonomous Driving
  - 3D occupancy prediction
  - depth guidance
  - region experts
  - MoE
  - visual perception
date: 2026-05-08
content_hash: cdb71d0a58a1ba6c
---

# Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2603.01007](https://arxiv.org/abs/2603.01007)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: 3D occupancy prediction, depth guidance, region experts, MoE, visual perception

## TL;DR

Dr.Occ proposes a unified 3D occupancy prediction framework with depth guidance and region guidance. It employs D2-VFormer to leverage high-quality depth priors from MoGe-2 for accurate 2D→3D geometric mapping, and R/R2-EFormer to adaptively assign region-specific experts inspired by MoE/MoR for handling spatial semantic anisotropy, achieving a +7.43% mIoU improvement over the BEVDet4D baseline.

## Background & Motivation

3D semantic occupancy prediction is a core task in autonomous driving perception, requiring the reconstruction of accurate voxel-level semantic scenes from surround-view camera images. Existing methods face two major challenges:

**Geometric Misalignment**: Existing view transformation methods (forward/backward/bidirectional projection) rely on low-resolution, noisy depth estimation for 2D→3D feature mapping, leading to projection errors and feature misalignment.

**Spatial Class Imbalance**: Different semantic categories exhibit strong positional preferences in 3D space — pedestrians concentrate near roadsides, vehicles cluster at road centers, and buildings and vegetation appear at higher elevations. This spatial anisotropy makes it difficult for a unified model to learn balanced representations.

**Key Observations**:
- With the development of large vision models (e.g., MoGe-2), high-quality pixel-level depth estimation is now available; however, naively concatenating depth maps or converting them to pseudo point clouds yields poor results — directly using them for forward projection actually degrades performance.
- Approximately 90% of occupancy voxels are empty, making direct fitting of all voxels inefficient. It is thus more effective to use depth to generate geometry-aware occupancy masks that focus computation on non-empty voxels.

## Method

### Overall Architecture

Dr.Occ introduces two key improvements over the standard occupancy prediction pipeline:

1. **D2-VFormer (Depth-guided 2D-to-3D View Transformer)**: Uses MoGe-2 depth priors to construct geometry-aware occupancy masks, guiding efficient and accurate voxel feature construction.
2. **R/R2-EFormer (Region-guided Expert Transformer)**: Inspired by MoE/MoR, divides the 3D space into regions along distance and height axes and adaptively assigns expert modules to handle the semantic distribution of different regions.

### Key Designs

#### 1. Depth-guided 2D-to-3D View Transformer (D2-VFormer)

**Depth Prior Acquisition**: MoGe-2 is used to simultaneously extract depth features $\mathbf{F}^{(D)}$ and depth maps $\{\mathbf{D}_i\}$. The depth maps are unprojected via camera projection into a pseudo point cloud $\mathcal{P}$, which is then voxelized to generate a geometry-aware occupancy mask $M(\mathbf{v})$:

$$M(\mathbf{v}) = \begin{cases} 1, & \mathbf{v} \in \text{Voxelize}(\mathcal{P}, r) \\ 0, & \text{otherwise} \end{cases}$$

**Three-Stage Progressive Refinement**:

**Stage 1: Forward Projection + Downsampling**. Following BEVStereo, 2D features are lifted into voxel space (covering ~30% of voxels). The voxel features and geometric mask are then downsampled by a factor of $\lambda$, yielding: (1) improved computational efficiency and (2) a coarser voxel resolution that naturally tolerates pixel-level depth errors.

**Stage 2: Backward Projection Densification**. Deformable cross-attention (DCA) is used to fuse multi-view image features and recover geometric completeness:

$$\mathbf{F}_{dense} = \text{DCA}(\mathbf{F}_{down}, \mathbf{F}^{(I)})$$

**Stage 3: Depth-guided Non-empty Voxel Refinement**. Selective two-step refinement is performed under the guidance of the geometric mask:

- **Geometric Refinement**: Depth features $\mathbf{F}^{(D)}$ are fused only for occupied voxels; non-occupied voxels are assigned a learnable empty embedding $\mathbf{e}_{empty}$.
- **Semantic Enhancement**: Multi-view image features are fused for occupied voxels:

$$\mathbf{F}_{out} = \text{DCA}(\mathbf{F}_{geo}, \mathbf{F}^{(I)}; \mathcal{M}_{down})$$

Core Idea: Computational resources are concentrated on the ~10% of semantically meaningful voxels, avoiding waste on the 90% that are empty.

#### 2. Region-guided Expert Transformer (R-EFormer)

**Spatial Anisotropy Analysis**: Statistical analysis reveals significant distributional differences across semantic categories along the height and distance dimensions:
- Road surfaces concentrate at low height and short range.
- Vegetation and buildings occupy higher positions at medium distances.
- Dynamic objects appear only within narrow spatial bands.

**Region Partition**: The 3D space is divided into $3\times3=9$ regions $\mathcal{R}_m$ along distance (near 0–10m / mid 10–30m / far ≥30m) and height (low −1~0.2m / mid 0.2~2.2m / high 2.2~5.4m), with a dedicated expert $E_m$ assigned to each region.

**Routing and Expert Selection**:

$$s_m = \text{Router}(\mathbf{F}_{out}), \quad \mathcal{S} = \text{TopK}(\{s_m\}_{m=1}^M, K)$$

Each expert applies the same DCA module but restricted to the binary mask $\mathcal{M}_m$ of its corresponding region:

$$\mathbf{F}_{final} = \sum_{m \in \mathcal{S}} w_m \cdot E_m(\mathbf{F}_{out}, \mathbf{F}^{(I)}; \mathcal{M}_m)$$

#### 3. Region-guided Recursive Expert Transformer (R2-EFormer)

R-EFormer requires manually defined regions and is sensitive to hyperparameters. R2-EFormer draws inspiration from Mixture-of-Recursions (MoR), using a single expert that iterates recursively $n$ times. At each iteration, a router progressively focuses on smaller salient regions:

$$\mathcal{M}^{(t)} = \begin{cases} \Omega, & t=1 \\ \text{TopK}(\mathcal{R}^{(t)}(\mathbf{F}^{(t-1)}, \mathcal{M}^{(t-1)}), k_t), & t>1 \end{cases}$$

The coverage ratio decreases progressively (100% → 75% → 50%), ensuring $\mathcal{M}^{(t)} \subset \mathcal{M}^{(t-1)}$.

R2-EFormer offers three advantages:
1. A single recursive expert reduces parameter count.
2. Adaptive region discovery reduces sensitivity to manually defined hyperparameters.
3. Progressive focus on high-confidence regions enhances semantic prediction.

### Loss & Training

- Standard occupancy prediction losses (focal loss for classification + scene-class CE)
- Image encoder: ResNet-50; depth estimator: moge-2-vits-normal
- Voxel resolution: 0.4m; spatial range: 80m×80m×6.4m; grid size: 200×200×16
- Forward projection feature downsampling: 1/16
- R-EFormer multi-head attention: 8 heads, $N_{ref}$=4 reference points
- AdamW: lr=$1 \times 10^{-4}$, weight decay=$1 \times 10^{-2}$, batch size 16 (8×L20 GPUs), 24 epochs

## Key Experimental Results

### Main Results

**Table 1: Occ3D-nuScenes mIoU (%) Comparison**

| Method | Backbone | mIoU (%) |
|------|---------|:---:|
| BEVFormer | R101 | 26.9 |
| SparseOcc | R50 | 30.9 |
| BEVDet4D* | R50 | 36.0 |
| FlashOcc* | R50 | 37.8 |
| FB-Occ* | R50 | 39.1 |
| ViewFormer* | R50 | 41.9 |
| COTR* | R50 | 43.1 |
| **BEVDet4D+Dr.Occ*** | **R50** | **43.4** |
| **COTR+Dr.Occ*** | **R50** | **44.1** |

Dr.Occ achieves **+7.43% mIoU** and **+3.09% IoU** over the BEVDet4D baseline, and further improves the state-of-the-art method COTR by **+1.0% mIoU**.

**Significant gains on foreground categories**: bicycle +20.4%, motorcycle +6.9%, pedestrian +13.4%, traffic cone +9.5%, validating the benefit of region experts for rare classes.

### Ablation Study

**Table 2: Component Ablation**

| D2-VFormer | R-EFormer | R2-EFormer | IoU (%) | mIoU (%) |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 70.36 | 36.01 |
| ✓ | ✗ | ✗ | 71.29 | 41.45 |
| ✓ | ✓ | ✗ | **73.45** | 43.03 |
| ✓ | ✗ | ✓ | 72.87 | **43.43** |

- D2-VFormer alone contributes +5.44% mIoU — geometric alignment is the largest source of gain.
- R-EFormer adds a further +1.58% mIoU on top of D2-VFormer.
- R2-EFormer yields slightly lower IoU but achieves the highest mIoU (43.43%), as recursive refinement is better suited to handling rare categories.

### Key Findings

1. The effective way to utilize high-quality depth priors (MoGe-2) is not direct projection but rather generating occupancy masks to guide the model's focus — a counterintuitive yet critical finding.
2. The observation that ~90% of voxels are empty directly motivates the computational efficiency optimization strategy.
3. Spatial anisotropy of semantic categories is an overlooked yet important problem, and region experts provide an effective solution.
4. R2-EFormer's adaptive recursion outperforms manual region definitions, particularly in challenging scenarios such as nighttime driving.

## Highlights & Insights

1. **Novel Use of Depth Priors**: Rather than brute-force projection of depth maps (which degrades performance), the method cleverly converts depth into occupancy masks to guide attention — reflecting a deep understanding of the problem's essence.
2. **First Application of MoE/MoR to 3D Perception**: The natural integration of region awareness with mixture of experts provides a new perspective on addressing long-tail problems in occupancy prediction.
3. **Plug-and-Play Design**: D2-VFormer and R/R2-EFormer can be independently integrated into different baselines; COTR+Dr.Occ validates generalizability.
4. **Statistical Analysis of Spatial Anisotropy**: The visualizations in Figure 4 intuitively demonstrate distributional differences of semantic categories along height and distance dimensions, providing data-driven support for region partitioning.

## Limitations & Future Work

1. The MoGe-2 depth estimator is not fine-tuned, potentially introducing domain gaps; joint fine-tuning or substitution with a driving-domain-specific depth estimator may further improve performance.
2. The region partition in R-EFormer depends on manually defined height/distance thresholds with limited generalizability; R2-EFormer addresses this but introduces additional recursive computation.
3. Evaluation is limited to nuScenes/Occ3D; validation on larger-scale datasets such as Waymo remains absent.
4. The three-stage progressive refinement in D2-VFormer increases model complexity, and real-time performance is insufficiently assessed.
5. Deeper exploitation of temporal information (e.g., video-level depth estimation consistency) is not explored.

## Related Work & Insights

- **BEVDet4D/BEVStereo**: Forward projection baselines on which Dr.Occ's geometric enhancement yields the most significant gains.
- **COTR**: A bidirectional projection state-of-the-art method whose performance is further improved by Dr.Occ modules.
- **MoGe/MoGe-2**: The depth estimation generalization capability of large vision models opens new opportunities for 3D perception.
- **MoE/MoR**: Efficient model scaling paradigms from NLP with broad application prospects in 3D vision.
- The depth-guided occupancy mask idea is generalizable to other tasks requiring 2D→3D mapping (e.g., 3D object detection, scene reconstruction).

## Rating

| Dimension | Score (1–5) |
|------|:---:|
| Novelty | 4 |
| Technical Depth | 5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 4 |
| Overall | 4.2 |

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)
- [\[CVPR 2026\] OccAny: Generalized Unconstrained Urban 3D Occupancy](occany_generalized_unconstrained_urban_3d_occupancy.md)
- [\[CVPR 2026\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2occ_resilient_3d_semantic_occupancy_prediction_f.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](test-time_3d_occupancy_prediction.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)

<!-- RELATED:END -->
