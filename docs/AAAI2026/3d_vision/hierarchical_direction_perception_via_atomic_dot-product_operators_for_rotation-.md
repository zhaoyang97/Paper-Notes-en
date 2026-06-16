---
title: >-
  [Paper Note] Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning
description: >-
  [AAAI2026][3D Vision][Point Cloud] This paper proposes DiPVNet, which leverages the dual properties of the atomic dot-product operator (directional selectivity + rotation invariance) to construct a local L2DP operator an…
tags:
  - "AAAI2026"
  - "3D Vision"
  - "Point Cloud"
  - "Rotation Invariance"
  - "Dot-Product Operator"
  - "Spherical Fourier Transform"
  - "Equivariance"
date: 2026-05-08
content_hash: e07be9a5ed3904e5
---

# Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning

**Conference**: AAAI2026
**arXiv**: [2511.08240](https://arxiv.org/abs/2511.08240)  
**Code**: [DiPVNet](https://github.com/wxszreal0/DiPVNet)  
**Area**: 3D Vision
**Keywords**: Point Cloud, Rotation Invariance, Dot-Product Operator, Spherical Fourier Transform, Equivariance

## TL;DR
This paper proposes DiPVNet, which leverages the dual properties of the atomic dot-product operator (directional selectivity + rotation invariance) to construct a local L2DP operator and a global DASFT module, achieving hierarchical direction-aware rotation-invariant point cloud learning.

## Background & Motivation
- 3D point cloud processing is widely applied in autonomous driving and embodied AI, but arbitrary rotations disrupt spatial distributions and cause feature inconsistency.
- The core challenge is that rotational perturbations destroy inherent multi-scale directional features in point clouds (local: edge orientations, normals; global: principal axes, structural symmetry).
- Explicit methods (e.g., ODF, spatial direction partitioning) rely on fixed partitions and cannot adapt to non-uniform distributions.
- Implicit methods (e.g., VNN) preserve equivariance/invariance but underutilize directional information; VNN uses only a single global direction vector for gating, failing to capture complex hierarchical directional structures.

## Core Problem
How to adaptively perceive multi-scale directional features of point clouds while preserving rotational symmetry, so as to improve rotationally robust discriminative representations?

## Method

### Overall Architecture
DiPVNet consists of three core components: (1) the L2DP operator for local directional feature extraction → (2) the DASFT module for constructing a global directional response spectrum → (3) Cross-Attention for fusing local and global features. An equivariant branch based on VNN Blocks is also retained.

### Key Designs

**1. Atomic Dot-Product Operator**

- Reveals the dual properties of the dot-product: directional selectivity (as a directional filter) + rotation invariance.
- Encapsulated as a differentiable atomic operator: $\Phi(\mathbf{a}, \{\mathbf{b}_i\}; \Theta) = \text{FFN}(\{\langle \mathbf{a} \cdot \mathbf{b}_i \rangle\}_{i=1}^K; \Theta)$

**2. Learnable Local Dot-Product (L2DP) Operator**

- Computes dot-products between center point $\mathbf{v}_j$ and its K nearest neighbors $\mathbf{g}_j^{(k)}$ with relative position encoding:
  $I_j^{(\mathcal{G}, \text{rel})} = \{\langle \mathbf{v}_j, \mathbf{g}_{jk} - \mathbf{v}_j \rangle \mid k=1,\dots,K\}$
- Here $\langle \mathbf{v}_j, \mathbf{g}_{jk} \rangle$ encodes directional information, while $\langle \mathbf{v}_j, \mathbf{v}_j \rangle$ injects positional encoding.
- Two aggregation strategies: DLP (Direct Linear Projection, preserving full neighbor interactions) and SAP (Statistics-Aware Projection, computing max/var/avg before projection, suitable for large neighborhoods).

**3. Direction-Aware Spherical Fourier Transform (DASFT)**

- Treats the point cloud as a discrete signal in 3D space and computes dot-products with spherical sampling vectors $\Omega = r \cdot \omega$.
- Proves this operation is equivalent to a direction-aware spherical Fourier transform: $\mathcal{F}(\mathcal{P}, \{\Omega\}) = \sum_{j=1}^n \exp(-ir\omega^\top v_j)$
- Constructs an energy spectrum $E(\mathcal{P}, \{\Omega\}) = |\mathcal{F}|^2$ and obtains rotation-invariant descriptors via spherical averaging.
- Uniformly samples $N_{\text{dir}} = 36$ directions; results are stable for $\geq 36$ directions.

**4. Feature Fusion**

- Cross-Attention: L2DP features serve as Query; DASFT features serve as Key/Value.
- VNN Block equivariant features are projected onto a learned canonical basis to generate scalar tokens, which are concatenated with the fused invariant features.

## Key Experimental Results

| Method | ModelNet40 (z/SO(3)) | ScanObjectNN (z/SO(3)) | ShapeNetPart (z/SO(3)) |
|--------|---------------------|----------------------|----------------------|
| VN-DGCNN | 89.5 | 83.5 | 81.4 |
| LGR-Net | 90.9 | 81.2 | 80.0 |
| TetraSphere | 90.5 | 87.3 | 82.3 |
| PaRot | 91.0 | - | - |
| **DiPVNet** | **91.4** | **87.5** | **82.5** |

- Ablation: DASFT only (Model A) = 89.5 (no gain); L2DP-DLP only (Model C) = 90.6; full model = 91.4.
- Gate fusion (90.9) < Cross-Attention fusion (91.4), indicating that dynamic feature calibration outperforms static weight assignment.

## Highlights & Insights
- Starting from the fundamental mathematical properties of the dot-product, the paper unifies directional perception and rotation invariance under a novel theoretical perspective.
- L2DP and DASFT capture local and global directional features respectively with strong complementarity (DASFT alone yields no gain, underscoring the importance of local features).
- The theoretical connection between DASFT and generalized harmonic analysis is rigorous, going beyond mere engineering design.
- Consistent performance is maintained under noise and large-angle rotations (z/z = z/SO(3) = SO(3)/SO(3)).

## Limitations & Future Work
- Built on the VN-DGCNN baseline, the model capacity is limited and has not been validated on large-scale pretrained point cloud models.
- Dataset scales for classification and segmentation are relatively small (ModelNet40 contains only 12k samples).
- Experiments on large outdoor scenes (e.g., S3DIS, full ScanNet scene segmentation) are absent.
- DASFT involves spherical sampling and Fourier transforms, but computational overhead is not thoroughly analyzed.
- Only coordinate information is processed; multi-modal point cloud features such as normals and colors are not considered.

## Related Work & Insights
- vs **VNN**: VNN uses a single global direction vector for gating, whereas DiPVNet achieves adaptive multi-scale directional perception via the dot-product operator.
- vs **SGMNet**: SGMNet also employs dot-products but its sorting mechanism disrupts spatial relationships and lacks a directional aggregation strategy.
- vs **TFN**: TFN constructs group-equivariant convolutional kernels via spherical harmonics with high computational cost; DiPVNet is more lightweight.
- vs **PaRot/TetraSphere**: DiPVNet surpasses both on ModelNet40 and ScanObjectNN.

## Inspiration & Connections
- The "atomic operator" paradigm can be extended to other geometric operations (e.g., cross-product for normal perception).
- The spherical frequency-domain analysis of DASFT may be applicable to point cloud generation or shape retrieval.
- The adaptive neighborhood direction learning in L2DP can be transferred to local feature extraction in 3D object detection.

## Rating
- Novelty: ⭐⭐⭐⭐ (Unique theoretical perspective grounded in the fundamental properties of the dot-product)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ablation, but lacks large-scene experiments)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous mathematical derivation, clear structure)
- Value: ⭐⭐⭐⭐ (New SOTA for rotation-invariant point cloud learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Enhancing Rotation-Invariant 3D Learning with Global Pose Awareness and Attention Mechanisms](enhancing_rotation-invariant_3d_learning_with_global_pose_awareness_and_attentio.md)
- [\[CVPR 2026\] 4D Local Modeling Toward Dynamic Global Perception for Ambiguity-free Rotation-Invariant Point Cloud Analysis](../../CVPR2026/3d_vision/4d_local_modeling_toward_dynamic_global_perception_for_ambiguity-free_rotation-i.md)
- [\[AAAI 2026\] Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation](learning_conjugate_direction_fields_for_planar_quadrilateral_mesh_generation.md)
- [\[CVPR 2026\] PointINS: Instance-Aware Self-Supervised Learning for Point Clouds](../../CVPR2026/3d_vision/pointins_instance-aware_self-supervised_learning_for_point_clouds.md)
- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](point-sra_self-representation_alignment_for_3d_representation_learning.md)

</div>

<!-- RELATED:END -->
