---
title: >-
  [Paper Note] QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition
description: >-
  [CVPR 2026][3D Vision][Quadrifocal tensor] This paper presents QuadSync, the first global synchronization algorithm for quadrifocal tensors. By constructing a block quadrifocal tensor and proving that it admits a Tucker…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Quadrifocal tensor"
  - "Tucker decomposition"
  - "Structure from Motion"
  - "global synchronization"
  - "multi-view geometry"
date: 2026-05-08
content_hash: dd10a1c1c0d0cbca
---

# QuadSync: Quadrifocal Tensor Synchronization via Tucker Decomposition

**Conference**: CVPR 2026
**arXiv**: [2602.22639](https://arxiv.org/abs/2602.22639)
**Code**: None
**Area**: 3D Vision
**Keywords**: Quadrifocal tensor, Tucker decomposition, Structure from Motion, global synchronization, multi-view geometry

## TL;DR

This paper presents QuadSync, the first global synchronization algorithm for quadrifocal tensors. By constructing a block quadrifocal tensor and proving that it admits a Tucker decomposition with multilinear rank $(4,4,4,4)$, the method recovers camera poses from four-view measurements via an ADMM-IRLS optimization framework, achieving superior synchronization accuracy over two-view and three-view methods in dense-view settings.

## Background & Motivation

### 1. State of the Field

Structure from Motion (SfM) is a central topic in 3D computer vision. Traditional SfM pipelines consist of feature detection and matching, relative pose estimation, synchronization, and reconstruction. Global synchronization methods avoid the error accumulation inherent in incremental approaches by processing all cameras simultaneously. Existing global methods are predominantly based on two-view measurements (fundamental/essential matrices), with a small number of recent works exploiting trifocal tensors (e.g., TrifocalSync).

### 2. Limitations of Prior Work

- **Limited two-view information**: Fundamental/essential matrices encode only the geometric relationship between two views, providing weak constraints.
- **Collinear degeneracy**: When camera centers are collinear, the rank of the multi-view fundamental matrix $\mathcal{F}^n$ drops from 6 to 4, and the multilinear rank of the trifocal tensor also degenerates, causing pose recovery to fail.
- **Underutilization of higher-order information**: Although the quadrifocal tensor encodes richer four-view geometric relationships, it has long been considered to have "only theoretical significance and no practical value."

### 3. Root Cause

Higher-order tensors (quadrifocal tensors) theoretically impose stronger geometric constraints, encoding complete four-view relationships and subsuming two-view and three-view information, yet no effective synchronization theory or practical algorithm has existed for them.

### 4. Paper Goals

- Establish an algebraic constraint theory for block quadrifocal tensors via low-rank decomposition.
- Globally recover camera poses from a collection of quadrifocal tensors.
- Jointly exploit two-view, three-view, and four-view measurements to improve synchronization accuracy.

### 5. Starting Point

From the perspective of Tucker decomposition, the paper proves that the block quadrifocal tensor possesses a low-rank structure with multilinear rank $(4,4,4,4)$, and that its factor matrices are precisely the stacked camera matrices. This reduces the synchronization problem to a constrained tensor decomposition optimization.

### 6. Core Idea

The paper constructs a block quadrifocal tensor $\mathcal{Q}^n \in \mathbb{R}^{3n \times 3n \times 3n \times 3n}$ and proves that it admits the Tucker decomposition $\mathcal{Q}^n = \mathcal{G}_Q \times_1 C \times_2 C \times_3 C \times_4 C$, where $C \in \mathbb{R}^{3n \times 4}$ is the stacked camera matrix. Compared with two-view (rank 6) and three-view (multilinear rank $(6,4,4)$) cases, the four modes of the quadrifocal tensor share the same factor matrix $C$, and the core tensor $\mathcal{G}_Q$ is a fixed sparse tensor, yielding the strongest constraints.

## Method

### Overall Architecture

1. **Input**: Trifocal tensors and essential matrices estimated from feature matching.
2. **Quadrifocal tensor estimation**: Reliable quadruples are selected via a four-cycle consistency heuristic on trifocal tensors; quadrifocal tensors are then computed directly from the camera matrices.
3. **Block quadrifocal tensor construction**: All quadrifocal tensors are stacked by index into $\mathcal{Q}^n$.
4. **QuadSync optimization**: An ADMM-IRLS framework solves for the scales and camera matrix.
5. **(Optional) Joint optimization**: Simultaneous synchronization of block quadrifocal tensors, block trifocal tensors, and block essential matrices.
6. **Output**: Global camera poses (projective reconstruction).

### Key Designs

#### Design 1: Tucker Decomposition Theory for Block Quadrifocal Tensors

- **Function**: Proves that the block quadrifocal tensor admits a Tucker decomposition under appropriate block scaling, with multilinear rank exactly $(4,4,4,4)$.
- **Mechanism**: Each element of the quadrifocal tensor satisfies $(Q_{ijkl})^{pqrs} = \det[P_i^p; P_j^q; P_k^r; P_l^s]$. Stacking all blocks yields $\mathcal{Q}^n = \mathcal{G}_Q \times_1 C \times_2 C \times_3 C \times_4 C$, where the core tensor $\mathcal{G}_Q \in \mathbb{R}^{4 \times 4 \times 4 \times 4}$ has entries in $\{-1, 0, 1\}$.
- **Design Motivation**: This decomposition directly reformulates synchronization as the extraction of factor matrices from a Tucker decomposition. Moreover, the multilinear rank $(4,4,4,4)$ holds unconditionally when camera centers are non-coincident, avoiding the collinear degeneracy that affects two-view (rank $6 \to 4$) and three-view ($(6,4,4) \to (5,4,4)$) cases.

#### Design 2: ADMM-IRLS Optimization Algorithm

- **Function**: Jointly solves for the unknown scales $\Lambda$ and the camera matrix $C$.
- **Mechanism**: Auxiliary variables $C_1 = C_2 = C_3 = C_4 = B$ are introduced to decouple the quartic coupling. The outer IRLS handles the $\ell_1$ norm for robustness, while the inner ADMM alternately solves for each variable. Each subproblem admits a closed-form or simple convex solution.
- **Design Motivation**: The original problem is quartic in $C$ and non-convex due to scale ambiguity. ADMM relaxes equality constraints via augmented Lagrangian terms, and IRLS converts the $\ell_1$ norm into iteratively reweighted least squares to reduce sensitivity to outliers.

#### Design 3: Joint Optimization Framework

- **Function**: Simultaneously synchronizes quadrifocal tensors, trifocal tensors, and essential matrices.
- **Mechanism**: All three share factor matrices—the block quadrifocal and block trifocal tensors share the stacked camera matrix $C$, while the block trifocal and block essential matrices share the line projection matrix $\mathcal{P}$. The objective is a weighted sum of three loss terms, normalized by the number of observed blocks for each tensor type.
- **Design Motivation**: Exploits the complementarity of different-order information: quadrifocal tensors provide the strongest constraints but are the hardest to estimate and the sparsest, while two-view data are the most abundant but provide the weakest constraints. Joint optimization achieves a balance between low- and high-order information.

#### Design 4: Four-Cycle Consistency Heuristic

- **Function**: Assesses the corruption level of a quadruple to filter reliable quadrifocal tensor estimates.
- **Mechanism**: Given four trifocal tensors $T_{ijk}, T_{jkl}, T_{kli}, T_{lij}$, the method progressively aligns projective coordinate frames around the cycle and measures the rotational/translational inconsistency upon closure.
- **Design Motivation**: No direct robust estimator exists for the quadrifocal tensor; the consistency of trifocal tensors is therefore used as an indirect quality proxy. Quadruples with rotation error $> 3°$ or translation error $> 0.2$ are discarded.

### Loss & Training

- **QuadSync loss**: $\sum_{(i,j,k,l) \in \Omega} \| \Lambda_{ijkl} \tilde{\mathcal{Q}}^n_{ijkl} - \llbracket \mathcal{G}_Q; C, C, C, C \rrbracket_{ijkl} \|_F$ ($\ell_1$ norm for robustness against outliers).
- **Constraints**: Scales $\Lambda$ are symmetric and normalized as $\|\Lambda\|_F^2 = 1$ to avoid trivial solutions.
- **Initialization**: HOSVD extracts the leading 4 singular vectors as an initial estimate of the camera matrix.
- **Hyperparameters**: $\rho = 0.01$ (QuadSync), 4 IRLS iterations, 1 ADMM iteration per IRLS; for joint optimization, $\rho = 0.00001$, 2 IRLS iterations.

## Key Experimental Results

### Main Results

#### Table 1: Average Position Error on ETH3D Dataset (11 Scenes)

| Method | # Best/Near-Best Scenes | Characteristic |
|--------|------------------------|----------------|
| NRFM | — | Two-view fundamental matrix synchronization |
| MPLS+LUD | — | Separate rotation + position synchronization |
| MPLS+BATA | — | Rotation + robust position synchronization |
| TrifocalSync | — | Trifocal tensor synchronization |
| MPLS+Cycle-Sync | — | Rotation + higher-order cycle synchronization (SOTA) |
| **QuadSync** | **7/11** | Quadrifocal tensor synchronization |
| **Joint Opt.** | **7/11** | Joint two/three/four-view synchronization |

#### Table 2: Average Position Error on EPFL Dataset (6 Scenes)

| Method | # Best/Near-Best Scenes | Characteristic |
|--------|------------------------|----------------|
| TrifocalSync | — | Three-view baseline |
| MPLS+Cycle-Sync | — | Current SOTA |
| **QuadSync** | **4/6** | Best performance under dense graphs |
| **Joint Opt.** | **4/6** | Complementary to QuadSync |

### Ablation Study

- **Density dependence**: When the quadrifocal tensor observation rate exceeds 70%, QuadSync/Joint Opt. significantly outperform SOTA; performance degrades below 30%.
- **Collinear configuration**: On a near-collinear subsequence from ETH3D SLAM (plant\_scene\_1), QuadSync successfully recovers poses while two-view methods (fundamental matrix) fail entirely.
- **Random sampling acceleration**: Randomly sampling $m = O(1)$ columns for the row updates of $C_i$ yields substantial speedups without accuracy loss, since the low rank is independent of the number of cameras.

### Key Findings

1. Quadrifocal tensor synchronization consistently matches or surpasses all baselines in position accuracy on dense view graphs (observation rate $> 70\%$).
2. Joint optimization further leverages the complementarity of different-order information, compensating for QuadSync's weaknesses in partially sparse scenes.
3. Quadrifocal tensors are naturally immune to collinear degeneracy—the multilinear rank $(4,4,4,4)$ is unaffected by camera collinearity.
4. Although the problem is non-convex, HOSVD initialization is empirically sufficient and does not require the complex initialization strategies needed by two-view methods.

## Highlights & Insights

1. **Solid theoretical contribution**: The paper fully establishes the Tucker decomposition theory for block quadrifocal tensors, proving multilinear rank, projective rank, scale identifiability, and related algebraic properties—constituting the first theoretical framework for quadrifocal tensor synchronization.
2. **Immunity to collinear degeneracy**: This is the most fundamental advantage over two-view and three-view methods—the four modes of the quadrifocal tensor symmetrically share the same factor matrix and do not degenerate under collinear configurations.
3. **Elegant structure of increasing constraint strength**: $\text{codim}(Q) = \Omega(n^4) > \text{codim}(T) = \Omega(n^3) > \text{codim}(E) = \Omega(n^2)$; the constraints imposed by higher-order measurements on the low-rank set grow exponentially.
4. **Unified perspective via joint framework**: By sharing factor matrices, synchronization of two-, three-, and four-view tensors is integrated into a single optimization problem, elegantly exploiting information at different orders.

## Limitations & Future Work

1. **Dependence on dense view graphs**: Quadrifocal tensor estimation requires four views to share sufficiently many inlier correspondences; performance degrades in sparse scenes due to low observation rates.
2. **Indirect quadrifocal tensor estimation**: The current approach estimates quadrifocal tensors indirectly from trifocal tensors, introducing additional noise. Direct robust estimation of quadrifocal tensors from point/line correspondences remains an open problem.
3. **High computational cost**: The block quadrifocal tensor is a fourth-order tensor of size $3n \times 3n \times 3n \times 3n$, with $O(n^4)$ variable dimensions, making it ill-suited for large-scale scenes.
4. **Absence of distributed methods**: The paper mentions the possibility of extending to large-scale datasets via distributed synchronization but provides no complete solution.
5. **Validation limited to projective/calibrated settings**: Experiments are restricted to calibrated cameras; performance in uncalibrated settings is unknown.

## Related Work & Insights

- **TrifocalSync** [35]: The direct predecessor of this work, establishing the Tucker decomposition and synchronization framework for block trifocal tensors. QuadSync is a natural fourth-order extension.
- **COLMAP/GLOMAP** [42, 39]: Mainstream incremental/global SfM pipelines; the proposed method can serve as an alternative synchronization module.
- **Cycle-Sync** [32]: The latest SOTA position synchronization method exploiting higher-order cycle consistency. Conceptually related but does not directly operate on higher-order tensors.
- **NRFM** [45]: Classical multi-view fundamental matrix synchronization (rank-6 constraint); the second-order counterpart of QuadSync.
- **Insights**: The trend of stronger constraints from higher-order geometric quantities (trifocal → quadrifocal) is clear. Future directions include generalizing to fifth-order or higher tensors, and deeply integrating this framework with learning-based feature matching (e.g., GlueStick).

## Rating

⭐⭐⭐⭐ The theoretical contribution is outstanding, establishing for the first time a complete algebraic framework for quadrifocal tensor synchronization with demonstrated practical utility. However, the dependence on dense view graphs and indirect estimation limits applicability to large-scale scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery](reparameterized_tensor_ring_functional_decomposition_for_multi-dimensional_data_.md)
- [\[ICCV 2025\] Proactive Scene Decomposition and Reconstruction](../../ICCV2025/3d_vision/proactive_scene_decomposition_and_reconstruction.md)
- [\[CVPR 2026\] GGPT: Geometry-Grounded Point Transformer](ggpt_geometry_grounded_point_transformer.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](../../ICCV2025/3d_vision/instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[NeurIPS 2025\] VisualSync: Multi-Camera Synchronization via Cross-View Object Motion](../../NeurIPS2025/3d_vision/visual_sync_multi-camera_synchronization_via_cross-view_object_motion.md)

</div>

<!-- RELATED:END -->
