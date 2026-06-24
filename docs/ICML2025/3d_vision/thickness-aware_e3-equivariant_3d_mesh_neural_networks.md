---
title: >-
  [Paper Note] Thickness-aware E(3)-Equivariant 3D Mesh Neural Networks
description: >-
  [ICML 2025][3D Vision][3D Mesh] This paper proposes T-EMNN, which introduces a thickness-aware message passing mechanism and a PCA-based data-driven coordinate system. While maintaining the computational efficiency of surface meshes, it models the thickness interaction between opposing surfaces to achieve E(3)-equivariant/invariant node-level 3D deformation prediction.
tags:
  - "ICML 2025"
  - "3D Vision"
  - "3D Mesh"
  - "E(3)-Equivariance"
  - "Thickness-aware"
  - "Static Analysis"
  - "Deformation Prediction"
date: 2026-05-08
content_hash: 56c942ac8127724d
---

# Thickness-aware E(3)-Equivariant 3D Mesh Neural Networks

**Conference**: ICML 2025  
**arXiv**: [2505.21572](https://arxiv.org/abs/2505.21572)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Mesh, E(3)-Equivariance, Thickness-aware, Static Analysis, Deformation Prediction

## TL;DR

This paper proposes T-EMNN, which introduces a thickness-aware message passing mechanism and a PCA-based data-driven coordinate system. While maintaining the computational efficiency of surface meshes, it models the thickness interaction between opposing surfaces to achieve E(3)-equivariant/invariant node-level 3D deformation prediction.

## Background & Motivation

Mesh-based 3D static analysis methods (such as MGN, EGNN, and EMNN) have emerged as highly efficient alternatives to traditional finite element methods (FEM). However, existing methods suffer from two core issues:

**Neglecting Thickness Information**: Existing methods focus solely on surface topology and geometry, ignoring the interactions between opposing surfaces. Experiments reveal that the deformation correlation between thickness node pairs is significantly higher than the average correlation among neighbors within a radius (showing a notably higher Pearson correlation and a lower L2 norm). This indicates that thickness modeling is crucial for precise prediction.

**Missing Spatial Information**: To avoid computational overhead, equivariant methods like EGNN/EMNN only utilize local geometric features such as relative displacement, failing to capture global spatial relationships. On the other hand, high-order methods like spherical harmonics are too computationally expensive for large-scale industrial meshes (averaging ~54K nodes and ~325K edges).

Core Motivation: To introduce both thickness interaction modeling and global spatial information while preserving E(3)-equivariance and computational efficiency.

## Method

### Overall Architecture

T-EMNN adopts an encode-process-decode architecture, consisting of four core modules:

1. **Data-Driven Coordinate Transformation**: Transforms the original coordinates into an E(3)-invariant coordinate system.
2. **Encoder**: Separately encodes geometric features, spatial features, and experimental conditions.
3. **Dual Processor**: Alternatingly stacks a surface processor and a thickness processor.
4. **Decoder**: Fuses geometric, spatial, and conditional embeddings to predict deformation, followed by an inverse transformation back to the original coordinates.

### Key Designs

#### 1. E(3)-Invariant Data-Driven Coordinate System

A four-step coordinate transformation is designed to achieve E(3)-invariance:

- **Step 1**: Center the coordinates at the centroid: $\tilde{\mathbf{x}}_i = \mathbf{x}_i^{\text{orig}} - \mathbf{x}_{\text{cm}}$ (canceling translation).
- **Step 2**: Perform PCA on the centered coordinates to generate three orthogonal principal axes $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$, forming a rotation matrix $\mathbf{R}$.
- **Step 3**: Use a reference vector $\mathbf{v} = \mathbf{x}_{\text{cm}} - \mathbf{x}_{\text{bbox}}$ (the direction from the centroid to the bounding box center) to determine the signs of the principal axes, ensuring consistency.
- **Step 4**: Transform the coordinates: $\mathbf{x}_i^{\text{inv}} = \mathbf{R}^\top \tilde{\mathbf{x}}_i$.

Key Property: The transformed coordinates are invariant under any translation $g$ and orthogonal matrix $Q$ (a complete proof is provided in Appendix H of the paper). Storing $\mathbf{x}_{\text{cm}}$ and $\mathbf{R}$ allows for inverse transformation back to original coordinates, yielding E(3)-equivariant final predictions.

#### 2. Thickness Node Pairs and Thickness Edges

**Definition of Thickness Node Pairs**: For a node $v_i$, project along the negative direction of its normal vector to find the nearest node $\mathcal{T}(v_i)$ on the opposing surface:

$$\mathcal{T}(v_i) = \arg\min_{v_j \in V, v_j \neq v_i} \|\mathbf{x}_j - (\mathbf{x}_i - d \cdot \mathbf{n}_i^{\text{node}})\|, \quad \text{s.t.} \ (\mathbf{x}_j - \mathbf{x}_i) \cdot \mathbf{n}_i^{\text{node}} < 0$$

**Thickness Edge Features**: $\mathbf{f}_{i,\text{thick}} = [t(v_i), \mathbf{n}_i \cdot \mathbf{n}_i^{\mathcal{T}}]$, containing:
- Thickness distance: $t(v_i) = \|\mathbf{x}_i - \mathbf{x}_{\mathcal{T}(v_i)}\|$
- Normal vector dot product: Quantifies the degree of normal alignment between opposing surfaces.

#### 3. Learnable Thickness Threshold and Activation Function

Not all thickness node pairs represent actual thickness (e.g., node pairs on the sides of a wide flat plate represent "width" rather than "thickness"). Therefore, a learnable threshold $\tau$ and a sigmoid activation are introduced:

$$I_i = \frac{1}{1 + e^{\alpha(t(v_i) - \tau)}}$$

- When $t(v_i) \leq \tau$, $I_i \approx 1$ (retaining the thickness edge).
- When $t(v_i) > \tau$, $I_i \approx 0$ (filtering out noisy edges).
- $\alpha = 3$ controls the transition sharpness, and $\tau$ is automatically learned through training (converging to 5.68, filtering out 3.83% of the edges).

#### 4. Dual Processor Message Passing

**Surface Processor**: Performs standard message passing on surface edges $E$.
$$\mathbf{e}_{ij}^{(l+1)} \leftarrow f_{\text{surf}}^M(\mathbf{e}_{ij}^{(l)}, \mathbf{z}_i^{(l)}, \mathbf{z}_j^{(l)})$$
$$\mathbf{z}_i^{\text{surf},(l)} \leftarrow f_{\text{surf}}^V(\mathbf{z}_i^{(l)}, \sum_{j \in \mathcal{N}(i)} \mathbf{e}_{ij}^{(l+1)})$$

**Thickness Processor**: Performs weighted message passing on thickness edges.
$$\mathbf{e}_{i,\text{thick}}^{(l+1)} \leftarrow I_i \cdot f_{\text{thick}}^M(\mathbf{e}_{i,\text{thick}}^{(l)}, \mathbf{z}_i^{\text{surf},(l)}, \mathbf{z}_{\mathcal{T}(v_i)}^{\text{surf},(l)})$$
$$\mathbf{z}_i^{(l+1)} \leftarrow f_{\text{thick}}^V(\mathbf{z}_i^{\text{surf},(l)}, \mathbf{e}_{i,\text{thick}}^{(l+1)})$$

Each node has only one thickness edge (connecting to the node on the opposing surface), resulting in extremely low computational overhead. Thickness edges achieve single-hop message passing between opposing surfaces, replacing paths on the surface mesh that would otherwise require 6+ steps.

#### 5. Encoder and Decoder

- **Geometric Encoder**: Encodes E(3)-invariant features (distance, radius, etc.) using an MLP.
- **Spatial Encoder**: $\mathbf{z}_i^{\text{coord}} = \phi_{\text{coord}}(\mathbf{x}_i^{\text{inv}})$, which encodes the transformed coordinates.
- **Condition Encoder**: $\mathbf{h}_c = \phi_{\text{cond}}(\mathbf{c})$, which encodes experimental conditions (temperature, pressure, etc.).
- **Decoder**: Concatenates geometric embedding + spatial embedding $\rightarrow$ combine $\rightarrow$ concatenates condition embedding $\rightarrow$ decodes $\rightarrow$ inverse-transforms back to original coordinates.

### Loss & Training

- 200 epochs, learning rate 0.001, weight decay 5e-4.
- ReduceLROnPlateau adaptive schedule is applied to the thickness threshold $\tau$ (patience=5, factor=0.5).
- 3 layers of message passing, with a hidden dimension of 32.
- Hardware: NVIDIA RTX 4090, PyTorch 2.0.1 + PyG 2.4.0.

## Key Experimental Results

### Main Results

Dataset: Industrial injection molding dataset, 504 samples, 28 geometries $\times$ 18 experimental conditions, averaging ~54K nodes.

| Method | R²(In-Dist)↑ | R²(OOD)↑ | RMSE(In-Dist)↓ | MAE(In-Dist)↓ |
|------|-------------|-----------|-----------------|---------------|
| MLP (original coordinates) | 0.8984 | 0.7393 | 0.2818 | 0.1164 |
| MLP (invariant coordinates) | 0.9154 | 0.9385 | 0.2546 | 0.1043 |
| MGN (w/o coordinates) | 0.0782 | -0.0903 | 1.2608 | 0.5607 |
| MGN + invariant coordinates | 0.9113 | 0.9446 | 0.2241 | 0.0938 |
| EGNN (w/o coordinates) | -14341.0 | -32260.9 | 153.05 | 54.36 |
| EGNN + invariant coordinates | 0.9129 | 0.9443 | 0.2270 | 0.0963 |
| EMNN + invariant coordinates | 0.9149 | 0.9473 | 0.2210 | 0.0937 |
| **T-EMNN** | **0.9228** | **0.9513** | **0.2132** | **0.0892** |

Key Findings: EGNN/EMNN models without coordinate embeddings perform extremely poorly (yielding large negative R² scores), highlighting that spatial information is crucial. Methods using original coordinates experience a significant performance drop in OOD settings, verifying the necessity of E(3)-invariance.

### Ablation Study

| Configuration | RMSE↓ | MAE↓ | R²↑ | Explanation |
|------|-------|------|-----|------|
| w/o thickness | 0.2156 | 0.0908 | 0.9148 | Remove thickness edge features |
| w/o dot product | 0.2191 | 0.0912 | 0.9134 | Remove normal vector dot product |
| **T-EMNN (Full)** | **0.2132** | **0.0892** | **0.9228** | Both features used |

Computational Efficiency Comparison:

| Method | Speed (it/s) | GPU Memory (MB) |
|------|------------|--------------|
| MGN + invariant coordinates | 22.29 | 3,952 |
| EMNN + invariant coordinates | 19.99 | 7,322 |
| **T-EMNN** | **20.21** | **3,714** |

### Key Findings

1. **Stable Convergence of Thickness Threshold**: Across 3 seeds, $\tau$ consistently converges to 5.68, filtering out 3.83% of noisy thickness edges. Fixed threshold experiments verify that the performance peaks around 5.68.
2. **Generality of Thickness Edges**: Adding the thickness processor to the MGN/EGNN/EMNN baselines consistently improves performance for all methods.
3. **Surface Mesh vs. Voxel Mesh**: Although voxel meshes can model internal structures, dense connectivity hinders geometric understanding, and significantly escalates GPU memory usage and inference time. T-EMNN achieves better performance with lower overhead using surface meshes combined with thickness edges.
4. **Generalization to Dynamic Scenes**: On the Deforming Plate dataset, T-EMNN with thickness edges (R²=0.7579) significantly outperforms the version without thickness edges (R²=0.7007).

## Highlights & Insights

- **Ingenious Thickness Modeling**: Without altering the mesh topology, the method only adds "virtual" thickness edges connecting nodes on opposing surfaces. By adding at most one edge per node, the computational overhead is minimal while performance gains are substantial.
- **Data-Driven Coordinate System**: Uses PCA and a bounding box reference vector to achieve simple E(3)-invariant coordinate transformation, avoiding high-computation schemes like spherical harmonics, making it highly engineering-friendly.
- **Learnable Threshold**: Replaces hard thresholds with a soft sigmoid threshold, and learns $\tau$ end-to-end to automatically distinguish "thickness" from "width".
- **Computational Efficiency**: Evaluated at a GPU memory footprint of only 3,714 MB, approximately half that of EMNN, making it suitable for large-scale industrial meshes.

## Limitations & Future Work

1. **Symmetry Issues**: When a shape is perfectly symmetric across its three principal axes, the PCA directions become ambiguous ($\mathbf{b}_i \cdot \mathbf{v} = 0$), causing the coordinate transformation to fail. The authors acknowledge that this scenario is rare in real-world industrial geometries.
2. **Single Dataset**: Validation was only performed on the injection molding dataset (comprising 28 geometries). Generalization to other industrial scenarios (such as aerospace structural components, composite materials, etc.) remains to be verified.
3. **Homogeneous Material Assumption**: The thickness processor assumes uniform materials. Multi-material or anisotropic material scenarios would require additional design.
4. **Limitation in Dynamic Scenes**: The data-driven coordinate system is specifically designed for static analysis, requiring fallback to original coordinate systems for dynamic scenarios.

## Related Work & Insights

- **EGNN** (Satorras et al., 2021): Guarantees E(3)-equivariance via message passing but fails to exploit global spatial information.
- **EMNN** (Trang et al., 2024): Introduces geometric features like area and normal vectors on top of EGNN, yet still suffers from a limited local receptive field.
- **MGN** (Pfaff et al., 2020): A classic method within the encode-process-decode framework that does not guarantee equivariance.
- **Insights**: This work demonstrates a general strategy of "adding virtual cross-surface connections on surface meshes," which can be generalized to medical imaging (thin-walled organ analysis), CAD deformation prediction, etc.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|-------------|
| Novelty | 4 | The combination of thickness edges, learnable threshold, and PCA coordinate system is clever. |
| Technical Quality | 4 | Solid theoretical foundation (including the equivariance proof) and sufficient ablation studies. |
| Experimental Thoroughness | 3 | Evaluated on only a single industrial dataset and one public dataset. |
| Value | 4 | Computationally highly efficient, directly applicable to industrial CAE scenarios. |
| Writing Quality | 4 | Clear structure with intuitive illustrations. |
| **Total Score** | **3.8** | A utility-driven equivariant mesh method with strong industrial applicability. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](../../NeurIPS2025/3d_vision/copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[ICML 2025\] SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space](se3-equivariant_diffusion_policy_in_spherical_fourier_space.md)
- [\[ICML 2025\] FlowDrag: 3D-aware Drag-based Image Editing with Mesh-guided Deformation Vector Flow Fields](flowdrag_3d-aware_drag-based_image_editing_with_mesh-guided_deformation_vector_f.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](../../ICCV2025/3d_vision/ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[CVPR 2025\] HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery](../../CVPR2025/3d_vision/heatformer_a_neural_optimizer_for_multiview_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
