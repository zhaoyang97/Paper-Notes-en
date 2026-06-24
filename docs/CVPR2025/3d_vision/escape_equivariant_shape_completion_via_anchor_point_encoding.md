---
title: >-
  [Paper Note] ESCAPE: Equivariant Shape Completion via Anchor Point Encoding
description: >-
  [CVPR 2025][3D Vision][Point Cloud Completion] ESCAPE proposes a rotation-equivariant point cloud completion method based on anchor distance encoding. By representing point clouds as distance matrices to high-curvature anchor points, the Transformer is enabled to predict the complete shape within a rotation-invariant distance space, and coordinates are subsequently recovered via optimization. This approach significantly outperforms existing methods under arbitrary input rotat…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Rotational Equivariance"
  - "Anchor Point Encoding"
  - "Shape Reconstruction"
  - "Transformer"
date: 2026-05-08
content_hash: 04bd33f2b99ef8a6
---

# ESCAPE: Equivariant Shape Completion via Anchor Point Encoding

**Conference**: CVPR 2025  
**arXiv**: [2412.00952](https://arxiv.org/abs/2412.00952)  
**Code**: None  
**Area**: Human Understanding  
**Keywords**: Point Cloud Completion, Rotational Equivariance, Anchor Point Encoding, Shape Reconstruction, Transformer

## TL;DR
ESCAPE proposes a rotation-equivariant point cloud completion method based on anchor distance encoding. By representing point clouds as distance matrices to high-curvature anchor points, the Transformer is enabled to predict the complete shape within a rotation-invariant distance space, and coordinates are subsequently recovered via optimization. This approach significantly outperforms existing methods under arbitrary input rotations (slashing CD-L1 on the PCN dataset from 26.65 to 10.58).

## Background & Motivation

1. **Background**: Point cloud completion is a crucial task in computer vision. Current mainstream methods (PoinTr, AdaPoinTr, SnowflakeNet, SeedFormer, AnchorFormer) are based on the Transformer architecture. Using an encoder-decoder framework, they predict complete shapes from partial point clouds, achieving outstanding performance on standard benchmarks.

2. **Limitations of Prior Work**: All existing methods rely on aligning objects to canonical coordinates, which assumes the orientation of the input point cloud is known and fixed. Under arbitrary rotations, their performance drops drastically—e.g., AdaPoinTr's CD-L1 spikes from around 6 to 33.52, performing worse than the earlier PoinTr (30.20).

3. **Key Challenge**: Existing methods use absolute coordinates as input features, which essentially memorizes the fixed orientation distribution of the training data rather than truly understanding the geometric structure. To achieve rotational equivariance, a rotation-independent geometric representation is required.

4. **Goal**: Design an end-to-end rotation-equivariant point cloud completion system that maintains stable completion quality under arbitrary rotation and translation.

5. **Key Insight**: Replacing absolute coordinates with the distance from point to anchor points. Distance is naturally rotation-invariant ($\|Rp - Ra\| = \|p - a\|$). Under general positions where the number of anchor points $k \geq d+1$ ($k \geq 4$ in 3D), the distance matrix uniquely determines the point positions (up to rigid transformation).

6. **Core Idea**: Transform the point cloud completion problem from "predicting points in 3D coordinate space" to "predicting distance matrices to anchor points in distance space". This keeps the entire Transformer processing naturally rotation-invariant, with coordinates recovered from predicted distances via optimization at the end.

## Method

### Overall Architecture
The input is a partial point cloud $P$ (2048 points), and the output is a complete point cloud (16384 points). The pipeline consists of three steps: (1) Select $k=8$ high-curvature anchor points from the input and calculate the distance matrix $D_p \in \mathbb{R}^{n \times k}$ from all points to the anchors; (2) Feed the distance matrix into a modified AdaPoinTr Transformer to predict the distance matrix $\hat{D}_c$ of the complete shape in distance space; (3) Recover the 3D coordinates from the predicted distances using Levenberg-Marquardt optimization. The entire pipeline maintains rotational equivariance.

### Key Designs

1. **High-Curvature Anchor Selection**:

    - Function: Select stable, consistent, and informative anchor points to provide a reference frame for distance encoding.
    - Mechanism: First, perform FPS (Farthest Point Sampling) starting from the centroid to sample $k$ cluster centers on the input point cloud (ensuring equivariance). Then, calculate the PCA curvature $\kappa_i = \min(\text{eig}(C_i))$ of the normal vectors within each cluster, and select the points with the highest curvature as anchors. High-curvature points correspond to geometric mutations (such as edges and corners), which exhibit semantic consistency across objects of the same category.
    - Design Motivation: Anchor points must remain consistent across different instances of the same category; randomly selected anchors are unstable across samples. High-curvature points are geometrically salient features that can be stably detected under different rotations. Selecting FPS starting from the centroid guarantees rotational equivariance.

2. **Distance-Space Transformer**:

    - Function: Complete the encoding and decoding of point cloud completion in the rotation-invariant distance space.
    - Mechanism: Modifying the AdaPoinTr architecture with three key changes: (a) Replacing absolute coordinates with the distance vector $d_{ij}$ to the anchors as input to the DGCNN feature extractor; (b) Replacing all coordinate information in self-attention layers with anchor distances; (c) Modifying the loss function for de-noising training to de-noise the distances with added noise. The key intuition is that two points adjacent in Euclidean space share similar distance vectors to the anchor points, making distance an effective encoder of spatial neighborhood relationships.
    - Design Motivation: Processing distances directly rather than coordinates makes all Transformer operations naturally unaffected by rotation. The error bound remains $O(1)$ (independent of network depth), whereas the errors of equivariant layers like Vector Neurons accumulate exponentially as $O(\alpha^L)$.

3. **Coordinate Optimization Recovery**:

    - Function: Recover 3D coordinates from the predicted distance matrix.
    - Mechanism: For each point $p=(x,y,z)$ to be recovered, solve the optimization problem $\min_p \sum_{j=1}^{k} (\|p - a_j\|_2 - \hat{d}_{ij})^2$, which finds the 3D coordinates that best match the predicted distances. Solve this using the Levenberg-Marquardt algorithm initialized from the anchor centroid. According to the reconstruction uniqueness theorem, the solution is unique (up to reflection symmetry) when $k \geq 4$ and the anchors are in general positions.
    - Design Motivation: Although the conversion from distance to coordinates is inevitable, since the anchor points themselves rotate along with the input, the recovered coordinates will rotate accordingly, thereby preserving the rotational equivariance of the entire pipeline.

### Loss & Training
Using Distance Matrix Chamfer Distance (DMCD) as the loss: $L = DMCD(\hat{D}_c, D_c)$, where DMCD calculates nearest-neighbor matching distances in the distance vector space (rather than coordinate space). Optimized using Adam with a learning rate of 0.001, decayed by 0.98 every 15 epochs, trained until validation loss no longer improves (up to 200 epochs). Training takes about 10 hours on a single RTX 3090.

## Key Experimental Results

### Main Results (PCN Dataset, Rotated Input)

| Category | SnowflakeNet | SeedFormer | PoinTr | AdaPoinTr | AnchorFormer | ESCAPE |
|------|-------------|-----------|--------|-----------|-------------|--------|
| Airplane | 72.71 | 76.19 | 13.03 | 12.10 | 11.88 | **8.6** |
| Car | 78.76 | 82.28 | 37.42 | 40.90 | 28.97 | **10.43** |
| Chair | 64.57 | 66.28 | 30.53 | 37.24 | 34.94 | **10.71** |
| Average | 88.85 | 92.15 | 30.20 | 33.52 | 26.65 | **10.58** |

### Ablation Study (Equivariant Encoding Comparison)

| Method | Airplane | Car | Chair | Lamp | Average |
|------|------|------|------|-----|------|
| SCARP | 104.4 | 135.9 | 147.1 | - | 124.0 |
| SnowflakeNet+VN | 10.65 | 11.92 | 17.86 | 22.82 | 18.62 |
| SnowflakeNet+PPF | 8.68 | 10.95 | 19.36 | 25.96 | 17.46 |
| ESCAPE (Ours) | **8.6** | **10.43** | **10.71** | **8.14** | **10.58** |

### Key Findings
- Existing SOTA methods collapse severely under rotated inputs—the more a method overfits to canonical coordinates (e.g., AdaPoinTr, AnchorFormer), the more it degrades.
- ESCAPE is the only method whose performance is unaffected by rotation, demonstrating the equivariance advantage of distance encoding.
- On the real-world OmniObject3D dataset (which naturally lacks canonical coordinates), ESCAPE's average CD-L1 is 18.82 vs AnchorFormer's 40.12, widening the gap to over 2x.
- On real LiDAR data from KITTI, ESCAPE's MMID (5.93) outperforms PoinTr (6.15) and SnowflakeNet (16.08).

## Highlights & Insights
- **Distance matrix as a geometric representation** is the most core contribution of this work—it shifts rotational equivariance from "requiring specialized network layers" to "naturally guaranteed at the representation level" with solid mathematical guarantees (reconstruction uniqueness theorem), which is more elegant than methods like Vector Neurons.
- **High-curvature anchor selection** ensures encoding consistency across samples—anchors of the same category tend to fall on semantically similar positions (e.g., chair legs, airplane wingtips), making the distance encoding comparable within the class.
- The approach of **transforming the "equivariance" problem into a "representation space selection" problem** can be transferred to other 3D tasks (such as registration, classification, segmentation), simply by replacing coordinate inputs with distance matrices.

## Limitations & Future Work
- The optimization recovery step (Levenberg-Marquardt) increases inference time and computational complexity, which is less efficient than directly outputting coordinates end-to-end.
- Anchor selection depends on normal estimation and FPS, making it sensitive to noise (isolated points in OmniObject3D cause abnormal metrics).
- Experiments were only conducted on 8 classes of PCN; the generalization to more complex geometries (such as highly non-convex shapes) has not been verified.
- The focus is solely on shape completion without expanding distance encoding to other 3D tasks (segmentation, detection).

## Related Work & Insights
- **vs AdaPoinTr/AnchorFormer**: These methods perform better under canonical coordinates (AdaPoinTr ~6 CD-L1) but crash under rotation. ESCAPE sacrifices about 4 points of canonical accuracy in exchange for rotational robustness.
- **vs Vector Neurons**: VN achieves rotational invariance through equivariant network layers, but the error accumulates exponentially. ESCAPE's distance encoding has a constant error $O(1)$, which is more suitable for deeper networks.
- **vs SCARP**: SCARP also aims for equivariant completion but only predicts coarse geometry, yielding a high CD-L1 of 124.0 vs ESCAPE's 10.58.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of using a distance matrix as a rotation-invariant representation is simple, powerful, and theoretically backed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets (PCN/OmniObject/KITTI) cover synthetic and real-world scenes with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, and experimental comparisons are comprehensive, though the details of the optimization recovery step could be slightly more thorough.
- Value: ⭐⭐⭐⭐ Points out the critical flaw of existing methods relying excessively on canonical coordinates, and the distance-encoding concept has broad transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Parametric Point Cloud Completion for Polygonal Surface Reconstruction](parametric_point_cloud_completion_for_polygonal_surface_reconstruction.md)
- [\[CVPR 2025\] GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors](genpc_zero-shot_point_cloud_completion_via_3d_generative_priors.md)
- [\[CVPR 2025\] PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors](pcdreamer_point_cloud_completion_through_multi-view_diffusion_priors.md)
- [\[CVPR 2025\] End-to-End HOI Reconstruction Transformer with Graph-based Encoding](end-to-end_hoi_reconstruction_transformer_with_graph-based_encoding.md)
- [\[NeurIPS 2025\] Learning Generalizable Shape Completion with SIM(3) Equivariance](../../NeurIPS2025/3d_vision/learning_generalizable_shape_completion_with_sim3_equivariance.md)

</div>

<!-- RELATED:END -->
