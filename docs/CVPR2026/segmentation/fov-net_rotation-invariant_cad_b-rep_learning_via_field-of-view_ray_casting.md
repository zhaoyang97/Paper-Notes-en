---
title: >-
  [Paper Note] FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting
description: >-
  [CVPR 2026][Segmentation][CAD B-rep learning] This paper presents FoV-Net, the first rotation-invariant framework for CAD B-rep learning that simultaneously captures local surface geometry and global structural context.…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "CAD B-rep learning"
  - "rotation invariance"
  - "ray casting"
  - "graph attention network"
  - "3D segmentation"
date: 2026-05-08
content_hash: 78dad0cd5e71a17f
---

# FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting

**Conference**: CVPR 2026
**arXiv**: [2602.24084](https://arxiv.org/abs/2602.24084)
**Code**: [GitHub](https://github.com/UGent-CVAMO/fovnet)
**Area**: Segmentation
**Keywords**: CAD B-rep learning, rotation invariance, ray casting, graph attention network, 3D segmentation

## TL;DR

This paper presents FoV-Net, the first rotation-invariant framework for CAD B-rep learning that simultaneously captures local surface geometry and global structural context. By introducing a Local Reference Frame UV grid (LRF UV) and a Field-of-View (FoV) ray casting descriptor, FoV-Net achieves robust classification and segmentation under arbitrary $\mathbf{SO}(3)$ rotations.

## Background & Motivation

CAD boundary representations (B-reps) are graph-structured, with faces as nodes and shared edges as connections, making them naturally suited for GNN-based processing. The central challenge in B-rep learning is designing descriptors that capture both local surface geometry and global structural context.

UV-Net pioneered the approach of sampling the UV parameter domain into a grid, storing absolute coordinates $(x,y,z)$ and normals $(n_x, n_y, n_z)$ in the global coordinate frame, establishing a foundation adopted by many subsequent methods. However, this reliance on global coordinates introduces severe **rotation sensitivity**. The authors reveal a striking phenomenon:

> Models achieving over 95% accuracy on aligned benchmarks may **collapse to 10%** under arbitrary $\mathbf{SO}(3)$ rotations.

This is unacceptable in real manufacturing pipelines, where CAD models originate from diverse sources and analysis must be robust to arbitrary orientations. Rotation augmentation provides partial relief but cannot cover all rotations and incurs computational overhead. More importantly, even on aligned data, rotation invariance is critical for segmentation: positional variation of faces across different locations may introduce spurious pose correlations, degrading generalization in low-data regimes.

## Method

### Overall Architecture

FoV-Net's input representation consists of two components:
1. **LRF UV grid**: encodes intrinsic surface geometry in each face's local reference frame
2. **FoV grid**: captures 3D structural context around each face via ray casting

Each descriptor is processed by a lightweight CNN to extract features, which are concatenated with face attributes and fused into a unified face embedding via an MLP. Information is then propagated over the B-rep graph using a Graph Attention Network (GAT).

### Key Designs

1. **Local Reference Frame UV Grid (LRF UV)**: The core innovation transforms the standard UV grid from the global coordinate frame into a per-face local reference frame. An orthonormal basis is constructed at the face centroid $\mathbf{o}$: $\mathbf{N}$ is the surface normal, $\mathbf{U}$ is the normalized projection of the U-direction tangent, and $\mathbf{V} = \mathbf{N} \times \mathbf{U}$. The transformation $\mathbf{p}' = \mathbf{R}_f^\top(\mathbf{p} - \mathbf{o})$ ensures that the same face produces identical descriptors under different poses. This exploits the advantage that B-rep faces naturally provide surface discretization and reference directions, avoiding the noise and consistency issues encountered in point cloud LRF construction. The result is an $n_u \times n_v \times 7$ tensor (relative coordinates + normals + trim mask). **Design Motivation**: decouple local geometry from global pose to guarantee rotation invariance.

2. **Field-of-View (FoV) Descriptor**: While LRF UV preserves local geometry, it discards structural context. FoV recovers this information via ray casting. Rays are cast from the face centroid $\mathbf{o}$ over the normal-hemisphere, discretized by elevation $n_{\text{el}}$ and azimuth $n_{\text{az}}$ (default $6 \times 12$, corresponding to 15° and 30° steps). Each ray records three quantities: (1) a hit flag, (2) distance to the intersection point, and (3) the dot product between the ray direction and the surface normal at the hit point (encoding the angle of incidence). Since ray origins and directions are defined in the local reference frame, they rotate with the face, guaranteeing rotation invariance.

    - **Outward Vision (OV)**: rays cast over the $\mathbf{N}$ hemisphere, probing the external environment of the face
    - **Inward Vision (IV)**: rays cast over the $-\mathbf{N}$ hemisphere, probing the interior structure of the solid. For watertight solids, inward rays yield dense intersections, providing rich distance information

   OV and IV together provide complementary, rotation-invariant structural context representations that are also inherently translation-invariant.

3. **Lightweight Network Architecture**:

    - OV/IV encoders: 2-layer CNN ($32 \to 64$ channels) + global average pooling + linear projection to 64-D; circular padding along the azimuth axis to handle periodicity
    - LRF UV encoder: 3-layer CNN ($32 \to 64 \to 128$) + global average pooling + 64-D projection
    - Face attributes: one-hot face type (6-D) + area = 7-D
    - 2-layer fusion MLP (hidden dim 256) → 64-D face embedding
    - 3-layer GAT (4 attention heads, 64-D) for message passing over the B-rep graph

   **Design Motivation**: edge features are omitted, as ablation experiments show negligible gains, reducing computational overhead.

### Loss & Training

- Classification: global max pooling + 2-layer MLP classification head
- Segmentation: face embeddings fed directly into a per-face prediction head
- Optimizer: Adam (lr=0.001, batch size 64) with early stopping (patience 30)
- Single NVIDIA RTX A5000 (24 GB); each experiment repeated 5 times, reported as mean ± std

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | FoV-Net (rotated) | FoV-Net (original) | UV-Net (rotated) | AAGNet (rotated) |
|---------|------|--------|-------------------|-------------------|------------------|-----------------|
| SolidLetters | Classification | Acc% | **96.35** | 96.35 | 8.94 | 14.03 |
| TraceParts | Classification | Acc% | **100.00** | 100.00 | 45.67 | 91.33 |
| Fusion360 | Segmentation | Acc% | **91.72** | 91.72 | 69.13 | 79.85 |
| Fusion360 | Segmentation | IoU% | **73.81** | 73.81 | 37.07 | 53.42 |
| MFCAD++ | Segmentation | Acc% | **99.33** | 99.33 | 35.44 | 80.13 |
| MFCAD++ | Segmentation | IoU% | **97.81** | 97.81 | 18.79 | 64.70 |

FoV-Net yields identical performance before and after rotation; UV-Net collapses from 97.10% to 8.94% on SolidLetters.

### Ablation Study

| Configuration | SolidLetters Acc% | Notes |
|---------------|-------------------|-------|
| FoV-Net (full) | 96.35 | LRF UV + FoV complementary |
| FoV grid only | 95.79 | Rich structural context |
| LRF UV only | 94.39 | Strong local geometry alone |
| OV only | 92.92 | Outward vision |
| IV only | 93.40 | Inward vision slightly better than OV |
| Face attributes only | 70.68 | Simple attributes insufficient |
| Topology only | 37.72 | Graph structure alone far from sufficient |

### Key Findings

- FoV grid and LRF UV are highly complementary; their combination outperforms either alone
- Even a very small FoV resolution (e.g., $4 \times 2$) yields effective features; however, a single ray ($1 \times 1$) collapses accuracy to 75%
- **Remarkable data efficiency**: FoV-Net reaches 80% accuracy on MFCAD++ with only 50 training samples, whereas UV-Net requires approximately 10,000
- Rotation augmentation improves robustness but at the cost of overall performance degradation — FoV-Net avoids this trade-off entirely

## Highlights & Insights

1. **Impactful problem revelation**: The paper is the first to systematically quantify the severity of rotation sensitivity in B-rep learning; the 95%→10% collapse is highly compelling
2. **Ray casting for descriptor construction**: A classic computer graphics technique is creatively introduced into B-rep learning, leveraging the precise intersection capabilities of CAD kernels to observe surrounding structure from the face centroid
3. **Complementary dual-vision (OV + IV)**: Outward rays probe the external environment while inward rays probe the interior structure; their combination provides complete 3D context
4. **LRF advantage for B-rep faces**: B-rep faces naturally provide parameterization and reference directions, making LRF construction far simpler and more reliable than for point clouds
5. **Data efficiency**: In the industrial CAD domain where IP constraints limit data availability, low data requirements represent a significant practical advantage

## Limitations & Future Work

- Ray casting relies on the PythonOCC CAD kernel (CPU) and only supports CPU-level parallelization, without GPU acceleration
- Current experiments focus on moderately complex single-part B-reps; scalability to larger assemblies remains to be validated
- The FoV grid uses equiangular 3D-to-2D mapping, which introduces polar distortion analogous to geographic projections; spherical CNNs may offer more uniform directional parameterization
- UV axis flip/swap ambiguities are not addressed (UV-Net mitigates this via D2-equivariant convolutions)
- Edge features are omitted, which may limit performance on tasks where edge features are critical, such as B-rep generation

## Related Work & Insights

- UV-Net established the UV grid paradigm but did not address rotation sensitivity — FoV-Net resolves this fundamentally via LRF transformation
- The strategy of combining LRF with global context, established in point cloud rotation-invariant methods, is successfully transferred to the B-rep domain
- Ray casting has prior applications in CAD for tool accessibility estimation; FoV-Net extends it as a general-purpose face descriptor
- The FoV descriptor holds significant potential for contrastive learning pretraining and unsupervised CAD retrieval

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First rotation-invariant B-rep learning framework; the ray casting descriptor is a highly original contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, classification + segmentation, rotated/original comparisons, detailed ablations, data efficiency analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is compelling, visualizations are clear, structure is rigorous
- Value: ⭐⭐⭐⭐⭐ Addresses a long-standing rotation sensitivity problem in B-rep learning with broad industrial application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection](pointer-cad_unifying_b-rep_and_command_sequences_via_pointer-based_edges_faces_s.md)
- [\[CVPR 2026\] Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)
- [\[CVPR 2026\] Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera](unified_spherical_frontend_learning_rotation-equivariant_representations_of_sphe.md)
- [\[CVPR 2026\] PRUE: A Practical Recipe for Field Boundary Segmentation at Scale](prue_a_practical_recipe_for_field_boundary_segmentation_at_scale.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)

</div>

<!-- RELATED:END -->
