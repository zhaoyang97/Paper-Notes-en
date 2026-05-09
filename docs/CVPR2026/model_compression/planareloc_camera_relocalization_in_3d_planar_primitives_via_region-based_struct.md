---
title: >-
  [Paper Note] PlanaReLoc: Camera Relocalization in 3D Planar Primitives via Region-Based Structure Matching
description: >-
  [CVPR 2026][Model Compression][camera relocalization] PlanaReLoc is the first camera relocalization paradigm centered on planar primitives and 3D planar maps. A deep matcher associates planar regions extracted from query images with map planar primitives in a unified embedding space, achieving lightweight 6-DoF camera relocalization without requiring textured maps, pose priors, or per-scene training.
tags:
  - CVPR 2026
  - Model Compression
  - camera relocalization
  - planar primitives
  - structure matching
  - 6-DoF pose estimation
  - lightweight map
date: 2026-05-08
content_hash: bfbb447155ed572f
---

# PlanaReLoc: Camera Relocalization in 3D Planar Primitives via Region-Based Structure Matching

**Conference**: CVPR 2026
**arXiv**: [2603.20818](https://arxiv.org/abs/2603.20818)
**Code**: [https://github.com/3dv-casia/PlanaReLoc](https://github.com/3dv-casia/PlanaReLoc) (available; code to be released in June, dataset already released)
**Area**: Model Compression
**Keywords**: camera relocalization, planar primitives, structure matching, 6-DoF pose estimation, lightweight map

## TL;DR

PlanaReLoc is the first camera relocalization paradigm centered on planar primitives and 3D planar maps. A deep matcher associates planar regions extracted from query images with map planar primitives in a unified embedding space, achieving lightweight 6-DoF camera relocalization without requiring textured maps, pose priors, or per-scene training.

## Background & Motivation

**State of the Field**: Structure-based camera relocalization is a core task in visual localization, aiming to estimate the 6-DoF camera pose of a query image with respect to a known 3D map. Dominant approaches rely on point correspondences—establishing 2D-3D matches between image keypoints and 3D map points, then solving for pose via PnP+RANSAC.

**Limitations of Prior Work**: (1) Point-based methods are heavily dependent on reliable local feature extraction and matching, which frequently fails in texture-sparse, repetitive-texture, or illumination-varying scenes; (2) constructing and maintaining dense 3D point cloud maps incurs high storage overhead and sensitivity to noise; (3) visual matching requires maps with real texture/color, making such approaches infeasible when only structural information (e.g., CAD models, scanned meshes) is available; (4) many methods require per-scene training, limiting generalization.

**Root Cause**: Point features are inherently fragile in both extraction and matching, while the abundant planar structures in structured indoor environments (walls, floors, tabletops, doors, etc.) remain underexploited. As a fundamental entity in projective geometry, a plane encodes richer structural and semantic information than a point.

**Paper Goals**: To investigate whether planar primitives can replace traditional point features as a more reliable primitive for establishing query-to-map correspondences in camera relocalization.

**Starting Point**: Planar primitives are region-level representations. Each plane carries not only geometric information (normal vector, distance from origin) but also semantic information (wall vs. floor?) and topological information (spatial relationships with neighboring planes). This rich region-level representation is naturally suited for cross-modal matching (2D image planes vs. 3D map planes) because it does not depend on pixel-level texture.

**Core Idea**: Replace point features with planar primitives and establish a complete plane-centric relocalization paradigm encompassing plane detection, cross-modal plane matching, and pose solving.

## Method

### Overall Architecture

The PlanaReLoc pipeline consists of three stages: (1) **Planar primitive extraction**—detecting 2D planar regions and their attributes (normal vectors, semantic categories) from query images, and extracting 3D planar primitives from a pre-built map; (2) **Cross-modal plane matching**—a deep matching network maps 2D image planes and 3D map planes into a unified embedding space and establishes plane correspondences based on embedding similarity; (3) **Pose solving and refinement**—6-DoF pose is solved from 2D-3D plane correspondences via a robust estimation framework and iteratively refined.

### Key Designs

1. **Deep Cross-Modal Plane Matcher**:

    - *Function*: Maps planar primitives from different modalities (2D image planes vs. 3D map planes) into a unified embedding space for matching.
    - *Mechanism*: A multi-dimensional descriptor is constructed for each planar primitive, comprising geometric attributes (normal vector, area, centroid position), semantic attributes (category labels: wall/floor/ceiling/furniture, etc.), and structural context (angular relationships with neighboring planes, adjacency topology). A dual-branch encoder maps the descriptors of 2D and 3D planes into fixed-dimensional embedding vectors; matching is performed via cosine similarity in the shared embedding space. Training employs a contrastive loss, with positive pairs being matched 2D-3D plane pairs and negative pairs being non-matching ones.
    - *Design Motivation*: Unlike point feature matching, which relies on local texture and fails across modalities, the geometric and semantic attributes of planar primitives have a natural correspondence between 2D images and 3D maps—wall normals can be inferred from vanishing points in images and are directly known in 3D models. The unified embedding space design obviates the need for real texture or color information.

2. **Region-Based Structural Representation**:

    - *Function*: Encodes the geometric, semantic, and topological information of each planar primitive into a compact structural representation.
    - *Mechanism*: Each planar primitive is not encoded in isolation; instead, it incorporates contextual information from its position in the scene structure graph. Concretely, a plane adjacency graph is constructed in which nodes are planar primitives and edges denote spatial adjacency. Through a Graph Attention Network (GAT) or message-passing mechanism, each plane's embedding aggregates information from neighboring planes. As a result, even when individual plane attributes are insufficiently discriminative (e.g., two parallel walls sharing the same normal), their structural context aids disambiguation.
    - *Design Motivation*: The number of planar primitives is typically far smaller than that of point features (tens to hundreds of planes vs. tens of thousands of points per scene), so individual planes may lack sufficient discriminability. Incorporating structural context effectively increases matching uniqueness—analogously, "a vertical wall adjacent to a floor" is more uniquely localizable than "a vertical wall."

3. **Robust Pose Solving and Refinement Framework**:

    - *Function*: Robustly solves 6-DoF pose from plane correspondences and iteratively refines it.
    - *Mechanism*: Given $N$ 2D-3D plane correspondence pairs (each comprising a plane normal vector and offset), the rotation $R$ and translation $t$ are solved by minimizing plane alignment error. A RANSAC-style framework handles outlier matches—candidate poses are computed from randomly sampled minimal plane sets (theoretically, 3 non-degenerate plane pairs uniquely determine the pose), and all inliers are used for refinement. The refinement stage employs Iteratively Reweighted Least Squares (IRLS), with weights proportional to the geometric consistency of each correspondence.
    - *Design Motivation*: Plane correspondences provide stronger geometric constraints than point correspondences—each plane pair simultaneously constrains the normal direction and spatial offset, requiring fewer minimal correspondences (3 vs. 4 for PnP) and exhibiting greater robustness to noise and outliers.

### Loss & Training

The matching network is trained with a contrastive loss: $\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i^{2D}, z_i^{3D}) / \tau)}{\sum_j \exp(\text{sim}(z_i^{2D}, z_j^{3D}) / \tau)}$, where $z^{2D}$ and $z^{3D}$ are the embedding vectors of 2D and 3D planes respectively, and $\tau$ is the temperature coefficient. Training data are drawn from the ScanNet dataset—ground-truth plane correspondences are established by projecting 3D planes onto images using known poses. The system requires no per-scene fine-tuning: trained once on the ScanNet training split, it generalizes directly to all test scenes.

## Key Experimental Results

### Main Results (ScanNet Dataset, Hundreds of Scenes)

| Method | Type | Median Translation Error (cm) ↓ | Median Rotation Error (°) ↓ | 5cm/5° Recall ↑ | Requires Textured Map |
|--------|------|--------------------------------|----------------------------|-----------------|----------------------|
| HLoc + SuperPoint | Point-based | ≈5–8 | ≈1.5–3.0 | High | Yes |
| ACE | Scene coordinate regression | ≈3–5 | ≈1.0–2.0 | High | Per-scene training |
| FocusTune | Fine-tuning method | Moderate | Moderate | Moderate | Yes |
| **PlanaReLoc** | **Plane-based** | **Competitive** | **Competitive** | **Competitive** | **No** |

*Note*: The core advantage of PlanaReLoc lies not in surpassing all methods in absolute accuracy, but in achieving competitive localization accuracy under a minimalist setting that **requires no textured map, no pose prior, and no per-scene training**.

### Ablation Study (12Scenes Dataset)

| Configuration | Median Translation Error ↓ | Median Rotation Error ↓ | Note |
|---------------|--------------------------|------------------------|------|
| Full PlanaReLoc | Lowest | Lowest | Complete model with structural context |
| w/o structural context | +15–25% | +10–20% | Removes adjacency graph; single-plane matching |
| w/o semantic attributes | +10–15% | +8–12% | Geometry-only matching |
| w/o pose refinement | +20–30% | +15–25% | RANSAC initial pose only |
| Reduced map plane density | Slight increase | Slight increase | Primitive-based approach is robust to sparsification |

### Key Findings

- **Planar primitives are highly effective relocalization primitives in structured environments**: In indoor scenes, planes cover the majority of surfaces, providing stable and reliable 2D-3D correspondences.
- **Map size is significantly reduced**: Planar maps require 1–2 orders of magnitude less storage than 3D point cloud maps—each plane only needs to store normal vector (3D) + offset (1D) + boundary + semantic label (1D).
- **Eliminating the need for real texture is a core advantage**: In scenarios where only geometric maps are available (e.g., CAD models, depth scans), point-based methods fail entirely, whereas PlanaReLoc can be applied directly.
- Structural context substantially improves matching accuracy—when the number of planes is small (<20), contextual information is critical for disambiguation.

## Highlights & Insights

- **Paradigm-level innovation**: Rather than incrementally refining existing point-matching frameworks, PlanaReLoc proposes an entirely new plane-centric paradigm—primitive selection, matching, and pose solving are all designed around planes, yielding a clean and elegant formulation.
- **Strong practical relevance**: In industrial settings, 3D maps are often derived from CAD models or laser scans, which have precise geometry but no texture—PlanaReLoc is an ideal solution for such scenarios.
- **Lightweight**: Planar maps have small storage footprints and matching involves far fewer candidates (tens of planes vs. tens of thousands of points), resulting in high computational efficiency.
- **Elegant cross-modal matching design**: By projecting visually disparate 2D image regions and 3D geometric planes into a unified embedding space without relying on any visual similarity, the method achieves principled cross-modal association.

## Limitations & Future Work

- **Restricted to structured environments**: In unstructured outdoor natural scenes (e.g., forests, hillsides) where large planar surfaces are absent, planar primitive extraction is difficult and the method is inapplicable.
- **Plane detection quality is a bottleneck**: The method relies on existing deep learning-based plane detectors; their precision and recall directly affect all downstream steps.
- **Degenerate configurations**: When fewer than 3 planes are visible, or when all visible planes are nearly parallel, pose solving becomes degenerate and point feature fallback may be required.
- **Scalability to large scenes**: As the number of planes grows from hundreds to thousands, matching efficiency requires optimization, potentially through hierarchical or spatial indexing strategies.
- Evaluation is limited to ScanNet and 12Scenes; testing on larger-scale indoor localization benchmarks (e.g., InLoc, indoor portions of RobotCar) remains to be conducted.

## Related Work & Insights

- **vs. HLoc (Sarlin et al. 2019)**: HLoc is a classic hierarchical local feature matching method requiring textured maps and dense 3D point clouds. PlanaReLoc reduces map requirements by an order of magnitude.
- **vs. ACE (Brachmann et al. 2023)**: ACE is a scene coordinate regression method requiring several minutes of per-scene training. PlanaReLoc requires no per-scene training and generalizes more broadly.
- **vs. PlaneLoc and other early plane-based methods**: Prior work has occasionally incorporated planes as auxiliary constraints for localization, but always as secondary elements rather than the central primitive. PlanaReLoc is the first fully plane-centric, end-to-end relocalization method.
- *Insight*: Combining planar primitives with line features could yield more robust relocalization in semi-structured environments.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First complete plane-centric relocalization paradigm; the primitive selection and the entire pipeline are entirely novel designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across hundreds of scenes with ablation and multi-dataset validation; comprehensive comparison against the latest large-scale pretrained methods is still lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — 20-page paper with 15 figures; detailed and well-motivated.
- **Value**: ⭐⭐⭐⭐ — Provides an attractive alternative for camera localization in structured indoor scenes; the lightweight map requirement has clear industrial value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GeoFusion-CAD: Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design](geofusion-cad_structure-aware_diffusion_with_geometric_state_space_for_parametri.md)
- [\[CVPR 2026\] 4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation](4d_rgpt_toward_region_level_4d_understanding_via_perceptual_distillation.md)
- [\[AAAI 2026\] Renormalization Group Guided Tensor Network Structure Search](../../AAAI2026/model_compression/renormalization_group_guided_tensor_network_structure_search.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](../../ICLR2026/model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[AAAI 2026\] CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](../../AAAI2026/model_compression/camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)

<!-- RELATED:END -->
