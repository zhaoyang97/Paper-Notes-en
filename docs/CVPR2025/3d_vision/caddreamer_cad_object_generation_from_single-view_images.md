---
title: >-
  [Paper Note] CADDreamer: CAD Object Generation from Single-view Images
description: >-
  [CVPR 2025][3D Vision][CAD Reconstruction] This paper proposes CADDreamer, which directly generates CAD models with compact B-rep representations, clear structures, and sharp edges from a single RGB image. Utilizing a semantic-enhanced multi-view diffusion model and a geometric-topological extraction module, it supports five primitive types: planes, cylinders, cones, spheres, and tori.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "CAD Reconstruction"
  - "Single-view 3D Generation"
  - "Boundary Representation"
  - "Diffusion Models"
  - "Geometric Optimization"
date: 2026-05-08
content_hash: 46e43bf92ef9580f
---

# CADDreamer: CAD Object Generation from Single-view Images

**Conference**: CVPR 2025  
**arXiv**: [2502.20732](https://arxiv.org/abs/2502.20732)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: CAD Reconstruction, Single-view 3D Generation, Boundary Representation, Diffusion Models, Geometric Optimization

## TL;DR

This paper proposes CADDreamer, which directly generates CAD models with compact B-rep representations, clear structures, and sharp edges from a single RGB image. Utilizing a semantic-enhanced multi-view diffusion model and a geometric-topological extraction module, it supports five primitive types: planes, cylinders, cones, spheres, and tori.

## Background & Motivation

Significant progress has been made in 3D generation based on diffusion models in recent years. However, the generated meshes are typically over-dense and unstructured triangular meshes. This contrasts sharply with the compact, structured, and sharp-edged CAD models created by human designers. This gap seriously limits the application of generative models in scenarios requiring high-quality structured 3D models, such as gaming, manufacturing, and product design.

Existing Image-to-CAD methods mainly fall into two categories: retrieval-and-assembly methods which rely on massive CAD databases and are restricted to implicit surfaces, and sketch-and-extrude methods which can directly generate B-reps but limit the generated objects to planes and cylinders. The **Key Challenge** is: **diffusion models lack an understanding of high-level geometric structures (primitive semantics), while noise and distortion make accurate primitive fitting and watertight B-rep construction highly challenging**.

**Key Insight** of this work: By encoding primitive semantics into the color space and leveraging the strong priors of pre-trained diffusion models, the model simultaneously understands low-level geometry (normal maps) and high-level structure (primitive semantic maps). A complete and watertight B-rep is then generated via geometric optimization and topology-preserving extraction.

## Method

### Overall Architecture

The method consist of two main modules: (1) Multi-view Generation Module: predicts multi-view normal maps and semantic primitive maps from a single-view RGB image, reconstructs a 3D mesh, and segments it into patches corresponding to the primitives via Graph Cut; (2) Geometric and Topological Extraction Module: rectifies primitive parameters through geometric optimization and computes primitive intersection lines, vertices, and faces using a topology-preserving extraction method to generate a watertight B-rep.

### Key Designs

1. **Semantic-enhanced Multi-view 2D Diffusion Model**:
    - **Function**: Jointly generates normal maps and semantic primitive maps from 6 viewpoints based on a single-view normal map.
    - **Mechanism**: Fine-tuned on the cross-domain diffusion model of Wonder3D, this module encodes 7 semantic labels (5 primitive types + background + feature lines) into the RGB color space. It utilizes cross-view and cross-domain attention mechanisms to guarantee the multi-view consistency of both geometry and semantics. The normal maps are fed into NeuS for 3D mesh reconstruction, while the semantic maps are used to segment the mesh via back-projection and Graph Cut.
    - **Design Motivation**: Directly encoding semantic information into the color space reuses the strong priors of pre-trained diffusion models. This allows the model to implicitly understand high-level CAD structures without requiring the design of an additional semantic segmentation branch.

2. **Geometric Optimization Algorithm (Primitive Stitching)**:
    - **Function**: Corrects inaccurate primitive parameters caused by reconstruction noise to restore the topological and geometric relationships between primitives.
    - **Mechanism**: It samples $k$ stitching points on the boundaries of the mesh segmentations and projects each stitching point onto the two adjacent primitive surfaces, minimizing the distance between these projected points $f_{stch}(v_i) = \|\pi(v_i, P_A) - \pi(v_i, P_B)\|$. Concurrently, geometric constraints are enforced to maintain relations of parallelism (shared axes), collinearity ($p_A = p_B + \vec{x}_B t$), and perpendicularity ($\vec{x}_C \cdot \vec{x}_D = 0$). The optimization is solved using L-BFGS.
    - **Design Motivation**: Even minor deviations in primitive parameters can cause the intersection computation to fail, resulting in dangling faces or non-watertight B-reps. Geometric relationship constraints ensure the structural integrity of the generated CAD model.

3. **Topology-preserving B-rep Construction**:
    - **Function**: Extracts topological representations (vertices, edges, faces) from the segmented mesh to guide the primitive intersection calculation, generating watertight B-reps.
    - **Mechanism**: Mesh patches are map to topological faces, shared boundaries between two patches to topological edges, and vertices where more than two patches meet to topological vertices. Guided by this topology, primitive intersection curves are calculated (selecting the intersection curve closest to the topological edge). The intersection of two adjacent intersection curves forms a CAD vertex, and the CAD edges are obtained by trimming the intersection curves with these vertices.
    - **Design Motivation**: Since the reconstructed mesh is watertight, the extracted topological representation is also watertight. Utilizing this topological guidance avoids incorrect intersection curve selection, ensuring the completeness of the final B-rep.

### Loss & Training

- The multi-view diffusion model is fine-tuned based on Wonder3D, separately fine-tuning two VAE decoders for normal map and primitive map generation.
- NeuS reconstruction removes the multi-view color input and texture reconstruction loss (as CAD models do not require textures).
- Primitive parameter extraction utilizes the RANSAC algorithm, and geometric optimization uses L-BFGS.
- The training set is curated from the ABC and DeepCAD datasets, selecting 30,000 seamless CAD models (29,000 for training and 1,000 for testing).

## Key Experimental Results

### Main Results

| Method | CD (↓) | NC (↑) | SEG(V) (↑) | SEG(P) (↑) |
|------|--------|--------|------------|------------|
| CRM | 3.97 | 64.4% | 40.2% | 49.3% |
| LRM | 4.26 | 63.6% | 38.4% | 46.8% |
| InstantMesh | 4.61 | 58.3% | 35.1% | 41.7% |
| SyncDreamer | 5.49 | 48.9% | 29.8% | 33.2% |
| **CADDreamer** | **1.27** | **92.6%** | **95.7%** | **97.9%** |

### B-rep Quality

| Method | HF (Dangling Face Ratio) (↓) | CD (↓) |
|------|-------------------|--------|
| CRM | 35.2% | 9.74 |
| LRM | 39.6% | 11.6 |
| InstantMesh | 43.6% | 13.1 |
| SyncDreamer | 58.5% | 15.4 |
| **CADDreamer** | **2.4%** | **1.36** |

### Key Findings

- CADDreamer outperforms all baselines by a large margin: its Chamfer Distance is 68% lower than the best baseline, and its Normal Consistency is 28 percentage points higher.
- The primitive segmentation accuracy reaches 97.9% (based on the number of primitives), indicating that the semantic-enhanced diffusion model can accurately understand CAD structures.
- The dangling face ratio is only 2.4%, significantly lower than the 35-58% of other methods, demonstrating the effectiveness of the geometric optimization and topology-preserving extraction.
- It successfully reconstructs high-quality CAD models on real-world images, demonstrating robust generalization capability.

## Highlights & Insights

- **Encoding semantics into the color space** is a highly clever design: it reuses the image generation capability of pre-trained diffusion models to understand high-level CAD structures, avoiding the need to train a semantic branch from scratch.
- **The complete pipeline from segmented mesh to watertight B-rep** solves a long-standing challenge: how to extract precise CAD models from highly noisy generated meshes.
- The method supports 5 primitive types, making it more general than sketch-and-extrude methods (which are limited only to planes and cylinders).
- The concept of "stitching" in geometric optimization can be generalized to other tasks requiring the recovery of geometric relationships.

## Limitations & Future Work

- Due to the inherent lack of information in a single-view input, some primitives may remain undetected under extreme occlusion or complex perspectives.
- The number and resolution of images limit the detection of highly fine-grained geometric features.
- Freeform surfaces (such as NURBS) are not supported; representation is limited to five basic geometric primitives.
- Topological extraction relies on the watertightness of the reconstructed mesh, which may fail when the mesh quality is poor.

## Related Work & Insights

- Multi-view diffusion models like Wonder3D and SyncDreamer provide foundational cross-view consistent generation capabilities.
- Point2CAD delivers a pipeline from point clouds to B-rep, but requires precise inputs.
- The approach of "RANSAC primitive extraction + geometric optimization" can be transferred to other reverse engineering tasks.
- This work demonstrates the immense potential of diffusion models in structured 3D generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First method to directly generate multi-primitive B-rep CAD models from a single-view image.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes both synthetic and real-world experiments, but lacks comparisons with more CAD reconstruction methods.
- **Writing Quality**: ⭐⭐⭐⭐ Clear pipeline and rich illustrations.
- **Value**: ⭐⭐⭐⭐⭐ Has direct application value for manufacturing and product design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] StdGEN: Semantic-Decomposed 3D Character Generation from Single Images](stdgen_semantic-decomposed_3d_character_generation_from_single_images.md)
- [\[CVPR 2025\] MEt3R: Measuring Multi-View Consistency in Generated Images](met3r_measuring_multi-view_consistency_in_generated_images.md)
- [\[CVPR 2026\] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](../../CVPR2026/3d_vision/brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](../../ICCV2025/3d_vision/ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[CVPR 2025\] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)

</div>

<!-- RELATED:END -->
