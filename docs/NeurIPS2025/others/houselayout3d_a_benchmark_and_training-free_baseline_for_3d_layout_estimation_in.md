---
title: >-
  [Paper Note] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild
description: >-
  [NeurIPS 2025][3D layout estimation] This paper introduces HouseLayout3D—the first real-world 3D layout estimation benchmark targeting large-scale multi-floor buildings—and MultiFloor3D…
tags:
  - "NeurIPS 2025"
  - "3D layout estimation"
  - "multi-floor buildings"
  - "benchmark"
  - "training-free"
  - "scene graph"
date: 2026-05-08
content_hash: 5f6e3cd9dec25e4c
---

# HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild

**Conference**: NeurIPS 2025
**arXiv**: [2512.02450](https://arxiv.org/abs/2512.02450)  
**Code**: [https://houselayout3d.github.io](https://houselayout3d.github.io)  
**Area**: LLM Evaluation
**Keywords**: 3D layout estimation, multi-floor buildings, benchmark, training-free, scene graph

## TL;DR

This paper introduces HouseLayout3D—the first real-world 3D layout estimation benchmark targeting large-scale multi-floor buildings—and MultiFloor3D, a training-free baseline that combines modern 3D reconstruction and segmentation models to surpass existing deep learning methods on multi-floor building layout estimation.

## Background & Motivation

Existing 3D layout estimation models are primarily trained on synthetic datasets containing only simple single-room or single-floor environments, giving rise to two critical issues:

**Inability to handle multi-floor buildings**: Current methods require pre-segmenting a scene into individual floors before processing each separately, discarding the global spatial context needed to understand cross-floor structures such as staircases.

**Lack of diversity in training data**: Although synthetic data can be generated at scale automatically, it fails to capture the complexity of real large-scale buildings—multiple rooms, multiple floors, non-Manhattan geometry, and partially open spaces.

Existing datasets (SceneCAD, ASE, Stru3D, etc.) are limited along multiple dimensions: they are either not real-world data, do not support multiple floors, or lack door and window annotations. HouseLayout3D is the first benchmark to satisfy all of these requirements simultaneously.

## Method

### Overall Architecture

MultiFloor3D is a four-stage, training-free pipeline:

1. **Mesh Reconstruction**: Reconstructs a 3D mesh from RGB images.
2. **Layout Skeleton Extraction**: Extracts structural geometric elements from the mesh.
3. **Layout Prototype Fitting**: Repairs defects in the skeleton through optimization.
4. **Scene Graph Generation**: Converts the prototype into a final 3D layout.

### Key Designs

1. **Mesh Reconstruction (Stage 1)**: DN-Splatter (based on 3D Gaussian Splatting) is used to obtain a triangular mesh and depth maps from pose-free 2D images. Camera poses are first estimated with COLMAP; a Metric3D depth model then supervises 3DGS training, and Poisson surface reconstruction generates the final mesh.

2. **Layout Skeleton Extraction (Stage 2)**: The mesh is partitioned into four semantic categories—structural components (walls, ceilings, floors, large furniture), geometrically imprecise surfaces (windows, mirrors), objects (small furniture), and staircases. OneFormer performs semantic segmentation on the input images; labels are transferred to the 3D mesh via back-projection and refined through superpoint clustering with majority voting. Only structural components are retained as the skeleton.

3. **Layout Prototype Fitting (Stage 3)**: This is the core innovation. To address artifacts in the skeleton (holes and unobserved regions), a set of 3D polygons is optimized via gradient descent:

    - **$\mathcal{L}_{\text{geo}}$ (Geometric Loss)**: Comprises $\mathcal{L}_{\text{prox}}$ (minimizing the distance from skeleton vertices to the nearest polygon) and $\mathcal{L}_{\text{empty}}$ (preventing polygons from occluding known empty space, detected via camera-ray and depth intersection).
    - **$\mathcal{L}_{\text{connect}}$ (Connectivity Loss)**: Encourages polygons to share boundaries, reducing small gaps.
    - **$\mathcal{L}_{\text{simple}}$ (Simplification Loss)**: Penalizes the length of non-shared edges, encouraging unnecessary edges to shrink until eliminated.
    - **Vertex Merging**: Periodically simplifies polygons by merging nearby vertices, applying the RDP algorithm to simplify boundaries, and merging nearby polygons with similar normals.
    - **Floor/Wall Hole Filling**: Projects object meshes onto the nearest floor plane to complete floor holes; extends wall polygons to ceilings/floors to fill wall holes.

4. **Scene Graph Generation (Stage 4)**:
    - Identifies building floors via height-based clustering of floor polygons.
    - Creates a 2D floor plan per floor by merging floor and ceiling polygons.
    - Applies Hov-SG's room segmentation algorithm to partition each floor into rooms, generating a scene graph with rooms as nodes and doors/openings as edges.
    - Detects staircases and adds cross-floor connection edges to the scene graph.
    - **Room Extrusion**: Triangulates each floor plan using 2D Constrained Delaunay Triangulation, casts upward rays to assign ceilings, and extrudes each floor triangle to its assigned ceiling plane to generate enclosed 3D rooms.

### Loss & Training

MultiFloor3D requires no training. Optimization occurs in Stage 3 during prototype fitting, where polygon vertex positions are updated via gradient descent:

$$\mathcal{L} = \mathcal{L}_{\text{geom}} + \mathcal{L}_{\text{connect}} + \mathcal{L}_{\text{simple}}$$

where $\mathcal{L}_{\text{geom}} = \mathcal{L}_{\text{prox}} + \mathcal{L}_{\text{empty}}$. Throughout optimization, vertices within each polygon are constrained to remain coplanar, and polygons are permitted to share vertices.

## Key Experimental Results

### HouseLayout3D Dataset Statistics

- 16 buildings, 33 individual floors, 317 rooms
- Over 26,000 frames of RGB-D data
- 292 doors, 379 windows, and 34 staircases annotated
- 1–5 floors and 4–40 rooms per building
- 4–10 hours of annotation effort per building

### Main Results (HouseLayout3D)

| Method | Structures F1@0.5 | Doors F1@0.5 | Windows F1@0.5 | Stairs F1@0.5 | Depth Δ₅ | Depth Δ₁₀ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| RoomFormer (per floor) | 0.24 | 0.23 | 0.07 | – | 24.9 | 32.9 |
| RoomFormer (per room) | 0.18 | 0.18 | 0.08 | – | 37.3 | 44.8 |
| SceneScript (per floor) | 0.28 | 0.23 | 0.16 | – | 22.5 | 33.8 |
| SceneScript (per room) | 0.23 | 0.31 | 0.11 | – | 23.5 | 32.9 |
| **MultiFloor3D** | **0.40** | **0.55** | **0.43** | **0.42** | **61.1** | **76.3** |

MultiFloor3D substantially outperforms all baselines across every metric without relying on ground-truth floor or room segmentation.

### ScanNet++ Results

| Method | #Vertices | Depth Δ₅ | Depth Δ₁₀ |
|------|:-:|:-:|:-:|
| DN-Splatter Mesh | 354k | 84.1 | 92.6 |
| RoomFormer | **32.5** | 36.8 | 48.9 |
| SceneScript | 41.2 | 55.1 | 68.5 |
| **MultiFloor3D** | 83.1 | **67.8** | **84.7** |

### Ablation Study

| Configuration | Avg F1 | #Vertices | Notes |
|------|:-:|:-:|------|
| Input Mesh + QSlim | 0.109 | 2000 | Direct simplification of raw mesh |
| Layout Skeleton + QSlim | 0.223 | 2000 | Skeleton extraction without fitting |
| Layout Prototype | 0.373 | 2553 | Fitting without scene graph |
| **MultiFloor3D** | **0.381** | **1957** | Full pipeline |
| w/o prototype fitting | 0.214 | 2270 | Stage 3 removed |
| w/o room segmentation | 0.359 | 2442 | Room segmentation removed |

### Key Findings

1. **Every stage contributes**: Each step yields meaningful improvement—from skeleton (0.223) to prototype (0.373) to the full pipeline (0.381).
2. **Prototype fitting is the most critical component**: Removing it causes F1 to drop from 0.381 to 0.214, the largest single degradation.
3. **Training-free approach surpasses trained methods**: MultiFloor3D uses no training data yet outperforms RoomFormer and SceneScript, both trained on approximately 100,000 synthetic samples.
4. **Fundamental limitation of baseline methods**: RoomFormer and SceneScript predict only rectangular primitives and cannot represent complex shapes such as slanted ceilings.
5. **Large margins on door and window detection**: Doors F1 improves from 0.23/0.31 to 0.55; Windows F1 improves from 0.07/0.16 to 0.43.
6. **Only method capable of staircase prediction**: All other baselines entirely lack staircase detection support.

## Highlights & Insights

- **First multi-floor building 3D layout benchmark**, filling a significant gap in the field.
- The **counterintuitive result that a training-free method outperforms trained methods** reveals severe generalization deficiencies in current end-to-end approaches.
- **Modular pipeline design** allows each stage to be independently replaced or upgraded.
- **Scene graph representation** naturally supports downstream applications such as navigation (the paper demonstrates an indoor navigation demo combining the proposed method with an LLM).
- The three geometric loss functions are elegantly designed: $\mathcal{L}_{\text{prox}}$ ensures fidelity, $\mathcal{L}_{\text{empty}}$ avoids occlusion of free space, $\mathcal{L}_{\text{connect}}$ enforces connectivity, and $\mathcal{L}_{\text{simple}}$ promotes compactness.

## Limitations & Future Work

1. **Long runtime**: Each HouseLayout3D scene requires 1–2 hours on an NVIDIA RTX 4090, whereas SceneScript/RoomFormer complete inference in 1–2 minutes.
2. **Interference from outdoor elements**: Outdoor content perceived through large windows may introduce artifacts.
3. **Dependency on multiple pretrained models**: The pipeline relies on COLMAP, Metric3D, OneFormer, and DN-Splatter, causing errors to accumulate across stages.
4. **Limited dataset scale**: With 16 buildings, the benchmark is small compared to training datasets of 100,000+ samples.
5. **No real-time inference**: The sequential pipeline is unsuitable for real-time applications.
6. **Sensitivity to depth estimation quality**: Inaccurate depth estimates at windows and reflective surfaces directly degrade downstream processing.

## Related Work & Insights

- Traditional Manhattan-world assumption methods (Scan2Bim, DuLaNet) constrain the diversity of scenes that can be handled.
- RoomFormer applies Transformers to 2D floor plan prediction; SceneScript introduces a structured scene language; yet both are fundamentally limited by synthetic training data.
- MultiFloor3D directly adopts Hov-SG's room segmentation algorithm.
- DN-Splatter's 3DGS + depth-supervised reconstruction strategy provides high-quality geometric input for the subsequent pipeline.
- The central insight of this work is that **when data is scarce, composing strong pretrained models may be more effective than end-to-end training**.

## Rating

- Novelty: ⭐⭐⭐⭐ (First multi-floor building benchmark + a new paradigm of training-free pipelines)
- Experimental Thoroughness: ⭐⭐⭐⭐ (HouseLayout3D + ScanNet++ + comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich illustrations)
- Value: ⭐⭐⭐⭐ (Lasting benchmark contribution; exposes fundamental limitations of existing methods)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation](../../ICCV2025/others/stroke2sketch_harnessing_stroke_attributes_for_training-free_sketch_generation.md)
- [\[NeurIPS 2025\] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)
- [\[NeurIPS 2025\] Active Measurement: Efficient Estimation at Scale](active_measurement_efficient_estimation_at_scale.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)

</div>

<!-- RELATED:END -->
