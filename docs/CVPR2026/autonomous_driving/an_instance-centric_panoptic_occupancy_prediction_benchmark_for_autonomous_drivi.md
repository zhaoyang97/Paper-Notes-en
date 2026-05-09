---
title: >-
  [Paper Note] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving
description: >-
  [CVPR 2026][Autonomous Driving][Panoptic Occupancy Prediction] This paper proposes ADMesh (a library of 15K+ high-quality 3D models) and CarlaOcc (a panoptic occupancy dataset with 100K frames at 0.05m resolution), providing for the first time instance-level annotations and physically consistent ground truth for 3D panoptic occupancy prediction in autonomous driving, along with occupancy quality evaluation metrics and a systematic benchmark.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Panoptic Occupancy Prediction
  - 3D Mesh Library
  - CARLA Simulation
  - Instance-Level Annotation
  - Occupancy Dataset Quality
date: 2026-05-08
content_hash: 9e192239e9377a7b
---

# An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2603.27238](https://arxiv.org/abs/2603.27238)
**Code**: [https://mias.group/CarlaOcc](https://mias.group/CarlaOcc)
**Area**: Autonomous Driving
**Keywords**: Panoptic Occupancy Prediction, 3D Mesh Library, CARLA Simulation, Instance-Level Annotation, Occupancy Dataset Quality

## TL;DR
This paper proposes ADMesh (a library of 15K+ high-quality 3D models) and CarlaOcc (a panoptic occupancy dataset with 100K frames at 0.05m resolution), providing for the first time instance-level annotations and physically consistent ground truth for 3D panoptic occupancy prediction in autonomous driving, along with occupancy quality evaluation metrics and a systematic benchmark.

## Background & Motivation

**Background**: 3D occupancy prediction is evolving from purely semantic occupancy toward fine-grained panoptic occupancy (joint semantic and instance prediction). Methods such as SparseOcc and PanoOcc have been proposed, but remain constrained by dataset quality.

**Limitations of Prior Work**: (1) Existing datasets lack instance-level annotations—SparseOcc/PaSCo generate pseudo-panoptic labels via heuristics (3D box grouping/clustering), introducing boundary artifacts and instance overlaps; (2) Existing ground truth relies on LiDAR point cloud aggregation and voxelization, resulting in coarse resolution (0.2–0.5 m), incomplete geometry (only sensor-visible surfaces), and physical inconsistencies (holes and fractures); (3) No unified high-quality 3D model library exists—current resources are fragmented and platform-dependent.

**Key Challenge**: Panoptic occupancy prediction requires precise instance-level geometric annotations, yet the generation pipelines of existing datasets (LiDAR aggregation → voxelization) are fundamentally incapable of providing physically consistent and complete ground truth.

**Key Insight**: Starting from 3D meshes rather than point clouds—meshes encode complete geometry and can be voxelized at arbitrary resolution.

**Core Idea**: Build a unified 3D model library (ADMesh) → reconstruct complete scene meshes via CARLA simulation → apply topology-aware voxelization to generate physically consistent panoptic occupancy labels.

## Method

### Overall Architecture
Four major components: (1) ADMesh 3D model library construction; (2) CarlaOcc dataset generation (scene mesh reconstruction → voxelization → sensor artifact correction); (3) occupancy quality evaluation metrics; (4) systematic benchmarking.

### Key Designs

1. **ADMesh: Unified 3D Model Library**:

    - Function: Integrates 15K+ high-quality 3D models from four sources—CARLA, BuildingNet, MeshFleet, and ShapeNet.
    - Mechanism: Develops an automated mesh export toolchain—traverses CARLA scenes → extracts component-level mesh assets → queries component hierarchy and transforms via the UE editor interface → integrates CARLA's native semantic annotation system → hierarchically assembles complete object-level meshes. A unified data organization framework ensures consistency in naming, coordinate systems, and semantic hierarchy.
    - Design Motivation: Simulation platform assets are fragmented, non-standardized, and platform-bound; a unified framework is needed to support large-scale dataset construction.

2. **Mesh-Based Scene Reconstruction**:

    - Function: Directly reconstructs the panoptic scene mesh for each frame from 3D meshes (rather than LiDAR point aggregation).
    - Mechanism:
        - **Static background**: Selects background meshes $\mathcal{S}_{bg}$ intersecting the occupancy region.
        - **Rigid foreground**: Matches models $\mathcal{S}_{fg}^r$ from ADMesh using a lookup table (LUT).
        - **Non-rigid foreground (pedestrians)**: A skeletal motion analyzer preprocesses walking animations into $D$ discrete phase template meshes; at runtime, the current skeletal state is matched to the nearest phase via geodesic distance: $d_k = \arg\min_d \mathcal{G}(\delta_k, \delta_d)$.
        - Merging: $\mathcal{M}^{pano} = \mathcal{S}_{bg} \cup \mathcal{S}_{fg}^r \cup \mathcal{S}_{fg}^n$
    - Design Motivation: Meshes preserve complete geometric information, avoiding the incompleteness caused by LiDAR sparse sampling and occlusion.

3. **Topology-Aware Mesh Displacement Strategy**:

    - Function: Generates overlap-free panoptic occupancy labels from the panoptic scene mesh.
    - Mechanism: Merges stuff meshes by semantic category (eliminating redundant boundaries), then sorts instances by world height and integrates them via layer-by-layer voxelization from bottom to top—ensuring lower structures do not overwrite higher ones.
    - Design Motivation: Independently voxelizing each mesh is computationally expensive and produces label conflicts.

4. **Instance-Guided Sensor Artifact Correction**:

    - Function: Corrects depth and semantic artifacts caused by transparent/semi-transparent objects in CARLA rendering.
    - Mechanism: Constructs a scene mesh containing only transparent objects → generates accurate depth via ray casting → repairs the original depth map by taking the per-point minimum.
    - Design Motivation: CARLA incorrectly renders depth and semantics of transparent objects, displaying the opaque objects behind them instead.

### Occupancy Quality Evaluation Metrics
- **Spatial Continuity Score ($s_{sc}$)**: Quantifies the spatial continuity of occupied voxels within the same semantic category (higher is better).
- **Temporal Consistency Score ($s_{tc}$)**: Quantifies the temporal stability of occupancy labels across adjacent frames.

## Key Experimental Results

### Dataset Quality Comparison

| Dataset | Synthetic | Resolution (m) | Instance Ann. | $s_{sc}$↑ | $s_{tc}$↑ |
|--------|------|-----------|----------|-----------|-----------|
| SemanticKITTI | No | 0.2 | No | 0.353 | 0.023 |
| Occ3D-nuScenes | No | 0.4 | No | 0.721 | 0.431 |
| SurroundOcc | No | 0.5 | No | 0.878 | 0.589 |
| CarlaSC | Yes | 0.4 | No | 0.887 | 0.775 |
| **CarlaOcc (Ours)** | Yes | **0.05** | **Yes** | **0.996** | **0.873** |

### Benchmark Model Evaluation (Semantic Occupancy mIoU)

| Model | Key Findings |
|------|----------|
| Multiple SOTA methods | Models trained on CarlaOcc benefit from finer-grained ground truth |
| Panoptic occupancy task | For the first time, evaluation on genuine instance-level annotations is possible |

### Key Findings
- CarlaOcc achieves substantially higher spatial continuity (0.996) and temporal consistency (0.873) than all existing datasets.
- The 0.05 m resolution is 4× finer than the finest existing dataset (SemanticKITTI at 0.2 m).
- The instance-guided artifact correction pipeline effectively rectifies rendering errors for transparent objects.
- The mesh-based generation pipeline entirely avoids information loss inherent in LiDAR aggregation.

## Highlights & Insights
- **Paradigm Shift from Point Clouds to Meshes**: Meshes encode complete geometric information, fundamentally resolving the resolution and completeness limitations of LiDAR aggregation pipelines. This has significant implications for the methodology of synthetic dataset construction.
- **Skeletal Motion Analyzer**: Provides an elegant solution for accurate reconstruction of non-rigid objects (pedestrians)—preprocessing animation phases combined with runtime geodesic matching.
- **Quality Evaluation Metrics**: For the first time, quantitative standards for spatial continuity and temporal consistency are defined to assess occupancy dataset quality.

## Limitations & Future Work
- Sim-to-real gap of synthetic data—whether models trained on CarlaOcc transfer to real driving scenarios remains an open question.
- ADMesh assets are primarily sourced from CARLA; asset diversity remains constrained by the simulation platform.
- The enormous voxel count at 0.05 m resolution imposes significant memory and computational overhead for model training and inference.
- Pedestrian animations cover only walking cycles; more complex human motions (e.g., bending, crouching) require future extension.

## Related Work & Insights
- **vs. Occ3D/SurroundOcc**: Real-world datasets based on LiDAR aggregation with incomplete geometry. CarlaOcc generates labels from meshes and is physically consistent, but incurs a sim-to-real gap.
- **vs. CarlaSC**: Also a CARLA-based synthetic dataset, but lacks instance annotations and has coarser resolution (0.4 m vs. 0.05 m).
- **vs. SparseOcc/PanoOcc**: Model-level algorithmic innovations; this paper provides dataset-level infrastructure.

## Rating
- Novelty: ⭐⭐⭐⭐ First instance-level panoptic occupancy benchmark; ADMesh and the mesh reconstruction pipeline are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive dataset quality evaluation, though downstream model benchmarking could be enriched.
- Writing Quality: ⭐⭐⭐⭐⭐ Pipeline description is clear and complete; dataset statistics are detailed.
- Value: ⭐⭐⭐⭐⭐ Provides foundational infrastructure for 3D panoptic occupancy research and advances the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2occ_resilient_3d_semantic_occupancy_prediction_f.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](../../ICCV2025/autonomous_driving/uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)
- [\[CVPR 2026\] DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving](dlwm_dual_latent_world_models_enable_holistic_gaussian-centric_pre-training_in_a.md)
- [\[CVPR 2026\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_openvocabulary_occupancy_predi.md)
- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth-_and_region-guided_3d_occupancy_from_surround-view_cameras_for_auton.md)

</div>

<!-- RELATED:END -->
