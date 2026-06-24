---
title: >-
  [Paper Note] A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering
description: >-
  [CVPR 2025][3D Vision][CAD Reverse Engineering] The A2Z dataset is constructed containing over 1 million complex CAD models and more than 10 million multimodal annotations (high-resolution 3D scans, freehand 3D sketches, text descriptions, and BRep topology labels), representing the largest dataset for CAD reverse engineering to date. Based on this, foundation models for BRep boundary and corner detection are trained.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "CAD Reverse Engineering"
  - "Boundary Representation (BRep)"
  - "Multimodal Annotation"
  - "Geometric Deep Learning"
  - "3D Scanning"
  - "Large-Scale Dataset"
date: 2026-05-08
content_hash: 044e766e0370abb5
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering

**Conference**: CVPR 2025  
**arXiv**: [2603.12605](https://arxiv.org/abs/2603.12605)  
**Code**: To be confirmed  
**Area**: 3D Vision  
**Keywords**: CAD Reverse Engineering, Boundary Representation (BRep), Multimodal Annotation, Geometric Deep Learning, 3D Scanning, Large-Scale Dataset

## TL;DR

The A2Z dataset is constructed containing over 1 million complex CAD models and more than 10 million multimodal annotations (high-resolution 3D scans, freehand 3D sketches, text descriptions, and BRep topology labels), representing the largest dataset for CAD reverse engineering to date. Based on this, foundation models for BRep boundary and corner detection are trained.

## Background & Motivation

### Background
- In industrial product design, Boundary Representation (BRep) is the de facto standard storage format for CAD models.
- BRep consists of hierarchical geometric primitives: solid shell $\to$ faces (planes, cylinders, etc.) $\to$ closed loops $\to$ edges (lines, circles, B-splines) $\to$ corners.
- Reconstructing CAD models from 3D scans through reverse engineering is a core demand in the industry.

### Limitations of Prior Work
- **Design-history methods are limited by DeepCAD’s 170K simple models**: The DeepCAD dataset primarily contains simple cuboid shapes, with extremely low geometric and topological diversity.
- **Small data scale for BRep methods**: Fusion-360 contains only ~8K samples, and ABCParts has only 32K samples.
- **The ABC dataset lacks companion annotations**: Despite having 1 million CAD models, it lacks 3D scans, text descriptions, BRep labels, etc.
- **Input point clouds used in existing methods come from low-fidelity random sampling** rather than realistic scans.
- This creates a **deadlock**: methods require data progress, and data requires methods to generate.

### Key Challenge
An extreme scarcity of large-scale, multimodal, and high-quality BRep annotated data severely hinders progress in research directions such as CAD reverse engineering, BRep reconstruction, sketch-to-BRep generation, and text-to-BRep modeling.

### Key Insight
To directly target the 1 million complex CAD models in the ABC dataset, systematically generating comprehensive multimodal annotations spanning 3D scans, sketches, text, and BRep topology to fill the data gap.

## Method

### Overall Architecture
The A2Z dataset includes four major categories of annotations, totaling nearly 5TB in storage:
1. High-resolution 3D scan meshes (with BRep labels)
2. 3D freehand sketches of multiple skill levels
3. Multi-view images, text descriptions, and tags
4. An additional 25K electronics enclosure CAD models

### Key Design 1: High-Resolution Scan Simulation (Sec. 3.1)
Transform ABC's low-polygon meshes into high-resolution meshes that simulate real 3D scans, processed through four steps:
- **Step-I Mesh Upsampling**: Two rounds of midpoint subdivision, yielding a high-resolution mesh with ~150K vertices and ~380K triangular faces.
- **Step-II Tangential Shrinkage near Small Holes**: Identifies small loops to simulate occlusions and missing data caused by sensor frustum limitations.
- **Step-III Surface Roughness**: Displaces vertices along the normal direction using a multi-octave Perlin noise field to inject millimeter-level errors.
- **Step-IV Dents and Bumps**: Randomly adds local Gaussian dents and sinusoidal bumps on planar BRep surfaces to simulate manufacturing defects.

### Key Design 2: Proximity-Aware Smooth Annotation (Sec. 3.2)
- Instead of traditional hard nearest-neighbor labeling rules, a **multi-scale, BRep edge-length-aware, SPH-weighted membership assignment strategy** is proposed.
- SPH kernels are used to aggregate weights across $K$ different scales to select the probabilistic soft label $\pi(x)$.
- High annotation coverage: boundary ID 99.37%, boundary type 97.67%, loops 99.99%, and face ID 99.93%.

### Key Design 3: 3D Sketch Generation with Multi-Skill Levels (Sec. 3.3)
- Uses a single skill parameter $\kappa\in\{1,...,5\}$ to control sketch quality, simulating different levels from amateur to professional artists.
- Designs distinct displacement fields for straight lines, circular arcs, and general curves respectively.
- Straight lines: mean-reverting random walk + endpoint tapering + bowing.
- Circular arcs: radial harmonic perturbation + intentional gaps + tangential jittering.
- Generates a total of approximately 5 million 3D sketches with BRep annotations.

### Key Design 4: VLM-Driven Text Annotation (Sec. 3.4)
- First to use a **VLM-critic system** (Qwen3-14B + InternVL-26B) to automatically generate high-quality descriptions.
- Takes 12 multi-view transparent rendered BRep images ($4\times3$ grid) as input, producing descriptions of $\le200$ words and $\le20$ tags.
- Leverages ImageNet/WordNet to build a hierarchical category structure (up to 4 levels deep) containing 6 major feature classes.

### Foundation Model Architecture (Sec. 4)
- Based on a DGCNN backbone + a two-head structure (boundary detection + corner detection).
- Point-wise binary classification (boundary/non-boundary) using focal loss.
- Trained on a subset of 300K CAD models using two Nvidia H100 GPUs for 20 epochs (4 days).

## Key Experimental Results

### Dataset Quality Evaluation

| Evaluation Method | Face IDs | Face Type | Sketch L2 | Sketch L3 | Sketch L4 | Sketch L5 |
|---------|---------|---------|----------|----------|----------|----------|
| Gemini | 8.43 | 7.88 | 8.21 | 7.87 | 7.97 | 8.31 |
| GPT-5 | 8.79 | 8.71 | 7.47 | 7.71 | 8.11 | 8.43 |
| Human | 8.37 | 8.05 | 8.36 | 9.12 | 9.03 | 9.61 |

- High-quality VLM annotations have a mean MLTD lexical diversity of 70.52 vs. 59.32 for low-quality ones ($+18.9\%$).

### Foundation Model Comparison (A2Z Test Set)

| Method | Boundary Recall | Boundary Precision | Corner Recall | Corner Precision |
|------|----------|-------------|----------|-------------|
| **Ours** | **0.978** | **0.901** | **0.732** | **0.891** |
| BRepDetNet* | 0.903 | 0.781 | 0.454 | 0.561 |
| ComplexGen* | 0.551 | 0.750 | 0.297 | 0.592 |
| PieNet* | 0.832 | 0.885 | - | - |

- Achieves zero-shot transfer on the unseen CC3D dataset: Boundary Recall 0.961, Precision 0.854.
- Baseline methods perform significantly better when retrained on A2Z annotations (with 10-30% recall gains), demonstrating the superior quality of the annotations.
- This method experiences the smallest performance drop when moving from seen chunks to unseen chunks.

## Highlights & Insights

1. **Unprecedented Scale**: Over 1 million CAD models and more than 10 million annotations, representing the largest known CAD reverse engineering dataset.
2. **Multimodal Completeness**: The only large-scale dataset providing BRep labels, 3D scans, sketches, and text descriptions simultaneously.
3. **Realistic Scan Simulation**: A four-step geometric transformation pipeline systematically simulates realistic scanning noise, including occlusions, roughness, and manufacturing defects.
4. The **VLM-critic system** is innovatively applied to CAD text annotation, offering high cost-effectiveness and guaranteed quality.
5. **Astonishing Annotation Coverage**: Both boundary and face annotation coverage exceed 97%, achieved through a rigorous multi-threshold SPH-weighted strategy.
6. **Strong Zero-Shot Transfer**: Achieves high performance on CC3D without training, proving the generalization value of the dataset.

## Limitations & Future Work

1. **No Design History**: A2Z does not include parameterized CAD operation sequences, rendering it inapplicable to methods that require design history.
2. **Simulated Rather Than Real 3D Scans**: Although various noises are introduced, a distribution discrepancy with real-world scanner data still exists.
3. **Room for Improvement in Corner Detection**: Even the best method achieves a corner Recall of only 0.732, presenting a challenge due to the extreme sparsity of corners.
4. **Relatively Simple Foundation Model Architecture**: Employs a DGCNN backbone, leaving advanced architectures like Transformers unexplored.
5. **Text Annotation Dependencies**: The text annotations rely heavily on VLM quality, and description accuracy for complex mechanical parts remains limited.

## Related Work & Insights

- Compared to DeepCAD (170K simple models with design history) and Fusion-360 (8K samples), A2Z significantly outperforms them in scale and multimodality.
- BRep detection methods such as ComplexGen, BRepDetNet, and PIE-Net can all benefit from the high-quality annotations in A2Z.
- The paradigm of using VLMs for industrial data annotation is worth extending to other domains.
- It provides a solid data foundation for emerging directions such as sketch-to-BRep and text-to-BRep.

## Rating
- Novelty: ⭐⭐⭐⭐ (Large-scale multimodal CAD dataset is a substantial contribution; scanning simulation and annotation methods are innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-angle data quality evaluation + foundation model benchmarks, though downstream task validation is limited)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and fully detailed technical descriptions)
- Value: ⭐⭐⭐⭐⭐ (Fills the gap in CAD reverse engineering data; open-sourcing will provide significant driving force)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning](../../NeurIPS2025/3d_vision/cosmobench_a_multiscale_multiview_multitask_cosmology_benchmark_for_geometric_de.md)
- [\[CVPR 2025\] Pano360: Perspective to Panoramic Vision with Geometric Consistency](pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)
- [\[CVPR 2025\] CADDreamer: CAD Object Generation from Single-view Images](caddreamer_cad_object_generation_from_single-view_images.md)
- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](../../NeurIPS2025/3d_vision/copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[CVPR 2025\] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)

</div>

<!-- RELATED:END -->
