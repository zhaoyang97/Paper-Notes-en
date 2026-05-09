---
title: >-
  [Paper Note] A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering
description: >-
  [CVPR2026][3D Vision][BRep learning] This work constructs A2Z, a large-scale multimodal CAD dataset comprising 1M+ complex models with 10M+ annotations (high-resolution 3D scans, hand-drawn 3D sketches, text descriptions, and BRep topology labels), providing an unprecedented data foundation for Scan-to-BRep reverse engineering and multimodal BRep learning. Foundation models trained on A2Z substantially outperform existing methods on edge and junction detection.
tags:
  - CVPR2026
  - 3D Vision
  - BRep learning
  - CAD reverse engineering
  - multimodal annotation
  - 3D scanning
  - geometric deep learning
  - foundation model
date: 2026-05-08
content_hash: 310cd3bc6763cab8
---

# A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering

**Conference**: CVPR2026  
**arXiv**: [2603.12605](https://arxiv.org/abs/2603.12605)  
**Code**: To be confirmed  
**Area**: 3D Vision / CAD Reverse Engineering  
**Keywords**: BRep learning, CAD reverse engineering, multimodal annotation, 3D scanning, geometric deep learning, foundation model

## TL;DR

This work constructs A2Z, a large-scale multimodal CAD dataset comprising 1M+ complex models with 10M+ annotations (high-resolution 3D scans, hand-drawn 3D sketches, text descriptions, and BRep topology labels), providing an unprecedented data foundation for Scan-to-BRep reverse engineering and multimodal BRep learning. Foundation models trained on A2Z substantially outperform existing methods on edge and junction detection.

## Background & Motivation

**BRep is the core CAD representation**: In industrial product design, Boundary Representation (BRep) is the standard format for feature modeling and topology exploration, organized as a hierarchical structure of faces, co-edges, and junctions.

**Existing datasets are small and unimodal**: DeepCAD contains only 170K simple models requiring design history; Fusion-360 has only 8K samples; ABCParts only 32K. No existing dataset simultaneously provides BRep labels, 3D scans, 3D sketches, and text descriptions.

**Lack of high-quality 3D scan data**: Point clouds used by current methods are randomly sampled from clean BRep patches, lacking the noise, occlusion, and surface defects of real scans, which is far from practical application scenarios.

**Limited access to design history**: The design history of the ABC dataset requires proprietary access to OnShape, forcing history-based methods to rely on DeepCAD's simple cuboid models, creating a deadlock.

**Gap in multimodal BRep learning**: Emerging tasks such as freehand 3D sketch-to-BRep generation and text-to-BRep modeling cannot advance due to the absence of large-scale multimodal annotated data.

**Existing BRep parsing methods lack robustness**: Methods such as ParseNet, ComplexGen, and SPFN are limited by small-scale training data and low-quality annotations, performing poorly on complex CAD models.

## Method

### Overall Architecture

A2Z is an ultra-large-scale multimodal annotated dataset (~5TB) built on 1M+ complex CAD models from the ABC dataset, containing four types of annotations:
- **High-resolution 3D scans**: Simulated scan meshes with BRep topology labels
- **3D hand-drawn sketches**: 5M sketches simulating artists of varying skill levels
- **Text descriptions and labels**: VLM-generated per-model descriptions (≤200 words) and tags (≤20)
- **Additional electronic enclosure CAD**: 25K charging port/enclosure models created by professional designers

### Key Designs

**High-resolution scan generation (Sec 3.1)**: A four-step pipeline converts low-polygon meshes into simulated scans:
- **Step-I Mesh upsampling**: Two rounds of midpoint subdivision yield ~150K vertices / ~380K triangles, achieving high-precision scanner-level density
- **Step-II Tangential contraction near small holes**: Small loops in BRep are identified ($L_\ell/L_{\max} < \tau_h$), and neighboring vertices are contracted inward tangentially to simulate visibility loss due to sensor frustum constraints
- **Step-III Surface roughness**: Multi-octave Perlin noise fields perturb vertices along normals to inject millimeter-scale inaccuracies while preserving sharp edges
- **Step-IV Dents and bumps**: Seed points are randomly placed on planar BRep faces, with Gaussian attenuation and sinusoidal bumps simulating machining defects

**Neighbor-aware smooth annotation (Sec 3.2)**: Traditional hard nearest-neighbor assignment is replaced by probabilistic soft labels weighted via multi-scale SPH:

$$p_i(\boldsymbol{x}) = \frac{\omega_i(\boldsymbol{x})}{\sum_{j \in \mathcal{N}(\boldsymbol{x})} \omega_j(\boldsymbol{x})}, \quad \pi(\boldsymbol{x}) = \arg\max_{i} p_i(\boldsymbol{x})$$

Each vertex stores parent edge ID, loop ID, paired loop ID, incident faces, and curve feature vectors. Annotation coverage exceeds 99%.

**3D hand-drawn sketch generation (Sec 3.3)**: Strokes are simulated using a single skill parameter $\kappa \in \{1,...,5\}$ to model artists of different proficiency levels:
- Line segments undergo mean-reverting random walks, bow-arc bending, and endpoint tapering
- Circular arcs use polar harmonic representations with low-frequency oscillations and tapered high-frequency harmonics
- General curves (ellipses, B-splines, etc.) receive multi-window bow-arcs and mean-reverting perturbations applied over arc-length segments

**VLM jury-based text annotation (Sec 3.4)**: A dual-model jury system using Qwen3-14B and InternVL-26B processes 12 multi-view BRep renderings (4×3 grid) to generate descriptions and tags, organized into a 6-category × 4-level tree taxonomy following ImageNet/WordNet conventions.

### Foundation Model & Loss & Training

A DGCNN-based point cloud encoder is equipped with two classification heads for edge detection and junction detection (binary classification), trained with Focal Loss to address severe class imbalance (edges are sparse; junctions are even sparser). The model is trained on 300K CAD models for 20 epochs on 2×H100 GPUs (4 days).

## Key Experimental Results

### Main Results: Edge and Junction Detection

| Model | Edge Recall | Edge Precision | Junction Recall | Junction Precision |
|------|-----------|---------------|-----------|---------------|
| **A2Z (Ours)** | **0.978** | **0.901** | **0.732** | **0.891** |
| BRepDetNet* | 0.903 | 0.781 | 0.454 | 0.561 |
| ComplexGen* | 0.551 | 0.750 | 0.297 | 0.592 |
| PieNet* | 0.832 | 0.885 | — | — |

> Results on A2Z Seen Chunks; * denotes retrained on A2Z. On Unseen Chunks, Ours achieves Edge Recall of 0.971, demonstrating significantly stronger generalization than baselines.

### Zero-Shot Generalization (CC3D Dataset, Never Seen During Training)

| Model | Edge Recall | Edge Precision | Junction Recall | Junction Precision |
|------|-----------|---------------|-----------|---------------|
| **A2Z (Ours)** | **0.961** | **0.854** | **0.633** | **0.810** |
| BRepDetNet* | 0.763 | 0.807 | 0.137 | 0.417 |
| ComplexGen* | 0.427 | 0.743 | 0.062 | 0.437 |

### Ablation Study

- **Impact of annotation quality**: Baseline methods retrained on A2Z improve by 10%–30% on edge detection and 4%–33% on junction detection, directly demonstrating the value of annotation quality
- **Annotation coverage**: Edge ID 99.37%, edge type 97.67%, loop 99.99%, face ID 99.93%
- **Text quality**: The VLM jury achieves an MLTD score of 70.52 vs. 59.32 for a single small model (+18.9%), with substantially improved bigram diversity
- **Human evaluation**: 10 annotators rated scan annotations 8.37/10 (face ID) and Level-5 sketches 9.61/10

### Key Findings

1. Junction detection is the most challenging task (junctions are extremely sparse relative to edges); the proposed model achieves the largest margin here (Recall 0.732 vs. runner-up 0.454)
2. Performance degradation from Seen to Unseen chunks is far smaller for the proposed method than for baselines, demonstrating the generalization benefit of large-scale high-quality data
3. PieNet consistently fails to train on junction detection, exposing fundamental architectural limitations

## Highlights & Insights

- **Largest multimodal CAD dataset to date**: 10M+ annotations across 1M+ complex CAD models, one to two orders of magnitude larger than existing datasets
- **Elegant four-step simulated scan pipeline**: Tangential contraction, Perlin noise, and Gaussian dents precisely simulate the physical characteristics of real scanners
- **SPH-weighted soft-label annotation**: Multi-scale smoothed particle hydrodynamics weights replace hard nearest-neighbor assignment, achieving 99%+ annotation coverage
- **Parameterized sketch skill levels**: A single parameter $\kappa$ controls five artist proficiency levels, generating 5M diverse sketches
- **Strong zero-shot generalization**: Achieves a large performance margin on the never-seen CC3D dataset

## Limitations & Future Work

- The dataset does not include design history, and thus cannot directly support CAD reconstruction methods based on construction sequences
- A gap remains between simulated scans and real physical scans; validation on large-scale real scan data has not been performed
- The foundation model only addresses edge/junction detection and has not been extended to subsequent parametric surface fitting and full BRep reconstruction
- Text annotations are automatically generated by VLMs and may contain hallucinations and inaccuracies
- The electronic enclosure subset contains only 25K models with limited category diversity
- The ~5TB dataset size poses a high barrier to adoption for small and mid-sized research teams

## Related Work & Insights

- **BRep learning methods**: PIE-Net, BRepDetNet, and ComplexGen perform edge/junction detection → topology graph → CAD wireframe; ParseNet, SPFN, and CPFN perform surface segmentation
- **CAD datasets**: DeepCAD (170K simple models with design history), Fusion-360 (8K), ABC (1M unannotated), CC3D (50K+ partially annotated), ABCPrimitive (5.6K with BRep labels)
- **Text-to-3D/CAD**: Text2CAD and CAD-MLLM provide design-history text annotations; HoLABRep aligns ABCPrimitive with text and sketches

## Rating

- Novelty: ⭐⭐⭐⭐ — First million-scale multimodal CAD dataset covering scans, sketches, text, and BRep topology; annotation methods (SPH soft labels, four-step simulated scanning, parameterized sketches) are each individually novel
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation including multi-baseline comparison, Seen/Unseen generalization, zero-shot transfer, and a three-party assessment system (human + GPT + Gemini)
- Writing Quality: ⭐⭐⭐⭐ — Mathematical formulations are rigorous, pipeline descriptions are clear, and figures are highly informative
- Value: ⭐⭐⭐⭐⭐ — Fills a fundamental gap in large-scale multimodal data for CAD reverse engineering; expected to become an important benchmark in the field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning](../../NeurIPS2025/3d_vision/cosmobench_a_multiscale_multiview_multitask_cosmology_benchmark_for_geometric_de.md)
- [\[CVPR 2026\] Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)
- [\[CVPR 2026\] SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations](scenescribe-1m_a_large-scale_video_dataset_with_comprehensive_geometric_and_sema.md)
- [\[AAAI 2026\] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling](../../AAAI2026/3d_vision/nurbgen_high-fidelity_text-to-cad_generation_through_llm-driven_nurbs_modeling.md)
- [\[CVPR 2026\] Pano360: Perspective to Panoramic Vision with Geometric Consistency](pano360_perspective_to_panoramic_vision_with_geometric_consistency.md)

</div>

<!-- RELATED:END -->
