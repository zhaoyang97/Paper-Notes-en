---
title: >-
  [Paper Note] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes
description: >-
  [3D Vision] This paper introduces PlaceIt3D, a language-guided object placement task in real 3D scenes, comprising a benchmark, a large-scale dataset, and a 3D LLM-based baseline method called PlaceWizard that performs joint reasoning over scenes, objects, and natural language instructions.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: ac70a68878e238ff
---

# PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes

| Info | Content |
|------|------|
| Conference | ICCV2025 |
| arXiv | [2505.05288](https://arxiv.org/abs/2505.05288) |
| Code | [nianticlabs/placeit3d](https://nianticlabs.github.io/placeit3d/) |
| Area | 3D Vision / Language-Guided Object Placement |
| Keywords | 3D scene understanding, object placement, multimodal large language models, point cloud, language-guided |

## TL;DR

This paper introduces PlaceIt3D, a language-guided object placement task in real 3D scenes, comprising a benchmark, a large-scale dataset, and a 3D LLM-based baseline method called PlaceWizard that performs joint reasoning over scenes, objects, and natural language instructions.

## Background & Motivation

### Problem Definition

Given a reconstructed 3D point cloud scene, a 3D asset, and a natural language instruction, the goal is to determine a placement position and orientation for the object that satisfies the instruction. The task involves four intertwined challenges:

**One-to-many ambiguity**: Valid placements are generally not unique; multiple locations may satisfy a given instruction.

**Precise geometric and physical reasoning**: Many constraints are inherently 3D geometric in nature and cannot be inferred from 2D projections alone.

**Cross-modal joint understanding**: The method must simultaneously comprehend the scene, the asset geometry, and the language instruction.

**Robustness to noisy point clouds**: No privileged metadata (e.g., scene graphs, clean geometry) is available at test time.

### Limitations of Prior Work

- **3D visual grounding** methods typically identify a single correct location and cannot handle one-to-many scenarios.
- **Synthetic scene generation** methods rely on privileged information such as layout graphs or scene graphs.
- **Image-level placement** methods predict only 2D placement regions and lack intrinsic 3D reasoning capability.
- The concurrent work FirePlace focuses on clean synthetic environments and cannot handle the noise inherent in reconstructed scenes.

## Method

### Task Formulation

**Input**: A point cloud scene $\mathbf{X} \in \mathbb{R}^{N \times 6}$ (containing 3D coordinates and color), a 3D asset, and a text instruction. **Output**: A 3D translation vector $\mathbf{t}$ and a yaw angle $\alpha$.

Simplifying assumptions: the vertical direction of both the scene and the asset is known; the asset is always placed on a horizontal surface; only rotation about the vertical axis is considered.

### Constraint Taxonomy

- **Physical feasibility**: The object must not intersect the scene mesh and must rest on a supporting surface.
- **Spatial constraints**: near/adjacent, on, between, above/below.
- **Rotation constraints**: the object faces a specified anchor.
- **Visibility constraints**: the object is within or occluded from the line of sight of an anchor.

A valid placement must satisfy all constraints simultaneously:

$$\mathcal{M} = \bigcap_{c \in \mathcal{C}} \mathcal{M}_c$$

### Overall Architecture of PlaceWizard

PlaceWizard is built upon Reason3D with the following key modules:

**1. Scene Encoding**
- A point encoder extracts features $F_X \in \mathbb{R}^{N \times d}$.
- Positional embedding features $F_X^{pos}$ are appended.
- **Uniform spatial pooling** replaces the original Superpoints: farthest point sampling selects $M$ center points, and each point is assigned to its nearest center. This preserves finer-grained spatial information and avoids Superpoints aggregating horizontal and vertical surfaces into a single feature.
- A Q-Former projects the features into the LLM embedding space.

**2. Asset Encoding**
- A Point-BERT encoder (pretrained on Objaverse) encodes the asset point cloud.
- Max-pooling over the sequence features yields a single embedding.
- The asset dimensions along the X/Y/Z axes are encoded separately.
- An MLP projects the features into the LLM embedding space.

**3. Placement Decoder**
- The LLM outputs three special tokens: [LOC], [ANC], and [ROT].
- Self-attention and cross-attention layers process the token features against scene and asset features.
- Three prediction heads:
    - **Placement mask head**: predicts $\mathcal{M}_{loc} \in [0,1]^N$, the valid placement region.
    - **Rotation head**: predicts $\mathcal{M}_{rot} \in [0,1]^{N \times 8}$, the validity of 8 discrete rotation angles at each point.
    - **Anchor mask head**: predicts $\mathcal{M}_{anc} \in [0,1]^N$, localizing the anchor object mentioned in the instruction (auxiliary task).

### Loss & Training

The total loss consists of four components:

$$\mathcal{L} = \mathcal{L}_{seg}(\bar{\mathcal{M}}_{loc}, \mathcal{M}_{loc}) + \mathcal{L}_{rot} + \mathcal{L}_{seg}(\bar{\mathcal{M}}_{anc}, \mathcal{M}_{anc}) + \mathcal{L}_L$$

- Segmentation loss: $\mathcal{L}_{seg} = \text{BCE} + \text{Dice}$
- Rotation loss: $\mathcal{L}_{rot} = \text{BCE}(\bar{\mathcal{M}}_{rot}, \mathcal{M}_{rot})$
- LLM loss: $\mathcal{L}_L = \text{CE}(\bar{Y}, Y)$

### Inference

The point with the maximum value in $\mathcal{M}_{loc}$ is selected; the translation vector is obtained by adding an offset of half the asset height. The rotation angle is determined by taking the argmax of $\mathcal{M}_{rot}$ at that point.

## Key Experimental Results

### Benchmark Statistics

The PlaceIt3D-benchmark contains 3,500 evaluation samples (142 ScanNet scenes × 20 assets); the PlaceIt3D-dataset contains 100,505 training samples (565 scenes × 20 assets).

### Main Results

| Method | Physical Feasibility | Spatial | Rotation | Visibility | Language Compliance | Global Accuracy | Complete Success |
|------|--------|------|------|--------|----------|-----------|-----------|
| OpenMask3D + rules | 61.6 | 28.6 | 6.5 | 53.4 | 21.8 | 29.2 | 11.7 |
| OpenMask3D + LLM | 5.8 | 35.3 | 10.5 | 61.5 | 18.4 | 26.7 | 1.6 |
| Reason3D (A) | 53.9 | 37.5 | 6.6 | 57.0 | 18.1 | 44.8 | 13.2 |
| **PlaceWizard (G)** | **58.8** | **56.6** | **17.3** | **61.2** | **25.9** | **54.9** | **15.0** |

PlaceWizard significantly outperforms both baselines on all global metrics. The LLM-based baseline lacks direct access to 3D geometry, resulting in extremely low physical feasibility (5.8%).

### Ablation Study

| Variant | Key Modification | Language Compliance | Global Accuracy | Complete Success |
|------|---------|----------|-----------|-----------|
| A (Reason3D) | Superpoints | 18.1 | 44.8 | 13.2 |
| B | Uniform pooling | 18.4 | 48.9 | 10.1 |
| C | + Positional embedding | 20.0 | 50.4 | 10.9 |
| E | + Anchor prediction | 22.2 | 42.5 | 12.3 |
| F | + Rotation prediction | 20.8 | 51.0 | 11.4 |
| G (PlaceWizard) | + Asset features in decoder | **25.9** | **54.9** | **15.0** |

Key findings:
- Uniform spatial pooling improves global accuracy by +4.1% over Superpoints.
- Positional embeddings contribute a further +1.5% improvement.
- Anchor prediction as an auxiliary task improves language compliance by +2.2%.
- Incorporating asset encoding into the decoder yields the largest single gain.

## Highlights & Insights

1. **Novel task definition**: This work is the first to systematically define language-guided object placement in real 3D scenes, explicitly accounting for one-to-many ambiguity.
2. **Complete ecosystem**: The paper simultaneously provides a benchmark (3,500 samples), a large-scale training set (100K+), and a baseline method.
3. **End-to-end design**: The approach avoids the expensive collision detection required by rule-based methods at inference time, offering greater scalability.
4. **Uniform spatial pooling** is an elegant design choice that resolves the issue of Superpoints aggregating large planar surfaces into a single feature vector.

## Limitations & Future Work

- Only horizontal surface placement is supported; scenarios such as "hang the clock on the wall" cannot be handled.
- The dataset is generated via programmatic rules without human verification, limiting quality on edge cases.
- Mismatches between language instructions and the actual scene content are not addressed.
- Performance on the strictest metric (complete success rate) remains low overall (maximum 15.0%), reflecting the inherent difficulty of the task.

## Related Work & Insights

- **Reason3D**: The backbone architecture upon which PlaceWizard is built, providing 3D visual grounding capability.
- **OpenMask3D**: An open-vocabulary 3D grounding method used as a comparison baseline.
- **ScanNet**: Provides real-world room-scale 3D scan data.
- The task has direct relevance to robotic manipulation and AR applications, such as enabling robots to place objects according to natural language instructions.

## Rating

⭐⭐⭐⭐ — Defines a valuable new task with complete contributions (task + data + method), though there remains considerable room for improvement in current method performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization](memorytalker_personalized_speech-driven_3d_facial_animation_via_audio-guided_sty.md)
- [\[ICCV 2025\] Demeter: A Parametric Model of Crop Plant Morphology from the Real World](demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)
- [\[ICCV 2025\] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)
- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)

</div>

<!-- RELATED:END -->
