---
title: >-
  [Paper Note] Demeter: A Parametric Model of Crop Plant Morphology from the Real World
description: >-
  [3D Vision] Demeter is a data-driven parametric plant morphology model that decomposes plant shape into four factors — topology, articulation, shape, and deformation — supporting shape generation, 3D reconstruction, and biophysical simulation.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: e8906ba6ab5780c4
---

# Demeter: A Parametric Model of Crop Plant Morphology from the Real World

## Basic Information

- **Conference**: ICCV 2025
- **arXiv**: 2510.16377
- **Code**: [Project Page](Project will be open-sourced)
- **Area**: 3D Vision
- **Keywords**: Parametric shape model, plant morphology, 3D reconstruction, PCA, biophysical simulation

## TL;DR

Demeter is a data-driven parametric plant morphology model that decomposes plant shape into four factors — topology, articulation, shape, and deformation — supporting shape generation, 3D reconstruction, and biophysical simulation.

## Background & Motivation

Parametric shape models (e.g., SMPL for humans, SMAL for animals) have achieved remarkable success in computer vision and graphics, yet an equally expressive model for plants remains absent. Existing plant modeling approaches primarily rely on:

**Procedural models** (L-systems, etc.): require extensive hand-crafted rules and are difficult to invert.

**General-purpose 3D reconstruction** (NeRF/3DGS, etc.): lack shape priors and perform poorly on thin structures and occluded scenes.

Plant modeling poses unique challenges:
- **Topological variation**: different individuals of the same species exhibit distinct branching structures (unlike fixed-topology human/animal models).
- **Multi-source variation**: shape variation originates from three independent factors — articulation, sub-component shape, and non-rigid deformation.
- **Data scarcity**: high-quality 3D annotated data from real agricultural field scenes is extremely limited.

## Method

### Overall Architecture

Demeter expresses plant shape as a function of a parametric primitive graph:

$$(\mathbf{V}, \mathbf{F}) = \mathcal{M}_\Phi(\Gamma, \theta, \beta, \gamma)$$

where $\Gamma$ denotes topology, $\theta$ articulation parameters, $\beta$ shape parameters, $\gamma$ deformation parameters, and $\Phi = \{\Phi_s, \Phi_d\}$ the learned PCA bases.

The vertex computation formula is:

$$\mathbf{v}_f = T(\theta; \Gamma) \cdot \mathcal{D}(\mathbf{v}_t + \mathcal{S}(\beta); \gamma)$$

### Four Parametric Components

**1. Topology ($\Gamma$)**

Represented as a tree data structure with $n = n_l + n_s + n_o$ nodes, each representing a leaf or stem. Each node has a type $\text{tp}(i) \in \{\text{leaf}, \text{stem}\}$ and a parent node $\text{pa}(i)$. Unlike SMPL's fixed topology, Demeter allows different individuals of the same species to have distinct topology graphs.

**2. Articulation ($\theta$)**

The articulation parameters for each node are $\theta_i = (\tau_i, d_i, s_i)$, representing the rotation quaternion, path length along the parent stem, and scale factor, respectively. Global transforms are computed via a forward kinematic chain:

$$\mathbf{T}_i(\theta; \Gamma) = \prod_{i' \in \text{ans}(i)} T(\tau_{i'}, d_{i'}, s_{i'})$$

**3. Shape ($\beta$)**

Defines template-based offsets in canonical space. For leaves, the template is a $m_{l_1} \times m_{l_2}$ 2D grid with control points defined by Catmull-Rom splines. PCA is used to compress the high-dimensional parameters:

$$\mathbf{v}_t + \mathcal{S}(\beta) = \Phi_s^T \beta + \mathbf{v}_t$$

where $\Phi_s \in \mathbb{R}^{2 m_{l_1} m_{l_2} \times |\beta|}$ is the PCA basis learned from 2D leaf scans.

**4. Deformation ($\gamma$)**

Converts leaf articulation into a 2D skeleton structure to drive 3D deformation. Forward kinematics is used to map joint angles to deformed 3D positions:

$$\mathcal{D}(\mathbf{v}_j; \gamma_i) = \prod_{j' \in \text{ans}(j)} T(\tau_{j'}, d_j, 1) \cdot \mathbf{v}_j$$

PCA dimensionality reduction is likewise applied: $\Phi_d^T \gamma$. The deformation preserves local rigidity (leaf area and stem length are approximately invariant).

### Learning Demeter from Real-World Data

**Data collection**: Multi-view RGB videos of approximately 600 soybean plants from a farm in Illinois are collected; 4DGS is used for dynamic reconstruction, followed by 2DGS for high-quality mesh extraction, with manual annotation of instance segmentation and topology.

**Learning procedure** follows a multi-stage strategy:
1. **Stage 1**: Shape parameters $\hat{\beta}_i$ are fitted from raw 3D point clouds by minimizing Chamfer distance; the PCA basis $\Phi_d$ is learned from aligned deformation templates.
2. **Stage 2**: Given initial articulation and estimated $\beta_i$, $\gamma_i$, articulation parameters are optimized via forward kinematics to fit the complete point cloud.

### 3D Reconstruction

- **Multi-view reconstruction**: Point cloud instance segmentation (PointTransformer-V3) → topology inference via minimum spanning tree → parameter fitting.
- **Single-image reconstruction**: SAM segmentation → Mask-RCNN instance prediction → depth estimation lifted to 3D → topology inference → Demeter fitting.

## Key Experimental Results

### Main Results: Quantitative 3D Reconstruction Comparison

| Method | Smooth | Disentangled | Learnable | Soybean CD ↓ | Maize CD ↓ | Soybean Size (KB) ↓ | Maize Size (KB) ↓ |
|---|---|---|---|---|---|---|---|
| NKSR | ✓ | | | 0.0030 | 0.0023 | 5785.6 | 3686.4 |
| SimpleProc | ✓ | ✓ | | 0.0376 | 0.0557 | **0.2754** | **0.2236** |
| **Demeter (Ours)** | ✓ | ✓ | ✓ | **0.0016** | 0.0071 | 3.375 | 1.766 |

Demeter achieves the best CD on soybean (0.0016), with storage far smaller than NKSR (3.375 KB vs. 5785.6 KB), while simultaneously offering disentanglement and learnability.

### Ablation / Comparison: Single-Image Reconstruction

| Method | IoU ↑ |
|---|---|
| One2345++ | 0.296 |
| Meshy | 0.206 |
| **Demeter (Ours)** | **0.328** |

- One2345++ produces coarse shapes but lacks fine detail.
- Meshy yields photo-realistic reconstructions but is misaligned with the input image.
- Demeter produces complete reconstructions that are faithful to the input.

### Key Findings

- **Cross-species generalization**: Demeter fits effectively across multiple species including pepper, rose, tobacco, and maize.
- **Biophysical simulation**: Meshes generated by Demeter are fed into the Helios simulator, successfully simulating diurnal variation in photosynthetic rate.
- **Parameter interpretability**: The four parameters — topology, articulation, shape, and deformation — can be controlled independently, enabling flexible morphological editing.

## Highlights & Insights

1. **Filling a domain gap**: Demeter is the first data-driven parametric shape model for plants, extending the success of SMPL to the plant domain.
2. **Variable-topology handling**: A tree-structured graph is employed to represent variable topologies, elegantly resolving the fixed-topology assumption that constrains existing parametric models.
3. **Multi-source variation disentanglement**: Shape variation is decomposed into three orthogonal sources — articulation, shape, and deformation — offering finer granularity than SMPL.
4. **Practical application value**: The model provides a computational foundation for crop yield analysis and environmental monitoring, as demonstrated through downstream tasks such as photosynthesis simulation.

## Limitations & Future Work

- The assumption that leaves are 2D shapes and stems are uniform-thickness curves makes the model unsuitable for structurally complex plants such as cacti, algae, and banyan trees.
- Organs such as flowers and fruits are currently not modeled.
- The 3D data collection and annotation pipeline is time-consuming; extension to new species requires re-acquisition.
- Single-image reconstruction IoU remains relatively low (0.328), leaving substantial room for improvement.

## Related Work & Insights

- **SMPL/FLAME/SMAL** and other parametric models for humans, faces, and animals provide the core design inspiration; Demeter innovatively addresses the variable-topology problem absent in these works.
- **L-system**-based procedural modeling methods are well-suited for forward generation but difficult to invert; Demeter's primitive-based design is better suited for inverse problems.
- **Catmull-Rom splines** for leaf shape representation is an elegant choice, as control points lie directly on the curve, facilitating optimization.
- This work may inspire parametric modeling research in other domains involving branching structures, such as coral and vasculature.

## Rating

⭐⭐⭐⭐ — A pioneering work that is the first to extend data-driven parametric shape modeling from humans/animals to plants, featuring a sophisticated design with tangible agricultural application value. The dataset contribution is notable, though the range of applicable species remains limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement](neuraleaf_neural_parametric_leaf_models_with_shape_and_deformation_disentangleme.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [\[ICCV 2025\] DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness](dso_aligning_3d_generators_with_simulation_feedback_for_physical_soundness.md)

</div>

<!-- RELATED:END -->
