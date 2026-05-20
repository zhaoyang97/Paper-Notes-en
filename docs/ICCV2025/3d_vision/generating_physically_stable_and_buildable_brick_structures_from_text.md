---
title: >-
  [Paper Note] Generating Physically Stable and Buildable Brick Structures from Text
description: >-
  [ICCV2025][3D Vision][text-to-3D] BrickGPT is the first method to generate physically stable and assemblable interlocking brick structures directly from text prompts. The core idea is to formulate brick assembly as an au…
tags:
  - "ICCV2025"
  - "3D Vision"
  - "text-to-3D"
  - "brick assembly"
  - "physical stability"
  - "autoregressive LLM"
  - "LEGO generation"
date: 2026-05-08
content_hash: 31331760e0885130
---

# Generating Physically Stable and Buildable Brick Structures from Text

**Conference**: ICCV2025
**arXiv**: [2505.05469](https://arxiv.org/abs/2505.05469)  
**Code**: [Project Page](https://avalovelace1.github.io/BrickGPT/)  
**Area**: 3D Vision
**Keywords**: text-to-3D, brick assembly, physical stability, autoregressive LLM, LEGO generation

## TL;DR

BrickGPT is the first method to generate physically stable and assemblable interlocking brick structures directly from text prompts. The core idea is to formulate brick assembly as an autoregressive text generation task, augmented at inference time with physics-aware validity checking and a rollback mechanism to ensure structural stability and buildability.

## Background & Motivation

- **Problem Definition**: Generating 3D structures composed of interlocking bricks directly from free-form text prompts, subject to two requirements: (1) *physical stability* — the structure must not collapse, float, or fracture when placed on a baseplate; and (2) *buildability* — the structure must be compatible with standard brick components and assemblable piece by piece by a human or robot.
- **Limitations of Prior Work**:
    - Standard text-to-3D methods (e.g., DreamFusion) produce digital designs that cannot be directly physically realized — they are difficult to assemble from standard components and may be physically unstable.
    - Existing brick design methods primarily convert given 3D objects into brick layouts (Luo et al.), or are restricted to a single object category (Ge et al.).
    - The few learning-based approaches (Thompson et al.) use graph generation but are limited to simple categories and a single brick type.
    - None support direct generation from text, and none incorporate physical constraints.
- **Motivation**: To leverage LLMs' sequence modeling and text understanding capabilities by reframing "next-token prediction" as "next-brick prediction," while ensuring physical stability through both training data curation and inference-time constraints.

## Method

### Overall Architecture

BrickGPT consists of three stages:
1. **Dataset Construction** (StableText2Brick): 47,000+ physically stable brick structures paired with textual descriptions.
2. **Model Fine-tuning**: Instruction fine-tuning on LLaMA-3.2-1B-Instruct.
3. **Physics-Aware Inference**: Per-brick rejection sampling + physics-aware rollback.

### Dataset: StableText2Brick

**Brick Representation**: Each structure $B = [b_1, b_2, \ldots, b_N]$, where each brick $b_i = [h_i, w_i, x_i, y_i, z_i]$ (dimensions + coordinates), within a $20 \times 20 \times 20$ grid world.

**Construction Pipeline**:
1. **Shape-to-Brick**: ShapeNetCore 3D meshes → voxelization → delete-and-rebuild algorithm to generate brick layouts.
2. **Structure Augmentation**: Multiple brick layout variants are generated for the same object via randomization, increasing diversity and the probability of obtaining stable structures.
3. **Stability Filtering**: A stability score $s_i \in [0,1]$ is computed for each brick via mechanical analysis (nonlinear programming); only structures where all $s_i > 0$ are retained.
4. **Description Generation**: 24-view rendering → GPT-4o generates geometric descriptions at 5 levels of detail (excluding color).

**Scale**: 28,000+ unique 3D objects, 21 common object categories, 47,000+ distinct brick structures.

### Model Fine-tuning

**Custom Brick Text Format**: Each brick is represented as one line `"{h}×{w} ({x},{y},{z})"`, substantially reducing token count compared to the LDraw format while retaining dimensional information to facilitate 3D reasoning. Bricks are ordered bottom-to-top in raster scan order.

**Autoregressive Generation**:

$$p(b_1, b_2, \ldots, b_N | \theta) = \prod_{i=1}^{N} p(b_i | b_1, \ldots, b_{i-1}, \theta)$$

### Physics-Aware Inference

**Stability Analysis** is based on mechanical modeling. A force model (gravity, normal forces, shear forces, knob connection forces) is established for each brick in the structure, and static equilibrium is achieved by solving a nonlinear program:

$$\arg\min_{\mathcal{F}} \sum_i^N \left\{ \left|\sum_j^{M_i} F_i^j\right| + \left|\sum_j^{M_i} \tau_i^j\right| + \alpha \mathcal{D}_i^{\max} + \beta \sum \mathcal{D}_i \right\}$$

Solved using Gurobi. Constraints include: non-negative forces, mutual exclusivity of conflicting forces, and Newton's third law.

**Per-Brick Rejection Sampling**: Each generated brick is checked for format validity, boundary compliance, and collision-freeness — lightweight constraints that do not significantly impact inference time.

**Physics-Aware Rollback**: After generation, stability scores are computed. If the structure is unstable, the state is rolled back to just before the first unstable brick: $B' = [b_1, \ldots, b_{\min \mathcal{I} - 1}]$, and generation continues from $B'$. This process iterates up to 100 times. The median number of rollbacks is only 2, with a median generation time of 40.8 seconds.

### Brick Coloring and Texturing

- **UV Texturing**: Merge visible bricks into a mesh → cubic projection to generate UV map → FlashTex generates the texture.
- **Uniform Color Assignment**: Voxelization → UV unwrapping → FlashTex texture generation → per-voxel/brick color averaging → matching against a standard color library.

## Key Experimental Results

### Main Results

| Method | Valid% | Stable% | Avg. Stability | Min. Stability | CLIP ↑ | DINO ↑ |
|--------|--------|---------|----------------|----------------|--------|--------|
| Pre-trained LLaMA (0-shot) | 0.0 | 0.0 | N/A | N/A | N/A | N/A |
| In-context learning (5-shot) | 2.4 | 1.2 | 0.675 | 0.479 | 0.284 | 0.814 |
| LLaMA-Mesh | 94.8 | 50.8 | 0.894 | 0.499 | 0.317 | 0.851 |
| Hunyuan3D-2 + stability | 100 | 88.4 | 0.976 | 0.813 | 0.324 | 0.868 |
| **BrickGPT** | **100** | **98.8** | **0.996** | **0.915** | **0.324** | **0.880** |

### Ablation Study

| Variant | Valid% | Stable% | Avg. Stability | Min. Stability | CLIP |
|---------|--------|---------|----------------|----------------|------|
| w/o rejection sampling & rollback | 37.2 | 12.8 | 0.956 | 0.325 | 0.329 |
| w/o rollback | 100 | 24.0 | 0.947 | 0.228 | 0.322 |
| **Full BrickGPT** | **100** | **98.8** | **0.996** | **0.915** | **0.324** |

### Key Findings

1. Pre-trained LLaMA (0-shot) completely fails to generate valid structures; 5-shot in-context learning achieves only 2.4% validity — demonstrating the necessity of fine-tuning and inference-time constraints.
2. Rejection sampling raises the validity rate from 37.2% to 100%; rollback raises the stability rate from 24.0% to 98.8%.
3. Even advanced text-to-3D methods such as Hunyuan3D-2, when converted to brick structures and augmented with stability analysis, achieve lower stability rates than BrickGPT (88.4% vs. 98.8%).
4. A median of only 2 rollbacks indicates that the model has already learned a strong stability prior during training.
5. Generated structures are validated through both manual human assembly and automated dual-arm robot assembly.

## Highlights & Insights

1. **Cross-Domain Innovation**: The LLM next-token prediction paradigm is elegantly repurposed as next-brick prediction — a concise and well-motivated adaptation.
2. **Graceful Integration of Physical Constraints**: Rather than applying full physical checks at every step (which would be overly restrictive), the method combines lightweight per-brick validity checks with post-generation rollback, effectively balancing efficiency and stability.
3. **End-to-End Physical Realizability**: Generated structures can be directly assembled in the physical world, including via dual-arm robotic assembly, completing a true closed loop from text to physical object.
4. **Dataset Contribution**: StableText2Brick (47,000+ structures with descriptions) constitutes a large-scale, high-quality dataset of physically stable brick structures.
5. **Custom Text Format Design**: By eliminating redundant information present in LDraw (e.g., orientation, scale), the custom format significantly reduces token count while retaining geometric expressiveness.

## Limitations & Future Work

1. Structures are constrained to a $20 \times 20 \times 20$ grid, limiting resolution and precluding finer-grained designs.
2. Training samples are capped at 4,096 tokens, which may cause truncation for large structures.
3. Training data is drawn from only 21 ShapeNetCore categories, offering limited coverage of naturally occurring objects.
4. Stability analysis depends on the commercial Gurobi solver.
5. Color and texture generation are decoupled post-processing steps, separate from geometry generation.

## Related Work & Insights

- **LLM → 3D**: LLaMA-Mesh demonstrates that LLMs can be fine-tuned to output 3D shapes in OBJ format; this work further adapts that paradigm to physically constrained brick structures.
- **Physics-Aware Generation**: The field has progressed from simple collision avoidance to structural stability analysis to full mechanical modeling with nonlinear programming, reflecting an increasing level of physical sophistication.
- **Realizability in Text-to-3D**: This work opens a new direction toward generating physically assemblable objects from text, with practical implications for manufacturing, education, and architecture.
- **Constrained Inference**: The rejection sampling + rollback inference paradigm is generalizable to other generation tasks requiring hard constraint satisfaction.

## Rating ⭐⭐⭐⭐

The topic is novel, representing the first method to generate physically stable brick structures from text. The method design is elegant, with well-motivated integration of physical constraints. The dataset is large-scale and high-quality. Experiments are comprehensive, with ablations clearly demonstrating the contribution of each component. Physical assembly validation — including robotic assembly — further strengthens the paper's claims. Limitations in resolution and category coverage are acknowledged, but as a pioneering work this is an impressive contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Stable Score Distillation](stable_score_distillation.md)
- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[ICCV 2025\] Bolt3D: Generating 3D Scenes in Seconds](bolt3d_generating_3d_scenes_in_seconds.md)
- [\[ICCV 2025\] A Recipe for Generating 3D Worlds from a Single Image](a_recipe_for_generating_3d_worlds_from_a_single_image.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)

</div>

<!-- RELATED:END -->
