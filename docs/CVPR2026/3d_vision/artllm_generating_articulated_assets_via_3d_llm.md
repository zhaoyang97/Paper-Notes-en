---
title: >-
  [Paper Note] ArtLLM: Generating Articulated Assets via 3D LLM
description: >-
  [CVPR 2026][3D Vision][Articulated Object] ArtLLM formulates articulated object generation as a language generation problem. A 3D multimodal LLM autoregressively predicts part layouts and kinematic joint parameters (disc…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Articulated Object"
  - "3D LLM"
  - "URDF"
  - "Autoregressive"
  - "Part-Aware Generation"
date: 2026-05-08
content_hash: e5315073ad906d9f
---

# ArtLLM: Generating Articulated Assets via 3D LLM

**Conference**: CVPR 2026
**arXiv**: [2603.01142](https://arxiv.org/abs/2603.01142)  
**Code**: [https://authoritywang.github.io/artllm](https://authoritywang.github.io/artllm)  
**Area**: 3D Vision / Articulated Object Generation
**Keywords**: Articulated Object, 3D LLM, URDF, Autoregressive, Part-Aware Generation

## TL;DR
ArtLLM formulates articulated object generation as a language generation problem. A 3D multimodal LLM autoregressively predicts part layouts and kinematic joint parameters (discretized as tokens) from point cloud input, followed by XPart-based high-fidelity part geometry synthesis. The method significantly outperforms existing approaches on PartNet-Mobility (mIoU 0.69) with inference in only 19 seconds.

## Background & Motivation

**Background**: Interactive digital environments (games, robotics, simulation) rely on articulated 3D objects whose functionality derives from part geometry and kinematic structure. Existing methods suffer from fundamental limitations.

**Limitations of Prior Work**:
- **Optimization-based reconstruction methods** (PARIS, VideoArtGS, ArtGS): require slow per-object joint fitting and typically handle only single-joint simple objects.
- **Retrieval-based assembly methods** (SINGAPO, CAGE, URDFormer): assemble objects from fixed part libraries, resulting in high geometric redundancy and poor generalization.

**Key Challenge**: General-purpose 3D generative models (Trellis, Hunyuan3D) can produce high-quality geometry, and part-level generation (XPart, OmniPart) has also advanced. However, these models **lack understanding of kinematic structure**—generated parts have no notion of how they should move, leading to a fundamental disconnect between geometry and motion.

**Key Insight**: A **unified solution that jointly understands geometry and articulation** is needed. LLMs are naturally suited to handle variable-length structured sequences; their sequence modeling and reasoning capabilities can be leveraged to autoregressively predict articulation blueprints.

**Core Idea**: Discretize URDF articulation structures into token sequences and train a 3D LLM to autoregressively generate a unified blueprint of "part layouts + kinematic joints" from point clouds, which then drives a part generation model to synthesize geometry.

## Method

### Overall Architecture
A three-stage pipeline:
1. **ArtLLM**: Point cloud input → 3D LLM → token sequence predicting part AABBs and joint parameters.
2. **Part Geometry Synthesis**: Predicted bounding boxes → XPart generates high-fidelity part meshes.
3. **Physics-Constrained Joint Limit Correction**: Collision detection → hierarchical search for precise joint limits.

### Key Designs

1. **Language Modeling of Articulation Structure**:

    - Each part is parameterized by its AABB: $\text{BBox}(x_{min}, y_{min}, z_{min}, x_{max}, y_{max}, z_{max})$
    - Joint definitions include type, parent-child connectivity, axis direction, axis position, and motion range.
    - Four joint types are supported: Revolute, Continuous, Prismatic, and Screw.
    - Generation order: all part bounding boxes are predicted first, followed by all joint definitions.

2. **Continuous Parameter Discretization (Quantization)**:

    - **Design Motivation**: LLMs inherently predict discrete tokens; direct regression of continuous values is numerically unstable.
    - Bounding box coordinates: quantized into 128 bins over $[-1,1]$.
    - Joint origin: 128 bins; rotation range: 48 bins over $[-2\pi, 2\pi]$; translation distance: 64 bins.
    - **Joint axis direction**: a discrete codebook of 128 entries is constructed—uniform sampling on the XY/YZ/XZ planes (covering axis-aligned directions) supplemented by FPS on a Fibonacci sphere.
    - This hierarchical codebook design provides denser coverage of principal axis-aligned directions while maintaining flexibility for arbitrary orientations.

3. **Multi-Task Multi-Stage SFT**:

    - Three tasks: part layout prediction only / joint prediction given layout / end-to-end prediction.
    - **Two-stage training**:
        - Stage 1: train part layout prediction only (point cloud encoder initialized from P3SAM pretrained weights).
        - Stage 2: joint SFT on all three tasks.
    - **Design Motivation**: first establish a foundation for part-level geometric understanding, then build motion reasoning on top of it.

4. **Physics-Constrained Joint Limit Correction**:

    - Single-timestep geometry prediction cannot account for motion dynamics, potentially causing part collisions.
    - For revolute joints: articulate the child part within the predicted range and compute collision volume against other static parts.
    - Angles where the derivative of collision volume spikes indicate collision events → hierarchical search for the precise angle → set as the corrected joint limit.
    - Analogous processing is applied to prismatic joints.

### Loss & Training
- Standard cross-entropy loss for SFT.
- Multi-task data mixing ratio: 3:2:5.
- Cosine learning rate schedule, max 1e-5, warmup ratio 0.03.
- Data augmentation: random scaling ($s \in [0.8, 1.05]$) and rotation (90°/180°/270°) with 75% probability.
- Stage 1: 8×H20 GPUs, 50 epochs (~8h); Stage 2: 8×H20 GPUs, 30 epochs (~15h).
- 3D encoder: Point Transformer v3; LLM backbone: Qwen3 0.6B.

## Key Experimental Results

### Main Results (PartNet-Mobility, 7 categories, 77 objects)

| Method | mIoU↑ | CD↓ | Type Acc↑ | Joint-Axis-Err↓ | Joint-Pivot-Err↓ | Range-IoU↑ | Graph Acc↑ | Time(s) |
|--------|-------|-----|-----------|-----------------|------------------|------------|------------|---------|
| URDFormer | 0.123 | 0.249 | 0.607 | 0.738 | 0.610 | 0.703 | 0.079 | 183 |
| SINGAPO | 0.433 | 0.044 | 0.765 | 0.245 | 0.257 | 0.526 | 0.456 | 84 |
| ArtAny | 0.338 | 0.072 | 0.846 | 0.453 | 0.536 | **0.865** | 0.614 | 522 |
| **ArtLLM** | **0.688** | **0.028** | **0.908** | **0.127** | **0.080** | 0.740 | **0.774** | **19** |

### Ablation Study

| Configuration | IoU | Type Acc | Axis Err | Pivot Err | Range IoU | Graph Acc |
|---------------|-----|----------|----------|-----------|-----------|-----------|
| Full | **0.473** | **0.898** | **0.141** | **0.135** | **0.582** | **0.780** |
| A: w/o Discretization | 0.352 | 0.823 | 0.277 | 0.235 | 0.575 | 0.775 |
| B: w/o Multi-Task | 0.464 | 0.825 | 0.289 | 0.131 | 0.510 | 0.737 |
| C: w/o Data Augmentation | 0.412 | 0.894 | 0.142 | 0.138 | 0.577 | 0.754 |
| D: w/o Multi-Stage | 0.463 | 0.890 | 0.143 | 0.175 | 0.511 | 0.780 |

### Key Findings
- ArtLLM is an order of magnitude faster at inference (19s vs. 84–522s), making it suitable for large-scale simulation environments.
- Discretization (A) has the largest impact on coordinate- and direction-related attributes (IoU: 0.352 vs. 0.473).
- Multi-task learning (B) improves all metrics except axis direction error, indicating complementary effects from training tasks of varying difficulty.
- Physics-constrained joint limit correction effectively eliminates self-collisions (qualitative results) without affecting inference speed.
- Real2Sim application succeeds: reconstructed articulated assets reproduce real robot manipulation behavior in the SAPIEN simulator.

## Highlights & Insights
- **Articulation as Language**: URDF kinematic structures are naturally mapped to token sequences, fully exploiting LLM sequence modeling capabilities.
- **Carefully Designed Discretization**: The hierarchical codebook for joint axis directions and the varying quantization precision for different physical quantities reflect a deep understanding of the problem structure.
- **Multi-Task Multi-Stage Training**: A simple yet effective approach for decoupling geometric understanding from motion reasoning.
- **End-to-End Practical Value**: A complete pipeline from images/text to simulation-ready articulated assets.

## Limitations & Future Work
- Training data covers a limited number of object categories (43), with insufficient generalization to complex categories such as vehicles and robots.
- Physical properties (mass, friction coefficients, etc.) are not modeled and represent a natural direction for future extension.
- Joint limit correction is a post-processing step; ideally, collision awareness should be integrated into the generation process.
- The method depends on XPart for part geometry synthesis; inaccurate bounding box predictions may cause geometric truncation.

## Related Work & Insights
- SINGAPO and URDFormer are direct competitors, both relying on fixed part libraries; ArtLLM eliminates this constraint entirely through generation.
- The 3D LLM encoder-projector architecture is similar to SpatialLM.
- The discretization + autoregressive paradigm is generalizable to other structured 3D prediction tasks (e.g., scene graph generation, assembly planning).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to use a 3D LLM for end-to-end generation of multi-joint articulated assets; a paradigm-level innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative comparisons, complete ablations, and Real2Sim validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed descriptions of the discretization design.
- Value: ⭐⭐⭐⭐⭐ Direct and significant applicability to robot learning and simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience](3drawagent_teaching_llm_to_draw_in_3d_with_early_contrastive_experience.md)
- [\[CVPR 2026\] Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface](real2edit2real_generating_robotic_demonstrations_via_a_3d_control_interface.md)
- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [\[CVPR 2026\] FreeArtGS: Articulated Gaussian Splatting Under Free-Moving Scenario](freeartgs_articulated_gaussian_splatting_under_free-moving_scenario.md)
- [\[ICLR 2026\] LaVCa: LLM-assisted Visual Cortex Captioning](../../ICLR2026/3d_vision/lavca_llm-assisted_visual_cortex_captioning.md)

</div>

<!-- RELATED:END -->
