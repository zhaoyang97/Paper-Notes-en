---
title: >-
  [Paper Note] TELA: Text to Layer-wise 3D Clothed Human Generation
description: >-
  [ECCV 2024][Human Understanding][3D Human Generation] TELA proposes a layer-wise 3D clothed human representation and a progressive optimization strategy to generate garment-decoupled 3D human models from text descriptions, supporting applications such as layer-by-layer clothing generation and virtual try-on.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "3D Human Generation"
  - "Garment Generation"
  - "Layer-wise Representation"
  - "Text-to-3D"
  - "Virtual Try-on"
date: 2026-05-08
content_hash: 1b52bf0c0dedf5db
---

# TELA: Text to Layer-wise 3D Clothed Human Generation

**Conference**: ECCV 2024  
**arXiv**: [2404.16748](https://arxiv.org/abs/2404.16748)  
**Code**: [http://jtdong.com/tela_layer/](http://jtdong.com/tela_layer/)  
**Area**: Human Understanding / 3D Generation  
**Keywords**: 3D Human Generation, Garment Generation, Layer-wise Representation, Text-to-3D, Virtual Try-on

## TL;DR
TELA proposes a layer-wise 3D clothed human representation and a progressive optimization strategy to generate garment-decoupled 3D human models from text descriptions, supporting applications such as layer-by-layer clothing generation and virtual try-on.

## Background & Motivation

**Background**: Text-to-3D clothed human generation is a crucial direction in 3D content creation. Existing methods (e.g., DreamAvatar, AvatarCLIP) typically encode the human body and clothing into a single unified model, generating the complete clothed human in a single-stage optimization.

**Limitations of Prior Work**: (1) Holistic generation methods cannot separate the human body from garments, making editing operations like re-dressing and virtual try-on impossible; (2) Single-stage optimization lacks fine-grained control over the generation pipeline, often causing geometric coupling between the body and garments; (3) Occlusion relationships between different clothing layers (e.g., undergarments vs. outerwear) are challenging to model correctly.

**Key Challenge**: Achieving both high-quality generation and garment editability requires decoupling the human body and clothing at the representation level, which significantly increases the difficulty of generation.

**Goal**: Design a layer-wise clothed human representation and optimization strategy to generate high-quality 3D human models with decoupled garments.

**Key Insight**: Decompose the clothed human into a minimally clothed body layer and progressively stacked clothing layers, employing a progressive inside-out generation strategy.

**Core Idea**: Generate the minimally clothed body first, then add clothing layers sequentially, ensuring correct geometric relations between layers via stratified compositional rendering and decoupling losses.

## Method

### Overall Architecture
Given a text description (e.g., "a woman in a blue dress"), the framework first generates a minimally clothed human body model (SMPL-based) and then progressively generates clothing layers, with each garment layer represented as an independent implicit/explicit surface model. The progressive optimization strategy ensures that each layer is correctly overlaid on top of the preceding layers.

### Key Designs

1. **Layer-wise Clothed Human Representation**:

    - Function: Represents the human body and individual clothing layers as independent, editable models.
    - Mechanism: The body layer is SMPL-based, while each garment layer is represented by an independent NeRF/SDF. Each layer possesses its own geometric and appearance parameters, allowing independent editing or replacement. The clothing layers are attached to the body layer, following physical inside-out occlusion relationships.
    - Design Motivation: Layer-wise representation is a prerequisite for garment editability—only decoupled representations can support re-dressing and virtual try-on.

2. **Stratified Compositional Rendering**:

    - Function: Merges multi-layer models into a final image for SDS optimization.
    - Mechanism: During rendering, layers are composed in an inside-out order, where outer garments occlude inner body layers. Alpha compositing is used to combine the color and density of each layer to ensure physically correct occlusion. This rendering process is differentiable, allowing gradients to propagate back to each layer.
    - Design Motivation: Standard single-layer rendering cannot handle occlusions in multi-layer geometries; stratified compositional rendering addresses this challenge.

3. **Garment-Body Decoupling Loss**:

    - Function: Prevents geometric entanglement between garment layers and the body layer.
    - Mechanism: A regularization loss is designed to ensure garment layers only have density near the human surface (without penetrating the body interior), while suppressing the body layer's appearance contribution to the final rendering in regions covered by clothes. This loss encourages the correct hierarchical relationship of "outer clothes, inner body".
    - Design Motivation: Optimization without explicit constraints easily leads to garment-body geometric blending; the decoupling loss is crucial to ensuring layer-wise quality.

### Loss & Training
The Score Distillation Sampling (SDS) loss is utilized to drive text-guided 3D generation, combined with a garment-body decoupling regularization loss. A progressive optimization strategy is adopted: the human body layer is optimized first, followed by sequential addition and optimization of garment layers.

## Key Experimental Results

### Main Results

| Method | Generation Quality | Decoupling | Editability |
|------|---------|--------|---------|
| Ours | High | Strong | Supports Re-dressing |
| DreamAvatar | High | None | Unsupported |
| AvatarCLIP | Medium | None | Unsupported |
| TADA | High | Weak | Limited |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Full TELA | Best | Layer-wise + Decoupling + Progressive |
| w/o Stratified Rendering | Poor Geometry | Incorrect Occlusion |
| w/o Decoupling Loss | Severe Entanglement | Garment-body Blending |
| Joint Optimization (Non-progressive) | Degraded Quality | Unstable Optimization |

### Key Findings
- Layer-wise representation is key to garment editability—existing holistic representation methods cannot support any editing.
- Decoupling loss is essential for preventing geometric entanglement; without it, garments and the body undergo severe blending.
- Progressive optimization is more stable than joint optimization as the inner layers provide a reliable initialization for the outer layers.

## Highlights & Insights
- **Representation-driven Problem Solving**: Instead of focusing solely on generating algorithm improvements, this work redesigns the representation format to enable editing—"right representation makes the right capability".
- **Progressive Inside-Out Optimization**: Simulated the real-world process of getting dressed (body first, then clothes). This physically intuitive optimization strategy is both natural and effective.
- **Practical Application of Virtual Try-On**: Once garments are decoupled, virtual try-on can be directly performed (e.g., putting Person A's clothes onto Person B), offering immediate commercial value.

## Limitations & Future Work
- Progressive multi-layer optimization is slow, requiring re-optimization for each additional clothing layer.
- The representation capacity may be insufficient for complex garment geometries (such as ribbons or intricate wrinkles).
- The physical properties of garments (such as fabric drape and elasticity) are not modeled.
- Only text inputs are supported; garment transfer using reference images as input is not supported.

## Related Work & Insights
- **vs DreamAvatar**: DreamAvatar generates a holistic human body, whereas TELA utilizes layer-wise generation to support editing.
- **vs TADA**: TADA has some decoupling capabilities but they are incomplete; TELA's layer-wise representation is more explicit.
- **vs AvatarCraft**: AvatarCraft focuses on animation rather than garment editing, entailing different objectives.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel combination of layer-wise representation and progressive optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison with multiple baselines, comprehensive ablation studies, and demonstrations of editing applications.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and systematic.
- Value: ⭐⭐⭐⭐ Provides a practical boost to 3D human generation and virtual try-on.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FreeCloth: Free-Form Generation Enhances Challenging Clothed Human Modeling](../../CVPR2025/human_understanding/freecloth_free-form_generation_enhances_challenging_clothed_human_modeling.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [\[ECCV 2024\] 3DFG-PIFu: 3D Feature Grids for Human Digitization from Sparse Views](3dfg-pifu_3d_feature_grids_for_human_digitization_from_sparse_views.md)
- [\[ECCV 2024\] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)

</div>

<!-- RELATED:END -->
