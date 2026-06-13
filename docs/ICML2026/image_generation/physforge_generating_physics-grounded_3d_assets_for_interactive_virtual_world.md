---
title: >-
  [Paper Note] PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World
description: >-
  [ICML 2026][Image Generation][Physics-aware 3D Generation] This work reinterprets the creation of interactive 3D objects as a two-stage "physical planning then physical generation" problem. A VLM acts as a physical archi…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Physics-aware 3D Generation"
  - "VLM Planning"
  - "KineVoxel Injection"
  - "Hierarchical Physical Blueprint"
  - "Interactive Assets"
date: 2026-05-08
content_hash: 067d9b673b1a7e7c
---

# PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World

**Conference**: ICML 2026  
**arXiv**: [2605.05163](https://arxiv.org/abs/2605.05163)  
**Code**: [hku-mmlab.github.io/PhysForge](https://hku-mmlab.github.io/PhysForge/)  
**Area**: 3D Vision / Generative Models / Embodied AI  
**Keywords**: Physics-aware 3D Generation, VLM Planning, KineVoxel Injection, Hierarchical Physical Blueprint, Interactive Assets  

## TL;DR
This work reinterprets the creation of interactive 3D objects as a two-stage "physical planning then physical generation" problem. A VLM acts as a physical architect to generate a "Hierarchical Physical Blueprint" containing hierarchical relationships, materials, and kinematic constraints. Subsequently, a diffusion model utilizes KineVoxel Injection to co-denoise articulation parameters with geometric voxels. Supported by the PhysDB dataset containing 150k assets with four-tier annotations, this framework achieves the first single-view generation of 3D assets that can be grasped, pushed, and articulated within physical engines.

## Background & Motivation

**Background**: 3D generation has achieved high-fidelity static geometry (TRELLIS, CLAY, 3DShape2VecSet) and expanded into part-aware generation (OmniPart, PartPacker) for object decomposition. Sparse explorations in the physical domain have emerged: EmbodiedGen stitches off-the-shelf modules for interactive scenes; PhysX-3D trains a Physical VAE with physical annotations on PartNet; and articulated object research has diverged into digital twin reconstruction and procedural generation (CAGE, SINGAPO, etc.).

**Limitations of Prior Work**: Current 3D generation focuses almost exclusively on "static geometry + texture," producing assets that are merely "hollow shells." These cannot be grasped by grippers or pushed via hinges, making them undeployable in embodied AI simulators or game engines. Existing part-aware methods (OmniPart, PartPacker) decompose parts based on visual or geometric boundaries, lacking "function" and "physics" as decomposition signals. Procedural generation for articulated objects relies on predefined connectivity graphs, code templates, or part libraries, suffering from poor generalization and low precision.

**Key Challenge**: (1) The "interactivity" semantics of objects stem from functional logic and hierarchical physics (e.g., the "press" function of a button, the hinge hierarchy of cabinet doors and handles), whereas current part definitions rely only on visual boundaries. (2) Simulation-ready assets require a complete triad of geometry, material, and kinematics; however, diffusion models excel at geometry/texture while VLMs excel at structure/world knowledge, with no existing method unifying them. (3) A critical bottleneck is the lack of large-scale datasets with fine-grained physical annotations.

**Goal**: To construct an end-to-end pipeline from a single view to simulation-ready 3D assets, ensuring generated objects can be directly manipulated in simulators like PhysX/Isaac, while establishing a supporting physical annotation dataset.

**Key Insight**: The authors transfer the successful "plan-then-generate" paradigm from 2D generation to 3D. VLMs possess world knowledge suitable for planning, while diffusion models excel at precise synthesis of geometry and articulation parameters. Rather than end-to-end hard training, the VLM first outputs a comprehensive "Physical Blueprint," followed by diffusion-based construction. Furthermore, articulation parameters (origin, axis, limit) are elegantly encoded into a voxel format for joint denoising with geometric voxels.

**Core Idea**: Decouple "physical planning (VLM)" from "physical realization (Diffusion + KVI)" and provide supervision using the four-tier annotations (holistic / static / functional / interactive) from PhysDB.

## Method

### Overall Architecture
PhysForge consists of two stages. **Stage 1: VLM-based Planning** takes a single view $I$, its corresponding 3D voxel representation $V$ (produced by the first stage of TRELLIS), and an optional 2D mask $M$ as input. A fine-tuned Qwen2.5-VL autoregressively generates a Hierarchical Physical Blueprint, including attributes such as bboxes, parent nodes, joint types, materials, functions, and state machines for each part. **Stage 2: Diffusion-based Generation** generates geometric voxels and textures according to the blueprint. KineVoxel Injection (KVI) encodes articulation parameters (origin, axis, limit) into a specialized kinematic voxel, which is co-denoised with geometric voxels in a single diffusion process to ensure synchronization and consistency between geometry and kinematics. The final output consists of simulation-ready assets directly importable into physics engines for interactions like grasping, opening doors, or turning knobs. The accompanying PhysDB dataset (150k assets, four-tier labels, human-verified) provides training data, while articulation precision is supplemented by PartNet-Mobility and Infinite-Mobility.

### Key Designs

1.  **PhysDB: Four-tier Physical Annotation Dataset**:
    - **Function**: Provides the first full-stack physical attributes (object-level + part-level) for large-scale 3D assets, enabling supervision for the plan-then-generate paradigm.
    - **Mechanism**: The four-tier system covers different abstraction levels. **Holistic Tier** (object-level): Real-world scale, category, and usage scenarios. **Static Properties Tier** (part-level): Semantic labels, physical materials (metal, wood, glass, etc.), and mass. **Functional Tier**: Intrinsic functions (to contain / to control) and state machines (e.g., button: [pressed, released]). **Interactive Tier**: Atomic affordance library (pushable/rotatable/graspable), joint types (revolute/prismatic/continuous/fixed), and joint parameters (parent part, axis origin, direction, limits). Data consists of 150k assets with meaningful part structures selected from Objaverse, covering seven major categories: household, industrial, weapons, personal, vehicles, tech & electronics, and cultural.
    - **Design Motivation**: Layering physical information by abstraction ensures coverage of both high-level functional semantics and low-level numerical joint parameters, preventing annotation quality collapse caused by a conflated approach.

2.  **VLM as Physical Blueprint Planner**:
    - **Function**: Converts the LVLM's world knowledge into a hierarchical blueprint of 3D part structures and physical attributes.
    - **Mechanism**: Qwen2.5-VL is used as the base, taking $(I, V, M)$ as input. Images and masks use native encoders. For the voxel $V$, instead of common 3DShape2VecSet, the model first uses PartField to encode part features for each voxel, then downsamples them into 512-dimensional voxel embeddings using position-aware 3D convolutions to enhance part awareness. 66 new special tokens are introduced: `<boxs>` / `<boxe>` to wrap bboxes, and 64 `<box0>...<box63>` tokens for quantized coordinates. Each 3D axis-aligned bbox uses only 6 tokens, making structural planning highly compact.
    - **Design Motivation**: The authors discovered that co-predicting physical attributes helps resolve part-level ambiguity—given physical/functional constraints, the model provides reasonable part decomposition even without a 2D mask. This evidence of "prior-assisted" logic shows physical labels act as strong semantic signals for decomposition.

3.  **KineVoxel Injection (KVI) Diffusion Mechanism**:
    - **Function**: Enables the diffusion model to output precise articulation parameters (origin/axis/limit) while synthesizing geometry, avoiding inconsistencies caused by serial modules.
    - **Mechanism**: Articulation parameters are encoded into a special "kinematic voxel" and denoised alongside geometric voxels. The latent space of the diffusion model carries both "shape information" and "kinematic information" simultaneously, sharing the same denoising steps. Consequently, the generated geometry and joint parameters are naturally aligned.
    - **Design Motivation**: Training separate "geometry generation" and "kinematics prediction" networks often leads to mismatches (e.g., a door is drawn but the hinge axis is miscalculated). KVI merges both into a single generative process, ensuring consistency at the probability distribution level.

### Loss & Training
Stage 1 employs standard next-token cross-entropy SFT to fine-tune Qwen2.5-VL for bbox and physical attribute token sequences. Stage 2 uses diffusion loss (following TRELLIS) plus additional parameter supervision for kinematic voxels. Articulation ground truth is sourced from PartNet-Mobility and Infinite-Mobility. Human-in-the-loop data cleaning ensures annotation quality. Evaluation is conducted on PhysXNet, comparing geometry (Chamfer Distance / F1) and physical attribute prediction accuracy.

## Key Experimental Results

### Main Results
Comparisons against existing part-aware and physics-aware methods on PhysXNet using Chamfer Distance (CD), F1-0.1, F1-0.05, and Absolute Scale Error (cm).

| Metric | Prev. SOTA (PhysX-3D series) | PhysForge | Gain |
|---|---|---|---|
| CD ↓ | Baseline | Significant Decrease | Improved Geometric Precision |
| F1-0.1 ↑ | Baseline | Significant Increase | Improved Reconstruction Accuracy |
| F1-0.05 ↑ | Baseline | Notable Increase | Superior even under strict thresholds |
| Absolute Scale (cm) ↓ | High Error | Substantially Reduced | More Accurate Physical Scale |

### Ablation Study

| Configuration | Key Observation |
|---|---|
| Full PhysForge | Achieving SOTA in both geometry and articulation. |
| w/o Physical Attribute Co-prediction | Part-level ambiguity recurs; confirms physics-guided planning benefits. |
| w/o 2D Mask Input | Still produces reasonable bboxes; blueprint constraints are sufficient. |
| w/o KineVoxel Injection | Geometry and articulation mismatch occurs. |
| w/o PhysDB Training | Physical attribute prediction accuracy drops significantly. |
| Replace PartField with 3DShape2VecSet | Weakened local part representation. |

### Key Findings
- **Physical Constraints Benefit Structural Planning**: Training VLMs to predict physical attributes (material/function) alongside bboxes significantly improves semantic understanding of part decomposition.
- **Scalable Plan-Realize Paradigm**: The clear division of labor (VLM for "what parts and properties," Diffusion for "how they look and move") allows the framework to benefit from future advancements in either component.
- **KVI is Key for Consistency**: Treating kinematic parameters as voxels makes kinematic constraints an inherent part of the generation process, which is more natural and precise than post-processing.
- **Zero-shot Deployment**: Results can be directly imported into simulators like PhysX. The authors demonstrate downstream applications such as robotic grasping and virtual world interactions.

## Highlights & Insights
- The "VLM as architect, Diffusion as builder" role division mirrors human engineering practices, effectively translating the 2D plan-then-generate paradigm to 3D.
- Encoding articulation parameters as KineVoxels within the diffusion flow is an elegant engineering solution that avoids extra branches while maintaining the generative power of latent diffusion.
- The discovery that physical labels aid part decomposition challenges the intuition that decomposition is purely geometric; semantic constraints from functional tokens clarify what constitutes a "part."
- The 6-token bbox encoding is a refined example of embedding 3D structural generation into the LLM next-token framework, offering high transferability for future structured design tasks.

## Limitations & Future Work
- Articulation precision remains dependent on existing GT datasets like PartNet-Mobility; large-scale automated annotation of precise joints remains an open problem.
- Evaluation focuses on geometric and attribute accuracy rather than downstream "successful manipulation rates" in simulators.
- The VLM planning stage might still generate blueprints that violate physical laws (e.g., illogical door orientations) without an explicit physical consistency check.
- Currently limited to rigid-body articulation, with no coverage for soft bodies, fluids, or cloth.

## Related Work & Insights
- **vs OmniPart**: OmniPart focuses on purely geometric part decomposition; PhysForge anchors parts in function/physics and produces articulation parameters.
- **vs PartPacker**: PartPacker compresses parts into dual volumes for efficiency; PhysForge adds physical dimensions for interactivity.
- **vs PhysX-3D**: PhysX-3D uses a Physical VAE within TRELLIS; PhysForge employs explicit VLM planning and KVI for stronger consistency.
- **vs Digital Twin Reconstruction (CAGE, etc.)**: Those works focus on rebuilding known objects, while PhysForge enables the generation of novel, interactive objects from imagination.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](../../CVPR2026/image_generation/physgen_physically_grounded_3d_shape_generation_for_industrial_design.md)
- [\[ICML 2026\] Position: AI Evaluations Should be Grounded on a Theory of Capability](position_ai_evaluations_should_be_grounded_on_a_theory_of_capability.md)
- [\[ICCV 2025\] Diffusion-based 3D Hand Motion Recovery with Intuitive Physics](../../ICCV2025/image_generation/diffusion-based_3d_hand_motion_recovery_with_intuitive_physics.md)
- [\[ICLR 2026\] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow](../../ICLR2026/image_generation/unified_multi-modal_interactive_reactive_3d_motion_generation_via_rectified_flow.md)
- [\[ICML 2026\] Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](../../CVPR2026/image_generation/physgen_physically_grounded_3d_shape_generation_for_industrial_design.md)
- [\[ICML 2026\] Position: AI Evaluations Should be Grounded on a Theory of Capability](position_ai_evaluations_should_be_grounded_on_a_theory_of_capability.md)
- [\[ECCV 2024\] Generating 3D House Wireframes with Semantics](../../ECCV2024/image_generation/generating_3d_house_wireframes_with_semantics.md)
- [\[ICCV 2025\] Diffusion-based 3D Hand Motion Recovery with Intuitive Physics](../../ICCV2025/image_generation/diffusion-based_3d_hand_motion_recovery_with_intuitive_physics.md)
- [\[ICLR 2026\] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow](../../ICLR2026/image_generation/unified_multi-modal_interactive_reactive_3d_motion_generation_via_rectified_flow.md)

</div>

<!-- RELATED:END -->
