---
title: >-
  [Paper Note] PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World
description: >-
  [ICML 2026][3D Vision][Physics-Aware 3D Generation] Reframes "creating interactive 3D objects" as a two-stage problem: "physical planning first, physical generation second." The VLM acts as a physical architect…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Physics-Aware 3D Generation"
  - "VLM Planning"
  - "KineVoxel Injection"
  - "Hierarchical Physical Blueprint"
  - "Interactive Assets"
date: 2026-05-08
content_hash: 4bcc3fc752410ba0
---

# PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World

**Conference**: ICML 2026  
**arXiv**: [2605.05163](https://arxiv.org/abs/2605.05163)  
**Code**: [hku-mmlab.github.io/PhysForge](https://hku-mmlab.github.io/PhysForge/)  
**Area**: 3D Vision / Generative Models / Embodied Intelligence  
**Keywords**: Physics-Aware 3D Generation, VLM Planning, KineVoxel Injection, Hierarchical Physical Blueprint, Interactive Assets

## TL;DR
Reframes "creating interactive 3D objects" as a two-stage problem: "physical planning first, physical generation second." The VLM acts as a physical architect, generating a "Hierarchical Physical Blueprint" with hierarchical relationships, materials, and kinematic constraints. A diffusion model then uses KineVoxel Injection to co-denoise articulation parameters and geometric voxels. Combined with the PhysDB dataset (150k assets with four-layer annotations), this approach achieves the first single-view-to-"simulation-ready" 3D asset generation capable of grasping, pushing, and articulating in physics engines.

## Background & Motivation

**Background**: 3D generation has advanced to produce high-fidelity static geometry (e.g., TRELLIS, CLAY, 3DShape2VecSet) and part-aware generation (e.g., OmniPart, PartPacker) for decomposable objects. Initial explorations into physical aspects include EmbodiedGen, which combines existing modules for interactive scenes, and PhysX-3D, which trains a Physical VAE on PartNet with physical annotations. For articulated objects, two directions have emerged: "digital twin reconstruction" and "procedural generation" (e.g., CAGE, SINGAPO).

**Limitations of Prior Work**: Current 3D generation focuses almost exclusively on "static geometry + texture," producing assets that are mere "shells"—unsuitable for grasping, pushing, or articulation in embodied AI simulators or game engines. Even part-aware methods (e.g., OmniPart, PartPacker) decompose objects based on visual/geometry boundaries, ignoring "function" and "physics" as decomposition signals. Procedural generation of articulated objects relies on predefined connection graphs, code templates, or part libraries, limiting generalization and precision.

**Key Challenge**:  
1. The semantics of "interactivity" stem from functional logic and hierarchical physics (e.g., a button's "press" function, a cabinet's door-handle hinge hierarchy), but existing methods define parts solely by visual boundaries.  
2. Simulation-ready assets require geometry, material, and kinematics, but diffusion models excel at geometry/texture, while VLMs excel at structure/world knowledge—no method unifies them.  
3. A major bottleneck is the lack of large-scale datasets with fine-grained physical annotations.

**Goal**: Develop an end-to-end pipeline for generating simulation-ready 3D assets from single views, ensuring deployability in simulators like PhysX/Isaac. Simultaneously, establish a supporting dataset with physical annotations.

**Key Insight**: The authors adapt the successful "plan-then-generate" paradigm from 2D generation to 3D. VLMs excel at planning with world knowledge, while diffusion models excel at synthesizing geometry and articulation parameters. Instead of end-to-end training, the VLM first outputs a complete "physical blueprint," and the diffusion model follows the blueprint. Articulation parameters (origin, axis, limit) are encoded as voxels and co-denoised with geometric voxels.

**Core Idea**: Decouple "physical planning (VLM)" and "physical realization (Diffusion + KVI)" while leveraging PhysDB's four-layer annotations (holistic, static, functional, interactive) for supervision.

## Method

### Overall Architecture
PhysForge operates in two stages.  
**Stage 1: VLM-based Planning**—Input a single view $I$, its 3D voxel representation $V$ (from TRELLIS Stage 1), and an optional 2D mask $M$. A fine-tuned Qwen2.5-VL autoregressively generates a Hierarchical Physical Blueprint, specifying each part's bbox, parent node, joint type, material, function, and state machine.  
**Stage 2: Diffusion-based Generation**—The blueprint guides the generation of geometric voxels and textures. KineVoxel Injection (KVI) encodes articulation parameters (origin, axis, limit) into kinematic voxels, which co-denoise with geometric voxels in the same diffusion process, ensuring synchronization and consistency. The final output is a simulation-ready asset, directly importable into physics engines for interactions like grasping, pushing doors, and rotating knobs. The PhysDB dataset (150k assets, four-layer annotations, human-verified) provides training data, with articulation precision supplemented by PartNet-Mobility and Infinite-Mobility.

### Key Designs

1. **PhysDB Four-Layer Physical Annotation Dataset**:  
    - **Function**: Provides "object-level + part-level" full-stack physical properties for the first time, enabling supervision for the plan-then-generate paradigm.  
    - **Mechanism**: The four-layer annotation system spans different abstraction levels.  
        - **Holistic Tier** (object-level): Real-world scale, category, usage context (e.g., kitchen/bedroom).  
        - **Static Properties Tier** (part-level): Semantic labels, physical materials (e.g., metal/wood/glass), mass.  
        - **Functional Tier**: Intrinsic functions (e.g., to contain/to control) and state machines (e.g., button: [pressed, released]).  
        - **Interactive Tier**: Atomic affordance library (e.g., pushable/rotatable/graspable), joint types (e.g., revolute/prismatic), and joint parameters (e.g., parent part, axis origin, direction, limits).  
    - **Design Motivation**: Layered abstraction avoids annotation collapse by separating high-level semantics (e.g., function) from low-level parameters (e.g., joint axes).

2. **VLM as Physical Blueprint Planner**:  
    - **Function**: Converts VLM world knowledge into hierarchical blueprints with 3D part structures and physical properties.  
    - **Mechanism**: Qwen2.5-VL is fine-tuned to process $(I, V, M)$. Images and masks use the native image encoder, while voxels $V$ are encoded via PartField for part features and downsampled into 512D voxel embeddings. The model introduces 66 new special tokens for compact bbox encoding. The VLM autoregressively outputs bboxes and physical attributes for each part.  
    - **Design Motivation**: Predicting physical attributes resolves part-level ambiguities, demonstrating the mutual benefit of multi-task training.

3. **KineVoxel Injection (KVI) Diffusion Mechanism**:  
    - **Function**: Ensures simultaneous generation of geometry and precise articulation parameters (origin/axis/limit).  
    - **Mechanism**: Encodes articulation parameters as kinematic voxels, co-denoised with geometric voxels in the diffusion process. This shared latent space ensures alignment between geometry and kinematics.  
    - **Design Motivation**: Avoids inconsistencies from separate geometry and articulation networks, ensuring probabilistic consistency.

### Loss & Training
Stage 1: Standard next-token cross-entropy fine-tuning for Qwen2.5-VL to output bbox and physical attribute tokens.  
Stage 2: Diffusion loss (following TRELLIS) with additional supervision for kinematic voxels. Articulation ground truth comes from PartNet-Mobility/Infinite-Mobility. Human-in-the-loop data cleaning ensures annotation quality. Evaluation uses PhysXNet for geometry (Chamfer Distance/F1) and physical property prediction accuracy.

## Key Experimental Results

### Main Results
Comparison on PhysXNet with part-aware and physics-aware baselines. Key metrics: Chamfer Distance (CD), F1-0.1, F1-0.05, Absolute Scale Error (cm).

| Metric | Prev. SOTA (PhysX-3D) | PhysForge | Trend |
|---|---|---|---|
| CD ↓ | Baseline | Significantly lower | Improved geometry accuracy |
| F1-0.1 ↑ | Baseline | Significantly higher | Improved reconstruction accuracy |
| F1-0.05 ↑ | Baseline | Noticeably higher | Superior under stricter thresholds |
| Absolute Scale (cm) ↓ | Large error | Substantially reduced | More accurate physical scale |

### Ablation Study

| Configuration | Key Observation | Explanation |
|---|---|---|
| Full PhysForge | Geometry + articulation fully SOTA | Complete two-stage + KVI |
| w/o Physical Attribute Prediction | Part-level ambiguity reappears | Validates "physics-guided planning resolves part ambiguity" |
| w/o 2D Mask Input | Still produces reasonable bboxes | Blueprint constraints are sufficient |
| w/o KineVoxel Injection | Geometry-articulation mismatch | Sequential articulation prediction fails |
| w/o PhysDB Training | Physical property accuracy drops | Dataset is essential |
| Replace PartField with 3DShape2VecSet | Weaker part representation | PartField better captures part features |

### Key Findings
- **Physical Constraints Aid Structural Planning**: Jointly predicting physical attributes (e.g., material/function) and bboxes improves part-level semantic understanding, even without 2D masks.  
- **Decoupled Plan-Realize Paradigm is Scalable**: VLM handles "what parts and their properties," while diffusion handles "how to realize shapes, textures, and articulation."  
- **KVI Ensures Geometry-Kinematics Consistency**: Encoding articulation parameters as voxels within diffusion ensures natural alignment.  
- **Zero-Shot Deployment**: Generated assets are directly usable in simulators, demonstrating practical readiness.

## Highlights & Insights
- The division of roles—VLM as "architect" and diffusion as "builder"—mirrors human engineering practices, effectively adapting the 2D plan-then-generate paradigm to 3D.  
- KVI elegantly integrates articulation parameters into diffusion, avoiding additional prediction branches while maintaining generative power.  
- The discovery that physical labels enhance part decomposition challenges traditional geometry-centric approaches.  
- The compact 6-token bbox encoding demonstrates the potential of LLMs for structured design generation.  
- The four-layer annotation system offers a refined breakdown of "physical interactivity," setting a potential standard for the field.

## Limitations & Future Work
- Articulation precision relies on existing datasets (PartNet-Mobility/Infinite-Mobility), as PhysDB's 150k samples lack precise joint parameters.  
- Evaluation focuses on geometry and physical property accuracy, lacking systematic assessment of downstream interaction success rates.  
- VLM planning may produce physically inconsistent blueprints, lacking explicit checks.  
- Current pipeline targets rigid-body articulation; extending to soft bodies, fluids, and fabrics requires new mechanisms.  
- Objaverse-based training data may have category biases, with limited validation on complex industrial machinery.

## Related Work & Insights
- **vs OmniPart**: OmniPart focuses on semantic part-aware generation, while PhysForge anchors parts in "function + physics" and generates articulation parameters.  
- **vs PartPacker**: PartPacker prioritizes efficiency via dual volumes, while PhysForge emphasizes physical expressiveness for interactivity.  
- **vs PhysX-3D**: PhysForge's explicit VLM planning and KVI diffusion ensure stronger physical consistency and articulated generation.  
- **vs EmbodiedGen**: EmbodiedGen integrates existing modules, while PhysForge offers an end-to-end pipeline.  
- **vs Digital Twin Reconstruction**: PhysForge generates novel interactive objects, not just reconstructs known ones.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  
- Experimental Thoroughness: ⭐⭐⭐⭐  
- Writing Quality: ⭐⭐⭐⭐  
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] WonderTurbo: Generating Interactive 3D World in 0.72 Seconds](../../ICCV2025/3d_vision/wonderturbo_generating_interactive_3d_world_in_072_seconds.md)
- [\[CVPR 2026\] ArtLLM: Generating Articulated Assets via 3D LLM](../../CVPR2026/3d_vision/artllm_generating_articulated_assets_via_3d_llm.md)
- [\[ICML 2026\] PhyScene3D: Physically Consistent Interactive 3D Tabletop Scene Generation](physcene3d_physically_consistent_interactive_3d_tabletop_scene_generation.md)
- [\[ICML 2026\] LabBuilder: Protocol-Grounded 3D Layout Generation for Interactable and Safe Laboratory](labbuilder_protocol-grounded_3d_layout_generation_for_interactable_and_safe_labo.md)
- [\[ICML 2026\] PhysHanDI: Physics-Based Reconstruction of Hand-Deformable Object Interactions](physhandi_physics-based_reconstruction_of_hand-deformable_object_interactions.md)

</div>

<!-- RELATED:END -->
