---
title: >-
  [Paper Note] PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement
description: >-
  [ICLR 2026][LLM Agent][3D scene generation] This paper proposes PhyScensis, an LLM agent framework augmented with a physics engine that generates high-complexity, physically accurate 3D scenes via a spatial and physical predicate-driven solver. It significantly outperforms prior methods in visual quality, semantic correctness, and physical accuracy, and is successfully applied to training robotic manipulation policies.
tags:
  - ICLR 2026
  - LLM Agent
  - 3D scene generation
  - physics engine
  - physical plausibility
  - predicate-based placement
  - probabilistic programming
  - robotic manipulation
date: 2026-05-08
content_hash: 761660db022d8dd5
---

# PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement

**Conference**: ICLR 2026
**arXiv**: [2602.14968](https://arxiv.org/abs/2602.14968)
**Code**: [Project Page](https://physcensis.github.io)
**Area**: LLM Agent
**Keywords**: 3D scene generation, physics engine, LLM agent, physical plausibility, predicate-based placement, probabilistic programming, robotic manipulation

## TL;DR

This paper proposes PhyScensis, an LLM agent framework augmented with a physics engine that generates high-complexity, physically accurate 3D scenes via a spatial and physical predicate-driven solver. It significantly outperforms prior methods in visual quality, semantic correctness, and physical accuracy, and is successfully applied to training robotic manipulation policies.

## Background & Motivation

Automatically generating interactive 3D environments is critical for scalable robotic simulation data collection. However, existing approaches suffer from multiple limitations:

**Procedural methods** (e.g., ProcTHOR): constrained by designer-defined rules and unable to cover open-ended scenes.

**Data-driven methods** (Transformer/Diffusion): limited by the sparse coverage of 3D datasets, particularly lacking fine-grained small-object placement.

**LLM agent methods** have critical shortcomings:
   - Image-driven methods (Architect, SceneTheis) are affected by occlusion and lack fine-grained control.
   - Direct position prediction methods (LayoutGPT, 3D-Generalist) are bottlenecked by LLMs' weak 3D spatial reasoning.
   - Predicate-plus-solver methods (e.g., LayoutVLM) rely solely on 2D AABB collision detection and lack feedback loops.

**Physical interactions are neglected**: relationships such as stacking, containment, and support are not modeled, resulting in object penetration and unstable configurations.

The core challenge is that complex physical scenes require (a) high object density, (b) rich support relationships, and (c) joint modeling of spatial positions and physical properties.

## Method

### Overall Architecture

A three-stage pipeline (Figure 2):
1. **LLM Agent**: generates an object list, spatial/physical predicates, and object descriptions (for asset retrieval) from a user text prompt.
2. **Solver**: a spatial solver handles 2D positional constraints; a physics solver handles 3D stacking/containment via a physics engine.
3. **Feedback System**: analyzes the generated scene and provides corrective signals to drive iterative refinement by the LLM agent.

### Key Designs

#### 1. Predicate System Definition

**Spatial predicates** (2D planar constraints):
- **Positional**: left/right/front/back-of (with specified distance), place-on-base (place on tabletop).
- **Alignment**: align-left/right/front/back, align-center.
- **Rotation**: facing-to, facing-same-as, random-rot, etc.
- **Symmetry**: symmetry-along.
- **Grouping**: group (creates a virtual group), copy-group (copies while preserving structure).

**Physical predicates** (3D interaction constraints):
- **Container insertion**: place-in (drops object into container via physics simulation).
- **Stacking**: place-on (controllable support ratio and stability).
- **Free placement**: place-anywhere (random position with no penetration and guaranteed support).

#### 2. Spatial Solver

Collision detection based on 2D convex hulls (rather than AABB), offering higher precision and faster execution:
- Checks whether each object is "fully solved" (x, y, and yaw are all determined or inferrable).
- If not fully solved, feeds back to the LLM agent requesting additional predicates.
- Iteratively optimizes predicate parameters by minimizing convex hull overlap area and out-of-bounds distance.

#### 3. Physics Solver

**place-in**: similar to Blender's physics-based placement — objects are released from above the container and settle under applied forces.

**place-on / place-anywhere** (Figure 4):
- **Occupancy grid heuristic**: the scene and candidate objects are voxelized into occupancy grids; grid search identifies candidate positions with no penetration and with the object's center of mass projection falling within the support convex hull.
- **Physics engine validation**: only candidates that show no significant displacement after simulation are retained.
- **Probabilistic programming stability assessment**: perturbations around the current state (3D position, Euler angles, mass, center-of-mass offset, friction coefficient) are sampled, and stability probability is estimated via Bayesian methods.

Stability is controllable: iteratively selecting configurations that are "unstable but not fallen" enables the extreme unstable arrangements shown in Figure 3.

#### 4. Feedback System (Three Types)

**Syntactic feedback**: checks predicate format correctness and whether all objects are fully solved.

**Solver failure feedback**: diagnoses causes such as penetration, out-of-table placement, and stacking failure; estimates scene crowdedness; and identifies empty regions (e.g., "empty area to the left of the rear tabletop behind the laptop").

**Success feedback**:
- Stability score (physics engine + probabilistic programming).
- VQA score (whether the scene appears tidy or cluttered).
- Heuristic metrics (surface coverage, compactness, object count).

### Loss & Training

PhyScensis is a generative framework rather than a training-based method and does not involve neural network loss functions. The optimization objectives are the collision/out-of-bounds penalty terms in the spatial solver and stability probability maximization/minimization in the physics solver.

## Key Experimental Results

### Main Results

**Quantitative comparison (Table 1)**:

| Method | VQA Score↑ | GPT Ranking↓ | Settle Distance↓ | Reaching (10 trials) | Placing (10 trials) |
|--------|-----------|-------------|-----------------|----------------------|---------------------|
| Architect | 0.493±0.392 | 2.607±0.673 | 0.405±0.471 | 3/10 | 0/10 |
| 3D-Generalist | 0.578±0.399 | 1.946±0.731 | 0.033±0.048 | 4/10 | 1/10 |
| **PhyScensis** | **0.704±0.425** | **1.429±0.562** | **0.003±0.008** | **9/10** | **3/10** |

PhyScensis achieves significant improvements across all metrics:
- VQA Score +21.8% (vs. 3D-Generalist).
- Settle Distance reduced by 91% (physical accuracy).
- Robot reaching success rate 9/10 vs. 4/10.

**User study (Table 4, 20 participants, 18 cases, 1–5 scale)**:

| Method | Text Alignment↑ | Naturalness & Physics↑ | Complexity↑ |
|--------|----------------|------------------------|------------|
| Architect | 2.68 | 2.65 | 2.69 |
| 3D-Generalist | 2.54 | 2.72 | 3.04 |
| **PhyScensis** | **4.04** | **3.98** | **3.82** |

### Ablation Study

**Feedback system ablation (Table 2)**:

| Variant | Retry Count↓ | Time Cost↓ |
|---------|-------------|-----------|
| No feedback | 1.69±1.92 | 132.29±78.38 |
| No empty-region reporting | 1.43±1.55 | 126.09±59.19 |
| With visual feedback added | **0.95±0.91** | 120.65±53.62 |
| Full framework | 1.04±1.41 | **106.41±55.53** |

The complete feedback system reduces time cost from 132 s to 106 s (20% speedup).

**Predicate/solver ablation (Table 3)**:

| Variant | VQA Score↑ | GPT Ranking↓ | Settle Distance↓ |
|---------|-----------|-------------|-----------------|
| Random placement | 0.415±0.363 | 2.706±0.666 | 0.004±0.003 |
| LLM-Only (direct position prediction) | 0.592±0.401 | 1.882±0.676 | 0.154±0.133 |
| **PhyScensis** | **0.704±0.425** | **1.411±0.492** | **0.003±0.008** |

Random placement achieves a low Settle Distance (since all objects rest on the tabletop without stacking) but performs poorly on VQA and GPT Ranking. LLM-Only exhibits high Settle Distance (physically inaccurate), while PhyScensis achieves both visual quality and physical accuracy.

**Robot experiment**:
- 300 scenes per method × 1 demonstration trajectory used to train a diffusion policy.
- Generalization evaluated on 10 manually designed scenes.
- Scenes generated by PhyScensis more closely match the real distribution, yielding better policy generalization.

### Key Findings

1. Physics engine integration reduces Settle Distance by two orders of magnitude (0.003 vs. 0.405).
2. Predicate-based methods substantially outperform direct LLM position prediction (VQA +19%).
3. The feedback system — especially empty-region identification — significantly improves iterative efficiency.
4. Generated scenes effectively support robotic policy training and generalize to manually designed scenes.

## Highlights & Insights

1. **Elegant integration of physics engine and LLM agent**: the LLM handles high-level semantic understanding and predicate generation, while the physics engine ensures low-level physical accuracy — leveraging the strengths of each.
2. **Probabilistic programming for stability control**: the framework not only generates stable scenes but can deliberately produce extreme unstable arrangements (for challenging robotic scenarios). This fine-grained controllability has not been demonstrated in prior work.
3. **Rich predicate system**: the hierarchical design of spatial and physical predicates covers the vast majority of real-world placement scenarios; advanced predicates such as copy-group support complex structured layouts.
4. **Validated on real robotic applications**: beyond academic scene-generation evaluation, the practical value of the generated scenes is demonstrated through imitation learning experiments.
5. **Convex hull collision detection**: provides more accurate 2D collision detection than AABB while being much faster than full 3D mesh intersection — a sound engineering trade-off.

## Limitations & Future Work

1. 3D assets rely on the BlenderKit dataset and a text-to-3D pipeline, limiting asset quality and diversity.
2. The occupancy grid resolution of the physics solver constrains the precision of continuous placement.
3. Robot placing success rate is only 3/10 — better than baselines but still low in absolute terms.
4. Generation speed (~106 s/scene) may be insufficient for large-scale data generation.
5. Experiments are limited to local scenes such as tabletops, shelves, and boxes, with no extension to room-scale environments.

## Related Work & Insights

Compared to 3D-Generalist (Sun et al., 2025b)'s VLM point-by-point specification approach, PhyScensis circumvents VLMs' spatial reasoning weaknesses through its predicate system. Compared to Architect (Wang et al., 2024b)'s image inpainting approach, it avoids penetration artifacts caused by depth estimation errors. Compared to ClutterGen (Jia & Chen, 2024)'s clutter generation, PhyScensis supports more complex stacking and semantic instruction following.

**Core insight**: positioning the LLM as a "predicate generator" rather than a "coordinate predictor" is the key design philosophy. LLMs excel at semantic understanding and logical reasoning but are weak at precise 3D spatial reasoning. Decoupling these two capabilities through a predicate intermediate representation is a generalizable paradigm for LLM–physical system collaboration.

## Rating

- Novelty: ⭐⭐⭐⭐ (the systematic combination of physics engine, LLM agent, and probabilistic programming for scene generation is pioneering)
- Experimental Thoroughness: ⭐⭐⭐⭐ (quantitative + qualitative + user study + robot experiments + ablations provide comprehensive coverage)
- Writing Quality: ⭐⭐⭐⭐ (method descriptions are clear, figures are well-crafted, and experimental analysis is thorough)
- Value: ⭐⭐⭐⭐ (directly valuable for robotic simulation data generation and embodied AI)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](../../AAAI2026/llm_agent/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[AAAI 2026\] Physics-Informed Autonomous LLM Agents for Explainable Power Electronics Modulation Design](../../AAAI2026/llm_agent/physics-informed_autonomous_llm_agents_for_explainable_power_electronics_modulat.md)
- [\[ICLR 2026\] FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](membership_privacy_risks_of_sharpness_aware_minimization.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](../../ACL2026/llm_agent/hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)

</div>

<!-- RELATED:END -->
