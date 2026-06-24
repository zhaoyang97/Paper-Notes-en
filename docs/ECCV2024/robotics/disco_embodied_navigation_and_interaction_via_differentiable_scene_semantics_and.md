---
title: >-
  [Paper Note] DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-Level Control
description: >-
  [ECCV2024][Robotics][embodied navigation] The DISCO framework is proposed, which significantly improves the performance of embodied navigation and interaction on the ALFRED benchmark (outperforming SOTA by +8.6% in unseen success rate, without requiring step-by-step instructions) through differentiable scene semantic representation and dual-level coarse-to-fine action control.
tags:
  - "ECCV2024"
  - "Robotics"
  - "embodied navigation"
  - "mobile manipulation"
  - "differentiable scene representation"
  - "affordance"
  - "dual-level control"
  - "ALFRED"
date: 2026-05-08
content_hash: e3977819a30effa5
---

# DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-Level Control

**Conference**: ECCV2024  
**arXiv**: [2407.14758](https://arxiv.org/abs/2407.14758)  
**Code**: [AllenXuuu/DISCO](https://github.com/AllenXuuu/DISCO)  
**Area**: Robotics  
**Keywords**: embodied navigation, mobile manipulation, differentiable scene representation, affordance, dual-level control, ALFRED

## TL;DR

The DISCO framework is proposed, which significantly improves the performance of embodied navigation and interaction on the ALFRED benchmark (outperforming SOTA by +8.6% in unseen success rate, without requiring step-by-step instructions) through differentiable scene semantic representation and dual-level coarse-to-fine action control.

## Background & Motivation

Building general household assistant agents is a long-term goal of embodied AI, requiring agents to possess capabilities in task planning, environmental modeling, and object interaction. Existing approaches are mainly divided into two categories:

1. **Neural policy methods** (e.g., Seq2Seq, E.T., M-Track): Learn end-to-end actions, but require a large volume of training trajectories and annotations, suffering from the contradiction between long-horizon tasks and memoryless perception.
2. **Map-based planning methods** (e.g., HLSM, FILM, Prompter): Build scene models to assist planning, but suffer from rigid execution and difficulty in adapting dynamically during runtime.

Both categories have limitations: neural policies are data-hungry and generalize poorly, while map planning methods rely on discrete cell representations that are sensitive to imperfect perception, requiring manual heuristic rules for patching.

## Core Problem

How to construct an embodied agent that can efficiently complete mobile manipulation tasks based on verb-noun instruction pairs (e.g., "Pickup Lettuce")? Specifically, three sub-problems need to be addressed:

1. How to establish a scene representation that is semantically rich, dynamically updated, queryable, and generalizable?
2. How to achieve efficient mobile manipulation under limited imitation data?
3. How to integrate primitive tasks into long-horizon embodied instruction following applications?

## Method

### Perception System

Starting from egocentric RGB frames, three neural networks are used to predict pixel-level information:

- **Depth Estimation**: A U-Net architecture where depth is discretized into 50 bins (10cm per bin), trained with cross-entropy loss.
- **Instance Segmentation**: Mask R-CNN with 85 object classes, pretrained on COCO and fine-tuned.
- **Affordance Estimation**: A U-Net architecture that predicts 1 navigation category + 7 interaction categories (e.g., pickable, openable), trained with binary cross-entropy loss.

All training data is collected through the AI2THOR simulator; the use of unseen scene data is strictly prohibited during training.

### Differentiable Scene Representation

The scene is modeled as a 20m × 20m space, discretized into an 80 × 80 grid (each cell is 25cm × 25cm). Each grid cell is assigned a 256-dimensional embedding $s_i$, while $N^o + N^a$ semantic query vectors $q_j$ (object classes + affordance classes) are initialized.

**Query Mechanism**: The probability of grid cell $i$ belonging to class $j$ is computed via dot product and sigmoid:

$$p_{i,j} = \sigma(s_i^T q_j)$$

**Online Optimization**: At each step, the egocentric frame is transformed into a semantic point cloud and projected onto a bird's-eye view, generating soft labels $y_i^j$ (normalized semantic point ratio). Cross-entropy loss is applied to visible grid cells, and both $s_i$ and $q_j$ are updated simultaneously via gradient descent (learning rate 0.01, 10 iterations per step).

Key Advantage: Compared to discrete cell representation, the continuous differentiable representation makes soft trade-offs between historical and current observations, mitigating imperfection in perception without requiring heuristic manual patching rules.

### Dual-Level Coarse-to-Fine Action Control

For a verb-noun primitive task (e.g., "Pickup Lettuce"), execution proceeds in three phases:

1. **Random Walk**: When the target object is not yet detected, the navigation affordance is used to query the reachability map to randomly select an accessible point, and BFS is used to plan the path.
2. **Coarse Control (Global Cues)**: Once the target object is found, the scene representation is queried to obtain the joint probability map of object and affordance distributions. The grid cell with the highest probability is selected as the target, and BFS plans a path to within 1m of the target.
3. **Fine Control (Local Cues)**: A ResNet50 neural policy is used, where the input is a concatenation of RGB + depth + target object mask, predicting actions via object-class-specific classifiers. It only handles short-horizon adjustments (within 4 steps) and is trained via imitation learning on expert trajectories.

Fine control requires only 316,935 frames of training data (ALFRED's default trajectories contain 1,051,308 frames), making it highly data-efficient.

### Application: Embodied Instruction Following

Integrated into ALFRED long-horizon tasks: a fine-tuned BERT parses natural language instructions into ALFRED internal parameters, which are mapped to a sequence of verb-noun subgoals via templates. For example, a `pick_clean_then_put` task is mapped to (Pick, Lettuce) → (Clean, Lettuce) → (Put, DiningTable).

## Key Experimental Results

The main results on the ALFRED test set:

| Setting | Seen SR | Unseen SR | Unseen GC |
|------|-----------|-----------|-----------|
| DISCO (with step-by-step instructions) | **59.5%** | **56.5%** | **66.8%** |
| DISCO (high-level goal only) | **58.0%** | **54.7%** | **65.5%** |
| Prompter (with step-by-step instructions) | 53.2% | 45.7% | 58.8% |
| CAPEAM (with step-by-step instructions) | 51.8% | 46.1% | 57.3% |

Key Findings:

- Unseen scene success rate outperforms SOTA by +10.4% (with instructions) / +11.0% (without instructions).
- **DISCO without step-by-step instructions still outperforms SOTA with step-by-step instructions** (54.7% vs 46.1%).
- PLWSR metric is 1.57x-1.75x of Prompter, proving higher execution efficiency.

Ablation Study (Validation Set):

| Ablation Item | Seen SR Change | Unseen SR Change |
|--------|------------|------------|
| Remove differentiable representation → discrete cell | -9.9% | -12.3% |
| Remove navigation affordance | -9.3% | -9.0% |
| Remove interaction affordance | -5.1% | -3.7% |
| Remove fine control | -4.3% | -3.3% |
| Remove coarse control | -43.8% | -45.9% |

## Highlights & Insights

1. **Elegant Differentiable Scene Representation**: The dot-product-sigmoid mechanism pairing grid embeddings and semantic queries enables differentiable optimization instead of heuristic updates of discrete cells, leading to significantly stronger generalization (12.3% increase in unseen scenes).
2. **Practical Dual-Level Control Paradigm**: Coarse control efficiently handles long-range navigation, while fine control focuses only on short-horizon adjustments, reducing data requirements by more than 3x.
3. **Outperforming SOTA Without Step-by-Step Instructions**: Demonstrates the robustness and planning capability of the method, reducing reliance on fine-grained human annotations.
4. **Natural Affordance Integration**: Properties like openable are automatically integrated into decision-making, avoiding hand-crafted rules.

## Limitations & Future Work

1. **Dependence on Simulator Ground Truth for Perception Training**: Depth, segmentation, and affordances are all trained using ground truth from AI2THOR, which may result in a severe drop in perception quality when transferring to the real world.
2. **Fixed Affordance Categories**: The 7 categories of interaction affordance are simulator-defined, lacking open-vocabulary generalization.
3. **2D Bird's-Eye View Scene Representation**: Ignores height dimension information, which may be insufficient for environments with multiple levels or stacked objects.
4. **Validated Only in AI2THOR/ALFRED**: High reliance on discrete action spaces and limited scene diversity, representing a massive gap before real-robot deployment.
5. **Brittle Language Understanding via Template Matching**: Instruction parsing via BERT + templates is brittle and does not support flexible natural language input.

## Related Work & Insights

| Method | Scene Representation | Control Method | Unseen SR |
|------|---------|---------|--------|
| FILM | Discrete 2D cell | Heuristic rules | 26.5% |
| HLSM | 3D voxel | Neural policy | 16.3% |
| Prompter | Discrete 2D cell + search | Heuristic rules | 45.7% |
| CAPEAM | Context memory | Context planning | 46.1% |
| **DISCO** | **Differentiable continuous embedding** | **Coarse-fine dual-level** | **56.5%** |

DISCO's advantages lie in: (1) differentiable representation is more robust than discrete cells; (2) dual-level control balances global efficiency of map planning and local flexibility of neural policies; (3) affordance integration eliminates manual rules.

## Insights & Connections

- **Transferable concepts from differentiable scene representations**: The paradigm of dot-product query + online gradient optimization can be applied to other tasks requiring dynamic spatial semantic modeling (e.g., semantic SLAM, scene graph construction).
- **Highly generalizable coarse-fine control paradigm**: The hierarchical strategy of global planning + local adjustment is worth emulating in broader robot manipulation scenarios.
- **Potential for integration with foundation models**: Currently, BERT + templates are used for instruction parsing. Replacing this with VLMs/LLMs might further enhance planning flexibility and open-world generalization.

## Rating
- Novelty: 8/10 — The integration of differentiable scene representation and dual-level control is novel, although the individual modules are relatively classic.
- Experimental Thoroughness: 9/10 — Comprehensive ablation studies and qualitative analyses render the results highly convincing.
- Writing Quality: 8/10 — Well-structured, with clearly articulated motivation and methods.
- Value: 8/10 — Marginally advances SOTA on ALFRED but remains limited by simulator environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](../../ICLR2026/robotics/janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.md)
- [\[ICLR 2026\] Differentiable Simulation of Hard Contacts with Soft Gradients for Learning and Control](../../ICLR2026/robotics/differentiable_simulation_of_hard_contacts_with_soft_gradients_for_learning_and_.md)
- [\[ICLR 2026\] Lifelong Embodied Navigation Learning](../../ICLR2026/robotics/lifelong_embodied_navigation_learning.md)
- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](../../NeurIPS2025/robotics/esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[CVPR 2026\] DynBridge: Bridging Imagination and Control through Interaction Dynamics for Robot Manipulation](../../CVPR2026/robotics/dynbridge_bridging_imagination_and_control_through_interaction_dynamics_for_robo.md)

</div>

<!-- RELATED:END -->
