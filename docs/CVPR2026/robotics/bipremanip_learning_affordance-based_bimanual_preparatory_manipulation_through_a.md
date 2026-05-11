---
title: >-
  [Paper Note] BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration
description: >-
  [CVPR 2026][Robotics][Bimanual collaborative manipulation] This paper proposes BiPreManip, a framework for bimanual preparatory manipulation based on visual affordance representations. The system first anticipates the pr…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "Bimanual collaborative manipulation"
  - "visual affordance"
  - "preparatory manipulation"
  - "anticipatory reasoning"
  - "point cloud"
date: 2026-05-08
content_hash: 6b0ab7e9c5207802
---

# BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration

**Conference**: CVPR 2026
**arXiv**: [2603.21679](https://arxiv.org/abs/2603.21679)
**Code**: [Project Page](https://sites.google.com/view/bipremanip)
**Area**: Robotic Manipulation / Human Understanding
**Keywords**: Bimanual collaborative manipulation, visual affordance, preparatory manipulation, anticipatory reasoning, point cloud

## TL;DR
This paper proposes BiPreManip, a framework for bimanual preparatory manipulation based on visual affordance representations. The system first anticipates the primary hand's target interaction region, then guides the assistive hand to perform preparatory actions (e.g., flipping a bottle so its cap faces the primary hand), achieving substantial improvements over baselines in both simulated and real-world environments.

## Background & Motivation
**Background**: Bimanual manipulation research has advanced considerably in recent years (ACT, RDT-1B, 3D FlowMatch Actor, etc.), covering symmetric, sequentially independent, and complementary-role collaboration paradigms.

**Limitations of Prior Work**: Existing methods assume both hands can directly interact with an object, yet many everyday scenarios require one hand to first *change the object's state* before the other hand can operate—e.g., pushing a tablet to the table edge before grasping it, or standing a pen upright before uncapping it.

**Key Challenge**: Preparatory manipulation demands asymmetric anticipatory coordination and long-horizon interdependent planning—the assistive hand must understand the primary hand's future intent while avoiding interference with its anticipated interaction region.

**Goal**: Define and address a novel problem category of "collaborative preparatory manipulation," enabling robots to learn bimanual coordinated behavior of prepare-then-operate.

**Key Insight**: Affordance-driven reasoning—first use an affordance map to anticipate the final goal action, then inversely derive the assistive hand's preparatory behavior.

**Core Idea**: Cross-arm reasoning is achieved via an anticipatory affordance map, such that every action of the assistive hand serves the primary hand's ultimate goal.

## Method

### Overall Architecture
Input: object point cloud + language instruction → Goal Affordance Network predicts anticipatory affordance → Pre-Affordance Network reasons about the assistive hand's preparatory action → Anticipatory Object Pose Predictor estimates the target object pose → Reorient Actor executes reorientation → Goal Affordance Network is invoked again to execute the final goal action.

### Key Designs

1. **Goal Affordance Network**:

    - PointNet++ encodes point cloud features $f_p$; CLIP encodes language instructions as $f_l$
    - An MLP fuses both to predict per-point affordance scores $s$ (likelihood of each point serving as a contact region)
    - A cVAE predicts the target gripper orientation $d_{\text{goal}} \in SO(3)$; combined with the contact point, this yields a 6D action $a_{\text{goal}} \in SE(3)$
    - **Key distinction**: The prediction is *anticipatory* rather than reactive—it imagines interactions that only become feasible after preparatory manipulation is complete
    - **Design Motivation**: Affordance representations naturally encode "where and how to interact," offering greater generalizability than directly learning action sequences

2. **Pre-Affordance Network**:

    - Conditioned on the anticipatory goal affordance, reasons about how the assistive hand should act
    - Fuses $(f_p, f_l, f_{p_{\text{goal}}}, f_{d_{\text{goal}}})$ to predict a pre-affordance map
    - A cVAE samples the assistive gripper orientation $d_{\text{pre}}$, yielding preparatory action $a_{\text{pre}} \in SE(3)$
    - **Design Motivation**: The assistive hand's preparatory behavior must align with the primary hand's future interaction space, precluding naive blind grasping

3. **Anticipatory Object Pose Predictor + Reorient Actor**:

    - Estimates the ideal object pose $T^{\text{obj}} = (t^{\text{obj}}, r^{\text{obj}}) \in SE(3)$ that enables collision-free contact at the target region for the primary hand
    - Transforms the point cloud: $O' = T^{\text{obj}} \cdot O$
    - The Reorient Actor receives the transformed point cloud and current grasp scene, predicting a 6D reorientation motion
    - **Design Motivation**: Many preparatory tasks require adjusting object orientation (e.g., rotating a bottle so its cap faces the primary hand); explicit modeling of this step is more controllable than end-to-end learning

### Loss & Training
- Affordance scores: supervised with $\ell_1$ loss; positive and negative samples are derived from demonstrations
- Gripper orientation: geodesic distance loss $\mathcal{L}_{\text{ori}} = \arccos\frac{\text{Tr}(d^\top d^*) - 1}{2}$ + KL regularization
- The anticipatory stage lacks direct annotations; supervision is constructed via pose transformations from the execution stage: $R_{\text{grp,ant}} = R_{\text{obj,init}} \cdot R_{\text{obj,fin}}^\top \cdot R_{\text{grp,fin}}$
- Both the pose predictor and reorientation actor are cVAEs trained with a combination of geodesic loss + $\ell_1$ + KL

## Key Experimental Results

### Main Results (Success Rate %, Trained / Unseen Objects)

| Category | BiPreManip | ACT | 3DFA | Heuristic | W2A |
|------|-----------|-----|------|-----------|-----|
| Bowl (Edge-Push) | **49/52** | 32/27 | 3/0 | 15/21 | 0/0 |
| Cap | **71/74** | 22/36 | 5/14 | 31/37 | 2/4 |
| Pen-Button (Artic.) | **67/72** | 15/9 | 14/25 | 27/34 | 0/0 |
| Lighter | **43/58** | 34/30 | 41/36 | 21/32 | 2/0 |
| Plate (PerAct2) | **85/82** | 30/26 | 71/68 | 81/78 | 4/4 |

### Ablation Study

| Configuration | Bottle | Pen-button | Pen-cap | Notes |
|------|--------|-----------|---------|------|
| Full model | **30/26** | **67/72** | **26/32** | Best |
| w/o Ant-Aff | 27/13 | 48/58 | 23/10 | Removing anticipatory affordance causes notable degradation |
| w/o ObjPosePred | 24/15 | 51/50 | 21/8 | Removing pose prediction leads to reorientation failures |

### Key Findings
- Across 18 object categories, BiPreManip significantly outperforms all baselines on the majority of tasks
- The method generalizes well to unseen objects; on some categories, success rates on unseen objects even exceed those on training objects
- Both the anticipatory affordance and the object pose predictor are critical components
- Real-world human-to-robot handover experiments further validate the practical utility of the approach

## Highlights & Insights
- The paper defines a novel problem category of "collaborative preparatory manipulation," filling an important gap in bimanual manipulation research
- Affordance-driven anticipatory reasoning is highly elegant—the "think before act" paradigm closely mirrors human behavior
- Parameter sharing ensures semantic consistency between the anticipatory and execution stages, guaranteeing coherence between "imagination" and "execution"
- The benchmark comprising 18 object categories and 882 instances is systematic and comprehensive

## Limitations & Future Work
- The cVAE's multimodal modeling capacity is limited; more complex manipulations may require diffusion-based models
- The method relies on complete point cloud observations and may degrade under heavy occlusion
- Only two-step preparation (grasp + reorientation) is currently supported; longer-horizon preparatory sequences remain unexplored
- Integration with language models could enable more complex task decomposition

## Related Work & Insights
- Single-arm affordance methods such as Where2Act provide the foundation but do not support coordinated reasoning
- The Transformer architecture of ACT suits general bimanual prediction but lacks anticipatory reasoning capability
- Key insight: for manipulation tasks requiring sequential coordination, explicitly modeling "intent" is more effective than end-to-end learning

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Novel problem definition + anticipatory affordance-driven bimanual coordination framework
- Experimental Thoroughness: ⭐⭐⭐⭐ Simulation + real world, 18 categories, multiple baselines and ablations
- Writing Quality: ⭐⭐⭐⭐ Clear logic, intuitive illustrations
- Value: ⭐⭐⭐⭐⭐ Advances bimanual manipulation toward more practical scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[CVPR 2026\] Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](actiongeometry_prediction_with_3d_geometric_prior.md)
- [\[CVPR 2026\] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](panoaffordancenet_towards_holistic_affordance_grou.md)
- [\[CVPR 2026\] STRNet: Visual Navigation with Spatio-Temporal Representation through Dynamic Graph Aggregation](strnet_visual_navigation_with_spatio-temporal_representation_through_dynamic_gra.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)

</div>

<!-- RELATED:END -->
