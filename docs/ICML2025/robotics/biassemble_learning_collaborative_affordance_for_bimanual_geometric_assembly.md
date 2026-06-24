---
title: >-
  [Paper Note] BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly
description: >-
  [ICML 2025][Robotics][Bimanual collaboration] The BiAssemble framework is proposed to decompose the geometric assembly task into three steps (pick-up -> alignment -> assembly) by learning collaboration-aware point-level affordances. It outperforms existing affordance and imitation learning methods in fractured object reassembly tasks and is validated on a real-world benchmark.
tags:
  - "ICML 2025"
  - "Robotics"
  - "Bimanual collaboration"
  - "geometric assembly"
  - "point-level affordance"
  - "fragment reassembly"
  - "long-horizon planning"
date: 2026-05-08
content_hash: 12e775543ef8388b
---

# BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly

**Conference**: ICML 2025  
**arXiv**: [2506.06221](https://arxiv.org/abs/2506.06221)  
**Code**: [https://sites.google.com/view/biassembly/](https://sites.google.com/view/biassembly/)  
**Area**: Robotics  
**Keywords**: Bimanual collaboration, geometric assembly, point-level affordance, fragment reassembly, long-horizon planning

## TL;DR
The BiAssemble framework is proposed to decompose the geometric assembly task into three steps (pick-up -> alignment -> assembly) by learning collaboration-aware point-level affordances. It outperforms existing affordance and imitation learning methods in fractured object reassembly tasks and is validated on a real-world benchmark.

## Background & Motivation

**Background**: Shape assembly is categorized into furniture assembly (functional component association) and geometric assembly (fractured fragment reassembly). The latter is widely applicable (e.g., cultural relic restoration, bone reassembly) but remains under-researched.

**Limitations of Prior Work**: (a) Existing methods only predict the target pose, ignoring collisions during actual manipulation; (b) fragments have arbitrary geometries with no semantic definitions, making grasping and manipulation extremely difficult; (c) the bimanual coordination and contact-rich assembly process within long-horizon action sequences are highly complex.

**Key Challenge**: Both the observation space (arbitrary geometric fragments) and the action space (long-horizon bimanual coordination) are extremely large.

**Goal**: How to enable bimanual robots to collaboratively assemble fractured fragments of arbitrary shapes?

**Key Insight**: Leverage point-level affordances to achieve geometric generalization, and decompose the long-horizon task into three sub-steps to reduce complexity.

**Core Idea**: Mimic human intuition—pick-up -> alignment (leaving a gap) -> gradual pushing. Each step utilizes affordance to perceive the constraints of subsequent steps.

## Method

### Overall Architecture
A three-step workflow:
1. **Pick-up**: Choose grasp points by learning point-level affordances (considering both grasp feasibility and subsequent assembly compatibility).
2. **Alignment**: Move the fragments to an aligned pose (finding a collision-free aligned pose through backward disassembly).
3. **Assembly**: Predict a collision-free direction to gradually push the fragments together.

### Key Designs

1. **Collaboration-Aware Point-Level Affordance**:

    - **Function**: Predicts a grasp score for each point on the fragment surface, considering both local geometry and subsequent operations.
    - **Mechanism**: $\text{Affordance} = \text{grasp feasibility} \times \text{alignment reachability} \times \text{assembly direction compatibility}$.
    - **Design Motivation**: It is insufficient to select points that are merely graspable geometrically — it must also be ensured that subsequent alignment and assembly can be successfully completed after the grasp.

2. **Collision-Free Aligned Pose Generation**:

    - **Function**: Find the aligned pose by backward disassembly from the fully assembled state.
    - **Mechanism**: Separate the fragments in the opposite direction of assembly, leaving a safety gap.
    - **Design Motivation**: Plunging directly into the target pose inevitably causes collisions; aligning first and then pushing helps avoid collisions.

3. **Real-World Reproducible Benchmark**:

    - **Function**: Create a globally accessible standardized fragment benchmark.
    - **Mechanism**: Utilize standard objects (e.g., a specific brand of mug) + a standardized breaking method, and provide 3D meshes.
    - **Design Motivation**: Fractured fragments with varying geometries make fair evaluation difficult; a standardized benchmark addresses this issue.

### Loss & Training
- The affordance model is trained in simulation and transferred to the real world.
- The simulation environment supports bimanual fragment assembly.

## Key Experimental Results

### Main Results

| Method | Assembly Success Rate | Grasp Success Rate |
|------|----------|----------|
| Imitation Learning | ~25% | ~60% |
| Single-step Affordance | ~40% | ~75% |
| **BiAssemble** | **~65%** | **~85%** |

### Ablation Study

| Configuration | Success Rate | Description |
|------|--------|------|
| No Collaboration Awareness | ~40% | Grasp points ignore subsequent feasibility |
| No Alignment Step (Direct Assembly) | ~20% | Massive collisions |
| Full BiAssemble | **~65%** | Three-step decomposition is optimal |

### Key Findings
- The three-step decomposition simplifies complex long-horizon tasks into learnable sub-tasks.
- Collaboration-aware affordance improves performance by ~25% compared to geometric-only affordance.
- Successful assembly of broken mugs in the real world demonstrates the feasibility of simulation-to-real (Sim-to-Real) transfer.

## Highlights & Insights
- **Task decomposition mimics human intuition**—humans naturally pick up, align, and then push fragments together; this decomposition is highly intuitive.
- The advantages of affordance-based methods in geometric generalization are validated once again.
- The real-world reproducible benchmark brings long-term value to this field.

## Limitations & Future Work
- Only two-fragment assembly is handled; scaling to multi-fragment assembly remains an open problem.
- 3D perception of fragments depends heavily on the quality of point clouds.
- Assembly direction prediction is still relatively simple; complex geometries may require more fine-grained planning.

## Related Work & Insights
- **vs. Pose-Only Prediction Methods**: Ignores the manipulation process, rendering it unexecutable.
- **vs. Furniture Assembly**: Fragments lack semantic labels, making the task significantly more challenging.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending affordance to bimanual geometric assembly is a novel application.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively evaluated across simulation and real-world environments with multiple object categories.
- Writing Quality: ⭐⭐⭐⭐ Clear illustrations and reasonable task decomposition.
- Value: ⭐⭐⭐⭐ Advances the practical feasibility of robotic fragment assembly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](../../CVPR2026/robotics/bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)
- [\[ICML 2025\] Geometric Contact Flows: Contactomorphisms for Dynamics and Control](geometric_contact_flows_contactomorphisms_for_dynamics_and_control.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](../../ICCV2025/robotics/selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[CVPR 2025\] ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning](../../CVPR2025/robotics/maniptrans_efficient_dexterous_bimanual_manipulation_transfer_via_residual_learn.md)
- [\[CVPR 2025\] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](../../CVPR2025/robotics/panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)

</div>

<!-- RELATED:END -->
