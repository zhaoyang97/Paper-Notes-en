---
title: >-
  [Paper Note] LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation
description: >-
  [CVPR 2025][Robotics][Language-conditioned manipulation] Proposed LaDA, which decouples 7-DoF robot actions into three types of motion primitives (translation, rotation, and gripper) and aligns them with language semantics. Using soft-label contrastive learning and adaptive loss weighting, it achieves a 93.6% average success rate on LIBERO with only 1.3B parameters.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Language-conditioned manipulation"
  - "action decoupling"
  - "motion primitives"
  - "contrastive learning"
  - "action representation"
  - "compositional generalization"
date: 2026-05-08
content_hash: ba067116b44591d2
---

# LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation

**Conference**: CVPR 2025  
**arXiv**: [2603.12967](https://arxiv.org/abs/2603.12967)  
**Code**: TBD  
**Area**: Robotic Manipulation  
**Keywords**: Language-conditioned manipulation, action decoupling, motion primitives, contrastive learning, action representation, compositional generalization

## TL;DR

Proposed LaDA, which decouples 7-DoF robot actions into three types of motion primitives (translation, rotation, and gripper) and aligns them with language semantics. Using soft-label contrastive learning and adaptive loss weighting, it achieves a 93.6% average success rate on LIBERO with only 1.3B parameters.

## Background & Motivation

**Heterogeneity of Vision-Language-Action**: In robotic manipulation, there exists a large semantic gap between high-level vision-language understanding ("pick up the red cup") and low-level motion control (joint angles/velocities).

**Limitations of Prior Work**: Most VLA methods directly predict actions as uninterpretable 7-dimensional vectors, ignoring the structured semantics within actions.

**Unexploited Shared Motion Primitives**: Different tasks share similar motion primitives (e.g., "push forward", "rotate right", "close gripper"), but existing methods fail to reuse these primitives across tasks.

**Demand for Compositional Generalization**: Recombining learned motion primitives in new tasks would substantially improve generalization capabilities to unseen scenarios.

**Natural Correspondence Between Language and Motion**: Verbs in language descriptions (push, pull, rotate) naturally correspond to specific types of motion primitives, a relationship that can be explicitly modeled.

**Core Idea**: Decoupling 7-DoF actions into three types of primitives (translation/rotation/gripper) and grounding each primitive with language semantics to achieve compositional generalization.

## Method

### Overall Architecture

LaDA consists of three core modules:

1. **Action Decoupling**: Decomposing 7-DoF ($\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, g$) into three types of motion primitives: translation, rotation, and gripper.
2. **Language Alignment**: Aligning each type of motion primitive with the language semantic space via soft-label contrastive learning.
3. **Adaptive Loss Weighting**: Balancing the training progress of the three decoupled branches.

### Key Design 1: Motion Primitive Decoupling

- Translation primitive: $[\Delta x, \Delta y, \Delta z]$, corresponding to spatial displacement.
- Rotation primitive: $[\Delta r_x, \Delta r_y, \Delta r_z]$, corresponding to orientation changes.
- Gripper primitive: $[g]$, corresponding to grasping/releasing actions.
- Each type of primitive is predicted by an independent action head, sharing a vision-language backbone.
- Decoupling enables the model to learn finer action semantics.

### Key Design 2: Soft-Label Contrastive Learning

- Unlike traditional contrastive learning that uses 0/1 hard labels, it utilizes the semantic similarity between language instructions as soft labels.
- For example, the translation primitives for "push the red cup" and "push the blue cup" should be similar, as their "push" semantics are close.
- Semantic similarity is calculated via the cosine similarity of sentence embeddings.
- This design allows the model to learn that semantically similar instructions should generate similar motion primitives.

### Key Design 3: Adaptive Loss Weighting

- The loss scales and training difficulties of the three primitive branches differ (e.g., rotation is typically harder to predict than translation).
- Dynamic adjustment of the loss weight for each branch avoids any single branch dominating the training process.
- Ensures balanced progress in training across translation, rotation, and gripper primitives.

## Key Experimental Results

### Main Results (LIBERO Benchmark)

| Task Suite | LaDA (1.3B) | CLIP-RT (2.6B) | OpenVLA | RoboFlamingo |
|---------|-------------|----------------|---------|---------------|
| Spatial | **96.4%** | — | — | — |
| Object | **97.8%** | — | — | — |
| Goal | **88.4%**| — | — | — |
| Long-horizon | **86.4%** | — | — | — |
| **Average** | **93.6%** | ~89% | ~85% | ~82% |

- LaDA has only 1.3B parameters, which is half of CLIP-RT.

### Ablation Study

| Configuration | Average SR | Description |
|------|---------|------|
| Full LaDA | **93.6%** | All components |
| W/o Decoupling (Unified Prediction) | ~88% | Unified 7-DoF prediction |
| W/o Soft Labels (Hard Label Contrastive) | ~90% | 0/1 hard labels |
| W/o Adaptive Weighting | ~91% | Fixed weights |
| W/o Language Alignment | ~87% | Regression loss only |

### Key Findings

- Language alignment is the most significant contributing factor (dropping it results in -6.6pp).
- Action decoupling contributes +5.6pp, showing that structured decomposition significantly benefits performance.
- The gap between soft labels and hard labels (+3.6pp) demonstrates the effectiveness of using linguistic similarity as a supervisory signal.
- LaDA outperforms CLIP-RT with half the parameters, indicating that action decoupling combined with language alignment is more effective than simply scaling up the model.

## Highlights & Insights

1. **Natural Correspondence Between Motion Primitives and Language**: It fully exploits the natural structure of human language descriptions of motion, leading to an elegant design.
2. **Ingenious Soft Label Design**: Using semantic similarity as a soft supervision signal for contrastive learning provides finer granularity than hard labels.
3. **High Efficiency**: Achieving SOTA performance with 1.3B parameters validates that structured design is more effective than brute-force model scaling.
4. **Potential for Compositional Generalization**: The three decoupled primitives can be "assembled" into actions required for new tasks.

## Limitations & Future Work

- Verified only in the LIBERO simulation environment; real-world robot experiments are missing.
- The tripartite decomposition into translation, rotation, and gripper is specifically designed for 7-DoF; more complex robots (such as dexterous hands) require redesigned primitives.
- Long-horizon success rate (86.4%) still has room for improvement, indicating that compositional reasoning over long sequences remains a challenge.
- Negative transfer has not been explored—certain similar yet distinct motion primitives might interfere with each other.

## Related Work & Insights

- **CLIP-RT**: Uses CLIP to align vision, language, and action but does not decouple the action structure, with a parameter size of 2.6B.
- **OpenVLA**: A generalist VLA model that directly predicts raw action vectors.
- **Insight**: The concept of action decoupling can be extended to broader embodied AI tasks, such as navigation and realistic character animation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of motion primitive decoupling and language alignment is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐☆ — Detailed ablation studies, but lacking real-world experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and intuitive methodology.
- **Value**: ⭐⭐⭐☆ — The model is small and highly efficient, but currently limited to simulation.
- **Overall Recommendation**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](../../CVPR2026/robotics/lada_robotic_manipulation.md)
- [\[CVPR 2025\] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](roboground_robotic_manipulation_with_grounded_vision-language_priors.md)
- [\[CVPR 2025\] MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)
- [\[CVPR 2025\] g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks](g3d-lf_generalizable_3d-language_feature_fields_for_embodied_tasks.md)

</div>

<!-- RELATED:END -->
