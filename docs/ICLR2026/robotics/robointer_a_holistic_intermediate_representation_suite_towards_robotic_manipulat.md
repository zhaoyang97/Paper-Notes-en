---
title: >-
  [Paper Note] RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation
description: >-
  [ICLR 2026][Robotics][Intermediate Representation] This paper presents RoboInter, a unified manipulation suite for intermediate representations, comprising: RoboInter-Tool (a semi-automatic annotation GUI), RoboInter-Data (230K episodes × 571 scenes with dense per-frame annotations across 10+ intermediate representation types), RoboInter-VQA (a 29-category embodied VQA benchmark), and RoboInter-VLA (a plan-then-execute framework supporting both modular and end-to-end variants), providing a complete infrastructure for enhancing VLA generalization through intermediate representations.
tags:
  - ICLR 2026
  - Robotics
  - Intermediate Representation
  - VLA
  - Manipulation Dataset
  - Embodied VQA
  - Plan-then-Execute
date: 2026-05-08
content_hash: 13b01566fecb2493
---

# RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation

**Conference**: ICLR 2026
**arXiv**: [2602.09973](https://arxiv.org/abs/2602.09973)
**Code**: [GitHub](https://github.com/RoboInter)
**Area**: Robot Learning / Datasets
**Keywords**: Intermediate Representation, VLA, Manipulation Dataset, Embodied VQA, Plan-then-Execute

## TL;DR

This paper presents RoboInter, a unified manipulation suite for intermediate representations, comprising: RoboInter-Tool (a semi-automatic annotation GUI), RoboInter-Data (230K episodes × 571 scenes with dense per-frame annotations across 10+ intermediate representation types), RoboInter-VQA (a 29-category embodied VQA benchmark), and RoboInter-VLA (a plan-then-execute framework supporting both modular and end-to-end variants), providing a complete infrastructure for enhancing VLA generalization through intermediate representations.

## Background & Motivation

**State of the Field**: Vision-Language-Action (VLA) systems integrate large-scale pretrained VLMs with robotic manipulation, yet existing manipulation datasets suffer from high annotation cost, embodiment specificity, and insufficient coverage. The plan-then-execute paradigm—generating high-level plans before translating them into low-level actions—has been validated as an effective approach to improving generalization, but critically relies on supervision signals from intermediate representations (subtasks, traces, grounding, etc.).

**Limitations of Prior Work**:

1. Existing datasets provide almost no dense intermediate representation annotations, limiting the development of plan-then-execute methods.
2. Prior annotation efforts are either small in scale (ShareRobot: only 51K), limited in annotation types (LLARVA: trace only), or rely on automatic annotation with uncontrolled quality (ECoT).
3. No systematic benchmark exists for evaluating VLMs' spatial and temporal reasoning in embodied settings.
4. Comparisons between modular and end-to-end VLA lack a unified framework and data support.

**Root Cause**: While the potential of the plan-then-execute paradigm has been demonstrated, large-scale, high-quality, and diverse intermediate representation annotations are lacking to fully unlock it.

**Paper Goals**: Build a complete intermediate representation ecosystem—from annotation tooling (RoboInter-Tool) to data (RoboInter-Data) to benchmarks (RoboInter-VQA) to model frameworks (RoboInter-VLA)—addressing the three bottlenecks of data, evaluation, and methodology in a unified suite.

## Method

### Overall Architecture

The RoboInter suite consists of four major components:

1. **RoboInter-Tool**: A lightweight PyQt5-based GUI supporting semi-automatic per-frame annotation, integrated with SAM2 for object segmentation and tracking.
2. **RoboInter-Data**: Dense per-frame annotation data covering 230K+ episodes, 571 scenes, and 10+ intermediate representation types.
3. **RoboInter-VQA**: An embodied VQA benchmark and dataset with 9 spatial and 20 temporal question categories.
4. **RoboInter-VLA**: A plan-then-execute framework with a VLM Planner and Executor, supporting three architectural variants.

### Key Design 1: Multi-Level Intermediate Representation Annotation

Annotations span three levels:

- **Task level**: Tasks decomposed into subtasks → 15 predefined primitive skills (Pick, Place, Push, Pull, etc.) → segment-level and video-level language annotations.
- **Object level**: Object grounding annotations via SAM2 + human review → 61M frames of object grounding, 190K affordance boxes, and placement proposals.
- **Execution level**: End-effector 2D trajectories (70M frames of trace annotations), contact points, 6D grasp poses, and gripper bounding boxes.

All annotations are temporally aligned with actions, states, and both third-person and wrist-view observations.

### Key Design 2: F-CoT — Flexible Chain-of-Thought Bridging Planning and Execution

The paper introduces Flexible Chain-of-Thought (F-CoT), composed of combinations of intermediate representations:

- Serves as VQA supervision signals for Planner training.
- Serves as action-aligned guidance signals for the Executor.
- Supports text form (Te-Modular) and visual prompt form (Im-Modular).
- Users may flexibly select subsets (e.g., subtask + trace, affordance + skill) according to task requirements.

### Key Design 3: Three Plan-then-Execute VLA Variants

| Variant | Architecture | Intermediate Representation Usage |
|---|---|---|
| IC-E2E | VLM as feature extractor for Executor | Implicit (via pretrained weights) |
| EC-E2E | VLM jointly generates intermediate representations and actions | Explicit (joint optimization of CoT + action) |
| Modular | Planner and Executor decoupled | Explicit (Planner output → Executor conditioning) |

The Executor adopts a Qwen2.5-VL backbone + DiT action head + information aggregator (compressing all token hidden states into a controllable-length conditioning feature).

## Key Experimental Results

### Main Results: VLM Capability Evaluation on Third-Party Benchmarks

| Model | Where2Place ↑ | RoboRefIt ↑ | RoboVQA ↑ | RefCOCOg ↑ |
|---|:---:|:---:|:---:|:---:|
| QwenVL2.5-7B | 18.9% | 75.8% | 38.4 | 87.2% |
| RoboBrain-2.0-7B | 63.6% | 8.8% | 31.6 | 62.9% |
| **RoboInter-Qwen-7B** | **65.8%** | **85.6%** | **74.4** | **88.4%** |
| RoboInter-LLaVAOV-7B | 66.3% | 89.3% | 74.5 | 87.3% |

RoboInter-VLM substantially outperforms baselines across all embodied benchmarks while maintaining stable general-purpose capabilities (TextVQA 83.0, MME 2281).

### Open-Loop Executor Evaluation

| Method | OLS@0.1 | OLS@0.05 | OLS@0.01 | mOLS |
|---|:---:|:---:|:---:|:---:|
| Vanilla | 0.6793 | 0.3608 | 0.0189 | 0.3086 |
| IC-E2E | 0.6984 | 0.3810 | 0.0204 | 0.3218 |
| EC-E2E | 0.7049 | 0.3930 | 0.0314 | 0.3340 |
| **Te-Modular** | **0.7124** | **0.4133** | **0.0584** | **0.3543** |
| Oracle+Executor | 0.7511 | 0.4640 | 0.0587 | 0.3861 |

Te-Modular (textual F-CoT + modular architecture) achieves the best results among learned methods, demonstrating that decoupling planning and execution allows each component to specialize more effectively.

### Ablation Study: Contribution of Intermediate Representation Types

| Intermediate Representation Combination | mOLS |
|---|:---:|
| Vanilla (no intermediate representation) | 0.3086 |
| + Subtask | 0.3146 |
| + Subtask + Primitive Skill | 0.3159 |
| + ... + Object Box | 0.3289 |
| + ... + Gripper Box | 0.3391 |
| + ... + Affordance | 0.3435 |
| + ... + **Trace** | **0.3861** |

Coarse-grained representations (Subtask, Skill) yield marginal gains, spatially precise signals (Object Box, Affordance) contribute significantly, and **Trace provides the largest gain** (dense temporal information directly constraining action generation).

### Real-World Closed-Loop Evaluation

| Model | ID Avg. Success | OOD Avg. Success | ID→OOD Drop |
|---|:---:|:---:|:---:|
| OpenVLA | 45.0% | 23.3% | 21.7% |
| π₀ | 63.3% | 45.0% | 18.3% |
| Vanilla | 65.0% | 38.3% | 26.7% |
| IC-E2E | 77.3% | 58.3% | 19.0% |
| **EC-E2E** | 68.3% | **60.0%** | **8.3%** |

EC-E2E achieves the best OOD performance with the smallest degradation (only 8.3%), demonstrating that explicit intermediate representation reasoning provides stronger generalization robustness.

## Highlights & Insights

### Strengths

1. **Exceptional systematicity**: From annotation tooling to data to benchmarks to model frameworks, the suite provides an all-in-one infrastructure for intermediate representation research.
2. **Substantial scale**: 230K episodes + 571 scenes + 61M frames of grounding annotations, far exceeding prior work.
3. **Comprehensive experimentation**: Full coverage of open-loop, closed-loop, cross-platform, SimplerEnv, and ablation evaluations.
4. **Deep insights**: Systematic validation of the effects of intermediate representation granularity and architectural design choices (modular vs. end-to-end) on performance.

### Limitations & Future Work

1. VQA data is generated from templates and reassembled annotations, limiting diversity to the design space of annotation templates.
2. Real-world experiments cover only 4 tasks; generalization to more complex, long-horizon tasks remains unexplored.
3. The modular Planner incurs high inference latency (~2.4s), requiring engineering optimizations such as asynchronous inference for practical deployment.

## Rating

⭐⭐⭐⭐⭐

RoboInter represents a milestone in robotic intermediate representation research. It not only provides the largest and most diverse multi-type intermediate representation dataset to date, but also establishes a complete experimental platform for the plan-then-execute paradigm through systematic VQA benchmark and VLA framework design. The ablation study clearly reveals a value hierarchy of intermediate representations—Trace > Affordance > Object Box > Subtask—an insight of significant guidance value for future embodied AI research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](../../CVPR2026/robotics/lada_robotic_manipulation.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](../../CVPR2026/robotics/language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[ICLR 2026\] When would Vision-Proprioception Policies Fail in Robotic Manipulation?](when_would_vision-proprioception_policies_fail_in_robotic_manipulation.md)
- [\[ICLR 2026\] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)

<!-- RELATED:END -->
