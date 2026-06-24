---
title: >-
  [Paper Note] PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation
description: >-
  [ICCV 2025][Robotics][Geometric Primitive Extraction] This paper proposes PASG (Primitive-Aware Semantic Grounding), a closed-loop framework that dynamically couples low-level geometric features with high-level task semantics through automated geometric primitive extraction (keypoints, functional axes, principal axes) and VLM-driven semantic anchoring. PASG achieves near-human-annotation performance on robotic manipulation tasks, and introduces the Robocasa-PA benchmark along…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Geometric Primitive Extraction"
  - "Semantic Anchoring"
  - "Robotic Manipulation"
  - "VLM"
  - "Closed-Loop Framework"
date: 2026-05-08
content_hash: f6a16760ca0af11b
---

# PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation

**Conference**: ICCV 2025
**arXiv**: [2508.05976](https://arxiv.org/abs/2508.05976)  
**Code**: None  
**Area**: Robotic Manipulation / Object Detection
**Keywords**: Geometric Primitive Extraction, Semantic Anchoring, Robotic Manipulation, VLM, Closed-Loop Framework

## TL;DR

This paper proposes PASG (Primitive-Aware Semantic Grounding), a closed-loop framework that dynamically couples low-level geometric features with high-level task semantics through automated geometric primitive extraction (keypoints, functional axes, principal axes) and VLM-driven semantic anchoring. PASG achieves near-human-annotation performance on robotic manipulation tasks, and introduces the Robocasa-PA benchmark along with the fine-tuned model Qwen2.5VL-PA.

## Background & Motivation

Robotic manipulation has long suffered from a **semantic-geometric gap between high-level task semantics and low-level geometric features**:

- VLMs excel at generating perceptual visual representations but lack semantic grounding capability in canonical space.
- Existing methods (e.g., ReKep, CoPa) rely on VLMs to detect primitives at the task level but lack validation mechanisms, causing detection errors to propagate and significantly degrade success rates.
- Manual annotation of geometric primitives is costly and limits generalizability.
- Existing frameworks have incomplete primitive definitions — typically including only keypoints while neglecting important directional information such as principal axes, leading to failures in grasping and transport tasks.

**Core Problem**: How to construct a unified framework that automatically extracts geometric primitives from objects and establishes dynamic mappings to task semantics?

## Method

### Overall Architecture

PASG consists of four core stages:
1. Automated geometric primitive extraction (VFM + topological analysis)
2. Semantic primitive identification (VLM inference of possible manipulation tasks and associated primitives)
3. Visual-semantic alignment (mapping semantic primitives to geometric detection results)
4. Dynamic self-refine matching (closed-loop optimization to correct missed or erroneous detections)

### Key Designs

1. **Semantic Primitive Definition**:

    - Each interaction primitive is defined as a triplet $(E, S, F)$: geometric entity, structural description, and functional role.
    - **Point primitives $\mathcal{P}$**: anchor points (alignment reference positions), grasp points (optimal end-effector grasping positions), and manipulation points (positions that trigger mechanisms).
    - **Axis primitives $\mathcal{A}$**: principal axes (geometric/symmetry axes of the object), functional axes (functional directions), and approach axes (end-effector approach directions).
    - **Design Motivation**: Unifying point and axis primitives covers all necessary spatial constraints in manipulation tasks.

2. **Automated Geometric Primitive Extraction**:

    - Semantic SAM is employed for multi-granularity semantic segmentation.
    - **Keypoint extraction**: Center points, corner points, and intersections of PCA axes with object boundaries are extracted from segmentation masks.
    - **Keypoint filtering**: DBSCAN clustering for redundancy removal + farthest point sampling to preserve globally representative features.
    - **Principal axis calibration**: The Z-axis is defined by connecting mask centroid projections from top and bottom views; X/Y axes are generated orthogonally.
    - Cross-view color normalization ensures consistent axis visualization.

3. **Dynamic Self-Refine Matching Mechanism**:

    - Core closed-loop algorithm: segmentation → alignment → detection → resampling.
    - When matching confidence falls below 0.5 or a primitive is labeled NONE, finer-grained segmentation resampling is triggered.
    - The multi-granularity segmentation hierarchy of Semantic SAM is leveraged for adaptive quality improvement.
    - Achieves a 98% matching success rate on the dataset, effectively preventing error propagation.
    - Maximum of $\tau_{max} = 5$ iterations.

### Loss & Training

- **LoRA** parameter-efficient fine-tuning is applied to the VLM (Qwen2.5VL-PA).
- Training data is automatically generated by the PASG pipeline: 5,583 samples for fine-tuning, 1,396 in-distribution test samples, and 1,364 out-of-distribution test samples.
- Three task types: type identification, task association, and task-to-primitive mapping.
- Benchmark evaluation uses multiple-choice accuracy.

## Key Experimental Results

### Main Results

**RoboTwin Manipulation Task Success Rate (%)**

| Method | Hammering | Container Placement | Dual-Bottle Pickup | Empty Cup Placement | Apple Pickup | Shoe Placement | Average |
|--------|-----------|---------------------|-------------------|---------------------|--------------|----------------|---------|
| Human Annotation | 79.0 | **93.0** | **95.0** | 73.0 | **85.0** | **83.0** | **84.67** |
| **PASG** | **82.0** | 89.0 | 70.0 | **76.0** | 81.0 | 69.0 | 77.83 |

**Spatial Semantic Reasoning Benchmark Accuracy (%)**

| Model | In-distribution Overall | Out-of-distribution Overall |
|-------|-------------------------|-----------------------------|
| GPT-4V | 39.04 | 39.00 |
| GPT-4O | 43.19 | 41.79 |
| Qwen-2.5VL | 43.91 | 43.33 |
| RoboPoint | 14.40 | 15.32 |
| **Qwen2.5VL-PA** | **77.79** | **79.69** |

### Ablation Study

**Data Efficiency Experiment** (Fine-tuning data size vs. accuracy gain)

| Training Data Ratio | # Samples | In-distribution Gain | Out-of-distribution Gain |
|--------------------|-----------|---------------------|--------------------------|
| 1% | 55 | Marginal | Marginal |
| 5% | 279 | ~+10% (absolute) | ~+10% (absolute) |
| 10% | 558 | ~+20% (absolute) | ~+20% (absolute) |
| 50% | 2,791 | Near full performance | Near full performance |
| 100% | 5,583 | +33.9% (absolute) | +36.4% (absolute) |

**Dataset Validation**:
- Semantic recognition accuracy: 91.6%
- Strict alignment accuracy: 75.8%
- Alignment validity: 91.5%
- Self-refine matching success rate: 98%

### Key Findings

- PASG **surpasses human annotation** on hammering and empty cup placement tasks (82% vs. 79%; 76% vs. 73%), indicating that the diversity of automated annotations leads to more flexible manipulation strategies.
- PASG generates richer and more diverse interaction primitives than human annotation — human annotators, constrained by cost, label only a few optimal points, whereas PASG generates a broader set of semantically meaningful primitives.
- Fine-tuned Qwen2.5VL-PA achieves approximately 10% absolute accuracy improvement with only 5% of the training data (20.6% relative gain).
- The performance gap between in-distribution and out-of-distribution settings remains within ±2%, demonstrating strong cross-domain generalization.
- PASG is the first closed-loop framework that simultaneously incorporates automated primitive extraction, semantic anchoring, and self-correction.

## Highlights & Insights

- **Object-centric rather than task-centric**: The framework first comprehensively understands an object's geometric and functional properties before mapping them to specific tasks, offering greater generality than existing task-level methods (ReKep, CoPa).
- **Closed-loop self-correction** cleverly leverages Semantic SAM's multi-granularity segmentation to adaptively improve detection quality.
- **Semantic hierarchy**: From low-level descriptions (e.g., "bottle neck edge") to high-level intent (e.g., "critical position for aligning the bottle during pouring"), a complete semantic chain is provided.
- Validates the possibility that **automated annotation can match or even surpass human annotation**.

## Limitations & Future Work

- PASG falls significantly short of human annotation on dual-bottle pickup and shoe placement tasks (70% vs. 95%; 69% vs. 83%), indicating remaining gaps in tasks requiring precise coordination.
- GPT-4o still faces challenges in complex spatial semantic understanding.
- Validation is limited to simulated environments (RoboTwin); large-scale deployment to real robotic scenarios has not been conducted.
- The dataset is sourced from RoboCasa and Objaverse, which may lack objects from industrial settings.
- Routing accuracy depends on GPT-4o's reasoning quality, introducing a degree of error propagation risk.

## Related Work & Insights

- **ReKep**: Uses VFM+VLM to detect keypoints at the task level but lacks validation mechanisms and semantic coupling.
- **CoPa**: Incorporates functional axes but similarly lacks adaptive correction.
- **OmniManip**: Employs constrained optimization and scene rendering for VLM validation, but at higher computational cost.
- **SOFAR**: Proposes direction-aware spatial understanding modules but relies on predefined directional priors rather than automatic extraction.
- **SoM** (Set-of-Mark): Inspires the annotation visualization scheme.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First closed-loop framework to integrate geometric primitive extraction, semantic anchoring, and self-correction; comprehensive object-level semantic primitive definition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers manipulation task evaluation, VQA benchmarks, data efficiency experiments, and human validation; however, real-robot experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with sufficient methodological detail; Table 1 intuitively illustrates differences from existing methods.
- **Value**: ⭐⭐⭐⭐ — Provides a practical and scalable solution for bridging semantics and geometry in robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](../../ICML2025/robotics/closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[ICLR 2026\] World-In-World: World Models in a Closed-Loop World](../../ICLR2026/robotics/world-in-world_world_models_in_a_closed-loop_world.md)
- [\[AAAI 2026\] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](../../AAAI2026/robotics/semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)
- [\[CVPR 2026\] Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of Humanoid Control](../../CVPR2026/robotics/iterative_closed-loop_motion_synthesis_for_scaling_the_capabilities_of_humanoid_.md)

</div>

<!-- RELATED:END -->
