---
title: >-
  [Paper Note] Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences
description: >-
  [ICLR 2026][Multimodal VLM][VLM spatial reasoning] This paper systematically evaluates VLMs' spatial reasoning capabilities over robot motion trajectories, proposing four image-querying methods that enable VLMs to select optimal motion paths based on user natural language descriptions. Results show that Qwen2.5-VL achieves 71.4% zero-shot accuracy, with smaller models achieving significant gains after fine-tuning.
tags:
  - ICLR 2026
  - Multimodal VLM
  - VLM spatial reasoning
  - robot motion planning
  - motion preferences
  - path selection
  - vision-language model evaluation
date: 2026-05-08
content_hash: d8d1dd85a424e61b
---

# Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences

**Conference**: ICLR 2026
**arXiv**: [2603.13100](https://arxiv.org/abs/2603.13100)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: VLM spatial reasoning, robot motion planning, motion preferences, path selection, vision-language model evaluation

## TL;DR

This paper systematically evaluates VLMs' spatial reasoning capabilities over robot motion trajectories, proposing four image-querying methods that enable VLMs to select optimal motion paths based on user natural language descriptions. Results show that Qwen2.5-VL achieves 71.4% zero-shot accuracy, with smaller models achieving significant gains after fine-tuning.

## Background & Motivation

In human-robot interaction scenarios, users need to express motion preferences in natural language, such as "move away from the window" or "follow a curved path." The semantic knowledge and spatial reasoning capabilities of VLMs may enhance the generalization of robot planners to novel tasks.

However, critical gaps remain in existing research:

**Motion preferences are overlooked**: Existing VLM-robot research primarily focuses on task planning (what to do) rather than motion planning (how to move)—the topological properties and stylistic preferences of paths have never been systematically studied.

**Spatial reasoning capability is unknown**: Whether VLMs can genuinely understand spatial constraints such as "far from/close to an object" or "curved/straight path" lacks quantitative evaluation.

**Integration pathways are unclear**: The optimal approach (image-querying strategy) for integrating VLMs into motion planning pipelines has not been explored.

## Method

### Overall Architecture

A "generate–score–select" pipeline is proposed to integrate VLMs into motion planning:

1. **Diverse path generation**: BiRRT + PRM is used to sample $n=50$ candidate paths.
2. **Path clustering**: K-means clustering groups candidate paths, selecting the path closest to each cluster centroid.
3. **Image visualization**: Candidate paths are rendered as colored dot trajectories overlaid on scene images.
4. **VLM scoring**: The VLM is provided with the image and user instruction, and is prompted to score and select the best-matching path.

### Key Designs

**Four image-querying methods**:

| Query Method | Description | Query Count | Image Count |
|---|---|---|---|
| **Single-image trajectory** | All candidate paths drawn in different colors on one image | 1 | 1 |
| **Multi-image trajectory** | Each path on a separate image, queried sequentially | k | k |
| **Single-image + visual context** | VLM first generates an image description, then makes a selection | 2 | 1 |
| **Screenshot gallery** | Screenshot sequences of robot motion along each path | 1 | 1 (gallery) |

**Two categories of motion preferences**:

- **Object proximity preferences**: Describe spatial relationships with environmental objects, e.g., "stay away from the lamp," "pass between objects B and C."
- **Path style preferences**: Describe geometric properties of the path, e.g., "straight," "curved," "zigzag," "shortest path."

**Dataset construction**: 558 motion planning problems with language constraints are constructed in the iGibson simulation environment (126 navigation + 432 manipulation), with manually annotated ground-truth paths for each problem.

### Loss & Training

Evaluation is primarily conducted in the zero-shot setting. Fine-tuning experiments apply SFT (supervised fine-tuning) on 98 training samples for LLaVa1.5-7B and Qwen2.5-VL-7B, with a test set of 28 problems.

## Key Experimental Results

### Main Results

**Overall accuracy across query methods (navigation tasks, Qwen2.5-VL-72B)**:

| Query Method | Accuracy | Token Cost |
|---|---|---|
| Single-image trajectory | **71.4%** | 687.3 |
| Multi-image trajectory | ~55% | High |
| Single-image + visual context | ~68% | Medium |
| Screenshot gallery | Slightly above random | High |

**VLM performance on navigation tasks (single-query method)**:

| Model | Proximity Preference | Path Style | Overall |
|---|---|---|---|
| **Qwen2.5-VL-72B** | **74.4%** | **63.9%** | **71.4%** |
| GPT-4o | Lower | Lower | Below Qwen |
| LLaVa1.5 | Lowest | Lowest | ~Random |

**Manipulation task performance**:

| Model | Proximity Preference | Path Style |
|---|---|---|
| Qwen2.5-VL-72B | **66.3%** | 65.5% |
| GPT-4o | Lower | **69.5%** |

### Ablation Study

**Fine-tuning results (28 test problems)**:

| Model | Before Fine-tuning (Zero-shot) | After Fine-tuning | Gain |
|---|---|---|---|
| Qwen2.5-VL-7B | ~55% | **75%** | +20% |
| LLaVa1.5-7B | ~15% | ~75% | **+60%** |
| Qwen2.5-VL-3B | ~45% | ~55% | +10% |

**Relationship between token count and accuracy**: Accuracy increases approximately linearly with token count (i.e., image resolution). Within the 200–800 token range, both Qwen2.5-VL-7B and 72B exhibit linear trends.

### Key Findings

1. **Single-image is optimal**: Placing all paths in a single image enables relative comparison by the VLM, outperforming sequential per-image scoring.
2. **Proximity preferences outperform path style**: Proximity-type accuracy is consistently higher than path-style accuracy, as VLMs are more adept at understanding object spatial relationships than path geometric properties.
3. **Navigation outperforms manipulation**: Overall success rate on navigation tasks (71.4%) exceeds that on manipulation tasks (65.5%), as path differences in manipulation scenes are more subtle.
4. **Visual context is ineffective**: Generating additional visual descriptions provides no benefit for large models and may introduce redundant information that conflicts with the model's built-in context tracking.
5. **Fine-tuning is highly efficient**: Only 98 samples suffice to yield significant improvements, demonstrating that VLM architectures possess rapid adaptability to novel motion preferences.

## Highlights & Insights

1. **Novel problem perspective**: This is the first work to systematically study VLMs' spatial reasoning over robot motion trajectories (rather than goals/sub-goals), extending from task planning to motion quality control.
2. **Comprehensive evaluation framework**: A full cross-evaluation covering 4 query methods × 3 VLMs × 2 preference types × 2 task types.
3. **Practical findings**: The simplest single-image method achieves the best trade-off between performance and computational cost, providing clear guidance for real-world deployment.
4. **Strong fine-tuning potential**: Few-shot fine-tuning substantially improves small model performance, lowering the deployment barrier.

## Limitations & Future Work

1. **VLM hallucination**: Models occasionally select non-existent colored paths (e.g., selecting "red" when no red path is present).
2. **Poor recognition of shortest/longest paths**: These cases are precisely the strength of classical optimal planners (RRT*, PRM*), suggesting hybrid approaches should be explored.
3. **Simulation-only evaluation**: Visual complexity, occlusion, and dynamic changes in real robot environments are not addressed.
4. **Accuracy insufficient for direct deployment**: A 71.4% accuracy rate still requires human intervention in safety-critical scenarios.
5. **Dynamic obstacles and multi-step planning not considered**: The dataset consists entirely of single-step path selection in static scenes.

## Related Work & Insights

- **Distinction from SayCan/PaLM-E**: Those works focus on task planning (what to do), while this work focuses on motion properties (how to move); the two are complementary.
- **Distinction from MotionGPT**: MotionGPT generates human body motion, whereas this work uses VLMs to evaluate robot motion, employing images as an intermediate representation.
- **Distinction from IMPACT**: IMPACT identifies affordance objects, while this work modulates motion style and constraints.
- **Insight**: The hybrid paradigm of VLMs as motion quality evaluators combined with classical planners for candidate generation warrants further exploration.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic evaluation of VLMs' spatial reasoning over robot motion trajectories
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across query methods, VLM types, preference types, and task types
- Writing Quality: ⭐⭐⭐ — Clear structure, though some figures and tables could be improved
- Value: ⭐⭐⭐⭐ — Provides a solid benchmark and feasibility evidence for integrating VLMs into motion planning pipelines

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[CVPR 2026\] Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](../../CVPR2026/multimodal_vlm/self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](../../CVPR2026/multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)

</div>

<!-- RELATED:END -->
