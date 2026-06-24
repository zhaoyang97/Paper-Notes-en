---
title: >-
  [Paper Note] ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models
description: >-
  [CVPR 2025][VLM Reasoning][Spatial Reasoning] This paper proposes Espire, a simulation-based diagnostic benchmark for embodied spatial reasoning. It decomposes VLM evaluation into localization and execution phases, systematically assessing the capabilities of VLMs across multiple spatial reasoning dimensions and granularities through a fully generative paradigm.
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "Spatial Reasoning"
  - "Embodied AI"
  - "VLM Benchmark"
  - "Robot Manipulation"
  - "6-DoF"
date: 2026-05-08
content_hash: 007aa1b482ba474a
---

# ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2603.13033](https://arxiv.org/abs/2603.13033)  
**Code**: [github.com/spatigen/espire](https://github.com/spatigen/espire)  
**Area**: Multimodal VLM  
**Keywords**: Spatial Reasoning, Embodied AI, VLM Benchmark, Robot Manipulation, 6-DoF

## TL;DR

This paper proposes Espire, a simulation-based diagnostic benchmark for embodied spatial reasoning. It decomposes VLM evaluation into localization and execution phases, systematically assessing the capabilities of VLMs across multiple spatial reasoning dimensions and granularities through a fully generative paradigm.

## Background & Motivation

**Background**: Although VLMs have made progress in spatial cognition and extensive work has been dedicated to enhancing their spatial intelligence (such as SpatialRGPT, RoboSpatial, etc.), evaluation methods lag behind model development.

**Limitations of Prior Work**: Existing benchmarks are mostly in the form of static VQA, which rely on distractor options and are prone to bias. Additionally, they are decoupled from actual robotic deployment scenarios, lacking evaluation of the execution phase.

**Key Challenge**: Discriminative VQA evaluation fails to reflect the "understand-then-act" capability of VLMs in real embodied scenarios, whereas real-world evaluations rely on specific hardware, making them difficult to scale and replicate.

**Goal**: To design a systematic and scalable benchmark that evaluates both the spatial localization and action execution capabilities of VLMs while supporting fine-grained diagnostic analysis.

**Key Insight**: Utilizing the Isaac Sim environment, physical robot tasks are decomposed into localization (2D point generation) and execution (6-DoF pose generation), unified into a fully generative paradigm.

**Core Idea**: By combining systematic spatial reasoning task design with physical simulation environments, this work constructs the first embodied spatial reasoning benchmark that simultaneously covers localization and execution, supporting multi-granularity diagnostic analysis.

## Method

### Overall Architecture

Espire is built on top of Isaac Sim, providing tabletop and bookshelf environments that cover pick and place tasks. Each task is decomposed into:
- **Localization**: Generating 2D points on the scene image to locate target objects.
- **Execution**: Generating a 6-DoF target pose (position + orientation) in SE(3).

The evaluation metrics are accuracy (localization success rate) and acceptance rate (the rate of physically feasible poses), the latter of which is verified using the cuRobo motion planner.

### Key Designs

**1. Systematic Design of Spatial Reasoning Tasks**
- **Function**: Three core factors are defined—spatial dimension $S$ (relationship/distance/attribute/orientation), frames of reference $F$ (relative/intrinsic/absolute), and referent objects $O$ (oriented/unoriented)—whose combination $C=(S,F,O)$ defines the task context.
- **Mechanism**: Instructions are represented by functional programs (e.g., `filterRel(left, unique(filter(book, G)))`), enabling flexible control over reasoning complexity.
- **Design Motivation**: To achieve systematic coverage of various spatial dimensions and granularities, such as progressive granularities from "left" to "leftmost" to "11 o'clock direction".

**2. Instruction Families and Template System**
- **Function**: Defining 148 spatial reasoning types distributed across 65 instruction families (31 pick + 34 place), with 3-4 handwritten templates per family to increase linguistic diversity.
- **Mechanism**: Diverse instructions and their corresponding functional programs are automatically generated through variable binding (e.g., binding `[R]` to Closest/Furthest).
- **Design Motivation**: Functional programs can be executed on 3D scene graphs to output the ground-truth, supporting scalable and automatic task generation.

**3. Simulation Environment Design**
- **Function**: Comprising tabletop (pick task) and shelf (place task) scenes, supporting randomized object layouts, lighting, textures, etc.
- **Mechanism**: The environment is rendered from randomized 3D scene graphs using photorealistic assets, realistic material textures, and randomized lighting/camera poses to mitigate the sim-to-real gap.
- **Design Motivation**: Categorizing tasks by difficulty levels (easy/medium/hard) to cover different degrees of clutter, ensuring a systematic and comprehensive evaluation.

### Loss & Training

As a benchmark paper, this work does not introduce a training strategy. The evaluation uses a zero-shot approach: given the scene image and a natural language instruction, the VLM directly generates 2D points or 3D poses, allowing up to 3 attempts for localization and 5 attempts for execution.

## Key Experimental Results

### Main Results

| Model | Pick Acc(%) | Pick Accept(%) | Pick Succ(%) | Place Acc(%) | Place Accept(%) | Place Succ(%) |
|---|---|---|---|---|---|---|
| Gemini2.5-Pro | 57.72 | 63.93 | 34.06 | 50.61 | 28.36 | 5.68 |
| InternVL3-78B | 28.31 | 63.01 | 17.26 | 23.66 | 40.94 | 9.67 |
| RoboBrain2.0-7B | 57.72 | 18.81 | 10.87 | 50.70 | 15.68 | 8.64 |
| Qwen3-VL-30B | 54.43 | 62.56 | 32.15 | 45.54 | 43.47 | 20.00 |
| Qwen3-VL-8B | 47.03 | 63.20 | 29.32 | 35.71 | 37.31 | 12.41 |
| Qwen3-VL-235B | 51.96 | 52.79 | 26.76 | 47.42 | 41.22 | 19.34 |

### Ablation Study

| Spatial Dimension | Pick Avg. Acc(%) | Place Avg. Acc(%) |
|---|---|---|
| Attribute | 49.33 | 47.20 |
| Distance | 45.37 | 33.33 |
| Orientation | 54.02 | 37.17 |
| Relationship | 49.81 | 47.03 |

Difficulty levels: All models exhibit a general decline in performance across easy $\rightarrow$ medium $\rightarrow$ hard levels. For example, Gemini 2.5-Pro's Pick Acc drops from 60.78% to 52.04%.

### Key Findings

1. **Localization vastly outperforms execution**: The localization accuracy of all models is significantly higher than their final success rate, indicating that while passive spatial understanding is acceptable, action-oriented spatial reasoning (especially 3D rotational geometry) is severely lacking.
2. **Place is generally harder than Pick**: Place tasks require considering constraints and occlusions in the target space.
3. **Model size does not equate to performance**: Qwen3-VL-30B (with 3B active parameters) outperforms Qwen3-VL-8B and 235B across most metrics.
4. **Distance is the weakest dimension**: All models show the worst performance in distance-based reasoning, demonstrating that current VLMs lack precise distance comprehension capabilities.
5. **Reflection helps localization but does not necessarily assist execution**: Reflective mechanisms can improve localization but may degrade execution performance, as 3D rotation comprehension remains the bottleneck for execution.

## Highlights & Insights

- The first fully generative embodied spatial reasoning benchmark that unifies localization and execution, pushing evaluation from passive understanding to active action.
- Systematic task design supporting fine-grained diagnostics across multiple dimensions and granularities, covering 148 spatial reasoning types.
- An automated functional program + scene graph mechanism that enables scalable task generation.
- Revealing critical shortcomings of VLMs in understanding 3D rotational geometry, pointing out directions for future data construction.

## Limitations & Future Work

- A gap still exists between simulation and the real world, which has not been completely eliminated despite various mitigation measures.
- The current reasoning depth is limited to 3-hop, which does not fully test compositional reasoning capabilities.
- The scene types are limited to tabletop and bookshelf, offering restricted coverage.
- The execution phase relies on ground-truth depth, which does not fully simulate real-world conditions.

## Related Work & Insights

- Compared to point generation works like Where2Place and SpatialVQA, Espire incorporates the execution dimension.
- Compared to simulation benchmarks like EmbodiedBench and LIBERO, Espire provides systematic spatial reasoning task designs and tool-free evaluations.
- **Insight**: Future work can construct training datasets containing rotational reasoning data to enhance the 6-DoF capabilities of VLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified fully generative localization + execution evaluation paradigm and the systematic spatial reasoning factor design are highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluates multiple leading VLMs with rich analysis dimensions (spatial dimensions, difficulty levels, reflection, etc.).
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, precise problem definitions, and reasonable chart/table designs.
- **Value**: ⭐⭐⭐⭐ Fills a significant gap in the evaluation of embodied spatial reasoning, offering diagnostic insights that are highly valuable to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/vlm_reasoning/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/vlm_reasoning/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/vlm_reasoning/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[NeurIPS 2025\] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](../../NeurIPS2025/vlm_reasoning/roborefer_towards_spatial_referring_with_reasoning_in_vision-language_models_for.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](../../ICLR2026/vlm_reasoning/spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)

</div>

<!-- RELATED:END -->
