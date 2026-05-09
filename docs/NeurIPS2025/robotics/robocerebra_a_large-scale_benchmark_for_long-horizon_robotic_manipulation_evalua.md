---
title: >-
  [Paper Note] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation
description: >-
  [NeurIPS 2025][Robotics][Long-horizon manipulation] This paper proposes RoboCerebra, a long-horizon robotic manipulation benchmark comprising 1,000 human demonstration trajectories (averaging 2,972 steps, approximately 6× longer than existing benchmarks). Through a hierarchical planning and execution framework and a multi-dimensional evaluation protocol, it systematically assesses VLMs across three System 2 cognitive dimensions: planning, reflection, and memory.
tags:
  - NeurIPS 2025
  - Robotics
  - Long-horizon manipulation
  - benchmark evaluation
  - System 2 reasoning
  - hierarchical planning
  - VLM evaluation
date: 2026-05-08
content_hash: 856836f2db8f20e9
---

# RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation

**Conference**: NeurIPS 2025  
**arXiv**: [2506.06677](https://arxiv.org/abs/2506.06677)  
**Code**: [robocerebra.github.io](https://robocerebra.github.io)  
**Area**: Robotics  
**Keywords**: Long-horizon manipulation, benchmark evaluation, System 2 reasoning, hierarchical planning, VLM evaluation

## TL;DR

This paper proposes RoboCerebra, a long-horizon robotic manipulation benchmark comprising 1,000 human demonstration trajectories (averaging 2,972 steps, approximately 6× longer than existing benchmarks). Through a hierarchical planning and execution framework and a multi-dimensional evaluation protocol, it systematically assesses VLMs across three System 2 cognitive dimensions: planning, reflection, and memory.

## Background & Motivation

Current applications of VLMs in robotic manipulation largely remain at the fast-reactive System 1 level—VLA models directly map multimodal inputs to low-level control signals. However, the true strength of VLMs lies in semantic abstraction, relational understanding, and contextual reasoning, which correspond precisely to slow-thinking System 2 processes (long-horizon planning, subgoal decomposition, and adaptive adjustment). Deficiencies in existing benchmark datasets directly impede research on System 2 capabilities:

**Insufficient temporal scale**: "Long-horizon" benchmarks such as LIBERO-Long and RoboCasa typically contain only 2–5 subtasks with fewer than 500 action steps, making it difficult to evaluate genuine long-horizon reasoning demands such as memory maintenance and temporal abstraction.

**Lack of dynamic scenes and memory requirements**: In real-world settings, objects are moved, occluded, and change state; robots must remember what was inside a previously explored cabinet or where an object was placed. Existing benchmarks include almost no such scenarios.

**Unidimensional evaluation**: Most benchmarks rely solely on binary task success rate, which cannot distinguish performance across specific cognitive dimensions such as planning capability, perceptual judgment, and memory utilization.

RoboCerebra aims to fill this gap by constructing a genuinely long-horizon task environment that comprehensively evaluates VLMs as System 2 high-level reasoners.

## Method

### Overall Architecture

RoboCerebra consists of three major components: (1) a large-scale simulation dataset generated via LLMs and executed by humans; (2) a Hierarchical Planning and Execution (HPE) framework in which a VLM handles high-level planning and a VLA handles low-level control; and (3) a multi-dimensional evaluation protocol that fixes System 1 to isolate the assessment of different VLMs' System 2 capabilities.

### Key Designs

1. **Top-down data generation pipeline**:

    - **Cascaded task generation**: Objects are randomly sampled from the LIBERO object library and converted into structured representations (category, functionality, spatial context), which are fed to GPT to generate high-level task descriptions (e.g., "heat milk in the microwave"), subsequently decomposed into subtask sequences. Affordance-aware prompt design ensures temporal consistency and physical feasibility.
    - **Scene initialization with dual validation**: Structured plans are parsed into simulator-executable code and verified through a dual-loop process of symbolic checking (object state consistency) and visual-language validation (GPT-4o judges spatial plausibility from multi-view renderings).
    - **Human demonstration and annotation**: Human operators execute tasks in simulation, producing diverse action trajectories annotated with fine-grained subtask temporal boundaries. A total of 400 hours were invested in trajectory collection and 200 hours in quality verification.

2. **Six subtask categories**:

    - **Ideal**: Static, fully observable baseline
    - **Memory Exploration**: Requires active environment exploration to build internal representations (e.g., inspecting cabinet compartment contents)
    - **Memory Execution**: Requires leveraging memory to accomplish goals after perceptual cues are removed
    - **Random Disturbance**: Introduces unexpected environmental changes (object displacement, collisions)
    - **Observation Mismatching**: Requires handling plan–perception inconsistencies
    - **Mix**: Combines memory and dynamic factors, requiring continuous replanning under uncertainty

3. **Hierarchical Planning and Execution (HPE) Framework**:

    - **VLM Planner**: Processes low-frequency observations, generates and updates subgoal-level subtasks, and stores them in a memory store. During training, contrastive learning on video–instruction pairs annotated with success/failure labels enables the VLM to evaluate task progress.
    - **VLA Controller**: Based on OpenVLA, trained at the subtask level to consume high-frequency visual inputs and execute fine-grained actions.
    - **Memory Store**: A shared state connecting both modules; the VLM updates memory and the next subgoal upon detecting subtask completion or deviation.

### Loss & Training

- VLA training: (image, instruction, action) triplets are sampled from long-horizon demonstrations; continuous actions are discretized into token sequences and trained with next-token prediction. 200K steps, batch size 64, 256×256 input resolution.
- VLM training: Subtask-level video clips paired with success/failure labels are used for contrastive learning, enabling the VLM to assess task completion status.

## Key Experimental Results

### Main Results

**Long-horizon task performance across different System 1 + System 2 combinations**

| Method | Avg. SR | Random | Obs. Mis. | Mem. Exp. | Mem. Exe. | Mix | Ideal |
|--------|---------|--------|-----------|-----------|-----------|-----|-------|
| OpenVLA (System 1 only) | 2.00% | 4.59% | 1.35% | 0.18% | 1.86% | 0.00% | 4.05% |
| OpenVLA* (fine-tuned) | 4.57% | 7.84% | 8.65% | 1.06% | 2.06% | 0.00% | 7.84% |
| Planner + OpenVLA* | 16.04% | 18.63% | 19.45% | 8.04% | 16.69% | 11.48% | 21.92% |
| HPE Framework | **16.55%** | 18.63% | 19.18% | **9.06%** | **17.83%** | **13.21%** | 21.10% |

**Comparison of different VLMs as System 2 Planners**

| Planner Model | Avg. SR | Mem. Exp. | Mix | Ideal |
|---------------|---------|-----------|-----|-------|
| GPT-4o | **16.04%** | **8.04%** | **11.48%** | **21.92%** |
| GPT-4o-Blind | 15.10% | 7.02% | 10.48% | 20.00% |
| Qwen2.5-VL | 11.19% | 2.63% | 6.67% | 16.71% |
| LLaVA-Next-Video | 11.37% | 1.07% | 3.70% | 19.73% |
| GT-plan (upper bound) | 25.16% | 19.47% | 19.26% | 31.23% |

### Ablation Study

**Multi-dimensional System 2 evaluation**

| Model | Planning Acc.↑ | Reflection↑ | SR↑ | Plan Length↓ | Plan Efficiency↑ |
|-------|---------------|-------------|-----|-------------|-----------------|
| GPT-4o | **68.33%** | 32.66% | **16.04%** | 10.67 | **1.50** |
| GPT-4o-Blind | 61.37% | 0.00% | 15.10% | 10.73 | 1.41 |
| Qwen2.5-VL-7B | 44.67% | 47.74% | 11.19% | 8.30 | 1.34 |
| Qwen2.5-VL-7B-SFT | 30.00% | **66.83%** | 9.33% | 6.95 | 1.32 |

### Key Findings

- **System 1 completely fails on long-horizon tasks**: Even fine-tuned OpenVLA achieves only 4–8% success rate under the Ideal setting and fails entirely (0%) on Mix tasks, confirming the indispensability of System 2.
- **System 2 gains are more pronounced on complex tasks**: The HPE framework improves Mix task performance (requiring memory + dynamic adaptation) from 0% to 13.21%, though on simpler Ideal tasks the reasoning overhead may cause it to underperform a pure planner approach.
- **Planning capability outweighs perceptual capability**: GPT-4o maintains 15.10% success rate even without visual input (Blind mode), while fine-tuning Qwen2.5-VL improves reflection ability (66.83%) but reduces planning accuracy (30%), ultimately yielding a lower overall success rate. This indicates that planning reasoning is more critical than perceptual judgment in current long-horizon tasks.
- **A 9% gap remains relative to GT-plan**: This highlights that insufficient environmental interaction and the visual domain gap remain the primary bottlenecks for VLMs.

## Highlights & Insights

- **Breakthrough in data scale**: An average trajectory length of 2,972 steps—6× longer than existing long-horizon benchmarks—genuinely challenges long-term memory and multi-step reasoning.
- **Decoupled evaluation of cognitive dimensions**: Fixing System 1 to isolate the assessment of System 2 planning, reflection, and memory capabilities offers a methodological contribution worth adopting.
- **System 1 + System 2 collaborative paradigm**: The work explicitly establishes that VLMs should serve as high-level reasoners rather than direct controllers, providing a clear architectural reference for robotic AI systems.

## Limitations & Future Work

- The interaction between System 1 and System 2 remains limited, lacking fine-grained bidirectional feedback mechanisms.
- The evaluation protocol could be further extended with execution-level signals, such as subtask ordering correctness and failure recovery capability.
- Transfer to the real world has not been validated, though the authors argue that since System 2 focuses on high-level reasoning rather than low-level control, the sim-to-real gap has limited impact.
- The data generation pipeline relies on GPT and human operators, making scaling to additional environments and task types costly.

## Related Work & Insights

- In contrast to LIBERO (≤500 steps), CALVIN (34 short-horizon tasks), and RoboCasa (LLM-generated but without human trajectories), RoboCerebra represents a qualitative leap in temporal span and annotation richness.
- Inspired by Kahneman's System 1/System 2 theory, the work introduces a cognitive science perspective for the design of robotic manipulation systems.
- Insight: Future work could explore enabling System 2 VLMs to receive failure feedback from System 1 during execution, achieving tighter closed-loop collaboration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematic construction of a long-horizon manipulation benchmark and the decoupled cognitive-dimension evaluation design are original contributions
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Cross-model comparison of multiple VLMs, multi-dimensional evaluation, and validation across different VLA backends, though absolute performance remains low
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed descriptions of dataset construction and evaluation methodology
- **Value**: ⭐⭐⭐⭐⭐ Fills an important gap in long-horizon manipulation evaluation and provides empirical grounding for the role of VLMs in robotics

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks](rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](../../CVPR2026/robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)
- [\[NeurIPS 2025\] SutureBot: A Precision Framework & Benchmark for Autonomous End-to-End Suturing](suturebot_a_precision_framework_benchmark_for_autonomous_end-to-end_suturing.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)

<!-- RELATED:END -->
