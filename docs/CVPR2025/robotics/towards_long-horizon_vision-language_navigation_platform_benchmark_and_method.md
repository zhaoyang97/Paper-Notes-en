---
title: >-
  [Paper Note] Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method
description: >-
  [CVPR 2025][Robotics][Long-horizon navigation] This paper defines the Long-Horizon Vision-Language Navigation (LH-VLN) task, constructs the NavGen automatic generation platform and the LHPR-VLN benchmark (comprising 3,260 multi-stage tasks with an average of 150 steps), and proposes the MGDM method that achieves multi-stage navigation through short-term memory blurring, long-term memory retrieval, and CoT feedback, outperforming NaviLLM by 23% in the ISR metric.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Long-horizon navigation"
  - "multi-stage task"
  - "vision-language navigation"
  - "memory mechanism"
  - "benchmark evaluation"
date: 2026-05-08
content_hash: 080d739aecc512c9
---

# Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method

**Conference**: CVPR 2025  
**arXiv**: [2412.09082](https://arxiv.org/abs/2412.09082)  
**Code**: [https://hcplab-sysu.github.io/LH-VLN](https://hcplab-sysu.github.io/LH-VLN)  
**Area**: Robotics / Vision-Language Navigation  
**Keywords**: Long-horizon navigation, multi-stage task, vision-language navigation, memory mechanism, benchmark evaluation

## TL;DR

This paper defines the Long-Horizon Vision-Language Navigation (LH-VLN) task, constructs the NavGen automatic generation platform and the LHPR-VLN benchmark (comprising 3,260 multi-stage tasks with an average of 150 steps), and proposes the MGDM method that achieves multi-stage navigation through short-term memory blurring, long-term memory retrieval, and CoT feedback, outperforming NaviLLM by 23% in the ISR metric.

## Background & Motivation

**Background**: Vision-Language Navigation (VLN) enables agents to navigate in 3D environments guided by natural language instructions. Existing benchmarks (e.g., R2R, VLN-CE) feature an average path length of only 55 steps, and instructions involve only a single target—well below the requirements of real-world scenarios.

**Limitations of Prior Work**: Navigation in real-world scenarios is usually multi-stage (e.g., "go to the kitchen to fetch a cup, then put it on the table in the living room"), which involves long-horizon planning of 150+ steps. Existing methods and benchmarks fail to evaluate this multi-stage long-horizon capability.

**Key Challenge**: Long-horizon multi-stage navigation requires handling dependencies between subtasks (e.g., task A must be completed before starting B). However, existing evaluation metrics (such as SR and SPL) only assess the final outcome, failing to measure the correctness of intermediate stages.

**Key Insight**: Three new metrics (ISR, CSR, and CGT) are defined to evaluate independent subtask success rate, conditional subtask success rate, and path difficulty-weighted success rate, respectively. GPT-4 and the NavGen platform are leveraged to automatically generate large-scale multi-stage tasks.

**Core Idea**: New task definition + new evaluation metrics + large-scale automatically generated benchmark = long-horizon multi-stage VLN.

## Method

### Key Designs

1. **NavGen Automatic Generation Platform**:

    - **Function**: Automatically generates multi-stage navigation tasks from 3D scenes.
    - **Mechanism**: Given a list of objects and the topological structure of a scene, GPT-4 automatically generates navigation instructions containing 2 to 4 subtasks. Each subtask has an independent start/end point and success evaluation criteria.

2. **MGDM (Memory-Guided Decision Model)**:

    - **Function**: Handles memory management in long-horizon navigation.
    - **Mechanism**: Short-term memory blurring (compressing recent observations into summaries to avoid information overload) + long-term memory retrieval (retrieving relevant experiences from history) + Chain-of-Thought feedback (utilizing CoT to analyze the current state and decide on the next action).

3. **Three New Evaluation Metrics**:

    - ISR (Independent Subtask Success Rate): Evaluates each subtask independently.
    - CSR (Conditional Subtask Success Rate): Accounts for dependencies between subtasks (if a preceding subtask fails, all subsequent ones are considered failed).
    - CGT: Weighs CSR with path difficulty weights.

### Loss & Training

Imitation learning + cross-entropy loss. LHPR-VLN contains 3,260 tasks covering 216 HM3D scenes, with 39% two-subtask tasks, 52.4% three-subtask tasks, and 8.6% four-subtask tasks.

## Key Experimental Results

### Main Results

| Method | ISR | CSR | CGT |
|------|-----|-----|-----|
| NaviLLM (Finetuned) | 3.81% | 1.67% | 2.54% |
| **MGDM** | **4.69%** | **3.30%** | **5.83%** |

All baselines show a success rate close to 0% on tasks with 2–3 subtasks, indicating that long-horizon navigation is highly challenging.

### Key Findings
- **Long-horizon navigation is extremely challenging for all existing methods**: The best method achieves only a ~5% success rate.
- **Subtask dependencies heavily impact performance**: CSR is significantly lower than ISR, indicating that error cascades are the primary cause of failure.
- **Memory mechanisms are crucial for long-horizon scenarios**: Methods without memory management fail completely.

## Highlights & Insights
- **The task definition and benchmark are the core contributions**—unveiling the fundamental limitations of existing VLN methods in long-horizon scenarios.
- **Extremely low absolute performance**—the 5% success rate indicates that long-horizon VLN is a highly unresolved, open problem.

## Limitations & Future Work
- The performance of baseline methods is extremely low, making it difficult to demonstrate the room for method improvement.
- Limited to the Habitat simulator.
- The memory mechanism is tailored for specific tasks, and its generalizability remains to be verified.

## Rating
- Novelty: ⭐⭐⭐⭐ The contribution framework consisting of new tasks, new metrics, and a new benchmark is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baselines are evaluated, but absolute performance is too low.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear.
- Value: ⭐⭐⭐⭐ Points the VLN community towards the long-horizon direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CoNavBench: Collaborative Long-Horizon Vision-Language Navigation Benchmark](../../ICLR2026/robotics/conavbench_collaborative_long-horizon_vision-language_navigation_benchmark.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[CVPR 2025\] g3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks](g3d-lf_generalizable_3d-language_feature_fields_for_embodied_tasks.md)
- [\[CVPR 2025\] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](roboground_robotic_manipulation_with_grounded_vision-language_priors.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)

</div>

<!-- RELATED:END -->
