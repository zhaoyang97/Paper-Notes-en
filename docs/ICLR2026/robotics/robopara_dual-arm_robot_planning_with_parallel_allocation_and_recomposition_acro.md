---
title: >-
  [Paper Note] RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks
description: >-
  [ICLR 2026][Robotics][dual-arm robot] This paper proposes RoboPARA, a two-stage framework that optimizes task parallelism for dual-arm robots via dependency graph construction and graph re-traversal scheduling, achieving 30–50% reduction in execution time and a 34% improvement in success rate over existing methods across multi-scenario benchmarks.
tags:
  - ICLR 2026
  - Robotics
  - dual-arm robot
  - parallel task planning
  - DAG dependency graph
  - LLM planning
  - multi-task scheduling
date: 2026-05-08
content_hash: 61589d97930e1191
---

# RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks

**Conference**: ICLR 2026
**arXiv**: [2506.06683](https://arxiv.org/abs/2506.06683)
**Code**: [https://github.com/AiDuanshiying/RoboPARA](https://github.com/AiDuanshiying/RoboPARA)
**Area**: Robotics / Task Planning
**Keywords**: dual-arm robot, parallel task planning, DAG dependency graph, LLM planning, multi-task scheduling

## TL;DR
This paper proposes RoboPARA, a two-stage framework that optimizes task parallelism for dual-arm robots via dependency graph construction and graph re-traversal scheduling, achieving 30–50% reduction in execution time and a 34% improvement in success rate over existing methods across multi-scenario benchmarks.

## Background & Motivation

**Background**: LLM-driven dual-arm robot task planning approaches (e.g., RoCo, FLTRNN) have made notable progress, yet these methods primarily optimize task success rate and completion time, mostly producing plans that execute sequentially on a single arm.

**Limitations of Prior Work**: Existing methods overlook inter-arm parallelism — when a task requires only one arm, the other remains completely idle. This leaves the collaborative potential of dual-arm systems underutilized and leads to low execution efficiency.

**Key Challenge**: Dual-arm parallel planning must simultaneously handle task dependencies (certain steps must execute in order) and parallelization opportunities (independent steps can be assigned concurrently to both arms), constituting a combinatorial optimization problem.

**Goal**: Maximize parallel utilization of both arms while preserving task correctness, thereby reducing overall execution time.

**Key Insight**: Drawing inspiration from everyday human behavior — boiling water while brushing teeth — the paper models task dependencies via a DAG (Directed Acyclic Graph) and then applies a graph-traversal scheduling algorithm to maximize parallelism.

**Core Idea**: Decouple dual-arm task planning into two stages — "LLM generates dependency graph → scheduling algorithm maximizes parallelism" — allowing the LLM to focus on understanding task semantics rather than directly planning parallel execution.

## Method

### Overall Architecture
RoboPARA adopts a two-stage architecture. The first stage employs an LLM augmented with RAG to generate a task dependency graph (DAG). The second stage applies a graph re-traversal algorithm to assign parallel tasks to both arms. The input is a set of multi-task instructions from the user; the output is a parallel execution plan for the dual-arm system.

### Key Designs

1. **Dependency Graph Generation with Error Correction (Dependency Graph-based Planning)**:

    - **Function**: Transforms multi-task instructions into a DAG, where nodes represent atomic operation steps and edges represent dependencies.
    - **Mechanism**: RAG retrieves detailed procedural knowledge for each task from a local memory module and integrates it into structured prompts, enabling the LLM to generate the DAG. Each edge $(u, v)$ indicates that step $v$ cannot begin until step $u$ is complete. A DAG validation and error-correction step is included to detect cyclic dependencies and redundant edges.
    - **Design Motivation**: The DAG structure naturally models task dependencies — nodes with no incoming edges can execute immediately, and when multiple dependency-free nodes exist simultaneously, they can be assigned to both arms in parallel.

2. **Graph Re-Traversal Dual-Arm Parallel Scheduling**:

    - **Function**: Optimizes the DAG traversal order to maximize the number of parallel execution steps across both arms.
    - **Mechanism**: Two arm queues $Q_L, Q_R$ are maintained. At each time step, the set of all nodes with in-degree zero, $S_t$, is identified, and a heuristic rule assigns these nodes to idle arms. Tasks requiring bimanual coordination are simultaneously assigned to both arms. Key heuristics include prioritizing tasks on the critical path and balancing workload between the two arms.
    - **Design Motivation**: The graph generation stage only ensures semantic correctness; a dedicated scheduling algorithm is required to optimize parallelism, approximating an NP-hard combinatorial problem.

3. **X-DAPT Benchmark Dataset**:

    - **Function**: Constructs the first evaluation dataset specifically focused on dual-arm task parallelism.
    - **Mechanism**: Covers 10 representative scenarios (kitchen, office, agricultural greenhouse, factory, etc.), each at three difficulty levels, comprising 1,000+ task bundles. Evaluation metrics include TEI (Time Efficiency Index), TFR (Task Failure Rate), PPR (Parallel Step Proportion), and APR (Average Parallelism Rate).
    - **Design Motivation**: Existing benchmarks do not assess parallelism and therefore cannot measure dual-arm collaborative efficiency.

### Loss & Training
No training is required. The framework relies entirely on zero-shot/few-shot inference of LLMs combined with deterministic scheduling algorithms.

## Key Experimental Results

### Main Results

| Method | TEI ↑ | TFR ↓ | PPR ↑ | APR ↑ | Scenario |
|---|---|---|---|---|---|
| RoboPARA | 0.953 | 0.033 | 0.543 | 0.283 | Kitchen |
| Embodied TaPA | 0.859 | 0.200 | 0.000 | 0.080 | Kitchen |
| RoCo | 0.836 | 0.067 | 0.008 | 0.041 | Kitchen |
| ChatGPT-Prompts | 0.817 | 0.081 | 0.000 | 0.010 | Kitchen |
| LLM-Planner | 0.858 | 0.200 | 0.000 | 0.077 | Kitchen |

### Ablation Study

| Configuration | PPR | APR | Notes |
|---|---|---|---|
| RoboPARA (Full) | 0.543 | 0.283 | Complete framework |
| w/o graph error correction | ~0.35 | ~0.18 | DAG generation may contain cycles/redundancies |
| w/o RAG retrieval | ~0.40 | ~0.20 | Task decomposition quality degrades |

### Key Findings
- RoboPARA achieves on average more than 4.5× parallel and collaborative steps, reducing execution time by 30–50%.
- On the most complex task combinations, RoboPARA outperforms all baselines by an average of 34% in success rate.
- All baseline methods exhibit near-zero parallel step ratios (PPR ≈ 0), confirming that existing approaches entirely neglect parallelism.
- Real-world deployment on a humanoid robot demonstrates behavior patterns closely resembling human activity.

## Highlights & Insights
- **Planning–Scheduling Decoupling**: Delegating task semantic understanding and dependency modeling to the LLM, while assigning the NP-hard scheduling problem to a deterministic algorithm, represents a principled division of responsibilities.
- **Parallelism as an Evaluation Dimension**: This work is the first to establish parallel degree as a core evaluation metric for dual-arm robots; the PPR/APR metrics are generalizable to multi-robot collaborative scenarios.

## Limitations & Future Work
- DAG generation still relies on the LLM's reasoning capability and may produce errors for complex cross-task dependencies.
- The scheduling algorithm uses heuristics rather than optimal solvers, potentially missing the globally optimal parallel plan.
- The framework assumes that execution time for each atomic operation is known or estimable in advance, which may not hold in practice.
- Dynamic replanning during execution (e.g., recovery strategies upon step failure) is not addressed.

## Related Work & Insights
- **vs. RoCo**: RoCo employs agent dialogue negotiation to decompose and assign tasks, but the resulting plans remain predominantly sequential; RoboPARA explicitly optimizes parallelism.
- **vs. FLTRNN**: FLTRNN uses an RNN architecture for long-horizon planning, focusing on task decomposition and memory management, without addressing dual-arm parallelism.

## Rating
- Novelty: ⭐⭐⭐⭐ First LLM planning framework to systematically address dual-arm parallelism
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive baseline comparisons and multi-scenario evaluations
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and thorough method description
- Value: ⭐⭐⭐⭐ Practically instructive for multi-arm and multi-robot task scheduling

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)
- [\[ICLR 2026\] SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models](synthworlds_controlled_parallel_worlds_for_disentangling_reasoning_and_knowledge.md)
- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[ICLR 2026\] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)

</div>

<!-- RELATED:END -->
