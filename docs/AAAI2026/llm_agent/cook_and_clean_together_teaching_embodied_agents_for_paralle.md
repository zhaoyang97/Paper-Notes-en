---
title: >-
  [Paper Note] Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution
description: >-
  [AAAI 2026 (Oral)][LLM Agent][Operations Research] This paper introduces ORS3D, a novel task that incorporates operations research (OR) knowledge into embodied AI task scheduling. Agents are required to exploit the waiti…
tags:
  - "AAAI 2026 (Oral)"
  - "LLM Agent"
  - "Operations Research"
  - "Task Scheduling"
  - "3D Grounding"
  - "Embodied Agents"
  - "MLLM"
date: 2026-05-08
content_hash: dde1b3c1ca31bbdd
---

# Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution

**Conference**: AAAI 2026 (Oral)  
**arXiv**: [2511.19430v1](https://arxiv.org/abs/2511.19430v1)  
**Code**: [https://github.com/H-EmbodVis/GRANT](https://github.com/H-EmbodVis/GRANT)  
**Area**: Embodied AI / 3D Scene Understanding / Task Planning  
**Keywords**: Operations Research, Task Scheduling, 3D Grounding, Embodied Agents, MLLM  

## TL;DR
This paper introduces ORS3D, a novel task that incorporates operations research (OR) knowledge into embodied AI task scheduling. Agents are required to exploit the waiting time of parallelizable sub-tasks to execute other tasks, thereby minimizing total completion time, while simultaneously localizing target objects in 3D scenes. The authors construct a 60K-scale dataset ORS3D-60K and propose the GRANT model, which connects to an external dynamic programming solver via a scheduling token mechanism, achieving a 30.53% improvement in time efficiency over baselines.

## Background & Motivation
Existing task planning methods in embodied AI (e.g., LEO, Grounded 3D LLM) suffer from two critical limitations: (1) they ignore efficiency optimization — generating only sequential step lists while overlooking the fact that sub-tasks can be executed in parallel (e.g., wiping a table while food is being heated in the microwave); (2) they lack 3D spatial grounding — despite operating in 3D environments, these methods often degrade to textual QA without performing spatial grounding. In practice, humans naturally exploit waiting time during household tasks — a form of parallel scheduling in operations research — a capability that is critical yet overlooked for embodied agents.

## Core Problem
How can embodied agents be equipped with OR-level task scheduling capabilities? Specific challenges include: (1) identifying which sub-tasks can be parallelized (e.g., microwave heating = parallelizable; wiping a table = non-parallelizable); (2) optimal scheduling — a classical 0-1 knapsack problem that packs non-parallel tasks into the waiting time windows of parallel tasks to maximize time utilization; (3) simultaneously performing 3D spatial grounding to localize target objects for each action step.

## Method

### Overall Architecture
**Input**: 3D scene point cloud + composite task instruction → 3D scene encoder (initialized with CLASP) generates scene tokens → LLM (Vicuna-1B + LoRA) comprehends the task and identifies sub-task types → scheduling token `<SCH>` invokes an external 0-1 knapsack DP solver to produce the optimal schedule → scheduling result is injected back into the LLM to generate step-by-step action descriptions → grounding token `<GRU>` localizes target objects in the scene via a 3D grounding head → **Output**: time-optimized scheduling plan with per-step 3D object localization.

### Key Designs
1. **ORS3D-60K Dataset**: Sourced from five real-world scene datasets (ScanNet, HM3D, ARKitScenes, 3RScan, MultiScan), covering 4,376 scenes and 60,825 composite tasks. GPT-4o is used to generate sub-task metadata (type and estimated duration) from 3D scene graphs; an OR solver then produces optimal schedules, which are converted into natural language steps. Each task contains 4–7 sub-tasks with an average text length of 311 words.
2. **Scheduling Token Mechanism (STM)**: The LLM first identifies sub-task types (parallelizable/non-parallelizable) and their estimated durations, generating constraint information $I$. Upon encountering the special `<SCH>` token, an external DP solver is invoked, modeling the problem as a 0-1 knapsack (waiting time of parallelizable sub-tasks = knapsack capacity; duration of non-parallel sub-tasks = item weights), and solving for the optimal schedule $S^*$ in milliseconds. The result is converted to text and injected back into the LLM for subsequent generation.
3. **3D Grounding Head**: The `<GRU>` token maps LLM outputs to the scene query space; cosine similarity selects the best-matching scene query, and a point-cloud mask is generated via dot product followed by sigmoid.
4. **Time Efficiency Metric (TE)**: $$\text{TE} = \frac{T_\text{worst} - T_\text{pred}}{T_\text{worst} - T_\text{opt}} \times 100\%$$ This normalizes the model's actual time savings as a proportion of the theoretically optimal savings.

### Loss & Training
- Language generation: cross-entropy loss (next-token prediction)
- 3D Grounding: sigmoid focal loss
- Trained for 10 epochs on 8× RTX 4090; AdamW optimizer; cosine learning rate schedule; $lr = 8\text{e-}4$

## Key Experimental Results

| Method | METEOR | ROUGE | TE (%) | Grounding Acc | Overall |
|--------|--------|-------|--------|---------------|---------|
| Grounded 3D LLM | 41.96 | 53.71 | 42.46 | 34.00 | 43.03 |
| **GRANT** | **42.82** | **62.78** | **72.99** | **35.38** | **53.49** |
| DeepSeek-R1 (text-only) | 32.40 | 41.50 | 72.63 | N/A | 36.63 |
| GPT-4o (text-only) | 49.16 | 62.19 | 45.27 | N/A | 39.15 |

### Ablation Study
- **STM is the core contribution**: removing scheduling content → TE 21.03%; adding scheduling text → 47.04%; with STM → 72.99% (+25.95 pp)
- **Sub-task type identification is critical**: GRANT achieves parallelizable sub-task F1 = 62.84%, substantially higher than the baseline at 59.72%
- **Task complexity has a significant impact**: Overall = 60.23% at 4 sub-tasks, dropping to 48.70% at 7 sub-tasks
- **The solver introduces near-zero overhead**: even with 50 sub-tasks, solving takes only 4 ms

## Highlights & Insights
- **Cross-disciplinary innovation: OR × Embodied AI** — the first work to incorporate OR knowledge (e.g., 0-1 knapsack) into task scheduling within 3D embodied environments.
- **Scheduling token design paradigm** — "LLM identifies constraints → special token invokes external solver → results injected back into LLM" — a generalizable pattern applicable to other settings requiring external solvers.
- **ORS3D-60K dataset** — a 60K-scale 3D task scheduling dataset that fills a critical gap in the community.
- **DeepSeek-R1 comparison** — DeepSeek-R1 achieves TE of 72.63% in text-only scheduling thanks to mathematical RL training, but cannot handle 3D grounding.

## Limitations & Future Work
- Evaluation is conducted solely on offline benchmarks; no deployment on physical robots has been performed.
- The external solver is non-differentiable, precluding end-to-end optimization.
- The current framework supports only scenarios with a single parallelizable sub-task.
- Future directions include internalizing the DP solver within the LLM for differentiable optimization.

## Related Work & Insights
Compared to LEO (ICML '24), which performs only sequential planning without parallel scheduling, GRANT achieves an Overall score of 53.49 vs. LEO's 38.14. Compared to GPT-4o, which offers stronger language understanding but lacks 3D grounding and exhibits limited scheduling capability (TE 45.27 vs. 72.99).

The paradigm of integrating external solvers into LLMs via special tokens is transferable to other tasks requiring precise computation. The application of OR knowledge in embodied AI can be extended to more complex scheduling scenarios, such as multi-agent collaboration and dynamic environments.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Cross-disciplinary innovation combining OR and embodied AI; the scheduling token elegantly embeds combinatorial optimization into Transformer generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Large-scale dataset, multi-baseline comparisons, and comprehensive ablations with broad task coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation framed around the sequential-to-parallel pain point, with rich figures illustrating scheduling effects.
- **Value**: ⭐⭐⭐⭐ Fills the gap in task scheduling efficiency for embodied AI; the scheduling token concept is generalizable to other agent scenarios requiring resource allocation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](../../ICLR2026/llm_agent/the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)
- [\[ACL 2026\] Don't Click That: Teaching Web Agents to Resist Deceptive Interfaces](../../ACL2026/llm_agent/dont_click_that_teaching_web_agents_to_resist_deceptive_interfaces.md)
- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](../../ICLR2026/llm_agent/agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](../../ICLR2026/llm_agent/hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [\[ICML 2026\] Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation](../../ICML2026/llm_agent/rule2drc_benchmarking_llm_agents_for_drc_script_synthesis_with_execution-guided_.md)

</div>

<!-- RELATED:END -->
