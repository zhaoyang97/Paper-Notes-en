---
title: >-
  [Paper Note] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation
description: >-
  [ACL 2026][code generation] This paper proposes CollabCoder, a plan-code co-evolution framework that employs a Collaborative Decision Module (CDM) to determine whether errors should be repaired at the plan level or the code level, and a Reasoning Trajectory module (RT) to enable self-improving debugging that learns from failures. CollabCoder outperforms strong baselines by 11–20% on challenging programming benchmarks while reducing API calls by 4–10.
tags:
  - ACL 2026
  - code generation
  - plan-code co-evolution
  - multi-agent
  - collaborative debugging
  - reasoning trajectory
date: 2026-05-08
content_hash: 2305d55846c6546c
---

# CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation

**Conference**: ACL 2026
**arXiv**: [2604.13946](https://arxiv.org/abs/2604.13946)
**Code**: [https://github.com/ihbkaiser/CollabCoder](https://github.com/ihbkaiser/CollabCoder)
**Area**: Code Generation / Multi-Agent Systems
**Keywords**: code generation, plan-code co-evolution, multi-agent, collaborative debugging, reasoning trajectory

## TL;DR

This paper proposes CollabCoder, a plan-code co-evolution framework that employs a Collaborative Decision Module (CDM) to determine whether errors should be repaired at the plan level or the code level, and a Reasoning Trajectory module (RT) to enable self-improving debugging that learns from failures. CollabCoder outperforms strong baselines by 11–20% on challenging programming benchmarks while reducing API calls by 4–10.

## Background & Motivation

**State of the Field**: LLM-based code generation has evolved from direct generation to a two-stage "plan-then-code" paradigm: the first stage generates a plan and code, and the second stage performs refinement or debugging. Recent multi-agent frameworks such as MapCoder and CodeSIM decompose the generation process into iterative pipelines of retrieval, planning, and debugging.

**Limitations of Prior Work**: (1) Debugging is predominantly reactive and lacks an error attribution mechanism, often producing repetitive and ineffective modifications. (2) The planning module remains fixed throughout the debugging process and cannot adapt based on code revisions or intermediate feedback. (3) The effective reasoning complexity of existing systems is $O(nk)$, resulting in high computational overhead.

**Root Cause**: When a code error originates from a logical mistake at the planning level, modifying only the code cannot address the root cause. However, existing methods cannot distinguish whether the error should be fixed in the plan or in the code.

**Paper Goals**: To design a framework in which plans and code can co-evolve collaboratively, adaptively diagnosing the source of errors and selecting the appropriate repair strategy.

**Starting Point**: Introduce a Collaborative Decision Module (CDM) that diagnoses errors from three complementary perspectives (plan analysis, code analysis, and plan-code alignment analysis), and a Reasoning Trajectory module (RT) that accumulates historical debugging experience.

**Core Idea**: Plans and code should co-evolve — debugging should not only fix the code but also revise the plan when necessary, and the debugging strategy should continuously learn from historical failures.

## Method

### Overall Architecture

CollabCoder consists of three interacting agents: a planning agent $A_{\text{plan}}$, a coding agent $A_{\text{code}}$, and a debugging agent $A_{\text{debug}}$. The debugging agent is further decomposed into the Collaborative Decision Module (CDM) and the Reasoning Trajectory module (RT). At each iteration, CDM analyzes the cause of failure and decides whether to update the plan or the code; RT accumulates debugging experience and guides the repair strategy.

### Key Designs

1. **Collaborative Decision Module (CDM)**:

    - **Function**: Analyzes errors from multiple perspectives and determines the repair strategy.
    - **Mechanism**: The analysis phase performs three complementary analyses — plan-level analysis $E_\pi^{(t)}$ (whether the plan logic is consistent with the failure), code-level analysis $E_c^{(t)}$ (implementation errors assuming the plan is correct), and plan-code alignment analysis $E_{\text{align}}^{(t)}$ (semantic consistency between the plan and the code). The decision phase applies a confidence-consistency aggregation function $D^{(t)} = \arg\max_{d} \sum_i w_i \cdot \phi_{i,d}^{(t)} \cdot \varphi_{H\setminus\{i\},d}^{(t)}$ to decide whether to update the plan or the code.
    - **Design Motivation**: A single perspective may produce misattribution; triangulated analysis combined with weighted aggregation provides more reliable error attribution.

2. **Reasoning Trajectory Module (RT)**:

    - **Function**: Accumulates debugging experience across iterations to avoid repetitive and ineffective repairs.
    - **Mechanism**: Maintains a persistent reasoning state $R^{(t)}$ that jointly considers historical debugging context $R^{(t-1)}$, current diagnostic signals $E_X^{(t)}$, the problem description, the current solution, and failure evidence to update the debugging strategy, which then guides the next round of repair.
    - **Design Motivation**: Prior methods treat each failure independently and cannot learn from historical repairs, leading to repeated attempts with ineffective strategies.

3. **Plan-Code Co-Evolution Pipeline**:

    - **Function**: Enables collaborative iterative refinement of both the plan and the code.
    - **Mechanism**: At each iteration, CDM determines the repair target ($D^{(t)} = 0$ to update the plan, $D^{(t)} = 1$ to update the code); RT generates the corresponding repair strategy, which is executed by the respective agent. After repair, the solution is re-tested and the next iteration begins, until all tests pass or the iteration limit is reached.
    - **Design Motivation**: Breaks the rigid pattern of "fixed plan + repeated code patching," allowing errors to be corrected at the appropriate level.

### Loss & Training

CollabCoder is a training-free inference-time framework and involves no gradient updates. Key hyperparameters: number of iterations $t = 5$; trust weights $w_\pi = 0.4,\ w_c = 0.3,\ w_{\text{align}} = 0.3$. Multiple backbone models are supported, including GPT-4o mini, Seed-Coder-8B, and Qwen2.5-Coder-32B.

## Key Experimental Results

### Main Results

**Code generation accuracy on Seed-Coder-8B (Pass@1 %)**

| Method | HE | HE-ET | MBPP | MBPP-ET | Avg | API Calls |
|------|-----|-------|------|---------|-----|-----------|
| CoT | 82.32 | 75.00 | 75.06 | 50.13 | 70.63 | 1.00 |
| MapCoder | 79.88 | 70.12 | 73.55 | 49.12 | 68.78 | 9.84 |
| CodeSIM | 90.24 | 76.20 | 82.00 | 53.65 | 75.51 | 6.69 |
| **CollabCoder** | 87.20 | 78.05 | 83.37 | 56.42 | **76.26** | **5.06** |

**Performance on challenging benchmarks (GPT-4o mini)**

| Method | LiveCodeBench | xCodeEval | API Calls |
|------|---------------|-----------|-----------|
| CodeSIM | 39.60 | 20.26 | 8.41 |
| ThinkCoder | 36.91 | 18.93 | 9.00 |
| **CollabCoder** | **47.65** | **22.37** | **4.76** |

### Ablation Study

| Configuration | Performance | Notes |
|------|------|------|
| Full CollabCoder | Best | CDM + RT complete |
| w/o CDM (code-only repair) | Degraded | Cannot correct plan-level errors |
| w/o RT (no historical experience) | Degraded | Repetitive ineffective repairs |
| w/o alignment analysis | Slightly degraded | Reduced error attribution accuracy |

### Key Findings

- The advantage is more pronounced on the harder LiveCodeBench and xCodeEval benchmarks: 11–20% improvement over CodeSIM with approximately 4 fewer API calls.
- CDM's error attribution accuracy improves steadily across iterations, demonstrating the effectiveness of the three-perspective analysis.
- The RT module significantly reduces repetitive ineffective repairs and improves debugging efficiency.
- CollabCoder matches the state of the art on simpler benchmarks (HumanEval, MBPP) and substantially surpasses it on complex benchmarks.

## Highlights & Insights

- The decision mechanism of "revise the plan or revise the code" addresses a core pain point in code debugging.
- Accumulated reasoning trajectory state avoids "trial-and-error loops" — a common failure mode in current multi-agent systems.
- CollabCoder achieves a dual gain in efficiency and effectiveness: fewer API calls and higher accuracy, with the advantage particularly salient on difficult tasks.

## Limitations & Future Work

- The trust weights in CDM are fixed hyperparameters and may not generalize to all task types.
- Improvements on simple tasks are limited, making the overhead potentially unjustified in those settings.
- The approach relies on the LLM's code analysis capabilities and may be less effective for programming paradigms that LLMs are inherently less proficient at.
- The RT history window is finite; critical information may be lost during long debugging sequences.

## Related Work & Insights

- **vs. MapCoder/CodeSIM**: These methods employ a fixed plan with multi-round code repair; CollabCoder allows plans and code to co-evolve.
- **vs. ThinkCoder**: ThinkCoder uses 20 debugging rounds yet underperforms CollabCoder's 5 rounds, demonstrating that adaptive decision-making is more effective than brute-force iteration.

## Rating

- Novelty: ⭐⭐⭐⭐ Plan-code co-evolution and the collaborative decision mechanism represent a significant improvement over existing code generation agents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks, three backbone models, efficiency analysis, and ablation studies — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear and the diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Achieves a dual gain in efficiency and effectiveness on complex programming tasks.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)

<!-- RELATED:END -->
