---
title: >-
  [Paper Note] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation
description: >-
  [ACL 2026][Code Intelligence][Code Generation] This paper proposes CollabCoder, a plan-code co-evolution framework. By utilizing a Collaborative Decision-Making (CDM) module to determine whether errors should be fixed at…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code Generation"
  - "Plan-Code Co-Evolution"
  - "Multi-Agent"
  - "Collaborative Debugging"
  - "Reasoning Trajectory"
date: 2026-05-08
content_hash: 16dde777a770ee2e
---

# CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation

**Conference**: ACL 2026  
**arXiv**: [2604.13946](https://arxiv.org/abs/2604.13946)  
**Code**: [https://github.com/ihbkaiser/CollabCoder](https://github.com/ihbkaiser/CollabCoder)  
**Area**: Code Generation / Multi-Agent Systems  
**Keywords**: Code Generation, Plan-Code Co-Evolution, Multi-Agent, Collaborative Debugging, Reasoning Trajectory

## TL;DR

This paper proposes CollabCoder, a plan-code co-evolution framework. By utilizing a Collaborative Decision-Making (CDM) module to determine whether errors should be fixed at the plan or code level, and integrating a Reasoning Trajectory (RT) module for self-improving debugging learned from errors, it achieves an 11-20% improvement over strong baselines on complex programming benchmarks while reducing API calls by 4-10.

## Background & Motivation

**Background**: LLM-based code generation has evolved from direct generation to a "plan-then-code" dual-phase paradigm: the first stage generates a plan and encodes it, while the second stage refines or debugs. Recent multi-agent frameworks like MapCoder and CodeSIM decompose the generation process into iterative workflows of retrieval, planning, and debugging.

**Limitations of Prior Work**: (1) Debugging is primarily reactive and lacks error attribution mechanisms, often leading to repetitive and ineffective modifications; (2) The planning module remains fixed throughout the debugging process, failing to adapt based on code changes and intermediate feedback; (3) Existing systems have an effective reasoning complexity of $O(nk)$, leading to high computational overhead.

**Key Challenge**: When code errors originate from logical flaws at the planning level, modifying the code alone cannot resolve the fundamental issue; however, existing methods cannot distinguish the source of the error—whether to modify the plan or the code.

**Goal**: To design a framework where planning and code can co-evolve, adaptively judging the source of errors and selecting corresponding repair strategies.

**Key Insight**: Introduce a Collaborative Decision-Making (CDM) module to diagnose errors from three complementary perspectives (plan analysis, code analysis, and plan-code alignment analysis), and incorporate a Reasoning Trajectory (RT) module to accumulate historical debugging experience.

**Core Idea**: Plans and code should evolve together—debugging should not only fix code but also revise plans when necessary, and debugging strategies should continuously learn from historical failures.

## Method

### Overall Architecture

CollabCoder consists of three interacting agents: a planning agent $A_{\text{plan}}$, a coding agent $A_{\text{code}}$, and a debugging agent $A_{\text{debug}}$. The debugging agent is further divided into the Collaborative Decision-Making (CDM) module and the Reasoning Trajectory (RT) module. In each iteration, CDM analyzes the cause of failure to decide whether to update the plan or the code, while RT accumulates debugging experience to guide the repair strategy.

### Key Designs

1.  **Collaborative Decision-Making (CDM)**:
    - **Function**: Analyzes errors from multiple perspectives and determines the recovery strategy.
    - **Mechanism**: The analysis phase performs three complementary analyses—plan-level analysis $E_\pi^{(t)}$ (whether the plan logic is consistent with the failure), code-level analysis $E_c^{(t)}$ (implementation errors assuming the plan is correct), and plan-code alignment analysis $E_{\text{align}}^{(t)}$ (semantic consistency between plan and code). The decision phase determines whether to update the plan or code via a confidence-consistency aggregation function $D^{(t)} = \arg\max_{d} \sum_i w_i \cdot \phi_{i,d}^{(t)} \cdot \varphi_{H\setminus\{i\},d}^{(t)}$.
    - **Design Motivation**: A single perspective may lead to misjudgment; three-angle analysis combined with weighted aggregation provides more reliable error attribution.

2.  **Reasoning Trajectory (RT)**:
    - **Function**: Accumulates debugging experience across iterations to avoid repetitive and ineffective fixes.
    - **Mechanism**: Maintains a persistent reasoning state $R^{(t)}$, updating the debugging strategy by jointly considering the historical debugging context $R^{(t-1)}$, current diagnostic signals $E_X^{(t)}$, problem description, current solution, and evidence of failure. The strategy guides the next round of repair operations.
    - **Design Motivation**: Previous methods treat each failure independently and fail to learn from historical repairs, leading to repeated attempts at invalid strategies.

3.  **Plan-Code Co-Evolution Process**:
    - **Function**: Implements collaborative iterative optimization of plans and code.
    - **Mechanism**: In each iteration, CDM decides the repair target ($D^{(t)} = 0$ to update the plan, $D^{(t)} = 1$ to update the code), and RT generates the corresponding repair strategy, which is executed by the respective agent. After repair, the system re-tests and enters the next iteration until all tests pass or the iteration limit is reached.
    - **Design Motivation**: Breaks the rigid pattern of "fixed plan + repeated code fixing," allowing errors to be corrected at the appropriate level.

### Loss & Training

CollabCoder is a training-free inference-time framework and does not involve gradient updates. Key hyperparameters: iteration count $t = 5$, trust weights $w_\pi = 0.4, w_c = 0.3, w_{\text{align}} = 0.3$. It supports various backbone models such as GPT-4o mini, Seed-Coder-8B, and Qwen2.5-Coder-32B.

## Key Experimental Results

### Main Results

**Code Generation Accuracy on Seed-Coder-8B (Pass@1 %)**

| Method | HE | HE-ET | MBPP | MBPP-ET | Avg | API Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CoT | 82.32 | 75.00 | 75.06 | 50.13 | 70.63 | 1.00 |
| MapCoder | 79.88 | 70.12 | 73.55 | 49.12 | 68.78 | 9.84 |
| CodeSIM | 90.24 | 76.20 | 82.00 | 53.65 | 75.51 | 6.69 |
| **CollabCoder** | 87.20 | 78.05 | 83.37 | 56.42 | **76.26** | **5.06** |

**Performance on Complex Benchmarks (GPT-4o mini)**

| Method | LiveCodeBench | xCodeEval | API Calls |
| :--- | :--- | :--- | :--- |
| CodeSIM | 39.60 | 20.26 | 8.41 |
| ThinkCoder | 36.91 | 18.93 | 9.00 |
| **CollabCoder** | **47.65** | **22.37** | **4.76** |

### Ablation Study

| Configuration | Performance | Description |
| :--- | :--- | :--- |
| Full CollabCoder | Optimal | Full version with CDM + RT |
| W/o CDM (Code repair only) | Decrease | Unable to correct plan-level errors |
| W/o RT (No history) | Decrease | Repetitive ineffective repairs |
| W/o Alignment Analysis | Slight Decrease | Reduces error attribution accuracy |

### Key Findings

- Advantages are more pronounced on difficult benchmarks like LiveCodeBench and xCodeEval: showing an 11-20% improvement over CodeSIM while reducing API calls by approximately 4.
- The accuracy of error attribution in CDM continues to improve across iterations, demonstrating the effectiveness of the three-angle analysis.
- The RT module significantly reduces the number of repetitive ineffective repairs and improves debugging efficiency.
- Performance is comparable to SOTA on simple benchmarks (HumanEval, MBPP) but significantly exceeds them on complex benchmarks.

## Highlights & Insights

- The decision mechanism of "whether to modify the plan or the code" addresses a core pain point in code debugging.
- State accumulation in the reasoning trajectory avoids the "trial-and-error loop"—a common issue in current multi-agent systems.
- A win-win for efficiency and effectiveness: fewer API calls and higher accuracy, with particularly significant advantages on difficult tasks.

## Limitations & Future Work

- Trust weights in CDM are fixed hyperparameters and may not be suitable for all task types.
- Improvements on simple tasks are limited, making the overhead potentially unjustifiable.
- Reliability depends on the LLM's code analysis capabilities; effectiveness may be limited for programming paradigms that the LLM itself is not proficient in.
- The historical window of RT is limited, potentially missing key information in long debugging sequences.

## Related Work & Insights

- **vs MapCoder/CodeSIM**: These methods use fixed plans with multi-round code repairs, whereas CollabCoder allows plan and code to co-evolve.
- **vs ThinkCoder**: ThinkCoder uses 20 rounds of debugging but is less effective than CollabCoder’s 5 rounds, indicating that adaptive decision-making is more effective than brute-force iteration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Plan-code co-evolution and collaborative decision mechanisms are significant improvements over existing code generation agents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Very comprehensive, covering six benchmarks, three backbone models, efficiency analysis, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear and diagrams are intuitive.
- **Value**: ⭐⭐⭐⭐ Achieves a win-win for efficiency and effectiveness on complex programming tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PaT: Planning-after-Trial for Efficient Test-Time Code Generation](pat_planning-after-trial_for_efficient_test-time_code_generation.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](learning_adaptive_parallel_execution_for_efficient_code_localization.md)

</div>

<!-- RELATED:END -->
