---
title: >-
  [Paper Note] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation
description: >-
  [ACL 2026 Findings][Code Intelligence][Code Generation] This paper proposes CollabCoder, a plan-code co-evolution framework. Through a Collaborative Decision-Making (CDM) module, it determines whether errors should be fixed at the plan level or the code level. Combined with a Reasoning Trajectory (RT) module for self-improving debugging learned from errors, it achieves an 11-20% improvement over strong baselines on complex programming benchmarks while reducing API calls by 4-…
tags:
  - "ACL 2026 Findings"
  - "Code Intelligence"
  - "Code Generation"
  - "Plan-Code Co-evolution"
  - "Multi-agent"
  - "Collaborative Debugging"
  - "Reasoning Trajectory"
date: 2026-05-08
content_hash: 28c4ada95af97bd2
---

# CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.13946](https://arxiv.org/abs/2604.13946)  
**Code**: [https://github.com/ihbkaiser/CollabCoder](https://github.com/ihbkaiser/CollabCoder)  
**Area**: Code Generation / Multi-Agent Systems  
**Keywords**: Code Generation, Plan-Code Co-evolution, Multi-agent, Collaborative Debugging, Reasoning Trajectory

## TL;DR

This paper proposes CollabCoder, a plan-code co-evolution framework. Through a Collaborative Decision-Making (CDM) module, it determines whether errors should be fixed at the plan level or the code level. Combined with a Reasoning Trajectory (RT) module for self-improving debugging learned from errors, it achieves an 11-20% improvement over strong baselines on complex programming benchmarks while reducing API calls by 4-10.

## Background & Motivation

**Background**: LLM code generation has evolved from direct generation to a "plan-then-code" two-stage paradigm: the first stage generates a plan and code, while the second stage refines or debugs. Multi-agent frameworks like MapCoder and CodeSIM have emerged, decomposing the generation process into iterative workflows of retrieval, planning, and debugging.

**Limitations of Prior Work**: (1) Debugging is primarily reactive and lacks an error attribution mechanism, often leading to repetitive and ineffective modifications; (2) The planning module remains fixed throughout the debugging process, unable to adjust based on code changes and intermediate feedback; (3) The effective reasoning complexity of existing systems is $O(nk)$, leading to high computational overhead.

**Key Challenge**: When a code error originates from a logical flaw at the planning level, modifying only the code cannot solve the fundamental problem; however, existing methods cannot distinguish the source of the error—whether to modify the plan or the code.

**Goal**: To design a framework where plans and code can co-evolve, adaptively judging the source of errors and selecting corresponding repair strategies.

**Key Insight**: Introduce a Collaborative Decision-Making (CDM) module to diagnose errors from three complementary perspectives (plan analysis, code analysis, and plan-code alignment analysis), and a Reasoning Trajectory (RT) module to accumulate historical debugging experience.

**Core Idea**: Plans and code should evolve together—debugging should not only fix code but also revise plans when necessary, and debugging strategies should continuously learn from historical failures.

## Method

### Overall Architecture

CollabCoder addresses two chronic issues in the "plan-then-code" paradigm: plans being frozen once set and reactive debugging that repeatedly modifies code without identifying the root cause. It enables plan-code co-evolution—the framework consists of a planning agent $A_{\text{plan}}$, a coding agent $A_{\text{code}}$, and a debugging agent $A_{\text{debug}}$. Inside the debugging agent are the Collaborative Decision-Making (CDM) and Reasoning Trajectory (RT) modules. In each iteration, after a code test failure, the CDM first diagnoses whether the failure is attributable to the plan or the code. The RT then provides specific repair strategies based on historical debugging experience, which are executed by the corresponding agent. Testing repeats until all tests pass or the iteration limit is reached. The input is the problem description; the intermediate state is a plan and code that are continuously revised; the output is the final solution passing the tests.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Problem Description P"] --> B["Planning Agent A_plan<br/>Generate / Revise Plan π"]
    B --> C["Coding Agent A_code<br/>Generate / Revise Code c"]
    C --> T{"Run Test<br/>All Passed?"}
    T -->|Yes| Z["Output Final Solution Passing Tests"]
    T -->|No (Failure Log F)| DBG
    subgraph DBG["Debugging Agent A_debug: One Round of Plan-Code Co-evolution"]
        direction TB
        subgraph CDM["Collaborative Decision-Making (CDM): Tri-perspective Error Localization"]
            direction TB
            E1["Plan-level Analysis E_π"] --> AGG["Confidence-Consistency Aggregation<br/>→ Decision D"]
            E2["Code-level Analysis E_c"] --> AGG
            E3["Plan-Code Alignment Analysis E_align"] --> AGG
        end
        AGG --> RT["Reasoning Trajectory Module (RT)<br/>Cumulative Experience → Repair Strategy"]
    end
    RT -->|D=0: Revise Plan| B
    RT -->|D=1: Revise Code| C
```

### Key Designs

**1. Collaborative Decision-Making (CDM): Localizing the True Source of Errors from Three Perspectives**

When code fails, the root cause may lie in implementation details or the plan's logic itself; looking at the code alone cannot distinguish them. During the analysis phase, CDM performs three complementary diagnoses in parallel—plan-level analysis $E_\pi^{(t)}$ to judge if the plan logic is consistent with the failure, code-level analysis $E_c^{(t)}$ to check for implementation errors assuming the plan is correct, and plan-code alignment analysis $E_{\text{align}}^{(t)}$ to verify semantic consistency between the plan and the code. In the decision phase, a confidence-consistency aggregation function $D^{(t)} = \arg\max_{d} \sum_i w_i \cdot \phi_{i,d}^{(t)} \cdot \varphi_{H\setminus\{i\},d}^{(t)}$ fuses the three diagnostic paths based on weights and mutual consistency to output whether to "Revise Plan" or "Revise Code." A single perspective is prone to misjudgment, but the weighted aggregation of three angles provides more reliable error attribution.

**2. Reasoning Trajectory Module (RT): Continuous Learning from Historical Failures**

Previous methods handle each failure in isolation, leading to repeated unsuccessful modifications. RT maintains a persistent reasoning state $R^{(t)}$. During each update, it jointly considers the historical debugging context $R^{(t-1)}$, the diagnostic signals $E_X^{(t)}$ from CDM, the problem description, the current solution, and evidence of failure to generate a strategy for the next fix. By distilling debugging experience into an accumulative state, the system avoids spinning in "trial-and-error loops."

**3. Plan-Code Co-evolution Mechanism: Fixing Errors at the Correct Level**

Rigid patterns of fixing code with a fixed plan are ineffective when the plan itself is logically flawed. In each iteration, CollabCoder uses CDM to output a repair target ($D^{(t)} = 0$ for plan, $D^{(t)} = 1$ for code), which RT equips with a strategy for the corresponding agent to execute. Plans and code can thus be revised alternately and converge cooperatively, ensuring errors are fixed at the level where they truly belong.

### Loss & Training

CollabCoder is a training-free inference-time framework and does not involve any gradient updates; all "learning" occurs within the state accumulation of the RT. The core hyperparameters are the iteration limit $t = 5$ and the trust weights for the three CDM diagnostic paths: $w_\pi = 0.4, w_c = 0.3, w_{\text{align}} = 0.3$. The framework is non-intrusive to the backbone model and can be applied directly to various models such as GPT-4o mini, Seed-Coder-8B, and Qwen2.5-Coder-32B.

## Key Experimental Results

### Main Results

**Code Generation Accuracy (Pass@1 %) on Seed-Coder-8B**

| Method | HE | HE-ET | MBPP | MBPP-ET | Avg | API Calls |
|------|-----|-------|------|---------|-----|-----------|
| CoT | 82.32 | 75.00 | 75.06 | 50.13 | 70.63 | 1.00 |
| MapCoder | 79.88 | 70.12 | 73.55 | 49.12 | 68.78 | 9.84 |
| CodeSIM | 90.24 | 76.20 | 82.00 | 53.65 | 75.51 | 6.69 |
| **CollabCoder** | 87.20 | 78.05 | 83.37 | 56.42 | **76.26** | **5.06** |

**Performance on Complex Benchmarks (GPT-4o mini)**

| Method | LiveCodeBench | xCodeEval | API Calls |
|------|---------------|-----------|-----------|
| CodeSIM | 39.60 | 20.26 | 8.41 |
| ThinkCoder | 36.91 | 18.93 | 9.00 |
| **CollabCoder** | **47.65** | **22.37** | **4.76** |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| Full CollabCoder | Optimal | Complete version with CDM + RT |
| W/O CDM (Code only) | Decrease | Unable to correct plan-level errors |
| W/O RT (No history) | Decrease | Repetitive ineffective repairs |
| W/O Alignment Analysis | Slight Decrease | Reduced accuracy in error attribution |

### Key Findings

- Advantages are more pronounced on high-difficulty benchmarks like LiveCodeBench and xCodeEval: 11-20% improvement over CodeSIM while reducing API calls by approximately 4.
- CDM's error attribution accuracy continuously improves during iterations, demonstrating the effectiveness of the tri-perspective analysis.
- The RT module significantly reduces the number of repetitive ineffective repairs, enhancing debugging efficiency.
- Performance is on par with SOTA on simple benchmarks (HumanEval, MBPP) and significantly exceeds them on complex benchmarks.

## Highlights & Insights

- The decision mechanism of "revising plan or code" addresses a core pain point in code debugging.
- State accumulation in the reasoning trajectory avoids the "trial-and-error loop"—a common issue in current multi-agent systems.
- Win-win for efficiency and effectiveness: fewer API calls and higher accuracy, with particularly strong advantages in difficult tasks.

## Limitations & Future Work

- Trust weights in CDM are fixed hyperparameters and may not suit all task types.
- Improvements on simple tasks are limited, making the overhead potentially unfavorable.
- Dependence on the code analysis capabilities of the LLM; effectiveness may be restricted for programming paradigms the LLM itself struggles with.
- The RT historical window is limited, potentially missing critical information in long debugging sequences.

## Related Work & Insights

- **vs MapCoder/CodeSIM**: These methods use fixed plans with multi-round code repairs, whereas CollabCoder allows plan-code co-evolution.
- **vs ThinkCoder**: ThinkCoder uses 20 rounds of debugging but performs worse than CollabCoder’s 5 rounds, indicating that adaptive decision-making is more effective than brute-force iteration.

## Rating

- Novelty: ⭐⭐⭐⭐ The plan-code co-evolution and collaborative decision-making mechanisms are significant improvements for existing code generation agents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks, three backbone models, efficiency analysis, and ablation studies make it very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Achieves a win-win for efficiency and effectiveness in complex programming tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PaT: Planning-after-Trial for Efficient Test-Time Code Generation](pat_planning-after-trial_for_efficient_test-time_code_generation.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ACL 2025\] Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](../../ACL2025/code_intelligence/tree_of_evolution_code_gen.md)
- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](learning_adaptive_parallel_execution_for_efficient_code_localization.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)

</div>

<!-- RELATED:END -->
