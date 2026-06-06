---
title: >-
  [Paper Note] CORE: Full-Path Evaluation of LLM Agents Beyond Final State
description: >-
  [NeurIPS 2025][LLM Agent][Agent Evaluation] This paper proposes CORE, a framework that encodes legitimate tool-calling paths for agent tasks using deterministic finite automata (DFA) and introduces five complementary met…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Agent Evaluation"
  - "Full-Path Evaluation"
  - "Deterministic Finite Automaton"
  - "Safety"
  - "Tool Calling"
date: 2026-05-08
content_hash: 58017974e5e414ad
---

# CORE: Full-Path Evaluation of LLM Agents Beyond Final State

**Conference**: NeurIPS 2025
**arXiv**: [2509.20998](https://arxiv.org/abs/2509.20998)  
**Code**: [https://github.com/Synkrasis-Labs/CORE](https://github.com/Synkrasis-Labs/CORE)  
**Area**: Agent
**Keywords**: Agent Evaluation, Full-Path Evaluation, Deterministic Finite Automaton, Safety, Tool Calling

## TL;DR
This paper proposes CORE, a framework that encodes legitimate tool-calling paths for agent tasks using deterministic finite automata (DFA) and introduces five complementary metrics (path correctness, order correctness, prefix criticality, harm rate, and efficiency) to evaluate agent behavior along the full execution path rather than the final state alone, revealing safety and efficiency differences invisible to conventional final-state evaluation.

## Background & Motivation

**Background**: LLM agents execute real-world tasks through sequences of function calls (API invocations, IoT control, robotic manipulation, etc.). Existing benchmarks (e.g., BFCL) judge agents primarily by whether the final state is correct—reaching the desired terminal state counts as "success."

**Limitations of Prior Work**: Final-state evaluation is severely insufficient: (a) agents that reach a correct terminal state via dangerous intermediate calls are still deemed successful (e.g., a robotic arm grasps the correct object but collides with others in the process); (b) redundant or inefficient paths receive the same score as optimal ones; (c) the "compensating pair" problem—an agent first errs then self-corrects, yielding a correct final state despite potentially harmful intermediate states (e.g., a mistaken fund transfer followed by reversal, where a network failure could leave the system stuck in the erroneous state); (d) unobservable harms—sensor granularity may be insufficient to detect harmful operations that leave correct terminal readings.

**Key Challenge**: In edge deployments (robotics, power grids, IoT), *how* an agent achieves a goal matters as much as *whether* it achieves it, yet existing evaluation attends only to the latter.

**Goal**: Establish an evaluation framework based on execution paths rather than final states, quantifying the correctness, safety, and efficiency of agent behavior.

**Key Insight**: Agent tasks are modeled as DFAs—each prompt induces a set of legitimate tool-calling paths (golden paths), against which the agent's actual execution path is compared.

**Core Idea**: Encode the space of legitimate execution paths for a task using a DFA, and evaluate full-path quality across five complementary metrics from distinct dimensions.

## Method

### Overall Architecture

The CORE framework comprises three layers:
1. **Task Modeling**: Encode the agent world $W = (T, Q)$ (tool set $T$, state set $Q$) and task $\theta = (p, q_0, A)$ as a DFA, define state transitions $\alpha: Q \times A \to Q$, and classify transitions into three types: progress, self-loop, and harmful.
2. **Path Processing**: Apply condensation to raw execution paths (removing self-loops that do not change state while retaining progress and harmful steps), and define golden paths (loop-free, harm-free optimal paths).
3. **Five-Dimensional Evaluation**: PC, PC-KTC, PrefixCrit, HarmRate, and Efficiency.

### Key Designs

1. **Path Correctness (PC)**:

    - Function: Measures alignment between the agent's condensed path and a golden path.
    - Mechanism: Uses normalized Levenshtein distance $NLD(x,y) = \frac{2 \cdot LD(x,y)}{|x|+|y|+LD(x,y)}$; $PC = 1 - NLD$, taking the maximum over all golden paths.
    - Advantage: Tolerates paths of unequal length; edit operations correspond to meaningful agent deviations; provides continuous rather than binary scores.

2. **Path Correctness – Kendall's Tau Composite (PC-KTC)**:

    - Function: Augments PC with an assessment of action ordering.
    - Mechanism: $PC\text{-}KTC = \lambda \cdot PC + (1-\lambda) \cdot \tau^+$, where $\tau^+$ is a normalized Kendall-$\tau$ rank correlation over shared progress tokens.
    - Design Motivation: An agent may execute the correct set of tools in the wrong order (e.g., watering before opening the valve); PC alone cannot capture this failure.

3. **Prefix Criticality (PrefixCrit)**:

    - Function: Imposes heavier penalties on harmful calls that occur earlier in the trajectory.
    - Mechanism: $\text{PrefixCrit}_\beta = 1 - c(\beta, N)\sum_{k=0}^{N-1} m_k \beta^k$, using exponential decay $\beta^k$ as weights so that earlier harmful steps carry greater weight.
    - Design Motivation: Early errors have greater causal impact—they may invalidate all subsequent steps.

4. **Harm Rate (HarmRate)**:

    - Function: Computes the proportion of harmful (DFA-undefined transition) calls in the condensed path.
    - $\text{HarmRate} = \frac{1}{N} \sum m_k$

5. **Efficiency**:

    - Function: Measures how efficiently the agent reaches the goal relative to the shortest legitimate path.
    - $\text{Eff} = \ell^* / n$, where $\ell^*$ is the length of the longest golden path not exceeding $n$.
    - Note: Efficiency is computed on the raw (uncondensed) path; every call incurs cost.

6. **Harm-Local Refinement (HLR)**:

    - Function: Expands the reference path set to reduce over-penalization caused by localized errors.
    - Performs local repairs at harmful positions (deletion or substitution with a legitimate read operation) to generate additional valid reference paths.

### Evaluation Setup
- DFAs are constructed over multiple worlds (agricultural patrol vehicle, bank assistant, etc.).
- Models evaluated: GPT-o4-mini, GPT-4o-mini, Qwen3 series (0.6B–8B), and Qwen2.5 series (0.5B–7B).

## Key Experimental Results

### Main Results: CORE Metric Comparison Across Models

| Model | Harmful (total) | Eff | PC | PC-KTC | PrefixCrit | BFCL State% | PC+HLR |
|-------|-----------------|-----|----|--------|------------|-------------|--------|
| GPT-o4-mini | 124 | 0.748 | 0.812 | 0.834 | 0.896 | 79.8 | 0.858 |
| GPT-4o-mini | 189 | 0.675 | 0.715 | 0.744 | 0.834 | 71.7 | 0.755 |
| Qwen3-8b | 111 | 0.591 | 0.744 | 0.777 | 0.897 | 80.5 | 0.775 |
| Qwen2.5-7b | 252 | 0.291 | 0.460 | 0.598 | 0.845 | 68.3 | 0.649 |
| Qwen2.5-0.5b | High | Low | Low | Low | Low | Low | Low |

### Ablation Study: CORE vs. BFCL Final-State Evaluation

| Comparison | Finding |
|------------|---------|
| Models with similar BFCL State% | CORE metrics may differ substantially—e.g., Qwen3-8b and GPT-o4-mini both achieve ~80% BFCL, yet their efficiency gap is 0.157. |
| Harmful calls | BFCL entirely ignores intermediate harmful calls; CORE's HarmRate and PrefixCrit expose hidden risks. |
| HLR correction | PC+HLR is fairer than PC alone—it grants appropriate credit to agents that deviate from golden paths due to localized errors. |

### Key Findings
- **Final-state evaluation substantially overestimates agent capability**: An agent with BFCL State% = 80% may achieve a PC of only 0.46 (Qwen2.5-7b), indicating that although the terminal state is correct, the execution path deviates severely.
- **Model scale does not imply safety**: Qwen2.5-7b produces far more harmful calls (252) than the smaller Qwen3-0.6b (157).
- **The efficiency dimension reveals deployment suitability**: GPT-o4-mini achieves both the highest accuracy and the highest efficiency (0.748), making it well suited for edge deployment.
- **Ordering matters**: The gap between PC and PC-KTC shows that many agents invoke the correct tools but in the wrong order.

## Highlights & Insights
- **Modeling agent tasks as DFAs** is an excellent attempt to bring formal-verification thinking into agent evaluation—transforming "correct execution" from a vague final-state judgment into a precisely definable path-matching problem.
- **The five metrics form a complementary matrix**: correctness (PC / PC-KTC) × safety (PrefixCrit / HarmRate) × efficiency (Eff), each dimension independently valuable.
- **The conceptualization of compensating pairs and unobservable harms** is highly illuminating—these can serve as important failure-mode categories in agent safety research.

## Limitations & Future Work
- DFA construction currently requires manual definition of the task's state space and transition function, making generalization to arbitrary tasks difficult.
- Validation is limited to relatively simple world models (farm patrol vehicle, bank assistant); DFA formulation for complex real-world tasks (e.g., web agents, code agents) remains unclear.
- Golden paths are predefined and cannot accommodate creative yet correct solutions.
- Comparisons with other agent benchmarks (e.g., SWE-Bench, WebArena) are absent.
- Automatically learning DFAs from agent traces is a promising direction for future exploration.

## Related Work & Insights
- **vs. BFCL**: BFCL checks only the final state and response format; CORE evaluates the full path. Experiments show that agents deemed "equivalent" under BFCL may differ substantially under CORE.
- **vs. SWE-Bench**: SWE-Bench uses test cases (also a form of final-state evaluation); CORE's path-evaluation paradigm could complement SWE-Bench.
- **vs. Formal Verification**: CORE borrows DFA and state-machine concepts but targets LLM agent evaluation rather than program verification.

## Rating
- Novelty: ⭐⭐⭐⭐ The DFA + five-dimensional metric framework is a novel design that fills an important gap in agent evaluation.
- Experimental Thoroughness: ⭐⭐⭐ Multi-model evaluation is conducted, but the world models are relatively simple and large-scale real-task validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are clear and are supported by intuitive examples.
- Value: ⭐⭐⭐⭐ Offers practical guidance for safe agent deployment, particularly in edge computing scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents](agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)
- [\[NeurIPS 2025\] AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)
- [\[AAAI 2026\] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](../../AAAI2026/llm_agent/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](../../ICML2026/llm_agent/bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)

</div>

<!-- RELATED:END -->
