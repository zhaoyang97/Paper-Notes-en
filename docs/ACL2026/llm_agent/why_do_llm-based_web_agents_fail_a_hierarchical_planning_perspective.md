---
title: >-
  [Paper Note] Why LLM Web Agents Fail: A Hierarchical Planning Perspective
description: >-
  [ACL 2026][LLM Agent][Web agent failure analysis] This paper systematically analyzes the failure causes of LLM web agents through a hierarchical planning framework (high-level planning, low-level execution…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Web agent failure analysis"
  - "hierarchical planning"
  - "natural language vs PDDL"
  - "execution bottlenecks"
date: 2026-05-08
content_hash: 6bfbeb715a5ea8ae
---

# Why LLM Web Agents Fail: A Hierarchical Planning Perspective

**Conference**: ACL 2026  
**arXiv**: [2603.14248](https://arxiv.org/abs/2603.14248)  
**Code**: https://github.com/Ziyu-Yao-NLP-Lab/llm-hierarchical-web-agents  
**Area**: LLM Agent / Web Navigation  
**Keywords**: Web agent failure analysis, hierarchical planning, natural language vs PDDL, execution bottlenecks

## TL;DR

This paper systematically analyzes the failure causes of LLM web agents through a hierarchical planning framework (high-level planning, low-level execution, and replanning). It finds that PDDL representation outperforms natural language planning, but low-level execution and perceptual grounding are the primary bottlenecks.

## Background & Motivation

**Background**: The performance of LLM web agents on long-horizon tasks remains significantly lower than human levels. However, existing evaluations focus primarily on end-to-end success rates, offering limited understanding of specific failure sources.

**Limitations of Prior Work**: End-to-end evaluation metrics (e.g., task success rate) mask the real issues—they cannot distinguish whether a failure stems from high-level planning errors, insufficient low-level execution, or the failure of replanning mechanisms.

**Key Challenge**: Different components have distinct bottlenecks, but existing methods optimize overall performance indiscriminately, leading to obscure improvement directions.

**Goal**: To establish a systematic hierarchical evaluation framework that decomposes web agent capabilities into three independent dimensions for diagnosis.

**Key Insight**: Inspired by automated planning (e.g., HTN planning), humans solve complex tasks via a three-layer process: "abstract strategy $\rightarrow$ concrete execution $\rightarrow$ dynamic replanning." LLM agents should be decomposable using the same logic.

**Core Idea**: Utilize a hierarchical planning framework instead of black-box end-to-end evaluation to precisely locate the failure causes of LLM agents.

## Method

### Overall Architecture

The proposed hierarchical planning evaluation framework consists of a four-stage process:

1.  **High-level Planning**: The LLM decomposes a natural language instruction into a sequence of high-level subgoals $P = [g_1, g_2, \ldots, g_n]$, where each $g_i$ represents an abstract, meaningful step.
2.  **Low-level Execution**: For each subgoal $g_i$, the agent generates a series of executable low-level actions $a_t \in \mathcal{A}$, producing an execution trajectory $\tau_i = (o_t, a_t, o_{t+1}, \ldots, o_{t+k})$.
3.  **Post-condition Check**: An LLM is used as a judge to verify if the execution result satisfies the intended effect of the subgoal, formulated as $\Phi(g_i, s') = 1$.
4.  **Replanning**: If a subgoal fails or hits a dead end, the agent decides between local adjustment (continuing from the last successful subgoal) or global replanning (generating a new plan from scratch).

### Key Designs

1.  **PDDL vs Natural Language Representation**:
    *   Function: Compares the impact of two high-level planning representations on plan quality.
    *   Mechanism: Natural language (NL) is flexible but prone to over-specification (implicit requirements) or over-decomposition. PDDL enforces clear plan semantics via formal structures (preconditions, effects).
    *   Design Motivation: NL plans often incorporate low-level details. This design tests whether PDDL can generate more abstract and executable plans through symbolic constraints.

2.  **Multi-dimensional Failure Mode Analysis**:
    *   Function: Measures performance across three dimensions: planning alignment, execution reliability, and replanning effectiveness.
    *   Mechanism: Defines six alignment metrics (Perfect Match / Partial / Missing / Decomposed / Unmatched / Matched Rate) to quantify the deviation of high-level plans from human reference plans. It also defines low-level execution metrics (subgoal completion rate / plan completion rate / task success rate / action efficiency) and evaluates performance changes before and after replanning.
    *   Design Motivation: Isolating capabilities allows for clearer improvement directions—for example, if low-level execution is poor despite high-quality planning, the focus should be on perception grounding rather than planning capacity.

3.  **LLM as a Judge for Verification**:
    *   Function: Automatically judges subgoal completion and overall task success.
    *   Mechanism: Uses gpt-5-nano as a judge to determine goal achievement based on execution trajectories and final webpage states. Human verification on 50 samples showed an accuracy of 82%-86%.
    *   Design Motivation: In real web environments, rule-based success judgment is fragile; LLMs can understand semantics rather than relying on mechanical matching.

## Key Experimental Results

### High-level Planning: Natural Language vs PDDL

| Metric | NL (Pre-replan) | PDDL (Pre-replan) | NL (Post-replan) | PDDL (Post-replan) |
| :--- | :--- | :--- | :--- | :--- |
| Perfect Match | 60.6% | 67.7% | 56.1% | 59.0% |
| Partial | 5.7% | 7.4% | 6.1% | 6.9% |
| Missing | 4.2% | 2.2% | 4.0% | 14.5% |
| Decomposed | 29.5% | 22.7% | 33.8% | 19.6% |
| Unmatched | 29.4% | 15.4% | 35.0% | 15.4% |
| Matched (Effective steps) | 70.6% | 84.6% | 65.0% | 84.6% |

**Key Findings**: PDDL plans achieve a higher Perfect Match rate (67.7% vs 60.6%) and fewer Unmatched steps (15.4% vs 29.4%). NL plans tend toward over-decomposition (29.5%), leading to redundant steps.

### Low-level Execution: Identifying the True Bottleneck

| Dataset Metric | gpt-5-nano (human plan) | gpt-5-nano (NL plan) | gpt-5-nano (PDDL plan) |
| :--- | :--- | :--- | :--- |
| Subgoal Completion Rate | 38.5% | 26.8% | 32.1% |
| Plan Completion Rate | 38.5% | - | - |
| Final Task Success Rate | 36.4% | 18.5% | 24.7% |

**Low-level Execution Failure Modes**:

| Failure Mode | Incidence | Root Cause |
| :--- | :--- | :--- |
| Hallucinated links (goto action) | 32.0% | LLM invents non-existent URLs |
| Redundant actions | 34.2% | Insufficient environment understanding; invalid operations |
| Out-of-domain links | 16.7% | Navigating away from target site (e.g., to Wikipedia via search) |
| Repetitive execution | 10.4% | Failure to learn from feedback; getting stuck in loops |

**Key Findings**: Even when provided with perfect human-annotated high-level plans, the LLM executor success rate is only 36.4%, indicating that **low-level execution and perceptual grounding are the true bottlenecks**.

### Effect of Replanning

| Configuration | Subgoal Completion | Task Success | Gain |
| :--- | :--- | :--- | :--- |
| Pre-replan (NL) | 26.8% | 18.5% | Base |
| Post-replan (NL) | 31.2% | 22.3% | +4.4pp |
| Pre-replan (PDDL) | 32.1% | 24.7% | Base |
| Post-replan (PDDL) | 35.5% | 28.9% | +3.4pp |

Single-turn replanning improves success rates by 4-5 percentage points, suggesting that while the replanning mechanism is effective, its impact is limited.

### Comparison Across LLMs

*   **gpt-5-nano**: Strongest performance, 36.4% task success rate (human plan).
*   **claude-haiku-4.5**: 29.2% success rate; highest rate of repetitive failures (16.7%), indicating weak feedback utilization.
*   **gemini-flash-2.5**: 17.3% success rate; worst low-level execution with the highest redundant action rate (41.2%), despite compact planning.

## Highlights & Insights

*   **Innovation in Hierarchical Diagnostic Framework**: Rather than aiming for incremental end-to-end gains, the paper provides a systematic method to isolate and evaluate three layers of capability. This approach, borrowed from automated planning, is applied to LLM web agent failure analysis for the first time.
*   **Quantitative Advantage of PDDL**: The paper quantitatively proves that formal representations outperform natural language. Although PDDL has a higher learning curve, it yields more precise plans with less redundancy and higher executability.
*   **Low-level Execution as the Core Issue**: The study challenges the common assumption that "improving LLM reasoning will automatically improve web agents." Evidence shows that even with perfect high-level planning, execution fails at a rate of 63.6%, shifting the focus toward perceptual grounding.
*   **Fine-grained Classification of Failure Modes**: Categorizing execution failures into hallucinated links, redundant actions, out-of-domain jumps, and repetitive loops (each accounting for 16-34%) provides a concrete roadmap for optimization.
*   **Limited Utility of Simple Replanning**: Gains from single-turn replanning are modest (+4-5pp), indicating a need for more sophisticated adaptation mechanisms beyond simple retries.

## Limitations & Future Work

**Author-acknowledged Limitations**:
*   Experiments focused on a limited set of high-level representations (NL and PDDL), action spaces (3 types), and agent configurations.
*   Does not account for multimodal settings (visual information).
*   High-level plan evaluation requires human-annotated reference plans, which limits flexibility.

**Self-identified Limitations**:
*   Evaluation is restricted to 104 tasks from Mind2Web-Live, which is a relatively small sample size.
*   While LLM-as-a-Judge has 82%-86% accuracy, it may still misjudge edge cases in highly complex web pages.
*   Replanning was explored for only one round; multi-round iterative convergence remains unstudied.
*   The potential of hybrid options (e.g., PDDL planning combined with neural network low-level executors) was not explored.

**Specific Improvement Directions**:
1.  **Perceptual Grounding**: Incorporate visual features or structured page representations (e.g., symbolic DOM trees) to mitigate link hallucinations.
2.  **Action Space Design**: Allow agents to explicitly express "uncertainty" or "need for clarification" rather than forced guessing.
3.  **Distributed Execution**: Decouple the planning module (using PDDL) from the execution module (using specialized tools or neural nets) to optimize each independently.
4.  **Multi-round Replanning Strategies**: Design adaptive feedback mechanisms that teach the agent when to backtrack versus when to proceed.

## Related Work & Insights

*   **vs. WebArena/Mind2Web (End-to-End Evaluation)**: These benchmarks only measure success rates without diagnosing the source of failure. This work redefines evaluation through a diagnostic lens rather than pure performance testing.
*   **vs. Prior PDDL Planning Work (Silver et al.)**: Previous research applied PDDL+LLM to classical planning; this paper introduces it to web agents with quantitative comparisons, finding formal representations maintain advantages in open-world web environments.
*   **vs. Low-level Execution Improvements (e.g., WALT tool calling)**: While those methods improve execution using APIs, they do not diagnose *why* execution fails. This framework is orthogonally compatible with those directions.
*   **vs. Adaptive Agents (e.g., Reflexion)**: While these methods use feedback for improvement, this paper quantifies the specific gain of single-round replanning (+4-5pp) and highlights the need for multi-layered feedback.

## Rating

*   Novelty: ⭐⭐⭐⭐ Applying hierarchical planning to web agent diagnostics is a fresh perspective. The quantitative PDDL vs. NL comparison is a significant contribution.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Rigorous design covering multiple LLMs and multi-dimensional metrics, though the task count (104) is somewhat low. Excellent failure mode analysis.
*   Writing Quality: ⭐⭐⭐⭐⭐ Extremely logical flow from motivation through framework to experiments and recommendations. High readability.
*   Value: ⭐⭐⭐⭐ Highly practical, providing clear guidance to the community by identifying low-level execution as the primary bottleneck over planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[AAAI 2026\] When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents](../../AAAI2026/llm_agent/when_refusals_fail_unstable_safety_mechanisms_in_long-context_llm_agents.md)
- [\[ICML 2026\] Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling](../../ICML2026/llm_agent/agent_jit_compilation_for_latency-optimizing_web_agent_planning_and_scheduling.md)
- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](synthagent_adapting_web_agents_with_synthetic_supervision.md)

</div>

<!-- RELATED:END -->
