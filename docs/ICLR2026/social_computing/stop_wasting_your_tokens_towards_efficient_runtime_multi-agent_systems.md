---
title: >-
  [Paper Note] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems
description: >-
  [ICLR 2026][Social Computing][Multi-agent systems] This paper proposes SupervisorAgent, a lightweight real-time adaptive supervision framework that actively intervenes at critical interaction nodes (error correction, guidance provision, observation purification) via an LLM-free adaptive filter, reducing token consumption of Smolagent on the GAIA benchmark by 29.68% without sacrificing success rate.
tags:
  - ICLR 2026
  - Social Computing
  - Multi-agent systems
  - Token efficiency
  - Runtime supervision
  - Adaptive filtering
  - Error correction
date: 2026-05-08
content_hash: 235389b2f52eca35
---

# Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems

**Conference**: ICLR 2026
**arXiv**: [2510.26585](https://arxiv.org/abs/2510.26585)
**Code**: None
**Area**: Social Computing
**Keywords**: Multi-agent systems, Token efficiency, Runtime supervision, Adaptive filtering, Error correction

## TL;DR

This paper proposes SupervisorAgent, a lightweight real-time adaptive supervision framework that actively intervenes at critical interaction nodes (error correction, guidance provision, observation purification) via an LLM-free adaptive filter, reducing token consumption of Smolagent on the GAIA benchmark by 29.68% without sacrificing success rate.

## Background & Motivation

- Multi-agent systems (MAS) excel at complex tasks but face an **efficiency–robustness paradox**:
    - **Error propagation**: A single hallucinated piece of information can corrupt the entire downstream reasoning chain.
    - **Inefficient behavior**: Agents fall into repetitive action loops or select unnecessarily complex execution paths.
    - **Context bloat**: Verbose tool returns (e.g., raw HTML) flood the context window.
- Existing methods primarily focus on **post-hoc failure attribution**, lacking **real-time proactive intervention**.

## Method

### Overall Architecture: Supervised Multi-Agent System (SMAS)

A meta-level control agent — SupervisorAgent — is added on top of the existing MAS to monitor three categories of high-risk interactions in real time:

1. **Agent–Agent interactions**: Inter-agent communication and delegation, prone to propagating hallucinated information.
2. **Agent–Tool interactions**: External tool calls that may return inaccurate, irrelevant, or outdated data.
3. **Agent–Memory interactions**: Memory retrieval that may surface stale or defective historical information.

### Adaptive Filter (When to Supervise)

A lightweight, **LLM-free** heuristic filter that triggers supervision only at critical nodes:

- **Error occurrence** $c_{error}$: Explicit errors such as tool call failures or code execution errors.
- **Inefficient behavior** $c_{inefficient}$: Repetitive action loops (e.g., repeated `page_down`).
- **Excessive observation** $c_{excessive}$: Tool returns exceeding a length threshold (e.g., raw HTML).

### Context Window

$$\mathcal{W} = (N, Q_g, Q_l, T_l, S)$$

- $N$: Name of the supervised agent
- $Q_g, Q_l$: Global goal and local task
- $T_l$: Local action trajectory
- $S$: Summary of the most recent interaction
- The extended version $\mathcal{W}_{ext}$ incorporates the global trajectory $T_g$ for diagnosing system-level inefficiencies.

### Multi-Level Intervention Action Space (How to Supervise)

| Action | Intensity | Trigger Condition | Description |
|--------|-----------|-------------------|-------------|
| **approve** | Lowest | $c_{inefficient}$ | Permits valid repetitive behavior to continue |
| **provide_guidance** | Moderate | $c_{error}, c_{inefficient}$ | Appends guidance hints to correct the reasoning path |
| **correct_observation** | High | $c_{error}, c_{excessive}$ | Replaces or purifies the raw observation content |
| **run_verification** | Highest | $c_{error}$ | Invokes a verification sub-agent for external fact-checking |

### Core Design Principles

- **Non-invasive**: Does not modify the underlying agent architecture.
- **Adaptive**: Supervision is triggered only at high-risk nodes, not at every interaction.
- **Memory-augmented**: SupervisorAgent maintains a more comprehensive view of system state than any individual agent.

## Key Experimental Results

### Main Results on GAIA Benchmark

| Method | Avg. Accuracy | Avg. Tokens (K) | L2 Tokens (K) |
|--------|--------------|-----------------|---------------|
| CodeAgent | 40.00 | 120.40 | — |
| Smolagent (pass@1) | — | Baseline | Baseline |
| **SMAS (pass@1)** | On par | **−29.68%** | **−35%** |

### Detailed Analysis on GAIA Level 2

| Metric | Smolagent | SMAS | Improvement |
|--------|-----------|------|-------------|
| Token consumption | Baseline | −35% | Significant |
| Variance | Baseline | −63% | Substantially reduced |
| Number of steps (case) | Baseline | −43% | Significant |

### Cross-Benchmark Validation

| Benchmark | Domain | Token Reduction | Accuracy Change |
|-----------|--------|----------------|-----------------|
| HumanEval | Code generation | **−23.74%** | Improved |
| MBPP | Code generation | Significant | On par / improved |
| AIME 2024 | Math reasoning | Reduced | On par |
| GSM8k-Hard | Math reasoning | Reduced | On par |
| DROP | QA | Reduced | On par |

### Key Findings

1. A 23.74% token reduction on HumanEval is achieved simultaneously with an improvement in accuracy.
2. SupervisorAgent is effective across GPT-4.1, Gemini-2.5-pro, and the Qwen3 model series.
3. The adaptive filter effectively controls supervision overhead, avoiding redundant checks on every interaction.
4. Case studies show that a single successful supervisory intervention can reduce token consumption by more than 70%.

## Highlights & Insights

- **Real-time proactive intervention vs. post-hoc analysis**: A paradigm shift from reactive to proactive supervision.
- **Pareto improvement**: Token consumption is reduced without sacrificing — and in some cases while improving — task success rate.
- **LLM-free filter**: A key innovation lies in using simple heuristics rather than an LLM to determine *when* to supervise.
- **Orthogonal to existing methods**: The framework can be stacked on top of any existing MAS framework.
- **Substantially reduced variance**: The resulting system exhibits more stable and reliable behavior.

## Limitations & Future Work

- The adaptive filter relies on predefined heuristic rules and may fail to capture emerging categories of high-risk interactions.
- The LLM calls made by SupervisorAgent itself incur costs, necessitating a trade-off between supervision benefit and supervision overhead.
- Validation is primarily conducted on the Smolagent framework; adaptation to other MAS frameworks may require non-trivial adjustments.
- The framework has limited applicability to purely conversational tasks that do not involve tool use.

## Related Work & Insights

- Failure attribution: Post-hoc analysis methods such as Aegis and AgenTracer.
- Efficiency optimization: AgentDropout (agent pruning), MetaAgent (design-time topology optimization).
- Context compression: Observation summarization and distillation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of runtime supervision is novel; the non-invasive design is practically valuable.
- **Technical Depth**: ⭐⭐⭐ — The method is relatively intuitive with moderate technical complexity.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 6 benchmarks × multiple backbone models × detailed case analysis.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly deployable; of significant value for reducing MAS operational costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/social_computing/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ICLR 2026\] Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)
- [\[ICLR 2026\] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI](functional_embeddings_enable_aggregation_of_multi-area_seeg_recordings_over_subj.md)

</div>

<!-- RELATED:END -->
