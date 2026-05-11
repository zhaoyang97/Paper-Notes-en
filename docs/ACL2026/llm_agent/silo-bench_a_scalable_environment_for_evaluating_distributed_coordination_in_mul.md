---
title: >-
  [Paper Note] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems
description: >-
  [ACL 2026][LLM Agent][multi-agent collaboration] This paper introduces SILO-BENCH, a role-agnostic benchmark for evaluating distributed coordination in multi-agent LLM systems. It comprises 30 algorithmic tasks across th…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "multi-agent collaboration"
  - "information silos"
  - "distributed coordination"
  - "communication-reasoning gap"
  - "scalable evaluation"
date: 2026-05-08
content_hash: e866d95e144bba84
---

# SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems

**Conference**: ACL 2026
**arXiv**: [2603.01045](https://arxiv.org/abs/2603.01045)
**Code**: [https://github.com/jwyjohn/acl26-silo-bench](https://github.com/jwyjohn/acl26-silo-bench)
**Area**: LLM Agent / Multi-Agent Systems
**Keywords**: multi-agent collaboration, information silos, distributed coordination, communication-reasoning gap, scalable evaluation

## TL;DR

This paper introduces SILO-BENCH, a role-agnostic benchmark for evaluating distributed coordination in multi-agent LLM systems. It comprises 30 algorithmic tasks across three communication complexity levels, with 54 configurations yielding 1,620 experiments. The benchmark reveals a critical **communication-reasoning gap**: agents spontaneously form reasonable communication topologies and actively exchange information, yet systematically fail to integrate distributed state into correct answers.

## Background & Motivation

**Background**: The limited context window of LLMs is a fundamental bottleneck for processing large-scale information. Multi-agent systems (MAS) address this by distributing global information across multiple agents, analogous to classical distributed computing paradigms such as MapReduce.

**Limitations of Prior Work**: (1) Existing multi-agent benchmarks either predefine fixed communication structures (CAMEL, MetaGPT) or focus on social simulation rather than computational collaboration (Generative Agents), both introducing inductive biases. (2) Role assignments (e.g., "doctor", "manager") entangle agents' reasoning capabilities with semantic role priors, making it difficult to isolate the contribution of communication architecture. (3) A core question remains unexplored: can LLMs compute globally correct answers through coordination under information silo conditions?

**Key Challenge**: While multi-agent systems theoretically overcome context limitations via distributed collaboration, it remains unknown whether LLMs genuinely possess "distributed reasoning" capabilities—i.e., the ability to start from local information and achieve global consistency through coordination.

**Goal**: (1) Construct a role-agnostic, configurable evaluation environment for distributed coordination; (2) Systematically study the effects of agent scale, communication protocols, and model capability on coordination performance; (3) Pinpoint the specific stage at which coordination fails.

**Key Insight**: Drawing on Yao's communication complexity theory, tasks are partitioned into three levels by optimal communication complexity—Aggregation ($O(N)$, star topology), Mesh Network ($O(N)$, chain propagation), and Global Shuffle ($O(N \log N)$ to $O(N^2)$, fully connected)—providing a theoretical anchor for task difficulty.

**Core Idea**: Multi-agent LLM systems exhibit a fundamental **communication-reasoning gap**: agents can spontaneously discover appropriate communication topologies and exchange information sufficiently, yet fail to correctly integrate the acquired distributed information into a global answer.

## Method

### Overall Architecture

The SILO-BENCH evaluation pipeline consists of four stages: (1) **Data Partitioning**—the global input is evenly distributed among $N$ agents; (2) **Agent Initialization**—each agent receives a task description, local data, and communication protocol; (3) **Collaborative Execution**—agents communicate in parallel over $R_{\max}$ rounds, with all agents simultaneously receiving messages, making independent decisions, and executing actions each round; (4) **Metric Computation**—four metrics are computed from submitted answers and communication logs.

### Key Designs

1. **Three-Level Communication Complexity Task Suite**

    - **Function**: Theoretically anchors task difficulty to ensure that performance differences are attributable to coordination demands.
    - **Mechanism**: Level I (Aggregation)—each agent independently processes local data before aggregation; optimal topology is star-shaped (e.g., global maximum, voting). Level II (Mesh)—agent $i$'s computation depends on neighboring agents $i{-}1$ and $i{+}1$; optimal topology is a linear chain (e.g., prefix sum, sliding average). Level III (Global Shuffle)—any agent's output may depend on any other agent's information (e.g., distributed sorting, graph connectivity).
    - **Design Motivation**: Task difficulty in existing benchmarks is ad hoc, making it impossible to distinguish whether failures stem from task complexity or coordination overhead.

2. **Four-Dimensional Evaluation Metrics**

    - **Function**: Simultaneously captures *what was done* and *how coordination proceeded*.
    - **Mechanism**: Success Rate $S$ measures the proportion of cases where all agents converge to the correct answer; Partial Correctness Score $P$ provides a continuous measure of answer quality (the gap $P - S$ isolates failures in the reasoning integration stage); Token Consumption $C$ quantifies computational cost; Communication Density $D$ captures the intensity of inter-agent interaction.
    - **Design Motivation**: Binary success rate underestimates partial progress; introducing PCS enables precise localization of the stage at which coordination breaks down.

3. **Role-Agnostic Design with Orthogonal Communication Protocols**

    - **Function**: Isolates the contribution of communication architecture, avoiding interference from semantic role priors.
    - **Mechanism**: All agents use the same model with no role assignment, receiving only task-structural prompts. Three communication protocols are varied orthogonally—P2P (directed messaging), BP (broadcast), and SFS (shared file system)—with agents autonomously deciding when, with whom, and what to share.
    - **Design Motivation**: Role-specialized multi-agent systems cannot disentangle whether performance derives from role heuristics or coordination capability; a role-agnostic design measures genuine distributed computation ability.

### Loss & Training

SILO-BENCH is an evaluation benchmark and does not involve training. Task instances are generated programmatically via Python generators under fixed random seeds, ensuring reproducibility and the ability to produce an unlimited number of novel instances.

## Key Experimental Results

### Main Results

**Average Performance of Three Models Across Dimensions**

| Model | SR (%) | PCS (%) | Token Consumption | Communication Density |
|---|---|---|---|---|
| DeepSeek-V3.1 | **36.9** | **47.1** | 323.0 | 0.82 |
| GPT-OSS-120B | 16.9 | 38.3 | 313.8 | 1.01 |
| Qwen3-Next-80B | 8.2 | 19.8 | 873.6 | 0.25 |

**By Task Difficulty Level (DeepSeek-V3.1)**

| Level | SR (%) | PCS (%) | SR–PCS Gap |
|---|---|---|---|
| Level I (Aggregation) | 62.0 | 88.0 | 26.0 |
| Level II (Mesh) | 35.1 | 59.7 | 24.6 |
| Level III (Global Shuffle) | 11.7 | 27.9 | 16.2 |

### Ablation Study

**Coordination Overhead Analysis: Multi-Agent vs. Single-Agent Baseline (GPT-OSS-120B)**

| Scale $k$ | Level I RCC | Level II RCC | Level III RCC |
|---|---|---|---|
| $k=2$ | 15.2% | 31.1% | 48.8% |
| $k=5$ | 30.3% | 32.9% | 70.0% |
| $k=10$ | 33.1% | 70.0% | 85.0% |
| $k=50$ | 45.9% | — | — |

### Key Findings

- **Communication-Reasoning Gap**: Agents spontaneously form task-adaptive communication topologies (e.g., Agent 0 self-organizes as a star aggregator on Level I tasks), yet fail to correctly integrate information once it is sufficiently gathered—at $N \geq 50$, Level III SR drops to 0% while PCS remains at 8–16%.
- Coordination overhead interacts **multiplicatively** with scale: Level I tasks sustain 40%+ SR at 100 agents, whereas Level III collapses entirely at 50 agents.
- Communication protocol preferences differ by model: DeepSeek favors broadcast (BP 40% vs. SFS 32%), while GPT favors point-to-point (P2P 20% vs. BP 14%).
- The single-agent performance gap from Level I to Level III is only ~15 percentage points, whereas the multi-agent gap expands to 50+ percentage points—demonstrating that failures stem from coordination rather than the tasks themselves.

## Highlights & Insights

- Anchoring task difficulty to communication complexity theory is an elegant design choice—it bridges distributed computing theory and LLM evaluation, lending interpretability to empirical findings.
- The gap between PCS and SR precisely identifies **reasoning integration** as the bottleneck: agents do not lack information, but rather the capacity to integrate it.
- The emergence of communication topologies is a striking finding: LLMs can spontaneously discover optimal communication patterns such as star and chain structures, suggesting they understand task structure but cannot exploit the information they collect.

## Limitations & Future Work

- Only homogeneous agents (all using the same model) are evaluated; heterogeneous agent combinations may exhibit different behavior.
- All tasks are algorithmic in nature; distributed collaboration in natural language reasoning or knowledge-intensive scenarios is not covered.
- The cap on communication rounds and token limits may artificially constrain agents' coordination capacity.
- Future work could explore hierarchical coordination strategies (e.g., leader election) or targeted training for distributed reasoning.

## Related Work & Insights

- **vs. CAMEL/MetaGPT**: These frameworks rely on role specialization and fixed workflows; SILO-BENCH is role-agnostic, isolating the pure contribution of communication architecture.
- **vs. LongBench/∞Bench**: These benchmarks evaluate single-agent long-context capability; SILO-BENCH evaluates whether multi-agent collaboration can serve as a substitute for long context.
- **vs. Debate-based systems**: Debate systems focus on opinion convergence; SILO-BENCH focuses on the computational correctness of distributed information processing.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic role-agnostic benchmark for distributed coordination, with an elegant theoretical grounding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 54 configurations × 30 tasks = 1,620 experiments, covering multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Logic is clear and the integration of theory with experiments is tight.
- **Value**: ⭐⭐⭐⭐⭐ — The discovery of the communication-reasoning gap has far-reaching implications for multi-agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[AAAI 2026\] Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems](../../AAAI2026/llm_agent/parallelism_meets_adaptiveness_scalable_documents_understanding_in_multi-agent_l.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)

</div>

<!-- RELATED:END -->
