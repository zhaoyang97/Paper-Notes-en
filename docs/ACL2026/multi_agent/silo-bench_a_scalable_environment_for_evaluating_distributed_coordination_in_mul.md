---
title: >-
  [Paper Note] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems
description: >-
  [ACL 2026][Multi-Agent][Paper Note] This paper proposes SILO-BENCH, a role-agnostic benchmark for evaluating distributed coordination in multi-agent LLM systems. Comprising 30 algorithmic tasks across three communication complexity levels and 1620 experiments over 54 configurations, it reveals a critical "communication-reasoning gap": while agents can sp
tags:
  - ACL 2026
  - Multi-Agent
date: 2026-05-08
content_hash: c225d0be076f0f58
---
# SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems

**Conference**: ACL 2026  
**arXiv**: [2603.01045](https://arxiv.org/abs/2603.01045)  
**Code**: [https://github.com/jwyjohn/acl26-silo-bench](https://github.com/jwyjohn/acl26-silo-bench)  
**Area**: LLM Agent / Multi-Agent Systems  
**Keywords**: Multi-Agent Collaboration, Information Silos, Distributed Coordination, Communication-Reasoning Gap, Scalable Evaluation

## TL;DR

This paper proposes SILO-BENCH, a role-agnostic benchmark for evaluating distributed coordination in multi-agent LLM systems. Comprising 30 algorithmic tasks across three communication complexity levels and 1620 experiments over 54 configurations, it reveals a critical "communication-reasoning gap": while agents can spontaneously form rational communication topologies and actively exchange information, they systematically fail to integrate distributed states into correct answers.

## Background & Motivation

**Background**: The finite context window of LLMs is a fundamental bottleneck for processing large-scale information. Multi-agent systems (MAS) break the token limits of single models by distributing global information across multiple agents, similar to traditional distributed computing paradigms like MapReduce.

**Limitations of Prior Work**: (1) Existing multi-agent benchmarks either predefined fixed communication structures (e.g., CAMEL, MetaGPT) or focused on social simulation rather than computational collaboration (e.g., Generative Agents), both of which introduce inductive bias; (2) Role assignment (e.g., "Doctor", "Manager") entangles reasoning capabilities with semantic role priors, making it difficult to isolate the contribution of the communication architecture itself; (3) The core problem remains unexplored: Can LLMs coordinate within information silos to compute a globally correct result?

**Key Challenge**: Theoretically, multi-agents can break context limits through distributed collaboration, but in practice, it is unknown whether LLMs truly possess "distributed reasoning" capabilities—the ability to start from local information and achieve global consistency through coordination.

**Goal**: (1) Build a role-agnostic, configurable evaluation environment for distributed coordination; (2) Systematically study the impact of agent scale, communication protocols, and model capabilities on coordination performance; (3) Pinpoint specific stages of coordination failure.

**Key Insight**: Based on Yao’s communication complexity theory, tasks are categorized into three levels based on optimal communication complexity—Aggregation ($O(N)$, Star topology), Grid Network ($O(N)$, Chain transmission), and Global Shuffle ($O(N \log N)$ to $O(N^2)$, Full connectivity)—to theoretically anchor task difficulty.

**Core Idea**: A fundamental "communication-reasoning gap" exists in multi-agent LLM systems—agents can spontaneously discover appropriate communication topologies and exchange information sufficiently, yet they cannot correctly integrate the acquired distributed information into a global answer.

## Method

### Overall Architecture

The evaluation pipeline of SILO-BENCH is divided into four stages: (1) Data Partitioning—globally distributing input evenly among $N$ agents; (2) Agent Initialization—each agent receives task descriptions, local data, and communication protocols; (3) Collaborative Execution—agents communicate in parallel within $R_{\max}$ rounds, where each round involves simultaneous message reception, independent decision-making, and action execution; (4) Metric Calculation—calculating four metrics from submitted answers and communication logs.

### Key Designs

**1. Three-level Communication Complexity Task System: Theoretically anchoring task difficulty with Yao's communication complexity**

The task difficulty of existing multi-agent benchmarks is often ad hoc, making it difficult to distinguish whether failure stems from task complexity or coordination overhead. SILO-BENCH categorizes 30 algorithmic tasks into three levels based on optimal communication complexity: Level I (Aggregation) involves agents processing local data independently before summarization (optimal topology: Star, complexity: $O(N)$), e.g., global maximum, voting; Level II (Grid) involves computation where agent $i$ depends on neighbors $i-1$ and $i+1$ (optimal topology: Linear Chain, complexity: $O(N)$), e.g., prefix sum, moving average; Level III (Global Shuffle) involves any agent's output potentially depending on any other agent's information (complexity: $O(N\log N)$ to $O(N^2)$), e.g., distributed sorting, graph connectivity. With a theoretical scale for difficulty, performance differences can be cleanly attributed to coordination requirements rather than incidental task difficulty.

**2. Four-dimensional Evaluation Metrics: Characterizing both "what was solved" and "how they coordinated"**

Binary success rates underestimate partial progress and fail to reveal where coordination breaks down. SILO-BENCH reports four metrics: Success Rate $S$ measures the proportion of all agents converging to the correct answer; Partial Correctness Score $P$ provides a continuous measure of answer quality. Crucially, the $P-S$ gap isolates instances where information was acquired but failed at the integration stage; Token Consumption $C$ quantifies computational cost; Communication Density $D$ captures the intensity of interactions between agents. The juxtaposition of PCS and SR allows for precise localization of coordination failure stages—leading to the core conclusion that agents do not lack information, but rather the ability to integrate it.

**3. Role-Agnostic + Orthogonal Protocol Design: Decoupling communication architecture contributions from semantic role priors**

Labeling agents with roles like "Doctor" or "Manager" entangles reasoning performance with semantic priors, making it impossible to determine if performance stems from role heuristics or actual coordination capability. SILO-BENCH uses identical models for all agents without assigning roles, providing only task structure prompts. Simultaneously, three communication protocols are treated as orthogonal variables—P2P (Point-to-Point), BP (Broadcast), and SFS (Shared File System)—allowing agents to autonomously decide when, with whom, and what to share. Removing role-based confounding variables allows for the measurement of pure distributed computing capabilities and isolated analysis of protocol impacts.

### Loss & Training

SILO-BENCH is an evaluation benchmark and does not involve training. Task instances are programmatically generated using Python generators with fixed random seeds to ensure reproducibility and the ability to generate infinite new instances.

## Key Experimental Results

### Main Results

**Average performance of three models across different dimensions**

| Model | Success Rate (SR%) | Partial Correctness (PCS%) | Token Consumption | Comm. Density |
|------|-----------|-------------|----------|---------|
| DeepSeek-V3.1 | **36.9** | **47.1** | 323.0 | 0.82 |
| GPT-OSS-120B | 16.9 | 38.3 | 313.8 | 1.01 |
| Qwen3-Next-80B | 8.2 | 19.8 | 873.6 | 0.25 |

**By Task Difficulty Level (DeepSeek-V3.1)**

| Level | SR% | PCS% | SR-PCS Gap |
|------|-----|------|----------|
| Level I (Aggregation) | 62.0 | 88.0 | 26.0 |
| Level II (Grid) | 35.1 | 59.7 | 24.6 |
| Level III (Global Shuffle) | 11.7 | 27.9 | 16.2 |

### Ablation Study

**Coordination Overhead Analysis: Multi-Agent vs. Single-Agent Baseline (GPT-OSS-120B)**

| Scale k | Level I RCC | Level II RCC | Level III RCC |
|-------|------------|-------------|-------------|
| k=2 | 15.2% | 31.1% | 48.8% |
| k=5 | 30.3% | 32.9% | 70.0% |
| k=10 | 33.1% | 70.0% | 85.0% |
| k=50 | 45.9% | — | — |

### Key Findings

- **Communication-Reasoning Gap**: Agents can spontaneously form task-appropriate communication topologies (e.g., Agent 0 becoming a star aggregator in Level I tasks), but fail to integrate information correctly after obtaining it—at $N \geq 50$, Level III SR drops to 0% while PCS remains at 8-16%.
- Coordination overhead interacts **multiplicatively** with scale: Level I tasks maintain 40%+ SR at 100 agents, while Level III completely collapses at 50 agents.
- Communication protocol preferences vary by model: DeepSeek prefers broadcasting (BP 40% vs SFS 32%), while GPT prefers point-to-point (P2P 20% vs BP 14%).
- The performance gap between Level I and Level III for single agents is only about 15 percentage points, but for multi-agents, it balloons to 50+ points—proving that failure stems from coordination rather than the task itself.

## Highlights & Insights

- Theoretically anchoring task difficulty with communication complexity is an elegant design—bridging distributed computing theory with LLM evaluation for interpretable findings.
- The gap between PCS and SR precisely identifies "reasoning integration" as the bottleneck—agents do not lack information; they lack the ability to synthesize it.
- The discovery of emergent communication topologies is surprising: LLMs can spontaneously discover optimal patterns like stars and chains, indicating they understand task structures but cannot utilize the collected information effectively.

## Limitations & Future Work

- Evaluation is limited to homogeneous agents (all agents use the same model); heterogeneous combinations might exhibit different behaviors.
- Tasks are purely algorithmic, lacking natural language reasoning or knowledge-intensive distributed collaboration scenarios.
- Limits on communication rounds and task tokens may artificially constrain agent coordination capabilities.
- Future work could explore hierarchical coordination strategies (e.g., leader election) or specialized distributed reasoning training.

## Related Work & Insights

- **vs. CAMEL/MetaGPT**: These use role specialization and fixed workflows; SILO-BENCH is role-agnostic, isolating the pure contribution of communication architectures.
- **vs. LongBench/∞Bench**: These evaluate single-agent long-context capabilities; SILO-BENCH evaluates whether multi-agent collaboration can substitute for long contexts.
- **vs. Debate-based systems**: Debate systems focus on opinion convergence, whereas SILO-BENCH focuses on the computational correctness of distributed information.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic role-agnostic benchmark for distributed coordination with elegant theoretical anchoring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 54 configurations × 30 tasks = 1620 experiments, covering multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Logically clear, with tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐⭐ The discovery of the "communication-reasoning gap" has profound implications for multi-agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ICML 2025\] Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](../../ICML2025/multi_agent/cross-environment_cooperation_enables_zero-shot_multi-agent_coordination.md)
- [\[AAAI 2026\] Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems](../../AAAI2026/multi_agent/parallelism_meets_adaptiveness_scalable_documents_understanding_in_multi-agent_l.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] Explicit Trait Inference for Multi-Agent Coordination](explicit_trait_inference_for_multi-agent_coordination.md)

</div>

<!-- RELATED:END -->
