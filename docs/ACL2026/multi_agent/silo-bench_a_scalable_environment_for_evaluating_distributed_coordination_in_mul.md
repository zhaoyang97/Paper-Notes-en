---
title: >-
  [Paper Note] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems
description: >-
  [ACL 2026][Multi-Agent][Multi-agent collaboration] This paper proposes SILO-BENCH, a role-agnostic benchmark for distributed coordination in multi-agent LLM systems. It comprises 30 algorithmic tasks…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Multi-agent collaboration"
  - "Information Silos"
  - "Distributed Coordination"
  - "Communication-Reasoning Gap"
  - "Scalable Evaluation"
date: 2026-05-08
content_hash: f425d754bd33e899
---

# SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems

**Conference**: ACL 2026  
**arXiv**: [2603.01045](https://arxiv.org/abs/2603.01045)  
**Code**: [https://github.com/jwyjohn/acl26-silo-bench](https://github.com/jwyjohn/acl26-silo-bench)  
**Area**: LLM Agent / Multi-Agent Systems  
**Keywords**: Multi-agent collaboration, Information Silos, Distributed Coordination, Communication-Reasoning Gap, Scalable Evaluation

## TL;DR

This paper proposes SILO-BENCH, a role-agnostic benchmark for distributed coordination in multi-agent LLM systems. It comprises 30 algorithmic tasks, three communication complexity levels, and 1,620 experiments across 54 configurations. The study reveals a critical "communication-reasoning gap": while agents can spontaneously form reasonable communication topologies and actively exchange information, they systematically fail to integrate distributed states into correct answers.

## Background & Motivation

**Background**: The finite context window of LLMs is a fundamental bottleneck for processing large-scale information. Multi-agent systems (MAS) break the token limits of single models by distributing global information across multiple agents, similar to traditional distributed computing paradigms like MapReduce.

**Limitations of Prior Work**: (1) Existing multi-agent benchmarks either predefine fixed communication structures (CAMEL, MetaGPT) or focus on social simulation rather than computational collaboration (Generative Agents), introducing inductive biases. (2) Role assignment (e.g., "doctor", "manager") entangles reasoning ability with semantic role priors, making it difficult to isolate the contribution of the communication architecture itself. (3) A core problem remains unexplored: Can LLMs compute a global correct answer through coordination within information silos?

**Key Challenge**: Theoretically, multi-agents can exceed context limits through distributed collaboration, but in practice, it is unknown whether LLMs truly possess "distributed reasoning" capabilities—the ability to reach global consistency starting from local information through coordination.

**Goal**: (1) Build a role-agnostic, configurable evaluation environment for distributed coordination; (2) Systematically study the impact of agent scale, communication protocols, and model capabilities on coordination performance; (3) Locate the specific stages where coordination fails.

**Key Insight**: Based on Yao’s communication complexity theory, tasks are categorized into three levels according to optimal communication complexity: aggregation ($O(N)$, star topology), mesh network ($O(N)$, chain passing), and global shuffle ($O(N \log N)$ to $O(N^2)$, fully connected), anchoring task difficulty theoretically.

**Core Idea**: A fundamental "communication-reasoning gap" exists in multi-agent LLM systems—agents can spontaneously discover appropriate communication topologies and exchange information sufficiently, but they cannot correctly integrate the acquired distributed information into a global answer.

## Method

### Overall Architecture

The SILO-BENCH evaluation pipeline consists of four stages: (1) Data Partitioning—uniformly distributing global input to $N$ agents; (2) Agent Initialization—each agent receives task descriptions, local data, and communication protocols; (3) Collaborative Execution—agents communicate in parallel within $R_{\max}$ rounds, where in each round all agents simultaneously receive messages, make independent decisions, and execute actions; (4) Metric Calculation—computing four metrics from submitted answers and communication logs.

### Key Designs

1. **Three-Level Communication Complexity Task System**:
    - **Function**: Anchors task difficulty based on theory to ensure performance differences are attributable to coordination requirements.
    - **Mechanism**: Level I (Aggregation)—each agent processes local data independently before aggregation (optimal topology is a star), e.g., global maximum, voting; Level II (Grid)—computation for agent $i$ depends on neighbors $i-1$ and $i+1$ (optimal topology is a linear chain), e.g., prefix sum, moving average; Level III (Global Shuffle)—output of any agent may depend on information from any other agent, e.g., distributed sorting, graph connectivity.
    - **Design Motivation**: Task difficulty in existing benchmarks is ad hoc, failing to distinguish if failure stems from the task itself or coordination overhead.

2. **Four-Dimensional Evaluation Metrics**:
    - **Function**: Captures both outcomes and coordination processes.
    - **Mechanism**: Success Rate $S$ measures the proportion of all agents converging to the correct answer; Partial Correctness Score $P$ provides a continuous measure of answer quality (the gap $P - S$ isolates failures in the reasoning integration stage); Token Consumption $C$ quantifies computational cost; Communication Density $D$ captures interaction intensity between agents.
    - **Design Motivation**: Binary success rates underestimate partial progress; introducing PCS allows precise localization of where coordination collapses.

3. **Role-Agnostic + Orthogonal Communication Protocol Design**:
    - **Function**: Isolates the contribution of the communication architecture and avoids interference from role semantic priors.
    - **Mechanism**: All agents use the same model with no role assignment, provided only with task structure prompts. Three communication protocols vary orthogonally: P2P (directed messages), BP (broadcast), and SFS (shared file system), with agents autonomously deciding when, with whom, and what to share.
    - **Design Motivation**: Role-based MAS cannot distinguish if performance comes from role heuristics or coordination ability; role-agnostic design measures true distributed computing capabilities.

### Loss & Training

SILO-BENCH is an evaluation benchmark and does not involve training. Task instances are programmatically generated using Python generators under fixed random seeds to ensure reproducibility and the ability to generate infinite new instances.

## Key Experimental Results

### Main Results

**Average performance of three models across different dimensions**

| Model | Success Rate (SR%) | Partial Correctness (PCS%) | Token Consum. | Comm. Density |
| :--- | :--- | :--- | :--- | :--- |
| DeepSeek-V3.1 | **36.9** | **47.1** | 323.0 | 0.82 |
| GPT-OSS-120B | 16.9 | 38.3 | 313.8 | 1.01 |
| Qwen3-Next-80B | 8.2 | 19.8 | 873.6 | 0.25 |

**Performance by Task Difficulty Level (DeepSeek-V3.1)**

| Level | SR% | PCS% | SR-PCS Gap |
| :--- | :--- | :--- | :--- |
| Level I (Aggregation) | 62.0 | 88.0 | 26.0 |
| Level II (Grid) | 35.1 | 59.7 | 24.6 |
| Level III (Global Shuffle) | 11.7 | 27.9 | 16.2 |

### Ablation Study

**Coordination Overhead Analysis: Multi-Agent vs. Single-Agent Baseline (GPT-OSS-120B)**

| Scale k | Level I RCC | Level II RCC | Level III RCC |
| :--- | :--- | :--- | :--- |
| k=2 | 15.2% | 31.1% | 48.8% |
| k=5 | 30.3% | 32.9% | 70.0% |
| k=10 | 33.1% | 70.0% | 85.0% |
| k=50 | 45.9% | — | — |

### Key Findings

- **Communication-Reasoning Gap**: Agents can spontaneously form task-appropriate communication topologies (e.g., Agent 0 becomes a star aggregator in Level I tasks), but fail to integrate information correctly once fully acquired—for $N \geq 50$, Level III SR drops to 0% while PCS remains at 8-16%.
- Coordination overhead interacts **multiplicatively** with scale: Level I tasks still yield 40%+ SR at 100 agents, while Level III completely collapses at 50 agents.
- Communication protocol preferences vary by model: DeepSeek prefers broadcasting (BP 40% vs SFS 32%), while GPT prefers point-to-point (P2P 20% vs BP 14%).
- The performance gap between Level I and Level III is only about 15 percentage points for single agents, but expands to 50+ points for multi-agents—proving failures stem from coordination rather than task difficulty.

## Highlights & Insights

- Using communication complexity theory to anchor task difficulty is an elegant design—connecting distributed computing theory with LLM evaluation makes experimental findings interpretable.
- The gap between PCS and SR precisely identifies "reasoning integration" as the bottleneck—agents do not lack information; they lack the ability to integrate it.
- The discovery of emergent communication topologies is surprising: LLMs can spontaneously discover optimal communication patterns like stars and chains, indicating they understand task structures but cannot utilize the collected information effectively.

## Limitations & Future Work

- Only evaluates homogeneous agents (all agents use the same model); heterogeneous agent combinations may behave differently.
- Tasks are algorithmic, lacking natural language reasoning or knowledge-intensive distributed collaboration scenarios.
- Limits on communication rounds and tokens may artificially constrain agent coordination capabilities.
- Future work could explore hierarchical coordination strategies (e.g., electing a leader) or specialized distributed reasoning training.

## Related Work & Insights

- **vs CAMEL/MetaGPT**: These use role specialization and fixed workflows; SILO-BENCH is role-agnostic, isolating the pure contribution of the communication architecture.
- **vs LongBench/∞Bench**: These evaluate single-agent long-context capabilities; SILO-BENCH evaluates whether multi-agent collaboration can substitute for long context.
- **vs Debate-based systems**: Debate systems focus on opinion convergence; SILO-BENCH focuses on the computational correctness of distributed information.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic role-agnostic distributed coordination benchmark with elegant theoretical anchoring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1,620 experiments across 54 configurations covering multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Logically clear, with tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Discovery of the "communication-reasoning gap" has profound implications for multi-agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems](../../AAAI2026/multi_agent/parallelism_meets_adaptiveness_scalable_documents_understanding_in_multi-agent_l.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] Explicit Trait Inference for Multi-Agent Coordination](explicit_trait_inference_for_multi-agent_coordination.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
