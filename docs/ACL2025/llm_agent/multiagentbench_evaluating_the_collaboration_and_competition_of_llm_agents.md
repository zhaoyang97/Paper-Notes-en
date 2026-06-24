---
title: >-
  [Paper Note] MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents
description: >-
  [ACL 2025][LLM Agent][multi-agent] This paper proposes the MultiAgentBench benchmark and the MARBLE framework to systematically evaluate the performance of LLM multi-agent systems in collaborative and competitive scenarios. Covering 6 interactive environments (Research, Minecraft, Database, Coding, Bargaining, and Werewolf), the study introduces milestone-based KPI metrics and coordination scores. The evaluation reveals that GPT-4o-mini achieves the highest overall task score…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "multi-agent"
  - "benchmark"
  - "collaboration"
  - "competition"
  - "coordination protocol"
  - "emergent behavior"
date: 2026-05-08
content_hash: 8393ac7a03391c0a
---

# MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents

## Basic Information

**Conference**: ACL 2025  
**Code**: [MultiagentBench/MARBLE](https://github.com/MultiagentBench/MARBLE)  
**Institution**: University of Illinois Urbana-Champaign  
**Area**: LLM Agent / Multi-Agent Systems  
**Keywords**: multi-agent, benchmark, collaboration, competition, coordination protocol, emergent behavior  

## TL;DR

This paper proposes the MultiAgentBench benchmark and the MARBLE framework to systematically evaluate the performance of LLM multi-agent systems in collaborative and competitive scenarios. Covering 6 interactive environments (Research, Minecraft, Database, Coding, Bargaining, and Werewolf), the study introduces milestone-based KPI metrics and coordination scores. The evaluation reveals that GPT-4o-mini achieves the highest overall task score, graph-structured coordination protocols perform best in research scenarios, and cognitive planning improves milestone completion rates by 3%.

## Background & Motivation

- **Limitations of Single-Agent Evaluation**: Existing benchmarks such as AgentBench, GAIA, and ToolBench primarily evaluate the reasoning and generation capabilities of individual LLM agents, neglecting coordination and competitive dynamics in multi-agent interactions.
- **Rise of Multi-Agent Systems**: LLM multi-agent systems have demonstrated great potential in software development (e.g., MetaGPT, ChatDev), scientific research, and gaming, but lack systematic evaluation standards.
- **Insufficient Evaluation Dimensions**: Evaluation must cover not only task completion but also coordination quality, communication efficiency, and planning capabilities.
- **Goal**: To construct a comprehensive multi-agent evaluation framework spanning both collaborative and competitive scenarios.

## Method

### Overall Architecture: MARBLE (Multi-agent cooRdination Backbone with LLM Engine)

MARBLE consists of four core modules:

#### 1. Agent Graph Module

Models agent relationships as a graph $G = (\mathcal{A}, E)$:
- $\mathcal{A} = \{a_1, a_2, \dots, a_n\}$: The set of agents.
- Each edge $(a_i, r, a_j)$ represents a relationship type: collaborates, supervises, or negotiates.
- Communication and coordination occur only between agents with explicit relationships.

#### 2. Cognitive Module

- Maintains each agent's internal state: persona, inter-agent relationships, and reasoning strategies.
- Integrates Theory of Mind (ToM) and social intelligence.
- Supports reasoning strategies such as CoT and ReACT.
- Simulates the human process of continuously updating mental models based on social cues.

#### 3. Coordination Engine

Supports four coordination protocols:

| Protocol | Type | Characteristics |
|------|------|------|
| **Star** | Centralized | Single planner assigns tasks, strong supervision but limited scalability |
| **Tree** | Centralized | Hierarchical structure, top-level planner delegates to downstream planners |
| **Graph-Mesh** | Decentralized | Direct communication between agents, concurrent planning, and distributed decision-making |
| **Chain** | Decentralized | Sequential propagation of decisions, suitable for tasks with dependencies |

#### 4. Planner Design

Four planning strategies:

1. **Vanilla Prompting**: Direct zero-shot prompt-based plan generation.
2. **Chain-of-Thought (CoT)**: Step-by-step reasoning using input tasks, agent profiles, and interaction history.
3. **Group Discussion**: Multi-agent collaborative deliberation sharing insights and constraints.
4. **Cognitive Self-Evolving Planning**: Generates expected outcomes, stores them in memory, compares them with actual performance, and iteratively refines them (similar to Reflexion).

### Benchmark Task Design

#### Shared-Goal Scenarios (Collaboration)

| Scenario | Description | Scale |
|------|------|------|
| **Research** | Multi-agent collaborative writing of research proposals | 100 test cases |
| **Minecraft** | Collaborative structure construction | 100 test cases |
| **Database** | 5 agents diagnosing different root causes | 100 test cases |
| **Coding** | Collective programming and module development | 100 test cases |

#### Adversarial-Goal Scenarios (Competition)

| Scenario | Description |
|------|------|
| **Werewolf** | Two opposing groups involving deceptive strategies |
| **Bargaining** | Resource negotiation to maximize individual payoff |

### Evaluation Metrics

#### Task Completion Metrics

- **KPI**: Milestone-based Key Performance Indicators, $\text{KPI}_{\text{overall}} = \frac{1}{NM}\sum_{j=1}^N n_j$
- **Task Score (TS)**: Final output quality score (evaluated via LLMs or heuristics-based rules).

#### Coordination Metrics

- **Communication Score ($C_{\text{score}}$)**: Evaluates communication quality using an LLM (5-point scale).
- **Planning Score ($P_{\text{score}}$)**: Evaluates task organization, role maintenance, and strategy adjustment (5-point scale).
- **Coordination Score (CS)**: The average of the above two scores.

## Experiments

### Experimental Setup

- **Models**: Meta-Llama-3.1-8B, Meta-Llama-3.1-70B, Meta-Llama-3.3-70B, GPT-3.5-turbo, GPT-4o-mini.
- **Parameters**: max_token_num=1024, temperature=0.7, top_p=1.0.
- **Iterations**: 5 rounds for Research, 20 rounds for Minecraft, maximum of 5 communication rounds.
- **Default Protocol**: Graph-Mesh.

### Main Results I: Model Performance Comparison

| Model | Research TS | Minecraft TS | Database TS | Coding TS | Bargaining TS | Werewolf TS |
|------|-----------|-------------|-----------|---------|-------------|-----------|
| Llama-3.1-8B | 80.87 | 6.12 | 34.00 | 59.90 | 72.81 | 12.64 |
| Llama-3.1-70B | 80.80 | 0.21 | 53.00 | 62.10 | 72.13 | 19.82 |
| Llama-3.3-70B | 80.00 | 9.15 | 28.50 | 56.60 | 73.15 | 36.33 |
| GPT-3.5-turbo | 70.20 | 5.05 | 45.00 | 55.50 | 71.67 | 15.69 |
| **GPT-4o-mini** | **84.13** | **33.60** | 45.00 | **65.10** | **74.47** | 14.06 |

**Key Findings**:
1. **GPT-4o-mini Achieves the Highest Task Score**: It leads in Research, Minecraft, Coding, and Bargaining.
2. **Coordination Score $\neq$ Task Score**: Llama-3.1-70B obtains a high coordination score of 75.00 in Minecraft, but its task score is only 0.21.
3. **Capabilities Vary by Task**: Llama-3.3-70B achieves the highest TS in Werewolf (36.33), but does not perform prominently in other tasks.

### Main Results II: Coordination Protocol Comparison

- **Graph implementation** performs best in the research scenario (highest task score and planning efficiency).
- **Tree implementation** performs the worst, suffering from high token consumption and the lowest task and coordination scores.
- **Star and Graph** exhibit similar task scores.

### Planning Strategy Comparison

- **Cognitive Evolving Planning** achieves the best coordination score, with a task score comparable to CoT.
- **Group Discussion** unexpectedly performs the worst—excessively large planning groups tend to hinder efficiency (similar to large organizational issues in reality).

### Ablation Study

**Number of Iterations**:
- Task and coordination scores increase between 1 and 7 rounds, plummet at 10 rounds, and the task score recovers at 20 rounds while the coordination score remains unchanged.
- Excessive iterations can lead to coordination degradation (due to communication overhead or instruction conflicts).

**Number of Agents**:
- KPI decreases as the number of agents increases (owing to more complex coordination).
- Coordination scores improve significantly from 1 to 3 agents, plateauing thereafter.
- The growth of the task score is more gradual.

### Emergent Behavior Analysis

Three key emergent patterns:
1. **Strategic Information Sharing**: Agents selectively disclose key information (e.g., the Seer concealing inspection results in Werewolf).
2. **Trust Polarization Collaboration**: Role identities drive cooperative division; over-suspicious Villagers may end up attacking their own allies.
3. **Role-Driven Strategy Iteration**: Characters (e.g., Seer, Witch) adaptively adjust their strategies during gameplay.

## Highlights & Insights

1. **First Systematic Multi-Agent Evaluation Framework**: Covers collaboration and competition across 6 scenarios and multiple coordination protocols, filling an important evaluation gap.
2. **Innovative Milestone-Based KPI**: Moves beyond binary task success/failure, tracking intermediate progress and individual contributions.
3. **Counter-Intuitive Findings**: Group Discussion performs the worst, high coordination scores do not equate to high task scores, and more agents are not always beneficial.
4. **Discovery of Emergent Behaviors**: LLM agents exhibit human-like social behavior patterns under information asymmetry and role-based conflicts.
5. **Model Capabilities Remain Critical**: Improvements in coordination cannot compensate for deficiencies in underlying foundational capabilities.

## Limitations & Future Work

1. **Limited Scenario Coverage**: Does not cover highly complex environments such as open worlds or task-oriented dialogues.
2. **Incomplete Model Coverage**: Excludes recent models like DeepSeek.
3. **Insufficiently Deep Ablation**: Does not adequately study memory mechanisms (long-term, short-term, or shared memory) or different workflow methodologies.
4. **Simplistic Competitive Mechanics**: Fails to cover complex dynamics like multi-party negotiation, repeated games, or stochastic factors.
5. **Evaluation Dependency on LLM**: KPI milestone detection and coordination scoring rely heavily on LLM evaluation, which may introduce bias.

## Related Work & Insights

- **Multi-Agent Systems**: MetaGPT (Hong et al., 2024), AgentVerse (Chen et al., 2023), ChatDev (Li et al., 2023)
- **Multi-Agent Collaboration**: Cognitive expansion (Zhuge et al.), collective expansion (Qian et al., 2024)
- **Agents in Games**: GameNGen (Valevski et al., 2024), CUISINEWORLD (Gong et al., 2023), Voyager (Wang et al., 2023)
- **Single-Agent Benchmarks**: AgentBench (Liu et al., 2023), GAIA (Mialon et al., 2023)

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: The first systematic evaluation benchmark for multi-agent systems, featuring a comprehensive design (+1)
- **Experimental Thoroughness**: 6 scenarios $\times$ 5 models $\times$ 4 coordination protocols $\times$ 4 planning strategies (+0.5)
- **Depth of Insight**: Emergent behavior analysis and counter-intuitive findings add to the academic contribution (+0.5)
- **Value**: The open-source framework is highly beneficial for the community (+0.5)
- **Deductions**: Reliability of LLM-based evaluation is questionable, model coverage is not broad enough, and some task designs are overly simplistic (-1)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [\[ACL 2025\] The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)
- [\[ICLR 2026\] Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration](../../ICLR2026/llm_agent/collaborative_gym_a_framework_for_enabling_and_evaluating_human-agent_collaborat.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[ACL 2025\] Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration](dpt_agent_dual_process.md)

</div>

<!-- RELATED:END -->
