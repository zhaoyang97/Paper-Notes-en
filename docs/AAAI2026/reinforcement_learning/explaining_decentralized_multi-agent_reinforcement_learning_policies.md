---
title: >-
  [Paper Note] Explaining Decentralized Multi-Agent Reinforcement Learning Policies
description: >-
  [AAAI 2026][Reinforcement Learning][Explainable AI] This paper proposes the first explainability method for decentralized multi-agent reinforcement learning (MARL) policies, comprising Hasse diagram-based policy summarization and query-based natural language explanations (When / Why Not / What). The approach is demonstrated across four MARL domains, showing both generality and computational efficiency. A user study confirms that it significantly improves human understanding of policies and question-answering performance.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Explainable AI
  - Multi-Agent Reinforcement Learning
  - Decentralized Policies
  - Hasse Diagram
  - Policy Summarization
date: 2026-05-08
content_hash: a543fd08d39731d3
---

# Explaining Decentralized Multi-Agent Reinforcement Learning Policies

**Conference**: AAAI 2026
**arXiv**: [2511.10409](https://arxiv.org/abs/2511.10409)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Explainable AI, Multi-Agent Reinforcement Learning, Decentralized Policies, Hasse Diagram, Policy Summarization

## TL;DR

This paper proposes the first explainability method for decentralized multi-agent reinforcement learning (MARL) policies, comprising Hasse diagram-based policy summarization and query-based natural language explanations (When / Why Not / What). The approach is demonstrated across four MARL domains, showing both generality and computational efficiency. A user study confirms that it significantly improves human understanding of policies and question-answering performance.

## Background & Motivation

**Gap in decentralized MARL explainability**: Multi-agent reinforcement learning has achieved substantial progress and has been applied to domains such as autonomous driving and multi-robot warehousing. However, existing explainability methods almost exclusively focus on centralized MARL—settings in which a joint policy can observe the global state. The unique challenges of decentralized settings, where each agent operates on local observations with an independent policy, have been largely overlooked.

**Core challenges of decentralized settings**:

**Uncertainty and non-determinism**: Decentralized execution introduces inherent uncertainty in task completion ordering; different agents may complete tasks asynchronously.

**Unobservable inter-agent coordination**: Because each agent observes only its own local state, coordinated behaviors across agents cannot be directly inferred from a single trajectory.

**Limitations of prior work**: Single-agent abstract policy graphs (Topin 2019) and centralized MARL macro-action abstractions (Boggess 2022) both assume a single input policy and cannot handle interactions among multiple independent policies.

**Motivating scenario**: In a search-and-rescue mission, multiple robots executing decentralized policies perform tasks collaboratively. Operators need to understand "which robots completed which tasks," "why a certain task was not completed under given conditions," and "what comes next after completing a task"—precisely the questions this paper addresses.

## Method

### Overall Architecture

The method consists of two main modules:

1. **Policy Summarization**: Constructs a Hasse diagram from execution trajectories of decentralized policies, compactly representing task partial-order relations and agent coordination patterns.
2. **Query-Based Explanations**: Supports three query types—"When" (when a task is completed), "Why Not" (why a task was not completed), and "What" (what happens after completing a task).

### Key Designs

#### 1. **Hasse Diagram Summarization (Algorithm 1: HDS)**: Constructing a partial-order graph from trajectories

**Core Idea**: The Hasse diagram $\mathcal{D} = (\mathcal{V}, \mathcal{E})$ is a directed acyclic graph in which each node represents a set of tasks completed simultaneously along with the agents executing them, and edges encode temporal ordering constraints.

**Construction procedure**:
- For each agent $i$, extract the task sequence $\mathsf{trace}(\omega^i)$ from its trajectory $\omega^i$.
- For each task $\tau$ in the sequence: if $\tau$ already exists in some node $v$ (indicating collaborative completion by multiple agents), add agent $i$ to that node; otherwise, create a new node.
- Add directed edges between nodes corresponding to consecutive tasks.
- Apply transitive reduction to remove redundant edges.

**Correctness and completeness guarantees (Theorem 1)**: For all paths $\rho$ and agents $i$, the projection of a path onto agent $i$, denoted $\rho^i$, is either empty or preserves the original task order (correctness); for each agent there exists a path that fully covers its task sequence (completeness).

Time complexity: $\mathcal{O}(N \cdot |T|^2 + |T|^4)$, where $N$ is the number of agents and $|T|$ is the number of tasks.

**Design Motivation**: The Hasse diagram simultaneously encodes three categories of critical information—task partial order (edges), agent collaboration (multi-agent annotations within nodes), and uncertainty (branching paths)—whereas existing methods require constructing separate graphs per agent and comparing them manually.

#### 2. **"When" Query Explanation (Algorithm 2)**: Identifying necessary and sufficient conditions for task completion

**Core Idea**: Given a query "when does agent group $\mathcal{G}_q$ complete task $\tau_q$," the method extracts relevant features from the Hasse diagram and distinguishes deterministic from uncertain conditions.

**Key innovation—uncertainty dictionary $U$**:
- In the Hasse diagram, if no reachability path exists between a node $v$ and the target node $v_\tau$ (i.e., their temporal order cannot be determined), the features associated with $v$ are marked as "uncertain."
- A partial comparability graph is used to identify these uncertain relations.

**Boolean formula generation**: Nodes are encoded as Boolean feature vectors; the Quine–McCluskey algorithm is applied to extract a minimal Boolean formula distinguishing target from non-target nodes, which is then translated into natural language via linguistic templates. Deterministic features are expressed with "must" and uncertain features with "may."

Example explanation: "For agents 2 and 4 to complete task C, agent 2 must complete task C, agent 4 must complete task C, and task A must be completed. Additionally, task B **may** need to be completed."

#### 3. **"Why Not" and "What" Query Explanations**:

- **"Why Not"** (Algorithm in Appendix B): Symmetric to "When"—the user-specified conditions are encoded as the target, completed cases as non-targets, and missing conditions are extracted.
- **"What"** (Algorithm 3): Analyzes the successors of the target node in the Hasse diagram, distinguishing deterministic successors (tasks in direct child nodes) from uncertain successors (tasks in partially incomparable nodes).

### Loss & Training

This is a post-hoc explanation method and involves no model training. Two MARL algorithms are used to train policies in the experiments:
- **SEAC** (Shared Experience Actor-Critic): Centralized Training with Decentralized Execution (CTDE).
- **IA2C** (Independent Advantage Actor-Critic): Decentralized Training with Decentralized Execution (DTDE).

All models are trained to convergence or for a maximum of 400 million steps.

## Key Experimental Results

### Main Results

Summarization compactness is evaluated across four benchmark domains (Search and Rescue, Level-Based Foraging, Multi-Robot Warehouse, Pressure Plate) against a baseline adapted from single-agent methods:

| Domain | $(N, |T|)$ | HDS Nodes | HDS Edges | Baseline Nodes | Baseline Edges |
|--------|-----------|-----------|-----------|----------------|----------------|
| SR | (9, 7) | 8 | 7.88 | 534 | 525 |
| LBF | (9, 9) | 10 | 10.83 | 723 | 714 |
| RW | (4, 19) | 20 | 19 | 1,274 | 1,270 |
| PP | (7, 6) | 7 | 6 | 265 | 258 |

Query explanation size comparison ("When" query, number of features):

| Domain | $(N, |T|)$ | HDS Certain Features | HDS Uncertain Features | Baseline Certain Features |
|--------|-----------|----------------------|------------------------|---------------------------|
| SR | (9, 7) | 9 | 2 | 54 |
| LBF | (9, 9) | 13 | 11 | 104 |
| RW | (4, 19) | 0 | 153 | 267 |
| PP | (7, 6) | 8 | 3 | 20 |

### Ablation Study

**Summarization user study** (20 participants, within-subjects design):

| Metric | HDS | Baseline | Statistical Test |
|--------|-----|----------|-----------------|
| Correct answers (max 6) | 4.25 (SD=0.83) | 3.1 (SD=1.04) | t(19)=4.2, p≤0.01, d=0.96 |
| Completeness rating (5-point scale) | Significantly higher | — | W=16.0, p≤0.04 |

**Explanation user study** (21 participants)—correct answer rates are significantly higher across all three query types:

| Query Type | HDE (Ours) | Baseline | Effect Size $d$ |
|------------|-----------|----------|-----------------|
| When | Significantly higher | — | d=2.16 |
| Why Not | Significantly higher | — | d=2.96 |
| What | Significantly higher | — | d=2.69 |

Subjective ratings are significantly superior to the baseline across all 7 dimensions (comprehension, satisfaction, detail, completeness, actionability, reliability, and trust).

### Key Findings

1. **HDS summaries are 1–2 orders of magnitude smaller than the baseline**: the baseline requires displaying separate policy graphs for each agent, resulting in hundreds to thousands of nodes and edges; HDS produces a single compact Hasse diagram per episode.
2. **Expressing uncertainty is critical**: in the highly asynchronous RW(4,19) domain, all features are uncertain—a situation the baseline is entirely unable to represent.
3. **Effect sizes in the user study are very large** ($d = 2.16$–$2.96$), indicating that uncertainty expression provides fundamentally better human understanding in decentralized settings.
4. **The method is agnostic to training paradigm**: it is effective under both CTDE and DTDE algorithms.
5. All summaries and explanations are generated in under one second.

## Highlights & Insights

1. **First work to address the gap in decentralized MARL explainability**: Hasse diagrams—a classical tool from partial-order theory—are creatively introduced into MARL policy summarization, elegantly capturing the essential characteristics of decentralized execution.
2. **Elegant design of the uncertainty dictionary**: the partial comparability graph distinguishes "definitely precedes/follows" from "temporally unordered," naturally expressed via "must/may."
3. **Combines theoretical guarantees with human evaluation**: Theorem 1 establishes correctness and completeness, while the user study validates practical utility.
4. **Scales to 19 tasks and 9 agents** with manageable computational complexity.

## Limitations & Future Work

1. **Restricted to gridworld domains**: all four benchmarks are gridworlds; applicability to continuous state/action spaces remains unverified.
2. **Task definition requires domain knowledge**: task recognition relies on manually engineered features derived from reward signals and state transitions.
3. **Quine–McCluskey complexity**: with large feature sets (e.g., 253 uncertain features in RW), worst-case complexity is $\mathcal{O}(3^{|\mathcal{F}_q|} / \ln |\mathcal{F}_q|)$.
4. **No LLM-augmented explanations**: natural language explanations are template-based and could be further refined by large language models.
5. **Real-time interactive explanation not considered**: the current approach is purely post-hoc.

## Related Work & Insights

- Single-agent abstract policy graphs (Topin 2019; McCalmon 2022) → extended in this work to a multi-agent partial-order structure.
- Query-based explanations (Hayes 2017) → augmented here with an uncertainty dimension.
- Centralized MARL explainability (Boggess 2022; Milani 2022) → this work removes the centralization assumption.
- The approach can inspire "explanation as human–machine interface" design: the explanation method can be directly embedded into decision support systems for human–multi-robot collaboration.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First work on decentralized MARL explainability; the combination of Hasse diagrams and the uncertainty dictionary is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four domains, two algorithms, and a user study, though validation in continuous environments and at larger scale is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, complete algorithmic descriptions, and tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐ — Fills an important gap, though further validation for real-world deployment is needed.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)
- [\[AAAI 2026\] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)
- [\[AAAI 2026\] Beyond Monotonicity: Revisiting Factorization Principles in Multi-Agent Q-Learning](beyond_monotonicity_revisiting_factorization_principles_in_multi-agent_q-learnin.md)
- [\[AAAI 2026\] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [\[AAAI 2026\] BAMAS: Structuring Budget-Aware Multi-Agent Systems](bamas_structuring_budget-aware_multi-agent_systems.md)

<!-- RELATED:END -->
