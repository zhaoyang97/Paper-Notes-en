---
title: >-
  [Paper Note] Multi-Agent Collaboration via Cross-Team Orchestration
description: >-
  [ACL 2025][Multi-Agent][Multi-Agent Collaboration] This paper proposes Cross-Team Orchestration (Croto), a scalable multi-team collaboration framework that organizes multiple independent agent teams for cross-team interaction, utilizing Hierarchy Partitioning and Greedy Aggregation mechanisms to fuse diverse solutions from various teams into superior results.
tags:
  - "ACL 2025"
  - "Multi-Agent"
  - "Multi-Agent Collaboration"
  - "Cross-Team Orchestration"
  - "Solution Aggregation"
  - "Software Development"
  - "Story Generation"
date: 2026-05-08
content_hash: b206ea4907e09c91
---

# Multi-Agent Collaboration via Cross-Team Orchestration

**Conference**: ACL 2025  
**arXiv**: [2406.08979](https://arxiv.org/abs/2406.08979)  
**Code**: [GitHub](https://github.com/OpenBMB/ChatDev/tree/macnet)  
**Area**: Others  
**Keywords**: Multi-Agent Collaboration, Cross-Team Orchestration, Solution Aggregation, Software Development, Story Generation

## TL;DR

This paper proposes Cross-Team Orchestration (Croto), a scalable multi-team collaboration framework that organizes multiple independent agent teams for cross-team interaction, utilizing Hierarchy Partitioning and Greedy Aggregation mechanisms to fuse diverse solutions from various teams into superior results.

## Background & Motivation

While LLM-based multi-agent collaboration has made progress in fields such as software development and story generation, existing approaches suffer from fundamental limitations:

**Single Team, Single Path**: Chain-like teams such as ChatDev execute sequentially through predefined phases, where each phase generates only one result, forming a unique decision path. If a specific configuration is prone to making the same error on a certain class of problems, self-correction becomes difficult.

**Simple Parallelism is Insufficient**: Directly running $n$ teams and integrating their final results fails to leverage the collaborative potential during intermediate stages. This is akin to exploring $n$ paths in parallel without taking advantage of their intersection points.

**Limitations of Graph-Structured Methods**: Graph-structured paradigms like GPTSwarm require task-specific customization for each node and edge, making them complex to use and difficult to generalize.

Core Problem: **How can multiple agent teams acquire and leverage insights from other teams while maintaining their independence to yield superior solutions?**

## Method

### Overall Architecture

Croto organizes multiple chain-like teams (Chain-as-a-Team) into a collaborative network:
1. Multiple teams process the same task in parallel, independently proposing solutions.
2. At predefined key phases (e.g., design, coding), each team pauses its workflow.
3. Solutions from all teams are extracted for cross-team interaction.
4. These solutions are consolidated into a superior plan through hierarchy partitioning and greedy aggregation.
5. The aggregated results are dispatched back to all teams, replacing their original solutions to guide subsequent stages.

### Key Designs

1. **Cross-Team Interaction Network**:

    - Function: Establishes communication channels among teams during key phases.
    - Mechanism: $\mathcal{N} = \{\mathcal{V}, \mathcal{E}\}$, where node $\mathcal{V}$ represents stages across all teams, and edge $\mathcal{E}$ connects key phases of the same name from different teams.
    - Design Motivation: Interacting only at critical decision points preserves team independence while enabling collaboration.

2. **Greedy Aggregation**:

    - Function: Synthesizes solutions $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$ from multiple teams into a superior solution $s^* = \alpha(\theta(\mathcal{S}))$.
    - Mechanism: Rather than simply voting for the best solution, it lets an aggregation agent extract the advantages and disadvantages of each option, greedily integrating the strengths while eliminating the weaknesses.
    - Pruning Mechanism $\theta$: Before aggregation, low-quality plans are filtered out (evaluated by a Quality metric) to reduce aggregation overhead.
    - Design Motivation: Direct aggregation of all solutions can introduce noise; pruning before aggregation yields better results.

3. **Hierarchy Partitioning**:

    - Function: Groups solutions and aggregates them hierarchically to avoid processing too many proposals at once.
    - Mechanism: Solutions are evenly partitioned (each group containing $u$ solutions) and aggregated within each group. The aggregated results are then grouped and aggregated again iteratively until a single final solution is obtained.
    - Equation: $s^* = \alpha_x(\tau_x(\alpha_{x-1}(\ldots\alpha_1(\tau_1(\mathcal{S}^0)))))$
    - Design Motivation: Solves the long-context issue—processing 8 solutions simultaneously may exceed the effective operational capacity of LLMs.

4. **Team Diversity Design**:

    - Different teams utilize varied temperature parameters and chain lengths.
    - Chain length diversity can be configured manually or evolved autonomously.
    - Ensures that each team explores a distinct decision space.

### Loss & Training

Croto is a training-free, inference-time framework. Key configurations include:
- 8 default teams, with temperature = 0.2
- A maximum of 5 agent communication rounds per phase
- GPT-3.5-Turbo is used as the base model
- Code generation and code completion phases are designated as key phases in software development tasks.

## Key Experimental Results

### Main Results — Software Generation (Table)

| Method | Paradigm | Completeness | Executability | Consistency | Quality |
|------|------|--------|----------|--------|------|
| GPT-Engineer | Single Agent | 0.502 | 0.358 | 0.768 | 0.543 |
| MetaGPT | Single Team | 0.483 | 0.415 | 0.739 | 0.545 |
| ChatDev | Single Team | 0.744 | 0.813 | 0.781 | 0.779 |
| AgentVerse | Single Team | 0.650 | 0.850 | 0.776 | 0.759 |
| GPTSwarm | Graph-Structured | 0.800 | 0.550 | 0.779 | 0.710 |
| **Croto** | **Cross-Team** | **0.795** | **0.928** | **0.796** | **0.840** |

### Ablation Study — Pruning Mechanism (Table)

| Configuration | Completeness | Executability | Consistency | Quality |
|------|--------|----------|--------|------|
| 8-team Croto (Without Pruning) | 0.706 | 0.828 | 0.792 | 0.775 |
| 8-team Croto (**+ Pruning**) | **0.795** | **0.928** | **0.796** | **0.840** |
| Δ | +0.089 | +0.100 | +0.004 | +0.065 |

### Key Findings

1. **Significant Boost in Executability**: Croto achieves an executability of 0.928, which is 7.8 percentage points higher than the strongest baseline, AgentVerse (0.850). This demonstrates that cross-team collaboration can effectively identify and rectify code defects.
2. **Interesting Trade-offs in Team Scaling**: Without pruning, a configuration with 4 teams yields the best performance (quality score of 0.789); performance degrades when scaling beyond 4 teams. This is attributed to the difficulty agents face when synthesizing features from an excessive number of solutions.
3. **Pruning is Critical for Scaling**: 8-team + pruning (Quality: 0.840) outperforms 4-team without pruning (0.789), validating that the pruning mechanism effectively mitigates noise introduced by excessive proposals.
4. **Executability vs. Completeness Trade-off**: As the number of teams increases, executability climbs whilst completeness drops, revealing an intrinsic trade-off.
5. **Generalizability to Story Generation**: Croto also exhibits significant improvements on the ROCStories generation task, proving that the framework's effectiveness extends beyond programming tasks.

## Highlights & Insights

1. **Balancing Independence and Collaboration**: The elegance of Croto lies in its design—teams work independently during their respective intra-team phases to maintain diversity, and interact only at designated key phases to exchange insights. This design outperforms both persistent collaboration and absolute isolation.
2. **Hierarchical Aggregation under Scale**: Decomposing "many-to-one" aggregation into a multi-tiered "many-to-few" process serves as a practical strategy to address the context window limitations of LLMs.
3. **Analogy to Evolutionary Algorithms**: The pipeline of 'multi-team proposal -> scoring/filtering -> aggregation/optimization -> dispatch/continuation' closely resembles population-based evolution in genetic algorithms.

## Limitations & Future Work

1. **High Token Costs**: Running 8 teams across multiple phases with aggregation processes generates substantial API call expenses for GPT-3.5-Turbo. The paper does not report specific token consumption comparisons.
2. **Heuristic Key Phase Definition**: Identifying which phases should prompt cross-team interactions depends on predefined settings; ideally, this process should be automatically determined.
3. **Bottleneck of the Aggregation Agent**: The quality of aggregation strictly relies on the synthesis capability of the LLM. If the LLM struggles to extract advantages from multiple proposals, the overall efficacy of the framework becomes limited.
4. **Small-Scale Evaluation**: The framework is evaluated only on 15 software development tasks and 10 story generation tasks, which represents a relatively small sample size.
5. **Limited Base Model Testing**: The experiments solely utilize GPT-3.5-Turbo, without verification on stronger or weaker models to observe consistent generalization.

## Related Work & Insights

- ChatDev serves as the single-team baseline for Croto, with Croto developed directly on top of the ChatDev repository.
- MACNET discovered that solution quality grows logistically with the number of agents; Croto bypasses this performance ceiling through cross-team collaboration.
- Analogous to ensemble learning concepts (i.e., aggregating multiple weak proposals into a single strong proposal), Croto's aggregation operates at the semantic level rather than relying on naive voting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of cross-team orchestration is innovative; both hierarchical partitioning and greedy aggregation designs are practical and insightful.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers both software development and story generation tasks, but the sample size is small (15 + 10) and only leverages a single LLM baseline.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly clear formal definitions (Definition 1-3) along with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Provides a practical collaborative paradigm for multi-agent systems, successfully integrated into the open-source ChatDev project.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](../../NeurIPS2025/multi_agent/multi-agent_collaboration_via_evolving_orchestration.md)
- [\[ACL 2025\] Preventing Rogue Agents Improves Multi-Agent Collaboration](preventing_rogue_agents_improves_multi-agent_collaboration.md)
- [\[ACL 2025\] Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](beyond_frameworks_multi_agent_collaboration.md)
- [\[ICML 2025\] Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](../../ICML2025/multi_agent/cross-environment_cooperation_enables_zero-shot_multi-agent_coordination.md)
- [\[CVPR 2025\] NADER: Neural Architecture Design via Multi-Agent Collaboration](../../CVPR2025/multi_agent/nader_neural_architecture_design_via_multi-agent_collaboration.md)

</div>

<!-- RELATED:END -->
