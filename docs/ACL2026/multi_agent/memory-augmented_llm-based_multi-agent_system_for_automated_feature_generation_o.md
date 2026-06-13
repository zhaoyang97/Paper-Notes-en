---
title: >-
  [Paper Note] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data
description: >-
  [ACL 2026][Multi-Agent][Automated Feature Engineering] The authors propose MALMAS, a memory-augmented LLM multi-agent system for automated feature generation on tabular data. It utilizes six specialized agents to explore…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Automated Feature Engineering"
  - "Multi-Agent Systems"
  - "Memory Augmentation"
  - "Tabular Data"
  - "AutoML"
date: 2026-05-08
content_hash: a6e806cf174772db
---

# Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data

**Conference**: ACL 2026  
**arXiv**: [2604.20261](https://arxiv.org/abs/2604.20261)  
**Code**: [GitHub](https://github.com/fxdong24/MALMAS)  
**Area**: LLM/NLP  
**Keywords**: Automated Feature Engineering, Multi-Agent Systems, Memory Augmentation, Tabular Data, AutoML

## TL;DR
The authors propose MALMAS, a memory-augmented LLM multi-agent system for automated feature generation on tabular data. It utilizes six specialized agents to explore different feature space dimensions combined with a three-level memory mechanism (Procedural/Feedback/Conceptual) for cross-iteration optimization, outperforming existing baselines on 16 classification and 7 regression datasets.

## Background & Motivation

**Background**: Automated feature generation is a critical component of AutoML, aiming to automatically construct high-quality features from raw tabular data. Traditional methods (e.g., DFS, OpenFE) rely on predefined operator libraries for combinatorial search, while recent LLM-based methods (e.g., CAAFE) introduce semantic information to guide feature transformations but still face limitations.

**Limitations of Prior Work**: (1) Traditional methods are constrained by fixed operator sets and cannot utilize task semantics, resulting in a narrow search space; (2) Although LLM methods introduce semantic signals, they rely on single generation strategies and fixed thinking patterns, leaving feature space exploration restricted; (3) Crucially, existing LLM methods lack feedback mechanisms from downstream learning objectives—the generation process is decoupled from model performance, leading to inefficient trial-and-error exploration.

**Key Challenge**: The contradiction between the high dimensionality and diversity of the feature space versus the limited exploration capability of a single agent, and the lack of a "generation $\rightarrow$ evaluation $\rightarrow$ optimization" closed loop.

**Goal**: Design a multi-agent collaborative and memory-driven automated feature generation framework capable of (1) broadly exploring the feature space through role specialization and (2) achieving cross-iteration experience accumulation and strategy adjustment through multi-level memory.

**Key Insight**: Starting from the classification of "golden features" in feature engineering practice, specialized agents are designed along three orthogonal dimensions (transformation complexity, data scope, and data type dependency), coupled with a three-level experience system: procedural memory (what was done), feedback memory (how effective it was), and conceptual memory (why it worked).

**Core Idea**: Decomposing feature generation into parallel exploration by multiple specialized agents, dynamic scheduling via a Router Agent, and iterative optimization driven by three-level memory.

## Method

### Overall Architecture
In each iteration: The Router Agent selects a subset of active agents from the agent pool $\rightarrow$ each active agent builds a prompt based on metadata and memory to generate features through multi-turn interactions with the LLM $\rightarrow$ the validation performance of generated features is evaluated on downstream models $\rightarrow$ the three-level memory is updated $\rightarrow$ the Summary Agent aggregates global conceptual memory $\rightarrow$ TopN features are selected and added to the dataset $\rightarrow$ move to the next iteration.

### Key Designs

1. **Parallel Architecture of Six Specialized Agents + Router Agent**:
    - **Function**: Achieve broad exploration of the feature space through role specialization.
    - **Mechanism**: Six agents are responsible for Unary, Cross-Compositional, Temporal, Aggregation-Construct, Local-Transform, and Local-Pattern transformations. The Router Agent dynamically selects which agents to activate in each round based on task metadata and accumulated memory.
    - **Design Motivation**: A single agent tends to produce feature homogenization; multiple agents explore complementary regions across transformation complexity, data scope, and data types.

2. **Three-Level Memory Mechanism (Procedural + Feedback + Conceptual)**:
    - **Function**: Transform per-round evaluation feedback into persistent learning signals.
    - **Mechanism**: **Procedural Memory** (ProcMem) records the full trace of transformations (base columns, transformation type, feature name, description, round) to avoid redundant exploration; **Feedback Memory** (FeedMem) associates each feature with downstream validation metrics for explicit credit assignment; **Conceptual Memory** (ConMem) consists of reusable heuristic rules distilled by the LLM from procedural and feedback memories.
    - **Design Motivation**: LLM generation without memory is stateless. The three-level memory abstracts from "what was done" $\rightarrow$ "how it performed" $\rightarrow$ "why it worked," enabling short-term error avoidance, mid-term guidance, and long-term strategy adaptation.

3. **Global Conceptual Memory and Cross-Agent Knowledge Transfer**:
    - **Function**: Facilitate coordination and knowledge sharing among agents.
    - **Mechanism**: After each round, the Summary Agent aggregates conceptual and feedback memories from all active agents to generate a GlobalMem. Subsequent Router decisions and agent prompt constructions reference this global memory.
    - **Design Motivation**: Local memory serves only individual agents; global memory propagates effective patterns to other agents to reduce overlapping exploration.

### Loss & Training
The goal is to maximize performance metrics of downstream models on the validation set (AUC for classification, NRMSE for regression). XGBoost is used as the downstream model, and the best features are retained through TopN-Features filtering in each round.

## Key Experimental Results

### Main Results (Classification AUC, Mean Rank across 16 Datasets)

| Method | Type | Mean Rank |
|------|------|-----------|
| DFS | Traditional | 3.69 |
| OpenFE | Traditional | 3.12 |
| CAAFE | LLM | 3.57 |
| OCTree | LLM | 4.81 |
| LLMFE | LLM | 3.75 |
| **Ours (MALMAS)** | **Multi-Agent + Memory** | **1.12** |

### Ablation Study (Contribution of Key Components)

| Configuration | Description |
|------|------|
| Single Agent (No Router) | Feature diversity drops; high homogenization |
| No Memory | Independent generation each round; high redundant exploration |
| No Global Memory | No knowledge transfer between agents; increased redundant features |
| No Feedback Memory | Unable to learn which transformations are effective from history |
| **Ours (MALMAS Full)** | Optimal performance, Mean Rank 1.12 |

### Key Findings
- **MALMAS achieves a mean rank of 1.12 across 16 classification datasets**, significantly outperforming the runner-up OpenFE (3.12).
- **Advantages are more pronounced on difficult datasets**: e.g., Titanic (0.872 vs. next best 0.849), Credit_G (0.775 vs. next best 0.758).
- **The memory mechanism is critical**: Conceptual memory abstracts "why a transformation is effective" into reusable rules to guide subsequent exploration.
- **Dynamic scheduling by the Router Agent** avoids the computational waste of activating all agents for every dataset.

## Highlights & Insights
- The **hierarchical design of three-level memory** is insightful: moving from operation traces to credit assignment to strategy abstraction corresponds to procedural memory $\rightarrow$ working memory $\rightarrow$ metacognition in cognitive psychology. This can be migrated to any multi-agent system requiring iterative optimization.
- **Dynamic scheduling via the Router Agent** addresses the computational inefficiency of "running every agent," achieving task-dependent resource allocation.
- **Designing agent roles based on "golden feature" classifications** is a good practice—encoding domain knowledge into the division of labor among agents.

## Limitations & Future Work
- Agent roles are manually designed; can optimal divisions be discovered automatically?
- The memory management lacks a forgetting mechanism, which may lead to context bloat over long iterations.
- The downstream model is fixed to XGBoost; effectiveness on deep learning models is unverified.
- End-to-end comparisons with full-pipeline AutoML methods (e.g., Auto-sklearn) are missing.
- Adversarial or debate mechanisms between agents could be explored to improve feature quality.

## Related Work & Insights
- **vs. CAAFE**: CAAFE uses a single LLM to generate features, limited by a single strategy. MALMAS significantly expands the exploration space via multi-agent specialization and memory feedback.
- **vs. OpenFE**: OpenFE uses operator search via tree models, which is efficient but limited to predefined operators. MALMAS leverages LLM semantic understanding to generate more diverse transformations.
- **vs. Generative Agents**: While the latter uses multi-agent systems and memory for social simulation, MALMAS introduces this paradigm to feature engineering, representing a new application direction for this approach.

## Rating
- Novelty: ⭐⭐⭐⭐ Multi-agent + three-level memory for feature generation is a new combination, though components are individually established.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage with 23 datasets, but ablation details could be more granular.
- Writing Quality: ⭐⭐⭐⭐ Framework diagrams are clear, though notation is slightly redundant.
- Value: ⭐⭐⭐⭐ Provides a practical contribution to the AutoML community; the three-level memory design has broad transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](efficient_multi-agent_system_training_with_data_influence-oriented_tree_search.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation](a_multi-agent_framework_for_feature-constrained_difficulty_control_in_reading_co.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)
- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](../../AAAI2026/multi_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)

</div>

<!-- RELATED:END -->
