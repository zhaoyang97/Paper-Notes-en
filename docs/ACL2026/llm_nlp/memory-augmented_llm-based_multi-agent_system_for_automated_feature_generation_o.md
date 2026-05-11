---
title: >-
  [Paper Note] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data
description: >-
  [ACL 2026][LLM/NLP][Automated Feature Engineering] This paper proposes MALMAS, a memory-augmented LLM-based multi-agent system for automated feature generation on tabular data. It employs six specialized agents to explor…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Automated Feature Engineering"
  - "Multi-Agent System"
  - "Memory Augmentation"
  - "Tabular Data"
  - "AutoML"
date: 2026-05-08
content_hash: 67864cb02774e3ed
---

# Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data

**Conference**: ACL 2026
**arXiv**: [2604.20261](https://arxiv.org/abs/2604.20261)
**Code**: [GitHub](https://github.com/fxdong24/MALMAS)
**Area**: LLM/NLP
**Keywords**: Automated Feature Engineering, Multi-Agent System, Memory Augmentation, Tabular Data, AutoML

## TL;DR
This paper proposes MALMAS, a memory-augmented LLM-based multi-agent system for automated feature generation on tabular data. It employs six specialized agents to explore different dimensions of the feature space in parallel, coordinated by a Router Agent, and leverages a three-tier memory mechanism (procedural/feedback/conceptual) for cross-iteration experience accumulation and strategy refinement. MALMAS outperforms existing baselines on 16 classification and 7 regression datasets.

## Background & Motivation

**Background**: Automated feature generation is a critical component of AutoML, aiming to construct high-quality features from raw tabular data. Traditional methods (e.g., DFS, OpenFE) rely on predefined operator libraries for combinatorial search, while recent LLM-based approaches (e.g., CAAFE) incorporate semantic information to guide feature transformations, albeit with remaining limitations.

**Limitations of Prior Work**: (1) Traditional methods are constrained by fixed operator sets and cannot leverage task semantics, resulting in a narrow search space. (2) LLM-based methods, while introducing semantic signals, depend on a single generation strategy with rigid thinking patterns, leaving feature space exploration still limited. (3) More critically, existing LLM-based methods lack feedback mechanisms from downstream learning objectives—the generation process is decoupled from model performance, yielding only inefficient trial-and-error exploration.

**Key Challenge**: A fundamental tension exists between the high dimensionality and diversity of the feature space and the limited exploratory capacity of a single agent, compounded by the absence of a closed "generate→evaluate→optimize" loop.

**Goal**: To design a multi-agent collaborative, memory-driven automated feature generation framework that (1) broadly explores the feature space through role specialization, and (2) accumulates experience and adjusts strategies across iterations via multi-tier memory.

**Key Insight**: Drawing from a taxonomy of "golden features" in feature engineering practice, the paper designs specialized agents along three orthogonal dimensions (transformation complexity, data scope, and data-type dependency), and introduces a three-tier experience system comprising procedural memory (what was done), feedback memory (how effective it was), and conceptual memory (why it worked).

**Core Idea**: Decompose feature generation into parallel exploration by multiple specialized agents, dynamic scheduling by a Router Agent, and iterative optimization driven by three-tier memory.

## Method

### Overall Architecture
Each iteration proceeds as follows: the Router Agent selects an active subset from the agent pool → each active agent constructs a prompt based on metadata and memory, then interacts with the LLM over multiple turns to generate features → the generated features are evaluated on a downstream model using validation performance → the three-tier memory is updated → a Summary Agent consolidates global conceptual memory → the top-N features are selected and added to the dataset → the next iteration begins.

### Key Designs

1. **Parallel Architecture of Six Specialized Agents + Router Agent**:

    - **Function**: Achieves broad exploration of the feature space through role specialization.
    - **Mechanism**: Six agents are each responsible for one of the following: unary transformations (Unary), cross-compositional combinations (Cross-Compositional), temporal features (Temporal), aggregation construction (Aggregation-Construct), local transformations (Local-Transform), and local patterns (Local-Pattern). The Router Agent dynamically selects which agents to activate each round based on task metadata and accumulated memory.
    - **Design Motivation**: A single agent tends to produce homogeneous features (feature homogenization); multiple agents explore complementary regions along three orthogonal dimensions—transformation complexity, data scope, and data type.

2. **Three-Tier Memory Mechanism (Procedural + Feedback + Conceptual)**:

    - **Function**: Converts per-round evaluation feedback into persistent learning signals.
    - **Mechanism**: **Procedural memory** (ProcMem) records complete traces of transformation operations (base columns, transformation type, feature name, description, and round index) to avoid redundant exploration. **Feedback memory** (FeedMem) associates each feature with downstream validation metrics, enabling explicit credit assignment. **Conceptual memory** (ConMem) has the LLM distill reusable heuristic rules from procedural and feedback memory.
    - **Design Motivation**: LLM generation without memory is stateless. The three tiers progressively abstract from "what was done" → "how effective it was" → "why it worked," enabling short-term error avoidance, medium-term guided search, and long-term strategy adaptation.

3. **Global Conceptual Memory and Cross-Agent Knowledge Transfer**:

    - **Function**: Facilitates inter-agent coordination and knowledge sharing.
    - **Mechanism**: At the end of each round, the Summary Agent consolidates the conceptual and feedback memories of all active agents into a global conceptual memory (GlobalMem). Both the Router's scheduling decisions and each agent's prompt construction in the subsequent round reference this global memory.
    - **Design Motivation**: Local memory serves only a single agent; global memory propagates effective patterns to other agents, reducing overlapping exploration.

### Loss & Training
The objective is to maximize downstream model performance on the validation set (AUC for classification, NRMSE for regression). XGBoost is used as the downstream model, and the top-N features are retained at each round through a TopN-Features selection step.

## Key Experimental Results

### Main Results (Classification AUC, Mean Rank over 16 Datasets)

| Method | Type | Mean Rank |
|--------|------|-----------|
| DFS | Traditional | 3.69 |
| OpenFE | Traditional | 3.12 |
| CAAFE | LLM | 3.57 |
| OCTree | LLM | 4.81 |
| LLMFE | LLM | 3.75 |
| **MALMAS** | **Multi-Agent + Memory** | **1.12** |

### Ablation Study (Contribution of Key Components)

| Configuration | Description |
|---------------|-------------|
| Single Agent (no Router) | Reduced feature diversity; severe homogenization |
| No Memory | Independent generation per round; extensive redundant exploration |
| No Global Memory | No inter-agent knowledge transfer; increased redundant features |
| No Feedback Memory | Cannot learn which transformations are effective from history |
| **MALMAS (Full)** | Best performance; Mean Rank 1.12 |

### Key Findings
- **MALMAS achieves a mean rank of 1.12 across 16 classification datasets**, substantially outperforming the second-best method OpenFE (3.12).
- **The advantage is more pronounced on difficult datasets**: e.g., Titanic (0.872 vs. runner-up 0.849), Credit_G (0.775 vs. runner-up 0.758).
- **Memory mechanisms are critical**: conceptual memory abstracts "why a transformation is effective" into reusable rules that guide subsequent exploration.
- **Dynamic scheduling by the Router Agent** avoids the inefficiency of uniformly activating all agents on every dataset.

## Highlights & Insights
- **The hierarchical design of three-tier memory** is highly instructive: from operation traces to credit assignment to strategy abstraction, it mirrors the cognitive psychology hierarchy of procedural memory → working memory → metacognition. This paradigm is transferable to any multi-agent system requiring iterative optimization.
- **Dynamic scheduling by the Router Agent** addresses the computational waste of running all agents exhaustively, enabling task-dependent resource allocation.
- **Designing agent roles based on a "golden feature" taxonomy** represents a sound practice of encoding domain knowledge into agent specialization.

## Limitations & Future Work
- The agent role decomposition is manually designed; whether the optimal division of labor can be discovered automatically remains an open question.
- The memory management lacks a forgetting mechanism, which may lead to context bloat over long iteration sequences.
- The downstream model is fixed as XGBoost; effectiveness on deep learning models has not been verified.
- End-to-end comparisons with full AutoML pipelines (e.g., Auto-sklearn) are absent.
- Adversarial or debate mechanisms among agents could be explored to further improve feature quality.

## Related Work & Insights
- **vs. CAAFE**: CAAFE employs a single LLM for feature generation, constrained by a single generation strategy. MALMAS substantially expands the exploration space through multi-agent specialization and memory-driven feedback.
- **vs. OpenFE**: OpenFE performs operator search using tree models, which is efficient but restricted to predefined operators. MALMAS leverages the semantic understanding of LLMs to generate more diverse transformations.
- **vs. Generative Agents**: Generative Agents use multi-agent memory architectures for social simulation; MALMAS applies a similar paradigm to feature engineering, representing a novel application direction for this framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of multi-agent systems and three-tier memory for feature generation is novel, though individual components are not new in isolation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across 23 datasets is broad, but ablation details are insufficiently granular.
- Writing Quality: ⭐⭐⭐⭐ Framework diagrams are clear, though notation is slightly redundant.
- Value: ⭐⭐⭐⭐ Offers practical contributions to the AutoML community; the three-tier memory design has broad transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](../../NeurIPS2025/llm_nlp/large_language_models_miss_the_multi-agent_mark.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](../../AAAI2026/llm_nlp/collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[ACL 2026\] EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution](evospark_endogenous_interactive_agent_societies_for_unified_long-horizon_narrati.md)

</div>

<!-- RELATED:END -->
