---
title: >-
  [Paper Note] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data
description: >-
  [ACL 2026][Multi-Agent][AutoML] MALMAS is proposed as a memory-augmented LLM multi-agent system for automated feature generation on tabular data. Through a workforce of six specialized Agents exploring different feature space dimensions and a three-level memory mechanism (Procedural/Feedback/Conceptual) for cross-iteration optimization, it outperform
tags:
  - ACL 2026
  - Multi-Agent
  - AutoML
date: 2026-05-08
content_hash: 18f1dbbbd54778cf
---
# Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data

**Conference**: ACL 2026  
**arXiv**: [2604.20261](https://arxiv.org/abs/2604.20261)  
**Code**: [GitHub](https://github.com/fxdong24/MALMAS)  
**Area**: LLM/NLP  
**Keywords**: Automated Feature Engineering, Multi-Agent System, Memory Augmentation, Tabular Data, AutoML

## TL;DR
MALMAS is proposed as a memory-augmented LLM multi-agent system for automated feature generation on tabular data. Through a workforce of six specialized Agents exploring different feature space dimensions and a three-level memory mechanism (Procedural/Feedback/Conceptual) for cross-iteration optimization, it outperforms existing baselines across 16 classification and 7 regression datasets.

## Background & Motivation

**Background**: Automated feature generation is a critical component of AutoML, aiming to construct high-quality features from raw tabular data automatically. Traditional methods (e.g., DFS, OpenFE) rely on predefined operator libraries for combinatorial search, while recent LLM-based methods (e.g., CAAFE) introduce semantic information to guide feature transformations, yet limitations persist.

**Limitations of Prior Work**: (1) Traditional methods are constrained by fixed operator sets and cannot leverage task semantics, resulting in a narrow search space; (2) Although LLM methods introduce semantic signals, they depend on single generation strategies and suffer from rigid thinking patterns, leading to restricted feature space exploration; (3) Crucially, existing LLM methods lack a feedback mechanism from downstream learning objectives—the generation process is decoupled from model performance, resulting in inefficient trial-and-error exploration.

**Key Challenge**: The contradiction between the high dimensionality and diversity of the feature space and the limited exploration capability of a single Agent, alongside the absence of a "generation $\rightarrow$ evaluation $\rightarrow$ optimization" closed loop.

**Goal**: Design a multi-Agent collaborative and memory-driven automated feature generation framework capable of (1) extensively exploring the feature space through role differentiation and (2) achieving cross-iteration experience accumulation and strategy adjustment through multi-level memory.

**Key Insight**: Starting from the classification of "golden features" in feature engineering practice, specialized Agents are designed along three orthogonal dimensions (transformation complexity, data scope, and data type dependency), coupled with a three-level experience system: Procedural Memory (what was done), Feedback Memory (how effective it was), and Conceptual Memory (why it worked).

**Core Idea**: Decomposing feature generation into parallel exploration by multiple specialized Agents + dynamic scheduling by a Router Agent + iterative optimization driven by three-level memory.

## Method

### Overall Architecture
Each iteration: The Router Agent selects a subset of active Agents for the current round from the Agent pool $\rightarrow$ each active Agent constructs prompts based on metadata and memory to generate features through multi-turn interactions with the LLM $\rightarrow$ evaluate the validation performance of generated features on a downstream model $\rightarrow$ update the three-level memory $\rightarrow$ the Summary Agent aggregates global conceptual memory $\rightarrow$ select TopN features to add to the dataset $\rightarrow$ proceed to the next round.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Tabular Data + Task Metadata"] --> B["Router Agent Dynamic Scheduling<br/>Reads metadata & memory to select active subset"]
    subgraph AG["Parallel Architecture of 6 Specialized Agents + Router"]
        direction TB
        C["Unary / Cross / Temporal / Aggregation / Local Transform / Local Pattern<br/>Each handles one type of feature transformation, covering three orthogonal dimensions"]
    end
    B --> AG
    AG --> D["Downstream Evaluation<br/>XGBoost Validation AUC / NRMSE"]
    subgraph MEM["Three-Level Memory Mechanism"]
        direction TB
        E["Procedural Memory ProcMem<br/>What was done"] --> F["Feedback Memory FeedMem<br/>Effectiveness & Credit Assignment"] --> G["Conceptual Memory ConMem<br/>Why it worked"]
    end
    D --> MEM
    MEM --> H["Global Conceptual Memory & Cross-Agent Knowledge Transfer<br/>Summary Agent aggregates GlobalMem"]
    H -->|"TopN features added to dataset, next round"| B
    H --> I["Output Enhanced Feature Set"]
```

### Key Designs

**1. Parallel Architecture of 6 Specialized Agents + Router: Functional specialization avoids homogenization.**

Individual LLMs tend to produce homogenized features when generating repeatedly, with logic getting stuck on a few transformation types. MALMAS decomposes feature construction into six specialized Agents: Unary, Cross-Compositional, Temporal, Aggregation-Construct, Local-Transform, and Local-Pattern. This set of roles covers transformation complexity, data scope, and data types, allowing them to explore complementary rather than overlapping feature regions. Not all Agents participate in every round; the Router Agent first reads task metadata and accumulated memory to dynamically select the subset of Agents to activate, focusing computing resources on directions most likely to yield results for the current dataset.

**2. Three-Level Memory Mechanism (Procedural + Feedback + Conceptual): Establishing an experience chain of "what $\rightarrow$ how $\rightarrow$ why" for stateless LLMs.**

Without memory, an LLM starts from scratch every round, repeatedly exploring the same transformations and failing to learn which operations are truly useful. MALMAS uses three levels of memory to crystallize feedback: Procedural Memory (ProcMem) records the full trace of each transformation (base columns, transformation type, feature name, description, round), allowing subsequent rounds to avoid redundant paths; Feedback Memory (FeedMem) binds each generated feature with its validation metric on the downstream model for explicit credit assignment; Conceptual Memory (ConMem) distills reusable heuristic rules from the first two. These three layers of abstraction correspond to short-term error avoidance, mid-term goal orientation, and long-term strategic adaptation, transforming iterations from blind trial-and-error into structured optimization.

**3. Global Conceptual Memory and Cross-Agent Knowledge Transfer: Broadcasting effective patterns to the entire team.**

Local memory serves only a single Agent; if Agent B cannot use experience learned by Agent A, redundant features will still be created. After each round, the Summary Agent aggregates the conceptual and feedback memories of all active Agents into a Global Conceptual Memory (GlobalMem). This is referenced by the Router for scheduling decisions and by each Agent for prompt construction in the next round. Consequently, effective patterns discovered by one Agent can propagate through the team, and the Router can assign tasks more intelligently, reusing the results of successful exploration across the system.

### Example: One Iteration on Titanic

Using the Titanic dataset as an example: At the start of a round, the Router reads task metadata (containing columns like Age, Fare, Pclass, Sex) and existing memory. It determines that temporal features are irrelevant and activates only Unary, Cross-Compositional, and Aggregation-Construct Agents. The Unary Agent performs a log transformation on Fare; the Cross Agent combines Pclass and Sex into a new category; the Aggregation Agent calculates the mean fare per class. The features generated by the three are evaluated by XGBoost. FeedMem records that "log(Fare) slightly increased AUC, while Pclass×Sex provided the maximum gain." The Summary Agent writes "The interaction between class and sex is effective for survival prediction" into GlobalMem. In the next round, the Router continues to emphasize the cross-compositional direction, while ProcMem ensures log(Fare) is not generated again. After several iterations, the AUC for Titanic converges to 0.872, surpassing the 0.849 of the next best method.

### Loss & Training
The objective is to maximize the performance metrics of the downstream model on the validation set (AUC for classification and NRMSE for regression). XGBoost is employed as the downstream model, and the optimal features are retained via TopN-feature selection in each round.

## Key Experimental Results

### Main Results (Classification AUC, Average Rank across 16 Datasets)

| Method | Type | Mean Rank |
|------|------|-----------|
| DFS | Traditional | 3.69 |
| OpenFE | Traditional | 3.12 |
| CAAFE | LLM | 3.57 |
| OCTree | LLM | 4.81 |
| LLMFE | LLM | 3.75 |
| **MALMAS** | **Multi-Agent+Memory** | **1.12** |

### Ablation Study (Contribution of Key Components)

| Configuration | Description |
|------|------|
| Single Agent (No Router) | Feature diversity drops; high homogenization |
| No Memory | Independent generation each round; massive redundant exploration |
| No Global Memory | No knowledge transfer between Agents; increased redundant features |
| No Feedback Memory | Inability to learn which transformations are effective from history |
| **MALMAS (Full)** | Optimal performance, Mean Rank 1.12 |

### Key Findings
- **MALMAS achieves an average rank of 1.12 across 16 classification datasets**, significantly ahead of the runner-up OpenFE (3.12).
- **Advantages are more pronounced on difficult datasets**: e.g., Titanic (0.872 vs. 0.849) and Credit_G (0.775 vs. 0.758).
- **The memory mechanism is crucial**: Conceptual memory abstracts "why a transformation is effective" into reusable rules to guide subsequent exploration.
- **Dynamic scheduling by the Router Agent** avoids the computational waste of activating all Agents for every dataset.

## Highlights & Insights
- **Hierarchical Design of Three-Level Memory**: The progression from operation traces to credit assignment to policy abstraction mirrors the cognitive psychology concepts of procedural memory $\rightarrow$ working memory $\rightarrow$ metacognition. This is transferable to any multi-Agent system requiring iterative optimization.
- **Dynamic Scheduling via Router Agent**: Solves the inefficiency of "running all Agents," achieving task-dependent resource allocation.
- **Design of Agent Roles based on "Golden Features"**: A strong practice of encoding domain knowledge into the division of labor between Agents.

## Limitations & Future Work
- Agent role division is manually designed; can optimal divisions be discovered automatically?
- Memory management lacks a forgetting mechanism, which may lead to context bloat in long-running iterations.
- The downstream model is fixed to XGBoost; effectiveness on deep learning models has not been verified.
- No end-to-end comparison with full AutoML frameworks (e.g., Auto-sklearn).
- Potential to explore adversarial/debate mechanisms between Agents to further improve feature quality.

## Related Work & Insights
- **vs CAAFE**: CAAFE uses a single LLM and is limited by a single generation strategy. MALMAS significantly expands the exploration space through Agent specialization and memory feedback.
- **vs OpenFE**: OpenFE uses operator search on tree models, which is efficient but limited to predefined operators. MALMAS leverages the semantic understanding of LLMs to generate more diverse transformations.
- **vs Generative Agents**: While the latter uses multi-Agent with memory for social simulation, MALMAS introduces a similar paradigm to feature engineering, representing a new application direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Multi-Agent + three-level memory for feature generation is a novel combination, though components are individually known.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage with 23 datasets, though ablation details could be more granular.
- Writing Quality: ⭐⭐⭐⭐ Framework diagrams are clear, though notation is slightly redundant.
- Value: ⭐⭐⭐⭐ Practical contribution to the AutoML community; the three-level memory design has high transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](efficient_multi-agent_system_training_with_data_influence-oriented_tree_search.md)
- [\[ACL 2025\] DocAgent: A Multi-Agent System for Automated Code Documentation Generation](../../ACL2025/multi_agent/docagent_a_multi-agent_system_for_automated_code_documentation_generation.md)
- [\[ACL 2026\] A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation](a_multi-agent_framework_for_feature-constrained_difficulty_control_in_reading_co.md)
- [\[AAAI 2026\] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](../../AAAI2026/multi_agent/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)

</div>

<!-- RELATED:END -->
