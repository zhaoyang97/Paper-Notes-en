---
title: >-
  [Paper Note] Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search
description: >-
  [ICML 2025][Self-Supervised Learning][Text-to-SQL] Alpha-SQL models zero-shot Text-to-SQL as a tree search problem. By combining a Monte Carlo Tree Search (MCTS) framework with an LLM-as-Action-Model and a self-supervised reward function, it achieves a 69.7% execution accuracy on the BIRD dataset using a 32B open-source model without any fine-tuning, surpassing the GPT-4o-based zero-shot SOTA by 2.5 percentage points.
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Text-to-SQL"
  - "Monte Carlo Tree Search (MCTS)"
  - "Large Language Models"
  - "Zero-Shot Inference"
  - "Test-Time Compute"
date: 2026-05-08
content_hash: e076302605c04461
---

# Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search

**Conference**: ICML 2025  
**arXiv**: [2502.17248](https://arxiv.org/abs/2502.17248)  
**Code**: [HKUSTDial/Alpha-SQL](https://github.com/HKUSTDial/Alpha-SQL)  
**Area**: Self-Supervised Learning  
**Keywords**: Text-to-SQL, Monte Carlo Tree Search (MCTS), Large Language Models, Zero-Shot Inference, Test-Time Compute

## TL;DR

Alpha-SQL models zero-shot Text-to-SQL as a tree search problem. By combining a Monte Carlo Tree Search (MCTS) framework with an LLM-as-Action-Model and a self-supervised reward function, it achieves a 69.7% execution accuracy on the BIRD dataset using a 32B open-source model without any fine-tuning, surpassing the GPT-4o-based zero-shot SOTA by 2.5 percentage points.

## Background & Motivation

The Text-to-SQL task aims to translate natural language queries into SQL statements, holding significant application value in data analysis and database interactions. Current LLM-based methods fall into two categories:

**Fine-tuning methods**: Fine-tuning LLMs on labeled data (e.g., CodeS, CHESS-SQL) yields strong performance but is highly costly. The training process must be repeated when new models emerge.

**Zero-shot methods**: Leveraging pre-trained LLM knowledge directly to generate SQL (e.g., DAIL-SQL, C3) avoids reliance on annotations but offers limited performance.

**Key Challenge**: Under zero-shot scenarios, models struggle to transfer general pre-trained knowledge to specific SQL generation tasks. LLMs face difficulties in accurately understanding schema relations, constructing complex SQL queries, and maintaining robustness across different contexts when handling complex mappings between natural language and database schemas.

**Design Motivation**: The task of SQL generation can be decomposed into smaller, manageable subtasks. By gradually constructing SQL through progressive construction, contextual guidance is provided at each step, thereby reducing step-by-step complexity. This process can naturally be modeled as a tree search problem.

## Method

### Overall Architecture

Alpha-SQL models Text-to-SQL as a search problem over a tree-structured search space. The core idea is to represent the SQL query construction process as a path from the root node to a leaf node:

- **Node (V)**: Represents a partial reasoning state in the SQL construction process. The root node $v_0$ is the empty query state (containing problem $q$ and database schema $\mathcal{D}$), intermediate nodes store incremental reasoning steps, and leaf nodes represent complete SQL queries or termination states.
- **Edge (E)**: Represents SQL construction actions (such as selecting tables, adding conditions, applying aggregation functions, etc.).
- **Path**: A path from the root to a leaf corresponds to a complete reasoning trajectory, generating a candidate SQL: $y = v_0 \oplus v_1 \oplus \cdots \oplus v_t$.

The search space $|\mathcal{S}|$ grows exponentially with database schema and query complexity (number of tables, number of columns, JOIN conditions, nested subqueries, etc.), making exhaustive search infeasible. Alpha-SQL utilizes the MCTS framework to explore this space efficiently.

### Key Designs

#### 1. LLM-as-Action-Model

At each step of MCTS, Alpha-SQL calls the LLM as an action model to generate reasoning actions based on the current context (question, schema, existing reasoning trajectory). Each step of reasoning is stored in the node in a Chain-of-Thought (CoT) format to maintain contextual coherence:

$$v_{i+1} = LLM(q, \mathcal{D}, Actions(v_0, \ldots, v_i), Prompt(a_i))$$

**Action Space Definition**: A total of 7 SQL-construction reasoning actions are defined, each with a specialized prompt template:

| Action | Name | Function Description |
|------|------|----------|
| $A_1$ | Question Rephrasing | Decomposes ambiguous questions into a structured "list of conditions + question" format to eliminate ambiguity. |
| $A_2$ | Schema Selection | Identifies the subset of tables and columns relevant to the question from the full schema via CoT reasoning. |
| $A_3$ | Column Value Identification | Identifies column value filtering conditions required for the WHERE clause (e.g., "Bob" → name='Bob'). |
| $A_4$ | Column Function Identification | Identifies required aggregate/scalar functions (e.g., COUNT, STRFTIME, etc.). |
| $A_5$ | SQL Generation | Adopts a Divide-and-Conquer CoT strategy, decomposing complex queries into subtasks and combining them. |
| $A_6$ | SQL Revision | Multi-round error correction based on execution result feedback, up to $N_{revision}$ rounds. |
| $A_7$ | Termination | Terminates the reasoning trajectory; must follow $A_5$ or $A_6$. |

**Action Order Constraints**: Strict sequential constraints exist between actions (defined by a valid transition matrix) to ensure logical consistency. Each action appears at most once in a single reasoning trajectory to prevent infinite loops. Calculations show over 3,000 valid reasoning paths.

#### 2. MCTS Candidate SQL Generation

Each MCTS rollout consists of four classic phases:

**(1) Selection**: Starting from the root node, nodes are selected using the UCT (Upper Confidence Bound for Trees) formula:

$$UCT(v, a) = \frac{Q(v, a)}{N(v, a)} + c \sqrt{\frac{\ln N(v)}{N(v, a)}}$$

where $N(v, a)$ is the action visit count and $Q(v, a)$ is the accumulated reward. If unvisited child nodes exist ($N(v,a)=0$), these nodes are prioritized.

**(2) Expansion**: Valid successor actions are determined based on the current node type, with each action sampled $N_{expansion}$ times (temperature $T_{expansion}$), generating $N_{expansion} \times |E_{valid}|$ child nodes. **Key Pruning**: If multiple sampled final results are identical (e.g., Schema Selection selects the same subset), only one node is kept, drastically reducing the branching factor.

**(3) Simulation**: Iterate selection and expansion until a termination node is reached; all newly expanded nodes are permanently retained in the tree.

**(4) Backpropagation**: The predicted SQL is evaluated at the termination node, backtracking from the termination node to the root, updating the $Q$ and $N$ values for all nodes along the path:

$$Q(v, a) = Q(v, a) + r, \quad N(v) = N(v) + 1$$

**Final SQL Selection**: After completing $N_{rollout}$ rollouts, all complete trajectories reaching the terminal node are collected. All candidate SQL queries are executed, and the one with the highest execution result consistency is chosen as the final output.

#### 3. Self-Supervised Reward Function

The reward function is the core evaluation mechanism of MCTS. Traditional methods (Outcome Reward Model, Progress Reward Model) require labeled data for training, which limits their generalization.

**Core Idea**: Similar to confidence in human reasoning — a confident individual will provide consistent answers across multiple attempts (high consistency = high confidence = high quality).

**Implementation**: For the predicted SQL $y$ of each reasoning path, $N_{reward}$ candidate SQL queries $\{y_i\}$ are generated using high-temperature sampling. After filtering out invalid queries, the self-consistency score is computed:

$$R(y, q, \mathcal{D}) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{Execute}(y, \mathcal{D}) = \text{Execute}(y_i, \mathcal{D})]$$

This function is entirely self-supervised and requires no annotated data.

#### 4. Database Value Retrieval

To bridge the semantic gap between user expressions and stored database values (e.g., "America" vs "United States"), a two-stage method is utilized:

- **Offline Preprocessing**: MinHash signatures are generated for TEXT type column values and stored locally.
- **Online Retrieval**: Keywords are extracted from the user question, and LSH (Locality Sensitive Hashing) is used to match MinHash signatures. After filtering via an edit distance threshold $\epsilon_{edit}$ and a semantic similarity threshold $\epsilon_{semantic}$, the matched values are injected into the schema prompt.

### Loss & Training

Alpha-SQL does not involve any model training or fine-tuning; it is executed entirely at inference time. Core hyperparameter settings are as follows:

| Parameter | Value | Description |
|------|-----|------|
| $N_{rollout}$ | 24 | MCTS search rollouts |
| $N_{expansion}$ | 3 | Number of samples per action |
| $T_{expansion}$ | 0.8 | Sampling temperature in expansion phase |
| $N_{reward}$ | 5 | Number of samples for reward calculation |
| $T_{reward}$ | 1.0 | Sampling temperature for reward calculation |
| $N_{revision}$ | 10 | Max iterations for SQL Revision |
| $\epsilon_{edit}$ | 0.3 | Edit distance similarity threshold |
| $\epsilon_{semantic}$ | 0.6 | Semantic similarity threshold |

## Key Experimental Results

### Main Results

Evaluation datasets: BIRD (1534 samples, more complex) and Spider (1034 samples). The evaluation metric is execution accuracy (EX).

**BIRD Development Set Results**:

| Method | Inference Model | Zero-Shot | Overall EX (%) |
|------|----------|--------|-------------|
| CodeS (SFT) | CodeS-7B | ✗ | 57.0 |
| CHESS-SQL | DS-Coder-33B | ✗ | 65.0 |
| Distillery | GPT-4o | ✗ | 67.2 |
| RSL-SQL | GPT-4o | ✓ | 67.2 |
| **Alpha-SQL** | **Qwen2.5-7B** | **✓** | **66.8** |
| **Alpha-SQL** | **Qwen2.5-14B** | **✓** | **68.4** |
| **Alpha-SQL** | **Qwen2.5-32B** | **✓** | **69.7** |

**Spider Development Set Results**:

| Method | Inference Model | Zero-Shot | All EX (%) |
|------|----------|--------|------------|
| CodeS (SFT) | CodeS-15B | ✗ | 84.9 |
| DIN-SQL | GPT-4 | ✓ | 82.8 |
| SuperSQL | GPT-4 | ✓ | 87.0 |
| **Alpha-SQL** | **Qwen2.5-14B** | **✓** | **87.0** |

### Ablation Study

Conducted on the SDS (Subsampled Development Set, 147 samples) using Qwen2.5-Coder-7B:

**Ablation on Action Space**:

| Configuration | EX (%) | Change | Description |
|------|--------|------|------|
| Alpha-SQL (Full) | 64.6 | — | All 7 reasoning actions |
| w/o $A_1$ (Question Rephrasing) | 63.9 | −0.7 | Question rephrasing helps eliminate ambiguity |
| w/o $A_2$ (Schema Selection) | 63.1 | −1.5 | High importance of Schema Selection |
| w/o $A_3$ (Column Value Identification) | 64.2 | −0.4 | Column value identification makes a minor contribution |
| w/o $A_4$ (Column Function Identification) | 64.0 | −0.6 | Function identification has a moderate contribution |
| w/o $A_6$ (SQL Revision) | 62.8 | −1.8 | **Largest impact**; error-correction from execution feedback is critical |

**Comparison with Baseline LLMs (SDS Dataset, direct prompting)**:

| Model | EX (%) | Notes |
|------|--------|------|
| QwQ-32B-Preview | 38.8 | Reasoning model |
| Phi-4 | 43.5 | — |
| Qwen2.5-Coder-7B | 47.6 | — |
| DeepSeek-R1 | 50.3 | Reasoning model |
| Deepseek-V3 | 51.2 | — |
| GPT-4o | 53.7 | — |
| Gemini-1.5-Pro | 56.2 | — |
| Gemini-2.0-Flash-Thinking | 60.8 | Reasoning model |
| **Phi-4 + Alpha-SQL** | **60.0** | **+16.5** |
| **Qwen-7B + Alpha-SQL** | **64.6** | **+17.0, surpassing all baselines** |

### Key Findings

1. **7B Model + Alpha-SQL surpasses GPT-4o direct inference**: Qwen2.5-Coder-7B (47.6%) combined with Alpha-SQL reaches 64.6%, far exceeding GPT-4o's 53.7%.
2. **Performance scales with the number of rollouts**: The number of MCTS rollouts is positively correlated with both the upper bound accuracy and final accuracy. With only 24 rollouts, high-quality SQL queries can be efficiently identified from over 3,000 possible paths.
3. **SQL Revision is the most critical action**: Removing it results in a 1.8% drop, indicating that database execution feedback is indispensable in Text-to-SQL.
4. **Plug-and-play advantage**: Alpha-SQL delivers a 15-17% improvement across different base models (Qwen, Phi-4), validating the generality of the framework.

## Highlights & Insights

1. **Innovative Problem Modeling**: Reformulating Text-to-SQL from "one-step generation" to a "tree search problem" and naturally integrating it into the MCTS framework represents an excellent application of test-time compute scaling in the structured data domain.
2. **Self-Supervised Reward Function**: The reward design, based on self-consistency of execution results, cleverly bypasses the dependency on annotated data, inspiring a general evaluation paradigm for zero-shot scenarios.
3. **Small Model, Big Capability**: Compensating for model capability deficiencies through search frameworks, a 7B model can match or even exceed GPT-4o, demonstrating the immense potential of inference-time scaling.
4. **Rational Action Space Design**: The 7 action types address the key difficulties of Text-to-SQL (schema understanding, value matching, function selection, error correction), and the sequential execution constraints ensure the logical consistency of reasoning trajectories.
5. **Practical Pruning Strategy**: Deduplicating expansion nodes with identical results drastically reduces search costs without losing information.

## Limitations & Future Work

1. **High Inference Cost**: Every step within the 24 MCTS rollouts requires LLM invocations. The reasoning cost per question is much higher than single-shot generation, which limits its applicability to real-time interactive scenarios.
2. **Dependence on External Embedding Models**: Database value retrieval semantic matching relies on OpenAI's `text-embedding-3-large`, adding dependencies on external services.
3. **Fixed Action Space**: The 7 action types are manually designed; whether they constitute the optimal set and whether better action combinations can be discovered automatically remains an open question.
4. **Evaluation Limitations**: Evaluation was restricted to BIRD and Spider, leaving more diverse database scenarios (such as multi-databases, streaming SQL, etc.) untested.
5. **Scalable Directions**: Future work could explore combining MCTS with reinforcement learning for online self-improvement, or introducing process reward models to replace the self-consistency reward.

## Related Work & Insights

- **CHASE-SQL**: Multi-step pipeline generation + candidate SQL verification, but relies on fine-tuning Gemini; Alpha-SQL replaces the static pipeline dynamically with MCTS.
- **rStar**: A pioneer work in employing MCTS + LLMs for general reasoning tasks; Alpha-SQL adapts it to the specific challenges of Text-to-SQL.
- **Tree of Thoughts**: Explores tree search to enhance LLM reasoning; Alpha-SQL integrates tree search deeper with SQL execution feedback.
- **Test-time computation scaling**: This study serves as a successful case of inference-time scaling on database tasks, further validating the trend of "search > larger models".

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐ | Deploying MCTS for Text-to-SQL is a novel combination, and the self-supervised reward design is ingenious. |
| Technical Depth | ⭐⭐⭐⭐ | The framework is thoroughly designed, with action space, pruning, and reward functions tightly integrated. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Comprehensive evaluation across multiple models, benchmarks, and ablation studies. |
| Practical Value | ⭐⭐⭐⭐ | Plug-and-play with no fine-tuning required, resulting in a low deployment barrier. |
| Writing Quality | ⭐⭐⭐⭐ | Clearly structured, rich in figures/tables, with standard formulas. |
| **Overall** | **⭐⭐⭐⭐** | **A landmark work in zero-shot Text-to-SQL and an excellent practice of MCTS + LLM.** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CObL: Toward Zero-Shot Ordinal Layering without User Prompting](../../ICCV2025/self_supervised/cobl_toward_zero-shot_ordinal_layering_without_user_prompting.md)
- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](../../ICML2026/self_supervised/from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](../../ICML2026/self_supervised/infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](../../NeurIPS2025/self_supervised/t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
