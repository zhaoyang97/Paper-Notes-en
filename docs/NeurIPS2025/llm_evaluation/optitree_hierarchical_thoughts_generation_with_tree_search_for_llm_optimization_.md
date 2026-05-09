---
title: >-
  [Paper Note] OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling
description: >-
  [NeurIPS 2025][LLM Evaluation][Operations Research Modeling] This paper proposes OptiTree, which organizes hierarchical classification and modeling thoughts for operations research (OR) problems by constructing a modeling tree, and employs tree search to adaptively decompose complex problems into sequences of simpler subproblems, achieving significant accuracy gains in optimization modeling tasks for LLMs (exceeding 10% on multiple challenging benchmarks).
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - Operations Research Modeling
  - LLM Reasoning
  - Tree Search
  - Subproblem Decomposition
  - Hierarchical Thoughts
date: 2026-05-08
content_hash: bd1489fc694a0e90
---

# OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2510.22192](https://arxiv.org/abs/2510.22192)
**Code**: [GitHub](https://github.com/MIRALab-USTC/OptiTree/tree/main)
**Area**: LLM Evaluation
**Keywords**: Operations Research Modeling, LLM Reasoning, Tree Search, Subproblem Decomposition, Hierarchical Thoughts

## TL;DR

This paper proposes OptiTree, which organizes hierarchical classification and modeling thoughts for operations research (OR) problems by constructing a modeling tree, and employs tree search to adaptively decompose complex problems into sequences of simpler subproblems, achieving significant accuracy gains in optimization modeling tasks for LLMs (exceeding 10% on multiple challenging benchmarks).

## Background & Motivation

Operations Research (OR) modeling refers to the process of transforming natural-language descriptions of real-world problems into mathematical optimization models, a task that traditionally requires extensive expert knowledge. Recent work has explored using LLMs to automate this process, falling into two main categories:

**Prompt-based methods** (CoE, OptiMUS, MCTS): decompose the modeling task into fixed steps — sequentially generating variables, constraints, and objective functions.

**Fine-tuning methods** (ORLM, LLMOPT): train specialized LLMs on large-scale modeling datasets.

**Key limitations of existing approaches**:

- Fixed-step decomposition ignores problem complexity and performs poorly on hard instances.
- The authors' analysis reveals that **over 70% of errors on Medium and Hard problems stem from incorrect variable definitions**.
- Structural relationships among problems are not exploited.

**Core observations** (three motivating findings):

**Observation 1**: Existing fixed-step decomposition exhibits high failure rates on complex problems, with variable definition being the primary bottleneck.

**Observation 2**: 69% of complex OR problems contain standard OR problems as subproblems, and LLMs can identify these subproblems with relatively high accuracy.

**Observation 3**: A naive subproblem-decomposition approach already improves modeling performance.

## Method

### Overall Architecture

The core idea of OptiTree is to **adaptively decompose complex OR problems into a sequence of simpler subproblems, and leverage the modeling experience (modeling thoughts) from subproblems to guide the modeling of the original problem**.

The overall pipeline consists of three stages:
1. **Modeling tree construction**: automatically build a hierarchical tree structure from a dataset.
2. **Tree search and thought retrieval**: given a new problem, search the tree to identify an appropriate subproblem chain.
3. **Global modeling thought synthesis**: integrate hierarchical thoughts from subproblems to generate the final model.

### Subproblem Definition and Identification

Given an OR problem $\mathcal{P}$ with optimization model:

$$\min_{\boldsymbol{x}} f(\boldsymbol{x}) \quad \text{s.t.} \quad g_i(\boldsymbol{x}; \beta_i) \leq 0, \quad i=1,\ldots,N$$

partitioning the variables as $\boldsymbol{x} = (\tilde{\boldsymbol{x}}_1, \tilde{\boldsymbol{x}}_2)$, a subproblem $\tilde{\mathcal{P}}$ is defined as a simplified problem involving only a subset of variables and constraints:

$$\min_{\tilde{\boldsymbol{x}}_1} f_1(\tilde{\boldsymbol{x}}_1) \quad \text{s.t.} \quad g_{i_k,1}(\tilde{\boldsymbol{x}}_1; \beta_{i_k,1}) \leq 0$$

**Identification**: an LLM distills the problem description into atomic-level statement thoughts $\mathcal{C}_{\mathcal{P}} = \{c_1, c_2, \ldots, c_{n_\mathcal{P}}\}$, and the subproblem relationship is determined via the semantic containment relation $\mathcal{C}_{\tilde{\mathcal{P}}} \subseteq_{\mathcal{S}} \mathcal{C}_{\mathcal{P}}$.

### Modeling Tree Structure

The **Modeling Tree** is a hierarchical tree structure where:

- **Root node**: represents an abstract class of combinatorial optimization problems.
- **Each node** contains: problem category name, statement thoughts $\mathcal{C}_{\mathcal{P}}$, and modeling thoughts $\mathcal{T}(\mathcal{P})$.
- **Parent-child relationship**: a parent node is a subproblem of its child node ($\mathcal{P}_j \subseteq_{\mathcal{S}} \mathcal{P}_i$).
- **Child nodes inherit** the basic variables and constraints of parent nodes and add specialized components.
- **Order-preserving property**: ancestor nodes are always subproblems of descendant nodes.

### Tree Search and Global Modeling Thought Construction

Given problem $\mathcal{P}$, the search proceeds layer by layer from the root:

$$\mathcal{P}^{(1)} = \underset{\mathcal{P}_t^{(0)}}{\text{argmax}} \; \mathbb{I}(\mathcal{P}_t^{(0)} \subseteq_{\mathcal{S}} \mathcal{P}) \cdot \text{Sim}_{\text{LLM}}(\mathcal{C}_{\mathcal{P}_t^{(0)}}, \mathcal{C}_{\mathcal{P}})$$

This yields a subproblem chain $\mathcal{P}^{(1)} \subseteq_{\mathcal{S}} \mathcal{P}^{(2)} \subseteq_{\mathcal{S}} \cdots \subseteq_{\mathcal{S}} \mathcal{P}^{(M)}$. The modeling thoughts $\mathcal{T}(\mathcal{P}^{(M)})$ of the largest subproblem $\mathcal{P}^{(M)}$ are combined with the problem description to synthesize global modeling thoughts $\mathcal{T}(\mathcal{P})$.

### Dynamic Construction and Update of the Modeling Tree

The tree is automatically constructed from 400 problems drawn from the OR-Instruct 3K dataset:
1. For each new problem, tree search is performed to find the largest subproblem.
2. If the modeling result is correct — the tree already covers this problem type.
3. If incorrect — **node expansion** is triggered to insert the new problem into the tree.
4. During expansion, the order-preserving property is maintained by checking subproblem relationships between the new problem and existing sibling nodes.

## Key Experimental Results

### Main Results

Modeling accuracy comparison across 7 benchmark datasets (based on DeepSeek-V3):

| Method | NL4Opt | MAMO EasyLP | MAMO ComplexLP | ComplexOR | IndustryOR | OptiBench | OptMATH |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CoT | 74.0 | 82.9 | 40.7 | 52.6 | 35.0 | 53.1 | 21.1 |
| CoE | 79.2 | 85.9 | 43.1 | 63.2 | 33.0 | 55.2 | 24.1 |
| OptiMUS | 80.6 | 87.1 | 45.2 | 79.0 | 36.0 | 58.8 | 32.5 |
| MCTS | 89.6 | 88.0 | 51.6 | 79.0 | 46.0 | 67.9 | 38.6 |
| DeepSeek-R1 | 86.1 | 79.5 | 57.3 | 68.4 | 38.0 | 70.2 | 33.1 |
| OpenAI-o1 | 87.1 | 87.6 | 54.5 | 73.6 | 40.0 | 71.5 | 34.9 |
| **OptiTree** | **98.3** | **96.9** | **81.5** | **84.2** | **54.0** | **74.7** | **52.4** |

### Ablation Study

| Variant | MAMO ComplexLP | IndustryOR |
|:---|:---:|:---:|
| OptiTree (full) | 81.5 | 54.0 |
| w/o Tree Search | notable drop | notable drop |
| w/o Modeling Thoughts | significant drop | significant drop |
| depth=1 | below full | below full |
| depth=3 | close to full | close to full |

Efficiency comparison (seconds/problem, IndustryOR):

| Method | Total Time |
|:---|:---:|
| CoE | 81.8 |
| OptiMUS | 57.8 |
| MCTS | 124.6 |
| OptiTree | **19.9** (search 8.4 + modeling 11.5) |

### Key Findings

1. **Largest gains on hard datasets**: MAMO ComplexLP improves from MCTS's 51.6% to 81.5% (+30%); OptMATH improves from 38.6% to 52.4% (+14%).
2. **Outperforms reasoning LLMs**: OptiTree + DeepSeek-V3 surpasses both DeepSeek-R1 and OpenAI-o1.
3. **High subproblem coverage**: on average, 88% of problems can be matched to a subproblem, with high manual verification accuracy.
4. **Clear efficiency advantage**: OptiTree's inference time is approximately 1/6 that of MCTS.
5. **Only 400 problems** are needed to construct a well-generalizing modeling tree.

## Highlights & Insights

1. **The subproblem decomposition insight is particularly compelling**: OR modeling is reframed from a flat pipeline of "generate variables → constraints → objective" to a hierarchical process of "find a structurally similar simpler problem → model incrementally," closely mirroring the reasoning process of human experts.
2. **The modeling tree design is elegant**: it serves simultaneously as a knowledge organization structure and a search space, naturally integrating RAG and tree search.
3. **Statement thoughts effectively mitigate LLM hallucination**: converting problems into atomic-level semantic descriptions before comparison is more reliable than directly comparing natural-language descriptions.
4. **Automated modeling tree construction**: no manual curation is required; the tree is built and updated in a data-driven manner, ensuring scalability.
5. **Dramatic reduction in search space**: the exponential space of variables and constraints is reduced to a finite set of subproblems.

## Limitations & Future Work

1. **Modeling tree quality depends on the construction dataset**: if the dataset covers a limited range of OR problem types, the tree's generalization ability is constrained.
2. **Focus on prompt-based methods only**: integration with fine-tuning methods is not explored; OptiTree's thoughts could in principle augment fine-tuning data.
3. **Subproblem identification relies on LLM capability**: identification accuracy may degrade for novel OR problem types unfamiliar to the LLM.
4. **Trade-off between tree depth and breadth**: overly deep trees may introduce cumulative errors, while overly shallow trees may yield insufficient decomposition.
5. **More complex scenarios such as multi-objective optimization are not considered**.

## Related Work & Insights

- **Buffer-of-Thoughts (BoT)**: the concept of a thought buffer is closely related to the storage and retrieval of modeling thoughts in this work.
- **RAG + tree search integration**: provides a paradigm reference for other LLM applications requiring structured knowledge retrieval.
- **Incremental modeling strategy**: can be generalized to tasks that require step-by-step construction, such as mathematical proof generation and code generation.

## Rating

- ⭐⭐⭐⭐ (4/5)
- **Novelty** ⭐⭐⭐⭐⭐: the combination of subproblem decomposition, modeling tree, and tree search is highly original and captures the structural properties of OR modeling.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: 7 datasets, multiple LLM backbones, and detailed ablation and efficiency analyses.
- **Writing Quality** ⭐⭐⭐⭐: motivation is clear, with three observations building progressively upon one another.
- **Value** ⭐⭐⭐⭐: directly applicable to OR practitioners and real-world LLM deployment.
- **Theoretical Depth** ⭐⭐⭐: the subproblem definition is formalized, but the proof of the order-preserving property is relatively straightforward.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[AAAI 2026\] MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](../../AAAI2026/llm_evaluation/mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)
- [\[AAAI 2026\] GDBA Revisited: Unleashing the Power of Guided Local Search for Distributed Constraint Optimization](../../AAAI2026/llm_evaluation/gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/llm_evaluation/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)

<!-- RELATED:END -->
