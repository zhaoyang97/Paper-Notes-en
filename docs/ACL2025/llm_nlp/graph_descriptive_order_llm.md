---
title: >-
  [Paper Note] Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?
description: >-
  [ACL 2025][LLM (Other)][Graph Reasoning] This paper presents the first systematic study on the impact of graph description orders (BFS, DFS, PageRank, PPR) when LLMs solve graph reasoning problems, demonstrating that ordered descriptions significantly outperform random ones, and different tasks prefer different permutation strategies.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Graph Reasoning"
  - "Graph Description Order"
  - "Prompt Engineering"
  - "BFS/DFS"
  - "PageRank"
  - "Graph Problems"
date: 2026-05-08
content_hash: e7def90cce7a324a
---

# Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?

**Conference**: ACL 2025  
**arXiv**: [2402.07140](https://arxiv.org/abs/2402.07140)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Graph Reasoning, Graph Description Order, Prompt Engineering, BFS/DFS, PageRank, Graph Problems  

## TL;DR

This paper presents the first systematic study on the impact of graph description orders (BFS, DFS, PageRank, PPR) when LLMs solve graph reasoning problems, demonstrating that ordered descriptions significantly outperform random ones, and different tasks prefer different permutation strategies.

## Background & Motivation

1. LLMs have made remarkable progress in reasoning tasks (mathematical reasoning, logical deduction), with graph problems becoming an important research direction due to their structural complexity.
2. Existing work has explored various graph encoding methods (adjacency matrices, edge lists, etc.) and carefully designed prompts, but has generally overlooked a critical factor: the **presentation order of edges/nodes in the graph description**.
3. Since graphs have no fixed textual representation, the permutation order of edges can affect the model's attention allocation, thereby influencing the reasoning process.
4. Prior studies defaulted to using random order when describing graph structures, without systematically investigating the impact of order.
5. Different description orders can provide the model with different "perspectives"—BFS offers a hierarchical view, DFS provides a depth view, and PageRank offers a global importance view.
6. This paper is the first to propose "graph description order" as a research question, constructing the GraphDO dataset for comprehensive experiments to fill this gap.

## Method

### Overall Architecture

The graph problem is modeled as an optimization problem: given a graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$, the goal is to find the optimal description order $o \in \mathcal{O}$ that maximizes the LLM's performance score on the graph task. The framework consists of three modules: a graph encoder (converting the graph into text), graph description ordering (determining the presentation sequence of edges), and task questioning (generating questions for corresponding graph tasks). An adjacency format (edge list) is adopted as the unified encoding method.

### Module 1: Graph Description Ordering Strategies

Four ordered description strategies are designed (compared with the random baseline):

- **BFS Order**: Starting from a random root node, traversing level by level, executed on the dual graph $\mathcal{G}^*$ to ensure all edges are included. This provides a hierarchical perspective of the graph structure.
- **DFS Order**: Starting from the root node, recursively exploring deeply and backtracking, providing a depth-first path perspective.
- **PageRank Order**: Sorting nodes in descending order of their PageRank scores, where $PR(v) = \alpha \sum_{u} \frac{PR(u)}{|\mathcal{N}(u)|} + (1-\alpha)$ ($\alpha=0.85$), providing a global perspective of node importance.
- **Personalized PageRank Order (PPR)**: Introducing a personalization vector $e_v$ (targeting task-related destination nodes) to provide a localized importance perspective.

### Module 2: Graph Task Design

Six types of graph reasoning tasks are designed, covering different complexity levels:

- **Local Reasoning**: Connectivity Detection (T1), Cycle Detection (T2)—binary classification tasks.
- **Global Reasoning**: Hamiltonian Path (T3), Shortest Path (T4), Topological Sort (T5)—requiring global comprehension of the graph structure.
- **Property Learning**: Node Classification (T6)—predicting the category of unknown nodes based on neighbor labels, which is closer to practical applications.

### Module 3: Prompt Style

Five prompt styles are designed to provide different levels of heuristic reasoning guidance: Zero-shot, Zero-shot CoT, Few-shot, CoT, and CoT-BAG (Build-a-Graph), gradually proceeding from no guidance to strong guidance.

### Training / Inference

This is a pure evaluation study and does not involve model training. GPT-3.5-Turbo-0613 is used as the default model with the decoding temperature set to 0. Generalization is also verified on open-source models including LLaMA2-7B/13B, Qwen2-7B, Mistral-7B, and Vicuna-7B.

## Experiments

### Table 1: Traditional Graph Tasks

| Task | Random | BFS | DFS | PR | PPR |
|------|--------|-----|-----|-----|-----|
| Connectivity | 78.36 | **89.43**(↑14.1%) | 85.21(↑8.8%) | 83.79(↑6.9%) | 82.50(↑5.3%) |
| Cycle Detection | 64.50 | **72.71**(↑12.7%) | 67.93(↑5.3%) | 69.14(↑7.2%) | 67.72(↑5.0%) |
| Hamiltonian Path | 31.50 | 42.86(↑36.1%) | **52.43**(↑66.4%) | 37.64(↑19.5%) | 37.93(↑20.4%) |
| Topological Sort | 46.93 | 55.43(↑18.1%) | **63.21**(↑34.7%) | 53.14(↑13.2%) | 54.93(↑17.1%) |
| Shortest Path | 30.14 | **51.29**(↑70.2%) | 45.43(↑50.7%) | 41.21(↑36.7%) | 42.86(↑42.2%) |

### Table 2: Node Classification Tasks

| Dataset (Ego) | Random | BFS | DFS | PR | PPR |
|-------------|--------|-----|-----|-----|-----|
| CORA | 70.00 | 72.00 | 71.33 | **75.33**(↑7.6%) | 73.33 |
| Citeseer | 67.33 | 68.67 | 68.66 | **71.33**(↑5.9%) | 69.33 |
| Pubmed | 72.00 | 74.00 | 77.33 | **82.67**(↑14.8%) | 77.33 |

### Key Findings

1. **Ordered descriptions globally outperform random ones**: All four ordered description methods outperform the random baseline across all tasks, with BFS achieving up to a 70.2% improvement on the shortest path task.
2. **Task complexity determines robustness**: Simpler tasks (connectivity, cycle detection) show lower variance and higher robustness to description order, whereas complex tasks (Hamiltonian path, shortest path) exhibit higher variance and sensitivity.
3. **Different tasks prefer different orders**: Local reasoning tasks (connectivity, cycle detection, shortest path) favor the hierarchical traversal of BFS; global reasoning tasks (Hamiltonian path, topological sort) prefer the deep exploration of DFS; node classification tasks favor the global importance ordering of PageRank (PR).
4. **CoT does not mitigate attention bias**: Even when utilizing CoT to guide "slow thinking", the performance variance introduced by the description order is not significantly reduced.
5. **Path ordering sanity check**: Designing two extreme orders, namely "shortest path order" and "longest path order", reveals that even with the highest overlap with the correct answer, the accuracy is far from 100%. This indicates that ordered descriptions indeed enhance the LLM's understanding of the graph structure rather than simply leaking the answer.

## Highlights & Insights

- Proposes the neglected research perspective of "graph description order" for the first time, with a simple yet insightful experimental design.
- Formulates the "attention bias" hypothesis to explain the phenomenon: the positional encoding and attention mechanisms of LLMs are inefficient when handling disordered graph descriptions.
- The analysis of task-to-order matching relationships provides highly practical guidelines for prompt engineering (e.g., using BFS for shortest paths, and DFS for Hamiltonian paths).
- In-depth sanity check experiments (comparing extreme orders) effectively rule out the alternative hypothesis of simple "answer leakage".
- Constructs the GraphDO dataset containing 8,500 cases across 6 task categories.

## Limitations & Future Work

- The impact of different graph structural typologies (sparse/dense, small-world, scale-free, etc.) remains unexplored.
- Lacks a rigid mathematical or theoretical description; the "attention bias" hypothesis remains qualitative.
- The graph scale is constrained by the LLMs' context window, and performance on large-scale graphs has not been evaluated.
- Does not investigate whether more advanced LLMs (such as GPT-4) still exhibit the same sensitivity to order.
- Generating traditional task data relies solely on Erdos-Renyi (ER) random graphs, which may not represent properties of real-world graphs.

## Related Work & Insights

- **LLM Reasoning**: Reasoning frameworks such as ReAct (Yao et al., 2022), Self-Refine (Madaan et al., 2023), Reflexion (Shinn et al., 2023), and LATS (Zhou et al., 2023).
- **LLM Graph Reasoning**: The NLGraph benchmark (Wang et al., 2023a) introduces Build-a-Graph and algorithmic prompting; Talk like a Graph (Fatemi et al., 2023) systematically evaluates graph encoding methods; GITA (Wei et al., 2024) integrates visual graphs into graph reasoning.
- **Prompt Engineering**: This work applies prompt engineering to the graph domain, focusing on how the sequential representation of graph structures affects prompt effectiveness.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to systematically investigate graph description order, addressing a unique perspective completely overlooked by prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experiments (6 tasks $\times$ 4 orders $\times$ 5 prompts $\times$ 6 models), yielding consistent and reliable conclusions.
- **Utility**: ⭐⭐⭐⭐ — Directly guides prompt design strategies for real-world graph reasoning tasks.
- **Clarity**: ⭐⭐⭐⭐ — Well-structured paper, progressively answering three research questions with rich visualizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2025\] Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)
- [\[ACL 2025\] A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive](theory_of_llm_sampling.md)
- [\[ACL 2025\] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](can_we_further_elicit_reasoning_in_llms_critic-guided_planning_with_retrieval-au.md)

</div>

<!-- RELATED:END -->
