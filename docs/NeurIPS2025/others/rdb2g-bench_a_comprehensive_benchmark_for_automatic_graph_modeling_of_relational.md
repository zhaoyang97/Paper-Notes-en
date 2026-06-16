---
title: >-
  [Paper Note] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases
description: >-
  [NeurIPS 2025][relational databases] This paper proposes RDB2G-Bench — the first benchmark framework for evaluating relational-database-to-graph modeling methods, comprising 5 real-world RDBs, 12 prediction tasks…
tags:
  - "NeurIPS 2025"
  - "relational databases"
  - "graph modeling"
  - "benchmark"
  - "graph neural networks"
  - "automatic modeling"
date: 2026-05-08
content_hash: 21790e3b8be9504c
---

# RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases

**Conference**: NeurIPS 2025
**arXiv**: [2506.01360](https://arxiv.org/abs/2506.01360)  
**Code**: [github.com/chlehdwon/RDB2G-Bench](https://github.com/chlehdwon/RDB2G-Bench)  
**Area**: LLM Evaluation
**Keywords**: relational databases, graph modeling, benchmark, graph neural networks, automatic modeling

## TL;DR

This paper proposes RDB2G-Bench — the first benchmark framework for evaluating relational-database-to-graph modeling methods, comprising 5 real-world RDBs, 12 prediction tasks, approximately 50,000 precomputed graph model–performance pairs, and a systematic comparison of 10 automatic graph modeling approaches.

## Background & Motivation

Relational databases (RDBs) are widely used in finance, healthcare, e-commerce, and other domains. Graph-based machine learning methods — which model rows of RDB tables as nodes and foreign-key relationships as edges and then apply GNNs — have demonstrated strong performance on RDB prediction tasks.

However, the design space for RDB-to-graph modeling is **enormous**:
- Table rows can be modeled as either nodes or edges
- Certain tables and foreign-key relationships can be selectively included or excluded
- Different modeling choices lead to substantial GNN performance differences (up to 10%)

Most existing work adopts fixed heuristic rules (e.g., modeling all rows as nodes), which are not necessarily optimal. Research on automatically searching for optimal graph models faces a fundamental obstacle: **evaluation cost is prohibitive** — training a GNN is required for every graph model evaluated.

The core motivation of this paper is to construct a precomputed benchmark dataset that enables researchers to evaluate graph modeling strategies without repeatedly training GNNs, thereby accelerating progress in this area.

## Method

### Overall Architecture

The construction and usage pipeline of RDB2G-Bench:
1. **Graph model design space definition**: selecting which tables and foreign keys to include (Step 1) + deciding how to represent rows (node/edge) (Step 2)
2. **Graph model generation**: enumerating valid graph models subject to defined constraints
3. **Performance metric collection**: training a GNN for each graph model and recording training/validation/test performance, runtime, and parameter count
4. **Benchmark evaluation**: rapid evaluation of 10 modeling methods against the precomputed dataset

### Key Designs

**Constraints on the Graph Model Space**

A valid graph model must satisfy:
1. It must include the task table (the table on which the prediction task is defined)
2. All selected tables must be connected to the task table via paths (with path length not exceeding the number of GNN layers)
3. Tables whose rows are modeled as edges must have exactly 2 foreign keys, and their primary keys must not be referenced by other tables

**Ten Automatic Modeling Methods**

Organized into three categories:

*Heuristic methods:*
- **Random**: randomly samples graph models and selects the best
- **AR2N (All-Rows-to-Nodes)**: models all rows as nodes and includes all tables and foreign keys — the most commonly used baseline

*Action-based search algorithms (4 actions: add/remove FK edges, row→edge/edge→row):*
- **Greedy Forward (GF)**: starts from the task table and greedily adds components
- **Greedy Backward (GB)**: starts from the complete graph and greedily prunes
- **Greedy Local (GL)**: starts from a random graph and performs greedy local search
- **Evolutionary Algorithm (EA)**: evolutionary strategy
- **Bayesian Optimization (BO)**: based on the BANANAS algorithm
- **Reinforcement Learning (RL)**: RNN-based policy gradient

*LLM-based methods:*
- **LLM**: uses Claude Sonnet-3.5 to directly generate action sequences
- **LLM-CoT**: incorporates Chain-of-Thought prompting

### Loss & Training

GNNs used are Heterogeneous GraphSAGE (for classification/regression tasks) and ID-GNN (for recommendation tasks). The training protocol follows RelBench, fixing the number of epochs at 20 and tuning only the learning rate. Each experiment is repeated 5–15 times to ensure reliability.

## Key Experimental Results

### Main Results

**Dataset Overview** (Figure 2a)

| RDB | Task | Task Type | # Tables | # Graph Models | Best | AR2N | Worst |
|-----|------|-----------|----------|----------------|------|------|-------|
| rel-avito | user-clicks | Classification (AUC%) | 8 | 944 | 67.93 | 64.66 | 60.89 |
| rel-avito | user-visits | Classification (AUC%) | 8 | 944 | 66.33 | 65.97 | 59.83 |
| rel-avito | ad-ctr | Regression (MAE↓) | 8 | 1304 | 0.039 | 0.040 | 0.044 |
| rel-event | user-repeat | Classification (AUC%) | 5 | 214 | 82.29 | 77.65 | 63.96 |
| rel-event | user-ignore | Classification (AUC%) | 5 | 214 | 82.82 | 82.22 | 74.29 |
| rel-f1 | driver-dnf | Classification (AUC%) | 9 | 722 | 74.56 | 73.14 | 67.40 |
| rel-f1 | driver-top3 | Classification (AUC%) | 9 | 722 | 81.88 | 78.11 | 75.37 |
| rel-f1 | driver-position | Regression (MAE↓) | 9 | 722 | 3.831 | 3.913 | 4.171 |
| rel-stack | post-related | Recommendation (MAP%) | 7 | 7979 | 12.04 | 10.82 | 0.006 |
| rel-trial | study-outcome | Classification (AUC%) | 15 | 36863 | 70.91 | 68.09 | 62.85 |

**Performance of Ten Methods on rel-f1** (Table 1, driver-top3 task AUC-ROC%)

| Method | Budget 1% | 2% | 3% | 5% |
|--------|----------|-----|-----|-----|
| Best | 81.88 | 81.88 | 81.88 | 81.88 |
| AR2N | 78.11 | 78.11 | 78.11 | 78.11 |
| GF | **81.88** | **81.88** | **81.88** | **81.88** |
| GB | 80.15 | 80.56 | 80.56 | 80.56 |
| BO | 79.42 | 79.80 | 80.13 | 80.35 |
| RL | 79.04 | 79.39 | 79.44 | 80.08 |
| LLM | 80.34 | 80.54 | 80.54 | 80.61 |
| Random | 79.63 | 80.17 | 80.43 | 80.60 |

### Ablation Study

**Five Key Observations**

| Observation | Core Finding |
|-------------|-------------|
| Obs 1 | The optimal graph model improves over AR2N by up to 10%, and is typically smaller and faster |
| Obs 2 | The benefit of modeling rows as edges (Row2Edge) varies by task; different tasks on the same RDB may yield opposite conclusions |
| Obs 3 | High-performing graph models share common substructures (e.g., specific foreign-key relationships and edge modeling strategies) |
| Obs 4 | Different tasks require different graph models; cross-task Spearman correlation is generally below 0.4 |
| Obs 5 | The effectiveness of graph models generalizes well across GNN architectures (Spearman correlation > 0.7–0.8) |

**Efficiency Gains**

| Evaluation Mode | Total Time | Speedup |
|-----------------|-----------|---------|
| On-the-fly (training GNN for each evaluation) | 850+ hours | 1× |
| RDB2G-Bench (precomputed lookup) | 2.20 hours | **389×** |

### Key Findings

1. **"More data is not always better"**: including all tables and foreign keys (AR2N) is generally not the optimal strategy. Selectively using fewer but more relevant tables can yield better performance and efficiency.
2. **Greedy Forward achieves the best performance under small budgets**: on the rel-f1 driver-top3 task, it finds the global optimum (81.88%) with only 1% of the budget.
3. **Complex methods (RL, EA) are unstable under small budgets**: they require more exploration to converge.
4. **LLM-based methods excel at short-term reasoning but struggle with long-term planning**: performance improves rapidly in early iterations but gains diminish as the budget increases.
5. **Simple methods (Greedy + Random) are competitive with complex methods**: indicating substantial room for improvement in this area.

## Highlights & Insights

- **Pioneering contribution**: the first benchmark specifically designed to evaluate RDB-to-graph modeling strategies, filling an important gap in the literature.
- **High practical value**: the 389× speedup makes large-scale evaluation feasible, and the precomputed dataset is directly usable.
- **Findings carry practical guidance**:
    - Avoid blindly applying AR2N — select graph modeling strategies tailored to the task
    - Greedy Forward is a practical default choice
    - Graph model effectiveness generalizes across GNNs, suggesting that graph modeling choices matter more than GNN architecture selection
- **Systematic experimental design**: 5 real-world RDBs × 12 tasks × 50k graph models provide sufficient scale.

## Limitations & Future Work

1. The graph model space considers only two dimensions — "node vs. edge" and "include vs. exclude" — without addressing hyperedge modeling.
2. Only the 5 RDBs provided by RelBench are used, limiting coverage of domains and schema complexity.
3. The precomputed dataset is tied to specific GNN architectures and training configurations; other settings may require rebuilding.
4. LLM-based methods are evaluated only with Claude Sonnet-3.5, without comparison to other models such as GPT-4.
5. The theoretical relationship between graph model design and downstream task semantics is not explored.

## Related Work & Insights

- **RelBench** (Fey et al. 2024): provides RDBs and prediction tasks, but all methods use the same graph modeling. RDB2G-Bench builds on this foundation to evaluate different graph modeling strategies.
- **AutoG** (Chen et al. 2025): leverages LLMs to explore effective graph models, but at high evaluation cost. RDB2G-Bench enables rapid evaluation for such approaches.
- **RDBench, 4DBInfer**: related benchmarks focusing on different ML methods under fixed graph modeling.
- Insight: the choice of graph modeling may have a greater impact on GNN performance than the choice of GNN architecture, warranting greater research attention.

## Rating

- **Novelty**: ★★★★☆ (first benchmark specifically targeting graph modeling strategy evaluation; novel problem formulation)
- **Experimental Scale**: ★★★★★ (50k graph models, 12 tasks, 10 methods, 10,400 GPU hours to construct)
- **Practicality**: ★★★★★ (precomputed dataset, open-source code, 389× speedup)
- **Clarity**: ★★★★☆ (well-structured, with 5 observations progressively developed and intuitive figures)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](face_faithful_automatic_concept_extraction.md)
- [\[NeurIPS 2025\] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)
- [\[ICML 2026\] iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](../../ICML2026/others/iworld-bench_a_benchmark_for_interactive_world_models_with_a_unified_action_gene.md)
- [\[NeurIPS 2025\] On Topological Descriptors for Graph Products](on_topological_descriptors_for_graph_products.md)
- [\[NeurIPS 2025\] Graph Alignment via Birkhoff Relaxation](graph_alignment_via_birkhoff_relaxation.md)

</div>

<!-- RELATED:END -->
