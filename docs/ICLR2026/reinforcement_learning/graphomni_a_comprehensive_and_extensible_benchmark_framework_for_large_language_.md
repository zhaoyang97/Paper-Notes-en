---
title: >-
  [Paper Note] GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks
description: >-
  [ICLR 2026][Reinforcement Learning][Graph Reasoning] This paper proposes GraphOmni, a benchmark framework that systematically evaluates the graph-theoretic reasoning capabilities of 11 LLMs across 241K queries spanning 7…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Graph Reasoning"
  - "LLM Benchmarking"
  - "Serialization Formats"
  - "Prompting Strategies"
date: 2026-05-08
content_hash: 75cbb1fc61c5190c
---

# GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks

**Conference**: ICLR 2026
**arXiv**: [2504.12764](https://arxiv.org/abs/2504.12764)  
**Code**: [GitHub](https://github.com/GAI-Community/GraphOmni)  
**Area**: Reinforcement Learning
**Keywords**: Graph Reasoning, LLM Benchmarking, Serialization Formats, Prompting Strategies, Reinforcement Learning

## TL;DR

This paper proposes GraphOmni, a benchmark framework that systematically evaluates the graph-theoretic reasoning capabilities of 11 LLMs across 241K queries spanning 7 graph types × 7 serialization formats × 9 prompting strategies, reveals complex interaction effects among these three dimensions, and introduces an RL-guided combinatorial search method that achieves approximately 90% of optimal accuracy at roughly 25% of the evaluation cost.

## Background & Motivation

**Background**: LLMs have demonstrated remarkable performance on natural language tasks, and using LLMs to reason over graph-structured data serialized as text has emerged as a new research direction. Several benchmarks (NLGraph, GraphQA, GraphInstruct, etc.) have attempted to evaluate LLM graph reasoning, yet they exhibit significant blind spots in dimensional coverage.

**Limitations of Prior Work**: As shown in Table 1, existing benchmarks typically focus on only one dimension at a time—NLGraph covers 5 prompting strategies but only 1 graph type and 1 serialization format; GraphQA includes 7 graph types but relies solely on plain-text serialization; GraphInstruct supports 3 serialization formats but only 1 prompting strategy. This "single-dimension tuning" evaluation paradigm cannot reveal interaction effects across dimensions, nor can it attribute performance gains to model capability, text representation, or instruction design. Moreover, most works lack random baselines, which may lead to inflated capability claims on class-imbalanced tasks (e.g., cycle detection with a 50% random baseline).

**Key Challenge**: Graph reasoning performance is highly sensitive to how graphs are presented to LLMs—for the same model and task, different serialization–prompt combinations can cause accuracy fluctuations of up to 40%. Existing work cannot quantify these interaction effects, let alone guide optimal configuration selection in practice.

**Goal**: (1) Construct a large-scale benchmark covering the full Cartesian product of three dimensions; (2) systematically quantify interaction effects among graph type, serialization format, and prompting strategy; (3) provide an automated mechanism for searching optimal configurations.

**Key Insight**: The authors' core observation is that "the three dimensions are not independent—a serialization format effective for open-source models may in fact hurt closed-source models." Consequently, a full factorial evaluation is necessary to draw reliable conclusions.

**Core Idea**: Through a 7×7×9 full factorial evaluation combined with RL-guided combinatorial search, this work provides the first systematic characterization of the complete three-dimensional interaction landscape of "graph structure × text representation × prompting strategy" in LLM graph reasoning.

## Method

### Overall Architecture

GraphOmni comprises four pluggable modules: **graph-theoretic tasks** (6 classical graph problems) → **graph generators** (7 graph types) → **serialization formats** (7 text representations) → **prompting schemes** (9 instruction strategies). Each query corresponds to a specific combination of these four dimensions along with three difficulty levels (Easy: 5–10 nodes / Medium: 10–20 nodes / Hard: 20–30 nodes), yielding a total of 241,726 query instances. The evaluation covers 7 open-source models (Llama-3, Llama-3.1, Mistral, Phi-4, Qwen-2.5 7B, Qwen-2.5 72B, Qwen-3 8B) and 4 closed-source models (Claude-3.5, GPT-4o, GPT-4o-mini, Gemini-2.0, o4-mini). The framework is designed to be modular and extensible, allowing researchers to conveniently add new graph generators, serialization methods, or prompting strategies.

### Key Designs

1. **Capability Stratification Across 6 Graph-Theoretic Tasks**:

    - The 6 tasks are carefully selected to cover distinct reasoning capability dimensions: **local properties** include connectivity detection (testing local link understanding) and cycle detection (testing circular pattern recognition); **global properties** include diameter computation (requiring the maximum of all-pairs shortest path lengths) and BFS traversal (testing ordered sequence generation); **numerical/path** tasks include shortest path (path planning ability) and triangle counting (precise enumeration of triplets, the most challenging task).
    - Each task has a dedicated random baseline: for example, the connectivity baseline is approximately 67% (reflecting the prior that most real graphs are connected), cycle detection is 50%, and triangle counting is approximately 2%.

2. **Topological Diversity Across 7 Graph Generators**:

    - The generators cover two variants of Erdős–Rényi random graphs (ERM with fixed edge count / ERP with edge probability), two variants of bipartite ER graphs (BERM/BERP with bipartite constraints), Barabási–Albert scale-free graphs (BAG with preferential attachment / BAF as a forest variant), and Scale-Free graphs (SF with degree-weighted random attachment).
    - This design ensures comprehensive coverage from uniform random structures to power-law distributions to hierarchical tree-like structures. The authors further confirm via statistical tests in the appendix that these generators produce structurally distinct graphs within the 5–30 node range.

3. **Full Factorial Cross of Serialization Formats and Prompting Strategies**:

    - The 7 serialization formats range from compact to verbose: Edge Set/List (concise edge representation), Adjacency Set/List/Matrix (adjacency-based representations), GMoL/GMaL (structured markup languages). Token overhead varies substantially across formats (e.g., Adjacency Matrix incurs far more tokens than Edge Set on dense graphs).
    - The 9 prompting strategies are stratified by degree of guidance: from 0-Shot (no prompting) to Algorithm (explicitly specifying algorithmic steps such as BFS/Dijkstra), with intermediate strategies including CoT, K-Shot, Instruct, their zero-shot variants, and LTM (Least-to-Most).
    - The key contribution is computing the full Cartesian product rather than comparing only partial combinations, which enables heatmap-based interaction analysis.

### RL-Guided Optimal Configuration Search (RL-Opt)

Motivated by the finding that "no universally optimal configuration exists," the authors propose a practical automated configuration method. The core idea is to formulate the problem of "selecting the best (serialization format, prompting strategy, model) combination for a given task" as a sequential decision-making problem:

- **State space**: The currently selected factor combination (e.g., Adjacency List serialization already chosen, prompting strategy yet to be selected).
- **Action space**: At each step, selecting one option from the candidates in the corresponding dimension.
- **Reward**: Accuracy of the selected combination on a validation set.
- **Search space**: $E = 7 \times 9 \times 5 = 315$ combinations (5 open-source models).
- A DQN is used to learn the Q-function approximating the optimal policy, with $\varepsilon$-greedy exploration ($\varepsilon$ decaying from 1.0 to 0.01).
- The objective is to identify a near-globally-optimal configuration by exploring only approximately 25% of combinations ($\text{Cost} = k/K \approx 0.22$), achieving $\text{Rate} = acc^*/acc_{\max} \approx 0.90$.

### Training and Evaluation Strategy

Evaluation uses binary accuracy: qualitative tasks (connectivity, cycle detection) are judged via key-phrase matching; quantitative tasks (triangle counting, diameter) compare extracted numerical values against ground truth; multi-solution tasks (BFS traversal, shortest path) use rule-based validation functions to verify solution correctness. All few-shot experiments uniformly use $k=5$ examples.

## Key Experimental Results

### Main Results (Table 3)

Core results across all 6 tasks × 3 difficulty levels × 11 models (representative models shown):

| Task | Difficulty | o4-mini | Claude-3.5 | Qwen-2.5 72B | Qwen-3 8B | Llama-3.1 8B | Random Baseline |
|------|-----------|---------|------------|-------------|-----------|-------------|----------------|
| BFS Traversal | Easy | **95.46** | 91.42 | 71.41 | 65.87 | 18.69 | 0.00 |
| BFS Traversal | Hard | **32.45** | 26.80 | 22.03 | 29.53 | 0.63 | 0.00 |
| Connectivity | Easy | **98.23** | 98.38 | 90.24 | 97.17 | 79.53 | 67.49 |
| Cycle Detection | Easy | **97.97** | 82.56 | 74.02 | 90.30 | 55.49 | 50.00 |
| Diameter | Easy | **98.88** | 83.71 | 78.50 | 77.56 | 41.27 | 11.20 |
| Diameter | Hard | 34.61 | **56.70** | 29.59 | 39.83 | 18.63 | 3.72 |
| Shortest Path | Easy | 95.08 | **94.35** | 90.03 | 77.69 | 38.75 | — |
| Triangle Counting | Easy | **84.54** | 43.41 | 36.57 | 41.36 | 14.97 | 2.13 |
| Triangle Counting | Hard | **16.27** | 8.63 | 4.73 | 19.54 | 3.07 | 0.47 |

### RL-Opt Search Results (Table 4)

| Task | Difficulty | Avg. Search Cost (Cost) | Avg. Optimality Rate (Rate) |
|------|-----------|------------------------|----------------------------|
| BFS Traversal | Easy | 0.220 | 0.974 |
| Connectivity | Easy | 0.224 | 0.988 |
| Cycle Detection | Easy | 0.223 | 0.976 |
| Diameter | Easy | 0.226 | 0.973 |
| Shortest Path | Medium | 0.216 | 0.986 |
| Triangle Counting | Easy | 0.228 | 0.906 |
| Triangle Counting | Hard | 0.224 | **0.732** |

### Scaling vs. Reasoning Comparison (Table 11)

Using Qwen-2.5 7B as baseline, comparing parameter scaling (72B) against inference optimization (Qwen-3 8B):

| Task | Difficulty | Qwen-2.5 7B | Qwen-2.5 72B (Scaled) | Qwen-3 8B (Reasoning) |
|------|-----------|------------|----------------------|----------------------|
| BFS Traversal | Easy | 21.46 | **71.41** (+50) | 65.87 (+44) |
| BFS Traversal | Hard | 1.38 | 22.03 | **29.53** (+7.5 vs. 72B) |
| Diameter | Hard | 15.27 | 29.59 | **39.83** (+10 vs. 72B) |
| Triangle Counting | Hard | 3.62 | 4.73 | **19.54** (+15 vs. 72B) |
| Connectivity | Hard | 81.19 | 84.09 | **92.89** |

### Key Findings

1. **o4-mini Leads Overall but Substantial Room for Improvement Remains**: o4-mini ranks first on most tasks (Easy BFS: 95.46%, triangle counting: 84.54%), yet all models drop sharply under Hard difficulty—triangle counting Hard reaches only 16.27%—indicating that LLMs currently have limited capacity for combinatorial enumeration in graph reasoning.

2. **Scaling Raises the Floor; Reasoning Raises the Ceiling**: The Qwen family comparison reveals two distinct improvement pathways—the 72B model substantially improves easy tasks (BFS Easy +50%) but provides almost no gain on Hard triangle counting (+1.1%), while the 8B reasoning model surpasses the 72B model on the hardest configurations (triangle Hard: 19.54% vs. 4.73%), suggesting that combinatorial reasoning ability benefits more from architectural/training paradigm optimization than from parameter scaling alone.

3. **High Variance Across Serialization–Prompt Combinations**: For the same model and task, different combinations can cause accuracy differences exceeding 40%. For example, GPT-4o on Diameter Easy achieves 0.715 with the best combination (Algorithm + Adjacency List) but only 0.167 with the worst, exhibiting pronounced heatmap patterns.

4. **Open-Source and Closed-Source Models Respond Differently to Prompting Strategies**: Open-source models clearly benefit from example-based prompts such as K-Shot and Instruct; closed-source models exhibit more complex behavior—CoT-style prompts already perform well under 0-shot settings, and adding few-shot examples may interfere (e.g., Algorithm prompts improve with examples, but K-Shot itself provides limited benefit for closed-source models).

5. **LLMs Harbor Fundamental Misconceptions About Graph Concepts**: Error analysis reveals that LLMs confuse "graph diameter" with "longest path" rather than "the maximum of all-pairs shortest paths," and approximate triangle counts as $\lfloor n/3 \rfloor$ (i.e., number of nodes divided by 3), indicating that LLMs have not genuinely internalized graph-theoretic definitions.

6. **Validation on Real-World Graphs and NP-Hard Tasks**: Experiments on IMDB-MULTI and ogbg-molhiv real-world datasets (~3.6K samples) and NP-hard problems (Hamiltonian cycle / Max-Cut) show consistent relative rankings, though real graphs—being mostly sparse and connected—cause some tasks (e.g., connectivity) to be underestimated in difficulty.

## Highlights & Insights

- **Full Factorial Design** is the paper's most fundamental methodological contribution. Unlike prior work that fixes all dimensions but one, GraphOmni performs a complete Cartesian product evaluation, enabling heatmap-based interaction analysis—a design that is relatively rare in NLP benchmark research and merits adoption in other evaluation efforts.

- **The Practical Value of RL-Opt Outweighs Its Technical Complexity**: The core idea is straightforward (DQN + ε-greedy search over 315 combinations), yet it addresses a highly practical question—"given a new task or model, how can one find the best configuration at minimal cost?" Achieving 90% optimality at 25% cost is highly attractive from an engineering perspective.

- **The Scaling vs. Reasoning Controlled Experiment** (Table 11) provides a clean comparison—the same model family scaled from 7B to 72B versus a reasoning-optimized model at the same parameter count—clearly demonstrating the complementary nature of these two improvement pathways, with direct implications for model selection decisions.

- **The Random Baseline Design**, though seemingly straightforward, is critically important: the 50% baseline for cycle detection exposes that some models (e.g., Llama-3.1 at 55.49%) are only marginally better than random guessing.

## Limitations & Future Work

- **Small Graph Scale (Maximum 30 Nodes)**: Although the appendix extends experiments to 50 nodes, only graphs with a few dozen nodes can be encoded within the same token budget. Real-world graphs routinely involve thousands of nodes, so the current evaluation effectively tests "precise reasoning on small graphs" rather than "approximate reasoning on large graphs."
- **Exclusively Synthetic Graph Generators**: Although validation experiments on IMDB/ogbg are included, all main experiments use synthetic graphs. Synthetic graph distributions are limited and lack real-network characteristics such as community structure and small-world properties.
- **No Visual Graph Modality**: All inputs are text-serialized. As VLMs continue to develop, visual representations of graphs (node-edge rendered images) may constitute a more natural input modality, yet this direction is entirely unexplored.
- **Search Space Assumptions in RL-Opt**: The method assumes zero prior knowledge, starting the search from scratch. In practice, partial prior knowledge often exists (e.g., "CoT generally outperforms 0-Shot"), and incorporating such priors could further reduce search cost.
- **Open-Source Models Concentrated in the 7B–14B Range**: Although Qwen-2.5 72B is included, large-scale open-source models such as Llama-3.1 70B and Mistral Large are absent from the comparison.
- **Extensibility to Dynamic and Heterogeneous Graphs**: The current framework considers only static homogeneous undirected graphs and could be extended to temporal graphs, directed graphs, heterogeneous information networks, and other more complex settings.

## Related Work & Insights

- **vs. NLGraph**: NLGraph covers 5 prompting strategies but only 1 graph type and 1 serialization format, with only 5,902 samples (2.4% of GraphOmni). NLGraph's finding that "Algorithm prompting is best" is refined by GraphOmni to "indeed best for open-source models, but task-dependent for closed-source models."
- **vs. GraphQA**: GraphQA includes 7 graph types but relies solely on plain-text description (1 serialization format), precluding analysis of serialization format effects. Moreover, the absence of random baselines means reported cycle detection accuracy may be overestimated.
- **vs. GraphArena**: GraphArena contains 10K samples but uses only 1 serialization format and 1 prompting strategy. GraphOmni's multi-combination evaluation on identical tasks reveals the unreliability of single-configuration results as reported by GraphArena.
- **Insights**: When handling graph-structured data in LLM agent systems, the serialization–prompt combination should be tuned for each specific task rather than relying on a fixed template. The RL-Opt paradigm is generalizable to the broader problem of "multi-hyperparameter LLM configuration search."

## Rating

- Novelty: ⭐⭐⭐⭐ The full factorial evaluation paradigm is novel, though RL-Opt itself is methodologically straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 241K queries × 11 models × 3 difficulty levels, with appendix extensions to large graphs, real-world graphs, and NP-hard problems.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and the 115-page appendix is exceptionally detailed, though the main text contains some data redundancy.
- Value: ⭐⭐⭐⭐ Provides the most comprehensive reference for LLM graph reasoning evaluation; RL-Opt offers practical engineering utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](../../ACL2026/reinforcement_learning/table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)

</div>

<!-- RELATED:END -->
