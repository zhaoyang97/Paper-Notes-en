---
title: >-
  [Paper Note] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks
description: >-
  [ACL 2026][Graph Learning][large graph reasoning] Existing LLM graph reasoning benchmarks are limited to small graphs of 20–50 nodes and require full graph visibility. This paper compresses real-world graphs of up to 2.3…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "large graph reasoning"
  - "random walk"
  - "LLM benchmark"
  - "graph property estimation"
  - "partial access"
date: 2026-05-08
content_hash: fd668279912e1d7b
---

# Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks

**Conference**: ACL 2026  
**arXiv**: [2605.01484](https://arxiv.org/abs/2605.01484)  
**Code**: https://zenodo.org/records/19632942  
**Area**: Graph Learning / LLM Evaluation / Estimation Algorithms  
**Keywords**: large graph reasoning, random walk, LLM benchmark, graph property estimation, partial access

## TL;DR
Existing LLM graph reasoning benchmarks are limited to small graphs of 20–50 nodes and require full graph visibility. This paper compresses real-world graphs of up to 2.39M nodes into prompts using "random walk statistics." It proposes EstGraph to evaluate LLM performance on four estimation tasks: node/edge count, community count, graph structure, and influential nodes. The study finds that LLMs achieve < 20% relative error on medium-scale graphs and can effectively identify graph structures.

## Background & Motivation

**Background**: Almost all existing LLM graph reasoning benchmarks, such as NLGraph, GraphQA, GraphArena, and GraphPattern, encode the entire graph as an edgelist or adjacency list within the prompt. These benchmarks focus on "algorithm execution" problems like shortest paths, connectivity, and Hamiltonian paths.

**Limitations of Prior Work**: (1) **Context window constraints**—Typical benchmarks are capped at 20–50 nodes, which is 4–6 orders of magnitude smaller than real-world graphs. (2) **Invalid full visibility assumption**—Graphs in social networks, Web, or P2P systems are often only partially accessible via APIs, making it impossible to retrieve the full graph at once. (3) **LLM collapse on large graphs**—Empirical tests on simple local tasks (e.g., edgelist to adjacency list conversion) show that as the number of nodes increases, LLMs suffer from omitted edges or hallucinations (Fig. 1). (4) **Misaligned task focus**—Large-scale graph analysis is concerned with global statistics like community structure, degree distribution, and influential nodes, rather than specific algorithm execution.

**Key Challenge**: As graph scale grows, the number of tokens required for encoding increases linearly, hitting the context window limit. Even if forced into a prompt, LLMs struggle to maintain a "globally consistent view." Meanwhile, traditional graph estimation algorithms (e.g., MH-walk, max-degree walk, return-time) require either unbiased sampling (unfeasible via many APIs) or global information like the maximum degree.

**Goal**: (1) Discard the "full visibility" assumption and introduce a new "partial access via random walks" setting. (2) Design four estimation tasks for large graphs covering size, community, structure, and influential nodes. (3) Construct task-specific "walk-statistics prompts" where prompt length is independent of graph scale. (4) Systematically compare LLMs against classical estimators on synthetic (up to 100k nodes) and real-world graphs (up to 2.39M nodes).

**Key Insight**: The authors noted that classical graph estimation literature (e.g., capture-recapture, Chapman estimator) naturally infers global properties from local random walks. By compressing walk results into statistics (degree distribution, revisit rates, node co-occurrence) before feeding them to LLMs, the models can use graph theory priors for reasoning. This bypasses the context limit while leveraging LLMs' world knowledge.

**Core Idea**: Replace "full graph encoding" with "task-specific random-walk statistics" in prompts, treating the LLM as an "estimator with graph theory commonsense" rather than an "algorithm executor."

## Method

EstGraph abstracts each estimation task as follows: sampling multiple srw/MH random walks from a large graph $G=(V, E)$ $\rightarrow$ computing statistics derived from the walks (node intersections, degree distribution histograms, revisit rates, etc.) $\rightarrow$ populating a prompt template with graph theory reasoning hints $\rightarrow$ LLM outputs scalar estimates (node/community counts) or rankings (top-k influential nodes).

### Overall Architecture

The four task categories share the same pipeline: ① Select a walking strategy (srw or MH) and budget (steps, starting points). ② Execute walks on the graph and record (node id, degree). ③ Convert results into statistics-only prompts (including graph theory prior hints). ④ LLM outputs estimates. ⑤ Comparison with classical estimators.

### Key Designs

1.  **Statistics-only Prompt: Reducing prompt length from $\Theta(n+m)$ to $\Theta(\log n)$**:

    - **Function**: Avoids cramming nodes/edges into the prompt, allowing evaluation to scale to million-node graphs.
    - **Mechanism**: For node/edge estimation, the Chapman estimator $\hat{N}=\frac{(|\mathcal{S}_1|+1)(|\mathcal{S}_2|+1)}{|\mathcal{C}|}-1$ is used (where $\mathcal{S}_1, \mathcal{S}_2$ are sets of nodes sampled from two independent MH walks and $\mathcal{C}$ is their intersection). Only aggregate values like $|\mathcal{S}_1|, |\mathcal{S}_2|, |\mathcal{C}|, \bar{d}$ are included in the prompt. Edges are inferred via $\hat{M}=\bar{d}\hat{N}/2$. For structure recognition, only degree histograms of visited nodes are used. For community estimation, node revisits and jump patterns in walk subgraphs are used.
    - **Design Motivation**: Encoding real graphs (ego-Twitter, twitch-gamers, email-EuAll, as-skitter, wiki-Talk) as edgelists requires $10^5$–$10^7$ tokens, exceeding any LLM context. Statistics-only prompts compress this to a few hundred tokens (Fig. 4), achieving a **reduction of up to 559×** and decoupling length from graph scale.

2.  **Four-Task Benchmark Covering Core Large Graph Estimation Needs**:

    - **Function**: Decomposes "large graph analysis" into four complementary tasks with ground truth and classical baselines.
    - **Mechanism**: ① **Size estimation** (nodes + edges)—Synthetic (BA/ER/GRP) + 5 SNAP real graphs, compared against uniform / MH / max-degree / return-walk. ② **Community count**—20 LFR synthetic graphs, compared against Louvain / Greedy / Label Propagation. ③ **Graph structure recognition**—4-way classification of synthetic graphs (BA / ER / LFR / Grid). ④ **Influential node ranking**—Predicting top-20 nodes for Betweenness / Closeness / PageRank on LFR graphs, evaluated via Precision@20.
    - **Design Motivation**: These tasks correspond to scale $\rightarrow$ modularity $\rightarrow$ global topology $\rightarrow$ node importance, which are the most common problems in real-world large graph analysis. Each has mature baselines for fair comparison with LLMs.

3.  **MH vs. srw Dual Sampling Protocol + Real-World Availability Constraints**:

    - **Function**: Reports results for both "ideal unbiased sampling" and "practically feasible sampling" to reveal deployment trade-offs.
    - **Mechanism**: MH-walk (with burn-in) is the gold standard for unbiased estimation but requires rejecting samples and partial global info. Simple random walk (srw) transitions uniformly among neighbors and can be implemented purely via APIs but has degree bias. The authors report both, marking methods requiring unbiased sampling as "real-world unavailable" with $\dagger$.
    - **Design Motivation**: Previous works often reported optimistic results using only MH. This work explicitly highlights the constraint that "only srw is available in real deployment," making conclusions more actionable. For example, on BA real graphs, LLM performance with srw is only 9% worse than with MH, proving feasibility for deployment.

### Loss & Training

This is a pure evaluation study with no training: all LLMs (gemini-2.5-pro, o3, sonnet-4, deepseek-v3.1) are accessed via zero-shot API prompting. Walk hyperparameters (steps, starting points, burn-in) are fixed. Results are reported as the median/mean/std of 5 independent walk sets per experiment.

## Key Experimental Results

### Main Results

Node count estimation for synthetic Large graphs (10k–100k nodes), median relative error %:

| Graph Type | uniform† | MH† | o3 (MH)† | o3 (srw) | gemini-2.5-pro (srw) | deepseek-v3.1 (srw) |
|------------|----------|------|----------|----------|----------------------|---------------------|
| BA Large   | 0.60     | 12.17| 13.08    | 25.47    | 52.56                | 26.97               |
| ER Large   | 0.77     | 2.39 | 3.41     | 5.57     | 8.08                 | 6.87                |
| GRP Large  | 0.56     | 2.51 | 2.81     | 4.94     | 16.84                | 4.94                |

Node count estimation for real-world large graphs (million-node scale), median relative error %:

| Dataset (Size) | MH†    | gemini-2.5-pro (MH) | o3 (srw) | deepseek-v3.1 (srw) |
|----------------|--------|---------------------|----------|---------------------|
| ego-Twitter    | 51.02  | 66.04               | 51.85    | 51.83               |
| twitch-gamers  | 59.62  | 36.64               | 52.41    | 52.41               |
| email-EuAll    | 136.20 | 19.06               | 28.84    | 29.99               |
| as-skitter     | 75.21  | 30.01               | 49.84    | 50.21               |
| wiki-Talk      | 181.04 | 64.37               | 33.03    | 34.38               |

On several real graphs, LLM performance **surpasses the classical MH baseline**. Gemini-2.5-pro reduced error from 136% to 19% on email-EuAll.

Structure recognition accuracy (4 classes):

| Model           | BA     | ER     | LFR    | Grid   |
|-----------------|--------|--------|--------|--------|
| gemini-2.5-pro  | 33.3%  | 73.3%  | 80.0%  | 100%   |
| o3              | 93.3%  | 93.3%  | 26.7%  | 100%   |
| sonnet-4        | 100%   | 13.3%  | 6.7%   | 100%   |
| DeepSeek-V3.1   | 80.0%  | 66.67% | 66.67% | 100%   |

Influential node ranking Precision@20 (%):

| Model           | Betweenness     | Closeness       | PageRank        |
|-----------------|-----------------|-----------------|-----------------|
| gemini-2.5-pro  | 6.5 ± 7.4       | 9.3 ± 8.4       | 27.5 ± 18.4     |
| o3              | **31.5 ± 14.2** | **35.0 ± 11.7** | **81.0 ± 19.9** |
| sonnet-4        | 15.3 ± 10.1     | 23.8 ± 16.1     | 42.8 ± 28.4     |
| DeepSeek-V3.1   | 23.0 ± 13.6     | 20.0 ± 16.4     | 28.5 ± 23.0     |

### Ablation Study

| Dimension                       | Observation                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| srw vs MH (BA Large)            | srw error is 78% higher than MH (synthetic) but only 9% higher (real-world).|
| LLM vs Baseline (BA Large, MH)  | o3 13% vs uniform 0.6% (uniform requires full nodelist and is unavailable). |
| Walk budget (Fig. 6)            | budget ↑ → size estimate error monotonically decreases.                     |
| Multiple runs (Median)          | Median < Mean significantly, suggesting long-tail over-estimation.          |
| Community count (Range 5–12)    | LLM Mean Absolute Error 1.9–2.6; Louvain ≈ 0.                               |
| Token compression ratio         | Statistics prompt vs edgelist: up to 559× reduction.                        |

### Key Findings

- **Small vs. Large Graph Differences**: On synthetic medium-scale graphs, LLM median error is < 20%, comparable to MH baselines. On real-world large graphs, LLMs are more stable, with gemini/o3 showing lower errors than MH on million-node graphs.
- **srw is Sufficient**: For actual deployment, the srw route is feasible as LLM errors are only a few percentage points higher than MH, avoiding the need for burn-in or rejection sampling.
- **Median ≠ Mean**: Occasional extreme over-estimation by LLMs inflates the mean (e.g., deepseek-v3.1 srw BA Large mean 35.38 vs median 26.97), implying a need for multiple runs and taking the median as a robust estimate.
- **Significant Model Disparity**: o3 is the strongest overall in structure recognition and influential node ranking; gemini-2.5-pro is most stable for real-world size estimation; sonnet-4 tends to classify any graph as BA.
- **PageRank > Betweenness/Closeness**: PageRank naturally aligns with srw visitation frequency, allowing LLMs to approximate it via degree distribution. Shortest-path metrics are difficult to infer from walks.
- **Massive Token Compression**: For million-node graphs like wiki-Talk, edgelist encoding requires tens of millions of tokens, while statistics prompts require only hundreds (reduction ≥ 500×).

## Highlights & Insights

- **"Estimation tasks" are the correct entry point for large graphs + LLMs**: Precise algorithm execution is meaningful for small graphs but collapses on large ones. Estimation tasks tolerate approximation, which naturally matches the approximate reasoning capabilities of LLMs.
- **Prompt = Task-Specific Statistics**: The authors use graph theory priors to design prompts following a "statistical summary + task hint" paradigm. This decouples prompt length from graph scale, offering a blueprint for other scenarios requiring large data processing (log analysis, streaming, massive tables).
- **Explicit Deployment Constraints (†)**: Distinguishing which baselines are available in real API settings vs. those that are not is a model for evaluation design. It avoids unfair comparisons and yields more credible results for real-world utility.
- **"Implicit Regularization" in LLMs for Large Graphs**: When data is noisy, LLMs' world knowledge can lead to more stable estimates than unsupervised estimators, as seen in wiki-Talk where LLM error was significantly lower than MH.
- **Transferable Heuristics**: The "sample $\rightarrow$ statistic $\rightarrow$ LLM reasoning" pipeline can be adapted for approximate queries on tables/logs/SQL. Replacing graph priors with medical or financial priors could create similar partial-access benchmarks in other domains.

## Limitations & Future Work

- **Narrow Task Coverage**: Only 4 estimation tasks were covered, excluding path finding, flow, link prediction, or anomaly detection.
- **Insufficient Hyperparameter Ablation**: Due to the cost of reasoning LLM APIs, a full grid ablation on walk steps, burn-in, and starting points was not conducted.
- **High Variance in LLM Outputs**: The gap between mean and median is often 2–10x, meaning a single estimate is unreliable; models also do not currently output confidence intervals.
- **Weak on BA Graphs**: LLMs generally underperform compared to MH on BA graphs with extremely long-tailed degree distributions, indicating difficulties with highly skewed statistics.
- **Influential Node Ranking (Betweenness) < 35%**: Shortest-path information cannot be directly inferred from random walks, necessitating more sophisticated approximation algorithms or LLM collaboration.
- **No Test on Text-Attributed Graphs**: Real graphs often contain node text (e.g., user profiles). This multi-modal advantage of LLMs was not evaluated.
- **No Training Data Provided**: The evaluation was pure zero-shot/reasoning; the potential of fine-tuning a "graph-stat-LLM" remains unexplored.

## Related Work & Insights

- **vs. NLGraph / GraphQA / GraphArena**: These assume full visibility, $\le$ 50 nodes, and ask for algorithm execution. EstGraph assumes partial access, up to 2.39M nodes, and asks for estimation.
- **vs. GraphPattern (Dai et al. 2025)**: GraphPattern measures motif recognition on local/small graphs; EstGraph measures global properties on large graphs.
- **vs. Talk like a Graph (Fatemi et al. 2024)**: Fatemi systematically compared encoding styles on small graph tasks; this work abandons "full graph encoding" for "sampling statistics," providing a new paradigm for large graphs.
- **vs. Classical Capture-Recapture / MH / Max-degree**: This work uses these methods' outputs as input features for the LLM, making them nested rather than strictly competing.
- **vs. Walk&Retrieve (Böckling et al. 2025)**: Walk&Retrieve uses knowledge graph walks for RAG; EstGraph uses walks for property estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ First to expand LLM graph evaluation from "full visibility small graphs" to "partial access million-node graphs" using a systematic "sampling statistics + LLM" benchmark.
- Experimental Thoroughness: ⭐⭐⭐ Covers 4 tasks across 4 LLMs on 8 datasets, but lacks full hyperparameter ablation and some task baselines.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation and comparison tables are clear and powerful; some prompt details require checking the appendix.
- Value: ⭐⭐⭐⭐ Provides a new dataset, practically applicable prompt patterns, and actionable insights for practitioners working with large-scale graphs and LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](../../AAAI2026/graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ICLR 2026\] GraphUniverse: Synthetic Graph Generation for Evaluating Inductive Generalization](../../ICLR2026/graph_learning/graphuniverse_synthetic_graph_generation_for_evaluating_inductive_generalization.md)

</div>

<!-- RELATED:END -->
