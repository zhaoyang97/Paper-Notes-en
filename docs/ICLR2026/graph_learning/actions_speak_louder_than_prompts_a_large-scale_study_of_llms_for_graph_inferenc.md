---
title: >-
  [Paper Note] Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference
description: >-
  [ICLR2026][Graph Learning][LLM-Graph Interaction] This paper presents a large-scale, controlled empirical study systematically comparing three "interaction modes" for LLMs to process textual graphs: direct prompting, ReAct-style tool calling, and Graph-as-Code (where the LLM writes code to query the graph). The study finds that **allowing the LLM to write code for graph operations** (rather than stuffing the graph into the prompt) is overall superior for node classification…
tags:
  - "ICLR2026"
  - "Graph Learning"
  - "LLM-Graph Interaction"
  - "Node Classification"
  - "Graph-as-Code"
  - "ReAct Tool-use"
  - "Dependency Analysis"
date: 2026-05-08
content_hash: 1cb9f7123aced59b
---

# Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=MgJUj9Sk3C](https://openreview.net/forum?id=MgJUj9Sk3C)  
**Code**: To be confirmed  
**Area**: Graph Learning / LLM Reasoning  
**Keywords**: LLM-Graph Interaction, Node Classification, Graph-as-Code, ReAct Tool-use, Dependency Analysis

## TL;DR
This paper presents a large-scale, controlled empirical study systematically comparing three "interaction modes" for LLMs to process textual graphs: direct prompting, ReAct-style tool calling, and Graph-as-Code (where the LLM writes code to query the graph). The study finds that **allowing the LLM to write code for graph operations** (rather than stuffing the graph into the prompt) is overall superior for node classification, especially on dense graphs with long text or high degrees, as it enables adaptive switching between structural, feature, and label signals.

## Background & Motivation
**Background**: In "text-rich graph" scenarios like fraud detection, recommendation, and information retrieval, node classification is a core task. While Graph Neural Networks (GNNs) have been the dominant paradigm, they often require task-specific training and lack cross-domain transferability. LLMs, with their world knowledge and text understanding, are emerging as alternatives by "verbalizing" the graph (serializing neighbors, edges, and labels into text) and placing it into prompts for inference.

**Limitations of Prior Work**: Most existing research reports performance only for specific domains or tasks, lacking a **principled understanding** of how LLMs should interact with graph data. It remains unclear when LLMs rely on features, structure, or labels. Furthermore, nearly all work relies on a single interaction mode—serializing the graph into the prompt—which **rapidly exhausts token budgets** when node degrees are high or text is long, preventing the inclusion of sufficient neighborhood information.

**Key Challenge**: Graph information (structure + features + labels) is highly structured and potentially massive, whereas prompting flattens it into a linear, inefficient, and lossy text sequence. Worse, mechanistic analyses (Guan et al. 2025) suggest LLMs under prompting often "imitate the prompt format" rather than executing true graph computation. In other words, the **interaction medium itself** may determine success more than model size or prompt phrasing, yet this dimension has never been systematically studied.

**Goal**: The authors aim to "factorize" all key variables affecting LLM-graph reasoning through a controlled large-scale evaluation to answer two questions: (1) which interaction mode is strongest and under what conditions, and (2) how different modes rely on structure, features, and labels.

**Key Insight**: The core observation is that "Actions Speak Louder than Prompts." Instead of forcing the model to passively read the entire graph in context, it is better to provide the model with **active agency**: allowing it to initiate tool calls or write code to query the graph on demand. Higher autonomy allows the model to tailor retrieval and reasoning to the structural characteristics of each instance.

**Core Idea**: Replace "stuffing the graph into prompts" with "letting the LLM write code to operate on the graph (Graph-as-Code)." Using controlled variables and dependency ablation experiments, the authors prove that the latter is more robust and capable of adaptive information dependency.

## Method

### Overall Architecture
The paper does not propose a new model but builds a **controlled evaluation framework**. The node classification task is fixed while systematically varying multiple axes to isolate the impact of each factor.

Task Formulation: Given a graph $G=(V,E,X,Y)$, partially known labels $Y_K$, graph structure (adjacency matrix $A$ or edge set $E$), and text features $X$ for all nodes, the goal is to predict labels $Y_Q$ for a query set $Q$. The three interaction modes are abstracted as functions $\phi_{\text{prompt}}, \phi_{\text{tool}}, \phi_{\text{code}}: \mathcal{T} \times \mathcal{T}^N \times \{0,1\}^{N\times N} \to \mathcal{T}$, which encode the "dialogue history + node features + graph structure" into a finite token sequence for an underlying $\text{LLM}_\theta$. The difference lies **only in the encoding and interaction method**, keeping the model constant to isolate the "interaction medium" variable.

The evaluation spans six axes: (1) **Interaction Mode** (prompting / tool-use / code), (2) **Dataset Domain** (citation, web links, e-commerce, social), (3) **Structural Properties** (homophilic vs. heterophilic), (4) **Feature Length** (short vs. long text), (5) **Model Scale** (Llama to GPT-5), and (6) **Reasoning Capabilities** (reasoning vs. non-reasoning variants). The primary model used is o4-mini. Beyond accuracy, the authors perform **dependency ablation**: truncating text features, removing edges, or removing labels to create 2D accuracy heatmaps that visualize what information each mode relies on.

### Key Designs

**1. Prompting $\phi_{\text{prompt}}$: Serializing k-hop neighborhoods into a single prompt**
This is the baseline mode. it stuffs all context into a single reasoning turn, including candidate categories, the target's description and known label, and the k-hop neighborhood grouped by distance. Each neighbor is listed with its description and label (masked if unknown). The hop distance $k$ controls the info volume (0-hop, 1-hop, 2-hop). Its fatal flaw is the **token budget**: on high-degree graphs (e.g., products, avg degree 61) or long-text graphs (e.g., wiki-cs, avg 3215 words), the 2-hop neighborhood quickly hits the context limit (**TokenLimit**). To mitigate this, a **budget prompt** variant is used, which caps the number of neighbors per hop through downsampling, though this introduces noise and information loss.

**2. GraphTool / GraphTool+ $\phi_{\text{tool}}$: ReAct-style think–act–observe loop**
Inspired by ReAct, this transforms node classification into an iterative "thought–action–observation" loop. At each step, the LLM reasons about "what is known and what is missing," then selects a **single action** from a fixed toolset. The environment executes the action on the graph and appends the result to the history until the model terminates with a label. The basic GraphTool offers four actions: submit label, get neighbors (topology only), get text description (features), and get label (if in training set). The enhanced GraphTool+ adds two batch retrieval actions for precise k-hop neighbors. This **on-demand retrieval** reduces irrelevant exposure and token consumption compared to prompting.

**3. Graph-as-Code $\phi_{\text{code}}$: LLM-generated code queries**
This is the strongest mode proposed. It pushes the ReAct paradigm to the extreme: instead of a fixed toolset, the graph is represented as a **typed table** indexed by node_id, containing features (text), neighbors (list of IDs), and labels. The LLM iteratively **generates a compact program → executes it → reasons over the output**. Its key advantage is **compositional access**: a snippet of code can collapse a sequence of tool calls into a single query (e.g., "count all 2-hop neighbors with label X"), saving steps and tokens while remaining transparent. This capability allows it to handle long-text and dense graphs without exceeding the context window.

**4. Dependency Ablation Analysis: Quantifying information reliance**
To understand which information types the model relies on, the authors designed controlled perturbations on 1000 sampled test nodes. They **truncate text to fixed token percentages** (feature ablation), **randomly delete edges** (structural ablation), and **randomly delete known labels** (label ablation). Results are plotted as 2D heatmaps—one axis for feature/label deletion and the other for edge deletion—to compare the robustness and information dependency patterns of Prompting vs. Graph-as-Code.

## Key Experimental Results

### Main Results
Results are reported by dataset type (short-text homophilic, heterophilic, and long-text homophilic) using o4-mini, comparing the three modes against classical baselines like Label Propagation (LP).

Short-text homophilic datasets (where token limits are triggered as scale grows):

| Dataset | cora | pubmed | arxiv | products |
|--------|------|--------|-------|----------|
| Label Propagation | 76.61 | 80.80 | 68.00 | 70.40 |
| 1-hop prompt | 81.92 | 91.30 | 73.80 | 82.20 |
| 2-hop prompt | 83.43 | 91.80 | 74.30 | **TokenLimit** |
| GraphTool+ | 81.40 | 91.90 | 73.30 | 78.50 |
| Ours (Graph-as-Code) | **85.16** | 89.90 | **74.40** | **82.70** |

On short-text homophilic graphs, Prompting and Graph-as-Code are competitive (Finding 1). However, on the products dataset (avg degree 61), 2-hop prompt fails due to **TokenLimit**, while Graph-as-Code achieves the highest score.

Long-text homophilic datasets (where Graph-as-Code pulls ahead):

| Dataset | reddit | computer | photo | wiki-cs |
|--------|--------|----------|-------|---------|
| 2-hop prompt | TokenLimit | TokenLimit | TokenLimit | TokenLimit |
| 2-hop budget prompt | 54.40 | 86.00 | 85.60 | 80.80 |
| GraphTool+ | 61.80 | 83.10 | 81.30 | 80.50 |
| Ours (Graph-as-Code) | **61.60** | **86.20** | **86.40** | **82.20** |

2-hop prompt hits **TokenLimit** across nearly all long-text graphs. Even the downsampled budget prompt is weaker, while Graph-as-Code remains superior due to on-demand retrieval (Finding 4).

Heterophilic datasets (challenging the notion that LLMs fail on low homophily):

| Dataset | cornell | texas | washington | wisconsin |
|--------|---------|-------|------------|-----------|
| Hom.(%) | 11.55 | 6.69 | 17.07 | 16.27 |
| Label propagation | 41.74 | 78.90 | 15.07 | 14.21 |
| 0-hop prompt | 81.57 | 53.20 | 80.14 | 84.78 |
| Ours (Graph-as-Code) | **92.70** | 73.60 | 81.96 | 89.17 |

Despite homophily as low as 6–17%, LLM interaction modes far outperform baselines (Finding 3), suggesting LLMs utilize non-local, feature-based cues rather than just "neighbor voting."

### Key Findings (Dependency Ablation)

| Finding | Content |
|------|------|
| Finding 5 | When within token limits, Prompting and Graph-as-Code show identical dependency on **features vs. structure**: homophilic graphs (cora/arxiv) suffer most from edge deletion, while heterophilic graphs (cornell) suffer most from feature truncation. |
| Finding 6 | Graph-as-Code is more robust to the deletion of any info type. When structure is fully removed but features are intact, Graph-as-Code maintains high accuracy while Prompting collapses, as the former can still access other nodes' features/labels without edges. |
| Finding 7 | When prompting hits token limits (e.g., photo), behaviors **diverge**. Prompting performance plummets, sometimes even improving if features are truncated (reducing noise), whereas Graph-as-Code stays selective and maintains high performance. |
| Finding 8–9 | In **label vs. structure** ablation, Prompting requires both to function. Graph-as-Code, however, **adaptively switches** to whatever signal is most informative; it only becomes fragile when multiple signals are severely degraded simultaneously. |

## Highlights & Insights
- **Interaction medium is a neglected critical variable**: The main contribution is establishing "how LLMs interact with graphs" as an independent research dimension. Switching to code generation significantly boosts performance for the same LLM, a lesson applicable to other structured data tasks (tables, databases).
- **Token budget is a structural ceiling for prompting**: High-degree and long-text graphs (products, wiki-cs) prove that even with expanding context windows, LLMs struggle with long, flattened inputs. Graph-as-Code's ability to "restructure graph info" ensures it remains relevant regardless of model growth.
- **Adaptive dependency switching is the root of robustness**: Graph-as-Code does not ignore structure; it only uses it when it is more informative than other signals. This capability explains its stability in heterophilic or information-sparse scenarios—a significant mechanistic insight.
- **Challenging the "Heterophily Collapse" myth**: Previous pessimistic conclusions about LLMs on heterophilic graphs appear to be limitations of the interaction mode (prompting) rather than the LLMs themselves.

## Limitations & Future Work
- **Task Scope**: Limited to node classification. It remains unclear if these conclusions generalize to link prediction, graph classification, or KB reasoning.
- **Cost of Graph-as-Code**: Writing code and iterative execution is slower and more expensive than single-turn prompting. The paper lacks a systematic cost/latency analysis.
- **External Dependencies**: Graph-as-Code requires a secure sandbox for execution, introducing engineering complexity and safety concerns regarding code correctness.
- **Feature Truncation as a Proxy**: Using truncation to simulate feature loss may not perfectly mirror real-world data noise or missingness.
- **Model Diversity**: While o4-mini was the primary model, more stable cross-family verification is needed to ensure the ranking of interaction modes holds across all LLMs.

## Related Work & Insights
- **vs. Graph Verbalization & Prompting (Fatemi et al. 2024; Huang et al. 2024a)**: Prior works suggested LLMs collapse on heterophilic graphs. This paper demonstrates those conclusions are largely artifacts of the prompting interaction mode.
- **vs. Tool-use / ReAct (Yao et al. 2023; Schick et al. 2023)**: While ReAct has been used for specific KB reasoning, this paper systematizes and parameterizes it (GraphTool) for general graph learning, quantifying its reliance on different info types.
- **vs. Learnable Graph-LLM Modules (Perozzi et al. 2024; Zhao et al. 2024)**: Instead of training new modules, this paper advocates for a training-free path by "giving LLMs agency," which offers better transferability and lighter weight.

## Rating
- Novelty: ⭐⭐⭐⭐ Does not propose a new model, but establishes "interaction mode" as a primary research dimension with new insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across six axes and robust dependency ablation via heatmaps.
- Writing Quality: ⭐⭐⭐⭐ Clearly organized findings, though dependency analyses are best understood when viewed alongside the original paper's heatmaps.
- Value: ⭐⭐⭐⭐⭐ Provides a practical guide for practitioners (use Graph-as-Code for dense/long-text graphs) and has broader implications for LLMs with structured data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](../../ACL2026/graph_learning/evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)
- [\[ICLR 2026\] AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM](atlaskv_augmenting_llms_with_billion-scale_knowledge_graphs_in_20gb_vram.md)
- [\[ICLR 2026\] Discrete Bayesian Sample Inference for Graph Generation](discrete_bayesian_sample_inference_for_graph_generation.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](../../AAAI2026/graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[ICLR 2026\] Training-Free Counterfactual Explanation for Temporal Graph Model Inference](training-free_counterfactual_explanation_for_temporal_graph_model_inference.md)

</div>

<!-- RELATED:END -->
