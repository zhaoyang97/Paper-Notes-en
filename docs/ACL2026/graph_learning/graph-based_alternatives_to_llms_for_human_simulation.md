---
title: >-
  [Paper Note] Graph-Based Alternatives to LLMs for Human Simulation
description: >-
  [ACL 2026][Graph Learning][Graph Neural Networks] This paper proposes GEMS (Graph-basEd Models for Human Simulation), which models closed-ended human behavior simulation tasks as link prediction problems on heterogeneous…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Human Simulation"
  - "Link Prediction"
  - "Heterogeneous Graphs"
  - "Survey Prediction"
date: 2026-05-08
content_hash: 49493f772556df3f
---

# Graph-Based Alternatives to LLMs for Human Simulation

**Conference**: ACL 2026  
**arXiv**: [2511.02135](https://arxiv.org/abs/2511.02135)  
**Code**: [GitHub](https://github.com/schang-lab/gems)  
**Area**: Graph Learning / Human Behavior Simulation  
**Keywords**: Graph Neural Networks, Human Simulation, Link Prediction, Heterogeneous Graphs, Survey Prediction

## TL;DR

This paper proposes GEMS (Graph-basEd Models for Human Simulation), which models closed-ended human behavior simulation tasks as link prediction problems on heterogeneous graphs. It matches or surpasses strong LLM baselines across three datasets and three evaluation settings while reducing the number of parameters by 3 orders of magnitude.

## Background & Motivation

**Background**: Human behavior simulation has attracted significant attention recently, with LLMs being almost the sole mainstream method in this field. Extensive work utilizes LLMs for closed-ended tasks such as predicting survey responses, social science experiment outcomes, voting results, and test scores.

**Limitations of Prior Work**: (1) LLMs are expensive to run and train; (2) Opaque pre-training processes lead to concerns regarding data leakage and social bias; (3) For closed-ended tasks involving selection from fixed options, the open-ended text generation advantage of LLMs may not be fully exploited.

**Key Challenge**: The essence of closed-ended simulation tasks is predicting individual choices from limited options, which is closer to a link prediction problem in recommendation systems than a natural language generation task. However, this perspective of relational structural modeling has been entirely overlooked in the field.

**Goal**: To explore whether a smaller, more transparent class of models (GNNs) can compete with LLMs on closed-ended human simulation tasks.

**Key Insight**: Represent individuals and options as nodes in a heterogeneous graph, represent observed choices as edges, and utilize the relational inductive bias of GNNs to learn representations for individuals, sub-groups, and options.

**Core Idea**: Replace the token prediction of LLMs with link prediction in GNNs, leveraging the relational structure of human choices rather than language understanding to simulate behavior.

## Method

### Overall Architecture

GEMS constructs a heterogeneous graph containing three types of nodes: sub-group nodes $\mathcal{S}$ (e.g., demographic groups like age/gender), individual nodes $\mathcal{U}$, and option nodes $\mathcal{C}$ (all answer options for each question). Two types of bidirectional edges connect the nodes: membership edges (individual $\to$ sub-group) and response edges (individual $\to$ option). A GNN encoder learns node embeddings through relation-aware message passing, and a decoder predicts the distribution of individuals over question options via dot product + softmax.

### Key Designs

1. **Heterogeneous Graph Construction and Link Prediction**:

    - **Function**: Transforms closed-ended simulation tasks into structured graph learning problems.
    - **Mechanism**: Individual nodes use unified features (de-identified), while sub-groups and option nodes use learnable embedding tables. The GNN aggregates neighborhood information through multi-layer message passing to produce output embeddings $z_w^O$. The decoder computes $p(c|u,q) = \text{softmax}(\text{Dot}(z_u^O, z_c^O) / \tau)$, and the training objective is self-supervised link prediction (randomly masking response edges and reconstructing them).
    - **Design Motivation**: Leverages the relational inductive bias that "similar individuals make similar choices," analogous to collaborative filtering in recommendation systems, but systematically studied for the first time in human simulation.

2. **Unified Framework for Three Evaluation Settings**:

    - **Function**: Covers three scenarios: missing response imputation, new individual prediction, and new question prediction.
    - **Mechanism**: Setting 1 (Imputation) randomly masks some responses of existing individuals; Setting 2 (New Individuals) completely hides all responses of certain individuals during training, retaining only demographic features; Setting 3 (New Questions) completely hides all responses to certain questions during training.
    - **Design Motivation**: These three settings cover core application scenarios of human simulation (survey completion, new population prediction, new survey design), providing a comprehensive evaluation dimension for comparing different methods.

3. **LLM-to-GNN Projection Layer (Setting 3 Only)**:

    - **Function**: Enables the GNN to generalize to new questions unseen during training.
    - **Mechanism**: Option nodes for new questions are isolated in the graph and cannot obtain embeddings through message passing. The solution is to learn a linear projection $z_c' = \mathbf{W}_{\text{proj}} h_{\text{LLM}}(c)$, mapping the hidden states of a frozen LLM to the GNN embedding space. During training, the MSE between the projection and the GNN output embeddings is minimized on seen option nodes.
    - **Design Motivation**: Adds only $d_{\text{LLM}} \times d_{\text{GNN}}$ parameters, far fewer than LLM fine-tuning. The first two settings do not require any language representation at all.

### Loss & Training

Link prediction utilizes cross-entropy loss, where masked response edges are positive samples and other options for the same question serve as implicit negative samples (normalized via softmax). The LLM-to-GNN projection is trained using ridge regression.

## Key Experimental Results

### Main Results

**Setting 1: Missing Response Imputation (Accuracy)**

| Method | OpinionQA | Twin-2K | Dunning-Kruger |
|------|-----------|---------|----------------|
| Zero-shot (Qwen3-8B) | 39.38 | 52.06 | 41.82 |
| Few-shot FT (8, best LLM) | 55.98 | 66.36 | 57.21 |
| **GEMS (SAGE)** | **57.00** | **66.62** | **57.89** |

### Ablation Study

**Setting 2: New Individual Prediction (Accuracy)**

| Method | OpinionQA | Twin-2K | Dunning-Kruger |
|------|-----------|---------|----------------|
| SFT (best LLM) | 50.56 | 61.85 | 56.66 |
| **GEMS (RGCN)** | **50.50** | **62.39** | **56.76** |

### Key Findings

- In Settings 1 and 2, GEMS uses no language representation at all, yet matches or exceeds the strongest LLM fine-tuning methods solely through graph structure.
- In Setting 3 (New Questions), an LLM-to-GNN projection is required, but there is still no need for runtime LLM queries.
- GEMS has approximately $10^3$ times fewer parameters than LLMs, and computational requirements are reduced by up to $10^2$ times.
- Three GNN architectures (RGCN, GAT, GraphSAGE) show similar performance, with SAGE being slightly superior.
- GEMS consistently outperforms Agentic CoT and SFT on OpinionQA, suggesting that relational structure is more critical than language understanding.

## Highlights & Insights

- The core insight is remarkably simple and powerful—closed-ended human simulation is essentially a recommendation system problem, where relational structure is more important than language understanding.
- The experimental design is rigorous, comparing 5 LLM methods × 3 models × 3 datasets × 3 settings under identical conditions.
- GEMS can be trained from scratch on domain data, avoiding the data leakage and bias issues associated with LLM pre-training.

## Limitations & Future Work

- Evaluated only on closed-ended tasks; cannot currently be extended to open-ended human simulation (e.g., dialogue generation, behavioral narratives).
- Setting 3 still requires a frozen LLM to extract text features, meaning it is not entirely detached from LLMs.
- Graph construction relies on predefined sub-groups (e.g., demographic variables); methods for automatic sub-group discovery have not been explored.
- No systematic comparison with classical discrete choice models (e.g., MNL, mixed logit).

## Related Work & Insights

- **vs LLM Fine-tuning (Suh et al., 2025)**: The latter fine-tunes LLMs on the same data but with 1000x more parameters; GEMS shows comparable performance in Setting 1.
- **vs RecSys GNNs**: Technically similar to graph recommendation (e.g., PinSage), but systematically applied to the field of human behavior simulation for the first time.
- **vs Agentic CoT**: The latter uses dual-agent chain-of-thought for reflection and prediction, but performs worse than simple SFT and significantly worse than GEMS in most settings.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic demonstration that GNNs can match LLMs in human simulation; the shift in perspective is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets × 3 settings × 5 LLM methods × 3 LLM models; comparisons are extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem definitions are clear, and experimental logic is easy to follow.
- Value: ⭐⭐⭐⭐ Provides an efficient and transparent alternative for the human simulation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)
- [\[ACL 2026\] Comparing Human and Large Language Model Interpretation of Implicit Information](comparing_human_and_large_language_model_interpretation_of_implicit_information.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[ACL 2026\] Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)

</div>

<!-- RELATED:END -->
