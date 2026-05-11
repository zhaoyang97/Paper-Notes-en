---
title: >-
  [Paper Note] Self-Adaptive Graph Mixture of Models
description: >-
  [AAAI 2026][Graph Learning][Graph Neural Networks] This paper proposes SAGMM (Self-Adaptive Graph Mixture of Models), a graph MoE framework that leverages architectural diversity by employing a Topology-Aware Attention G…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Mixture of Experts"
  - "Adaptive Gating"
  - "Expert Pruning"
  - "Topology-Aware Attention"
date: 2026-05-08
content_hash: 6d251d2594460e67
---

# Self-Adaptive Graph Mixture of Models

**Conference**: AAAI 2026
**arXiv**: [2511.13062](https://arxiv.org/abs/2511.13062)
**Code**: [SAGMM](https://github.com/ast-fri/SAGMM)
**Area**: Graph Learning
**Keywords**: Graph Neural Networks, Mixture of Experts, Adaptive Gating, Expert Pruning, Topology-Aware Attention

## TL;DR

This paper proposes SAGMM (Self-Adaptive Graph Mixture of Models), a graph MoE framework that leverages architectural diversity by employing a Topology-Aware Attention Gating (TAAG) mechanism to adaptively select and combine heterogeneous GNN experts, coupled with an adaptive pruning mechanism. SAGMM consistently outperforms individual GNNs and existing MoE methods across 16 benchmarks spanning node classification, graph classification, regression, and link prediction.

## Background & Motivation

**Background**: GNN architectures have achieved substantial progress, yet performance improvements are approaching saturation. Recent studies show that well-tuned classical models (GCN, GAT, GraphSAGE) can match or even surpass state-of-the-art Graph Transformers on node classification tasks, suggesting that different models capture distinct subregions of the representation space while each individually offering limited coverage.

**Limitations of Prior Work**: (1) Selecting the optimal GNN for a specific dataset requires extensive trial-and-error and hyperparameter tuning, with many trained models ultimately discarded. (2) Existing graph MoE methods (e.g., GMoE, DA-MoE) employ variants of the same base model as experts, resulting in limited architectural diversity. (3) Gating mechanisms rely on simple linear projections, neglecting graph topology information. (4) Top-$k$ routing forces all nodes to activate the same number of experts, which is misaligned with the actual needs of individual nodes.

**Key Challenge**: A single GNN cannot cover all graph structural patterns (No Free Lunch theorem), yet naive expert mixing lacks topology-aware adaptive selection capability.

**Core Idea**: A pool of architecturally heterogeneous GNNs (GCN, GAT, GraphSAGE, GIN, etc.) serves as experts, with a topology-aware sparse attention gating mechanism enabling each node to adaptively determine the number and combination of relevant experts.

## Method

### Overall Architecture

SAGMM comprises three core components: (1) a heterogeneous expert pool of diverse GNN architectures; (2) Topology-Aware Attention Gating (TAAG), which performs dynamic routing based on local and global graph structural information; and (3) adaptive expert pruning, which removes low-importance experts during training. Input features are first augmented with structural context, after which TAAG computes attention scores for each node–expert pair. Selected experts process the input, and their outputs are aggregated via weighted summation using the gating weights.

### Key Designs

1. **Heterogeneous Expert Pool**:

    - **Function**: Employs GNN models with fundamentally different architectures as experts to maximize diversity in inductive biases.
    - **Mechanism**: Experts are selected along three dimensions—propagation strategy (spectral vs. spatial), aggregation mechanism (mean vs. attention), and training paradigm (transductive vs. inductive)—yielding a pool including GCN (spectral filtering), GAT (attention-weighted aggregation), GraphSAGE (inductive sampling), GIN (WL-test approximation), and JKNet (jumping knowledge connections). The message-passing update for each expert is $H^{(l+1)}_{e_i} = f_{e_i}(H^{(l)}_{e_i}, A)$.
    - **Design Motivation**: Heterogeneous experts provide complementary perspectives—GCN excels on homophilic graphs, GAT captures attention patterns, and GraphSAGE suits large-scale graphs. The No Free Lunch theorem guarantees that no single model is globally optimal.

2. **Topology-Aware Attention Gating (TAAG)**:

    - **Function**: Enables each node to adaptively select the number and combination of experts based on local and global graph structural information.
    - **Mechanism**: Input features are augmented as $\mathbf{X'} = \frac{1}{3}(\mathbf{X} + \mathbf{X}^{(1)} + \mathbf{X}^{(2)}) \| \mathbf{X}^{(g)}$, where $\mathbf{X}^{(1)}, \mathbf{X}^{(2)}$ are 1-hop and 2-hop aggregated features, and $\mathbf{X}^{(g)}$ denotes the $p$ smallest eigenvectors of the normalized Laplacian (global positional encoding). Simple Global Attention (SGA) is applied to compute gating scores $Z$ with linear complexity $O(n)$; a learnable threshold $T$ and sigmoid activation then filter the scores: $M = \text{sign}(\text{ReLU}(\sigma(Z) - \sigma(T)))$, activating $k_u = |\{j | M_{u,j} > 0\}|$ experts per node $u$.
    - **Design Motivation**: Overcomes the limitation of Top-$k$ routing, which enforces a fixed number of experts. The linear complexity of SGA enables scalability to large graphs, while the combination of local and global features allows the gate to perceive both neighborhood structure and global graph position.

3. **Adaptive Expert Pruning**:

    - **Function**: Dynamically removes low-contribution experts during training to improve computational efficiency.
    - **Mechanism**: Importance scores are updated recursively as $I_t(e_i) = (1-\alpha)I_{t-1}(e_i) + \alpha \gamma(e_i)$, where $\gamma(e_i) = \|\sum_u G_{e_i,u} H_{e_i,u,:}\|$ measures the cumulative weighted contribution of expert $e_i$ to the model output. Experts falling below threshold $\eta$ are removed.
    - **Design Motivation**: The smoothing factor $\alpha$ prevents premature removal of experts that become useful in later training stages. Experiments show that pruning largely preserves performance while improving computational efficiency.

### Loss & Training

The training objective combines task loss, importance loss (encouraging uniform expert utilization), and diversity loss (encouraging orthogonal expert activation patterns). Two variants are supported: end-to-end training and SAGMM-PE (pretrained frozen experts), where only the gating mechanism and task head are trained.

## Key Experimental Results

### Main Results (Node Classification)

| Method | Deezer | YelpChi | ogbn-proteins | ogbn-arxiv | Pokec |
|--------|--------|---------|---------------|------------|-------|
| GCN | 57.70 | 85.62 | 69.74 | 71.74 | 76.52 |
| GAT | 58.59 | 85.42 | 69.56 | 71.42 | 78.87 |
| GraphSAGE | 64.40 | 89.23 | 73.21 | 71.46 | 79.82 |
| Graph CNN | 63.65 | 89.25 | 77.54 | 72.04 | 80.21 |
| GMoE-GCN | 61.11 | 85.75 | 74.48 | 71.88 | 76.04 |
| DA-MoE | 62.15 | 85.53 | 75.22 | 71.96 | 64.87 |
| **SAGMM** | **64.73** | **91.06** | **78.15** | **72.80** | **82.25** |

### Ablation Study

| Configuration | ogbn-arxiv | ogbn-proteins | Note |
|---------------|-----------|---------------|------|
| SAGMM (full) | 72.80 | 78.15 | — |
| Replace with Top-2 gating | 72.18 | 76.45 | TAAG advantage |
| Replace with Top-4 gating | 72.01 | 77.02 | Fixed $k$ inferior to adaptive |
| Single expert (best) | 72.04 | 77.54 | Mixture outperforms individual |
| Without pruning | 72.65 | 77.89 | Pruning slightly improves efficiency |

### Key Findings

- SAGMM consistently outperforms all individual GNNs and MoE baselines across 16 benchmarks covering node classification, graph classification, regression, and link prediction.
- Different nodes activate different numbers of experts in practice (e.g., the distribution of $k_u$ on ogbn-proteins ranges from 1 to 7), validating the necessity of adaptive selection.
- SAGMM-PE (pretrained frozen experts) achieves comparable performance with only 50–70% of the training data, demonstrating strong data efficiency.
- TAAG gating outperforms Top-$k$ gating by an average of 0.5–1.5%; the combination of local and global features is identified as the critical factor.

## Highlights & Insights

- **Automated Model Selection**: SAGMM automates the trial-and-error process of selecting the best GNN for a given dataset as end-to-end learning, offering substantial practical value.
- **Exploitation of Architectural Heterogeneity**: Unlike homogeneous MoE methods (e.g., GMoE using variants of the same GCN), SAGMM leverages fundamentally different architectures (GCN + GAT + SAGE + GIN + ...), yielding stronger complementarity.
- **Learnable Threshold as Top-$k$ Replacement**: The threshold $T$ automatically learns how many experts each node should activate, outperforming a manually specified fixed $k$.
- **Reuse of Pretrained Experts**: SAGMM-PE can incorporate previously discarded trained models at zero additional cost, avoiding computational waste.

## Limitations & Future Work

- **Sensitivity to Expert Pool Composition**: The paper fixes the expert pool contents (GCN + GAT + SAGE, etc.); the effect of different pool compositions on performance is not thoroughly analyzed.
- **Training Overhead**: End-to-end training requires simultaneous forward passes through all experts, incurring considerable memory and computational costs.
- **Efficiency on Very Large Graphs**: Although SGA operates in $O(n)$, precomputing Laplacian eigenvectors remains expensive for extremely large graphs.
- **Heterogeneous and Dynamic Graphs**: Experiments cover only homophilic/heterophilic static graphs; temporal graphs and heterogeneous information networks are not evaluated.
- **Integration with Graph Transformers**: The expert pool does not include Graph Transformers, potentially overlooking their unique representational capabilities.

## Related Work & Insights

- **vs. GMoE**: GMoE uses variants of the same GCN as experts, resulting in limited architectural diversity and topology-agnostic gating. SAGMM's heterogeneous pool combined with TAAG is superior both conceptually and empirically.
- **vs. DA-MoE**: DA-MoE uses GNN layers as experts but does not support dynamic selection; SAGMM's adaptive $k_u$ and pruning offer greater flexibility.
- **vs. Traditional Ensemble Learning**: SAGMM is not a simple bagging/boosting approach but achieves input-conditioned expert selection through attention-based gating.
- **Implications for LLM MoE**: The idea of replacing Top-$k$ routing with a learnable threshold in TAAG may also inspire improved routing strategies in LLM-based MoE architectures.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of a heterogeneous expert pool and topology-aware adaptive gating constitutes a meaningful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 16 benchmarks, 4 task types, comprehensive ablation and variant analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, thorough background analysis, and concise theoretical discussion.
- Value: ⭐⭐⭐⭐ — Direct practical value for automating model selection in graph learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[AAAI 2026\] On Stealing Graph Neural Network Models](on_stealing_graph_neural_network_models.md)
- [\[AAAI 2026\] Adaptive Initial Residual Connections for GNNs with Theoretical Guarantees](adaptive_initial_residual_connections_for_gnns_with_theoretical_guarantees.md)
- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](self-correction_distillation_for_structured_data_question_answering.md)

</div>

<!-- RELATED:END -->
