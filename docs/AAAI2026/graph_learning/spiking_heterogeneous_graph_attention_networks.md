---
title: >-
  [Paper Note] Spiking Heterogeneous Graph Attention Networks
description: >-
  [AAAI 2026][Graph Learning][Heterogeneous Graphs] This paper proposes SpikingHAN, the first framework to introduce **Spiking Neural Networks (SNNs)** into heterogeneous graph learning. It employs a single-layer graph convolution with shared parameters to aggregate meta-path-based neighborhood information, fuses multiple meta-path semantics via semantic-level attention, and encodes the resulting representations into 1-bit binary spike sequences. SpikingHAN achieves competitive…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Heterogeneous Graphs"
  - "Spiking Neural Networks"
  - "Meta-path"
  - "Graph Attention"
  - "Energy-Efficient Computing"
date: 2026-05-08
content_hash: 2a069dc5b027decc
---

# Spiking Heterogeneous Graph Attention Networks

**Conference**: AAAI 2026
**arXiv**: [2601.02401](https://arxiv.org/abs/2601.02401)  
**Code**: [https://github.com/QianPeng369/SpikingHAN](https://github.com/QianPeng369/SpikingHAN)  
**Area**: Graph Learning / Spiking Neural Networks
**Keywords**: Heterogeneous Graphs, Spiking Neural Networks, Meta-path, Graph Attention, Energy-Efficient Computing

## TL;DR

This paper proposes SpikingHAN, the first framework to introduce **Spiking Neural Networks (SNNs)** into heterogeneous graph learning. It employs a single-layer graph convolution with shared parameters to aggregate meta-path-based neighborhood information, fuses multiple meta-path semantics via semantic-level attention, and encodes the resulting representations into 1-bit binary spike sequences. SpikingHAN achieves competitive node classification performance on three datasets with fewer parameters, faster inference, and lower energy consumption.

## Background & Motivation

1. **Background**: Heterogeneous Graph Neural Networks (HGNNs) such as HAN effectively handle multi-type nodes and edges, capturing rich structural and semantic information. SNNs have demonstrated success in computer vision and homogeneous graph learning (e.g., SpikingGCN, SpikeGCL).
2. **Limitations of Prior Work**: HGNNs tend to be architecturally complex — HAN assigns an independent attention module to each meta-path, causing parameters, memory, and computational cost to scale significantly with the number of meta-paths, making deployment on resource-constrained devices difficult.
3. **Key Challenge**: Heterogeneous graph learning requires effective modeling of heterogeneous information, yet existing methods incur prohibitive computational overhead. The binary, sparse communication of SNNs is naturally suited for low-energy scenarios but has not been explored on heterogeneous graphs.
4. **Goal**: Design an energy-efficient heterogeneous graph neural network capable of operating under resource-constrained conditions.
5. **Key Insight**: Replace HAN's multiple independent aggregation modules with a single-layer GCN with shared parameters to simplify the model, then encode heterogeneous information into spike sequences via SNNs to achieve binarization.
6. **Core Idea**: Simplified heterogeneous graph convolution with shared parameters + SNN spike encoding = energy-efficient heterogeneous graph learning.

## Method

### Overall Architecture

The framework consists of three components: (1) **Shared Convolution and Attention** — a single-layer GCN with shared parameters aggregates neighborhood information for each meta-path, and semantic-level attention fuses multi-meta-path semantics; (2) **Fully Connected Layer and SNN** — a linear transformation produces the input current for the SNN, which performs integrate-and-fire-reset operations to generate spike sequences; (3) **Spike Aggregation and Prediction** — average pooling over multi-timestep spike outputs yields firing rates as classification predictions.

### Key Designs

1. **Single-Layer Graph Convolution with Shared Parameters**

    - Function: Aggregate neighborhood information for each meta-path with minimal model complexity via shared convolution.
    - Mechanism: Unlike HAN, which assigns an independent aggregation module to each meta-path $\Phi_p$, SpikingHAN applies a single shared weight matrix $W_1$ across all meta-paths using standard GCN convolution: $h_i^{\Phi_p} = \sigma(\sum_{j \in N_i^{\Phi_p}} h_j^0 \cdot W_1 / \sqrt{\tilde{D}_i \cdot \tilde{D}_j})$. This produces $P$ sets of meta-path-specific node embeddings $\{h^{\Phi_1}, ..., h^{\Phi_P}\}$.
    - Design Motivation: Independent modules cause parameter count to grow linearly with the number of meta-paths and are prone to overfitting. Shared parameters substantially reduce complexity while preserving expressiveness — semantic differences across meta-paths are distinguished by the subsequent attention mechanism.

2. **Semantic-Level Attention and SNN Encoding**

    - Function: Adaptively learn the importance of different meta-paths, fuse them, and encode the result into spike representations.
    - Mechanism: Semantic attention weights are computed as $\beta_{\Phi_p} = \text{softmax}(\frac{1}{|V|}\sum_i q^T \cdot \tanh(h_i^{\Phi_p} \cdot W_2 + b))$, and the weighted aggregation is $H = \sum_p \beta_{\Phi_p} \cdot h^{\Phi_p}$. This is then passed into a simplified fully connected layer without nonlinear activation or bias, $H \cdot W_3$, serving as the SNN input current. The SNN uses the PLIF (Parametric LIF) model with a learnable membrane time constant $\tau_m$, generating spikes through integrate-and-fire-reset dynamics. The final classification uses the average spike output over $T$ timesteps: $\hat{y} = \frac{1}{T}\sum_t \Theta(V^t)$.
    - Design Motivation: The binary communication of SNNs (0/1 spikes) substantially reduces computational and memory overhead compared to floating-point representations. The simplified fully connected layer (no nonlinearity, no bias) is motivated by findings from LightGCN — depth is not a critical factor in graph-based learning.

### Loss & Training

Semi-supervised learning with cross-entropy loss; backpropagation through time via surrogate gradients.

## Key Experimental Results

### Main Results

Node classification on three heterogeneous graph datasets (Mi-F1, 20% training ratio):

| Method | Type | DBLP | ACM | IMDB |
|--------|------|------|-----|------|
| GAT | Homogeneous GNN | 91.4 | 91.1 | 58.5 |
| SpikingGCN | Homogeneous SNN | 88.7 | 91.2 | 51.3 |
| HAN | Heterogeneous GNN | 93.1 | 92.8 | 61.3 |
| PHGT | Heterogeneous GNN | **93.7** | **93.4** | **63.2** |
| **SpikingHAN** | Heterogeneous SNN | **93.7** | 93.3 | 62.9 |

### Ablation Study

| Configuration | Result | Notes |
|---------------|--------|-------|
| Independent GCN vs. Shared GCN | Shared is superior | Fewer parameters and reduced overfitting |
| IF vs. LIF vs. PLIF | PLIF is best | Learnable time constant offers greater flexibility |
| Timesteps T=2/4/8/16 | T=4 is optimal | Too few: insufficient information; too many: wasteful computation |
| Hard reset vs. Subtract-threshold reset | Subtract-threshold is better | Retains more membrane potential information |

### Key Findings

- SpikingHAN matches PHGT on DBLP (93.7%) and closely approaches it on ACM (93.3% vs. 93.4%), demonstrating that SNNs do not sacrifice accuracy.
- The parameter count is approximately one-third that of HAN, with faster inference and significantly lower energy consumption.
- The learnable $\tau_m$ in PLIF contributes substantially to performance — different datasets require different temporal dynamics.
- Performance on IMDB is slightly weaker (62.9% vs. 63.2%), possibly because IMDB's noisier labels cause the binary spike representations to lose some discriminative information.

## Highlights & Insights

- **First introduction of SNNs into heterogeneous graphs**: Fills a gap in the literature and demonstrates the feasibility and efficiency advantages of this combination.
- **Minimalist shared-parameter design**: A single-layer shared GCN combined with semantic attention suffices to match the performance of multi-layer independent modules, embodying a "less is more" design philosophy.
- **Practical value of 1-bit representations**: Spike encoding compresses floating-point embeddings to 1-bit, which is highly significant for deployment on edge devices.

## Limitations & Future Work

- Evaluation is limited to node classification; tasks such as link prediction and graph classification are not explored.
- Shared parameters may lack sufficient expressiveness when semantic differences across meta-paths are substantial.
- Actual energy consumption benefits on real neuromorphic hardware have not been validated.
- The performance gap on IMDB suggests that SNN binarization is more sensitive to noisy labels.

## Related Work & Insights

- **vs. HAN**: HAN performs independent aggregation per meta-path; SpikingHAN uses shared parameters and SNNs, resulting in fewer parameters and lower energy consumption.
- **vs. SpikingGCN/SpikeGCL**: These methods handle only homogeneous graphs; SpikingHAN is the first to support heterogeneous graphs with multi-type nodes and edges.
- **vs. PHGT**: The Transformer-based approach achieves the highest accuracy but at substantial computational cost; SpikingHAN trades a marginal performance gap for significant efficiency gains.

## Rating

- Novelty: ⭐⭐⭐⭐ First combination of SNNs and heterogeneous graphs
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multi-dimensional efficiency analysis, and complete ablation study
- Writing Quality: ⭐⭐⭐⭐ Clear structure with intuitive comparative figures against HAN
- Value: ⭐⭐⭐⭐ Practically significant for edge deployment of heterogeneous graph learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Kernelized Edge Attention: Addressing Semantic Attention Blurring in Temporal Graph Neural Networks](kernelized_edge_attention_addressing_semantic_attention_blurring_in_temporal_gra.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)

</div>

<!-- RELATED:END -->
