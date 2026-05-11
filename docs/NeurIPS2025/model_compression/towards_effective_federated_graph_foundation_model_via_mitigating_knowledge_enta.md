---
title: >-
  [Paper Note] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement
description: >-
  [NeurIPS 2025][Model Compression][Federated Graph Learning] This work is the first to propose the Federated Graph Foundation Model (FedGFM) paradigm…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Federated Graph Learning"
  - "Graph Foundation Model"
  - "Knowledge Entanglement"
  - "Vector Quantization"
  - "Prompt Learning"
date: 2026-05-08
content_hash: 562f51ae3a5f5c41
---

# Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement

**Conference**: NeurIPS 2025
**arXiv**: [2505.12684](https://arxiv.org/abs/2505.12684)
**Code**: N/A
**Area**: Model Compression
**Keywords**: Federated Graph Learning, Graph Foundation Model, Knowledge Entanglement, Vector Quantization, Prompt Learning

## TL;DR

This work is the first to propose the Federated Graph Foundation Model (FedGFM) paradigm, which integrates the distributed collaborative capability of federated graph learning with the cross-domain generalization capability of graph foundation models. Two modules — AncDAI (Anchor-based Domain-Aware Initialization) and AdaDPP (Adaptive Domain-sensitive Prompt Pool) — are introduced to mitigate knowledge entanglement, achieving state-of-the-art performance on 8 cross-task, cross-domain datasets against 20 baselines.

## Background & Motivation

Graph machine learning faces inherent limitations under two dominant paradigms:

**Limitations of Federated Graph Learning (FGL)**:

**Data heterogeneity**: Graphs across different clients vary greatly in feature dimensionality, label space, and topological patterns; most FGL methods are restricted to collaborative training on subsets of a single dataset.

**Task heterogeneity**: Existing FGL methods assume a unified graph granularity and downstream task (node-level / subgraph-level / graph-level), making multi-task collaboration difficult.

**Limitations of Graph Foundation Models (GFM)**:

**Multi-domain data silos**: Training GFMs requires multi-domain graph data, but in practice data is distributed across institutions and cannot be shared due to privacy regulations.

**Neglect of cross-institution resources**: Centralized training cannot leverage storage and compute resources distributed across multiple institutions.

**Complementary relationship**: FGL provides a distributed training paradigm for GFMs, while GFMs provide a unified feature encoding and pre-train–fine-tune framework for FGL. Combining the two is therefore a natural choice.

**Knowledge entanglement challenge**: Naively distributing gVQ-VAE pre-training to a federated setting causes multi-domain knowledge to be encoded into indistinguishable representations. Empirical evidence (Figure 2(b)) shows that GFT (centralized training) produces clearly differentiated inter-domain cosine similarities, whereas GFT* (federated training) yields inter-domain similarities close to 1, indicating a collapse of domain-specific representations.

## Method

### Overall Architecture

FedGFM+ adopts a federated pre-training followed by fine-tuning paradigm:
1. **Federated pre-training**: Each client performs self-supervised learning on private graphs; the server aggregates local models to build a global graph foundation model.
2. **Fine-tuning**: The global model serves as a GFM and is adapted to specific downstream tasks via supervised learning.
3. **Dual-perspective disentanglement**: AncDAI (global) and AdaDPP (local) collaboratively mitigate knowledge entanglement.

The backbone network is gVQ-VAE, chosen because: (1) it jointly encodes graph structure and textual attributes into discrete semantic representations; and (2) it has an extremely small parameter count (e.g., GFT has only 7M parameters), making it naturally suited to communication-constrained federated settings.

### Key Designs

#### 1. **AncDAI: Anchor-based Domain-Aware Initialization (Global Perspective)**

**Core Idea**: Before pre-training, domain prototypes from each client are used as semantic anchors to initialize the global codebook, injecting domain-discriminative inductive bias.

Steps:
1. Each client encodes its local graph using the globally initialized model: $\mathbf{Z}^k = f_{\theta^{glb}}(\mathbf{X}^k, \mathbf{A}^k)$
2. Mean pooling yields the domain prototype: $\mathbf{p}^k = \frac{1}{|\mathcal{V}^k|} \sum_{i \in \mathcal{V}^k} \mathbf{z}_i^k$
3. Theoretical guarantee: Even with randomly initialized shared parameters, domain prototypes remain distinguishable across clients (Theorem B.1).
4. Perturbed embeddings are generated around each anchor: $\tilde{\mathbf{p}}_i^k = \mathbf{p}^k + \sigma \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(\mathbf{0}, \mathbf{1})$
5. Synthetic embeddings from all domains are aggregated to initialize the global codebook.

**Design Motivation**: To endow the codebook with a domain-aware structure from the outset, laying the foundation for maintaining inter-domain separation throughout federated training.

#### 2. **AdaDPP: Adaptive Domain-sensitive Prompt Pool (Local Perspective)**

**Pre-training phase**: Each client independently learns a set of domain-specific prompts $\Phi^k = \{\phi_i^k\}_{i=1}^\lambda$, which are excluded from federated aggregation. Node features are augmented as:

$$\tilde{x}_i^k = x_i^k + \sum_{j=1}^\lambda \alpha_j^k \phi_j^k, \quad \alpha_j^k = \frac{e^{(\mathbf{w}_j^k)^T x_i^k}}{\sum_{t=1}^\lambda e^{(\mathbf{w}_t^k)^T x_i^k}}$$

**Fine-tuning phase**: Prompts from all clients are collected to build a global prompt pool $\rho$. For a target graph, the most relevant prompts are selected via an attention mechanism:

$$\tilde{x}_i^{tgt} = x_i^{tgt} + \sum_{p=1}^K \sum_{j=1}^\lambda \alpha_j^p \phi_j^p$$

**Design Motivation**: Excluding prompts from federated aggregation prevents information mixing; combining them at fine-tuning time enables adaptive transfer of cross-domain knowledge, realizing a "preserve domain specificity first, then adaptively transfer" strategy.

### Loss & Training

Pre-training loss (self-supervised reconstruction, Eq. 2):
$$\mathcal{L}_{pretrain} = \mathcal{L}_{feat} + \mathcal{L}_{topo} + \text{codebook alignment} + \text{commitment loss}$$

- $\mathcal{L}_{feat}$: Node feature reconstruction (cosine similarity)
- $\mathcal{L}_{topo}$: Topological reconstruction (adjacency matrix)
- A straight-through estimator is used to enable end-to-end gradient flow.

Federated aggregation follows the FedAvg strategy: $\Theta^g \leftarrow \frac{N_k}{N} \sum_{k=1}^K \Theta^k$

## Key Experimental Results

### Main Results

Performance comparison on 8 cross-domain, cross-task datasets:

| Method | Cora | PubMed | OGB-arxiv | WikiCS | FB15K | WN18RR | HIV | PCBA |
|---|---|---|---|---|---|---|---|---|
| GCN | 80.17 | 84.70 | 72.50 | 77.24 | 71.24 | 82.27 | 65.37 | 63.41 |
| FedAvg | 81.45 | 85.22 | 71.53 | 77.67 | 73.14 | 83.55 | 66.05 | 68.52 |
| GFT* (federated variant) | 81.07 | 84.24 | 73.19 | 78.81 | 73.52 | 86.30 | 66.32 | 72.81 |
| GQT* (federated variant) | 81.92 | 85.59 | 74.07 | 77.52 | 73.40 | 85.66 | 67.93 | 73.22 |
| **FedGFM+** | **83.79** | **88.52** | **76.31** | **80.70** | **75.25** | **89.25** | **69.39** | **77.68** |

Gains over the best baseline: node classification ≥2.70%, edge classification ≥2.18%, graph classification ≥3.09%.

### Ablation Study

| Configuration | Cora | PubMed | OGB-arxiv | HIV | PCBA | Note |
|---|---|---|---|---|---|---|
| w/o AncDAI | 81.55 | 85.56 | 75.19 | 67.52 | 74.81 | Largest performance drop |
| w/o AdaDPP | 83.17 | 87.42 | 75.83 | 67.84 | 76.72 | Without prompt pool |
| **FedGFM+** | **83.79** | **88.52** | **76.31** | **69.39** | **77.68** | Full method |

### Key Findings

1. **Knowledge entanglement is the core bottleneck**: Naïve federated GFM variants (e.g., GFT*) can suffer negative transfer, underperforming isolated supervised models.
2. **AncDAI contributes more**: Removing AncDAI leads to a larger performance drop than removing AdaDPP, indicating that global initialization is critical for combating entanglement.
3. **Communication efficiency**: GFM parameters are on the order of millions (vs. billions for LLMs), making federated communication overhead acceptable.
4. **Hyperparameter robustness**: Performance remains stable across a wide range of codebook sizes and prompt counts.

## Highlights & Insights

1. **Paradigm-level contribution**: FedGFM is the first systematic proposal to combine FGL and GFM, integrating their complementary strengths.
2. **Precise problem formulation**: Empirical analysis (Figure 2) clearly identifies knowledge entanglement as a non-trivial challenge.
3. **Dual-perspective solution**: The combination of global (initialization) and local (prompts) strategies is well-motivated and coherent.
4. **Theoretical support**: The work proves that domain prototypes remain distinguishable under random initialization (Theorem B.1) and that the initialization strategy provides structured inductive bias (Theorem B.2).

## Limitations & Future Work

1. **Privacy risks**: The exchange of prototypes and prompts may expose partial semantic information; formal privacy analysis is needed.
2. Current experiments assign each dataset to 3 clients; real-world scenarios may involve a much larger number of clients with greater heterogeneity.
3. Differential privacy (DP) or secure computation could be incorporated to protect the transmission of prototypes and prompts.
4. Only the gVQ-VAE backbone is evaluated; whether the approach generalizes to larger-scale GFM architectures warrants further exploration.

## Related Work & Insights

- The work transfers federated foundation model training ideas from NLP/CV to the graph domain, where heterogeneity is considerably stronger.
- Excluding prompt learning from federated aggregation is analogous to local fine-tuning strategies in personalized federated learning.
- The domain prototype concept in AncDAI can be generalized to other federated learning scenarios that require handling domain heterogeneity.
- The codebook initialization strategy has broader applicability to federated training of VQ-VAE-based models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The FedGFM paradigm is proposed for the first time; both the problem formulation and method design are original.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8 datasets, 20 baselines, 3 task types, ablation and hyperparameter analysis.)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured, though the method involves multiple interleaved concepts — federated learning, GFM, VQ-VAE, and prompt learning — making the paper information-dense.)
- Value: ⭐⭐⭐⭐ (Opens a promising research direction, but privacy and scalability challenges remain before real-world deployment.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning](graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)
- [\[NeurIPS 2025\] PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation](ppg-distill_efficient_photoplethysmography_signals_analysis_via_foundation_model.md)
- [\[NeurIPS 2025\] Mitigating Semantic Collapse in Partially Relevant Video Retrieval](mitigating_semantic_collapse_in_partially_relevant_video_retrieval.md)
- [\[NeurIPS 2025\] Graph Your Own Prompt](graph_your_own_prompt.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)

</div>

<!-- RELATED:END -->
