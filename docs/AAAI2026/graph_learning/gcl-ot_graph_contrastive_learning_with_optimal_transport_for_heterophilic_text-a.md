---
title: >-
  [Paper Note] GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs
description: >-
  [AAAI 2026][Graph Learning][Graph Contrastive Learning] This paper proposes GCL-OT, the first framework to introduce Optimal Transport (OT) into graph contrastive learning for heterophilic text-attributed graphs. Three dedicated modules — RealSoftMax similarity estimation, a filter-prompt mechanism, and OT-guided latent homophily mining — address three multi-granularity heterophily challenges: partial heterophily, complete heterophily, and latent homophily, respectively.
tags:
  - AAAI 2026
  - Graph Learning
  - Graph Contrastive Learning
  - Optimal Transport
  - Heterophilic Graphs
  - Text-Attributed Graphs
  - Multi-Granularity Heterophily
  - RealSoftMax
  - Sinkhorn
date: 2026-05-08
content_hash: 01e22b7889c47535
---

# GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs

**Conference**: AAAI 2026
**arXiv**: [2511.16778](https://arxiv.org/abs/2511.16778)
**Code**: [github.com/users-01/GCL-OT](https://github.com/users-01/GCL-OT)
**Area**: Graph Learning / Graph Contrastive Learning
**Keywords**: Graph Contrastive Learning, Optimal Transport, Heterophilic Graphs, Text-Attributed Graphs, Multi-Granularity Heterophily, RealSoftMax, Sinkhorn

## TL;DR

This paper proposes GCL-OT, the first framework to introduce Optimal Transport (OT) into graph contrastive learning for heterophilic text-attributed graphs. Three dedicated modules — RealSoftMax similarity estimation, a filter-prompt mechanism, and OT-guided latent homophily mining — address three multi-granularity heterophily challenges: partial heterophily, complete heterophily, and latent homophily, respectively.

## Background & Motivation

**Text-Attributed Graphs (TAGs)** model textual entities as nodes and relations as edges, and are widely used in academic citation networks, e-commerce recommendation, and similar applications. Existing methods typically combine GNNs with language models (LMs) and align structural and textual representations via contrastive learning.

However, **heterophilic TAGs** are prevalent in practice (e.g., opposite-attract patterns in dating networks), and existing methods face three major challenges:

**Partial Heterophily**: Node text aligns only partially with neighbor semantics; existing methods fail to capture fine-grained semantic mismatches.

**Complete Heterophily**: Node text is entirely unrelated to its neighbors (e.g., random co-purchase relations); SOTA models are misled by such noise.

**Latent Homophily**: Semantically similar but unconnected latent neighbors are overlooked due to missing edges or implicit links.

Traditional InfoNCE supports only 1:1 hard alignment, and DGI's N:1 scheme is equally insufficient. **OT naturally supports N:N soft alignment**, enabling flexible partial-mass assignment for each node pair.

## Method

### Overall Architecture

GCL-OT consists of four stages:
1. **Multi-view Feature Encoding**: LLM-augmented text → PLM encodes text (token-level + sentence-level); GNN encodes structural information.
2. **Hierarchical OT Alignment**: RealSoftMax for partial heterophily + Filter-Prompt for complete heterophily.
3. **Latent Homophily Mining**: OT transport matrix as auxiliary supervision.
4. **Joint Optimization**: Contrastive loss + node classification loss.

### Key Design 1: RealSoftMax Similarity Estimation (Partial Heterophily)

For the $k$-th neighbor embedding of node $v_i$ and the $w$-th word embedding of node $v_j$, the bidirectional similarity is defined as:

$$s_{ij} = \frac{1}{2}\left(\mathbb{E}_k[\text{RSM}_\beta(\{h_{ik}^{\mathcal{N}} \cdot h_{jw}^{\varpi}\}_{w=1}^W)] + \mathbb{E}_w[\text{RSM}_\beta(\{h_{iw}^{\varpi} \cdot h_{jk}^{\mathcal{N}}\}_{k=1}^K)]\right)$$

where $\text{RSM}_\beta$ degenerates to $\max$ as $\beta \to 0$ and to $\text{mean}$ as $\beta \to \infty$, enabling smooth interpolation. The first term identifies the most relevant word for each neighbor; the second term does the reverse. This bidirectional formulation emphasizes informative interactions and suppresses background noise.

### Key Design 2: Filter-Prompt Global Filtering (Complete Heterophily)

The structural-textual global similarity matrix $S$ is extended to an $(N+1) \times (N+1)$ matrix:

$$\bar{S} = \begin{bmatrix} S & z \\ z^\top & z_{N+1} \end{bmatrix}$$

where $z$ is a learnable prompt vector. If the maximum similarity of an embedding falls below the corresponding value in $z$, that embedding is aligned with the prompt vector rather than with other nodes in the OT alignment, thereby adaptively excluding irrelevant noise.

The OT problem is solved efficiently via Low-Rank Sinkhorn (LRSinkhorn), reducing complexity from $O(N^2)$ to $O(Nr)$.

### Key Design 3: OT-Guided Latent Homophily Mining

The global OT transport matrix $\hat{Q}^*$ is used as a soft supervision signal to construct a contrastive objective:

$$P = I + \hat{Q}^*$$

After normalization, $P$ serves as soft labels. The $\mathcal{L}_\text{LHM}$ loss encourages embeddings of semantically similar but unconnected nodes to be closer together, preventing latent positive pairs from being treated as negatives.

### Loss & Training

The total loss consists of three components:

$$\mathcal{L} = \mathcal{L}_{NC} + \lambda \mathcal{L}_{GCL\text{-}OT}$$

- $\mathcal{L}_{NC}$: Cross-entropy loss for node classification.
- $\mathcal{L}_{GCL\text{-}OT} = \mathcal{L}_{MHA} + \mathcal{L}_{LHM}$: Multi-level hierarchical alignment loss + latent homophily mining loss.

Theoretical analysis proves that both $\mathcal{L}_{MHA}$ and $\mathcal{L}_{LHM}$ provide tighter mutual information lower bounds than standard InfoNCE.

## Key Experimental Results

### Main Results: Node Classification Accuracy (%)

| Method | Cora | PubMed | Products | Wisconsin | Cornell | Texas |
|--------|------|--------|----------|-----------|---------|-------|
| GCN | 89.11 | 85.33 | 75.64 | 46.98 | 44.36 | 54.21 |
| TAPE-RevGAT | 92.80 | 96.04 | 79.76 | 87.77 | 88.46 | 85.90 |
| ENGINE-LLAMA | 91.48 | 95.24 | 80.05 | 85.50 | 77.36 | 75.68 |
| **GCL-OT-GCN** | **93.54** | 96.08 | 81.50 | 88.68 | 88.64 | **89.47** |
| **GCL-OT-SAGE** | **93.73** | **96.62** | 81.73 | **89.26** | 88.21 | **90.01** |

GCL-OT consistently outperforms all baselines across 9 datasets, with particularly notable gains on heterophilic datasets (Wisconsin/Cornell/Texas).

### Ablation Study

| Variant | Effect |
|---------|--------|
| w/o $\mathcal{L}_{GCL\text{-}OT}$ | Significant performance drop; contrastive learning is critical for structure–text alignment. |
| w/o $\mathcal{L}_{MHA}$ | Substantial degradation on heterophilic datasets (Texas, Cornell). |
| w/o $\mathcal{L}_{LHM}$ | Performance drop on datasets with weak structural signals (Amazon). |
| Full model | Best across all datasets; the three components are mutually complementary. |

### Key Findings

1. **Heterophily is the core challenge**: Replacing GCL-OT with homophilic InfoNCE leads to significant performance degradation in mixed-label neighborhoods and strong semantic heterophily settings.
2. **Robustness advantage**: Under edge perturbation (500 edges removed), GCL-OT+GCN achieves a relative improvement of 24.74% over vanilla GCN; under text perturbation, accuracy remains stable at ~85%.
3. **Effective in unsupervised settings**: Label-free contrastive pre-training followed by linear probing achieves 72.83% on Wisconsin, surpassing dedicated methods such as PolyGCL and HeterGCL.
4. **Training efficiency**: GCL-OT+GCN achieves top accuracy with training time comparable to TAPE.

## Highlights & Insights

- **First work to introduce Optimal Transport into graph contrastive learning for heterophilic TAGs**, representing a conceptually innovative and methodologically complete contribution.
- **Fine-grained analysis of multi-granularity heterophily** (partial / complete / latent) — each type receives a dedicated module.
- The temperature parameter $\beta$ in RealSoftMax elegantly interpolates between $\max$ and $\text{mean}$, yielding a simple yet effective formulation.
- The Filter-Prompt mechanism cleverly employs learnable prompt vectors to "absorb" completely heterophilic noise.
- Theoretical guarantees: tighter mutual information bounds and lower Bayes error upper bounds compared to InfoNCE are formally established.

## Limitations & Future Work

- Dependence on LLM-based text augmentation (GPT-3.5) increases deployment costs and privacy risks.
- The accuracy loss from the low-rank approximation in LRSinkhorn under extreme heterophily is not thoroughly analyzed.
- Only lightweight PLMs such as DistilBERT are evaluated; the performance ceiling with stronger LLM encoders remains unexplored.
- Subgraph sampling may discard global information, and scalability on large graphs is limited (maximum ~170K nodes).
- Hyperparameters ($\lambda$, $\beta$, Sinkhorn iterations) require tuning, although the paper reports low sensitivity to $\beta$.

## Related Work & Insights

- **TAG Learning**: TAPE (He et al. 2024) augments GNNs with LLM-generated explanations; ConGraT (Brannon et al. 2024) performs contrastive pre-training over text and graph modalities; SimTeG demonstrates that high-quality text embeddings alone can substantially improve GNN performance.
- **Heterophilic Graph GCL**: PolyGCL contrasts views using spectral polynomial filters with different homophily levels; HeterGCL exploits label-inconsistent signals via structural and semantic modules.
- **OT + GCL**: THESAURUS aligns node embeddings with semantic prototypes using Gromov-Wasserstein distance; FOSSIL performs subgraph-level contrastive learning with a fused GW distance.

## Rating

⭐⭐⭐⭐ (4/5)

The problem is clearly defined (multi-granularity heterophily), the method is elegantly designed (three targeted modules), and the experiments are comprehensive (9 datasets, supervised + unsupervised settings, ablation, and robustness studies). Theoretical analysis provides formal guarantees in terms of mutual information and Bayes error. Points are deducted primarily for the reliance on LLM augmentation and insufficient large-scale scalability validation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/graph_learning/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](../../NeurIPS2025/graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](../../NeurIPS2025/graph_learning/dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[ICLR 2026\] GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning](../../ICLR2026/graph_learning/graph_homophily_booster_reimagining_the_role_of_discrete_features_in_heterophili.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)

<!-- RELATED:END -->
