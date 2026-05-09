---
title: >-
  [Paper Note] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory
description: >-
  [AAAI2026][Graph Learning][hypergraph learning] This paper proposes CIF, which leverages hypergraphs to extract intra-class structural commonalities from a small number of training samples, guiding memory bank construction and retrieval for few-shot multimodal industrial anomaly detection, achieving state-of-the-art performance.
tags:
  - AAAI2026
  - Graph Learning
  - hypergraph learning
  - few-shot anomaly detection
  - multimodal industrial anomaly detection
  - memory bank
  - training-free message passing
date: 2026-05-08
content_hash: f75de15bc2642054
---

# Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory

**Conference**: AAAI2026
**arXiv**: [2511.05966](https://arxiv.org/abs/2511.05966)
**Code**: [Sunny5250/CIF](https://github.com/Sunny5250/CIF)
**Area**: Graph Learning
**Keywords**: hypergraph learning, few-shot anomaly detection, multimodal industrial anomaly detection, memory bank, training-free message passing

## TL;DR

This paper proposes CIF, which leverages hypergraphs to extract intra-class structural commonalities from a small number of training samples, guiding memory bank construction and retrieval for few-shot multimodal industrial anomaly detection, achieving state-of-the-art performance.

## Background & Motivation

Industrial anomaly detection is a critical technique for ensuring product quality. Most existing methods require large amounts of normal samples to construct feature distributions, yet in real industrial settings normal samples are often scarce. Few-shot anomaly detection has thus been proposed, with the central challenge being: **a small number of training samples cannot cover the diverse normal patterns observed in test samples**, resulting in insufficient memory bank coverage and high false positive rates.

Existing methods such as PatchCore perform anomaly detection via nearest-neighbor search over patch features; GraphCore employs GNNs to aggregate neighborhood information for extracting isometry-invariant visual features. However, ordinary graphs can only model pairwise relations and struggle to capture higher-order associations among multiple patches. The authors observe that **same-category samples in single-semantic industrial images exhibit highly consistent structural patterns**, and hypergraphs are better suited to model such higher-order structural commonalities, thereby enhancing memory bank coverage under the few-shot setting.

## Core Problem

1. In the few-shot setting, memory bank features are sparse and cannot adequately cover the normal patterns of test samples.
2. A distribution gap exists between test features and memory bank features.
3. Direct nearest-neighbor matching lacks structural guidance, leading to high false positive rates.

## Method

CIF (Commonality In Few) is a hypergraph-based few-shot unsupervised multimodal industrial anomaly detection framework consisting of four core modules:

### 1. Semantics-Aware Hypergraph Construction (SAHC)

Traditional hard clustering (K-Means) cannot assign nodes to multiple hyperedges, while fuzzy clustering (Fuzzy C-Means) leads to imbalanced hyperedge distributions on single-semantic industrial images. SAHC proceeds as follows:

- Patch features $X = [x_1, x_2, \ldots, x_N]$ are extracted using a pretrained feature extractor, with each patch treated as a node.
- A foreground mask is extracted from 3D point clouds to filter foreground nodes $V_{\text{fore}}$.
- K-Means clustering is applied to foreground nodes to obtain $|\mathcal{E}|$ cluster centers as hyperedge centers.
- Cosine similarities between all foreground nodes and hyperedge centers are computed, min-max normalized, and thresholded to determine node membership.
- An incidence matrix $\mathbf{H} \in \mathbb{R}^{|V| \times |\mathcal{E}|}$ (soft assignment) and a hard incidence matrix $\mathbf{H}_{\text{hard}}$ (each node assigned only to its most similar hyperedge) are generated.

Key design: the hypergraph constructed from RGB image features is uniformly used to support both 2D and 3D modalities, since hypergraphs built from 3D point cloud features suffer from severe imbalance.

### 2. Structure-Guided Memory Sampling (SGMS)

Intra-class structural commonalities guide memory bank construction and compression:

- **Node assignment**: distances between hyperedge features of new training samples and those of the memory bank are computed, and nodes of each hyperedge in the new sample are merged into the most similar memory bank hyperedge.
- **Hyperedge update**: hyperedge features are recomputed after merging.
- **Memory sampling**: greedy coreset sampling is performed independently within each hyperedge (rather than globally), ensuring representative features from each structural region. If fewer than one node remains after sampling within a hyperedge, the node with the smallest maximum distance to all other nodes is retained.

### 3. Bidirectional Training-Free Hypergraph Message Passing (Bi-TF-MP)

To bridge the distribution gap between test features and memory bank features:

- A **joint hypergraph** is constructed by concatenating the test sample hypergraph, the memory bank hypergraph, and cross-domain hyperedges.
- Cross-domain hyperedges are built by finding the top-$k$ most similar memory nodes for each test node, and vice versa.
- Joint incidence matrix: $\mathbf{H}^{joint} = [\widetilde{\mathbf{H}}^{test} | \widetilde{\mathbf{H}}^{mem} | \mathbf{H}^{cross}]$
- A training-free message passing kernel $\mathbf{S}$ from TF-MP is adopted; $L$ layers of propagation allow nodes to exchange information with $L$-hop neighbors.
- A retention coefficient $\alpha = 0.9$ ensures that each node's own information dominates, performing only mild distribution alignment.

### 4. Hyperedge-Guided Memory Search (HGMS)

A two-stage search reduces false positives:

- **Stage 1 (structure matching)**: cosine similarities between updated hyperedge features of the test sample and those of the memory bank are computed; top-$k$ most similar memory hyperedges are selected for each test hyperedge.
- **Stage 2 (patch matching)**: patch-level nearest-neighbor search is performed within the matched hyperedge subset, yielding anomaly scores $\mathcal{A}_{ij} = \min_{m \in \mathcal{M}_{sub\,i}} \|X^{test}_{ij} - m\|_2$.
- Scores from the conventional global patch-level search are also retained, and element-wise multiplication of the two yields the final anomaly score.

**Feature extractors**: DINO for 2D and PointMAE for 3D, both pretrained models requiring no fine-tuning.

## Key Experimental Results

Evaluated on two multimodal datasets, MVTec 3D-AD and Eyecandies:

**MVTec 3D-AD (I-AUROC / AUPRO)**:

| Setting | CIF (training-free) | Best training method | Patchcore+FPFH (training-free) |
|---------|--------------------|--------------------|-------------------------------|
| 1-shot | 72.0 / 86.1 | M3DM 73.9 / CFM 91.4 | 59.9 / 88.3 |
| 2-shot | 73.2 / 87.2 | M3DM 76.5 / CFM 92.5 | 61.4 / 88.6 |
| 4-shot | 77.6 / 89.6 | CFM 80.1 / CFM 94.0 | 64.3 / 90.4 |

- Achieves comprehensive state-of-the-art among training-free methods; I-AUROC exceeds Patchcore+FPFH by approximately 12–20%.
- The gap with training-based methods is small (only 2.6% at 1-shot), with no training required.

**Eyecandies (I-AUROC)**: 1-shot 69.5, 2-shot 73.6, 4-shot 75.1, surpassing training-based methods M3DM and CFM across all settings.

**Ablation Study (MVTec 3D-AD, 1-shot)**:

| Module | I-AUROC | AUPRO |
|--------|---------|-------|
| No modules | 68.6 | 76.1 |
| +SGMS | 71.2 | 85.0 |
| +SGMS+HGMS | 71.7 | 86.0 |
| +SGMS+HGMS+Bi-TF-MP (full) | 72.0 | 86.1 |

SGMS contributes the most (I-AUROC +3.8%, AUPRO +11.7%), demonstrating the effectiveness of structure-guided sampling.

## Highlights & Insights

1. **First exploration of hypergraphs for industrial anomaly detection**: Hypergraphs capture higher-order patch associations and are better suited than ordinary graphs for modeling structural commonalities in single-semantic industrial images.
2. **Fully training-free**: No fine-tuning or parameter training is required; only pretrained features and hypergraph message passing are used.
3. **Structural priors throughout the pipeline**: Hypergraph structural information is consistently utilized across memory bank construction, feature alignment, and retrieval matching.
4. **Pronounced few-shot advantage**: The largest improvement over baselines is observed at 1-shot, indicating that structural information is most valuable when data is scarce.

## Limitations & Future Work

1. **Limited anomaly localization**: AUPRO does not reach the optimum on either dataset; the authors acknowledge that localization remains a weakness.
2. **Hyperparameter sensitivity**: The number of hyperedges must be manually configured per dataset (4 for MVTec 3D-AD, 8 for Eyecandies), lacking an adaptive mechanism.
3. **Dependence on 3D point clouds for foreground segmentation**: This limits applicability in purely 2D settings.
4. **Gap with training-based methods**: A non-trivial gap remains compared to methods such as CFM, particularly on localization metrics.
5. **Scalability to larger datasets**: Validation is currently limited to small- to medium-scale datasets.

## Related Work & Insights

| Method | Type | Structural Modeling | Training Required | 1-shot I-AUROC |
|--------|------|--------------------|--------------------|----------------|
| PatchCore+FPFH | Training-free | None | No | 59.9 |
| GraphCore | Training-based | Ordinary graph (pairwise) | GNN training | — |
| M3DM | Training-based | None | Yes | 73.9 |
| CFM | Training-based | None | Yes | 67.5 |
| **CIF** | **Training-free** | **Hypergraph (higher-order)** | **No** | **72.0** |

Compared to GraphCore, CIF replaces ordinary graphs with hypergraphs to capture higher-order associations. Compared to training-based methods such as M3DM and CFM, CIF approaches their performance without any training.

The advantages of hypergraphs in structured data modeling are transferable to other vision tasks requiring higher-order relational modeling. Training-free message passing offers a lightweight feature alignment strategy applicable to other few-shot scenarios. Structure-guided sampling within hyperedges (independent per-hyperedge sampling) better preserves representativeness than global sampling, a principle generalizable to other memory bank methods. The two-stage search strategy that incorporates structural priors into the retrieval phase is broadly applicable for reducing false positives.

## Rating
- **Novelty**: 8/10 — First application of hypergraphs to few-shot industrial anomaly detection; the overall framework is elegantly designed.
- **Experimental Thoroughness**: 7/10 — Two datasets with detailed ablations, but additional datasets and unimodal experiments are absent.
- **Writing Quality**: 8/10 — Clear structure with well-illustrated figures and tables.
- **Value**: 7/10 — A training-free method achieving near training-based performance, though localization metrics leave room for improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/graph_learning/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](../../NeurIPS2025/graph_learning/moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[AAAI 2026\] BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks](bugsweeper_function-level_detection_of_smart_contract_vulnerabilities_using_grap.md)

</div>

<!-- RELATED:END -->
