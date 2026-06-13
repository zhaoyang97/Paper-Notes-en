---
title: >-
  [Paper Note] On Stealing Graph Neural Network Models
description: >-
  [AAAI2026][Graph Learning][GNN model stealing] This paper demonstrates that under strict query budgets (e.g., only 100 queries), an attacker can efficiently steal a GNN model via a two-stage approach: (1) locally obtaini…
tags:
  - "AAAI2026"
  - "Graph Learning"
  - "GNN model stealing"
  - "model security"
  - "self-supervised learning"
  - "query selection"
  - "black-box attack"
  - "inductive/transductive"
date: 2026-05-08
content_hash: 0362e76db9067a23
---

# On Stealing Graph Neural Network Models

**Conference**: AAAI2026
**arXiv**: [2511.07170](https://arxiv.org/abs/2511.07170)  
**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pręgowska, Tomasz P. Michalak
**Code**: [m-podhajski/OnStealingGNNs](https://github.com/m-podhajski/OnStealingGNNs)  
**Area**: Graph Learning
**Keywords**: GNN model stealing, model security, self-supervised learning, query selection, black-box attack, inductive/transductive

## TL;DR

This paper demonstrates that under strict query budgets (e.g., only 100 queries), an attacker can efficiently steal a GNN model via a two-stage approach: (1) locally obtaining an encoder (randomly initialized or SSL-trained) without interacting with the victim, and (2) strategically selecting queries via K-means clustering. On the Physics dataset, the proposed method achieves 91% accuracy with only 100 queries, whereas the current state-of-the-art requires approximately 5,000 queries plus additional access to victim embeddings to reach comparable performance.

## Background & Motivation

### Security Threats to GNN Models

Graph neural networks are widely deployed in node classification, link prediction, graph classification, and recommender systems, yet they face security threats common to all neural networks. In model stealing attacks, an adversary queries a victim model's API to collect input–output pairs and trains a functionally equivalent surrogate model. A typical defense is to limit the number of queries; however, existing GNN stealing research generally assumes unlimited query access, overlooking the strict query budgets imposed in real-world deployments.

### Overly Permissive Assumptions in Prior Methods

Existing GNN stealing methods (Shen et al., Podhajski et al.) rely on the victim model returning intermediate representations such as embeddings to steal the encoder, and assume an unlimited query budget. Data-free methods, while not requiring data, still demand a large number of queries (e.g., 100 queries × 250 nodes = 25,000 queried nodes). These assumptions are unrealistic in practice—real-world APIs typically return only class labels and enforce strict query quotas.

### A Key Observation from Self-Supervised Learning

An important finding from SSL research: under the inductive setting, a randomly initialized GCN encoder paired with a trained MLP head can achieve performance close to a fully trained model. For example, DGI reports that a random encoder reaches 93.3% on Reddit while SSL training achieves only 94.0%; on Physics, BGRL shows a gap of only 2 percentage points. This suggests that an attacker may not need to query the victim at all to obtain an encoder—random initialization suffices in the inductive setting, and local SSL training is effective in the transductive setting.

## Core Problem

Under the hardest black-box scenario—where the attacker can only obtain class labels (not embeddings) and queries are strictly limited—how can a GNN model be stolen efficiently?

## Method

### Overall Architecture (Three Stages)

1. **Local Encoder Acquisition**: Obtain a feature extractor locally without interacting with the victim.
2. **Strategic Query Selection**: Use the embedding space produced by the encoder to select the most informative nodes to query.
3. **MLP Head Training**: Train an MLP using the class labels obtained from queries, then combine it with the encoder to form the surrogate model.

### Threat Model

- **Black-box setting**: The attacker has no knowledge of the victim's parameters, architecture, or training data $\mathbf{G}_V$.
- **Query budget**: At most $q_n$ queries are allowed, each returning only a class label.
- **Data assumption**: The attacker possesses an unlabeled graph $\mathbf{G}_D$ drawn from the same distribution as the victim's training data.

### Stage 1: Encoder Acquisition

**Inductive setting**: A randomly initialized GCN is used directly as the encoder, with no interaction with the victim whatsoever. T-SNE visualizations confirm that a random encoder already produces structured embeddings under the inductive setting, with nodes of each class forming distinct clusters.

**Transductive setting**: Self-supervised learning (LaGraph) is applied to locally train an encoder on the attacker's full dataset $\mathbf{G}_D$. Transductive graphs are typically small (e.g., Cora with 2,708 nodes), making SSL training computationally inexpensive. SSL yields substantial gains in this setting: on Cora, accuracy improves from 69.3% to 82.3% (+13.0%).

### Stage 2: Query Selection

Using the embeddings $\mathbf{H} = f(\mathbf{X}_D, \mathbf{A}_D) \in \mathbb{R}^{n \times b}$ produced by the encoder, K-means is applied to partition nodes into $q_n$ clusters. The node closest to each cluster centroid is selected as a query node $\{v_1', \ldots, v_{q_n}'\}$. This ensures that queries cover the entire input space, maximizing information gain per query—analogous to diversity sampling in active learning.

### Stage 3: MLP Training

The victim is queried on the selected nodes to obtain labels $\{y_1, \ldots, y_{q_n}\}$. The MLP component $g$ is trained with a cross-entropy loss:
$$\hat{y} = f_s(\mathbf{X}, \mathbf{A}) = g(f(\mathbf{X}, \mathbf{A}))$$
where $f$ is the encoder, $g$ is the MLP head, and $f_s$ is the final surrogate model.

## Key Experimental Results

### Inductive Setting (Target: SAGE, Surrogate: GCN, $q_n=100$)

| Method | Reddit Acc | CS Acc | Physics Acc | Photo Acc | WikiCS Acc |
|--------|-----------|--------|------------|-----------|------------|
| Target (victim) | 94.8 | 93.9 | 96.0 | 93.0 | 72.5 |
| E2E | 47.0±4.5 | 73.6±3.9 | 89.9±1.1 | 81.2±0.8 | 61.6±1.3 |
| Shen et al.* | 77.2±5.1 | 77.7±0.8 | 90.6±0.5 | 84.4±0.8 | 64.9±1.0 |
| Podhajski et al.* | 79.9±4.1 | 78.0±0.5 | 89.9±0.2 | 84.0±1.0 | 64.0±1.1 |
| datafree | 13.6±4.1 | 24.8±2.8 | 55.5±5.0 | 24.9±2.8 | 38.6±2.1 |
| **R-init+Select (Ours)** | **82.5±1.2** | **78.4±2.1** | **91.2±0.4** | **86.8±1.0** | **65.5±1.8** |

*Methods marked with * require additional access to victim embeddings (weaker threat model).

### Transductive Setting (Target: GCN, Surrogate: GCN, $q_n=10$)

| Method | Cora Acc | Cora Fid | Citeseer Acc | Citeseer Fid | Pubmed Acc | Pubmed Fid |
|--------|---------|---------|-------------|-------------|-----------|----------|
| Target | 83.3 | — | 72.1 | — | 80.0 | — |
| E2E | 47.5±3.7 | 45.7±1.0 | 37.2±6.1 | 41.1±7.5 | 61.0±4.9 | 67.5±5.0 |
| datafree | 18.1±2.7 | 21.1±3.9 | 22.1±3.3 | 23.1±3.8 | 33.2±2.9 | 33.4±3.0 |
| SSL+Random | 56.1±2.7 | 56.8±3.0 | 51.3±5.1 | 57.6±5.5 | 66.1±7.3 | 72.7±9.0 |
| **SSL+Select (Ours)** | **69.9±1.2** | **72.5±1.3** | **66.3±1.9** | **72.4±2.3** | **67.0±6.0** | **80.1±4.7** |

### Random Initialization vs. SSL-Trained Encoder

| Setting | Dataset | Random Acc | SSL-Trained Acc | Gain |
|---------|---------|-----------|----------------|------|
| Inductive | Reddit | 93.3 | 94.0 | +0.7 |
| Inductive | Physics | 93.7 | 95.7 | +2.0 |
| Transductive | Cora | 69.3 | 82.3 | +13.0 |
| Transductive | Citeseer | 61.9 | 71.8 | +9.9 |

### Robustness Under Defense

Under a 10% prediction-flipping defense, the proposed method retains the highest performance across all settings, indicating that such defenses provide limited protection.

## Highlights & Insights

- **First study of GNN stealing under strict query budgets**: The problem is decomposed into two independent stages—encoder acquisition and head stealing—revealing a serious security threat that had been previously overlooked.
- **Effectiveness of randomly initialized encoders**: In the inductive setting, a randomly initialized GCN encoder already produces high-quality feature representations, enabling an attacker to obtain the core of the surrogate model without any interaction with the victim.
- **Extreme resource efficiency**: The attack requires only 100 queries and a single commodity CPU (AMD EPYC 7742), compared to Shen et al., which requires ~5,000 queries, a GPU, and victim embeddings—representing approximately a 15× improvement in query efficiency.
- **Effectiveness of K-means query selection**: Compared to random selection and other active learning strategies (farthest-first, coreset herding, etc.), K-means consistently achieves the best results across all datasets.
- **Coverage of both inductive and transductive settings**: This is the first GNN stealing method demonstrated to be effective under both paradigms.

## Limitations & Future Work

- **Data distribution assumption**: The attacker is assumed to possess unlabeled graph data $\mathbf{G}_D$ drawn from the same distribution as the victim's training data, which may not fully hold in practice.
- **Node-level tasks only**: The current work focuses solely on node classification and does not address other GNN tasks such as graph classification or link prediction.
- **Surrogate architecture selection**: Although the method does not require architectural matching, using GCN as the surrogate for all victim architectures may be suboptimal.
- **Limited defense discussion**: Only prediction-flipping defense is evaluated; more sophisticated mechanisms such as watermarking or differential privacy are not considered.

## Related Work & Insights

- **vs. Shen et al. / Podhajski et al.**: Both methods require the victim to return embeddings (a weaker threat model) and assume unlimited queries. The proposed method surpasses them using only class labels and 100 queries.
- **vs. datafree**: While data-free methods require no data, they demand a large number of queries (25,000 nodes); under a budget of 100 queries, their performance degrades severely (55.5% on Physics), whereas the proposed method achieves 91.2%.
- **vs. wu2021model**: Limited to the transductive setting and assumes unlimited queries; the proposed method achieves better results under stricter constraints.

The effectiveness of randomly initialized GNN encoders is related to the Weisfeiler-Lehman graph isomorphism test—the structure-awareness of GNNs derives partly from the architecture itself rather than from learned weights alone. The K-means query selection strategy aligns conceptually with diversity sampling in active learning, and further active learning strategies may be worth exploring. The security threats identified in this work have direct practical implications for the deployment of GNN-as-a-Service (GNNaaS) systems.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to study GNN stealing under strict query budgets; the insight of decoupling encoder acquisition from head stealing is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 datasets, two settings, multiple victim/surrogate architectures, defense evaluation, and McNemar's test.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, the method is presented progressively, and Table 1 provides an at-a-glance comparison of methods.
- Value: ⭐⭐⭐⭐ — Concretely exposes security vulnerabilities in GNNs and serves as a meaningful warning for the secure deployment of GNN-based services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](../../ICML2026/graph_learning/full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [\[ICML 2026\] Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design](../../ICML2026/graph_learning/beyond_model_base_retrieval_weaving_knowledge_to_master_fine-grained_neural_netw.md)
- [\[AAAI 2026\] BugSweeper: Function-Level Detection of Smart Contract Vulnerabilities Using Graph Neural Networks](bugsweeper_function-level_detection_of_smart_contract_vulnerabilities_using_grap.md)
- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)

</div>

<!-- RELATED:END -->
