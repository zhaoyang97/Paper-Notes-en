---
title: >-
  [Paper Note] Spatio-Temporal Directed Graph Learning for Account Takeover Fraud Detection
description: >-
  [NeurIPS 2025 (Workshop on New Perspective in Graph Machine Learning)][Graph Learning][GraphSAGE] This paper proposes ATLAS, a framework that reformulates account takeover (ATO) fraud detection as a node classification p…
tags:
  - "NeurIPS 2025 (Workshop on New Perspective in Graph Machine Learning)"
  - "Graph Learning"
  - "GraphSAGE"
  - "spatio-temporal directed graph"
  - "fraud detection"
  - "label propagation"
  - "causal inference"
date: 2026-05-08
content_hash: d36ce0e445e67d67
---

# Spatio-Temporal Directed Graph Learning for Account Takeover Fraud Detection

**Conference**: NeurIPS 2025 (Workshop on New Perspective in Graph Machine Learning)  
**arXiv**: [2509.20339](https://arxiv.org/abs/2509.20339)  
**Code**: N/A  
**Area**: Graph Learning / Fraud Detection  
**Keywords**: GraphSAGE, spatio-temporal directed graph, fraud detection, label propagation, causal inference

## TL;DR

This paper proposes ATLAS, a framework that reformulates account takeover (ATO) fraud detection as a node classification problem on spatio-temporal directed graphs. By constructing causal directed graphs via temporal windows and nearest-neighbor constraints, and combining lag-aware label propagation with a GraphSAGE encoder, ATLAS achieves a +6.38% AUC improvement and over 50% reduction in user friction on a production graph at Capital One with 100M nodes and 1B edges.

## Background & Motivation

**Background**: ATO fraud detection in consumer finance is a high-stakes task—attackers steal credentials to control legitimate accounts and initiate high-risk transactions (HRTs). Production systems predominantly rely on tabular gradient boosting models such as XGBoost, which score each session independently. Although deep architectures including fully connected networks, RNNs, and Transformers have been explored, none has consistently outperformed XGBoost under the same latency and reliability constraints.

**Limitations of Prior Work**: The i.i.d. assumption underlying XGBoost's row-wise scoring ignores two critical structures: (1) *relational structure*—multiple suspicious sessions may share the same device fingerprint, IP address, or account ID, forming fraud rings; and (2) *temporal structure*—causal ordering and temporal recency are essential for assessing the risk of a given session. These cross-session signals cannot be captured by any row-wise model.

**Key Challenge**: Production environments impose strict latency requirements (<250ms), while graph models must perform neighborhood sampling and message passing over graphs with 100M+ nodes. Additionally, labels are delayed—fraud labels are confirmed through adjudication after the event (adjudication time $\tau_u$), and training must strictly avoid using future information unavailable at inference time (data leakage).

**Goal**: How can cross-session relational and temporal structures be leveraged to improve ATO detection while satisfying production latency constraints? How can training–inference consistency be ensured without leakage?

**Key Insight**: ATO detection is reformulated from tabular classification to node classification on a spatio-temporal directed graph. The key observation is that sessions sharing identifiers (account, device, IP) exhibit directed causal relationships that can be exploited via GNN message passing, with strict temporal constraints guaranteeing causality.

**Core Idea**: A causal directed session graph is constructed using temporal window and nearest-neighbor constraints. Combined with lag-aware label propagation that supplies high-signal neighborhood features to GraphSAGE, ATLAS upgrades ATO detection from independent row scoring to graph-structured reasoning.

## Method

### Overall Architecture

ATLAS comprises three core components: (1) temporally-respectful directed session graph construction; (2) inference-consistent lag-aware label propagation; and (3) an inductive GraphSAGE-based GNN encoder. The input consists of HRT session nodes with tabular features, and the output is a fraud risk probability $s_v \in [0,1]$ for each session.

### Key Designs

1. **Temporally-Respectful Directed Graph Construction**

    - **Function**: Organizes independent sessions into a causal directed graph, exposing cross-session relational and temporal patterns.
    - **Mechanism**: Each node is uniquely identified by (account\_id, device\_id, ip\_address, timestamp). A directed edge $u \to v$ is added if sessions $u$ and $v$ share any identifier and $t_u < t_v$. Edges are typed by identifier $m \in \{\text{account}, \text{device}, \text{IP}\}$. Connectivity is governed by two constraints: a temporal window $T$ (connecting only nodes with $0 < t_v - t_u \leq T$) and a nearest-neighbor cap $K$ (retaining at most $K$ most recent predecessors per edge type per node).
    - **Design Motivation**: The temporal window enforces causal ordering (yielding a DAG), while the nearest-neighbor cap controls node degree to meet latency budgets. The three edge types allow the model to distinguish different co-occurrence patterns (e.g., sessions sharing a device vs. an IP).

2. **Lag-Aware Label Propagation**

    - **Function**: Provides each node with high-signal features derived from historically confirmed fraud labels, while strictly preventing data leakage.
    - **Mechanism**: For a target node $v$, its predecessors within the temporal window are collected as $R(v)$ (at most $K$ nodes), and filtered to the subset $A(v)$ where adjudication time $\tau_u \leq t_v$ (i.e., labels that are genuinely known at inference time). Four aggregate statistics are computed from $A(v)$: $n^{\text{lab}}_v$ (number of neighbors with known labels), $n^{\text{fraud}}_v$ (number of known fraudulent neighbors), $r_v$ (empirical fraud rate), and $a_v$ (indicator of any upstream fraud). These are concatenated to the node's raw features: $h^{(0)}_v = [x_v; \ell_v]$.
    - **Design Motivation**: Fraud labels are delayed due to adjudication, so using all neighbor labels naively causes training–inference inconsistency. The lag filter ensures that labels visible at training time match those available at inference time. The four simple aggregation features already encode the critical signal of whether an upstream fraud link exists.

3. **GraphSAGE Encoder (Multiple Variants)**

    - **Function**: Learns node representations through neighborhood sampling and message passing.
    - **Mechanism**: Three variants are explored: (1) *Homogeneous GraphSAGE*—standard mean aggregation $m_v^{(k)} = \text{AGG}(\{h_u^{(k-1)}: u \in S^{(k)}(v)\})$; (2) *Relational GraphSAGE*—separate aggregation per edge type followed by fusion $m_v^{(k)} = \sum_m \Phi_m^{(k)}(\text{AGG}_m(\cdot))$; (3) *Temporal attention variant*—attention aggregation incorporating time differences $\Delta t$ and edge-type embeddings. The neighborhood sampler applies the same $(T, K)$ constraints at both training and inference. In practice, shallow networks ($L \in \{2,3\}$) with moderate fanout suffice.
    - **Design Motivation**: Inductive learning supports a continuously growing graph (new sessions arrive over time). Neighborhood sampling enables mini-batch training and maintains consistency with inference to avoid bias. Relational and attention variants are theoretically more expressive but experiments show the homogeneous variant is already sufficient.

### Loss & Training

Weighted binary cross-entropy loss is applied to handle extreme class imbalance. Decision thresholds are calibrated against a target friction envelope. A time-based split is used: 8 months for training, 2 months for validation, and 5 months for testing (non-overlapping). Numerical features are standardized using training-set statistics only. PyTorch Geometric's NeighborLoader is used for efficient out-of-core neighborhood sampling.

## Key Experimental Results

### Main Results

| Model | AUC Overall | AUC Segment 1 | AUC Segment 2 |
|------|-------------|---------------|---------------|
| XGBoost | 79.83 | 78.88 | 82.45 |
| GNN (w/o label propagation) | 82.27 (+3.06%) | 81.59 (+3.43%) | 83.82 (+1.66%) |
| **GNN + Label Propagation** | **84.46 (+5.8%)** | **83.92 (+6.38%)** | **85.45 (+3.63%)** |

### Ablation Study

| Configuration | Key Finding |
|------|---------|
| K: 1→10 | AUC improves steadily with more nearest neighbors; more historical sessions are beneficial. |
| T: 1→120 days | AUC improves consistently with a larger temporal window; longer temporal context is valuable. |
| GNN vs. GNN+LP | Label propagation contributes +2.2% AUC, the largest single improvement. |
| Relational/attention variants vs. homogeneous | Additional architectural complexity yields limited gain; most improvement stems from graph structure itself. |

### Key Findings

- **Label propagation is the dominant contributor**: GNN alone outperforms XGBoost by +3.06%; adding label propagation yields a further +2.8%, for a total gain of +5.8%. This demonstrates that whether upstream neighbors have been flagged as fraudulent is an extremely strong signal.
- **Simple architectures suffice**: Homogeneous GraphSAGE with label propagation achieves optimal performance; relational and attention variants bring no significant additional gain. The improvements are primarily attributable to graph-structural modeling rather than more complex GNN architectures.
- **Hyperparameter analysis is intuitively consistent**: Both larger $K$ and larger $T$ yield consistent improvements, confirming that more historical context and more associated sessions are valuable.
- Production deployment achieves over 50% reduction in user friction—substantially reducing disruption to legitimate users while improving fraud capture.

## Highlights & Insights

- **Graph-structural reformulation yields far greater gains than model architecture upgrades**: Over many years, DNNs, RNNs, and Transformers all failed to surpass XGBoost; yet once the problem is modeled as a graph, a simple GraphSAGE achieves a significant breakthrough. This demonstrates that on relational data, the correct data representation matters more than model complexity.
- **Lag-aware label propagation is elegantly designed**: The condition $\tau_u \leq t_v$ rigorously guarantees training–inference consistency, while four simple aggregation statistics encode the critical fraud-chain signal. This concise yet leakage-free feature design is applicable to any online system with delayed labels.
- **Industrial viability**: Deployable inference within latency constraints is achieved on a graph with 100M nodes and 1B edges, demonstrating the feasibility of GNNs in real-world financial systems.

## Limitations & Future Work

- **Workshop paper scope**: The experimental section is relatively brief, lacking detailed ablation results (e.g., tabular comparison of different GNN variants, isolated ablation of individual label propagation features).
- **Data confidentiality**: Due to data sensitivity, descriptive statistics of the dataset are not reported, and experiments cannot be reproduced.
- **Static graph assumption**: Although NeighborLoader is used to handle a growing graph, temporal drift in graph structure (concept drift) and model update strategies are not discussed.
- **Single product line evaluation**: Validation is conducted on only two segments of a single Capital One digital product; generalizability to additional product lines and institutions remains to be demonstrated.
- **No comparison with other graph-based fraud detection methods**: Methods such as temporal GNNs (TGAT, TGN) or heterogeneous graph approaches (HGT) are not compared; the only baseline is XGBoost.

## Related Work & Insights

- **vs. XGBoost tabular methods**: XGBoost's row-wise independent scoring cannot capture relational structure, yet remains the industry baseline. ATLAS complements XGBoost by incorporating cross-session associations that tabular models inherently miss.
- **vs. temporal graph methods (TGN/TGAT)**: These methods are more general but are not designed for the delayed-label and extreme class-imbalance challenges of fraud detection. ATLAS's lag-aware label propagation is an important innovation tailored to the financial domain.
- **vs. GCN/GAT**: GraphSAGE is chosen primarily for its inductive learning capability and mini-batch sampling support, which are well-suited to a continuously growing large-scale graph.
- **Inspiration**: The lag-aware label propagation approach is transferable to any online system with delayed feedback (e.g., advertising fraud, review manipulation), and the graph-structural reformulation methodology is applicable to other structured anomaly detection scenarios.

## Rating

- Novelty: ⭐⭐⭐ — The spatio-temporal directed graph modeling and lag-aware label propagation are clear and practical, though GraphSAGE itself is not a novel method.
- Experimental Thoroughness: ⭐⭐⭐ — Large-scale real-data validation is convincing, but comparisons with other graph methods are absent and ablation details are insufficient.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, mathematical notation is rigorous, and the descriptions of graph construction and label propagation are precise.
- Value: ⭐⭐⭐⭐ — The industrial-scale GNN deployment case study has high practical reference value; 50%+ friction reduction represents a substantial business impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] WSGG: Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](../../CVPR2026/graph_learning/wsgg_spatiotemporal_world_scene_graph.md)
- [\[NeurIPS 2025\] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction](tami_taming_heterogeneity_in_temporal_interactions_for_temporal_graph_link_predi.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] GraphFaaS: Serverless GNN Inference for Burst-Resilient, Real-Time Intrusion Detection](graphfaas_serverless_gnn_inference_for_burst-resilient_real-time_intrusion_detec.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)

</div>

<!-- RELATED:END -->
