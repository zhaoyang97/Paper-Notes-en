---
title: >-
  [Paper Note] Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning
description: >-
  [AAAI 2026][Autonomous Driving][Road network representation learning] This paper proposes DST (Dual-branch Spatial-Temporal), a road network representation learning framework that jointly models spatial heterogeneity and…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Road network representation learning"
  - "self-supervised learning"
  - "graph neural networks"
  - "hypergraph"
  - "spatial-temporal modeling"
date: 2026-05-08
content_hash: 8db39539fb2c7806
---

# Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning

**Conference**: AAAI 2026
**arXiv**: [2511.06633](https://arxiv.org/abs/2511.06633)  
**Code**: [Available](https://github.com/chaser-gua/DST)  
**Area**: Autonomous Driving
**Keywords**: Road network representation learning, self-supervised learning, graph neural networks, hypergraph, spatial-temporal modeling

## TL;DR

This paper proposes DST (Dual-branch Spatial-Temporal), a road network representation learning framework that jointly models spatial heterogeneity and temporal dynamics via a spatial branch (mix-hop transition matrix + graph–hypergraph contrastive learning) and a temporal branch (Transformer encoder + next-token prediction + weekday/weekend classification). DST achieves state-of-the-art performance on three downstream tasks across three cities.

## Background & Motivation

Road Network Representation Learning (RNRL) aims to learn general low-dimensional vector representations for road segments, enabling downstream tasks such as traffic inference, travel time estimation, and trajectory destination prediction. Existing methods face two key challenges:

**1. Spatial heterogeneity**: Road segment similarity depends not only on geographic distance but also on functional attributes, trajectory reachability, and other multi-dimensional relationships. For example, two distant road segments may share similar functions (e.g., both in residential areas), while adjacent segments may serve entirely different functions. The neighborhood smoothing mechanism of GNNs struggles to capture such long-range functional similarity.

**2. Temporal dynamics**: Different types of road segments exhibit significantly different traffic patterns across time periods, with weekday and weekend patterns being markedly distinct. Road network topology alone is insufficient to characterize this temporal variation.

Early methods (Node2Vec, GCN, GAE) are limited to simple graph structures; more recent methods (Toast, JCLRNT, TrajRNE) leverage trajectory information but lack temporal modeling; DyToast incorporates sinusoidal temporal features but suffers performance degradation on speed inference tasks.

## Method

### Overall Architecture

DST adopts a dual-branch architecture in which the spatial and temporal branches are **pre-trained independently** and subsequently fused for downstream tasks:

- **Spatial branch**: GNN + hypergraph contrastive learning → captures spatial topology and high-order semantic relationships of the road network
- **Temporal branch**: Transformer encoder + dual self-supervised tasks → models 24-hour traffic dynamic patterns
- **Fusion strategy**: Concatenation of three representations for downstream tasks

### Key Designs

**1. Mix-hop Transition Matrix Weighting**

Multi-hop reachability relationships between road segments are extracted from trajectory data to construct a mix-hop transition matrix:

$$P_{hop}[r_i, r_j] = \sum_{\tau \in \mathcal{T}} \sum_{1 \leq i < j \leq m} m - (j - i)$$

Weighting strategy: smaller hop counts receive higher initial weights, while larger hop counts receive lower weights, thereby emphasizing adjacent connections while incorporating reachable long-distance connections. After row normalization, the matrix is used to initialize a learnable weight matrix for feature weighting prior to the GNN:

$$Z_{hop} = \tilde{P}_{hop} \cdot Z_{\mathcal{I}}$$

**2. Spatial Semantic Graph–Hypergraph Contrastive Learning**

Two views are constructed for road network representation:

**Graph view**: Multi-layer GAT encodes the road network topology, utilizing edge features $X_\mathcal{E}$ (connectivity attributes between road segments) to produce spatial representations $Z_\mathcal{G}$.

**Hypergraph view**: Three types of hyperedges are constructed to capture high-order relationships:
- $\mathcal{E}_{\mathcal{H}_1}$: Functional region hyperedges — road segments are clustered via spectral clustering; segments in the same cluster share a hyperedge
- $\mathcal{E}_{\mathcal{H}_2}$: Same-type hyperedges — road segments of the same type share a hyperedge regardless of geographic distance
- $\mathcal{E}_{\mathcal{H}_3}$: Adjacent unidirectional hyperedges — geographically adjacent unidirectional roads share a hyperedge (Tobler's First Law of Geography)

HGNN+ encodes the hypergraph to produce semantic representations $Z_\mathcal{H}$.

Contrastive learning maximizes the mutual information between representations of the same road segment across the two views:

$$\mathcal{L}_{\mathcal{GH}} = -\frac{1}{N}\sum_{r_i \in \mathcal{R}} \left[\frac{1}{|\mathcal{H}(r_i)|}\sum_{r_j \in \mathcal{H}(r_i)} I(v_{r_i}, h_{r_j})\right]$$

**3. Temporal Dynamics Modeling**

Traffic dynamics are defined as $\mathcal{D}_\mathcal{R} \in \mathbb{R}^{N \times 24 \times 2}$, representing the 24-hour visit count sequence for each road segment with two channels (weekday/weekend).

The Transformer encoder takes the last hidden state as a compressed sequence representation:

$$Z_\mathcal{D} = \text{TransEnc}(\text{PosEnc}(\mathcal{D}_\mathcal{R}))[-1]$$

Two self-supervised tasks are jointly optimized:
- **Dynamic prediction** (regression): predicts next-step traffic volume from historical sequences
$$\mathcal{L}_{reg} = \frac{1}{N \times C}\sum_{i=1}^{N \times C}\|y_i - \hat{y}_i\|^2$$
- **Dynamic classification**: distinguishes whether the input sequence corresponds to a weekday or weekend
$$\mathcal{L}_{cls} = -\frac{1}{N \times C}\sum_{i}\sum_c y_i(c)\log(\hat{y}_i(c))$$

### Loss & Training

Total temporal branch loss: $\mathcal{L}_d = \lambda_{reg} \cdot \mathcal{L}_{reg} + \lambda_{cls} \cdot \mathcal{L}_{cls}$

The spatial and temporal branches are **pre-trained separately**, and the three representations ($Z_\mathcal{G}, Z_\mathcal{H}, Z_\mathcal{D}$) are concatenated for downstream tasks. Sensitivity analysis shows that increasing the weight of $\lambda_{reg}$ improves performance, since the two tasks differ substantially in initial loss magnitude and require weight adjustment for task balance.

## Key Experimental Results

### Main Results

**Table 1: Destination Prediction and Travel Time Estimation (three cities)**

| Method | Beijing ACC@1↑ | Beijing MRR↑ | Porto ACC@1↑ | Xi'an ACC@1↑ |
|--------|---------------|-------------|-------------|-------------|
| Node2Vec | 0.1954 | 0.2884 | 0.2201 | 0.4088 |
| TrajRNE | 0.6728 | 0.7603 | 0.6728 | 0.8260 |
| JCLRNT | 0.4222 | 0.5528 | 0.5133 | 0.6752 |
| **DST** | **0.7288** | **0.8213** | **0.6766** | **0.8335** |

**Table 2: Speed Inference Task**

| Method | Beijing MAE↓ | Porto MAE↓ | Xi'an MAE↓ |
|--------|-------------|-----------|-----------|
| JCLRNT | 2.8512 | 3.7475 | 4.5138 |
| TrajRNE | 3.0756 | 4.7854 | 5.1898 |
| **DST** | **2.4595** | **3.4259** | **4.4987** |

DST achieves state-of-the-art results across all three downstream tasks in all three cities. For destination prediction, DST outperforms TrajRNE by 8.3% (ACC@1) in Beijing.

### Ablation Study

- w/o $P_{hop}$ (removing the mix-hop matrix): the most severe degradation occurs on speed inference, indicating that multi-hop movement relationships are critical for understanding road segment function
- w/o $hg_2$ (removing same-type hyperedges): significant degradation on speed inference, confirming that high-order relationships among same-type road segments are key supplementary information
- w/o tm (removing the temporal branch): the most severe degradation occurs on trajectory-related tasks (destination prediction, travel time estimation), demonstrating that temporal dynamics are indispensable
- w/o $hg_1$, w/o $hg_3$: moderate degradation, indicating that the three hyperedge types are mutually complementary

### Key Findings

1. Spatial semantic hypergraph and mix-hop matrix contribute most to speed inference — this task demands the highest level of road segment functional understanding.
2. The temporal branch yields substantial gains on trajectory-related tasks — traffic dynamics encode critical information about travel patterns.
3. In zero-shot cross-city transfer experiments (trained on Beijing, tested on Porto), DST remains highly competitive, achieving ACC@1 = 0.6424, far surpassing JCLRNT's 0.0167.
4. DST is robust to hyperparameter choices; smaller traffic batch sizes yield slightly better results due to the sparsity of traffic sequences, where large batches introduce noise.

## Highlights & Insights

- The **dual-branch divide-and-conquer strategy** is clean and effective: spatial and temporal information are each processed by the most suitable architecture (GNN vs. Transformer), avoiding interference from heterogeneous input coupling.
- The three hyperedge types collectively cover multiple dimensions of high-order road network relationships (functional region, type consistency, geographic adjacency), complementing one another.
- Strong zero-shot cross-city transfer capability suggests that the learned representations generalize well, reducing deployment costs for new cities.
- The use of weekday/weekend classification as a regularization task is an elegant design choice that guides the model toward learning discriminative temporal representations.

## Limitations & Future Work

- The independent pre-training and concatenation fusion of the two branches is relatively simplistic; joint training or attention-based fusion may yield further improvements.
- The mix-hop transition matrix relies on trajectory data quality; GPS noise and map-matching errors may degrade matrix quality.
- The temporal branch uses only hourly visit counts at 24-hour granularity, without considering finer-grained intervals (e.g., 15-minute) or longer temporal spans.
- The three hyperedge types are manually designed; data-driven hyperedge construction methods warrant exploration.

## Related Work & Insights

- **JCLRNT** and **TrajRNE** are pioneering works that leverage trajectories to enhance road network representations; DST builds upon them by incorporating hypergraphs and a temporal branch.
- The hypergraph contrastive learning paradigm can be applied to representation learning on other spatial networks (e.g., power grids, water networks).
- The dual-branch pre-training and fusion paradigm shares conceptual similarities with multimodal pre-training approaches such as CLIP (visual/text branches).

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 4 |
| Overall | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](../../NeurIPS2025/autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [\[AAAI 2026\] Minimum-Cost Network Flow with Dual Predictions](minimum-cost_network_flow_with_dual_predictions.md)
- [\[CVPR 2026\] Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens](../../CVPR2026/autonomous_driving/le_mumo_jepa_multi-modal_self-supervised_representation_learning_with_learnable_.md)
- [\[NeurIPS 2025\] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](../../NeurIPS2025/autonomous_driving/how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)
- [\[CVPR 2026\] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR](../../CVPR2026/autonomous_driving/terraseg_self-supervised_ground_segmentation_for_any_lidar.md)

</div>

<!-- RELATED:END -->
