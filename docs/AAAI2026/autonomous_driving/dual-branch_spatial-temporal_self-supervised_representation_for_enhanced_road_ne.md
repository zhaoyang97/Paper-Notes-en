---
title: >-
  [Paper Note] Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning
description: >-
  [AAAI 2026][Autonomous Driving][Road Network Representation Learning] This paper proposes the DST (Dual-branch Spatial-Temporal) road network representation learning framework. By jointly modeling the spatial heterogeneity and temporal dynamics of road networks through a spatial branch (mix-hop transition matrix + graph-hypergraph contrastive learning) and a temporal branch (Transformer encoder + next-token prediction + weekday/weekend classification)…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Road Network Representation Learning"
  - "Self-Supervised Learning"
  - "Graph Neural Networks"
  - "Hypergraph"
  - "Spatial-Temporal Modeling"
date: 2026-05-08
content_hash: 1aa1446ddc3d3162
---

# Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning

**Conference**: AAAI 2026  
**arXiv**: [2511.06633](https://arxiv.org/abs/2511.06633)  
**Code**: [Available](https://github.com/chaser-gua/DST)  
**Area**: Autonomous Driving  
**Keywords**: Road Network Representation Learning, Self-Supervised Learning, Graph Neural Networks, Hypergraph, Spatial-Temporal Modeling

## TL;DR

This paper proposes the DST (Dual-branch Spatial-Temporal) road network representation learning framework. By jointly modeling the spatial heterogeneity and temporal dynamics of road networks through a spatial branch (mix-hop transition matrix + graph-hypergraph contrastive learning) and a temporal branch (Transformer encoder + next-token prediction + weekday/weekend classification), the method achieves SOTA performance on three downstream tasks across three cities.

## Background & Motivation

Road Network Representation Learning (RNRL) aims to learn general low-dimensional vector representations for road segments, empowering downstream tasks such as traffic inference, travel time estimation, and trajectory destination prediction. Existing methods face two major challenges:

**1. Spatial Heterogeneity**: Road segment similarity depends not only on geographic distance but also on multi-dimensional relationships such as functional attributes and trajectory accessibility. For example, two distant road segments may have similar functions (e.g., both being residential roads), while adjacent road segments may have completely different functions. The neighborhood smoothing mechanism of GNNs struggles to capture such long-distance functional similarities.

**2. Temporal Dynamics**: Different types of road segments exhibit significantly different traffic patterns at different times, and patterns during weekdays and weekends are distinctly different. Relying solely on road network topology cannot fully characterize these dynamic changes in the temporal dimension.

Early methods (Node2Vec, GCN, GAE) are limited to simple graph structures; recent methods (Toast, JCLRNT, TrajRNE) have started to utilize trajectory information but lack temporal modeling; DyToast incorporates triangular temporal features but suffers from performance degradation in speed inference tasks.

## Method

### Overall Architecture

DST adopts a dual-branch architecture where the spatial and temporal branches are **pre-trained independently** and ultimately fused for downstream tasks:

- **Spatial Branch**: GNN + Hypergraph Contrastive Learning $\rightarrow$ Captures road network spatial topology and high-order semantic relationships
- **Temporal Branch**: Transformer Encoder + Dual-task Self-supervised Learning $\rightarrow$ Models 24-hour traffic dynamic patterns
- **Fusion Strategy**: Concatenating three representations for downstream tasks

### Key Designs

**1. Mix-hop Transition Matrix Weighting**

Extract multi-hop reachability relationships between road segments from trajectory data to construct the mix-hop transition matrix:

$$P_{hop}[r_i, r_j] = \sum_{\tau \in \mathcal{T}} \sum_{1 \leq i < j \leq m} m - (j - i)$$

Weighting strategy: Smaller hops are assigned larger initial weights, while larger hops get smaller weights, which stresses both adjacent connections and reachable long-distance connections. After row normalization, it is used as the initialization of a learnable weight matrix for feature weighting before the GNN:

$$Z_{hop} = \tilde{P}_{hop} \cdot Z_{\mathcal{I}}$$

**2. Spatial Semantic Graph-Hypergraph Contrastive Learning**

Constructing road network representations from two views:

**Graph View**: A multi-layer GAT encodes the road network topological structure while utilizing edge features $X_\mathcal{E}$ (connectivity attributes between road segments) to generate the spatial representation $Z_\mathcal{G}$.

**Hypergraph View**: Capturing high-order relationships by constructing three types of hyperedges:
- $\mathcal{E}_{\mathcal{H}_1}$: Functional Area Hyperedges — Spectral clustering is performed on road segments; road segments within the same cluster share a hyperedge.
- $\mathcal{E}_{\mathcal{H}_2}$: Same-type Hyperedges — Road segments of the same road type share a hyperedge regardless of their physical distance.
- $\mathcal{E}_{\mathcal{H}_3}$: Adjacent One-way Hyperedges — Geographically adjacent one-way roads share a hyperedge (Tobler's First Law of Geography).

The hypergraph is encoded using HGNN+ to generate the semantic representation $Z_\mathcal{H}$.

Contrastive learning is employed to maximize the mutual information of the same road segment representations between the two views:

$$\mathcal{L}_{\mathcal{GH}} = -\frac{1}{N}\sum_{r_i \in \mathcal{R}} \left[\frac{1}{|\mathcal{H}(r_i)|}\sum_{r_j \in \mathcal{H}(r_i)} I(v_{r_i}, h_{r_j})\right]$$

**3. Temporal Dynamics Modeling**

Traffic dynamics are defined as $\mathcal{D}_\mathcal{R} \in \mathbb{R}^{N \times 24 \times 2}$, representing the 24-hour volume sequence for each road segment (with separate channels for weekdays and weekends).

The Transformer encoder takes the last hidden state as the compressed sequence representation:

$$Z_\mathcal{D} = \text{TransEnc}(\text{PosEnc}(\mathcal{D}_\mathcal{R}))[-1]$$

Two joint self-supervised tasks are constructed:
- **Dynamics Prediction** (Regression): Predict the traffic volume of the next time step using historical sequences.
$$\mathcal{L}_{reg} = \frac{1}{N \times C}\sum_{i=1}^{N \times C}\|y_i - \hat{y}_i\|^2$$
- **Dynamics Classification**: Distinguish whether the input sequence is for weekdays or weekends.
$$\mathcal{L}_{cls} = -\frac{1}{N \times C}\sum_{i}\sum_c y_i(c)\log(\hat{y}_i(c))$$

### Loss & Training

Total loss for the temporal branch: $\mathcal{L}_d = \lambda_{reg} \cdot \mathcal{L}_{reg} + \lambda_{cls} \cdot \mathcal{L}_{cls}$

The spatial and temporal branches are **pre-trained separately**, and the three representations ($Z_\mathcal{G}, Z_\mathcal{H}, Z_\mathcal{D}$) are ultimately fused via concatenation for downstream tasks. Parameter sensitivity analysis shows that increasing the proportion of $\lambda_{reg}$ improves performance, as the initial loss scales of the two tasks differ significantly and require weight adjustment to achieve task balance.

## Key Experimental Results

### Main Results

**Table 1: Destination Prediction and Travel Time Estimation (Three Cities)**

| Method | Beijing ACC@1↑ | Beijing MRR↑ | Porto ACC@1↑ | Xi'an ACC@1↑ |
|------|---------------|-------------|-------------|-------------|
| Node2Vec | 0.1954 | 0.2884 | 0.2201 | 0.4088 |
| TrajRNE | 0.6728 | 0.7603 | 0.6728 | 0.8260 |
| JCLRNT | 0.4222 | 0.5528 | 0.5133 | 0.6752 |
| **DST** | **0.7288** | **0.8213** | **0.6766** | **0.8335** |

**Table 2: Speed Inference Task**

| Method | Beijing MAE↓ | Porto MAE↓ | Xi'an MAE↓ |
|------|-------------|-----------|-----------|
| JCLRNT | 2.8512 | 3.7475 | 4.5138 |
| TrajRNE | 3.0756 | 4.7854 | 5.1898 |
| **DST** | **2.4595** | **3.4259** | **4.4987** |

DST consistently achieves the best performance across three cities and three downstream tasks. Specifically, destination prediction performance in Beijing is improved by 8.3% (ACC@1) compared to TrajRNE.

### Ablation Study

- w/o $P_{hop}$ (without mix-hop matrix): Speed inference performance degrades the most, indicating that multi-hop motion relationships are crucial for understanding road segment functions.
- w/o $hg_2$ (without same-type hyperedge): Speed inference performance degrades significantly, showing that high-order relationships of the same-type road segments are critical supplementary information.
- w/o tm (without temporal branch): Trajectory-related tasks (destination prediction, travel time) degrade the most, proving that temporal dynamics are an indispensable supplement.
- w/o $hg_1$, w/o $hg_3$: Moderate degradation, showing that the three types of hyperedges complement each other.

### Key Findings

1. The spatial semantic hypergraph and the mix-hop matrix contribute the most to the speed inference task, which has the highest requirement for understanding road segment functionality.
2. The temporal branch significantly improves trajectory-related tasks, as traffic dynamics contain key information about mobility patterns.
3. In zero-shot cross-city transferability experiments (trained on Beijing $\rightarrow$ tested on Porto), DST demonstrates strong competitiveness with an ACC@1 of 0.6424, far outperforming JCLRNT's 0.0167.
4. DST is robust to hyperparameters; a smaller traffic batch size is slightly better (due to the sparsity of traffic sequences, where large batches introduce noise).

## Highlights & Insights

- **The dual-branch divide-and-conquer strategy** is straightforward and effective: spatial and temporal branches utilize their most suitable architectures respectively (GNN vs. Transformer), avoiding coupling interference of heterogeneous inputs.
- The design of three types of hyperedges covers multiple dimensions of high-order relationships in road networks (functional areas, type consistency, and physical adjacency), complementing each other.
- The zero-shot cross-city transferability indicates that the learned representations have good generalization capability, reducing deployment costs in new cities.
- The design of weekday/weekend classification as a regularization task is clever, guiding the model to learn discriminative temporal representations.

## Limitations & Future Work

- The dual-branch independent pre-training followed by concatenation is relatively simple; joint training or attention-based fusion might yield better results.
- The mix-hop transition matrix relies on trajectory data quality; GPS noise and map-matching errors may affect the quality of the matrix.
- The temporal branch only uses volume at a 24-hour granularity, without considering more fine-grained intervals (e.g., 15-minute intervals) or longer time spans.
- The three types of hyperedges in the hypergraph are manually designed; data-driven hyperedge generation methods could be explored.

## Related Work & Insights

- **JCLRNT** and **TrajRNE** are pioneers in leveraging trajectories to enhance road network representation; DST builds upon them by incorporating hypergraph and temporal branches.
- The idea of hypergraph contrastive learning can be applied to representation learning in other spatial networks (e.g., power grids, water networks).
- The paradigm of dual-branch pre-training + fusion shares similarities with multi-modal pre-training (such as CLIP's vision/text alignment).

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 4 |
| Overall Rating | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](../../NeurIPS2025/autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [\[AAAI 2026\] Minimum-Cost Network Flow with Dual Predictions](minimum-cost_network_flow_with_dual_predictions.md)
- [\[NeurIPS 2025\] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](../../NeurIPS2025/autonomous_driving/how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)
- [\[CVPR 2026\] STUR3D: Spatio-Temporal Unified Representation Learning for 3D Object Detection](../../CVPR2026/autonomous_driving/stur3d_spatio-temporal_unified_representation_learning_for_3d_object_detection.md)
- [\[CVPR 2026\] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR](../../CVPR2026/autonomous_driving/terraseg_self-supervised_ground_segmentation_for_any_lidar.md)

</div>

<!-- RELATED:END -->
