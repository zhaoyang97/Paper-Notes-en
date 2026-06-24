---
title: >-
  [Paper Note] SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions
description: >-
  [Autonomous Driving] Inspired by the human sketching process, this work models lane topology as a chain of sequential graph expansions, incrementally constructing directed lane graphs via an autoregressive transformer, thereby overcoming the inability of DAG-based methods to represent cycles and bidirectional lanes.
tags:
  - "Autonomous Driving"
date: 2026-05-08
content_hash: 75272afb040b9ff2
---

# SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2507.04822](https://arxiv.org/abs/2507.04822)
- **Code**: Not released
- **Area**: Autonomous Driving
- **Keywords**: Lane topology, graph generation, sequential modeling, autoregressive, incremental adjacency matrix expansion

## TL;DR

Inspired by the human sketching process, this work models lane topology as a chain of sequential graph expansions, incrementally constructing directed lane graphs via an autoregressive transformer, thereby overcoming the inability of DAG-based methods to represent cycles and bidirectional lanes.

## Background & Motivation

Accurate lane topology is critical for autonomous driving. Existing methods suffer from three categories of limitations:

**Fragmentation in detection-based methods**: Pixel-level methods (HDMapNet) cannot infer global topology; segment-level methods (TopoNet) detect centerlines in segments, causing discontinuities and misalignment; path-level methods (LaneGAP) introduce redundancy and rely on extensive post-processing.

**Structural degradation in generative methods**: RNTR converts lane graphs into DAGs, resulting in sequence redundancy and an inability to represent cycles or bidirectional lanes. LaneGraph2Seq generates node positions and centerline shapes separately, lacking holistic topological understanding.

**Core insight — how humans draw graphs**: Humans start from a single node and incrementally add new elements, continuously expanding the graph while establishing connections to existing nodes. This motivates modeling lane graph construction as a progressive expansion process.

## Method

### Mathematical Formulation of Lane Graphs

A lane graph is represented as a directed graph $G = (V, E)$, where $V$ denotes intersection nodes $v_n = (x_n, y_n)$ and $E$ denotes centerlines. Two matrices are used:

- **Adjacency matrix $A$**: $A(i,j) = 1$ indicates a direct directed centerline from $v_i$ to $v_j$
- **Geometry matrix $M$**: $M(i,j) = (\sigma_x^{ij}, \sigma_y^{ij})$ stores the control point of a quadratic Bézier curve

### Graph Serialization

Let $S_n$ denote the subgraph composed of the first $n$ nodes. Centerlines incoming to node $n$ from existing nodes are defined as $F_n = \sum_{k=0}^{n-1} M(k,n)$ ("from" relations), and outgoing centerlines to existing nodes as $T_n = \sum_{k=0}^{n-1} M(n,k)$ ("to" relations):

$$S_n = S_{n-1} + (v_n + F_n + T_n)$$

Each time a new node $n$ is introduced, the adjacency matrix expands from $n \times n$ to $(n+1) \times (n+1)$, focusing solely on the topological relationships between the new node and existing nodes. Node traversal follows a depth-first search order.

**Key advantage**: Unlike DAG-based methods, SeqGrowGraph can flexibly model cycles, bidirectional lanes, and non-trivial topologies.

### Discretization

Coordinates are quantized into discrete bins (resolution 0.5 m): node coordinate range 0–200, node index range 200–350, Bézier control point range 350–570. Special tokens: `<to>` = 571, `<SEP>` = 572, etc.

### Model Architecture and Objective

- **BEV encoder**: LSS encoder based on ResNet-50, pre-trained on a centerline segmentation task
- **Sequence decoder**: 6-layer Transformer generating token sequences autoregressively
- **Weighted loss**: Node position tokens are assigned weight 2; all other tokens weight 1

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log p(\hat{x}_t | x_1, x_2, \ldots, x_{t-1})$$

## Key Experimental Results

### Main Results on nuScenes

| Method | L-P | L-R | L-F | R-P | R-R | R-F |
|------|-----|-----|-----|-----|-----|-----|
| TopoNet | 52.5 | 47.1 | 49.6 | 46.9 | 10.8 | 17.5 |
| LaneGAP | 49.9 | 57.0 | 53.2 | 74.1 | 34.9 | 47.5 |
| RNTR | 57.1 | 42.7 | 48.9 | 63.7 | 45.2 | 52.8 |
| LaneGraph2Seq | 46.9 | 43.7 | 45.2 | 63.7 | 36.3 | 46.2 |
| **SeqGrowGraph** | **63.6** | **50.8** | **56.4** | **75.5** | **61.4** | **67.8** |

### Results on Argoverse 2

| Method | L-P | L-R | L-F | R-P | R-R | R-F |
|------|-----|-----|-----|-----|-----|-----|
| RNTR* | 50.7 | 29.4 | 37.2 | 68.1 | 29.6 | 41.3 |
| **SeqGrowGraph** | SOTA | — | — | — | — | — |

### Key Findings

1. **Substantial gains on reachability metrics**: R-F improves from 52.8 (RNTR) to 67.8 (+15.0), demonstrating that incremental graph expansion better preserves topological connectivity.
2. **High landmark precision**: L-P of 63.6 substantially surpasses all baselines, indicating accurate node position prediction.
3. **End-to-end without post-processing**: Continuous centerline results are generated in a single step.
4. **Expressiveness for complex topologies**: Cycles and bidirectional lanes, absent in DAG-based methods, are correctly modeled.

## Highlights & Insights

1. **Human sketching intuition**: The progressive graph expansion paradigm closely mirrors how humans draw maps.
2. **Incremental adjacency matrix expansion**: At each step, only the relationships between the new node and existing nodes need to be described, keeping sequences compact.
3. **DFS serialization**: Depth-first traversal ensures that adjacent nodes appear close to each other in the sequence.
4. **Bézier curve geometry representation**: A quadratic Bézier curve requires only one control point to describe centerline curvature.
5. **Weighted loss design**: Assigning higher weight to node position tokens reflects the principle that "correct edges depend on correct nodes."

## Limitations & Future Work

1. Sequence length grows quadratically as the number of nodes increases.
2. The use of only quadratic Bézier curves limits expressiveness for high-curvature roads.
3. The discretization resolution of 0.5 m constrains fine-grained positional accuracy.

## Related Work & Insights

- **Online HD mapping**: HDMapNet, VectorMapNet, MapTR
- **Lane topology**: STSU, TopoNet, RNTR, LaneGraph2Seq
- **Language models for graphs**: Research on encoding graph structures as learnable sequences

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The incremental graph expansion serialization approach represents an entirely new paradigm
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous mathematical formulation with elegant serialization design
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated on two large-scale datasets under two data splits
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear illustrations make the progressive expansion process easy to follow

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters](rescue_crowd_evacuation_simulation_via_controlling_sdm-united_characters.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[CVPR 2025\] T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving](../../CVPR2025/autonomous_driving/t2sg_traffic_topology_scene_graph_for_topology_reasoning_in_autonomous_driving.md)
- [\[AAAI 2026\] Fine-Grained Representation for Lane Topology Reasoning](../../AAAI2026/autonomous_driving/fine-grained_representation_for_lane_topology_reasoning.md)
- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](../../NeurIPS2025/autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)

</div>

<!-- RELATED:END -->
