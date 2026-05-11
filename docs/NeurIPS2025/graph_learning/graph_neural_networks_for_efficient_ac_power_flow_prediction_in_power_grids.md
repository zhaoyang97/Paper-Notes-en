---
title: >-
  [Paper Note] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids
description: >-
  [NeurIPS 2025][Graph Learning][GNN] This work models power networks as graph structures (buses as nodes, transmission lines as edges) and investigates four GNN architectures — GCN, GAT, SAGEConv…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "GNN"
  - "AC Power Flow"
  - "Power Systems"
  - "Graph Convolution"
  - "Optimal Power Flow"
date: 2026-05-08
content_hash: fed495c1148118b7
---

# Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids

**Conference**: NeurIPS 2025
**arXiv**: [2502.05702](https://arxiv.org/abs/2502.05702)
**Code**: [GitHub](https://github.com/Amirtalebi83/GNN-OptimalPowerFlow)
**Area**: Graph Learning / Power Systems
**Keywords**: GNN, AC Power Flow, Power Systems, Graph Convolution, Optimal Power Flow

## TL;DR

This work models power networks as graph structures (buses as nodes, transmission lines as edges) and investigates four GNN architectures — GCN, GAT, SAGEConv, and GraphConv — for predicting AC power flow solutions (voltage magnitudes and phase angles). Experiments on IEEE 14/30/57/118-bus test systems demonstrate that GNNs can efficiently substitute traditional Newton-Raphson solvers.

## Background & Motivation

The Optimal Power Flow (OPF) problem is one of the most fundamental optimization problems in power system operations: minimizing generation cost subject to load demand and various grid constraints (generator capacity, voltage magnitude/angle limits, and line thermal limits).

**Limitations of Prior Work**: Traditional OPF solvers (e.g., Newton-Raphson, IPOPT interior-point methods) face significant challenges in scalability and computational efficiency. As grid size grows (more buses and transmission lines) and renewable energy penetration introduces increased variability, computational complexity escalates dramatically, making real-time dispatch infeasible. AC OPF is further complicated by the nonlinearity of power flow equations, rendering it a non-convex optimization problem that is considerably harder to solve than the linearized DC OPF.

**Key Challenge**: Real-time grid management requires millisecond-level response, whereas traditional AC OPF solvers may take seconds or even minutes on large-scale networks. Early MLP-based approaches improved speed but failed to exploit the graph topology of the grid, limiting their accuracy and generalization.

**Key Insight**: Power networks are inherently graph-structured — buses are nodes, transmission lines are edges, and the admittance matrix encodes the electrical coupling between nodes. This structure aligns naturally with the message-passing mechanism of GNNs, which aggregate neighborhood information to model inter-node electrical interactions.

**Core Idea**: Train GNNs to learn the mapping from load conditions to power flow solutions (voltage magnitudes and phase angles) over large amounts of simulation data, thereby replacing the iterative solving process and enabling real-time prediction.

## Method

### Overall Architecture

**Input**: Graph-structured data of the power network. Each bus (node) carries a 7-dimensional feature vector comprising active power $P$, reactive power $Q$, voltage magnitude $V$, voltage phase angle $\delta$, and bus type (one-hot encoding of Slack/PV/PQ). Edges are determined by the transmission line topology (bidirectional connections).

**Output**: Voltage magnitude $V$ and phase angle $\delta$ for each bus (i.e., the solution to the AC power flow equations).

**Intermediate stages**: 2-layer GNN message passing (12-dimensional hidden features per layer) → 128-dimensional fully connected layer → output of dimension $2 \times n_{\text{bus}}$.

### Key Designs

1. **AC Power Flow Equation Modeling**:

    - Function: Reformulates the traditional nonlinear power flow equations as a GNN-learnable regression problem.
    - Core equations:
    $P_i = V_i \sum_{j=1}^{n} V_j (G_{ij} \cos(\delta_i - \delta_j) + B_{ij} \sin(\delta_i - \delta_j))$
    $Q_i = V_i \sum_{j=1}^{n} V_j (G_{ij} \sin(\delta_i - \delta_j) - B_{ij} \cos(\delta_i - \delta_j))$
      where $G_{ij}$ (conductance) and $B_{ij}$ (susceptance) are the real and imaginary parts of the admittance matrix.
    - Design Motivation: The power injection at each bus depends on the voltages and phase angles of neighboring buses — a neighborhood interaction that directly corresponds to the message-passing mechanism of GNNs.

2. **Bus Type Encoding**:

    - Function: Encodes bus type information (PV, PQ, Slack) as node features.
    - Mechanism: One-hot encoding is used to distinguish three bus types:
        - Slack bus: $V$ and $\delta$ are known; $P$ and $Q$ are solved.
        - PV bus (generator bus): $P$ and $V$ are known; $Q$ and $\delta$ are solved.
        - PQ bus (load bus): $P$ and $Q$ are known; $V$ and $\delta$ are solved.
    - Design Motivation: Different bus types play fundamentally different roles and carry different constraints in power flow computation. Explicit encoding enables the GNN to learn these distinctions.

3. **Comparison of Four GNN Architectures**:

    - **GCN**: Baseline model; aggregates mean neighborhood features per node. Simple but struggles to capture long-range dependencies.
    - **GAT**: Introduces attention mechanisms to dynamically assign different weights to different neighbors, allowing the model to focus on critical buses (e.g., generator buses).
    - **SAGEConv**: Samples a fixed number of neighbors for aggregation; designed for inductive learning and scalable to large-scale grids.
    - **GraphConv**: Incorporates self-loops to strengthen learning from nodes' own features, well-suited for scenarios where node-level information is particularly important.

4. **Data Generation and Augmentation**:

    - Function: Uses pandapower to generate large-scale training data with realistic load variations.
    - Mechanism: Applies ±40% load perturbations to IEEE standard test systems, simulating intra-day fluctuations (morning peak 60–70%, noon 110–120%, evening peak 110–120%, nighttime 60–70%) and seasonal variations (winter 1.2–1.4×, summer 1.1–1.3×).
    - Design Motivation: Ensures model robustness and generalization across diverse operating conditions.

### Loss & Training

- **Loss Function**: Mean Squared Error (MSE), measuring the deviation between predicted and ground-truth voltage magnitudes and phase angles.
- **Optimizer**: Adam, learning rate $5 \times 10^{-5}$, L2 regularization $\lambda = 1 \times 10^{-6}$.
- **Learning Rate Scheduling**: ReduceLROnPlateau combined with exponential decay (decay factor 0.9 every 10 epochs).
- **Regularization**: Dropout rate 0.2, Batch Normalization.
- **Early Stopping**: Based on validation loss, patience = 20 (100 in some experiments).
- **Data Scale**: 10 scenarios × 10,000 samples per IEEE system = 100,000 data points.
- **Feature Normalization**: z-score standardization (bus type features kept as categorical without normalization).

## Key Experimental Results

### Main Results

Experiments compare four GNN architectures on IEEE 14/30/57/118-bus systems:

| Architecture | NRMSE (14-bus) | NRMSE (30-bus) | NRMSE (57-bus) | NRMSE (118-bus) |
|------|:-:|:-:|:-:|:-:|
| GCN | Higher | Highest | Highest | Higher |
| GAT | Medium | Significant improvement | Significant improvement | Medium |
| SAGEConv | **Lowest** | Low | Low | **Lowest** |
| GraphConv | Lowest | **Lowest** | **Lowest** | Lowest |

All models achieve NRMSE below 0.05 and $R^2$ scores close to 1.

### Ablation Study

| Architecture | Strengths | Weaknesses | Best Use Case |
|------|------|------|----------|
| GCN | Simple, fast training | Highest NRMSE; limited long-range dependency capture | Baseline comparison |
| GAT | Attention focuses on critical nodes | Higher computational cost than GCN | Medium-scale grids requiring node importance differentiation |
| SAGEConv | Best scalability; superior on large grids | Slightly higher test loss than GraphConv | Large-scale grids (118-bus and beyond) |
| GraphConv | Best node feature enhancement; lowest loss | No notable disadvantages | Scenarios where node-level information is critical |

### Key Findings

1. **All GNN architectures perform well**: All four architectures achieve $R^2 \approx 1$ and NRMSE < 0.05 across all IEEE systems, confirming the suitability of GNNs for power flow prediction.
2. **Training loss decreases as grid size increases**: This suggests that larger grids provide richer topological information for GNNs to exploit.
3. **SAGEConv and GraphConv are top performers**: SAGEConv excels on large-scale systems (118-bus), while GraphConv achieves the lowest test loss on small-to-medium systems.
4. **GAT attention mechanism is effective**: Compared to GCN, GAT achieves significantly lower NRMSE on 30-bus and 57-bus systems, attributed to its ability to focus attention on critical nodes such as generator buses.
5. **Batch size impact**: Larger batch sizes improve convergence and reduce training noise.
6. **Consistent train/validation loss convergence**: Indicates effective learning and good generalization.

## Highlights & Insights

1. **Natural problem–method alignment**: The most central insight of this work is the natural correspondence between the graph structure of power networks and the message-passing mechanism of GNNs — the electrical coupling encoded in the admittance matrix directly maps to GNN neighborhood aggregation.
2. **Bus type encoding**: Explicitly incorporating Slack/PV/PQ bus types as node features is a more refined treatment than prior baseline approaches, reflecting effective integration of domain knowledge.
3. **Comprehensive architecture comparison**: A systematic evaluation of four GNN architectures across grids of varying scales provides clear guidance for architecture selection in future work.
4. **Realistic load variation modeling**: The data generation strategy incorporating ±40% intra-day and seasonal load variations enhances the practical relevance of the experiments.

## Limitations & Future Work

1. **No runtime comparison with traditional solvers**: Despite claiming superiority over traditional solvers, the paper provides no concrete computation time benchmarks.
2. **No evaluation on larger grids**: The largest test case is 118 buses; real-world grids can scale to thousands or tens of thousands of buses (e.g., Polish 9241-bus system), leaving scalability unverified.
3. **Physical constraint satisfaction not enforced**: Predicted solutions may violate physical laws such as Kirchhoff's laws; no AC-feasibility regularization is incorporated.
4. **Static topology assumption**: Only fixed network topologies are considered; dynamic topology changes (e.g., line failures, topology switching) are not addressed.
5. **Shallow GNN depth**: Only 2 GNN layers with 12-dimensional hidden features, which may limit the capture of long-range dependencies.
6. **Ground truth depends on Newton-Raphson**: Training data quality is bounded by the accuracy of the traditional solver.
7. **No comparison with advanced ML methods**: Direct comparison with existing approaches such as TGNNs and physics-constrained GNNs is absent.
8. **Imprecise NRMSE reporting**: Results are presented as trends in figures rather than exact numerical values.

## Related Work & Insights

- **Chen & Yu (2020)**: Among the earliest applications of GNNs to OPF, using imitation learning to predict IPOPT outputs → this work extends the idea to multiple GNN architectures.
- **Donti et al. (2021)**: Topology-Aware GNN with AC-feasibility regularization → the absence of physical constraints in this work represents a clear direction for improvement.
- **López-García (2023)**: Typed GNN differentiating bus types → this work achieves a similar effect via one-hot encoding.
- **Yang et al. (2020)**: PPO + GNN for OPF → combining reinforcement learning is a promising future direction.
- **Donon et al. (2020)**: Graph Neural Solver for directly solving power flow equations → complementary to the regression approach in this paper.

**Insights**: The systematic architecture comparison presented in this work provides a valuable template for applying graph learning to physical systems. Incorporating domain knowledge (bus types, admittance matrix) into graph representations is a key factor in improving GNN performance in scientific computing applications.

## Rating

- Novelty: ⭐⭐⭐ The systematic comparison of multiple GNN architectures is valuable, though the core idea of using GNNs for power flow prediction has precedent in prior work.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive evaluation across four architectures and four grids, but lacking runtime comparisons and tests on larger systems.
- Writing Quality: ⭐⭐⭐ Well-structured with thorough related work coverage, though some experimental results are not reported with sufficient numerical precision.
- Value: ⭐⭐⭐ Provides a practical reference for GNN applications in power systems; open-sourced code enhances reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Underappreciated Power of Vision Models for Graph Structural Understanding](the_underappreciated_power_of_vision_models_for_graph_structural_understanding.md)
- [\[NeurIPS 2025\] BLISS: Bandit Layer Importance Sampling Strategy for Efficient Training of Graph Neural Networks](bliss_bandit_layer_importance_sampling_strategy_for_efficient_training_of_graph_.md)
- [\[ICLR 2026\] On the Expressive Power of GNNs for Boolean Satisfiability](../../ICLR2026/graph_learning/on_the_expressive_power_of_gnns_for_boolean_satisfiability.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Interferometer Simulations](graph_neural_networks_for_interferometer_simulations.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
