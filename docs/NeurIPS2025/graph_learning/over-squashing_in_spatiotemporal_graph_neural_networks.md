---
title: >-
  [Paper Note] Over-squashing in Spatiotemporal Graph Neural Networks
description: >-
  [NeurIPS 2025][Graph Learning][Over-squashing] This paper provides the first formal treatment of over-squashing in spatiotemporal graph neural networks (STGNNs)…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Over-squashing"
  - "Spatiotemporal GNN"
  - "Causal Convolution"
  - "Information Propagation"
  - "Graph Rewiring"
date: 2026-05-08
content_hash: bb040f6b99ee7157
---

# Over-squashing in Spatiotemporal Graph Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2506.15507](https://arxiv.org/abs/2506.15507)  
**Code**: To be confirmed  
**Area**: Graph Learning / Spatiotemporal Graph Neural Networks / Theoretical Analysis
**Keywords**: Over-squashing, Spatiotemporal GNN, Causal Convolution, Information Propagation, Graph Rewiring

## TL;DR
This paper provides the first formal treatment of over-squashing in spatiotemporal graph neural networks (STGNNs), uncovering a counterintuitive "temporal sink" phenomenon in causal convolutions—whereby the earliest timestep exerts the greatest influence on the final representation—and proves that time-and-space (T&S) and time-then-space (TTS) architectures are equivalent in terms of information bottlenecks, offering theoretical justification for the computationally efficient TTS design.

## Background & Motivation

**Background**: STGNNs combine GNNs with sequential models (RNNs/TCNs) to handle data where node features on a graph evolve over time (e.g., traffic forecasting, energy systems). While over-squashing has been thoroughly studied in static GNNs, it remains entirely unexplored in spatiotemporal settings.

**Limitations of Prior Work**:
- The over-squashing theory developed for static GNNs does not transfer directly to STGNNs—the temporal dimension introduces an additional axis of information propagation, making the compression effect substantially more complex.
- The architectural choice between T&S (time-and-space, alternating processing) and TTS (time-then-space, sequential processing) lacks theoretical guidance and is largely empirical.
- The information propagation dynamics of causal convolutions have not been theoretically characterized.

**Key Challenge**: In STGNNs, information must propagate across both spatial and temporal dimensions simultaneously; each dimension can produce its own bottleneck, and the two bottlenecks compound.

**Goal**: (1) Formalize spatiotemporal over-squashing; (2) analyze the temporal information propagation characteristics of TCNs; (3) compare T&S and TTS with respect to information propagation.

**Key Insight**: Decompose STGNN information propagation into orthogonal spatial and temporal components, analyze each via Jacobian sensitivity, and combine the results into a joint spatiotemporal bound.

**Core Idea**: The sensitivity bound for spatiotemporal over-squashing factors into a product of spatial and temporal topology terms, and T&S and TTS are equivalent under a fixed computational budget.

## Method

### Overall Architecture
Consider an STGNN composed of $L$ spatiotemporal message-passing (STMP) layers, each containing $L_T$ temporal processing layers (TCN) and $L_S$ spatial processing layers (MPNN). The Jacobian $\nabla_i^u \mathbf{h}_t^{v(L)}$ is used to analyze how the initial feature of node $u$ at time $t-i$ influences the final representation of node $v$ at time $t$.

### Key Designs

1. **Temporal Over-squashing in TCNs (Theorem 4.1 + Proposition 4.2)**:

    - Function: Establishes an upper bound on sensitivity in TCNs and reveals a counterintuitive temporal preference.
    - Mechanism: $\|\frac{\partial \mathbf{h}_{t-j}^{(L_T)}}{\partial \mathbf{h}_{t-i}^{(0)}}\| \leq (c_\sigma \mathsf{w})^{L_T} (\mathbf{R}^{L_T})_{ij}$, where $\mathbf{R}$ is the temporal topology matrix of causal convolution (a lower-triangular Toeplitz matrix).
    - **Counterintuitive Finding**: The structure of $\mathbf{R}^l$ implies that the **earliest timesteps** (rather than the most recent) exert the greatest influence on the final output, because early timesteps accumulate exponentially more propagation paths across $l$ stacked layers. Recent timesteps can only maintain information through self-loops, with influence decaying at $O(l^{-(i-j)})$.
    - Analogy to the attention sink phenomenon in Transformers: when the receptive field exceeds the sequence length, the earliest tokens receive disproportionately large influence.

2. **Temporal Graph Rewiring Strategies**:

    - **Dilated Convolution**: Uses a distinct $\mathbf{R}^{(l)}$ per layer (dilation rate $d^{(l)} = P^{l-1}$), enabling exponential receptive field growth with a more uniform influence distribution. However, beyond $\log_P T$ layers the dilation rate must be reset, reintroducing bias.
    - **Row-Normalized Convolution**: Row-normalizes $\mathbf{R}$ to obtain $\mathbf{R}_N$ such that each row sums to one. This causes the influence received at the final timestep to approach a uniform distribution, making it particularly suited to forecasting tasks.

3. **Joint Spatiotemporal Bound for MPTCN (Theorem 5.1)**:

    - Function: Proves that the spatiotemporal sensitivity bound factors into a product of spatial and temporal components.
    - Mechanism: $\|\nabla_i^u \mathbf{h}_t^{v(L)}\| \leq (c_\xi \theta_\mathsf{m})^{LL_S} (c_\sigma \mathsf{w})^{LL_T} (\mathbf{S}^{LL_S})_{uv} (\mathbf{R}^{LL_T})_{i0}$
    - **Key Corollary**: The bound depends only on the total budgets $LL_S$ and $LL_T$, not on the value of $L$ individually. This establishes that TTS ($L=1$) and T&S ($L>1$) are equivalent in information propagation capacity—the computational advantage of TTS comes without any penalty in terms of information bottlenecks.

## Key Experimental Results

### Synthetic Experiments

| Task | Standard Convolution | Dilated Convolution | Normalized Convolution |
|------|---------|---------|----------|
| CopyFirst (recall earliest value) | Succeeds (benefits from sink) | Succeeds | Succeeds |
| CopyLast (recall most recent value) | Fails when $L_T>5$ | Partially succeeds | Succeeds |
| RocketMan (joint spatiotemporal) | Consistent with theoretical predictions | — | — |

### Real-World Data (Traffic Forecasting MAE)

| Model | $L$ | METR-LA | PEMS-BAY |
|------|-----|---------|----------|
| MPTCN $\mathbf{R}$ | 6 (T&S) | 3.19 | 1.66 |
| MPTCN $\mathbf{R}$ | 1 (TTS) | 3.14 | 1.63 |
| MPTCN $\mathbf{R}_N$ | 1 (TTS) | **Best** | **Best** |

### Key Findings
- **TTS ≈ T&S**: Under a fixed budget, $L=1$ (TTS) achieves performance comparable to or better than $L>1$ (T&S), validating Theorem 5.1.
- **Standard Causal Convolution Exhibits a Sink Effect**: As the number of layers increases, the model becomes biased toward information from the earliest timesteps, causing failure on the CopyLast task.
- **Normalized Convolution Effectively Mitigates Temporal Sink**: Row normalization drives influence toward a uniform distribution, substantially improving information retention from recent timesteps.
- **Both Dimensions of Spatiotemporal Over-squashing Must Be Addressed**: Optimizing only the spatial or temporal dimension in isolation is insufficient.

## Highlights & Insights
- **First Characterization of the "Temporal Attention Sink" in TCNs**: Multi-layer causal convolutions develop a bias toward the earliest inputs—a finding that parallels the attention sink phenomenon in Transformers and suggests a universal character to information compression bottlenecks.
- **Theoretical Proof that TTS = T&S**: Provides formal justification for the practical preference for the computationally efficient TTS architecture (TTS reduces spatial computational complexity to $O(T)$ compared to $O(T \times L)$ for T&S).
- **Temporal Graph Rewiring as a Dual of Spatial Graph Rewiring**: Extends the graph rewiring framework from the spatial domain to the temporal domain, establishing an elegant duality.

## Limitations & Future Work
- **Analysis Restricted to TCN as the Temporal Component**: The cases of RNNs or Transformers as temporal processors are not analyzed.
- **Sensitivity Upper Bounds Are Not Tight**: The bounds are necessary but not sufficient conditions; the actual degree of over-squashing may depend on bound tightness.
- **Normalized Convolution Benefits Only the Final Timestep**: It may not help tasks that require readout at intermediate timesteps (e.g., interpolation).
- **No Consideration of Adaptive Temporal Topology**: The proposed temporal rewiring strategies are fixed; learnable temporal attention mechanisms are not explored.

## Related Work & Insights
- **vs. Di Giovanni et al.**: Their work analyzes over-squashing in static GNNs; this paper extends the framework to spatiotemporal settings by incorporating a temporal dimension.
- **vs. WaveNet/TCN**: Dilated convolutions have been widely adopted in practice, but this paper provides the first theoretical explanation of their advantages from an over-squashing perspective.
- **Implications for STGNN Design**: (1) TTS is sufficient and does not require T&S; (2) standard causal convolution exhibits a sink bias and requires dilation or normalization for long-sequence tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First formalization of spatiotemporal over-squashing; the counterintuitive findings carry significant theoretical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic and real-world data, though real-world experiments are limited to traffic forecasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[NeurIPS 2025\] GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Interferometer Simulations](graph_neural_networks_for_interferometer_simulations.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
