---
title: >-
  [Paper Note] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks
description: >-
  [NeurIPS 2025][Graph Learning][Graph Neural Networks] A GNN-based self-supervised framework is proposed that infers latent synaptic connectivity via a structure learning module while simultaneously predicting future spiking activity via a spike prediction module. The approach substantially outperforms statistical inference baselines on both simulated ring attractor network data and real mouse head-direction cell recordings.
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Neural Circuit Inference"
  - "Self-Supervised Learning"
  - "Continuous Attractor Networks"
  - "Synaptic Connectivity Inference"
date: 2026-05-08
content_hash: 0089a703cd570479
---

# Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2509.17174](https://arxiv.org/abs/2509.17174)  
**Code**: Unavailable  
**Area**: Graph Learning
**Keywords**: Graph Neural Networks, Neural Circuit Inference, Self-Supervised Learning, Continuous Attractor Networks, Synaptic Connectivity Inference

## TL;DR

A GNN-based self-supervised framework is proposed that infers latent synaptic connectivity via a structure learning module while simultaneously predicting future spiking activity via a spike prediction module. The approach substantially outperforms statistical inference baselines on both simulated ring attractor network data and real mouse head-direction cell recordings.

## Background & Motivation

Inferring synaptic connection strengths among neural populations is a central challenge in computational neuroscience. Existing statistical inference methods face two fundamental difficulties:

**Partial observability**: Recording all neurons in a circuit is infeasible, and unobserved neurons introduce spurious correlations.

**Model mismatch**: Inference models (e.g., GLMs, Ising models) may fail to accurately characterize the underlying generative dynamical system.

In strongly recurrent networks, even weakly connected or disconnected neurons exhibit strong activity correlations due to "co-activation" — a consequence of pattern formation principles. Existing methods (GLMs, maximum entropy models, etc.) struggle to explain away such spurious correlations, leading to biased inference.

**Core Insight**: Neurons can be naturally modeled as nodes in a graph with synaptic connections as edges. The message-passing mechanism of GNNs is inherently suited to capturing the dynamics of interacting neurons. By adopting future spike prediction as a proxy task, the framework self-supervisedly extracts connectivity structure from latent representations.

## Method

### Overall Architecture

The model consists of two functionally decoupled modules: a structure learning module that infers the connectivity matrix $\mathbf{w}$, and a spike prediction module that leverages $\mathbf{w}$ to predict future spiking activity. Both modules are jointly optimized via a self-supervised objective, without requiring ground-truth connectivity labels.

### Key Designs

1. **Structure Learning Module**: Learns pairwise connection strengths from each neuron's spike train.

   A 1D convolution is applied to the spike train $\mathbf{x}_i$ of neuron $i$ to extract temporal features, followed by a fully connected layer to obtain embedding $\mathbf{z}_i$:
   $\mathbf{z}_i = f_{\text{out}}(\text{vec}(f_{\text{Conv1D}}(\mathbf{x}_i)))$

   The embeddings of each neuron pair are concatenated and passed through a two-layer MLP to estimate connection strength:
   $w_{ij} = \text{MLP}([\mathbf{z}_i, \mathbf{z}_j])$

   **Design Motivation**: Rather than inferring connectivity directly from a correlation matrix, the module learns feature representations in a kernel space to distinguish true connections from spurious correlations.

2. **Spike Prediction Module (GNN Message Passing)**: Uses the inferred connectivity as edge weights within a GNN to predict future spikes.

   An encoder embeds the recent spike history window: $\mathbf{h}_i^t = f_{\text{enc}}(\mathbf{x}_i^{t-\ell+1:t})$

   Message computation and aggregation:
   $\mathbf{m}_{ij}^t = \phi([\mathbf{h}_i^t, \mathbf{h}_j^t])$
   $\mathbf{h}_i^{t+1} = \psi\left(\sum_{j \in \mathcal{N}(i)} w_{ij} \cdot \mathbf{m}_{ij}^t, \mathbf{h}_i^t\right)$

   The decoder outputs firing rate parameters: $\log(\lambda_i^{t+1}) = f_{\text{dec}}(\mathbf{h}_i^{t+1})$, with spikes generated via a Poisson process.

   **Design Motivation**: GRU gating retains long-range temporal dependencies, while the connectivity weights $w_{ij}$ serve directly as message scaling factors, tightly coupling prediction accuracy with the quality of connectivity inference.

3. **Hidden Neuron Handling**: Auxiliary nodes representing unobserved neurons are added, with embeddings initialized via interpolation. These nodes participate in message passing under a transductive setting, implicitly incorporating information from unobserved components.

### Loss & Training

The Poisson negative log-likelihood is minimized:

$$\Theta^* = \arg\min_\Theta \sum_{i=1}^{N} \sum_{t=1}^{T} \left(\lambda_i^t - \mathbf{x}_i^t \log \lambda_i^t\right)$$

Note that $\mathbf{w}$ is not a free parameter but is deterministically derived from observed data through the structure learning module and updated indirectly as $\Theta$ is optimized. The Adam optimizer is used with a learning rate of $5 \times 10^{-4}$ and exponential decay.

## Key Experimental Results

### Main Results: Connectivity Inference in Fully Observed Networks

| Method | Δ↓ (Thresh.) | $\mathcal{L}_{\text{bps}}$↑ (Thresh.) | Δ↓ (LNP) | $\mathcal{L}_{\text{bps}}$↑ (LNP) |
|------|-------------|-------------|----------|-------------|
| **GNN (Ours)** | **0.061** | **0.882** | **0.049** | **0.876** |
| GLM | 0.244 | 0.695 | 0.238 | 0.712 |
| seqNMF | 0.789 | – | 0.796 | – |
| TCA | 0.762 | – | 0.761 | – |

The GNN's inference error is more than 70% lower than that of the GLM.

### Fully Observed Networks with External Input Drive

| Method | Δ↓ (Thresh.) | $\mathcal{L}_{\text{bps}}$↑ (Thresh.) | Δ↓ (LNP) | $\mathcal{L}_{\text{bps}}$↑ (LNP) |
|------|-------------|-------------|----------|-------------|
| **GNN** | **0.073** | **0.916** | **0.058** | **0.924** |
| GLM | 0.259 | 0.724 | 0.245 | 0.748 |

### Weakly Correlated Networks (Single-Bump Activity Pattern)

| Method | Δ↓ (Thresh.) | $\mathcal{L}_{\text{bps}}$↑ (Thresh.) | Δ↓ (LNP) | $\mathcal{L}_{\text{bps}}$↑ (LNP) |
|------|-------------|-------------|----------|-------------|
| **GNN** | **0.048** | **2.652** | **0.043** | **2.668** |
| GLM | 0.125 | 2.534 | 0.117 | 2.576 |

### Key Findings

1. Baseline methods consistently exhibit "side-trough" artifacts in the inferred weight profiles — i.e., spuriously strong connections are inferred for distal, unconnected neurons — whereas the GNN effectively suppresses these artifacts.
2. Increasing the number of hidden neurons initially improves performance but eventually saturates, as excessive hidden nodes introduce structural ambiguity.
3. On real mouse head-direction (HD) cell data, the inferred connectivity patterns are consistent with the theoretical predictions of the continuous attractor model.

## Highlights & Insights

1. **Self-supervised paradigm**: Ground-truth connectivity labels are not required; the connectivity structure is indirectly inferred solely through the spike prediction task.
2. **Dual-module decoupled design**: The separation of structure learning and dynamics prediction renders the connectivity matrix an interpretable latent representation.
3. **Biological verifiability**: The ring attractor hypothesis is validated on real HD cell recordings.

## Limitations & Future Work

- Connection strengths are assumed to be stationary throughout the recording period, precluding the modeling of synaptic plasticity.
- Validation is currently limited to ring network topologies; generalization to arbitrary topologies remains to be demonstrated.
- The same observed dynamics may correspond to multiple circuit configurations, posing a non-identifiability problem.
- The framework is extensible to 2D continuous attractor networks, such as grid cell systems.

## Related Work & Insights

- **Statistical inference methods**: GLM, maximum-entropy Ising model, Minimum Probability Flow (MPF)
- **GNN-based physical inference**: Neural Relational Inference (NRI) and related work
- **Neural system models**: Ring attractor networks, head-direction cell models

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic application of a GNN-based self-supervised framework to neural circuit inference
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-condition ablations on synthetic data combined with real-data validation
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with progressively presented experiments
- Value: ⭐⭐⭐⭐ — A meaningful contribution at the intersection of computational neuroscience and graph neural networks

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport](../../ICML2025/graph_learning/hgot_self-supervised_heterogeneous_graph_neural_network_with_optimal_transport.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)

</div>

<!-- RELATED:END -->
