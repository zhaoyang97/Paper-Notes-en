---
title: >-
  [Paper Note] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding
description: >-
  [NeurIPS 2025][LLM Pretraining][Neural architecture encoding] This paper proposes FGP (Flow-based Generative Pre-training), which trains an encoder to reconstruct a *flow surrogate* — a lightweight representation of arch…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "Neural architecture encoding"
  - "information flow"
  - "generative pre-training"
  - "NAS"
  - "graph neural networks"
date: 2026-05-08
content_hash: c67eb8354a48ef90
---

# Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding

**Conference**: NeurIPS 2025
**arXiv**: [2510.18360](https://arxiv.org/abs/2510.18360)
**Code**: [GitHub](https://github.com/kswoo97/FGPAnom)
**Area**: Neural Architecture Search / Representation Learning
**Keywords**: Neural architecture encoding, information flow, generative pre-training, NAS, graph neural networks

## TL;DR

This paper proposes FGP (Flow-based Generative Pre-training), which trains an encoder to reconstruct a *flow surrogate* — a lightweight representation of architectural information flow — enabling encoders of arbitrary structure to capture information flow without specialized asynchronous message-passing designs. FGP achieves up to **106%** improvement in Precision@1% on performance prediction.

## Background & Motivation

Neural architecture encoders are a core component of NAS, mapping architectures to vector representations for predicting task performance. State-of-the-art encoders (e.g., FlowerFormer) employ specialized asynchronous message-passing structures to simulate forward and backward information flow, yielding strong results but running up to **57×** slower than simple GNN encoders — a significant efficiency bottleneck.

Existing generative pre-training methods (e.g., Arch2vec reconstructing edges, GMAE predicting masked operations) are transferred directly from other domains but lack clear learning signals in the architecture domain: unlike chemical bonds, there are no compositional rules governing operations, making nearly every operation a plausible candidate for a masked position and providing little effective guidance to the model.

This paper addresses two challenges:

**Efficiency**: enabling simple encoders to learn information flow features without expensive flow-based encoder architectures.

**Pre-training objective**: designing novel generative pre-training tasks oriented toward architectural information flow, providing stronger learning signals than operation masking.

## Method

### Overall Architecture

FGP consists of two stages:

1. **Constructing the Pre-training Target (Flow Surrogate)**: For each architecture graph, a vector representing information flow is generated via a one-time random vector propagation, requiring no training.
2. **Generative Pre-training**: An encoder of arbitrary structure is trained to reconstruct the flow surrogate, internalizing information flow knowledge; the encoder is subsequently fine-tuned on downstream tasks (performance prediction, NAS).

### Key Designs

#### 1. **Topological Order Assignment**

The directed acyclic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ of an architecture is partitioned into topological layers $\mathcal{V}^{(1)}, \mathcal{V}^{(2)}, \ldots, \mathcal{V}^{(T)}$, where $\mathcal{V}^{(1)}$ contains nodes with no incoming edges (corresponding to inputs) and $\mathcal{V}^{(T)}$ contains nodes with no outgoing edges (corresponding to outputs). The topological order determines the sequence of message propagation.

#### 2. **Flow Surrogate Generation**

The core idea is to perform a simplified forward and backward propagation of random vectors over the architecture graph, producing a vector that serves as the architecture's "information flow fingerprint."

**Simulated Forward Propagation**: fp-messages are propagated in order $\mathcal{V}^{(1)} \to \mathcal{V}^{(T)}$. The operation embedding of node $v_i$ is $\mathbf{h}_i = \mathbf{P} \mathbf{X}_{i,:}$ (where $\mathbf{P}$ is a random matrix). Each node first aggregates incoming messages:

$$\mathbf{m}_i = \sum_{v_j \in \mathcal{N}^{(i)}} \mathbf{f}_j$$

and then produces an operation-conditioned fp-message:

$$\mathbf{f}_i = \alpha \mathbf{m}_i + (1 - \alpha) \text{ReLU}([\mathbf{h}_i \| \mathbf{m}_i] \mathbf{W})$$

where $\mathbf{W} \in \mathbb{R}^{2k \times k}$ is a fixed projection matrix and $\alpha$ is a weighting hyperparameter.

**Simulated Backward Propagation**: bp-messages are propagated in reverse order $\mathcal{V}^{(T)} \to \mathcal{V}^{(1)}$, initialized from the corresponding node's fp-message (simulating gradient dependence on forward outputs), using the same aggregation and transformation mechanism but propagated along outgoing edges in reverse.

The flow surrogate is obtained by summing the bp-messages of all order-1 nodes: $\mathbf{s} = \sum_{v_i \in \mathcal{V}^{(1)}} \mathbf{b}_i$.

**Design Motivation**: This process simulates the complete pipeline of how inputs traverse each operational layer in a real neural network and how gradients are back-propagated. Different architectures yield distinct flow surrogates due to differences in topology and operation types, naturally differentiating their information flow characteristics.

#### 3. **Generative Pre-training Loss**

An encoder $f_\theta$ encodes the architecture graph into $\mathbf{z} \in \mathbb{R}^d$, and an MLP decoder $g_\phi$ maps it back to the flow surrogate space:

$$\mathcal{L}_{rec} = \| \mathbf{s} - g_\phi(f_\theta(\mathbf{X}, \mathcal{E})) \|_2^2$$

The total training objective combines the reconstruction loss with an auxiliary loss (e.g., zero-cost proxy prediction):

$$\mathcal{L} = \lambda_1 \mathcal{L}_{rec} + \lambda_2 \mathcal{L}_{aux}$$

**Design Motivation**: Reconstructing the flow surrogate compels the encoder to retain information flow characteristics in its embeddings, while the auxiliary objective provides additional performance-correlated supervision signals.

### Loss & Training

- **Pre-training phase**: Minimizes $\mathcal{L}_{rec} + \mathcal{L}_{aux}$ over all architectures without using performance labels.
- **Fine-tuning phase**: Minimizes the supervised performance prediction loss on a small labeled training set.
- Pre-training is applicable to arbitrary GNN encoders (ResGatedGCN, GIN, FlowerFormer, etc.) without modifying the encoder architecture.

## Key Experimental Results

### Main Results — Performance Prediction

Fine-tuned with 1% training data; evaluated on NAS-Bench-101/201/301 across three encoders × six methods.

| Encoder | Pre-training | NB-101 Kendall τ | NB-201 Kendall τ | NB-101 Prec@1% | NB-201 Prec@1% |
|---|---|---|---|---|---|
| ResGatedGCN | N/A (none) | 65.0 | 73.4 | 18.2 | 29.7 |
| ResGatedGCN | ZC-Proxy | 68.3 | 79.9 | 26.2 | 44.3 |
| ResGatedGCN | **FGP** | **74.8** | **82.2** | **37.5** | **48.9** |
| GIN | N/A | 62.8 | 65.7 | 26.9 | 25.0 |
| GIN | **FGP** | **67.8** | **79.2** | **33.2** | **35.6** |
| FlowerFormer | N/A | 74.0 | 77.3 | 35.3 | 35.6 |
| FlowerFormer | **FGP** | **76.3** | **83.6** | **40.6** | **48.3** |

FGP achieves the best results in **23 out of 27 settings**. ResGatedGCN + FGP improves Precision@1% on NB-101 by **106%** over the no-pretraining baseline.

### Ablation Study

| Configuration | NB-101 Kendall τ | NB-201 Kendall τ | NB-101 Prec@1% | Notes |
|---|---|---|---|---|
| w/o $\mathcal{L}_{rec}$ | 68.3 | 79.9 | 26.2 | Removing flow reconstruction → degenerates to ZC-Proxy |
| w/o $\mathcal{L}_{aux}$ | 71.5 | 74.5 | 35.6 | Removing auxiliary loss → Kendall τ drops |
| w/o Forward | 73.5 | 81.4 | 36.3 | Backward only → still effective but incomplete |
| w/o Backward | 72.4 | 81.4 | 31.0 | Forward only → backward contributes more |
| **FGP (full)** | **74.8** | **82.2** | **37.5** | Forward + backward + auxiliary loss is optimal |

### Key Findings

1. **Non-flow encoders + FGP can surpass flow encoders**: ResGatedGCN + FGP matches or exceeds FlowerFormer (without pre-training) on most metrics, while being orders of magnitude faster.
2. **NAS experiments**: Predictors pre-trained with FGP consistently identify better architectures at every step within the NPENAS search framework.
3. **PCA visualizations** demonstrate that the flow surrogate effectively separates high-performing architectures from low-performing ones.
4. Simulated backward propagation contributes more to performance than simulated forward propagation.

## Highlights & Insights

- The paper elegantly reframes "learning information flow" from a model architecture design problem into a pre-training objective design problem, replacing expensive asynchronous message-passing with a one-time computed flow surrogate.
- The flow surrogate requires no labels to compute, making it naturally suited for unsupervised pre-training.
- The method is highly general and can be plugged into any GNN encoder as a pre-training step.

## Limitations & Future Work

- The flow surrogate relies on fixed random matrices $\mathbf{P}$ and $\mathbf{W}$, which may not be optimal; learnable initialization or ensembling over multiple random samples could be explored.
- Thorough validation is currently limited to the NAS-Bench series (primarily CV tasks); results for NLP and graph learning domains are relegated to the appendix.
- For very large search spaces (e.g., the continuous space of NB-301), FGP's gains are relatively modest, potentially requiring richer flow representations.

## Related Work & Insights

- **FlowerFormer**: The state-of-the-art flow-based encoder and the primary baseline; FGP further improves its performance when combined.
- **Arch2vec / GMAE**: Existing generative pre-training methods that reconstruct edges or predict masked operations; this paper argues that such approaches lack effective learning signals in the architecture domain.
- **ZC-Proxy**: Uses zero-cost proxies as a pre-training objective; FGP's auxiliary loss also incorporates zero-cost proxies.
- **Insight**: For representation learning on structured data, designing domain-specific pre-training objectives is more effective than directly transferring generic methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of simulating information flow to construct generative pre-training targets is novel, though the core techniques (message passing, graph reconstruction) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three NAS-Bench datasets × three encoders × five baselines, with full ablations, visualizations, and NAS application experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with intuitive figures; the challenge-solution narrative flows clearly.
- **Value**: ⭐⭐⭐⭐ Provides a practical pre-training method for the NAS community that balances efficiency and effectiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](superposition_yields_robust_neural_scaling.md)

</div>

<!-- RELATED:END -->
