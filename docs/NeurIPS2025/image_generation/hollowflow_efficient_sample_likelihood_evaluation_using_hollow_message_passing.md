---
title: >-
  [Paper Note] HollowFlow: Efficient Sample Likelihood Evaluation using Hollow Message Passing
description: >-
  [NeurIPS 2025][Image Generation][Continuous Normalizing Flows] This paper proposes HollowFlow, a framework that enforces a block-diagonal structure on the Jacobian of the velocity field via Non-Backtracking Graph Neural Networks (NoBGNN) and Hollow Message Passing, reducing the number of backward passes required for likelihood computation in Continuous Normalizing Flows from $\mathcal{O}(n)$ to a constant $\mathcal{O}(d)$, achieving up to $10^2\times$ sampling speedup.
tags:
  - NeurIPS 2025
  - Image Generation
  - Continuous Normalizing Flows
  - Boltzmann Generators
  - Message Passing
  - Non-Backtracking Graph Neural Networks
  - Likelihood Computation
date: 2026-05-08
content_hash: e80e9d423980ff8f
---

# HollowFlow: Efficient Sample Likelihood Evaluation using Hollow Message Passing

**Conference**: NeurIPS 2025
**arXiv**: [2510.21542](https://arxiv.org/abs/2510.21542)
**Code**: Unavailable
**Area**: Flow Models / Scientific Computing
**Keywords**: Continuous Normalizing Flows, Boltzmann Generators, Message Passing, Non-Backtracking Graph Neural Networks, Likelihood Computation

## TL;DR

This paper proposes HollowFlow, a framework that enforces a block-diagonal structure on the Jacobian of the velocity field via Non-Backtracking Graph Neural Networks (NoBGNN) and Hollow Message Passing, reducing the number of backward passes required for likelihood computation in Continuous Normalizing Flows from $\mathcal{O}(n)$ to a constant $\mathcal{O}(d)$, achieving up to $10^2\times$ sampling speedup.

## Background & Motivation

Boltzmann Generators (BG) are fundamental tools in scientific computing that aim to learn a surrogate model $\rho_1(\mathbf{x})$ approximating the Boltzmann distribution $\mu(\mathbf{x}) = Z^{-1}\exp(-\beta u(\mathbf{x}))$. A key requirement of BGs is the ability to perform unbiased estimation via importance sampling, which demands efficient computation of sample likelihoods $\rho_1(\mathbf{x}_i)$ under the model.

Continuous Normalizing Flows (CNF) are the dominant implementation of BGs, where the density change is given by the divergence:

$$\Delta \log \rho^{\text{CNF}} = -\int_0^1 \nabla \cdot b_\theta(\mathbf{x}(t), t) \, dt$$

**Core bottleneck**: Computing $\nabla \cdot b_\theta$ requires $\mathcal{O}(N)$ backward passes (where $N = nd$ is the total system dimensionality), which is prohibitively expensive for large systems (e.g., an $n=55$ particle Lennard-Jones system with $N=165$ dimensions). Although stochastic estimators such as Hutchinson's estimator can reduce cost, they suffer from high variance and are inexact.

Existing HollowNet techniques can reduce backward passes for divergence computation to a single pass by enforcing a hollow Jacobian structure, but are limited to feedforward networks and cannot be directly extended to the **equivariant graph neural networks** required for molecular systems.

The central problem addressed in this paper is: how to generalize the efficient likelihood computation of HollowNet to graph neural network architectures while preserving equivariance.

## Method

### Overall Architecture

HollowFlow consists of three components:
1. **Non-Backtracking Graph Neural Network (NoBGNN)** as a conditioner, ensuring that each node's hidden state $h_i$ contains no information about itself.
2. **Transformer network** $\tau_i$ mapping $h_i$ and $\mathbf{x}_i$ to output $b_i$.
3. **Continuous Normalizing Flow** parameterizing the velocity field $b_\theta$ using the above architecture, trained via Conditional Flow Matching.

The core principle is to decompose the Jacobian of the velocity field into block-hollow and block-diagonal components:

$$\mathbf{J}_{b(\mathbf{x})} = \mathbf{J}_{c(\mathbf{x})} + \mathbf{J}_{\tau(\mathbf{x})}$$

where $\mathbf{J}_c$ is block-hollow (zero diagonal blocks) and $\mathbf{J}_\tau$ is block-diagonal (each block of size $d \times d$). The divergence requires computing only the diagonal elements of $\mathbf{J}_\tau$, necessitating only $d$ backward passes.

### Key Designs

1. **Hollow Message Passing (HoMP)**: A non-backtracking GNN is constructed based on the line graph. Given the original graph $G = (N, E)$, the line graph $L(G) = (N^{lg}, E^{lg})$ is constructed such that nodes in the line graph correspond to edges in the original graph, and edges in the line graph satisfy:
$$E^{lg} = \{(i,j,k) \mid (i,j) \in E,\ (j,k) \in E \text{ and } i \neq k\}$$
The non-backtracking condition is enforced by the constraint $i \neq k$, preventing information from returning to its source node. Initial node features are set as $n_{ij}^{lg} = n_i$, and message passing proceeds on the line graph:
$$m_{lij}^t = \phi(h_{li}^t, h_{ij}^t), \quad m_{ij}^t = \sum_{l \in \mathcal{N}^{lg}(i,j)} m_{lij}^t, \quad h_{ij}^{t+1} = \psi(h_{ij}^t, m_{ij}^t)$$
Readout projects back to the original graph: $b_j = \sum_{i \in \mathcal{N}(j)} R(h_{ij}^{T^{lg}}, n_j)$.

2. **Multi-step Non-Backtracking Guarantee**: Single-step message passing is naturally non-backtracking, but multi-step propagation requires additional handling. A backtracking array $B(t) \in \mathbb{R}^{n \times n \times n}$ is introduced to track information propagation paths:
$$B(t)_{ijk} = \begin{cases} 1 & \text{if } \partial h_{ij}^t / \partial \mathbf{x}_k \neq 0 \\ 0 & \text{else} \end{cases}$$
Before each message passing step, edges that could cause backtracking are removed: $E^{lg} \leftarrow E^{lg} \setminus \{(i,j,k) \in E^{lg} \mid B(t)_{ijk} = 1\}$.

3. **Euclidean Equivariance**: Each node $\mathbf{x}_i \in \mathbb{R}^d$ is embedded into equivariant vector features $\mathbf{v}_i$ and invariant scalar features $s_i$, using message and update functions from $\mathcal{G}$-equivariant GNNs such as PaiNN or E3NN. Since only the underlying graph structure is modified, equivariance is naturally preserved. The $d$-dimensional node inputs induce a natural $d \times d$ block structure in the Jacobian.

### Computational Complexity Analysis

Key result (Theorem 2): For a $k$-nearest-neighbor graph ($k$NN), the per-step inference complexity of HollowFlow is:

$$\text{RT}^{step}(L(G_k)) = \mathcal{O}(n(T^{lg} k^2 + dk))$$

Compared to $\mathcal{O}(Tn^3 d)$ for a standard fully connected GNN, the speedup factor is $\mathcal{O}\left(\frac{Tn^2 d}{T^{lg}k^2 + dk}\right)$.

Choosing $k \leq \mathcal{O}(\sqrt{n})$ ensures that the forward pass overhead does not exceed that of a fully connected GNN.

## Key Experimental Results

### LJ13 System (13 particles, 39 dimensions)

| Model | ESS↑(%) | ESSrem↑(%) | EffSUrem↑ |
|-------|---------|------------|-----------|
| Baseline (fully connected) | 2.132 | **40.73** | 1 |
| HollowFlow k=6 | 3.300 | 20.20 | **3.260** |
| HollowFlow k=12 | 4.069 | 19.72 | 1.627 |
| HollowFlow k=2 | 0.054 | 2.92 | 1.059 |

### LJ55 System (55 particles, 165 dimensions)

| Model | ESS↑(%) | ESSrem↑(%) | EffSUrem↑ |
|-------|---------|------------|-----------|
| Baseline (fully connected) | 0.048 | **2.96** | 1 |
| HollowFlow k=7 | 0.006 | 0.53 | **93.737** |
| HollowFlow k=27 | 0.007 | 0.64 | 9.466 |
| HollowFlow k=55 | 0.020 | 0.74 | 4.365 |

### Key Findings

- HollowFlow yields lower raw ESS than the baseline (due to reduced expressiveness from the $k$NN graph constraint), but delivers **substantially greater effective throughput when accounting for computation time**.
- LJ13: $k=6$ achieves the best trade-off, with approximately $3.3\times$ practical sampling speedup.
- LJ55: $k=7$ achieves **approximately $94\times$** equivalent speedup (on the order of $10^2$), consistent with the theoretical $\mathcal{O}(n^2)$ prediction.
- Runtime analysis shows that the computational bottleneck in the baseline is the backward pass (divergence computation), whereas in HollowFlow the bottleneck shifts to the forward pass.
- The choice of $k$ involves a trade-off: too small limits expressiveness, while too large diminishes the speedup advantage.

## Highlights & Insights

- Generalizing HollowNet from scalar-level to block-level ($d \times d$ blocks) is a natural and conceptually deep extension.
- The backtracking array $B(t)$ provides an elegant solution to the multi-step non-backtracking guarantee.
- The framework is highly general: any equivariant GNN or attention-based architecture can be adapted into a NoBGNN (demonstrated for attention mechanisms in the appendix).
- Theoretical and empirical speedups are in strong agreement, validating the correctness of the analysis.

## Limitations & Future Work

- The $k$NN graph introduces a locality assumption that may be inappropriate for systems with long-range interactions (e.g., molecules involving Coulombic forces).
- Edge removal after each message passing step may introduce additional overhead in large-scale systems.
- The baseline ESS is not state-of-the-art; however, improvements to the baseline should transfer directly to HollowFlow.
- Experiments are limited to Lennard-Jones systems; validation on real molecular systems remains to be done.

## Related Work & Insights

- The original HollowNet (Chen & Duvenaud) handles only scalar inputs; extending it to equivariant vector inputs is a key contribution of this work.
- Non-backtracking graphs have been applied to community detection and alleviating over-squashing, but this is the first application to efficient likelihood computation.
- HollowFlow is complementary to BG emulator approaches (BioEmu, AlphaFlow, etc.): the latter sacrifice exact likelihoods for qualitative correctness, whereas HollowFlow maintains exact likelihood computation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Creatively extends HollowNet to graph neural networks with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Theoretical speedup predictions are validated across two systems with detailed runtime analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, rigorous theoretical derivations, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐⭐ Addresses the scalability bottleneck of Boltzmann Generators, with significant implications for the scientific computing community.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](../../ICLR2026/image_generation/sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)
- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [\[NeurIPS 2025\] GSPN-2: Efficient Parallel Sequence Modeling](gspn-2_efficient_parallel_sequence_modeling.md)
- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](efficient_rectified_flow_for_image_fusion.md)
- [\[NeurIPS 2025\] Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation](hallucination_as_an_upper_bound_a_new_perspective_on_text-to-image_evaluation.md)

<!-- RELATED:END -->
