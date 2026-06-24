---
title: >-
  [Paper Note] Beyond Self-Repellent Kernels: History-Driven Target Towards Efficient Nonlinear MCMC on General Graphs
description: >-
  [ICML 2025 (Oral)][Optimization][MCMC] This paper proposes the History-Driven Target (HDT) framework, which embeds self-repellent dynamics into any MCMC sampler by modifying the target distribution (rather than the transition kernel). It achieves the $O(1/\alpha)$ variance reduction while resolving the three major limitations of SRRW: high computational overhead, restriction to reversible chains, and high memory footprint.
tags:
  - "ICML 2025 (Oral)"
  - "Optimization"
  - "MCMC"
  - "Graph Sampling"
  - "Nonlinear Markov Chains"
  - "Stochastic Approximation"
  - "Self-Repellent Random Walk"
date: 2026-05-08
content_hash: c6a752d556a54c25
---

# Beyond Self-Repellent Kernels: History-Driven Target Towards Efficient Nonlinear MCMC on General Graphs

**Conference**: ICML 2025 (Oral)  
**arXiv**: [2505.18300](https://arxiv.org/abs/2505.18300)  
**Code**: None  
**Area**: Optimization  
**Keywords**: MCMC, Graph Sampling, Nonlinear Markov Chains, Stochastic Approximation, Self-Repellent Random Walk

## TL;DR

This paper proposes the History-Driven Target (HDT) framework, which embeds self-repellent dynamics into any MCMC sampler by modifying the target distribution (rather than the transition kernel). It achieves the $O(1/\alpha)$ variance reduction while resolving the three major limitations of SRRW: high computational overhead, restriction to reversible chains, and high memory footprint.

## Background & Motivation

Random walks on graphs are fundamental tools in physics, statistics, machine learning, and social sciences, with typical applications including social network sampling, web crawling, robot exploration, and mobile networks. The Metropolis-Hastings (MH) random walk is the most classic graph sampling method but suffers from slow mixing and local trapping issues.

Recently, the Self-Repellent Random Walk (SRRW) was proposed as a breakthrough method: it constructs a nonlinear Markov chain and dynamically adjusts transition probabilities based on historical visit frequencies. This biases the sampler towards under-sampled states, achieving near-zero variance. The core formulation of SRRW is:

$$K_{ij}[\mathbf{x}] = Z_i^{-1} P_{ij} (x_j / \mu_j)^{-\alpha}$$

where $\alpha \geq 0$ controls the self-repelling strength and $Z_i$ is the normalization constant.

However, SRRW has three key limitations:

**High computational overhead**: In each step, the transition probabilities of all neighbors of the current node must be pre-calculated to obtain the normalization constant $Z_i$, including the self-transition probability $P_{ii}$ (which requires traversing the acceptance rates of all neighbors). This completely destroys the lightweight "compute-on-demand" nature of the MH algorithm. The larger the neighborhood (e.g., in dense graphs), the more severe the overhead becomes.

**Restriction to reversible chains**: The theoretical guarantees of SRRW strictly depend on time-reversible Markov chains, making it incompatible with non-reversible MCMC methods that offer better acceleration (e.g., lifted chains, 2-cycle MCMC, MHDA).

**Memory constraints**: The dimension of the empirical measure $\mathbf{x}$ equals the size of the state space, which is memory-prohibitive for large graphs or exponential state spaces.

## Method

### Overall Architecture

The core idea of HDT is to shift the self-repelling mechanism from the transition kernel to the target distribution. Instead of modifying the transition kernel $\mathbf{P}$ of the MCMC, it replaces the original target $\boldsymbol{\mu}$ with a history-driven target $\boldsymbol{\pi}[\mathbf{x}]$, which is then used as the target distribution for an arbitrary MCMC sampler. Consequently, any MCMC method (reversible or non-reversible) can be plugged directly into the framework as a "base sampler" simply by replacing $\boldsymbol{\mu}$ with $\boldsymbol{\pi}[\mathbf{x}]$.

### Key Designs

#### Design Principles of the HDT Distribution

The authors propose four conditions to constrain the form of $\boldsymbol{\pi}[\mathbf{x}]$:

- **C1 (Scale Invariance)**: $\pi_i[\mathbf{x}, \boldsymbol{\mu}] = \pi_i[\tilde{\mathbf{x}}, \tilde{\boldsymbol{\mu}}]$, which means the framework can work with unnormalized quantities.
- **C2 (Local Dependence)**: The unnormalized term $\tilde{\pi}_i$ depends only on $\mu_i$ and $x_i$, requiring no neighbor information.
- **C3 (Fixed Point)**: $\boldsymbol{\pi}[\boldsymbol{\mu}] = \boldsymbol{\mu}$, meaning that the target distribution remains unchanged upon convergence.
- **C4 (History Dependence)**: Assigns higher probability to under-sampled states and lower probability to over-sampled states.

**Lemma 3.1** proves that the distribution satisfying C1-C4 must and can only have the following form:

$$\pi_i[\mathbf{x}] \propto \mu_i (x_i / \mu_i)^{-\alpha}, \quad \alpha > 0$$

This formulation is remarkably elegant. When $\alpha = 0$, it degrades to the original target $\boldsymbol{\mu}$. The key advantage lies in C2: the unnormalized term $\tilde{\pi}_i = \tilde{\mu}_i (\tilde{x}_i / \tilde{\mu}_i)^{-\alpha}$ only involves local information of the current and candidate states, completely eliminating the dependence on neighbor information.

#### Algorithm 1: HDT-MCMC Framework

Input: Graph $\mathcal{G}$, parameter $\alpha$, unnormalized target $\tilde{\boldsymbol{\mu}}$, number of iterations $T$, base sampler. In each step, execute:

1. Sample $X_{n+1}$ using the base sampler (reversible or non-reversible) with target $\boldsymbol{\pi}[\mathbf{x}]$.
2. Update the visit count: $\tilde{x}(X_{n+1}) \leftarrow \tilde{x}(X_{n+1}) + 1$.

Taking MH as the base sampler for instance, the acceptance rate becomes:

$$A_{ij}[\mathbf{x}] = \min\left\{1, \frac{\tilde{\mu}_j (\tilde{x}_j / \tilde{\mu}_j)^{-\alpha} Q_{ji}}{\tilde{\mu}_i (\tilde{x}_i / \tilde{\mu}_i)^{-\alpha} Q_{ij}}\right\}$$

This requires only four local quantities $\tilde{\mu}_i, \tilde{\mu}_j, \tilde{x}_i, \tilde{x}_j$, making the computational cost exactly the same as standard MH. Furthermore, this acceptance rate naturally embeds the self-repellent effect: if state $j$ is under-sampled relative to $i$ ($\tilde{x}_j / \tilde{\mu}_j < \tilde{x}_i / \tilde{\mu}_i$), then $A_{ij}[\mathbf{x}] \geq A_{ij}$, meaning it is more easily accepted.

#### Key Comparison with SRRW

The stationary distribution of SRRW contains the neighbor summation term $\sum_{j \in \bar{\mathcal{N}}(i)} P_{ij}(x_j/\mu_j)^{-\alpha}$, violating C2 and causing high computational overhead. The design of HDT decouples the neighbor dependency, which is the core innovation.

#### Compatibility with Diverse Samplers

The HDT framework seamlessly integrates with: reversible methods (MH, MTM, MHDR) and non-reversible methods (MHDA, 2-cycle Markov chains, non-reversible MH).

### Loss & Training

#### Theoretical Guarantees

**Theorem 3.3 (Ergodicity and CLT)**: HDT-MCMC satisfies (a) $\mathbf{x}_n \to \boldsymbol{\mu}$ a.s. and (b) $\sqrt{n}(\mathbf{x}_n - \boldsymbol{\mu}) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathbf{V}_{\text{HDT}}(\alpha))$, where:

$$\mathbf{V}_{\text{HDT}}(\alpha) = \frac{1}{2\alpha + 1} \mathbf{V}_{\text{base}}$$

The variance is reduced at an $O(1/\alpha)$ rate. **Corollary 3.4** ensures that the known performance ranking of samplers remains invariant under HDT.

**Lemma 3.6 (Cost-Based Comparison)**: $C_{\text{HDT}} \mathbf{V}_{\text{HDT}}(\alpha) \preceq \frac{2}{\mathbb{E}[|\bar{\mathcal{N}}(i)|]} \cdot C_{\text{SRRW}} \mathbf{V}_{\text{SRRW}}(\alpha)$, showing a greater advantage in dense graphs.

#### LRU Cache Strategy

To address the memory constraint on large graphs, only recently visited states are tracked. For a neighbor $j$ not in the cache, its value is approximated by the local neighborhood average:

$$\hat{x}_j = \tilde{\mu}_j \cdot |\bar{\mathcal{N}}(i) \cap \mathcal{C}|^{-1} \sum_{k \in \bar{\mathcal{N}}(i) \cap \mathcal{C}} \tilde{x}_k / \tilde{\mu}_k$$

## Key Experimental Results

### Main Results

Evaluated on Facebook (4039 nodes, average degree 43.7) and p2p-Gnutella04 (10876 nodes, average degree 7.4) under a uniform target over 1000 independent runs.

| Method | Facebook TVD | Facebook NRMSE | p2p TVD | Type |
|------|-------------|---------------|---------|------|
| MHRW | 0.520 | 0.079 | 0.545 | Reversible Baseline |
| HDT-MHRW | 0.371 | 0.028 | 0.403 | HDT Reversible |
| MTM | 0.487 | 0.056 | 0.514 | Reversible Baseline |
| HDT-MTM | 0.285 | 0.062 | 0.328 | HDT Reversible |
| MHDA | 0.513 | 0.068 | 0.522 | Non-reversible Baseline |
| HDT-MHDA | 0.365 | 0.027 | 0.388 | HDT Non-reversible |

### Ablation Study

| Configuration | TVD (Facebook) | Description |
|------|---------------|------|
| α=0 (MHRW) | 0.520 | No self-repelling effect |
| α=1 | ~0.45 | Minor improvement |
| α=5 | 0.371 | Significant improvement |
| α=10 | ~0.36 | Larger α brings continuous improvement |
| LRU r=0.1 (10% memory) | < MHRW | Still outperforms the baseline with a 90% memory reduction |
| LRU r=0.01 (100K graph) | ≈ r=0.1 | Still effective at extremely low memory |

### Key Findings

1. **Fair Comparison on Computational Budget**: Under the same budget, HDT-MHRW consistently outperforms SRRW, with a much wider gap on the Facebook graph (average degree 43.6) than on p2p-Gnutella04 (average degree 7.4), validating Lemma 3.6.
2. **Robustness to Initialization**: Different initial nodes and pseudo-visit counts have minimal impact on HDT-MCMC.
3. **Large-Scale Graphs**: On ego-Gplus (~100K nodes), the performance of LRU cache with $r=0.01$ and $r=0.1$ is comparable, both outperforming MHRW.

## Highlights & Insights

1. **Paradigm Shift**: Shifting self-repulsion from the transition kernel to the target distribution. A seemingly simple change of perspective resolves three core problems simultaneously.
2. **Uniqueness Proof**: Lemma 3.1 uses Cauchy's functional equation to prove that the distribution form satisfying the four conditions is unique, constraining the design space to a single-parameter family.
3. **Universal Plug-and-Play**: "Bring Your Own MCMC" — completely black-box to the base sampler.
4. **Lyapunov Analysis**: Proves global stability using $V(\mathbf{x}) = \sum_i \mu_i (x_i/\mu_i)^{-\alpha}$ combined with LaSalle's invariance principle; the use of the Cauchy-Schwarz inequality is remarkably elegant.

## Limitations & Future Work

1. **Exponential State Spaces Unexplored**: Application to high-dimensional problems (such as the Ising model) is left for future work.
2. **Lack of Theoretical Guarantees for LRU**: The caching scheme is heuristic and lacks a theoretical analysis of convergence.
3. **Adaptive Selection of $\alpha$**: How to automatically select the optimal $\alpha$ based on the graph structure is not discussed in depth.
4. **Limited to Discrete Spaces**: Extension of the framework to continuous spaces is not addressed.

## Related Work & Insights

- **SRRW (Doshi et al., 2023, ICML)**: Implements self-repulsion by directly modifying the transition kernel; the direct predecessor of this work.
- **MTM with locally balanced weights (Chang et al., 2022, NeurIPS)**: A multi-try Mutually-dependent MH variant.
- **MHDA (Lee et al., 2012)**: Non-reversible non-backtracking MH.
- **Stochastic Approximation Theory (Delyon 2000, Fort 2015)**: Convergence analysis frameworks.

Core Insight: When an adaptive mechanism is deeply coupled with a specific component of an algorithm, causing limitations, one can consider transferring that mechanism to another component. This might yield equivalent or even better effects at a lower cost.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐⭐ | Paradigm shift, from modifying the kernel to modifying the target |
| Theoretical Depth | ⭐⭐⭐⭐⭐ | Complete proofs for uniqueness, convergence, CLT, and cost-based CLT |
| Experimental Thoroughness | ⭐⭐⭐ stars | Multi-graph, multi-sampler, multi-metric; lacks high-dimensional experiments |
| Practicality | ⭐⭐⭐⭐ | Plug-and-play; large-scale scenarios still need validation |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clear motivation, thorough comparisons, and intuitive illustrations |

**Overall Rating**: 4.5/5 — Well-deserved ICML Oral. It addresses the core pain points of SRRW with an extremely simple design, backed by solid theoretical guarantees and thorough experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts](../../NeurIPS2025/optimization/flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)
- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](../../ICML2026/optimization/a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[ICLR 2026\] Symmetry-Aware Bayesian Optimization via Max Kernels](../../ICLR2026/optimization/symmetry-aware_bayesian_optimization_via_max_kernels.md)
- [\[NeurIPS 2025\] From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning](../../NeurIPS2025/optimization/from_linear_to_nonlinear_provable_weak-to-strong_generalization_through_feature_.md)
- [\[NeurIPS 2025\] Escaping Saddle Points without Lipschitz Smoothness: The Power of Nonlinear Preconditioning](../../NeurIPS2025/optimization/escaping_saddle_points_without_lipschitz_smoothness_the_power_of_nonlinear_preco.md)

</div>

<!-- RELATED:END -->
