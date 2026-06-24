---
title: >-
  [Paper Note] A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication
description: >-
  [AAAI 2026 Oral][Optimization][Semi-Decentralized Learning] This paper presents a unified convergence analysis framework to systematically compare, for the first time, two server-to-device communication primitives in semi-decentralized federated learning — S2S (returning the aggregated model only to sampled devices) and S2A (broadcasting to all devices). The analysis reveals distinct regimes in which S2S is superior under high inter-component heterogeneity and S2A is superior…
tags:
  - "AAAI 2026 Oral"
  - "Optimization"
  - "Semi-Decentralized Learning"
  - "Sampled-to-Sampled"
  - "Sampled-to-All"
  - "Convergence Analysis"
  - "Data Heterogeneity"
date: 2026-05-08
content_hash: dda40f4967eb4a0a
---

# A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.11560](https://arxiv.org/abs/2511.11560)  
**Code**: [https://github.com/arodio/SemiDec](https://github.com/arodio/SemiDec)  
**Area**: Distributed Optimization / Federated Learning  
**Keywords**: Semi-Decentralized Learning, Sampled-to-Sampled, Sampled-to-All, Convergence Analysis, Data Heterogeneity  

## TL;DR
This paper presents a unified convergence analysis framework to systematically compare, for the first time, two server-to-device communication primitives in semi-decentralized federated learning — S2S (returning the aggregated model only to sampled devices) and S2A (broadcasting to all devices). The analysis reveals distinct regimes in which S2S is superior under high inter-component heterogeneity and S2A is superior under low heterogeneity, and provides practical guidelines for system configuration.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: In federated learning (FL), device-to-server (D2S) communication is costly due to limited bandwidth and high latency. Fully decentralized learning avoids the central server by relying on device-to-device (D2D) communication, but fails to converge when the communication graph is disconnected. **Semi-decentralized learning** combines both paradigms: devices primarily communicate via D2D within components to reach intra-component consensus, and occasionally interact with a central server via D2S to propagate information across components.

After server aggregation, the aggregated model can be distributed in two ways: (i) S2S — returning the aggregated model only to the sampled devices; (ii) S2A — broadcasting the aggregated model to all devices. Each strategy has intuitive trade-offs (S2A propagates information faster but introduces bias; S2S is unbiased but leaves residual disagreement), yet a rigorous theoretical and empirical comparison has been lacking.

### Starting Point

**Goal**: Which strategy, S2S or S2A, is preferable in semi-decentralized FL, and under what conditions? Prior work lacks a convergence analysis of S2S under non-convex objectives, and no unified framework exists for fairly comparing the two strategies across varying system parameters (sampling rate, aggregation frequency, network connectivity, data heterogeneity).

## Method

### Overall Architecture
Consider $n$ devices distributed across $C$ disconnected components, where devices within each component communicate via D2D, and all devices occasionally interact through a central server via D2S. Each round consists of three steps:
1. **Local SGD**: Each device performs one step of stochastic gradient descent.
2. **D2D Mixing**: Each device averages model parameters with its neighbors, governed by mixing matrix $W$.
3. **D2S Aggregation** (every $H$ rounds): The server randomly samples $K$ devices, computes an aggregated model, and distributes it via either S2S or S2A.

### Key Designs

1. **Bias-Disagreement Duality**: The mixing matrix $W_{\text{S2S}}$ is doubly stochastic, preserving the global average (zero bias), but non-sampled devices are not updated, leaving a residual disagreement scaled by $\frac{n-K}{n-1}$ relative to the post-D2D disagreement. The mixing matrix $W_{\text{S2A}}$ is column-stochastic but not row-stochastic, eliminating disagreement (all devices share the same model) at the cost of introducing a broadcast bias of magnitude $\frac{n-K}{K(n-1)}$ times the post-D2D disagreement. These two error sources scale differently with respect to step size, sampling rate, aggregation period, and network connectivity.

2. **Orthogonal Decomposition**: A component projection operator $\Pi_C$ is introduced to orthogonally decompose the global disagreement into **intra-component disagreement** $\|X(I-\Pi_C)\|_F^2$ and **inter-component disagreement** $\|X(\Pi_C-\Pi)\|_F^2$. Only D2D communication can reduce intra-component disagreement, whereas inter-component disagreement can only be mitigated through D2S aggregation — a key distinction between S2S and S2A.

3. **Unified Convergence Framework**: A hierarchical heterogeneity assumption is proposed to separately quantify intra-component heterogeneity $\bar{\zeta}_{\text{intra}}$ and inter-component heterogeneity $\bar{\zeta}_{\text{inter}}$. Through an Alternating Disagreement Recursion (Lemma 6) and an Alternating Convergence Recursion (Lemma 7), the convergence rates of both S2S and S2A are analyzed in a unified manner for both convex and non-convex objectives.

### Main Theoretical Results

**Theorem 1 (S2S)**: In the non-convex setting, the heterogeneity term in the iteration complexity scales as $\mathcal{O}(\epsilon^{-3/2})$, with the intra-component term amplified by $\frac{n-1}{K-1}\cdot\frac{1}{p}$ and the inter-component term amplified by $\frac{n-1}{K-1}\cdot H$.

**Theorem 2 (S2A)**: In the non-convex setting, due to broadcast bias, the heterogeneity term scales as $\mathcal{O}(\epsilon^{-2})$ (slower), with an additional bias-related term weighted by $\frac{n-K}{K(n-1)}$, causing S2A to converge more slowly under high heterogeneity.

## Key Experimental Results

| Dataset | Setting | S2S Win Rate | Max Lead |
|---------|---------|-------------|----------|
| MNIST+CIFAR-10 | Varying sampling rate, H=5 | 60% (96 configs) | S2S up to +8.4pp (ring, K/n=0.2) |
| MNIST+CIFAR-10 | Varying aggregation period, K/n=0.2 | 60% (96 configs) | S2S up to +8.5pp (ring, H=5) |
| CIFAR-100 | R3, K/n=0.2, H=20 | S2S | +13.6pp |

Experimental setup: 100 devices split into 2 components (50 each), topologies include ring/grid/complete graph, MNIST uses a linear classifier (7,850 parameters), CIFAR-10 uses a CNN (~1.1M parameters).

### Ablation Study
- **Three regimes**: R1 (low heterogeneity) → S2A slightly better; R2 (high intra-, low inter-) → mixed results; R3 (high inter-component heterogeneity) → S2S substantially better (>90% of configurations).
- **Sampling rate**: The gap between S2S and S2A narrows as K/n increases, and the two coincide when $K=n$.
- **Aggregation period**: Both strategies degrade as $H$ increases, but S2A's inter-component term grows quadratically in $H$, while S2S grows only linearly.
- **Network connectivity**: S2S is more advantageous under sparse topologies (small $p$), as S2A's bias is harder to correct.
- **Dynamic topology**: Random regular graphs favor S2S more than fixed graphs (the gap widens from +8.58pp to +11.52pp).
- **Server momentum (FedAvgM)**: Does not alter the relative performance of S2S vs. S2A, but slightly reduces the periodic accuracy drops observed with S2A.

## Highlights & Insights
- The **bias-disagreement duality** perspective is highly intuitive: S2A eliminates disagreement but introduces bias; S2S is unbiased but leaves residuals. Which error dominates under a given parameter regime determines which strategy is preferable.
- The **orthogonal decomposition** that decouples intra- and inter-component heterogeneity is the central technical contribution enabling the theoretical analysis, and offers inspiration for other hierarchical optimization problems.
- The unified framework covers convex/non-convex objectives and static/dynamic topologies, and directly yields actionable configuration recommendations for practitioners.
- Experiments reveal that S2A exhibits **periodic accuracy drops** following each D2S aggregation event, which are particularly pronounced under high inter-component heterogeneity and explain why S2A is eventually overtaken by S2S in later training stages.

## Limitations & Future Work
- Only FedAvg/FedAvgM are considered as server optimizers; the interaction with more advanced methods (e.g., SCAFFOLD, ProxSkip) remains unexplored.
- The analysis assumes uniform random sampling without replacement; importance-based non-uniform sampling is not addressed.
- The network model assumes intra-component connectivity; more complex partially connected scenarios are not considered.
- Theoretical bounds involve large constants (e.g., 72, 210), which may be overly conservative for practical guidance.
- The interaction between communication compression (gradient quantization, sparsification) and S2S/S2A is not investigated.

## Related Work & Insights
- **vs. Fully Decentralized SGD (D-SGD)**: D-SGD is a special case with only D2D and no D2S (i.e., $H\to\infty$); this paper extends the analysis frameworks of Koloskova et al. (2020) and Le Bars et al. (2023).
- **vs. Hierarchical Federated Learning (HFL)**: HFL assumes tree-structured topologies and full device participation at each aggregation round; this paper supports arbitrary D2D topologies and partial device sampling.
- **vs. Chen, Wang, Brinton (2024)**: Their analysis of S2S is limited to convex objectives and assumes the server knows component membership; this paper relaxes both assumptions and extends to the non-convex setting.
- **vs. Lin et al. (2021), Guo et al. (2021)**: They analyze S2A under full participation; this paper handles partial sampling and establishes the effect of broadcast bias.

## Related Work & Insights
- The bias-disagreement duality is transferable to other hierarchical or multi-level communication scenarios (e.g., edge-cloud collaboration).
- The orthogonal decomposition of intra- and inter-component heterogeneity directly informs algorithm design for medical FL settings (e.g., different hospitals as different components).
- Integration with personalization methods in FL is a natural direction — the superiority of S2S under high inter-component heterogeneity suggests that preserving component independence may be preferable to enforcing global consensus.

## Rating
- Novelty: ⭐⭐⭐⭐ First to compare S2S and S2A in a unified framework and reveal the bias-disagreement duality, though methodologically the work combines existing analytical tools.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9,600 experiments across multiple datasets, topologies, and heterogeneity regimes, with comprehensive ablations and thorough outlier analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with tight theory-experiment correspondence; the regime visualization in Figure 1 is particularly intuitive.
- Value: ⭐⭐⭐⭐ Provides clear configuration guidelines for practical deployment of semi-decentralized FL, with both academic and engineering significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Improved Convergence Analysis of Topology Dependence in Decentralized SGD](../../ICML2026/optimization/improved_convergence_analysis_of_topology_dependence_in_decentralized_sgd.md)
- [\[ICLR 2026\] Communication-Efficient Decentralized Optimization via Double-Communication Symmetric ADMM](../../ICLR2026/optimization/communication-efficient_decentralized_optimization_via_double-communication_symm.md)
- [\[ICLR 2026\] On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning](../../ICLR2026/optimization/on_the_surprising_effectiveness_of_a_single_global_merging_in_decentralized_lear.md)
- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](../../NeurIPS2025/optimization/a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[ICLR 2026\] Unified Analyses for Hierarchical Federated Learning: Topology Selection under Data Heterogeneity](../../ICLR2026/optimization/unified_analyses_for_hierarchical_federated_learning_topology_selection_under_da.md)

</div>

<!-- RELATED:END -->
