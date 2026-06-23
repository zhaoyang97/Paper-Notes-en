---
title: >-
  [Paper Note] Unified Analyses for Hierarchical Federated Learning: Topology Selection under Data Heterogeneity
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] This paper establishes the **first unified non-convex convergence framework** for four two-tier topologies of Hierarchical Federated Learning (HFL) (Star-Star / Star-Ring / Ring-Star / Ring-Ring). By utilizing a common set of assumptions and an "effective learning rate," it compares their convergence bounds in a single
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 912633b99b5bf636
---
# Unified Analyses for Hierarchical Federated Learning: Topology Selection under Data Heterogeneity

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ojdYVjLX7S](https://openreview.net/forum?id=ojdYVjLX7S)  
**Area**: Federated Learning / Distributed Optimization Theory  
**Keywords**: Hierarchical Federated Learning, convergence analysis, topology selection, data heterogeneity, ring aggregation

## TL;DR
This paper establishes the **first unified non-convex convergence framework** for four two-tier topologies of Hierarchical Federated Learning (HFL) (Star-Star / Star-Ring / Ring-Star / Ring-Ring). By utilizing a common set of assumptions and an "effective learning rate," it compares their convergence bounds in a single table, derives three actionable topology selection principles, and validates them on CIFAR-10, CINIC-10, Fashion-MNIST, and SST-2.

## Background & Motivation

**Background**: Federated Learning (FL) enables devices to collaboratively train models without exchanging raw data. However, single-tier FL faces communication bottlenecks, synchronization delays, and single points of failure in large-scale deployments. Hierarchical Federated Learning (HFL) supports massive numbers of devices by inserting an intermediate aggregation layer (edge servers/cluster heads) between clients and the global server to distribute coordination pressure.

**Limitations of Prior Work**: Each layer of aggregation in HFL can use **parallel updates (Star)** or **sequential updates (Ring)**, resulting in four topological combinations: Star-Star, Star-Ring, Ring-Star, and Ring-Ring. Existing theoretical analyses almost exclusively cover the simplest Star-Star case, and different works use inconsistent assumptions (some even using the strong "uniformly bounded gradient" assumption). Consequently, it is **impossible to horizontally compare which topology is superior**. Ring-Ring has never even been analyzed.

**Key Challenge**: Practitioners need to know "which topology to choose given specific data heterogeneity, grouping structures, and network conditions," but the lack of a unified framework leads to arbitrary decisions. The difficulty lies in HFL's **cascaded heterogeneity** (non-trivial interleaving of inter-group distribution $\delta$ and intra-group distribution $\zeta$), **cross-layer dependencies** (updates in one layer directly affecting error propagation in another, making the effective learning rate topology-dependent), and the fact that Star/Ring updates have fundamentally different statistical properties (Star is unbiased high-variance parallel averaging; Ring is biased low-variance sequential propagation). These properties stack layer by layer.

**Goal**: To place the four topologies under a unified set of non-convex assumptions, provide comparable convergence bounds, and answer "which topology to choose for which scenario."

**Key Insight**: The authors observe that global data heterogeneity can be precisely decomposed into "inter-group + intra-group" parts using the **Law of Total Variance**. By defining a **topology-dependent effective learning rate $\tilde\eta$** that absorbs architectural parameters (number of groups $G$, group rounds $P$, clients per group $M$, local steps $K$), the four bounds can be expressed in a unified form for item-by-item comparison.

**Core Idea**: Use a "unified non-convex assumption + effective learning rate + inter/intra-group heterogeneity decomposition" toolkit to derive and compare the convergence bounds of the four HFL topologies, extracting topology selection rules from the error terms.

## Method

### Overall Architecture

This work is not a new algorithm but a **unified convergence analysis framework**. It formalizes the optimization objective of two-tier HFL: the global goal is to minimize
$$F(x) = \frac{1}{G}\sum_{g=1}^{G} F_g(x) = \frac{1}{G}\sum_{g=1}^{G}\frac{1}{M}\sum_{m=1}^{M} F_{g,m}(x),$$
where $F_g$ is the average of all client objectives within group $g$, and $F_{g,m}$ is the local objective of the $m$-th client in group $g$. The difference between the four topologies lies solely in **whether each of the two layers uses Star or Ring aggregation**: Star allows members to update in parallel from the same starting point and averages them; Ring allows members to update sequentially from the previous member's result, propagating like a relay.

The framework involves three core steps: ① Use the Law of Total Variance to decompose global heterogeneity into two bounded quantities: inter-group $\delta$ and intra-group $\zeta$ (Assumptions 3, 4); ② Define an effective learning rate $\tilde\eta$ for each topology that absorbs architectural parameters (e.g., $\tilde\eta=PK\eta$ for Star-Star, $\tilde\eta=GPMK\eta$ for Ring-Ring); ③ Under $L$-smoothness, bounded variance, and bounded inter/intra-group heterogeneity—**assumptions that are identical for all four topologies**—derive unified convergence bounds (Theorem 1 for full participation, Theorem 2 for partial participation), simplified into comparable convergence rates using the optimal $\tilde\eta$ (Corollaries 1, 2). The framework also extends to realistic settings like random-shuffle execution and partial participation (sampling $S_1$ groups and $S_2$ clients per group).

### Key Designs

**1. Heterogeneity Decomposition: Separating "Inter-group $\delta$ vs. Intra-group $\zeta$" to locate the actual bottleneck**

The challenge in HFL is that data heterogeneity is hierarchical—there are differences between group distributions and differences between clients within a group, both of which affect convergence. This paper uses the **Law of Total Variance** to provide an identity:
$$\frac{1}{G}\sum_{g}\frac{1}{M}\sum_{m}\|\nabla F_{g,m}(x)-\nabla F(x)\|^2 = \underbrace{\frac{1}{G}\sum_{g}\|\nabla F_g(x)-\nabla F(x)\|^2}_{\text{Inter-group }\delta^2} + \underbrace{\frac{1}{G}\sum_{g}\frac{1}{M}\sum_{m}\|\nabla F_{g,m}(x)-\nabla F_g(x)\|^2}_{\text{Intra-group }\zeta^2},$$
meaning global heterogeneity is exactly the **orthogonal partition** of inter-group $\delta^2$ (Assumption 3) and intra-group $\zeta^2$ (Assumption 4). This decomposition allows $\delta$ and $\zeta$ to appear as separate terms in the convergence bound, making it clear which type of heterogeneity decays slower and acts as the bottleneck.

**2. Topology-Dependent Effective Learning Rate $\tilde\eta$: Aligning the four topologies for comparison**

The update mechanisms of the four topologies differ significantly. This paper defines an **effective learning rate $\tilde\eta$** that absorbs all architectural parameters: $\tilde\eta=PK\eta$ for Star-Star, $PMK\eta$ for Star-Ring, $GPK\eta$ for Ring-Star, and $GPMK\eta$ for Ring-Ring. After substitution, all bounds follow a unified structure: "Optimization term $\frac{A}{\tilde\eta R}$ + Noise term + Heterogeneity term." For instance, the full participation bound for Ring-Ring is:
$$\mathbb{E}\|\nabla F(\bar x^{(R)})\|^2 \lesssim \frac{A}{\tilde\eta R} + \frac{L\tilde\eta\sigma^2}{GPMK} + \frac{L^2\tilde\eta^2\sigma^2}{GPMK} + \frac{L^2\tilde\eta^2\zeta^2}{G^2P^2} + L^2\tilde\eta^2\delta^2.$$
Larger $\tilde\eta$ speeds up optimization but amplifies error terms. With the optimal $\tilde\eta$ (Corollary 1), all four topologies share the same asymptotic rate $O(1/\sqrt{GPMKR})$, with differences residing in the lower-order topology-dependent term $T$. This also explains the observed learning rate law: $\eta_{\text{Star-Star}} > \eta_{\text{Star-Ring}} \approx \eta_{\text{Ring-Star}} > \eta_{\text{Ring-Ring}}$.

**3. Three Topology Selection Principles: Top-tier dominance, Inter-group bottleneck, and Structural matching**

By comparing the terms (Table 1 / Corollary 1), three regularities emerge. First, **Top-tier Dominance**: Changing the top layer from Star to Ring (Ring-Star vs. Star-Star) adds a factor of $G$ (or $S_1$ in partial participation) to the denominator of the SGD variance and intra-group heterogeneity terms. This means the Ring top layer suppresses errors by an additional $G^2$ factor, making it more robust. Second, **Inter-group Heterogeneity Bottleneck**: In all topologies, the inter-group term decays slowly at $O\big(\frac{(L^2A^2\delta^2)^{1/3}}{R^{2/3}}\big)$ without $G/P$ in the denominator, while the intra-group term enjoys an extra $\frac{1}{G^{2/3}P^{2/3}}$ acceleration under a Ring top layer. Thus, $\delta$ is the primary bottleneck. Third, **Topology-Structure Matching**: Ring-Star is suitable for "many small groups" (IoT), while Star-Ring is better for "few large groups" (hospital networks) due to deeper local refinement. Star-Star consistently performs worst.

### Loss & Training

Ours does not introduce a new loss function, following standard FL local empirical risk minimization. The theory rests on four assumptions: $L$-smoothness (Asm 1), unbiased stochastic gradients with variance bounded by $\sigma^2$ (Asm 2), inter-group heterogeneity bounded by $\delta^2$ (Asm 3), and intra-group heterogeneity bounded by $\zeta^2$ (Asm 4). These assumptions are consistent across all topologies and sufficiently general.

## Key Experimental Results

Setup: $N=100$ clients divided into $G=10$ groups, four data partitions (inter/intra-group IID or Non-IID using Dirichlet distribution), with individual learning rate tuning for each topology.

### Main Results: Final Test Accuracy of the Four Topologies

| Dataset/Model | Heterogeneity (Inter/Intra) | Star-Star | Star-Ring | Ring-Star | Ring-Ring |
|------------|------------------|-----------|-----------|-----------|-----------|
| CIFAR-10 / ResNet-18 | IID / IID | 88.48 | 90.30 | 90.40 | **91.53** |
| CIFAR-10 / ResNet-18 | Non-IID / Non-IID | 86.78 | 87.40 | 90.01 | **90.33** |
| CINIC-10 / ResNet-18 | Non-IID / Non-IID | 73.63 | 74.21 | **77.11** | 76.78 |
| Fashion-MNIST / ResNet-10 | Non-IID / Non-IID | 88.04 | 92.18 | 92.27 | **93.33** |
| SST-2 / MLP | Non-IID / IID | 68.12 | 73.85 | 79.13 | **81.08** |

**Topologies with a Ring top layer (Ring-Star / Ring-Ring) outperform Star top layers in almost all settings**, while Star-Star remains consistently the worst.

### Impact of Inter-group vs. Intra-group Heterogeneity

| Scenario (CIFAR-10/ResNet-18, Star-Ring) | Accuracy | Gain relative to IID-IID |
|------|------|------|
| Inter-IID + Intra-IID | 90.30 | — |
| Inter-IID + Intra-Non-IID | 89.55 | −0.75 |
| Inter-Non-IID + Intra-IID | 88.22 | **−2.08** |

### Key Findings
- **Inter-group heterogeneity is far more lethal than intra-group heterogeneity**: Changing inter-group data from IID to Non-IID causes 2-12x more accuracy drop than intra-group changes, strongly supporting the theoretical conclusion that $\delta$ is the main bottleneck.
- **Topology preference reverses with the number of groups $G$**: Star-Ring performs best with small $G$ (few large groups) due to longer intra-group update chains, whereas Ring-Star excels with large $G$ (many small groups) as sequential inter-group updates provide fine-grained global alignment.
- **Ring is more aggressive and volatile**: The Ring top layer allows updates to propagate sequentially and traverse the loss landscape faster, but it is more sensitive to data skew and exhibits larger fluctuations.

## Highlights & Insights
- **The value of the unified framework lies in "comparability"**: Unlike prior works analyzing single topologies under disparate assumptions, this paper enables horizontal comparison through unified assumptions and effective learning rates.
- **Elegant use of the Law of Total Variance**: Decomposing global heterogeneity into inter and intra-group terms turns the "bottleneck" question into a fact readable directly from the convergence bound.
- **Counter-intuitive Conclusion**: HFL's main value is scalability rather than convergence acceleration; a well-configured single-tier FL might converge faster than two-tier HFL.
- **Transferable Analysis Paradigm**: The strategy of absorbing architectural parameters into an effective learning rate can be extended to other multi-tier or decentralized optimization comparisons.

## Limitations & Future Work
- Restricted to two-tier HFL; multi-tier (≥3 layers) unified analysis is not yet covered.
- Convergence bounds rely on standard $L$-smoothness and bounded heterogeneity assumptions, which may not directly apply to adaptive optimizers or compressed communication.
- Experimental scale is relatively small ($N=100$); large-scale real-world network/bandwidth heterogeneity performance is only discussed analytically.
- The stability issues of Ring topologies (high volatility under data skew) lack theoretical characterization and automated tuning solutions.

## Related Work & Insights
- **vs. Star-Star Analysis (Wang et al. 2022, Jiang & Zhu 2024)**: These only analyze Star-Star. Ours extends identical assumptions to all four topologies and provides the first analysis for Ring-Ring.
- **vs. Early Star-Ring/Ring-Star Analysis (Chen et al. 2020, Yan et al. 2025)**: Previous works relied on "uniformly bounded gradients" and provided loosed $O(1/\sqrt{R})$ bounds; Ours derives a tighter $O(1/\sqrt{GPMKR})$ bound under more general assumptions.
- **vs. Single-tier FL Theory (Li & Lyu 2023)**: Ours recovers standard FedAvg rates as special cases when $P=M=1, \zeta=0$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified framework for all four HFL topologies with consistent assumptions and tighter bounds.
- Experimental Thoroughness: ⭐⭐⭐⭐ Good coverage across datasets and models, though scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from analytical challenges to actionable principles.
- Value: ⭐⭐⭐⭐⭐ Directly useful for theoretical guidance on HFL system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Non-Convex Federated Optimization under Cost-Aware Client Selection](non-convex_federated_optimization_under_cost-aware_client_selection.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](../../AAAI2026/optimization/data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](../../AAAI2026/optimization/tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](deepafl_deep_analytic_federated_learning.md)
- [\[ICLR 2026\] Federated Learning with Profile Mapping under Distribution Shifts and Drifts](federated_learning_with_profile_mapping_under_distribution_shifts_and_drifts.md)

</div>

<!-- RELATED:END -->
