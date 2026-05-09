---
title: >-
  [Paper Note] Locally Optimal Private Sampling: Beyond the Global Minimax
description: >-
  [NeurIPS 2025][AI Safety][Local Differential Privacy] Under local differential privacy (LDP), this paper proposes a **local minimax** framework that leverages neighborhood constraints defined by a public distribution $P_0$ to derive closed-form optimal samplers, achieving **consistent and significant improvements over the global minimax sampler** both theoretically and empirically.
tags:
  - NeurIPS 2025
  - AI Safety
  - Local Differential Privacy
  - Private Sampling
  - Minimax Optimality
  - $f$-Divergence
  - Public Data
date: 2026-05-08
content_hash: 3de63e373cb3c793
---

# Locally Optimal Private Sampling: Beyond the Global Minimax

**Conference**: NeurIPS 2025
**arXiv**: [2510.09485](https://arxiv.org/abs/2510.09485)
**Code**: [GitHub](https://github.com/hradghoukasian/private_sampling)
**Area**: AI Security
**Keywords**: Local Differential Privacy, Private Sampling, Minimax Optimality, $f$-Divergence, Public Data

## TL;DR

Under local differential privacy (LDP), this paper proposes a **local minimax** framework that leverages neighborhood constraints defined by a public distribution $P_0$ to derive closed-form optimal samplers, achieving **consistent and significant improvements over the global minimax sampler** both theoretically and empirically.

## Background & Motivation

**Background**: Local differential privacy (LDP) allows users to randomize data on-device before sharing, eliminating the need to trust a central server. It has been widely deployed by Google, Apple, and Microsoft. Private sampling refers to the task of generating a single sample from a distribution $P$ subject to LDP constraints.

**Limitations of Prior Work**: Park et al. (NeurIPS'24) established a global minimax framework that finds the worst-case optimal sampler. However, the global minimax criterion is **overly pessimistic**—it assumes the adversary may select an arbitrary distribution, whereas real-world data distributions are unlikely to be worst-case (degenerate) distributions.

**Key Challenge**: The global minimax optimal sampler applies a uniform transformation across all input distributions and cannot exploit the practical prior knowledge that the data distribution lies near a known reference distribution, leading to suboptimal privacy–utility tradeoffs.

**Goal**: Establish a **local minimax** framework—when the data distribution $P$ is known to reside near a known public distribution $P_0$, can better private samplers be designed?

**Key Insight**: The neighborhood $N_\gamma(P_0) = \{P: E_\gamma(P\|P_0) = E_\gamma(P_0\|P) = 0\}$ is defined via the $E_\gamma$-divergence (hockey-stick divergence), and the minimax optimal sampler is derived under this constraint. A key observation is that **the local minimax risk equals the global minimax risk restricted to distributions within the neighborhood**.

**Core Idea**: Extend the global minimax framework from pure LDP to functional LDP, then show that the locally minimax optimal sampler is the global optimal sampler restricted to the $P_0$-neighborhood, yielding a closed-form solution that is independent of the choice of $f$-divergence.

## Method

### Overall Architecture

The theoretical construction proceeds in two layers:
1. **Global minimax under functional LDP**: Generalizes the results of Park et al. to general functional LDP (encompassing approximate DP and Gaussian LDP).
2. **Local minimax**: Derives the neighborhood-constrained optimal sampler from the global result.

### Key Designs

#### 1. Global Minimax Optimal Sampler (Theorem 3.4)

- **Function**: Finds the minimax optimal sampler satisfying $g$-FLDP for the distribution class $\tilde{\mathcal{P}}_{c_1,c_2,h}$ and any $f$-divergence.
- **Core Formula**:
$$q_g^\star(P)(x) = \lambda^\star p(x) + (1 - \lambda^\star) h(x)$$
  where $\lambda^\star_{c_1,c_2,g} = \inf_{\beta \geq 0} \frac{e^\beta + \frac{c_2 - c_1}{1-c_1}(1 + g^*(-e^\beta)) - 1}{(1-c_1)e^\beta + c_2 - 1}$
- **Interpretation**: With probability $\lambda^\star$, draw from the original distribution $P$; with probability $1 - \lambda^\star$, draw from the reference distribution $h$—a natural privacy–utility mixture.

#### 2. Locally Minimax Optimal Sampler — Functional LDP (Theorem 4.1)

- **Function**: Solves for the minimax optimal sampler over the neighborhood $N_\gamma(P_0)$.
- **Mechanism**: Shows that the local minimax risk equals the global minimax risk obtained by setting $c_1 = 1/\gamma$, $c_2 = \gamma$, and $h = p_0$.
- **Formula**: The local solution follows directly by substituting these values of $c_1$, $c_2$, and $h$ into the global result.
- **Design Motivation**: The neighborhood $N_\gamma(P_0)$ constrains the density ratio $p(x)/p_0(x) \in [1/\gamma, \gamma]$, which naturally corresponds to the distribution class parameters in the global framework.

#### 3. Locally Minimax Optimal Sampler — Pure LDP (Theorem 5.1)

- **Function**: Derives a stronger, pointwise optimal sampler under pure $\varepsilon$-LDP.
- **Core Formula**:
$$q(x) = \text{clip}\left(\frac{1}{r_P} p(x); \frac{\gamma+1}{\gamma+e^\varepsilon} p_0(x), \frac{(\gamma+1)e^\varepsilon}{\gamma+e^\varepsilon} p_0(x)\right)$$
- **Key Advantage**: This is a **nonlinear sampler** that adapts pointwise to each input distribution $P$ via clipping, strictly dominating the linear sampler of Theorem 4.1 (Proposition 5.2).
- **Design Motivation**: The linear sampler applies a uniform transformation across all distributions, whereas the nonlinear sampler minimizes the $f$-divergence for each specific $P$ in an instance-optimal manner.

### Theoretical Properties

- **$f$-Divergence Independence**: The structure of the optimal sampler does not depend on the specific choice of $f$-divergence (KL, TV, Hellinger, etc.).
- **Pointwise Dominance**: $D_f(P \| \mathbf{Q}^\star_\varepsilon(P)) \leq D_f(P \| \mathbf{Q}^\star_{g_\varepsilon}(P))$ for all $P \in N_\gamma(P_0)$.
- **Generality**: Pure LDP, approximate LDP, and Gaussian LDP are all special cases of functional LDP.

## Key Experimental Results

### Discrete Setting ($k=20$, uniform reference)

| $\varepsilon$ | Metric | Global Optimal | Local Optimal |
|-----------|------|---------|---------|
| 0.1 | KL | ~0.18 | ~0.065 |
| 0.5 | KL | ~0.14 | ~0.035 |
| 1.0 | KL | ~0.10 | ~0.018 |
| 2.0 | KL | ~0.04 | ~0.004 |
| 0.1 | TV | ~0.14 | ~0.09 |
| 1.0 | TV | ~0.08 | ~0.04 |

The local sampler consistently outperforms the global sampler across all privacy levels and divergence metrics.

### Continuous Setting (1D Laplace mixture, 100 client distributions)

| $\varepsilon$ | Metric | Global Worst-Case | Local Worst-Case |
|-----------|------|---------|---------|
| 0.1 | KL | ~0.18 | ~0.04 |
| 0.5 | KL | ~0.12 | ~0.02 |
| 1.0 | KL | ~0.07 | ~0.008 |
| 2.0 | KL | ~0.025 | ~0.002 |
| 0.1 | Hellinger² | ~0.06 | ~0.02 |
| 1.0 | Hellinger² | ~0.028 | ~0.003 |

### Key Findings

1. **Consistent and significant improvement**: The local sampler reduces the worst-case $f$-divergence by a factor of 2–10 across all settings.
2. **Larger gains at low privacy budgets (small $\varepsilon$)**: At $\varepsilon=0.1$, KL divergence drops from 0.18 to 0.04.
3. **Pure LDP nonlinear sampler outperforms functional LDP linear sampler**: Pointwise clipping yields additional gains.
4. **Effectiveness under Gaussian LDP**: Experimental results are provided in the appendix.

## Highlights & Insights

1. **Elegant theoretical reduction**: The local minimax problem reduces to a specially parameterized global minimax problem in a single step.
2. **$f$-Divergence independence**: A remarkably strong theoretical guarantee—optimality holds regardless of the divergence used to measure utility.
3. **Clipping as optimality**: The nonlinear sampler under pure LDP is essentially a clipping operation—both intuitive and concise.
4. **Effective use of public data**: Unlike Zamanlooy et al., the public data here **genuinely reduces the minimax risk**.
5. **From global pessimism to local realism**: The paper provides theoretical justification that leveraging prior information can substantially improve the privacy–utility tradeoff.

## Limitations & Future Work

1. **Evaluation limited to synthetic data**: Experiments on high-dimensional real-world datasets are absent.
2. **Computational cost**: The clipping sampler incurs high computational overhead in high dimensions due to the normalization constant $r_P$.
3. **Selection of neighborhood parameter $\gamma$**: How to determine $P_0$ and $\gamma$ in practice remains an open problem.
4. **Single-sample setting**: The framework addresses single-sample generation only; extensions to the multi-sample regime are not considered.
5. **Integer constraint $\gamma \in \mathbb{N}$**: A technical limitation, though it can be overcome via amplification.
6. **Generalization from $E_\gamma$-divergence neighborhoods to general $f$-divergence neighborhoods**: Left for future work.

## Related Work & Insights

- **Relation to Park et al. (NeurIPS'24)**: This work builds directly on their global framework; the key innovation is the introduction of neighborhood constraints.
- **Distinction from Zamanlooy et al.**: Their approach requires samplers to preserve the public prior (a hard constraint), whereas this paper treats public data as the center of a neighborhood (a soft constraint)—the latter is more flexible and provably reduces the minimax risk.
- **Feldman et al.**: Employ the same $E_\gamma$-neighborhood definition, but in the context of density estimation rather than sampling.
- **Broader inspiration**: Application of local minimax to other DP problems (distribution estimation, learning, generation).

## Rating

⭐⭐⭐⭐ (4/5)
- The theoretical contributions are elegant and refined; the reduction from global to local minimax is impressive.
- However, the work lacks experiments on real-world data and validation of high-dimensional scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Last-Click: An Optimal Mechanism for Ad Attribution](beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](private_zeroth-order_optimization_with_public_data.md)
- [\[NeurIPS 2025\] Private Continual Counting of Unbounded Streams](private_continual_counting_of_unbounded_streams.md)

</div>

<!-- RELATED:END -->
