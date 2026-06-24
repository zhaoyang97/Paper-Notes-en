---
title: >-
  [Paper Note] Exploring Large Action Sets with Hyperspherical Embeddings using von Mises-Fisher Sampling
description: >-
  [ICML2025][Reinforcement Learning][Large action space exploration] Ours proposes vMF-exp, which achieves scalable exploration over large-scale action sets (million-scale) by sampling von Mises-Fisher distributed vectors on the hypersphere and then performing nearest-neighbor retrieval. It is theoretically proven to be asymptotically equivalent to Boltzmann exploration under the uniform distribution assumption, and has been successfully deployed in the Deezer music recommendat…
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Large action space exploration"
  - "von Mises-Fisher distribution"
  - "Hyperspherical embeddings"
  - "Boltzmann exploration"
  - "Recommendation systems"
date: 2026-05-08
content_hash: fbf36fa432fd97e6
---

# Exploring Large Action Sets with Hyperspherical Embeddings using von Mises-Fisher Sampling

**Conference**: ICML2025  
**arXiv**: [2507.00518](https://arxiv.org/abs/2507.00518)  
**Code**: [deezer/vMF-exploration](https://github.com/deezer/vMF-exploration)  
**Area**: Reinforcement Learning  
**Keywords**: Large action space exploration, von Mises-Fisher distribution, Hyperspherical embeddings, Boltzmann exploration, Recommendation systems

## TL;DR

Ours proposes vMF-exp, which achieves scalable exploration over large-scale action sets (million-scale) by sampling von Mises-Fisher distributed vectors on the hypersphere and then performing nearest-neighbor retrieval. It is theoretically proven to be asymptotically equivalent to Boltzmann exploration under the uniform distribution assumption, and has been successfully deployed in the Deezer music recommendation system.

## Background & Motivation

### Core Problem

In reinforcement learning, when the action space is extremely large (such as million-scale song catalogs in recommendation systems), achieving efficient action exploration is a critical dilemma. Existing methods face a triple trade-off:

**Random/ε-greedy exploration**: Scalable but ignores embedding information, failing to satisfy the order-preserving property (P3) and assigning equal probabilities to a large number of irrelevant actions.

**Boltzmann exploration (B-exp)**: Satisfies order preservation and unrestricted radius, but requires computing the softmax probability for all $n$ actions, with a complexity of $\mathcal{O}(n)$ which is not scalable.

**Truncated Boltzmann exploration (TB-exp)**: Retrieves $m \ll n$ candidates via ANN first and then performs softmax (i.e., the Wolpertinger architecture). This is scalable but artificially limits the exploration radius.

### Three Desired Properties

The paper formalizes three properties that an ideal exploration strategy should satisfy simultaneously:

- **P1 Scalability**: The sampling time does not exceed the ANN retrieval time (sub-linear complexity with respect to $n$).
- **P2 Unrestricted Radius**: The exploration probability for all actions is non-zero, without being restricted by design to a specific neighborhood.
- **P3 Order Preserving**: $\langle V, X_i \rangle > \langle V, X_j \rangle \Rightarrow P(i|V) > P(j|V)$, meaning larger embedding similarity leads to a higher probability of being selected.

Existing methods cannot simultaneously satisfy P1+P2+P3, which serves as the core motivation of this work.

### Application Scenario

Taking the "inspiration playlist" recommendation on music streaming platforms like Deezer as an example: after a user selects a song, the system needs to recommend the next song step-by-step from millions of tracks. The RL framework allows dynamic adjustment of the policy based on user feedback (like/skip) but requires efficient exploration over the entire candidate space.

## Method

### Problem Setting

- $n$ actions, with each action $i$ represented by a unit-norm embedding vector $X_i \in \mathcal{S}^{d-1}$.
- State vector $V \in \mathcal{S}^{d-1}$ (also normalized to the hypersphere).
- An ANN search engine is available to retrieve nearest neighbors in sub-linear time.

### vMF-exp Algorithm (Two Steps)

**Step 1**: Given the state vector $V$, sample a perturbed vector $\tilde{V}$ from the von Mises-Fisher distribution $\text{vMF}(\kappa, V)$:

$$f_{\text{vMF}}(\tilde{V} | \kappa, V, d) = C_d(\kappa) \cdot e^{\kappa \langle V, \tilde{V} \rangle}$$

where $C_d(\kappa) = \frac{\kappa^{d/2-1}}{(2\pi)^{d/2} I_{d/2-1}(\kappa)}$, and $I_{d/2-1}$ is the modified Bessel function.

**Step 2**: Use the ANN search engine to retrieve the nearest neighbor action $i_{\tilde{V}}^\star$ of $\tilde{V}$ in the embedding space for exploration.

### Role of Hyperparameter $\kappa$

- $\kappa = 0$: vMF degenerates to a uniform distribution on the hypersphere, equivalent to random exploration.
- Larger $\kappa$: Sampling concentrates more around $V$, narrowing the exploration range.
- Smaller $\kappa$: Sampling is more dispersed, expanding the exploration range.

### Why It Works—Voronoï Tessellation Perspective

Each action $X_i$ corresponds to a Voronoï cell $\mathcal{S}_{\text{Voronoï}}(X_i | \mathcal{X}_n)$ on the hypersphere. The probability of selecting action $i$ in vMF-exp equals the integral of the vMF distribution over this cell:

$$P_{\text{vMF-exp}}(i | V, \mathcal{X}_n, \kappa) = \int_{\tilde{V} \in \mathcal{S}_{\text{Voronoï}}(X_i)} f_{\text{vMF}}(\tilde{V} | \kappa, V, d) \, d\tilde{V}$$

This probability is determined by two factors: (1) the similarity between $X_i$ and $V$ (the average value of the vMF density in that region); (2) the area of the Voronoï cell (reflecting the distinctiveness of the action compared to other actions).

### Property Verification

| Property | Random Exploration | B-exp | TB-exp | vMF-exp |
|------|---------|-------|--------|---------|
| P1 Scalable | ✓ | ✗ | ✓ | ✓ |
| P2 Unrestricted Radius | ✓ | ✓ | ✗ | ✓ |
| P3 Order Preserving | ✗ | ✓ | ✓ | ✓* |

*P3 is asymptotically satisfied under the assumption of a uniform embedding distribution.

## Theoretical Analysis

### Core Theorem (Proposition 4.1)

Under the uniform distribution assumption $\mathcal{X}_n \sim \mathcal{U}(\mathcal{S}^{d-1})$:

$$\lim_{n \to +\infty} \frac{P_{\text{B-exp}}(a | n, d, V, \kappa)}{P_{\text{vMF-exp}}(a | n, d, V, \kappa)} = 1$$

That is, as the number of actions approaches infinity, vMF-exp and B-exp assign the same exploration probability to each action.

### Asymptotic Approximation Expression

The zero-order approximation $P_0$ shared by both methods:

$$P_0(a | n, d, V, \kappa) = \frac{f_{\text{vMF}}(A | V, \kappa) \cdot \mathcal{A}(\mathcal{S}^{d-1})}{n}$$

- Convergence rate of B-exp: $P_{\text{B-exp}} = P_0 + o(1/(n\sqrt{n}))$
- Convergence rate of vMF-exp: $P_{\text{vMF-exp}} = P_0 + \mathcal{O}(1/n^{1+2/(d-1)})$ ($d > 2$)

### High-Dimensional Correction (Proposition 4.4)

When the dimension $d$ is large ($d \geq 20$), a first-order correction term $P_1$ is required to better approximate the true probability of vMF-exp. The sign of the correction term depends on $\langle V, A \rangle$: actions similar to $V$ have a slightly lower probability under vMF-exp sampling compared to B-exp, and vice-versa—meaning vMF-exp tends to explore more thoroughly in high dimensions.

## Key Experimental Results

### Monte Carlo Simulation

- Conducted 8 million repeated experiments for different combinations of $(d, \kappa, \langle V, A \rangle)$.
- When $d$ is small ($d \leq 8$), $P_{\text{vMF-exp}}$ and $P_{\text{B-exp}}$ are almost indistinguishable.
- When $d \geq 16$, the first-order approximation $P_1$ clearly outperforms the zero-order approximation $P_0$.
- All results are consistent with theoretical predictions.

### GloVe Real Data Verification

- Used 1 million GloVe word embedding vectors (normalized).
- Despite not satisfying the i.i.d. uniform distribution assumption, the theoretical approximation remains accurate.
- Verified that vMF-exp simultaneously satisfies P1, P2, and P3.

### Deezer Production Environment Deployment

- Successfully deployed for several months in the "inspiration playlist" recommendation system on the global music streaming platform Deezer.
- Explored million-scale candidate songs with a sampling latency of only a few milliseconds.
- Verified positive outcomes through global A/B testing.
- Confirmed the scalability and practicality of the method in real large-scale systems.

## Highlights & Insights

1. **Clever Continuous-to-Discrete Bridging**: Formulates the exploration problem in a discrete action space as continuous vMF sampling + nearest-neighbor retrieval on the hypersphere, entirely bypassing the $\mathcal{O}(n)$ bottleneck of softmax.
2. **Theoretical Rigor**: Not only proves asymptotic equivalence, but also provides convergence rates and high-dimensional correction formulas for different dimensions.
3. **Voronoï Geometric Intuition**: Provides an intuitive probabilistic explanation via Voronoï tessellation—actions in low-density regions have larger cells and thus are explored more, which is a beneficial "sparsity bias."
4. **Industrial Validation**: A complete validation pipeline spanning theory, Monte Carlo simulation, real public database testing, and production deployment.
5. **Connection to Thompson Sampling**: Shows an interesting structural similarity with Thompson Sampling (sampling from a distribution followed by greedy selection).

## Limitations & Future Work

1. **Theoretical Guarantee Limited to Uniform Distribution Assumption**: Actual embeddings usually do not satisfy the i.i.d. uniform assumption (e.g., they have clustering structures). Although experiments show that the theoretical approximation remains effective, there is no rigorous guarantee.
2. **Slow Convergence in High Dimensions**: As $d$ increases, vMF-exp requires a larger $n$ to approximate B-exp, with the second-order error term decaying at a rate of $\mathcal{O}(1/n^{2/(d-1)})$.
3. **In-depth Analysis of ANN Error Lacking**: The theory assumes exact nearest neighbors, while the approximation errors of ANN under extreme scale might affect exploration quality.
4. **Hyperparameter $\kappa$ Tuning**: How to adaptively select $\kappa$ for specific applications is not discussed in detail.
5. **Clustered Embeddings Unconsidered**: Embeddings in recommendation systems often have clustered structures (e.g., music genres). Over such distributions, the Voronoï cell area distribution is uneven, and the extent to which P3 is satisfied warrants further study.

## Related Work & Insights

- **Wolpertinger Architecture**: Performs policy selection after retrieving via ANN; vMF-exp can be viewed as a more elegant alternative.
- **Directional Statistics**: Introducing the vMF distribution from directional statistics into RL exploration is a key innovation of this work.
- **Thompson Sampling**: vMF-exp shares a "sample-then-exploit" philosophy with Thompson Sampling.
- **RL in Recommendation Systems**: Platforms like YouTube have adopted TB-exp; this paper provides a theoretically guaranteed alternative that does not require truncation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of introducing the vMF distribution to large action space exploration is simple and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four-tier validation: theory + simulation + real public data + industrial deployment.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem definitions, rigorous theoretical derivations, and fluent presentation.
- Value: ⭐⭐⭐⭐ — Highly practical for large-scale recommendation system scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](../../ICLR2026/reinforcement_learning/ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [\[ICML 2025\] Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation](demystifying_the_paradox_of_importance_sampling_with_an_estimated_history-depend.md)
- [\[ICML 2025\] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments](fast_and_robust_task_sampling_with_posterior_and_diversity_synergies_for_adaptiv.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[ICML 2025\] Action-Dependent Optimality-Preserving Reward Shaping (ADOPS)](action-dependent_optimality-preserving_reward_shaping.md)

</div>

<!-- RELATED:END -->
