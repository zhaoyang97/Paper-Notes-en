---
title: >-
  [Paper Note] Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits
description: >-
  [NeurIPS 2025][Gaussian process] This paper proves that the GP-UCB algorithm achieves nearly-optimal regret bounds in noise-free GP bandit problems…
tags:
  - "NeurIPS 2025"
  - "Gaussian process"
  - "bandit problem"
  - "Bayesian optimization"
  - "regret bound"
  - "noise-free optimization"
date: 2026-05-08
content_hash: a63fa9c17a1ba292
---

# Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits

**Conference**: NeurIPS 2025
**arXiv**: [2502.19006](https://arxiv.org/abs/2502.19006)  
**Code**: None  
**Area**: Other
**Keywords**: Gaussian process, bandit problem, Bayesian optimization, regret bound, noise-free optimization

## TL;DR

This paper proves that the GP-UCB algorithm achieves nearly-optimal regret bounds in noise-free GP bandit problems, including the first $O(1)$ constant cumulative regret under the SE kernel and the Matérn kernel (with $d > \nu$), thereby closing the long-standing gap between the theory of GP-UCB and its empirical performance.

## Background & Motivation

1. **Background**: In noise-free GP bandit problems, the learner seeks to minimize regret for a black-box objective function via noise-free observations. GP-UCB is the most well-known adaptive algorithm, yet existing theoretical bounds remain significantly weaker than known lower bounds.

2. **Limitations of Prior Work**: Existing nearly-optimal algorithms (e.g., REDS, PE) rely on non-adaptive sampling schemes (e.g., uniform sampling or maximum variance reduction), which are theoretically optimal but often underperform adaptive methods such as GP-UCB in practice.

3. **Key Challenge**: Despite its strong empirical performance, GP-UCB's theoretical regret bound is $O(\sqrt{T\ln^d T})$ (SE kernel), which is far weaker than the lower bound $O(1)$ and the $O(\ln T)$ bound achieved by nearly-optimal non-adaptive algorithms.

4. **Goal**: Establish nearly-optimal regret bounds for GP-UCB, bridging the gap between theory and experiment.

5. **Key Insight**: The paper proposes novel algorithm-agnostic upper bounds on the posterior standard deviation (Lemmas 3–5), bridging information-gain analyses from the noisy setting to the noise-free setting.

6. **Core Idea**: Through a refined analysis of the posterior standard deviation, the paper proves that GP-UCB achieves cumulative regret of $O(1)$ (SE kernel) and $\tilde{O}(T^{(d-\nu)/d})$ (Matérn kernel) in the noise-free setting, matching known lower bounds.

## Method

### Overall Architecture

GP-UCB selects at each step the point that maximizes $\mu(\bm{x}; \mathbf{X}_{t-1}) + \beta^{1/2}\sigma(\bm{x}; \mathbf{X}_{t-1})$. The central analytical objective is to derive tighter upper bounds on $\sum_{t=1}^T \sigma(\bm{x}_t; \mathbf{X}_{t-1})$ and $\min_{t \in [T]} \sigma(\bm{x}_t; \mathbf{X}_{t-1})$.

### Key Designs

**1. Novel Upper Bounds on Posterior Standard Deviation (Lemma 3)**

- **Function**: Provides a key technical tool for the regret analysis of GP-UCB.
- **Mechanism**: For an arbitrary input sequence, the following are established: $\sum_{t=1}^T \sigma(\bm{x}_t; \mathbf{X}_{t-1}) = O(1)$ under the SE kernel; and $O(T^{(d-\nu)/d})$, $O(\ln^2 T)$, or $O(1)$ under the Matérn kernel ($\nu > 1/2$), depending on the relationship between $d$ and $\nu$.
- **Design Motivation**: These bounds are algorithm-agnostic and directly yield regret bounds via $R_T \leq 2B\sum_t \sigma(\bm{x}_t; \mathbf{X}_{t-1})$.

**2. Bridging from Information-Gain Analysis**

- **Function**: Extends established analytical techniques from the noisy setting to the noise-free setting.
- **Mechanism**: The paper leverages known upper bounds on the maximum information gain (MIG) $\gamma_T(\lambda^2)$, establishes the relationship between lower bounds on the cumulative posterior standard deviation and MIG, and extracts tighter results by taking the noise-free limit carefully.
- **Design Motivation**: The noise-free setting can be viewed as the limit where observation noise vanishes, but extracting this limit requires careful treatment.

**3. Key Inequality Chain**

- **Function**: Tightly relates regret to the posterior standard deviation.
- **Mechanism**: Using the noise-free confidence bound $|f(\bm{x}) - \mu(\bm{x}; \mathbf{X}_t)| \leq B\sigma(\bm{x}; \mathbf{X}_t)$ and the UCB selection rule, the paper obtains $R_T \leq 2B\sum_t \sigma(\bm{x}_t; \mathbf{X}_{t-1})$ and $r_T \leq 2B\min_t \sigma(\bm{x}_t; \mathbf{X}_{t-1})$.
- **Design Motivation**: Existing analyses apply Cauchy–Schwarz and introduce slack; this work directly bounds the cumulative standard deviation, avoiding such looseness.

### Loss & Training

There is no training process — this is a purely theoretical work. The GP-UCB algorithm uses $\beta^{1/2} = B$ (the RKHS norm bound) and requires no additional tuning.

## Key Experimental Results

### Main Results

Comparison of theoretical cumulative regret bounds (SE kernel and Matérn kernel):

| Algorithm | SE kernel | Matérn ($d>\nu$) | Matérn ($d=\nu$) | Matérn ($d<\nu$) |
|------|------|------------|------------|------------|
| GP-UCB [prior] | $O(\sqrt{T\ln^d T})$ | $\tilde{O}(T^{(\nu+d)/(2\nu+d)})$ | — | — |
| PE [nearly-optimal] | $O(\ln T)$ | $\tilde{O}(T^{(d-\nu)/d})$ | $O(\ln^{2+\alpha} T)$ | $O(\ln T)$ |
| **GP-UCB [Ours]** | **$O(1)$** | $\tilde{O}(T^{(d-\nu)/d})$ | $O(\ln^2 T)$ | **$O(1)$** |
| Lower bound | — | $\Omega(T^{(d-\nu)/d})$ | $\Omega(\ln T)$ | $\Omega(1)$ |

### Ablation Study

Empirical comparison (Figure 1, averaged over 3,000 independent runs):

| Algorithm | SE kernel | Matérn-5/2 | Matérn-3/2 |
|------|---------|---------------|---------------|
| REDS (non-adaptive optimal) | suboptimal | suboptimal | suboptimal |
| PE (non-adaptive optimal) | suboptimal | suboptimal | suboptimal |
| **GP-UCB** | **best** | **best** | **best** |

### Key Findings

- **GP-UCB achieves $O(1)$ cumulative regret under the SE kernel**: The first proof that a fully adaptive algorithm achieves constant regret under the SE kernel.
- **Matérn kernel results match lower bounds**: $\tilde{O}(T^{(d-\nu)/d})$ for $d > \nu$ and $O(1)$ for $d < \nu$.
- **Posterior standard deviation bounds are algorithm-agnostic**: They can be directly applied to convert other UCB algorithms in the noisy setting into nearly-optimal noise-free variants.
- **Theory validates empirical observation**: GP-UCB does indeed outperform non-adaptive nearly-optimal algorithms in practice.

## Highlights & Insights

- **Closes a long-standing theory–practice gap**: The empirical superiority of GP-UCB finally has rigorous theoretical support.
- **Technical contributions are independent of GP-UCB**: The posterior standard deviation bounds in Lemmas 3–5 are broadly applicable to other confidence-bound algorithms.
- **Algorithm simplicity**: GP-UCB itself is extremely simple, requiring only $\beta^{1/2} = B$ with no additional techniques.
- **Constant regret under the SE kernel**: This implies that GP-UCB "almost never errs" after sufficiently many steps.

## Limitations & Future Work

- A logarithmic gap remains between the upper and lower bounds in the $d = \nu$ case ($O(\ln^2 T)$ vs. $\Omega(\ln T)$).
- The analysis relies on known MIG upper bounds; the assumption of uniformly bounded eigenfunctions on general compact domains is debated.
- Only the frequentist (deterministic) setting is considered; analysis under the Bayesian setting may differ.
- Extension to settings with noise that decays over time is a natural direction.

## Related Work & Insights

- The results fill the gap between the suboptimal GP-UCB analysis of Lyu & Tsang (2019) and the lower bounds of Li (2024).
- Technically related to the non-adaptive MVR analysis of Iwazaki (2025), the present work can be viewed as a refined specialization thereof to the noise-free setting.
- Insight: The same algorithm can exhibit fundamentally different optimality properties under different assumptions — in-depth analysis of simple algorithms is a worthwhile investment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First proof of near-optimality for GP-UCB; a landmark result.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; experiments serve as validation only.
- Writing Quality: ⭐⭐⭐⭐⭐ Theorem statements are clear; proof structure is rigorous.
- Value: ⭐⭐⭐⭐⭐ Significant advancement for GP bandit and Bayesian optimization theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Reward Redistribution via Gaussian Process Likelihood Estimation](../../AAAI2026/others/reward_redistribution_via_gaussian_process_likelihood_estimation.md)
- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](infrequent_exploration_in_linear_bandits.md)
- [\[NeurIPS 2025\] 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)
- [\[NeurIPS 2025\] Optimism Without Regularization: Constant Regret in Zero-Sum Games](optimism_without_regularization_constant_regret_in_zero-sum_games.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)

</div>

<!-- RELATED:END -->
