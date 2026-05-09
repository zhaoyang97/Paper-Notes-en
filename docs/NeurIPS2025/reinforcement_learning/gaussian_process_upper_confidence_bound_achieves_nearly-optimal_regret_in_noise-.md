---
title: >-
  [Paper Note] Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits
description: >-
  [NeurIPS 2025][Reinforcement Learning][Gaussian Process Bandits] This paper proves that GP-UCB achieves nearly-optimal regret in the noise-free GP bandit problem, establishing for the first time $O(1)$ constant cumulative regret under the SE kernel and $O(1)$ cumulative regret under the Matérn kernel (when $d < \nu$), thereby closing a long-standing gap between the theory and practice of GP-UCB.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Gaussian Process Bandits
  - GP-UCB
  - Regret Bounds
  - Noise-Free Optimization
  - RKHS
  - Bayesian Optimization
date: 2026-05-08
content_hash: e483ec9d867fb8b1
---

# Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits

**Conference**: NeurIPS 2025
**arXiv**: [2502.19006](https://arxiv.org/abs/2502.19006)
**Authors**: Shogo Iwazaki (LY Corporation)
**Code**: Not released
**Area**: Reinforcement Learning
**Keywords**: Gaussian Process Bandits, GP-UCB, Regret Bounds, Noise-Free Optimization, RKHS, Bayesian Optimization

## TL;DR

This paper proves that GP-UCB achieves nearly-optimal regret in the noise-free GP bandit problem, establishing for the first time $O(1)$ constant cumulative regret under the SE kernel and $O(1)$ cumulative regret under the Matérn kernel (when $d < \nu$), thereby closing a long-standing gap between the theory and practice of GP-UCB.

## Background & Motivation

### Problem Definition

The noise-free GP bandit problem: a learner makes noise-free observations of a black-box objective function $f$ over a compact domain $\mathcal{X} \subset \mathbb{R}^d$, with the goal of minimizing cumulative regret $R_T = \sum_{t=1}^T f(\mathbf{x}^*) - f(\mathbf{x}_t)$ or simple regret $r_T = f(\mathbf{x}^*) - f(\hat{\mathbf{x}}_T)$, where $f$ belongs to the reproducing kernel Hilbert space (RKHS) associated with a known kernel.

### Existing Challenges

**Theory–practice disconnect**: GP-UCB is the most classical adaptive GP bandit algorithm and consistently outperforms alternatives in practice, yet existing theoretical analyses (Lyu et al. 2019; Kim et al. 2024) yield regret bounds that are strictly sub-optimal.

**Limitations of non-adaptive algorithms**: Algorithms with known nearly-optimal guarantees (e.g., REDS, PE) rely on non-adaptive sampling strategies (uniform sampling or maximum variance reduction), achieving theoretical optimality at the cost of practical performance inferior to GP-UCB.

**Open conjecture**: Vakili (2022) conjectured that the cumulative sum of posterior standard deviations admits tighter upper bounds under the Matérn kernel, but this conjecture had not been confirmed for general algorithms.

### Core Motivation

Given that GP-UCB performs best in practice, can its near-optimality be established theoretically without modifying the algorithm? Answering this question requires fundamentally new analytical tools.

## Method

### Assumptions

- **Assumption 1**: The objective function $f$ belongs to the RKHS of a known kernel $k$ satisfying $k(\mathbf{x}, \mathbf{x}) \leq 1$ and $\|f\|_k \leq B < \infty$.
- Two kernel families are considered: the SE kernel $k_{\text{SE}}(\mathbf{x}, \tilde{\mathbf{x}}) = \exp(-\|\mathbf{x} - \tilde{\mathbf{x}}\|_2^2 / 2\ell^2)$ and the Matérn kernel with smoothness parameter $\nu > 1/2$.

### GP-UCB Algorithm (Algorithm 1)

The noise-free variant of standard GP-UCB selects at each step:

$$\mathbf{x}_t = \arg\max_{\mathbf{x} \in \mathcal{X}} \mu(\mathbf{x}; \mathbf{X}_{t-1}) + B \cdot \sigma(\mathbf{x}; \mathbf{X}_{t-1})$$

where $\mu$ and $\sigma$ denote the GP posterior mean and standard deviation, respectively, and the confidence parameter is set to $\beta^{1/2} = B$ (a deterministic bound requiring no probabilistic relaxation). The algorithm itself is unchanged; the contribution lies entirely in a more refined analysis.

### Core Technical Contribution: Lemma 3 (Algorithm-Independent Posterior Standard Deviation Bound)

This constitutes the central technical breakthrough of the paper. For **any** input sequence $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \mathcal{X}$ (independent of any particular algorithm):

**SE kernel**:
- $\min_{t \in [T]} \sigma(\mathbf{x}_t; \mathbf{X}_{t-1}) = O(\sqrt{T} \exp(-\frac{1}{2} C T^{1/(d+1)}))$ (exponential decay)
- $\sum_{t=1}^T \sigma(\mathbf{x}_t; \mathbf{X}_{t-1}) = O(1)$ (constant!)

**Matérn kernel** ($\nu > 1/2$):
- $\min_{t \in [T]} \sigma(\mathbf{x}_t; \mathbf{X}_{t-1}) = O(T^{-\nu/d} \ln^{\nu/d} T)$
- $\sum_{t=1}^T \sigma = O(T^{(d-\nu)/d} \ln^{\nu/d} T)$ ($d > \nu$); $O(\ln^2 T)$ ($d = \nu$); $O(1)$ ($d < \nu$)

### Proof Sketch

1. **From regret to posterior std**: Using the UCB selection rule and noise-free confidence bounds (Lemma 7), the relations $R_T \leq 2B \sum_t \sigma(\mathbf{x}_t; \mathbf{X}_{t-1})$ and $r_T \leq 2B \min_t \sigma(\mathbf{x}_t; \mathbf{X}_{t-1})$ are established.
2. **Bridging noisy and noise-free settings**: The noisy GP posterior variance $\sigma_{\lambda^2}^2$ is introduced as an upper bound for the noise-free posterior variance, exploiting the monotonicity of variance with respect to the noise parameter.
3. **Elliptical Potential Count Lemma (Lemma 6)**: For any $\lambda > 0$, the cardinality of the set $\{t : \sigma_{\lambda^2}(\mathbf{x}_t) / \lambda > 1\}$ is at most $3\gamma_T(\lambda^2)$.
4. **Backward induction to prove Lemma 5**: The time step with minimal posterior std is identified and removed; monotonicity arguments then yield $\sum_t \sigma \leq \bar{T} - 1 + \sum_{t=\bar{T}}^T \lambda_t$ by induction.
5. **Substituting MIG upper bounds**: $\lambda_t^2 = t \exp(-\tilde{C}t^{1/(d+1)})$ is set for the SE kernel, and $\lambda_t^2 = O(t^{-2\nu/d} \ln^{2\nu/d} t)$ for the Matérn kernel.

## Key Experimental Results

### Table 1: Cumulative Regret Comparison (Noise-Free Algorithms)

| Algorithm | SE Regret | Matérn ($d > \nu$) | Matérn ($d = \nu$) | Matérn ($d < \nu$) | Type |
|---|---|---|---|---|---|
| GP-UCB (Lyu 2019) | $O(\sqrt{T \ln^d T})$ | $\tilde{O}(T^{(\nu+d)/(2\nu+d)})$ | same | same | D |
| REDS (Salgia 2024) | N/A | $\tilde{O}(T^{(d-\nu)/d})$ | $O(\ln^{5/2} T)$ | $O(\ln^{3/2} T)$ | P |
| PE (Iwazaki 2025) | $O(\ln T)$ | $\tilde{O}(T^{(d-\nu)/d})$ | $O(\ln^{2+\alpha} T)$ | $O(\ln T)$ | D |
| **GP-UCB (Ours)** | **$O(1)$** | $\tilde{O}(T^{(d-\nu)/d})$ | $O(\ln^2 T)$ | **$O(1)$** | **D** |
| Lower bound (Li et al.) | N/A | $\Omega(T^{(d-\nu)/d})$ | $\Omega(1)$ | $\Omega(1)$ | N/A |

**Highlight**: GP-UCB achieves $O(1)$ constant regret under the SE kernel and the Matérn kernel with $d < \nu$; under the SE kernel it even surpasses the previously best-known PE algorithm ($O(\ln T)$). In the $d > \nu$ regime, the bound matches the lower bound up to logarithmic factors.

### Table 2: Simple Regret Comparison

| Algorithm | SE Regret | Matérn Regret | Type |
|---|---|---|---|
| GP-EI (Bull 2011) | N/A | $\tilde{O}(T^{-\min(1,\nu)/d})$ | D |
| GP-UCB (Lyu 2019) | $O(\sqrt{\ln^d T / T})$ | $\tilde{O}(T^{-\nu/(2\nu+d)})$ | D |
| MVR (Iwazaki 2025) | $O(\exp(-\frac{1}{2}T^{1/(d+1)}\ln^{-\alpha}T))$ | $\tilde{O}(T^{-\nu/d})$ | D |
| **GP-UCB (Ours)** | $O(\sqrt{T}\exp(-\frac{1}{2}CT^{1/(d+1)}))$ | $\tilde{O}(T^{-\nu/d})$ | **D** |
| Lower bound (Bull 2011) | N/A | $\Omega(T^{-\nu/d})$ | N/A |

**Highlight**: The simple regret of GP-UCB matches the lower bound $\Omega(T^{-\nu/d})$ exactly under the Matérn kernel (up to logarithmic factors) and decays super-exponentially under the SE kernel. Experiments confirm that GP-UCB consistently outperforms non-adaptive algorithms in practice.

## Highlights & Insights

1. **Analysis, not algorithm design**: The paper proposes no new algorithm; it proves that the classical GP-UCB is already sufficient. This exemplifies the paradigm of "the algorithm was right all along — the theory needed to catch up."
2. **Algorithm-independent results**: Lemma 3 holds for arbitrary input sequences and is independent of any specific algorithm, making it directly applicable to GP-TS, EI, GP level-set estimation, multi-objective optimization, and other settings.
3. **Significance of constant regret**: Under the SE kernel, $R_T = O(1)$ implies that total regret remains bounded regardless of runtime — GP-UCB effectively identifies the optimum almost immediately.
4. **Resolution of open conjectures**: The posterior standard deviation cumulative sum bound conjectured by Vakili (2022) is confirmed (up to logarithmic factors), and the general open problem posed by Li et al. is resolved.
5. **Noisy-to-noise-free bridge**: The MIG analysis framework transfers tools from the noisy setting to the noise-free setting, offering a broadly extensible methodological contribution.

## Limitations & Future Work

1. **Missing lower bound for SE kernel**: The optimal simple regret lower bound under the SE kernel remains unknown; the authors conjecture it should be $O(\sqrt{T}\exp(-CT^{2/d}))$, whereas the current upper bound contains $d+1$ rather than $d$ in the exponent.
2. **Logarithmic gap**: Under the Matérn kernel with $d = \nu$, the upper bound is $O(\ln^2 T)$ while the conjectured lower bound is $\Omega(\ln T)$, leaving a gap of one logarithmic factor.
3. **Dependence on MIG bound quality**: The results rely on the MIG upper bounds of Vakili et al. (2021), whose assumption of uniform boundedness of eigenfunctions over general compact domains has been questioned (Janz et al. 2022).
4. **Counter-intuitive behavior in the Bayesian setting**: In the Bayesian setting, GP-UCB requires $\beta^{1/2} = O(\sqrt{\ln T})$, yielding only $O(\sqrt{\ln T})$ rather than $O(1)$ under the SE kernel — an inconsistency with the frequentist setting.
5. **GP-UCB vs. EI in practice**: Although GP-UCB is theoretically nearly-optimal for simple regret, EI still performs better empirically for simple regret minimization, suggesting that constant or logarithmic factors in the theoretical bounds may be large.
6. **Restricted to SE and Matérn kernels**: While the proof strategy generalizes to other kernels (e.g., Neural Tangent Kernel), concrete results are not provided in this work.

## Related Work & Insights

- **Original GP-UCB analysis (Srinivas et al. 2010)**: Established the analytical framework for GP bandits, but yields non-tight regret in the noise-free setting.
- **Noise-free pioneer (Lyu et al. 2019)**: First analyzed noise-free GP-UCB, but obtained regret of $O(\sqrt{T \ln^d T})$, far from optimal.
- **Nearly-optimal non-adaptive algorithms (Iwazaki 2025; Salgia 2024)**: Achieve near-optimality but rely on non-adaptive sampling, leading to inferior practical performance.
- **Lower bounds (Vakili 2022; Li et al.)**: Established cumulative regret lower bounds for the noise-free setting; the upper bounds in this paper match them.
- **Extensible directions**: Lemma 3 can be directly applied to analyze GP-TS (Chowdhury 2017), contextual bandits (Krause 2011), level-set estimation (Gotovos 2013), robust optimization (Bogunovic 2018), and related settings under noise-free conditions; extensions to deep learning-related kernels such as NTK also merit exploration.

## Rating

| Dimension | Score | Comment |
|---|---|---|
| Theoretical Contribution | ⭐⭐⭐⭐⭐ | Closes the fundamental theory–practice gap for GP-UCB and resolves open conjectures |
| Technical Difficulty | ⭐⭐⭐⭐ | The backward induction proof of Lemma 3 is elegant; the MIG bridging approach is insightful |
| Practicality | ⭐⭐⭐⭐ | Directly provides theoretical guarantees for the most widely used GP-UCB without modifying the algorithm |
| Experimental Validation | ⭐⭐⭐ | Experiments are limited in scale ($d=2$) but clearly demonstrate the adaptive vs. non-adaptive gap |
| Writing Quality | ⭐⭐⭐⭐⭐ | Well-structured, with progressively developed proofs and comprehensive comparison tables |
| Overall Recommendation | ⭐⭐⭐⭐½ | A landmark theoretical contribution resolving a core open problem in the GP bandits literature |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [\[NeurIPS 2025\] Improved Regret Bounds for GP-UCB in Bayesian Optimization](improved_regret_bounds_for_gaussian_process_upper_confidence_bound_in_bayesian_o.md)
- [\[NeurIPS 2025\] Dynamic Regret Reduces to Kernelized Static Regret](dynamic_regret_reduces_to_kernelized_static_regret.md)
- [\[NeurIPS 2025\] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)
- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](../../ICLR2026/reinforcement_learning/nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)

</div>

<!-- RELATED:END -->
