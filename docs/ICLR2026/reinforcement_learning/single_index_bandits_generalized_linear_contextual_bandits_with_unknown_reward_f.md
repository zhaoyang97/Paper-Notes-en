---
title: >-
  [Paper Note] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions
description: >-
  [ICLR 2026][Reinforcement Learning][Contextual multi-armed bandits] This paper introduces the Single Index Bandit (SIB) problem — extending generalized linear bandits to the setting where the reward function is unknown —…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Contextual multi-armed bandits"
  - "generalized linear models"
  - "single index models"
  - "Stein's method"
  - "regret bounds"
date: 2026-05-08
content_hash: 4722f9da54366b75
---

# Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions

**Conference**: ICLR 2026
**arXiv**: [2506.12751](https://arxiv.org/abs/2506.12751)  
**Code**: None  
**Area**: Reinforcement Learning / Online Learning
**Keywords**: Contextual multi-armed bandits, generalized linear models, single index models, Stein's method, regret bounds

## TL;DR

This paper introduces the Single Index Bandit (SIB) problem — extending generalized linear bandits to the setting where the reward function is unknown — and proposes a family of efficient algorithms (STOR/ESTOR/GSTOR) based on Stein's method, achieving near-optimal regret $\tilde{O}(\sqrt{T})$ under monotone increasing reward functions.

## Background & Motivation

- **Background**: Generalized linear bandits (GLB) are an important extension of contextual bandits and have been widely applied in recommendation systems, clinical trials, and precision medicine. However, all existing methods assume the reward function (link function) is known.
- **Limitations of Prior Work**: Misspecification of the reward function causes existing GLB algorithms to fail entirely, potentially incurring linear regret. In practice, the underlying functional form is typically unknown and not identifiable.
- **Key Challenge**: Existing UCB and Thompson Sampling methods both require solving (quasi-)maximum likelihood estimators, which inherently depend on the explicit form of the reward function. Similarly, all existing theoretical analyses rely on vector-valued martingale concentration inequalities that explicitly involve the reward function — techniques that completely break down when the reward function is unknown.
- **Goal**: Design efficient bandit algorithms with sublinear regret guarantees when the reward function is entirely unknown.
- **Key Insight**: Drawing on the single index model (SIM) from statistical learning, Stein's method is employed to bypass dependence on the functional form of the reward, enabling direct estimation of the unknown parameter direction.
- **Core Idea**: Leveraging Stein's identity $\mathbb{E}[y_i S(x_i)] = \mu_* \theta_*$, the direction of $\theta_*$ can be estimated without knowledge of $f(\cdot)$, enabling robust optimization under unknown reward functions.

## Method

### Overall Architecture

- **Problem Setup**: At time $t$, the agent selects an arm $x_t$ from the arm set $\mathcal{X}_t = \{x_{t,a} \in \mathbb{R}^d : a \in [K]\}$ and observes reward $y_t = f(x_t^\top \theta_*) + \eta_t$, where both $f(\cdot)$ and $\theta_*$ are unknown.
- **Three-level progression**: STOR (EtC framework, $\tilde{O}(T^{2/3})$) → ESTOR (epoch scheduling, $\tilde{O}(\sqrt{T})$) → GSTOR (general non-monotone $f$, $\tilde{O}(T^{3/4})$).

### Key Designs

#### 1. Stein-based Parameter Estimator

- **Function**: Estimates the direction of $\theta_*$ without knowledge of $f(\cdot)$.
- **Mechanism**: Stein's identity is used to establish $\mathbb{E}[y_i S(x_i)] = \mu_* \theta_*$, where $S(x) = -\nabla_x \log p(x)$ is the score function of the distribution and $\mu_* = \mathbb{E}[f'(X^\top \theta_*)]$. The estimator is:
$$\hat{\theta} = \arg\min_{\theta \in \Theta} \|\theta\|_2^2 - \frac{2}{n} \sum_{i=1}^n \phi_\tau(y_i \cdot S(x_i))^\top \theta + \lambda \|\theta\|_1$$
  where $\phi_\tau$ is an element-wise truncation function controlling the variance-bias tradeoff under heavy-tailed noise.
- **Estimation accuracy**: $\|\hat{\theta} - \mu_* \theta_*\|_2 = \tilde{O}(\sqrt{d/n})$, achieving minimax optimality.
- **Design Motivation**: No iterative optimization is required; the closed-form solution runs in $O(nd)$ time and $O(d)$ space, far more efficient than MLE solvers used in GLB.

#### 2. STOR: Explore-then-Commit Baseline

- **Function**: The simplest EtC-based implementation.
- **Mechanism**: Randomly explore for the first $T_1$ rounds to collect samples and compute $\hat{\theta}$; greedily select $x_t = \arg\max_{x \in \mathcal{X}_t} x^\top \hat{\theta}$ in all remaining rounds.
- **Regret bound**: $R_T = \tilde{O}(d^{2/3} T^{2/3})$, suboptimal due to the inherent limitations of the EtC framework.

#### 3. ESTOR: Epoch-Scheduled Improved Algorithm

- **Function**: Achieves near-optimal regret through carefully designed epoch scheduling.
- **Mechanism**: Uses exponentially growing epoch lengths $e_i = (2^i - 1)T_0$. At the start of each epoch, $\hat{\theta}_i$ is updated using data from the previous epoch, and the score function distribution is recomputed as $p_i(x) = K \cdot p(x) \cdot F_i(x^\top \hat{\theta}_i)^{K-1}$.
- **Regret bound**: $R_T = \tilde{O}(dK^{3/2}\sqrt{T})$, achieving near-optimal $\tilde{O}_T(\sqrt{T})$ in $T$.
- **Design Motivation**: Short initial epochs enable rapid exploration; long later epochs accumulate samples for accurate estimation, yielding geometrically decreasing estimation error.
- **Computational efficiency**: $O(dT)$ time and $O(d)$ space, among the most efficient of existing GLB algorithms.

#### 4. Sparse High-Dimensional Extension

- **Function**: Extends the method to settings where $\theta_*$ has at most $s \ll d$ nonzero entries.
- **Mechanism**: Incorporates $\ell_1$ regularization $\lambda > 0$ into the estimator; knowledge of the sparsity level $s$ is not required.
- **Regret bound**: The regret of ESTOR replaces $d$ with $s$, yielding $R_T = \tilde{O}(sK^{3/2}\sqrt{T})$.

#### 5. GSTOR: General Reward Functions

- **Function**: Handles general continuously differentiable reward functions that may be non-monotone.
- **Mechanism**: A two-phase explore-then-exploit strategy — the first phase uses the Stein estimator to estimate $\hat{\theta}$; the second phase applies kernel regression $\hat{f}(z) = \frac{\sum_i y_i K_h(z - x_i^\top \hat{\theta}_0)}{\sum_i K_h(z - x_i^\top \hat{\theta}_0)}$ to approximate the unknown link function, followed by greedy exploitation.
- **Regret bound**: $\mathbb{E}(R_T) = O(d^{3/8} T^{3/4})$.

### Loss & Training

The estimator loss is a simple $\ell_2 + \ell_1$ regularized quadratic form with a closed-form solution when $\lambda = 0$.

## Key Experimental Results

### Main Results

Comparison across four link functions ($T=10{,}000$, $d=10$):

| Method | Linear $f(x)=x$ | Poisson $f(x)=e^x$ | Square $f(x)=\text{sign}(x)x^2+2x$ | Fifth $f(x)=x^5$ |
|---|---|---|---|---|
| LinUCB/UCB-GLM | Optimal under correct specification | Acceptable under correct specification | Linear regret under misspecification | Linear regret under misspecification |
| ESTOR | On par with correctly specified LinUCB | On par with UCB-GLM | **Significantly outperforms** misspecified GLB | **Significantly outperforms** misspecified GLB |
| Runtime | Hundreds of times faster than UCB-GLM | Thousands of times faster than GLM-TSL | Same as left | Same as left |

### Ablation Study

- **Misspecification experiment**: Under Square/Fifth link functions, fitting GLB algorithms with incorrect link functions leads to severe performance degradation.
- **High-dimensional sparse experiment**: ESTOR maintains $\sqrt{T}$ regret rate under $d=100, s=5$.
- **Real data**: On Forest Cover Type and Yahoo News datasets, all SIB algorithms consistently outperform GLB methods.

### Key Findings

1. When the link function is correctly specified, ESTOR matches the performance of optimal algorithms with known reward functions (LinUCB, UCB-GLM).
2. When the link function is misspecified, GLB algorithms degrade severely while ESTOR/STOR remain robust.
3. ESTOR substantially outperforms all GLB baselines in computational efficiency (by factors of hundreds to thousands).
4. On real data, where the underlying link function is typically unknown, the advantage of SIB methods is even more pronounced.

## Highlights & Insights

1. **High value of problem formulation**: The SIB problem is formally introduced for the first time, filling an important theoretical gap in the GLB literature.
2. **Elegant application of Stein's method**: The score function is used to bypass dependence on $f(\cdot)$; the identity $\mathbb{E}[yS(x)] = \mu_* \theta_*$ is remarkably elegant.
3. **Dual theoretical and practical contributions**: Near-optimal regret bounds are established alongside a highly practical algorithm (closed-form solution, no iteration, $O(d)$ space).
4. **Novel use of truncation**: Truncation techniques from heavy-tailed noise handling are repurposed to manage ambiguity arising from unknown reward functions.

## Limitations & Future Work

1. **Distributional assumption**: The arm set is assumed to be i.i.d. sampled from a fixed distribution $\mathcal{D}$; adversarial arm selection is not supported, representing a significant theoretical limitation.
2. **Gaussian assumption in GSTOR**: The algorithm for general reward functions relies on a Gaussian design assumption, which may not hold in practice.
3. **Dependence on $K$**: ESTOR has a worst-case $K^{3/2}$ dependence on the number of arms, though this improves to $\sqrt{\log K}$ in the Gaussian case.
4. **Optimality of $T^{3/4}$ unclear**: Whether the $T^{3/4}$ bound for general non-monotone functions can be improved remains an open problem.

## Related Work & Insights

- **GLB literature**: UCB-GLM (Li et al., 2017) and GLM-TSL (Kveton et al., 2020) are standard baselines, but both require known link functions.
- **SIM statistical learning**: Stein's method has been applied in low-rank matrix bandits (Kang et al., 2022); this paper is the first to introduce it to the linear/GLB setting.
- **Realizable contextual bandits**: Such methods require powerful regression oracles, whereas SIM lacks oracles with finite-sample guarantees satisfying the required conditions.
- **Insights**: The potential of Stein's method in online learning warrants further exploration, particularly in practical settings where the model is only partially known.

## Rating

⭐⭐⭐⭐⭐ (5/5)

- **Novelty**: ⭐⭐⭐⭐⭐ Novel problem formulation, strong methodological originality, significant theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on both synthetic and real data with thorough comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with clear progressive structure.
- **Value**: ⭐⭐⭐⭐ Efficient and easy-to-implement algorithm, though distributional assumptions limit the scope of applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates](../../ICML2026/reinforcement_learning/practical_and_optimal_algorithm_for_linear_contextual_bandits_with_rare_paramete.md)
- [\[ICLR 2026\] Revisiting Matrix Sketching in Linear Bandits: Achieving Sublinear Regret via Dyadic Block Sketching](revisiting_matrix_sketching_in_linear_bandits_achieving_sublinear_regret_via_dya.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](../../NeurIPS2025/reinforcement_learning/tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](../../NeurIPS2025/reinforcement_learning/generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [\[ICLR 2026\] Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)

</div>

<!-- RELATED:END -->
