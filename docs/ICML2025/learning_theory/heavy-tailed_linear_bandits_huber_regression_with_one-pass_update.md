---
title: >-
  [Paper Note] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update
description: >-
  [ICML2025][Others/Bandits][heavy-tailed bandits] This paper proposes Hvt-UCB, a one-pass Huber regression algorithm based on Online Mirror Descent for linear bandits with heavy-tailed noise. It reduces the per-round computational complexity from $\mathcal{O}(t\log T)$ to $\mathcal{O}(1)$ while maintaining an optimal and instance-dependent regret bound.
tags:
  - "ICML2025"
  - "Others/Bandits"
  - "heavy-tailed bandits"
  - "Huber regression"
  - "online mirror descent"
  - "linear bandits"
  - "one-pass algorithm"
date: 2026-05-08
content_hash: f2eecaf22ac4f035
---

# Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update

**Conference**: ICML2025  
**arXiv**: [2503.00419](https://arxiv.org/abs/2503.00419)  
**Code**: None  
**Area**: Others/Bandits  
**Keywords**: heavy-tailed bandits, Huber regression, online mirror descent, linear bandits, one-pass algorithm

## TL;DR

This paper proposes Hvt-UCB, a one-pass Huber regression algorithm based on Online Mirror Descent for linear bandits with heavy-tailed noise. It reduces the per-round computational complexity from $\mathcal{O}(t\log T)$ to $\mathcal{O}(1)$ while maintaining an optimal and instance-dependent regret bound.

## Background & Motivation

**Problem Setting**: In stochastic linear bandits (SLB), the reward observed in each round is $r_t = X_t^\top \theta_* + \eta_t$, where the noise $\eta_t$ may follow a heavy-tailed distribution (with only finite $(1+\varepsilon)$-th moment, $\varepsilon \in (0,1]$). This is common in scenarios such as financial markets and online advertising.

**Limitations of Prior Work**:

| Method Type | Representative Algorithms | Main Limitations |
|---------|---------|---------|
| Truncation | TOFU, CRTM | Requires absolute moment bounded assumption, suboptimal in noiseless cases |
| Median-of-Means (MOM) | MENU, CRMM | Relies on fixed arm set and repetitive pulling mechanism |
| Adaptive Huber Regression | HEAVY-OFUL | Needs to store all historical data, per-round calculation is $\mathcal{O}(t\log T)$ |

**Core Motivation**: Although HEAVY-OFUL (Huang et al., 2023) achieves an optimal and instance-dependent regret bound without relying on additional noise assumptions, its computational cost is prohibitively high—requiring a traversal of all historical data to solve a strongly convex optimization problem in each round. The authors ask: **Can we design a one-pass algorithm based on Huber loss that significantly reduces the computational cost while maintaining optimal regret?**

## Method

### Overall Architecture

The **Hvt-UCB** algorithm embeds adaptive Huber regression into the Online Mirror Descent (OMD) framework, replacing full-batch optimization with single-step projected gradient updates.

### Huber Loss

$$f_\tau(x) = \begin{cases} \frac{x^2}{2} & |x| \leq \tau \\ \tau|x| - \frac{\tau^2}{2} & |x| > \tau \end{cases}$$

The Huber loss is quadratic (maintaining strong convexity) when $|x| \leq \tau$ and linearized when $|x| > \tau$ (reducing outlier impact). Based on this, the bandit loss is defined as:

$$\ell_t(\theta) = f_{\tau_t}\!\left(\frac{r_t - X_t^\top \theta}{\sigma_t}\right)$$

where $\sigma_t$ is the normalization factor and $\tau_t$ is the dynamic robustification parameter.

### OMD One-Pass Update (Key Designs)

Unlike HEAVY-OFUL which solves a full-batch optimization $\hat\theta_t = \arg\min_\theta \lambda\|\theta\|^2 + \sum_{s=1}^{t} \ell_s(\theta)$ in each round, this paper adopts the OMD update:

$$\hat\theta_{t+1} = \arg\min_{\theta \in \Theta} \left\{ \langle \theta, \nabla\ell_t(\hat\theta_t) \rangle + \mathcal{D}_{\psi_t}(\theta, \hat\theta_t) \right\}$$

where $\mathcal{D}_{\psi_t}$ is the Bregman divergence, and the regularizer is chosen as the local norm:

$$\psi_t(\theta) = \frac{1}{2}\|\theta\|_{V_t}^2, \quad V_t = \lambda I + \frac{1}{\alpha}\sum_{s=1}^{t}\frac{X_s X_s^\top}{\sigma_s^2}$$

Equivalently, one step of projected gradient descent is executed in each round:

$$\tilde\theta_{t+1} = \hat\theta_t - V_t^{-1}\nabla\ell_t(\hat\theta_t), \quad \hat\theta_{t+1} = \Pi_\Theta(\tilde\theta_{t+1})$$

$V_t^{-1}$ is recursively updated via the Sherman-Morrison-Woodbury formula, costing only $\mathcal{O}(d^2)$ per round.

### Recursive Design of Parameter Settings

The configuration of the normalization factor $\sigma_t$ and the robustification parameter $\tau_t$ is crucial:

- $\sigma_t = \max\!\left\{\nu_t,\; \sigma_{\min},\; \sqrt{\frac{2\beta_{t-1}}{\tau_0\sqrt{\alpha}\, t^{\frac{1-\varepsilon}{2(1+\varepsilon)}}}}\|X_t\|_{V_{t-1}^{-1}}\right\}$
- $\tau_t = \tau_0 \cdot \frac{\sqrt{1+w_t^2}}{w_t} \cdot t^{\frac{1-\varepsilon}{2(1+\varepsilon)}}$

where $w_t = \frac{1}{\sqrt{\alpha}}\|X_t/\sigma_t\|_{V_{t-1}^{-1}}$ and $\beta_t$ is the confidence bound radius. This recursive design ensures that the denoised data falls into the quadratic region of the Huber loss, thereby guaranteeing the Hessian lower bound.

### UCB Arm Selection Strategy

$$X_t = \arg\max_{\mathbf{x} \in \mathcal{X}_t}\left\{\langle \mathbf{x}, \hat\theta_t \rangle + \beta_{t-1}\|\mathbf{x}\|_{V_{t-1}^{-1}}\right\}$$

## Theoretical Results

### Main Theorem (Theorem 1): Instance-Dependent Regret Bound

$$\text{Reg}_T \leq \widetilde{\mathcal{O}}\!\left(d\, T^{\frac{1-\varepsilon}{2(1+\varepsilon)}}\sqrt{\sum_{t=1}^T \nu_t^2} + d\, T^{\frac{1-\varepsilon}{2(1+\varepsilon)}}\right)$$

### Corollary (Corollary 1): Optimal Regret under Unified Moment Bound

When only the global moment bound $\nu$ is known: $\text{Reg}_T \leq \widetilde{\mathcal{O}}(d\, T^{1/(1+\varepsilon)})$, matching the lower bound $\Omega(d\, T^{1/(1+\varepsilon)})$.

### Comparison Summary of Methods

| Method | Regret | Complexity per Round | Extra Assumptions |
|------|--------|---------|---------|
| MENU (MOM) | $\widetilde{\mathcal{O}}(dT^{1/(1+\varepsilon)})$ | $\mathcal{O}(\log T)$ | Fixed arm set + repetitive pulling |
| TOFU (Truncation) | $\widetilde{\mathcal{O}}(dT^{1/(1+\varepsilon)})$ | $\mathcal{O}(t)$ | Absolute moment bounded |
| HEAVY-OFUL (Huber) | Instance-dependent, optimal | $\mathcal{O}(t\log T)$ | None |
| **Hvt-UCB (Ours)** | **Instance-dependent, optimal** | $\mathcal{O}(1)$ | **None** |

### Estimation Error Decomposition (Core Proof Idea)

Through OMD analysis, the estimation error is decomposed into three terms:

1. **Generalization gap term**: $\sum_s \langle \nabla\tilde\ell_s - \nabla\ell_s, \hat\theta_s - \theta_* \rangle$ (deviation caused by noise, controlled by the self-normalized concentration inequality)
2. **Stability term**: $\sum_s \|\nabla\ell_s(\hat\theta_s)\|_{V_s^{-1}}^2$ (error introduced by the one-pass update, controlled through the potential lemma)
3. **Negative term**: $-\frac{1}{2}\sum_s \|\hat\theta_s - \theta_*\|_{X_sX_s^\top/\sigma_s^2}^2$ (originating from the quadratic nature of the loss, used to cancel out the positive parts of the first two terms)

The key difficulty in the heavy-tailed case lies in: the non-quadratic region of the Huber loss makes the analysis of the generalization gap and stability terms more complex, requiring a carefully designed normalization factor to wrap the data into the quadratic region with high probability.

## Highlights & Insights

- **Qualitative Leap in Computational Efficiency**: From $\mathcal{O}(t\log T)$ to $\mathcal{O}(1)$, without needing to store historical data, which is highly practical for large-scale scenarios
- **No Loss in Theory**: The computational speedup does not sacrifice the quality of the regret, maintaining the optimal instance-dependent bound
- **No Extra Assumptions**: Does not require restrictive conditions such as fixed arm sets, repetitive pulling, or bounded absolute moments
- **Inherent Elegance of Recursive Parameter Design**: Embedding the prior round's confidence bound $\beta_{t-1}$ into the setup of the current round's $\sigma_t$, which is cleaner than HEAVY-OFUL
- **New Application of the OMD Framework**: Adapting OMD, which is originally used for regret minimization in adversarial online learning, to parameter estimation in stochastic bandits, building a bridge between parameter gap and loss gap
- **When $\varepsilon=1$ (finite variance)**: Recovers the classic $\widetilde{\mathcal{O}}(d\sqrt{T})$ bound, unifying light-tailed and heavy-tailed scenarios

## Limitations & Future Work

- **Purely Theoretical Work**: Lacks numerical experiments for verification, making it impossible to evaluate the actual difference in performance
- **Requires Known Moment Parameters**: The algorithm needs the moment bound of each round $\nu_t$ (or the global bound $\nu$) and the moment order $\varepsilon$ as inputs, which may be difficult to obtain in practical scenarios
- **Projection Step**: Although per-round complexity is $\mathcal{O}(1)$ w.r.t. $t, T$, the implicit $\mathcal{O}(d^3)$ projection cost can still be significant in high-dimensional scenarios
- **Limited to Linear Bandits**: Has not yet been extended to generalized linear bandits or reinforcement learning and other more complex scenarios (the paper mentions this as future work in the discussion section)
- **Logarithmic Factor Gap**: There is still a gap in logarithmic factors between the regret bound and the lower bound

## Related Work & Insights

- **HEAVY-OFUL** (Huang et al., 2023): The most direct baseline to this work, also based on adaptive Huber regression but solved in a full-batch manner
- **OL2M** (Zhang et al., 2016): The first one-pass algorithm for logistic bandits, using Online Newton Step, but it relies on global strong convexity and is not applicable to Huber loss
- **GLOC** (Jun et al., 2017): A one-pass generalized linear bandit algorithm based on online-to-confidence-set conversion
- **OFUL** (Abbasi-Yadkori et al., 2011): Classic sub-Gaussian linear bandit algorithm; least squares naturally supports one-pass updates
- **Online Learning with Gradient Variation** (Zhao et al., 2020b, 2024): Provides technical inspiration for eliminating the positive terms in the OMD stability term

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of adapting OMD for parameter estimation in heavy-tailed bandits is novel, and the recursive normalization design is elegant.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical with no experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear proof ideas, and the case study is structured from shallow to deep.
- Value: ⭐⭐⭐⭐ — Fills the theoretical gap in the computational efficiency of heavy-tailed bandits.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](../../NeurIPS2025/learning_theory/infrequent_exploration_in_linear_bandits.md)
- [\[NeurIPS 2025\] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression](../../NeurIPS2025/learning_theory/transfer_learning_for_benign_overfitting_in_high-dimensional_linear_regression.md)
- [\[ICML 2026\] A Perturbation Approach to Unconstrained Linear Bandits](../../ICML2026/learning_theory/a_perturbation_approach_to_unconstrained_linear_bandits.md)
- [\[ICLR 2026\] Learning under Quantization for High-Dimensional Linear Regression](../../ICLR2026/learning_theory/learning_under_quantization_for_high-dimensional_linear_regression.md)
- [\[ICML 2025\] Near Optimal Best Arm Identification for Clustered Bandits](near_optimal_best_arm_identification_for_clustered_bandits.md)

</div>

<!-- RELATED:END -->
