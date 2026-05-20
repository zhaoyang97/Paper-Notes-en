---
title: >-
  [Paper Note] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update
description: >-
  [NeurIPS 2025][Reinforcement Learning][Generalized linear bandits] This paper proposes the GLB-OMD algorithm, which, for the first time in the generalized linear bandit (GLB) setting…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Generalized linear bandits"
  - "online learning"
  - "confidence sets"
  - "online mirror descent"
  - "mix loss"
date: 2026-05-08
content_hash: d4446f5f8fd90b30
---

# Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update

**Conference**: NeurIPS 2025
**arXiv**: [2507.11847](https://arxiv.org/abs/2507.11847)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Generalized linear bandits, online learning, confidence sets, online mirror descent, mix loss

## TL;DR

This paper proposes the GLB-OMD algorithm, which, for the first time in the generalized linear bandit (GLB) setting, simultaneously achieves a near-optimal regret bound of $\mathcal{O}(\log T\sqrt{T/\kappa_*})$ and $\mathcal{O}(1)$ per-round time and space complexity. The key technical contribution is constructing tight confidence sets for an online mirror descent (OMD) estimator via mix loss.

## Background & Motivation

Generalized linear bandits (GLB) are an important extension of the contextual multi-armed bandit framework, modeling richer reward distributions (e.g., Bernoulli, Poisson) through a nonlinear link function $\mu$. In practice, GLB is widely applicable to recommendation systems, personalized medicine, dynamic pricing, and related domains.

The central challenge in GLB is the **trade-off between computational efficiency and statistical efficiency**:
- **Statistical efficiency**: The regret bound of the classical GLM-UCB algorithm involves a potentially exponentially large constant $\kappa = 1/\inf \mu'$; for instance, in logistic bandits $\kappa = \mathcal{O}(e^S)$.
- **Computational efficiency**: Methods based on maximum likelihood estimation (MLE) require $\mathcal{O}(t)$ per-round time and space complexity.

Existing methods are either computationally efficient but statistically suboptimal (GLOC), or statistically optimal but computationally expensive (OFUGLB, RS-GLinCB). The goal of this paper is to design a **jointly efficient** algorithm that achieves optimal regret with constant per-round computational cost.

## Method

### Overall Architecture

The GLB-OMD algorithm follows the optimism in the face of uncertainty (OFU) principle. At each round $t$:
1. Construct an ellipsoidal confidence set $\mathcal{C}_t(\delta)$ centered at the OMD estimator $\theta_t$.
2. Select the arm $X_t = \arg\max_{\mathbf{x}} \{\mathbf{x}^\top \theta_t + \beta_t(\delta) \|\mathbf{x}\|_{H_t^{-1}}\}$ that maximizes the UCB.
3. Upon observing the reward, update the estimator via OMD.

The key innovation lies in replacing MLE with OMD as the parameter estimator and constructing confidence sets for OMD that are as tight as those available for MLE.

### Key Designs

#### Online Mirror Descent (OMD) Estimator

The per-round update rule is:
$$\theta_{t+1} = \arg\min_{\theta \in \Theta} \tilde{\ell}_t(\theta) + \frac{1}{2\eta}\|\theta - \theta_t\|^2_{H_t}$$

where $\tilde{\ell}_t(\theta)$ is a second-order approximation of the original loss (retaining curvature information while admitting efficient optimization), and $H_t = \lambda I_d + \sum_{s=1}^{t-1} \nabla^2 \ell_s(\theta_{s+1})$ is the accumulated curvature matrix.

**Computational advantage**: The optimization problem reduces to quadratic minimization over a Euclidean ball with complexity $\mathcal{O}(d^3)$, independent of $t$. $H_t$ can be updated incrementally as $H_{t+1} = H_t + \nabla^2 \ell_t(\theta_{t+1})$, requiring no storage of historical data.

#### Confidence Sets via Mix Loss

The key technical contribution is the analysis of the inverse regret of the OMD estimator. The inverse regret, defined as $\sum \ell_s(\theta_*) - \sum \ell_s(\theta_{s+1})$, directly determines the radius of the confidence set.

Classical approaches introduce an intermediate online learning iterate $\tilde{\theta}_s$ for decomposition, which is difficult to unify in the GLB setting. This paper instead uses the **mix loss** as an intermediate quantity:
$$m_s(P_s) = -\ln\left(\mathbb{E}_{\theta \sim P_s}[e^{-\ell_s(\theta)}]\right)$$

The analysis proceeds in two steps:
1. **Lemma 2** (Ville's inequality): $\sum \ell_s(\theta_*) - \sum m_s(P_s) \leq \log(1/\delta)$ holds with high probability.
2. **Lemma 3**: Choosing $P_s = \mathcal{N}(\theta_s, \frac{3}{2}\eta H_s^{-1})$ as a Gaussian distribution centered at the OMD estimator enables control of $\sum m_s(P_s) - \sum \ell_s(\theta_{s+1})$.

The resulting confidence set (Theorem 1) is $\mathcal{C}_t^{\text{OL}}(\delta) = \{\theta \mid \|\theta - \theta_t\|_{H_t} \leq \beta_t(\delta)\}$, with radius $\beta_t(\delta) = \mathcal{O}(SR\sqrt{d(S^2R + \ln(t/\delta))})$, which is **independent of $\kappa$**.

### Loss & Training

**Loss function**: The negative log-likelihood of the GLM, $\ell_t(\theta) = (m(X_t^\top \theta) - r_t \cdot X_t^\top \theta)/g(\tau)$, where $m$ is the log-partition function.

**Self-concordance assumption** (Assumption 3): $|\mu''(z)| \leq R \cdot \mu'(z)$, satisfied by common distributions including Bernoulli ($R=1$), Poisson ($R=1$), and Gaussian ($R=0$). This condition is essential for exploiting the local curvature of the link function to improve the dependence on $\kappa$.

**Regret bound** (Theorem 2):
$$\text{Reg}_T \lesssim dSR\sqrt{S^2R + \log T} \cdot \sqrt{\frac{T\log T}{\kappa_*}} + \kappa d^2 S^2 R^3 \log T (S^2R + \log T)$$

The leading term is $\tilde{\mathcal{O}}(d\sqrt{T/\kappa_*})$; the lower-order term retains dependence on $\kappa$ but is dominated at large $T$.

## Key Experimental Results

### Main Results (Logistic Bandit)

| Method | Regret ($S=3$, $\kappa=21$) | Regret ($S=5$, $\kappa=137$) | Per-round Time |
|--------|----------------------------|------------------------------|----------------|
| GLOC | High | Very high | **Fastest** |
| GLM-UCB | Medium-high | High | $\mathcal{O}(t)$ |
| OFUGLB | **Lowest** | **Lowest** | Slowest ($\mathcal{O}(t)$) |
| RS-GLinCB | Low | Low | Grows with $\kappa$ |
| ECOLog | Low | Low | $\mathcal{O}(\log t)$ |
| **GLB-OMD** | Low (close to OFUGLB) | Low | **$\mathcal{O}(1)$** |

**Poisson Bandit** (unbounded rewards):

| Method | Runtime relative to OFUGLB | Regret increase |
|--------|---------------------------|-----------------|
| **GLB-OMD** | **~1/1000** | Modest |
| OFUGLB | 1× (baseline) | Baseline |

### Ablation Study

- **Dependence on $\kappa$**: As $S$ increases from 3 to 7 (with $\kappa$ growing from 21 to 963), GLB-OMD maintains constant per-round cost, whereas RS-GLinCB's update frequency grows with $\kappa$.
- **Comparison with logistic bandit-specific methods** (Table 2): GLB-OMD simultaneously improves upon ECOLog's $\mathcal{O}(\log t)$ time complexity and the additional $\mathcal{O}(\sqrt{\log T})$ regret factor of OFUL-MLogB.

### Key Findings

1. GLB-OMD achieves the optimal balance between computational and statistical efficiency.
2. The OMD estimator, even with one-pass updates, matches the statistical efficiency of MLE.
3. Nonlinear link functions do not necessarily entail higher computational cost — GLB-OMD has the same per-round cost as LinUCB.
4. Results on the Covertype real-world dataset are consistent with synthetic experiments.

## Highlights & Insights

- **Elegant use of mix loss**: Importing the mix loss concept from online prediction into bandit confidence set construction yields a more flexible analytical framework than direct decomposition.
- **Time-varying Gaussian bridge**: Selecting a time-varying Gaussian distribution centered at the OMD estimator as the reference distribution for the mix loss cleverly accommodates the online nature of OMD.
- **Unified framework**: Prior methods for logistic bandits exploit the specific structure of the logistic function, whereas this paper applies generally to all GLMs satisfying self-concordance.
- **Practical value**: Constant space complexity enables direct deployment in resource-constrained online systems.

## Limitations & Future Work

1. The leading term retains a dependence on $S^2$, which MLE-based methods have already fully eliminated (Lee et al., 2024).
2. The lower-order term still has linear dependence on $\kappa$; this may be improvable to $\kappa_\mathcal{X}$ via warm-up strategies.
3. Whether geometry-aware bounds analogous to those for logistic bandits can be obtained remains an open problem.
4. Relaxing the self-concordance assumption or improving the dependence on $d$ in the finite-arm setting warrants further investigation.
5. Combining OMD with rare-update techniques may enable further acceleration.

## Related Work & Insights

- **ECOLog** (Faury et al., 2022): A jointly efficient method for logistic bandits, but requires a warm-up phase and has $\mathcal{O}(\log t)$ per-round time.
- **OFUL-MLogB** (Zhang & Sugiyama, 2023): Achieves $\mathcal{O}(1)$ time but incurs an extra $\mathcal{O}(\sqrt{\log T})$ regret factor.
- **RS-GLinCB** (Sawarni et al., 2024): A rare-update strategy reduces amortized cost but still requires $\mathcal{O}(t)$ space.
- **Kirschner et al., 2025 & Clerico et al., 2025**: Concurrent works also employ mix loss but target the batch MLE setting.
- Insight: Mix loss techniques may prove useful in other online decision-making problems, such as RL with function approximation.

## Rating

- **Novelty**: ★★★★★ — First to achieve joint computational and statistical optimality in GLB.
- **Experimental Thoroughness**: ★★★★☆ — Covers logistic and Poisson bandits as well as real-world data.
- **Value**: ★★★★★ — $\mathcal{O}(1)$ per-round cost is highly valuable for online systems.
- **Writing Quality**: ★★★★★ — Theoretically rigorous with a clear and well-organized presentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality](improved_regret_and_contextual_linear_extension_for_pandoras_box_and_prophet_ine.md)
- [\[NeurIPS 2025\] Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits](gaussian_process_upper_confidence_bound_achieves_nearly-optimal_regret_in_noise-.md)
- [\[NeurIPS 2025\] Tractable Multinomial Logit Contextual Bandits with Non-Linear Utilities](tractable_multinomial_logit_contextual_bandits_with_non-linear_utilities.md)
- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](../../ICLR2026/reinforcement_learning/single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [\[NeurIPS 2025\] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)

</div>

<!-- RELATED:END -->
