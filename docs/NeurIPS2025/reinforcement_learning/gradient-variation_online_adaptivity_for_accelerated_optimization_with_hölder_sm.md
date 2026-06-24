---
title: >-
  [Paper Note] Gradient-Variation Online Adaptivity for Accelerated Optimization with Hölder Smoothness
description: >-
  [NeurIPS 2025][Reinforcement Learning][online learning] This paper develops gradient-variation adaptive online learning algorithms for Hölder smooth function classes, achieving regret that smoothly interpolates between the smooth and non-smooth extremes. Via online-to-batch conversion, it provides the first universal method for strongly convex optimization that attains accelerated convergence in the smooth case and near-optimal convergence in the non-smooth case.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "online learning"
  - "Hölder smoothness"
  - "gradient variation"
  - "acceleration"
  - "universality"
  - "online-to-batch conversion"
date: 2026-05-08
content_hash: de5eeb77dd35d293
---

# Gradient-Variation Online Adaptivity for Accelerated Optimization with Hölder Smoothness

**Conference**: NeurIPS 2025
**arXiv**: [2511.02276](https://arxiv.org/abs/2511.02276)  
**Authors**: Yuheng Zhao, Yu-Hu Yan, Kfir Yehuda Levy, Peng Zhao (Nanjing University / Technion)
**Code**: To be confirmed  
**Area**: Reinforcement Learning
**Keywords**: online learning, Hölder smoothness, gradient variation, acceleration, universality, online-to-batch conversion

## TL;DR

This paper develops gradient-variation adaptive online learning algorithms for Hölder smooth function classes, achieving regret that smoothly interpolates between the smooth and non-smooth extremes. Via online-to-batch conversion, it provides the first universal method for strongly convex optimization that attains accelerated convergence in the smooth case and near-optimal convergence in the non-smooth case.

## Background & Motivation

- Smoothness is central to both accelerated optimization and gradient-variation regret minimization.
- **Hölder smoothness** unifies smooth and non-smooth settings: a function is $(L_\nu, \nu)$-Hölder smooth if $\|\nabla\ell(\mathbf{x}) - \nabla\ell(\mathbf{y})\| \leq L_\nu \|\mathbf{x} - \mathbf{y}\|^\nu$
    - $\nu = 1$: standard smoothness; $\nu = 0$: Lipschitz continuity.
- Universal methods for the convex case have been addressed (Kavis et al., 2019), but **achieving accelerated convergence under smoothness and near-optimality under non-smoothness simultaneously for the strongly convex setting** remains an open problem.
- The key mechanism is to leverage gradient-variation adaptivity in online learning and obtain accelerated offline optimization via online-to-batch conversion.

## Core Problem

1. Can gradient-variation regret be achieved under Hölder smoothness assumptions, interpolating smoothly between the smooth and non-smooth extremes?
2. Can this yield a **universal method** for offline (strongly) convex optimization that adaptively accelerates without knowledge of smoothness parameters?

## Method

### "Approximate Smoothness" Lemma for Hölder Smoothness (Lemma 1)

A $(L_\nu, \nu)$-Hölder smooth function satisfies: for any $\delta > 0$, letting $L = \delta^{(\nu-1)/(1+\nu)} L_\nu^{2/(1+\nu)}$:

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\|^2 \leq L^2 \|\mathbf{x} - \mathbf{y}\|^2 + 4L\delta$$

This reformulates Hölder smoothness as standard smoothness plus a constant perturbation, serving as a key analytical bridge for subsequent analysis.

### Convex Case: Optimistic OGD with AdaGrad Step Size

An optimistic online gradient descent scheme maintains two decision sequences:

$$\mathbf{x}_t = \Pi_\mathcal{X}[\hat{\mathbf{x}}_t - \eta_t M_t], \quad \hat{\mathbf{x}}_{t+1} = \Pi_\mathcal{X}[\hat{\mathbf{x}}_t - \eta_t \nabla f_t(\mathbf{x}_t)]$$

The key innovation is an AdaGrad-style step size that achieves **implicit clipping** without knowledge of smoothness parameters:

$$\eta_{t+1} = \frac{D}{2\sqrt{A_t}}, \quad A_t = \|\nabla f_1(\mathbf{x}_1)\|^2 + \sum_{s=2}^{t} \|\nabla f_s(\mathbf{x}_s) - M_s\|^2$$

**Virtual clipping principle**: $\eta_{t+1}$ decreases monotonically and eventually becomes smaller than $1/L$ automatically. Before automatic clipping takes effect, $\sqrt{A_\tau}$ is itself small, keeping uneliminated gradient-variation terms under control.

### Convex Hölder Regret (Theorem 1)

$$\text{Reg}_T \leq O\left(D\sqrt{V_T} + L_\nu D^{1+\nu} T^{(1-\nu)/2} + D\|\nabla f_1(\mathbf{x}_1)\|\right)$$

where the gradient variation is $V_T = \sum_{t=2}^T \sup_\mathbf{x} \|\nabla f_t(\mathbf{x}) - \nabla f_{t-1}(\mathbf{x})\|^2$.

Verification at the extremes:
- $\nu = 1$ (smooth): recovers the optimal $O(D\sqrt{V_T} + LD^2)$
- $\nu = 0$ (Lipschitz): recovers the optimal $O(GD\sqrt{T})$
- **No knowledge of $L_\nu$ or $\nu$ is required** (strong universality)

### Universal Method for the Strongly Convex Case

**Three-step scheme**:
1. Strongly convex gradient-variation regret: $O\big(\frac{1}{\lambda}\log V_T + \frac{1}{\lambda}L_\nu^2 (\log T)^{(1-\nu)/(1+\nu)}\big)$
2. **Detection mechanism**: a guess-and-check procedure that adaptively determines whether the function is closer to smooth or non-smooth
3. A carefully designed online-to-batch (O2B) conversion

### Stabilized Online-to-Batch Conversion

Translates online regret into offline convergence rates:

$$\mathbb{E}[\ell(\bar{\mathbf{x}}_T)] - \ell(\mathbf{x}_\star) \leq \frac{\mathbb{E}[\text{Reg}_T^\alpha]}{\alpha_{1:T}}$$

By setting $\alpha_t = t$ and exploiting gradient variation to obtain $O(1)$ weighted regret, the method ultimately achieves $O(1/T^2)$ convergence.

## Key Experimental Results

### Convex + Stochastic Optimization Convergence Rate Comparison

| Method | Convergence Rate | Universality |
|--------|-----------------|--------------|
| Kavis et al., 2019 | $O(L/T^2 + \sigma/\sqrt{T})$ smooth; $O(1/\sqrt{T})$ Lipschitz | Weak universal |
| Rodomanov et al., 2024 | $O(L_\nu / T^{(1+3\nu)/2} + \sigma/\sqrt{T})$ | Strong universal |
| **Ours (Theorem 2)** | $O(L_\nu / T^{(1+3\nu)/2} + \sigma/\sqrt{T})$ | **Strong universal** |

### Strongly Convex Optimization Convergence Rate Comparison

| Method | Smooth Rate | Non-smooth Rate | Universality |
|--------|------------|-----------------|--------------|
| Levy, 2017 | $O(\exp(-T/\kappa) \cdot T/\kappa)$ (non-accelerated) | $\tilde{O}(1/\lambda T)$ | Weak (non-accelerated) |
| **Ours (Theorem 4)** | $O(\exp(-T/6\sqrt{\kappa}))$ (**accelerated**) | $\tilde{O}(1/\lambda T)$ | **Weak universal** |

**First** universal method for strongly convex optimization to achieve accelerated convergence in the smooth case.

### Online Learning Regret Comparison

| Setting | Smooth Optimal | Lipschitz Optimal | Ours (Hölder) |
|---------|---------------|-------------------|----------------|
| Convex | $O(\sqrt{V_T} + LD^2)$ | $O(GD\sqrt{T})$ | $O(D\sqrt{V_T} + L_\nu D^{1+\nu} T^{(1-\nu)/2})$ |
| Strongly convex | $O(\frac{1}{\lambda}\log V_T)$ | $O(\frac{1}{\lambda}\log T)$ | $O(\frac{1}{\lambda}\log V_T + \frac{L_\nu^2}{\lambda}(\log T)^{(1-\nu)/(1+\nu)})$ |

### Universality Definitions (Definition 1)

| Type | Description |
|------|-------------|
| Weak universal | Simultaneously adaptive to smooth and non-smooth (Lipschitz) settings |
| Strong universal | Simultaneously adaptive to $(L_\nu, \nu)$-Hölder smoothness for all $\nu \in [0,1]$ |

## Highlights & Insights

- ⭐⭐⭐⭐ First to achieve accelerated rates in universal strongly convex optimization, resolving a long-standing open problem
- ⭐⭐⭐⭐ The Hölder "approximate smoothness" lemma elegantly unifies the analytical framework
- ⭐⭐⭐⭐ Virtual clipping via AdaGrad step sizes — adaptive without any smoothness parameter knowledge
- ⭐⭐⭐ First to achieve optimal gradient-variation regret in the convex case without dependence on $L$
- Establishes a clear bridge between gradient-variation online adaptivity and accelerated offline optimization

## Limitations & Future Work

- Strong convexity results only achieve weak universality (smooth vs. non-smooth); extension to strong universality over the full Hölder parameter range remains open
- The detection mechanism adds complexity, and practical implementation efficiency remains to be verified
- Numerical experiments are absent
- Whether the detection step can be eliminated in favor of direct online adaptive methods for strongly convex strong universality is an open question
- Connections to deep learning optimizers merit further exploration

## ⭐ Recommendation: ⭐⭐⭐⭐⭐

This work represents a significant advance in optimization theory, translating online learning adaptivity into accelerated offline optimization. The unified Hölder smoothness perspective and the virtual clipping technique carry broad technical implications, and the results are expected to have lasting impact on the optimization theory community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[ICML 2026\] LABO: LLM-Accelerated Bayesian Optimization through Broad Exploration and Selective Experimentation](../../ICML2026/reinforcement_learning/labo_llm-accelerated_bayesian_optimization_through_broad_exploration_and_selecti.md)
- [\[NeurIPS 2025\] Robust and Diverse Multi-Agent Learning via Rational Policy Gradient](robust_and_diverse_multi-agent_learning_via_rational_policy_gradient.md)
- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)
- [\[NeurIPS 2025\] Bandit and Delayed Feedback in Online Structured Prediction](bandit_and_delayed_feedback_in_online_structured_prediction.md)

</div>

<!-- RELATED:END -->
