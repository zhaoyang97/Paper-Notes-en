---
title: >-
  [Paper Note] Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization
description: >-
  [NeurIPS 2025][AI Safety][Bilevel Optimization] This paper proposes a novel search direction and proves that first-order and zeroth-order online bilevel optimization algorithms built upon it achieve sublinear stochastic bilevel regret guarantees without requiring window smoothing, while improving efficiency through reduced oracle dependence, parallel updates, and zeroth-order Hessian/Jacobian estimation.
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Bilevel Optimization"
  - "Online Learning"
  - "Zeroth-Order Optimization"
  - "Stochastic Regret"
  - "Hypergradient Estimation"
date: 2026-05-08
content_hash: 391fbdc4be15f451
---

# Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2511.01126](https://arxiv.org/abs/2511.01126)  
**Code**: None  
**Area**: AI Safety
**Keywords**: Bilevel Optimization, Online Learning, Zeroth-Order Optimization, Stochastic Regret, Hypergradient Estimation

## TL;DR

This paper proposes a novel search direction and proves that first-order and zeroth-order online bilevel optimization algorithms built upon it achieve sublinear stochastic bilevel regret guarantees without requiring window smoothing, while improving efficiency through reduced oracle dependence, parallel updates, and zeroth-order Hessian/Jacobian estimation.

## Background & Motivation

Bilevel optimization is an important framework in machine learning, widely applied in hyperparameter optimization, meta-learning, adversarial training, and related settings. Its formulation is:

$$\min_{x} f(x, y^*(x)) \quad \text{s.t.} \quad y^*(x) = \arg\min_{y} g(x, y)$$

Online bilevel optimization (OBO) further requires optimization when the objective functions vary over time, imposing higher demands on algorithmic adaptivity.

Existing OBO methods rely on **window-smoothed regret** to measure performance. Window-smoothed regret averages function values over a time window to "smooth out" rapid changes in the objective, but this metric may fail to accurately reflect true system performance when functions change rapidly. Moreover, existing methods typically require a large number of gradient oracle calls and are inefficient in the zeroth-order (gradient-free) setting.

## Method

### Overall Architecture

This paper proposes a unified online bilevel optimization framework comprising the following key components:

1. **Novel Search Direction**: A more efficient hypergradient approximation direction.
2. **First-Order Variant (SOBO-FO)**: An online bilevel optimization algorithm utilizing first-order gradient information.
3. **Zeroth-Order Variant (SOBO-ZO)**: An online bilevel optimization algorithm using only function value queries.
4. **Stochastic Regret Analysis**: Proof of sublinear regret without reliance on window smoothing.

### Key Designs

**Novel Search Direction**: In bilevel optimization, the implicit gradient (hypergradient) of the outer objective requires computing the Hessian and Jacobian of the inner problem. The authors design an approximate search direction $d_t$ satisfying:

$$d_t \approx \nabla_x f_t - \nabla_{xy}^2 g_t \cdot [\nabla_{yy}^2 g_t]^{-1} \cdot \nabla_y f_t$$

The key improvement lies in **parallelizing** the solution of the linear system $[\nabla_{yy}^2 g_t] v = \nabla_y f_t$ with the updates of the inner and outer variables, rather than requiring the inner problem to be fully solved before updating the outer variable as in prior methods.

**Reduced Oracle Complexity**: Oracle calls are reduced through the following strategies:
- Simultaneously updating the inner variable $y_t$, outer variable $x_t$, and auxiliary variable $v_t$ for the linear system at each iteration.
- Avoiding full inner optimization at every step.

**Zeroth-Order (ZO) Method**: When gradient information is unavailable (e.g., black-box attack scenarios), stochastic finite differences are used to estimate:
- Gradient: $\hat{\nabla} f(x) \approx \frac{d}{\delta} [f(x + \delta u) - f(x)] u$, where $u$ is a random direction.
- Hessian-vector products and Jacobian-vector products are similarly estimated via finite differences.

### Loss & Training

This paper is a theoretical optimization work and does not involve specific loss function design. The regret metric is the stochastic bilevel regret:

$$\text{Regret}_T = \sum_{t=1}^{T} \mathbb{E}[f_t(x_t, y^*(x_t))] - \min_{x \in \mathcal{X}} \sum_{t=1}^{T} f_t(x, y^*(x))$$

**Theoretical Results**:
- First-order algorithm: $\text{Regret}_T = O(\sqrt{T})$
- Zeroth-order algorithm: $\text{Regret}_T = O(d^{3/2}\sqrt{T})$, where $d$ is the problem dimension.

## Key Experimental Results

### Main Results

**Experiment 1: Online Parametric Loss Tuning**

Online hyperparameter tuning is conducted on MNIST and CIFAR-10, with the outer loop optimizing validation loss and the inner loop optimizing training loss.

| Method | MNIST Regret (T=500) | MNIST Regret (T=1000) | CIFAR-10 Regret (T=500) | CIFAR-10 Regret (T=1000) |
|------|-------------------|--------------------|-----------------------|----------------------|
| SOBO-FO (Ours) | **12.3** | **18.7** | **25.6** | **38.2** |
| OBO-FO (Window Smoothing) | 15.8 | 25.4 | 31.2 | 49.5 |
| Online SGD | 22.1 | 38.6 | 42.8 | 68.3 |
| SOBO-ZO (Ours) | 16.5 | 24.1 | 30.8 | 45.7 |

**Experiment 2: Black-box Adversarial Attacks**

Adversarial examples are generated on a CIFAR-10 classifier using the zeroth-order method.

| Method | Attack Success Rate ↑ | Query Count ↓ | $L_\infty$ Perturbation |
|------|------------|-----------|----------------|
| SOBO-ZO (Ours) | **92.3%** | **850** | 8/255 |
| ZO-BiAdam | 89.7% | 1200 | 8/255 |
| SimBA | 85.4% | 1500 | 8/255 |
| Sign-OPT | 87.1% | 1100 | 8/255 |

### Ablation Study

**Oracle Call Comparison (per iteration)**:

| Method | Gradient Oracle | Hessian Oracle | Jacobian Oracle | Total |
|------|------------|---------------|----------------|------|
| SOBO-FO (Ours) | 3 | 1 | 1 | **5** |
| Prior FO Methods | 3 | K (inner iterations) | K | **2K+3** |
| SOBO-ZO (Ours) | O(d) | O(d) | O(d) | **O(d)** |
| Prior ZO Methods | O(d) | O(Kd) | O(Kd) | **O(Kd)** |

By parallelizing inner-loop solving with outer-loop updates, the proposed methods significantly reduce oracle dependence.

### Key Findings

1. **Window Smoothing Is Unnecessary**: This work is the first to prove that OBO can achieve sublinear stochastic regret without window smoothing.
2. **Substantial Efficiency Gains**: Oracle calls are reduced from $O(K)$ to $O(1)$ (first-order) and from $O(Kd)$ to $O(d)$ (zeroth-order).
3. **Practical Zeroth-Order Method**: SOBO-ZO performs comparably to first-order methods in black-box settings.
4. **Theory-Experiment Consistency**: The convergence trend of regret aligns with the theoretically predicted $O(\sqrt{T})$ rate.

## Highlights & Insights

- **Solid Theoretical Contributions**: The 88-page paper provides rigorous and complete theoretical analysis, from the novel search direction design to full regret bound proofs.
- **Practical Zeroth-Order Variant**: The zeroth-order method extends the framework to black-box optimization scenarios (e.g., adversarial attacks).
- **Parallelization Insight**: Parallelizing linear system solving with variable updates is the key to efficiency improvement.
- **Unified Framework**: Both first-order and zeroth-order algorithms are analyzed within the same theoretical framework.

## Limitations & Future Work

1. **Strong Convexity Assumption**: The inner problem is required to be strongly convex, limiting applicability (e.g., neural network training is generally non-convex).
2. **Small-Scale Experiments**: Validation is limited to MNIST and CIFAR-10; large-scale experimental evaluation is absent.
3. **Constant Factors**: The constant in $O(\sqrt{T})$ may be large; practical convergence speed depends on the problem's condition number.
4. **Comparison with Practical HPO Tools**: No comparison against practical tools such as Optuna or Ray Tune is provided.
5. **Exploitation of Structured Dynamics**: When the pattern of function variation is known (e.g., periodic), it remains open whether such structure can be leveraged.

## Related Work & Insights

- **Online Bilevel Optimization**: The direct predecessor works, particularly window-smoothed regret methods.
- **Bilevel Optimization**: Classical methods including Franceschi et al. (2018) and Ghadimi & Wang (2018).
- **Zeroth-Order Optimization**: Nesterov & Spokoiny (2017) and stochastic finite-difference methods.
- **Online Learning**: Hazan (2016) and online convex optimization theory.
- **Hyperparameter Optimization**: Bilevel optimization applications in meta-learning.

## Rating

- **Novelty**: 4/5 — Novel search direction and regret guarantees without window smoothing.
- **Technical Quality**: 5/5 — Rigorous and complete theoretical analysis.
- **Writing Quality**: 3/5 — The 88-page paper is information-dense but excessively long.
- **Value**: 3/5 — Theoretical significance outweighs practical applicability.
- **Overall**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](../../ICML2026/ai_safety/privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[NeurIPS 2025\] Fairness-Regularized Online Optimization with Switching Costs](fairness-regularized_online_optimization_with_switching_costs.md)
- [\[NeurIPS 2025\] Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization](dual-flow_transferable_multi-target_instance-agnostic_attacks_via_in-the-wild_ca.md)
- [\[ICML 2025\] Connecting Thompson Sampling and UCB: Towards More Efficient Trade-offs Between Privacy and Regret](../../ICML2025/ai_safety/connecting_thompson_sampling_and_ucb_towards_more_efficient_trade-offs_between_p.md)

</div>

<!-- RELATED:END -->
