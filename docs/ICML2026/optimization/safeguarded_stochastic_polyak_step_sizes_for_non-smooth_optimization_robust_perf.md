---
title: >-
  [Paper Note] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization
description: >-
  [ICML 2026][Optimization & Theory][Stochastic Polyak step size] SPSsafe extends the Stochastic Polyak Step Size (SPS) to non-smooth stochastic optimization without requiring interpolation assumptions or prior knowledge of optimal values. Combined with momentum (IMA, an equivalent form of SHB), it maintains rigorous convergence guarantees. It demonstrates superior robustness over ex
tags:
  - ICML 2026
  - Optimization & Theory
  - Stochastic Polyak step size
date: 2026-05-08
content_hash: 572fee5eae988965
---
# SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization

**Conference**: ICML 2026  
**arXiv**: [2512.02342](https://arxiv.org/abs/2512.02342)  
**Code**: To be confirmed  
**Area**: Optimization / Adaptive Step Sizes / Non-smooth Optimization  
**Keywords**: Stochastic Polyak step size, non-smooth optimization, stochastic subgradient, momentum, robust training

## TL;DR
SPSsafe extends the Stochastic Polyak Step Size (SPS) to non-smooth stochastic optimization without requiring interpolation assumptions or prior knowledge of optimal values. Combined with momentum (IMA, an equivalent form of SHB), it maintains rigorous convergence guarantees. It demonstrates superior robustness over existing adaptive methods (AdaGrad, Adam, DecSPS) in DNN training and prevents gradient norm collapse (resisting gradient vanishing).

## Background & Motivation

**Background**: Adaptive optimization is a staple in modern ML, with AdaGrad and Adam being the defaults. The Polyak step size approach has seen a recent resurgence (e.g., Loizou 2021), using function values rather than just gradients to determine step sizes, showing strong performance on overparameterized models. It has been extended to SGD, Mirror Descent, Local SGD, and SHB.

**Limitations of Prior Work**: (1) Existing SPS analyses almost exclusively assume convexity and smoothness; guarantees for non-smooth regimes (common with ReLU in DNNs, L1 regularization, or ranking losses) are scarce. (2) Existing non-smooth SPS variants either require an interpolation assumption (requiring $x^* \in X^*$ such that $f_i(x^*) = f_i^*$ for all $i$, which is overly strong) or require oracle knowledge of $f_i(x^*)$ (unavailable in practice). (3) The combination of momentum and SPS remains largely underexplored in non-smooth settings.

**Key Challenge**: To satisfy three conditions simultaneously—the adaptive advantages of Polyak step sizes, convergence guarantees for non-smooth functions, and the absence of interpolation or oracle optimal value assumptions—no existing SPS variant succeeds.

**Goal**: (1) Propose SPSsafe for the stochastic subgradient method (SSM), providing rigorous convergence in non-smooth convex optimization without interpolation or oracle values; (2) Incorporate momentum (IMA, equivalent to SHB) while maintaining theoretical guarantees; (3) Empirically demonstrate robustness in DNN training and the prevention of gradient vanishing.

**Key Insight**: The root cause of classical SPS failure in non-smooth regimes is that the denominator of the naive Polyak step size $(f_i(x_t)-f_i^*)/\|g_t^i\|^2$ is the squared subgradient norm. When subgradients become small (common in non-smooth regimes and late-stage DNN training), $\|g_t^i\|^2 \to 0$ causes $1/\|g_t^i\|^2$ to pull the step size toward infinity, leading to divergence. However, the numerator $f_i(x_t)-f_i^*$ remains a good proxy for "distance to optimality." This paper’s safeguard thus only modifies the denominator—setting a lower bound $M$ for $\|g_t^i\|^2$ to block explosions caused by small subgradients while keeping the numerator intact to preserve Polyak adaptivity.

**Core Idea**: The SPSsafe step size is $\gamma_t = \dfrac{f_i(x_t) - \ell_i^*}{\max\{\|g_t^i\|^2,\,M\}}$, where $\ell_i^* \leq \inf f_i$ is a known lower bound (e.g., 0 for non-negative losses like cross-entropy) and $M>0$ is the unique safeguard constant. Notably, this is **not** like the older SPSmax that caps the overall step size $\min\{\text{Polyak},\gamma_b\}$ (which can degrade the step size to a constant); instead, it floors the denominator. A single hyperparameter $M$ replaces both the capping constant $\gamma_b$ in SPSmax and the oracle value $f_i(x^*)$ required by SPS\*. This is then integrated into the IMA momentum framework to maintain theoretical properties.

## Method

### Overall Architecture

Consider $\min_x f(x) = \tfrac{1}{n}\sum_i f_i(x)$, where $f_i$ is convex, Lipschitz, non-smooth, and has a lower bound $\ell_i^*$ (e.g., cross-entropy = 0).

Both algorithms use the same family of SPSsafe step sizes (flooring the denominator at $\max\{\|g_t^i\|^2,\,M\}$):
- **SSM with SPSsafe**: $x_{t+1} = x_t - \gamma_t g_t^i$, $g_t^i \in \partial f_i(x_t)$, with step size $\gamma_t = \dfrac{f_i(x_t) - \ell_i^*}{\max\{\|g_t^i\|^2,\,M\}}$.
- **IMA with SPSsafe** (Momentum, equivalent to SHB): $z_{t+1} = z_t - \eta_t g_t^i$, $x_{t+1} = \tfrac{\lambda_{t+1}}{\lambda_{t+1}+1} x_t + \tfrac{1}{\lambda_{t+1}+1} z_{t+1}$, with step size $\eta_t = \dfrac{[f_i(x_t) - \ell_i^* + \lambda_t\langle g_t^i,\, x_t - x_{t-1}\rangle]_+}{\max\{\|g_t^i\|^2,\,M\}}$.

The safeguard constant $M>0$ floors the denominator to prevent step size explosion when subgradients are small. Both variants achieve $O(1/\sqrt{T}+\sigma^2)$ convergence to an optimal neighborhood under convex non-smooth (Lipschitz) conditions. The IMA version additionally provides a last-iterate guarantee, all without requiring interpolation or knowledge of $f_i(x^*)$.

### Key Designs

**1. Flooring the subgradient norm $\max\{\|g_t^i\|^2, M\}$ (Core Innovation): Preventing step size explosion from small subgradients**

In smooth settings, classical SPS is naturally bounded—Lipschitz gradients ensure $\|g_t^i\|^2$ does not become arbitrarily small relative to the numerator, keeping step sizes controlled. In non-smooth regimes or late-stage DNN training, subgradients can become extremely small. The denominator in naive Polyak step sizes $(f_i(x_t)-\ell_i^*)/\|g_t^i\|^2$ approaches 0, causing the step size to diverge. SPSsafe fixes this by **flooring the denominator without touching the numerator**:

$$\gamma_t=\frac{f_i(x_t)-\ell_i^*}{\max\{\|g_t^i\|^2,\,M\}}.$$

This is the opposite of SPSmax, which caps the entire step size $\min\{\text{Polyak},\gamma_b\}$. If $\gamma_b$ is too small, SPSmax degrades to a constant step size, losing Polyak adaptivity. By flooring the denominator, SPSsafe **never degrades to a constant**, preserving adaptivity. This floor is also key to the convergence proof, as $M$ allows the step size to be controlled, enabling the $O(1/\sqrt{T}+\sigma^2)$ bound. The paper notes this can be interpreted as adaptive gradient clipping, providing the first theoretical guarantee for "Polyak-style clipped SSM."

**2. Using $\ell_i^*$ as a lower bound instead of $f_i^*$: Bypassing oracle assumptions**

Previous non-smooth SPS variants (like SPS\*) required either interpolation or the exact value of each $f_i(x^*)$, which is unattainable in practice. SPSsafe utilizes the fact that lower bounds $\ell_i^*\le\inf f_i$ are often known (e.g., 0 for most loss functions). For SSM, since $\ell_i^*\le\inf f_i\le f_i(x_t)$, the numerator is naturally non-negative. For the momentum version (IMA), the term $\lambda_t\langle g_t^i,x_t-x_{t-1}\rangle$ can be negative, so the numerator is truncated with $[\cdot]_+$. This converts an "oracle-dependent SPS" into a "lower-bound-dependent SPS," leaving only $\ell_i^*$ and $M$ as configurable parameters.

**3. IMA Momentum Framework equivalent to SHB: Integrating momentum without compromising theory**

Momentum is crucial in practice, but its non-smooth analysis is often difficult. SPSsafe uses the dual-sequence form of IMA (a $z$ sequence for subgradient updates and an $x$ sequence for averaging), which is equivalent to SHB with momentum: $x_{t+1}=x_t-\hat\gamma_t g_t^i+\beta(x_t-x_{t-1})$. Substituting the SPSsafe step size into this framework makes the Lyapunov analysis cleaner. The momentum version achieves both Cesàro mean convergence and last-iterate guarantees without ever needing $f_i(x^*)$, a feat previously impossible for adaptive momentum methods.

## Key Experimental Results

### Convex Non-smooth Benchmark

| Method | Logistic + L1 | SVM hinge loss | Ranking loss |
|------|----------|----------|----------|
| AdaGrad | 0.083 | 0.142 | 0.295 |
| Adam | 0.076 | 0.135 | 0.281 |
| DecSPS | 0.082 | 0.148 | 0.302 |
| **SPSsafe (SSM)** | **0.069** | **0.121** | **0.258** |
| **SPSsafe (IMA)** | **0.063** | **0.114** | **0.247** |

SPSsafe consistently leads across three non-smooth convex benchmarks, with the IMA version performing best.

### DNN Training (CIFAR-10 + ResNet-18)

| Method | Test Accuracy | Training Stability |
|------|--------|--------|
| SGD + Tuned LR | 94.7 | Medium (requires tuning) |
| Adam | 93.8 | High |
| AdaGrad | 92.5 | Medium |
| **SPSsafe** | **94.5** | **High (no tuning)** |
| **SPSsafe + IMA** | **95.1** | **High** |

SPSsafe + IMA reaches 95.1% on ResNet-18 without manual tuning of the learning rate, matching or slightly exceeding fine-tuned SGD.

### Gradient Norm Tracking (Anti-Gradient Vanishing)

| Method | Late-stage Grad Norm | Gradient Vanishing? |
|------|-------------|----------|
| Adam | Near 0 | ✓ Severe |
| AdaGrad | Near 0 | ✓ Severe |
| **SPSsafe** | Maintains $\sim 10^{-2}$ | ✗ No |

The gradient norm in SPSsafe does not collapse in late training stages, meaning the model remains in "active learning," which is beneficial for fine-tuning or adversarial robustness.

### Key Findings
- **Safeguarding is essential for theory**: Removing the denominator floor ($M\to0$) causes Polyak step sizes to explode or diverge as subgradients vanish.
- **Consistent across Convex and DNN settings**: SOTA performance on three convex benchmarks and ResNet.
- **Anti-gradient vanishing is a byproduct**: Stable late-stage gradients are favorable for fine-tuning and adversarial training.
- **No LR tuning required**: Polyak-style step sizes are naturally adaptive, eliminating the need for LR sweeps.

## Highlights & Insights
- **Flooring the denominator vs. capping the step size**: Unlike SPSmax, which caps the step size and risks degrading to a constant, SPSsafe floors the denominator, ensuring adaptivity is preserved. This can be interpreted as adaptive gradient clipping.
- **Removal of interpolation/oracle assumptions**: Eliminating these strong assumptions makes the method truly practical for non-smooth optimization.
- **Gradient vanishing mitigation**: Traditional adaptive optimizers tend to reduce late-stage gradients (a noise floor effect); SPSsafe avoids this collapse, potentially making deeper networks more trainable.
- **Analytical convenience of IMA-SHB equivalence**: Using dual-sequence equivalence makes momentum analysis cleaner and provides tighter guarantees.

## Limitations & Future Work
- The safeguard constant $M$ and lower bound $\ell_i^*$ still require manual setting; an adaptive $M$ might be more robust.
- Convergence is only proved for convex non-smooth cases; non-convex guarantees remain open theoretically despite empirical success.
- Lack of extensive verification on large-scale LLMs/Transformers; may eventually require combination with Adam-style second-moment estimates.
- Computing $f_i(x_t)$ requires a forward pass at each step, making it slightly more expensive than pure gradient methods (though same complexity as SGD).
- Distributed or communication-compressed scenarios were not explored.

## Related Work & Insights
- **vs. Loizou 2021 (Classical SPS)**: Classical SPS needs interpolation; this does not.
- **vs. Garrigos 2023 (SPS*)**: SPS\* needs an $f_i(x^*)$ oracle; this uses a lower bound.
- **vs. AdaGrad / Adam**: These follow the adaptive second-moment route; SPS uses function values, offering a complementary approach.
- **vs. DecSPS, SPSL, Oikonomou-Loizou 2025**: Previous momentum versions still rely on partial assumptions; this provides the first complete theory for non-smooth momentum.
- **Insight**: The safeguard idea (capping the adaptive denominator) can be extended to other adaptive algorithms for non-smooth extensions; using $\ell_i^*$ instead of oracle values is applicable to all Polyak-style methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple safeguard but provides the first rigorous non-smooth guarantee for SPS; momentum + non-smooth combination is also new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers convex benchmarks, DNN training, and gradient norm analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear algorithms, solid theoretical grounding, and alignment with experiments.
- Value: ⭐⭐⭐⭐ Non-smooth optimization is ubiquitous in ML; SPSsafe offers a plug-in adaptive solution without LR tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICLR 2026\] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization](../../ICLR2026/optimization/faster_gradient_methods_for_highly-smooth_stochastic_bilevel_optimization.md)
- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)

</div>

<!-- RELATED:END -->
