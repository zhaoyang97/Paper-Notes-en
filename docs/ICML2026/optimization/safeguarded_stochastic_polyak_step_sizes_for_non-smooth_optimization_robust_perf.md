---
title: >-
  [Paper Note] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization
description: >-
  [ICML 2026][Optimization][Stochastic Polyak step size] SPSsafe extends Stochastic Polyak Step Size (SPS) to non-smooth stochastic optimization without requiring interpolation assumptions or knowledge of optimal values. Combined with momentum (IMA = SHB equivalent form), it maintains rigorous convergence guarantees. It is more robust than existing adaptive methods (AdaGrad, Adam, DecSPS, etc.) for DNN training and avoids gradient norm collapse (anti-gradient vanishing).
tags:
  - "ICML 2026"
  - "Optimization"
  - "Stochastic Polyak step size"
  - "Non-smooth optimization"
  - "Stochastic subgradient"
  - "Momentum"
  - "Robust training"
date: 2026-05-08
content_hash: 98c8ca6df48670d7
---

# SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization

**Conference**: ICML 2026  
**arXiv**: [2512.02342](https://arxiv.org/abs/2512.02342)  
**Code**: TBD  
**Area**: Optimization / Adaptive Step Sizes / Non-smooth Optimization  
**Keywords**: Stochastic Polyak step size, Non-smooth optimization, Stochastic subgradient, Momentum, Robust training

## TL;DR
SPSsafe extends Stochastic Polyak Step Size (SPS) to non-smooth stochastic optimization without requiring interpolation assumptions or knowledge of optimal values. Combined with momentum (IMA = SHB equivalent form), it maintains rigorous convergence guarantees. It is more robust than existing adaptive methods (AdaGrad, Adam, DecSPS, etc.) for DNN training and avoids gradient norm collapse (anti-gradient vanishing).

## Background & Motivation

**Background**: Adaptive optimization is standard in modern ML (AdaGrad, Adam are defaults). Polyak step size methods have seen a revival (Loizou 2021, etc.), using function values instead of pure gradients to determine step sizes, showing strong performance on overparameterized models. Extensions exist for SGD, Mirror Descent, Local SGD, and SHB.

**Limitations of Prior Work**: (1) Existing SPS analyses mostly assume convexity + smoothness; guarantees for non-smooth regimes (common in DNNs with ReLU, L1 regularization, ranking loss) are scarce. (2) Existing non-smooth SPS variants either require an interpolation assumption (requires $x^* \in X^*$ such that $f_i(x^*) = f_i^*$ for all $i$, which is too strong) or knowledge of $f_i(x^*)$ (not available in practice). (3) The combination of momentum and SPS is even more underexplored in non-smooth settings.

**Key Challenge**: Achieving the adaptive advantage of Polyak step sizes + non-smooth convergence guarantees + avoiding interpolation assumptions or oracle knowledge of optimal values simultaneously. Such an SPS variant does not currently exist.

**Goal**: (1) Propose SPSsafe for the stochastic subgradient method (SSM) with rigorous convergence in non-smooth convex optimization without interpolation or optimal values; (2) Incorporate momentum (IMA equivalent to SHB) while maintaining theory; (3) Empirically demonstrate robustness in DNN training and resistance to gradient vanishing.

**Key Insight**: The root cause of classical SPS failure in non-smooth settings is the denominator of the naive Polyak step $(f_i(x_t)-f_i^*)/\|g_t^i\|^2$, which is the squared subgradient norm. When subgradients become small (common in non-smooth regimes or late DNN training), $\|g_t^i\|^2 \to 0$ causes $1/\|g_t^i\|^2$ to pull the step size to infinity, causing divergence. The numerator $f_i(x_t)-f_i^*$ is a good proxy for "distance to opt." This paper’s safeguard thus only modifies the denominator—setting a lower bound $M$ for $\|g_t^i\|^2$ to block explosions caused by small subgradients while keeping the numerator intact to preserve Polyak adaptability.

**Core Idea**: The SPSsafe step size is $\gamma_t = \dfrac{f_i(x_t) - \ell_i^*}{\max\{\|g_t^i\|^2,\,M\}}$, where $\ell_i^* \leq \inf f_i$ is a known lower bound (e.g., 0 for cross-entropy) and $M>0$ is a single safeguard constant. Note it is **not** like the old SPSmax which caps the step size $\min\{\text{Polyak},\gamma_b\}$ (which can degrade to a constant); instead, it floors the denominator. The single hyperparameter $M$ replaces both the capping constant $\gamma_b$ of SPSmax and the oracle value $f_i(x^*)$ required by SPS\*. This is then integrated into the IMA momentum framework.

## Method

### Overall Architecture

Consider $\min_x f(x) = \tfrac{1}{n}\sum_i f_i(x)$, where $f_i$ is convex, Lipschitz, and non-smooth with lower bound $\ell_i^*$ (e.g., cross-entropy = 0).

Both algorithms use the same SPSsafe step size family (floored denominator $\max\{\|g_t^i\|^2,\,M\}$):
- **SSM with SPSsafe**: $x_{t+1} = x_t - \gamma_t g_t^i$, $g_t^i \in \partial f_i(x_t)$, with step size $\gamma_t = \dfrac{f_i(x_t) - \ell_i^*}{\max\{\|g_t^i\|^2,\,M\}}$.
- **IMA with SPSsafe** (Momentum, SHB equivalent): $z_{t+1} = z_t - \eta_t g_t^i$, $x_{t+1} = \tfrac{\lambda_{t+1}}{\lambda_{t+1}+1} x_t + \tfrac{1}{\lambda_{t+1}+1} z_{t+1}$, with step size $\eta_t = \dfrac{[f_i(x_t) - \ell_i^* + \lambda_t\langle g_t^i,\, x_t - x_{t-1}\rangle]_+}{\max\{\|g_t^i\|^2,\,M\}}$.

The safeguard constant $M>0$ floors the denominator, blocking step size explosion when subgradients are small; $\ell_i^* \le \inf f_i$ is the known lower bound. Both variants achieve $O(1/\sqrt{T}+\sigma^2)$ convergence to an optimal neighborhood under convex non-smooth (Lipschitz) conditions. The IMA version also provides last-iterate guarantees without interpolation or $f_i(x^*)$.

### Key Designs

**1. Subgradient norm flooring $\max\{\|g_t^i\|^2, M\}$ (Core Innovation): Blocking explosions caused by small subgradients**

Classical SPS is naturally bounded in smooth scenarios—gradient Lipschitzness ensures $\|g_t^i\|^2$ does not become arbitrarily small relative to the numerator, preventing step size instability. In the non-smooth regime (or late DNN training), subgradients can become extremely small. The denominator $\to 0$ in the naive Polyak step $(f_i(x_t)-\ell_i^*)/\|g_t^i\|^2$ causes the step size to diverge. SPSsafe's critical fix is to **only floor the denominator while keeping the numerator unchanged**:

$$\gamma_t=\frac{f_i(x_t)-\ell_i^*}{\max\{\|g_t^i\|^2,\,M\}}.$$

This is the opposite of SPSmax, which caps the entire step size $\min\{\text{Polyak},\gamma_b\}$. If $\gamma_b$ is chosen too small, SPSmax degrades to a constant step size, losing Polyak’s adaptability. SPSsafe preserves the numerator, so it **never degrades to a constant step size**, maintaining adaptability throughout training. This flooring is also the mathematical key to proving $O(1/\sqrt{T}+\sigma^2)$ convergence in non-smooth settings. The paper interprets this as a form of adaptive gradient clipping, providing the first theory for "Polyak-style clipped SSM."

**2. $\ell_i^*$ as a lower bound instead of $f_i^*$: Bypassing the "oracle" optimal value requirement**

Prior non-smooth SPS variants (like SPS\*) required either interpolation or knowledge of each $f_i(x^*)$, which is unattainable in practice. SPSsafe utilizes the fact that lower bounds $\ell_i^*\le\inf f_i$ are often known (e.g., 0 for non-negative losses). For SSM, since $\ell_i^*\le\inf f_i\le f_i(x_t)$, the numerator is naturally non-negative. For the IMA momentum version, the additional term $\lambda_t\langle g_t^i,x_t-x_{t-1}\rangle$ might make the numerator negative, so a $[\cdot]_+$ truncation is applied. This reduces "oracle-dependent SPS" to "lower-bound-dependent SPS," making the method truly practical.

**3. IMA momentum framework equivalent to SHB: Maintaining momentum without breaking theory**

Momentum is crucial in practice but difficult to analyze in non-smooth settings. SPSsafe uses the dual-sequence IMA form (a $z$ sequence for subgradient updates and an $x$ sequence for averaging), which is equivalent to the Heavy Ball (SHB) method: $x_{t+1}=x_t-\hat\gamma_t g_t^i+\beta(x_t-x_{t-1})$. By substituting the SPSsafe step size $\eta_t=[f_i(x_t)-\ell_i^*+\lambda_t\langle g_t^i,x_t-x_{t-1}\rangle]_+/\max\{\|g_t^i\|^2,M\}$, momentum becomes cleanly provable via Lyapunov-style analysis. The momentum version provides not only Cesàro mean convergence but also last-iterate guarantees without requiring $f_i(x^*)$.

## Key Experimental Results

### Main Results

**Convex Non-smooth Benchmarks**

| Method | Logistic + L1 | SVM hinge loss | ranking loss |
|------|----------|----------|----------|
| AdaGrad | 0.083 | 0.142 | 0.295 |
| Adam | 0.076 | 0.135 | 0.281 |
| DecSPS | 0.082 | 0.148 | 0.302 |
| **SPSsafe (SSM)** | **0.069** | **0.121** | **0.258** |
| **SPSsafe (IMA)** | **0.063** | **0.114** | **0.247** |

SPSsafe leads consistently across three non-smooth convex benchmarks, with the IMA (momentum) version performing best.

**DNN Training (CIFAR-10 + ResNet-18)**

| Method | Test Accuracy | Training Stability |
|------|--------|--------|
| SGD + Tuned LR | 94.7 | Medium (needs tuning) |
| Adam | 93.8 | High |
| AdaGrad | 92.5 | Medium |
| **SPSsafe** | **94.5** | **High (no tuning)** |
| **SPSsafe + IMA** | **95.1** | **High** |

SPSsafe + IMA achieves 95.1% accuracy on ResNet without manual LR tuning, matching or exceeding fine-tuned SGD.

**Gradient Norm Tracking (Anti-Gradient Vanishing)**

| Method | Late-Stage Grad Norm | Gradient Vanishing? |
|------|-------------|----------|
| Adam | Approaches 0 | ✓ Severe |
| AdaGrad | Approaches 0 | ✓ Severe |
| **SPSsafe** | Maintains $\sim 10^{-2}$ | ✗ No |

SPSsafe avoids gradient norm collapse in late stages, suggesting the model remains in an active learning state, which is beneficial for fine-tuning or adversarial robustness.

### Key Findings
- **Safeguard is essential for theory**: Removing the denominator floor ($M\to0$) causes Polyak step sizes to explode or diverge as subgradients vanish.
- **Consistency**: Exhibits SOTA performance across convex non-smooth benchmarks and DNNs.
- **Anti-gradient vanishing**: Gradient norms do not collapse, improving fine-tuning and adversarial training potential.
- **Auto-adaptive**: Polyak-style step sizes are inherently adaptive, removing the need for LR sweeps.

## Highlights & Insights
- **Flooring denominator vs. Capping step size**: Unlike SPSmax, which caps the step size $\min\{\text{Polyak},\gamma_b\}$ and risks constant-step degradation, SPSsafe floors the denominator $\max\{\|g\|^2,M\}$. This preserves adaptability and provides the first theoretical guarantee for Polyak-style clipped SSM.
- **No Interpolation or Oracle**: Previous non-smooth SPS analyses required strong assumptions. This paper removes them to make the method practical.
- **Anti-gradient vanishing**: While traditional adaptive optimizers let gradients shrink to the noise floor, SPSsafe persists, potentially making deep networks more trainable.
- **Analytical convenience of IMA**: Equivalent to SHB, it makes momentum analysis clean and provable.

## Limitations & Future Work
- The safeguard constant $M$ and lower bound $\ell_i^*$ still require manual selection; adaptively choosing $M$ might be more robust.
- Convergence is strictly proven for convex settings; non-convex guarantees remain theoretically open (though empirically functional).
- Verification on large-scale LLMs/Transformers is pending; may require combination with Adam-style second-moment estimation in the future.
- Computing $f_i(x_t)$ requires a forward pass per step, making it slightly more expensive than pure gradient-only methods.

## Related Work & Insights
- **vs. Loizou 2021 (classical SPS)**: Classical SPS needs interpolation; Ours does not.
- **vs. Garrigos 2023 (SPS*)**: SPS* requires an $f_i(x^*)$ oracle; Ours uses known lower bounds.
- **vs. AdaGrad / Adam**: Traditional methods use adaptive second-moments; SPS uses function values—complementary approaches.
- **Insights**: The safeguard philosophy (capping adaptive quantities via their bounds) can be extended to other non-smooth variants of adaptive algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐ (Simple safeguard enables rigorous non-smooth theory; momentum integration is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Convex benchmarks + DNN + gradient analysis)
- Writing Quality: ⭐⭐⭐⭐ (Algorithm is clear; theory matches experiment)
- Value: ⭐⭐⭐⭐ (Non-smoothness is ubiquitous via ReLU and L1; SPSsafe provides a plug-in adaptive solution)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Flatland: The Adventures of Gradient Descent with Large Step Sizes](flatland_the_adventures_of_gradient_descent_with_large_step_sizes.md)
- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICLR 2026\] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization](../../ICLR2026/optimization/faster_gradient_methods_for_highly-smooth_stochastic_bilevel_optimization.md)
- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)

</div>

<!-- RELATED:END -->
