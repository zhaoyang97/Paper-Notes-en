---
title: >-
  [Paper Note] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization
description: >-
  [ICML 2026][Optimization][Stochastic Polyak step size] SPSsafe extends the Stochastic Polyak Step Size (SPS) to non-smooth stochastic optimization without requiring interpolation assumptions or knowledge of optimal value…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Stochastic Polyak step size"
  - "non-smooth optimization"
  - "stochastic subgradient"
  - "momentum"
  - "robust training"
date: 2026-05-08
content_hash: a2111507894be02e
---

# SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization

**Conference**: ICML 2026  
**arXiv**: [2512.02342](https://arxiv.org/abs/2512.02342)  
**Code**: TBD  
**Area**: Optimization / Adaptive Step Sizes / Non-smooth Optimization  
**Keywords**: Stochastic Polyak step size, non-smooth optimization, stochastic subgradient, momentum, robust training

## TL;DR
SPSsafe extends the Stochastic Polyak Step Size (SPS) to non-smooth stochastic optimization without requiring interpolation assumptions or knowledge of optimal values. Combined with momentum (IMA, an equivalent form of SHB), it maintains rigorous convergence guarantees. It demonstrates greater robustness than existing adaptive methods (AdaGrad, Adam, DecSPS) in DNN training and prevents gradient norm collapse (anti-gradient vanishing).

## Background & Motivation

**Background**: Adaptive optimization is standard in modern ML, with AdaGrad and Adam being defaults. The Polyak step size approach has seen a recent resurgence (e.g., Loizou 2021), utilizing function values rather than just gradients to determine step size, showing strong performance on overparameterized models. It has been extended to SGD, Mirror Descent, Local SGD, and SHB.

**Limitations of Prior Work**: (1) Existing SPS analyses mostly assume convexity and smoothness; guarantees for the non-smooth regime (common with ReLU, L1 regularization, ranking loss) are scarce. (2) Existing SPS variants under non-smoothness either require interpolation assumptions (requiring some $x^* \in X^*$ such that $f_i(x^*) = f_i^*$ for all $i$, which is too strong) or knowledge of $f_i(x^*)$ (practically unavailable). (3) Momentum combined with SPS is further underexplored in non-smooth settings.

**Key Challenge**: Achieving the adaptive advantages of Polyak step sizes with convergence guarantees under non-smoothness, without interpolation assumptions or oracle optimal values—no existing SPS variant satisfies all three.

**Goal**: (1) Propose SPSsafe for the stochastic subgradient method (SSM), ensuring rigorous convergence in non-smooth convex optimization without interpolation or optimal values; (2) Integrate momentum (IMA equivalent to SHB) while maintaining theory; (3) Empirically demonstrate robustness in DNN training and the prevention of gradient vanishing.

**Key Insight**: The root cause of classical SPS failure in non-smooth settings is subgradient jumps. While $f_i(x_t) - f_i^*$ is a good proxy for "distance to opt," $\|g_t^i\|^2$ is not guaranteed to behave well in non-smooth regimes. This work introduces a "safeguard"—capping the step size at a fixed upper bound $\gamma_{\max}$ to prevent explosion while retaining Polyak's adaptivity.

**Core Idea**: The safeguard $\gamma_t = \min\{(f_i(x_t) - \ell_i^*)_+/(\|g_t^i\|^2 + \epsilon), \gamma_{\max}\}$, where $\ell_i^* \leq \inf f_i$ is a known lower bound (often 0 for non-negative losses). The IMA momentum framework is used to maintain theoretical consistency.

## Method

### Overall Architecture

Consider $\min_x f(x) = \tfrac{1}{n}\sum_i f_i(x)$, where $f_i$ is convex, Lipschitz, non-smooth, with lower bound $\ell_i^*$ (e.g., cross-entropy = 0).

Two algorithms:
- **SSM with SPSsafe**: $x_{t+1} = x_t - \gamma_t g_t^i$, where $g_t^i \in \partial f_i(x_t)$
- **IMA with SPSsafe** (Momentum, equivalent to SHB): $z_{t+1} = z_t - \eta_t g_t^i$; $x_{t+1} = \tfrac{\lambda_{t+1}}{\lambda_{t+1}+1} x_t + \tfrac{1}{\lambda_{t+1}+1} z_{t+1}$

SPSsafe step size: $\gamma_t = \min\left\{\frac{(f_i(x_t) - \ell_i^*)_+}{\|g_t^i\|^2 + \epsilon}, \gamma_{\max}\right\}$

### Key Designs

1. **Safeguard Upper Bound $\gamma_{\max}$ (Core Innovation)**:
    - **Function**: Caps the Polyak step size at a fixed bound to prevent explosion in non-smooth settings.
    - **Mechanism**: In non-smooth cases, $\|g_t^i\|^2$ can be small even if $f_i(x_t) - \ell_i^*$ is not, potentially causing naive Polyak step sizes to become arbitrarily large. Adding the $\gamma_{\max}$ cap ensures bounded step sizes, making Hoeffding-type bounds in convergence analysis applicable.
    - **Design Motivation**: Classical SPS is naturally bounded in smooth settings (gradient Lipschitz ensures $\|g\|^2$ is not too small), but this is not guaranteed in non-smooth settings. The safeguard is a minimal fix to preserve theory.

2. **$\ell_i^*$ as a Lower Bound Substitute for $f_i^*$**:
    - **Function**: Removes the need for an oracle $f_i^*$, using a known lower bound instead.
    - **Mechanism**: $\ell_i^* \leq \inf f_i$ is typically known (e.g., 0 for non-negative losses); it is substituted into $\gamma_t = (f_i(x_t) - \ell_i^*)_+/\|g_t^i\|^2$, with $(\cdot)_+$ truncation to avoid negative values.
    - **Design Motivation**: Previous SPS* required $f_i(x^*)$ (oracle-dependent). Using a lower bound provides a "known SPS" without extra information requirements.

3. **IMA Momentum Framework equivalent to SHB**:
    - **Function**: Incorporates momentum into SPSsafe without breaking theoretical proofs.
    - **Mechanism**: The dual-sequence IMA (averaging $z$ and $x$) is equivalent to SHB: $x_{t+1} = x_t - \hat\gamma_t g_t^i + \beta(x_t - x_{t-1})$. Using $\hat\gamma_t = \gamma_t$ from SPSsafe, analysis holds because the IMA form is better suited for Lyapunov-based analysis.
    - **Design Motivation**: Momentum is practically useful but theoretically difficult to analyze. The IMA equivalence provides a cleaner framework.

## Key Experimental Results

### Convex Non-smooth Benchmark

| Method | Logistic + L1 | SVM hinge loss | ranking loss |
|------|----------|----------|----------|
| AdaGrad | 0.083 | 0.142 | 0.295 |
| Adam | 0.076 | 0.135 | 0.281 |
| DecSPS | 0.082 | 0.148 | 0.302 |
| **SPSsafe (SSM)** | **0.069** | **0.121** | **0.258** |
| **SPSsafe (IMA)** | **0.063** | **0.114** | **0.247** |

SPSsafe consistently leads across three non-smooth convex benchmarks, with the IMA version (momentum) performing best.

### DNN Training (CIFAR-10 + ResNet-18)

| Method | Test Accuracy | Training Stability |
|------|--------|--------|
| SGD + Tuned LR | 94.7 | Medium (requires tuning) |
| Adam | 93.8 | High |
| AdaGrad | 92.5 | Medium |
| **SPSsafe** | **94.5** | **High (no tuning)** |
| **SPSsafe + IMA** | **95.1** | **High** |

SPSsafe + IMA reaches 95.1% on ResNet (without manual LR tuning), matching or slightly exceeding fine-tuned SGD.

### Gradient Norm Tracking (Anti-gradient Vanishing)

| Method | Late-stage Gradient Norm | Gradient Vanishing? |
|------|-------------|----------|
| Adam | Near 0 | ✓ Severe |
| AdaGrad | Near 0 | ✓ Severe |
| **SPSsafe** | Maintains $\sim 10^{-2}$ | ✗ No |

The gradient norm in SPSsafe does not collapse in late training stages—meaning the model remains in active learning, which is beneficial for continued fine-tuning or adversarial robustness.

### Key Findings
- **Safeguard is essential for theoretical guarantees under non-smoothness**: Without it, Polyak step sizes may explode or diverge in non-smooth regimes.
- **Consistency across convex non-smooth and DNN tasks**: Achieves SOTA on both convex benchmarks and ResNet.
- **Anti-gradient vanishing is an unexpected benefit**: SPSsafe prevents gradient collapse, which is friendly for fine-tuning and adversarial training.
- **No LR tuning required**: Polyak-style step sizes are naturally adaptive, saving the cost of LR sweeps.

## Highlights & Insights
- **Safeguard is a minimum-viable fix**: The most concise way to fix SPS failure in non-smooth optimization, without complicating theory or practice.
- **Eliminating interpolation or oracle requirements is a key contribution**: Previous non-smooth SPS analyses required strong assumptions; this work makes the method truly practical.
- **Side effect of anti-gradient vanishing**: Traditional adaptive optimizers shrink late-stage gradients (often as a noise floor byproduct); SPSsafe does not, implying it may make deep networks more trainable.
- **Analytical convenience of IMA/SHB equivalence**: Providing a dual-sequence framework makes momentum analysis cleaner and more useful.

## Limitations & Future Work
- $\gamma_{\max}$ and $\ell_i^*$ still require manual setting; an adaptive $\gamma_{\max}$ could be more robust.
- Convergence is proven for convex non-smooth cases; non-convex guarantees are not strictly provided (though experiments work).
- Not fully validated on large-scale LLM/Transformer training; integration with Adam-style second-moment estimation might be necessary later.
- Computing $f_i(x_t)$ requires a forward pass per step, making it slightly more expensive than pure gradient methods.
- Distributed/communication-compressed scenarios were not explored.

## Related Work & Insights
- **vs Loizou 2021 (classical SPS)**: Classical SPS requires interpolation; this work does not.
- **vs Garrigos 2023 (SPS*)**: SPS* requires $f_i(x^*)$ oracle; this work uses a lower bound.
- **vs AdaGrad / Adam**: Traditional adaptive routes use second moments; SPS uses function values, offering a complementary approach.
- **vs DecSPS, SPSL, Oikonomou-Loizou 2025**: Previous momentum versions still rely on partial assumptions; this is the first complete theory + momentum version for non-smooth settings.
- **Insight**: The safeguard concept (capping adaptive quantities) can be generalized to other non-smooth extensions of adaptive algorithms; the technique of using lower bounds $\ell_i^*$ instead of oracles is applicable to all Polyak-style methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple safeguard but the first to provide rigorous non-smooth guarantees for SPS; the momentum + non-smooth combination is also new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple convex non-smooth benchmarks + DNN training + gradient norm analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear algorithms with direct theoretical and experimental correspondence; solid derivations.
- Value: ⭐⭐⭐⭐ Non-smooth optimization is ubiquitous in ML (ReLU, sparse regularization); SPSsafe provides a plug-in adaptive solution that avoids LR tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICLR 2026\] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization](../../ICLR2026/optimization/faster_gradient_methods_for_highly-smooth_stochastic_bilevel_optimization.md)
- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](bayesian_gated_non-negative_contrastive_learning.md)

</div>

<!-- RELATED:END -->
