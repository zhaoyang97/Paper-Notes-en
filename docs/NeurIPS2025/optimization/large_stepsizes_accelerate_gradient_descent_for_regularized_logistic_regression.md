---
title: >-
  [Paper Note] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression
description: >-
  [NeurIPS 2025][Optimization][gradient descent] This paper proves that applying GD with large stepsizes (entering the Edge of Stability regime) to $\ell_2$-regularized logistic regression on linearly separable data accele…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "gradient descent"
  - "large stepsizes"
  - "edge of stability"
  - "logistic regression"
  - "acceleration"
  - "condition number"
date: 2026-05-08
content_hash: 0e001113c44a313f
---

# Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression

**Conference**: NeurIPS 2025
**arXiv**: [2506.02336](https://arxiv.org/abs/2506.02336)  
**Code**: None  
**Area**: Optimization
**Keywords**: gradient descent, large stepsizes, edge of stability, logistic regression, acceleration, condition number

## TL;DR
This paper proves that applying GD with large stepsizes (entering the Edge of Stability regime) to $\ell_2$-regularized logistic regression on linearly separable data accelerates the step complexity from the classical $\widetilde{O}(\kappa)$ to $\widetilde{O}(\sqrt{\kappa})$, matching the acceleration rate of Nesterov momentum in the small-regularization regime.

## Background & Motivation
**Background**: $\ell_2$-regularized logistic regression is a classical strongly convex optimization problem. Standard GD with small stepsizes monotonically decreases the objective, achieving step complexity $O(\kappa \ln(1/\varepsilon))$, which is known to be accelerable to $O(\sqrt{\kappa} \ln(1/\varepsilon))$ via Nesterov momentum.

**Limitations of Prior Work**: Wu et al. (2024) demonstrated that large-stepsize GD can accelerate unregularized logistic regression, but their analysis relies on the special structure where the minimizer lies at infinity. Whether large stepsizes retain an acceleration effect for strongly convex problems with finite minimizers remained unknown.

**Key Challenge**: (a) With regularization, the minimizer is finite, and excessively large stepsizes can cause instability and divergence; (b) prior results are valid only for optimization error $\varepsilon < 1/n$, whereas statistical error is typically $\gg 1/n$.

**Key Insight**: The paper analyzes the two-phase behavior of GD in the Edge of Stability (EoS) regime—an initial phase of non-monotone oscillation followed by a stable phase of exponential convergence.

## Method

### Problem Setup

Minimize the $\ell_2$-regularized logistic loss:

$$\widetilde{\mathcal{L}}(w) = \frac{1}{n}\sum_{i=1}^n \ln(1 + \exp(-y_i x_i^\top w)) + \frac{\lambda}{2}\|w\|^2$$

Data satisfy Assumption 1: $\|x_i\| \le 1$, $y_i x_i^\top w^* \ge \gamma > 0$ (linear separability with margin $\gamma$).

The condition number is $\kappa = \Theta(1/\lambda)$, since the smoothness parameter is $\approx 1+\lambda$ and the strong convexity parameter is $\lambda$.

### Theorem 1: Convergence under Small Regularization (Matching Nesterov)

**Conditions**: $\lambda \le \gamma^2 / (C_1 n \ln n)$, stepsize $\eta \le \min\{\gamma/\sqrt{C_1\lambda},\ \gamma^2/(C_1 n \lambda)\}$.

**Two-phase behavior**:
1. **EoS phase** (first $\tau$ steps): The objective oscillates non-monotonically for $\tau = O(\max\{\eta, n, n\ln n / \eta\}/\gamma^2)$ steps.
2. **Stable phase** ($t \ge \tau$): Exponential convergence $\widetilde{\mathcal{L}}(w_t) - \min \widetilde{\mathcal{L}} \le C_3 e^{-\lambda\eta(t-\tau)}$.

**Corollary 2**: Taking the maximum admissible stepsize $\eta = \gamma/\sqrt{C_1\lambda}$ (when $\lambda \lesssim \gamma^2/n^2$), the step complexity is

$$t = O\left(\frac{\ln(1/\varepsilon)}{\gamma\sqrt{\lambda}}\right) = O\left(\frac{\ln(1/\varepsilon)\sqrt{\kappa}}{\gamma}\right)$$

This **matches Nesterov acceleration** using only constant-stepsize GD.

### Theorem 3: Lower Bound in the Stable Regime

A two-dimensional dataset is constructed to show that within the stable regime (where $\widetilde{\mathcal{L}}$ decreases monotonically), the step complexity is at least $\Omega(\ln(1/\varepsilon)/(\lambda \ln^2(1/\lambda)))$, i.e., of order $\Omega(\kappa)$—demonstrating that acceleration must originate from the EoS phase.

### Theorem 4: Improvement under General Regularization

**Conditions**: $\lambda \le \gamma^2/C_1$, $\eta \le (\gamma^2/(C_1\lambda))^{1/3}$.

**Step complexity** (Corollary 5):

$$t = O\left(\frac{\ln(1/\varepsilon)}{(\gamma\lambda)^{2/3}}\right)$$

While this does not match Nesterov's $O(\sqrt{\kappa})$, it still improves upon the classical $O(\kappa)$. This result holds for arbitrary $\lambda$ without the restriction $\lambda \lesssim 1/(n\ln n)$.

### Technical Core: EoS Phase Bound (Lemma 1)

$$\frac{1}{t}\sum_{k=0}^{t-1} \mathcal{L}(w_k) \le O\left(\frac{\eta^2 + \ln^2(e + \gamma^2\min\{\eta t, 1/\lambda\})}{\gamma^2\min\{\eta t, 1/\lambda\}}\right)$$

$$\|w_t\| \le O\left(\frac{\eta + \ln(e + \gamma^2\min\{\eta t, 1/\lambda\})}{\gamma}\right)$$

The proof exploits the self-boundedness of the logistic loss $\|\nabla^2\mathcal{L}(w)\| \le \mathcal{L}(w)$ and the interaction between the strongly convex regularization term and the phase transition.

### Statistical Learning Setting (Section 3)

For data satisfying Assumption 2 (bounded separable distribution) with $\lambda = \Theta(1/n)$, the number of steps required by each algorithm to achieve population risk $\widetilde{O}(1/n)$ is:

| Algorithm | Steps | $\lambda$ | $\eta$ |
|-----------|-------|-----------|--------|
| GD (unregularized, early stopping) | $O(n)$ | 0 | $\Theta(1)$ |
| GD (small stepsize) | $O(n\ln n)$ | $1/n$ | 1 |
| **GD (large stepsize)** | $O(n^{2/3}\ln n)$ | $1/n$ | $\Theta(n^{1/3})$ |
| Nesterov momentum | $O(n^{1/2}\ln n)$ | $1/n$ | 1 |
| Adaptive GD | $O(1/\gamma^2)$ | 0 | $\Theta(\ln n)$ |

The combination of **large stepsizes and regularization** reduces the GD step count from $O(n)$ to $O(n^{2/3})$, providing the first evidence that large stepsizes retain an acceleration effect in the presence of statistical uncertainty.

### Critical Stepsize for Convergence (Section 4, 1D Results)

Under additional data assumptions, the critical stepsize threshold for local convergence is derived to be $\Theta(1/(\lambda\ln(1/\lambda)))$:
- Below this threshold by a constant factor: GD converges locally/globally.
- Above this threshold by a constant factor: GD diverges for almost all initializations.

## Key Experimental Results

Figure 2 illustrates GD behavior with different stepsizes on a two-dimensional separable dataset:

| Stepsize | Behavior | Steps to $10^{-6}$ error |
|----------|----------|--------------------------|
| $\eta = 1$ (small) | Stable regime, monotone decrease | ≈3000 |
| $\eta = 10$ (medium) | EoS regime, oscillation then convergence | ≈500 |
| $\eta = 50$ (large) | EoS regime, faster convergence | ≈200 |
| $\eta = 100$ (too large) | Diverges | ∞ |

The sharpness (largest eigenvalue of the Hessian) oscillates around $2/\eta$ during the EoS phase, consistent with the observations of Cohen et al. (2020).

## Highlights & Insights
1. **Constant-stepsize GD alone matches Nesterov acceleration**—no momentum, no knowledge of $\varepsilon$ required; the method is remarkably simple.
2. This is the first work to prove global convergence and acceleration in the EoS regime for strongly convex problems with finite minimizers.
3. The two-phase analysis framework (EoS → stable) is clearly structured and potentially generalizable to other problems.
4. Acceleration results in the statistical learning setting bridge the gap between pure optimization theory and learning theory.

## Limitations & Future Work
1. Theorem 1 applies only to small regularization $\lambda \lesssim 1/(n\ln n)$; Theorem 4 relaxes this but does not achieve the $\sqrt{\kappa}$ rate.
2. The analysis strictly requires linear separability; Meng et al. (2024) have already constructed counterexamples for the non-separable case.
3. The critical stepsize result is limited to the 1D setting; extension to high dimensions remains an open problem.
4. Large-scale numerical experiments validating the tightness of the theoretical predictions are lacking.
5. Corollaries 2 and 5 yield incomparable bounds in the overlapping parameter regime, suggesting the analysis admits further improvement.

## Related Work & Insights
- **vs. Wu et al. (2024)**: The latter studies the unregularized case ($\lambda=0$, minimizer at infinity); this paper extends large-stepsize acceleration to regularized settings with finite minimizers.
- **vs. Altschuler & Parrilo (2025) Silver stepsize**: The latter achieves $\widetilde{O}(\kappa^{0.79})$ for arbitrary smooth strongly convex functions; this paper achieves $\widetilde{O}(\kappa^{0.5})$ on a narrower problem class using a constant stepsize.
- **vs. Nesterov momentum**: Both achieve $\sqrt{\kappa}$ acceleration, but the present work requires no momentum term and the stepsize is independent of $\varepsilon$.
- **vs. Zhang et al. (2025) Adaptive GD**: The latter requires fewer optimization steps but has worse dependence on $\gamma$ in population risk.

The EoS phenomenon is ubiquitous in deep learning (Cohen et al. 2020); this paper provides a rigorous theoretical foundation for understanding its acceleration mechanism. The synergistic effect of large stepsizes and regularization suggests that practitioners should jointly increase the learning rate and apply regularization during training. The two-phase analysis framework is expected to generalize to other loss functions such as hinge loss and cross-entropy.

## Rating
- ⭐ Novelty: 5/5 — The finding that "constant large stepsizes = Nesterov acceleration" is highly surprising.
- ⭐ Experimental Thoroughness: 2/5 — Relies primarily on theory and low-dimensional toy examples; experiments are insufficient.
- ⭐ Writing Quality: 4/5 — Theoretical exposition is clear with a well-structured two-phase proof.
- ⭐ Value: 4/5 — Strong contribution to core optimization theory, though the applicable scope is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)
- [\[NeurIPS 2025\] Gradient Descent as Loss Landscape Navigation: a Normative Framework for Deriving Learning Rules](gradient_descent_as_loss_landscape_navigation_a_normative_framework_for_deriving.md)

</div>

<!-- RELATED:END -->
