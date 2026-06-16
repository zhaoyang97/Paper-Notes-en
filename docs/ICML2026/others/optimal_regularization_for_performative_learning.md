---
title: >-
  [Paper Note] Optimal Regularization for Performative Learning
description: >-
  [ICML 2026][Others][Paper Note] Under a high-dimensional ridge regression framework, this paper systematically characterizes the scaling law of optimal regularization strength in "performativity" scenarios where model deployment drives data distribution shifts. The optimal $\lambda$ is proportional to the performative strength $\bar b$. In over-param
tags:
  - ICML 2026
  - Others
date: 2026-05-08
content_hash: 209942b58d00e779
---
# Optimal Regularization for Performative Learning

**Conference**: ICML2026  
**arXiv**: [2510.12249](https://arxiv.org/abs/2510.12249)  
**Code**: https://github.com/totilas/regularization-vs-perf  
**Area**: others (High-dimensional learning theory / Performative learning / Ridge regression)  
**Keywords**: Performative learning, Ridge regularization, High-dimensional statistics, Repeated Risk Minimization, Spurious features

## TL;DR
Under a high-dimensional ridge regression framework, this paper systematically characterizes the scaling law of optimal regularization strength in "performativity" scenarios where model deployment drives data distribution shifts. The optimal $\lambda$ is proportional to the performative strength $\bar b$. In over-parameterized regimes, proper regularization can even leverage performative effects to reduce risk.

## Background & Motivation

**Background**: Performative learning (Perdomo et al. 2020) investigates the feedback loop where a deployed model $\theta$ alters the next sampled data distribution $\mathcal{D}(\theta)$, typically seen when strategic users modify their features to obtain loans. Research follows two main lines: explicitly estimating the performative operator (Miller 2021, Izzo 2022, Cyffers 2024), or directly implementing Repeated Risk Minimization (RRM).

**Limitations of Prior Work**: The first line is only feasible for low-dimensional examples as it requires multiple deployments for distribution alignment. While RRM is more practical (as deployment often happens once), existing analyses primarily cover strongly convex losses in low-dimensional settings. In modern over-parameterized regimes where feature dimension $p$ is proportional to sample size $n$, existing theory remains silent—despite these regimes being the habitat for phenomena like double descent and benign overfitting.

**Key Challenge**: Regularization appears to be a low-cost solution, but in high dimensions, it may encourage models to rely on spurious features (Bombari & Mondelli 2025). If performative effects amplify these spurious features, blindly increasing $\lambda$ might push the model toward worse performance. Thus, determining "how much regularization and in which direction" remains an open question in performative learning.

**Goal**: To characterize the impact of ridge regularization on the RRM fixed-point risk in high-dimensional linear regression under (i) the population limit and (ii) the proportional regime $p/n = \kappa > 1$, and to provide a closed-form for the optimal $\lambda^*$.

**Key Insight**: The authors model the performative effect as an additional linear term in the labels: $y = x^\top \theta^*_{\text{pop}} + x^\top D\theta + w$, where $D = \text{diag}(b, c)$ models performative strengths for predictive and spurious features separately. This allows the use of high-dimensional random matrix tools (Han & Xu 2023) to find deterministic equivalents while retaining analytical control over which features are performatively amplified.

**Core Idea**: Treat performative effects as perturbations in a known direction. The paper proves that in high-dimensional linear regression, "optimal regularization scales proportionally with performative strength $\bar b$," leading to a practical rule for selecting $\lambda$ without estimating $D$.

## Method

The paper is a theoretical analysis; the "pipeline" involves expressing the RRM fixed-point risk as an analytical function of $\lambda$, $D$, and $\Sigma$, and then minimizing it with respect to $\lambda$.

### Overall Architecture

Let features $x \in \mathbb{R}^p$ ($p=2d$), where the first $d$ dimensions are predictive and the latter $d$ are spurious. Ground truth parameters are $\theta^*_{\text{pop}} = (a^\top, 0)^\top$; the performativity matrix is $D = \text{diag}(b, c)$, where $b$ corresponds to predictive features and $c$ to spurious features. Labels are generated as $y = x^\top \theta^*_{\text{pop}} + x^\top D\theta + w$, where $w \sim \mathcal{N}(0, \sigma^2)$.

The $k$-th iteration of RRM yields $\theta_k = \arg\min_\theta \tfrac{1}{2n}\sum_i \ell(x_i^{(k-1)}, y_i^{(k-1)}; \theta) + \tfrac{\lambda}{2}\|\theta\|_2^2$, with data drawn from the previous distribution $\mathcal{D}(\theta_{k-1})$. The evaluation risk is defined on the initial distribution $\mathcal{D}(\theta=0)$ unpolluted by performativity, with excess risk $\mathcal{R}(\Sigma, \theta, \theta^*_{\text{pop}}) = \|\Sigma^{1/2}(\theta - \theta^*_{\text{pop}})\|_2^2$.

In the population scenario, RRM converges to the fixed point $\theta^\infty = (I_p + \lambda\Sigma^{-1} - D)^{-1}\theta^*_{\text{pop}}$. In the over-parameterized scenario with finite data, "deterministic equivalents" are derived using high-dimensional random matrix theory.

### Key Designs

**1. Population Limit: Optimal $\lambda$ is proportional to average performative strength $\bar b$**

Under the population limit ($n \to \infty$), the authors seek a practical formula for selecting $\lambda$. By letting $F = D - \lambda\Sigma^{-1}$ and performing a second-order Taylor expansion of the excess risk around $F$, they obtain the dominant term:

$$\widetilde{\mathcal{R}}_{\text{pop}}(D,\lambda,\Sigma) = \tfrac{1}{d}\text{Tr}[\text{diag}(b^2)\Sigma_1] - 2\lambda\bar b + \tfrac{\lambda^2}{d}\text{Tr}(S_1),$$

where $\bar b = \tfrac{1}{d}\sum_i b_i$ is the average performative strength on the predictive side, and $S_1$ is the Schur complement of the covariance. This is an explicit quadratic form for $\lambda$, where the first-order condition yields the minimizer $\lambda^*_{\text{pop}} = \bar b\, d / \text{Tr}(S_1)$. Its practical value lies in the fact that optimal regularization depends only on the "average strength" $\bar b$ and the covariance structure, requiring **no coordinate-wise estimation of $D$**—a step where previous performative algorithms often fail when $p > 100$. The conclusion is intuitive: positive feedback ( $\bar b > 0$) requires stronger regularization to suppress, whereas "self-negating" performativity ( $\bar b < 0$) requires negative regularization, providing a new physical explanation for negative $\lambda$ often encountered in over-parameterization.

**2. Over-parameterized Regime: Aligned performativity can reduce risk**

In the proportional regime $p/n = \kappa > 1$, finite data renders population formulas invalid, requiring "deterministic equivalents" from high-dimensional random matrix theory. Building on the risk frameworks of Han & Xu (2023) and Ildiz et al. (2025), the authors perform a two-step RRM iteration (sufficient for first-order convergence to the fixed point) to derive the deterministic equivalent risk $\mathcal{R}_{\text{eq}}(\Sigma, \theta^*_{\text{pop}}, D, \lambda)$. This depends on an auxiliary scalar $\tau$ determined by the fixed-point equation $\kappa^{-1} - \lambda/\tau = \tfrac{1}{p}\text{Tr}[(\Sigma + \tau I_p)^{-1}\Sigma]$. Expanding the risk to the first order of $\bar b, \bar c$, the signs of four auxiliary functions $B_1, B_2, C_1, C_2$ determine whether performativity helps or hurts. A key result is $B_2(\kappa, \sigma) \le 0$: provided performativity amplifies existing trends ($\bar b > 0$), the risk under optimal regularization is **actually lower than without performativity**. This contradicts the population case; the mechanism is that variance dominates the over-parameterized regime, and performativity stacks the signal in the same direction, effectively boosting the signal-to-noise ratio.

**3. Noise level flips the direction of regularization**

The direction of regularization adjustment depends on the noise level. Analyzing the sign of $B_1(\sigma, \kappa)$ identifies a critical noise level $\sigma_{B_1}^2(\kappa) = 1/2 - 7\kappa^{-1}/18 + O(\kappa^{-2})$. At low noise ($\sigma < \sigma_{B_1}$), $B_1 \ge 0$, and the optimal $\lambda$ moves in the same direction as $\bar b$. At high noise ($\sigma > \sigma_{B_1}$), $B_1 \le 0$, and the direction flips. For performativity on spurious features $\bar c$, the direction is always opposite to $\bar c$ as long as $\kappa \ge 2$, though the impact is heavily suppressed by the correlation $\rho^2$ between spurious and predictive features. This aligns with Bayesian intuition: at high noise, the model should "revert to the prior," so performative corrections must move in the opposite direction to avoid overconfidence. This rule was directly validated by curve flipping in LSAC experiments with small samples ($n=100, d=22$).

### Loss & Training
The study utilizes square loss $\ell(x, y; \theta) = (y - x^\top \theta)^2$ with ridge regularization $\tfrac{\lambda}{2}\|\theta\|_2^2$. Theoretical analysis does not require hyperparameter training; in experiments, 4–5 iterations of RRM are sufficient to approach the fixed point, consistent with theoretical predictions that two iterations suffice.

## Key Experimental Results

### Main Results: Synthetic + Real Datasets

| Setting | Data | Key Findings | Consistency with Theory |
|------|------|---------|------------|
| Population, $d=100, \Sigma=I_p$ | Synthetic | $\lambda^*$ grows linearly with $\bar b$; $\lambda^* < 0$ when $\bar b < 0$ | Corollary 4.2 |
| Proportional, low noise $\kappa=1.1, \sigma=0.2$ | Synthetic | Larger $\lambda^*$ and **lower** risk at $\bar b=0.2$ compared to $\bar b=0$ | Theorem 5.2 ($B_2 \le 0$) |
| Proportional, high noise $\kappa=1.1, \sigma=0.7$ | Synthetic | $\bar b=0.2$ decreases $\lambda^*$, yet risk still decreases | $B_1 \le 0$ flip |
| Housing ($n=4000, d=8$) | Real | $\lambda^*$ increases with $\bar b$, risk worsens with $\bar b$ | Population behavior |
| LSAC ($n=4000, d=22$) | Real | Same as above | Population behavior |
| LSAC ($n=100, d=22$) | Real | $\lambda^*$ decreases as $\bar b$ increases, risk improves with $\bar b$ | Over-parameterized high-noise prediction |

### Ablation Study: Alternative Regularization Forms

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Ridge (Main focus) | $\lambda^* \propto \bar b$ | Primary theoretical target |
| Dropout | Same patterns | Indicates relationship is independent of specific norm |
| Lasso | Same patterns | Qualitative conclusions hold under $\ell_1$ |
| Elastic Net | Same patterns | Consistent across hybrid regularization |
| NN + GiveMeSomeCredit (Mofakhami 2023) | $\ell_2$ reg mitigates accuracy drop from $\delta$, optimal $\lambda$ increases with $\delta$ | Qualitative conclusions extend to non-linear models |

### Key Findings
- **Most significant conclusion**: Over-parameterization combined with performativity in the signal direction ($\bar b > 0$) leads to an optimal risk that is **better** than the non-performative baseline, overturning the intuition that performativity is always harmful.
- **Minimal impact of spurious performativity $\bar c$**: Since it is multiplied by $\rho^2$ in the risk expansion, it is negligible in experiments, suggesting that spurious performative modeling can be safely ignored in practice.
- **Sample size determines the applicable curve**: On the same dataset (LSAC), $n \gg d$ follows the "increase regularization" logic of the population regime, while $n \approx d$ shifts to over-parameterized high-noise logic—a direct practical guide.
- **Weak requirements for RRM convergence**: Two-step iteration is sufficient to reach the fixed point (in a first-order sense), naturally fitting industrial scenarios with very few deployments.

## Highlights & Insights
- **First to systematically characterize performative learning using high-dimensional statistical tools**: Mapping the deterministic equivalent framework of Han & Xu (2023) to RRM fixed points creates a new analytical path. These tools can likely be extended to performative classification or adversarial robustness.
- **Practical rule where "optimal $\lambda$ is proportional to average performative strength"**: Does not require coordinate-wise estimation of $D$, only the estimation of a scalar $\bar b$, making the method viable in high dimensions where previous algorithms fail.
- **Physical explanation for negative regularization**: While negative $\lambda$ is often treated as a quirk of over-parameterization, this paper suggests that when performativity is "self-negating," negative regularization acts to counteract that decay.
- **Honest presentation of failure cases**: The authors explicitly state that the impact of $c$ (spurious performativity) was unobservable in experiments rather than claiming it as a key contribution—a level of restraint commendable in ICML submissions.

## Limitations & Future Work
- **Strong linear assumptions**: Analysis relies on labels being linear in $\theta$ and features being Gaussian (sub-Gaussian extensions possible). Real performative feedback is likely non-linear. Neural network experiments provide only qualitative evidence.
- **Label drift focus**: Does not cover feature drift. Although centering can absorb some feature drift, real-world strategic classification often involves fundamental feature changes.
- **Optimization methods limited**: Focuses on "adding regularization" without exploring other implicit methods like early stopping, pruning, or data augmentation—noted by the authors as a future direction.
- **Ambiguity in test distribution**: Highlighting a choice between evaluating on $\mathcal{D}(0)$ or $\mathcal{D}(\theta)$; the former ignores social evolution while the latter encourages distribution manipulation. Defining "fair" performative evaluation remains open.

## Related Work & Insights
- **vs. Perdomo et al. (2020)**: They proved RRM convergence under strong convexity; this work provides scaling laws for optimal regularization and extends analysis to over-parameterized regimes.
- **vs. Cyffers et al. (2024)**: They provided numerical evidence that "performative optimal ≈ regularized non-performative optimal" in classification; this paper proves this intuition as a theorem in regression and reveals the reverse phenomenon under over-parameterization.
- **vs. Hastie et al. (2022) / Patil et al. (2024)**: These works studied the optimality of ridge in standard/OOD regression, including negative $\lambda$; this paper shows negative $\lambda$ can emerge due to performativity even in the population regime.
- **vs. Bombari & Mondelli (2025)**: They noted that high-dimensional regularization might worsen reliance on spurious features; this work validates that the optimal $\lambda$ direction for spurious features is reversed, engaging with their concerns.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First use of high-dimensional statistical tools in performative learning; identifies counter-intuitive "performativity improves risk" phenomenon.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered synthetic, Housing, LSAC, and neural networks, though real performative datasets are still lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear correspondence between theorems and figures; explains both intuition and formulas effectively.
- Value: ⭐⭐⭐⭐ Provides actionable rules for selecting $\lambda$, useful for engineers in strategic classification or recommendation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Revisiting the Predictability of Performative, Social Events](../../ICML2025/others/revisiting_the_predictability_of_performative_social_events.md)
- [\[ICML 2026\] Guaranteed Optimal Compositional Explanations for Neurons](guaranteed_optimal_compositional_explanations_for_neurons.md)
- [\[NeurIPS 2025\] Tight Lower Bounds and Improved Convergence in Performative Prediction](../../NeurIPS2025/others/tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)
- [\[ICML 2025\] Cross-regularization: Adaptive Model Complexity through Validation Gradients](../../ICML2025/others/cross-regularization_adaptive_model_complexity_through_validation_gradients.md)
- [\[ICML 2026\] Learning Permutation-Invariant Macroscopic Dynamics](learning_permutation-invariant_macroscopic_dynamics.md)

</div>

<!-- RELATED:END -->
