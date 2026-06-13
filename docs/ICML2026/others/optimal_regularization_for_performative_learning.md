---
title: >-
  [Paper Note] Optimal Regularization for Performative Learning
description: >-
  [ICML2026][Performative learning] Under a high-dimensional ridge regression framework, this work systematically characterizes the scaling law of the optimal regularization strength in "performativity" scenarios where mod…
tags:
  - "ICML2026"
  - "Performative learning"
  - "Ridge regularization"
  - "High-dimensional statistics"
  - "Repeated Risk Minimization"
  - "Spurious features"
date: 2026-05-08
content_hash: e939502f44cdb923
---

# Optimal Regularization for Performative Learning

**Conference**: ICML2026  
**arXiv**: [2510.12249](https://arxiv.org/abs/2510.12249)  
**Code**: https://github.com/totilas/regularization-vs-perf  
**Area**: others (High-dimensional learning theory / Performative learning / Ridge regression)  
**Keywords**: Performative learning, Ridge regularization, High-dimensional statistics, Repeated Risk Minimization, Spurious features

## TL;DR
Under a high-dimensional ridge regression framework, this work systematically characterizes the scaling law of the optimal regularization strength in "performativity" scenarios where model deployment drives data distribution shifts: the optimal $\lambda$ is proportional to the performative intensity $\bar b$. In overparameterized regimes, appropriate regularization can even leverage performative effects to inversely reduce risk.

## Background & Motivation

**Background**: Performative learning (Perdomo et al. 2020) investigates a feedback loop where the deployed model $\theta$ alters the subsequent data distribution $\mathcal{D}(\theta)$. A typical example is strategic users modifying their features to obtain loans. Research follows two main paths: explicit estimation of the performative operator (Miller 2021, Izzo 2022, Cyffers 2024), and direct Repeated Risk Minimization (RRM).

**Limitations of Prior Work**: The first path is computationally feasible only for low-dimensional, small-scale examples. While RRM is more practical (as deployment often occurs only once), existing analysis is largely limited to strongly convex losses in low-dimensional settings. When entering the modern overparameterized regime where dimension $p$ and sample size $n$ are of the same order, existing theories are virtually silent—yet overparameterization is where phenomena like double descent and benign overfitting reside.

**Key Challenge**: Regularization appears to be a low-cost countermeasure, but in high dimensions, it can encourage the model to rely on spurious features (Bombari & Mondelli 2025). If performative effects amplify these spurious features, blindly increasing $\lambda$ might worsen the model. Thus, the question of "how much regularization to add and in which direction" remains open in performative learning.

**Goal**: Characterize the influence of ridge regularization on the RRM fixed-point risk under high-dimensional linear regression in two scenarios: (i) the population limit and (ii) the proportional regime $p/n=\kappa>1$, and provide closed-form expressions for the optimal $\lambda^*$.

**Key Insight**: The authors model the performative effect as an additional linear term in the labels: $y = x^\top \theta^*_{\text{pop}} + x^\top D\theta + w$, where $D=\text{diag}(b,c)$ separately models the performative intensity of predictive and spurious features. This allows utilizing high-dimensional random matrix tools from Han & Xu (2023) to derive deterministic equivalents while maintaining analytical control over which features are amplified.

**Core Idea**: By treating the performative effect as a perturbation with a known direction, it is proved that the "optimal regularization scales proportionally with performative intensity $\bar b$." A practical $\lambda$ selection rule is derived that does not require estimating $D$.

## Method

The entire paper is a theoretical analysis; the "pipeline" consists of expressing the RRM fixed-point risk as an analytical function of $\lambda$, $D$, and $\Sigma$, and then minimizing it with respect to $\lambda$.

### Overall Architecture

Let features $x\in\mathbb{R}^p$ ($p=2d$), where the first $d$ dimensions are predictive and the latter $d$ are spurious. Ground truth parameters are $\theta^*_{\text{pop}} = (a^\top, 0)^\top$. The performative matrix is $D=\text{diag}(b,c)$, where $b$ corresponds to predictive features and $c$ to spurious features. Labels are generated as $y = x^\top \theta^*_{\text{pop}} + x^\top D\theta + w$, where $w\sim\mathcal{N}(0,\sigma^2)$.

At RRM iteration $k$, the solution is $\theta_k = \arg\min_\theta \tfrac{1}{2n}\sum_i \ell(x_i^{(k-1)}, y_i^{(k-1)};\theta) + \tfrac{\lambda}{2}\|\theta\|_2^2$, with data sampled from the previous distribution $\mathcal{D}(\theta_{k-1})$. Evaluation risk is defined on the initial distribution $\mathcal{D}(\theta=0)$, with excess risk $\mathcal{R}(\Sigma,\theta,\theta^*_{\text{pop}}) = \|\Sigma^{1/2}(\theta-\theta^*_{\text{pop}})\|_2^2$.

In the population case, RRM converges to the fixed point $\theta^\infty = (I_p + \lambda\Sigma^{-1} - D)^{-1}\theta^*_{\text{pop}}$. In the overparameterized regime with finite data, deterministic equivalents are derived using high-dimensional random matrix theory.

### Key Designs

1. **Population Limit: Optimal $\lambda$ proportional to $\bar b$**

    - **Function**: Provides a closed-form expression for the optimal regularization strength under the $n\to\infty$ assumption.
    - **Mechanism**: Let $F = D - \lambda\Sigma^{-1}$. A second-order Taylor expansion of the excess risk with respect to $F$ yields the dominant term $\widetilde{\mathcal{R}}_{\text{pop}}(D,\lambda,\Sigma) = \tfrac{1}{d}\text{Tr}[\text{diag}(b^2)\Sigma_1] - 2\lambda\bar b + \tfrac{\lambda^2}{d}\text{Tr}(S_1)$, where $\bar b = \tfrac{1}{d}\sum_i b_i$ and $S_1$ is the Schur complement of the covariance. This is an explicit quadratic form in $\lambda$, with a minimum at $\lambda^*_{\text{pop}} = \bar b d / \text{Tr}(S_1)$.
    - **Design Motivation**: This result is highly practical—the optimal regularization depends only on the "average intensity" of performativity $\bar b$ and the feature covariance structure, **obviating the need for coordinate-wise estimation of $D$**. It also reveals an intuitive picture: "rich-get-richer" performative feedback ($\bar b>0$) requires stronger regularization to suppress, while "self-decaying" performativity ($\bar b<0$) requires negative regularization.

2. **Overparameterized Regime: Performativity can inversely reduce risk**

    - **Function**: Provides a deterministic equivalent for risk in the proportional regime $p/n=\kappa>1$.
    - **Mechanism**: Based on the high-dimensional risk characterization by Han & Xu (2023) and Ildiz et al. (2025), two-step RRM iteration is performed (sufficient for a first-order approximation of the fixed point) to obtain the deterministic equivalent $\mathcal{R}_{\text{eq}}(\Sigma,\theta^*_{\text{pop}},D,\lambda)$, which depends on an auxiliary scalar $\tau$ solving the fixed point equation $\kappa^{-1} - \lambda/\tau = \tfrac{1}{p}\text{Tr}[(\Sigma+\tau I_p)^{-1}\Sigma]$. After expanding to first order in $\bar b$ and $\bar c$, the signs of four auxiliary functions $B_1, B_2, C_1, C_2$ determine the directional influence of performativity.
    - **Design Motivation**: A key finding is $B_2(\kappa,\sigma)\le 0$. If performativity "amplifies existing trends" ($\bar b>0$), the risk under optimal regularization is **actually lower than without performativity**. This is opposite to the population scenario. The mechanism is that variance dominates in the overparameterized regime; performativity superimposes signals in a consistent direction, effectively increasing the signal-to-noise ratio.

3. **Noise levels flip the direction of regularization**

    - **Function**: Explains why regularization should move in opposite directions under different noise levels.
    - **Mechanism**: Analyzing the sign of $B_1(\sigma,\kappa)$ yields a critical noise level $\sigma_{B_1}^2(\kappa) = 1/2 - 7\kappa^{-1}/18 + O(\kappa^{-2})$. When $\sigma < \sigma_{B_1}$ (low noise), $B_1\ge 0$ and the optimal $\lambda$ moves in the same direction as $\bar b$. When $\sigma > \sigma_{B_1}$, $B_1\le 0$ and the optimal $\lambda$ moves in the opposite direction. For performativity on spurious features $\bar c$, the optimal $\lambda$ always moves opposite to $\bar c$ as long as $\kappa\ge 2$, though the effect is dampened by $\rho^2$ (spurious-predictive feature correlation).
    - **Design Motivation**: The authors provide a Bayesian intuition: under high noise, the model should "revert to the prior," so the correction brought by performativity must move in the opposite direction to balance overconfidence. This rule is directly validated by the curve inversion in the small-sample LSAC dataset ($n=100, d=22$).

### Loss & Training
The study uses square loss $\ell(x,y;\theta) = (y - x^\top\theta)^2$ with ridge regularization $\tfrac{\lambda}{2}\|\theta\|_2^2$. Theoretical analysis requires no hyperparameter training. Experiments show 4–5 iterations of RRM are sufficient to approach the fixed point, matching the theoretical prediction that "two-step iteration is sufficient."

## Key Experimental Results

### Main Results: Synthetic Data + Real-world Datasets

| Setting | Data | Key Observation | Consistency with Theory |
|------|------|---------|------------|
| Population regime, $d=100, \Sigma=I_p$ | Synthetic | $\lambda^*$ grows linearly with $\bar b$; $\lambda^*<0$ when $\bar b<0$ | Corollary 4.2 |
| Proportional regime low noise $\kappa=1.1, \sigma=0.2$ | Synthetic | $\lambda^*$ is larger and risk is **lower** at $\bar b=0.2$ than $\bar b=0$ | Theorem 5.2 ($B_2\le 0$) |
| Proportional regime high noise $\kappa=1.1, \sigma=0.7$ | Synthetic | $\bar b=0.2$ decreases $\lambda^*$, risk still drops | $B_1\le 0$ inversion |
| Housing ($n=4000, d=8$) | Real | $\lambda^*$ increases with $\bar b$, risk worsens with $\bar b$ | Population behavior |
| LSAC ($n=4000, d=22$) | Real | Same as above | Population behavior |
| LSAC ($n=100, d=22$) | Real | $\lambda^*$ decreases with $\bar b$, risk improves with $\bar b$ | Overparameterized high noise |

### Ablation Study: Alternative Regularization Forms

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Ridge (Main focus) | $\lambda^*\propto\bar b$ | Target of theoretical analysis |
| Dropout | Same pattern | Suggests relationship is norm-independent |
| Lasso | Same pattern | Qualitative conclusions hold under $\ell_1$ |
| Elastic Net | Same pattern | Consistent with hybrid regularization |
| Neural Net + GiveMeSomeCredit (Mofakhami 2023) | $\ell_2$ reg. mitigates accuracy drop from large $\delta$; optimal $\lambda$ increases with $\delta$ | Qualitative conclusions extend to non-linear models |

### Key Findings
- **Most Critical Conclusion**: The combination of overparameterization and strong signal performativity ($\bar b>0$) makes the optimal risk **better** than the zero-performativity baseline, overturning the intuition that "performativity is always bad."
- **Small impact from spurious feature performativity $\bar c$**: Since it is multiplied by $\rho^2$ in the risk expansion, it is barely observable in empirical results; this suggests that performative modeling for spurious features can be safely ignored in practice.
- **Sample size determines the regime**: The same dataset (LSAC) follows population logic when $n\gg d$, but switches to overparameterized high-noise logic when $n \approx d$—offering direct practical guidance.
- **Weak requirements for RRM convergence**: Two steps suffice to reach the fixed point (in a first-order sense), naturally fitting industrial scenarios where deployment frequency is extremely low.

## Highlights & Insights
- **First application of high-dimensional statistical tools to performative learning**: Mapping the deterministic equivalent framework of Han & Xu (2023) to RRM fixed points creates a new analytical path. These tools can likely be applied to performative classification or performative adversarial robustness.
- **Practical "optimal $\lambda$ scales with average performative intensity" rule**: Does not require coordinate-wise estimation of $D$, only a scalar $\bar b$, making the method viable in high dimensions—where previous performative algorithms failed for $p>100$.
- **Interpretation of negative regularization**: While negative $\lambda$ is often treated as a quirk of overparameterization, this work provides a new explanation—when performativity is "self-decaying," negative regularization exactly offsets this decay.
- **Honest presentation of negative results**: The authors explicitly note that the impact of $c$ (spurious performativity) was not observed in experiments, avoiding exaggerated claims for that specific portion.

## Limitations & Future Work
- **Strong linear assumptions**: Analysis relies on labels being linear in $\theta$ and Gaussian features; real-world performative feedback is likely non-linear. Neural network experiments provide only qualitative evidence.
- **Limited to label shifts**: Does not cover feature shifts. While the authors suggest feature shifts can be partially absorbed by centering, real scenarios in strategic classification often involve feature modification.
- **Fixed optimization method**: Focusing on "adding regularization," the study does not explore other implicit regularization methods like early stopping, pruning, or data augmentation.
- **Ambiguity in test distribution**: Performance must be evaluated on either $\mathcal{D}(0)$ or $\mathcal{D}(\theta)$. The former ignores social evolution, while the latter encourages distribution manipulation. Defining a "fair" metric remains an open problem.

## Related Work & Insights
- **vs Perdomo et al. (2020)**: They proved RRM convergence under strong convexity; this work provides scaling laws for optimal regularization and extends analysis to overparameterized regimes.
- **vs Cyffers et al. (2024)**: They provided numerical evidence that "performatively optimal $\approx$ regularized non-performatively optimal" in classification; this work proves that intuition as a theorem in regression and reveals inverse phenomena in overparameterized cases.
- **vs Hastie et al. (2022) / Patil et al. (2024)**: These works studied the optimality of ridge in standard/OOD regression, including conditions for negative $\lambda$; this work shows negative $\lambda$ can also arise due to performativity in the population regime.
- **vs Bombari & Mondelli (2025)**: They noted that high-dimensional regularization might increase reliance on spurious features; this work validates that the optimal $\lambda$ direction for spurious features is opposite, establishing a dialogue with their concerns.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First introduction of high-dimensional statistics to performative learning; discovery of counter-intuitive risk improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + Housing + LSAC + Neural Nets; lacks real performative deployment data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear correspondence between theorems and figures; well-explained intuition.
- Value: ⭐⭐⭐⭐ Provides executable $\lambda$ selection rules useful for engineers in strategic classification or recommender systems.

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Tight Lower Bounds and Improved Convergence in Performative Prediction](../../NeurIPS2025/others/tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)
- [\[ICML 2026\] Guaranteed Optimal Compositional Explanations for Neurons](guaranteed_optimal_compositional_explanations_for_neurons.md)
- [\[AAAI 2026\] Optimal Welfare in Noncooperative Network Formation under Attack](../../AAAI2026/others/optimal_welfare_in_noncooperative_network_formation_under_attack.md)
- [\[ICML 2026\] Learning Permutation-Invariant Macroscopic Dynamics](learning_permutation-invariant_macroscopic_dynamics.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](../../AAAI2026/others/area-optimal_control_strategies_for_heterogeneous_multi-agen.md)

</div>

<!-- RELATED:END -->
