---
title: >-
  [Paper Note] Optimal Regularization for Performative Learning
description: >-
  [ICML2026][Performative learning] This paper systematically characterizes the scaling laws of optimal regularization strength within a high-dimensional ridge regression framework under "performativity," where model deployment drives data distribution shifts. The optimal $\lambda$ is found to be proportional to the performative strength $\bar b$, and in overparameterized regimes, appropriate regularization can even leverage performative effects to reduce risk.
tags:
  - "ICML2026"
  - "Performative learning"
  - "Ridge regularization"
  - "High-dimensional statistics"
  - "Repeated Risk Minimization"
  - "Spurious features"
date: 2026-05-08
content_hash: 449fe4c4ea344231
---

# Optimal Regularization for Performative Learning

**Conference**: ICML2026  
**arXiv**: [2510.12249](https://arxiv.org/abs/2510.12249)  
**Code**: https://github.com/totilas/regularization-vs-perf  
**Area**: others (High-dimensional learning theory / Performative learning / Ridge regression)  
**Keywords**: Performative learning, Ridge regularization, High-dimensional statistics, Repeated Risk Minimization, Spurious features

## TL;DR
This paper systematically characterizes the scaling laws of optimal regularization strength within a high-dimensional ridge regression framework under "performativity," where model deployment drives data distribution shifts. The optimal $\lambda$ is found to be proportional to the performative strength $\bar b$, and in overparameterized regimes, appropriate regularization can even leverage performative effects to reduce risk.

## Background & Motivation

**Background**: Performative learning (Perdomo et al. 2020) investigates a feedback loop where the deployed model $\theta$ alters the subsequent sampled data distribution $\mathcal{D}(\theta)$. A typical example is strategic users modifying their features to obtain loans. Research follows two main lines: explicit estimation of the performative operator (Miller 2021, Izzo 2022, Cyffers 2024), and direct Repeated Risk Minimization (RRM).

**Limitations of Prior Work**: The first line is computationally feasible only in low-dimensional cases and requires multiple deployment rounds for distribution alignment. While RRM is more practical as deployment often occurs only once, existing analysis is largely restricted to strongly convex losses in low-dimensional settings. Theoretical understanding remains limited in modern overparameterized regimes where the feature dimension $p$ is of the same order as the sample size $n$—the domain of phenomena like double descent and benign overfitting.

**Key Challenge**: While regularization seems to be a low-cost mitigation strategy, it can encourage models to rely on spurious features in high dimensions (Bombari & Mondelli 2025). If performative effects amplify these spurious features, blindly increasing $\lambda$ might worsen model performance. Thus, determining the optimal magnitude and direction of regularization in performative learning remains an open question.

**Goal**: This paper aims to characterize the impact of ridge regularization on the risk of RRM fixed points under (i) the population limit and (ii) the proportional regime $p/n=\kappa>1$ within high-dimensional linear regression, providing closed-form expressions for the optimal $\lambda^*$.

**Key Insight**: The authors model the performative effect as an additional linear term in the labels: $y = x^\top \theta^*_{\text{pop}} + x^\top D\theta + w$, where $D=\text{diag}(b,c)$ separately models performative strengths for predictive and spurious features. This allows the use of high-dimensional random matrix tools (Han & Xu 2023) for finding deterministic equivalents while maintaining analytical control over which features are amplified by performativity.

**Core Idea**: By treating performative effects as perturbations in a known direction, it is demonstrated that the optimal regularization scales proportionally with performative strength $\bar b$ in high-dimensional linear regression. This leads to a practical rule for selecting $\lambda$ without requiring the estimation of $D$.

## Method

The paper provides a theoretical analysis where the risk of the RRM fixed point is expressed as an analytical function of $\lambda$, $D$, and $\Sigma$, followed by minimization with respect to $\lambda$.

### Overall Architecture

Features $x\in\mathbb{R}^p$ ($p=2d$) consist of $d$ predictive dimensions and $d$ spurious dimensions. The ground truth parameters are $\theta^*_{\text{pop}} = (a^\top, 0)^\top$. The performative matrix $D=\text{diag}(b,c)$ maps to predictive and spurious features respectively. Labels are generated as $y = x^\top \theta^*_{\text{pop}} + x^\top D\theta + w$, with $w\sim\mathcal{N}(0,\sigma^2)$.

At the $k$-th iteration of RRM, $\theta_k = \arg\min_\theta \tfrac{1}{2n}\sum_i \ell(x_i^{(k-1)}, y_i^{(k-1)};\theta) + \tfrac{\lambda}{2}\|\theta\|_2^2$, where data is sampled from $\mathcal{D}(\theta_{k-1})$. The risk is evaluated on the initial distribution $\mathcal{D}(\theta=0)$, with excess risk defined as $\mathcal{R}(\Sigma,\theta,\theta^*_{\text{pop}}) = \|\Sigma^{1/2}(\theta-\theta^*_{\text{pop}})\|_2^2$.

In the population setting, RRM converges to the fixed point $\theta^\infty = (I_p + \lambda\Sigma^{-1} - D)^{-1}\theta^*_{\text{pop}}$. In the overparameterized setting with finite data, "deterministic equivalents" are derived using high-dimensional random matrix theory.

### Key Designs

**1. Population Limit: Optimal $\lambda$ is proportional to average performative strength $\bar b$**

In the $n\to\infty$ population case, the authors derive a practical formula for selecting $\lambda$. By setting $F = D - \lambda\Sigma^{-1}$ and performing a second-order Taylor expansion of the excess risk around $F$, the dominant term is:

$$\widetilde{\mathcal{R}}_{\text{pop}}(D,\lambda,\Sigma) = \tfrac{1}{d}\text{Tr}[\text{diag}(b^2)\Sigma_1] - 2\lambda\bar b + \tfrac{\lambda^2}{d}\text{Tr}(S_1),$$

where $\bar b = \tfrac{1}{d}\sum_i b_i$ is the average performative strength of predictive features and $S_1$ is the Schur complement of the covariance. This is an explicit quadratic form in $\lambda$, with the first-order condition yielding $\lambda^*_{\text{pop}} = \bar b\, d / \text{Tr}(S_1)$. Crucially, the optimal regularization depends only on the average strength $\bar b$ and covariance structure, **obviating the need for coordinate-wise estimation of $D$**. This provides an intuitive result: positive feedback ($\bar b>0$) requires stronger regularization, while "self-negating" performativity ($\bar b<0$) may require negative regularization.

**2. Overparameterized Regime: Aligned performativity can reduce risk**

In the proportional regime $p/n=\kappa>1$, population formulas fail. The authors utilize the high-dimensional risk framework (Han & Xu 2023, Ildiz et al. 2025) to perform a two-step iteration on RRM, obtaining a deterministic equivalent risk $\mathcal{R}_{\text{eq}}(\Sigma,\theta^*_{\text{pop}},D,\lambda)$. This depends on an auxiliary scalar $\tau$ determined by the fixed-point equation $\kappa^{-1} - \lambda/\tau = \tfrac{1}{p}\text{Tr}[(\Sigma+\tau I_p)^{-1}\Sigma]$. Expanding the risk to the first order of $\bar b$ and $\bar c$ reveals that the signs of four auxiliary functions $B_1,B_2,C_1,C_2$ determine whether performativity is beneficial. A key finding is $B_2(\kappa,\sigma)\le 0$: if performativity amplifies existing trends ($\bar b>0$), the risk under optimal regularization is **actually lower than in the non-performative case**. This contrasts with the population case because, in overparameterized regimes where variance dominates, performativity effectively increases the signal-to-noise ratio.

**3. Noise levels flip the direction of regularization**

The direction of optimal regularization depends on noise levels. Analyzing $B_1(\sigma,\kappa)$ reveals a critical noise threshold $\sigma_{B_1}^2(\kappa) = 1/2 - 7\kappa^{-1}/18 + O(\kappa^{-2})$. For low noise ($\sigma < \sigma_{B_1}$), $B_1\ge 0$ and the optimal $\lambda$ moves in the same direction as $\bar b$. For high noise ($\sigma > \sigma_{B_1}$), $B_1\le 0$ and the direction flips. For performativity on spurious features $\bar c$, the optimal $\lambda$ is always opposite to $\bar c$ when $\kappa\ge 2$, though the impact is suppressed by the correlation coefficient $\rho^2$ between spurious and predictive features.

### Loss & Training
The study utilizes square loss $\ell(x,y;\theta) = (y - x^\top\theta)^2$ with ridge regularization $\tfrac{\lambda}{2}\|\theta\|_2^2$. Theoretical analysis does not require hyperparameter training. In experiments, 4–5 rounds of RRM suffice to approach the fixed point, consistent with theoretical predictions.

## Key Experimental Results

### Main Results: Synthetic + Real Datasets

| Setting | Data | Key Findings | Theoretical Consistency |
|------|------|---------|------------|
| Population regime, $d=100, \Sigma=I_p$ | Synthetic | $\lambda^*$ increases linearly with $\bar b$; $\lambda^*<0$ when $\bar b<0$ | Corollary 4.2 |
| Proportional regime, low noise $\kappa=1.1, \sigma=0.2$ | Synthetic | $\lambda^*$ is larger at $\bar b=0.2$ vs $\bar b=0$, but risk is **lower** | Theorem 5.2 ($B_2\le 0$) |
| Proportional regime, high noise $\kappa=1.1, \sigma=0.7$ | Synthetic | $\bar b=0.2$ decreases $\lambda^*$, risk still decreases | $B_1\le 0$ flip |
| Housing ($n=4000, d=8$) | Real | $\lambda^*$ increases with $\bar b$, risk worsens with $\bar b$ | Population behavior |
| LSAC ($n=4000, d=22$) | Real | Same as above | Population behavior |
| LSAC ($n=100, d=22$) | Real | $\lambda^*$ decreases with $\bar b$, risk improves with $\bar b$ | Overparameterized high-noise prediction |

### Ablation Study: Alternative Regularization

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Ridge (Primary) | $\lambda^*\propto\bar b$ | Main target of theory |
| Dropout | Similar trend | Relationship is independent of specific norm |
| Lasso | Similar trend | Qualitative conclusions hold under $\ell_1$ |
| Elastic Net | Similar trend | Consistent across mixed regularization |
| Neural Net + GiveMeSomeCredit (Mofakhami 2023) | $\ell_2$ mitigates accuracy drop from increased $\delta$; optimal $\lambda$ increases with $\delta$ | Qualitative conclusions extend to non-linear models |

### Key Findings
- **Crucial Conclusion**: Overparameterization combined with performativity in the signal direction ($\bar b>0$) can lead to an optimal risk that is **better** than the non-performative baseline, challenging the intuition that performativity is always detrimental.
- **Minimal impact of spurious performativity $\bar c$**: Its effect is scaled by $\rho^2$ in the risk expansion and is negligible in practice, suggesting that performative modeling of spurious features can often be ignored.
- **Sample size dictates the relevant curve**: On the same dataset (LSAC), $n\gg d$ follows population logic (increase regularization), while $n \approx d$ shifts to overparameterized high-noise logic.
- **Weak requirements for RRM convergence**: Two iterations are sufficient to reach the fixed point (to the first order), aligning well with industrial settings where deployment rounds are limited.

## Highlights & Insights
- **First application of high-dimensional statistics to performative learning**: Merging the deterministic equivalent framework (Han & Xu 2023) with RRM fixed points provides a new analytical path applicable to performative classification and robustness.
- **Practical rule for optimal $\lambda$**: Since the rule only requires estimating a scalar $\bar b$ rather than the full matrix $D$, the method remains functional in high dimensions.
- **Physical interpretation of negative regularization**: Provides a new explanation for negative $\lambda$—it serves to counteract "self-negating" performativity.
- **Honest presentation of negative results**: The authors explicitly note that the impact of spurious performative strength $c$ was not observed in experiments, avoiding overstatement of its contribution.

## Limitations & Future Work
- **Strong linear assumptions**: Analysis relies on labels being linear in $\theta$ and features being Gaussian/sub-Gaussian; real-world performativity is likely non-linear.
- **Label shift focus**: Does not explicitly cover feature shift, though the authors suggest some feature shifts can be absorbed via centering.
- **Optimization scope**: Focuses on explicit regularization without exploring implicit methods like early stopping or data augmentation.
- **Evaluation distribution**: Choosing between $\mathcal{D}(0)$ and $\mathcal{D}(\theta)$ for evaluation remains a dilemma between ignoring social evolution and encouraging distribution manipulation.

## Related Work & Insights
- **vs Perdomo et al. (2020)**: Extends their RRM convergence proof in strongly convex settings to scaling laws for optimal regularization in overparameterized regimes.
- **vs Cyffers et al. (2024)**: Provides theoretical proofs in regression for their numerical evidence in classification that "performative optimal ≈ regularized non-performative optimal."
- **vs Hastie et al. (2022) / Patil et al. (2024)**: While they explore ridge optimality in OOD regression, this work demonstrates that negative $\lambda$ can arise from performativity even in population settings.
- **vs Bombari & Mondelli (2025)**: Complements their concern about high-dimensional regularization exacerbating spurious feature dependence by showing that optimal $\lambda$ directions for spurious features may be reversed.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to use high-dimensional statistical tools in performative learning; discovered that performativity can improve risk.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered synthetic, Housing, LSAC, and neural networks, though real performative datasets remain scarce.
- Writing Quality: ⭐⭐⭐⭐⭐ High correspondence between theorems and figures; clear explanation of intuition and formalisms.
- Value: ⭐⭐⭐⭐ Provides actionable rules for selecting $\lambda$, useful for practitioners in strategic classification and recommendation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Tight Lower Bounds and Improved Convergence in Performative Prediction](../../NeurIPS2025/others/tight_lower_bounds_and_improved_convergence_in_performative_prediction.md)
- [\[ICML 2025\] Revisiting the Predictability of Performative, Social Events](../../ICML2025/others/revisiting_the_predictability_of_performative_social_events.md)
- [\[ICLR 2026\] Spurious Correlation-Aware Embedding Regularization for Worst-Group Robustness](../../ICLR2026/others/spurious_correlation-aware_embedding_regularization_for_worst-group_robustness.md)
- [\[ICML 2026\] Guaranteed Optimal Compositional Explanations for Neurons](guaranteed_optimal_compositional_explanations_for_neurons.md)
- [\[ICML 2025\] Cross-regularization: Adaptive Model Complexity through Validation Gradients](../../ICML2025/others/cross-regularization_adaptive_model_complexity_through_validation_gradients.md)

</div>

<!-- RELATED:END -->
