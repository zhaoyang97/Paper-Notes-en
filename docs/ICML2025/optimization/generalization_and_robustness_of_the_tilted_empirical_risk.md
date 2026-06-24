---
title: >-
  [Paper Note] Generalization and Robustness of the Tilted Empirical Risk
description: >-
  [ICML2025][Optimization][tilted empirical risk] This paper provides systematic generalization error bounds and robustness guarantees for Tilted Empirical Risk (TER) under negative tilt parameters ($\gamma < 0$). Under the condition of unbounded loss functions with bounded $(1+\epsilon)$-th order moments, it establishes a convergence rate of $O(n^{-\epsilon/(1+\epsilon)})$ using uniform and information-theoretic approaches, and presents a data-driven scheme for selecting the t…
tags:
  - "ICML2025"
  - "Optimization"
  - "tilted empirical risk"
  - "generalization error"
  - "robustness"
  - "information-theoretic bounds"
  - "unbounded loss functions"
  - "KL regularization"
date: 2026-05-08
content_hash: 8cc1cf1d3707a049
---

# Generalization and Robustness of the Tilted Empirical Risk

**Conference**: ICML2025  
**arXiv**: [2409.19431](https://arxiv.org/abs/2409.19431)  
**Code**: None  
**Area**: Optimization  
**Keywords**: tilted empirical risk, generalization error, robustness, information-theoretic bounds, unbounded loss functions, KL regularization

## TL;DR

This paper provides systematic generalization error bounds and robustness guarantees for Tilted Empirical Risk (TER) under negative tilt parameters ($\gamma < 0$). Under the condition of unbounded loss functions with bounded $(1+\epsilon)$-th order moments, it establishes a convergence rate of $O(n^{-\epsilon/(1+\epsilon)})$ using uniform and information-theoretic approaches, and presents a data-driven scheme for selecting the tilt parameter.

## Background & Motivation

- **Tilted Empirical Risk (TER)**, proposed by Li et al. (2021), is a non-linear risk metric: $\hat{R}_\gamma(h,S) = \frac{1}{\gamma}\log\left(\frac{1}{n}\sum_{i=1}^n \exp(\gamma \ell(h,Z_i))\right)$, which degenerates to standard empirical risk as $\gamma \to 0$.
- Empirical studies have shown that TER under **negative tilt ($\gamma < 0$)** is effective at resisting outliers/noisy labels, whereas under **positive tilt ($\gamma > 0$)** it can handle class imbalance and fairness constraints.
- **Theoretical Gap**: Existing generalization analyses are either restricted to bounded loss functions (Lee et al. 2020) or assume that both the loss and its derivatives are bounded (Wang et al. 2023), failing to cover common unbounded loss scenarios such as regression.
- Existing generalization bounds for linear empirical risk under unbounded loss (Cortes et al. 2019) exhibit a convergence rate of $O(\log(n) \cdot n^{-\epsilon/(1+\epsilon)})$. This work aims to eliminate the $\log(n)$ factor and generalize it to TER.

## Method

### Overall Architecture

This work defines the **tilted generalization error** as the difference between the population risk and the TER:

$$\text{gen}_\gamma(h,S) := R(h,\mu) - \hat{R}_\gamma(h,S)$$

The core mechanism is to decompose this term into three parts for individual bounding: (1) the difference $I_1$ between the population risk and the tilted population risk; (2) the non-linear generalization error $\hat{\text{gen}}_\gamma$; (3) the distribution shift term.

### Key Design 1: Uniform Bounds (Section 3.1)

**Assumption**: The $(1+\epsilon)$-th order moment of the loss is uniformly bounded, i.e., $\mathbb{E}_\mu[\ell^{1+\epsilon}(h,Z)] \le \kappa_u^{1+\epsilon}$, where $\epsilon \in (0,1]$.

- **Proposition 3.2** (Bound on $I_1$): $0 \le R(h,\mu) - R_\gamma(h,\mu^{\otimes n}) \le |\gamma|^\epsilon \kappa_u^{1+\epsilon}$
- **Proposition 3.3/3.4**: Leverages Bernstein's inequality and properties of logarithmic functions to derive upper and lower bounds on the tilted generalization error.
- **Theorem 3.5** (Main Theorem): For a finite hypothesis space, with probability $\ge 1-\delta$:

$$\sup_{h \in \mathcal{H}} |\text{gen}_\gamma(h,S)| \le \frac{2e^{|\gamma|\kappa_u}}{(1-\zeta)|\gamma|}\sqrt{\frac{2^\epsilon |\gamma|^{1+\epsilon}\kappa_u^{1+\epsilon} B(\delta)}{n}} + \frac{4e^{|\gamma|\kappa_u}B(\delta)}{3n|\gamma|(1-\zeta)} + |\gamma|^\epsilon \kappa_u^{1+\epsilon}$$

where $B(\delta) = \log(\text{card}(\mathcal{H})) + \log(2/\delta)$. When $\gamma \asymp n^{-1/(1+\epsilon)}$, the convergence rate is $O(n^{-\epsilon/(1+\epsilon)})$.

### Key Design 2: Information-Theoretic Bounds (Section 3.2)

Relaxes the uniform assumption to algorithm-dependent moment conditions (Assumption 3.8):

- **Theorem 3.11**: The absolute value of the expected tilted generalization error satisfies:

$$|\bar{\text{gen}}_\gamma(H,S)| \le D(\gamma)\sqrt{\frac{2\kappa_t^{1+\epsilon}|\gamma|^{1+\epsilon} I(H;S)}{n}} + |\gamma|^\epsilon \kappa_t^{1+\epsilon}$$

where $D(\gamma) = \frac{e^{|\gamma|\kappa_t}}{|\gamma|(1-\zeta)}$, and $I(H;S)$ is the mutual information between the hypothesis and the training set.

### Key Design 3: Robustness under Distribution Shift (Section 4)

Models the shift between the training distribution $\tilde{\mu}$ (introduced by noise/outliers) and the true distribution $\mu$:

- **Proposition 4.2**: The difference in tilted population risk between two distributions can be bounded by the total variation distance $\mathbb{TV}(\mu, \tilde{\mu})$.
- **Theorem 4.3**: Under distribution shift, the absolute tilted generalization error incurs an additional term: $\frac{\mathbb{TV}(\mu,\tilde{\mu})}{\gamma^2} \cdot \frac{e^{|\gamma|\kappa_u} - e^{|\gamma|\kappa_s}}{\kappa_u - \kappa_s}$.
- **Core Insight**: Distinct from ERM, TER under negative tilt can yield finite bounds using TV distance even for unbounded losses, whereas ERM requires KL divergence and may diverge.

### Key Design 4: KL-Regularized TERM (Section 6)

- The optimal solution is the **tilted Gibbs posterior**: $P_{H|S}^\gamma \propto \pi_H \cdot \left(\frac{1}{n}\sum_i e^{\gamma \ell(H,Z_i)}\right)^{-\alpha/\gamma}$.
- Degenerates to the standard Gibbs posterior as $\gamma \to 0$.
- **Theorem 6.3**: Achieves a faster convergence rate of $O(n^{-\epsilon})$ when $\gamma = O(1/n)$ compared to the unregularized version.

### Data-Driven Selection of the Tilt Parameter (Section 5)

The optimal tilt parameter is chosen by minimizing the terms dependent on $\gamma$ in the upper bound of Theorem 4.3:

$$\gamma_{\text{data}} = \arg\min_{\gamma < 0}\left[|\gamma|^\epsilon \kappa_u^{1+\epsilon} + \frac{\mathbb{TV}(\mu,\tilde{\mu})}{\gamma^2}\cdot \frac{e^{|\gamma|\kappa_u} - e^{|\gamma|\kappa_s}}{\kappa_u - \kappa_s}\right]$$

## Key Experimental Results

### Logistic Regression + Gaussian Outliers (Table 1, n=1000)

| Outlier Ratio ρ | γ* (Grid Search) | R(TERM) | R(ERM) | γ_data | R (Data-Driven) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1% | -0.53 | 0.00±0.000 | 0.05±0.001 | -1.40 | 0.00±0.000 |
| 17.6% | -2.98 | 0.15±0.004 | 0.22±0.001 | -4.91 | 0.16±0.003 |
| 35.0% | -3.86 | 0.16±0.004 | 0.30±0.002 | -3.33 | 0.20±0.002 |
| 52.5% | -2.10 | 0.11±0.001 | 0.28±0.002 | -1.93 | 0.14±0.001 |
| 70.0% | -1.23 | 0.14±0.002 | 0.18±0.000 | -2.28 | 0.15±0.002 |

### Logistic Regression + Pareto Outliers (Table 2, n=1000)

| Outlier Ratio ρ | γ* (Grid Search) | R(TERM) | R(ERM) | γ_data | R (Data-Driven) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1% | -1.40 | 0.00±0.000 | 0.03±0.001 | -0.70 | 0.01±0.000 |
| 17.58% | -3.33 | 0.00±0.003 | 0.02±0.000 | -0.88 | 0.01±0.000 |
| 35.05% | -1.05 | 0.00±0.000 | 0.01±0.002 | -0.70 | 0.01±0.000 |
| 52.53% | -1.05 | 0.01±0.000 | 0.01±0.002 | -1.06 | 0.01±0.001 |
| 70.0% | -0.88 | 0.00±0.002 | 0.02±0.001 | -0.70 | 0.01±0.000 |

### Key Findings

- **TERM Significantly Outperforms ERM**: Especially under high outlier ratios, the population risk of ERM is substantially higher than that of TERM (e.g., for Gaussian outliers with $\rho=35\%$, ERM = 0.30 vs. TERM = 0.16).
- **Effective Data-Driven Selection**: The performance of $\gamma_{\text{data}}$ is close to the grid-search optimal $\gamma^*$, with smaller variance.
- **More Favorable under Pareto Outliers**: Under Pareto noise with unbounded second moments, the advantages of TERM and data-driven methods are even more pronounced.
- **Robustness-Generalization Trade-off**: Excessively large $|\gamma|$ enhances robustness but deteriorates generalization, indicating the existence of an optimal point.

## Highlights & Insights

1. **First to Establish a Complete Generalization Theory for TER under Unbounded Losses**: Bridges the gap between the empirical success of TER and its theoretical guarantees.
2. **Improved Convergence Rate**: Eliminates the $\log(n)$ factor present in Cortes et al. (2019), achieving $O(n^{-\epsilon/(1+\epsilon)})$.
3. **Theoretical Characterization of Robustness**: First to prove that negative-tilted TER possesses finite-bounded robustness to distribution shifts under the TV distance metric, which ERM fails to guarantee.
4. **The Tilted Gibbs Posterior** is a Novel Concept: It is the optimal solution for KL-regularized TERM, which degenerates to the standard Gibbs posterior as $\gamma \to 0$ and delivers a faster $O(n^{-\epsilon})$ convergence rate.
5. **High Practicality of Data-Driven Tilt Selection**: Optimizing based on the theoretical upper bound yields a near-optimal $\gamma$.

## Limitations & Future Work

1. **Only Negative Tilt ($\gamma < 0$) is Analyzed**: The generalization analysis for positive tilt (handling class imbalance/fairness) remains unaddressed.
2. **Simple Experimental Setup**: Validations are restricted to logistic/linear regression, lacking experiments in complex scenarios such as deep learning.
3. **Finite Hypothesis Space Assumption**: The uniform bounds require a finite $\text{card}(\mathcal{H})$; though generalizable via $\epsilon$-nets / VC dimension, the tightness in practice has not been explored.
4. **Mutual Information $I(H;S)$ Must Be Bounded**: The utility of the information-theoretic bounds relies on whether the mutual info can be effectively estimated or controlled.
5. **Difficulty in Estimating TV Distance in High Dimensions**: The data-driven method requires estimating the TV distance and moment parameters, which poses practical challenges.
6. **Lack of Comparison with Other Robust Optimization Methods**: No empirical comparison is made against mainstream robust methods such as Distributionally Robust Optimization (DRO).

## Related Work & Insights

- **Li et al. (2021, 2023a)**: Introduction and application of TER (class imbalance, outliers/noise, fairness).
- **Cortes et al. (2019)**: VC-dimension generalization bounds for unbounded losses, with a convergence of $O(\log(n)n^{-\epsilon/(1+\epsilon)})$, which this paper improves.
- **Xu & Raginsky (2017)**: Information-theoretic generalization bounds framework, which this work extends to non-linear empirical risk.
- **Haddouche & Guedj (2022)**: PAC-Bayes bounds under bounded second moments, for which this work provides a complementary perspective.
- **Behnamnia et al. (2025)**: Parallel work on TER in off-policy learning.
- **Insights**: The theoretical advantages (robustness guarantees) of TER could potentially guide applications in areas like adversarial training and robust federated learning.

## Rating

- Novelty: ⭐⭐⭐⭐ (First to establish complete generalization/robustness theory for TER under unbounded losses)
- Experimental Thoroughness: ⭐⭐ (Experiments are overly simple, restricted only to linear models)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous theoretical derivations and clear structure)
- Value: ⭐⭐⭐⭐ (Provides significant contributions to the theoretical foundation of TERM, though practical impact remains to be validated)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](tilted_sharpness-aware_minimization.md)
- [\[ICML 2025\] SAFER: A Calibrated Risk-Aware Multimodal Recommendation Model for Dynamic Treatment Regimes](safer_a_calibrated_risk-aware_multimodal_recommendation_model_for_dynamic_treatm.md)
- [\[ICML 2025\] A Generalization Result for Convergence in Learning-to-Optimize](a_generalization_result_for_convergence_in_learning-to-optimize.md)
- [\[NeurIPS 2025\] MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](../../NeurIPS2025/optimization/mecefo_enhancing_llm_training_robustness_via_fault-tolerant_optimization.md)
- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](../../ICLR2026/optimization/elastic_optimal_transport_theory_application_and_empirical_evaluation.md)

</div>

<!-- RELATED:END -->
