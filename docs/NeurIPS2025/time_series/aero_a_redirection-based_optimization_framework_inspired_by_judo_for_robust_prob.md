---
title: >-
  [Paper Note] AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting
description: >-
  [NeurIPS 2025][Time Series][Gradient redirection] AERO proposes an optimization paradigm inspired by the judo principle of "redirecting force rather than resisting it…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Gradient redirection"
  - "adversarial optimization"
  - "energy conservation"
  - "probabilistic forecasting"
  - "quantile regression"
date: 2026-05-08
content_hash: 2a9fe5eb60aa9931
---

# AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2506.02415](https://arxiv.org/abs/2506.02415)  
**Code**: None  
**Area**: Optimization / Time Series Forecasting
**Keywords**: Gradient redirection, adversarial optimization, energy conservation, probabilistic forecasting, quantile regression

## TL;DR
AERO proposes an optimization paradigm inspired by the judo principle of "redirecting force rather than resisting it," attempting to redirect adversarial perturbations into beneficial optimization signals. The framework is theoretically grounded in 15 axioms and 4 theorems, constructing an energy-conservation-based gradient redirection system. However, the actual implementation is substantially simplified to momentum SGD with Gaussian noise injection, and validation is conducted solely on a single private solar energy price prediction dataset without any baseline comparisons.

## Background & Motivation

Traditional optimizers (SGD, Adam, etc.) tend to become unstable in nonlinear dynamical systems characterized by high noise and uncertainty. Adversarial optimization methods (e.g., PGD, minimax training) enhance robustness by "resisting" worst-case perturbations, but this resistance itself can introduce instability. Improvements such as SAM and Lookahead enhance generalization and stability but remain primarily heuristic and lack a unified theoretical framework.

AERO's starting point draws from judo philosophy—"do not resist force; instead, use the opponent's strength by redirecting it." Mapped to optimization, perturbations (adversarial or noisy gradients) should not be eliminated but projected onto beneficial directions to assist optimization. The paper attempts to provide a theoretical foundation for this redirection-based optimization paradigm from the perspectives of physics (energy conservation, momentum transfer) and multi-agent cooperation.

## Method

### Overall Architecture
At the theoretical level, AERO comprises 15 "redirection axioms" and 4 derived theorems, organized into four modules: (1) Core Redirection Dynamics (Axioms A1–A5)—defining how perturbations are redirected via rotation matrices; (2) Adaptivity and Context-Sensitivity (A6–A10)—defining time-varying adaptive redirection strategies; (3) System Dynamics and Conservation (A11–A13)—energy conservation constraints and controlled instability; (4) Multi-Agent Cooperation (A14–A15)—cross-quantile gradient collaboration.

### Key Designs

1. **Redirected Gradient Computation (Theoretical Level)**:
    - Function: Extracts useful components from adversarial/noisy gradients for optimization updates.
    - Mechanism: $R_t^{(q)} = \text{proj}_{G_t^{(q)}}(G_{\text{adv}}^{(q)} + \delta_{t+1}^{(q)}) + \sum_{j \neq q} \beta_{qj} G_t^{(j)}$. The first term projects the adversarial gradient plus predictive perturbation onto the true gradient direction; the second term provides cross-quantile collaboration signals.
    - Design Motivation: Projection retains components of the perturbation that align with the optimization direction ("borrowing force") while discarding perpendicular and opposing components ("deflecting force"). Cross-quantile collaboration allows learning signals from different quantiles to mutually reinforce one another.

2. **Energy Conservation and Momentum Redistribution**:
    - Function: Prevents optimization updates from becoming excessively large or small, maintaining training stability.
    - Mechanism: $E_{\text{learn}}^{(q)} = \lambda \|R_t^{(q)}\|^2 + (1-\lambda)\|G_t^{(q)}\|^2$, where $\lambda$ balances energy allocation between the redirected and original gradients. Momentum satisfies approximate conservation across quantiles: $\sum_q v_t^{(q)} \approx \text{const}$.
    - Design Motivation: Inspired by energy conservation in physics, this prevents learning energy from becoming overly concentrated in certain quantiles.

3. **Anticipatory Dynamics**:
    - Function: Predicts future perturbations to proactively adjust the optimization direction.
    - Mechanism: $\delta_{t+1}^{(q)} = \text{PredictiveVariance}(x_t, \theta_t^{(q)})$, estimating output variance using current inputs and parameters as the anticipated perturbation.
    - Design Motivation: Analogous to "anticipating the opponent's next move" in judo, enabling preemptive adjustment of strategy.

### Actual Implementation (AERO-Shared)
The actual implementation diverges substantially from the theoretical framework. The update rule simplifies to:

$$g' = \nabla\mathcal{L} + \beta \cdot \mathcal{N}(0, I),\quad m_t = \mu \cdot m_{t-1} + (1-\mu) \cdot g',\quad \theta_{t+1} = \theta_t - \eta \cdot m_t$$

This is essentially **momentum SGD with Gaussian noise injection**—gradients augmented with random noise, then smoothed via exponential moving average. The projection, cross-quantile collaboration, and predictive variance components described in the theory are all absent from the implementation.

### Loss & Training
Standard quantile (pinball) loss is applied to train a QRNN model across three quantiles ($\tau \in \{0.1, 0.5, 0.9\}$). In experiments, AERO-Shared serves as a drop-in replacement for Adam.

## Key Experimental Results

### Main Results
QRNN + AERO convergence on a private 15-minute solar energy price dataset (one year of data):

| Epoch | Train Loss | Test Loss |
|:-----:|:----------:|:---------:|
| 1 | 146.78 | 142.23 |
| 10 | 1.15 | 1.11 |
| 25 | 0.21 | 0.22 |
| 50 | 0.0435 | 0.0485 |

Paired t-test: T-statistic = 1.70, p-value = 0.0955 (> 0.05), concluding no significant overfitting.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Ablation analysis | N/A | **The paper explicitly states ablation studies were "omitted for brevity"** |
| Comparison with Adam/SGD | N/A | **The paper claims AERO outperforms baselines but states "full ablation study omitted"** |
| Sensitivity to $\beta, \mu$ | N/A | Listed as future work |

### Key Findings
- The AERO-Shared optimizer achieves rapid convergence on a single dataset (loss decreasing from 146 to 0.04).
- The train–test loss gap is not statistically significant (p = 0.0955).
- **No quantitative comparison with any existing optimizer is provided**; claims of superiority over Adam/SGD are entirely unsupported by data.

## Highlights & Insights
- **Conceptually inspiring**: The optimization philosophy of "redirect rather than resist" represents an interesting shift in perspective—leveraging noise rather than fighting it in settings where noise cannot be eliminated.
- **Attempts a physics-grounded unification of optimizer design**: The 15 axioms draw on physical concepts such as energy conservation and momentum distribution, envisioning a comprehensive theoretical framework for optimization.
- **Cross-quantile collaboration is a valuable idea**: Allowing gradient signals from different quantiles to mutually reinforce one another in probabilistic forecasting is a direction genuinely worth exploring in quantile regression.

## Limitations & Future Work
- **Extremely weak empirical evaluation**: Only a single private dataset is used, with no quantitative comparison against any standard optimizer (Adam/SGD/SAM, etc.) and no ablation experiments.
- **Severe disconnect between theory and implementation**: The theoretical framework involves 15 axioms, 4 theorems, and complex components including projection, collaboration, and predictive variance; the actual implementation reduces to gradient noise injection plus momentum smoothing.
- **Theorems are overly trivial**: Theorem 1 is a standard application of KKT conditions for constrained convex optimization; Theorem 2 is standard Robbins–Monro convergence; Theorem 3 follows directly from summing $\|\rho_t\| \leq \epsilon_{\max}$—none offer theoretical insights beyond established results.
- **Axiom system lacks validation**: The 15 axioms are proposed as design guidelines, but no experiments verify the individual contribution of each to performance.
- **Increased computational cost without demonstrated benefit**: AERO introduces additional forward and backward passes, "approximately doubling training time," with no evidence that this overhead yields meaningful gains.
- **Hyperparameter sensitivity**: The paper acknowledges sensitivity to $\epsilon$, $\alpha$, and $\lambda$, but provides no tuning guidance.
- **Limited domain scope**: Testing is confined to solar energy price prediction; generalizability is entirely unverified.

## Related Work & Insights
- **vs. SAM**: SAM validates the value of flat minima search for generalization through large-scale experiments; AERO lacks comparable empirical validation.
- **vs. gradient noise injection**: AERO-Shared is implementationally equivalent to gradient perturbation methods already extensively studied (Neelakantan et al., 2015), which benefit from more rigorous theoretical analysis.
- **vs. adversarial training (PGD)**: PGD identifies worst-case perturbations by maximizing the inner loss before minimizing the outer loss—a more principled approach than AERO's stochastic noise injection.
- **Insight**: Translating physical metaphors (judo, energy conservation) into algorithms of practical value demands more rigorous methodology—the chain from axioms to algorithm to experimental validation cannot have gaps.

## Rating
- Novelty: ⭐⭐⭐ Conceptually interesting, but the 15 axioms are unwieldy and the actual innovation (gradient noise injection + momentum) is not novel.
- Experimental Thoroughness: ⭐ A single dataset, no baseline comparisons, no ablation experiments—far below the standard expected at top venues.
- Writing Quality: ⭐⭐⭐ The theoretical exposition is detailed but disproportionately lengthy; the experimental section is too sparse to balance the paper.
- Value: ⭐⭐ The concept is inspiring but lacks empirical support; in its current form, it is difficult to serve as a reliable reference baseline for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)
- [\[NeurIPS 2025\] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)
- [\[ICML 2026\] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](../../ICML2026/time_series/parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)
- [\[ICLR 2026\] From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](../../ICLR2026/time_series/from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)

</div>

<!-- RELATED:END -->
