---
title: >-
  [Paper Note] Efficient Adaptive Experimentation with Noncompliance
description: >-
  [NeurIPS 2025][Optimization][adaptive experimentation] This paper proposes AMRIV — the first semiparametrically efficient, multiply robust ATE estimator for adaptive experiments with noncompliance — combined with a varia…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "adaptive experimentation"
  - "instrumental variables"
  - "noncompliance"
  - "semiparametric efficiency"
  - "causal inference"
date: 2026-05-08
content_hash: 21f4ef89a8687a1e
---

# Efficient Adaptive Experimentation with Noncompliance

**Conference**: NeurIPS 2025
**arXiv**: [2505.17468](https://arxiv.org/abs/2505.17468)
**Code**: [GitHub](https://github.com/CausalML/Adaptive-IV)
**Area**: Optimization
**Keywords**: adaptive experimentation, instrumental variables, noncompliance, semiparametric efficiency, causal inference

## TL;DR

This paper proposes AMRIV — the first semiparametrically efficient, multiply robust ATE estimator for adaptive experiments with noncompliance — combined with a variance-optimal instrumental variable allocation strategy and sequential inference guarantees.

## Background & Motivation

**Background**: Adaptive experimentation efficiently estimates treatment effects by adjusting allocation strategies based on accumulated data, and has been formally recognized by the FDA. Adaptive ATE estimation with direct treatment assignment is already supported by mature tools (A2IPW, Neyman allocation, etc.).

**Limitations of Prior Work**: In many real-world settings, **treatment cannot be directly assigned** and can only be **encouraged via instrumental variables (IVs)**. Examples include:
   - TripAdvisor experiments: the registration interface can be randomized (IV), but whether a user subscribes (treatment) is voluntary.
   - Clinical trials: a physician can recommend a drug (IV), but patient adherence (treatment) is uncontrolled.
   - **Noncompliance** causes misalignment between treatment and IV, leading to bias in conventional methods.

**Key Challenge**:
   - Theory for adaptive experimentation is mature in **direct assignment** settings, but nearly absent when **only the IV is assignable and treatment is endogenous**.
   - Existing IV methods (DeepIV, MRIV, etc.) are either non-adaptive or do not pursue semiparametric efficiency.

**Goal**: To bring the full modern semiparametric toolkit — efficient influence functions, adaptive policy learning, robust imputation estimation, and anytime-valid inference — into the adaptive IV setting with noncompliance.

**Key Insight**: Building on the unconfounded compliance assumption and multiply robust influence functions of Wang & Tchetgen Tchetgen, extended to the adaptive setting.

**Core Idea**: Derive the semiparametric efficiency bound and optimal allocation strategy for adaptive experiments with IVs, and construct a sequential estimator that achieves this bound.

## Method

### Overall Architecture

**Problem Setup**: A $T$-round sequential experiment where at each round the covariates $X_t$ are observed, an instrumental variable $Z_t \sim \pi_t(\cdot|X_t, \mathcal{H}_{t-1})$ is assigned, and the treatment $A_t = A_t(Z_t)$ and outcome $Y_t$ are observed. The goal is to estimate the ATE $\tau = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]$.

**Key Assumptions**:
- **Assumption 1 (Standard IV)**: Exclusion restriction, independence ($Z \perp U|X$), and relevance ($\text{Cov}(Z,A|X) \neq 0$).
- **Assumption 2 (Unconfounded Compliance)**: $Y(1) - Y(0) \perp A(1) - A(0) | X$.

**ATE Identification**: $\tau = \mathbb{E}_X\left[\frac{\delta^Y(X)}{\delta^A(X)}\right]$, where $\delta^Y(X)$ and $\delta^A(X)$ denote the IV-induced changes in outcome and treatment, respectively.

### Key Designs

#### 1. Semiparametric Efficiency Bound (Theorem 1)

$$V_{\text{eff}}(\pi) = \mathbb{E}\left[\frac{1}{\delta^A(X)^2}\left(\frac{\sigma^2(1,X)}{\pi(X)} + \frac{\sigma^2(0,X)}{1-\pi(X)}\right) + (\delta(X) - \tau)^2\right]$$

where the residual variance is $\sigma^2(z,X) = \text{Var}(Y - A\delta(X) | Z=z, X)$.

#### 2. Optimal Instrumental Variable Allocation (Corollary 2)

$$\pi^*(X) = \frac{\sqrt{\sigma^2(1,X)}}{\sqrt{\sigma^2(1,X)} + \sqrt{\sigma^2(0,X)}}$$

**Key Insights**:
- The optimal policy tilts toward the arm with greater residual variance.
- The residual variance **depends jointly on outcome noise and compliance noise** — unlike the Neyman allocation in the standard ATE setting.
- When $\delta^A(X) \to 1$ (perfect compliance), it reduces to the classical Neyman allocation.
- When $\delta^A(X) \to 0$ (low compliance), it approaches uniform allocation.

#### 3. AMRIV Estimator

$$\hat{\tau}_T^{\text{AMRIV}} = \frac{1}{T}\sum_{t=1}^T \phi(X_t, Z_t, A_t, Y_t; \pi_t, \hat{\eta}_t)$$

The influence function $\phi$ is a recentered efficient influence function based on the adaptive policy $\pi_t$ and sequentially estimated nuisance parameters $\hat{\eta}_t$:

$$\phi = \frac{2Z-1}{Z\pi(X) + (1-Z)(1-\pi(X))} \cdot \frac{1}{\delta^A(X)} [Y - A\delta(X) - \mu^Y(0,X) + \mu^A(0,X)\delta(X)] + \delta(X)$$

**Key Property**: All nuisance estimates use only historical data $\mathcal{H}_{t-1}$, ensuring a near-martingale structure.

#### 4. Algorithm Components

- **Burn-in Phase**: Fixed policy $\pi_{\text{init}}$ (e.g., uniform randomization) for $T_0$ rounds.
- **Adaptive Phase**: Plug-in optimal policy $\tilde{\pi}_t$ with clipping $\pi_t = \text{clip}(\tilde{\pi}_t, 1/k_t, 1-1/k_t)$.
- **Residual Variance Estimation**: Two-stage cross-fitting to remove finite-sample bias.
- **Nuisance Learners**: Any nonparametric regression method (k-NN, random forests, neural networks) is applicable.

### Theoretical Guarantees

#### Theorem 3 (Asymptotic Normality)

$$\sqrt{T}(\hat{\tau}_T^{\text{AMRIV}} - \tau) \xrightarrow{d} \mathcal{N}(0, V_{\text{eff}}(\pi))$$

Semiparametric efficiency is achieved when $\pi = \pi^*$. Only $L_2$ consistency is required (no Donsker conditions needed).

#### Theorem 4 (Convergence Rate)

$$|\hat{\tau}_T^{\text{AMRIV}} - \tau| = O_p(T^{-1/2}) + O_p(\|\hat{\delta}_T^A - \delta^A\|_2 \cdot \|\hat{\delta}_T - \delta\|_2)$$

#### Corollary 5 (Multiple Robustness)

AMRIV is consistent as long as either $\hat{\delta}$ or $\hat{\delta}^A$ is $L_2$-consistent. This robustness is stronger than that of static MRIV, since the adaptive control of $\pi_t$ provides additional robustness against misspecification of $\mu^Y$ and $\mu^A$.

## Key Experimental Results

### Synthetic Experiments ($T=2000$, 1000 trajectories)

**One-sided noncompliance**: $\mu^A(0,X) = 0$, compliance rate $\delta^A(x) = \sigma(-2x)$

| Metric | AMRIV | AMRIV-NA | DM | DM-NA | A2IPW | Oracle |
|--------|-------|----------|----|-------|-------|--------|
| Efficiency (Norm. MSE) | Near Oracle | Constant gap | Grows with $T$ | Grows with $T$ | Biased | 1.0 |
| Consistency | ✓ Converges | ✓ Converges | ✓ | ✓ | ✗ Does not converge | ✓ |
| 95% CI Coverage | **Nominal level** | **Nominal level** | Under-covers | Under-covers | Severely under-covers | — |

### Key Findings

1. **AMRIV approaches the Oracle baseline** (which uses true nuisance parameters); the adaptive version substantially outperforms its non-adaptive counterpart.
2. **A2IPW is biased and fails to converge** — because it does not correct for unobserved confounding in treatment selection.
3. **AMRIV-MS (misspecified version) remains consistent** but achieves slightly below-nominal coverage.
4. **DM methods diverge under misspecification of $\delta$**, whereas AMRIV-MS continues to converge — demonstrating multiple robustness.
5. Adaptive design is **especially beneficial in low-compliance regions**: more allocation is directed to $Z=1$ to compensate for sparse treatment uptake.

### Semi-Synthetic Data (TripAdvisor)

Results are consistent with the synthetic experiments: adaptive IV allocation improves efficiency, and AMRIV achieves optimal coverage and consistency.

## Highlights & Insights

- **Fills an important gap**: First work to bring the complete semiparametric adaptive experimentation toolkit to the IV/noncompliance setting.
- **Fine-grained analysis of the efficiency bound**: Reveals the non-trivial structure by which the optimal allocation simultaneously balances outcome variance and compliance variance.
- **Elegant generalization of Neyman allocation**: Perfect compliance → classical Neyman; low compliance → approaches uniform.
- **Stronger multiple robustness than static methods**: Adaptive control of $\pi_t$ provides an additional dimension of robustness.
- **Anytime-valid inference**: Supports sequential stopping decisions (via asymptotic confidence sequences).

## Limitations & Future Work

1. **Assumption 2 (unconfounded compliance) is strong**: It may not hold in practice; in such cases, the quantity estimated is ACLATE rather than ATE.
2. **Computational cost**: All nuisance estimates must be updated each round (though mini-batch updates are feasible).
3. **Restricted to binary IV and binary treatment**: Extension to multi-valued settings is non-trivial.
4. **Boundedness assumption (Assumption 3)** may not hold under heavy-tailed distributions.
5. **Selection of the clipping parameter $k_t$** lacks systematic guidance.

## Related Work & Insights

- Generalizes the adaptive ATE methods of A2IPW (Kato et al.) and Cook et al. to the IV setting.
- Builds on the static semiparametric framework of MRIV (Wang & Tchetgen Tchetgen).
- **Core Insight**: In settings where treatment cannot be directly controlled — which is extremely common in medicine and social science — adaptively allocating encouragement/instruments can substantially improve the efficiency of causal effect estimation.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The problem motivation is compelling (noncompliance is a pervasive practical challenge), the theoretical contributions are complete (efficiency bound + optimal policy + convergence rate + multiple robustness + anytime-valid inference), and the experiments, though simulation-based, provide thorough validation. Limitations include the strong compliance assumption and restriction to binary settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[NeurIPS 2025\] Oracle-Efficient Combinatorial Semi-Bandits](oracle-efficient_combinatorial_semi-bandits.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[NeurIPS 2025\] DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization](dartquant_efficient_rotational_distribution_calibration_for_llm_quantization.md)
- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)

</div>

<!-- RELATED:END -->
