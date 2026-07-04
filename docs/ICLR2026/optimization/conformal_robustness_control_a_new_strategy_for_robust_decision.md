---
title: >-
  [Paper Note] Conformal Robustness Control: A New Strategy for Robust Decision
description: >-
  [ICLR 2026][Optimization][Conformal Prediction] Targeting the pain point of "overly conservative coverage constraints" when using conformal prediction for robust decision-making, this paper proposes Conformal Robustness Control (CRC). It optimizes prediction set construction **directly under explicit robustness constraints** (rather than requiring set coverage). The problem is solved using smooth proxies and alternating Lagrangian gradients, with non-asymptotic theoretical gu…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Conformal Prediction"
  - "Conditional Robust Optimization"
  - "Risk-Sensitive Decision Making"
  - "Prediction Set Optimization"
  - "Finite-Sample Guarantees"
date: 2026-05-08
content_hash: e05b9a76de714f31
---

# Conformal Robustness Control: A New Strategy for Robust Decision

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bt4Ahpemmi](https://openreview.net/forum?id=bt4Ahpemmi)  
**Code**: To be confirmed  
**Area**: Robust Optimization / Conformal Prediction / Decision Theory  
**Keywords**: Conformal Prediction, Conditional Robust Optimization, Risk-Sensitive Decision Making, Prediction Set Optimization, Finite-Sample Guarantees

## TL;DR
Targeting the pain point of "overly conservative coverage constraints" when using conformal prediction for robust decision-making, this paper proposes Conformal Robustness Control (CRC). It optimizes prediction set construction **directly under explicit robustness constraints** (rather than requiring set coverage). The problem is solved using smooth proxies and alternating Lagrangian gradients, with non-asymptotic theoretical guarantees and test-time finite-sample calibration. CRC achieves lower risk certificates and decision losses in tasks like portfolio optimization, stocks, and battery energy storage while precisely hitting target robustness levels.

## Background & Motivation
**Background**: In risk-sensitive scenarios like portfolio optimization, medical diagnosis, and traffic planning, decision-makers must choose a decision $z(X)$ before the outcome $Y$ is known. The goal is to ensure the probability of the decision loss $\phi(Y, z(X))$ not exceeding a risk certificate $r(X)$ is sufficiently high, satisfying a $(1-\alpha)$ level robustness constraint: $P\{\phi(Y, z(X)) \le r(X)\} \ge 1-\alpha$, while minimizing $r(X)$ for efficiency. Conditional Robust Optimization (CRO) is the mainstream framework: first construct a prediction set $U(X)$, then solve the minmax problem $z_U(X) := \arg\min_{z\in\mathcal Z}\max_{y\in U(X)} \phi(y,z)$, with a corresponding risk certificate $r_U(X) := \max_{y\in U(X)}\phi(y, z_U(X))$.

**Limitations of Prior Work**: To satisfy robustness targets, leading methods (Johnstone & Cox 2021; Sun et al. 2023) use **conformal prediction** to construct $U(X)$ as a prediction set with coverage $P\{Y\in U(X)\} \ge 1-\alpha$, then substitute it into the minmax problem. The issue is that coverage is a **sufficient but not necessary condition** for robustness. Forcing the prediction set to "contain the true value" is often more restrictive than "guaranteeing the decision does not fail," leading to oversized prediction sets, inflated risk certificates, and overly conservative decisions. Figure 1 provides an example: CRO reached 98% robustness and 90% coverage for a nominal 90% level, with a certificate of 1.93; whereas for actual 90% robustness, only 56% coverage was needed, reducing the certificate to 1.25.

**Key Challenge**: The mapping from coverage constraints $\Rightarrow$ robustness constraints is a one-way implication. All the "excess coverage" in between is converted into unnecessary conservatism and efficiency loss.

**Goal**: To replace the "coverage" constraint with the "robustness" constraint itself—directly minimizing the expected risk certificate $E[r_U(X)]$ under the constraint $P\{\phi(Y,z_U(X))\le r_U(X)\}\ge 1-\alpha$, and addressing two consequent problems: (1) How to optimize this non-differentiable constraint containing an indicator function; (2) How to provide finite-sample robustness guarantees.

**Key Insight**: Since the actual requirement is the robustness constraint, one should bypass coverage and treat the prediction set itself as the optimization variable to be learned end-to-end under robustness constraints.

**Core Idea**: Construct prediction sets using "explicit robustness control" instead of "coverage control" (Conformal Robustness Control) to significantly improve decision efficiency without sacrificing robustness.

## Method

### Overall Architecture
The goal of CRC is to solve the Risk-Averse Decision Policy Optimization (RA-DPO) problem:
$$\min_{z(\cdot), r(\cdot)} E[r(X)] \quad \text{s.t.} \quad P\{\phi(Y, z(X)) \le r(X)\} \ge 1-\alpha.$$
Optimizing over arbitrary functions $z(\cdot)$ and $r(\cdot)$ is difficult. The CRO framework introduces a prediction set $U(\cdot)$ from which decisions and certificates are derived, rewriting the problem to optimize only over the prediction set:
$$\min_{U(\cdot)} E[r_U(X)] \quad \text{s.t.} \quad P\{\phi(Y, z_U(X)) \le r_U(X)\} \ge 1-\alpha. \tag{4}$$
A key step is the proof that this rewriting **does not lose optimality** (Theorem 3.1): the optimal expected risk certificate of RA-DPO equals the optimal value of problem (4). This provides the theoretical basis for parameterizing the prediction set.

The prediction set is then parameterized as $U_\theta(\cdot)$ (e.g., box or ellipsoid), turning the problem into constrained optimization over $\theta$. Using sample average approximation for the objective and constraints, a differentiable version is created via smooth proxies, solved using alternating gradient descent (Algorithm 1). Since the trained set only has asymptotic robustness, a **test-time calibration** (Cal-CRC, Algorithm 2) is added using full conformal prediction to ensure robustness for individual test points holds exactly for **finite samples**. The pipeline is: Parameterize prediction set $\rightarrow$ Empirical optimization under robustness constraints (with theoretical guarantees) $\rightarrow$ Test-time conformal calibration.

### Key Designs

**1. Replacing Coverage Constraints with Robustness Constraints: De-conservatism at the Source**

This is the foundation of the work, addressing the excessive conservatism caused by coverage being a sufficient but not necessary condition. Traditional CRO (RA-CPO) optimizes expected certificates under the coverage constraint $P\{Y\in U(X)\}\ge 1-\alpha$. Kiyani et al. (2025) provided a closed-form solution for optimal sets under coverage constraints, but it depends on minimizing a VaR function, which is generally intractable in continuous decision spaces $\mathcal Z$. CRC swaps this for the robustness constraint (equation (4)). Because the robustness constraint is **looser** than the coverage constraint (coverage $\Rightarrow$ robustness, but not vice versa), the feasible region is larger, leading to lower optimal risk certificates—proven in Appendix B.3. Intuitively, coverage requires the "true value to fall in the set," while robustness only requires the "decision loss to be covered by the certificate," allowing for smaller sets and more aggressive, efficient decisions.

**2. Prediction Set Parameterization + Empirical Optimization: Learning the Set as a Solvable Problem**

To ensure tractability in continuous decision spaces, CRC parameterizes $U(\cdot)$ as $U_\theta(\cdot)$. In regression, two common shapes are used: box sets $U_\theta(x)=\{y: h^{lo}_\theta(x)\le y\le h^{hi}_\theta(x)\}$ and ellipsoidal sets $U_\theta(x)=\{y: (y-\mu_\theta(x))^\top \Sigma_\theta^{-1}(x)(y-\mu_\theta(x))\le 1\}$ (capturing correlations). Given i.i.d. labeled data $D_n=\{(X_i,Y_i)\}_{i=1}^n$, replacing expectations and probabilities with sample averages yields the empirical problem:
$$\hat\theta = \arg\min_{\theta} \frac1n\sum_{i=1}^n r_\theta(X_i) \quad \text{s.t.} \quad \frac1n\sum_{i=1}^n \mathbf 1\{\phi(Y_i, z_\theta(X_i))\le r_\theta(X_i)\}\ge 1-\alpha. \tag{6}$$
This step concretizes "learning a prediction set under robustness constraints" as standard empirical constrained optimization.

**3. Smooth Proxies + Alternating Lagrangian Gradients with Non-asymptotic Guarantees**

Equation (6) contains an indicator function $\mathbf 1\{\cdot\}$ which is non-smooth. CRC uses the Lagrangian $L(\lambda;\theta)=f(\theta)+\lambda g(\theta)$, where $f(\theta)=\frac1n\sum_i r_\theta(X_i)$ (differentiable via implicit differentiation if the CRO sub-problem is convex) and $g(\theta)=1-\alpha-\frac1n\sum_i \mathbf 1\{\phi(Y_i,z_\theta(X_i))\le r_\theta(X_i)\}$. Replacing the indicator with a Gaussian error function proxy $\tilde{\mathbf 1}\{a\le b\}=\frac12(1+\mathrm{erf}(\frac{b-a}{\sqrt2\sigma}))$ results in a smooth constraint $\tilde g(\theta)$. $\min_\theta\max_{\lambda\ge0}\tilde L(\lambda;\theta)$ is then solved via alternating gradients (Algorithm 1). Theorem 3.2 (Robustness Gap) guarantees $P\{\phi(Y,z_{\hat\theta}(X))\le r_{\hat\theta}(X)\mid D_n\}\ge 1-\alpha-\Delta_n$, where $\Delta_n=O(\sqrt{d\log n/n})$. Theorem 3.3 (Risk Certificate Optimality) ensures the expected certificate of $\hat\theta$ converges at the same rate.

**4. Cal-CRC: Test-time Finite-Sample Robustness Calibration**

Theorem 3.2 only provides asymptotic robustness. For **specific test points** $X_{n+1}$, CRC splits data into $D_{train}$ and $D_{cal}$, runs Algorithm 1 on $D_{train}$ for $U_{\hat\theta_0}(\cdot)$, then performs full conformal calibration using a radius parameter $t\in\mathbb R^+$ to control set size. For box sets, $U_{\theta,t}(x)=\{y: h^{lo}_\theta(x)-t\le y\le h^{hi}_\theta(x)+t\}$; for ellipsoids, $U_{\theta,t}(x)=\{y:(y-\mu_\theta(x))^\top\Sigma_\theta^{-1}(x)(y-\mu_\theta(x))\le t\}$, both forming nested set families. For each candidate label $y$, the smallest threshold $\hat t^y$ satisfying the empirical robustness rate $\ge 1-\alpha$ is calculated via an augmented calibration set (Eq 7). Theorem 4.1 proves that under exchangeability, $P\{\phi(Y_{n+1}, z_{U_{Cal}}(X_{n+1}))\le r_{U_{Cal}}(X_{n+1})\}\ge 1-\alpha$, providing finite-sample guarantees similar to classical conformal prediction.

### Loss & Training
The training goal is the empirical constrained optimization in Eq (6). Implementation uses the smooth Lagrangian $\tilde L(\lambda;\theta)=f(\theta)+\lambda\tilde g(\theta)$ with alternating $\theta$-gradient descent and $\lambda$-projected gradient ascent (Algorithm 1). Two mild regularity conditions are required: Condition 3.1 (Lipschitzness of $\phi, z_\theta, r_\theta$ and boundedness of $r_\theta$) and Condition 3.2 (consistent boundedness of the density of $V_\theta(X,Y)=\phi(Y,z_\theta(X))-r_\theta(X)$). Test-time calibration (Cal-CRC) is layered on top.

## Key Experimental Results

### Main Results
Tasks: (i) Synthetic portfolio optimization; (ii) Real US stock portfolio optimization; (iii) Battery energy storage control. Baselines: CRO (with conformal sets), E2E (end-to-end minimizing expected risk certificate). Metrics: Risk certificate (mean $r_U(X)$), decision loss (mean $\phi(Y,z_U(X))$), robustness, and coverage.

US Stocks Problem (Loss $\phi(y,z)=-y^\top z$, 15 stocks randomized per run):

| Method | Risk Certificate ($\alpha{=}0.1$) | Decision Loss | Robustness (%) | Risk Certificate ($\alpha{=}0.2$) | Decision Loss | Robustness (%) |
|------|------|------|------|------|------|------|
| **CRC-B** | **1.160** | -0.055 | 90.9 | **0.731** | -0.059 | 80.6 |
| CRO-B | 3.794 | -0.051 | 99.9 | 3.017 | -0.054 | 99.5 |
| E2E-B | 2.129 | -0.046 | 96.7 | 1.512 | -0.041 | 92.7 |
| **CRC-E** | **1.028** | -0.077 | 90.8 | **0.701** | -0.075 | 80.6 |
| CRO-E | 6.345 | -0.069 | 99.9 | 6.195 | -0.046 | 99.8 |
| E2E-E | 4.995 | -0.071 | 98.6 | 4.503 | -0.070 | 96.4 |

CRC outperforms in risk certificates and decision losses while pinning robustness precisely around the $1-\alpha$ target. CRO/E2E push robustness to 96%–99.9%, which is typical over-conservatism; this excess robustness comes at the cost of much higher risk certificates (CRO-E 6.345 vs CRC-E 1.028).

### Ablation Study
Rather than module removal, the paper validates via **varying the nominal level $\alpha$** and **sample size $n$**:

| Setting | Observation | Explanation |
|------|---------|------|
| Varying $\alpha$ ($n{=}1500$) | CRC-E risk/loss consistently lower than baselines | More efficient across all robustness levels |
| Coverage at varying $\alpha$ | CRC coverage is much lower than robustness level | Validates "coverage is unnecessary" motivation |
| Varying $n$ ($\alpha{=}0.1$) | CRC-E stable dominance across metrics | Consistent advantage across sample sizes |

### Key Findings
- **Empirical proof of over-conservatism**: CRO/E2E robustness yields 96%–99.9% (far exceeding target), resulting in high risk certificates; CRC pushes robustness back to $1-\alpha$, yielding lower risk and loss.
- **Coverage < Robustness**: In synthetic experiments, CRC coverage is significantly lower than the robustness level, confirming that not chasing coverage leads to higher efficiency.
- **Stability**: Conclusions remain consistent across box/ellipsoid shapes and the three tasks. Ellipsoid versions generally yield lower risk certificates by modeling correlations.

## Highlights & Insights
- **Correcting the Constraint**: The paper points out that "conformal + decision" work usually assumes "robustness requires coverage." This work proves replacing it with a robustness constraint does not lose optimality (Theorem 3.1) and eliminates conservatism.
- **Clean Handling of Non-differentiability**: Use of erf smooth proxies $\rightarrow$ Lagrangian $\rightarrow$ alternating gradients with implicit differentiation provides a reusable template for learning sets under risk constraints.
- **Two-stage Training + Calibration**: Training provides asymptotic optimality, while calibration provides finite-sample guarantees. Decoupling efficiency from statistical validity via a one-dimensional radius $t$ is an insightful trick.
- **Theoretical Coherence**: Using covering numbers to prove that robustness and optimality gaps converge at $O(\sqrt{d\log n/n})$ provides clear rates for convergence.

## Limitations & Future Work
- **Dependency on i.i.d. / Exchangeability**: Theorem 4.1's guarantee relies on data exchangeability. In non-stationary environments like real stock markets, these guarantees might fail.
- **Prediction Set Shape Constraints**: Experiments use box/ellipsoid shapes to keep the minmax sub-problem convex. Multimodal distributions might be poorly captured by these shapes.
- **Differentiability Assumptions**: Theoretical and efficient solving require the CRO sub-problem to be a smooth convex program. Non-convex losses would require re-evaluation of Lipschitz conditions and implicit differentiation.
- **Calibration Overhead**: Cal-CRC requires iterating over candidate labels $y$, which might be expensive in high-dimensional continuous spaces despite discretization.
- **Future Directions**: Extending to weighted conformal prediction for distribution shift; exploring more flexible (neural) set parameterizations; designing more efficient calibration searches.

## Related Work & Insights
- **vs. CRO + Conformal Prediction (Sun et al. 2023; Johnstone & Cox 2021)**: These construct sets under coverage constraints, leading to over-conservatism; CRC uses robustness constraints for strictly lower risk certificates (Appendix B.3).
- **vs. RA-CPO / RAC (Kiyani et al. 2025)**: RA-CPO provides closed-form solutions based on VaR which are intractable for continuous decision spaces; CRC parameterizes for better scalability.
- **vs. End-to-End Decision Learning (Chenreddy & Delage 2024; Yeh et al. 2024)**: E2E minimizes downstream risk directly, but CRC experiments show E2E remains more conservative than CRC.
- **vs. Conformal Risk Control (Angelopoulos et al. 2024b)**: CRC can be seen as controlling a specific risk, but this risk is non-monotonic w.r.t. parameters, and CRC optimizes a complex certificate function $r(\cdot)$ rather than a single threshold.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Decoupling robustness from coverage with theoretical equivalence is a clear and well-supported insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + Stocks + Battery tasks, but limited mainly to investment/control scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent motivation-method-theory-experiment loop; Figure 1 is very intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides a practical framework that is both robust and efficient; the smooth proxy + nested calibration paradigm is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Gen-DFL: Decision-Focused Generative Learning for Robust Decision Making](gen-dfl_decision-focused_generative_learning_for_robust_decision_making.md)
- [\[ICLR 2026\] Multilevel Control Functional](multilevel_control_functional.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/optimization/one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[ICLR 2026\] LEGACY: A Lightweight Dynamic Gradient Compression Strategy for Distributed Deep Learning](legacy_a_lightweight_dynamic_gradient_compression_strategy_for_distributed_deep_.md)
- [\[ICLR 2026\] Proving the Limited Scalability of Centralized Distributed Optimization via a New Lower Bound Construction](proving_the_limited_scalability_of_centralized_distributed_optimization_via_a_ne.md)

</div>

<!-- RELATED:END -->
