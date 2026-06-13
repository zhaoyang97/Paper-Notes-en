---
title: >-
  [Paper Note] Inference of Online Newton Methods with Nesterov's Accelerated Sketching
description: >-
  [ICML2026][Online Newton methods] This paper equips online Newton methods with a Nesterov-accelerated sketch-and-project solver…
tags:
  - "ICML2026"
  - "Online Newton methods"
  - "Nesterov's accelerated sketching"
  - "uncertainty quantification"
  - "covariance estimation"
  - "Lyapunov equation"
date: 2026-05-08
content_hash: 0406c84ae92933b8
---

# Inference of Online Newton Methods with Nesterov's Accelerated Sketching

**Conference**: ICML2026  
**arXiv**: [2604.23436](https://arxiv.org/abs/2604.23436)  
**Code**: Not yet publicly available  
**Area**: Optimization / Online Learning / Statistical Inference  
**Keywords**: Online Newton methods, Nesterov's accelerated sketching, uncertainty quantification, covariance estimation, Lyapunov equation  

## TL;DR
This paper equips online Newton methods with a Nesterov-accelerated sketch-and-project solver, reducing the per-transition cost to $O(d^2)$. It provides the first characterization of the asymptotic normality of the last iterate under the dual uncertainty of "data randomness + solver randomness" and introduces a streaming, inversion-free covariance estimator, making online Newton with accelerated sketching truly viable for statistical inference.

## Background & Motivation

**Background**: Parameter estimation and uncertainty quantification (confidence intervals) for streaming data typically follow two paths. One is SGD + Polyak-Ruppert averaging: iterations are cheap, but maintaining the covariance matrix for confidence intervals requires $O(d^2)$ memory/time, and it is highly sensitive to condition numbers and heteroscedastic noise (prior work observed significant undercoverage by SGD when $d=20$). The second is second-order/online Newton methods: curvature information makes estimates statistically superior and robust to ill-conditioning, but solving the Newton system is $O(d^3)$, making it impractical for online use.

**Limitations of Prior Work**: Recent works such as Na & Mahoney (2025) and Kuang et al. (2025) used **unaccelerated sketch-and-project solvers** to reduce the Newton step cost to $O(d^2)$, providing last-iterate asymptotic normality and streaming covariance estimation that outperforms SGD. However, sketch-and-project solvers have Nesterov-accelerated versions (Gower et al. 2018, Derezi'nski et al. 2025): while unaccelerated error decays at $1-\mu_t$, the accelerated version decays at $1-\sqrt{\mu_t/\nu_t}$. **Computationally, acceleration is strictly faster without increasing the per-iteration cost**.

**Key Challenge**: Accelerated sketching speeds up "computation," but what does it change in statistical inference? Does acceleration increase the asymptotic covariance of the last iterate, thereby canceling out computational gains? Existing inference analyses for online Newton methods do not cover accelerated sketching.

**Goal**: To embed Nesterov's accelerated sketching into online Newton methods and provide a triple-feature approach: (i) global almost sure convergence, (ii) analytical characterization of last-iterate asymptotic normality and the limiting covariance, and (iii) a fully online, inversion-free consistent covariance estimator.

**Key Insight**: Accelerated sketching upgrades the solver from a $d$-dimensional recurrence of symmetric projection matrices to a **randomized, time-varying, asymmetric $2d$-dimensional state-co-state recurrence**, which cannot directly reuse the symmetric projection geometry of Kuang et al. (2025). The authors utilize the Cayley–Hamilton theorem and similarity matrix theory combined with Kronecker products to derive the spectral radius recurrence, study the contraction of the $(1,1)+(1,2)$ blocks, and address two new technical difficulties: the conditional deterministic randomness of accelerated parameters $(\alpha_t,\beta_t,\gamma_t)$ and fourth-order moment bounds.

**Core Idea**: By using sketch-and-project with Nesterov acceleration to approximately solve the Newton system, the solver's randomness is incorporated into the limiting covariance of the Lyapunov equation. This provides the first analytical characterization of the "computation vs. statistics" trade-off—showing that accelerated sketching does not destroy asymptotic normality but adds a correction term determined by the sketching distribution to the covariance.

## Method

### Overall Architecture
Consider the stochastic optimization problem $\min_{x\in\mathbb{R}^d} f(x)=\mathbb{E}_{\xi\sim P}[F(x;\xi)]$. The online Newton iteration is $x_{t+1}=x_t+\varphi_t\Delta x_t$, where $\Delta x_t$ should solve $B_t\Delta x_t = -g_t$. The pipeline consists of three steps:

1. **Outer layer**: Obtain sample $\xi_t$, compute gradient $g_t=\nabla F(x_t;\xi_t)$ and Hessian estimate $H_t=\nabla^2 F(x_t;\xi_t)$, and maintain the Hessian average $B_t=(1-1/t)B_{t-1}+H_{t-1}/t$ via $O(1)$ incremental updates.
2. **Inner layer**: Invoke the NASketch solver to run $\tau$ steps of sketch-and-project + Nesterov momentum to output an approximate solution $\Delta x_t$; each step involves sketching projections of $O(d s)$ magnitude, totaling $O(d^2)$.
3. **Inference**: Given the last iterate $x_t$, use a **fully online** consistent estimator $\widehat{\Sigma}_t$ to estimate the limiting covariance $\Sigma^\star$ and construct confidence intervals.

### Key Designs

1. **NASketch Solver with Nesterov Acceleration**:
    - **Function**: Obtains high-precision approximate solutions for the Newton system $B\Delta x=-g$ within $O(d^2)$ cost.
    - **Mechanism**: Maintains a state-co-state pair $(z_j, v_j)$, computes the sketching direction $\omega_j = BS_j(S_j^\top B^2 S_j)^\dagger S_j^\top(B y_j + g)$ at the midpoint $y_j=\alpha v_j+(1-\alpha)z_j$ ($S_j\in\mathbb{R}^{d\times s}$ is the sketch matrix, $s\ll d$), and updates $z_{j+1}=y_j-\omega_j$ and $v_{j+1}=\beta v_j+(1-\beta)y_j-\gamma\omega_j$. Parameters are set as $\alpha=1/(1+\gamma\nu)$, $\beta=1-\sqrt{\mu/\nu}$, and $\gamma=1/\sqrt{\mu\nu}$, where $\mu, \nu$ are spectral parameters of the sketching distribution. Setting $\alpha=0.5,\beta=0,\gamma=1$ recovers the unaccelerated version.
    - **Design Motivation**: Acceleration changes the rate from $1-\mu_t$ to $1-\sqrt{\mu_t/\nu_t}$. Using the Cayley–Hamilton theorem and Kronecker products, the authors prove that the marginal blocks $(1,1)+(1,2)$ still undergo geometric contraction.

2. **Lyapunov Equation Characterization of Limiting Covariance**:
    - **Function**: Explicitly incorporates the impact of both "data randomness" and "accelerated sketching solver randomness" into the limiting covariance.
    - **Mechanism**: Under standard smoothness and moment conditions, it is proved that $1/\sqrt{\varphi_t}\cdot(x_t-x^\star)\xrightarrow{d}\mathcal{N}(0,\Sigma^\star)$, where $\Sigma^\star$ solves the Lyapunov equation $A^\star\Sigma^\star+\Sigma^\star (A^\star)^\top + Q^\star = 0$. $A^\star$ is determined by $\nabla^2 f(x^\star)$ and the limiting linear operator of the accelerated sketching, while $Q^\star$ absorbs data noise and sketching randomness.
    - **Design Motivation**: Unlike prior online inference where sketching randomness was assumed to "asymptotically vanish," this paper assumes a fixed $\tau$ steps for sketching, meaning algorithmic randomness **does not** vanish and must be modeled. The Lyapunov equation characterizes the "computation vs. statistics" trade-off: aggressive acceleration (smaller $\nu_t$) speeds up the solver but increases the sketching noise term in $Q^\star$.

3. **Inversion-free Online Covariance Estimator**:
    - **Function**: Streamlines the estimation of $\Sigma^\star$ for confidence intervals **without inverting any matrices at each step**.
    - **Mechanism**: Expands the Lyapunov equation into cumulative updates along the iteration sequence. The estimator $\widehat\Sigma_t$ satisfies $\widehat\Sigma_t\xrightarrow{p}\Sigma^\star$ by (i) replacing $\nabla^2 f(x^\star)$ with the Hessian average $B_t$, (ii) replacing expectations with sample averages of sketching operators, and (iii) using residuals for noise variance. This requires a fourth-order moment bound $\mathbb{E}[\|x_t-x^\star\|^4]=O(\varphi_t^2)$, which is technically more demanding than standard second-order bounds.
    - **Design Motivation**: To ensure engineering feasibility, $O(d^3)$ matrix inversions are avoided. The estimator uses only quantities already computed (iterates, sketching directions, Hessian average), maintaining an $O(d^2)$ per-step cost.

### Loss & Training
- **Step size**: $\varphi_t=c_\varphi/t^\alpha$, $\alpha\in(1/2,1)$, paired with $1/t$ decay for the Hessian average $B_t$.
- **Acceleration parameters**: Derived from $(\mu_t, \nu_t)$. While they are **conditionally deterministic random** variables, the authors prove $|\alpha_t-\alpha^\star|=O_p(\sqrt{\varphi_t})$, ensuring their randomness only contributes higher-order terms.
- **Sketching steps $\tau$**: Fixed constant (typically $\tau=5\sim 10$). The fact that $\tau$ does not grow with $t$ is the fundamental reason why sketching randomness persists and complicates the analysis.

## Key Experimental Results

### Main Results
Four online inference methods were compared on synthetic linear, logistic, and quantile regressions: Averaged SGD (ASGD), unaccelerated sketched Newton (SN), the proposed accelerated version (NA-SN, $\nu_t=1$), and a non-accelerated variant (NA-SN with $\mu_t\nu_t=1$). Dimensions $d\in\{20,50,100,200\}$, iterations $T=10^5$, 90% confidence level.

| Setup | Method | Coverage Rate | Avg Interval Width (×$10^{-2}$) | per-step Time (ms) |
|------|------|--------|----------------------------|---------------------|
| $d=100$ Linear Reg. | ASGD | 0.78 | 6.4 | 0.9 |
| $d=100$ Linear Reg. | SN | 0.89 | 4.1 | 1.4 |
| $d=100$ Linear Reg. | **NA-SN (Ours)** | **0.90** | **3.9** | **1.5** |
| $d=200$ Logistic Reg. | ASGD | 0.71 | 9.8 | 2.1 |
| $d=200$ Logistic Reg. | SN | 0.88 | 5.6 | 3.0 |
| $d=200$ Logistic Reg. | **NA-SN (Ours)** | **0.90** | **5.2** | **3.1** |

Accelerated NA-SN aligns closely with nominal coverage levels and produces 5%–8% narrower interval widths than unaccelerated SN. The per-step time is almost identical to SN, while providing significantly better coverage (10–20 percentage points higher) than ASGD in ill-conditioned cases.

### Ablation Study
| Configuration | Last-iterate Error $\|x_T-x^\star\|^2$ (×$10^{-3}$) | Coverage Rate | Notes |
|------|----------------------------------------|--------|------|
| Full: NA-SN + Hessian Avg + Online Cov. Est. | 4.2 | 0.90 | Complete model |
| w/o Nesterov Acceleration (Degrades to SN) | 4.5 | 0.88 | Slower inner convergence |
| w/o Hessian Avg (Using single-step $H_t$) | 7.6 | 0.81 | Amplified cov. fluctuations |
| w/o Online Cov. Est. (Using plug-in $\nabla^2 f(x_T)^{-1}$) | 4.2 | 0.86 | Plug-in underestimates variance |
| Sketching steps $\tau=1$ | 5.1 | 0.87 | Sketching noise dominates $Q^\star$ |

### Key Findings
- **Acceleration provides a "Win-Win"**: Accelerated sketching not only reduces inner-layer solver error but also slightly narrows the last-iterate confidence interval width, as $\nu_t=1$ minimizes the sketching term in the Lyapunov equation.
- **Hessian averaging is critical**: Removing it drops the coverage rate to 81%, showing that the consistency of the covariance estimator relies heavily on the rate $B_t\to\nabla^2 f(x^\star)$.
- **Small $\tau$ is sufficient**: $\tau=5\sim 10$ approaches the accuracy of $\tau=20$, validating the core assumption that sketching steps can remain constant.
- **Robust to ill-conditioning**: As the condition number increases from $10$ to $10^4$, ASGD coverage drops from 0.88 to 0.62, while NA-SN maintains 0.89–0.91.

## Highlights & Insights
- **First Lyapunov equation for dual-source randomness**: Most prior works ignore sketching randomness or assume it vanishes. This paper provides a realistic characterization of limiting covariance, which can be generalized to other randomized linear solvers.
- **Cayley–Hamilton + Kronecker Product analysis**: The proof of contraction for the state-co-state joint system is the first "accelerated version contraction proof" in the sketch-and-project literature.
- **Inversion-free Covariance Estimator**: By expanding the Lyapunov equation into an additive form, the $O(d^2)$ per-step cost makes this approach suitable for any inference task where the covariance is the solution to a Lyapunov equation.

## Limitations & Future Work
- **Dependency on available Hessian**: Analysis assumes $H_t=\nabla^2 F(x_t;\xi_t)$ is accessible; extension to quasi-Newton or finite differences for deep networks is not provided.
- **Unconstrained and smooth focus**: Handling constraints, non-smoothness (e.g., L1), and non-convexity remains for future work.
- **Hyperparameter $(\mu_t,\nu_t)$ estimation**: In practice, these must be estimated online, and the second-order impact of estimation errors on the limiting covariance is not yet characterized.
- **Regression-centric experiments**: Lack of large-scale real-world datasets (e.g., recommendation systems or policy gradients).

## Related Work & Insights
- **vs. Polyak-Juditsky Averaged SGD**: SGD is minimax optimal in covariance but sensitive to ill-conditioning; Ours matches it in well-conditioned settings and is strictly better in ill-conditioned ones.
- **vs. Kuang et al. 2025 (SN)**: This work strictly generalizes SN and handles the technical difficulty of asymmetric momentum systems.
- **vs. Leluc & Portier 2023 (Preconditioned SGD)**: They use deterministic preconditioners; this paper treats the solver as stochastic and incorporates its randomness into the final inference.
- **vs. Derezi'nski et al. 2025**: They focus on computational rates of accelerated sketching; this paper places it in a statistical inference framework to answer what the "statistical cost" of acceleration is.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Online Linear Regression with Paid Stochastic Features](../../AAAI2026/others/online_linear_regression_with_paid_stochastic_features.md)
- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)
- [\[CVPR 2026\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](../../CVPR2026/others/rooftop_wind_field_reconstruction_using_sparse_sen.md)
- [\[ICML 2026\] Industrializing Prediction-Powered Inference: The GLIDE Library for Reliable GenAI and Agentic Systems Evaluation](industrializing_prediction-powered_inference_the_glide_library_for_reliable_gena.md)
- [\[ICML 2026\] Amortized Simulation-Based Inference in Generalized Bayes via Neural Posterior Estimation](amortized_simulation-based_inference_in_generalized_bayes_via_neural_posterior_e.md)

</div>

<!-- RELATED:END -->
