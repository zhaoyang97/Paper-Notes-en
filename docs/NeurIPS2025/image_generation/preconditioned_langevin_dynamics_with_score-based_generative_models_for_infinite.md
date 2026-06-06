---
title: >-
  [Paper Note] Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems
description: >-
  [NeurIPS 2025][Image Generation][Preconditioned Langevin dynamics] This paper rigorously analyzes score-based generative model (SGM)-driven Langevin posterior samplers in infinite-dimensional Hilbert spaces…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Preconditioned Langevin dynamics"
  - "score-based generative models"
  - "infinite-dimensional Bayesian inverse problems"
  - "function space"
  - "optimal preconditioner"
date: 2026-05-08
content_hash: 7df2e7300cdf4413
---

# Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems

**Conference**: NeurIPS 2025
**arXiv**: [2505.18276](https://arxiv.org/abs/2505.18276)  
**Code**: Unavailable  
**Area**: Diffusion Models / Image Generation
**Keywords**: Preconditioned Langevin dynamics, score-based generative models, infinite-dimensional Bayesian inverse problems, function space, optimal preconditioner

## TL;DR
This paper rigorously analyzes score-based generative model (SGM)-driven Langevin posterior samplers in infinite-dimensional Hilbert spaces, derives for the first time convergence bounds that explicitly depend on score approximation errors, and identifies an optimal preconditioner that jointly depends on the forward operator and score errors, guaranteeing uniform convergence rates across all posterior modes.

## Background & Motivation

Bayesian inverse problems arise broadly in applications such as X-ray CT, seismic tomography, and inverse heat conduction, where the central task is to estimate unknown parameters from noisy observations. When the unknown is an object in function space, the problem is naturally posed in an infinite-dimensional setting, and it is essential to design inference algorithms that remain stable as discretization is refined.

Existing approaches face a deep tension:

**Limitations of finite-dimensional methods**: Most SGM-based posterior sampling methods assume the posterior is supported on a finite-dimensional space, and their error estimates diverge as dimension increases. The "discretize-then-Bayesianize" paradigm can lead to instability or even theoretically ill-posed results.

**Challenges specific to infinite dimensions**: In infinite-dimensional Hilbert spaces, a Lebesgue reference measure does not exist and density functions cannot be defined; prior covariances must be trace-class; and the drift of standard Langevin samplers may diverge at fine scales.

**Key Insight**: The "apply-algorithm-then-discretize" paradigm should be adopted. Embedding SGMs as learned priors within Langevin samplers requires understanding the interplay among score approximation errors, preconditioner operators, trace-class priors, and linear forward maps. Figure 1 of the paper illustrates two key counterexamples: (1) using an identity prior covariance yields samples that appear stable but have infinite energy and thus do not belong to the Hilbert space; (2) using a trace-class prior causes the standard Langevin drift to diverge at fine scales.

## Method

### Overall Architecture
The paper considers the linear Bayesian inverse problem $y = AX_0 + n$, where $X_0 \sim \mu$ is a random variable on a Hilbert space $H$ and $A:H\to\mathbb{R}^N$ is a linear operator. The core sampler is a preconditioned Langevin SDE:

$$dX_t = S_\theta(X_t, \tau; \mu) dt + C\nabla_X \log \rho(y - AX_t) dt + \sqrt{2C} dW_t$$

where $S_\theta$ is a neural network approximation of the score function, $C$ is the preconditioner operator, and $W_t$ is a Wiener process on $H$.

### Key Designs

1. **Rigorous Definition of the Infinite-Dimensional Score Function**:

    - Since densities do not exist in infinite dimensions, the score is expressed via conditional expectation: $S(X,\tau;\mu) = -(1-e^{-\tau})^{-1}(X - e^{-\tau/2}\mathbb{E}[X_0|X_\tau = X])$
    - Proposition 2.1 reformulates the score as $S(X,\tau;\mu) = -e^{\tau/2}\mathbb{E}[C(C_\mu C_\tau^{-1})^{-1}\nabla\Phi(X_0)|X_\tau=X] - CC_\tau^{-1}X$
    - **Design Motivation**: This avoids the density-based definitions used in finite dimensions.

2. **Error Analysis in the Gaussian Setting (Theorem 3.1)**:

    - Score approximation errors are assumed to have a component-wise form $\varepsilon_j^a(\tau)X^{(j)} + \varepsilon_j^b(\tau)$
    - KL divergence bounds are derived for each mode, with explicit dependence on the preconditioner $C$ and score errors
    - A sufficient condition for globally bounded convergence is given: $\sum_j |\lambda_j^{-1}\varepsilon_j^b(\tau)| < \infty$
    - **Key Insight**: The error $\varepsilon_j^b$ introduces bias that is amplified by the preconditioner factor $\lambda_j^{-1}$

3. **Optimal Preconditioner (Theorem 4.1)**:

    - For observed modes $j \leq N$: $\lambda_j^{(0)} = [\mu_j^{-1} + \sigma^{-2}(A_N^\top A_N)_{jj}]^{-1}$
    - For unobserved modes $j > N$: $\lambda_j^{(0)} = \mu_j$ (i.e., $C = C_\mu$)
    - A first-order correction incorporating score errors: $\lambda_j^{(1)} = \lambda_j^{(0)3}\mu_j^{-2} - \lambda_j^{(0)2}\mu_j^{-1} - \varepsilon_j^a \lambda_j^{(0)}$
    - **Design Motivation**: Proposition 4.1 analyzes the mean-reversion rate of each mode and selects $C$ to equalize convergence rates across all modes
    - **Key Finding**: $C$ cannot be the identity operator — it must be trace-class to ensure the diffusion process is well-defined on the Hilbert space

4. **Extension to Non-Gaussian Priors (Section 5)**:

    - The prior is absolutely continuous with respect to a Gaussian reference measure: $d\mu/d\mathcal{N}(0,C_\mu) \propto \exp(-\Phi(X))$
    - Proposition 5.1 decomposes the score into a Gaussian part plus a nonlinear correction term
    - The stationary distribution (Proposition 5.2) is absolutely continuous with respect to a Gaussian measure with approximate mean and covariance
    - The optimal preconditioner additionally accounts for the convexity constant of the potential: $\lambda_j = [\mu_j^{-1} + \sigma^{-2}(A_N^\top A_N)_{jj} + C_{\phi_j}]^{-1}$

### Experimental Validation Strategy
Two linear inverse problems with analytically tractable solutions are used to validate the theory: the KL expansion of a Brownian sheet and an inverse source problem for the heat equation.

## Key Experimental Results

### Main Results: Inverse Source Reconstruction for the Heat Equation

| Configuration | Preconditioned Langevin | Standard Langevin | Notes |
|------|----------------|--------------|------|
| $\tau=10^{-3}$, low score error | Accurate reconstruction | Fine-scale divergence | Optimal $C$ absorbs error |
| $\tau=10^{-1}$, high score error | Degraded but stable | Severe deterioration | Clear robustness advantage of preconditioning |

### Ablation Study: Discretization Invariance (Brownian Sheet)

| Number of Observed Modes $M^2$ | Preconditioned Method | Notes |
|------------------|-----------|------|
| $75^2$ | Stable reconstruction | KL coefficients accurately recovered |
| $200^2$ | Equally stable | Confirms discretization invariance |

### Key Findings
- The optimal preconditioner yields rapid, uniform decay of per-mode autocorrelation functions (Figure 4), whereas standard Langevin exhibits highly heterogeneous convergence rates across modes
- Bias introduced by score error $\varepsilon_j^b$ cannot be eliminated through preconditioning, but the influence of $\varepsilon_j^a$ can be controlled by an appropriate preconditioner
- Practical recommendation: $C$ should be set as close to the prior covariance $C_\mu$ as possible, with corrections informed by the posterior covariance or score error estimates

## Highlights & Insights
- **Deep value of the infinite-dimensional perspective**: The preconditioner $C$ is not a technical fix but an intrinsic requirement of the infinite-dimensional setting — trace-class operators must be used consistently throughout, from forward diffusion to Langevin sampling. This insight directly guides discretization strategy choices in practice.
- **Practical utility of the error analysis**: Theorem 3.1 provides verifiable sufficient conditions, enabling practitioners to determine whether a given score approximation guarantees global convergence.
- **The conditional expectation representation in Proposition 2.1** elegantly circumvents the non-existence of densities in infinite dimensions, laying a rigorous foundation for subsequent analysis.

## Limitations & Future Work
- The theoretical analysis relies on the assumption of a linear inverse problem and joint diagonalizability of the operators; deriving optimal preconditioners for nonlinear inverse problems remains an important open problem.
- The structural assumptions on score approximation errors (Assumptions 1 and 2) are difficult to verify precisely for practical deep learning models, and more practical error estimation methods need to be developed.
- Only small-scale validation experiments are presented; large-scale quantitative comparisons on real high-dimensional imaging inverse problems (e.g., medical image reconstruction) are lacking.
- The effect of specific discretization choices on practical performance — such as the interaction between time step size and the preconditioner — is not thoroughly discussed.

## Related Work & Insights
- **vs [Song et al. 2023] finite-dimensional analysis**: Finite-dimensional error estimates diverge with dimension and cannot guarantee stability under discretization refinement; this paper derives globally bounded conditions in infinite dimensions.
- **vs [Pidstrigach et al. 2023]**: Both derive an optimal $C$ minimizing the Wasserstein-2 distance, but this paper provides a more operational interpretation via mean-reversion rates and additionally incorporates the effect of score errors.
- **vs [Cardoso et al. 2025] nonlinear inverse problems**: The nonlinear setting is more complex and does not admit identification of an optimal preconditioner; this paper exploits explicit formulas available in the linear setting to obtain finer-grained analysis.
- **vs classical MCMC preconditioning**: Classical methods such as pCN and DILI do not involve SGM score approximation errors; this paper is the first to reveal the critical interaction between score errors and the preconditioner.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First convergence bounds depending on score errors and optimal preconditioner derived in infinite dimensions
- Experimental Thoroughness: ⭐⭐⭐ Only small-scale validation experiments; lacks quantitative evaluation in large-scale application settings
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, but the information density is very high and the paper is not accessible to readers without a strong mathematical background
- Value: ⭐⭐⭐⭐ Provides a solid theoretical foundation for SGM-based posterior sampling in inverse problems, with important implications for practice

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PID-controlled Langevin Dynamics for Faster Sampling of Generative Models](pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)
- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)

</div>

<!-- RELATED:END -->
