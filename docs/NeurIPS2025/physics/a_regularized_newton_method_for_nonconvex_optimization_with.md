---
title: >-
  [Paper Note] A Regularized Newton Method for Nonconvex Optimization with Global and Local Complexity Guarantees
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][Regularized Newton method] This paper proposes a novel class of regularizers constructed from current and historical gradients, combined with a conjugate gradient method equipped with negative-curvature detection to solve the regularized Newton equation. Within an adaptive framework that requires no prior knowledge of the Hessian Lipschitz constant, the method simultaneously achieves, for the first time…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "Regularized Newton method"
  - "nonconvex optimization"
  - "optimal global complexity"
  - "quadratic local convergence"
  - "adaptive algorithm"
date: 2026-05-08
content_hash: ee0b39a6a26901fb
---

# A Regularized Newton Method for Nonconvex Optimization with Global and Local Complexity Guarantees

**Conference**: NeurIPS 2025
**arXiv**: [2502.04799](https://arxiv.org/abs/2502.04799)  
**Code**: No public code (MATLAB implementation; experiments based on the CUTEst benchmark)  
**Area**: Scientific Computing
**Keywords**: Regularized Newton method, nonconvex optimization, optimal global complexity, quadratic local convergence, adaptive algorithm

## TL;DR
This paper proposes a novel class of regularizers constructed from current and historical gradients, combined with a conjugate gradient method equipped with negative-curvature detection to solve the regularized Newton equation. Within an adaptive framework that requires no prior knowledge of the Hessian Lipschitz constant, the method simultaneously achieves, for the first time, the optimal global iteration complexity of $O(\epsilon^{-3/2})$ and a quadratic local convergence rate.

## Background & Motivation
Finding an $\epsilon$-stationary point ($\|\nabla\varphi(x^*)\|\leq\epsilon$) is a central problem in nonconvex optimization. Newton-type methods are powerful due to their quadratic local convergence near solutions with positive-definite Hessians, but classical Newton's method offers no global convergence guarantee. Several globalization strategies have been developed:

- **Cubic regularization** (Nesterov & Polyak 2006): achieves the optimal $O(\epsilon^{-3/2})$ iteration complexity while preserving quadratic local convergence, but the subproblem is harder to solve than in quadratic regularization.
- **Quadratic regularization (Levenberg–Marquardt)**: the subproblem reduces to solving a linear system, efficiently solvable by conjugate gradients, making it well-suited for large-scale problems.
- **Trust-region methods**: e.g., CAT (Hamad & Hinder 2024) simultaneously achieves optimal global order and quadratic local convergence, but incurs large memory overhead.

Quadratic regularization methods face a long-standing **global–local trade-off**: using $\rho_k\propto\|g_k\|^{1/2}$ yields optimal global complexity but only a local convergence order of $3/2$; using $\rho_k\propto\|g_k\|$ yields quadratic local convergence but with no clear global rate. Prior work (e.g., Gratton et al. 2024) achieves a near-optimal $O(\epsilon^{-3/2}\log(1/\epsilon))$ global rate and superlinear local convergence, but cannot attain both simultaneously. **Bridging this gap** is the core motivation of this paper.

## Core Problem
Can one design a parameter-free quadratic-regularization Newton method that **simultaneously** achieves:
1. Optimal global iteration complexity $O(\epsilon^{-3/2})$ (without logarithmic factors);
2. Quadratic local convergence rate;
3. No prior knowledge of the Hessian Lipschitz constant $L_H$;
4. No computation of the minimum eigenvalue?

## Method

### Overall Architecture
The algorithm, named **ARNCG** (Adaptive Regularized Newton-CG), follows the pipeline below:

**Input**: Initial point $x_0$, parameters $\mu,\beta,\tau_-,\tau_+,\gamma,m_{\max},M_0$, and regularizer sequences $\{\omega_k^t, \omega_k^f\}$.

Each iteration consists of three levels:
1. **Main loop**: invokes NewtonStep with the trial-step regularizer $\omega_k^t$; switches to the fallback regularizer $\omega_k^f$ if the gradient increases unexpectedly.
2. **NewtonStep subroutine**: calls CappedCG to solve the regularized Newton equation $(H+2\rho I)d=-g$; executes different line-search strategies based on the returned status (SOL/NC/TERM).
3. **CappedCG subroutine**: standard conjugate gradient with negative-curvature detection, requiring only one additional Hessian–vector product throughout the entire CG process.

**Output**: An $\epsilon$-stationary point $x^*$ satisfying $\|\nabla\varphi(x^*)\|\leq\epsilon$.

### Key Designs

1. **Novel regularizer construction**: Two families of regularizers are proposed, incorporating an extra parameter $\theta>0$ and a ratio $\delta_k$ to modulate the global–local trade-off:

    - **Type I**: $\omega_k^f=\|g_k\|^{1/2}$, $\omega_k^t=\omega_k^f\cdot\min(1, (g_k/g_{k-1})^\theta)$
    - **Type II**: $\omega_k^f=\epsilon_k^{1/2}$ (where $\epsilon_k=\min_{i\leq k}\|g_i\|$), $\omega_k^t=\omega_k^f\cdot(\epsilon_k/\epsilon_{k-1})^\theta$

   Key insight: when $\theta=0$, the method reduces to classical gradient-square-root regularization (local order $3/2$); when $\theta>0$, the additional factor $\delta_k^\theta$ decays rapidly once the iterates enter superlinear convergence, progressively lifting the local convergence order from $3/2$ toward $2$; full quadratic convergence is achieved for $\theta>1$.

2. **Capped CG with negative-curvature detection**: Based on Royer et al. (2020), the subroutine returns one of three statuses:

    - **SOL**: an approximate solution to the regularized Newton equation has been found.
    - **NC**: a negative-curvature direction satisfying $d^T H d \leq -\rho\|d\|^2$ has been detected.
    - **TERM**: the regularization parameter is too small, causing early termination due to excessive iterations.

   Each CG iteration requires exactly one Hessian–vector product, and negative-curvature monitoring incurs only one additional Hessian–vector product over the entire CG run.

3. **Adaptive Lipschitz constant estimation**: The parameter $M_k$ adaptively estimates $L_H$ and is updated based on line-search outcomes:

    - Successful step with sufficient decrease → decrease $M_k$ (set $\mathcal{J}_{-1}$)
    - Successful step with moderate decrease → keep $M_k$ (set $\mathcal{J}_0$)
    - Failed step → increase $M_k$ (set $\mathcal{J}_1$)

   After $O(\log(M_0/L_H))$ iterations, $M_k$ stabilizes at $O(L_H)$.

4. **Two-phase line search**: For the SOL case, the method first attempts a standard Armijo line search (Criterion 2.4); if that fails, it switches to a step-scaled secondary line search (Criterion 2.5), ensuring a uniform bound on the number of function evaluations.

5. **Fallback step mechanism**: When a trial step causes an unexpected gradient increase ($g_{k+1/2}>g_k$ and $g_k\leq g_{k-1}$), the method switches to the fallback regularizer $\omega_k^f$. This ensures the validity of the subsequence transition lemma (Lemma 3.2). Experiments show that fallback steps are rarely triggered in practice and can be relaxed or removed.

### Loss & Training
- Regularization coefficient: $\rho_k=(M_k\cdot\omega_k)^{1/2}$, where $\omega_k$ is selected according to the trial or fallback step.
- CG solve tolerance: $\xi=\rho_k=M_k^{1/2}\omega_k$ (adapted adaptively across iterations).
- Recommended experimental parameters: $\mu=0.3$, $\beta=0.5$, $\tau_-=0.3$, $\tau_+=1.0$, $\gamma=5$, $M_0=1$, $\eta=0.01$, $\theta\in[0.5,1]$, $m_{\max}=1$.

## Key Experimental Results

The method is evaluated on 124 unconstrained optimization problems from the CUTEst benchmark with more than 100 variables (dimensions ranging from 100 to 123,200), and compared against CAT (trust-region) and AN2CER (regularized Newton):

| Metric | ARNCG g | ARNCG ε | CAT | AN2CER |
|--------|---------|---------|-----|--------|
| Success rate (%) | 87.10 | 86.29 | 85.48 | 81.45 |
| Time (shifted geo-mean) | 16.71 | 18.28 | 23.34 | 36.70 |
| Hessian evaluations | 80.86 | 85.03 | 88.47 | 170.10 |
| Gradient evaluations | 86.41 | 90.78 | 96.61 | 172.02 |
| Function evaluations | 119.51 | 125.29 | 125.56 | 176.80 |
| Memory (largest problem) | ~6 GB | ~6 GB | ~74 GB | — |

### Ablation Study
- **Fallback parameter $\lambda$**: has minimal impact on success rate; larger $\lambda$ triggers fallback steps more frequently, increasing computational cost. Setting $\lambda=0$ (disabling fallback) is recommended.
- **Regularization parameter $\theta$**: the range $\theta\in[0.5,1]$ offers the best balance between computational efficiency and local convergence speed; excessively large $\theta$ imposes stricter CG tolerance requirements, increasing Hessian–vector product counts.
- **Line-search parameter $m_{\max}$**: $m_{\max}=1$ suffices (at most 4 function evaluations per iteration); increasing $m_{\max}$ yields no significant benefit.
- **Secondary line search and TERM status**: rarely triggered in practice.
- **Comparison of the two regularizer families**: although ARNCG g carries an extra double-logarithmic factor in theory, its empirical performance is comparable or slightly better than ARNCG ε.

## Highlights & Insights
- **Theoretical breakthrough**: This work is the first to prove that a quadratic-regularization Newton method can simultaneously achieve the optimal $O(\epsilon^{-3/2})$ global iteration complexity and quadratic local convergence, closing a long-standing theoretical gap in the field.
- **Local convergence rate lifting technique** (Lemma 3.9): a recursive argument elegantly demonstrates how $\delta_k^\theta$ progressively elevates the local convergence order from $3/2$ to $2$. For $\theta\in(0,1]$, the method attains order $1+\nu_\infty$ (where $\nu_\infty$ is the positive root of a specific equation); for $\theta>1$, strict quadratic convergence is reached after finitely many steps.
- **Practical efficiency**: no prior knowledge of the Hessian Lipschitz constant is required, no minimum eigenvalue computation is needed, each CG iteration requires only one Hessian–vector product, and memory usage is far superior to factorization-based methods on large-scale problems (6 GB vs. 74 GB).
- **The choice $\omega_k^f=\epsilon_k$** eliminates the $\log\log(1/\epsilon)$ factor; the key is that the monotonicity of $\epsilon_k$ enables tighter control of $V_k$.

## Limitations & Future Work
- **Limited experimental scope**: evaluation is confined to the CUTEst benchmark; although a PINN experiment is included, large-scale applicability in deep learning remains to be demonstrated.
- **Extension to Hölder-continuous Hessians**: the paper notes that the technique generalizes to this setting, but since the Hölder exponent is unknown in practice, additional estimation steps are required and this extension has not been completed.
- **Convex setting**: applicability to the convex settings studied by Doikov & Nesterov (2021) and related works remains an open problem.
- **Stochastic optimization**: extension to inexact methods (e.g., Yao et al. 2023) and stochastic optimization settings is an open direction.
- **Necessity of the fallback step**: experiments show that fallback steps are almost never triggered in practice; whether they can be eliminated entirely from the theory warrants further investigation.
- **Adaptive CG tolerance strategy**: the current CG tolerance $\xi=\rho_k$ is designed for local convergence; during the global phase, a looser fixed tolerance $\eta$ could reduce computational cost.

## Related Work & Insights

| Method | Global Complexity | Local Conv. Rate | Requires $\epsilon$ | Requires Min. Eigenvalue |
|--------|-------------------|-----------------|---------------------|--------------------------|
| Royer et al. 2020 | $O(\epsilon^{-3/2})$ | N/A | ✓ | ✗ |
| Gratton et al. 2024 (AN2CER) | $O(\epsilon^{-3/2}\log(1/\epsilon))$ | N/A | ✗ | ✓ |
| Zhu & Xiao 2024 | $O(\epsilon^{-3/2})$ | Superlinear (requires global strong convexity) | ✓ | ✗ |
| CAT (Hamad & Hinder 2024) | $O(\epsilon^{-3/2})$ | Quadratic | ✗ | ✗ |
| **Ours (ARNCG)** | $O(\epsilon^{-3/2})+\tilde{O}(1)$ | Quadratic ($\theta>1$) | ✗ | ✗ |

Compared to CAT, ARNCG uses significantly less memory (avoiding full Hessian construction) and is slightly faster on CUTEst. Compared to AN2CER, ARNCG requires no minimum eigenvalue computation and achieves a global complexity bound free of logarithmic factors.

The idea of exploiting historical information (via $g_{k-1}$) within the regularizer to "boost" local convergence rates raises the question of whether an analogous technique could be applied to deep learning optimizers (e.g., leveraging the momentum term in Adam) to design adaptive methods with provably good global and local convergence properties. The subsequence partition technique—analyzing iterate sequences by segmenting according to the monotonicity of gradient norms—may also prove useful in the analysis of other adaptive optimization algorithms. The efficient use of Hessian–vector products in large-scale problems (e.g., PINN training) makes this method a strong baseline for second-order optimization in scientific computing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First resolution of the global–local complexity trade-off in quadratic-regularization Newton methods, an open problem in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparisons on the CUTEst benchmark, but validation on practical ML applications is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations, clear structure, and well-explained key insights.
- Value: ⭐⭐⭐⭐⭐ — Closes an important theoretical gap with an elegant and practically effective algorithm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ARROW: An Adaptive Rollout and Routing Method for Global Weather Forecasting](../../ICLR2026/physics/arrow_an_adaptive_rollout_and_routing_method_for_global_weather_forecasting.md)
- [\[NeurIPS 2025\] 3DID: Direct 3D Inverse Design for Aerodynamics with Physics-Aware Optimization](3did_direct_3d_inverse_design_for_aerodynamics_with_physics-aware_optimization.md)
- [\[ICML 2026\] Spectrally Regularized Latent Flow Matching for Turbulence Generation](../../ICML2026/physics/spectrally_regularized_latent_flow_matching_for_turbulence_generation.md)
- [\[ICLR 2026\] Towards a Transferable Acceleration Method for Density Functional Theory](../../ICLR2026/physics/towards_a_transferable_acceleration_method_for_density_functional_theory.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)

</div>

<!-- RELATED:END -->
