---
title: >-
  [Paper Note] A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control
description: >-
  [ICLR2026][Optimization][Stochastic Optimal Control] For a class of stochastic optimal control (SOC) problems where the "uncontrolled drift is the gradient of a potential function," this paper proves that the linearized HJB operator is unitarily equivalent to a Schrödinger operator with a purely discrete spectrum. Consequently, long-horizon optimal control can be directly determined by the **principal eigenfunction** of this operator (with correction terms decaying exponentia…
tags:
  - "ICLR2026"
  - "Optimization"
  - "Stochastic Optimal Control"
  - "HJB Equation"
  - "Schrödinger Operator"
  - "Eigenfunction Learning"
  - "Long-Horizon Planning"
date: 2026-05-08
content_hash: 3075ebbbf421e046
---

# A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=lcEw5NcSij](https://openreview.net/forum?id=lcEw5NcSij)  
**Code**: TBD  
**Area**: Optimization / Stochastic Optimal Control / Spectral Methods  
**Keywords**: Stochastic Optimal Control, HJB Equation, Schrödinger Operator, Eigenfunction Learning, Long-Horizon Planning

## TL;DR
For a class of stochastic optimal control (SOC) problems where the "uncontrolled drift is the gradient of a potential function," this paper proves that the linearized HJB operator is unitarily equivalent to a Schrödinger operator with a purely discrete spectrum. Consequently, long-horizon optimal control can be directly determined by the **principal eigenfunction** of this operator (with correction terms decaying exponentially over the time horizon). Based on this, closed-form solutions for symmetric LQR are provided, and a relative eigenfunction loss is proposed to eliminate "implicit reweighting" bias, reducing the memory/time complexity of long-horizon SOC from $O(Td)$ to $O(d)$ while improving control accuracy by approximately one order of magnitude.

## Background & Motivation
**Background**: Stochastic optimal control investigates how to drive a system characterized by stochastic differential equations (SDEs) to minimize the expected total cost, with applications in rare event sampling in molecular dynamics, robotics, and finance. Under the "affine control" setting (where control enters the state linearly), the optimal control is exactly equal to the gradient of the value function, allowing for significant simplification of the corresponding Hamilton-Jacobi-Bellman (HJB) equation. Since high-dimensional problems cannot be solved with grid-based PDE solvers due to the curse of dimensionality, neural networks are commonly used to solve the HJB, primarily via two routes: Forward-Backward SDE (FBSDE) methods and Iterative Diffusion Optimization (IDO, which relies on simulating controlled trajectories + automatic differentiation to update parameters).

**Limitations of Prior Work**: These methods **deteriorate significantly as the time horizon $T$ increases**. Memory and single-step runtime grow at least linearly with $T$. Theoretically, error estimates for FBSDE worsen with $T$, and for IDO, the variance of weights in importance sampling can explode exponentially with $T$. In other words, long-horizon planning is a shared weakness of this family of methods, a phenomenon replicated by the authors in their experiments.

**Key Challenge**: Existing methods treat SOC as a **dynamic programming problem that rolls back along the time axis**, inherently tying the computational cost to $T$. However, intuitively, when the time is far from the terminal moment ($t \ll T$), the optimal control should converge to a "steady state" that is nearly independent of $T$—an intuition that current methods fail to exploit.

**Goal**: To identify a sub-class of SOC problems where long-horizon control can be characterized by a **$T$-independent steady-state object**, thereby escaping the curse of computational cost increasing with $T$.

**Key Insight**: Under affine control, the HJB can be linearized via the Cole–Hopf transformation $\psi := \exp(-V)$ into $\partial_t\psi = L\psi$, where the optimal control is $u^* = \partial_x\log\psi$. The solution to this linear PDE can leverage the logic of finite-dimensional linear ODEs—expanding the initial value in the operator's eigenbasis and applying the exponential operator to the eigenvalues. If $L$ has a **discrete spectrum**, $\psi$ can be written as an eigenfunction series $\psi_\tau=\sum_i e^{-\lambda_i\tau}\langle\phi_i,\psi_0\rangle\phi_i$, where the "principal eigenfunction" $\phi_0$ corresponding to the lowest eigenvalue dominates the long-horizon behavior.

**Core Idea**: Under the assumption that "drift is a gradient field," $L$ is **unitarily equivalent** to a Schrödinger operator $S=-\Delta+V$, which guarantees the existence of a discrete spectrum. Thus, long-horizon optimal control can be provided by a single principal eigenfunction $\phi_0$, with correction terms decaying exponentially as $T-t$ increases—transforming a time-rolling control problem into a $T$-independent eigenvalue problem.

## Method

### Overall Architecture
The method presented is a chain of "theoretical reduction + numerical solving." The input is an affine control SOC problem: SDE dynamics $dX_t^u=(b(X_t^u)+\sigma u)\,dt+\sqrt{\beta^{-1}}\sigma\,dW_t$, with a cost functional containing a running cost $f$ and a terminal cost $g$. The output is the high-dimensional, long-horizon optimal feedback control $u^*(x,t)$.

The entire process consists of four steps: (1) **Linearization**—using a Cole–Hopf type transformation to convert the nonlinear HJB equation into a linear PDE dominated by a linear operator $L$; (2) **Spectral Verification**—proving that under the "gradient drift" assumption, $L$ is unitarily equivalent to a Schrödinger operator, ensuring a discrete spectrum, an orthogonal eigenbasis, and a simple lowest eigenvalue corresponding to a strictly positive eigenfunction (the principal eigenfunction $\phi_0$); (3) **Eigenfunction Control**—proving that the optimal control is given by the eigensystem, where the long-horizon component is determined solely by $\phi_0$ with exponentially decaying corrections (symmetric LQR even yields closed-form solutions); (4) **Hybrid Solving**—directly using the control provided by $\phi_0$ when far from the terminal time, and switching back to existing FBSDE/IDO solvers for a correction term in short-range intervals near the terminal time, such that the total computational cost only scales with the short-range interval.

### Key Designs

**1. Cole–Hopf Linearization + Schrödinger Unitary Equivalence: Converting SOC into a Discrete-Spectrum Eigenvalue Problem**

The pain point is that the HJB equation is generally nonlinear, making it difficult to utilize linear structures like "eigen-expansion." The authors first parameterize $V(x,t)=-\beta^{-1}\log\psi$, which cancels the nonlinear term of the HJB to obtain the linear PDE:
$$\partial_\tau\psi + L\psi = 0,\quad L\psi = -\mathrm{Tr}(\sigma\sigma^\top\nabla^2\psi) - 2\beta\, b^\top\nabla\psi + 2\beta^2 f\cdot\psi,$$
where $\tau=(2\beta)^{-1}(T-t)$ and the optimal control is $u^*=\partial_x\log\psi$. A key step is assuming **(A1) the drift is the gradient of some potential function** $b=-\nabla E$. On the weighted space $L^2(\mu)$ (with $\mu(x)=e^{-2\beta E(x)}$), $L$ becomes symmetric, and there exists a unitary operator $U:\psi\mapsto e^{-\beta E}\psi$ such that
$$ULU^{-1} = -\Delta + \beta^2\|\nabla E\|^2 - \beta\Delta E + 2\beta^2 f =: S = -\Delta + \mathcal V.$$
This means $L$ is unitarily equivalent to the Schrödinger operator $S$ well-studied in quantum mechanics. Under mild conditions (A2) (effective potential $\mathcal V$ is locally $L^2$, lower-bounded, and approaches $+\infty$ at infinity), $S$ (and thus $L$) possesses a countable orthogonal eigenbasis, eigenvalues that are lower-bounded without finite accumulation points, and a simple lowest eigenvalue with a strictly positive eigenfunction—precisely the spectral properties needed to reduce SOC to an eigenvalue problem. This step is the theoretical foundation: it is not a "metaphor from physics" but an **exact reduction** of the control operator's spectral problem via unitary equivalence.

**2. Eigenfunction Control Formula and Symmetric LQR Closed-Form Solution: Long-Horizon Control Determined by a Single $\phi_0$**

With the discrete spectrum established, the authors prove (Theorem 3) that the optimal control can be written as a series of the eigensystem:
$$u^*(x,t)=\beta^{-1}\Big(\nabla\log\phi_0(x) + \nabla\log\big(1+\textstyle\sum_{i>0}c_i\, e^{-\frac{1}{2\beta}(\lambda_i-\lambda_0)(T-t)}\tfrac{\phi_i(x)}{\phi_0(x)}\big)\Big),$$
where $\lambda_0<\lambda_1\le\cdots$. When $t\ll T$, all correction terms for $i>0$ decay exponentially at the rate $\lambda_i-\lambda_0$ (the spectral gap), making the **long-horizon optimal control $\beta^{-1}\nabla\log\phi_0$**—a value independent of $T$. This directly addresses the issue of computational cost being tied to $T$.

Furthermore, when $E$ and $f$ are quadratic forms (Symmetric LQR), the corresponding Schrödinger operator is exactly the **Hamiltonian of a quantum harmonic oscillator**, which has classical closed-form solutions for its eigenvalues and eigenfunctions (expressed via Hermite polynomials $H_{\alpha_i}$):
$$\phi_\alpha(x)\propto \exp\!\big(-\tfrac{\beta}{2}x^\top(-A+U^\top\Lambda^{1/2}U)x\big)\prod_i H_{\alpha_i}\big(\sqrt{\beta}(\Lambda^{1/4}Ux)_i\big),\quad \lambda_\alpha=\beta\big(-\mathrm{Tr}(A)+\textstyle\sum_i\Lambda_{ii}^{1/2}(2\alpha_i+1)\big).$$
Combined with Theorem 3, this provides closed-form optimal control for systems with symmetric linear drift + quadratic running cost + **arbitrary terminal cost**—whereas the classical LQR solution requires the terminal cost to be quadratic as well.

**3. Relative Eigenfunction Loss: Eliminating Implicit Reweighting Bias in Existing Losses**

For general gradient drifts without closed-form solutions, neural networks must learn $\phi_0$. Existing losses have critical flaws. The authors parameterize the eigenfunction as $\phi=\exp(-\beta V_0)$ (where $V_0$ is the network, naturally ensuring $\phi>0$). Substituting this, it can be calculated that both PINN residual loss $\|L\phi-\lambda\phi\|^2$ and variational/Ritz loss $\langle\phi,L\phi\rangle$ carry an $e^{-\beta V_0}$ factor:
$$R^\rho_{\mathrm{PINN}}(e^{-\beta V_0}) = 4\beta^4\big\|e^{-\beta V_0}(\mathcal K V_0-\tfrac{\lambda_0}{2\beta^2})\big\|_\rho^2 + \alpha R_{\mathrm{reg}}.$$
In regions where $V_0$ is large (i.e., high value function areas that require the most precise control), this factor approaches 0, causing the loss to **ignore** errors in those regions and leading to control failure in high-cost areas. The solution is straightforward—converting the residual to a **relative residual**:
$$R^\rho_{\mathrm{Rel}}(\phi)=\Big\|\tfrac{L\phi}{\phi}-\lambda\Big\|_\rho^2 + \alpha R_{\mathrm{reg}}(\phi).$$
By substituting $\phi=e^{-\beta V_0}$, the exponential factor that "blinds" the loss is eliminated: $R^\rho_{\mathrm{Rel}}(e^{-\beta V_0})=4\beta^4\|\mathcal K V_0-\tfrac{\lambda_0}{2\beta^2}\|_\rho^2+\alpha R_{\mathrm{reg}}$, ensuring sensitivity even in regions where $\phi$ is small. In practice, the variational loss is used first to estimate $\lambda_0$ and provide a good initialization for $V_0$, followed by fine-tuning with the relative loss.

**4. Hybrid Solver: Eigenfunctions for Long-Horizon, SDE Solvers for Short-Horizon, Complexity $O(Td)\to O(d)$**

Eigenfunction control performs exceptionally well as $t\to 0$ (long-horizon) but is inaccurate as $t$ approaches the terminal time $T$. In contrast, FBSDE/IDO maintains roughly constant error across $[0,T]$ but requires simulating the SDE at every step. The authors combine these strengths: choosing a cutoff time $T_{\mathrm{cut}}<T$, the control is formulated as:
$$u_\theta(x,t)=\begin{cases}\beta^{-1}\nabla\log\phi_0^{\theta_0}, & t\le T_{\mathrm{cut}},\\[2pt]\beta^{-1}\big(\nabla\log\phi_0^{\theta_0}+e^{-\frac{1}{2\beta}(\lambda_1-\lambda_0)(T-t)}v^{\theta_1}(x,t)\big), & T_{\mathrm{cut}}<t\le T.\end{cases}$$
The long-horizon segment uses the learned $\phi_0$, while the short-horizon segment runs IDO/FBSDE only on $[T_{\mathrm{cut}},T]$ to learn an additive correction network $v^{\theta_1}$. This reduces the total time complexity from $O(Td)$ to $O(d)$.

### Loss & Training
Training occurs in two stages: first, coarse training with variational loss (Deep Ritz) to obtain the $\lambda_0$ estimate and $V_0$ initialization, followed by fine-tuning the principal eigenfunction with the relative loss $R_{\mathrm{Rel}}$. Simultaneously, the first two eigenvalues $\lambda_0, \lambda_1$ are estimated to set $T_{\mathrm{cut}}$ and the correction decay factor. The short-range correction network $v^{\theta_1}$ is integrated into the standard IDO/FBSDE training workflow, optimized only on $[T_{\mathrm{cut}},T]$.

## Key Experimental Results

### Main Results
The authors evaluate the method on four high-dimensional ($d=20$) long-horizon benchmarks: QUADRATIC (isotropic/anisotropic), DOUBLE WELL, and RING. These were adapted from the short-horizon problems of Nüsken & Richter (2021) by extending the time horizon. The results show that the relative loss is significantly superior to existing eigenfunction losses in approximating $\nabla\log\phi_0$. The hybrid algorithm, combining the learned eigenfunction with IDO, achieves lower $L^2$ control error in every setting—**typically by an order of magnitude**—while reducing memory/run-time complexity from $O(Td)$ to $O(d)$.

The distribution of error over $t\in[0,T]$ confirms the design motivation: the pure eigenfunction method is optimal as $t\to0$ (long-horizon) and degrades near $T$; IDO has constant error; the hybrid method leverages both, achieving the lowest overall $L^2$ error.

| Method | Algorithm / Loss | Control Objective (Lower is better) |
|------|-------------|----------------------|
| EIGF (Ours) | Relative Loss | **73.33 ± 0.02** |
| IDO | Log-variance | 74.52 ± 0.02 |
| IDO | Adjoint Matching | 74.69 ± 0.02 |
| IDO | Relative Entropy | 75.63 ± 0.02 |
| IDO | SOCM | Did not converge |

The table above shows the "Opinion Dynamics (De Groot Model)" task: $N=10$ agents, state $X_t\in\mathbb R^{10}$, symmetric interaction matrix $L$, non-quadratic running cost $f(x)=\sum_i(x_i^2-1)^2$, zero terminal cost, and $T=10$. Our method achieves the lowest objective value, whereas the SOCM baseline fails to converge.

### Ablation Study

| Configuration | Key Phenomenon | Description |
|------|----------|------|
| Relative loss (Ours) | Correct control direction in high $V_0$ | Eliminates $e^{-\beta V_0}$ implicit reweighting |
| PINN residual loss | Control collapse in high $V_0$ | Loss is "blind" to high-cost regions |
| Variational / Ritz loss | Control collapse in high $V_0$ | Also carries the $e^{-\beta V_0}$ factor |
| Pure eigenfunction | Good at $t\to0$, poor near $T$ | Lacks short-range correction |
| Hybrid method (Ours) | Lowest overall $L^2$ error | Long-horizon $\phi_0$ + Short-horizon IDO |

### Key Findings
- **Implicit reweighting is the root cause of existing loss failures**: PINN/variational losses fail in high-cost areas—where control is most needed—due to the $e^{-\beta V_0}$ factor. The relative loss removes this factor and the resulting bias.
- **Task division is optimal**: Errors over $t$ show that eigenfunctions and SDE solvers are complementary; the hybrid approach is globally superior.
- **Diminishing returns for additional eigenfunctions**: Learning the full spectrum is costly and yields diminishing returns. Using just the principal eigenfunction $\phi_0$ (with short-range correction) is sufficient.

## Highlights & Insights
- **Unitary equivalence brings SOC into the quantum mechanics toolbox**: This is an exact reduction, not an analogy. Once $L\cong S=-\Delta+\mathcal V$ is proven, the full suite of spectral properties (discrete spectrum, orthogonal basis, etc.) can be directly utilized.
- **The "long-horizon = steady-state eigenfunction" perspective** is highly transferable: Any problem solved by rolling back in time that can be linearized into an operator with a spectral gap can potentially use a principal eigenfunction to characterize long-horizon behavior, swapping $T$-dependence for spectral-gap dependence.
- **The relative loss is a versatile trick**: Replacing $\|L\phi-\lambda\phi\|$ with the relative residual $\|L\phi/\phi-\lambda\|$ addresses the problem of exponential factors in parameterization blinding the loss, which is applicable to other PDE learning tasks.
- **Closed-form solutions for Symmetric LQR with arbitrary terminal costs** serve as an independent theoretical contribution, removing the "terminal cost must be quadratic" constraint of classical LQR.

## Limitations & Future Work
- **Limitations**: The method is currently limited to gradient drift problems. When $L$ is non-symmetric, real eigenvalues are no longer guaranteed (though the principal eigenfunction may still be real and non-degenerate). Furthermore, $T_{\mathrm{cut}}$ lacks an a priori determination method and is a hyperparameter dependent on the spectral gap $\lambda_1-\lambda_0$ and the specific application.
- **Autonomous Observations**: Experiments focused on $d=20$ synthetic benchmarks and a $N=10$ opinion model; the scale is still relatively small. The advantage of eigenfunctions diminishes in systems with very small spectral gaps (weak mixing), where corrections decay slowly, requiring a larger $T_{\mathrm{cut}}$.
- **Future Directions**: Generalizing the unitary equivalence framework to non-gradient drifts (general non-symmetric $L$) and designing adaptive strategies for $T_{\mathrm{cut}}$ based on online spectral gap estimation.

## Related Work & Insights
- **vs. IDO / FBSDE**: These treat SOC as dynamic programming, where computational cost scales linearly with $T$ and performance degrades over long horizons. This work uses eigenfunctions for long-horizon steady states, calling SDE solvers only for short-range intervals.
- **vs. Existing Eigenfunction Losses (PINN, Deep Ritz)**: These losses exhibit implicit reweighting under $\phi=e^{-\beta V_0}$ parameterization, failing in high-cost regions. The relative loss eliminates this factor.
- **vs. Early Schrödinger Operator Approaches**: This work specifically targets **finite-time SOC with arbitrary terminal costs** and provides a practical algorithm combining learnable eigenfunctions with short-range corrections.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses unitary equivalence for exact reduction of long-horizon SOC to the Schrödinger spectral problem and provides closed-form LQR solutions for arbitrary terminal costs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple settings + opinion model showing an order-of-magnitude improvement; however, the scale ($d=20$) is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical chain (linearization → spectral verification → eigenfunction control → hybrid algorithm) and lucid failure analysis.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for long-horizon SOC that avoids $T$-dependence; the relative loss trick is independently valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control](../../NeurIPS2025/optimization/mdns_masked_diffusion_neural_sampler_via_stochastic_optimal_control.md)
- [\[ICLR 2026\] Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise](dual_optimistic_ascent_pi_control_is_the_augmented_lagrangian_method_in_disguise.md)
- [\[ICLR 2026\] Differentiable Model Predictive Control on the GPU](differentiable_model_predictive_control_on_the_gpu.md)
- [\[ICLR 2026\] Nesterov Finds GRAAL: Optimal and Adaptive Gradient Method for Convex Optimization](nesterov_finds_graal_optimal_and_adaptive_gradient_method_for_convex_optimizatio.md)
- [\[ICLR 2026\] Multilevel Control Functional](multilevel_control_functional.md)

</div>

<!-- RELATED:END -->
