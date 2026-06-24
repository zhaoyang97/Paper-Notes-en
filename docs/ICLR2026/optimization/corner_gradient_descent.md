---
title: >-
  [Paper Note] Corner Gradient Descent
description: >-
  [ICLR2026][Optimization][Stochastic Gradient Descent] This paper proposes a "contour perspective" that equates generalized (S)GD with arbitrary linear memory to a contour $\gamma$ on the complex plane (via the response mapping $\Psi=P/Q$). It proves that by forming a "sharp corner" at the origin with an exterior angle $\theta\pi$ ($1<\theta<2$), the loss convergence rate of SGD on power-law quadratic problems can be accelerated from $O(t^{-\zeta})$ to $O(t^{-\theta\zeta})$. T…
tags:
  - "ICLR2026"
  - "Optimization"
  - "Stochastic Gradient Descent"
  - "convergence acceleration"
  - "power-law spectrum"
  - "complex plane contour"
  - "infinite-memory optimizer"
date: 2026-05-08
content_hash: 9244d8682bac21da
---

# Corner Gradient Descent

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=nOXCfIdhD9](https://openreview.net/forum?id=nOXCfIdhD9)  
**Code**: Jupyter notebook included with supplementary materials (no independent repository)  
**Area**: optimization  
**Keywords**: Stochastic Gradient Descent, convergence acceleration, power-law spectrum, complex plane contour, infinite-memory optimizer  

## TL;DR
This paper proposes a "contour perspective" that equates generalized (S)GD with arbitrary linear memory to a contour $\gamma$ on the complex plane (via the response mapping $\Psi=P/Q$). It proves that by forming a "sharp corner" at the origin with an exterior angle $\theta\pi$ ($1<\theta<2$), the loss convergence rate of SGD on power-law quadratic problems can be accelerated from $O(t^{-\zeta})$ to $O(t^{-\theta\zeta})$. The authors provide a precise phase diagram for the optimal acceleration factor $\theta_{\max}=\min(2,\nu,\frac{2}{\zeta+1/\nu})$ and utilize rational approximation to compress the ideal infinite-memory "corner algorithm" into a practical finite-memory version, validated on synthetic problems and MNIST.

## Background & Motivation
**Background**: On ill-conditioned quadratic problems, the loss convergence of deterministic Gradient Descent (GD) follows a power law $L_t=O(t^{-\zeta})$. Classical results (Polyak’s Heavy Ball + Nemirovskiy–Polyak/Brakhage’s non-stationary Jacobi scheduling) show that adding momentum and a time-varying "Jacobi schedule" to GD can double the exponent, accelerating it to $O(t^{-2\zeta})$, which is the best achievable result for non-adaptive scheduling.

**Limitations of Prior Work**: Modern machine learning relies on mini-batch Stochastic Gradient Descent (SGD), where the aforementioned acceleration fails. Non-stationary Jacobi Heavy Ball **eventually diverges** under SGD due to the continuous accumulation of sampling noise. In other words, the "exponent doubling" acceleration in the deterministic world collapses in the stochastic world. To date, no known algorithm can accelerate SGD to $O(t^{-2\zeta})$ on infinite-dimensional problems.

**Key Challenge**: The root cause is the coexistence of two forces in SGD: **signal propagation** (purely reducing error in the absence of noise) and **noise propagation** (sampling noise injected at each step and amplified by the algorithm). Any means of accelerating signal convergence (e.g., increasing effective learning rate or momentum) tends to amplify noise simultaneously, causing the convergence curve to stagnate or diverge once noise dominates. Yarotsky & Velikanov (2024) proved that all **stationary** finite-memory generalized SGD methods (including momentum and its variants) share the same phase diagram as vanilla SGD and can only provide $O(t^{-\zeta})$ in the signal-dominated phase, **failing to accelerate the exponent**.

**Goal**: To construct a stationary algorithm that can truly lift the exponent from $\zeta$ to $\theta\zeta$ (with $\theta$ approaching 2) in the signal-dominated phase, and to characterize the precise boundary of "how far acceleration can go" under sampling noise constraints.

**Key Insight**: Instead of manually tuning matrix/recursive forms (parameters like $\alpha, b, c, D$), the authors adopt a purely geometric perspective—compressing all "fingerprints" of generalized SGD into a response mapping $\Psi(\mu)=P(\mu)/Q(\mu)$ on the complex plane, which maps the unit circle to a contour $\gamma$. All information about the loss is determined by the shape of this contour. Designing an algorithm thus becomes "designing a contour," where the **local geometry** of the contour (especially near the origin) directly determines the convergence exponent.

**Core Idea**: Create a sharp corner at the origin of the contour with an exterior angle $\theta\pi$. Use this single geometric parameter, the "sharpness $\theta$," to simultaneously characterize acceleration and noise amplification, then analytically solve for the optimal $\theta$ by balancing the two.

## Method

### Overall Architecture
This work focuses on quadratic loss $L(w)=\frac12 w^T H w - w^T q$ on infinite-dimensional Hilbert spaces, where the Hessian $H$ has a discrete positive spectrum $\lambda_k\downarrow 0$ satisfying two power-law conditions: the capacity condition $\lambda_k=\Lambda k^{-\nu}(1+o(1))$ and the source condition $\sum_{k:\lambda_k<\lambda}\lambda_k(e_k^Tw^*)^2=Q\lambda^\zeta(1+o(1))$. Intuitively, $\zeta$ represents the "inverse of the effective condition number" (smaller is harder), and $\nu$ represents the "inverse of the effective dimension."

The workflow follows: **(1) Equating stationary (S)GD with memory size $M$ to a complex plane contour**—proving that all information for loss evolution is contained in the response mapping $\Psi=P/Q$ via the characteristic polynomial $\chi(\mu,\lambda)=P(\mu)-\lambda Q(\mu)$; **(2) Designing "corner" contours**, proving that the exterior angle $\theta\pi$ directly changes the decay exponent of the signal propagator from $\zeta$ to $\theta\zeta$, while changing the noise propagator exponent to $2-\theta/\nu$; **(3) Balancing signal and noise** by taking the lower bound of the two exponents to find the optimal $\theta$ and the acceleration phase diagram; **(4) Implementation via rational approximation**, using fast rational approximations of $z^\theta$ to compress the ideal irrational mapping into a feasible finite-memory algorithm.

The theoretical backbone is the "propagator expansion" of the average SGD loss: under the Spectrally-Expressible (SE) approximation (and taking $\tau_2=0$),

$$L_t=\tfrac12\Big(V_{t+1}+\sum_{m=1}^{t}\sum_{0<t_1<\dots<t_m<t+1}U_{t+1-t_m}\cdots U_{t_2-t_1}V_{t_1}\Big),$$

where the **signal propagator** $V_t$ describes error reduction in the absence of noise, and the **noise propagator** $U_t$ describes fluctuations from sampling noise injected at each step. The asymptotic exponent of $L_t$ is determined by **whichever decays slower** between $U_t$ and $V_t$ (Theorem 1), which is the mathematical source of acceleration being hindered by noise. The batch size $|B|$ only affects the coefficient of $U_t$ (denominator $|B|$); as $|B|\to\infty$, $U_t\equiv 0$, reducing to deterministic GD.

### Key Designs

**1. Contour Perspective: Encoding Generalized (S)GD as a Response Mapping $\Psi$**

Directly tuning algorithm parameters $(\alpha, b, c, D)$ obscures what determines the convergence rate. The authors geometricize this via a residue/contour integral identity. Noting that terms like $(\,1\ 0^T)(\mu-S_\lambda)^{-1}(\cdot)$ recur in propagators, simple calculation yields:

$$(\,1\ 0^T)(\mu-S_\lambda)^{-1}\binom{-\alpha}{c}=\frac{Q(\mu)}{P(\mu)-\lambda Q(\mu)}=\frac{1}{\Psi(\mu)-\lambda},\qquad \Psi(\mu)=\frac{P(\mu)}{Q(\mu)}.$$

Thus, the noise propagator can be written as a contour integral depending only on $\Psi$: $U_t=\frac{\tau_1}{|B|}\sum_k\lambda_k^2\big|\frac{1}{2\pi i}\oint_{|\mu|=1}\frac{\mu^{t-1}d\mu}{\Psi(\mu)-\lambda_k}\big|^2$, and similarly for $V_t$. The key insight is that since $P$ can be any monic polynomial of degree $(M{+}1)$ with $P(1)=0$, and $Q$ can be any polynomial of degree $\le M$, "designing a stationary SGD with memory" is essentially "designing a rational function $\Psi$." Viewing $\Psi$ as a curve $\gamma = \Psi(\{|\mu|=1\})$, the stability condition is equivalent to $\gamma$ not touching the spectrum $[0, \lambda_{\max}]$. Conversely, given a Jordan contour $\gamma$, there exists a unique conformal mapping $\Psi_\gamma$ satisfying $\Psi(\infty)=\infty$ and $\Psi(1)=0$, allowing the **reconstruction of the algorithm** from the contour.

**2. Corner Algorithm: Using an Exterior Angle $\theta\pi$ to Pull the Exponent from $\zeta$ to $\theta\zeta$**

With the contour perspective, the authors seek the local geometry required for acceleration. In the signal-dominated phase, the loss coefficient $C_L$ decreases as the effective learning rate $\alpha_{\text{eff}}=-(\frac{d\Psi}{d\mu}(1))^{-1}$ increases, which means making $-\frac{d\Psi}{d\mu}(1)$ as small as possible. The natural choice is a power-law singularity at $\mu=1$ (corresponding to the origin of the contour):

$$\Psi(\mu)=-c_\Psi(\mu-1)^\theta(1+o(1)),\qquad \mu\to 1,$$

which corresponds to the contour opening an exterior angle $\theta\pi$ at the origin. Since $-\frac{d\Psi}{d\mu}\sim c\theta(\mu-1)^{\theta-1}$, making the derivative approach $+0$ at $\mu=1$ (increasing $\alpha_{\text{eff}}$) requires $\theta>1$. Stability prevents $\theta>2$. Thus, the feasible interval is $\theta \in [1, 2]$. Theorem 3 proves that under a power-law spectrum, the corner mapping yields strictly power-law propagators: **Signal** $V_t=C_V t^{-\theta\zeta}(1+o(1))$ and **Noise** $U_t=C_U t^{\theta/\nu-2}(1+o(1))$. The sharpness $\theta$ pulls both; a sharper corner (larger $\theta$) speeds up signal decay (good) but slows down noise decay (bad).

**3. Acceleration Phase Diagram: Balancing Signal and Noise for Optimal $\theta_{\max}$**

The exponent of $L_t$ is $\min(\theta\zeta, 2-\theta/\nu)$. The optimal $\theta$ occurs when $\theta\zeta=2-\theta/\nu$. Additionally, the noise exponent $2-\theta/\nu$ must be $>1$ (otherwise total noise $U_\Sigma = \sum_t U_t = \infty$). Theorem 4 characterizes the optimal acceleration in the signal-dominated phase ($\nu>1, 0<\zeta<2-1/\nu$):

$$\theta_{\max}=\min\Big(2,\ \nu,\ \frac{2}{\zeta+1/\nu}\Big),$$.

The phase is divided into: **(I) Full Acceleration** $\theta_{\max}=2$ (where exponent doubling is approached); **(II) Signal/Noise Balance** $\theta_{\max}=\frac{2}{\zeta+1/\nu}<2$; **(III) Limited by $U_\Sigma$ Finiteness** $\theta_{\max}=\nu<2$.

**4. Finite-Memory Approximation: Practical Implementation via Rational Approximation of $z^\theta$**

Ideal corner mappings $\Psi$ are irrational functions requiring infinite memory. The authors use results from Newman (1964) / Gopal & Trefethen (2019) showing that $z^\theta$ can be approximated by $M$-degree rational functions with $O(e^{-c\sqrt{M}})$ error. By discretizing a $\theta$-corner mapping integral using a trapezoidal rule with $M$ nodes, they obtain the rational function $\Psi^{(M)}=P/Q$. Proposition 2 provides explicit memory-$M$ algorithm parameters. The overhead is $MW$ extra scalars for memory ($W$ is weight count), which is negligible when $|B|\gg M$.

## Key Experimental Results

Experiments use the memory $M=5$ approximation.

### Synthetic Indicator Problem
Fitting $y(x)=\mathbf{1}_{[1/4,3/4]}(x)$ with a shallow ReLU network ($N=10^5$). $\zeta=1/4, \nu=4$, falling into Phase I ($\theta_{\max}=2$).

| Algorithm | Measured Loss Exponent | Theoretical Value |
|------|------|------|
| Plain SGD | $-0.247$ | $\zeta=0.25$ |
| Corner SGD ($\theta=1.8$) | $-0.408$ | $\theta\zeta=0.45$ |

### Main Results: MNIST
Single-hidden-layer ReLU network ($H=1000$). Empirical $\zeta\approx0.25, \nu\approx1.3$, falling into Phase III ($\theta_{\max}=\nu$).

| Configuration | $|B|=1000$ Loss Exponent | $|B|=100$ Loss Exponent |
|------|------|------|
| Plain SGD | $-0.274$ | $-0.271$ |
| Corner SGD $\theta=1.3$ | $-0.409$ | $-0.411$ |
| Corner SGD $\theta=1.8$ | $-0.414$ | $-0.281$ (Unstable) |

## Key Findings
- Corner SGD with $\theta=1.3$ consistently accelerates the exponent by approximately $\times 1.5$ ($0.27\to0.41$).
- $\theta=1.8$ becomes unstable at $|B|=100$, confirming that larger $\theta$ amplifies noise and requires larger batches.
- In finite-dimensional problems like MNIST, $U_\Sigma$ is always finite, allowing $\theta > \nu$ to gain extra acceleration if $|B|$ is sufficiently large.

## Highlights & Insights
- **From Algorithm Tuning to Contour Drawing**: The framework maps generalized SGD with linear memory to complex plane contours. Traditional methods like Heavy Ball are low-order special cases (circles/ellipses).
- **Single Parameter $\theta$ for Trade-offs**: The exterior angle $\theta$ encodes the "acceleration vs. noise" trade-off into a univariate optimization, leading to a closed-form $\theta_{\max}$.
- **Irrational Target + Rational Approximation**: This paradigm allows designing ideal limit objects and compressing them into $M=5$ memory implementations, which is transferable to other optimizer designs.

## Limitations & Future Work
- **Theoretical Limit**: Finite-memory stationary algorithms eventually revert to $O(t^{-\zeta})$ as $t\to\infty$. True asymptotic acceleration likely requires non-stationary approximations.
- **Assumptions**: Relies on SE approximation and $\tau_2=0$. While the authors suggest Theorem 3 holds for large batches when $\tau_2 \neq 0$, a complete proof is pending.
- **Scope**: Primarily established for quadratic/linear regimes (or wide NTK regimes). Feature learning in highly non-linear models may introduce different mechanisms.

## Related Work & Insights
- **vs. Non-stationary Jacobi Heavy Ball**: Jacobi scheduling achieves $O(t^{-2\zeta})$ in GD but diverges in SGD. This work uses **stationary infinite-memory** algorithms to achieve stable $O(t^{-\theta\zeta})$ under noise.
- **vs. Yarotsky & Velikanov (2024)**: While they proved finite-memory stationary SGD cannot accelerate exponents, this work breaks through by moving to the infinite-memory limit via contour geometry.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A fresh perspective using contour geometry to unify generalized SGD acceleration.
- Experimental Thoroughness: ⭐⭐⭐ Validated on synthetic and MNIST, but limited to approximately linear regimes.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theory but high mathematical barrier.
- Value: ⭐⭐⭐⭐ Provides fundamental answers to "how much can SGD accelerate" with a transferable implementation paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Convergence Direction of Gradient Descent](on_the_convergence_direction_of_gradient_descent.md)
- [\[ICLR 2026\] Fast Data Mixture Optimization via Gradient Descent](fast_data_mixture_optimization_via_gradient_descent.md)
- [\[ICLR 2026\] Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference](adaptive_gradient_descent_on_riemannian_manifolds_and_its_applications_to_gaussi.md)
- [\[ICLR 2026\] Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking](egalitarian_gradient_descent_a_simple_approach_to_accelerated_grokking.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](../../ICML2026/optimization/on_the_convergence_rate_of_lora_gradient_descent.md)

</div>

<!-- RELATED:END -->
