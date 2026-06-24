---
title: >-
  [Paper Note] Quantum Optimization via Gradient-Based Hamiltonian Descent
description: >-
  [ICML2025][Optimization][Quantum Hamiltonian Descent] Integrates gradient information into the Quantum Hamiltonian Descent (QHD) framework to propose gradient-based QHD, achieving convergence rates at least one order of magnitude faster and a higher success probability of finding global optima than the original QHD and classical methods (NAG, SGDM) in both convex and non-convex optimization.
tags:
  - "ICML2025"
  - "Optimization"
  - "Quantum Hamiltonian Descent"
  - "Gradient Information"
  - "Lyapunov Convergence Analysis"
  - "Schrödinger Equation"
  - "Non-Convex Optimization"
  - "Quantum Algorithms"
date: 2026-05-08
content_hash: 2053bd92e6847660
---

# Quantum Optimization via Gradient-Based Hamiltonian Descent

**Conference**: ICML2025  
**arXiv**: [2505.14670](https://arxiv.org/abs/2505.14670)  
**Code**: [jiaqileng/Gradient-Based-QHD](https://github.com/jiaqileng/Gradient-Based-QHD)  
**Area**: Quantum Optimization / Optimization  
**Keywords**: Quantum Hamiltonian Descent, Gradient Information, Lyapunov Convergence Analysis, Schrödinger Equation, Non-Convex Optimization, Quantum Algorithms

## TL;DR

Integrates gradient information into the Quantum Hamiltonian Descent (QHD) framework to propose gradient-based QHD, achieving convergence rates at least one order of magnitude faster and a higher success probability of finding global optima than the original QHD and classical methods (NAG, SGDM) in both convex and non-convex optimization.

## Background & Motivation

- **Bottlenecks of Classical Optimization**: First-order gradient methods (GD, NAG) easily fall into local extrema or saddle points in highly non-convex landscapes, failing to guarantee global convergence.
- **Quantum Hamiltonian Descent (QHD)**: Leng et al. proposed to quantize the Bregman Lagrangian corresponding to Nesterov's accelerated gradient method via canonical quantization, and evolve the quantum wave function using the Schrödinger equation to escape local extrema via quantum tunneling. However, the original QHD only utilizes function values (zero-order information), resulting in slow convergence rates.
- **Inspiration from High-Resolution ODEs**: The high-resolution ODE framework by Shi et al. reveals that the acceleration mechanism of NAG stems from the coupling of gradient terms in the Lyapunov function. Inspired by this, this work explicitly embeds gradient information into the quantum Hamiltonian to construct stronger quantum optimization dynamics.

## Method

### 1. Classical Lagrangian with Gradients

Adding a gradient coupling term on the basis of the standard Bregman Lagrangian:

$$
\mathcal{L}(t,X,\dot{X}) = \frac{t^3}{2}|\dot{X}|^2 - \alpha t^3 \dot{X}^\top \nabla f(X) - \frac{\beta t^3}{2}|\nabla f(X)|^2 - (t^3 + \gamma t^2)f(X)
$$

where $\alpha, \beta, \gamma$ are adjustable parameters. When $\alpha = \beta = \gamma = 0$, it degenerates to the original QHD.

### 2. Legendre Transform to Obtain Hamiltonian

$$
H(t,X,P) = \frac{1}{2}\|t^{-3/2}P + \alpha t^{3/2}\nabla f\|^2 + \frac{\beta t^3}{2}\|\nabla f\|^2 + (t^3 + \gamma t^2)f(X)
$$

### 3. Canonical Quantization → Gradient-Based QHD

Substituting the classical momentum $P \mapsto \hat{p} = -i\nabla$ yields the gradient-containing quantum Hamiltonian operator:

$$
\hat{H}(t) = \frac{1}{2}\sum_{j=1}^d A_j^2 + \frac{\beta}{2}t^3\|\nabla f\|^2 + (t^3 + \gamma t^2)f
$$

where $A_j = t^{-3/2}\hat{p}_j + \alpha t^{3/2}\hat{v}_j$, and $\hat{v}_j$ is the multiplicative operator of the gradient component. The evolution is driven by the time-dependent Schrödinger equation:

$$
i\partial_t \Psi(t,x) = \hat{H}(t)\Psi(t,x)
$$

### 4. Discretized Quantum Algorithm (Algorithm 1)

Decomposing $\hat{H}(t_k)$ into three terms: kinetic energy $H_{k,1} = -\frac{1}{2t_k^3}\Delta$, gradient coupling $H_{k,2} = \frac{\alpha}{2}\{-i\nabla, \nabla f\}$, and potential energy $H_{k,3}$, and then gradually evolving via product formulas (operator splitting):

$$
e^{-ih\hat{H}(t_k)} \approx e^{-ihH_{k,1}} e^{-ihH_{k,2}} e^{-ihH_{k,3}}
$$

Finally, measuring the position observable of the quantum state outputs the candidate solution $\xi \in \mathbb{R}^d$.

### 5. Convergence Theory

**Theorem 1 (Function Value Convergence, Convex Case)**: When $\beta=0$, $\gamma \geq \max(3\alpha, 0)$,

$$
\mathbb{E}[f(X_t)] \leq \frac{\mathscr{K}_0 + \mathscr{D}_0}{t^2 + \omega t} = O(t^{-2})
$$

where $\omega = \gamma - 3\alpha$. The proof relies on constructing the quantum operator Lyapunov function $\mathcal{E}(t) = \langle \hat{O}(t)\rangle_t$ and proving its monotonic decrease.

**Theorem 4 (Gradient Norm Convergence)**: When $\beta > 0$ and the gradient norm function $G(x) = \|\nabla f\|^2$ satisfies the convexity condition,

$$
\mathbb{E}[\|\nabla f(X_t)\|^2] \leq \frac{2(\mathscr{K}_0 + \mathscr{D}_0')}{\beta t^2} = O(t^{-2})
$$

### 6. Gate Complexity

**Theorem 6**: $K$ iterations require $O(K)$ zero-order quantum oracle queries and $\widetilde{O}(\alpha d h K L)$ first-order quantum oracle queries. The complexity per step scales linearly with the problem dimension $d$, making the algorithm scalable to large-scale problems.

## Key Experimental Results

| Test Function | Type | Landscape Characteristics | Grad-QHD vs QHD | Grad-QHD vs NAG/SGDM |
|---|---|---|---|---|
| Convex Function $(x+y)^4/256+(x-y)^4/128$ | Convex, non-strongly convex | Singular Hessian at global extremum | Function value/gradient norm converges significantly faster | Substantial lead |
| Styblinski-Tang | Non-convex | 3 local extrema + 1 global extremum | Significantly higher success probability, more concentrated solution distribution | One order of magnitude higher |
| Michalewicz | Non-convex | Flat plateau + hidden global extremum in a narrow valley | Superior in both function value and success probability | Clear advantage |
| Cube-Wave | Non-convex | 10+ local extrema, 4 global extrema | Terminal function value is about 1 order of magnitude lower | Terminal function value is about 2 orders of magnitude lower |
| Rastrigin | Non-convex | Highly oscillatory landscape | Superior in both convergence rate and success probability | Substantial lead |

**Experimental Setup**: Step size $h=0.2$. Classical methods use 1000 independent runs to estimate expectations, while the quantum method calculates them directly via numerical integration of the wave function. Parameters $\alpha=-0.1$ (convex) / $\alpha=-0.05$ (non-convex), $\beta=0$, $\gamma=5$. Experiments were completed on a MacBook M4.

## Highlights & Insights

1. **Elegant Combination of Physical Intuition and Optimization**: Porting the gradient coupling acceleration mechanism from the high-resolution ODE framework to quantum mechanics, forming a complete theoretical pipeline from Lagrangian → Hamiltonian → canonical quantization.
2. **Non-Trivial Lyapunov Analysis**: Constructing active Lyapunov functions and proving monotonicity under the quantum mechanical (non-commuting operators) framework requires handling complex commutation relations (Lemma 3), featuring technical difficulty far exceeding the classical case.
3. **Parameter Flexibility**: $\alpha, \beta, \gamma$ are not limited to the corresponding values in high-resolution ODEs, offering a broader design space for dynamical exploration.
4. **Asymptotic Comparability**: The gate complexity per step is $\widetilde{O}(d)$, asymptotically comparable to the $O(d)$ of classical NAG, without introducing exponential quantum overhead.
5. **Global Optimization Capability**: Utilizing quantum tunneling combined with gradient guidance, the success probability on multiple non-convex benchmarks far exceeds that of classical methods.

## Limitations & Future Work

1. **Numerical Experiments Limited to 2D**: Due to the exponential overhead of quantum state simulation, all experiments are conducted in two dimensions, leaving high-dimensional advantages unverified.
2. **Lack of Discretization Convergence Theory**: Although continuous-time convergence is proven, the rigorous convergence analysis of the discrete-time algorithm (Algorithm 1) is left for future work.
3. **Lack of Theoretical Guidance for Step Size Selection**: Theoretical analysis suggests $h \sim t_k^{-3/2}$, but in experiments, a fixed step size already works, leaving a gap between theory and practice.
4. **Requirement of First-Order Quantum Oracles**: Compared to the original QHD (zero-order), an additional gradient oracle is required, the construction cost of which depends on the problem.
5. **No Verification on Practical Quantum Hardware**: The algorithm assumes a fault-tolerant quantum computer; feasibility on current NISQ devices has not been explored.
6. **Convexity Assumptions**: Theoretical convergence guarantees require either $f$ to be convex or $G(x) = \|\nabla f\|^2$ to be convex; theoretical results for the non-convex case remain an open problem.

## Related Work & Insights

- **Original QHD Work** [Leng et al.]: The direct foundation upon which this work is extended.
- **High-Resolution ODEs** [Shi et al.]: Continuous understanding of the NAG acceleration mechanism, serving as the inspiration for the Lyapunov function design.
- **Bregman Lagrangian** [Wibisono et al.]: A variational perspective unifying accelerated methods.
- **Quantum Simulation** [Childs et al., Berry et al.]: Algorithmic foundation for time-dependent Hamiltonian simulation.
- **QSVT** [Gilyén et al.]: Key quantum subroutine for implementing the gradient coupling term $H_{k,2}$.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Integrates gradient information into the quantum Hamiltonian framework, presenting a novel and elegant theoretical construction.
- Experimental Thoroughness: ⭐⭐⭐ — Diverse benchmarks but limited to 2D, lacking verification on high-dimensional and real-world problems.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear and complete logic spanning from physical motivation to mathematical derivation and algorithmic implementation.
- Value: ⭐⭐⭐⭐ — Provides a strong new direction for quantum optimization algorithm design, although practical utility is constrained by quantum hardware.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](../../NeurIPS2025/optimization/isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[ICML 2025\] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression](benefits_of_early_stopping_in_gradient_descent_for_overparameterized_logistic_re.md)
- [\[ICLR 2026\] Fast Data Mixture Optimization via Gradient Descent](../../ICLR2026/optimization/fast_data_mixture_optimization_via_gradient_descent.md)
- [\[ICLR 2026\] Corner Gradient Descent](../../ICLR2026/optimization/corner_gradient_descent.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)

</div>

<!-- RELATED:END -->
