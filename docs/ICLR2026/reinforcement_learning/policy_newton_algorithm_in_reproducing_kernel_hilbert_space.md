---
title: >-
  [Paper Note] Policy Newton Algorithm in Reproducing Kernel Hilbert Space
description: >-
  [ICLR 2026][Reinforcement Learning][RKHS] This paper proposes the first second-order policy optimization method in RKHS, dubbed Policy Newton in RKHS. By optimizing a cubic-regularized auxiliary objective, the method bypasses the infinite-dimensional Hessian inversion. Leveraging the Representer Theorem, the infinite-dimensional optimization is equivalently transformed into a finite-dimensional problem where the dimension scales with the trajectory data $NT$. The authors theo…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "RKHS"
  - "Second-order Optimization"
  - "Policy Newton"
  - "Representer Theorem"
  - "Quadratic Convergence"
date: 2026-05-08
content_hash: f5e72effd849246e
---

# Policy Newton Algorithm in Reproducing Kernel Hilbert Space

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PYnLd91wZY](https://openreview.net/forum?id=PYnLd91wZY)  
**Code**: To be confirmed (Anonymous supplementary material with full implementation provided)  
**Area**: Reinforcement Learning / Policy Optimization Theory  
**Keywords**: RKHS, Second-order Optimization, Policy Newton, Representer Theorem, Quadratic Convergence

## TL;DR
This paper proposes the first second-order policy optimization method in RKHS, dubbed Policy Newton in RKHS. By optimizing a cubic-regularized auxiliary objective, the method bypasses the infinite-dimensional Hessian inversion. Leveraging the Representer Theorem, the infinite-dimensional optimization is equivalently transformed into a finite-dimensional problem where the dimension scales with the trajectory data $NT$. The authors theoretically prove convergence to local optima with a local quadratic convergence rate. Empirically, the method demonstrates faster convergence and higher returns compared to first-order RKHS methods and parametric second-order methods.

## Background & Motivation

**Background**: Representing reinforcement learning policies in a Reproducing Kernel Hilbert Space (RKHS) is a powerful non-parametric approach. Defined in infinite-dimensional function spaces, it possesses universal approximation capabilities. Facilitated by the Representer Theorem, policy updates occur within a finite-dimensional subspace spanned by observed data points, allowing the model scale to adapt to task complexity. This representation is particularly attractive in safety-critical scenarios with limited samples or high robustness requirements, having achieved success in meta-RL and distributed RL.

**Limitations of Prior Work**: Optimization of RKHS policies has remained almost exclusively first-order. The current standard, RKHS Policy Gradient (Paternain et al., 2020), updates policies by adding functions derived from gradients to the RKHS. Consequently, it inherits the common pitfalls of first-order methods: slow convergence on complex optimization terrains with high curvature or narrow valleys. Moreover, gradient directions are not scale-invariant, often leading to pathologically scaled search directions.

**Key Challenge**: In parametric policies, second-order (Newton) methods significantly accelerate convergence and provide well-scaled updates by incorporating Hessian curvature information. However, directly extending Newton methods to RKHS faces a fundamental hurdle: the counterpart of the Hessian in RKHS is the second-order Fréchet derivative of the expected cumulative return. This is an **operator** acting on function spaces and is infinite-dimensional. A standard Newton step requires explicitly inverting this operator, which is generally infeasible in infinite dimensions. Fundamentally, this Hessian operator is trace-class and thus compact in RKHS; compact operators in infinite-dimensional spaces lack bounded inverses, rendering the standard Newton step itself ill-posed.

**Goal**: To develop the first implementable RKHS second-order policy optimization algorithm with convergence guarantees under the RL setting, where the data distribution shifts with the policy. Existing RKHS Newton research focuses on online learning/regret settings with i.i.d. data, which cannot be directly transferred to RL.

**Key Insight + Core Idea**: Instead of confronting the infinite-dimensional Hessian inversion head-on, the authors **optimize a cubic-regularized auxiliary objective** to derive the Newton step. They then use the **Representer Theorem** to prove that this infinite-dimensional optimization is equivalent to a problem in a finite-dimensional Euclidean space, with dimensions scaling only with the trajectory data $N \times T$. The core mechanism is "replacing infinite-dimensional Hessian inversion with the solution of a finite-dimensional cubic-regularized subproblem."

## Method

### Overall Architecture

The methodology addresses one question: how to obtain a second-order policy update step given an infinite-dimensional RKHS Hessian that cannot be explicitly represented or inverted? The approach follows three stages.

First, the policy is modeled directly as a function $h$ in the RKHS rather than a parameter vector $\theta$. For discrete action spaces, the policy is $\pi_h(a_t\mid s_t)=\frac{1}{Z}e^{T h(s_t,a_t)}$, where $Z=\sum_{a'}e^{T h(s_t,a')}$ is the normalization constant and $T$ is temperature. The first-order Fréchet derivative $\nabla_h J(\pi_h)$ remains an element in the RKHS expressible via kernel sections, while the second-order Fréchet derivative $\nabla_h^2 J(\pi_h)$ becomes an operator acting on the function space, viewed as an element in the tensor product space $\mathcal H_K\otimes\mathcal H_K$.

Second, acknowledging that $\nabla_h^2 J$ is not explicitly computable and its inverse is non-existent, the method **solves a cubic-regularized auxiliary function** to obtain the RKHS Newton step $\Delta h$ (see Key Design 1). This step transforms "inversion" into "optimization," though the variable remains in the infinite-dimensional space $\mathcal H_K$.

Third, the **Representer Theorem** collapses this infinite-dimensional optimization into a finite-dimensional problem. The optimal solution must take the form of a linear combination of data kernel sections $\bar h_\alpha=\sum_{i,t}\alpha^i_t K((s^i_t,a^i_t),\cdot)$, requiring only the finite coefficients $\bar\alpha\in\mathbb R^{NT}$. Substituting this into the auxiliary objective and utilizing the reproducing property, the quadratic and cubic terms are derived in explicit algebraic forms of $\bar\alpha$ (Theorem 3.3, see Key Design 2). Finally, this finite-dimensional quadratic problem with cubic regularization is solved via the Conjugate Gradient method (Key Design 3). After solving for $\bar\alpha$, $\Delta h$ is reconstructed, and the policy is updated as $h_{m+1}=h_m+\eta\Delta h$. The loop of sampling, estimation, solving, and updating continues until convergence. Theoretical guarantees for convergence and quadratic rates are provided in Section 4 (Key Design 4).

### Key Designs

**1. Cubic-Regularized Auxiliary Function: Bypassing Infinite-Dimensional Inversion via Optimization**

This design targets the core challenge: the RKHS Hessian operator is compact and lacks a bounded inverse, making the standard Newton step $[\nabla_h^2\hat J]^{-1}\nabla_h\hat J$ undefined. Drawing inspiration from cubic regularization in finite-dimensional Newton methods (Nesterov–Polyak), the Newton step $\Delta h$ is defined as the minimizer of the following auxiliary functional:

$$\Delta h=\arg\min_{\bar h\in\mathcal H_K}\left\langle\nabla_h\hat J(h_k),\bar h\right\rangle+\frac12\left\langle\nabla_h^2\hat J(h_k)\circ\bar h,\bar h\right\rangle+\frac{\beta}{6}\|\bar h\|^3.$$

Crucially, while the Hessian operator $\nabla_h^2\hat J(h_k)=\frac1N\sum_{\omega}H_h(\omega;\pi_h)$ is not computable, the **value of the bilinear form** $\langle\nabla_h^2\hat J\circ\bar h,\bar h\rangle$ is easily calculated using the definition of the tensor-product RKHS. Thus, the operator itself never needs to be materialized—only its inner product when acting on specific functions. The cubic term $\frac\beta6\|\bar h\|^3$ provides regularization, ensuring the subproblem is well-defined, unique, and stable, with $\beta$ as a hyperparameter.

**2. Dimension Reduction via Representer Theorem: Equivalence to an $NT$-Dimensional Problem**

The optimization variable in Design 1 is still in the infinite-dimensional $\mathcal H_K$. The Representer Theorem (Schölkopf et al., 2001) guarantees that the minimizer of a functional in the form of "cost + norm regularization" can be expressed as a finite linear combination of kernel sections of training samples. Substituting $\bar h_\alpha=\sum_{i=1}^N\sum_{t=1}^T\alpha^i_t K((s^i_t,a^i_t),\cdot)$, the functional search is reduced to a search for the coefficient vector $\bar\alpha\in\mathbb R^{NT}$. Utilizing the reproducing property $\langle K(x,\cdot),K(y,\cdot)\rangle=K(x,y)$, Theorem 3.3 explicitly transforms the auxiliary objective into a finite-dimensional quadratic optimization with cubic regularization:

$$\bar\alpha^*=\arg\min_{\bar\alpha\in\mathbb R^{NT}}\left\langle v,\bar\alpha\right\rangle+\frac12\left\langle H\bar\alpha,\bar\alpha\right\rangle+\frac{\beta}{6}\|\bar\alpha\|_2^3.$$

Here, $v$ is the projection of the functional gradient $\nabla_h\hat J$ onto the data kernel subspace. The second-order coefficient matrix is decomposed as $H=\frac{T^2}{N}bc^\top-\frac{T}{N}\sum_l\Psi_l(\omega)\Sigma^{(l)}$, where the first term $\frac{T^2}{N}bc^\top$ captures the rank-one part of the curvature from the outer product of the first-order gradient, and the second term containing $\Sigma^{(l)}$ represents the covariance structure of the policy's action distribution ($\Sigma^{(l)}_{ij}=\mathrm{Cov}_{a'\sim\pi(\cdot\mid s_l)}[K'_{il},K'_{jl}]$). All components are computable from kernel values $K_{il}=K((s_i,a_i),(s_l,a_l))$. Complexity depends solely on the data size $N\times T$, adapting the model scale to the data.

**3. Solving the Subproblem via Conjugate Gradient: Avoiding Instability of Analytical Solutions**

There are two ways to solve the finite-dimensional problem: (1) solving for the critical point analytically, or (2) using iterative optimizers (GD/Newton/CG). The authors explicitly choose the **Conjugate Gradient (CG) method**. While the analytical approach is conceptually direct, it introduces significant instability during training, potentially leading to exponential error growth in complex environments. CG provides a practical compromise, maintaining efficiency while being numerically robust. The complete algorithm (Algorithm 1) involves: sampling $N$ trajectories via $\pi_{h_m} \to$ estimating $v$ and $H \to$ solving for $\bar\alpha$ via CG $\to$ reconstructing $\Delta h \to$ updating $h_{m+1}=h_m+\eta\Delta h$.

**4. Convergence and Local Quadratic Rate: Preserving Second-Order Properties after Reduction**

It must be proved that the "speed" of second-order optimization is not lost during dimension reduction. The paper first proves **convergence to stationary points** in a stochastic setting. Assuming Hessian Lipschitz continuity (constant $L\le\beta$), they derive a Taylor upper bound $J(h_2)\le J(h_1)+\langle\nabla_h J,h_2-h_1\rangle+\frac12\langle\nabla_h^2 J\circ(h_2-h_1),h_2-h_1\rangle+\frac L6\|h_2-h_1\|^3$. By bounding the update step norm and noting Monte Carlo estimation converges at $O(1/\sqrt N)$, they prove $\lim_{M,N\to\infty}\mathbb E[\|\nabla J(h_{R+1})\|]=0$ for a randomly selected iteration $R$. In an ideal deterministic setting (assuming access to true gradients/Hessians and bounded regularized Hessian inverses), they prove **local quadratic convergence**: $\|h_{k+1}-h^*\|\le C_q\|h_k-h^*\|^2$ for some constant $C_q>0$. This highlights the core advantage over the first-order RKHS Policy Gradient.

### Loss & Training

The objective is to minimize the expected discounted cost $J(\pi)=\mathbb E_{\omega\sim p(\omega;\pi)}\big[\sum_{t=0}^{T-1}\gamma^{t-1}r(s_t,a_t)\big]$ (where costs are negative rewards). $v$ and $H$ are estimated via $N$ Monte Carlo trajectories. The cubic regularization strength $\beta$ should satisfy $\beta\ge L$ (Hessian Lipschitz constant). The subproblem is solved via CG, and the update uses a learning rate $\eta$. Gaussian kernels are used throughout. For continuous actions (e.g., Hopper), a continuous Gaussian policy is used (derivation in Appendix I).

## Key Experimental Results

Experiments consist of: (a) A simplified Asset Allocation toy environment for validating theoretical quadratic convergence; (b) Standard Gymnasium RL benchmarks (Discrete CartPole/Lunar Lander, Continuous Inverted Pendulum/Hopper) for performance. Gaussian kernels are used for all; parametric baselines use linear + polynomial feature expansion for fair representation.

### Main Results (Toy Environment: Quadratic Convergence Validation)

| Method | Convergence Behavior | Final Policy |
|------|----------|----------|
| Policy Gradient (Parametric) | Slow convergence | Converged but slow |
| Policy Newton (Parametric 2nd-order) | Faster than 1st-order, but unstable | Trapped in sub-optimal local max |
| Policy Gradient in RKHS | Quick convergence | Near-optimal |
| **Policy Newton in RKHS (Ours)** | **Clear quadratic convergence** | **Approaches global optimum** |

Key Observation: Since all methods have identical representation capabilities in the toy environment (finite space, direct optimization of discrete probabilities), differences stem **purely from optimization geometry rather than expressivity**. RKHS updates span data-dependent kernel bases, providing richer descent directions and escaping sub-optimal attractors that trap parametric methods.

### Training Performance (Gymnasium benchmarks, mean of 5 runs + 95% CI)

| Environment | Action Space | Performance (Ours) | Comparison |
|------|---------|---------|---------|
| CartPole | Discrete | Fast convergence, high sample efficiency | Significantly outperforms 1st-order RKHS and Parametric Newton |
| Lunar Lander | Discrete | Fast convergence, high sample efficiency | Same as above |
| Inverted Pendulum | Continuous | Fast and stable | Outperforms baselines |
| Hopper | Continuous | Reaches high return in fewer iterations | Leading sample efficiency |

### Key Findings
- **Real Quadratic Convergence**: In the toy environment, Policy Newton in RKHS exhibits a clear quadratic convergence curve approaching the optimum. Parametric Policy Newton, while faster than first-order, oscillates and settles at sub-optimal points, suggesting that second-order information is more stable in RKHS.
- **Dual Advantage of Curvature and Expressivity**: The authors attribute superior performance to two factors: the curvature information from the Hessian operator helping traverse complex terrains, and the flexible RKHS representation providing diverse descent directions.
- **Discrete/Continuous Universality**: The framework is effective for discrete actions and extends successfully to continuous high-dimensional control (Hopper) via Gaussian policy expansions.

## Highlights & Insights
- **"Solve instead of Invert" Paradigm**: Porting cubic-regularized Newton logic to infinite-dimensional operators bypasses the fundamental hurdle of uninvertible compact operators. The trick lies in observing that the second-order term only ever appears as a bilinear inner product.
- **Representer Theorem as a Lossless Dimensionality Reduction**: The transformation from infinite dimensions to $NT$ dimensions is proved to fully preserve the quadratic convergence rate (Theorem 3.3)—a vital step in merging kernel methods with second-order optimization.
- **Diagnostic Experimental Design**: By equalizing representation power in toy environments, the authors decouple "optimization geometry" from "expressivity," cleanly proving that RKHS provides superior descent directions.
- The decomposition $H=\frac{T^2}{N}bc^\top-\frac TN\sum_l\Psi_l\Sigma^{(l)}$ provides clear physical meaning to the curvature matrix (rank-one outer product + covariance), which could be adapted to other kernelized second-order methods.

## Limitations & Future Work
- **Theoretical Conditions**: Quadratic convergence is proved only under ideal deterministic settings (true gradients/Hessians and bounded inverses). Full analysis under stochastic assumptions remains future work.
- **Complexity Scaling**: The dimensionality $N \times T$ poses a bottleneck as data grows, similar to standard kernel methods like SVM. Sparse or approximate schemes were not provided.
- **Complex Environments**: The authors acknowledge potential robustness issues in high-complexity tasks like Humanoid. A suggested direction is combining neural networks with the Policy Newton in RKHS framework.
- **Internal Observation**: The evaluation environments are relatively small, and comparisons with modern deep RL (PPO/SAC) are missing. The toy environment's explicit global optimum is a simplification for theory validation; its generalizability requires caution.

## Related Work & Insights
- **vs. RKHS Policy Gradient (Paternain et al., 2020; Lever & Stafford, 2015)**: These are first-order methods using gradient functions for updates (linear convergence). This work is the first second-order RKHS method, achieving local quadratic convergence and better escape from sub-optimal terrains.
- **vs. Parametric Policy Newton (Li et al., 2023; Maniyar et al., 2024)**: These calculate cubic-regularized steps in finite-dim parameter spaces. This work elevates second-order optimization to infinite-dim RKHS, showing better optimization geometry given equivalent representation power.
- **vs. RKHS Online Newton (Calandriello et al., 2017; Lu et al., 2016)**: They use iterative approximations of the Hessian inverse in i.i.d. online settings. This work addresses the RL setting (distribution shift) with convergence guarantees, bridging the gap between non-parametric policies and second-order optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First second-order policy optimization framework in RKHS. The "Cubic Regularization + Representer Theorem" combination is a elegant solution to infinite-dimensional Hessian inversion.
- Experimental Thoroughness: ⭐⭐⭐ Validated convergence and efficiency on toy + 4 Gym benchmarks, though environments are small and lack comparison with modern deep RL baselines.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are well-structured, moving logically from motivation to reduction to convergence with clear physical intuition.
- Value: ⭐⭐⭐⭐ Merges kernel methods and second-order optimization, providing a theoretical tool for faster convergence in sample-constrained or safety-critical non-parametric RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [\[AAAI 2026\] QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation](../../AAAI2026/reinforcement_learning/qimeng-kernel_macro-thinking_micro-coding_paradigm_for_llm-based_high-performanc.md)
- [\[ICML 2026\] Global Policy-Space Response Oracles for Two-Player Zero-Sum Games](../../ICML2026/reinforcement_learning/global_policy-space_response_oracles_for_two-player_zero-sum_games.md)
- [\[ICLR 2026\] Grouping Nodes with Known Value Differences: A Lossless UCT-based Abstraction Algorithm](grouping_nodes_with_known_value_differences_a_lossless_uct-based_abstraction_alg.md)
- [\[ICLR 2026\] Exo-Plore: Exploring Exoskeleton Control Space through Human-Aligned Simulation](exo-plore_exploring_exoskeleton_control_space_through_human-aligned_simulation.md)

</div>

<!-- RELATED:END -->
