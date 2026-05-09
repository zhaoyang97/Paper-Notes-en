---
title: >-
  [Paper Note] Extragradient Method for $(L_0, L_1)$-Lipschitz Root-finding Problems
description: >-
  [NeurIPS 2025][Optimization][Extragradient method] Under the $\alpha$-symmetric $(L_0,L_1)$-Lipschitz condition (a relaxation of the classical $L$-Lipschitz assumption), this paper proposes an adaptive step size strategy $\gamma_k = 1/(c_0 + c_1\|F(x_k)\|^\alpha)$ for the extragradient (EG) method, establishing the first complete convergence guarantees for three classes of root-finding problems: strongly monotone (linear convergence), monotone (sublinear convergence), and weak Minty (local convergence).
tags:
  - NeurIPS 2025
  - Optimization
  - Extragradient method
  - $(L_0
  - L_1)$-Lipschitz
  - root-finding
  - adaptive step size
  - min-max optimization
date: 2026-05-08
content_hash: 54100d26b911cec1
---

# Extragradient Method for $(L_0, L_1)$-Lipschitz Root-finding Problems

**Conference**: NeurIPS 2025
**arXiv**: [2510.22421](https://arxiv.org/abs/2510.22421)
**Code**: [github.com/isayantan/L0L1extragradient](https://github.com/isayantan/L0L1extragradient)
**Area**: Optimization Theory / Variational Inequalities
**Keywords**: Extragradient method, $(L_0,L_1)$-Lipschitz, root-finding, adaptive step size, min-max optimization

## TL;DR
Under the $\alpha$-symmetric $(L_0,L_1)$-Lipschitz condition (a relaxation of the classical $L$-Lipschitz assumption), this paper proposes an adaptive step size strategy $\gamma_k = 1/(c_0 + c_1\|F(x_k)\|^\alpha)$ for the extragradient (EG) method, establishing the first complete convergence guarantees for three classes of root-finding problems: strongly monotone (linear convergence), monotone (sublinear convergence), and weak Minty (local convergence).

## Background & Motivation

**Background**: The extragradient (EG) method is a classical algorithm for solving variational inequalities and root-finding problems $F(x_*) = 0$, widely applied in min-max optimization (GAN training, distributionally robust optimization, multi-agent games). Existing convergence theory almost universally relies on the $L$-Lipschitz assumption $\|F(x) - F(y)\| \leq L\|x-y\|$ on the operator $F$.

**Limitations of Prior Work**: (a) The $L$-Lipschitz assumption is overly restrictive for many problems — for instance, $F(x) = x^2$ does not satisfy a Lipschitz condition for any finite $L$; (b) the $(L_0, L_1)$-smoothness assumption $\|\nabla^2 f(x)\| \leq L_0 + L_1\|\nabla f(x)\|$ introduced by Zhang et al. (2020) for minimization has been empirically validated on LSTMs and Transformers, yet has not been extended to the EG method and variational inequality settings.

**Key Challenge**: For non-Lipschitz operators (e.g., cubic min-max problems), the Jacobian norm $\|\mathbf{J}(x)\|$ grows with $\|F(x)\|$, and constant step size EG may diverge — adaptive step sizes are needed, but no theoretical framework exists.

**Goal**: To establish a complete convergence theory for EG under the $\alpha$-symmetric $(L_0,L_1)$-Lipschitz condition (covering strongly monotone, monotone, and weak Minty problem classes) and to design a corresponding adaptive step size scheme.

**Key Insight**: By exploiting the equivalent characterization $\|\mathbf{J}(x)\| \leq L_0 + L_1\|F(x)\|^\alpha$ (Theorem 2.1), the paper designs adaptive step sizes inversely proportional to $\|F(x_k)\|^\alpha$.

**Core Idea**: The step size is inversely proportional to the current operator norm $\|F(x_k)\|^\alpha$, automatically shrinking when far from the solution (ensuring stability) and growing when close to the solution (accelerating convergence).

## Method

### Overall Architecture
The two-step EG iteration: $\hat{x}_k = x_k - \gamma_k F(x_k)$ (extrapolation), $x_{k+1} = x_k - \omega_k F(\hat{x}_k)$ (update). The core contribution of this paper is the design of $(\gamma_k, \omega_k)$ for different problem classes and the corresponding convergence proofs.

### Key Designs

1. **$\alpha$-Symmetric $(L_0,L_1)$-Lipschitz Condition (Assumption 1.1)**:

    - Function: Relaxes the classical Lipschitz assumption.
    - Core definition: $\|F(x) - F(y)\| \leq (L_0 + L_1 \max_{\theta \in [0,1]} \|F(\theta x + (1-\theta)y)\|^\alpha)\|x-y\|$
    - Equivalent Jacobian characterization (Theorem 2.1): For min-max problems, $\|\mathbf{J}(x)\| \leq L_0 + L_1\|F(x)\|^\alpha$
    - Reduces to standard $L$-Lipschitz when $L_1 = 0$; $\alpha = 1$ corresponds to the $(L_0, L_1)$-smoothness of Zhang et al.
    - Significance: Allows the Lipschitz "constant" to grow with the operator norm, more accurately capturing the structure of practical problems.

2. **Key Lemma for Eliminating the Path Maximum (Proposition 3.1)**:

    - Function: Eliminates the $\max_\theta$ in the assumption to yield a tractable upper bound.
    - For $\alpha = 1$: $\|F(x) - F(y)\| \leq (L_0 + L_1\|F(x)\|)\exp(L_1\|x-y\|) \cdot \|x-y\|$
    - For $\alpha \in (0,1)$: $\|F(x) - F(y)\| \leq (K_0 + K_1\|F(x)\|^\alpha + K_2\|x-y\|^{\alpha/(1-\alpha)}) \cdot \|x-y\|$
    - where $K_0 = L_0(2^{\alpha^2/(1-\alpha)} + 1)$, $K_1 = L_1 \cdot 2^{\alpha^2/(1-\alpha)}$

3. **Adaptive Step Size Strategy**:

    - **Unified form**: $\gamma_k = \frac{1}{c_0 + c_1\|F(x_k)\|^\alpha}$
    - **Strongly monotone, $\alpha = 1$**: $\gamma_k = \omega_k = \frac{0.21}{L_0 + L_1\|F(x_k)\|}$ (Theorem 3.2, linear convergence)
    - **Monotone, $\alpha = 1$**: $\gamma_k = \omega_k = \frac{0.45}{L_0 + L_1\|F(x_k)\|}$ (Theorem 3.5, sublinear convergence)
    - **Weak Minty, $\alpha = 1$**: $\gamma_k = \frac{0.56}{L_0 + L_1\|F(x_k)\|}$, $\omega_k = \gamma_k/2$ (Theorem 3.8, local sublinear convergence)
    - For $\alpha \in (0,1)$, the constants $c_0, c_1$ in the step size depend on $K_0, K_1, K_2$.

4. **Refined Analysis for Eliminating Exponential Dependence**:

    - Issue: The convergence rate in Theorem 3.2 contains the factor $\exp(L_1\|x_0 - x_*\|)$.
    - **Corollary 3.3**: After $K'$ iterations, $\|F(x_k)\| \leq L_0/L_1$ is established, after which the step size satisfies $\geq \nu/2L_0$. The total iteration count is decomposed into two terms:
        - Term I: $\frac{2L_0}{\nu\mu}\log(1/\varepsilon)$ (standard complexity, no exponential dependence)
        - Term II: A "warmup" phase independent of $\varepsilon$
    - This is a substantive improvement over Vankov et al. (2024), whose convergence rate depends exponentially on $\|x_0 - x_*\|$.

### Problem Instance Verification
- **Example 1** (Logistic regression): $f(x) = \log(1 + \exp(-a^\top x))$, with $L = \|a\|^2$ but $L_0 = 0, L_1 = \|a\|$ — when $\|a\| \gg 1$, the $(L_0, L_1)$ bound is far tighter than the $L$ bound.
- **Example 2**: $F(x) = (u_1^2, u_2^2)$, which does not satisfy a Lipschitz condition for any finite $L$, but satisfies the $1/2$-symmetric $(0,2)$-Lipschitz condition.
- **Example 3** (Bilinear coupling): $\frac{1}{p+1}\|w_1\|^{p+1} + w_1^\top \mathbf{B} w_2 - \frac{1}{p+1}\|w_2\|^{p+1}$, which satisfies the 1-symmetric $(L_0, L_1)$-Lipschitz condition for any $p > 1$.

## Key Experimental Results

### Strongly Monotone Problem: Adaptive Step Size vs. Vankov et al.

| Method | Initial Step Size | Final Step Size | Convergence Speed |
|--------|------------------|-----------------|-------------------|
| Vankov et al. (2024) | ~0.02 | ~0.02 (constant) | Slower |
| **Ours** | ~0.02 | >0.032 (increasing) | **Faster** |

Key observation: The proposed step size **automatically increases** as $\|F(x_k)\|$ decreases, whereas Vankov's step size remains constant.

### Monotone Problem (Eq. (20), Convex-Concave Min-Max)

| Step Size Strategy | $c$ or $(c_0, c_1)$ | Convergence | Remarks |
|--------------------|---------------------|-------------|---------|
| Constant $\gamma = 1/c$ | $c = 10^5$ (optimal grid search) | Slower | Diverges for $c < 10^5$ |
| Adaptive (Ours) | $(c_0, c_1) = (10, 10)$ (optimal) | **Fastest** | All 9 combinations outperform the constant |

### Weak Minty (GlobalForsaken Problem)

| Algorithm | Step Size | Convergence to $(0,0)$ |
|-----------|-----------|------------------------|
| AdaptiveEG+ | $\gamma = 0.1$ (fixed) | Moderate |
| EG+ | $\gamma = 0.1$ (fixed) | Moderate |
| **Ours (EG)** | $(c_0, c_1) = (1, 1)$ | **Significantly fastest** |

### Key Findings
- The core advantage of adaptive step sizes is **automatic small-to-large adjustment**: small step sizes ensure safety when far from the solution, and larger step sizes accelerate convergence when close — no manual tuning required.
- On the monotone problem, the **vast majority** of the 9 adaptive parameter combinations outperform the optimal constant step size, indicating low sensitivity to hyperparameters.
- The case $\alpha < 1$ (e.g., $\alpha = 1/2$) enables coverage of operator classes where $L$-Lipschitz conditions fail entirely.

## Highlights & Insights
- **Unified step size paradigm**: $\gamma_k = 1/(c_0 + c_1\|F(x_k)\|^\alpha)$ is a single formula covering all three problem classes (strongly monotone / monotone / weak Minty), differing only in coefficients — elegant and concise.
- **Technique for eliminating exponential dependence**: The two-phase analysis in Corollary 3.3 (first establishing that the step size grows to a safe level, then applying fixed step size analysis) is generalizable to other adaptive methods.
- **Practical utility of Theorem 2.1**: The abstract path-maximum condition is equivalently reformulated as a Jacobian norm condition — the latter is directly verifiable in practice (e.g., via scatter plots), greatly enhancing the operationalizability of the assumption.

## Limitations & Future Work
- **Deterministic analysis only**: Stochastic EG variants (SVRE-EG, etc.) are not addressed; large-scale min-max training typically relies on stochastic gradients.
- **Local convergence for weak Minty**: Requires the initial point to be sufficiently close to the solution (condition (17): $\Delta_1 > 0$); no global guarantee is provided.
- **Exponential dependence in the monotone case**: Although the $\exp(L_1\|x_0 - x_*\|)$ factor in Theorem 3.5 is eliminated in Theorem 3.6, the latter requires a warmup of at least $K+1 \geq 2L_1^2\|x_0 - x_*\|^2/\nu^2$ iterations.
- **Selection of $\alpha$**: While the theory covers $\alpha \in (0,1]$, how to determine the optimal $\alpha$ for practical problems is not discussed.
- **Small-scale experiments**: Only low-dimensional synthetic problems ($d \leq 2$) are considered; large-scale validation on GAN or DRO training is absent.

## Related Work & Insights
- **vs. Vankov et al. (2024)**: Analyzes EG only for the strongly monotone case with $\alpha = 1$, using a step size defined by a $\min$ over multiple conditions (more conservative). The proposed step size is simpler, the theory tighter, and the coverage broader.
- **vs. Gorbunov et al. (2025)**: Analyzes adaptive gradient descent under $(L_0, L_1)$-smoothness for minimization problems. This paper is the first to extend this framework to EG and root-finding.
- **vs. Zhang et al. (2020)**: The originators of the $(L_0, L_1)$-smoothness definition, focusing on convergence of clipped GD in the minimization setting. This paper extends the framework to variational inequalities.
- **vs. Diakonikolas et al. (2021)**: Convergence analysis of EG under weak Minty operators, but restricted to $L$-Lipschitz. This paper relaxes the assumption to $(L_0, L_1)$-Lipschitz.

## Rating
- Novelty: ⭐⭐⭐⭐ — First complete convergence theory for EG under $(L_0,L_1)$-Lipschitz conditions, covering three problem classes.
- Experimental Thoroughness: ⭐⭐⭐ — Synthetic experiments clearly validate the theory, but large-scale experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Table 1 provides an exceptionally clear summary; the theoretical development is well-structured and layered.
- Value: ⭐⭐⭐⭐ — Fills a theoretical gap for EG under generalized Lipschitz conditions; the step size strategy has direct practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Escaping Saddle Points without Lipschitz Smoothness: The Power of Nonlinear Preconditioning](escaping_saddle_points_without_lipschitz_smoothness_the_power_of_nonlinear_preco.md)
- [\[NeurIPS 2025\] Revisiting Orbital Minimization Method for Neural Operator Decomposition](revisiting_orbital_minimization_method_for_neural_operator_decomposition.md)
- [\[NeurIPS 2025\] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](rethinking_neural_combinatorial_optimization_for_vehicle_routing_problems_with_d.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[ICLR 2026\] Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise](../../ICLR2026/optimization/dual_optimistic_ascent_pi_control_is_the_augmented_lagrangian_method_in_disguise.md)

</div>

<!-- RELATED:END -->
