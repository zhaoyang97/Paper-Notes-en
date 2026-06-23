---
title: >-
  [Paper Note] DR-Submodular Maximization with Stochastic Biased Gradients: Classical and Quantum Gradient Algorithms
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] This paper systematically investigates continuous DR-submodular maximization under "stochastic biased gradients." It extends the analytical Lyapunov framework from exact gradients to gradients with bias and noise. Consequently, it proves a $1/e$ approximation for a new class of constraints (convex sets with a greatest
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 8cac425bab7cb161
---
# DR-Submodular Maximization with Stochastic Biased Gradients: Classical and Quantum Gradient Algorithms

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=0CZAimzcVr](https://openreview.net/forum?id=0CZAimzcVr)  
**Code**: To be confirmed  
**Area**: optimization  
**Keywords**: DR-submodular maximization, biased stochastic gradients, Lyapunov framework, zeroth-order optimization, quantum acceleration  

## TL;DR
This paper systematically investigates continuous DR-submodular maximization under "stochastic biased gradients." It extends the analytical Lyapunov framework from exact gradients to gradients with bias and noise. Consequently, it proves a $1/e$ approximation for a new class of constraints (convex sets with a greatest element), surpassing the $1/4$ hardness of general convex constraints. The authors provide both classical ($O(\epsilon^{-3})$ iterations) and quantum ($O(\epsilon^{-1})$ iterations) zeroth-order algorithms, demonstrating quantum acceleration theoretically and numerically.

## Background & Motivation
**Background**: Continuous DR-submodular (Diminishing Returns submodular) maximization is a unified abstraction for problems such as machine learning, social network influence maximization, and non-convex/non-concave quadratic programming. The function $F$ is defined on $[0,1]^d$, where DR-submodularity implies the Hessian is non-positive everywhere ($\partial^2 F/\partial x_i\partial x_j\le 0$), making maximization NP-hard. A mature technical route involves continuous relaxation followed by first-order methods (Frank-Wolfe variants, projected gradient ascent): the monotone case reaches $(1-e^{-1})$, the non-monotone case with general convex constraints reaches $1/4$ (proven to be a tight hardness), and down-closed convex constraints reach $0.385$ or even $0.401$.

**Limitations of Prior Work**: Almost all aforementioned approximation algorithms assume an **exact gradient oracle**. However, in real-world scenarios like influence maximization or wireless network resource allocation, gradients are often estimated via sampling, which is naturally noisy and frequently **biased** (due to privacy perturbations, sampling limits, or having only a value oracle). A widely used workaround is the smoothing technique of Hazan et al. $\hat F(x)=\mathbb E_{u\sim B}[F(x+ru)]$, which provides an unbiased estimate for $\nabla\hat F$, allowing the application of "unbiased gradient" algorithms—at the cost of optimizing $\hat F$ instead of the original function $F$. Since $\|\nabla\hat F-\nabla F\|\ne 0$, additional bounds are required for the gap between $\hat F$ and $F$.

**Key Challenge**: Existing theories only cover the clean settings of "exact gradients" or "unbiased stochastic gradients," whereas real-world gradients are "biased + noisy." Directly analyzing biased gradients without detouring through smoothing functions is more practical but remained a blank in the DR-submodular field.

**Goal**: (1) Establish a unified analysis framework capable of handling "stochastic biased gradients" directly; (2) Cover general convex and down-closed convex constraints under this framework while exploring new constraint classes that break the $1/4$ hardness; (3) Implement the framework as practical zeroth-order algorithms (using only function values) and investigate potential quantum acceleration.

**Core Idea**: Generalize the **Lyapunov framework** used by Du (2022) for exact gradients by introducing bounded assumptions for bias $b(x)$ and noise $n(x)$. This generalizes the sufficient condition of a "monotonically non-decreasing potential function" into an "approximately non-decreasing" one under biased settings, where errors are explicitly quantified within the approximation ratio and iteration complexity. This allows for deriving algorithms with theoretical guarantees for different constraints and oracles in a modular fashion.

## Method

### Overall Architecture
This paper does not propose a single algorithm but rather an "algorithm generator": given a constraint type and a gradient oracle, applying the same Lyapunov/potential function derivation yields an algorithm with approximation ratio and iteration complexity guarantees. The pipeline consists of three layers.

**Layer 1—Gradient Model**. The stochastic biased gradient is expressed as $g(x(t))=\nabla F(x(t))+b(x(t))+n(x(t),\xi)$, where $b$ is the bias and $n$ is zero-mean noise ($\mathbb E_\xi[n]=0$ but non-zero variance). To guarantee convergence, relative bounded assumptions are applied: $(m,\eta_b)$-bounded bias $\|b(x)\|_2\le m\|\nabla F(x)\|_2+\eta_b(t)$, and $(M,\eta_n)$-bounded noise $\mathbb E[\|n(x)\|_2]\le M\|\nabla F(x)\|_2+\eta_n(t)$. Combined, these define the "total variance between the estimated and true gradient" $A(x(t),t)\triangleq (m+M)\|\nabla F(x)\|_2+\eta_b(t)+\eta_n(t)$, which carries all subsequent error terms.

**Layer 2—Lyapunov/Potential Function Framework**. The iterative algorithm is viewed as a numerical discretization of a continuous dynamical system $\dot x(t)=f(x(t),g(x(t)))$, with the goal of maximizing the expected approximation ratio $\mathbb E[F(x(T))]/F(x^*)$. Since this fractional optimization is intractable, a Lyapunov function $\mathcal E(x(t))=a(t)\mathbb E[F(x(t))]-b(t)F(x^*)$ is constructed to transform "fractional optimization" into "difference optimization." Ensuring $\mathcal E$ is non-decreasing leads to $\mathbb E[F(x(T))]\ge \frac{b(T)-b(0)}{a(T)}F(x^*)$, where the approximation ratio is $\frac{b(T)-b(0)}{a(T)}$. After discretization, this corresponds to the potential function $P(x(t_j))=a(t_j)\mathbb E[F(x(t_j))]-b(t_j)F(x^*)$.

**Layer 3—Instantiation for Specific Constraints**. Given a constraint, explicit coefficients $p(x(t)),q(x(t)),v(x(t))$ for the optimal solution upper bound are derived using DR-submodular inequalities (e.g., Lemma 1 for convex sets with a greatest element). These are substituted back into the variational/sequence optimization problem (10)/(13) to solve for parameter functions $a(t),b(t)$, yielding the approximation ratio and iteration count $N$. By connecting this engine to "exact gradient / quantum zeroth-order estimation / classical zeroth-order estimation" oracles, three sets of algorithms are obtained.

### Key Designs

**1. Generalizing the Lyapunov framework to biased stochastic gradients: Absorbing bias and noise with "approximate non-decrease"**

The limitation was that the original Lyapunov framework (Du, 2022) only held under exact gradients—where the potential function could strictly increase. This work's approach is to **explicitly retain** the bias and noise terms in the upper bound of the optimal solution (Assumption 3/4). The upper bound for the discrete potential function is written as $F(x^*)\le p(x(t_j))F(x(t_j))+q(x(t_j))\mathbb E_{t_j}[\langle g(x(t_j)),v(x(t_j))\rangle]-q(x(t_j))\mathbb E_{t_j}[\|b+n\|_2\|v\|_2]$, where the last term represents the "erosion" of the upper bound by bias and noise. Thus, the potential function is relaxed from "strictly non-decreasing" to "approximately non-decreasing," with the per-step cumulative error proportional to the total variance $A(x(t_j),t_j)$. The reconstructed sequence optimization (13) optimizes two objectives: the first term $\frac{b(t_N)-b(t_0)}{a(t_N)}$ determines the approximation ratio, while the second term $-\frac{1}{a(t_N)}\sum_j[2(b(t_{j+1})-b(t_j))q\,\mathbb E[A]\|v\|_2+\frac{L_1}{2}\mathbb E[a(t_{j+1})\|x(t_{j+1})-x(t_j)\|_2^2]]$ determines iteration complexity. This design ensures that when the oracle improves to an exact gradient ($A\to 0$), the second term vanishes, and "approximate non-decrease" automatically reverts to "strict non-decrease," returning the complexity to $O(\epsilon^{-1})$. The framework is **smoothly compatible** with both exact and biased settings, quantifying the trade-off that reducing variance influence requires increasing iteration steps.

**2. Introducing "convex sets with a greatest element" to prove $1/e$ approximation and break $1/4$ hardness**

The $1/4$ approximation ratio for non-monotone DR-submodular functions under general convex constraints has been proven tight (Mualem & Feldman, 2023). To perform better, structure must be added to the constraints. This paper identifies a realistic class: **convex sets with a greatest element**—where there exists $x_{\text{large}}\in C$ such that $\forall x\in C, x\le x_{\text{large}}$. This naturally occurs in resource allocation, such as wireless networks where individual users satisfy lower bounds $C=\{x\in[0,1]^d\mid Ax\ge b\}$ ($A,b$ non-negative). Using the upper bound from Lemma 1 $(1-\theta(t))F(x^*)\le F(x^*\vee x(t))\le F(x(t))+\langle\nabla F(x(t)),x(t)\vee x^*-x(t)\rangle$ (where $\theta(t)=\|x(t)\|_\infty$), coefficients are solved as $p(x(t))=q(x(t))=\frac{1}{1-\theta(t)}$ and $v(x(t))=v_{\max}(t)-x(t)$, with $v_{\max}(t)=\arg\max_{v\in C}\langle g(x(t)),v\rangle$. Substituting into the continuous-time equation yields $\dot x(t)=\frac{\dot a(t)}{a(t)}v(x(t))$. For $T=1$, the explicit solutions are $a(t)=e^t$ and $b(t)=\|1-x(0)\|_\infty t$, resulting in an approximation ratio of $\frac{b(T)-b(0)}{a(T)}=\|1-x(0)\|_\infty\frac{a(0)}{a(T)}\ln\frac{a(T)}{a(0)}\le \frac{\|1-x(0)\|_\infty}{e}$, which is at the $1/e$ level. The discrete algorithm uses a Frank-Wolfe-style convex combination update $x(t_{j+1})=(1-\tfrac1N e^{-1/N})x(t_j)+\tfrac1N e^{-1/N}v(t_j)$, requiring $N=O(1/\epsilon)$ steps under exact gradients. This result "exceeds hardness" because the $1/4$ lower bound only applies to **general** convex constraints; the "greatest element" structure allows the analysis to utilize the feasibility of $x^*\vee x(t)\in C$, bypassing general obstacles.

**3. Quantum zeroth-order gradient estimation: Compressing iteration complexity to $O(\epsilon^{-1})$ to match classical first-order**

When only function values can be queried, classical zeroth-order estimates exhibit high variance, leading to high iteration complexity. This work utilizes an improved quantum Jordan's algorithm (van Apeldoorn et al., 2023) combined with quantum state tomography to estimate gradients. It characterizes variance for the realistic case where a finite-qubit system yields a $\beta$-approximate value oracle $F_\beta$. Theorem 2 shows that to ensure $\mathbb E[\|g(x)-\nabla F(x)\|_\infty]\le\sigma$, the value precision required is only $\beta\le\frac{\sigma^2}{288\pi L_1 d(8\lceil\ln(36L_0 d/\sigma)\rceil+1)}$, and the query complexity of the value oracle is $O(8\lceil\ln(36L_0 d/\sigma)\rceil)$—logarithmic with respect to dimension $d$. Using the continuity constant $L_0=\|\nabla F(0)\|_2+\|\nabla F(1)\|_2$, the total variance $\mathbb E[A(x(t_j),t_j)]$ of the quantum estimate is controlled at the $\sigma$ level. Plugging this back into the Key Design 1 framework, Theorem 3 states: the quantum zeroth-order algorithm maintains the same approximation ratios as Theorem 1 but requires only $O(\epsilon^{-1})$ iterations, **matching classical first-order methods**. This is the source of "quantum acceleration"—the variance of the quantum gradient estimate is small enough that it does not penalize iteration count.

**4. Classical zeroth-order gradient estimation + variance reduction: $O(\epsilon^{-3})$ matching unbiased stochastic gradients**

As a baseline, the paper also presents a purely classical zeroth-order algorithm. A two-point smoothing estimate $\nabla\hat F(x)=\frac{d}{2r}\mathbb E_{u\sim S}[(F(x+ru)-F(x-ru))u]$ is used, which is unbiased relative to the smoothed function $\hat F$ but biased relative to the original $F$. Lemma 2 bounds this bias and noise: bias $\|b(x)\|_2\le \frac{dL_1 r}{2}$ (scales with sampling radius $r$) and noise $\mathbb E[\|n(x)\|_2]\le\sqrt{16\sqrt{2\pi}d L_0^2}$. Since variance remains high, momentum-based variance reduction $d_{t_j}=(1-\rho_{t_j})d_{t_j-1}+\rho_t g(x(t_j))$ is applied. Theorem 4 proves that when $\|x(t_{j+1})-x(t_j)\|_2\le\frac{\sqrt D}{N}$, then $\mathbb E[\|\nabla F(x(t_j))-d_{t_j}\|_2^2]\le\frac{Q(t_j)}{(j+9)^{2/3}}$, meaning variance decays at $j^{-2/3}$. Theorem 5 then shows that the classical zeroth-order algorithm achieves the same approximation ratio as Theorem 1 with an iteration complexity of $O(\epsilon^{-3})$, matching the performance of "unbiased stochastic gradient" methods. Comparing this with Design 3 clearly illustrates that quantum methods reduce complexity from $\epsilon^{-3}$ to $\epsilon^{-1}$, quantifying the quantum advantage.

## Key Experimental Results

### Main Results
Numerical experiments compare the three algorithms across three standard test problems (DR-submodular Quadratic Programming, Finite Sum of DR-submodular Quadratic Functions, Regular Coverage Function) and three constraint types (General Convex / Down-closed Convex / Convex with Greatest Element). Due to the high cost of classical simulation of quantum algorithms, the dimension $d$ is set to 3.

| Constraint / Problem | Quantum Zeroth-Order vs Classical Zeroth-Order | Quantum Zeroth-Order vs Classical First-Order |
|--------------|----------------------|----------------------|
| General Convex / QP & Finite Sum | Faster convergence | Comparable solution quality |
| Down-closed Convex / QP | Faster convergence, better quality | Slightly inferior solution quality |
| Greatest Element / QP & Finite Sum | Better complexity and quality | Similar convergence rate |
| Greatest Element / Regular Coverage | Continues to converge faster | Decreased solution quality |

The conclusion is that the quantum zeroth-order algorithm consistently outperforms the classical zeroth-order algorithm in convergence speed and approaches classical first-order performance in most settings—consistent with the $O(\epsilon^{-1})$ vs $O(\epsilon^{-3})$ theory.

### Quantum Wall-Clock Time Estimation

| Dimension | Error $\epsilon_{\text{ALG}}$ | Circuit Depth | Qubits | Clock Time |
|------|------|----------|----------|----------|
| $d=10$ | $0.001$ | $\approx 543$ | $\approx 230$ | $\approx 54\mu s$ |
| $d=100$ | $0.1$ | $\approx 512$ | $\approx 1600$ | $\approx 51\mu s$ |
| $d=100$ | $0.001$ | $\approx 1002$ | $\approx 2600$ | $\approx 100\mu s$ |

Assuming single/double-qubit gate times of $50/100$ ns and parallelizable gates within the same depth, phase estimation takes $\approx 100\mu s$ for $d=100, \epsilon_{\text{ALG}}=0.001$. If multiple quantum states for the value oracle are prepared in parallel across several quantum computers, the total clock time would be several hundred $\mu s$. The authors suggest this will decrease further as hardware evolves.

### Key Findings
- Quantum acceleration is not just theoretical: Complexity is $O(\epsilon^{-1})$ vs $O(\epsilon^{-3})$, and numerically, quantum zeroth-order converges faster than classical zeroth-order in all settings.
- The "convex set with greatest element" constraint provides both theoretical gains ($1/e$ breaking $1/4$) and the best numerical performance (approaching classical first-order).
- Regular coverage functions are a weaker scenario: solution quality drops for quantum zeroth-order, indicating acceleration is most stable on quadratic problems.

## Highlights & Insights
- **Treating "bias" as a first-class citizen**: Previous works either assumed exact gradients or used smoothing to switch the target to $\hat F$. This work retains bias/noise terms in the upper bound analysis, allowing the framework to be compatible with exact gradients (recovering exact cases as $A\to0$). This "explicit error accounting" is transferable to other optimization problems requiring biased oracle analysis.
- **Trading constraint structure for approximation ratio**: Since $1/4$ is a tight hardness for general convex constraints, improvement requires added structure. The "greatest element" is a lightweight yet realistic structure (resource allocation lower bounds) that allows the analysis to leverage $x^*\vee x(t)\in C$ to achieve $1/e$. This suggests that when hitting a hardness wall, one should ask if the real constraints have more structure than "general convex."
- **The value of quantum gradient estimation lies in variance**: Quantum acceleration stems from Jordan's algorithm estimating gradients with sufficiently low variance, thus avoiding iteration penalties. The clean interface design—inserting the "quantum oracle" into a classical framework solely through the $\mathbb E[A]\le\sigma$ bound—is a reusable paradigm.

## Limitations & Future Work
- **Small quantum experiment dimension**: Due to high simulation costs, experiments are limited to $d=3$. Clock times are estimates rather than real hardware measurements; practical feasibility depends on future hardware.
- **Solution quality drop on regular coverage**: Quantum zeroth-order advantages are primarily in quadratic problems; degradation in coverage functions suggests acceleration is not robust across all DR-submodular instances.
- **Tightness of greatest element constraints**: The hardness for "convex sets with greatest element" in Table 1 remains Unknown; whether $1/e$ is optimal or if higher ratios exist remains an open question.
- **Bounded bias/noise assumption**: Analysis relies on Assumption 1/2. Unbounded cases lead to non-convergence; whether real-world oracles satisfy these must be verified on a case-by-case basis.

## Related Work & Insights
- **vs Du (2022) Lyapunov Framework**: Du unified DR-submodular approximation algorithm design under exact gradients; this work generalizes it to biased stochastic gradients, adding bias/noise characterization and a new constraint axis.
- **vs Hazan et al. (2016) Smoothing Technique**: Smoothing targets $\hat F$ to gain unbiased gradients, requiring bounds on the $\hat F-F$ gap. This work analyzes the original function $F$ directly under bias, avoiding target switching.
- **vs Mokhtari et al. (2020) / Pedramfar et al. (2023) Stochastic Zeroth-Order**: They achieve $O(\epsilon^{-3})$ under unbiased gradients. This work's classical zeroth-order algorithm maintains $O(\epsilon^{-3})$ in the harder biased setting and adds a quantum version at $O(\epsilon^{-1})$.
- **vs Augustino et al. (2025) Quantum Convex Optimization**: While Augustino focuses on convex optimization, this work migrates quantum Jordan's gradient estimation to non-convex DR-submodular maximization and re-characterizes variance for finite-qubit $\beta$-approximate oracles.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of DR-submodular under biased gradients, introducing a new constraint to break hardness and quantum acceleration.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; numerical experiments are limited to $d=3$ due to quantum simulation costs.
- Writing Quality: ⭐⭐⭐⭐ Clear framework hierarchy, well-aligned theorem-algorithm-complexity mappings, though high theoretical density may be challenging for some.
- Value: ⭐⭐⭐⭐ Provides a unified analytical tool for submodular optimization under biased oracles with forward-looking quantum results.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Submodular Maximization for Sums of Concave over Modular Functions](efficient_submodular_maximization_for_sums_of_concave_over_modular_functions.md)
- [\[ICML 2026\] Differentially Private Submodular Maximization with a Knapsack Constraint](../../ICML2026/optimization/differentially_private_submodular_maximization_with_a_knapsack_constraint.md)
- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](../../ICML2026/optimization/a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](../../NeurIPS2025/optimization/online_two-stage_submodular_maximization.md)
- [\[ICML 2026\] Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions](../../ICML2026/optimization/budget-feasible_mechanisms_for_submodular_welfare_maximization_in_procurement_au.md)

</div>

<!-- RELATED:END -->
