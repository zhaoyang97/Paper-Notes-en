---
title: >-
  [Paper Note] Convergence of Muon with Newton-Schulz
description: >-
  [ICLR2026][Optimization][Muon optimizer] This paper provides the first non-convex convergence guarantee for the practical Muon optimizer (which uses Newton-Schulz approximation instead of exact SVD polar decomposition). It proves that the convergence rate matches the idealized SVD version up to a constant factor that decays doubly exponentially with the number of Newton-Schulz steps $q$, and that Muon suffers $\sqrt{r}$ times less rank-dependent loss than its vector counterpa…
tags:
  - "ICLR2026"
  - "Optimization"
  - "Muon optimizer"
  - "Newton-Schulz"
  - "polar decomposition"
  - "matrix optimization"
  - "convergence analysis"
date: 2026-05-08
content_hash: 32eacc6eae29f5a2
---

# Convergence of Muon with Newton-Schulz

**Conference**: ICLR2026  
**arXiv**: [2601.19156](https://arxiv.org/abs/2601.19156)  
**Code**: To be confirmed  
**Area**: Optimization/Theory  
**Keywords**: Muon optimizer, Newton-Schulz, polar decomposition, matrix optimization, convergence analysis

## TL;DR
This paper provides the first non-convex convergence guarantee for the practical Muon optimizer (which uses Newton-Schulz approximation instead of exact SVD polar decomposition). It proves that the convergence rate matches the idealized SVD version up to a constant factor that decays doubly exponentially with the number of Newton-Schulz steps $q$, and that Muon suffers $\sqrt{r}$ times less rank-dependent loss than its vector counterpart SGD-M.

## Background & Motivation

**Background**: The Muon optimizer updates matrix parameters by orthogonalizing the momentum matrix (rather than processing it vectorially like Adam), showing superior performance in LLM training. In practice, Newton-Schulz (NS) iterations are used to approximate polar decomposition, avoiding expensive SVD.

**Limitations of Prior Work**: Existing theoretical analyses of Muon (Shen et al., Li & Hong) replace NS with exact SVD—yet SVD is never used in practice. How the NS approximation error affects convergence and why only a few NS steps are sufficient remain theoretically unexplored.

**Key Challenge**: While Muon achieves SVD-level performance in practice with a few NS steps (leading to faster wall-clock time), a theoretical gap exists where practical performance far exceeds current theoretical understanding.

**Key Insight**: The authors directly analyze the polar decomposition error $\varepsilon_q$ of the NS approximation and prove that it decays doubly exponentially with the number of steps.

**Core Idea**: The doubly exponential decay of the NS approximation error $\varepsilon_q$ allows Muon to reach SVD-level convergence rates with very few NS steps. Since the per-step computation of NS is much lower than SVD, it results in faster wall-clock time.

## Method

### Overall Architecture
The paper does not modify the Muon algorithm itself but rather incorporates the practical version (using Newton-Schulz iterations instead of exact SVD for orthogonalization) into a non-convex optimization analysis framework to quantify how much convergence is lost due to approximation errors. In each step of Muon, the stochastic gradient $G_t$ first undergoes momentum accumulation $M_t = \beta M_{t-1} + G_t$, is pre-scaled to $X_{t,0} = M_t/\alpha_t$, and then undergoes $q$ steps of NS iterations $X_{t,j} = p_\kappa(X_{t,j-1}X_{t,j-1}^\top)\,X_{t,j-1}$ to drive the momentum towards an orthogonal matrix. Finally, the weights are updated as $W_t = W_{t-1} - \eta O_t$ using the approximated orthogonal direction $O_t$. The goal is to derive a stationarity bound under the nuclear norm $\frac{1}{T}\sum_t \mathbb{E}\big[\|\nabla f(W_{t-1})\|_*\big] \leq \epsilon$.

### Key Designs

**1. Non-convex convergence bound for NS-Muon: Isolating approximation error into a constant factor**

Previous Muon theories analyzed orthogonalization as exact SVD. This paper directly analyzes Muon with $q$-step NS iterations, proving that under standard smoothness and bounded variance assumptions, the number of iterations required to reach an $\epsilon$-stationary point is:

$$T = O\!\left(\frac{C_q\, L D}{\epsilon^2}\right),$$

where $L$ is the smoothness constant and $D$ is the initial sub-optimality gap. The critical contribution is compressing all degradation caused by the NS approximation into a single constant factor $C_q$. When $C_q \to 1$, the bound reverts to the rate of the idealized SVD version, reducing the problem of "approximation quality" to how close $C_q$ is to 1.

**2. Doubly exponential decay of polar approximation error: Explaining why a few NS steps suffice**

To determine the closeness of $C_q$ to 1, the authors trace $C_q$ back to the NS approximation error $\varepsilon_q$ of the polar decomposition and prove that $\varepsilon_q \leq \varepsilon_0^{(2\kappa+1)^q}$. This error shrinks doubly exponentially with the number of steps $q$ and the polynomial order $\kappa$. For example, with $\kappa=2$ (where the base is $5$), the error after $q=3$ steps is already less than $\varepsilon_0^{125}$ (approximately $10^{-100}$), theoretically justifying why $q=3 \sim 5$ suffices in practice. This also explains the wall-clock advantage, as NS involves only matrix multiplications efficient on GPUs, whereas SVD is a costly $O(mn\min(m,n))$ dense decomposition.

**3. $\sqrt{r}$ rank advantage over SGD-M: A matter of metric**

After establishing that NS-Muon is nearly as fast as SVD-Muon, the paper addresses why Muon's "orthogonalized momentum" is superior to element-wise SGD-M. It proves that Muon's convergence rate is $\sqrt{r}$ times faster ($r=\min(m,n)$ being the rank dimension). This stems from the choice of metric: SGD-M is bounded under the Frobenius norm, while Muon’s orthogonalization naturally aligns with the nuclear norm. Analyzing in the nuclear norm exploits the low-rank structure of matrix parameters to provide more efficient search directions, particularly for large high-rank layers like attention.

## Key Experimental Results

### Main Results (Convergence Comparison)

| Method | Metric | Convergence Rate | Rank Dependency |
|------|------|--------|--------|
| SGD-M | Frobenius Gradient | $O(1/\sqrt{T})$ | $\sqrt{r}$ loss |
| Muon (SVD) | Nuclear Gradient | $O(1/\sqrt{T})$ | No $\sqrt{r}$ loss |
| **Muon (NS, $q$ steps)** | Nuclear Gradient | $O(C_q/\sqrt{T})$ | No $\sqrt{r}$ loss |

### Ablation Study ($C_q$ vs. steps $q$)

| NS steps $q$ | $C_q$ at $\kappa=2$ | $C_q$ at $\kappa=3$ |
|-------------|---------------------|---------------------|
| 1 | Large | Medium |
| 3 | $\approx 1.01$ | $\approx 1.001$ |
| 5 | $\approx 1.0$ | $\approx 1.0$ |

### Key Findings
- **3-5 NS steps match SVD**: $C_q$ converges to 1 doubly exponentially, providing a solid theoretical basis for empirical choices.
- **$\sqrt{r}$ advantage of Muon over SGD-M**: The advantage is significant for high-rank matrix parameters (e.g., large attention layers).
- **Wall-clock advantage explained**: Per-step costs of NS are much lower than SVD, and since iteration counts are nearly identical, total time is reduced.

## Highlights & Insights
- **First theoretical guarantee for practical Muon**: Closes the practice-theory gap by moving beyond "pretend" SVD analysis.
- **Doubly exponential decay insight**: Proving $\varepsilon_q \leq \varepsilon_0^{5^q}$ (for $\kappa=2$) reveals why 3 steps achieve near-perfect precision ($< 10^{-100}$).
- **Nuclear norm metric**: Choosing the nuclear norm over the Frobenius norm in matrix space naturally aligns with polar decomposition and reveals the rank advantage.
- **Inspiration for future matrix optimizers**: Provides a general analysis framework for NS-based approximations in other matrix optimizers.

## Limitations & Future Work
- Purely theoretical contribution without new experiments (though the goal was to explain existing practice).
- Assumes standard smoothness and bounded variance, not yet covering Adam-style adaptivity.
- Lacks a direct comparison between Muon and second-order methods like Shampoo or SOAP.

## Related Work & Insights
- **vs. Shen et al. / Li & Hong**: These works analyze SVD-Muon. This paper is the first to analyze NS-Muon, matching actual practice.
- **vs. Shampoo/SOAP**: These are second-order preconditioners that maintain curvature. Muon is not second-order but rather orthogonalizes momentum; the mechanisms are distinct and potentially complementary.
- **vs. Orthogonal-SGDM**: Orthogonal-SGDM performs orthogonalization before momentum. Muon performs momentum before orthogonalization and replaces SVD with NS.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First convergence theory for practical NS-Muon with elegant doubly exponential results.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical, but appropriately so as it explains existing empirical successes.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear research questions, progressive theorems, and rigorous narrative.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed theoretical foundation for a popular matrix optimizer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Newton Method Revisited: Global Convergence Rates up to $O(1/k^3)$ for Stepsize Schedules and Linesearch Procedures](newton_method_revisited_global_convergence_rates_up_to_o1k3_for_stepsize_schedul.md)
- [\[ICLR 2026\] The Polar Express: Optimal Matrix Sign Methods and their Application to the Muon Algorithm](the_polar_express_optimal_matrix_sign_methods_and_their_application_to_the_muon_.md)
- [\[ICLR 2026\] Convergence of Regret Matching in Potential Games and Constrained Optimization](convergence_of_regret_matching_in_potential_games_and_constrained_optimization.md)
- [\[ICLR 2026\] Error Feedback for Muon and Friends](error_feedback_for_muon_and_friends.md)
- [\[ICLR 2026\] Muon Outperforms Adam in Tail-End Associative Memory Learning](muon_outperforms_adam_in_tail-end_associative_memory_learning.md)

</div>

<!-- RELATED:END -->
