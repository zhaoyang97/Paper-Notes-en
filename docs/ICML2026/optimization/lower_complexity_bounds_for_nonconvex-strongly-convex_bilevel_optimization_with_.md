---
title: >-
  [Paper Note] Lower Complexity Bounds for Nonconvex-Strongly-Convex Bilevel Optimization with First-Order Oracles
description: >-
  [ICML2026][Optimization][Bilevel optimization] This paper provides the first complexity lower bounds for smooth "nonconvex-strongly-convex" bilevel optimization under standard (deterministic/stochastic) first-order oracles that are strongly related to the condition number $\kappa$. The established bounds are $\Omega(\kappa^{3/2}\epsilon^{-2})$ for the deterministic case and $\Omega(\kappa^{5/2}\epsilon^{-4})$ for the stochastic case. These results demonstrate that bilevel pro…
tags:
  - "ICML2026"
  - "Optimization"
  - "Bilevel optimization"
  - "First-order oracles"
  - "Lower complexity bounds"
  - "zero-chain"
  - "Condition number"
date: 2026-05-08
content_hash: e04bc83fc8bac56f
---

# Lower Complexity Bounds for Nonconvex-Strongly-Convex Bilevel Optimization with First-Order Oracles

**Conference**: ICML2026  
**arXiv**: [2511.19656](https://arxiv.org/abs/2511.19656)  
**Code**: None (Theoretical paper)  
**Area**: Optimization Theory / Bilevel Optimization / Lower Bounds  
**Keywords**: Bilevel optimization, First-order oracles, Lower complexity bounds, zero-chain, Condition number

## TL;DR
This paper provides the first complexity lower bounds for smooth "nonconvex-strongly-convex" bilevel optimization under standard (deterministic/stochastic) first-order oracles that are strongly related to the condition number $\kappa$. The established bounds are $\Omega(\kappa^{3/2}\epsilon^{-2})$ for the deterministic case and $\Omega(\kappa^{5/2}\epsilon^{-4})$ for the stochastic case. These results demonstrate that bilevel problems are inherently more difficult than single-level nonconvex and min-max optimization, exposing a significant power gap in $\kappa$ between existing upper and lower bounds.

## Background & Motivation

**Background**: Bilevel optimization takes the form $\min_{\mathbf{x}} H(\mathbf{x}):=f(\mathbf{x};\mathbf{y}^*(\mathbf{x}))$, where $\mathbf{y}^*(\mathbf{x})=\arg\min_{\mathbf{y}} g(\mathbf{x};\mathbf{y})$. This framework is widely used in meta-learning, reinforcement learning, hyperparameter optimization, and federated learning. This paper focuses on the widely analyzed "nonconvex-strongly-convex" setting: where the lower-level function $g$ is $\mu$-strongly convex and smooth with respect to $\mathbf{y}$, while the upper-level function $f$ is smooth but potentially nonconvex. Recent years have seen extensive research on **upper bounds** (convergence rates), including AID/ITD methods relying on second-order information (Hessian/Jacobian-vector products) and "fully first-order" algorithms like penalty-based or value-function methods.

**Limitations of Prior Work**: While upper bounds are well-studied, progress on **lower bounds** has significantly lagged. The difficulty lies in the inherent complexity of the bilevel structure—naively constructed hard instances often yield "trivial" lower bounds that are no stronger than the classic single-level bound of $\Omega(\epsilon^{-2})$, failing to capture the true dependence on the condition number $\kappa=L_g/\mu$ and precision $\epsilon$.

**Key Challenge**: Existing lower bounds bypass the "standard first-order oracle" setting. For instance, the bound by Ji and Liang (2023) relies on second-order oracles and requires the hyper-objective $H(\mathbf{x})$ to be convex/strongly convex (too restrictive for general bilevel problems). The finite-sum lower bound $\Omega(n+\sqrt{n}\epsilon^{-2})$ by Dagréou et al. (2024) lacks any dependence on $\kappa$. The bound by Kwon et al. (2024) is built on a $\mathbf{y}^*$-aware oracle that can directly return estimates of the true solution $\mathbf{y}^*$, effectively reducing the problem to single-level optimization. **Under standard (stochastic) first-order oracles with direct access to $f$ and $g$, the true lower bound for bilevel optimization has remained an open problem.**

**Goal**: To provide non-trivial first-order complexity lower bounds dependent on $\kappa$ and $\epsilon$ without relying on second-order information, convexity of $H$, or $\mathbf{y}^*$-aware oracles, thereby answering how much harder bilevel optimization is compared to single-level or min-max optimization.

**Key Insight**: The authors adopt the classic worst-case construction ideas from Nesterov and Carmon et al. (2020), using hard instances that satisfy the **zero-chain** property. This forces any zero-respecting first-order algorithm to "activate" variables coordinate-by-coordinate, translating the requirement of "discovering $T$ coordinates" into "at least $T$ oracle calls."

**Core Idea**: The strongly convex part is encoded using a tri-diagonal Laplacian matrix $A$, and the nonconvex part uses Carmon's $\Psi/\Phi$ hard functions. By **concatenating the zero-chains of the upper and lower levels**, the coordinate activation speed is further slowed by the condition number of the strongly convex lower level, effectively "stacking" the powers of $\kappa$ into the lower bound.

## Method

As a pure theory paper, the "method" refers to the **construction of hard instances and the proof of lower bounds**. The general approach is to first establish a universal framework for lower bound arguments (zero-chain + zero-respecting algorithm class), then design worst-case instances for the function class $\mathcal{F}(L_f, L_g, \mu, \Delta)$ for both deterministic and stochastic oracles.

### Overall Architecture

The lower bound argument follows three steps: **(1) Define the algorithm class**: All existing first-order bilevel algorithms (penalty, primal-dual, value-function, etc.) are abstracted as iterations satisfying the **zero-respecting** property (Definition 4), where the support of the next iterates $\mathbf{x}^{t+1}, \mathbf{y}^{t+1}$ is limited to the union of the supports of previous iterates and their gradients. **(2) Construct a hard instance**: Design $\{f, g\}$ within $\mathcal{F}(L_f,L_g,\mu,\Delta)$ that forms a global zero-chain, ensuring that starting from $\mathbf{x}=\mathbf{0}$, at most one new coordinate is activated per iteration. **(3) Translate to lower bounds**: Prove that achieving $\|\nabla H(\mathbf{x})\|_2 < \epsilon$ requires activating at least $T$ coordinates, where $T$ is proportional to $\kappa^{3/2}\epsilon^{-2}$ (deterministic) or $\kappa^{5/2}\epsilon^{-4}$ (stochastic).

The core construction involves coupling the upper level $f$ and lower level $g$ along coordinate blocks $\mathbf{y}^{(0)}, \dots, \mathbf{y}^{(T)}$:

$$f(\mathbf{x};\widetilde{\mathbf{y}})=\sum_{i=1}^{T}\frac{\lambda^2 L_f}{L}\Big[\Psi\big(-\tfrac{C_l}{\lambda}y^{(i-1)}_n\big)\Phi\big(-\tfrac{C_r}{\lambda}y^{(i)}_1\big)-\Psi\big(\tfrac{C_l}{\lambda}y^{(i-1)}_n\big)\Phi\big(\tfrac{C_r}{\lambda}y^{(i)}_1\big)\Big]$$

$$g(\mathbf{x};\widetilde{\mathbf{y}})=\sum_{i=0}^{T}\Big[\tfrac{L_g n^2}{2(4n^2+1)}(\mathbf{y}^{(i)})^\top\big(\tfrac{1}{n^2}I_n+A\big)\mathbf{y}^{(i)}-L_g(\mathbf{b}_x^{(i)})^\top\mathbf{y}^{(i)}\Big]$$

The lower level $g$ is a quadratic strongly convex function where the optimal solution $\mathbf{y}^*(\mathbf{x})$ depends implicitly on $\mathbf{x}$ via a linear system. The upper level $f$ couples through the boundary coordinates $y_n^{(i-1)}, y_1^{(i)}$ of the lower-level optimal solution, embedding the difficulty of solving the lower level into the propagation path of the zero-chain.

### Key Designs

**1. Zero-respecting algorithms + zero-chain reduction: Forcing iteration count to coordinate activation count**  
The lower bound must hold for *all* first-order algorithms. The authors use zero-chains (Definition 5): if $\mathrm{supp}(\mathbf{x}) \subset \{1, \dots, i-1\}$, then $\mathrm{supp}(\nabla f(\mathbf{x})) \subset \{1, \dots, i\}$. This implies that after $t$ steps, a zero-respecting algorithm starting at $\mathbf{0}$ satisfies $\mathrm{supp}(\mathbf{x}^t) \subset \{1, \dots, t\}$. The complexity problem is thus transformed into a combinatorial counting problem of coordinate activation.

**2. Tri-diagonal Laplacian matrix $A$: Injecting $\kappa$ via the strongly convex lower level**  
The lower-level strongly convex component uses a 1D discrete Laplacian matrix $A$ (diagonal 2, sub-diagonal -1). This structure naturally maintains the zero-chain. Solving for $\mathbf{y}^*$ requires solving a tri-diagonal linear system where the "information propagation speed" is severely hampered by the condition number $\kappa=L_g/\mu$. This is the source of $\kappa$ in the lower bound and why the bilevel bound $\Omega(\kappa^{3/2}\epsilon^{-2})$ is stronger than the min-max bound $\Omega(\sqrt{\kappa}\epsilon^{-2})$ by a factor of $\kappa$.

**3. Carmon's $\Psi/\Phi$ hard functions: Creating coupling in the nonconvex upper level**  
The upper-level nonconvexity is handled by two component functions $\Psi(x)$ (which is zero for $x \le 1/2$) and $\Phi(x)$. These ensure that the gradient remains large at non-activated coordinates while maintaining the zero-chain property ($\Psi(0)=\Psi'(0)=0$). These functions are bounded and smooth, ensuring the construction stays within the class $\mathcal{F}(L_f, L_g, \mu, \Delta)$.

**4. Stochastic case: Bounded hypercubes for variance control**  
In the stochastic setting, the oracle returns unbiased estimates with variance $\le \sigma^2$. The authors extend the deterministic construction by using **two bounded hypercubes** to constrain the ranges of $\mathbf{x}$ and $\mathbf{y}$, thereby controlling the variance budget. This approach is more concise than parallel works that eliminate the coupling variable $\mathbf{y}$. The resulting stochastic lower bound is $\Omega(\kappa^{5/2}\epsilon^{-4})$.

## Key Experimental Results

This paper contains no numerical experiments; the results are theoretical lower bounds. The following tables summarize the comparisons.

### Main Results: Comparison of Optimal Lower Bounds

| Setting | Single-level Nonconvex | Nonconvex-Strongly-Convex min-max | **Ours: Bilevel** | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| Det. First-order | $\Omega(\epsilon^{-2})$ | $\Omega(\sqrt{\kappa}\epsilon^{-2})$ | $\Omega(\kappa^{3/2}\epsilon^{-2})$ | $\times\kappa^{3/2}$ / $\times\kappa$ |
| Stoch. First-order | $\Omega(\epsilon^{-4})$ | $\Omega(\kappa^{1/3}\epsilon^{-2})$ | $\Omega(\kappa^{5/2}\epsilon^{-4})$ | $\times\kappa^{5/2}$ / $\times\kappa^{13/6}$ |

> ⚠️ The min-max stochastic lower bound is often cited as $\Omega(\kappa^{1/3}\epsilon^{-2})$, which is at a different precision scale than the $\epsilon^{-4}$ here. The gain factor follows the original logic of the paper.

### Comparison with Existing Bilevel Bounds

| Work | Type | Oracle / Assumption | Complexity | 含 $\kappa$ |
| :--- | :--- | :--- | :--- | :--- |
| Ji & Liang (2023) | Lower | 2nd-order, $H$ is convex | $\sqrt{\kappa}$ gap | Yes (Restricted) |
| Dagréou et al. (2024) | Lower | Finite-sum 1st-order | $\Omega(n+\sqrt{n}\epsilon^{-2})$ | No |
| Kwon et al. (2024) | Lower | $\mathbf{y}^*$-aware stoch. | — | Reduced case |
| **Ours** | **Lower** | **Standard 1st-order** | $\Omega(\kappa^{3/2}\epsilon^{-2})$ / $\Omega(\kappa^{5/2}\epsilon^{-4})$ | **Yes** |
| Chen et al. (2025) | Upper | 1st-order penalty | $\kappa^{3.5}\epsilon^{-2}$ | Yes |

### Key Findings
- **Bilevel > min-max > Single-level**: The deterministic bound of $\kappa^{3/2}$ vs. $\sqrt{\kappa}$ vs. $\kappa^0$ establishes that bilevel optimization is fundamentally harder, with the difficulty quantified by powers of $\kappa$.
- **Large Gap Persists**: There remains a significant gap between the deterministic upper bound $\kappa^{3.5}\epsilon^{-2}$ and this lower bound $\Omega(\kappa^{3/2}\epsilon^{-2})$.
- **Lower Level Quadraticity**: The lower bounds hold even when the lower-level objective is limited to quadratic functions.

## Highlights & Insights
- **Quantifying Bilevel Difficulty**: The paper elegantly uses a zero-chain framework to show that bilevel optimization adds a factor of $\kappa$ in difficulty relative to min-max problems.
- **Minimalist Construction**: Unlike parallel works that introduce auxiliary variables, this paper uses a "Keep $\mathbf{y}$ + Bounded Hypercubes" approach, resulting in a cleaner construction.
- **Transferable Toolset**: The combination of tri-diagonal Laplacian encoding and Carmon functions can be extended to analyze other nested structures like tri-level optimization.

## Limitations & Future Work
- **Bounds may not be tight**: The gap between upper and lower bounds persists, suggesting that either the lower bounds can be tightened or more efficient algorithms exist.
- **Setup Restricted to Nonconvex-Strongly-Convex**: The construction relies on the uniqueness of $\mathbf{y}^*$ provided by strong convexity; it does not directly apply to convex or nonconvex lower levels.
- **Existential Lower Bound**: The paper proves the existence of a worst-case instance but does not provide matching algorithms.

## Related Work & Insights
- **vs Li et al. (2021) (min-max bounds)**: The construction is heavily inspired by their work but upgrades the "saddle point chain" to a dual-layer chain, increasing the $\kappa$ dependency from $\sqrt{\kappa}$ to $\kappa^{3/2}$.
- **vs Chen & Zhang (2025) (Parallel work)**: Both independently arrived at first-order bilevel lower bounds with large $\kappa$ dependency. This paper's construction is generally more simplified.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Fills the open problem of standard first-order lower bounds for bilevel optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical, but provides systematic comparisons with existing results.
- Writing Quality: ⭐⭐⭐⭐ Clear reasoning and well-motivated components for the hard instance.
- Value: ⭐⭐⭐⭐⭐ Quantifies the inherent difficulty of bilevel optimization and points to future research directions for closing the complexity gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Fully First-Order Layer for Differentiable Optimization](a_fully_first-order_layer_for_differentiable_optimization.md)
- [\[NeurIPS 2025\] A Single-Loop First-Order Algorithm for Linearly Constrained Bilevel Optimization](../../NeurIPS2025/optimization/a_single-loop_first-order_algorithm_for_linearly_constrained_bilevel_optimizatio.md)
- [\[ICLR 2026\] Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm](../../ICLR2026/optimization/bilevel_optimization_with_lower-level_uniform_convexity_theory_and_algorithm.md)
- [\[ICLR 2026\] Strongly Convex Sets in Riemannian Manifolds](../../ICLR2026/optimization/strongly_convex_sets_in_riemannian_manifolds.md)
- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](../../ICML2025/optimization/improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)

</div>

<!-- RELATED:END -->
