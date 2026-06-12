---
title: >-
  [Paper Note] Optimism Without Regularization: Constant Regret in Zero-Sum Games
description: >-
  [NeurIPS 2025][Fictitious Play] This paper provides the first proof that Optimistic Fictitious Play without regularization achieves $O(1)$ constant regret in $2\times2$ zero-sum games…
tags:
  - "NeurIPS 2025"
  - "Fictitious Play"
  - "Optimistic Learning"
  - "Zero-Sum Games"
  - "Regret Bounds"
  - "No Regularization"
date: 2026-05-08
content_hash: d092618b5496449d
---

# Optimism Without Regularization: Constant Regret in Zero-Sum Games

**Conference**: NeurIPS 2025
**arXiv**: [2506.16736](https://arxiv.org/abs/2506.16736)  
**Code**: Not available  
**Area**: Other
**Keywords**: Fictitious Play, Optimistic Learning, Zero-Sum Games, Regret Bounds, No Regularization

## TL;DR

This paper provides the first proof that Optimistic Fictitious Play without regularization achieves $O(1)$ constant regret in $2\times2$ zero-sum games, matching the optimal rate of regularized Optimistic FTRL. It further establishes an $\Omega(\sqrt{T})$ regret lower bound for Alternating Fictitious Play, separating the capabilities of optimism and alternation in the unregularized setting.

## Background & Motivation

Regret bounds for online learning algorithms in games constitute a central problem in game theory and machine learning. The key background is as follows:

**Is regularization necessary?**
- In general adversarial settings, regularization is essential for no-regret learning (e.g., FTRL requires a bounded step size $\eta$).
- However, in the special structure of zero-sum games, **unregularized algorithms can also achieve sublinear regret**.

**History of Fictitious Play (FP)**:
- FP is the most classical unregularized algorithm (Brown, 1951), equivalent to Follow-the-Leader with step size $\eta \to \infty$.
- FP can incur $\Omega(T)$ regret in general settings (highly sensitive to oscillation).
- In zero-sum games, Robinson (1951) proved sublinear regret for FP, though the upper bound is slow: $O(T^{1-1/n})$ for $n\times n$ games.
- Recent work has progressively improved FP's regret bounds, achieving $O(\sqrt{T})$ on diagonal matrices and generalized rock-paper-scissors matrices.

**Regularized optimistic algorithms**:
- Optimistic FTRL (including Optimistic MWU and Optimistic GD) achieves $O(1)$ constant regret in zero-sum games.
- However, the standard RVU bound proof technique **critically relies** on a finite step size upper bound (i.e., nonzero regularization).

This motivates the core question: **Can $O(1)$ regret be achieved in zero-sum games with no regularization ($\eta \to \infty$)? Can a variant of FP attain $O(1)$?**

This question has both theoretical significance and direct applications in equilibrium computation for combinatorial games and self-play training in multi-agent reinforcement learning.

## Method

### Overall Architecture

The paper studies Optimistic Fictitious Play (OFP)—the optimistic variant of FP—with the following update rule:

$$x_1^{t+1} = \arg\max_{x \in \{e_i\}_m} \langle x, \sum_{k=0}^t Ax_2^k + Ax_2^t \rangle$$

Compared to standard FP, OFP additionally introduces a bias toward the most recent feedback (the $+Ax_2^t$ term), which is equivalent to Optimistic FTRL without regularization.

### Key Designs

1. **Geometric perspective in dual space**: An energy function $\Psi(y) = \max_{x \in \Delta_m \times \Delta_n} \langle x, y \rangle$ is defined as the support function of $\Delta_m \times \Delta_n$, where $y^t$ denotes the cumulative payoff vector. The key relationship is:

$$\text{Reg}(T) = \Psi(y^{T+1})$$

That is, regret equals exactly the energy of the dual iterate (Proposition 3.4). Proving constant regret is thus equivalent to proving energy boundedness.

2. **Unified skew subgradient descent perspective (Proposition 3.6)**: Both FP and OFP can be written as skew subgradient descent on the energy $\Psi$, differing only in the predicted dual vector:

$$y^{t+1} = y^t + J \partial\Psi(\tilde{y}^{t+1}), \quad J = \begin{pmatrix} 0 & A \\ -A^\top & 0 \end{pmatrix}$$

where $\tilde{y}^{t+1} = y^t$ (FP) or $\tilde{y}^{t+1} = 2y^t - y^{t-1}$ (OFP).

   - **Standard FP**: Due to the anti-symmetry of $J$, the single-step energy change satisfies $\Delta\Psi \geq 0$ (non-decreasing energy), growing by at least a constant over $\sqrt{T}$ steps.
   - **Optimistic FP**: When the predicted and actual dual vectors map to the same primal vertex (i.e., $\partial\Psi(y^{t+1}) = \partial\Psi(\tilde{y}^{t+1})$), $\Delta\Psi \leq 0$ (energy non-increasing).

3. **Exploiting the structure of $2\times2$ games**: Using Assumption 1 (any $2\times2$ zero-sum game can be transformed WLOG into the form $\det A = 0$, $a, d > \max\{0, b, c\}$), the dual vectors $y^t$ all lie in the same two-dimensional subspace, enabling planar geometric analysis of OFP's trajectory.

### Core Proof Idea for the Main Theorem

**Theorem 3.5** (Energy Boundedness): In a $2\times2$ zero-sum game, the dual energy of OFP satisfies:

$$\Psi(y^{T+1}) \leq 8a_{\max}\left(1 + 2\frac{a_{\max}}{a_{\text{gap}}}\right)^2$$

where $a_{\max} = \|A\|_\infty$ and $a_{\text{gap}} = \min_{(i,j),(k,\ell)} |A_{ij} - A_{k\ell}|$.

The core argument is that in dual space, the trajectory of $y^t$ is confined to a bounded region: whenever the energy grows beyond a certain threshold, the optimistic prediction causes the algorithm to "rebound" back into a low-energy region.

### Lower Bound for Alternating FP

**Theorem 3.2**: On the Matching Pennies game, for almost all initializations, Alternating FP incurs $\Omega(\sqrt{T})$ regret. This separates the capabilities of optimism and alternation in the unregularized setting: optimism achieves $O(1)$, while alternation alone cannot improve beyond $O(\sqrt{T})$.

## Key Experimental Results

### Main Results: Empirical Regret Comparison

| Game | FP Regret Growth | OFP Regret Growth | AFP Regret Growth |
|---|---|---|---|
| Matching Pennies ($2\times2$) | $\sim\sqrt{T}$ | **Constant** | $\sim\sqrt{T}$ |
| $15\times15$ Identity Matrix | $\sim\sqrt{T}$ | **Constant** | $\sim\sqrt{T}$ |
| $15\times15$ Generalized RPS | $\sim\sqrt{T}$ | **Constant** | $\sim\sqrt{T}$ |

### Ablation Study: Summary of Regret Bounds in $2\times2$ Games

| Algorithm Variant | Bounded $\eta$ (FTRL) | $\eta \to \infty$ (FP) |
|---|---|---|
| Standard | $O(\sqrt{T})$ | $O(\sqrt{T})$ |
| Optimistic | $O(1)$ | **$O(1)$ ⭐ Ours** |
| Alternating | $O(T^{1/5})$ | **$\Omega(\sqrt{T})$ ⭐ Ours** |

### Key Findings

- **Constant regret of OFP holds empirically in higher-dimensional games**: Although the theory is proven only for $2\times2$, OFP exhibits constant regret on $15\times15$ games, strongly suggesting the result generalizes.
- **Alternation cannot substitute for optimism**: In the unregularized setting, the two are strictly separated ($O(1)$ vs. $\Omega(\sqrt{T})$), whereas with regularization, alternation can achieve $O(T^{1/5})$.
- **The energy function provides an intuitive geometric picture**: The dual trajectory of OFP oscillates within a bounded region without diverging, whereas the energy of FP and AFP grows monotonically.

## Highlights & Insights

- **First result to match the optimal rate of regularized algorithms within the unregularized regime**, challenging the implicit assumption that regularization is a necessary condition for $O(1)$ regret.
- **The energy analysis in dual space** introduces a novel proof technique fundamentally different from the standard RVU bound approach.
- **The separation between optimism and alternation** is theoretically elegant: under the same FP framework, the two improvement strategies exhibit qualitatively different behavior in the large step size limit.
- The results have potential implications for equilibrium computation algorithms (e.g., MCCFR variants in combinatorial games) and self-play RL.

## Limitations & Future Work

- **Restricted to $2\times2$ zero-sum games**: The proof relies heavily on the structure of $2\times2$ games (dual vectors lying in a two-dimensional subspace); extending to general $n\times n$ games is the primary open problem.
- **Assumes a unique interior Nash equilibrium**: Degenerate games (e.g., those with pure strategy NE) are outside the scope.
- **Constants depend on game parameters**: $8a_{\max}(1+2a_{\max}/a_{\text{gap}})^2$ can be very large when $a_{\text{gap}}$ is small (near-degenerate).
- **Last-iterate convergence not addressed**: Only time-average convergence is proven; the last-iterate behavior of OFP remains unknown.
- **The lower bound for Alternating FP holds only for specific initializations**: Coverage is limited to almost all irrational initializations.

## Related Work & Insights

- The paper has deep connections to energy conservation in continuous-time Hamiltonian flows: discretization causes energy growth in standard FP, while the optimistic correction "restores" approximate conservation.
- The results may inspire the design of unregularized algorithms for extensive-form games.
- The findings provide theoretical reference for understanding convergence behavior in self-play training (e.g., AlphaStar, OpenAI Five).
- The methodology bears connections to work on unregularized acceleration in Frank-Wolfe optimization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First proof that a FP variant achieves $O(1)$ regret—a surprising result.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are primarily validatory ($T=10{,}000$); large-scale or application-oriented experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are elegant, intuitions are well-explained, and Table 1 provides a clear landscape summary.
- Value: ⭐⭐⭐⭐ Significant theoretical contribution to online learning and game theory, though the restriction to $2\times2$ limits immediate applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games](efficient_kernelized_learning_in_polyhedral_games_beyond_full-information_from_c.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](evolutionary_prediction_games.md)
- [\[NeurIPS 2025\] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](maszero_designing_multiagent_systems_with_zero_supervision.md)
- [\[NeurIPS 2025\] Gaussian Process Upper Confidence Bound Achieves Nearly-Optimal Regret in Noise-Free Gaussian Process Bandits](gaussian_process_upper_confidence_bound_achieves_nearly-optimal_regret_in_noise-.md)
- [\[ICML 2026\] Optimal Regularization for Performative Learning](../../ICML2026/others/optimal_regularization_for_performative_learning.md)

</div>

<!-- RELATED:END -->
