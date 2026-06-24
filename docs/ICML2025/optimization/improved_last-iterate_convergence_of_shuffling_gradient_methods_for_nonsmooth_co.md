---
title: >-
  [Paper Note] Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization
description: >-
  [ICML 2025][Optimization][shuffling gradient] This paper provides the first proof that the last-iterate convergence rates of Random Reshuffle (RR) and Single Shuffle (SS) in nonsmooth (strongly) convex finite-sum optimization are strictly better than that of Proximal GD. Specifically, RR achieves $\tilde{O}(GD_\star / (n^{1/4}\sqrt{K}))$, closely matching the lower bound $\Omega(1/(n^{1/4}\sqrt{K}))$.
tags:
  - "ICML 2025"
  - "Optimization"
  - "shuffling gradient"
  - "Random Reshuffle"
  - "last-iterate convergence"
  - "nonsmooth convex"
  - "suffix average"
date: 2026-05-08
content_hash: b4fe1ea71272a484
---

# Improved Last-Iterate Convergence of Shuffling Gradient Methods for Nonsmooth Convex Optimization

**Conference**: ICML 2025  
**arXiv**: [2505.23056](https://arxiv.org/abs/2505.23056)  
**Code**: None  
**Area**: Optimization Theory  
**Keywords**: shuffling gradient, Random Reshuffle, last-iterate convergence, nonsmooth convex, suffix average

## TL;DR

This paper provides the first proof that the last-iterate convergence rates of Random Reshuffle (RR) and Single Shuffle (SS) in nonsmooth (strongly) convex finite-sum optimization are strictly better than that of Proximal GD. Specifically, RR achieves $\tilde{O}(GD_\star / (n^{1/4}\sqrt{K}))$, closely matching the lower bound $\Omega(1/(n^{1/4}\sqrt{K}))$.

## Background & Motivation

**Background**: Shuffling gradient methods (RR/SS/IG) are the most widely used practical algorithms for solving finite-sum optimization $\min_x F(x) = f(x) + \psi(x)$, where $f = \frac{1}{n}\sum_{i=1}^n f_i$. Unlike SGD which randomly samples a single component, shuffling methods traverse all $n$ components in a permuted order in each epoch. In smooth convex optimization, the last-iterate of RR/SS has been shown to achieve the optimal convergence rate.

**Limitations of Prior Work**: Liu & Zhou (2024b) established the first last-iterate convergence results, but their bound in the nonsmooth case is $O(GD_\star/\sqrt{K})$—which holds for any shuffling strategy, is comparable to Proximal GD, and completely fails to capture the acceleration benefits brought by the permutation randomness in RR/SS. Koren et al. (2022) proved a rate of $O(1/(n^{1/4}\sqrt{K}))$ for the average iterate under RR, but improving the last-iterate has remained an open problem.

**Key Challenge**: Nonsmoothness renders standard analysis techniques (which rely on gradient Lipschitz continuity) inapplicable, while the correlation between gradients introduced by the permutation structure makes the analytical complexity far exceed that of independently sampled SGD.

**Goal**: To resolve the open question posed by Liu & Zhou (2024b)—whether the last-iterate of RR/SS in nonsmooth (strongly) convex optimization can be proven to be faster than that of Proximal GD.

**Key Insight**: By conducting a refined analysis of the permuted structure of RR/SS, recursion inequalities are established at the step level (rather than only at the epoch level) to extract the extra $n^{-1/4}$ or $n^{-1/2}$ acceleration factors brought by permutation randomness.

**Core Idea**: The last-iterate of RR achieves $\tilde{O}(GD_\star/(n^{1/4}\sqrt{K}))$, providing the first proof that shuffling is strictly faster than Proximal GD under nonsmoothness.

## Method

### Overall Architecture

The algorithm studied in this work is the General Proximal Gradient Method (Algorithm 1): at each step $t$, index $\mathsf{i}(t) \in [n]$ is selected, and $x_{t+1} = \arg\min_x \psi(x) + \langle \nabla f_{\mathsf{i}(t)}(x_t), x\rangle + \|x - x_t\|^2/(2\eta_t)$ is performed. The way the indices are generated determines RR (reshuffled randomly every epoch), SS (using the same single permutation throughout), or IG (fixed permutation). Unlike prior works, Algorithm 1 performs proximal updates at every step (rather than only at the end of an epoch) and is applicable to any $T \in \mathbb{N}$ (without requiring $T = Kn$).

### Key Designs

1. **General convex last-iterate convergence of RR (Theorem 4.2)**:

    - Function: Establishes the first accelerated convergence rate for the last-iterate under RR.
    - Mechanism: Setting $T = Kn$ and step size $\eta_t = \eta/\sqrt{t}$ yields $\mathbb{E}[F(x_{Kn+1}) - F(x_\star)] = \tilde{O}(G_{f,2} D_\star / (n^{1/4}\sqrt{K}))$. The logarithmic factor can be eliminated when $K = \Omega(\log n)$. The key technical innovation lies in a step-by-step analysis of function value changes, leveraging the "exactly traversing all $n$ components per epoch" property of the permutation structure to extract the $n^{-1/4}$ acceleration factor.
    - Design Motivation: Compared to $O(1/\sqrt{K})$ in Liu & Zhou (2024b), the proposed rate of $\tilde{O}(1/(n^{1/4}\sqrt{K}))$ is $\Theta(n^{1/4})$ times faster, demonstrating for the first time the acceleration effect of RR permutation randomness in nonsmooth settings.

2. **Suffix average optimality of RR (Corollary 4.3)**:

    - Function: As a corollary of Theorem 4.2, first proves that the suffix average (average over the last epoch) also achieves $\tilde{O}(1/(n^{1/4}\sqrt{K}))$.
    - Mechanism: The bound of the average of the last $n$ iterations is directly implied by the last-iterate bound. This result closely matches the lower bound $\Omega(1/(n^{1/4}\sqrt{K}))$ from Koren et al. (2022), filling the gap between upper and lower bounds.
    - Design Motivation: Koren et al. only proved the upper bound for the average iterate but lacked corresponding last-iterate and suffix average results. Starting from the last-iterate, this work resolves both issues simultaneously.

3. **Last-iterate convergence of SS (Theorems 4.5, 4.6)**:

    - Function: Establishes last-iterate convergence rates for Single Shuffle.
    - Mechanism: Under general convexity: $\tilde{O}(GD_\star/(n^{1/4}K^{1/4}) \lor GD_\star/\sqrt{n})$. In the special case of constrained optimization ($\psi = I_{\mathcal{C}}$), it can be improved to $\tilde{O}(GD_\star/(n^{1/4}K^{1/4}) \land GD_\star/\sqrt{K})$, which outperforms Liu & Zhou for any $K$. Under strong convexity: $\tilde{O}(\mu D_\star^2/(n^2K^2) + G^2/(\mu\sqrt{nK}) + G^2/(\mu n))$.
    - Design Motivation: SS only randomly selects a permutation once at the start, making its randomness weaker than RR (which reshuffles every epoch). Although the convergence rate of SS is indeed weaker than RR ($K^{1/4}$ vs $\sqrt{K}$), it is still strictly superior to Proximal GD, revealing that "even a single permutation" provides acceleration.

### Loss & Training

This work is purely theoretical. The assumptions are: each $f_i$ is convex and $G_i$-Lipschitz (only on $\text{dom}\psi$), and $\psi$ is proper closed convex and possibly $\mu$-strongly convex ($\mu \ge 0$). The step size is chosen as $\eta_t = \eta/\sqrt{t}$ (general convex) or $\eta_t = \eta/t$ (strongly convex), where $\eta$ is optimally chosen based on $G, D_\star, n$. The paper also proposes a general sufficient condition that guarantees last-iterate convergence, which is not restricted to shuffling strategies.

## Key Experimental Results

### Main Convergence Rates Comparison ($T = Kn$, General Convex $\mu=0$)

| Method | Sampling | Convergence Rate | Output |
|------|------|--------|------|
| Proximal GD | ANY | $O(GD_\star/\sqrt{K})$ | $x_{Kn+1}$ |
| Koren et al. 2022 | RR | $O(GD_\star/(n^{1/4}\sqrt{K}))$ | $x_{Kn+1}^{\text{avg}}$ |
| **Ours Thm 4.2** | RR | $\tilde{O}(GD_\star/(n^{1/4}\sqrt{K}))$ | $x_{Kn+1}$ |
| **Ours Cor 4.3** | RR | $\tilde{O}(GD_\star/(n^{1/4}\sqrt{K}))$ | $x_{Kn+1}^{\text{suffix}}$ |
| **Ours Thm 4.5** | SS | $\tilde{O}(GD_\star/(n^{1/4}K^{1/4}) \lor GD_\star/\sqrt{n})$ | $x_{Kn+1}$ |
| Lower Bound (Koren 2022) | RR/SS | $\Omega(1/(n^{1/4}\sqrt{K}))$ | $x_{Kn+1}^{\text{suffix}}$ |

### Strongly Convex $\mu > 0$ Convergence Rates

| Method | Sampling | Convergence Rate |
|------|------|--------|
| Proximal GD / Liu&Zhou | ANY | $\tilde{O}(\mu D_\star^2/K^2 + G^2/(\mu K))$ |
| **Ours Thm 4.4** | RR | $\tilde{O}(\mu D_\star^2/(n^2K^2) + G^2/(\mu\sqrt{n}K))$ |
| **Ours Thm 4.7** | SS | $\tilde{O}(\mu D_\star^2/(n^2K^2) + G^2/(\mu\sqrt{nK}) + G^2/(\mu n))$ |
| Lower Bound | ANY | $\Omega(1/(nK))$ |

### Key Findings

- RR undergoes an acceleration factor of $\Theta(n^{1/4})$ under general convexity and $\Theta(n^{1/2})$ under strong convexity compared to Proximal GD—the larger $n$ is, the more pronounced the acceleration, accurately reflecting the advantage of traversing $n$ components every epoch.
- The bound of last-iterate directly implies the optimality of suffix average—indicating that the last-iterate analysis is sufficiently refined and does not require additional averaging operations.
- SS is weaker than RR but still superior to GD, suggesting that "even a single random permutation" can provide acceleration.
- The logarithmic factors can be eliminated when $K = \Omega(\log n)$ (which typically holds).

## Highlights & Insights

- **First proof of shuffling's strict advantage under nonsmoothness**: This ends the theoretical debate on whether shuffling outperforms GD in nonsmooth settings. The acceleration originates from the permutation guaranteeing uniform coverage of all components in each epoch, which reduces the variance of gradient estimates.
- **Last-iterate implying suffix average optimality**: This elegant theoretical result indicates that the last-iterate analysis kills two birds with one stone. In practice, the last iteration can be directly used without extra storage or computation of historical averages.
- **Breakthrough in analysis techniques**: The recursive technique analyzing function value changes step-by-step (rather than epoch-by-epoch) and new methods for handling permutation dependencies can be extended to other nonsmooth finite-sum optimization analyses.

## Limitations & Future Work

- Whether the rate of $\tilde{O}(G^2/(\mu\sqrt{n}K))$ for strongly convex RR is optimal remains an open question—since the lower bound is only $\Omega(1/(nK))$.
- Whether IG (Incremental Gradient / deterministic permutations) outperforms GD in nonsmooth scenarios is not addressed.
- The assumption that $f_i$ is convex and Lipschitz restricts its direct applicability to non-convex deep learning.
- The selection of step size relies on prior knowledge of parameters such as $G$, $D_\star$, and $n$; adaptive step-size analysis is a natural next step.

## Related Work & Insights

- **vs Liu & Zhou (2024b)**: Optimal in smooth cases but only achieved GD level ($O(1/\sqrt{K})$) in nonsmooth settings. This work improves the nonsmooth last-iterate to $\tilde{O}(1/(n^{1/4}\sqrt{K}))$, filling in the final puzzle piece.
- **vs Koren et al. (2022)**: Only established the upper bound for average iterate and the lower bound for suffix average. This work is the first to prove that the last-iterate also achieves near-optimal rates, while implying a matching upper bound for suffix average.
- **vs Mishchenko et al. (2020)**: Classic shuffling analysis in smooth strongly convex settings. This work extends it to the nonsmooth setting.

## Rating

- Novelty: ⭐⭐⭐⭐ First proof that shuffling last-iterate outperforms GD under nonsmoothness, resolving an important open problem.
- Experimental Thoroughness: ⭐⭐ Purely theoretical work without experimental validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear summary of results in Table 1, rigorous theorem statements.
- Value: ⭐⭐⭐⭐ Fills a crucial gap in shuffling gradient theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] From Average-Iterate to Last-Iterate Convergence in Games: A Reduction and Its Applications](../../NeurIPS2025/optimization/from_average-iterate_to_last-iterate_convergence_in_games_a_reduction_and_its_ap.md)
- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)
- [\[ICLR 2026\] Clipped Gradient Methods for Nonsmooth Convex Optimization under Heavy-Tailed Noise: A Refined Analysis](../../ICLR2026/optimization/clipped_gradient_methods_for_nonsmooth_convex_optimization_under_heavy-tailed_no.md)
- [\[ICML 2025\] Subspace Optimization for Large Language Models with Convergence Guarantees](subspace_optimization_for_large_language_models_with_convergence_guarantees.md)
- [\[ICLR 2026\] High-Probability Bounds for the Last Iterate of Clipped SGD](../../ICLR2026/optimization/high-probability_bounds_for_the_last_iterate_of_clipped_sgd.md)

</div>

<!-- RELATED:END -->
