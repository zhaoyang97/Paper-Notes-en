---
title: >-
  [Paper Note] Product Distribution Learning with Imperfect Advice
description: >-
  [NeurIPS 2025][distribution learning] This paper studies the problem of learning product distributions over the Boolean hypercube given an imperfect advice distribution, and proposes an efficient algorithm that achieves sub-linear dependence on dimension $d$ in sample complexity when the advice is of sufficient quality.
tags:
  - NeurIPS 2025
  - distribution learning
  - algorithms with predictions
  - product distributions
  - sample complexity
  - tolerant testing
date: 2026-05-08
content_hash: bc1ea4ca50344e16
---

# Product Distribution Learning with Imperfect Advice

**Conference**: NeurIPS 2025
**arXiv**: [2511.10366](https://arxiv.org/abs/2511.10366)
**Code**: None
**Area**: Learning Theory / Distribution Learning
**Keywords**: distribution learning, algorithms with predictions, product distributions, sample complexity, tolerant testing

## TL;DR

This paper studies the problem of learning product distributions over the Boolean hypercube given an imperfect advice distribution, and proposes an efficient algorithm that achieves sub-linear dependence on dimension $d$ in sample complexity when the advice is of sufficient quality.

## Background & Motivation

Distribution learning is one of the foundational problems in statistical learning theory. Given i.i.d. samples drawn from an unknown distribution $P$, the goal is to recover a distribution that is close to $P$ in total variation distance. For product distributions over $\{0,1\}^d$, the known optimal sample complexity is $\Theta(d/\varepsilon^2)$, achieved by a simple empirical mean estimator.

In practice, however, learners often have access to prior information from related datasets or previously learned models, which can be viewed as "advice." Under the framework of algorithms with predictions, this paper investigates whether a learner, given both samples and an advice mean vector $\mathbf{q}$, can leverage high-quality advice to reduce sample complexity.

The key challenge is that the quality of $\mathbf{q}$ is unknown; the algorithm must accelerate when the advice is accurate and remain robust when it is not.

## Method

### Overall Architecture

The algorithm TestAndOptimizeMean proceeds in two phases:

1. **Advice quality estimation**: Uses the ApproxL1 algorithm to estimate $\|\mathbf{p} - \mathbf{q}\|_1$.
2. **Learning phase**: Based on the estimate, either performs constrained optimization exploiting the advice or falls back to standard empirical mean estimation.

If the advice is detected to be sufficiently good ($\lambda < \varepsilon\sqrt{d}$), an $\ell_1$-constrained LASSO estimator is applied; otherwise, the algorithm degrades to standard empirical mean estimation.

### Key Designs

**Tolerant Mean Tester (TMT)**

This is the core component of the algorithm. Given advice $\mathbf{q}$ and samples drawn from $\text{Ber}(\mathbf{p})$:
- If $\|\mathbf{p} - \mathbf{q}\|_2 \leq \varepsilon$, output Accept.
- If $\|\mathbf{p} - \mathbf{q}\|_2 \geq 2\varepsilon$, output Reject.

The sample complexity is $O(\sqrt{d}/\varepsilon^2 \cdot \log(1/\delta))$. The test statistic is defined as:

$$Z = \sum_{i=1}^d Z_i, \quad Z_i = (X_i - mq_i)^2 - X_i$$

where $X_i \sim \text{Poi}(mp_i)$. The test exploits $\mathbb{E}[Z] = m^2\|\mathbf{p}-\mathbf{q}\|_2^2$ and applies Chebyshev's inequality for the decision.

**ApproxL1 Algorithm**

To estimate the $\ell_1$ distance, the algorithm partitions the $d$ coordinates into $d/k$ blocks of size $k$. Within each block:
1. Binary search with TMT calls determines $\|\mathbf{p}_{B_j} - \mathbf{q}_{B_j}\|_2$.
2. The $\ell_1$–$\ell_2$ norm relationship ($\|\cdot\|_1 \leq \sqrt{k}\|\cdot\|_2$) converts within-block $\ell_2$ estimates to $\ell_1$ estimates.
3. Estimates across all blocks are aggregated to approximate the global $\ell_1$ distance.

The key benefit of blocking is that within each block of size $k$, the ratio of $\ell_1$ to $\ell_2$ norms is reduced from $\sqrt{d}$ to $\sqrt{k}$, yielding tighter estimates.

**Constrained LASSO Estimator**

When $\|\mathbf{p}-\mathbf{q}\|_1 \leq r$ is known, the algorithm solves:

$$\hat{\mathbf{p}} = \arg\min_{\|\mathbf{b}-\mathbf{q}\|_1 \leq r} \frac{1}{n}\sum_{i=1}^n \|\mathbf{y}_i - \mathbf{b}\|_2^2$$

Only $O(r^2/\varepsilon^4 \cdot \log(d/\delta))$ samples suffice to achieve $\|\hat{\mathbf{p}} - \mathbf{p}\|_2 \leq \varepsilon$.

### Loss & Training

This is a theoretical work and involves no empirical training. The core optimization objective is a constrained least-squares (LASSO) problem solvable in $\text{poly}(n,d)$ time.

## Key Experimental Results

### Main Results (Theoretical)

This paper is a purely theoretical contribution. The central results consist of matching upper and lower bounds:

| Setting | Sample Complexity Upper Bound | Sample Complexity Lower Bound |
|---|---|---|
| No advice | $\Theta(d/\varepsilon^2)$ | $\Theta(d/\varepsilon^2)$ |
| Perfect advice $\mathbf{q}=\mathbf{p}$ | $\Theta(\sqrt{d}/\varepsilon^2)$ | $\Omega(\sqrt{d}/\varepsilon^2)$ |
| $\tau$-balanced + good advice | $\tilde{O}(d^{1-\eta}/\varepsilon^2)$ | — |
| $\tau$-balanced + $\|\mathbf{p}-\mathbf{q}\|_1 = \lambda$ | $\tilde{O}(d/\varepsilon^2 \cdot (d^{-\eta} + \min\{1, \lambda^2/(d^{1-4\eta}\tau^6\varepsilon^2)\}))$ | $\tilde{\Omega}(\min\{\lambda^2/\varepsilon^4, d/\varepsilon^2\})$ |
| Unbalanced + $O(1)$-good advice | $\Theta(d/\varepsilon^2)$ (no improvement possible) | $\tilde{\Omega}(d/\varepsilon)$ |

### Ablation Study (Theoretical Lower Bound Analysis)

| Setting | Conclusion |
|---|---|
| Unbalanced distribution ($p_i = O(1/d)$), advice $\ell_1$ error $O(\varepsilon)$ | Still requires $\Omega(d/\varepsilon)$ samples; sub-linearity is impossible |
| Balanced distribution, $\|\mathbf{p}-\mathbf{q}\|_1 \geq \varepsilon\sqrt{d}$ | Still requires $\Omega(d/\varepsilon^2)$ samples; advice is unhelpful |
| Balanced distribution, $\|\mathbf{p}-\mathbf{q}\|_1 = \lambda \ll \varepsilon\sqrt{d}$ | Sample complexity becomes sub-linear |

### Key Findings

1. **Balance is a necessary condition**: When coordinate values can approach 0 or 1, sub-linear sample complexity is unachievable even when the advice is close in $\ell_1$. This contrasts sharply with identity testing, where $O(\sqrt{d}/\varepsilon^2)$ samples suffice without any balance assumption.
2. **Adaptivity**: The algorithm requires no prior knowledge of advice quality and adapts automatically—accelerating with good advice and not degrading with poor advice.
3. **Choice of $\ell_1$ distance**: Motivated by the compressed sensing literature, the $\ell_1$ distance is a natural measure of advice quality, as it characterizes sparse discrepancies.

## Highlights & Insights

- Extending the algorithms-with-predictions framework from online problems to distribution learning represents a pioneering contribution to statistical learning theory.
- The combination of blocking, tolerant testing, and LASSO is elegant, with each component having a clear technical motivation.
- The proof of the necessity of balance relies on a sophisticated combination of the Gilbert–Varshamov bound and Fano's method.
- A natural comparison with concurrent work on the Gaussian setting (Bhattacharyya et al. 2025) highlights fundamental differences between discrete and continuous distributions under the advice framework.

## Limitations & Future Work

1. A gap remains between upper and lower bounds (the $\tau$-dependence in the upper bound is $\tau^{-6}$; whether this can be improved is an open question).
2. The results are restricted to product distributions and do not cover more complex distribution families such as Bayesian networks or Ising models.
3. Advice is provided in the form of a mean vector; how to obtain high-quality advice in practice remains an open problem.
4. Extensions to the communication complexity or distributed settings are not discussed.

## Related Work & Insights

- **Algorithms with predictions**: Rich prior work exists for ski-rental, online scheduling, and secretary problems; this paper is among the few to bring the framework into statistical learning.
- **Distribution testing**: Identity testing results by Daskalakis–Pan and Canonne et al. ($\Theta(\sqrt{d}/\varepsilon^2)$) provide the foundation for the tester design in this work.
- **Compressed sensing**: The idea of restricting the search space to an $\ell_1$ ball is directly borrowed from the LASSO literature.
- Future work may explore extending this framework to unstructured distribution learning over discrete domains $[n]$.

## Rating

- **Novelty**: ★★★★☆ (Introducing the algorithms-with-predictions framework to distribution learning is theoretically original)
- **Theoretical Depth**: ★★★★★ (Rigorous upper and lower bound analysis with technically sophisticated blocking)
- **Practicality**: ★★☆☆☆ (Purely theoretical; practical applicability is limited)
- **Clarity**: ★★★★☆ (Technical overview is clear with well-organized proof structure)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Radar: Benchmarking Language Models on Imperfect Tabular Data](radar_benchmarking_language_models_on_imperfect_tabular_data.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[NeurIPS 2025\] Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis](double_descent_meets_out-of-distribution_detection_theoretical_insights_and_empi.md)
- [\[NeurIPS 2025\] Harnessing Feature Resonance under Arbitrary Target Alignment for Out-of-Distribution Node Detection](harnessing_feature_resonance_under_arbitrary_target_alignment_for_out-of-distrib.md)

<!-- RELATED:END -->
