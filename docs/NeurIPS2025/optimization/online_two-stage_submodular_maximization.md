---
title: >-
  [Paper Note] Online Two-Stage Submodular Maximization
description: >-
  [NeurIPS 2025][Optimization][submodular maximization] This paper introduces the Online Two-Stage Submodular Maximization (O2SSM) problem for the first time…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "submodular maximization"
  - "online learning"
  - "two-stage optimization"
  - "regret minimization"
  - "matroid constraints"
date: 2026-05-08
content_hash: 5f6c935e1df6dda8
---

# Online Two-Stage Submodular Maximization

**Conference**: NeurIPS 2025
**arXiv**: [2510.19480](https://arxiv.org/abs/2510.19480)  
**Code**: [GitHub](https://github.com/jasonNikolaou/online-two-stage-sub-max)  
**Area**: Optimization / Submodular Functions / Online Learning
**Keywords**: submodular maximization, online learning, two-stage optimization, regret minimization, matroid constraints

## TL;DR
This paper introduces the Online Two-Stage Submodular Maximization (O2SSM) problem for the first time, and proposes the RAOCO algorithm for Weighted Threshold Potential (WTP) functions. By combining fractional relaxation with randomized pipage rounding, RAOCO achieves sublinear $(1-1/e)^2$-regret in polynomial time, while also improving the approximation ratio for the offline problem.

## Background & Motivation
**Background**: Two-Stage Submodular Maximization (2SSM) is a classical problem introduced by Balkanski et al. (2016), in which the first stage selects a constrained ground set $S$ and the second stage maximizes a randomly drawn submodular objective over $S$. Applications include recommendation systems (prefetching $\ell$ items so that $k$ can be recommended upon user arrival), data summarization, and dictionary learning.

**Limitations of Prior Work**: Existing work assumes the function family $\mathcal{F}$ is known in advance (offline setting). When $\mathcal{F}$ is unknown and objective functions arrive in an online sequence, the problem becomes significantly more challenging.

**Key Challenge**: O2SSM faces two compounding difficulties: (1) the first-stage objective $F(\cdot)$ is non-submodular, precluding direct application of existing online submodular maximization techniques; and (2) even when the function is known, computing $F(\cdot)$ is NP-hard, so oracle access cannot be assumed.

**Key Insight**: The paper restricts attention to the WTP function subclass and constructs a concave continuous relaxation that enables the problem to be solved via Online Convex Optimization (OCO).

**Core Idea**: Construct an efficiently computable concave relaxation for the WTP objective, combined with randomized pipage rounding, to reduce O2SSM to an OCO problem.

## Method

### Overall Architecture
At each round, the RAOCO algorithm: (1) applies an OCO strategy (FTRL/OGA) in the fractional domain $\tilde{\mathcal{X}}_\ell$ to compute a fractional restricted set $\tilde{x}_t$; (2) applies Randomized Pipage Rounding to convert the fractional solution to an integral set $x_t$; (3) upon observing the objective function $f_t$, constructs the concave relaxation $\tilde{F}_t$ for use in the next OCO step.

### Key Designs

1. **Construction and Properties of the Concave Relaxation**:

    - *Function*: Relaxes the NP-hard discrete reward function $F(x) = \max_{y \in \mathcal{Y}(x)} f(y)$ into a tractable continuous optimization problem.
    - *Mechanism*: For a WTP function $f(y) = \sum_j c_j \min\{b_j, y \cdot w_j\}$, relaxing the domain to $[0,1]^n$ naturally preserves concavity. The relaxed objective $\tilde{F}(\tilde{x}) = \max_{\tilde{y} \in \tilde{\mathcal{Y}}(\tilde{x})} \tilde{f}(\tilde{y})$ is also concave and Lipschitz continuous, with supergradients computable in polynomial time via linear programming.
    - *Design Motivation*: Concavity combined with computable supergradients enables direct application of FTRL/OGA, circumventing the dual obstacles of non-submodularity and NP-hardness.

2. **Approximation Guarantees of Randomized Pipage Rounding**:

    - *Function*: Converts fractional solutions to integral solutions without loss in expectation.
    - *Mechanism*: The paper proves that the distribution produced by pipage rounding satisfies a submodular dominance property, enabling the application of a Contention Resolution Scheme (CRS) to obtain the approximation guarantee: $\mathbb{E}[F(\Xi(\tilde{x}))] \geq c_\mathcal{M} \cdot (1-1/e) \cdot \tilde{F}(\tilde{x})$.
    - *Design Motivation*: Standard CRS requires coordinate-independent sampling, which pipage rounding does not satisfy; the submodular dominance property provides an alternative guarantee.

3. **The $\alpha$-Regret Framework**:

    - Performance is measured against the best $\alpha$-approximate offline algorithm with full knowledge of the function sequence, rather than against the exact optimum.
    - Main theorem: RAOCO achieves $(1-1/e)^2$-regret $= O(\ell GM^2\sqrt{nT})$ for general matroid constraints, and a stronger $(1-1/e)(1-e^{-k}k^k/k!)$-regret for rank-$k$ uniform matroids.

## Key Experimental Results

### Main Results (7 Datasets)

| Dataset | n | ℓ | k | Description |
|--------|---|---|---|------|
| Wikipedia | 407 | 20 | 5 | Representative article selection |
| Images | 150 | 20 | 5 | Image collection summarization |
| MovieRec | 1000 | 10 | 4 | Movie recommendation |
| Influence | 34 | 8 | 3 | Influence maximization |
| HotpotQA | 111,140 | 150 | 4 | QA corpus summarization |

Across all datasets, all three RAOCO variants (OGA, FTRL-L2, FTRL-H) outperform or match the one-stage baseline (1S-OGA), with particularly notable advantages on the Influence, Images, and Coverage datasets. On the Coverage dataset, RAOCO achieves nearly twice the average reward of 1S-OGA.

### Ablation Study

| Method | Type | Description |
|------|------|------|
| RAOCO-OGA/FTRL | Online two-stage | Proposed method; consistently optimal |
| 1S-OGA | Online one-stage | Ignores two-stage structure; suboptimal |
| OFLN-CO | Offline | Balkanski et al. (2016); requires $k>1$ |
| OFLN-RGR | Offline | Stan et al. (2017); coincides with OPT on all datasets |
| Random | Online | Random selection; lower-bound reference |

### Key Findings
- Two-stage-aware online algorithms significantly outperform one-stage online algorithms, validating the value of explicitly modeling the two-stage structure.
- RAOCO's online performance closely approaches the offline optimal algorithm (OFLN-RGR), demonstrating the effectiveness of the concave relaxation and rounding strategy.
- The three OCO strategies (OGA, FTRL-L2, FTRL-H) perform comparably, indicating that the algorithm is robust to the choice of strategy.

## Highlights & Insights
- **Solid theoretical contributions**: The paper provides the first polynomial-time algorithm with sublinear regret for O2SSM, while also improving the approximation ratio for offline 2SSM under cardinality constraints.
- **Elegant concave relaxation**: The min-sum structure of WTP functions ensures that domain relaxation naturally preserves concavity, avoiding the difficulty that general submodular function relaxations are typically non-concave.
- **Novel use of submodular dominance in rounding analysis**: This bypasses the coordinate independence requirement of standard CRS.
- **Strong theory-practice alignment**: Experimental results are consistent with theoretical guarantees, confirming that the online algorithm reliably approaches the offline optimum.

## Limitations & Future Work
- The approach is restricted to the WTP function class and cannot handle general monotone submodular functions.
- Full-information feedback is assumed (the complete objective function is observed each round); extension to bandit feedback remains an open problem.
- The experimental scale is moderate (maximum $n = 111{,}140$); scalability to larger settings requires further investigation.

## Related Work & Insights
- **vs. Balkanski et al. (2016)**: Their offline algorithm uses basic randomized rounding; this paper employs pipage rounding with CRS to improve the approximation ratio.
- **vs. Stan et al. (2017)**: They propose an offline algorithm for general matroid constraints (approximation ratio 0.43); the online version in this paper achieves 0.40, a close match.
- **vs. Si Salem et al. (2024)**: They address one-stage online WTP maximization; this paper generalizes the setting to two stages.
- The work has direct applicability to prefetching and caching strategy design in recommendation systems.

## Rating
- Novelty: ⭐⭐⭐⭐ First formulation and solution of O2SSM; novel combination of theoretical tools.
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven datasets covering diverse applications, with comparisons and variant analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear, theoretical derivations are rigorous, and notation is consistent throughout.
- Value: ⭐⭐⭐⭐ Offers both theoretical and practical contributions to online combinatorial optimization and recommendation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[NeurIPS 2025\] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)
- [\[NeurIPS 2025\] Optimistic Online-to-Batch Conversions for Accelerated Convergence and Universality](optimistic_online-to-batch_conversions_for_accelerated_convergence_and_universal.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)
- [\[NeurIPS 2025\] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)

</div>

<!-- RELATED:END -->
