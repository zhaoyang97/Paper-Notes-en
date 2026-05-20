---
title: >-
  [Paper Note] A Unified Approach to Submodular Maximization Under Noise
description: >-
  [NeurIPS 2025][Optimization][submodular maximization] This paper proposes a unified meta-algorithm framework that takes any exact submodular maximization algorithm satisfying a "robustness" condition as a black box and a…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "submodular maximization"
  - "noisy value oracle"
  - "meta-algorithm"
  - "approximation algorithms"
  - "matroid constraints"
date: 2026-05-08
content_hash: 0735ba0fe8a0ed62
---

# A Unified Approach to Submodular Maximization Under Noise

**Conference**: NeurIPS 2025
**arXiv**: [2510.21128](https://arxiv.org/abs/2510.21128)  
**Code**: None  
**Area**: Combinatorial Optimization / Theory
**Keywords**: submodular maximization, noisy value oracle, meta-algorithm, approximation algorithms, matroid constraints

## TL;DR

This paper proposes a unified meta-algorithm framework that takes any exact submodular maximization algorithm satisfying a "robustness" condition as a black box and automatically converts it into an algorithm that maintains its approximation ratio (losing only $o(1)$) under a persistent noisy value oracle. This achieves, for the first time, optimal approximation ratios for non-monotone submodular maximization under matroid constraints and in the unconstrained setting.

## Background & Motivation

**Background**: Submodular maximization is a foundational problem in machine learning, combinatorial optimization, and economics. Submodular functions exhibit diminishing marginal returns and arise broadly in feature selection, influence maximization, recommender systems, and related applications. The standard model assumes access to an exact value oracle for querying $f(S)$, under which a rich body of approximation algorithms has been developed: the greedy algorithm achieves the optimal $(1-1/e)$-approximation for monotone functions under cardinality constraints, and the double greedy algorithm achieves the optimal $1/2$-approximation for unconstrained non-monotone functions.

**Limitations of Prior Work**: In practice, exact value oracles are rarely available. In machine learning, for instance, the true population loss is inaccessible and only noisy estimates can be obtained. Hassidim et al. (2017) introduced a *persistent noise* model in which repeated queries to the same set always return the same noisy value, precluding denoising via repeated sampling. They showed that the standard greedy algorithm achieves only an $o(1)$-approximation under this model—noise completely destroys the original algorithm's guarantees.

**Key Challenge**: Prior work addresses each concrete problem setting (cardinality constraints, matroid constraints) with separately designed algorithms and analyses. Hassidim (2017) handles only the monotone + cardinality case; Huang (2022) extends to general matroids but with a factor-of-2 gap in the approximation ratio ($(1-1/e)/2$ instead of $(1-1/e)$); the non-monotone case remains entirely open. Each new setting requires designing a new algorithm from scratch, and no unified theory exists.

**Goal**: (1) Can a general framework be designed that directly reuses existing exact algorithms to address the noisy setting? (2) Can the optimal $(1-1/e)$-approximation be achieved under matroid constraints? (3) Can noisy maximization of non-monotone submodular functions be addressed for the first time?

**Key Insight**: Prior work employs *deterministic* surrogate functions to smooth noise, but deterministic surrogates cannot uniformly approximate the true function in all settings. The key insight of this paper is to use a *random* surrogate function—smoothing via randomly sampled sets—to achieve uniform approximation of the true function in expectation.

**Core Idea**: Construct a random surrogate function for denoising, then run any "robust" exact-value algorithm as a black box on the denoised surrogate, automatically inheriting the original algorithm's approximation guarantee.

## Method

### Overall Architecture

The meta-algorithm (Algorithm 1) proceeds as follows. Given a noisy oracle $\tilde{f}$ and a matroid constraint $\mathcal{I}$: (1) select a matroid base $B_0$; (2) sample uniformly at random a subset $H$ of size $h$ from $B_0$ (the smoothing set); (3) construct the surrogate function $F^{H,t}(S) = \mathbb{E}_{H' \sim \binom{H}{t}}[f(S \cup H')]$ and approximate its noisy version via sampling; (4) run an arbitrary robust algorithm $\mathcal{A}$ on the contracted matroid $\mathcal{I}_H$ to maximize the approximate noisy surrogate; (5) return $S_H \cup H'$, where $H'$ is a randomly chosen subset of $H$ of size $t$.

### Key Designs

1. **Random Surrogate Function**:

    - **Function**: Transforms the noisy value oracle into an approximate oracle for the true function.
    - **Mechanism**: Given smoothing set $H$ and parameter $t$, the surrogate is defined as $F^{H,t}(S) = \frac{1}{\binom{h}{t}} \sum_{H' \in \binom{H}{t}} f(S \cup H')$, i.e., the average over all size-$t$ subsets of $H$ unioned with $S$. Because each $f(S \cup H')$ corresponds to a distinct set query, the noise terms are independent, and averaging reduces variance. The choice of $t$ is critical: $t$ must be large enough so that $\binom{h}{t}$ is sufficiently large to ensure denoising, yet not so large that adding many elements substantially degrades the function value in the non-monotone case.
    - **Design Motivation**: Hassidim (2017) uses a deterministic surrogate (averaging over all subsets of $H$), but this requires algorithm-specific tailored analysis. Randomly choosing $H$ ensures that the surrogate differs from the true function in expectation only by a factor depending on $h/r$, enabling black-box use of the inner algorithm.

2. **Robustness Condition**:

    - **Function**: Specifies which exact algorithms are usable by the meta-algorithm.
    - **Mechanism**: Algorithm $\mathcal{A}$ is $\beta(\varepsilon)$-robust to an $\varepsilon$-approximate oracle $\hat{f}$ (satisfying $|\hat{f}(S) - f(S)| \leq \varepsilon$ for all queried sets) if the solution quality degrades by at most $\beta(\varepsilon)$ when using the approximate oracle. Crucially, this is an "interface condition": it suffices to verify that an existing algorithm satisfies it, without modifying the algorithm itself.
    - **Design Motivation**: This decouples noise handling (the meta-algorithm's responsibility) from optimization logic (the existing algorithm's responsibility). The paper verifies that both the measured continuous greedy and double greedy algorithms satisfy the robustness condition.

3. **Smoothing Lemma**:

    - **Function**: Establishes the relationship between the optimal value of the random surrogate and that of the original function.
    - **Mechanism**: Using the basis exchange property of matroids, the paper proves $\mathbb{E}_H[\max_{S \in \mathcal{I}_H} F^{H,t}(S)] \geq (1 - h/(r-h) - t/(h-t)) f(O^*)$ for the non-monotone case; the $t/(h-t)$ term disappears in the monotone case. This ensures that maximizing the surrogate on the contracted matroid does not deviate excessively from the optimum of the original problem.
    - **Design Motivation**: This is the theoretical cornerstone of the meta-algorithm. When $h = o(r)$, the approximation factor approaches 1, guaranteeing that the meta-algorithm's output is close to optimal.

### Parameter Selection and Query Complexity

Parameters are set as $m \geq O(\nu^2 f_{max}^2 / \varepsilon^2 \cdot (n + \log(1/\delta)))$, $t \geq \log_2(4m)$, and $h = t^2$. The total number of oracle queries is $m \cdot Q(\mathcal{A})$, where $Q(\mathcal{A})$ is the query complexity of the inner algorithm. The noise distribution is assumed to be sub-exponential, a broad class that includes bounded, Gaussian, and exponential distributions.

## Key Experimental Results

### Main Results

| Submodular Function | Constraint | Prev. SOTA | Ours | Notes |
|---|---|---|---|---|
| Monotone | Cardinality | $(1-1/e)$ [Hassidim 2017] | $(1-1/e)$ | Optimal, matches prior work |
| Monotone | Matroid | $(1-1/e)/2$ [Huang 2022] | **$(1-1/e)$** | 2× improvement, optimal |
| Non-monotone | Unconstrained | None | **$1/2$** | First result, optimal |
| Non-monotone | Matroid | None | **$1/e$** | First result |

### Ablation Study (Theoretical)

| Variant | Effect | Remarks |
|---|---|---|
| Deterministic surrogate → Random surrogate | Eliminates algorithm-specific analysis | Core contribution |
| All subsets of $H$ vs. subsets of size $t$ | The latter is necessary for non-monotone case | Controls the penalty from adding elements to non-monotone functions |
| Increasing $h, t$ | Better denoising but worse approximation factor | Balanced when $h = o(r)$ |

### Key Findings

- **Unification**: A single meta-algorithm with different black-box inner algorithms covers all four settings (monotone/non-monotone × cardinality/matroid/unconstrained), eliminating the need to design separate algorithms for each problem variant.
- **Tightness**: All approximation ratios achieved are optimal even in the noiseless setting (matching information-theoretic lower bounds); noise incurs only an $o(1)$ loss.
- **Robustness Verification as Key Technical Work**: The generality of the meta-algorithm hinges on verifying the robustness of specific inner algorithms. The paper carefully establishes the robustness of measured continuous greedy and double greedy, which constitutes a technically demanding part of the contribution.

## Highlights & Insights

- **Elegant Black-Box Reduction**: The meta-algorithm requires no knowledge of the inner algorithm's internals—only that it satisfies the robustness condition. This design ensures that any future exact algorithm can automatically yield a noise-tolerant variant, greatly enhancing the framework's extensibility.
- **Ingenuity of the Random Surrogate**: Randomization circumvents the inherent limitations of deterministic surrogates—deterministic surrogates require algorithm-specific tailored analysis, whereas the random surrogate requires only a single unified smoothing lemma at the expectation level.
- **Practical Implications**: In machine learning, submodular optimization almost always operates with noisy estimates (e.g., cross-validation estimates in feature selection). The framework implies that standard implementations of greedy-type algorithms can be used directly, with denoising handled by a surrogate function layer prepended to the pipeline.

## Limitations & Future Work

- **High Query Complexity**: The query complexity of the meta-algorithm is $m \cdot Q(\mathcal{A})$, where $m = \tilde{O}(n^5/\varepsilon^2)$. For large-scale problems, this may be impractical. Reducing query complexity is an important open problem.
- **Persistent Noise Assumption**: The model assumes persistent noise (repeated queries to the same set return the same value). In some applications noise may be non-persistent, in which case the problem becomes easier (direct repeated-sampling denoising suffices) and the techniques of this paper do not directly apply.
- **Sub-Exponential Noise Assumption**: The noise distribution parameters $(\nu, \alpha)$ are assumed known. In practice these parameters may need to be estimated, introducing additional uncertainty.
- **Purely Theoretical Contribution**: No empirical validation is provided. While standard for theoretical papers, the lack of experiments on practical application scenarios (e.g., budget-constrained feature selection, social influence maximization) is a limitation.

## Related Work & Insights

- **vs. Hassidim et al. (2017)**: They design a specialized denoising greedy algorithm for the monotone + cardinality setting, achieving $(1-1/e)$. The proposed meta-algorithm matches this result in the same setting while additionally covering all other settings.
- **vs. Huang et al. (2022)**: They apply a local search algorithm with a deterministic surrogate for matroid constraints, but achieve only $(1-1/e)/2$. This paper achieves the optimal $(1-1/e)$ via a random surrogate combined with measured continuous greedy, eliminating the factor-of-2 gap.
- **vs. Approximate Submodular Optimization (Horel & Singer)**: That line of work studies functions that are themselves approximately submodular (adversarial noise), whereas this paper studies functions that are exactly submodular but whose value oracle is noisy (stochastic noise). The problem settings are fundamentally different—adversarial noise admits exponential lower bounds, making that setting harder.
- **vs. Online Submodular Maximization**: Online settings also require "robust" algorithms to handle uncertainty, suggesting methodological similarities. A deeper unification between the two directions may be possible.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The random surrogate + black-box reduction framework is elegant and powerful, unifying previously fragmented results.
- **Experimental Thoroughness**: ⭐⭐⭐ A purely theoretical work with tight theoretical results, but no empirical validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is clearly structured with natural motivation, rigorous technical presentation, and proof sketches that are easy to follow.
- **Value**: ⭐⭐⭐⭐ Significant contribution to the submodular optimization theory community, unifying scattered results and opening new directions for the non-monotone setting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](online_two-stage_submodular_maximization.md)
- [\[NeurIPS 2025\] perturbation bounds for low-rank inverse approximations under noise](perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[NeurIPS 2025\] Effective Policy Learning for Multi-Agent Online Coordination Beyond Submodular Objectives](effective_policy_learning_for_multi-agent_online_coordination_beyond_submodular_.md)

</div>

<!-- RELATED:END -->
