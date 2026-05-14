---
title: >-
  [Paper Note] Skirting Additive Error Barriers for Private Turnstile Streams
description: >-
  [ICLR 2026][AI Safety][differential privacy] This paper proves that the polynomial pure additive error lower bounds in differentially private turnstile streams—$\Omega(T^{1/4})$ for distinct elements counting and $\Omega…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "differential privacy"
  - "continual release"
  - "turnstile stream"
  - "distinct elements"
  - "F2 moment"
date: 2026-05-08
content_hash: 9d696f9fa1ee86d7
---

# Skirting Additive Error Barriers for Private Turnstile Streams

**Conference**: ICLR 2026
**arXiv**: [2602.10360](https://arxiv.org/abs/2602.10360)
**Code**: None
**Area**: Differential Privacy / Streaming Algorithms
**Keywords**: differential privacy, continual release, turnstile stream, distinct elements, F2 moment

## TL;DR

This paper proves that the polynomial pure additive error lower bounds in differentially private turnstile streams—$\Omega(T^{1/4})$ for distinct elements counting and $\Omega(T)$ for $F_2$ moment estimation—can be circumvented by introducing multiplicative error. The paper achieves $(\text{polylog}(T), \text{polylog}(T))$ mixed error for distinct elements and $(1+\eta, \text{polylog}(T))$ mixed error for $F_2$ moments, both requiring only polylogarithmic space.

## Background & Motivation

**Background**: Differentially private continual release is a fundamental model for privacy-preserving data processing, where data arrives as a stream and the algorithm must privately publish the underlying statistics at each time step. The turnstile model allows both insertions and deletions, making it significantly harder than the insertion-only setting. Two foundational streaming problems—distinct elements counting and $F_2$ moment estimation—are classical cornerstones of the streaming algorithms literature.

**Limitations of Prior Work**: In the private continual release setting over turnstile streams, a polynomial gap exists between known upper and lower bounds for both problems, even without space constraints. The best known pure additive error upper bound for distinct elements is $\tilde{O}(T^{1/3})$, with a lower bound of $\Omega(T^{1/4})$; closing this gap is a difficult open problem. The situation for $F_2$ moments is worse—sensitivity arguments alone force any pure additive error algorithm to incur $\Omega(T)$ error. These polynomial-level unavoidable errors severely limit practical applicability.

**Key Challenge**: While these lower bounds appear insurmountable, a careful analysis of the hard instances underlying the constructions reveals that whenever the lower bounds hold, the true statistic is itself far larger than the additive error. Specifically, the $\Omega(T^{1/4})$ lower bound arises in scenarios where the number of distinct elements greatly exceeds $T^{1/4}$, meaning the lower bound poses no barrier to achieving a constant multiplicative approximation.

**Key Insight**: The paper introduces the $(\alpha, \beta)$ mixed error model, which permits both a multiplicative factor $\alpha$ and an additive term $\beta$. When the true value $Y_t \gg \beta$, the output is essentially an $\alpha$-multiplicative approximation; when $Y_t$ is small (below the noise floor), a larger relative error is tolerated. This relaxation is natural in non-private sublinear-space streaming (where multiplicative error is inherent) and is equally well-motivated in the private setting.

**Core Idea**: Polynomial pure additive error can be replaced by polylogarithmic additive error at the cost of introducing some multiplicative error.

## Method

### Overall Architecture

The central approach is to use differentially private continual counting (DP continual counting) as a fundamental primitive, and to carefully reduce distinct elements and $F_2$ moment estimation to collections of continual counting instances via classical streaming techniques (MinHash, domain reduction, Johnson–Lindenstrauss dimensionality reduction), thereby exploiting the polylogarithmic additive error guarantee of continual counting. The two problems admit distinct algorithmic strategies.

### Key Designs

1. **MinHash Algorithm (Theorem 3.1) — Strict Turnstile Streams**:

    - **Function**: Estimates distinct elements with mixed error via hash buckets and continual counters.
    - **Mechanism**: A hash function $h: [n] \to [n]$ is chosen, and elements are assigned to $K+1$ buckets according to the least significant non-zero bit $\texttt{lsb}(h(a))$. Since the probability of $\texttt{lsb} = k$ is $2^{-(k+1)}$, the expected number of distinct elements in bucket $k$ decreases geometrically as $2^{-(k+1)} \cdot D_t$. A DP continual counter $C[k]$ is maintained for each bucket, yielding an estimate $\hat{f}_t[k]$ with additive error $\tau = O(\log^{1.5}(T)/\sqrt{\rho})$. The classical (non-private) estimator identifies the largest non-empty bucket $\ell$ and reports $\hat{D} = 2^\ell$. Under privacy constraints, the largest bucket whose count exceeds the noise threshold $\tau$ is selected. Because it is impossible to distinguish "a bucket count elevated by $\tau$ frequency-one elements" from "a single high-frequency element," this introduces $O(\tau)$ multiplicative error. The total error is $(O(\log^2(T)/\sqrt{\rho}),\, O(\log^2(T)/\sqrt{\rho}))$ with space $O(\log n \cdot \log^2(T))$.
    - **Design Motivation**: The classical minimum-hash estimator $1/(\min h)$ is unsuitable in the private setting—the minimum hash value may change frequently due to single events, yielding unfavorable sensitivity. The bucket-plus-continual-counter transformation reduces the problem to multiple low-sensitivity counting instances.

2. **Domain Reduction Algorithm (Theorem 4.1) — General Turnstile Streams**:

    - **Function**: Achieves more general mixed error estimation via hash-based dimensionality reduction and collision detection.
    - **Mechanism**: A hash function reduces the domain $[n]$ to a sufficiently small domain so that many elements collide. Continual counters on the reduced domain estimate the number of elements per bucket, and the reduced domain size serves as a proxy for the number of distinct elements. This approach incurs larger error $(O(\log^{10}(T)/\rho^2),\, O(\log^{10}(T)/\rho^2))$ than MinHash but applies to general turnstile streams (where frequencies may be negative). The paper also establishes an important reduction (Theorem 4.2): any algorithm achieving sublinear pure additive error $n^{0.99}$ over domain size $n$ can be converted into an algorithm with $(1+\eta)$ multiplicative and $\text{polylog}(T)$ additive error, providing a pathway toward improving the multiplicative factor to a constant.
    - **Design Motivation**: MinHash requires strict turnstile streams (non-negative frequencies at all times); domain reduction removes this restriction.

3. **$F_2$ Moment Estimation (JL Reduction, Theorem 5.1)**:

    - **Function**: Reduces $F_2$ moment estimation to a low-dimensional continual counting problem.
    - **Mechanism**: Johnson–Lindenstrauss dimensionality reduction maps the $n$-dimensional frequency vector $x_t \in \mathbb{R}^n$ to an $m$-dimensional space ($m = O(\log(T)/\eta^2)$), exploiting the norm-preservation guarantee $(1-\eta)\|x_t\|_2^2 \leq \|Ax_t\|_2^2 \leq (1+\eta)\|x_t\|_2^2$. A continual counter is maintained per coordinate of the projected space, and the sum of squared coordinates estimates $F_2$. A Rademacher random matrix (entries $\pm 1/\sqrt{m}$) is used; each stream update affects a single domain element $a_i$, so the increment to each projected coordinate is $\pm s_i/\sqrt{m}$ and can be directly fed to the continual counter. The total error is $(1+\eta,\, O(\text{polylog}(T)/(\varepsilon^2\eta^3)))$ with space $O(\text{polylog}(T)/\eta^2)$.
    - **Design Motivation**: The sensitivity of $F_2$ is $\Omega(T)$ in the original space (a single event change can shift the squared $\ell_2$ norm of the frequency vector by $O(T)$), making pure additive error $\Omega(T)$ unavoidable. After JL projection, the problem reduces to low-dimensional counting where small additive error suffices for a good $(1+\eta)$ multiplicative approximation. This simultaneously improves the additive error of Epasto et al. on insertion-only streams from $O(\log^7(T))$ to $O(\log^4(T))$ and extends the result to turnstile streams.

### Privacy Guarantees

All algorithms employ the zero-concentrated differential privacy (zCDP) framework, implementing continual counting via the Gaussian Binary Tree Mechanism. $\rho$-zCDP can be converted to $(\varepsilon,\delta)$-DP (with $\rho = O(\varepsilon^2/\log(1/\delta))$) and admits tighter composition than standard DP.

## Key Experimental Results

### Theoretical Comparison for Distinct Elements Counting

| Source | Error $(\alpha, \beta)$ | Space | Privacy | Notes |
|--------|------------------------|-------|---------|-------|
| Jain et al. '23 upper bound | $(1, \tilde{O}(T^{1/3}))$ | $O(T)$ | Item-level | Based on recomputation |
| Jain et al. '23 lower bound | $(1, \Omega(T^{1/4}))$ | — | Event-level | Pure additive lower bound |
| Epasto et al. '23 | $(1+\eta, O_\eta(\log^2 T))$ | polylog$(T)$ | Event-level | **Insertion-only** |
| **MinHash (Thm 3.1)** | $(O(\log^2 T), O(\log^2 T))$ | $O(\log^3 T)$ | Event-level | **Strict turnstile** |
| **Domain Red. (Thm 4.1)** | $(O(\log^{10} T), O(\log^{10} T))$ | poly$(T)$ | Event-level | **General turnstile** |

Core conclusion: the pure additive error is reduced from $\Omega(T^{1/4})$ (polynomial) to $\text{polylog}(T)$ (polylogarithmic) at the cost of introducing $\text{polylog}(T)$ multiplicative error. When the true number of distinct elements exceeds $\tilde{O}(T^{1/3})$, the present results strictly dominate the previous best upper bound.

### Theoretical Comparison for $F_2$ Moment Estimation

| Source | Error $(\alpha, \beta)$ | Space | Notes |
|--------|------------------------|-------|-------|
| Sensitivity lower bound | $(1, \Omega(T))$ | — | Pure additive unavoidable |
| Epasto et al. '23 | $(1+\eta, \tilde{O}_\eta(\log^7 T))$ | $O_\eta(\log^2 T)$ | **Insertion-only** |
| **Theorem 5.1** | $(1+\eta, \tilde{O}_\eta(\log^4 T))$ | $O_\eta(\log^2 T)$ | **General turnstile** |

The additive error is reduced from $\Omega(T)$ to $\text{polylog}(T)$ at the cost of only a $1+\eta$ multiplicative factor. Compared to Epasto et al., the additive error improves from $\log^7(T)$ to $\log^4(T)$ and the result is extended from insertion-only to turnstile streams.

### Key Findings

- **Blind spot in lower bound constructions**: In the hard instances underlying the $\Omega(T^{1/4})$ additive lower bound, the true number of distinct elements is far greater than $T^{1/4}$, so the lower bound poses no barrier to multiplicative approximation—this is the central observation of the paper.
- **Explicit source of multiplicative error in MinHash**: The $O(\tau)$ multiplicative error arises from the inability to distinguish "multiple low-frequency elements" from "a single high-frequency element" falling into the same bucket, pinpointing the technical bottleneck for reducing the multiplicative factor to a constant.
- **Significance of the Theorem 4.2 reduction**: This establishes that any algorithm achieving pure additive error $n^{0.99}$ can be converted to a $(1+\eta)$ multiplicative plus polylog additive algorithm—useful both for improving upper bounds and for proving lower bounds on pure additive error $o(n)$.

## Highlights & Insights

- **Shifting perspective from "breaking" to "skirting" lower bounds**: Rather than attempting to improve additive error within the lower bound framework, the paper switches to a mixed error model that renders the lower bounds inapplicable. This approach is increasingly prevalent in differential privacy (Aamand et al., Ghazi et al.), and the paper systematically demonstrates its power in the streaming setting.
- **Continual counting as a universal primitive**: All three algorithms reduce complex streaming statistics to collections of continual counting instances, leveraging the $O(\log^{1.5}(T))$ additive error of the Binary Tree Mechanism. This "reduce to counting, then exploit small counter error" design pattern may be applicable to a broader class of private streaming problems.
- **Elegant application of JL reduction in private streams**: The $\Omega(T)$ sensitivity of $F_2$ in the original space is unavoidable, but JL projection reduces the per-coordinate sensitivity to $O(1/\sqrt{m})$, enabling effective use of continual counters—dimensionality reduction simultaneously reduces sensitivity.

## Limitations & Future Work

- **Multiplicative factor not yet constant**: The multiplicative error of MinHash is $O(\log^2 T)$, still far from the practically desirable $(1+\eta)$ constant multiplicative factor. The paper identifies the bottleneck as bucket count ambiguity caused by high-frequency elements, and closing this gap is the main open problem.
- **Event-level privacy only**: Concurrent work (Aryanfard et al.) proves that under item-level privacy, the product of multiplicative and additive errors $\alpha\beta$ must be $\Omega(T^{1/3})$, so the present results do not extend to the stronger item-level setting.
- **Suboptimal space for domain reduction**: Theorem 4.1 uses poly$(T)$ space, which is far worse than the polylog space of MinHash, and the error exponent ($\log^{10}$) is also larger.
- **No empirical evaluation**: As a purely theoretical work, no validation on real-world data streams is provided.

## Related Work & Insights

- **vs. Jain et al. (NeurIPS '23)**: Proved the $\Omega(T^{1/4})$ pure additive lower bound. The core contribution of the present paper is showing that this lower bound can be circumvented via mixed error, rather than improving the pure additive error within its framework.
- **vs. Epasto et al. (2023)**: Achieved $(1+\eta, O(\log^2 T))$ error on insertion-only streams. The present paper extends analogous results to the significantly harder turnstile setting and improves the $F_2$ additive error from $\log^7$ to $\log^4$.
- **vs. Cummings et al. (2025)**: Used multiplicative error to obtain polylog space, but the additive error remains polynomial at $T^{1/3}$. The present paper is the first to achieve both additive and multiplicative error simultaneously polylogarithmic.
- **Concurrent work (Aryanfard et al., Andersson et al., Epasto et al.)**: Independently study mixed error extensions to graph problems, constant-factor improvements, and space lower bounds, respectively—complementary rather than overlapping with the present work.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective shift of "skirting" lower bounds is insightful; the combination of MinHash and JL reduction in private streams is elegant.
- Experimental Thoroughness: ⭐⭐ A purely theoretical work with no experiments, though the theoretical results comprehensively cover both distinct elements and $F_2$.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, result comparison tables are well organized, and the discussion of open problems is substantive.
- Value: ⭐⭐⭐⭐ Advances the theoretical frontier of private streaming algorithms; the mixed error framework offers insights applicable to a broader range of problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD](back_to_square_roots_an_optimal_bound_on_the_matrix_factorization_error_for_mult.md)
- [\[NeurIPS 2025\] Private Continual Counting of Unbounded Streams](../../NeurIPS2025/ai_safety/private_continual_counting_of_unbounded_streams.md)
- [\[CVPR 2026\] LogitDynamics: Reliable ViT Error Detection from Layerwise Logit Trajectories](../../CVPR2026/ai_safety/logitdynamics_vit_error_detection.md)
- [\[ICLR 2026\] Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature](dataless_weight_disentanglement_in_task_arithmetic_via_kronecker-factored_approx.md)
- [\[ICLR 2026\] Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective](adaptive_methods_are_preferable_in_high_privacy_settings_an_sde_perspective.md)

</div>

<!-- RELATED:END -->
