---
title: >-
  [Paper Note] Skirting Additive Error Barriers for Private Turnstile Streams
description: >-
  [AI Safety] This paper demonstrates that in the differentially private turnstile streaming model, allowing multiplicative error circumvents known polynomial additive error lower bounds, reducing the additive error for distinct elements and $F_2$ moment estimation from polynomial to $\mathrm{polylog}(T)$.
tags:
  - AI Safety
date: 2026-05-08
content_hash: ceccff1a7a67d5cf
---

# Skirting Additive Error Barriers for Private Turnstile Streams

## Basic Information

- **Title**: Skirting Additive Error Barriers for Private Turnstile Streams
- **Authors**: Anders Aamand (BARC, University of Copenhagen), Justin Y. Chen (MIT), Sandeep Silwal (University of Wisconsin-Madison)
- **Conference**: ICLR 2026
- **arXiv**: [2602.10360](https://arxiv.org/abs/2602.10360)
- **Area**: Differential Privacy, Streaming Algorithms, Continual Release

## TL;DR

This paper demonstrates that in the differentially private turnstile streaming model, allowing multiplicative error circumvents known polynomial additive error lower bounds, reducing the additive error for distinct elements and $F_2$ moment estimation from polynomial to $\mathrm{polylog}(T)$.

## Background & Motivation

### Problem Setting
Differentially private continual release (DP continual release) is a central setting in private data stream processing: data arrives one element at a time, and the algorithm must privately release statistics at each time step. This paper focuses on the **turnstile streaming** model (elements may be inserted or deleted) and studies two classical stream statistics:

1. **Distinct Elements**: the number of elements with nonzero frequency in the stream
2. **$F_2$ Moment Estimation**: the sum of squared element frequencies in the stream

### Known Lower Bounds and Gaps
- Distinct elements: the known lower bound for pure additive error is $\Omega(T^{1/4})$, while the best upper bound is $\tilde{O}(T^{1/3})$, leaving a polynomial gap between the two
- $F_2$ moment: sensitivity analysis yields a pure additive error lower bound of $\Omega(T)$, rendering pure additive error schemes practically unusable

### Core Insight
The authors observe that in the known instances establishing additive error lower bounds, the true statistic itself is much larger than the additive error. This suggests that allowing estimates to incur both **multiplicative and additive error** may circumvent these lower bounds. This approach has precedent in insertion-only streams (Epasto et al., NeurIPS '23), but had not been explored in the turnstile setting.

## Method

### Error Definition
The paper introduces an $(\alpha, \beta)$ mixed error framework: for statistic $Y_t$, the estimate $\hat{Y}_t$ satisfies

$$Y_t / p - r \leq \hat{Y}_t \leq q \cdot Y_t + s$$

where $p \cdot q = \alpha$ (multiplicative factor) and $r + s = \beta$ (additive factor). Intuitively, when the true value $Y_t$ greatly exceeds the noise floor $\beta$, the algorithm produces an estimate that is approximately an $\alpha$-factor approximation.

### Algorithm 1: MinHash Method (Theorem 3.1, Strict Turnstile)

**Core Idea**: Leverages the classical idea of estimating set size via minimum hash values.

1. **Bucketing**: Elements are assigned to buckets by the least significant nonzero bit ($\mathrm{lsb}$) of their hash value; the $k$-th bucket contains elements with $\mathrm{lsb}(h(a)) = k$, and the expected number of elements grows geometrically.
2. **Private Continual Counting**: A DP continual counter (based on the Gaussian binary tree mechanism) is maintained for each bucket, with additive error $\tau = O(\log^{1.5}(T)/\sqrt{\rho})$.
3. **Estimation**: The algorithm finds the largest bucket index $\ell$ whose counter exceeds $\tau$ and outputs $\hat{D} = 2^\ell$.
4. **Median Amplification**: $O(\log T)$ independent instances are run in parallel and the median is taken to boost the success probability.

**Key Error Analysis**:
- Small non-empty buckets are not missed: when $D_t > 6\tau$, Chebyshev's inequality guarantees a sufficiently large non-empty bucket exists.
- Large empty buckets are not selected: an empty bucket's counter value is at most $\tau$ and will not be chosen.
- Source of multiplicative error: the algorithm cannot distinguish whether a high counter value in a bucket arises from $\tau$ low-frequency elements or a single high-frequency element.

**Result**: $\left(O(\log^2(T)/\sqrt{\rho}),\, O(\log^2(T)/\sqrt{\rho})\right)$ error, $O(\log n \cdot \log^2(T))$ space.

### Algorithm 2: Domain Reduction Method (Theorem 4.1, General Turnstile)

**Core Idea**: Maps the high-dimensional domain to a low-dimensional one via a hash function and uses collision detection to estimate distinct elements.

1. **Domain Reduction**: For multiple scales $m = 2^i$, the domain $[n]$ is mapped to $[m]$; a random sign function $g$ is used to handle frequencies.
2. **Anti-concentration (Lemma 4.1)**: When $m \leq \|x\|_0/\ell$, the Erdős–Littlewood–Offord bound guarantees that the reduced coordinates are sufficiently large ($\geq \sqrt{\ell}/1000$).
3. **Sparsity (Lemma 4.2)**: When $m \geq \ell \cdot \|x\|_0$, nearly all reduced coordinates are zero.
4. **Threshold Detection**: Continual counters detect which scales have coordinates exceeding the noise threshold, thereby estimating $\|x\|_0$.

**Result**: $\left(O(\log^{10}(T)/\rho^2),\, O(\log^{10}(T)/\rho^2)\right)$ error, applicable to general turnstile streams.

### Algorithm 3: $F_2$ Moment Estimation (Theorem 5.1)

**Core Idea**: Johnson–Lindenstrauss dimensionality reduction combined with continual counting.

1. **JL Reduction**: An $m \times n$ Rademacher random matrix $A$ maps the frequency vector $x \in \mathbb{R}^n$ to $y = Ax \in \mathbb{R}^m$, preserving $\|y\|_2^2 \approx \|x\|_2^2$.
2. **Continual Counting**: A continual counter is maintained for each reduced coordinate $y_t^i$, with per-step updates of $\pm 1/\sqrt{m}$.
3. **$F_2$ Estimation**: The output $\sum_i (\hat{y}_t^i)^2$ serves as the estimate of $F_2$.
4. **Error Control**: Through careful error decomposition (large vs. small coordinates), the final error is split into a multiplicative component from the JL projection and an additive component from counter noise.

**Result**: $(1+\eta)$ multiplicative $+\, \tilde{O}_\eta(\log^4(T)/(\eta^3\rho))$ additive error, $O(\log^2(T)/\eta^2)$ space.

## Experiments

This is a purely theoretical work and contains no empirical evaluation. The following two tables summarize the theoretical results and comparisons with prior work.

### Table 1: Error Comparison for the Distinct Elements Problem

| Source | Error $(\alpha, \beta)$ | Space | Privacy Model | Notes |
|--------|-------------------------|-------|---------------|-------|
| Jain et al. '23 | $(1,\, \tilde{O}(T^{1/3}))$ | $O(T)$ | Item-level | — |
| Cummings et al. '25 | $(1+\eta,\, \tilde{O}(T^{1/3}))$ | $\tilde{O}_\eta(T^{1/3})$ | Event-level | — |
| Jain et al. '23 | $(1,\, \tilde{\Omega}(T^{1/4}))$ | — | Event-level | Lower bound |
| Epasto et al. '23 | $(1+\eta,\, O_\eta(\log^2(T)))$ | $\mathrm{polylog}(T)$ | Event-level | Insertion-only |
| **Theorem 3.1 (Ours)** | **$(O(\log^2(T)),\, O(\log^2(T)))$** | **$O(\log^3(T))$** | **Event-level** | **Strict turnstile** |
| **Theorem 4.1 (Ours)** | **$(O(\log^{10}(T)),\, O(\log^{10}(T)))$** | **$\mathrm{poly}(T)$** | **Event-level** | **General turnstile** |

### Table 2: Error Comparison for $F_2$ Moment Estimation

| Source | Error $(\alpha, \beta)$ | Space | Privacy Model | Notes |
|--------|-------------------------|-------|---------------|-------|
| Epasto et al. '23 | $(1+\eta,\, \tilde{O}_\eta(\log^7(T)))$ | $O_\eta(\log^2(T))$ | Event-level | Insertion-only |
| Lemma 5.1 (Ours) | $(1,\, \Omega(T))$ | — | Event-level | Lower bound |
| **Theorem 5.1 (Ours)** | **$(1+\eta,\, \tilde{O}_\eta(\log^4(T)))$** | **$O_\eta(\log^2(T))$** | **Event-level** | **General turnstile** |

## Highlights & Insights

1. **Conceptual Breakthrough**: This work is the first to systematically demonstrate that in turnstile streams, permitting multiplicative error can completely circumvent polynomial additive error lower bounds, reducing the additive error to $\mathrm{polylog}(T)$.
2. **Exceptional Space Efficiency**: The MinHash algorithm requires only $\mathrm{polylog}(n, T)$ space, in sharp contrast to prior algorithms requiring polynomial space.
3. **Unified Paradigm**: The core primitive underlying all algorithms is DP continual counting; the target problems are reduced to counting via different streaming transformations.
4. **Complete Theoretical Framework**: The paper introduces a canonical $(\alpha, \beta)$ mixed error definition, providing a unified vocabulary for future research.
5. **Reduction Result (Theorem 4.2)**: An algorithm with pure additive error $o(n)$ can be lifted to one with $(1+\eta)$ multiplicative error and polylogarithmic additive error, pointing a clear direction for future improvements.

## Limitations & Future Work

1. **Event-level Privacy Only**: All upper bound results hold only under event-level DP; concurrent work [Aryanfard et al.] shows that the same results are impossible under item-level DP (the product $\alpha \cdot \beta$ must be at least $T^{1/3}$).
2. **Polylogarithmic Multiplicative Error**: For distinct elements, the multiplicative error is poly-logarithmic rather than constant; whether constant multiplicative error is achievable remains an open problem.
3. **Complementary Limitations of the Two Algorithms**: The MinHash method applies only to strict turnstile streams (non-negative frequencies), while the domain reduction method supports general turnstile streams but requires polynomial space.
4. **Purely Theoretical, No Experiments**: The absence of empirical validation on real data leaves the practical performance and constant factors of the algorithms unknown.
5. **Restricted Adversarial Model**: The algorithms assume a non-adaptive adversary (the stream is independent of the algorithm's randomness); polynomial space is required under an adaptive adversary.

## Related Work & Insights

- **DP Continual Counting**: Dwork et al. 2010 and Chan et al. 2011 introduced the binary tree mechanism; subsequent work has optimized constant factors and composition bounds.
- **Insertion-Only Streams**: Epasto et al. 2023 first achieved $\mathrm{polylog}$ additive error with multiplicative error in the insertion-only model.
- **Turnstile Distinct Elements**: Jain et al. 2023 established the $\Omega(T^{1/4})$ additive lower bound and the $O(T^{1/3})$ upper bound; Henzinger et al. 2024 gave optimal bounds depending on flippancy.
- **Space-Efficient Schemes**: Cummings et al. 2025 achieved $\tilde{O}(T^{1/3})$ space but still with polynomial additive error.
- **Mixed Error Paradigm**: Aamand et al. 2025 and Ghazi et al. 2025 similarly exploit multiplicative error to break additive lower bounds in other DP problems.
- **Concurrent Work**: Aryanfard et al. 2025 (item-level lower bounds), Andersson et al. 2026 (constant factor improvements), Epasto et al. 2026 (space lower bounds).

## Rating

- **Novelty**: 9/10 — First to achieve $\mathrm{polylog}$ additive error in turnstile streams; conceptual contribution is significant.
- **Technical Depth**: 8/10 — Three distinct technical approaches (MinHash + continual counting, domain reduction + anti-concentration, JL + continual counting), each with its own ingenuity.
- **Practicality**: 5/10 — Purely theoretical; polylogarithmic multiplicative error may be too large for practice, and no experiments are provided.
- **Clarity**: 8/10 — Well-structured, with thorough motivation and valuable discussion of open problems.
- **Overall**: 8/10 — Achieves significant conceptual and technical advances in the important area of differentially private streaming algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](../../AAAI2026/ai_safety/problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](../../AAAI2026/ai_safety/learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)

</div>

<!-- RELATED:END -->
