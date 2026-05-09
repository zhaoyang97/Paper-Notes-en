---
title: >-
  [Paper Note] Sample-Adaptivity Tradeoff in On-Demand Sampling
description: >-
  [NeurIPS 2025][Multi-distribution learning] This paper systematically studies the tradeoff between sample complexity and adaptive rounds in on-demand sampling. In the realizable setting, it proves that the optimal sample complexity of $r$-round algorithms is $dk^{\Theta(1/r)}/\varepsilon$. In the agnostic setting, it proposes the LazyHedge algorithm that achieves near-optimal sample complexity in only $\widetilde{O}(\sqrt{k})$ rounds, and introduces the OODS abstract framework to establish nearly tight round complexity lower bounds.
tags:
  - NeurIPS 2025
  - Multi-distribution learning
  - sample complexity
  - round complexity
  - adaptive sampling
  - online optimization
  - Boosting
date: 2026-05-08
content_hash: c5aa661741daf178
---

# Sample-Adaptivity Tradeoff in On-Demand Sampling

**Conference**: NeurIPS 2025
**arXiv**: [2511.15507](https://arxiv.org/abs/2511.15507)
**Authors**: Nika Haghtalab (UC Berkeley), Omar Montasser (Yale), Mingda Qiao (UMass Amherst)
**Code**: Not released
**Area**: Other
**Keywords**: Multi-distribution learning, sample complexity, round complexity, adaptive sampling, online optimization, Boosting

## TL;DR

This paper systematically studies the tradeoff between sample complexity and adaptive rounds in on-demand sampling. In the realizable setting, it proves that the optimal sample complexity of $r$-round algorithms is $dk^{\Theta(1/r)}/\varepsilon$. In the agnostic setting, it proposes the LazyHedge algorithm that achieves near-optimal sample complexity in only $\widetilde{O}(\sqrt{k})$ rounds, and introduces the OODS abstract framework to establish nearly tight round complexity lower bounds.

## Background & Motivation

### State of the Field
Adaptive data collection—dynamically adjusting sampling strategies based on intermediate learning signals—has become an important paradigm in modern machine learning. The Multi-Distribution Learning (MDL) framework allows algorithms to adaptively select sampling from $k$ distributions to minimize worst-case error. However, adaptivity requires sequential data collection, limiting parallelization and scalability. This raises the core question: **To what extent is adaptive sampling necessary? What is the quantitative tradeoff between the number of rounds and sample complexity?**

### Limitations of Prior Work
- **Realizable setting**: It is known that fully adaptive ($O(\log k)$ rounds) achieves optimal sample complexity $\widetilde{O}((d+k)/\varepsilon)$, and completely non-adaptive (1 round) requires $\Omega(dk/\varepsilon)$, but the intermediate regime is entirely unexplored.
- **Agnostic setting**: Optimal sample complexity $\widetilde{O}((d+k)/\varepsilon^2)$ has been established by works such as HJZ (2022) and ZZC+ (2024), but these algorithms require at least $\text{poly}(1/\varepsilon)$ rounds, and some have round complexity as large as their sample complexity (collecting one sample per round).
- Whether algorithms with a constant number of rounds, or rounds depending only on $k$ and independent of $\varepsilon$, can recover optimal sample complexity was previously entirely unknown.

### Root Cause
The paper aims to fill the theoretical gap between fully adaptive and completely non-adaptive regimes, providing a precise characterization of the sample–round tradeoff. It also develops a new algorithmic framework (OODS) to understand the intrinsic difficulty of this tradeoff.

## Method

### Problem Formulation: Multi-Distribution Learning (MDL)
- Given $k$ unknown distributions $D_1, \ldots, D_k$ and a hypothesis class $\mathcal{H}$ (VC dimension $d$).
- Goal: Learn a predictor $\hat{h}$ such that $\max_{i \in [k]} \text{err}(\hat{h}, D_i) \leq \text{OPT} + \varepsilon$.
- An $r$-round algorithm adaptively samples over $r$ rounds, adjusting strategies between rounds based on collected samples.
- Realizable setting: $\text{OPT}=0$ (a perfect hypothesis exists); agnostic setting: $\text{OPT}$ is arbitrary.

### Realizable Setting: Distribution-Level Boosting via AdaBoost (Algorithm 1)

**Core Idea**: Run an AdaBoost variant at the distribution level, leveraging the margin amplification property.

Algorithm:
1. Initialize uniform weights $q_1(j) = 1/k$; set margin parameter $\theta = r/(2\log k)$.
2. Set weak learner error rate $p = \frac{1}{2}(4k^{2/r})^{-1/(1-\theta)}$.
3. For each round $t = 1, \ldots, r$:
    - Draw $m = O(d/(\varepsilon p))$ samples from the mixture distribution $q_t = \sum_j q_t(j) D_j$.
    - Call PAC learner $\mathbb{A}$ to obtain hypothesis $h_t$.
    - Draw a small number of samples from each $D_j$ to estimate $\text{err}(h_t, D_j)$.
    - Update weights via AdaBoost rule using thresholded loss $\mathbb{1}[\text{err}(h_t, D_j) > \tau/2]$.
4. Output majority vote $F(x) = \mathbb{1}[\frac{1}{r}\sum_{t=1}^r h_t(x) \geq 1/2]$.

**Key Property**: After $r$ rounds of boosting, for each distribution $D_j$, at least $1/2 + \theta/2$ fraction of the weak learners have error $\leq \tau$, thereby guaranteeing that the majority vote achieves error $\leq \varepsilon$ on all distributions.

**Sample Complexity** (Theorem 1):

$$O\left(k^{2/r} \log k \cdot \frac{d}{\varepsilon} + \frac{k \log(k) \log(k/\delta)}{\varepsilon}\right)$$

Special cases: $r = \log k$ recovers the optimal $\widetilde{O}((d+k)/\varepsilon)$; $r = 3$ or $4$ already achieves $\widetilde{O}((d\sqrt{k}+k)/\varepsilon)$.

### Realizable Setting Lower Bound (Theorem 2)

$$\Omega\left(\frac{dk^{1/r}}{r\log^2 k}\right)$$

**Proof Construction** (illustrated for $r=2$, giving $\Omega(d\sqrt{k})$):
- Instance space $\mathcal{X} = \mathbb{F}_2^d$, hypothesis class is linear functions, $h^\star$ chosen randomly.
- Construct $k$ distributions, each uniform over a random subspace $V_i$.
- Distribution difficulty is set at 3 levels ($\Theta(d)$, $\Theta(d/\sqrt{k})$, $\Theta(d/k)$) via a random permutation.
- Linear independence of subspaces implies samples from different distributions provide no cross information.
- A 2-round algorithm must "skip" one identification step, incurring an $\Omega(d\sqrt{k})$ cost.

### Agnostic Setting: LazyHedge Algorithm (Algorithm 2)

**Key Insight**: The Hedge algorithm of ZZC+ (2024) requires $T = \Theta(\log k / \varepsilon^2)$ rounds; lazy updates reduce the actual number of sampling rounds.

**Core Innovation — Lazy Upper Bound Updates**:
1. Maintain weights $w^{(t)}$ and an upper bound vector $\bar{w}^{(t)}$ (serving as a proxy for the dataset size of each distribution).
2. At the start of each iteration, check whether $w^{(t)} \in \mathcal{O}(\bar{w}^{(t-1)})$ holds.
3. If **yes**: Run ERM and Hedge updates using existing data without collecting new samples.
4. If **no**: Update the upper bound $\bar{w}_i^{(t)} = C \cdot \max\{w_i^{(1)}, \ldots, w_i^{(t)}\}$ and collect additional samples.

**Two Observable Regions**:
- Box version: $\mathcal{O}(\bar{w}) = \{w \in \Delta^{k-1}: w_i \leq \bar{w}_i, \forall i\}$
    - Rounds: $O(k\log k)$: each coordinate becomes a "culprit" at most $O(\log_C k)$ times.
- Ellipsoid version: $\mathcal{O}(\bar{w}) = \{w \in \Delta^{k-1}: \sum_i w_i^2/\bar{w}_i \leq 1\}$
    - Rounds: $\widetilde{O}(\sqrt{k})$: leveraging $\sum_i \max_t w_i^{(t)} \leq \widetilde{O}(1)$ (Lemma 3), updates are split into Type I (coordinates exceeding $1/\sqrt{k}$) and Type II.

**Final Result** (Proposition 1): $\min\{\widetilde{O}(\sqrt{k}), O(k\log k)\}$ rounds, with sample complexity $\widetilde{O}((d+k)/\varepsilon^2)$.

### OODS Framework: On-Demand Sampling Optimization

**Abstraction**: Models the sample–round tradeoff in MDL as a general convex optimization problem.
- Maximize an unknown concave function $f$ over $\Delta^{k-1}$.
- Each round selects an upper bound $\bar{w}^{(t)}$, within which a first-order oracle may be queried inside $\mathcal{O}(\bar{w}^{(t)})$.
- The cumulative sum $\sum_i \bar{w}_i^{(r)}$ corresponds to the sample cost.

**Upper Bound** (Theorem 4): LazyHedge achieves sample cost $\widetilde{O}(s)$ with $\widetilde{O}(k/s)$ rounds (Box) and $\widetilde{O}(\sqrt{k/s})$ rounds (Ellipsoid).

**Lower Bound** (Theorem 9): Constructs $f(w) = \min_{j \in [m]}\{w_{i_j^\star} + j/m^2\}$ and uses random critical indices to prove:
- When $\varepsilon \leq O(1/k)$: Box $\geq \Omega(\sqrt{k/s})$, Ellipsoid $\geq \Omega((k/s)^{1/4})$.
- When $\varepsilon \leq e^{-\Omega(k)}$: Box $\geq \Omega(k/s)$, Ellipsoid $\geq \Omega(\sqrt{k/s})$.

## Key Experimental Results

This paper is a purely theoretical work with no experimental section. The core contribution is establishing precise complexity characterizations.

### Table 1: Sample–Round Tradeoff in the Realizable Setting

| Rounds $r$ | Sample Complexity Upper Bound (Theorem 1) | Sample Complexity Lower Bound (Theorem 2) | Notes |
|------------|------------------------------------------|------------------------------------------|-------|
| $1$ | $O(dk/\varepsilon)$ | $\Omega(dk/\varepsilon)$ | Non-adaptive; known to be tight |
| $3$ | $\widetilde{O}((d\sqrt{k}+k)/\varepsilon)$ | $\Omega(dk^{1/3}/\varepsilon)$ | 3 rounds already significantly better than 1 |
| $r$ | $O(k^{2/r}\log k \cdot d/\varepsilon)$ | $\Omega(dk^{1/r}/(r\log^2 k))$ | Matching at the $k^{\Theta(1/r)}$ level |
| $\log k$ | $\widetilde{O}((d+k)/\varepsilon)$ | $\Omega((d+k)/\varepsilon)$ | Recovers the known optimum |

### Table 2: Agnostic Setting — Comparison with Prior Work

| Source | Sample Complexity | Round Complexity |
|--------|------------------|-----------------|
| HJZ (2022) | $\widetilde{O}((\log|\mathcal{H}|+k)/\varepsilon^2)$ | $\widetilde{O}((\log|\mathcal{H}|+k)/\varepsilon^2)$ |
| AHZ (2023) | $\widetilde{O}(d/\varepsilon^4 + k/\varepsilon^2)$ | $O(\log(k)/\varepsilon^2)$ |
| ZZC+ (2024) | $\widetilde{O}((d+k)/\varepsilon^2)$ | $O(\log(k)/\varepsilon^2)$ |
| Peng (2024) | $\widetilde{O}((d+k)/\varepsilon^2) \cdot (\log k)^{O(\log(1/\varepsilon))}$ | $(\log k)^{O(\log(1/\varepsilon))}$ |
| **Ours** | $\widetilde{O}((d+k)/\varepsilon^2)$ | $\min\{\widetilde{O}(\sqrt{k}), O(k\log k)\}$ |

This paper is the first to achieve near-optimal sample complexity with round complexity independent of the accuracy parameter $\varepsilon$ and VC dimension $d$.

## Highlights & Insights

- **First characterization of the sample–round tradeoff**: Provides a nearly tight $dk^{\Theta(1/r)}/\varepsilon$ characterization in the realizable setting, revealing that just $r=3$ rounds can reduce sample complexity from $dk$ to $d\sqrt{k}$ order.
- **LazyHedge algorithm**: Through a lazy upper bound update strategy, reduces the round complexity in the agnostic setting from $\text{poly}(1/\varepsilon)$ to $\widetilde{O}(\sqrt{k})$, achieving for the first time round complexity independent of accuracy.
- **OODS abstract framework**: Introduces the on-demand sampling optimization problem, unifying the core structure of existing MDL algorithms and establishing round lower bounds independent of MDL, revealing the intrinsic obstacles to further reducing rounds.
- **Novel application of AdaBoost**: Cleverly leverages AdaBoost's margin amplification property to perform boosting at the distribution level rather than the traditional sample level.
- **Elegance of lower bound constructions**: The realizable lower bound employs random linear functions over finite fields with multi-level difficulty stratification; the OODS lower bound is constructed via concave functions with layered critical indices.

## Limitations & Future Work

- **Gap between upper and lower bounds in the realizable setting**: Upper bound $k^{2/r}$ vs. lower bound $k^{1/r}$—a factor-of-2 gap in the exponent remains unclosed.
- **Lack of MDL-level round lower bounds in the agnostic setting**: The OODS lower bounds do not directly imply MDL lower bounds; Open Question 1 (whether $\text{polylog}(k/\varepsilon)$ rounds suffice) remains unresolved.
- **Purely theoretical work**: No empirical validation of algorithm performance in practical settings such as federated learning or domain adaptation.
- **Binary classification only**: Label space $\mathcal{Y} = \{0,1\}$; the framework is not extended to multiclass classification or regression.
- **Restriction to VC classes**: The analysis relies on VC dimension theory and does not address infinite hypothesis classes or function approximation settings common in deep learning.
- **Gap in the Ellipsoid version of OODS**: Lower bound $\Omega((k/s)^{1/4})$ vs. upper bound $\widetilde{O}(\sqrt{k/s})$—a significant gap remains.

## Related Work & Insights

- **Blum et al. (2017), Chen et al. (2018)**: Optimal $O(\log k)$-round algorithms for realizable MDL; this paper reveals the precise degradation behavior $k^{\Theta(1/r)}$ for $r < \log k$.
- **Haghtalab et al. (2022)**: The first near-optimal sample complexity algorithm for agnostic MDL, but with round complexity $\widetilde{O}((\log|\mathcal{H}|+k)/\varepsilon^2)$, which can be prohibitively large.
- **Zhu et al. (2024)**: Achieves $\widetilde{O}((d+k)/\varepsilon^2)$ sample complexity for agnostic MDL with $O(\log(k)/\varepsilon^2)$ rounds, which still depends on $\varepsilon$.
- **Peng (2024)**: Achieves $(\log k)^{O(\log(1/\varepsilon))}$ rounds but incurs an additional factor of the same order in sample complexity; this paper improves simultaneously on both metrics.
- **Agarwal et al. (2017)**: Pioneering work on adaptivity tradeoffs (e.g., multi-armed bandits); this paper focuses on MDL and provides a finer-grained characterization.
- **Jun et al. (2016), Gao et al. (2019)**: Adaptivity tradeoffs in batched multi-armed bandits; these correspond structurally to the OODS framework introduced in this paper.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic characterization of the sample–round tradeoff in MDL; the OODS framework is an entirely new abstraction.
- Experimental Thoroughness: ⭐⭐ — Purely theoretical work; no experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, progressing from simple to complex; excellent intuitive explanations.
- Value: ⭐⭐⭐⭐⭐ — Provides a solid theoretical foundation for understanding the intrinsic difficulty of adaptive data collection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[NeurIPS 2025\] Structure-Aware Spectral Sparsification via Uniform Edge Sampling](structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)
- [\[ICLR 2026\] FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff](../../ICLR2026/others/fire_frobenius_isometry_reinitialization.md)
- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](../../AAAI2026/others/toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)

</div>

<!-- RELATED:END -->
