---
title: >-
  [Paper Note] Improved Differentially Private Algorithms for Rank Aggregation
description: >-
  [AAAI 2026][differential privacy] This paper presents improved approximation algorithms for rank aggregation under differential privacy. It introduces the first study of differentially private footrule rank aggregation w…
tags:
  - "AAAI 2026"
  - "differential privacy"
  - "rank aggregation"
  - "Kemeny ranking"
  - "footrule distance"
  - "approximation algorithms"
date: 2026-05-08
content_hash: 5be801944806212a
---

# Improved Differentially Private Algorithms for Rank Aggregation

**Conference**: AAAI 2026
**arXiv**: [2511.11319](https://arxiv.org/abs/2511.11319)  
**Code**: None  
**Area**: Other
**Keywords**: differential privacy, rank aggregation, Kemeny ranking, footrule distance, approximation algorithms

## TL;DR

This paper presents improved approximation algorithms for rank aggregation under differential privacy. It introduces the first study of differentially private footrule rank aggregation with a near-optimal algorithm (which also yields a 2-approximation for the Kemeny problem), and improves the additive error of the Kemeny PTAS by combining two-way marginal queries with an unbiasedness technique (reducing the exponent of $m$ from 3 to 65/22).

## Background & Motivation

Rank aggregation is a fundamental problem of combining multiple users' rankings over candidates into an optimal consensus ranking, with broad applications in electoral voting, search result ranking, and related settings. Two classical optimality criteria exist: the Kemeny ranking based on Kendall tau distance (NP-hard) and the footrule ranking based on positional displacement (solvable in polynomial time).

Since input rankings typically contain sensitive user information (e.g., voting preferences), privacy-preserving rank aggregation is of critical importance. Differential privacy (DP), as the gold standard for privacy protection, provides rigorous mathematical guarantees. Alabi et al. (AAAI'22) proposed a PTAS and a 5-approximation algorithm for Kemeny rank aggregation under DP, but two core tensions remain:

- The PTAS achieves a multiplicative factor of $\alpha = 1+\xi$ but incurs a large additive error $\beta = \tilde{O}(m^3/\varepsilon n)$
- The 5-approximation has a smaller additive error $\beta = \tilde{O}(m^{2.5}/\varepsilon n)$ but a large multiplicative factor $\alpha = 5+\xi$

Moreover, footrule rank aggregation under DP has never been studied. The core idea of this paper is to first solve the footrule problem (leveraging an improved binary tree mechanism), yielding both a near-optimal DP algorithm for footrule and, via the known relationship between footrule and Kendall tau distance, a 2-approximation for the Kemeny problem; the PTAS is further improved using a bucketing plus two-way marginal query technique.

## Method

### Overall Architecture

The paper proposes two main algorithms: (1) a footrule rank aggregation algorithm based on approximate median (ApxMed), which can be adapted into a $(2, \beta)$-approximation for the Kemeny problem; and (2) an improved PTAS for Kemeny rank aggregation, handled separately for the "large $n$" and "small $n$" regimes.

### Key Designs

1. **Modified Binary Tree Mechanism**:

    - Function: Efficiently answers approximate median (ApxMed) queries under differential privacy, i.e., for each candidate position $j$, estimates $\gamma_j(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n |x_i - j|$
    - Mechanism: Extends the classical binary tree mechanism. The original mechanism answers range frequency queries; this paper extends each node $t$ to store two aggregated values: $v_t^{agg} = \frac{1}{n}\sum_{x_i \in I(t)}|x_i - r(t)|$ (sum of positional deviations) and $u_t^{agg} = \frac{1}{n}\sum_{x_i \in I(t)} 2^{\ell(t)}$ (scaled count). The estimate of $\gamma_j$ is reconstructed using sibling node information.
    - Design Motivation: In the naive approach, each data point $\pi_i(q)$ affects up to $m$ output values, causing excessive fragmentation of the privacy budget and large noise. The binary tree structure ensures each data point appears in only $O(\log m)$ intervals, substantially reducing noise. A weight factor $\kappa^{d-\ell(t)}$ is further introduced to reduce logarithmic error factors.
    - Accuracy Guarantee: Under $\varepsilon$-DP, $\beta = O(m\log m / \varepsilon n)$; under $(\varepsilon,\delta)$-DP, $\beta = O(m\sqrt{\log m \log(1/\delta)} / \varepsilon n)$

2. **From ApxMed to Footrule Rank Aggregation (Algorithm 2)**:

    - Function: Solves footrule rank aggregation using parallel ApxMed.
    - Mechanism: Models footrule rank aggregation as minimum-weight bipartite matching. For each candidate $q$ and position $j$, the cost $\gamma_q^j = \frac{1}{n}\sum_{i=1}^n |\pi_i(q) - j|$ is assigned. All weights are privately estimated using $m$-parallel ApxMed, and the minimum-weight bipartite matching is then solved.
    - Design Motivation: Since the Spearman footrule distance is between 1 and 2 times the Kendall tau distance, a $(1,\beta)$-approximation for footrule directly yields a $(2,\beta)$-approximation for Kemeny, reducing the multiplicative factor from 5 to 2.

3. **Improved Kemeny PTAS — Large $n$ Case (Leveraging Unbiasedness)**:

    - Function: When $n = \Omega_{\varepsilon,\delta}(m\log m)$, perturbs the weight matrix with unbiased Gaussian noise.
    - Mechanism: Independent Gaussian noise $\mathcal{N}(0, \sigma^2)$ is added to the weight matrix $\mathbf{w}^\Pi$ to obtain the privatized matrix $\tilde{\mathbf{w}}$. A key observation is that for entries where $w_{uv} < w_{vu}$, one can replace $w_{uv}$ with 0 and $w_{vu}$ with $w_{vu} - w_{uv}$, adding noise only to the latter, thereby avoiding truncation bias while ensuring non-negativity with high probability.
    - Design Motivation: Direct noise addition may produce negative values, which the non-private PTAS cannot handle; truncation-based fixes introduce $O(m^3)$ bias. The proposed unbiasedness transformation ensures non-negativity with $\sigma = O(1/\log m)$.

4. **Improved Kemeny PTAS — Small $n$ Case (Reduction to 2-Way Marginals, Algorithm 3)**:

    - Function: For small $n$, encodes Kemeny ranking via bucketing as two-way marginal queries.
    - Mechanism: The ranking domain $[m]$ is partitioned into $B$ buckets. Within-bucket comparisons are handled using standard Laplace/Gaussian mechanisms (with reduced sensitivity); cross-bucket comparisons are encoded as two-way marginal queries $t_{uv} = \frac{1}{n}\sum_{i} \mathbf{1}[\iota(\pi_i(u)) < \iota(\pi_i(v))]$, exploiting the superior accuracy guarantees of DP two-way marginal algorithms.
    - Design Motivation: The error of DP two-way marginal algorithms, $O_{\varepsilon,\delta}(m^{1/4}/\sqrt{n})$, significantly outperforms standard noise addition for small $n$. By optimizing the number of buckets $B$, the additive error of the PTAS under $(\varepsilon,\delta)$-DP is reduced from $m^3$ to $m^{65/22} \approx m^{2.955}$.

### Summary of Theoretical Results

The core theoretical contributions of this paper are summarized as follows:

- **Footrule Rank Aggregation** (Theorem 4.1): $(1, \beta)$-approximation; under $\varepsilon$-DP, $\beta = O(m^3 \log m / \varepsilon n)$; under $(\varepsilon,\delta)$-DP, $\beta = O(m^{2.5}\sqrt{\log m \log(1/\delta)} / \varepsilon n)$; both are near-optimal.
- **Kemeny 2-Approximation** (Theorem 4.2): $(2, \beta)$-approximation with the same additive error as footrule.
- **Kemeny PTAS** (Theorem 5.1): $(1+\xi, \beta)$-approximation; under $(\varepsilon,\delta)$-DP, $\beta = \tilde{O}_\xi(m^{65/22}\sqrt{\log(1/\delta)} / \varepsilon n)$.

## Key Experimental Results

### Main Results (Theoretical Bound Comparison)

This paper is purely theoretical and contains no experimental data. Core results are summarized in theorem form below:

| Privacy Model | Approximation Ratio $\alpha$ | Additive Error $\beta$ (exponent of $m$) | Source |
|---|---|---|---|
| $(\varepsilon,\delta)$-DP | $1+\xi$ | $m^3$ | Alabi et al. |
| $(\varepsilon,\delta)$-DP | $5+\xi$ | $m^{2.5}$ | Alabi et al. |
| $(\varepsilon,\delta)$-DP | $1+\xi$ | $m^{65/22} \approx m^{2.955}$ | **Ours** |
| $(\varepsilon,\delta)$-DP | $2$ | $m^{2.5}$ | **Ours** |
| $(\varepsilon,\delta)$-DP | any $\alpha \geq 1$ | $m^{2.5}$ (lower bound) | Alabi et al. |

### Footrule Results Under Different DP Models

| DP Model | Approx. Ratio | Additive Error | Optimality |
|---|---|---|---|
| $\varepsilon$-DP | 1 | $O(m^3 \log m / \varepsilon n)$ | Near-optimal |
| $(\varepsilon,\delta)$-DP | 1 | $O(m^{2.5}\sqrt{\log m \log(1/\delta)} / \varepsilon n)$ | Near-optimal |
| $\varepsilon$-LDP | 1 | $O(m^{2.5}\sqrt{\log m} / \varepsilon\sqrt{n})$ | — |

### Key Findings

- The additive errors for footrule rank aggregation under $\varepsilon$-DP and $(\varepsilon,\delta)$-DP are near-optimal (matching the lower bounds up to logarithmic factors).
- The 2-approximation algorithm is non-interactive under the LDP model (whereas the algorithm of Alabi et al. requires $O(\log m)$ rounds of communication).
- The exponent of $m$ in the PTAS additive error is reduced from 3 to 65/22, representing a meaningful step toward the lower bound of 2.5.

## Highlights & Insights

- This paper is the first to study footrule rank aggregation under differential privacy and cleverly leverages the footrule–Kendall tau distance relationship to improve the multiplicative approximation ratio for the Kemeny problem.
- The modified binary tree mechanism is elegantly designed: by storing aggregated positional deviation values (rather than only frequencies), the classical range query mechanism is extended to median estimation, which has independent research value.
- The bucketing-plus-two-way-marginal reduction for the small $n$ case is the most technically sophisticated contribution of the paper, cleverly exploiting the extensive literature on DP two-way marginal queries.
- The unbiasedness technique—replacing $w_{uv}$ with $w_{vu} - w_{uv}$—provides a concise and effective solution to the problem of Gaussian noise introducing negative values.

## Limitations & Future Work

- A gap remains between the PTAS additive error exponent of 65/22 ≈ 2.955 and the lower bound of 2.5, especially under $\varepsilon$-DP where the gap is larger (27/7 ≈ 3.857 vs. lower bound 3).
- As a purely theoretical work, the paper lacks empirical validation on real-world datasets (e.g., actual voting data, recommender system rankings).
- The improved PTAS requires separate handling for large $n$ and small $n$ regimes, making implementation considerably complex.
- No PTAS result is provided under the LDP model; only a 2-approximation is given.

## Related Work & Insights

- Alabi et al. (AAAI'22) established the theoretical foundation for DP rank aggregation; this paper provides systematic improvements upon that work.
- The binary tree mechanism (Dwork et al., Chan et al.) is a classical tool for handling continuous queries in DP; this paper demonstrates how to extend it to new problem settings.
- DP algorithms for two-way marginal queries (Nikolov, Dwork et al.) exhibit strong performance at small sample sizes, and this paper cleverly applies them to rank aggregation.
- Insight: Algorithmic improvements in DP problems often arise from deep structural analysis (e.g., bucketing, unbiasedness transformations) rather than straightforward application of generic DP mechanisms.

## Rating

- Novelty: ⭐⭐⭐⭐ (Solid theoretical contributions, though incremental in nature)
- Experimental Thoroughness: ⭐⭐ (Purely theoretical; no empirical validation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous proofs and clear structure)
- Value: ⭐⭐⭐⭐ (Narrows the gap between upper and lower bounds; first study of footrule under DP)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Missing Mass for Differentially Private Domain Discovery](../../ICLR2026/others/missing_mass_for_differentially_private_domain_discovery.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/others/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[AAAI 2026\] Private Frequency Estimation via Residue Number Systems](private_frequency_estimation_via_residue_number_systems.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)

</div>

<!-- RELATED:END -->
