---
title: >-
  [Paper Note] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor
description: >-
  [NeurIPS 2025][AI Safety][Hypothesis Selection] This paper presents the first hypothesis selection algorithm under the central differential privacy model that simultaneously achieves nearly-linear time complexity and the optimal approximation factor $\alpha=3$, resolving an open problem posed by Bun et al. (NeurIPS 2019).
tags:
  - NeurIPS 2025
  - AI Safety
  - Hypothesis Selection
  - Differential Privacy
  - Nearly-Linear Time
  - Optimal Approximation Factor
  - Exponential Mechanism
date: 2026-05-08
content_hash: 2119be9f9a0c1596
---

# Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor

**Conference**: NeurIPS 2025
**arXiv**: [2506.01162](https://arxiv.org/abs/2506.01162)
**Code**: None
**Area**: AI Safety / Differential Privacy
**Keywords**: Hypothesis Selection, Differential Privacy, Nearly-Linear Time, Optimal Approximation Factor, Exponential Mechanism

## TL;DR

This paper presents the first hypothesis selection algorithm under the central differential privacy model that simultaneously achieves nearly-linear time complexity and the optimal approximation factor $\alpha=3$, resolving an open problem posed by Bun et al. (NeurIPS 2019).

## Background & Motivation

Hypothesis selection is a fundamental problem in statistical inference: given $n$ candidate distributions and samples drawn from an unknown distribution $P$, the goal is to identify the candidate distribution closest to $P$. This problem has been thoroughly studied in the non-private setting, where $O(\log n)$ samples and $\tilde{O}(n)$ time are known to suffice.

Three key performance metrics are considered: (1) sample complexity, (2) time complexity, and (3) approximation factor $\alpha$ (the multiplicative ratio of the TV distance between the output hypothesis and $P$ relative to the optimal hypothesis). It is known that $\alpha=3$ is the information-theoretically optimal approximation factor.

Under differential privacy constraints, prior work exhibits a fundamental trade-off:
- Bun et al. (2019) tournament algorithm: $\alpha=9$, sample complexity $O(\log n)$, but time $O(n^2 \cdot s)$
- AAAK (2021) minimum distance estimator: $\alpha=3$ (optimal), sample complexity $O(\log n)$ (optimal), but time still $O(n^2 \cdot s)$

Root Cause: **No known algorithm simultaneously achieves nearly-linear time and the optimal approximation factor.** Bun et al. explicitly posed this as an open problem. This paper provides an affirmative answer at the cost of relaxing sample complexity from $O(\log n)$ to $O(\log^3 n)$.

## Method

### Overall Architecture

The algorithm is built upon the minimum distance estimation (MDE) framework. The core idea is to maintain a proxy distance $\hat{W}_A(H_j)$ for each hypothesis $H_j$ (an approximate lower bound on the maximum half-distance), iteratively adding "prompting hypotheses" to progressively refine the proxy distances, and ultimately selecting the hypothesis with the smallest proxy distance as output.

Algorithm pipeline: initialize proxy distances to 0 → sample a list of hypotheses via the exponential mechanism each round → use the Sparse Vector Technique (SVT) to find a prompting hypothesis → update proxy distances → terminate if no prompting hypothesis is found.

### Key Designs

1. **Exponential Mechanism-Based Hypothesis Distribution (Replacing Bucketing)**: Non-private algorithms use "buckets" to manage candidate hypotheses, but bucket membership is highly sensitive to data (changing one sample may alter the bucket assignments of all hypotheses). This paper instead uses the exponential mechanism to maintain a distribution $Q(H_j) \propto \exp(-\epsilon_1 \hat{W}_A(H_j) / (2\Delta(\hat{W}_A)))$, redefining the notion of prompting from "relative to hypotheses in a bucket" to "relative to hypotheses drawn from distribution $Q$." This modification allows the algorithm to operate under the privacy guarantees of the exponential mechanism.

2. **Score Function Design (Reducing Sensitivity)**: To determine whether hypothesis $H_i$ is a prompting hypothesis, one must check whether it can improve the proxy distances of a large number of other hypotheses. Direct counting approaches yield excessively high sensitivity. This paper instead designs a quantile-based score function: $\text{score}_{\eta,\mathcal{K},D}(H_i)$ ranks the lift values of $H_i$ over the list $\mathcal{K}$ and returns the $\lceil\eta/2 \cdot |\mathcal{K}|\rceil$-th largest value. This substantially reduces sensitivity—even when individual lift values shift due to data changes, the variation in the quantile remains bounded.

3. **Sparse Vector Technique (SVT) for Finding Prompting Hypotheses**: In each round, the algorithm must identify a hypothesis with sufficiently high score among $n$ candidates, which is equivalent to $n$ threshold queries. SVT allows the first query exceeding the threshold to be found with privacy cost independent of $n$, substantially conserving the privacy budget.

### Loss & Training

This is a theoretical algorithm paper and does not involve conventional loss functions. The privacy budget is allocated as follows: sampling $k$ hypotheses per round consumes $k \cdot \epsilon_1$, and SVT consumes $\epsilon_2$. Through a careful composition analysis, the total privacy loss across $T$ rounds is exactly $\epsilon$.

A key component is the round complexity analysis: by showing that the normalization term of the exponential mechanism must decrease each round, subject to the constraint that proxy distances are bounded above by 1, it is established that the algorithm terminates in at most $O(\log n)$ rounds.

## Key Experimental Results

### Main Results (Theoretical Comparison)

| Algorithm | Approximation Factor $\alpha$ | Sample Complexity | Time Complexity |
|-----------|-------------------------------|-------------------|-----------------|
| Private Scheffé Tournament (BKSW'19) | 9 | $O(\frac{\log n}{\sigma^2} + \frac{n^2\log n}{\sigma\epsilon})$ | $O(n^2 \cdot s)$ |
| BKSW'19 (Thm 3.5) | >54 | $\tilde{O}(\frac{\log n}{\sigma^2} + \frac{\log n}{\sigma\epsilon})$ | $\tilde{O}(n^2 \cdot s)$ |
| AAAK'21 | **3** | $O(\frac{\log n}{\sigma^2} + \frac{\log n}{\sigma\epsilon})$ | $O(n^2 \cdot s)$ |
| **Ours** | **3** | $O(\frac{\log^3 n}{\sigma^2\epsilon})$ | $\tilde{O}(n \cdot s / \sigma)$ |

### Ablation Study

| Component | Function | Remarks |
|-----------|----------|---------|
| Exponential mechanism replacing bucketing | Reduces privacy sensitivity | Avoids high sensitivity from discrete membership |
| Quantile score function | Reduces sensitivity from $O(1)$ to $O(1/s)$ | Key technical contribution |
| SVT | Privacy cost independent of $n$ | Finds one prompting hypothesis per round |
| Round complexity analysis | Proves $O(\log n)$ rounds | Argued via decrease of exponential mechanism normalization term |

### Key Findings

- The first differentially private hypothesis selection algorithm achieving nearly-linear time and optimal approximation factor simultaneously.
- Sample complexity is $O(\log^3 n / (\sigma^2\epsilon))$, incurring an additional $O(\log^2 n / \sigma)$ factor over the known optimum.
- The layered allocation of the privacy budget (each component providing independent privacy guarantees), while leading to suboptimal sample complexity, renders the analysis tractable.

## Highlights & Insights

- The technique of replacing counting statistics with quantile statistics to reduce sensitivity is highly elegant and may inspire the design of other privacy-preserving algorithms.
- The substitution of the exponential mechanism for bucketing gracefully resolves the high sensitivity issues arising from discrete structures.
- The round complexity proof (via tracking the decay of the exponential mechanism's normalization term) constitutes a novel technical contribution.

## Limitations & Future Work

- Sample complexity is $O(\log^3 n)$ rather than the optimal $O(\log n)$, incurring an additional $O(\log^2 n / \sigma)$ factor.
- The dependence on the confidence parameter $\beta$ is polynomial $1/\beta^2$, rather than the ideal $\log(1/\beta)$.
- The work is purely theoretical, with no empirical validation of practical runtime or usability.
- Two open problems remain unresolved: achieving optimal sample complexity and improving the dependence on $\beta$.

## Related Work & Insights

- **vs. Bun et al. (BKSW'19)**: Reduces time from $O(n^2)$ to $\tilde{O}(n)$ while improving the approximation factor from 9 to 3.
- **vs. Aden-Ali et al. (AAAK'21)**: Preserves $\alpha=3$ while reducing time from $O(n^2)$ to $\tilde{O}(n)$, at the cost of increased sample complexity.
- **vs. Aliakbarpour et al. (ABS'24)**: Extends their non-private nearly-linear time $\alpha=3$ algorithm to the private setting.
- **vs. Local Differential Privacy**: The sample lower bound in the local model is $\Omega(n)$, exponentially higher than in the central model.

## Rating

- Novelty: ⭐⭐⭐⭐ Solid technical contributions (quantile score, exponential mechanism replacing bucketing, round complexity analysis), though the overall framework builds on prior work.
- Experimental Thoroughness: ⭐⭐ Purely theoretical; no empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Rigorous in style typical of theory papers; the overview is clear, though technical details are dense.
- Value: ⭐⭐⭐⭐ Resolves a well-defined open problem, though the scope of application is relatively narrow.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Differentially Private High-dimensional Variable Selection via Integer Programming](differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] Locally Optimal Private Sampling: Beyond the Global Minimax](locally_optimal_private_sampling_beyond_the_global_minimax.md)
- [\[NeurIPS 2025\] Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy](spectral_perturbation_bounds_for_low-rank_approximation_with_applications_to_pri.md)
- [\[NeurIPS 2025\] Preserving Task-Relevant Information Under Linear Concept Removal](preserving_task-relevant_information_under_linear_concept_removal.md)

<!-- RELATED:END -->
