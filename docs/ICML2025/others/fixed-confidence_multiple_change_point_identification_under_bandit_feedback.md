---
title: >-
  [Paper Note] Fixed-Confidence Multiple Change Point Identification under Bandit Feedback
description: >-
  [ICML2025][change point detection] This paper proposes the problem of multiple change point identification in piecewise-constant bandits under the fixed-confidence setting, establishes instance-dependent lower bounds on sample complexity, and designs MCPI (Multiple Change Point Identification), a simple, computationally efficient, and asymptotically optimal algorithm.
tags:
  - "ICML2025"
  - "change point detection"
  - "multi-armed bandits"
  - "pure exploration"
  - "fixed-confidence"
  - "Track-and-Stop"
  - "sample complexity"
date: 2026-05-08
content_hash: 121bd29aede27235
---

# Fixed-Confidence Multiple Change Point Identification under Bandit Feedback

**Conference**: ICML2025  
**arXiv**: [2507.08994](https://arxiv.org/abs/2507.08994)  
**Code**: None  
**Area**: Others  
**Keywords**: change point detection, multi-armed bandits, pure exploration, fixed-confidence, Track-and-Stop, sample complexity

## TL;DR

This paper proposes the problem of multiple change point identification in piecewise-constant bandits under the fixed-confidence setting, establishes instance-dependent lower bounds on sample complexity, and designs MCPI (Multiple Change Point Identification), a simple, computationally efficient, and asymptotically optimal algorithm.

## Background & Motivation

Piecewise-constant functions widely exist in areas such as chemistry, manufacturing, communication systems, materials science, and marine geology. In practice, there is a frequent need to rapidly locate the abrupt change points of these functions with high confidence. For example:

- **Communication Systems**: Determining the critical point where system overload occurs.
- **Materials Development**: Finding the locations where a new material undergoes phase transitions under different experimental conditions.
- **Oceanography**: Detecting the edge of undersea cliffs.

In these scenarios, data collection is highly costly, and there are strict requirements on the reliability of the results. Existing work primarily focuses on: (1) single change point identification under a fixed budget; (2) change point detection along the temporal dimension in non-stationary bandits. This paper is the first to systematically study **multiple change point identification under fixed confidence**, filling a theoretical gap.

## Method

### Problem Formulation

Consider a $K$-armed bandit, where the means $(\mu_i)_{i=1}^K$ are piecewise constant over the action space $\mathcal{A}=[K]$. Let the set of change points be $\underline{x}^* \subset [K-1]$ with $|\underline{x}^*| = N$. In each round, an action $a_t \in [K]$ is selected, and a noisy reward is observed:

$$y_t = \mu_{a_t} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

The jump size at the $j$-th change point is defined as $\Delta_j = |\mu_{x_j^*} - \mu_{x_j^*+1}|$.

### Two Objectives

| Objective | Definition | Applicable Scenarios |
|------|------|----------|
| **Exact-$(N,\delta)$** | The returned $N$ change points are exactly equal to the true set of change points. | Known number of change points |
| **Any-$(N,\delta)$** | The returned $N$ change points are a subset of the true change point set $\underline{x}^*$ (which may contain $m \geq N$ points). | Unknown number of change points |

Both settings require $\mathbb{P}(\text{Correct}) > 1 - \delta$ and a finite stopping time.

### Instance-Dependent Lower Bounds

**Single Change Point Lower Bound** (Corollary 4.3):

$$\mathbb{E}[\tau] \geq \frac{8\sigma^2}{\Delta^2} \log\frac{1}{4\delta}$$

The optimal sampling allocation $\alpha^*$ has an explicit solution: sample only the two adjacent actions on both sides of the change point with equal probability (half each), and do not sample other actions.

**Multiple Change Point Lower Bound** (Theorem 5.2, Exact Setting):

$$\mathbb{E}[\tau] \geq 4\sigma^2 \log\frac{1}{4\delta} \sum_{i=1}^{N} \frac{1}{\Delta_i^2}$$

**Unknown Number of Change Points Lower Bound** (Theorem 5.5, Any Setting):

$$\mathbb{E}[\tau] \geq 8\sigma^2 \log\frac{1}{4\delta} \sum_{i=1}^{N} \frac{1}{\Delta_{(i)}^2}$$

where $\Delta_{(i)}$ denotes the jump size of the $i$-th change point sorted in descending order.

### Key Insights

The optimal strategy should concentrate sampling on the two adjacent actions on both sides of the change points, and the sampling budget allocated around each change point is **inversely proportional to the square of its jump size**:

$$\alpha_j^* = \alpha_{j+1}^* = \frac{1/\Delta_j^2}{2 \sum_{i=1}^N 1/\Delta_i^2}$$

### MCPI Algorithm (Algorithm 2)

MCPI is a variant of the Track-and-Stop framework, consisting of three components:

**1. Sampling Rule**

- **Forced Exploration**: If an action has been sampled fewer than $\sqrt{t}$ times (i.e., $T_i(t) < \sqrt{t}$), prioritize sampling this action.
- **Tracking**: Estimate the $N$ change point positions $\hat{x}_1, \dots, \hat{x}_N$, and sample uniformly on both sides of each estimated change point, choosing the action whose current sampling ratio is furthest below its target allocation.

The change point estimator utilizes the maximum empirical mean difference:

$$\hat{x}_t(S) = \arg\max_{a \in S} |\hat{\mu}_a(t) - \hat{\mu}_{a+1}(t)|$$

**2. Stopping Rule**

Based on the Chernoff stopping time, a test statistic is computed for each estimated change point $\hat{x}_j$:

$$Z_j(t) = \frac{T_{\hat{x}_j}(t) \cdot T_{\hat{x}_j+1}(t)}{2(T_{\hat{x}_j}(t) + T_{\hat{x}_j+1}(t))} \hat{\Delta}_{\hat{x}_j}^2$$

The algorithm stops when the $Z_j(t)$ for all change points exceed the threshold $\beta(t, \delta/N)$.

**3. Recommendation Rule**

Return the change point estimates $\hat{x}_1, \dots, \hat{x}_N$ at the stopping time.

### Asymptotic Optimality (Theorem 5.7)

MCPI is asymptotically optimal under both Exact and Any objectives:

$$\limsup_{\delta \to 0} \frac{\mathbb{E}[\tau_\delta]}{\log(1/\delta)} \leq 8\sigma^2 \sum_{i=1}^N \frac{1}{\Delta_i^2}$$

This matches the lower bound (up to a constant factor of 2) and does not require the total number of change points $N$ to be known in advance.

## Summary of Theoretical Results

| Result | Complexity/Bound | Condition |
|------|-----------|------|
| Single change point lower bound | $\frac{8\sigma^2}{\Delta^2}\log\frac{1}{4\delta}$ | $N=1$, known |
| Multiple change points lower bound (Exact) | $4\sigma^2 \log\frac{1}{4\delta} \sum_i \Delta_i^{-2}$ | Known $N$ |
| Multiple change points lower bound (Any) | $8\sigma^2 \log\frac{1}{4\delta} \sum_i \Delta_{(i)}^{-2}$ | Unknown $N$, $m \geq N$ |
| CPI upper bound (Single change point) | $\frac{8\sigma^2}{\Delta^2}\log\frac{1}{\delta}$ | Asymptotically optimal |
| MCPI upper bound (Multiple change points) | $8\sigma^2 \log\frac{1}{\delta} \sum_i \Delta_i^{-2}$ | Asymptotically optimal, both objectives |
| Cost of known vs unknown $N$ | At most $\times 2$ | Asymptotic sense |

## Highlights & Insights

- **Explicit Optimal Solution**: Unlike the generic Track-and-Stop which requires numerically solving an optimization problem, this work exploits the piecewise-constant structure to **analytically solve** the optimal sampling allocation, bringing dual advantages in computational efficiency and interpretability.
- **Clear Intuition**: The optimal strategy only samples on both sides of the change points, allocating more samples to smaller jumps—this is highly consistent with intuition.
- **Unified Framework**: A single algorithm, MCPI, achieves asymptotic optimality simultaneously under both Exact and Any objectives.
- **Connection to Classical Statistics**: The form of the stopping time is intrinsically connected to the CUSUM/GLR test in offline change point analysis.
- **Superior to Clustering Bandits**: Empirical results show that MCPI significantly outperforms methods that treat the problem as clustering bandits (Yang et al., 2022), as the latter do not exploit the spatial structure of piecewise-constant functions.

## Limitations & Future Work

- **Gaussian Noise Assumption**: The theoretical results rely on the assumption of Gaussian noise with known variance. Extending this to sub-Gaussian or more general distributions remains an open problem.
- **One-Dimensional Action Space**: This work only considers one-dimensional piecewise-constant functions; extending it to high-dimensional change surfaces is not addressed.
- **Synthetic Experiments Only**: Experiments were conducted solely in synthetically generated environments, lacking validation on real-world datasets.
- **Asymptotic Optimality**: The matching of upper and lower bounds is asymptotic ($\delta \to 0$), and finite-sample performance guarantees could be further strengthened.
- **Cost of Forced Exploration**: Although the $\sqrt{t}$ forced exploration guarantees consistency, it introduces extra sampling overhead in finite-time horizons.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic study of fixed-confidence multiple change point identification in bandits.
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Solid matching of upper and lower bounds with elegant analytic solutions.
- Experimental Thoroughness: ⭐⭐⭐ — Only synthetic experiments, lacks real-world datasets.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, well-motivated.
- Overall Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](../../NeurIPS2025/others/fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [\[ICLR 2026\] Change Point Localization and Inference in Dynamic Multilayer Networks](../../ICLR2026/others/change_point_localization_and_inference_in_dynamic_multilayer_networks.md)
- [\[ICML 2025\] Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation](bipartite_ranking_from_multiple_labels_on_loss_versus_label_aggregation.md)
- [\[ICML 2025\] Fixing the Loose Brake: Exponential-Tailed Stopping Time in Best Arm Identification](fixing_the_loose_brake_exponential-tailed_stopping_time_in_best_arm_identificati.md)
- [\[ICML 2025\] Adversarial Combinatorial Semi-bandits with Graph Feedback](adversarial_combinatorial_semi-bandits_with_graph_feedback.md)

</div>

<!-- RELATED:END -->
