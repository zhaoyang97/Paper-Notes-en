---
title: >-
  [Paper Note] Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems
description: >-
  [ICML2025][Others/Online Algorithms][Online Knapsack Problem] A family of online knapsack algorithms based on simple predictions (point or interval predictions of the critical value) is proposed. These algorithms achieve near-Pareto optimal trade-offs between consistency and robustness, alongside a general reduction from fractional to integer solutions.
tags:
  - "ICML2025"
  - "Others/Online Algorithms"
  - "Online Knapsack Problem"
  - "learning-augmented algorithms"
  - "competitive ratio"
  - "consistency-robustness trade-off"
  - "prediction-augmented"
date: 2026-05-08
content_hash: d33aa6df86799dcd
---

# Near-Optimal Consistency-Robustness Trade-Offs for Learning-Augmented Online Knapsack Problems

**Conference**: ICML2025  
**arXiv**: [2406.18752](https://arxiv.org/abs/2406.18752)  
**Code**: None  
**Area**: Others/Online Algorithms  
**Keywords**: Online Knapsack Problem, learning-augmented algorithms, competitive ratio, consistency-robustness trade-off, prediction-augmented

## TL;DR

A family of online knapsack algorithms based on simple predictions (point or interval predictions of the critical value) is proposed. These algorithms achieve near-Pareto optimal trade-offs between consistency and robustness, alongside a general reduction from fractional to integer solutions.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: The online knapsack problem (OKP) is a classic online resource allocation task: items arrive sequentially with value $v_i$ and weight $w_i$, and the decision maker must immediately and irrevocably decide to accept or reject each item upon arrival. The goal is to maximize the total value under the knapsack capacity constraint. This problem is widely applied in scenarios such as online advertising, dynamic pricing, and supply chain management.

**Limitations of Prior Work**:

### Background

**Background**: The basic OKP does not admit online algorithms with bounded competitive ratios without additional assumptions.

### Key Challenge

**Key Challenge**: Classic methods assume bounded unit values $v_i \in [L, U]$, yielding an optimal competitive ratio of $\ln(U/L) + 1$ (the ZCL algorithm).

### Mechanism

**Mechanism**: Existing learning-augmented methods either rely on **complex frequency predictions** (requiring predictions of the total weight for each possible value) or only achieve optimal trade-offs on **simplified variants**.

### Supplementary Notes

**Supplementary Notes**: For general OKP, the **Pareto-optimal consistency-robustness trade-off** remains an open problem.

**Core Problem**: Can near-Pareto optimal learning-augmented algorithms for the general online knapsack problem be designed using simple and practical predictions?

## Method

### Prediction Model: Critical Value Prediction

The key design of the paper centres on constructing predictions around the **critical value $\hat{v}$**—the minimum unit value among (partially) accepted items in the offline optimal solution.

**Point Prediction**: Directly predicts the exact value of $\hat{v}$.

**Interval Prediction**: Provides lower and upper bounds $\ell \leq \hat{v} \leq u$, where the prediction quality degrades as $u/\ell$ increases.

This simple prediction model is strictly weaker than frequency predictions, but is much easier to obtain in practice.

### Algorithm Framework

#### 1. PP-b: Basic Point-Prediction Algorithm (2-competitive)

The core idea is **reserve-then-greedy**: partitioning the knapsack capacity into two equal parts:

- One half is reserved for high-value items ($v_i > \hat{v}$)
- The other half is reserved for critical-value items ($v_i = \hat{v}$)

This guarantees obtaining at least half of the optimal profit from both parts.

#### 2. PP-a: Advanced Point-Prediction Algorithm ($(1 + \min\{1, \hat{\omega}\})$-competitive)

The key novelty is the **reserve-while-greedy** strategy:

- For items with $v_i > \hat{v}$, accept $x_i = w_i / (1 + \hat{\omega}_{i-1})$
- For items with $v_i = \hat{v}$, dynamically adjust the acceptance fraction: $x_i = \frac{\min\{w_i, 1-\hat{\omega}_{i-1}\}}{1+\hat{\omega}_i} - s_{i-1} \cdot \frac{\min\{w_i, 1-\hat{\omega}_{i-1}\}}{1+\hat{\omega}_i}$
- For items with $v_i < \hat{v}$, reject directly

Here, $\hat{\omega}_i$ is the lower-bound estimate of the total weight of critical-value items maintained by the algorithm. The **prebuying strategy** allows accepting a larger fraction of high-value items when $\hat{\omega}$ is small, breaking the barrier of 2-competitiveness.

#### 3. IPA: Interval Prediction Algorithm ($(2 + \ln(u/\ell))$-competitive)

Processing three value ranges separately:

| Value Range | Strategy |
|---------|------|
| $v_i > u$ | Accept a fraction of $\frac{1}{\alpha+1}$ |
| $v_i \in [\ell, u]$ | Delegated to sub-algorithm $\mathcal{A}$ (e.g., ZCL) with a fraction of $\frac{\alpha}{\alpha+1}$ |
| $v_i < \ell$ | Reject |

Using ZCL as the sub-algorithm, setting $\alpha = \ln(u/\ell) + 1$ yields a total competitive ratio of $2 + \ln(u/\ell)$.

#### 4. MIX: Hybrid Algorithm for Untrusted Predictions

Linearly mixing the trusted prediction algorithm ALG with the worst-case optimal ZCL:

$$x_i = \lambda \hat{x}_i + (1 - \lambda) \tilde{x}_i, \quad \lambda \in (0,1)$$

- **Consistency**: $c/\lambda$ (where $c$ is the inner algorithm's competitive ratio)
- **Robustness**: $\frac{\ln(U/L) + 1}{1 - \lambda}$

#### 5. Fr2Int: Fractional-to-Integer Reduction

The potential unit value space is partitioned into discrete intervals. Using an arbitrary OFKP algorithm as a subroutine, the fractional solution is transformed into an integer solution with a negligible loss under the small-weight assumption, preserving the competitive ratio.

## Theoretical Results

| Setting | Algorithm | Upper Bound (Competitive Ratio) | Lower Bound | Optimal? |
|------|------|---------------|------|---------|
| Trusted Point Prediction | PP-b | 2 | $1 + \min\{1, \hat{\omega}\}$ | Optimal in General |
| Trusted Point Prediction | PP-a | $1 + \min\{1, \hat{\omega}\}$ | $1 + \min\{1, \hat{\omega}\}$ | ✅ Instance-Optimal |
| Trusted Interval Prediction | IPA+ZCL | $2 + \ln(u/\ell)$ | $2 + \ln(u/\ell)$ | ✅ Optimal |
| Untrusted Prediction | MIX | $c/\lambda$-consistent, $\frac{\ln(U/L)+1}{1-\lambda}$-robust | $\Omega(\frac{\ln(U/L)+1}{1-\lambda})$-robust | ✅ Near-Optimal |
| Integer Knapsack | Fr2Int | Fractional Competitive Ratio + Small Loss | — | ✅ General Reduction |

**Key Findings**:

1. Even with perfect critical value predictions, 1-competitiveness is impossible (as the optimal acceptance weight of critical items is unknown online).
2. The prebuying strategy of PP-a achieves an instance-dependent optimal competitive ratio, which fundamentally improves upon the traditional reserve-then-greedy paradigm.
3. A simple linear mixture in MIX suffices to achieve a near-Pareto optimal trade-off, which is uncommon in the learning-augmented literature.
4. Simple critical-value predictions perform comparably to or even better than complex frequency predictions in experiments.

## Highlights & Insights

- **Minimalist Design of Prediction Model**: Only a single scalar (or interval) is required instead of complex frequency distribution predictions, significantly lowering the barrier to obtaining machine learning predictions.
- **Reserve-while-greedy Paradigm**: PP-a's dynamic capacity allocation strategy represents a fundamental improvement over the classic reserve-then-greedy paradigm, achieving instance optimality.
- **Tight Lower and Upper Bounds**: Algorithms under all three prediction settings match (or nearly match) their corresponding lower bounds, exhibiting strong theoretical completeness.
- **General Fractional-to-Integer Reduction**: Fr2Int breaks the limitation that existing integer knapsack algorithms must be designed based on thresholding, demonstrating broader applicability.
- **Robustness Under Adversarial Models**: All results hold under the adaptive adversary model, which is the strongest framework in online algorithm analysis.

## Limitations & Future Work

- OFKP algorithms permit fractional acceptance, while practical applications often require integer decisions; the Fr2Int reduction relies on the **small-weight assumption** $w_i \ll 1$.
- There exists a constant gap between the lower and upper bounds of the consistency-robustness trade-off, which asymptotically matches as $U/L \to \infty$ but has room for improvement in finite cases.
- The paper does not address how to **train ML models** to generate critical value predictions, which serves as a major bottleneck in practical deployment.
- Better algorithms may exist under **stochastic adversary models**, whereas this paper only considers the strongest adaptive adversaries.
- The scale of synthetic and real-world datasets (Bitcoin, Google workload) used in the experiments is relatively limited.

## Related Work & Insights

- **Im et al. (2021)**: Proposed the Sentinel algorithm using frequency predictions, which requires a more complex prediction model and lacks a proof of Pareto optimality.
- **Sun et al. (2021a)**: Achieved optimality using critical value predictions in online search (a special case of OKP), but the method does not generalize to OKP.
- **Balseiro et al. (2023)**: Achieved Pareto optimality on a simplified OKP with unit weights and discrete values.
- **Zhou et al. (2008)**: The ZCL algorithm—the optimal online knapsack algorithm with no predictions, which is adopted as a robust subroutine here.
- **Böckenhauer et al. (2014b)**: Investigated the advice complexity of OKP, which requires strong advice and does not account for untrusted advice.

## Rating

⭐⭐⭐⭐

The paper demonstrates outstanding theoretical depth and completeness, presenting an inspiring design philosophy for simplified prediction models. Highly rigorous with tight upper and lower bounds for all results. However, discussion on how to acquire ML predictions in real-world scenarios is insufficient, and its applicability is somewhat restricted to the online algorithms theory community.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](../../ICML2026/learning_theory/towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[ICML 2025\] Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](../../NeurIPS2025/learning_theory/learning-augmented_online_bipartite_fractional_matching.md)
- [\[ICML 2025\] Near Optimal Best Arm Identification for Clustered Bandits](near_optimal_best_arm_identification_for_clustered_bandits.md)
- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](../../ICML2026/learning_theory/parsimonious_learning-augmented_online_metric_matching.md)

</div>

<!-- RELATED:END -->
