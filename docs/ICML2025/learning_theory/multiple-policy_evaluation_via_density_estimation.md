---
title: >-
  [Paper Note] Multiple-Policy Evaluation via Density Estimation
description: >-
  [ICML 2025][multiple-policy evaluation] This paper proposes the CAESAR algorithm, which simultaneously evaluates $K$ policies through a two-phase approach (coarse estimation of the visitation distribution + density ratio estimation under the optimal sampling distribution). This achieves a non-asymptotic, instance-dependent sample complexity. The core technique is "coarse estimation"—obtaining a constant-factor distribution approximation using only $O(1/\epsilon)$ samples.
tags:
  - "ICML 2025"
  - "multiple-policy evaluation"
  - "density estimation"
  - "importance sampling"
  - "coarse estimation"
  - "sample complexity"
date: 2026-05-08
content_hash: dc15ce0fd742e19f
---

# Multiple-Policy Evaluation via Density Estimation

**Conference**: ICML 2025  
**arXiv**: [2404.00195](https://arxiv.org/abs/2404.00195)  
**Code**: None  
**Area**: Others  
**Keywords**: multiple-policy evaluation, density estimation, importance sampling, coarse estimation, sample complexity

## TL;DR

This paper proposes the CAESAR algorithm, which simultaneously evaluates $K$ policies through a two-phase approach (coarse estimation of the visitation distribution + density ratio estimation under the optimal sampling distribution). This achieves a non-asymptotic, instance-dependent sample complexity. The core technique is "coarse estimation"—obtaining a constant-factor distribution approximation using only $O(1/\epsilon)$ samples.

## Background & Motivation

**Background**: Policy evaluation is a fundamental problem in reinforcement learning, serving as a core step in methods such as policy iteration and policy gradient. In practical applications, it is often necessary to evaluate $K$ policies trained with different hyperparameters simultaneously to select the optimal policy. While offline evaluation of a single policy has been extensively studied, with methods including importance sampling, model-based methods, and doubly robust estimators.

**Limitations of Prior Work**: The most naive approach is to run single-policy evaluation $K$ times independently, but this leads to a sample complexity that scales linearly with $K$, completely ignoring the similarity between policies. Dann et al. (2023) proposed an online method using trajectory synthesis to reuse samples from overlapping policy regions. However, this requires building generative models as an intermediate step, which may be impractical.

**Key Challenge**: The main challenge of multiple-policy evaluation lies in designing a shared data collection policy such that a single dataset can efficiently support the evaluation of all $K$ policies. This requires the sampling distribution to cover the visitation distributions of all target policies simultaneously, rather than sampling for each policy individually.

**Goal**: Design an optimal data collection (behavior) distribution and utilize offline evaluation techniques to simultaneously estimate the values of all $K$ policies from the shared data.

**Key Insight**: The authors observe that obtaining a constant-factor approximation (i.e., "coarse estimation") of the target policies' visitation distributions is sufficient to compute a near-optimal sampling distribution. Crucially, coarse estimation requires only $O(1/\epsilon)$ samples—significantly lower than the $O(1/\epsilon^2)$ required for precise estimation.

**Core Idea**: Use low-cost coarse estimation to guide optimal sampling, and then perform precise evaluation under the optimal sampling distribution using a DualDICE-style density ratio estimator.

## Method

### Overall Architecture

The proposed CAESAR (Coarse and Adaptive EStimation with Approximate Reweighing) framework consists of two phases. The inputs are $K$ target policies $\pi_1, \ldots, \pi_K$ and a tabular MDP environment. Phase I uses a small number of samples to obtain a coarse estimation of the visitation distribution. Phase II constructs an optimal sampling distribution based on this coarse estimation, collects a shared dataset from it, and utilizes the IDES algorithm to estimate the density ratios and expected returns for all policies.

### Key Designs

1. **Coarse Estimation**:

    - **Function**: Approximates the visitation distribution $d_h^{\pi_k}(s,a)$ of each policy with constant-factor accuracy.
    - **Mechanism**: Collects $O(1/\epsilon)$ trajectories for each target policy and utilizes simple concentration inequalities to obtain a distribution estimate with multiplicative accuracy. Specifically, the coarse estimate $\hat{d}$ satisfies $c_1 d(s,a) \leq \hat{d}(s,a) \leq c_2 d(s,a)$ for constants $c_1, c_2$. The key advantage is that multiplicative accuracy requires a linear (rather than quadratic) sample complexity.
    - **Design Motivation**: The sample cost of Phase I is a lower-order term that can be considered "free" preprocessing. The flexibility of coarse estimation lies in the fact that the constant multipliers can be arbitrary, making the formulation more concise and elegant.

2. **Optimal Sampling Distribution Construction**:

    - **Function**: Computes a mixture behavior distribution $\mu^*$ based on the coarse estimation to minimize the maximum density ratio across all $K$ policies.
    - **Mechanism**: The optimal sampling problem is formulated as $\min_\mu \max_{k \in [K]} \sum_{s,a} (d_h^{\pi_k}(s,a))^2 / \mu_h(s,a)$. Due to the multiplicative accuracy of the coarse estimation, replacing $d$ with $\hat{d}$ yields an approximately optimal solution whose gap to the true optimum is bounded by a constant factor.
    - **Design Motivation**: The optimal sampling distribution effectively leverages the visitation overlap between policies. When multiple policies visit certain state-action pairs with high frequency, the sampling distribution naturally concentrates on these regions.

3. **IDES (Importance Density EStimation)**:

    - **Function**: Estimates the marginal importance density ratio $w_h^{\pi_k}(s,a) = d_h^{\pi_k}(s,a) / \mu_h(s,a)$ of each policy using data from the optimal sampling distribution.
    - **Mechanism**: Inspired by DualDICE, a step-wise quadratic loss function is designed. For each timestep $h$, the density ratio is estimated by minimizing $L_h(w) = \mathbb{E}_\mu[(w_h(s,a) - T_h w_{h+1}(s,a))^2]$, where $T_h$ is an operator related to the transition kernel. This is solved using stochastic gradient descent (SGD), combined with a Median of Means estimator to convert expectation results into high-probability guarantees.
    - **Design Motivation**: Directly estimating the marginal density ratios avoids the exponential variance (with respect to the horizon $H$) inherent in standard importance sampling. Compared to the original version of DualDICE (designed for infinite horizons), IDES non-trivially extends the method to finite-horizon MDPs.

### Loss & Training

The core loss of IDES is a step-wise quadratic objective: $L_h(w) = \mathbb{E}_\mu[(w_h(s,a) - r_h(s,a) \cdot w_h(s,a) - \gamma \sum_{s'} P_h(s'|s,a) \sum_{a'} \pi(a'|s') w_{h+1}(s',a'))^2]$. This loss is optimized independently at each timestep $h$, where strong convexity and smoothness guarantee the convergence rate of SGD.

## Key Experimental Results

### Sample Complexity Theoretical Comparison

| Method | Sample Complexity | Leverages Policy Similarity | Non-asymptotic | Known Upper Bound |
|------|----------|-------------|--------|---------|
| Naive $K$-times Evaluation | $O(K H^3 / \epsilon^2)$ | No | Yes | Yes |
| Dann et al. (2023) | Depends on trajectory synthesis | Yes (Online) | Yes | No |
| Yin & Wang (2020) | $O(H^3/\epsilon^2 \cdot \text{coverage term})$ | No | No (Asymptotic) | No |
| **CAESAR** | $O(H^4/\epsilon^2 \cdot \sum_h \max_k \sum_{s,a} d^2/\mu^*)$ | **Yes (Offline)** | **Yes** | **Yes** |

### Ablation Study

| Component | Contribution to Sample Complexity | Description |
|------|-------------|------|
| Phase I Coarse Estimation | $O(1/\epsilon)$ (Lower-order) | Linear instead of quadratic, negligible |
| Optimal Distribution Computation | Based on coarse estimation, no extra sampling | Solved via convex optimization |
| Phase II Precise Evaluation | $O(H^4/\epsilon^2 \cdot \text{instance term})$ (Main term) | Accounts for the vast majority of total cost |
| MARCH (Deterministic Policies) | Polynomial in $S, A$ | Despite exponential number of deterministic policies |

### Key Findings

1. The cost of coarse estimation in Phase I $O(1/\epsilon)$ is a lower-order term compared to precise estimation $O(1/\epsilon^2)$, making the upfront cost of the two-stage approach negligible.
2. The optimal sampling distribution effectively exploits policy overlap. When the $K$ policies are highly similar, the sample complexity approaches that of single-policy evaluation.
3. The known upper bound advantage of CAESAR implies that the required sample size can be estimated before execution, whereas results from Yin & Wang can only be verified ex-post.
4. The $\beta$-distance can be applied to the MARCH algorithm, achieving polynomial-complexity coarse estimation even for an exponential number of deterministic policies.

## Highlights & Insights

1. **The concept of coarse estimation is simple yet powerful**: Obtaining multiplicative-accuracy approximations requires only linear sample complexity. This technique is highly general and can be reused in other two-phase estimation problems.
2. **The separation of exploration and exploitation is elegantly designed**: The roles of Phase I (exploration for coarse information) and Phase II (utilization for precise evaluation) are clearly divided.
3. **Independent contribution of IDES**: As a rigorous extension of DualDICE to finite-horizon MDPs, IDES holds independent value and can be used outside multiple-policy evaluation.
4. **Practical significance of known upper bounds**: Not relying on the unknown true distribution for the upper bound allows the algorithm to confidently predict sample requirements.

## Limitations & Future Work

- The method is only applicable to tabular MDPs; extending it to function approximation scenarios remains an important open challenge.
- The $H^4$ dependence on the horizon is still relatively large, and whether a gap exists with the lower bound remains unknown.
- This is a theoretically oriented work that lacks large-scale empirical evaluations to verify practical performance.
- Computing the optimal sampling distribution requires knowing the MDP transitions, limiting its direct applicability to unknown environments.

## Related Work & Insights

- The marginal density ratio estimation idea from DualDICE (Nachum et al., 2019) is extended to finite-horizon multi-policy scenarios.
- It shares connections with the exploration objectives of Amortila et al. (2024), but coarse estimation simplifies the optimization process.
- Coarse estimation has broad applicability. Any two-phase problem requiring coarse exploration followed by precise exploitation can benefit from this approach.
- Zhang & Zanette (2023) and Li et al. (2023) also utilized multiplicative-accuracy estimation, but this study provides a cleaner and more general theoretical framework.

## Rating

⭐⭐⭐⭐ Solid theoretical work. Both the coarse estimation technique and the IDES algorithm offer independent value. Limitations include the tabular MDP assumption and the lack of empirical results. Overall, it represents a significant advancement in the theory of offline multiple-policy evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Score-Based Density Estimation from Pairwise Comparisons](../../ICLR2026/learning_theory/score-based_density_estimation_from_pairwise_comparisons.md)
- [\[ICLR 2026\] Minimax-Optimal Aggregation for Density Ratio Estimation](../../ICLR2026/learning_theory/minimax-optimal_aggregation_for_density_ratio_estimation.md)
- [\[ICML 2025\] Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)
- [\[ICLR 2026\] A Minimum Variance Path Principle for Accurate and Stable Score-Based Density Ratio Estimation](../../ICLR2026/learning_theory/a_minimum_variance_path_principle_for_accurate_and_stable_score-based_density_ra.md)
- [\[ICML 2025\] On Fine-Grained Distinct Element Estimation](on_fine-grained_distinct_element_estimation.md)

</div>

<!-- RELATED:END -->
