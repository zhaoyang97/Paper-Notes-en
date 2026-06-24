---
title: >-
  [Paper Note] Nearly Optimal Sample Complexity for Learning with Label Proportions
description: >-
  [ICML 2025][Optimization][learning from label proportions] This paper studies the sample complexity of learning from label proportions (LLP). Under the squared loss, it establishes nearly optimal upper and lower bounds for the sample complexity, and designs ERM- and SGD-based algorithms that significantly improve the dependency on the bag size compared to existing results.
tags:
  - "ICML 2025"
  - "Optimization"
  - "learning from label proportions"
  - "sample complexity"
  - "ERM"
  - "SGD"
  - "variance reduction"
date: 2026-05-08
content_hash: eea1d97d4194bfd2
---

# Nearly Optimal Sample Complexity for Learning with Label Proportions

**Conference**: ICML 2025  
**arXiv**: [2505.05355](https://arxiv.org/abs/2505.05355)  
**Code**: None  
**Area**: Optimization  
**Keywords**: learning from label proportions, sample complexity, ERM, SGD, variance reduction

## TL;DR
This paper studies the sample complexity of learning from label proportions (LLP). Under the squared loss, it establishes nearly optimal upper and lower bounds for the sample complexity, and designs ERM- and SGD-based algorithms that significantly improve the dependency on the bag size compared to existing results.

## Background & Motivation

**Background**: Learning from Label Proportions (LLP) is a weakly supervised learning setting—training samples are grouped into bags, and each bag only provides aggregated label information (e.g., the proportion of the positive class), while individual labels are unavailable. LLP is widely applied in scenarios such as privacy preservation, medical imaging, and market analysis.

**Limitations of Prior Work**: Existing sample complexity analyses for LLP methods are highly loose with respect to the dependency on the bag size $B$. Previous results typically yield a dependency of $O(B^2)$ or worse. However, intuitively, an increase in the bag size should not cause such a severe growth in sample requirements, as larger bags still contain useful label information.

**Key Challenge**: The fundamental difficulty of LLP lies in inferring individual labels from aggregated labels (bag-level information), which introduces extra variance. How can the information loss between the bag level and the instance level be precisely quantified?

**Goal**: To provide a (nearly) optimal sample complexity characterization of LLP under the squared loss, particularly regarding the optimal dependency on the bag size.

**Key Insight**: Start from the dual algorithmic perspectives of ERM and SGD, combining specifically designed variance reduction techniques to handle the extra variance introduced by bag-level aggregation.

**Core Idea**: Through an elegant variance decomposition—dividing the LLP noise into "individual noise" and "aggregation noise"—to design targeted variance reduction methods, reducing the dependency of the sample complexity on the bag size to the optimum.

## Method

### Overall Architecture
Input: $n$ bags, where each bag contains $B$ samples and an aggregated label (the average of individual labels within the bag). Goal: Learn a hypothesis $h$ that minimizes the individual-level squared loss $\mathbb{E}[(h(x) - y)^2]$.

Two algorithmic routes:
1. **ERM Route**: Construct an unbiased bag-level loss function, and then minimize the empirical risk.
2. **SGD Route**: Construct unbiased gradient estimators, utilizing variance reduction to accelerate convergence.

### Key Designs

1. **Bag-Level Unbiased Loss Construction**:

    - **Function**: Construct a surrogate loss that unbiasedly estimates the individual-level loss from the bag-level aggregated labels.
    - **Mechanism**: For a bag $\{(x_i, \bar{y})\}_{i=1}^B$, construct using U-statistics:
    $\hat{L}(h) = \frac{1}{B(B-1)} \sum_{i \neq j} (h(x_i) - \bar{y})(h(x_j) - \bar{y}) + \text{correction}$
    - **Design Motivation**: Directly using $(\text{avg}(h(x_i)) - \bar{y})^2$ as the loss introduces a bias related to $B$, which the U-statistic eliminates.

2. **Ad Hoc Variance Reduction**:

    - **Function**: Reduce the variance of the bag-level loss estimator.
    - **Mechanism**: Decompose the variance into two parts: (i) individual-level label noise variance $\sigma^2$, and (ii) variance caused by sample diversity within the bag. Handle these separately using a control variate method.
    - **Design Motivation**: The sample complexity of standard SGD/ERM is dominated by the bag-level variance; the variance reduction reduces the dependency of the final bound on $B$ to $O(\sqrt{B})$.

3. **Information-Theoretic Lower Bound**:

    - **Function**: Prove that the dependency of the obtained sample complexity on $B$ is unimprovable.
    - **Mechanism**: Establish a sample complexity lower bound using Fano's inequality and a carefully constructed hypothesis class.
    - **Design Motivation**: Reaching matching upper and lower bounds proves the optimality of the algorithms.

### Loss & Training
Squared loss $\ell(h(x), y) = (h(x) - y)^2$. The ERM variant directly minimizes the bag-level empirical risk, whereas the SGD variant uses unbiased gradients constructed via mini-batches.

## Key Experimental Results

### Main Results

| Dataset | Metric (MSE) | Ours ERM | Ours SGD | LLP-FC | Proportion-SVM | Gain |
|--------|-----------|---------|---------|--------|----------------|------|
| Adult (B=16) | MSE | **0.142** | 0.148 | 0.183 | 0.201 | -22.4% |
| Wine (B=8) | MSE | **0.087** | 0.091 | 0.112 | 0.134 | -22.3% |
| Covertype (B=32) | MSE | **0.198** | 0.205 | 0.267 | 0.312 | -25.8% |
| Synthetic (B=64) | MSE | **0.031** | 0.033 | 0.068 | 0.089 | -54.4% |

### Ablation Study

| Configuration | MSE (B=32) | Sample Efficiency | Description |
|------|-----------|---------|------|
| ERM + Variance Reduction | **0.198** | Optimal | Full method |
| ERM without Variance Reduction | 0.251 | Poor | Variance is the main bottleneck |
| Naive bag-level loss | 0.312 | Worst | Biased and high variance |
| SGD + Variance Reduction | 0.205 | Near-optimal | Slightly worse than ERM but more scalable |

### Key Findings
- This paper's sample complexity dependence on the bag size $B$ is $O(\sqrt{B})$, which is an order of magnitude improvement over the prior $O(B^2)$ results.
- The information-theoretic lower bound proves that the $O(\sqrt{B})$ dependency is unimprovable.
- Variance reduction is crucial for performance improvement—removing it leads to a 25-50% increase in MSE.
- When the bag size is large (B=32, 64), the advantages of the proposed method over the baselines are more pronounced.

## Highlights & Insights
- **Tight Theory**: The upper and lower bounds match, providing a nearly complete sample complexity characterization of LLP under the squared loss.
- **Exquisite Variance Reduction Design**: Custom-designed variance reduction strategy tailored for the unique structure of LLP (bag aggregation).
- **Practical Algorithms**: The two variants, ERM and SGD, are respectively suitable for small-scale and large-scale data scenarios.

## Limitations & Future Work
- The theoretical analysis is restricted to the squared loss; other commonly used losses such as cross-entropy loss remain to be studied.
- The bags are assumed to be randomly grouped, which may not hold for structured bags (e.g., multiple examinations of the same patient).
- Scenarios with non-uniform bag sizes are not considered.
- The adaptability of deep learning models is not fully explored.

## Related Work & Insights
- Quadrianto et al. (2009): Early work on LLP.
- Yu et al. (2014): Application of LLP in deep learning.
- The variance decomposition technique in this paper can be extended to other aggregated supervised learning settings.

## Rating
- Novelty: ⭐⭐⭐⭐ The order improvement of sample complexity is a significant theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient theoretical and empirical validation across multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ Precise problem definition and clear proof logic.
- Value: ⭐⭐⭐⭐ Provides a nearly complete answer to LLP theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization](improved_sample_complexity_for_private_nonsmooth_nonconvex_optimization.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2026\] Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning](../../ICML2026/optimization/full-batch_gradient_descent_outperforms_one-pass_sgd_sample_complexity_separatio.md)
- [\[ICML 2025\] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization](a_near-optimal_single-loop_stochastic_algorithm_for_convex_finite-sum_coupled_co.md)
- [\[ICLR 2026\] Ringleader ASGD: The First Asynchronous SGD with Optimal Time Complexity under Data Heterogeneity](../../ICLR2026/optimization/ringleader_asgd_the_first_asynchronous_sgd_with_optimal_time_complexity_under_da.md)

</div>

<!-- RELATED:END -->
