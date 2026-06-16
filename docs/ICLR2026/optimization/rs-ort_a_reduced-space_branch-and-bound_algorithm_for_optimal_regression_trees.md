---
title: >-
  [Paper Note] RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees
description: >-
  [ICLR 2026][Optimization][optimal decision trees] This paper proposes RS-ORT, an algorithm that reformulates regression tree training as a two-stage optimization problem and applies branch-and-bound over a reduced space…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "optimal decision trees"
  - "regression trees"
  - "branch-and-bound"
  - "mixed-integer programming"
  - "interpretable machine learning"
date: 2026-05-08
content_hash: 985edf8d1379bcc6
---

# RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees

**Conference**: ICLR 2026
**arXiv**: [2510.23901](https://arxiv.org/abs/2510.23901)  
**Area**: Optimization
**Keywords**: optimal decision trees, regression trees, branch-and-bound, mixed-integer programming, interpretable machine learning

## TL;DR

This paper proposes RS-ORT, an algorithm that reformulates regression tree training as a two-stage optimization problem and applies branch-and-bound over a reduced space (branching only on tree structure variables). Combined with closed-form leaf predictions, threshold discretization, and exact last-layer subtree resolution, RS-ORT is the first exact method to achieve globally optimal regression tree learning on datasets with up to 2 million samples containing continuous features.

## Background & Motivation

Decision trees are widely used in high-stakes domains such as healthcare and finance due to their strong interpretability. Traditional heuristic methods (ID3, C4.5, CART) are prone to suboptimal solutions, while learning truly optimal decision trees is NP-complete.

**Limitations of existing optimal regression tree methods:**

**MIP-based methods** (Bertsimas & Dunn 2019): guarantee global optimality but are computationally intractable; search space scales with sample size.

**DPB methods** (Zhang et al. 2023, current SOTA): scalable to 2 million samples, but **require binarization of continuous features**, sacrificing global optimality and potentially producing unnecessarily deep trees.

**Evolutionary search** (evtree): no optimality guarantees.

**Other MIP variants**: restrict tree size or the number of splits to maintain scalability.

**Core challenge**: Continuous features cause the search space of regression trees to expand dramatically. Even for a small dataset (Concrete) with depth $D=2$, $n=1030$, and $P=8$ features, the number of distinct tree structures exceeds $1.39 \times 10^{11}$.

## Method

### Two-Stage Optimization Reformulation

Regression tree training is decomposed into:

- **First-stage variables** $m = (a, b, c, d)$: tree structure (split feature $a$, threshold $b$, leaf prediction $c$, split indicator $d$)
- **Second-stage variables**: sample-dependent variables (leaf assignment $z$, loss terms $f, L$)

Objective function:

$$\min_{a,b,c,d} \sum_{i \in \mathcal{N}} Q_i(m). \quad Q_i(m) = \min_{z_i, L_{i*}} \frac{L_{i*}}{\hat{L}} + \frac{\lambda}{n}\sum_{t \in \mathcal{T}_D} d_t$$

### Reduced-Space Branch-and-Bound (RS-BB)

**Core design**: Branching is performed only over first-stage (tree structure) variables; second-stage variables are obtained by solving subproblems at each BB node.

**Key property**: The dimensionality of the search space is independent of the number of training samples, depending only on tree depth and the number of features.

**Convergence guarantee (Theorem 2)**:

$$\lim_{t \to \infty} \alpha_t = \lim_{t \to \infty} \beta_t = f^*$$

That is, branching solely on structure variables is sufficient to guarantee convergence to the global optimum.

### Three Acceleration Strategies

**Strategy 1: Implicit Leaf Predictions (Theorem 3)**

Given a fixed tree structure, the optimal prediction at each leaf node has a closed-form solution:

$$c_t^* = \frac{1}{|\mathcal{S}_t|}\sum_{i \in \mathcal{S}_t} y_i$$

This eliminates $|\mathcal{T}_L|$ continuous variables and exponentially reduces the number of BB nodes.

**Strategy 2: Threshold Discretization**

Continuous thresholds $b_t$ are restricted to the observed feature values in the training data, since any threshold between two consecutive sorted values induces the same split. Within the BB procedure, binary search is performed by selecting the median index of the feasible interval, eliminating at least one feasible split point per branching step.

**Strategy 3: Exact Last-Layer Subtree Resolution**

Once all structure variables above depth $D-2$ are fixed, the optimal depth-1 subtree for each parent node $P$ can be solved exactly via CART (max_depth=1). The split is accepted if the gain $\Delta(P) > \lambda|P|/\hat{L}$; otherwise, the node remains a leaf.

### Parallelization

The decomposable structure of the problem allows both lower- and upper-bound computations to be parallelized across the sample dimension, supporting large-scale distributed computation (40–1000 CPU cores are used in experiments).

## Key Experimental Results

### Main Results: Performance on Continuous-Feature Datasets

| Dataset | $n$ | $P$ | Method | Train RMSE | Test RMSE | Gap (%) | Time (s) |
|--------|------|-----|------|-----------|-----------|---------|----------|
| Concrete | 1,030 | 8 | **RS-ORT** | 11.96 | **11.80** | **<0.01** | 631 |
| | | | Bertsimas | 11.96 | 11.80 | <0.01 | 10047 |
| | | | OSRT | 11.96 | 11.80 | <0.01 | 116 |
| | | | CART | 12.01 | 12.57 | - | - |
| CPU ACT | 8,192 | 21 | **RS-ORT** | **5.99** | **6.03** | 8.08 | 14400 |
| | | | Bertsimas | 6.01 | 6.03 | 100.00 | 14400 |
| | | | OSRT | OoM | OoM | OoM | OoM |
| Seoul Bike | 8,760 | 12 | **RS-ORT** | **478.67** | **495.92** | **<0.01** | 10116 |

### Key Comparison: OSRT vs. RS-ORT

| Dimension | OSRT (SOTA) | RS-ORT |
|------|-------------|--------|
| Continuous features | Requires binarization | Handled directly |
| Search space | Scales with sample size | Independent of sample size |
| Maximum scale | 2M (binarized) | 2M (continuous) |
| Global optimality guarantee | Only after binarization | Under continuous features |
| Parallelization | Limited | Highly decomposable |

### Key Findings

1. On all small-to-medium datasets, RS-ORT achieves the same training/test RMSE as the best baseline, with clear advantages on large datasets.
2. On the Household dataset (2 million samples, continuous features), RS-ORT finds a globally optimal tree within 4 hours — **the first time any exact method has succeeded at this scale with continuous features**.
3. Trees produced by RS-ORT are typically 2–3 levels shallower than those from competing methods, while achieving better test performance.
4. OSRT runs out of memory on CPU ACT (8,192 samples, 21 features), whereas RS-ORT handles it via multi-core parallelization.

## Highlights & Insights

1. **Decoupling search space from sample size**: This is the fundamental distinction from all existing MIP-based methods, making the algorithm naturally amenable to large datasets.
2. **Three complementary acceleration strategies**: Closed-form solutions reduce variables, discretization shrinks the feasible domain, and last-layer exact resolution prunes the search — each exploiting the problem's special structure.
3. **Direct handling of continuous features**: Avoids information loss from binarization and prevents unnecessary increases in tree depth.
4. **High practical engineering value**: In interpretable AI settings (healthcare, compliance auditing), a globally optimal shallow tree is more valuable than a near-optimal deep one.
5. **Naturally parallelizable**: The sample-level decomposability of lower-bound computation enables parallelization without additional communication overhead.

## Limitations & Future Work

1. High computational resource requirements: large datasets demand hundreds to thousands of CPU cores, limiting accessibility for general users.
2. The fixed depth-2 experimental setting restricts evaluation of performance on deeper trees.
3. The 4-hour time limit may not suit real-time or interactive applications.
4. Only univariate (axis-aligned) splits are considered; extension to multivariate splits is not addressed.
5. Selection of the regularization parameter $\lambda$ relies on prior knowledge; the paper provides no automated tuning strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of two-stage decomposition and reduced-space BB represents a significant innovation for regression trees.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive experiments spanning small to 2-million-sample datasets; first exact method to tackle large-scale optimal regression trees with continuous features.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem formulation, detailed algorithmic description, and complete theoretical proofs.
- **Value**: ⭐⭐⭐⭐⭐ — Significant advancement for interpretable AI and the optimal decision tree literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Non-Asymptotic Analysis of Efficiency in Conformalized Regression](non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[CVPR 2026\] Label-Free Cross-Task LoRA Merging with Null-Space Compression](../../CVPR2026/optimization/label-free_cross-task_lora_merging_with_null-space_compression.md)
- [\[NeurIPS 2025\] Uncertainty Quantification for Reduced-Order Surrogate Models Applied to Cloud Microphysics](../../NeurIPS2025/optimization/uncertainty_quantification_for_reduced-order_surrogate_models_applied_to_cloud_m.md)
- [\[AAAI 2026\] A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](../../AAAI2026/optimization/a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)

</div>

<!-- RELATED:END -->
