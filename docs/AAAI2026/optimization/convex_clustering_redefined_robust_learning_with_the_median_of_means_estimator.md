---
title: >-
  [Paper Note] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator
description: >-
  [AAAI 2026][Optimization][Convex Clustering] This paper proposes COMET (Convex Clustering with Median of Means Estimator), which integrates the Median of Means (MoM) estimator into the convex clustering framework. Through random binning, truncated distance regularization, and ADAM optimization, COMET achieves robust clustering under noise and outliers without requiring a pre-specified number of clusters. The paper establishes weak consistency guarantees theoretically and demonstrates comprehensive superiority over existing methods on both synthetic and real-world datasets.
tags:
  - AAAI 2026
  - Optimization
  - Convex Clustering
  - Median of Means Estimator
  - Robust Clustering
  - Outlier Detection
  - ADAM Optimization
date: 2026-05-08
content_hash: ab92afd6e5825f57
---

# Convex Clustering Redefined: Robust Learning with the Median of Means Estimator

**Conference**: AAAI 2026  
**arXiv**: [2511.14784](https://arxiv.org/abs/2511.14784)  
**Code**: [https://tinyurl.com/2v3dx75x](https://tinyurl.com/2v3dx75x)  
**Area**: Optimization  
**Keywords**: Convex Clustering, Median of Means Estimator, Robust Clustering, Outlier Detection, ADAM Optimization

## TL;DR
This paper proposes COMET (Convex Clustering with Median of Means Estimator), which integrates the Median of Means (MoM) estimator into the convex clustering framework. Through random binning, truncated distance regularization, and ADAM optimization, COMET achieves robust clustering under noise and outliers without requiring a pre-specified number of clusters. The paper establishes weak consistency guarantees theoretically and demonstrates comprehensive superiority over existing methods on both synthetic and real-world datasets.

## Background & Motivation

Clustering is a fundamental task in unsupervised learning, yet traditional methods face several core challenges:

**k-means family**: Requires a pre-specified number of clusters $k$, is sensitive to initialization, and degrades on high-dimensional and noisy data.

**Convex Clustering**: Guarantees globally optimal solutions via convex optimization, but struggles with high-dimensional noisy data — strong fusion regularization (large $\gamma$) can cause inappropriate merging of outliers with genuine clusters.

**Robustness Gap**: Existing convex clustering methods (e.g., Robust Convex Clustering) rely on k-NN combined with a Gaussian kernel for weight assignment; arbitrary selection of bandwidth $\phi$ can lead to cluster collapse or the formation of spurious clusters.

**Key Challenge**: The regularization term in convex clustering encourages merging of adjacent centroids, but in noisy or outlier-contaminated data, the neighborhood structure induced by outliers misleads the fusion process.

**Core Idea**: Exploit the inherent robustness of the MoM estimator — randomly partition the data into $L$ subsets and take the median of per-subset losses rather than their mean, so that outliers corrupt only a minority of subsets and are automatically suppressed by the median operation. This is further combined with a truncated distance $\min(\mu, \|u_i - u_j\|^2)$ to limit the fusion influence of distant point pairs.

## Method

### Overall Architecture

The COMET pipeline proceeds as follows:
1. Construct a k-NN graph and Gaussian weights $w_{ij}$.
2. Apply random binning to partition the data into $L$ subsets.
3. Run ADAM gradient descent for $N$ iterations on the MoM objective.
4. Build a connectivity graph based on pairwise distances between optimized centroids.
5. Extract connected components as clusters; small components are merged into a noise cluster.

### Key Designs

1. **MoM Objective Function**:

    - Function: Replaces the mean with the median in computing the data-fitting term, suppressing the influence of outliers.
    - Mechanism: Randomly partition $n$ data points into $l = O(n)$ subsets $B_1, \ldots, B_l$, each of size $b = \lfloor n/l \rfloor$; take the median of the convex clustering loss over each subset:
      $$\mathrm{MoM}_B(U) = \mathrm{Median}\!\left(\left\{\frac{1}{2b}\sum_{i \in B_j}\|x_i - u_i\|^2\right\}_{j=1}^l\right)$$
    - Design Motivation: Outliers contaminate at most a small fraction of subsets; the median operation naturally discards contaminated subsets. The MoM estimator admits rigorous breakdown point analysis and concentration guarantees.

2. **Truncated Distance Regularization**:

    - Function: Limits the contribution of distant point pairs to the fusion term.
    - Mechanism: Replaces $\|u_i - u_j\|^2$ with $\min(\mu, \|u_i - u_j\|^2)$; when two centroids are farther apart than $\mu$, the gradient vanishes and no further fusion is attempted.
    - Design Motivation: Prevents outliers (points with large distances) from misleading cluster centroids via fusion regularization; $\mu$ serves as a hyperparameter controlling sensitivity to outlier detection.

3. **Random Binning**:

    - Function: Randomly reshuffles the assignment of data points to subsets at each iteration.
    - Mechanism: Inspired by the random features method (Rahimi & Recht), simplified to fixed-size bins with reassignment at each iteration.
    - Design Motivation: Avoids bias introduced by fixed binning; randomization yields more stable median estimates.

### Loss & Training

The overall objective function is:
$$C(U) = \mathrm{MoM}_B(U) + \frac{\gamma}{2}\sum_{i,j} w_{ij}\min\!\left\{\mu,\, \|u_i - u_j\|^2\right\}$$

The gradient is:
$$g_i = \frac{1}{b}(u_i - x_i)\,\mathbb{1}(i \in B_{l_t}) + \gamma\sum_j w_{ij}(u_i - u_j)\,\mathbb{1}(\|u_i - u_j\|^2 < \mu)$$

- The ADAM optimizer is used (rather than ADMM or AMA) because the objective is non-convex.
- Centroids are initialized as $u_i = x_i$.
- After optimization, clusters are extracted via connected components using threshold $\eta_1$.

## Key Experimental Results

### Main Results

**Results on Real-World Datasets (ARI under 10% Noise)**:

| Dataset ($k$) | KM | MKM | CC | RCC | RConv | RBKM | **COMET** |
|---|---|---|---|---|---|---|---|
| Newthyroid (3) | 0.34 | 0.40 | 0.69 | 0.00 | 0.81 | 0.11 | **0.97** |
| Wisconsin (2) | 0.52 | 0.47 | 0.81 | 0.01 | 0.85 | 0.15 | **0.87** |
| Wine (3) | 0.66 | 0.59 | 0.59 | 0.00 | 0.22 | 0.01 | **0.79** |
| Dermatology (6) | 0.61 | 0.56 | 0.21 | 0.00 | 0.66 | 0.004 | **0.81** |
| Lung-Discrete (7) | 0.44 | 0.50 | 0.07 | 0.41 | 0.39 | 0.01 | **0.71** |
| ORLRaws10p (10) | 0.33 | 0.33 | 0.53 | 0.00 | 0.54 | 0.02 | **0.73** |

### Ablation Study

**Computational Complexity Comparison**:

| Algorithm | Complexity | Notes |
|---|---|---|
| **COMET** | **$O(Nnkd)$** | $N$: iterations, $n$: data points, $k$: neighbors, $d$: dimension |
| Convex Clustering | $O(N(n^2d + d\varepsilon))$ | The $n^2$ term renders it infeasible on large datasets |
| RCC | $O(N(n^2d + nkd))$ | Also contains an $n^2$ term |
| RConv | $O(Nnkd)$ | Comparable to COMET |

**Noise Robustness on Brain Dataset (ARI)**:

| Noise % | KM | MKM | CC | RCC | RConv | **COMET** |
|---|---|---|---|---|---|---|
| 0% | 0.28 | 0.23 | 0.64 | 0.00 | 0.56 | **0.65** |
| 5% | 0.31 | 0.31 | 0.64 | 0.00 | 0.56 | **0.66** |
| 10% | 0.26 | 0.26 | 0.64 | 0.00 | 0.56 | **0.66** |
| 15% | 0.22 | — | — | 0.00 | — | **Consistently stable** |

### Key Findings

- **COMET significantly outperforms all competing methods on every real-world dataset** (confirmed via Wilcoxon Rank-Sum test).
- **RCC completely fails**: ARI $\approx 0.00$ on nearly all datasets, with severely inflated cluster count estimates (e.g., Newthyroid: $k^* = 212$ vs. $k = 3$).
- **Noise robustness of COMET**: As noise increases from 0% to 15%, ARI remains stable or even slightly improves, whereas KM and MKM degrade substantially.
- **Automatic cluster number estimation**: COMET's estimated $k^*$ is slightly larger than the ground-truth $k$ (e.g., Newthyroid: $k^* = 4.14$ vs. $k = 3$) but with minimal variance ($\pm 0.36$), far superior to CC ($k^* = 14.14$) and RCC ($k^* = 212$).
- **Advantage on high-dimensional data**: Performance gains are more pronounced on high-dimensional datasets such as ORLRaws10p ($d = 10304$) and Lung-Discrete ($d = 325$).

## Highlights & Insights

- **Complete theoretical guarantees**: Theorem 1 provides a finite-sample error bound; Corollary 1.1 establishes weak consistency under the condition $d = o(n)$; Corollary 1.2 derives an $O(1/\sqrt{n})$ convergence rate.
- **Two-layer robustness**: The MoM estimator provides robust estimation of the data-fitting term, while truncated distance regularization enforces robustness in the fusion term. This dual-layer defense renders outliers nearly unable to influence clustering outcomes.
- **No pre-specified $k$ required**: Cluster number is determined automatically via connected components, offering an advantage over k-means variants that rely on auxiliary methods such as Gapstat.
- **Fair experimental design**: The k-means family is also evaluated with Gapstat-estimated $k$ rather than the ground-truth value, ensuring a fair comparison.

## Limitations & Future Work

- The objective function is non-convex due to the MoM and $\min$ operations, so only local optimality can be guaranteed; ADAM mitigates this in practice, but global optimality is theoretically lost.
- The method involves multiple hyperparameters ($\gamma$, $\mu$, $\eta_1$, $\phi$, $k$, $N$), and tuning may require grid search.
- The estimated $k^*$ is systematically slightly larger than the ground-truth value, indicating that the separation of noise points could be further refined.
- Computational complexity is comparable to RConv; scalability to very large datasets ($n > 10^6$) has not been validated.
- Evaluation is conducted only on datasets with known class labels (using ARI/AMI); practical utility in fully unlabeled settings remains to be verified.
- The strategy of merging small clusters into a "noise cluster" is overly simplistic and may inadvertently discard genuine small clusters.

## Related Work & Insights

- The robustness theory of MoM estimators (breakdown point analysis) provides a rigorous mathematical foundation for clustering robustness.
- The regularization path (clusterpath) concept from convex clustering can be combined with COMET to enable automatic selection of optimal $\gamma$.
- The truncated distance idea is analogous to the Huber loss in regression and can be generalized to robustify other convex optimization problems.
- The random binning strategy can be combined with mini-batch SGD to further improve efficiency on large-scale data.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)

</div>

<!-- RELATED:END -->
