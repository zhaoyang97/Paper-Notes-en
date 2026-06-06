---
title: >-
  [Paper Note] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator
description: >-
  [AAAI 2026][Optimization][Convex Clustering] This paper integrates the Median of Means (MoM) estimator into the convex clustering framework…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Convex Clustering"
  - "Median of Means"
  - "Robust Clustering"
  - "Outlier Detection"
  - "Adam Optimization"
date: 2026-05-08
content_hash: 140da55be4ca1140
---

# Convex Clustering Redefined: Robust Learning with the Median of Means Estimator

**Conference**: AAAI 2026
**arXiv**: [2511.14784](https://arxiv.org/abs/2511.14784)  
**Code**: [https://tinyurl.com/2v3dx75x](https://tinyurl.com/2v3dx75x)  
**Area**: Clustering / Robust Optimization
**Keywords**: Convex Clustering, Median of Means, Robust Clustering, Outlier Detection, Adam Optimization

## TL;DR
This paper integrates the Median of Means (MoM) estimator into the convex clustering framework, proposing the COMET algorithm. By combining random binning with median aggregation, COMET achieves robustness to noise and outliers without requiring prior knowledge of the number of clusters $k$. Weak consistency is established theoretically, and experiments on multiple real-world datasets demonstrate substantial improvements over six baselines, including k-means, MoM k-means, and convex clustering.

## Background & Motivation

### Strengths and Limitations of Convex Clustering
Traditional clustering methods such as k-means formulate the problem as a non-convex optimization, suffering from three inherent drawbacks: (1) the number of clusters $k$ must be specified in advance; (2) results are sensitive to initialization; and (3) performance degrades on high-dimensional or noisy data. **Convex clustering** (sum-of-norms clustering) reformulates the problem as a convex optimization, guaranteeing a globally unique solution and fundamentally eliminating initialization sensitivity. Its objective function is:

$$\min_{\mathbf{u}} \frac{1}{2} \left[ \|\mathbf{x}_i - \mathbf{u}_i\|_2^2 + \gamma \sum_{i,j} \theta_{ij} \|\mathbf{u}_i - \mathbf{u}_j\|_p^2 \right]$$

The first term keeps each point close to its centroid, while the second term encourages centroid fusion through the regularization parameter $\gamma$, automatically determining the number of clusters. However, when the fusion regularization is too strong, convex clustering tends to incorrectly merge outliers with true clusters, leading to poor performance on high-dimensional noisy data.

### Robustness Guarantees of Median of Means
The core idea of the MoM estimator is to randomly partition data into $L$ disjoint subsets $B_1, \ldots, B_L$, compute a statistic independently on each subset, and then take the median as the final estimate. Since outliers typically contaminate only a small number of subsets, the median operation effectively suppresses their influence. This property holds even under adversarial conditions, providing strong robustness and concentration inequality guarantees.

### Motivation
How can the robustness of MoM be combined with the stability of convex clustering? Existing approaches such as Robust Convex Clustering and Robust Continuous Clustering address robustness to some extent but still underperform under high noise levels. This paper aims to construct a clustering framework that is **naturally robust to noise and outliers without requiring prior knowledge of $k$**.

## Method

### Overall Architecture: The COMET Algorithm
COMET (**Co**nvex clustering with **Me**dian of means es**T**imator) comprises three key components:

#### 1. Random Binning
The index set $\{1,2,\ldots,n\}$ is randomly partitioned into $l = \mathcal{O}(n)$ subsets $B = \{B_i\}_{i=1}^l$, each containing $b = \lfloor n/l \rfloor$ samples. This strategy is inspired by random feature methods, though here it is simplified for data binning rather than kernel approximation.

#### 2. MoM Cost Function
The "contribution" of point $\mathbf{x}_r$ to the convex cost function is defined as:

$$f_U(\mathbf{x}_r) = \frac{1}{2}\|\mathbf{x}_r - \mathbf{u}_r\|_2^2 + \frac{\gamma}{2}\sum_{i,j} w_{ij}\|\mathbf{u}_i - \mathbf{u}_j\|_2^2$$

Rather than directly minimizing the global mean $\frac{1}{n}\sum_r f_U(\mathbf{x}_r)$, the MoM objective is minimized:

$$C(\mathbf{U}) = \text{Median}\left(\left\{\frac{1}{b}\sum_{r \in B_j} f_U(\mathbf{x}_r)\right\}_{j=1}^l\right)$$

By taking the median, the influence of outliers on the cost function is effectively suppressed.

#### 3. Distance Truncation Mechanism
A hyperparameter $\mu$ is introduced to truncate pairwise distances: $\|\mathbf{u}_i - \mathbf{u}_j\|_2^2$ is replaced by $\min(\mu, \|\mathbf{u}_i - \mathbf{u}_j\|_2^2)$. When the distance between any pair of centroids exceeds the threshold $\mu$, the corresponding fusion edge is severed, preventing spurious merging of outliers with true clusters. The final cost function is:

$$C(\mathbf{U}) = \text{MoM}_B(\mathbf{U}) + \frac{\gamma}{2}\sum_{i,j} w_{ij} \min\{\mu, \|\mathbf{u}_i - \mathbf{u}_j\|_2^2\}$$

#### 4. Adam Gradient Descent Optimization
As the introduction of truncation renders the objective non-convex, the Adam optimizer is employed to minimize the cost function. The gradient is:

$$g_i = \frac{1}{b}(\mathbf{u}_i - \mathbf{x}_i)\mathbb{1}(i \in B_{l_t}) + \gamma \sum_j w_{ij}(\mathbf{u}_i - \mathbf{u}_j)\mathbb{1}(\|\mathbf{u}_i - \mathbf{u}_j\|_2^2 < \mu)$$

#### 5. Post-Processing Cluster Assignment
After optimization, a graph is constructed with $\{\mathbf{u}_i\}$ as vertices, connecting $i$ and $j$ if $\|\mathbf{u}_i - \mathbf{u}_j\| < \eta_1$. Each connected component is treated as a cluster; components smaller than half the average cluster size are merged and labeled as noise.

### Weight Design
The standard $k$-nearest neighbor Gaussian kernel scheme is adopted: $w_{ij} = \mathbb{1}_{ij,k} \cdot e^{-\phi\|\mathbf{x}_i - \mathbf{x}_j\|_2^2}$, where $\phi$ is the bandwidth parameter.

### Theoretical Guarantees
- **Theorem 1**: Under the assumption of bounded noise $|\epsilon_i| \leq M$, a finite-sample error bound is established, providing a probabilistic upper bound on $\|\hat{\mathbf{u}} - \mathbf{u}\|$.
- **Corollary 1.1**: When $d = o(n)$, centroid estimation is weakly consistent, i.e., $\frac{1}{2ndb}\|\hat{\mathbf{u}} - \mathbf{u}\|^2 \xrightarrow{p} 0$.
- **Corollary 1.2**: The convergence rate is $O(1/\sqrt{n})$.
- **Computational Complexity**: $\mathcal{O}(Nnkd)$, on par with Robust Convex Clustering and more efficient than standard convex clustering at $\mathcal{O}(N(n^2d + d\epsilon))$.

## Key Experimental Results

### Experimental Setup
- **Baselines**: k-means (KM), MoM k-means (MKM), Convex Clustering (CC), Robust Continuous Clustering (RCC), Robust Convex Clustering (RConv), Robust Bregman k-means (RBKM)
- **Evaluation Metrics**: Adjusted Rand Index (ARI), Adjusted Mutual Information (AMI)
- **Noise Injection**: $p\%$ noise points sampled uniformly within the minimum bounding hypercube of the data are added
- For fair comparison, GapStat is used to automatically estimate the number of clusters for methods requiring $k$

### Table 1: Performance on Real-World Datasets (10% Noise)

| Dataset | Metric | KM | MKM | CC | RCC | RConv | RBKM | **COMET** |
|--------|------|-----|------|------|------|-------|------|-----------|
| Newthyroid (k=3) | ARI | 0.34±0.21 | 0.40±0.26 | 0.69±0.04 | 0.00±0.00 | 0.81±0.21 | 0.11±0.03 | **0.97±0.01** |
| Wine (k=3) | ARI | 0.66±0.31 | 0.59±0.29 | 0.59±0.15 | 0.00±0.00 | 0.22±0.28 | 0.01±0.02 | **0.79±0.15** |
| Dermatology (k=6) | ARI | 0.61±0.17 | 0.56±0.17 | 0.21±0.00 | 0.00±0.00 | 0.66±0.01 | 0.004±0.02 | **0.81±0.06** |
| Lung-Discrete (k=7) | ARI | 0.44±0.09 | 0.50±0.10 | 0.07±0.03 | 0.41±0.12 | 0.39±0.05 | 0.01±0.01 | **0.71±0.02** |
| ORLRaws10p (k=10) | ARI | 0.33±0.11 | 0.33±0.10 | 0.53±0.00 | 0.00±0.00 | 0.54±0.00 | 0.02±0.01 | **0.73±0.00** |

COMET achieves the highest ARI on all datasets with substantially lower standard deviations, demonstrating excellent algorithmic stability. Wilcoxon rank-sum tests confirm that the differences between COMET and the baselines are statistically significant (marked with †).

### Table 2: Performance on the Brain Dataset under Varying Noise Levels

| Noise (%) | Metric | KM | MKM | CC | RCC | RConv | RBKM | **COMET** |
|---------|------|-----|------|------|------|-------|------|-----------|
| 0% | ARI | 0.28±0.10 | 0.23±0.11 | 0.64±0.00 | 0.00±0.00 | 0.56±0.00 | 0.01±0.01 | **0.65±0.00** |
| 5% | ARI | 0.31±0.13 | 0.31±0.13 | 0.64±0.00 | 0.00±0.00 | 0.56±0.01 | 0.01±0.01 | **0.66±0.00** |
| 10% | ARI | 0.26±0.10 | 0.26±0.10 | 0.64±0.02 | 0.00±0.00 | 0.56±0.06 | 0.016±0.02 | **0.66±0.03** |
| 15% | ARI | 0.22±0.09 | 0.10±0.08 | 0.63±0.02 | 0.00±0.00 | 0.55±0.06 | 0.02±0.02 | **0.66±0.03** |
| 20% | ARI | 0.19±0.11 | 0.08±0.07 | 0.63±0.04 | 0.00±0.00 | 0.63±0.03 | 0.02±0.02 | **0.65±0.02** |

The Brain dataset contains 42 brain tumor samples, 5597 features, and 5 classes. COMET maintains an ARI above 0.65 across all noise levels, while k-means and MKM degrade significantly as noise increases (dropping to 0.08–0.19 at 20% noise). RCC and RBKM nearly fail completely under all conditions.

## Highlights & Insights

1. **Elegant Integration of MoM and Convex Clustering**: Through random binning and median aggregation, the robust statistical theory of MoM is grafted onto convex clustering while preserving the advantage of not requiring prior knowledge of $k$.
2. **Edge-Severing via Distance Truncation**: The truncation function $\min(\mu, \|\cdot\|^2)$ controlled by hyperparameter $\mu$ provides a second layer of robustness—point pairs that are too far apart are no longer fused.
3. **Solid Theoretical Foundation**: Finite-sample error bounds and weak consistency are established using the Hanson-Wright inequality, with a convergence rate of $O(1/\sqrt{n})$.
4. **Computational Efficiency**: Complexity of $\mathcal{O}(Nnkd)$ is on par with existing robust methods while outperforming classical convex clustering.
5. **Comprehensive Experiments**: Evaluation covers synthetic data and 6 real-world datasets of varying scale and dimensionality, 5 noise levels, 6 baselines, and Wilcoxon statistical tests.

## Limitations & Future Work

1. **Non-Convex Optimization**: The introduction of truncation renders the objective non-convex; Adam can only find local optima, forfeiting the globally unique solution that is the core advantage of convex clustering.
2. **Large Number of Hyperparameters**: Six hyperparameters require tuning—$N, k(\text{kNN}), \phi, \gamma, \mu, \eta_1$—setting a relatively high barrier for practical use.
3. **High-Dimensional Limitations**: Theoretical consistency requires $d = o(n)$, which may not hold in modern high-dimensional small-sample settings (e.g., genomics where $d \gg n$).
4. **Noise Distribution Assumptions**: Only uniformly distributed noise is tested; behavior under structured noise (e.g., subspace noise, heavy-tailed distributions) remains unknown.
5. **Cluster Count Estimation Bias**: On some datasets (e.g., ORLRaws10p, Wisconsin), the estimated number of clusters deviates from the ground truth (14 vs. 10, 3 vs. 2).
6. **Scalability Not Validated**: Although the theoretical complexity is reasonable, performance on large-scale data ($n > 10^5$) is not reported.

## Related Work & Insights

- **Convex Clustering**: First proposed by [Pelckmans & De Moor 2005]; regularization path algorithms developed by [Hocking et al. 2011]; ADMM/AMA optimization introduced by [Chi & Lange 2015].
- **Robust Clustering**: [Wang et al. 2016] proposed Robust Convex Clustering, enhancing robustness via feature selection; [Shah & Koltun 2017] proposed Robust Continuous Clustering for non-convex continuous optimization.
- **MoM Clustering**: [Brunet et al. 2022] applied bootstrap MoM to k-means; [Paul et al. 2021] unified robust centroid clustering under general Bregman divergences.
- **SDP Relaxation**: [Mixon et al. 2017] proved that semidefinite programming relaxation achieves exact recovery under the random unit ball model.

## Rating

- **Novelty**: 3/5 — Both MoM and convex clustering are mature techniques; the combination is innovative but neither component is entirely new.
- **Technical Depth**: 4/5 — Theoretical analysis (finite-sample bounds, consistency) is rigorous, with professional use of the Hanson-Wright inequality framework.
- **Experimental Thoroughness**: 4/5 — Multiple datasets, baselines, noise levels, and statistical tests are included, though large-scale experiments are absent.
- **Writing Quality**: 3/5 — Formula derivations are dense, and the interactions among hyperparameters are insufficiently explained.
- **Value**: 3/5 — Robust clustering has practical value, but high-dimensional limitations and hyperparameter complexity may hinder adoption.
- **Overall**: 3.5/5 — A solid contribution with strong theory and experiments; the MoM-based approach to mitigating outlier influence is valuable, but the combinatorial novelty is limited and the convexity guarantee of convex clustering is sacrificed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption](../../ICML2026/optimization/convex_basins_in_single-index_model_loss_landscapes_applications_to_robust_recov.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](../../ICLR2026/optimization/convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[AAAI 2026\] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](../../NeurIPS2025/optimization/improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[NeurIPS 2025\] Preference Learning with Response Time: Robust Losses and Guarantees](../../NeurIPS2025/optimization/preference_learning_with_response_time_robust_losses_and_guarantees.md)

</div>

<!-- RELATED:END -->
