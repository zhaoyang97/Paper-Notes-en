---
title: >-
  [Paper Note] Revisiting Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model
description: >-
  [ICML 2025][Labeled Stochastic Block Model] For the Labeled Stochastic Block Model (LSBM), this work proposes the IAC (Instance-Adaptive Clustering) algorithm. Using a two-stage strategy of a single spectral clustering followed by iterative likelihood refinement, it is the first to achieve community recovery matching the instance-specific information-theoretic lower bound with an $\mathcal{O}(n(\log n)^3)$ complexity, while simultaneously providing dual guarantees in expectat…
tags:
  - "ICML 2025"
  - "Labeled Stochastic Block Model"
  - "Instance-Optimal"
  - "Spectral Clustering"
  - "Maximum Likelihood Refinement"
  - "Information-Theoretic Lower Bound"
date: 2026-05-08
content_hash: aa8aa10763bf32d8
---

# Revisiting Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model

**Conference**: ICML 2025  
**arXiv**: [2306.12968](https://arxiv.org/abs/2306.12968)  
**Code**: None  
**Area**: Others  
**Keywords**: Labeled Stochastic Block Model, Instance-Optimal, Spectral Clustering, Maximum Likelihood Refinement, Information-Theoretic Lower Bound

## TL;DR

For the Labeled Stochastic Block Model (LSBM), this work proposes the IAC (Instance-Adaptive Clustering) algorithm. Using a two-stage strategy of a single spectral clustering followed by iterative likelihood refinement, it is the first to achieve community recovery matching the instance-specific information-theoretic lower bound with an $\mathcal{O}(n(\log n)^3)$ complexity, while simultaneously providing dual guarantees in expectation and with high probability.

## Background & Motivation

**LSBM Model**: The Labeled Stochastic Block Model is a natural generalization of the classic SBM. While SBM only involves binary interactions ("edge/no-edge") between nodes, LSBM allows edges to carry labels from an arbitrary label set $\mathcal{L} = \{0, 1, \ldots, L\}$, capturing rich interactions such as ratings in recommender systems or relationship types in social networks. Each pair of nodes $(v, w) \in \mathcal{I}_i \times \mathcal{I}_j$ observes label $\ell$ with probability $p(i, j, \ell)$.

**Instance-optimality vs. Minimax**: Existing works (e.g., Gao et al., 2017; Zhang & Zhou, 2016) only provide minimax optimal guarantees—which are optimal for the worst-case scenario but can be far from optimal for most instances. Instance-optimality implies that the algorithm adaptively achieves the optimal error rate for each specific LSBM instance, which is determined by the instance-specific KL divergence $D(\alpha, p)$.

**Existing Lower Bound (Yun & Proutiere, 2016)**: The expected number of misclassified nodes for any algorithm is at least $n\exp(-nD(\alpha, p))$, where $D(\alpha, p) = \min_{i \neq j} D_{L+}(\alpha, p(i), p(j))$ characterizes the information-theoretic difficulty of distinguishing the two hardest clusters. However, no prior algorithms have matched this lower bound, particularly in expectation.

**Computational Efficiency Bottleneck**: The algorithms proposed by Gao et al. (2017) and Xu et al. (2020) require performing spectral clustering once for each node ($n$ times in total), leading to $\Omega(n^2)$ complexity. The goal of this work is to achieve matching performance with only a single spectral clustering run.

## Method

### Overall Architecture

IAC is a two-stage algorithm: **Stage 1 (Spectral Clustering Initialization)** performs a one-time spectral clustering on the observed label adjacency matrices to obtain a rough cluster estimate and an automatic estimate of the number of clusters $\hat{K}$; **Stage 2 (Likelihood Refinement)** utilizes the estimated parameters $\hat{p}(i,j,\ell)$ to perform maximum likelihood reallocation for each node over $\log n$ iterations. The key theoretical guarantee is that when $\liminf_{n \to \infty} \frac{nD(\alpha,p)}{\log(n/s)} \geq 1$, IAC misclassifies at most $s$ nodes with high probability and in expectation.

### Key Designs

1. **Improved Spectral Clustering Initialization**:
    - **Function**: Extracts rough cluster estimates from the observed label adjacency matrices while automatically determining the number of clusters $K$.
    - **Mechanism**:
        - Concatenates the $L$ label adjacency matrices into $\bar{A} = [A_\Gamma^1, \ldots, A_\Gamma^L] \in \mathbb{R}^{n \times Ln}$ (unlike previous works that used randomly weighted sums $\sum_\ell w_\ell A_\Gamma^\ell$).
        - Truncates high-degree nodes first (removing the $\lfloor n\exp(-n\tilde{p}) \rfloor$ nodes with the largest degrees) to prevent them from distorting spectral properties.
        - Employs the **iterative power method** instead of direct SVD, combined with **singular value thresholding** to automatically determine $\hat{K}$, stopping when singular values fall below $\sqrt{n\tilde{p}} \log(n\tilde{p})$.
        - Expands the candidate center set for the k-means step to $\lceil(\log n)^2\rceil$ random nodes.
    - **Design Motivation**: Matrix concatenation provides tighter control over failure probability than random weighting, which is the key improvement to obtain guarantees in expectation (rather than just high probability). The iterative power method with $(\log n)^2$ matrix multiplications controls the overall computational complexity to $\mathcal{O}(n(\log n)^3)$.
    - **Guarantee**: $\hat{K} = K$ and the number of misclassified nodes is $\leq C/\bar{p}$ with probability $\geq 1 - \exp(-cn\bar{p})$.

2. **Iterative Likelihood Refinement**:
    - **Function**: Iteratively optimizes cluster assignments for each node based on the rough initialization.
    - **Mechanism**:
        - Estimates parameters $\hat{p}(i,j,\ell) = \frac{\sum_{u \in S_i} \sum_{v \in S_j} A_{uv}^\ell}{|S_i||S_j|}$.
        - For each node $v$, computes the log-likelihood of belonging to each cluster $k$ as $\sum_{i \in [\hat{K}]} \sum_{w \in S_i} \sum_{\ell=0}^L A_{vw}^\ell \log \hat{p}(k, i, \ell)$, and assigns it to the cluster with the maximum likelihood.
        - Performs $\lceil \log n \rceil$ iterations.
    - **Design Motivation**: Likelihood refinement optimizes along the non-linear optimal decision boundary, which is defined by the asymmetric KL divergence of $D(\alpha, p)$. Pure spectral methods use Euclidean distance in k-means and fail to capture this information-geometric non-linearity. Leveraging sparsity to traverse only labeled node pairs limits the complexity per round to $\mathcal{O}(\log n)$.
    - **Contraction Guarantee**: The number of misclassified nodes within $H$ is contracted to a factor of $1/\sqrt{n\bar{p}}$ in each iteration (Proposition 4.6).

3. **Analysis Framework of "Well-Behaved" Node Set $H$**:
    - **Function**: Columns/Nodes are partitioned into the "well-behaved" set $H$ (which can be correctly classified) and $\mathcal{I} \setminus H$ (which are inevitably misclassified).
    - **Core Definition**: $v \in H$ must satisfy three conditions: (H1) degree regularity $e(v, \mathcal{I}) \leq C_{H1} n\bar{p}$; (H2) likelihood separability $\sum_{i,\ell} e(v, \mathcal{I}_i, \ell) \log \frac{p(k,i,\ell)}{p(j,i,\ell)} \geq \frac{n\bar{p}}{\log^4(n\bar{p})}$; (H3) limited interactions with $H^c$, i.e., $e(v, \mathcal{I} \setminus H) \leq \frac{2n\bar{p}}{\log^5(n\bar{p})}$.
    - **Design Motivation**: Proposition 4.5 proves that $\mathbb{E}[|\mathcal{I} \setminus H|] / s \leq 1 + o(1)$, indicating that the size of the set outside $H$ matches the information-theoretic lower bound. Combining this with Proposition 4.6, which proves that all nodes inside $H$ are correctly classified, yields the overall instance-optimality guarantee.

### Loss & Training

IAC requires no training. In the likelihood refinement stage, it constructs the log-likelihood score function using the empirically estimated label probabilities $\hat{p}(i,j,\ell)$, without requiring prior knowledge of the true model parameters (including the number of clusters $K$).

## Key Experimental Results

### Main Results

| Algorithm | Instance-Optimal? | Guarantee in Expectation? | High Probability Guarantee? | Computational Complexity | Number of Spectral Clusterings |
|------|----------|----------|------------|-----------|----------|
| Ours (IAC) | ✅ | ✅ | ✅ | $\mathcal{O}(n(\log n)^3)$ | 1 |
| Yun & Proutiere (2016) | ✅ | ❌ (High probability only) | ✅ | $\mathcal{O}(n \cdot \text{polylog}(n))$ | 1 |
| Gao et al. (2017) | ❌ (minimax) | ✅ | ✅ | $\mathcal{O}(n^2)$+ | $n$ |
| Xu et al. (2020) | ✅ (Homogeneous LSBM only) | — | ✅ | $\mathcal{O}(n^2)$+ | $n$ |
| Zhang & Zhou (2016) | ❌ (minimax) | ✅ | — | No polynomial algorithm | — |

### Numerical Meaning of Key Theorems

| Condition / Result | Formula | Significance |
|----------|------|------|
| Information-theoretic lower bound | $\mathbb{E}[\varepsilon^\pi(n)] \geq n\exp(-nD(\alpha,p))$ | Unbeatable by any algorithm |
| IAC upper bound matching | $\limsup \frac{\mathbb{E}[\varepsilon^{IAC}(n)]}{s} \leq 1$ when $\frac{nD(\alpha,p)}{\log(n/s)} \geq 1$ | Matches the lower bound |
| Spectral clustering error | $O(1/\bar{p})$ misclassified nodes | Rough but sufficient for initialization |
| Single-round refinement contraction | $\frac{\text{Misclassified (Round }t+1\text{)}}{\text{Misclassified (Round }t\text{)}} \leq \frac{1}{\sqrt{n\bar{p}}}$ | Exponential convergence |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| IAC vs. Gao et al. (2017)'s penalized local MLE | IAC achieves higher empirical accuracy | Validated on multiple simple SBM instances |
| Spectral clustering only (no likelihood refinement) | Misclassification $O(1/\bar{p})$ | May already be near-optimal for simple SBMs |
| Spectral clustering + Likelihood refinement | Misclassification matches the lower bound | Significant improvement in asymmetric LSBMs |

### Key Findings

- IAC is the first LSBM clustering algorithm to simultaneously offer instance-optimality, guarantees in expectation, and an $\mathcal{O}(n \cdot \text{polylog}(n))$ complexity.
- The matrix concatenation approach (rather than random weighting) is a key technical innovation to obtain guarantees in expectation.
- The number of misclassified nodes in the spectral clustering phase is $O(1/\bar{p})$, and the likelihood refinement phase converges exponentially at a rate of $1/\sqrt{n\bar{p}}$—reducing the misclassifications within $H$ to 0 after $\log n$ rounds.
- The asymmetry of $D(\alpha, p)$ in the information-theoretic lower bound (determined by the min-max structure of the KL divergence) explains why pure Euclidean distance (e.g., k-means) cannot achieve optimality—requiring likelihood refinement to optimize along non-linear boundaries.

## Highlights & Insights

- **Information-Theoretic Tightness**: The upper and lower bounds match across the entire range of $s = o(n)$ , including exact recovery ($s = 1/2$) and almost exact recovery ($s = o(n)$), unifying previously fragmented results.
- **Automatic Cluster Number Discovery**: The singular value thresholding method does not require prior knowledge of $K$ and estimates the correct number of clusters with high probability $1 - \exp(-cn\bar{p})$.
- **Orders of Magnitude Improvement in Computational Efficiency**: Reduces the number of spectral clustering runs from $n$ to 1, cutting the overall complexity down from $\Omega(n^2)$ to $\mathcal{O}(n(\log n)^3)$, which makes the method practical for large-scale graphs.
- **New Analytical Tool**: The three-condition definition of the "well-behaved" node set $H$ decouples the proof of instance-optimality into two independent parts (complete correctness inside $H$ + matching the lower bound for node counts outside $H$), serving as an elegant analytical framework.

## Limitations & Future Work

- Assumes a fixed number of clusters $K$ (does not grow with $n$), which is not applicable to scenarios where the number of clusters increases with the graph scale—Gao et al. (2017) allows $K$ to depend on $n$.
- Restricted to the sparse regime $\bar{p} = \mathcal{O}(\log n / n)$, without covering dense graphs—Gao et al.'s guarantees also apply to the dense regime.
- Assumption (A1) requires bounded ratios of all label probabilities ($p(i,j,\ell) / p(i,k,\ell) \leq \eta$), which may not hold in real-world networks.
- Assumption (A3) excludes excessively sparse labels, restricting how non-uniform the label probabilities can be.
- Experimental validation is limited to simple synthetic SBM instances in the appendix, lacking validation on large-scale real-world networks (e.g., social networks, recommender systems).
- Fails to discuss robustness against model misspecification.

## Related Work & Insights

- **Line of SBM Theories**: Moving from detectability (Mossel et al., 2015a) $\to$ asymptotic exact recovery (Abbe et al., 2016) $\to$ minimax optimality (Gao et al., 2017; Zhang & Zhou, 2016) $\to$ the instance-optimality in this work, representing a progressive deepening of community detection theory.
- **Limitations of Spectral Methods**: Zhang (2024) analyzed tight bounds of spectral methods in SBMs. The authors of this work conjecture that in general LSBMs, spectral methods fail to achieve instance-optimality due to the limitations of Euclidean distance (k-means)—making combining them with likelihood refinement necessary.
- **Multiplex/Node-attributed SBM**: The results of this work are directly applicable to special cases of LSBM such as Censored SBM and signed networks. However, Multiplex SBM (Barbillon et al., 2016) can express cluster structures that LSBM cannot, representing an important direction for extension.
- **Inspiration**: Applying change-of-measure arguments (originating from multi-armed bandit theory) to the information-theoretic lower bounds of graph clustering demonstrates a deep connection between online learning and statistical learning.

## Rating

- Novelty: ⭐⭐⭐⭐ First to simultaneously achieve instance-optimality, expected guarantees, and low complexity for LSBM.
- Experimental Thoroughness: ⭐⭐ Primarily theoretical, with experiments validated only on simple synthetic instances in the appendix.
- Writing Quality: ⭐⭐⭐⭐ The theorems are clearly stated, the proof ideas are well introduced, and comparisons with existing works are comprehensive.
- Value: ⭐⭐⭐⭐ An important contribution to community detection theory, completely solving the instance-optimal clustering problem for LSBM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](../../NeurIPS2025/others/estimation_of_stochastic_optimal_transport_maps.md)
- [\[ICML 2025\] Revisiting the Predictability of Performative, Social Events](revisiting_the_predictability_of_performative_social_events.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](../../CVPR2026/others/mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)
- [\[ICML 2025\] Sampling from Binary Quadratic Distributions via Stochastic Localization](sampling_from_binary_quadratic_distributions_via_stochastic_localization.md)
- [\[ICML 2025\] Time-Aware World Model for Adaptive Prediction and Control](time-aware_world_model_for_adaptive_prediction_and_control.md)

</div>

<!-- RELATED:END -->
