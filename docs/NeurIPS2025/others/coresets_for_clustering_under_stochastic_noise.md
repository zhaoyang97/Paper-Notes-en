---
title: >-
  [Paper Note] Coresets for Clustering Under Stochastic Noise
description: >-
  [NeurIPS 2025][Coreset] This paper presents the first systematic study of $(k,z)$-clustering coreset construction under noisy data. It proposes a novel surrogate error metric $\mathsf{Err}_\alpha$ to replace the traditional $\mathsf{Err}$, achieving a $\text{poly}(k)$-fold reduction in coreset size and a $\text{poly}(k)$-fold tightening of quality guarantees under mild data assumptions, along with a noise-aware cluster-wise sampling algorithm.
tags:
  - "NeurIPS 2025"
  - "Coreset"
  - "Clustering"
  - "Stochastic Noise"
  - "k-Means"
  - "Surrogate Error Metric"
date: 2026-05-08
content_hash: ae191575875580eb
---

# Coresets for Clustering Under Stochastic Noise

**Conference**: NeurIPS 2025
**arXiv**: [2510.23438](https://arxiv.org/abs/2510.23438)  
**Area**: Others
**Keywords**: Coreset, Clustering, Stochastic Noise, k-Means, Surrogate Error Metric

## TL;DR
This paper presents the first systematic study of $(k,z)$-clustering coreset construction under noisy data. It proposes a novel surrogate error metric $\mathsf{Err}_\alpha$ to replace the traditional $\mathsf{Err}$, achieving a $\text{poly}(k)$-fold reduction in coreset size and a $\text{poly}(k)$-fold tightening of quality guarantees under mild data assumptions, along with a noise-aware cluster-wise sampling algorithm.

## Background & Motivation

**Background**: Coresets are a fundamental data compression tool for clustering problems — a weighted subset $S \subseteq P$ satisfying $\text{cost}_z(S, C) \in (1 \pm \varepsilon)\text{cost}_z(P, C)$ for all center sets $C$. The noiseless setting has a rich theory with optimal size bounds.

**Limitations of Prior Work**: Real-world data is almost always corrupted by noise (sensor errors, privacy mechanisms, transmission distortion), yet existing coreset constructions **rely entirely on noiseless assumptions**. The fundamental question of how a coreset built from noisy data $\hat{P}$ guarantees quality with respect to the true data $P$ has remained largely unaddressed.

**Key Challenge**: (a) The true data $P$ is unobservable, precluding direct quality evaluation; (b) noise uniformly inflates clustering costs, making the traditional metric $\mathsf{Err}$ overly pessimistic; (c) noise may alter cluster assignments of individual points.

**Key Insight**: The paper introduces the approximation-ratio metric $\mathsf{Err}_\alpha$, which focuses on relative performance within the neighborhood of optimal center sets, rendering it naturally immune to the uniform cost inflation induced by noise.

## Method

### Noise Models
**Model I**: Each point remains unchanged with probability $1-\theta$ and, with probability $\theta$, receives independent per-coordinate noise $\xi_{p,j} \sim D_j$ (zero-mean, unit variance, satisfying Bernstein's condition). This covers Gaussian mechanisms (differential privacy), sensor errors, and related settings.

**Model II**: Every point receives per-coordinate noise $\xi_{p,j} \sim D_j$ with variance $\sigma^2$; this is a more general formulation.

### Two Surrogate Metrics

**Traditional metric $\mathsf{Err}$**:

$$\mathsf{Err}(S, P) = \sup_{C \in \mathcal{C}} \frac{|\text{cost}_z(S, C) - \text{cost}_z(P, C)|}{\text{cost}_z(S, C)}$$

**New metric $\mathsf{Err}_\alpha$ (core innovation)**:

$$\mathsf{Err}_\alpha(S, P) = \sup_{C \in \mathcal{C}_\alpha(S)} \frac{r_P(C)}{r_S(C)} - 1$$

where $\mathcal{C}_\alpha(S) = \{C: r_S(C) \leq \alpha\}$ denotes the set of $\alpha$-approximate center sets on $S$, and $r_P(C) = \text{cost}_z(P, C) / \mathsf{OPT}_P$.

**Key Distinction**: $\mathsf{Err}$ takes the supremum over all center sets and is highly sensitive to uniform noise-induced cost inflation; $\mathsf{Err}_\alpha$ compares only **relative rankings** over approximately optimal center sets, where the uniform inflation cancels between numerator and denominator.

### Illustrative Example
Consider 1-Means on $\mathbb{R}$, with $n/2$ points at $-1$, $n/2$ points at $+1$, corrupted by $\mathcal{N}(0,1)$ noise:
- $\mathsf{Err}(\hat{P}, P) \approx 2/3$ → $r_P(\hat{P}, 1) \leq 25/9$ (overly pessimistic)
- $\mathsf{Err}_1(\hat{P}, P) \leq 1/n$ → $r_P(\hat{P}, 1) \leq 1 + 1/n$ (near-perfect)

### Theorem 3.1: Coreset via $\mathsf{Err}$
Using any standard algorithm that guarantees $\mathsf{Err}(S, \hat{P}) \leq \varepsilon$, the coreset size is $\tilde{O}(\min\{k^{1.5}\varepsilon^{-2}, k\varepsilon^{-4}\})$ (identical to the noiseless case), but the quality guarantee degrades to:

$$r_P(S, \alpha) \leq \left(1 + \varepsilon + O\left(\frac{\theta nd}{\mathsf{OPT}_P} + \sqrt{\frac{\theta nd}{\mathsf{OPT}_P}}\right)\right)^2 \cdot \alpha$$

The additional noise error $O(\theta nd / \mathsf{OPT}_P)$ cannot be eliminated.

### Theorem 3.3: Coreset via $\mathsf{Err}_\alpha$ (Main Contribution)
**Assumption 3.2**: (1) $\gamma$-cost stability ($\mathsf{OPT}_P(k-1)/\mathsf{OPT}_P(k) \geq 1 + \gamma$); (2) absence of strong outliers ($r_i \leq 8\bar{r}_i$).

**Algorithm 1 (CN$_\alpha$)**:
1. Partition $\hat{P}$ into $k$ clusters $\hat{P}_i$ according to $\hat{C}$
2. Compute per-cluster mean radius $\hat{r}_i = \sqrt{\text{cost}(\hat{P}_i, \hat{c}_i) / |\hat{P}_i|}$
3. **Remove extreme noise points** (ball clipping): $P_i' = \hat{P}_i \cap B(\hat{c}_i, R_i)$, where $R_i = 3\hat{r}_i + O\!\left(\sqrt{d}\log\frac{1+\theta kd}{\sqrt{\alpha-1}}\right)$
4. Uniformly sample from each cluster $|S_i| = O\!\left(\frac{\log k}{\varepsilon - \Delta} + \frac{(\alpha-1)\log k}{(\varepsilon - \Delta)^2}\right)$, where $\Delta = \frac{\sqrt{\alpha-1}\,\theta nd}{\alpha\,\mathsf{OPT}_P}$

The resulting quality guarantee is:

$$r_P(S, \alpha) \leq \left(1 + \varepsilon + O\left(\frac{\theta kd}{\mathsf{OPT}_P} + \frac{\sqrt{\alpha-1}}{\alpha} \cdot \frac{\sqrt{\theta kd \cdot \mathsf{OPT}_P} + \theta nd}{\mathsf{OPT}_P}\right)\right) \cdot \alpha$$

### Comparison of the Two Approaches

| Dimension | $\mathbf{CN}$ (Thm 3.1) | $\mathbf{CN}_\alpha$ (Thm 3.3) | Improvement Factor |
|-----------|------------------------|-----------------------------|-------------------|
| Coreset size | $\tilde{O}(\min\{k^{1.5}/\varepsilon^2, k/\varepsilon^4\})$ | $\tilde{O}(k/\varepsilon)$ | $\sqrt{k}/\varepsilon$ |
| Noise error in $r_P$ | $O(\theta nd / \mathsf{OPT}_P)$ | $O(\theta kd / \mathsf{OPT}_P)$ | $n/k$ |
| Assumptions required | None | Assumption 3.2 | — |

### Technical Core
The noise tolerance of $\mathsf{Err}_\alpha$ stems from the separation between **center drift** and **cost inflation**:
- **Center drift**: $\mathsf{C}(\hat{P}_i) = \mathsf{C}(P_i) + \frac{1}{|P_i|}\sum_p \xi_p$, controlled via concentration inequalities for independent noise, contributing $O(\theta d / \mathsf{OPT}_{P_i})$
- **Cost inflation**: $\text{cost}(P_i, \mathsf{C}(\hat{P}_i)) = \mathsf{OPT}_{P_i} + \frac{1}{|P_i|}\|\sum_p \xi_p\|_2^2$, likewise controllable

Cluster-wise analysis followed by aggregation replaces the global factor $n$ with $k$ in the total error.

## Key Experimental Results

### Coreset Quality on Real Datasets ($k$-Means, $\mathbf{CN}$ vs. $\mathbf{CN}_\alpha$)

| Dataset | $n$ | $d$ | $k$ | $\theta$ | $r_P$ of $\mathbf{CN}$ | $r_P$ of $\mathbf{CN}_\alpha$ ↓ |
|---------|-----|-----|-----|---------|----------------------|-------------------------------|
| MNIST | 60000 | 784 | 10 | 0.1 | 1.32 | **1.08** |
| CIFAR-10 features | 50000 | 512 | 10 | 0.1 | 1.28 | **1.05** |

### Robustness Under Non-i.i.d. Noise (Table 7)
When the noise covariance matrix $\Sigma \neq \sigma^2 I_d$, $\mathbf{CN}_\alpha$ still substantially outperforms $\mathbf{CN}$, demonstrating practical utility beyond the theoretical assumptions.

### Performance When Assumptions Are Violated
Even when datasets violate Assumption 3.2 (e.g., weakly separated data with small $\gamma$), $\mathbf{CN}_\alpha$ continues to perform well empirically (Table 2), indicating that the theoretical assumptions are conservative.

### Key Findings
- $\mathsf{Err}_\alpha$ yields tighter quality guarantees than $\mathsf{Err}$ in all tested scenarios
- Coreset size improvements are significant for large $k$ (factor of $\sqrt{k}/\varepsilon$)
- Ball clipping (Line 3 of Algorithm 1) is a key design: it stabilizes cluster structure without information loss

## Highlights & Insights
- **The $\mathsf{Err}_\alpha$ metric** constitutes a fundamental improvement to coreset evaluation: in noisy settings, it relaxes the requirement from "uniform approximation over all $C$" to "rank preservation over approximately optimal $C$," which is both natural and necessary
- **Cluster-wise sampling combined with ball clipping** is a clean and effective algorithmic design: by controlling the influence of extreme noise points, the analysis is localized from global to per-cluster
- **The separation of center drift from cost inflation** is the core insight: noise uniformly inflates costs without altering the relative advantage of optimal centers, and $\mathsf{Err}_\alpha$ precisely captures this invariance
- Theory and experiments are well-aligned: although Assumption 3.2 is required for theoretical guarantees, the algorithm remains effective when the assumption is not satisfied

## Limitations & Future Work
- The $\gamma$-cost stability in Assumption 3.2 is difficult to satisfy when clusters are heavily overlapping, and the required $\gamma$ grows with $\alpha$ and $\theta$
- $\mathsf{OPT}_P$ must be estimated — while $\text{cost}(\hat{P}, \hat{C})$ serves as a proxy, its accuracy degrades under high noise
- The primary analysis targets $k$-Means ($z=2$); extensions to $k$-Median ($z=1$) and other values of $z$ are sketched in the appendix but lack depth
- The $\sqrt{\theta nd / \mathsf{OPT}_P}$ term introduced by Cauchy–Schwarz in the $\mathsf{Err}$ analysis may not be tight — whether it can be eliminated remains open
- Experiments are limited in scale, without coverage of million-scale datasets or high-dimensional embedding spaces

## Rating
- Novelty: ⭐⭐⭐⭐ New metric + new algorithm + first systematic study of noisy coresets
- Theoretical Depth: ⭐⭐⭐⭐ Complete analysis of both metrics with explicit quantification of improvements
- Experimental Thoroughness: ⭐⭐⭐ Multi-dataset validation, though experiments lack large scale
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, intuitive examples are effective, notation is heavy
- Overall: ⭐⭐⭐⭐ Fills a theoretical gap in noisy coresets; practical value grows as noise-aware pipelines become more prevalent

## Related Work & Insights
- **vs. Standard coresets (Feldman–Langberg, Cohen-Addad et al.)**: All assume noiseless data, making $\mathsf{Err}$ directly applicable. This paper is the first to reveal the systematic over-pessimism of that metric under noise
- **vs. Robust coresets [38,52,54]**: These treat noise as identifiable outliers to be filtered. This paper does not assume noise is separable and constructs coresets directly from noisy data
- **vs. Ben-David & Haghtalab (2014)**: That work studies robustness of clustering algorithms (e.g., fast convergence of $k$-means under Gaussian perturbations), targeting algorithm solving rather than coreset construction
- **Complementary to Coreset for Robust Geometric Median (2510.24621)**: Both papers extend coreset theory from different angles — the former addresses size dependence under deterministic outliers, the latter addresses cost inflation under stochastic noise. The technical tools differ, but both challenge the limits of classical analysis
- The design paradigm of $\mathsf{Err}_\alpha$ — comparing relative performance only within the neighborhood of approximate optima — is transferable to noisy graph partition coresets, privacy-preserving facility location, and related problems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)
- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](../../AAAI2026/others/enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)
- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](../../ICLR2026/others/distributed_algorithms_for_euclidean_clustering.md)

</div>

<!-- RELATED:END -->
