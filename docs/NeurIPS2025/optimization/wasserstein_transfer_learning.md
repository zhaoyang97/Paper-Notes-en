---
title: >-
  [Paper Note] Wasserstein Transfer Learning
description: >-
  [NeurIPS 2025][Optimization][Transfer Learning] This paper proposes WaTL, the first transfer learning framework for distributional outputs in Wasserstein space. It adopts a three-step procedure — weighted auxiliary estimation, bias correction, and projection — combined with adaptive source selection, to transfer knowledge from source domains and improve distributional regression in the target domain.
tags:
  - NeurIPS 2025
  - Optimization
  - Transfer Learning
  - Wasserstein Space
  - Fréchet Regression
  - Optimal Transport
  - Distributional Data Analysis
date: 2026-05-08
content_hash: c575e4d45119d92a
---

# Wasserstein Transfer Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.17404](https://arxiv.org/abs/2505.17404)
**Code**: [GitHub](https://github.com/h7nian/WaTL)
**Area**: Optimization
**Keywords**: Transfer Learning, Wasserstein Space, Fréchet Regression, Optimal Transport, Distributional Data Analysis

## TL;DR

This paper proposes WaTL, the first transfer learning framework for distributional outputs in Wasserstein space. It adopts a three-step procedure — weighted auxiliary estimation, bias correction, and projection — combined with adaptive source selection, to transfer knowledge from source domains and improve distributional regression in the target domain.

## Background & Motivation

Transfer learning has achieved remarkable success on Euclidean data such as images and text. However, existing methods almost universally assume Euclidean structure and are ill-suited for regression models whose **outputs are probability distributions**. In applications such as mortality analysis, temperature studies, and physical activity monitoring, the observations themselves are probability distributions, which naturally reside in Wasserstein space.

The Wasserstein space is a geodesic metric space that **lacks a conventional linear structure** — the sum of two density functions is not a valid density. Standard transfer learning methods (e.g., high-dimensional linear model transfer, nonparametric regression transfer) therefore cannot be directly applied. Although prior work has employed optimal transport metrics for domain adaptation, the problem of transfer learning when the output is a probability distribution has not been systematically studied.

The central motivation of this paper is: when target-domain data are scarce, how can one leverage multiple source domains — using the Wasserstein metric to quantify inter-domain discrepancy — to achieve transfer learning in the space of probability distributions?

## Method

### Overall Architecture

WaTL builds upon **Fréchet regression**, defining the target regression function as the conditional Fréchet mean:

$$m_G^{(0)}(x) = \arg\min_{\mu \in \mathcal{W}} E\{s_G^{(0)}(x) d_{\mathcal{W}}^2(\nu^{(0)}, \mu)\}$$

The core mechanism is a three-step procedure: weighted aggregation → bias correction → Wasserstein projection.

### Key Designs

1. **Weighted Auxiliary Estimator (Step 1)**: Information from the target domain and all source domains is aggregated with weights proportional to sample sizes. Specifically, for each domain $k$, one computes $\hat{f}^{(k)}(x) = n_k^{-1} \sum_{i=1}^{n_k} s_{iG}^{(k)}(x) F_{\nu_i^{(k)}}^{-1}$, and the weighted average is $\hat{f}(x) = \frac{1}{n_0 + n_{\mathcal{A}}} \sum_{k=0}^{K} n_k \hat{f}^{(k)}(x)$. This step exploits all available domains but may introduce bias.

2. **Bias Correction (Step 2)**: The aggregated estimate is regularized using target-domain data. One solves $\hat{f}_0(x) = \arg\min_{g \in L^2(0,1)} \frac{1}{n_0}\sum_{i=1}^{n_0} s_{iG}^{(0)}(x) \|F_{\nu_i^{(0)}}^{-1} - g\|_2^2 + \lambda \|g - \hat{f}(x)\|_2$. The regularization parameter $\lambda$ balances target-domain fidelity against the contribution of auxiliary information.

3. **Wasserstein Projection (Step 3)**: The corrected estimate is projected onto the Wasserstein space to ensure the output is a valid probability distribution: $\hat{m}_G^{(0)}(x) = \arg\min_{\mu \in \mathcal{W}} \|F_\mu^{-1} - \hat{f}_0(x)\|_2$. Uniqueness of the projection is guaranteed by the fact that $\mathcal{W}$ is a closed convex subset of $L^2(0,1)$.

4. **Adaptive Source Selection (AWaTL)**: When the informativeness of source domains is unknown, AWaTL ranks sources by computing the empirical discrepancy score $\hat{\psi}_k = \|\hat{f}^{(0)}(x) - \hat{f}^{(k)}(x)\|_2$, and selects the $L$ sources with the smallest discrepancy as the informative set. $L$ is determined via cross-validation.

### Loss & Training

- The regularization parameter $\lambda$ is selected via five-fold cross-validation over the range $[0, 3]$.
- The theoretically optimal order of regularization is $\lambda \asymp n_0^{-1/2+\epsilon}$.
- The core theoretical result (Theorem 2) establishes the convergence rate: $d_{\mathcal{W}}^2(\hat{m}_G^{(0)}(x), m_G^{(0)}(x)) = O_p(n_0^{-1/2+\epsilon}(\psi + (n_0+n_\mathcal{A})^{-1/2}))$.

## Key Experimental Results

### Main Results (Simulation Study)

| Target sample size $n_0$ | Only Target RMSPR | Only Source RMSPR | WaTL RMSPR ($\tau=100$) | WaTL RMSPR ($\tau=200$) |
|---|---|---|---|---|
| 200 | ~0.30 | ~0.25 | ~0.15 | ~0.12 |
| 400 | ~0.18 | ~0.25 | ~0.12 | ~0.10 |
| 800 | ~0.12 | ~0.25 | ~0.09 | ~0.08 |

### Real Data Experiment (NHANES Physical Activity Data)

| Method | Female RMSPR | Male RMSPR |
|---|---|---|
| Only Target | Higher | Higher |
| WaTL | Significantly lower | Significantly lower |

| Configuration | Description |
|---|---|
| Source selection (AWaTL) | When $\psi > 0.6$, selection rate of sources 1 and 2 approaches 100% |
| Negative transfer threshold | Negative transfer occurs when $\psi_1 \geq 0.9$ |
| Effect of source sample size | WaTL performance improves consistently as $\tau$ increases from 100 to 200 |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| K=1, $\psi_1 < 0.9$ | WaTL outperforms Only Target | Transfer is beneficial when source domain is sufficiently similar |
| K=1, $\psi_1 \geq 0.9$ | Only Target performs better | Negative transfer occurs when source domain is too dissimilar |
| AWaTL, L=2 | Selection rate → 100% | Informative sources are correctly identified as non-informative source discrepancy grows |

### Key Findings

- WaTL yields the greatest advantage when target sample sizes are small, reducing RMSPR by approximately 50%.
- AWaTL achieves 100% correct identification of informative sources when inter-domain discrepancy $\psi > 0.6$.
- WaTL gains increase monotonically with source sample size, corroborating the theoretical convergence rate.

## Highlights & Insights

- **WaTL is the first work to extend transfer learning to Wasserstein space**, filling a theoretical gap in transfer learning for distributional data.
- The theoretical results are general: the convergence rate in Theorem 1 extends to other metric space outputs, including networks, positive definite matrices, and trees.
- The adaptive selection mechanism of AWaTL elegantly addresses the negative transfer problem.
- The identity $d_\mathcal{W}^2(\mu_1, \mu_2) = \int_0^1 (F_{\mu_1}^{-1}(u) - F_{\mu_2}^{-1}(u))^2 du$ is exploited to convert the Wasserstein metric into an $L^2$ metric for tractable computation.

## Limitations & Future Work

- The current framework handles only **univariate distributions**; multivariate extensions would require Sinkhorn or Sliced Wasserstein distances.
- The theoretical analysis assumes fully observed distributions, whereas in practice only finite samples are available.
- Regularization parameter selection relies on cross-validation, incurring non-trivial computational cost.
- Optimal weighting strategies for imbalanced source-domain sample sizes are not addressed.

## Related Work & Insights

- This work elegantly integrates Fréchet regression with the transfer learning framework, providing a paradigm for transfer learning in other non-Euclidean spaces.
- The proof techniques are grounded in empirical process theory, offering a technical reference for subsequent work.
- The framework has implications for the transfer of distributional features in multimodal learning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic study of transfer learning in Wasserstein space; the problem formulation is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Combines simulation and real data, though real-data experiments are relatively limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and clearly presented.
- Value: ⭐⭐⭐⭐ Opens a new direction for distributional data analysis.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[NeurIPS 2025\] Learning from Interval Targets](learning_from_interval_targets.md)
- [\[NeurIPS 2025\] Learning Parameterized Skills from Demonstrations](learning_parameterized_skills_from_demonstrations.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](learning_theory_for_kernel_bilevel_optimization.md)

<!-- RELATED:END -->
