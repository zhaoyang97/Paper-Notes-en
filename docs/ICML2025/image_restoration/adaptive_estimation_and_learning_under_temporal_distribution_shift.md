---
title: >-
  [Paper Note] Adaptive Estimation and Learning under Temporal Distribution Shift
description: >-
  [ICML 2025][Image Restoration][distribution_shift] Proposes an estimation algorithm based on wavelet soft-thresholding that achieves optimal pointwise estimation error bounds under temporal distribution shift without prior knowledge. It establishes a connection between sequence non-stationarity and sparsity in the wavelet domain, applying it to binary classification and total variation denoising under distribution shift.
tags:
  - "ICML 2025"
  - "Image Restoration"
  - "distribution_shift"
  - "wavelet_denoising"
  - "nonstationary_estimation"
  - "binary_classification"
  - "total_variation"
date: 2026-05-08
content_hash: 8e3d96624b9ad98d
---

# Adaptive Estimation and Learning under Temporal Distribution Shift

**Conference**: ICML 2025  
**arXiv**: [2505.15803](https://arxiv.org/abs/2505.15803)  
**Code**: Not released  
**Area**: Statistical Learning Theory / Distribution Shift  
**Keywords**: distribution_shift, wavelet_denoising, nonstationary_estimation, binary_classification, total_variation  

## TL;DR

Proposes an estimation algorithm based on wavelet soft-thresholding that achieves optimal pointwise estimation error bounds under temporal distribution shift without prior knowledge. It establishes a connection between sequence non-stationarity and sparsity in the wavelet domain, applying it to binary classification and total variation denoising under distribution shift.

## Background & Motivation

Traditional statistical estimation assumes that data is independent and identically distributed (i.i.d.). However, in practical scenarios such as stock price prediction, temperature forecasting, and network traffic monitoring, the data distribution often changes over time. This paper relaxes the "identical distribution" assumption and studies how to utilize historical observation sequences to estimate the true value at the latest time step under temporal distribution shift.

The core challenge lies in the fact that sliding window averaging is the most natural estimation method, but selecting the window size faces a classical bias-variance tradeoff—small windows have small bias but large variance, while large windows have small variance but large bias. More critically, the optimal window depends on the stationarity/smoothness of the true sequence, which is typically unknown in practice. Mazzetto & Upfal (2023) first addressed this problem, but their metric based on "local stability" can only capture piecewise constant structures, limiting its expressiveness for more complex trend patterns.

## Method

### Overall Architecture

This paper proposes using the classical wavelet soft-thresholding denoising algorithm to solve the estimation problem under temporal distribution shift (Algorithm 1). The core idea is to transform apparently non-stationary signals in the time domain into the wavelet domain, leveraging sparsity in the wavelet domain for efficient estimation.

Algorithm workflow: Given an observation sequence $y_n, \ldots, y_1$, (1) compute empirical wavelet coefficients $\tilde{\boldsymbol{\beta}} = \boldsymbol{W}\boldsymbol{y}$; (2) apply soft-thresholding to each coefficient $\hat{\beta}_i = \text{sign}(\tilde{\beta}_i) \max\{|\tilde{\beta}_i| - \lambda, 0\}$, with threshold $\lambda = 2\sigma\sqrt{2\log(\log n / \delta)}$; (3) perform the inverse wavelet transform to reconstruct the signal and retrieve the last coordinate as the estimator.

The fundamental difference from prior work is that previous methods maintain an adaptive window size to average relevant historical observations, whereas the wavelet approach does not require an explicit window and instead implicitly exploits the most relevant parts of the data.

### Key Designs

#### Key Design 1: General Estimation Error Bound Based on Wavelet Sparsity (Lemma 1)

Lemma 1 provides the core theoretical guarantee of the algorithm. For the observation model $y_i = \theta_i + \epsilon_i$ (where $\epsilon_i$ is i.i.d. $\sigma$-sub-Gaussian noise), let $\mathcal{I}$ be the index set of wavelet coefficients affecting the final reconstructed value $\theta_1$. Then with probability at least $1 - \delta$:

$$|\hat{\theta}_1 - \theta_1| \leq \sum_{i \in \mathcal{I}} 6|W_{i,n}| \cdot (|\beta_i| \wedge \lambda)$$

where $\beta_i$ is the true wavelet coefficient and $\lambda$ is the threshold parameter. The key insight of this bound is that the estimation error is determined by the sparsity of the true wavelet coefficients. When the true signal is sparse in the wavelet domain (i.e., most $\beta_i$ are small), each term $|\beta_i| \wedge \lambda$ takes the value $|\beta_i|$ instead of $\lambda$, substantially reducing the error.

#### Key Design 2: Variational Bound and Optimality Under Haar Wavelets (Theorem 2)

When utilizing the Haar wavelet system, Theorem 2 proves that the algorithm recovers the variational bound of Mazzetto & Upfal (2023):

$$|\hat{\theta}_1 - \theta_1| \leq \kappa \cdot U(r^*)$$

where $U(r) = \max_{t \in S(r)} |\bar{\theta}_{t:1} - \theta_1| \vee \sigma/\sqrt{r}$, $r^*$ is the optimal time point minimizing $U(r)$, and $\kappa = (4\sqrt{2\log(\log n/\delta)} \vee 2\sqrt{2})(\log_2 n + 1)$.

$U(r)$ characterizes a bias-variance tradeoff: $|\bar{\theta}_{t:1} - \theta_1|$ is the bias term (historical mean deviating from the current value), and $\sigma/\sqrt{r}$ is the variance term. The algorithm automatically finds the optimal $r^*$ without requiring prior knowledge of the non-stationarity level.

#### Key Design 3: ERM Algorithm for Binary Classification Under Distribution Shift (Theorem 9)

The wavelet estimator is applied to control the excess risk in binary classification. For each function $f$ in the hypothesis class $\mathcal{F}$, Algorithm 1 estimates its loss on the latest distribution, followed by Empirical Risk Minimization (ERM). The excess risk bound is:

$$L_{\hat{f}} - L_{f_*} \leq 8\sqrt{\frac{2d\log(2n)}{n}} + 2\sqrt{\frac{2\log(3/\delta)}{n}} + \sqrt{3} \cdot \sup_{f \in \mathcal{F}} \sum_{i \in \mathcal{I}} 6|W_{i,n}| \cdot (|\beta_i^f| \wedge \lambda')$$

Key advantage: It requires only a single ERM invocation (whereas prior methods required $O(\log n)$), and the objective function remains differentiable with respect to differentiable surrogate losses, enabling gradient descent optimization.

#### Key Design 4: Connection to Total Variation Denoising (Theorem 11)

It is proved that any algorithm satisfying a bound of the form in Theorem 2 or Corollary 3 automatically achieves the optimal estimation rate for TV denoising when run iteratively:

$$R_{\text{sq}} = \tilde{O}(n^{1/3} C^{2/3} \sigma^{4/3}), \quad R_{\text{abs}} = \tilde{O}(n^{2/3} C^{1/3} \sigma^{2/3})$$

This reveals that the algorithms of Mazzetto & Upfal (2023) and Han et al. (2024) are also optimal for TV denoising, a fact that was previously undiscovered.

## Key Experimental Results

### Table 1: MSE of Different Algorithms Under Known Noise Standard Deviation (Synthetic Data, Doppler Signal)

| Noise Level | AVG | HAAR (Ours) | DB8 (Ours) |
|:--------:|:---:|:----------:|:---------:|
| 0.2 | 0.511 | 0.068 | **0.007** |
| 0.5 | 0.509 | 0.155 | **0.032** |
| 1.0 | 0.508 | 0.155 | **0.129** |

> DB8 reduces MSE by more than an order of magnitude on the Doppler signal (from 0.511 to 0.007 at noise level 0.2).

### Table 2: MSE of Different Algorithms Under Unknown Noise Standard Deviation (Synthetic Data, Doppler Signal)

| Noise Level | AVG | ARW | Aligator | HAAR (Ours) | DB8 (Ours) |
|:--------:|:---:|:---:|:--------:|:----------:|:---------:|
| 0.2 | 0.512 | 0.384 | 0.234 | 0.053 | **0.020** |
| 0.5 | 0.512 | 0.400 | 0.235 | 0.065 | **0.044** |
| 1.0 | 0.514 | 0.407 | 0.235 | **0.088** | 0.129 |

> Even without knowing the noise level, the wavelet methods still significantly outperform the baselines. HAAR outperforms DB8 under high noise because higher-order wavelets exhibit larger variance when noise is substantial.

### Table 3: Model Selection Experiment on Real-world Data (Dubai Real Estate Dataset)

| Method | 79%-1% Split | 75%-5% Split |
|:----:|:----------:|:----------:|
| ARW | 0.0790 | **0.0719** |
| HAAR (Ours) | **0.0722** | 0.0736 |
| DB8 (Ours) | 0.0762 | 0.0768 |

> When validation data is scarce (1% validation split), HAAR outperforms ARW; in highly non-stationary real-world data, the lower-order Haar wavelet is remarkably more suitable.

## Key Findings

1. **Equivalence between Non-stationarity and Wavelet Sparsity**: The temporal non-stationarity level of a sequence directly corresponds to the sparsity of its wavelet coefficients, a connection more general than the concept of "local stability".
2. **Adaptivity**: The algorithm does not require any prior knowledge about the extent of the distribution shift and automatically adapts to the non-stationarity level of the sequence.
3. **Computational Efficiency Improvement**: In binary classification scenarios, only 1 ERM invocation is needed (vs. $O(\log n)$ in prior work), at the sole cost of inflating the excess risk bound by an $O(\log n)$ factor.
4. **New Optimal Algorithms for TV Denoising**: Algorithms satisfying pointwise error bounds automatically become optimal for TV denoising, an implication previously unrecognized.
5. **Advantages and Limitations of Higher-order Wavelets**: DB8 performs vastly better than HAAR on smooth trends, but HAAR is more robust in highly non-stationary/high-noise scenarios.

## Highlights & Insights

- **The Power of Perspective Shift**: The transition from "selecting the optimal window" to "denoising in the wavelet domain" translates a challenging adaptive problem into classical wavelet theory, demonstrating the profound value of signal processing tools in statistical learning theory.
- **Consistency between Theory and Practice**: The wavelet sparsity advantage predicted by Lemma 1 is clearly verified in experiments—the Doppler signal is truly sparse in the DB8 wavelet domain, where the MSE of AVG is 0.51 while DB8 achieves 0.007.
- **Differentiability Maintenance**: The soft-thresholding-based ERM objective maintains differentiability, which is crucial for practical applications in the deep learning era.
- **Wavelet Selection Analogy to Kernel Selection**: The authors analogize the choice of wavelet basis to the choice of kernel functions in Gaussian processes, both of which require guidance from prior knowledge of the application scenario.

## Limitations & Future Work

1. **Wavelet Basis Selection** lacks a principled method, currently relying on prior intuition regarding data trends.
2. **Noise Assumptions** require i.i.d. sub-Gaussian noise, whereas noise in real-world scenarios may be heteroscedastic or correlated.
3. Lacks a **concrete construction** showing Lemma 1 is strictly superior to Theorem 2, providing only empirical comparisons.
4. Higher-order wavelets introduce numerical instability and larger variance, potentially failing in **small-sample, high-noise** scenarios.
5. Does not address how to extend the method to related problems such as **change-point detection**.

## Related Work & Insights

- **Mazzetto & Upfal (2023)**: First to address this open problem; based on local stability metrics, it is the direct target of improvement in this paper.
- **Han et al. (2024) (ARW)**: Another adaptive method that does not require prior knowledge of the noise standard deviation.
- **Donoho (1995)**: Classical work on wavelet soft-thresholding; this paper migrates it from signal denoising to distribution shift estimation.
- **Baby et al. (2021) (Aligator)**: Achieves optimal MSE rates for TV denoising via online learning, but lacks pointwise error guarantees.
- **Insights**: Classical tools from signal processing (wavelet transforms, soft-thresholding) still hold profound application value in modern machine learning theory problems.

## Rating

| Dimension | Rating (1-5) | Explanation |
|:----:|:---------:|:----:|
| Theoretical Depth | 5 | Multi-layered theoretical system: general bounds, Haar bounds, classification applications, and TV denoising connections. |
| Novelty | 4 | Novel perspective, though the core tool (wavelet thresholding) is classical; the innovation lies in establishing the connections. |
| Experimental Thoroughness | 4 | Validated on both synthetic and real-world data, though the real-world dataset scale is relatively small. |
| Practicality | 3 | Beautiful theoretical results, but practical guidance on wavelet selection is limited. |
| Writing Quality | 4 | Clear logic and well-defined contributions, though notations are somewhat heavy. |

**Overall Score**: 4.0/5 — A theoretically solid work with a unique perspective, successfully introducing wavelet theory to the distribution shift estimation problem and revealing key cross-domain connections.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RegionFuse: Region-Adaptive Pixel Distribution Learning for Infrared and Visible Image Fusion](../../CVPR2026/image_restoration/regionfuse_region-adaptive_pixel_distribution_learning_for_infrared_and_visible_.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](../../ICCV2025/image_restoration/learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[NeurIPS 2025\] Adaptive Discretization for Consistency Models](../../NeurIPS2025/image_restoration/adaptive_discretization_for_consistency_models.md)
- [\[NeurIPS 2025\] MAP Estimation with Denoisers: Convergence Rates and Guarantees](../../NeurIPS2025/image_restoration/map_estimation_with_denoisers_convergence_rates_and_guarantees.md)
- [\[NeurIPS 2025\] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](../../NeurIPS2025/image_restoration/improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)

</div>

<!-- RELATED:END -->
