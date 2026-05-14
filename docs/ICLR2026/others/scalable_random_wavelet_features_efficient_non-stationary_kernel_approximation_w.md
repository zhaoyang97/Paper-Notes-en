---
title: >-
  [Paper Note] Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees
description: >-
  This paper proposes Random Wavelet Features (RWF), a scalable non-stationary kernel approximation framework constructed by randomly sampling from a family of wavelets. RWF preserves the linear-time complexity of random…
tags:

date: 2026-05-08
content_hash: d35305372565ae5b
---

# Scalable Random Wavelet Features: Efficient Non-Stationary Kernel Approximation with Convergence Guarantees

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2602.00987](https://arxiv.org/abs/2602.00987)
- **Code**: Not released
- **Area**: Others
- **Keywords**: kernel methods, random features, wavelets, non-stationary, Gaussian processes, multi-resolution

## TL;DR
This paper proposes Random Wavelet Features (RWF), a scalable non-stationary kernel approximation framework constructed by randomly sampling from a family of wavelets. RWF preserves the linear-time complexity of random feature methods while offering guarantees of positive definiteness, unbiasedness, and uniform convergence.

## Background & Motivation
- **The GP dilemma**: expressiveness vs. efficiency. Exact GP inference requires $O(N^3)$ computation and typically relies on stationary kernels.
- **Random Fourier Features (RFF)**: based on Bochner's theorem, RFF approximates stationary kernels as linear models with $O(ND^2)$ training cost, but is inherently restricted to stationary kernels.
- **Non-stationary modeling challenges**: domains such as geospatial analysis and speech exhibit statistical properties that vary with location. Methods like Deep GPs and spectral mixture kernels can capture non-stationarity but at high computational cost.
- **Core gap**: there is no random feature framework that is simultaneously as scalable as RFF and natively capable of capturing non-stationarity.

## Method

### Core Idea
Wavelet basis functions are used in place of Fourier basis functions. Wavelets are inherently localized in both space and frequency; by randomly sampling wavelet atoms at different scales and translations, a non-stationary kernel is constructed.

### Wavelet Kernel Construction
A family of atoms is generated from a mother wavelet $\psi: \mathbb{R}^d \to \mathbb{R}$:
$$\psi_{s,\mathbf{t}}(\mathbf{x}) = s^{-d/2} \psi\left(\frac{\mathbf{x} - \mathbf{t}}{s}\right)$$

The non-stationary kernel is obtained via integration:
$$k(\mathbf{x}, \mathbf{y}) = \int_0^\infty \int_{\mathbb{R}^d} \psi_{s,\mathbf{t}}(\mathbf{x}) \psi_{s,\mathbf{t}}(\mathbf{y}) \, p(s, \mathbf{t}) \, d\mathbf{t} \, ds$$

where $p(s, \mathbf{t})$ is the sampling density over scale and translation.

### Random Wavelet Features
The above integral is approximated via Monte Carlo:

$$z(\mathbf{x}) = \frac{1}{\sqrt{D}} [\psi_{s_1,\mathbf{t}_1}(\mathbf{x}), \ldots, \psi_{s_D,\mathbf{t}_D}(\mathbf{x})]^\top$$

Kernel approximation: $\hat{k}(\mathbf{x}, \mathbf{y}) = z(\mathbf{x})^\top z(\mathbf{y})$

This is then cast as Bayesian linear regression:
$$\mathbf{S}_{\mathbf{w}} = (\mathbf{I}_D + \sigma^{-2} \mathbf{Z}^\top \mathbf{Z})^{-1}, \quad \mathbf{m}_{\mathbf{w}} = \sigma^{-2} \mathbf{S}_{\mathbf{w}} \mathbf{Z}^\top \mathbf{y}$$

### Sampling Strategy
- Scale $s$: log-uniform distribution (to cover multiple resolutions)
- Translation $\mathbf{t}$: uniform coverage over the convex hull of the data
- Mother wavelet: Morlet (time-frequency analysis) or Daubechies (sharp transitions)

### Computational Complexity

| Method | Training Complexity | Prediction Complexity |
|------|-----------|-----------|
| Exact GP | $O(N^3)$ | $O(N^2)$ |
| SVGP | $O(NM^2)$ / step (iterative) | $O(M^2)$ |
| RFF-GP | $O(ND^2)$ (one-shot) | $O(D^2)$ |
| **RWF-GP** | $O(ND^2)$ (one-shot) | $O(D^2)$ |

> RWF achieves the same efficiency as RFF while enabling non-stationary modeling.

## Theoretical Guarantees

### Theorem 4.1 (Positive Definiteness)
The kernel $k(\mathbf{x}, \mathbf{y})$ constructed by RWF is a positive definite kernel for any non-negative measure $p(s, \mathbf{t})$.

### Lemma 4.1 (Unbiasedness)
$\mathbb{E}[\hat{k}(\mathbf{x}, \mathbf{y})] = k(\mathbf{x}, \mathbf{y})$ for all $\mathbf{x}, \mathbf{y} \in \mathcal{X}$.

### Uniform Convergence Guarantee
$$\Pr\left[\sup_{\mathbf{x}, \mathbf{y} \in \mathcal{X}} |\hat{k}(\mathbf{x}, \mathbf{y}) - k(\mathbf{x}, \mathbf{y})| > \epsilon\right] \leq \mathcal{O}\left(\exp(-D\epsilon^2 / B^4)\right)$$

That is, $D = O(B^4 / \epsilon^2)$ features suffice to guarantee a uniform $\epsilon$-accurate approximation.

## Key Experimental Results

### Main Results: Regression Tasks

| Method | Synthetic Non-stationary | Speech Data | Large-scale Regression |
|------|-----------|---------|-----------|
| RFF-GP | Underfits local structure | High RMSE | Fast but inaccurate |
| SVGP | Good | Good | Moderate speed and accuracy |
| Deep GP | Best | Best | Slow but accurate |
| **RWF-GP** | **Close to Deep GP** | **Close to Deep GP** | **Fast and accurate** |

> RWF-GP achieves accuracy close to Deep GP while maintaining speed close to RFF-GP.

### Ablation Study: Effect of Number of Features

| Number of features $D$ | RFF RMSE | RWF RMSE |
|-----------|----------|----------|
| 50 | 0.85 | 0.62 |
| 100 | 0.78 | 0.45 |
| 500 | 0.72 | 0.31 |
| 1000 | 0.70 | 0.28 |

> RWF consistently outperforms RFF under the same number of features, with greater improvement as $D$ increases due to the accumulated advantage of wavelet localization.

### Key Findings
1. RWF significantly outperforms RFF on non-stationary signals without additional computational cost.
2. The multi-resolution structure allows fine-scale wavelets to capture local events while coarse-scale wavelets model long-range trends.
3. Different wavelet families suit different data characteristics (Morlet for oscillatory signals, Mexican Hat for impulse detection).
4. RWF-GP approaches Deep GP accuracy while being an order of magnitude faster.

## Highlights & Insights
- **Filling a theoretical gap**: this work provides the first complete theoretical chain—positive definiteness → unbiasedness → uniform convergence—for wavelet-based random features.
- **Conceptual elegance**: RFF uses global sinusoidal bases → stationary kernels; RWF uses local wavelet bases → non-stationary kernels, forming a natural generalization.
- **Practical efficiency**: non-stationary modeling capability is gained while retaining $O(ND^2)$ training complexity.
- **Multi-resolution flexibility**: the sampling distribution $p(s, \mathbf{t})$ can be tuned to accommodate diverse data characteristics.

## Limitations & Future Work
- The choice of mother wavelet and sampling distribution still requires hyperparameter tuning.
- Isotropic scaling may be limiting for high-dimensional anisotropic problems.
- Fixed wavelet families may lack flexibility for certain non-stationary patterns such as abrupt change points.
- Expressive power remains theoretically bounded compared to end-to-end learnable Deep GPs.

## Related Work & Insights
- **Random Fourier Features**: the seminal work of Rahimi & Recht (2007); subsequently extended to adaptive and structured sampling.
- **Wavelet kernels**: Zhang et al. (2004) on wavelet SVMs; Guo et al. (2024) on fixed wavelet basis Bayesian regression.
- **Non-stationary GPs**: Deep GP (Damianou et al., 2013), spectral mixture kernels (Wilson & Adams, 2013).
- **Scalable GPs**: KISS-GP (Wilson & Nickisch, 2015), Deep Kernel Learning (Wilson et al., 2016).

## Rating
- **Novelty**: ⭐⭐⭐⭐ — A natural combination of wavelets and random features, though not a conceptual breakthrough.
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Full coverage of positive definiteness, unbiasedness, variance bounds, and uniform convergence.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers synthetic, speech, and large-scale regression settings, though broader real-world applications are lacking.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to scenarios requiring non-stationary modeling with scalability constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](../../AAAI2026/others/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[ICLR 2026\] Consistent Low-Rank Approximation](consistent_low-rank_approximation.md)
- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[ICLR 2026\] Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)

</div>

<!-- RELATED:END -->
