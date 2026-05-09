---
title: >-
  [Paper Note] Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry
description: >-
  [ICLR 2026][SPD manifold] This paper reveals a simple product structure on the Cholesky manifold and, building upon it, proposes two fast and numerically stable SPD metrics (PCM and BWCM) with closed-form expressions for all Riemannian operators, achieving simultaneous improvements in accuracy, efficiency, and numerical stability for SPD deep learning.
tags:
  - ICLR 2026
  - SPD manifold
  - Riemannian metric
  - Cholesky decomposition
  - product geometry
  - SPD neural networks
date: 2026-05-08
content_hash: 697be345485df979
---

# Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry

**Conference**: ICLR 2026
**arXiv**: [2407.02607](https://arxiv.org/abs/2407.02607)
**Code**: [github.com/GitZH-Chen/PCM_BWCM](https://github.com/GitZH-Chen/PCM_BWCM)
**Area**: Other
**Keywords**: SPD manifold, Riemannian metric, Cholesky decomposition, product geometry, SPD neural networks

## TL;DR

This paper reveals a simple product structure on the Cholesky manifold and, building upon it, proposes two fast and numerically stable SPD metrics (PCM and BWCM) with closed-form expressions for all Riemannian operators, achieving simultaneous improvements in accuracy, efficiency, and numerical stability for SPD deep learning.

## Background & Motivation

### SPD Matrix Learning

Symmetric positive definite (SPD) matrices are widely used in medical imaging, EEG analysis, signal processing, and computer vision. SPD matrices form a non-Euclidean manifold $\mathcal{S}_{++}^n$, where standard Euclidean methods are inapplicable, and Riemannian metrics are required to define fundamental operations such as distances, geodesics, and logarithmic/exponential maps.

### Existing SPD Metrics

The predominant metrics in current literature include:
- **AIM** (Affine-Invariant Metric): theoretically well-founded but computationally expensive (requires SVD), $O(n^3)$ complexity
- **LEM** (Log-Euclidean Metric): requires matrix logarithm, numerically unstable
- **PEM** (Power-Euclidean Metric): requires matrix power, relatively flexible
- **LCM** (Log-Cholesky Metric): based on Cholesky decomposition, computationally fast and stable, currently a common choice in practice
- **BWM** (Bures-Wasserstein Metric): derived from optimal transport, some operators lack closed-form expressions
- **GBWM** (Generalized BWM): a generalization of BWM

### Advantages and Limitations of LCM

LCM transforms SPD operations into lower-triangular matrix operations via Cholesky decomposition, offering closed-form operators, high efficiency, and numerical stability. However, LCM applies the **logarithmic map** (log/exp) to the diagonal part, which can cause numerical overflow (e.g., $\log(10^{-15})$) or excessive stretching when diagonal entries are very small.

### Core Insight of This Paper

The Cholesky metric underlying LCM (diagonal log metric) actually admits a **product structure**: the strictly lower-triangular part uses the Euclidean metric, while the diagonal part is a product of $n$ Riemannian metrics on $\mathbb{R}_{++}$. This implies that **replacing the metric on $\mathbb{R}_{++}$** immediately yields new Cholesky metrics and, by pullback, new SPD metrics.

## Method

### Overall Architecture

```
SPD manifold → Cholesky decomposition → Cholesky manifold = Strictly lower-triangular (Euclidean) × Diagonal (ℝ₊₊ⁿ)
                                                                              ↓
                                                              Replace metric on ℝ₊₊
                                                                              ↓
                                                          θ-DPM (power metric) / M-DBWM (BW metric)
                                                                              ↓
                                                          Pullback to SPD manifold → θ-PCM / (θ,M)-BWCM
```

### Key Designs

#### 1. Revealing the Product Structure

The Cholesky manifold $\mathcal{L}_{++}^n$ decomposes as:
$$\{\mathcal{L}_{++}^n, g^{\text{DL}}\} = \{\mathcal{SL}^n, g^E\} \times \underbrace{\{\mathbb{R}_{++}, g^{\mathbb{R}_{++}}\} \times \cdots \times \{\mathbb{R}_{++}, g^{\mathbb{R}_{++}}\}}_{n}$$

where $\mathcal{SL}^n$ is the space of strictly lower-triangular matrices (Euclidean), and $\mathbb{R}_{++}$ corresponds to each diagonal entry. The diagonal metric of LCM takes the form $g_p(v,w) = p^{-2}vw$, which is a unified representation of AIM/LEM/LCM on $\mathcal{S}_{++}^1$.

#### 2. Two Novel Cholesky Metrics

**θ-DPM (Diagonal Power Metric)**: replaces the metric on $\mathbb{R}_{++}$ with the power-Euclidean metric ($\theta$-EM)

$$g_L^{\theta\text{-DE}}(X,Y) = \langle \lfloor X \rfloor, \lfloor Y \rfloor \rangle + \langle \mathbb{L}^{\theta-1}\mathbb{X}, \mathbb{L}^{\theta-1}\mathbb{Y} \rangle$$

**M-DBWM (Diagonal Bures-Wasserstein Metric)**: replaces the metric on $\mathbb{R}_{++}$ with the Bures-Wasserstein metric

$$g_L^{\mathbb{M}\text{-DBW}}(X,Y) = \langle \lfloor X \rfloor, \lfloor Y \rfloor \rangle + \frac{1}{4}\langle \mathbb{L}^{-1}\mathbb{X}, \mathbb{M}^{-1}\mathbb{Y} \rangle$$

#### 3. Fully Closed-Form Riemannian Operators

Both proposed metrics admit closed-form geodesics, logarithmic maps, exponential maps, parallel transport, distances, and weighted Fréchet means. For example, the distance under θ-DPM is:

$$d^2(L,K) = \|\lfloor K \rfloor - \lfloor L \rfloor\|_F^2 + \frac{1}{\theta^2}\|\mathbb{K}^\theta - \mathbb{L}^\theta\|_F^2$$

The key distinction: LCM uses $\log(\mathbb{K}) - \log(\mathbb{L})$, whereas θ-DPM uses $\mathbb{K}^\theta - \mathbb{L}^\theta$—**power functions replace logarithmic/exponential functions**, which is the source of numerical stability.

#### 4. Diagonal Power Deformation

A diagonal power deformation $\text{DPow}_\theta$ is defined to continuously interpolate between existing and proposed metrics:
- $\theta \to 0$: the deformed metric converges to the Log-Cholesky metric (LCM)
- $\theta = 1$: recovers the metrics proposed in this paper

This provides an adjustable trade-off parameter.

#### 5. Gyrovector Space Structure

Closed-form expressions for gyroaddition and gyroscalar multiplication are defined under the new metrics:

$$L \oplus K = \lfloor L \rfloor + \lfloor K \rfloor + (\mathbb{L}^\beta + \mathbb{K}^\beta - I)^{1/\beta}$$

All axioms of gyrocommutative groups and gyrovector spaces are satisfied, providing an algebraic foundation for constructing SPD neural networks.

### Loss & Training

The proposed metrics are applied to two types of SPD network components:

**SPD MLR Classifier** (a Riemannian generalization of point-to-hyperplane distance):

$$p(y=k|S) \propto \exp\left[\langle \lfloor K \rfloor - \lfloor L_k \rfloor, \lfloor A_k \rfloor \rangle + \frac{1}{2\theta}\langle \mathbb{K}^\theta - \mathbb{L}_k^\theta, \mathbb{A}_k \rangle\right]$$

**SPD Residual Block** (a generalization based on the Riemannian exponential map):
$$Y = \text{Exp}_X(Q \cdot \text{diag}(f(\text{spec}(X))) \cdot Q^T)$$

## Key Experimental Results

### Main Results

**Table 2: SPDNet + SPD MLR Classifier**

| Metric | Radar (Acc/Time) | HDM05 3-Block (Acc/Time) | FPHA (Acc/Time) |
|------|:-:|:-:|:-:|
| AIM | 94.53 / 0.80s | 61.14 / 19.23s | 85.57 / 7.14s |
| LEM | 93.55 / 0.76s | 60.28 / 3.50s | 85.90 / 0.98s |
| LCM | 93.49 / 0.72s | 62.33 / 2.90s | 86.37 / 0.74s |
| **θ-PCM** | **95.79 / 0.72s** | **65.75 / 2.76s** | **89.40 / 0.69s** |
| θ-BWCM | 93.93 / 0.71s | 67.40 / 2.87s | 86.27 / 0.70s |

**Table 3: GyroSPD Backbone**

| Metric | Radar | HDM05 | FPHA |
|------|:-:|:-:|:-:|
| LCM | 96.29 | 68.37 | 89.83 |
| **θ-PCM** | **97.04** | **71.93** | **91.17** |
| **θ-BWCM** | 96.21 | **72.74** | **91.00** |

On HDM05 (action recognition), θ-BWCM surpasses LCM by +5.1% accuracy (and +4.37% under the GyroSPD backbone).

### Ablation Study

**Numerical Stability: Small Eigenvalue Test (Table 5)**

| $\epsilon$ (minimum eigenvalue) | DLM failure rate | θ-DPM failure rate | θ-DBWM failure rate |
|:-:|:-:|:-:|:-:|
| $10^{-1}$ | 0.62% | **0%** | **0%** |
| $10^{-3}$ | 51.32% | **0%** | **0%** |
| $10^{-5}$ | 99.39% | **0%** | **0%** |
| $10^{-10}$ | 100% | **0%** | **0%** |
| $10^{-20}$ | 100% | **0%** | **0%** |

The Log-Cholesky metric (DLM/LCM) fails nearly 100% of the time under small eigenvalues (producing Inf/NaN), whereas the proposed metrics exhibit **zero failure** across all tested ranges.

**Ablation on Deformation Parameter $\theta$**: Sweeping $\theta$ from $-2$ to $1.5$ reveals a pronounced optimal value on HDM05 (a dataset with highly imbalanced Cholesky diagonal entries), while the impact is smaller on Radar/FPHA (datasets with more balanced diagonal entries).

### Key Findings

1. θ-PCM and θ-BWCM generally surpass LCM in accuracy, despite sharing the same Cholesky product structure origin.
2. The computational speed of the proposed metrics is comparable to LCM (far faster than AIM by a factor of 10–25×), with greater advantages at higher dimensions ($256 \times 256$).
3. In residual block experiments (Table 4), θ-PCM achieves the best accuracy across all datasets.
4. Numerical stability is a decisive advantage—zero failure rate across all eigenvalue ranges.

## Highlights & Insights

1. **Revealing the product structure**: seemingly simple yet highly instructive—it reduces the metric design problem to the choice of a metric on $\mathbb{R}_{++}$.
2. **Power functions replacing logarithms**: the core numerical insight—$x^\theta$ behaves far more mildly than $\log(x)$ as $x \to 0^+$.
3. **Theoretical completeness**: closed-form expressions for all Riemannian operators are provided, along with verification of gyrovector space axioms and continuity of the deformation.
4. **Strong practicality**: the proposed metrics can be directly plugged into existing SPD network frameworks (SPDNet, GyroSPD, RResNet) without architectural modifications.

## Limitations & Future Work

1. Experiments are limited to small-to-medium-scale SPD matrices ($n \leq 93$); performance at very large scales (e.g., $n > 1000$) remains to be validated.
2. Only classification tasks are considered; other SPD learning tasks such as regression and generation are not addressed.
3. The selection of $\theta$ and $\mathbb{M}$ currently relies on grid search; theoretical guidance for optimal selection is lacking.
4. The product structure assumes the standard Euclidean metric on the strictly lower-triangular part; whether more flexible metrics can be adopted there remains an open question.
5. Comparison with BWM on the full SPD manifold is not entirely fair, as BWM does not rely on Cholesky decomposition.

## Related Work & Insights

- **LCM** (Lin, 2019): the direct foundation of this work; this paper reveals the product structure underlying its metric.
- **GyroSPD** (Nguyen & Yang, 2023): provides the gyrovector space framework; this paper extends its algebraic structure.
- **SPD ResNet** (Katsman et al., 2024): provides the residual block framework; this paper directly adapts the proposed metrics to it.
- **Thanwerdas & Pennec (2022)**: theoretical framework for deformation metrics on SPD manifolds; this paper realizes analogous ideas at the Cholesky level.
- Inspiration: **identifying product structures on arbitrary manifolds** may be a general strategy for designing efficient metrics.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ★★★★☆ |
| Technical Depth | ★★★★★ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★★ |
| Value | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](../../NeurIPS2025/others/finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[ICLR 2026\] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)
- [\[ICLR 2026\] Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields](refine_now_query_fast_a_decoupled_refinement_paradigm_for_implicit_neural_fields.md)

</div>

<!-- RELATED:END -->
