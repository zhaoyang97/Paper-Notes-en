---
title: >-
  [Paper Note] Tilted Sharpness-Aware Minimization
description: >-
  [ICML2025][Optimization][SAM] This paper proposes Tilted SAM (TSAM), which utilizes exponential tilting to smooth the min-max objective of SAM into a soft optimization that weights multiple local solutions within a neighborhood by their loss values. Theoretically, TSAM is smoother and exhibits a stronger preference for flat minima. Empirically, it consistently outperforms SAM and its variants across both vision and language tasks.
tags:
  - "ICML2025"
  - "Optimization"
  - "SAM"
  - "exponential tilting"
  - "flat minima"
  - "generalization"
  - "Hamiltonian Monte Carlo"
date: 2026-05-08
content_hash: be4b27136ca7ac11
---

# Tilted Sharpness-Aware Minimization

**Conference**: ICML2025  
**arXiv**: [2410.22656](https://arxiv.org/abs/2410.22656)  
**Code**: [github.com/litian96/TSAM](https://github.com/litian96/TSAM)  
**Area**: Optimization / Sharpness-Aware Minimization  
**Keywords**: SAM, exponential tilting, flat minima, generalization, Hamiltonian Monte Carlo

## TL;DR

This paper proposes Tilted SAM (TSAM), which utilizes exponential tilting to smooth the min-max objective of SAM into a soft optimization that weights multiple local solutions within a neighborhood by their loss values. Theoretically, TSAM is smoother and exhibits a stronger preference for flat minima. Empirically, it consistently outperforms SAM and its variants across both vision and language tasks.

## Background & Motivation

SAM searches for flat minima by optimizing the worst-case loss within a neighborhood, thereby improving the generalization performance of overparameterized models. Its standard min-max objective is:

$$\min_{\theta} L^s(\theta) := \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$$

However, SAM suffers from two core issues:

**Optimization difficulties**: On highly non-convex loss landscapes, a single-step gradient ascent struggle to find the true maximizer perturbation $\epsilon$.

**Information waste**: Focusing solely on the single worst-case local solution ignores other directions in the neighborhood that also yield high losses, potentially leaving the loss landscape near the local minimum sharp.

The authors show that even when SAM finds a solution with low worst-case loss, the **average loss** over its neighborhood remains higher than that of TSAM's solution, suggesting that optimizing only the worst-case scenario is suboptimal.

## Method

### Core Objective: Tilted SAM

TSAM introduces a tilting parameter $t \geq 0$ and performs an exponentially weighted aggregation of the neighborhood losses using the LogSumExp operator:

$$\min_{\theta} L^t(\theta) := \frac{1}{t} \log \left( \mathbb{E}_{\mu(\epsilon)} \left[ e^{t L(\theta + \epsilon)} \right] \right)$$

where $\mu(\epsilon)$ denotes the probability distribution of the perturbation (e.g., a uniform ball $\|\epsilon\| \leq \rho$). TSAM unifies three optimization paradigms:

| Parameter $t$ | Degenerate Form | Meaning |
|:---:|:---:|:---|
| $t \to \infty$ | SAM | Focuses solely on the worst-case perturbation |
| $t = 0$ | Average perturbation loss | Uniformly weights all directions in the neighborhood |
| $0 < t < \infty$ | **TSAM** | Softly weighted by loss values, with higher weight assigned to high-loss directions |

### Tilted Weights and Gradients

The $t$-tilted weight is defined as:

$$w^t(\theta + \epsilon) := \frac{e^{t L(\theta + \epsilon)}}{\mathbb{E}[e^{t L(\theta + \epsilon)}]}$$

The gradient of TSAM is the weighted average of the neighborhood gradients using the tilted weights:

$$\nabla L^t(\theta) = \mathbb{E}\left[ w^t(\theta + \epsilon) \nabla L(\theta + \epsilon) \right]$$

This serves as a **softened version** of SAM (which assigns all weight to a single worst-case direction).

### Theoretical Properties

1. **Smoothness** (Lemma 3.3): The smoothness parameter of TSAM is $\beta(t) = O(t)$, which is bounded for finite $t$; whereas the smoothness parameter of SAM ($t \to \infty$) is unbounded. Thus, TSAM is easier to optimize.
2. **Preference for Flatness** (Theorem 3.6): For GLM-type losses, if $\theta_1$ is sharper than $\theta_2$, then $L^t(\theta_1) - L^t(\theta_2)$ is monotonically increasing with respect to $t$—the larger $t$ is, the more TSAM "penalizes" sharp solutions.
3. **Small $t$ Approximation**: For small $t$, $L^t(\theta) \approx \mathbb{E}[L(\theta+\epsilon)] + \frac{t}{2} \mathrm{var}(L(\theta+\epsilon))$, indicating joint minimization of both the mean and the variance of the neighborhood loss.
4. **Generalization Bound** (Theorem 3.7): There exists a finite optimal $t^*$ that yields the tightest generalization upper bound.

### Optimization Algorithm: HMC-Based Sampling

The key challenge of TSAM is to efficiently sample $\epsilon$ from the distribution $p(\epsilon) \propto e^{\delta L(\theta + \epsilon)}$. The authors employ an Euler discretization based on Hamiltonian Monte Carlo:

1. Randomly initialize $s$ perturbations $\epsilon_j$ and momenta $p_j$.
2. Perform a single-step gradient ascent for each sample: $\epsilon \leftarrow \epsilon + \beta' \nabla L(\theta + \epsilon)$.
3. Aggregate gradients using tilted weights to update the model:

$$\theta^{i+1} \leftarrow \theta^i - \eta \frac{\sum_{j} e^{(t-\delta) L(\theta^i + \epsilon_j)} \nabla L(\theta^i + \epsilon_j)}{\sum_{j} e^{(t-\delta) L(\theta^i + \epsilon_j)}}$$

In practice, taking only $s = 3 \sim 5$ samples is sufficient to obtain significant improvements.

## Key Experimental Results

### Image Classification (Test Accuracy %)

| Method | CIFAR100 (ResNet18) | CIFAR100 (WRN) | DTD (ViT) | Noisy CIFAR100 (ResNet18) | TinyImageNet (ResNet18) |
|:---|:---:|:---:|:---:|:---:|:---:|
| ERM | 71.39 | 73.22 | 66.38 | 61.01 | 71.10 |
| SAM | 76.52 | 78.44 | 67.87 | 69.00 | 72.43 |
| ESAM1 | 77.40 | 80.22 | 68.18 | 69.20 | 73.24 |
| RSAM | 77.35 | 79.02 | 68.35 | 69.31 | 73.57 |
| **TSAM** | **77.78** | **80.85** | **68.82** | **69.98** | **73.55** |

### GLUE Benchmark (DistilBERT Fine-tuning)

| Method | CoLA | SST-2 | MNLI | QNLI | AVG |
|:---|:---:|:---:|:---:|:---:|:---:|
| ERM | 80.34 | 90.48 | 79.6 | 87.72 | 77.15 |
| SAM | 80.48 | 91.74 | 81.1 | 86.42 | 77.56 |
| **TSAM** | **80.81** | **91.86** | 81.1 | **87.81** | **78.01** |

### Flatness Verification (CIFAR100, ResNet18 Top-5 Hessian Eigenvalues)

| Method | Top-5 Eigenvalues |
|:---|:---|
| ERM | 342, 305, 261, 253, 211 |
| SAM | 233, 198, 183, 154, 146 |
| **TSAM** (t=20) | **141, 113, 106, 93, 90** |

The largest eigenvalue of TSAM's Hessian is only 41% of ERM's, which is significantly flatter than that of SAM.

## Highlights & Insights

1. **Unified Framework**: TSAM smoothly interpolates between min-avg and min-max optimization via a single parameter $t$, offering a theoretically unified perspective on the SAM family.
2. **Theoretical Rigor**: The paper proves that TSAM preserves convexity/Lipschitz properties, maintains controllable smoothness, favors flat solutions, and guarantees the existence of an optimal $t$.
3. **Efficient Sampling**: Only 3-5 perturbation samples are required to approximate the tilted gradient effectively, keeping the additional computational overhead acceptable.
4. **Broad Applicability**: TSAM consistently outperforms SAM and its variants across three architectures (CNN, ViT, BERT) and two tasks (vision and text).
5. **Noise Robustness**: The improvement is especially pronounced under label noise (+8.97% over ERM), indicating that TSAM possesses stronger robustness to distribution shifts.

## Limitations & Future Work

1. **Computational Overhead**: Each step requires $s$ additional forward and backward passes, leading to a training time roughly $s$ times that of SAM, which may be less practical for large-scale models.
2. **Sensitivity to Hyperparameter $t$**: The optimal $t$ must be searched from $\{0,1,5,20,100\}$ on a validation set, increasing tuning costs.
3. **Theory-Practice Gap**: The theoretical preference for flat minima strictly holds only for GLM-type losses, remaining empirical for deep neural networks.
4. **HMC Approximation Quality**: To prioritize efficiency, only a 1-step Euler discretization is performed and samples are directly accepted, which does not guarantee sampling consistency.
5. **No Integration with Adaptive Optimizers**: The integration of TSAM with adaptive optimizers (e.g., Adam/AdaSAM) is not evaluated and is left for future work.

## Related Work & Insights

- **SAM Family**: SAM (Foret 2020), RSAM (Liu 2022), PGN (Zhao 2022), VASSO (Li & Giannakis 2023), etc.
- **Exponential Tilting**: Tilted ERM (Li et al., 2023) performs exponential weighting in the data space; TSAM generalizes it to the parameter space.
- **Average-perturbed Loss**: Related to average-perturbed sharpness (Wen 2022) and noise-perturbed loss (Zhang 2024), where TSAM serves as an interpolation between them and SAM.

## Rating

- Novelty: ⭐⭐⭐⭐ — The approach of generalizing exponential tilting from the data space to the parameter space is simple and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation covering CNN, ViT, and BERT architectures under noisy and OOD settings, with comparison against multiple SAM variants.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear theoretical derivations, rigorous experimental design, and complete ablation studies.
- Value: ⭐⭐⭐⭐ — Provides a smoother alternative to SAM, with insightful theoretical analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](../../ICML2026/optimization/stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICML 2025\] Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation](sassha_sharpness-aware_adaptive_second-order_optimization_with_stable_hessian_ap.md)
- [\[ICLR 2026\] Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models](../../ICLR2026/optimization/bi-lora_efficient_sharpness-aware_minimization_for_fine-tuning_large-scale_model.md)
- [\[ICLR 2026\] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization](../../ICLR2026/optimization/towards_understanding_the_calibration_benefits_of_sharpness-aware_minimization.md)
- [\[ICML 2025\] Generalization and Robustness of the Tilted Empirical Risk](generalization_and_robustness_of_the_tilted_empirical_risk.md)

</div>

<!-- RELATED:END -->
