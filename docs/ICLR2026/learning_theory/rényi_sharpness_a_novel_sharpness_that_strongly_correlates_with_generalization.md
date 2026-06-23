---
title: >-
  [Paper Note] Rényi Sharpness: A Novel Sharpness That Strongly Correlates with Generalization
description: >-
  [ICLR 2026][learning_theory][Sharpness-Aware Minimization] This paper identifies that the true determinant of generalization is the "average dispersion/unevenness" of the Hessian spectrum. It defines **Rényi sharpness** (the negative Rényi entropy of the normalized Hessian spectrum) using information theory and demonstrates its strong correlation with generalization across var
tags:
  - ICLR 2026
  - learning_theory
  - Sharpness-Aware Minimization
date: 2026-05-08
content_hash: 3df13b4e4a047a9e
---
# Rényi Sharpness: A Novel Sharpness That Strongly Correlates with Generalization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=rRIhUpobCv](https://openreview.net/forum?id=rRIhUpobCv)  
**Code**: Available (Anonymous link provided in the paper)  
**Area**: Learning Theory / Generalization / Loss Landscape  
**Keywords**: Sharpness, Generalization Bound, Hessian Spectrum, Rényi Entropy, Sharpness-Aware Minimization

## TL;DR
This paper identifies that the true determinant of generalization is the "average dispersion/unevenness" of the Hessian spectrum. It defines **Rényi sharpness** (the negative Rényi entropy of the normalized Hessian spectrum) using information theory and demonstrates its strong correlation with generalization across various scenarios (Kendall’s $\tau$ typically between 0.6–0.9, significantly outperforming older metrics like trace, SAM, and PAC-Bayes). Based on this, it derives generalization bounds and a competitive RSAM training regularization algorithm.

## Background & Motivation

**Background**: Why deep networks generalize under over-parameterization has long been attributed to the "flatness" of the loss landscape. Intuitively, models residing in flat minima (low sharpness) exhibit small loss variations under minor data perturbations, thus favoring generalization. Sharpness is typically quantified by functionals of the loss Hessian $H$, most commonly trace sharpness $\mathrm{tr}(H)$ and maximum eigenvalue sharpness $\lambda_{\max}(H)$.

**Limitations of Prior Work**: Recent empirical studies have found that the correlation between these sharpness metrics and generalization is weak or even contradictory. Dinh et al. (2017) demonstrated that equivalent models can be constructed with arbitrarily large sharpness but identical generalization performance. Andriushchenko et al. (2023) found that sharpness is almost uncorrelated with generalization on modern architectures, while Kaur et al. (2023) noted that $\lambda_{\max}(H)$ fails to predict generalization even under standard training. A significant gap exists between intuition and reality.

**Key Challenge**: The authors argue that the problem is "looking only at a part of the spectrum." The Hessian spectrum can be divided into three types of eigenvalues: Top (few in number but having a high impact on loss changes), Middle (individually small but extremely numerous), and Tail (near zero, almost no impact). $\mathrm{tr}(H)$ essentially reflects only the average magnitude of the middle eigenvalues, while $\lambda_{\max}(H)$ focuses solely on the single largest eigenvalue. Both lose significant information, leading to inaccurate predictions.

**Goal**: To find a sharpness metric that **simultaneously integrates the contributions of all three categories of eigenvalues**, ensuring it is both strongly and stably correlated with generalization while being able to guide model training.

**Key Insight**: The authors' crucial observation is that what truly matters is not the mean or maximum of the spectrum, but the **unevenness/average dispersion** of the spectrum. A more uniform spectrum (where all eigenvalues are close and no specific direction is extreme) is more conducive to generalization because data perturbations in any direction only cause small loss changes. Information-theoretic entropy is the ideal tool to characterize "how uneven a non-negative vector is."

**Core Idea**: Treat the normalized Hessian spectrum as a "virtual probability vector" and use **Rényi entropy** to measure its unevenness. The negative value is defined as Rényi sharpness; higher entropy (a more uniform spectrum) implies lower sharpness and better generalization. Compared to variance (where tail eigenvalues near zero dominate but are irrelevant to generalization) and Shannon entropy (which has no tunable parameters), Rényi entropy introduces a free order $\alpha$, allowing flexible adjustment of sensitivity to large and small eigenvalues based on spectral morphology.

## Method

### Overall Architecture
This paper combines the proposal of a new metric with theoretical analysis and algorithmic implementation. Instead of a multi-stage pipeline, it unfolds around a core definition: first **defining** Rényi sharpness and proving its reparameterization invariance; then **deriving** two generalization bounds with Rényi sharpness as the primary term to formalize the "low sharpness $\to$ good generalization" relationship; next, addressing two practical issues—**selecting the order $\alpha$** (choosing between 0.5 and 1.5 based on spectral shape) and **fast estimation** (using SLQ to reduce $O(n^3)$ spectral decomposition to matrix-vector multiplications); and finally, implementing it as a training regularizer, **RSAM**.

### Key Designs

**1. Rényi sharpness: Quantifying "spectral unevenness" using Rényi entropy**

To address the failure of old metrics that only focus on local parts of the spectrum, this work seeks a quantity reflecting the unevenness of the entire spectrum. Rényi entropy for a probability vector $p=[p_1,\dots,p_n]$ and order $\alpha$ is defined as:

$$H_\alpha(p) = \frac{1}{1-\alpha}\log\sum_{i=1}^n p_i^\alpha,\quad 0<\alpha<\infty,\ \alpha\neq 1$$

It is a concave function of $p$ and reaches its maximum when $p$ is uniform. Extending this to a positive definite matrix $H$, the eigenvalues are normalized into virtual probabilities $\lambda_i(H)/\mathrm{Tr}(H)$:

$$H_\alpha(H) = \frac{1}{1-\alpha}\log\sum_{i=1}^n\Big(\frac{\lambda_i(H)}{\mathrm{Tr}(H)}\Big)^\alpha$$

**Rényi sharpness is defined as the negative Rényi entropy of the normalized spectrum, $-H_\alpha(H)$** (calculated using layer-wise Hessians; rare negative eigenvalues are handled by absolute values). A more uniform spectrum $\to$ higher entropy $\to$ lower sharpness $\to$ better generalization. Unlike variance, tail eigenvalues near zero do not dominate the result (as they contribute little to generalization), and the parameter $\alpha$ allows for focus tuning.

**2. Reparameterization Invariance and Generalization Bounds: Transforming the metric into a theoretically sound indicator**

A robust generalization indicator must be invariant to "superficial" network transformations. Dinh et al. (2017) attacked old sharpness metrics using layer-wise scaling transformations in ReLU networks. Rényi sharpness is immune to such transformations: for an $L$-layer network with positive homogeneous activations, a scaling $\tilde W_l = c_l W_l$ satisfying $\prod_{l=1}^L c_l = 1$ leaves the Rényi entropy of the normalized Hessian spectrum invariant (Proposition 2.2).

Using this invariance and the technique of converting data perturbations into multiplicative weight perturbations, the paper provides two generalization bounds. The first (Theorem 3.2) takes the form:

$$\mathbb{E}_Q[L(D,\theta)] \le \mathbb{E}_Q[L(S,\theta)] + 2\sqrt{\frac{2L_0 + CV^2/n\,\exp\!\big(-\tfrac1n(H_\alpha(H)-A)\big) + \log\frac{2N}{\delta}}{N-1}}$$

Both bounds demonstrate that the generalization gap is controlled by the Rényi entropy of the normalized Hessian spectrum. Higher entropy (lower Rényi sharpness $-H_\alpha$) leads to tighter bounds.

**3. Selection of Order $\alpha$: Choosing between 0.5 and 1.5 based on Hessian spectral morphology**

The authors measured spectral shapes across layers using PyHessian and found that while deep network Hessians are heavy-tailed, they fall into two categories: ① **Zero-dominant, multi-cluster**: Besides near-zero values and a few large eigenvalues, there is a group of intermediate "non-negligible but smaller" eigenvalues; ② **Zero-dominant, uniform**: Only near-zero and a few large eigenvalues exist.

The selection logic is determined by the penalty strength of $\alpha$ on unevenness (larger $\alpha$ emphasizes high probability mass): for multi-cluster spectra, it is necessary to distinguish subtle differences between clusters, so $\alpha < 1$ is used (empirically $\alpha=0.5$). For uniform spectra, the focus is on differences between the dominant eigenvalues, where $\alpha < 1$ would smooth the differences; thus $\alpha \ge 1$ is used (empirically $\alpha=1.5$). A rule of thumb: select $\alpha < 1$ when inter-cluster separation exceeds intra-cluster amplification, and $\alpha > 1$ for single-cluster cases.

**4. Fast SLQ Estimation and RSAM Regularization: Making Rényi sharpness computationally feasible**

Spectral decomposition for large Hessians is computationally prohibitive. The authors rewrite Rényi entropy using matrix function traces:

$$H_\alpha(H) = \frac{1}{1-\alpha}\log\frac{\mathrm{Tr}(H^\alpha)}{\mathrm{Tr}(H)^\alpha}$$

This requires estimating $\mathrm{Tr}(H)$ and $\mathrm{Tr}(H^\alpha)$. Using Hutchinson’s stochastic trace estimator and Stochastic Lanczos Quadrature (SLQ), the decomposition is transformed into several matrix-vector products (Algorithm 1), significantly reducing complexity. For training, Hessian calculations are avoided by approximating $H$ with gradient magnitudes $\mathrm{GM} = \big(\mathrm{Diag}(\tfrac1N\sum_i\nabla_\theta l)\big)^2$, yielding a differentiable regularizer. RSAM follows the SAM approach: perturb weights along the regularized gradient, then compute loss at the perturbed point.

### Loss & Training
The RSAM training objective consists of the original classification loss plus a Rényi regularization term, implemented via a two-step perturbation. The order $\alpha$ is chosen according to the spectral rules (typically 0.5 or 1.5). Key hyperparameters include the perturbation radius $\rho$ and warm-up length. Warm-up (standard SGD) is used for the first few epochs (or until a certain accuracy threshold) to ensure stability before switching to RSAM.

## Key Experimental Results

### Main Results

**Correlation Experiments**: On ResNet18/34 and Simple ViT across CIFAR10/100 and TinyImageNet, various minima were generated by varying learning rates and optimizers. Kendall’s $\tau$ was used to measure the correlation between sharpness and the generalization gap. Rényi sharpness significantly outperformed traditional metrics (Table: ResNet18-CIFAR10 results):

| Metric | Kendall $\tau$ | Note |
|------|-----------|------|
| Rényi sharpness (Layer-wise, best layer) | +0.77 | Most layers between 0.6–0.78 |
| trace $\mathrm{tr}(H)$ | -0.05 | Nearly uncorrelated |
| parameter norm | -0.01 | Uncorrelated |
| Fisher-Rao norm | -0.25 | Weakly negatively correlated |
| PAC-Bayes flat | +0.13 | Weakly correlated |
| SAM $\ell_\infty$ | -0.17 | Weakly negatively correlated |
| ASAM $\ell_\infty$ | +0.20 | Weakly correlated |

Across tasks, Rényi sharpness $\tau$ typically falls between 0.6–0.9, while trace, SAM, and PAC-Bayes metrics often stay within $\pm 0.3$.

**RSAM Training Experiments**: Compared with SAM, ASAM, Eigen-SAM, etc., RSAM achieved the highest test accuracy in most tasks:

| Dataset | Model | SGD | SAM | ASAM | RSAM (Ours) |
|--------|------|-----|-----|------|-----------|
| CIFAR10 | WRN-28-10 | 96.36 | 96.95 | 96.79 | **97.13** |
| CIFAR100 | ResNet56 | 72.60 | 74.86 | 75.20 | **75.71** |
| TinyImageNet | ResNet50 | 59.62 | 60.70 | 62.56 | **63.33** |
| CIFAR100 | ViT-B-16 (Fine-tune) | 88.27 | 89.38 | 88.78 | **89.58** |

### Ablation Study

| Dimension | Key Result | Description |
|----------|---------|------|
| Selection Rule Match Rate | 1451/1630 matches | Empirically optimal $\alpha$ highly consistent with spectral morphology rules. |
| $\alpha$ Value | 0.5 (multi-cluster) / 1.5 (uniform) | Robust across datasets; mismatch weakens correlation. |
| warm-up | Essential | Applying Rényi regularization too early is unstable. |
| Global vs Layer-wise | Global | A single global regularization term is sufficient and avoids hyperparameter explosion. |

### Key Findings
- The correlation between Rényi sharpness and generalization ($\tau \approx 0.6–0.9$) is much stronger than all traditional metrics and remains stable across architectures and datasets.
- The selection rule for $\alpha$ is substantiated by 1630 statistical cases, showing spectral morphology determines the optimal parameter.
- The performance gain of RSAM over ASAM is sometimes small, which may be due to the use of gradient magnitude as a Hessian approximation.

## Highlights & Insights
- **Refining "spectral unevenness" into a precise entropy metric**: While old metrics focused on parts of the spectrum, Rényi entropy integrates the entire spectrum, explaining the significant jump in correlation.
- **Reparameterization invariance as a theoretical and empirical pillar**: It makes the metric immune to scaling attacks while serving as a key tool for deriving generalization bounds.
- **Dynamic focus tuning via $\alpha$**: Using a parameterized entropy instead of a fixed measure allows the metric to adapt to different data distributions and spectral shapes.
- **Pragmatic implementation**: Combining SLQ with gradient magnitude approximations makes an expensive spectral functional viable for actual training.

## Limitations & Future Work
- RSAM uses an approximation (gradient magnitude and first-order expansion). Using more precise estimators might further improve performance.
- Generalization bounds rely on strong assumptions, such as data separability and positive definite Hessians, which may not hold strictly in all neural networks.
- Experiments focused on medium-scale image classification; validation on large-scale pre-training or NLP tasks is pending.
- The selection rule simplifies spectra into two categories; more complex morphologies could benefit from more nuanced $\alpha$ selection.

## Related Work & Insights
- **vs Trace/$\lambda_{\max}$ Sharpness**: These reflect only the mean or extreme of the spectrum, losing information. Rényi sharpness captures the unevenness of the entire spectrum.
- **vs SAM/ASAM**: These optimize the worst-case perturbation loss, essentially favoring top eigenvalues. RSAM regularizes the entire spectrum and achieves higher accuracy in most tested tasks.
- **vs PAC-Bayes / Fisher-Rao**: These metrics often show unstable correlations in large-scale studies, whereas Rényi sharpness is significantly more robust.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Precisely defines sharpness via spectral unevenness using Rényi entropy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Excellent correlation and training results, though scale is medium.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from definition to theory to algorithm.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretically guaranteed and empirically strong answer to the controversy of sharpness as a generalization predictor.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] Quantitative Bounds for Length Generalization in Transformers](quantitative_bounds_for_length_generalization_in_transformers.md)
- [\[ICLR 2026\] Understanding the Dynamics of Forgetting and Generalization in Continual Learning via the Neural Tangent Kernel](understanding_the_dynamics_of_forgetting_and_generalization_in_continual_learnin.md)
- [\[ICLR 2026\] Provable Separations between Memorization and Generalization in Diffusion Models](provable_separations_between_memorization_and_generalization_in_diffusion_models.md)
- [\[ICLR 2026\] Pretrain–Test Task Alignment Governs Generalization in In-Context Learning](pretraintest_task_alignment_governs_generalization_in_in-context_learning.md)
- [\[ICLR 2026\] Bound by Semanticity: Universal Laws Governing the Generalization-Identification Tradeoff](bound_by_semanticity_universal_laws_governing_the_generalization-identification_.md)

</div>

<!-- RELATED:END -->
