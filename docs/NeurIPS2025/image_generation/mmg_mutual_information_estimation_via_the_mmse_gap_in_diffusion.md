---
title: >-
  [Paper Note] MMG: Mutual Information Estimation via the MMSE Gap in Diffusion
description: >-
  [NeurIPS 2025][Image Generation][mutual information estimation] Leveraging the information-theoretic formulation of diffusion models, this paper proves that mutual information equals one-half of the integral over all signal-to-noise ratios of the gap between conditional and unconditional denoising MMSE. The proposed MMG estimator, combined with adaptive importance sampling and the orthogonality principle, significantly improves estimation accuracy and stability.
tags:
  - NeurIPS 2025
  - Image Generation
  - mutual information estimation
  - diffusion models
  - MMSE
  - denoising
  - importance sampling
date: 2026-05-08
content_hash: 62274a830d604478
---

# MMG: Mutual Information Estimation via the MMSE Gap in Diffusion

**Conference**: NeurIPS 2025
**arXiv**: [2509.20609](https://arxiv.org/abs/2509.20609)
**Code**: [GitHub](https://github.com/fengsxy/Diffusion_MI)
**Area**: Information Theory / Diffusion Models
**Keywords**: mutual information estimation, diffusion models, MMSE, denoising, importance sampling

## TL;DR

Leveraging the information-theoretic formulation of diffusion models, this paper proves that mutual information equals one-half of the integral over all signal-to-noise ratios of the gap between conditional and unconditional denoising MMSE. The proposed MMG estimator, combined with adaptive importance sampling and the orthogonality principle, significantly improves estimation accuracy and stability.

## Background & Motivation

Mutual information (MI) is the most general measure of dependence between random variables, yet estimating MI from samples remains a fundamental challenge:

**Limitations of traditional methods**: KDE- and k-NN-based methods suffer severely from the curse of dimensionality in high-dimensional settings.

**Sample complexity of variational methods**: Variational lower-bound approaches such as MINE and InfoNCE exhibit sample complexity or variance that grows exponentially with the true MI, and are constrained by batch size.

**Difficulty of intermediate steps in score matching**: Although MINDE exploits diffusion models, it relies on accurate approximation of the log-density gradient (score function), which is a challenging intermediate step.

**Core motivation**: Given the remarkable progress of diffusion models in density estimation, can one bypass score matching and estimate MI directly from the denoising objective itself?

## Method

### Overall Architecture

Starting from the information-theoretic formulation of diffusion models, the paper derives an exact relationship between MI and the MMSE gap, approximates the MMSE with a neural network denoiser, and estimates MI via numerical integration.

### Key Designs

1. **Derivation of the MMSE Gap Formula**:

   Given a Gaussian noise channel $z_\gamma = \sqrt{\gamma/(1+\gamma)} x + \sqrt{1/(1+\gamma)} \epsilon$ with signal-to-noise ratio $\gamma$, starting from the ITD (Information-Theoretic Diffusion) result:

    $-\log p(x) = \frac{d}{2}\log(2\pi e) - \frac{1}{2}\int_0^\infty d\gamma \left(\frac{d}{1+\gamma} - \text{mmse}(x|\gamma)\right)$

   Writing the analogous expression for the conditional distribution $p(x|y)$ and subtracting yields the pointwise mutual information:

    $\log p(x|y) - \log p(x) = \frac{1}{2}\int_0^\infty d\gamma \left(\text{mmse}(x|\gamma) - \text{mmse}(x|\gamma, y)\right)$

   Taking expectations gives the exact MI expression:

    $I(x;y) = \frac{1}{2}\int_0^\infty d\gamma \left(\text{mmse}_x(\gamma) - \text{mmse}_{x|y}(\gamma)\right)$

   That is, **MI equals one-half the area between the conditional and unconditional MMSE curves**.

2. **Model Training**: A single denoising network $\hat{x}_\theta(z_\gamma, \gamma, y)$ is trained with $y$ replaced by a null token with 50% probability (analogous to classifier-free guidance), enabling the same network to learn both conditional and unconditional denoising. The loss is the standard MSE denoising objective.

3. **Adaptive Importance Sampling**:

    - The integral over SNR is estimated via Monte Carlo with a logistic proposal distribution $q(\gamma)$.
    - A preliminary model is first trained to analyze the transition region of the conditional MMSE curve.
    - The location parameter $\mu$ is set to the log-SNR at which the MMSE curve crosses the $d/2$ error threshold.
    - The scale parameter $\sigma$ is derived from the position of the $d/4$ threshold.
    - This concentrates samples on the critical transition region where the denoiser shifts from ineffective to effective.

4. **Orthogonality Principle**:
   Based on the orthogonality property of MMSE estimation, the MMSE gap can be equivalently expressed as the squared $\ell_2$ norm of the difference between conditional and unconditional denoiser outputs:

    $\text{MMSE Gap} = \mathbb{E}[\|\hat{x}(z_\gamma, y) - \hat{x}(z_\gamma)\|^2]$

   Advantages: (a) guaranteed non-negativity (a single squared term), avoiding numerical instability from subtracting two large numbers; (b) a smoother integrand with lower variance.

### Four Estimator Variants

| Variant | Adaptive Sampling | Orthogonality Principle |
|---|---|---|
| MMG | ✗ | ✗ |
| MMG-adaptive | ✓ | ✗ |
| MMG-orthogonal | ✗ | ✓ |
| MMG-orthogonal-adaptive | ✓ | ✓ |

## Key Experimental Results

### Main Results: Success Rate on 40-Task Benchmark

| Method | Successful Tasks (/40) |
|---|---|
| MINE | 30 |
| InfoNCE | 33 |
| NWJ | 30 |
| DoE (Gaussian) | 27 |
| MINDE-j | 35 |
| MINDE-c | 35 |
| MMG | 33 |
| MMG-adaptive | 35 |
| MMG-orthogonal | 37 |
| **MMG-orthogonal-adaptive** | **39** |

### High-MI Regime Comparison

In the high mutual information range MI $\in [10, 15]$ (3×3 sparse mixture of Gaussians):

| Setting | MMG-adaptive | MINDE | MMG-orthogonal |
|---|---|---|---|
| Original distribution | Most accurate | Severe underestimation | Conservative bias |
| Half-cube transform | Most accurate | Marked underestimation | Conservative bias |
| Spiral transform | All methods challenged | Underestimation | Conservative bias |

MINDE exhibits severe underestimation in high-MI regimes because score matching must approximate sharp, high-frequency score functions, a task at which neural networks fail due to spectral bias.

### Self-Consistency Tests (MNIST, 28×28)

| Test | Expected Result | MMG Performance |
|---|---|---|
| Baseline: $I(A;B_r)/I(A;B)$ | Monotonically approaches 1 with $r$ | ✓ Passed |
| Data Processing: $I(A;[B_{r+k},B_r])/I(A;B_{r+k})$ | Identically 1 | ✓ Passed |
| Additivity: $I([A^1,A^2];[B_r^1,B_r^2])/I(A^1;B_r^1)$ | Identically 2 | ✓ Passed |

### Key Findings

- **Bias–variance trade-off**: The orthogonality principle yields low variance but introduces a conservative bias (the distance between denoiser approximations is smaller than the true distance), whereas direct estimation has low bias but high variance. In low-MI regimes the primary challenge is variance → use the orthogonal variant; in high-MI regimes the primary challenge is bias → use the adaptive variant.
- **Generality across variable types**: The theoretical formula holds for arbitrary discrete, continuous, or mixed variables.
- **No guaranteed bound direction**: Because the MMSE terms appear with both signs in the formula, the estimator cannot be guaranteed to be either an upper or a lower bound.
- Adaptive sampling substantially improves accuracy by concentrating computation on the denoiser transition region.

## Highlights & Insights

- **Elegant derivation**: Starting from the ITD density–MMSE relationship, the MMSE gap representation of MI follows naturally from a simple subtraction, yielding a concise and beautiful theoretical chain.
- The geometric intuition of **"MI = area between two MMSE curves"** is exceptionally clear, far more transparent than the abstract expressions of variational bounds.
- The introduction of the orthogonality principle resolves the numerical instability of subtracting two large quantities, representing a critical engineering contribution.
- Identifying the bias–variance trade-off and providing two families of variants for users to select based on their regime reflects solid experimental insight.
- The release of a unified MI estimation library constitutes a significant contribution to the community.

## Limitations & Future Work

- The directionality of the estimate cannot be guaranteed (neither an upper nor a lower bound), relying on the assumption that the neural network approximates the global optimum.
- The conservative bias of the orthogonal variant in high-MI regimes is systematic; no mechanism currently exists for automatic detection and switching.
- Training two denoisers (or one conditional denoiser with 50% dropout) is computationally more expensive than simple variational methods.
- The two-stage training of adaptive sampling adds methodological complexity.
- Scalability to extremely high-dimensional data remains to be validated.

## Related Work & Insights

- **Core distinction from MINDE**: MINDE relies on the score function (density gradient), whereas MMG directly uses the denoising objective, bypassing the difficulties of gradient approximation.
- The framework can be extended to other information-theoretic quantities (e.g., conditional entropy, channel mutual information, alternative MI decompositions).
- **Implication for the diffusion model theory community**: The MMSE perspective provides an information-theoretic toolkit for understanding diffusion models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Although the MMSE gap formula has antecedents in Guo (2011), integrating it with modern diffusion model implementations and adaptive sampling constitutes an important contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — The 40-task benchmark, high-MI case studies, self-consistency tests, and ablation analyses are exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, and the geometric intuition figure (Figure 1) is outstanding.
- **Value**: ⭐⭐⭐⭐ — MI estimation is a foundational tool, and the open-source library lowers the barrier to adoption.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](../../ICLR2026/image_generation/learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[NeurIPS 2025\] Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [\[NeurIPS 2025\] ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)
- [\[NeurIPS 2025\] Diffusion-Classifier Synergy: Reward-Aligned Learning via Mutual Boosting Loop for FSCIL](diffusion-classifier_synergy_reward-aligned_learning_via_mutual_boosting_loop_fo.md)

<!-- RELATED:END -->
