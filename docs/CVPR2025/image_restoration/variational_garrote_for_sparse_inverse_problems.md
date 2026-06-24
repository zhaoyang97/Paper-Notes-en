---
title: >-
  [Paper Note] Variational Garrote for Sparse Inverse Problems
description: >-
  [CVPR 2025][Image Restoration][Sparse Regularization] This paper systematically compares the performance of $\ell_1$ regularization (LASSO) and Variational Garrote (VG, a probabilistic approximation of $\ell_0$) across three inverse problems: signal resampling, denoising, and sparse-view CT reconstruction. It is demonstrated that VG typically achieves lower generalization errors in highly underdetermined regimes (such as low sampling rates or highly sparse angles) because the…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Sparse Regularization"
  - "Variational Garrote"
  - "LASSO"
  - "Inverse Problems"
  - "CT Reconstruction"
  - "Signal Denoising"
date: 2026-05-08
content_hash: d7f74f124e967b98
---

# Variational Garrote for Sparse Inverse Problems

**Conference**: CVPR 2025  
**arXiv**: [2603.12562](https://arxiv.org/abs/2603.12562)  
**Code**: None  
**Area**: Image Restoration / Sparse Reconstruction  
**Keywords**: Sparse Regularization, Variational Garrote, LASSO, Inverse Problems, CT Reconstruction, Signal Denoising

## TL;DR
This paper systematically compares the performance of $\ell_1$ regularization (LASSO) and Variational Garrote (VG, a probabilistic approximation of $\ell_0$) across three inverse problems: signal resampling, denoising, and sparse-view CT reconstruction. It is demonstrated that VG typically achieves lower generalization errors in highly underdetermined regimes (such as low sampling rates or highly sparse angles) because the spike-and-slab prior aligns better with the true sparse distribution.

## Background & Motivation
**Background**: Sparse regularization is a core tool for solving underdetermined inverse problems. LASSO ($\ell_1$) is widely used due to its convexity and theoretical guarantees, corresponding to a Laplace prior. The ideal $\ell_0$ sparsity corresponds to a spike-and-slab prior but is NP-hard to solve.

**Limitations of Prior Work**: LASSO imposes continuous shrinkage, which biases large coefficients and produces approximately sparse rather than truly sparse solutions. Under highly sparse signals and severely underdetermined conditions, support recovery is inaccurate.

**Key Challenge**: Different regularizers implicitly correspond to different prior distributions, and the reconstruction performance depends on how well the prior matches the data's sparse structure. However, the impact of this "prior-data alignment" has lacked systematic empirical study in practical inverse problems.

**Goal**: This study aims to compare the reconstruction behavior of $\ell_1$ (LASSO) and probabilistic $\ell_0$ (VG) under different information bottlenecks within a unified framework.

**Key Insight**: A model-agnostic and fair comparison is achieved using the train-generalization error curve (scanning regularization strengths to compare the minimum generalization error of each method at its optimal regularization strength).

**Core Idea**: When the underlying distribution of coefficients is highly sparse, a spike-and-slab prior (VG) yields more accurate support recovery than a Laplace prior (LASSO), thereby obtaining lower reconstruction errors in highly underdetermined regimes.

## Method

### Overall Architecture
Three inverse problems are unified as sparse linear regression: $\mathbf{y} = \mathbf{A}\mathbf{x} + \epsilon$, where $\mathbf{A}$ represents the subsampling operator (resampling), the identity matrix (denoising), or the discrete Radon transform (CT), respectively. Sparse regression is performed on the coefficients $\mathbf{w}$ in the transform domain (DCT or pixel domain): $\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{y} - \mathbf{\Theta w}\|_2^2 + \lambda \mathcal{R}(\mathbf{w})$.

### Key Designs

1. **Variational Garrote (VG)**:

    - **Function**: Approximates $\ell_0$ sparsity using hidden binary gating variables $s_i \in \{0,1\}$.
    - **Mechanism**: The regression model is formulated as $y_\mu = \sum_i w_i s_i X_{i\mu} + \xi_\mu$. A Bernoulli prior is imposed on $s_i$ as $p(s_i|\gamma) = e^{\gamma s_i}/(1+e^\gamma)$. Since exact inference is intractable, a mean-field variational approximation $q(\mathbf{s}) = \prod_i q_i(s_i)$ is used, introducing activation probabilities $m_i = q(s_i=1)$.
    - The variational free energy consists of: reconstruction energy $E_{\text{rec}}$ (including a variance term with $m_i(1-m_i)w_i^2$), a prior term $\Omega = -\gamma \sum_i m_i$, and an entropy term $H$.
    - **Design Motivation**: Decouple support selection ($m_i$) and coefficient magnitude ($w_i$) to mitigate the shrinkage bias of LASSO.

2. **Fair Comparison Protocol**:

    - **Function**: Enables model-agnostic comparison via the train-generalization error curve.
    - **Mechanism**: For each information bottleneck setting, regularization hyperparameters (LASSO $\lambda$; VG $\gamma$) are scanned over a wide range to plot the training error vs. generalization error curve, taking the minimum generalization error (MGE) as the optimal performance of each method.
    - **Design Motivation**: The parameter $\lambda$ in LASSO and $\gamma$ in VG cannot be compared directly, but the bias-variance tradeoff curves are universally comparable.

3. **Three Tasks Covering Different Information Bottlenecks**:

    - Signal resampling: Sampling rate $R = M/N$ from 5% to 50%, sparse in the DCT domain.
    - Signal denoising: Noise amplitude $\alpha$ from $10^{-2}$ to $10^0$, sparse in the DCT domain.
    - Sparse-view CT: Number of projection angles $K$ from 10 to 120, sparse in the pixel domain.

### Loss & Training
All tasks utilize the AdamW optimizer with an initial learning rate of 0.3, decreasing via ReduceLROnPlateau and terminating at $10^{-5}$, with up to 50,000 iterations. Parameters are initialized with small Gaussian noise. The signal experiments are optimized using batches over 100 independent mask/noise realizations for stability.

## Key Experimental Results

### Main Results: Signal Resampling

| Sampling Rate R | LASSO MGE | VG MGE | Gain |
|---------|-----------|--------|------|
| R=0.05 | ~0.5 | ~0.35 | VG is significantly better |
| R=0.10 | ~0.15 | ~0.05 | VG is significantly better |
| R=0.20 | ~0.03 | ~0.01 | VG is better |
| R=0.50 | ~0.001 | ~0.001 | Comparable |

VG demonstrates the most substantial advantage under low sampling rates ($R < 0.2$), which is precisely the most critical regime for support recovery.

### Sparse-view CT Reconstruction

| Dataset | K=40 Angles FBP MSE | LASSO MSE | VG MSE |
|--------|---------------|-----------|--------|
| Shepp-Logan | High | Medium | **Lowest** |
| LIDC (Lung CT) | High | Medium | **Lowest** |
| BraTS (Brain MRI) | High | Medium | **Lowest** |
| Walnut | High | Medium | **Lowest** |

The MSE ranking is consistent: FBP > LASSO > VG, with VG exhibiting smaller variance.

### Ablation Study / Key Findings

| Observation | Explanation |
|------|------|
| Abrupt changes in VG training error | Gating variables lead to the collective activation/deactivation of frequency components, exhibiting phase transition behavior |
| No abrupt change in denoising | Noise blurs the effective spectral support, such that minor hyperparameter variations no longer trigger discrete activation |
| Homogeneous regions vs. boundaries | VG reduces errors in homogeneous regions but has slightly weaker boundary sharpness—suggesting potential complementarity with TV |

### Key Findings
- VG consistently outperforms LASSO in highly underdetermined regions across all tasks, with its advantages concentrated in low-sampling, low-angle, and low-SNR regimes.
- The advantage of VG stems from the better alignment of the spike-and-slab prior with the true sparse distribution.
- LASSO exhibits a smooth error-curve (continuous shrinkage), whereas VG shows jumps (discrete support switching).
- In CT tasks, VG improves reconstruction in homogeneous areas but is slightly inferior in edge sharpness—suggesting its potential complementarity with Total Variation (TV) regularization.

## Highlights & Insights
- **Prior Comparison Under a Unified Framework**: This work unifies signal resampling, denoising, and CT reconstruction as sparse regressions, isolating the impact of prior selection. This perspective of "prior-data alignment" serves as general guidance for selecting regularization methods.
- **Phase Transition Behavior of VG**: The sudden jumps in training error reveal the essence of $\ell_0$-type methods: support selection is a discrete decision, presenting a sharp contrast to the continuous shrinkage of LASSO.
- **Transferable Ideas**: The spike-and-slab prior of VG could be applied to the final layer weights of deep networks, introducing structured sparsity assumptions.

## Limitations & Future Work
- VG involves non-convex optimization and lacks global convergence guarantees, meaning results may depend heavily on initialization.
- The computational complexity is higher than that of LASSO due to the introduction of an additional set of gating variables.
- The CT experiments perform sparse regression in the pixel domain. Since natural images are not strictly sparse in the pixel domain, the results are more applicable to medical images with large background areas.
- Comparison with deep learning methods (U-Net, diffusion model-based reconstruction) was not conducted.
- Joint regularization combining VG + TV was not explored—though the paper highlights this as a natural future extension.

## Related Work & Insights
- **vs LASSO**: The Laplace prior of LASSO suffers from shrinkage bias on large coefficients, whereas VG eliminates this bias by decoupling support selection from magnitude estimation.
- **vs Ridge/Tikhonov**: Gaussian priors do not produce sparse solutions and are thus unsuitable for the sparse inverse problems considered in this work.
- **vs Deep Priors (DIP/Diffusion)**: This work focuses on prior selection within traditional regularization. Comparison with learned priors remains a valuable path for future work.
- Offers reference value for researchers working on compressed sensing and medical image reconstruction—suggesting the consideration of VG as an alternative to LASSO under highly underdetermined conditions.

## Rating
- **Novelty**: ⭐⭐⭐ VG itself is not new, but the systematic comparison in inverse problems is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three tasks, multiple datasets, statistical repetitions, and a fair comparison protocol.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivations and a smooth narrative under a unified framework.
- **Value**: ⭐⭐⭐ Rather theoretically oriented; practical impact might be limited, but the insights are valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FiRe: Fixed-points of Restoration Priors for Solving Inverse Problems](fire_fixed-points_of_restoration_priors_for_solving_inverse_problems.md)
- [\[CVPR 2026\] Dual Ascent Diffusion for Inverse Problems](../../CVPR2026/image_restoration/dual_ascent_diffusion_for_inverse_problems.md)
- [\[ICLR 2026\] Flower: A Flow-Matching Solver for Inverse Problems](../../ICLR2026/image_restoration/flower_a_flow-matching_solver_for_inverse_problems.md)
- [\[ICML 2026\] Solving Inverse Problems with Flow-based Models via Model Predictive Control](../../ICML2026/image_restoration/solving_inverse_problems_with_flow-based_models_via_model_predictive_control.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](../../CVPR2026/image_restoration/gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)

</div>

<!-- RELATED:END -->
