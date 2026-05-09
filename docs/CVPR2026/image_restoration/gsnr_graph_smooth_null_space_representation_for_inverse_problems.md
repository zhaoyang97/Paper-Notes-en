---
title: >-
  [Paper Note] GSNR: Graph Smooth Null-Space Representation for Inverse Problems
description: >-
  [CVPR 2026][Image Restoration][Inverse Problems] This paper proposes Graph Smooth Null-Space Representation (GSNR), which employs spectral graph theory to construct a null-space-constrained Laplacian matrix and selects the $p$ smoothest spectral modes as the null-space projection basis. GSNR provides structured null-space constraints for inverse problem solvers including PnP, DIP, and diffusion models, achieving up to 4.3 dB PSNR gains on deblurring, compressed sensing, demosaicing, and super-resolution.
tags:
  - CVPR 2026
  - Image Restoration
  - Inverse Problems
  - Null-Space Representation
  - Graph Smoothness
  - Spectral Graph Theory
  - Plug-and-Play
date: 2026-05-08
content_hash: e01dd2773b3afbb6
---

# GSNR: Graph Smooth Null-Space Representation for Inverse Problems

**Conference**: CVPR 2026
**arXiv**: [2602.20328](https://arxiv.org/abs/2602.20328)
**Code**: N/A
**Area**: Image Restoration / Inverse Problems
**Keywords**: Inverse Problems, Null-Space Representation, Graph Smoothness, Spectral Graph Theory, Plug-and-Play

## TL;DR

This paper proposes Graph Smooth Null-Space Representation (GSNR), which employs spectral graph theory to construct a null-space-constrained Laplacian matrix and selects the $p$ smoothest spectral modes as the null-space projection basis. GSNR provides structured null-space constraints for inverse problem solvers including PnP, DIP, and diffusion models, achieving up to 4.3 dB PSNR gains on deblurring, compressed sensing, demosaicing, and super-resolution.

## Background & Motivation

The central challenge in imaging inverse problems lies in the **ill-posedness of the null space**: for an underdetermined system $y = Hx^* + \omega$, the null space of the sensing matrix $H$ admits infinitely many solutions consistent with the measurements. Any signal $x$ can be decomposed into a range-space component $x_r = P_r x$ (observable) and a null-space component $x_n = P_n x$ (unobservable).

Existing methods suffer from two categories of limitations: (1) Generic priors (e.g., PnP denoisers, score functions from diffusion models) operate over the entire image space without distinguishing observable from unobservable components — denoisers may freely modify null-space components, introducing bias and hallucinations; (2) Existing null-space methods (e.g., NSN, NPN) attempt to learn low-dimensional projections within the null space, but **blindly learning arbitrary null-space subspaces** may waste model capacity and introduce bias, as these methods have no notion of which null-space directions are "meaningful."

The core insight is that **natural images are not uniformly distributed within the null space — they occupy a low-dimensional, structured subset.** Motivated by graph-based smooth representations of images, spectral graph theory can be leveraged to select the **smoothest null-space directions** as the projection basis. These directions are both easier to predict from measurements and efficiently cover the natural image variations within the null space.

## Method

### Overall Architecture

Given an inverse problem $y = Hx + \omega$ and a graph Laplacian $L$, GSNR constructs a null-space-constrained Laplacian $T = P_n L P_n$, and takes the eigenvectors corresponding to its $p$ smallest eigenvalues to form the projection matrix $S$. A predictor $G(y) \approx Sx^*$ is trained to estimate the null-space component from measurements. During reconstruction, $\|G(y) - Sx\|^2$ is incorporated as a regularization term into any solver (PnP, DIP, or diffusion models).

### Key Designs

1. **Null-Space-Constrained Laplacian and Graph Smooth Projection**:

    - Function: Selects the most informative null-space directions in a principled manner.
    - Mechanism: Constructs a graph Laplacian $L$ (4/8-nearest-neighbor image grid with edge weights encoding local pixel similarity), then projects it onto the null space to obtain $T = P_n L P_n$. The eigen-decomposition of $T$ yields a frequency ordering within the null space — the smallest eigenvalues correspond to the smoothest (low-frequency) null-space modes. The top $p$ smoothest modes form the projection matrix $S \in \mathbb{R}^{p \times n}$.
    - Design Motivation: Natural images are spatially smooth; their null-space components should likewise be preferentially described by smooth modes. Theorems 1 & 2 establish that graph-smooth modes achieve high coverage at small $p$ — a small number of modes suffices to capture the majority of null-space variance.

2. **Null-Space Component Predictor**:

    - Function: Predicts a low-dimensional null-space representation from measurements $y$.
    - Mechanism: A network $G(y) \approx Sx^*$ is trained to predict $p$-dimensional null-space coefficients. Proposition 1 theoretically demonstrates that graph-smooth null-space components are more predictable from measurements than those spanned by arbitrary null-space bases, owing to the stronger correlation between smooth modes and the range space.
    - Design Motivation: Predictability and coverage are two key criteria for null-space representations. GSNR simultaneously optimizes both: high coverage (large variance with small $p$) and high predictability.

3. **Plug-and-Play Integration**:

    - Function: Integrates the GSNR regularization term into arbitrary inverse problem solvers.
    - Mechanism: The term $\|G(y) - Sx\|^2$ is appended as an additional regularizer to the variational objective. For PnP solvers, the null-space penalty is incorporated into the data-fidelity step of proximal gradient descent; for DIP, it augments implicit regularization; for diffusion models, it provides null-space guidance during posterior sampling.
    - Design Motivation: GSNR constrains only the unobservable components (null space), making it complementary rather than conflicting with existing priors that operate over the full image space.

### Loss & Training

The predictor $G$ is trained with an L2 loss: $\min_G \mathbb{E}\|G(y) - Sx^*\|^2$. The projection matrix $S$ is obtained via eigen-decomposition of the null-space-constrained Laplacian and requires no learning. The regularization weight $\eta$ at reconstruction time requires tuning.

## Key Experimental Results

### Main Results

**Image Deblurring**

| Method | PSNR↑ | Gain | Notes |
|--------|-------|------|-------|
| PnP Baseline | X dB | — | No null-space constraint |
| PnP + GSNR | X+Y dB | **up to +4.3 dB** | Significant improvement |
| End-to-End Supervised Model | Z dB | — | Supervised training |
| PnP + GSNR | Z+1 dB | **up to +1 dB** | Surpasses end-to-end model |

**Consistency Across Tasks**

| Task | PnP Gain | DIP Gain | Diffusion Gain |
|------|----------|----------|----------------|
| Deblurring | Significant | Significant | Significant |
| Compressed Sensing | Significant | Significant | Significant |
| Demosaicing | Significant | Significant | Significant |
| Super-Resolution | Significant | Significant | Significant |

### Ablation Study

| Configuration | PSNR | Notes |
|---------------|------|-------|
| No null-space constraint | Baseline | Standard PnP/DIP/Diffusion |
| Random null-space basis (NPN) | +marginal | Low coverage |
| **GSNR (graph-smooth basis)** | **+maximum** | High coverage + high predictability |

### Key Findings

- GSNR consistently improves performance across four tasks and three solvers, demonstrating the universal value of structured null-space constraints.
- Graph-smooth bases achieve higher coverage than randomly learned bases at small $p$ — 30% of modes can capture 80%+ of null-space variance.
- The coverage/predictability curve serves as an operational diagnostic tool for selecting $p$.
- Null-space regularization reduces hallucinations by preventing denoisers from freely modifying unobservable components.

## Highlights & Insights

- **"Constrain only the invisible"** represents an elegant design philosophy: existing priors that constrain the full image space may conflict with data-fidelity terms, whereas GSNR applies structure exclusively within the sensor's blind region.
- **Spectral graph theory provides principled direction selection**: the projection matrix is derived directly from problem structure rather than learned, yielding theoretical clarity.
- **The coverage/predictability diagnostic curve** is a practical tool that transforms regularization strength selection from ad hoc tuning into objective evaluation.

## Limitations & Future Work

- Construction of the graph Laplacian requires neighborhood pixel similarity estimation, which may be unreliable for severely degraded images.
- The computational cost of null-space eigen-decomposition may be prohibitive for high-resolution images.
- The generalization of the predictor $G$ depends on the diversity of training data.
- The theoretical analysis assumes a linear forward model; applicability to nonlinear forward operators remains to be validated.

## Related Work & Insights

- **vs. NPN**: NPN also learns null-space projections but does so by blindly learning arbitrary directions; GSNR provides principled direction selection via graph smoothness.
- **vs. PnP/RED**: These methods operate over the full image space without distinguishing observable from unobservable components.
- **vs. Total Variation (TV)**: TV enforces smoothness over the entire image and may over-smooth; GSNR applies smoothness exclusively within the null space.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to introduce graph smoothness into null-space representation, with substantial theoretical contributions (three theorems/propositions).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across four tasks × three solvers, with tight correspondence between theory and experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations with clear motivation and intuition.
- Value: ⭐⭐⭐⭐⭐ Provides a general, plug-and-play null-space regularization framework for inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Variational Garrote for Sparse Inverse Problems](variational_garrote_for_sparse_inverse_problems.md)
- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](../../NeurIPS2025/image_restoration/learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)
- [\[CVPR 2026\] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [\[CVPR 2026\] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)
- [\[NeurIPS 2025\] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](../../NeurIPS2025/image_restoration/improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)

</div>

<!-- RELATED:END -->
