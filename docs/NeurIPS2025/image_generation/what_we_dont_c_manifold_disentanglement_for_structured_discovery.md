---
title: >-
  [Paper Note] What We Don't C: Manifold Disentanglement for Structured Discovery
description: >-
  [NeurIPS 2025][Image Generation][Manifold Disentanglement] This paper proposes WWDC (What We Don't C), a method that employs conditionally guided latent flow matching to remove known information from existing VAE representations, enabling unknown features to be more readily discovered and accessed in the residual manifold, thus facilitating iterative scientific discovery.
tags:
  - NeurIPS 2025
  - Image Generation
  - Manifold Disentanglement
  - Flow Matching
  - VAE
  - Classifier-Free Guidance
  - Structured Discovery
date: 2026-05-08
content_hash: 951a1f5bce57d8e2
---

# What We Don't C: Manifold Disentanglement for Structured Discovery

**Conference**: NeurIPS 2025
**arXiv**: [2511.09433](https://arxiv.org/abs/2511.09433)
**Code**: Available
**Area**: Representation Learning, Flow Matching, Disentanglement
**Keywords**: Manifold Disentanglement, Flow Matching, VAE, Classifier-Free Guidance, Structured Discovery

## TL;DR
This paper proposes WWDC (What We Don't C), a method that employs conditionally guided latent flow matching to remove known information from existing VAE representations, enabling unknown features to be more readily discovered and accessed in the residual manifold, thus facilitating iterative scientific discovery.

## Background & Motivation

### State of the Field

**State of the Field**: The core challenge of representation learning lies in how to access meaningful information within learned representations.

### Limitations of Prior Work

**Limitations of Prior Work**: Standard disentanglement methods (e.g., β-VAE, contrastive learning) attempt to separate all factors of variation into distinct dimensions, but struggle to achieve this on complex data.

### Root Cause

**Root Cause**: A key insight: rather than pursuing complete disentanglement, it is more practical to remove known information from existing representations, allowing the "unknown" to surface more readily.

### Starting Point

**Starting Point**: In scientific domains such as astronomy, dominant signals (e.g., galaxy morphology classes) often obscure secondary yet important signals.

## Method

### Overall Architecture
- Uses existing pre-trained VAE representations as the target distribution.
- Trains a conditional flow matching model with known features as guidance conditions.
- During reverse flow mapping to the base distribution, known features are suppressed while residual structure retains unknown features.
- Iterative loop: discover new features → incorporate into conditions → remove and continue exploration.

### Key Designs
1. **Conditional Flow Matching**:

    - Interpolates between the base distribution (standard Gaussian) and the target distribution (VAE latent space) using Gaussian optimal transport paths.
    - Trains a velocity field $u_t^\theta$ to approximate optimal transport trajectories.
    - Loss: $\mathcal{L}_{CFM} = \mathbb{E}_{t,X_0,X_1} \|u_t^\theta(X_t) - (X_1-X_0)\|^2$

2. **Classifier-Free Guidance (CFG)**:

    - Randomly drops conditioning information with probability $p_{cfg}$ (replacing it with a null vector).
    - At inference, combines guided and unguided velocities via weighted composition: $u_t^{CFG} = (1-\omega)u_t(x_1|x_t) + \omega u_t(x_1|x_t,y)$
    - The guidance weight ω controls the degree to which conditional information is removed.

3. **Reverse-Flow Representation**:

    - Maps VAE samples from $t=1$ back to $t=0$ (base distribution) via reverse flow.
    - Due to the optimal transport property of the guided flow, the base distribution preserves global structure while removing conditional information.
    - The KL constraint of the VAE keeps the latent space approximately Gaussian, naturally aligning it with the base distribution and reducing structural distortion.

### Loss & Training
- Conditional flow matching loss with $p_{cfg} \in [0.1, 0.2]$.
- Operates on frozen pre-trained VAE representations without retraining the VAE.
- The midpoint method is used for ODE solving.
- Different VAE configurations are used for different datasets (MNIST: β=1e-4, z∈R^64; Galaxy10: β=1e-6, z∈R^(4×32×32)).

## Key Experimental Results

### 2D Gaussian Experiment


### Main Results

| Guidance Weight ω | Class Mutual Information (t=0) | Distance Linear Interpretability R² (t=0) |
|-------------------|-------------------------------|------------------------------------------|
| 0 (no guidance)   | High                          | ~0.3                                     |
| 0.5               | Moderate                      | ~0.6                                     |
| 1.0 (full guidance) | ~0                          | ~1.0                                     |

### Colored MNIST Experiment


### Ablation Study

| Representation Space | Digit Classification Accuracy | Blue Regression R² |
|----------------------|------------------------------|--------------------|
| VAE (original)       | High                         | Low                |
| WWDC (guided removal of digit + red + green) | Significantly reduced | **Improved** |

### Galaxy10 Experiment
- Uses the "Round" class as the guidance target, successfully separating galaxy morphology features.
- Residual maps clearly reveal the removed structural features (spiral arms, bar structures, etc.).
- Background features and imaging artifacts are fully preserved throughout the guidance process.

### Key Findings
- Guidance weight ω=1 can almost completely remove the mutual information of conditional information.
- Removed information does not affect the recovery of unguided features: blue channel values remain recoverable by a linear model after removing digit, red, and green information.
- The near-Gaussian nature of VAE latent space and its natural alignment with the flow matching base distribution are critical to the method's effectiveness.
- Style transfer capability: after reverse flow, applying a different condition during forward flow preserves stylistic attributes such as stroke width, position, and color.

## Highlights & Insights
- The core idea is highly creative: rather than "discovering everything," the approach is to "remove the known and let the unknown emerge."
- The method is computationally efficient: it reuses existing VAEs and only requires training a lightweight flow matching model.
- The vision of an iterative discovery loop (Figure 1): annotate → conditional guidance → discover new features → continue the cycle.
- The application to astronomy has genuine impact: systematic exploration of complex features in large-scale survey data.
- The theoretical analysis is concise and effective: the mechanisms of information removal and preservation are explained through the lens of optimal transport and CFG.

## Limitations & Future Work
- The completeness of information removal depends on the accuracy of the flow matching model; in practice, conditional information may not be entirely eliminated.
- Linear probing as an evaluation method may be insensitive to nonlinear features.
- The Galaxy10 experiment uses only simple discrete class labels as conditions; the effectiveness of continuous features (e.g., redshift) remains unexplored.
- Alternative VAE backbones from other representation learning approaches (e.g., JEPA, contrastive learning) have not been explored.
- Performance at scale on larger datasets and higher-dimensional latent spaces has yet to be validated.

## Related Work & Insights
- The concept of "manifold disentanglement" draws a meaningful distinction from conventional "dimension-wise disentanglement."
- CFG in diffusion/flow models is here given a new role: not to enhance generation quality, but to achieve information separation.
- The approach offers important inspiration for systematic methods in scientific discovery: when one knows what matters, actively removing it can help uncover what is unknown.
- The method can be directly applied to galaxy feature exploration in large-scale astronomical surveys such as LSST.
- The natural alignment between the near-Gaussian VAE latent space and the flow matching base distribution is a key prerequisite for the method's success.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Highly original concept)
- Technical Contribution: ⭐⭐⭐⭐ (Simple yet effective methodology)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Complete validation chain from toy examples to real data)
- Writing Quality: ⭐⭐⭐⭐⭐ (Fluid narrative, excellent figures)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[NeurIPS 2025\] Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery](score-informed_neural_operator_for_enhancing_ordering-based_causal_discovery.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[NeurIPS 2025\] Why Diffusion Models Don't Memorize: The Role of Implicit Regularization](why_diffusion_models_dont_memorize_the_role_of_implicit_regularization.md)
- [\[NeurIPS 2025\] Diffusion-Based Electromagnetic Inverse Design of Scattering Structured Media](diffusion-based_electromagnetic_inverse_design_of_scattering_structured_media.md)

<!-- RELATED:END -->
