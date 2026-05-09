---
title: >-
  [Paper Note] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Models] Under a Mixture of Low-Rank Gaussians (MoLRG) data model, this paper theoretically proves that the unimodal dynamics of representation quality across noise levels arise from a trade-off between denoising strength and class discriminability, and empirically demonstrates that the emergence of unimodal dynamics serves as a reliable indicator of model generalization.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion Models
  - Representation Learning
  - Unimodal Dynamics
  - Low-Rank Gaussian Mixture
  - Generalization-Memorization
date: 2026-05-08
content_hash: aba84b8c22c32358
---

# Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models

**Conference**: NeurIPS 2025
**arXiv**: [2502.05743](https://arxiv.org/abs/2502.05743)
**Code**: None
**Area**: Image Generation / Diffusion Model Theory
**Keywords**: Diffusion Models, Representation Learning, Unimodal Dynamics, Low-Rank Gaussian Mixture, Generalization-Memorization

## TL;DR

Under a Mixture of Low-Rank Gaussians (MoLRG) data model, this paper theoretically proves that the unimodal dynamics of representation quality across noise levels arise from a trade-off between denoising strength and class discriminability, and empirically demonstrates that the emergence of unimodal dynamics serves as a reliable indicator of model generalization.

## Background & Motivation

Diffusion models have achieved remarkable success not only in generative tasks but also, as recent studies reveal, in **representation learning**. Internal feature extractors of trained diffusion models can serve as powerful self-supervised learners, matching or even surpassing dedicated self-supervised methods on downstream tasks such as classification, semantic segmentation, and image alignment.

A widely observed phenomenon is **unimodal representation dynamics**: the quality of learned representations (measured by downstream task performance) follows a unimodal trend across noise levels—optimal features emerge at intermediate noise levels, while performance degrades for both fully noisy and fully clean inputs (as shown in Figure 1).

**Root Cause**: Although this phenomenon has been broadly observed, its underlying mechanism has lacked theoretical understanding. Specifically: (1) Why do intermediate noise levels yield the best representations? (2) What drives this unimodal pattern? (3) How does it relate to the model's generalization ability?

This paper addresses these questions under the **Mixture of Low-Rank Gaussians (MoLRG)** data assumption, using simplified yet analytically tractable network architectures.

## Method

### Overall Architecture

A three-step theoretical analysis framework:
1. Assume the data distribution follows MoLRG (capturing the low-dimensional manifold structure of natural images).
2. Design an analytically tractable denoising autoencoder (DAE) architecture that mimics the structural properties of U-Net.
3. Define a signal-to-noise ratio (SNR) metric for representation quality and derive its closed-form expression as a function of noise level.

### Key Designs

1. **MoLRG Data Model** (Assumption 1): The data distribution is a $K$-class noisy low-rank Gaussian mixture:
    $x_0 = U_k^\star a + \delta \tilde{U}_k^\star e, \quad \text{with probability } \pi_k$
    - $U_k^\star \in \mathcal{O}^{n \times d}$: orthonormal basis of the $k$-th class subspace (class-specific attributes)
    - $\tilde{U}_k^\star$: bases of other classes' subspaces (class-agnostic fine-grained attributes, e.g., background)
    - $\delta$: data noise level

   Motivation: (1) MoLRG captures the intrinsic low-dimensionality of real images; (2) the KL penalty in latent diffusion models encourages the latent space to approximate a Gaussian distribution; (3) the noise term models class-agnostic complexity.

2. **Network Parameterization**: The DAE and feature representation are parameterized as:
    $x_\theta(x_t, t) = U h_\theta(x_t, t), \quad h_\theta(x_t, t) = D(x_t, t) U^\top x_t$
    $D(x_t, t) = \text{diag}(\beta_1^t I_d, \ldots, \beta_K^t I_d)$

    where $\beta_l^t$ implements data- and time-dependent expert selection via softmax weights $w_l(x_t, t)$. This can be interpreted as a **shallow U-Net with block-wise Mixture-of-Experts (MoE)**, comprising three components: low-dimensional projection, expert weighting, and symmetric reconstruction.

3. **SNR Representation Quality Metric** (Definition 1):
    $\text{SNR}(\hat{x}_\theta, t) = \mathbb{E}_k\left[\frac{\mathbb{E}_{x_t}[\|U_k^\star \hat{h}_\theta(x_t, t)\|^2 | k]}{\mathbb{E}_{x_t}[\|\hat{x}_\theta(x_t, t) - U_k^\star \hat{h}_\theta(x_t, t)\|^2 | k]}\right]$
    The numerator measures the projection energy of the feature onto the correct class subspace (signal); the denominator measures the residual energy after removing the correct class projection (noise).

### Main Theoretical Results

**Proposition 1**: The optimal DAE admits a closed-form solution as a weighted projection:
$$\hat{x}_\theta^\star(x_t, t) = \sum_{l=1}^K w_l^\star(x_t, t)(\zeta_t U_l^\star U_l^{\star\top} + \xi_t \tilde{U}_l^\star \tilde{U}_l^{\star\top}) x_t$$
where $\zeta_t = 1/(1+\sigma_t^2)$ and $\xi_t = \delta^2/(\delta^2+\sigma_t^2)$. As $\sigma_t$ increases, $\xi_t$ decays much faster than $\zeta_t$, exhibiting a **coarse-to-fine generation transition**.

**Theorem 1 (Core)**: The SNR of the optimal DAE is approximated as:
$$\text{SNR}(\hat{x}_\theta^\star, t) \approx \frac{C_t}{(K-1)} \cdot \left(\frac{1 + \frac{\sigma_t^2}{\delta^2}h(\hat{w}_t^+, \delta)}{1 + \frac{\sigma_t^2}{\delta^2}h(\hat{w}_t^-, \delta)}\right)^2$$

Physical intuition behind unimodality:
- **Denoising rate** $\sigma_t^2/\delta^2$: monotonically increasing in $\sigma_t$
- **Positive-class confidence rate** $h(\hat{w}_t^+, \delta)$: monotonically decreasing in $\sigma_t$
- At small $\sigma_t$: class confidence remains stable and increasing denoising rate improves SNR
- At large $\sigma_t$: class confidence collapses sharply, $h(\hat{w}_t^+)$ approaches $h(\hat{w}_t^-)$, and SNR decreases
- An intermediate equilibrium exists where class-agnostic components are maximally suppressed and class-relevant features are best preserved → SNR peak

## Key Experimental Results

### Theoretical Validation

| Dataset | SNR Unimodal | Feature Probing Unimodal | Alignment | Notes |
|--------|---------|-------------------|---------|------|
| MoLRG Synthetic | ✓ | ✓ | ✓ | Theory and experiment match perfectly |
| CIFAR-10 | ✓ | ✓ | ✓ | SNR peak aligns with probing accuracy peak |
| TinyImageNet | ✓ | ✓ | ✓ | Same as above |

### Generalization-Memorization Experiments

| Training Data Size | UNet-32 Generalization Score | Representation Dynamics | Notes |
|-----------|-----------------|---------|------|
| $2^{15}$ (large) | High | Unimodal | Good generalization → unimodal |
| $2^{12}$ (medium) | Medium | Weakly unimodal | Transition regime |
| $2^8$ (small) | Low | Monotonically decreasing | Memorization → unimodal disappears |

### Training Dynamics Experiments ($N=2^{12}$)

| Training Stage | FID | Peak Probing Acc | Representation Dynamics | Notes |
|---------|-----|-----------------|---------|------|
| Early (Iter≤7.5M) | Decreasing | Increasing | Unimodal | Generalization phase |
| Late (Iter=15M+) | Increasing | Decreasing | Monotonically decreasing | Memorization phase |

### Key Findings

- The presence of unimodal representation dynamics is a **reliable indicator** of good generalization in diffusion models.
- The transition from unimodal to monotonically decreasing dynamics precisely corresponds to the generalization-to-memorization phase transition.
- FID and peak probing accuracy exhibit a consistent negative correlation.
- The transition in representation dynamics can serve as an **early stopping criterion** to prevent overfitting under limited data.

## Highlights & Insights

- First theoretical explanation of unimodal representation dynamics, revealing its origin in the trade-off between denoising strength and class discriminability.
- The SNR metric is concise and effective, aligning with probing accuracy on both synthetic and real data.
- The MoLRG data assumption, while simplified, is physically well-motivated (low-dimensional manifold + KL regularization + subspace structure).
- The connection between the generalization-memorization phase transition and representation dynamics constitutes an important practical insight.

## Limitations & Future Work

- The MoLRG data assumption limits the direct applicability of the theory, as real image distributions are far more complex than Gaussian mixtures.
- The network parameterization is highly simplified (equivalent to a shallow U-Net) and cannot be directly extended to practical deep U-Net architectures.
- The theory assumes orthogonal subspaces, equal dimensionality, and uniform mixing weights; relaxing these assumptions remains an open problem.
- The representation quality definition (SNR) relies on ground-truth subspace bases, which require PCA approximation in practical settings.

## Related Work & Insights

- **vs. Chen et al. (2024)**: The latter analyzes the optimization dynamics of diffusion learning in two-layer CNNs, focusing on the contrast between denoising and classification objectives, without addressing representation quality variation across timesteps.
- **vs. Wang et al. (2024)**: The latter also studies the effect of timesteps on diffusion representation learning but focuses on attribute classification and counterfactual generation without theoretical explanation.
- **vs. REPA (2024)**: REPA improves training efficiency by aligning diffusion features with pretrained self-supervised model features; the theoretical framework proposed in this paper can provide deeper understanding of this approach.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First theoretical explanation of one of the most important empirical phenomena in diffusion model representation learning
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic validation on both synthetic and real data; the generalization-memorization experiment is elegantly designed
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical intuition is clearly articulated (denoising rate vs. class confidence trade-off); figures are well-crafted
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation for diffusion model representation learning, potentially guiding more principled feature extraction and early stopping strategies

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [\[NeurIPS 2025\] Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems](preconditioned_langevin_dynamics_with_score-based_generative_models_for_infinite.md)
- [\[NeurIPS 2025\] Cross-fluctuation Phase Transitions Reveal Sampling Dynamics in Diffusion Models](cross-fluctuation_phase_transitions_reveal_sampling_dynamics_in_diffusion_models.md)
- [\[NeurIPS 2025\] Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics](elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)
- [\[NeurIPS 2025\] PID-controlled Langevin Dynamics for Faster Sampling of Generative Models](pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)

<!-- RELATED:END -->
