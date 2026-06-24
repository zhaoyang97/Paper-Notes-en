---
title: >-
  [Paper Note] Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model
description: >-
  [CVPR 2025][Image Generation][Distortion-Perception Tradeoff] This paper proposes a variance-scaled reverse diffusion process that controls the variance of reverse sampling via a single parameter $\lambda \in [0,1]$. This allows a single pre-trained score network to flexibly traverse the optimal solution of the distortion-perception tradeoff curve, with its optimality mathematically proven under conditional Gaussian distributions.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Distortion-Perception Tradeoff"
  - "Diffusion Models"
  - "Inverse Problems"
  - "Variance Scaling"
  - "MMSE Estimation"
date: 2026-05-08
content_hash: 2ccbbe5854640d71
---

# Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model

**Conference**: CVPR 2025  
**arXiv**: [2503.20297](https://arxiv.org/abs/2503.20297)  
**Code**: None  
**Area**: Diffusion Models  
**Keywords**: Distortion-Perception Tradeoff, Diffusion Models, Inverse Problems, Variance Scaling, MMSE Estimation

## TL;DR

This paper proposes a variance-scaled reverse diffusion process that controls the variance of reverse sampling via a single parameter $\lambda \in [0,1]$. This allows a single pre-trained score network to flexibly traverse the optimal solution of the distortion-perception tradeoff curve, with its optimality mathematically proven under conditional Gaussian distributions.

## Background & Motivation

**Background**: In the field of image restoration, a fundamental conflict exists where distortion (e.g., MSE/PSNR) and perceptual quality (i.e., whether the image looks natural) present an irreconcilable tradeoff. Existing algorithms either optimize MSE (such as regression methods) or focus on perceptual quality (such as GANs), operating at the two extremes of the DP plane.

**Limitations of Prior Work**: (1) Conditional GAN-based methods (e.g., PSCGAN) can traverse the tradeoff by multisample averaging or adjusting injected noise, but their training relies on paired data, and different noise levels require retraining; (2) Methods that linearly combine two extreme estimators require deploying two separate models, offering poor flexibility; (3) Neither approach can mathematically guarantee reaching the optimal DP curve.

**Key Challenge**: Existing methods are either inflexible (requiring retraining) or sub-optimal (unable to guarantee being on the DP curve), and none can be applied to different types of inverse problems with a single model.

**Goal**: Using a single pre-trained score-based diffusion model, flexibly and optimally traverse the entire DP tradeoff curve from the MMSE point to the perfect perception point at inference time by tuning a single parameter $\lambda$.

**Key Insight**: The reverse process of score-based diffusion models has two key properties: (1) when the reverse variance is set to 0, the mean propagation converges to the MMSE estimator; (2) when the true posterior variance is used, sampling comes from the posterior distribution (perfect perception). These two endpoints correspond exactly to the two extremes of the DP tradeoff.

**Core Idea**: By scaling the variance of the reverse process as $\lambda \mathbf{C}_k$ ($\lambda=0$ corresponds to MMSE, $\lambda=1$ corresponds to posterior sampling), one can continuously interpolate between the two extremes. This is proven to be the optimal solution for the DP tradeoff under conditional Gaussian distributions.

## Method

### Overall Architecture

The core of the method is modifying the standard conditional reverse diffusion process. Given a pre-trained score network $s_\theta(\mathbf{x}_k, k)$ and observation $\mathbf{y}$, the scaled variance $\lambda \mathbf{C}_k$ is used instead of the original variance in each reverse sampling step. The user only needs to choose $\lambda \in [0,1]$: $\lambda=0$ yields the minimum distortion (MMSE), and $\lambda=1$ yields the optimal perceptual quality. The entire process uses the exact same score network without any retraining.

### Key Designs

1. **Variance-Scaled Reverse Diffusion Process**:

    - **Function**: Controls the stochasticity of sampling by scaling the variance at each step in the reverse process, thereby adjusting the balance between distortion and perception.
    - **Mechanism**: Defines the scaled joint inference distribution $p_\lambda(\mathbf{x}_k|\mathbf{x}_{k+1}, \mathbf{y}) = \mathcal{N}(\boldsymbol{\mu}_k(\mathbf{x}_{k+1}, \mathbf{y}), \lambda \mathbf{C}_k)$, where the mean $\boldsymbol{\mu}_k$ remains unchanged (still determined by the conditional score), and only the variance is scaled. As $\bar{\alpha}_T \to 0$, the mean of the final marginal distribution converges to the MMSE estimator $\mathbb{E}[X_0|\mathbf{y}]$, and the variance converges to $\lambda \cdot \text{Cov}[X_0|\mathbf{y}]$.
    - **Design Motivation**: The elegance of this design lies in keeping the mean unchanged—no matter what value $\lambda$ takes, the mean always tends toward the MMSE, only the "dispersion" of sampling differs. This ensures that reconstruction results do not deviate from the optimal mean estimation.

2. **Conditional Gaussian Optimality Proof (Theorem 2)**:

    - **Function**: Proves that variance-scaled sampling achieves the theoretical optimum of the DP tradeoff under conditional Gaussian distributions.
    - **Mechanism**: For a conditional Gaussian distribution $p_{X|Y}(\mathbf{x}|\check{\mathbf{y}}) \sim \mathcal{N}(\boldsymbol{\mu}_{\check{\mathbf{y}}}, \boldsymbol{\Sigma}_{\check{\mathbf{y}}})$, using MSE and Wasserstein-2 distance as metrics, the optimal DP function is $D(P) = \text{Tr}(\boldsymbol{\Sigma}) + (\sqrt{\text{Tr}(\boldsymbol{\Sigma})} - P)^2$ (when $P \leq \sqrt{\text{Tr}(\boldsymbol{\Sigma})}$). At perfect perception ($P=0$), the optimal MSE is precisely twice that of the MMSE.
    - **Design Motivation**: This theoretical guarantee provides a solid foundation for the variance-scaling method, while quantifying the cost of improving perceptual quality—striving for perfect perception at most doubles the MSE.

3. **DPS-Based Conditional Score Approximation**:

    - **Function**: Implements the actual computation of variance-scaled reverse sampling.
    - **Mechanism**: Approximates the conditional score using the Denoising Posterior Sampling (DPS) framework: $\nabla_{\mathbf{x}_k} \log p(\mathbf{x}_k|\mathbf{y}) \approx s_\theta(\mathbf{x}_k, k) + \frac{1}{\sigma_n^2} \nabla_{\mathbf{x}_k} \|\mathbf{y} - \mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_k))\|_2^2$. Tweedie's formula is used to estimate $\hat{\mathbf{x}}_0$. A hyperparameter $\zeta_{k,\lambda}$ is introduced to control the weight of the conditional score.
    - **Design Motivation**: The DPS framework naturally fits this work's requirements—it only requires a pre-trained unconditional score network to handle arbitrary measurement operators $\mathcal{A}$, avoiding the need for retraining for different inverse problems.

### Loss & Training

This method does not involve additional training. An off-the-shelf, pre-trained score network is utilized. During inference, one only needs to set the value of $\lambda$ in Algorithm 1: each step performs $\mathbf{x}_{k-1} = \frac{1}{\sqrt{\alpha_k}}(\mathbf{x}_k + (1-\alpha_k)(\hat{s} + \zeta_{k,\lambda} \hat{c}(\hat{\mathbf{x}}_0))) + \lambda \tilde{\sigma}_k \mathbf{z}$, where $\lambda$ is the only parameter requiring tuning.

## Key Experimental Results

### Main Results

Mixture of Gaussians experiment:
- $\lambda=0$: Trajectories deterministically converge to the MMSE point.
- Increasing $\lambda$: Reconstruction distribution gradually approaches the true posterior.
- $\lambda=1$: Reconstruction distribution matches the true posterior, with MSE approximately twice that of $\lambda=0$ (consistent with theory).

FFHQ 256×256 Face Dataset (Gaussian Blur + Super-Resolution):

| Method | Flexibility | Retraining Required | DP Curve Coverage |
|------|--------|-----------|-------------|
| PSCGAN-N | Limited | Yes (each noise level) | Partial |
| PSCGAN-z | Limited | Yes | Partial |
| DiffPIR | Fixed point | No | Single point |
| **Ours** | **Fully flexible** | **No** | **Full curve** |

### Ablation Study

DP tradeoff on 2D datasets (Pinwheel, S-curve, Moon):

| $\lambda$ | MSE Trend | Wasserstein-2 Trend | Behavior |
|-----------|---------|------------------|------|
| 0 | Minimum | Maximum | Deterministic, converges to MMSE |
| 0.3 | Slightly increases | Significantly decreases | Mild stochasticity |
| 0.8 | Moderately increases | Near zero | Stronger stochasticity |
| 1.0 | ~2×MMSE | Zero | Perfect perception, matches posterior |

### Key Findings

- The variance-scaling method covers a larger range of the DP tradeoff compared to PSCGAN on all 2D datasets.
- On FFHQ, a single score network can simultaneously handle two inverse problems (Gaussian blur and super-resolution) with different measurement operators, whereas PSCGAN requires separate training for each problem type.
- The MSE at $\lambda=1$ is approximately twice that of $\lambda=0$, highly consistent with theoretical predictions.
- The hyperparameter $\zeta_{k,\lambda}$ may need adjustments for different $\lambda$, but the overall method is relatively insensitive to it.

## Highlights & Insights

- The **elegant unification of theory and practice** is the greatest highlight of this paper. Theorems 1 and 2 provide a solid theoretical foundation, while the practical modification in Algorithm 1 is extremely simple—just scaling the standard deviation of noise in standard reverse sampling.
- The **"2x MMSE" Rule** is an important theoretical insight—the cost of pursuing perfect perceptual quality is at most doubling the MSE, providing quantitative guidance for tradeoff decisions in practical applications.
- The flexibility of **single-model multi-tasking** is highly practical: one score network + different $\lambda$ + different measurement operators = comprehensive coverage of solutions under various DP tradeoffs for multiple inverse problems.

## Limitations & Future Work

- Theoretical optimality strictly holds only under conditional Gaussian distributions, while it provides an approximate optimum for real image distributions (which are non-Gaussian).
- The approximation of the conditional score in the DPS framework (Jensen gap) may be imprecise in high-noise environments.
- The hyperparameter $\zeta_{k,\lambda}$ requires heuristic tuning, lacking theoretical guidance.
- The sampling process requires a full $T$-step reverse diffusion, resulting in high computational costs. Faster methods like DiffPIR can complete sampling in fewer steps but cannot flexibly traverse the DP tradeoff.
- Only Gaussian blur and super-resolution inverse problems are evaluated; the performance on tasks like inpainting or compression remains to be verified.

## Related Work & Insights

- **vs PSCGAN**: PSCGAN traverses the DP tradeoff via multi-sample averaging or tuning generator noise, but requires paired training data and retraining for each noise level. The proposed method is fully controlled during inference and guarantees theoretical optimality.
- **vs [Ohayon 2023]**: That work theoretically analyzes the DP tradeoff in Wasserstein space and proposes a linear combination of two extreme estimators, but requires training/deploying two separate models. This work naturally achieves continuous traversal via variance scaling using a single model.
- **vs DiffPIR**: DiffPIR accelerates sampling based on DDIM, but can only reach a fixed point on the DP plane. The proposed method can reach any point on the curve by varying $\lambda$.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to theoretically prove that a single diffusion model can optimally traverse the DP tradeoff.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid theoretical validation, but real-world image experiments are limited to FFHQ 256×256.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations, but the paper is highly theory-oriented and experimental evaluations could be further expanded.
- Value: ⭐⭐⭐⭐⭐ Provides a unified theoretical framework and practical tool for applying diffusion models to inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](../../ICML2026/image_generation/stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)
- [\[CVPR 2025\] CustAny: Customizing Anything from A Single Example](custany_customizing_anything_from_a_single_example.md)
- [\[CVPR 2025\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)

</div>

<!-- RELATED:END -->
