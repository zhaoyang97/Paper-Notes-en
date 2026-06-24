---
title: >-
  [Paper Note] Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization
description: >-
  [ICML2025][Image Generation][Distribution Matching] This paper proposes an expressive, score-based prior distribution (SAUB) that sidesteps prior density estimation via the Score Function Substitution (SFS) technique, combined with Gromov-Wasserstein geometry-preserving constraints to achieve stable and efficient distribution matching, yielding superior performance in fair classification, domain adaptation, and domain translation tasks.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Distribution Matching"
  - "Score-based Prior"
  - "VAE"
  - "Gromov-Wasserstein"
  - "Geometry-Preserving Regularization"
  - "CLIP Semantic Space"
date: 2026-05-08
content_hash: daf80b2bc7c283e0
---

# Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization

**Conference**: ICML2025  
**arXiv**: [2506.14607](https://arxiv.org/abs/2506.14607)  
**Code**: [inouye-lab/SAUB](https://github.com/inouye-lab/SAUB)  
**Area**: Generative Models / Distribution Matching  
**Keywords**: Distribution Matching, Score-based Prior, VAE, Gromov-Wasserstein, Geometry-Preserving Regularization, CLIP Semantic Space

## TL;DR

This paper proposes an expressive, score-based prior distribution (SAUB) that sidesteps prior density estimation via the Score Function Substitution (SFS) technique, combined with Gromov-Wasserstein geometry-preserving constraints to achieve stable and efficient distribution matching, yielding superior performance in fair classification, domain adaptation, and domain translation tasks.

## Background & Motivation

Distribution Matching (DM) is a general technique for domain-invariant representation learning widely applied in tasks such as fair classification, domain adaptation, and domain translation. Existing methods suffer from the following issues:

- **Non-parametric methods** (e.g., MMD, Sinkhorn): Poor scalability and low efficiency in high-dimensional scenarios.
- **Adversarial methods** (e.g., GAN): Unstable training, prone to mode collapse, and highly sensitive to hyperparameters.
- **Likelihood-based methods** (e.g., VAE): Typically employ fixed Gaussian priors, which lack expressiveness and lead to distorted encoder transformations.
- **Normalizing Flow priors**: Require the latent space to share the same dimensionality as the input, which limits flexibility.
- **LSGM** (Latent Score-based Generative Models): Must backpropagate through the diffusion model U-Net to compute Jacobians, suffering from unstable gradients at low noise levels.

Core Motivation: For gradient-based DM training, the likelihood objective actually **does not require the prior density itself**, but only its score function (the gradient of the log-probability). This key insight opens the door to replacing explicit density estimation with score-based models.

## Method

### 1. VAUB Objective

Under the VAUB (Variational Alignment Upper Bound) framework, the distribution matching objective can be decomposed into three terms:

$$\mathcal{L}_{\text{VAUB}} = \sum_d \left\{ \underbrace{\mathbb{E}_{q_\theta}[-\log p_\varphi(x|z,d)]}_{\text{Reconstruction Term}} - \underbrace{\mathbb{E}_{q_\theta}[-\log q_\theta(z|x,d)]}_{\text{Entropy Term}} + \underbrace{\mathbb{E}_{q_\theta}[-\log Q_\psi(z)]}_{\text{Cross-Entropy Term}} \right\}$$

where $q_\theta(z|x,d)$ is the encoder, $p_\varphi(x|z,d)$ is the decoder, and $Q_\psi(z)$ is the shared prior across domains.

### 2. Score Function Substitution (SFS) Technique

**Core Idea**: The encoder gradient of the cross-entropy term can be equivalently substituted using the score function:

$$\nabla_\theta \mathbb{E}_{z_\theta \sim q_\theta(z|x)}[-\log Q_\psi(z_\theta)] = \nabla_\theta \mathbb{E}_{z_\theta}\left[-\left(\nabla_{\bar{z}} \log Q_\psi(\bar{z})\big|_{\bar{z}=z_\theta}\right)^\top z_\theta\right]$$

Key point: The score function evaluation is treated as a **constant** (detached from the computation graph) when computing the encoder gradient, which results in:

- No need to compute the prior density $Q_\psi(z)$.
- No need to backpropagate through the score network (avoiding the instability of LSGM).
- Only requires a single forward pass to evaluate the score function.

### 3. SAUB Objective (Score-based Prior Alignment Upper Bound)

$$\mathcal{L}_{\text{SAUB}} = \sum_d \mathbb{E}_{z \sim q_\theta(z|x,d)}\left[-\log p_\varphi(x|z,d) + \log q_\theta(z|x,d) - \left(\nabla_{\bar{z}} \log Q_\psi(\bar{z})\big|_{\bar{z}=z}\right)^\top z\right]$$

It satisfies $\nabla_{\theta,\varphi} \mathcal{L}_{\text{VAUB}} = \nabla_{\theta,\varphi} \mathcal{L}_{\text{SAUB}}$, meaning the gradients with respect to the encoder and decoder parameters are mathematically identical.

### 4. Alternating Optimization Algorithm

The overall training is formulated as a bilevel optimization problem:

- **Upper level**: Fix the score prior, and optimize the encoder $\theta$ and decoder $\varphi$ (SAUB objective).
- **Lower level**: Fix the encoder/decoder, and update the score model $\psi$ using denoising score matching.

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{q_\theta}\left[\|S_\psi(\tilde{z}, \sigma_i) - \nabla_{\tilde{z}} \log q_{\sigma_i}(\tilde{z}|z)\|_2^2\right]$$

The variational bound becomes tight when the score prior perfectly matches the marginal posterior of the encoder.

### 5. Gromov-Wasserstein Geometry-Preserving Regularization

The total loss incorporates Gromov-Wasserstein (GW) constraints:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DM}} + \lambda_{\text{GW}} \cdot \mathbb{E}\left[\|d_X(x,x') - d_Z(z,z')\|_2^2\right]$$

Metric space choices:

- **GW-EP** (Euclidean-Preserving): Uses $L_2$ distance in both input and latent spaces, suitable for low-dimensional data.
- **GW-SP** (Semantic-Preserving): Uses pre-trained CLIP embeddings to compute distances in the input space, suitable for high-dimensional image data.

## Key Experimental Results

### Domain Adaptation (MNIST ↔ USPS)

| Model | MNIST→USPS | USPS→MNIST |
|------|-----------|-----------|
| ADDA | 89.4% | 90.1% |
| DANN | 77.1% | 73.0% |
| VAUB | 40.7% | 45.3% |
| **Ours w/o GW** | 88.1% | 85.5% |
| **Ours w/ GW-EP** | 91.4% | 92.7% |
| **Ours w/ GW-SP** | **96.1%** | **97.4%** |

The GW-SP variant achieves comprehensive improvements in bidirectional domain adaptation, outperforming ADDA by +6.7% / +7.3%.

### Domain Translation (CelebA Hair Color Conversion, Black ↔ Blonde Hair)

| Task / Model | Top-1 Retrieval | SSIM ↑ | LPIPS ↓ |
|----------|-----------|--------|---------|
| No GW | 5.0% | 0.393 | 0.431 |
| GW-EP | 4.0% | 0.428 | 0.371 |
| **GW-SP** | **9.0%** | **0.542** | **0.285** |

GW-SP nearly doubles the retrieval accuracy in the semantic space and substantially reduces LPIPS, demonstrating that semantic preservation is significantly superior to Euclidean-space alternatives.

### Stability Comparison (SFS vs LSGM)

- LSGM exhibits catastrophic instability at $\sigma_{\min}=0.001$ (divergent NLL and spiked reconstruction loss).
- SFS remains stable across all noise levels, achieving an NLL at $\sigma_{\min}=0.01$ that outperforms the best results of LSGM.
- SFS does not require backpropagating through the score network, yielding higher VRAM efficiency.

### Fair Classification (UCI Adult Dataset)

Under the same Demographic Parity (DP) gap, SAUB + GW-EP outperforms baselines such as FCRL, CVIB, VAUB, and LAFTR-DP, achieving a near-zero DP gap while maintaining high classification accuracy.

## Highlights & Insights

1. **SFS Technique is Extremely Simple**: The core idea is formulated in a single equation—substituting the cross-entropy gradient with the detached inner product of the score function—yet it entirely circumvents explicit density estimation and Jacobian computations.
2. **Key Difference from LSGM**: LSGM requires backpropagating through a U-Net (implicitly estimating Hessians), leading to instability at low noise levels; SFS only requires a forward score evaluation, ensuring consistently stable gradients.
3. **GW-SP is a Practical Innovation**: Introducing CLIP semantic embeddings into the GW distance metric space addresses the long-standing limitation of inadequate semantics in Euclidean distances for high-dimensional images.
4. **Strong Generalization**: The unified framework adaptively handles three completely distinct types of tasks: fair classification (tabular data), domain adaptation (handwritten digits), and domain translation (facial images).
5. **Convincing Synthetic Experiments**: Achieves excellent latent space alignment with only 20 samples, whereas Gaussian/MoG priors require 100+ samples.

## Limitations & Future Work

1. **Limited Image Generation Quality**: The domain translation experiment employs a simple VAE + diffusion architecture, leaving a quality gap behind the SOTA. The authors acknowledge this as a proof-of-concept.
2. **Insufficient Theoretical Guarantees for Alternating Optimization**: Approximating bilevel optimization with heuristic alternating optimization lacks formal convergence analysis.
3. **GW-SP Relies on Pre-trained CLIP**: The semantic preserving effect is bottlenecked by the quality of the CLIP model and is inapplicable to non-visual modalities.
4. **Small Experimental Scale**: Domain adaptation is evaluated only on MNIST ↔ USPS, lacking evaluation on larger benchmarks such as VisDA or DomainNet.
5. **Training Overhead of Score Models**: Although more efficient than LSGM, training the score network alternately still increases the overall training runtime.
6. **Evaluation Limited to Two-Domain Scenarios**: Scalability and performance for multi-domain (>2) distribution matching remain unexplored.

## Rating

- Novelty: ⭐⭐⭐⭐ — The SFS technique is clever and practical; the combination of a score-based prior and GW-SP is highly novel.
- Experimental Thoroughness: ⭐⭐⭐ — Covers multiple tasks but at a relatively small scale, lacking large-scale benchmarks.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations with a logically progressive motivation.
- Value: ⭐⭐⭐⭐ — Provides a stable and efficient new paradigm for distribution matching; the score prior concept has broad generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICML 2025\] Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers](elucidating_flow_matching_ode_dynamics_with_respect_to_data_geometries_and_denoi.md)
- [\[ICLR 2026\] Bridging the Distribution Gap to Harness Pretrained Diffusion Priors for Super-Resolution](../../ICLR2026/image_generation/bridging_the_distribution_gap_to_harness_pretrained_diffusion_priors_for_super-r.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](../../ICCV2025/image_generation/balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] Learning Few-Step Diffusion Models by Trajectory Distribution Matching](../../ICCV2025/image_generation/learning_few-step_diffusion_models_by_trajectory_distribution_matching.md)

</div>

<!-- RELATED:END -->
