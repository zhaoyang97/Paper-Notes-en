---
title: >-
  [Paper Note] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models
description: >-
  [ICLR 2026][Image Generation][Model-heterogeneous federated learning] FedVTC proposes that, in model-heterogeneous federated learning, each client generates synthetic data via a Variational Transposed Convolution network (VTC) from aggregated feature distribution statistics to fine-tune the local model. Without requiring a public dataset, the method significantly improves generalization while reducing communication and memory overhead.
tags:
  - ICLR 2026
  - Image Generation
  - Model-heterogeneous federated learning
  - variational transposed convolution
  - synthetic data fine-tuning
  - feature distribution alignment
  - communication efficiency
date: 2026-05-08
content_hash: 84258f5dceafdd48
---

# Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models

**Conference**: ICLR 2026
**arXiv**: [2508.01669](https://arxiv.org/abs/2508.01669)
**Code**: N/A
**Area**: Federated Learning / Generative Models
**Keywords**: Model-heterogeneous federated learning, variational transposed convolution, synthetic data fine-tuning, feature distribution alignment, communication efficiency

## TL;DR
FedVTC proposes that, in model-heterogeneous federated learning, each client generates synthetic data via a Variational Transposed Convolution network (VTC) from aggregated feature distribution statistics to fine-tune the local model. Without requiring a public dataset, the method significantly improves generalization while reducing communication and memory overhead.

## Background & Motivation
**State of the Field**: Data heterogeneity in federated learning leads to poor generalization of local models. Conventional methods improve this through regularization or weight adjustment, but uniformly assume homogeneous client model architectures.

**Limitations of Prior Work**:
- Knowledge distillation methods require a public dataset, which is typically unavailable in practice.
- Performing knowledge distillation in feature space can only debias the classification head, without improving the feature extractor.
- Prototype-sharing methods regularize only the feature extractor while neglecting the classification head.
- Proxy-model methods incur high communication and memory overhead.

**Root Cause**: Model heterogeneity precludes parameter sharing for aggregation, yet clients still need to benefit from global information to improve generalization.

**Paper Goals**: Simultaneously debias both the feature extractor and the classification head without relying on a public dataset or sharing model parameters.

**Starting Point**: Clients share only the statistics of the feature distribution (mean + covariance), which are then used to guide the generation of synthetic data for full-model fine-tuning.

**Core Idea**: Generate synthetic images from the global feature distribution via a variational transposed convolution network, and perform full-model fine-tuning on the local model to simultaneously debias both the feature extractor and the classification head.

## Method

### Overall Architecture
Each client $k$ maintains a local model $f_k = h_k \circ g_k$ (feature extractor + classification head) and a VTC model $\psi_k$. The training pipeline proceeds as follows: (1) locally train $f_k$ and $\psi_k$; (2) upload per-class mean prototypes $\mathbf{c}_k^y$ and standard deviations $\boldsymbol{\sigma}_k$ to the server; (3) the server aggregates them into global prototypes $\mathbf{c}^y$ and global standard deviations $\boldsymbol{\sigma}$; (4) clients sample latent variables from the global distribution and generate synthetic data via the VTC; (5) the local model is fine-tuned on the synthetic data.

### Key Designs

1. **Variational Transposed Convolution Network (VTC)**:

    - **Function**: Generates synthetic image samples from low-dimensional Gaussian latent variables.
    - **Mechanism**: Similar to a VAE decoder, but employs transposed convolutions as the upsampling architecture. The input $\mathbf{v} = \mathbf{z} + \boldsymbol{\sigma}_k \odot \boldsymbol{\epsilon}$ (reparameterization trick) yields synthetic image $\mathbf{x}' = \psi_k(\mathbf{v})$.
    - **Training Objective**: Maximize the ELBO = reconstruction loss + KL divergence (aligning the local feature distribution to global prototypes).

2. **Distribution Matching Regularization (DM Loss)**:

    - **Function**: Enhances the robustness of the VTC to diverse input latent variables.
    - **Mechanism**: Introduces a Distribution Matching loss to ensure the VTC generates high-quality samples even when presented with latent variables sampled from the global distribution rather than the local one.
    - **Design Motivation**: During local training, the VTC is exposed only to the local feature distribution; without this regularization, generation quality degrades when latent variables are sampled from the global distribution.

3. **Full-Model Fine-Tuning Strategy**:

    - **Function**: Fine-tunes the entire local model $f_k$ using synthetic data (not just the classification head).
    - **Mechanism**: Synthetic data passes through the full forward pass of the model, enabling simultaneous debiasing of both the feature extractor and the classification head.
    - **Design Motivation**: Compared to feature-space distillation (which debiases only the classification head) and prototype sharing (which debiases only the feature extractor), FedVTC unifies both objectives through image-level synthetic data.

4. **Communication Efficiency**:

    - **Function**: Clients exchange only feature distribution statistics with the server.
    - **Transmitted Content**: Per-class mean prototypes $\mathbf{c}_k^y \in \mathbb{R}^p$ and standard deviations $\boldsymbol{\sigma}_k \in \mathbb{R}^p$.
    - **Communication Cost**: Far lower than transmitting model parameters or full generative models.

### Loss & Training
- VTC training loss: $\mathcal{L}_e = \mathcal{L}_{rc} + D_{KL} + \mathcal{L}_{DM}$
- Reconstruction loss: $\mathcal{L}_{rc} = \|\mathbf{x}' - \mathbf{x}\|_2^2$
- KL divergence: aligns the local feature distribution to global prototypes.
- Local model fine-tuning: cross-entropy loss on synthetic data.
- The VTC and local model are trained alternately, avoiding additional GPU memory consumption.

## Key Experimental Results

### Main Results — Generalization Accuracy in Model-Heterogeneous FL

| Method | MNIST | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|--------|-------|----------|-----------|---------------|
| FedGH (representation sharing) | Moderate | Moderate | Low | Low |
| FedKD (knowledge distillation) | Requires public data | Requires public data | Requires public data | Requires public data |
| **FedVTC** | **Highest** | **Highest** | **Highest** | **Highest** |

### Ablation Study

| Configuration | Generalization Accuracy |
|---------------|------------------------|
| FedVTC (full) | Best |
| w/o DM Loss | Degraded (VTC is not robust to global distribution sampling) |
| w/o full-model fine-tuning (classification head only) | Significant drop |
| w/o KL alignment | Significant drop |

### Key Findings
- **Full-model fine-tuning vs. partial alignment**: Fine-tuning the entire model with synthetic data substantially outperforms aligning only the feature space or the classification head.
- **DM Loss is critical**: Without DM Loss, the quality of VTC-generated synthetic data degrades severely under global distribution sampling.
- **Communication efficiency**: FedVTC incurs far lower communication cost than methods requiring model parameter or proxy model transmission.
- **More pronounced advantage on large-scale datasets (Tiny-ImageNet)**: Demonstrates strong scalability of the proposed method.

## Highlights & Insights
- **Synthetic data as a knowledge transfer medium**: Rather than transmitting model parameters or raw data, FedVTC indirectly transfers global knowledge by sharing distribution statistics and generating synthetic data locally—elegantly balancing privacy protection and knowledge sharing.
- **Unified debiasing of both components**: Prior methods debias either the feature extractor or the classification head; FedVTC naturally unifies both through image-level operations.
- **Lightweight design**: The VTC is a simple transposed convolution network trained alternately with the local model, requiring no additional GPU memory.

## Limitations & Future Work
- The quality of VTC-generated images may be limited (simple transposed convolution vs. stronger generators such as diffusion models).
- Per-class feature distributions are assumed to be Gaussian, which may not capture more complex real-world distributions.
- Covariance estimation may be inaccurate when the feature dimension $p$ is large.
- Privacy risks are not analyzed—whether feature means and covariances can be exploited to reconstruct raw data remains an open question.

## Related Work & Insights
- **vs. FedGH/FedTGP**: These methods share prototypes to regularize only the feature extractor while neglecting the classification head; FedVTC simultaneously debiases both via synthetic data.
- **vs. FedZKD/FedGen**: These methods perform knowledge distillation in feature space, debiasing only the classification head; FedVTC operates at the image level.
- **vs. FedMAN**: This method transmits a hypernetwork-based proxy model with high communication overhead; FedVTC transmits only distribution statistics.

## Rating
- Novelty: ⭐⭐⭐⭐ Using the VTC as a synthetic data generator within federated learning is a novel combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 4 datasets with multiple heterogeneous baselines and ablations; reasonably comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Logic is clear and comparisons are well articulated.
- Value: ⭐⭐⭐⭐ Addresses a practical pain point in model-heterogeneous FL with high communication efficiency.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICCV 2025\] FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning](../../ICCV2025/image_generation/feddifrc_unlocking_the_potential_of_text-to-image_diffusion_models_in_heterogene.md)
- [\[CVPR 2026\] Heterogeneous Decentralized Diffusion Models](../../CVPR2026/image_generation/heterogeneous_decentralized_diffusion_models.md)
- [\[ICLR 2026\] Bridging Degradation Discrimination and Generation for Universal Image Restoration](bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)
- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)

<!-- RELATED:END -->
