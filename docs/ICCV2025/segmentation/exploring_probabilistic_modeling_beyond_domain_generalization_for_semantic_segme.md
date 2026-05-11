---
title: >-
  [Paper Note] Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation
description: >-
  [ICCV 2025][Segmentation][domain generalization] This paper proposes PDAF (Probabilistic Diffusion Alignment Framework), which explicitly estimates a Latent Domain Prior (LDP) via probabilistic diffusion modeling to prov…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "domain generalization"
  - "semantic segmentation"
  - "diffusion models"
  - "latent domain prior"
  - "probabilistic modeling"
date: 2026-05-08
content_hash: 84916a8606dbf230
---

# Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.21367](https://arxiv.org/abs/2507.21367)
**Code**: [https://pdaf-iccv.github.io](https://pdaf-iccv.github.io)
**Area**: Image Segmentation
**Keywords**: domain generalization, semantic segmentation, diffusion models, latent domain prior, probabilistic modeling

## TL;DR

This paper proposes PDAF (Probabilistic Diffusion Alignment Framework), which explicitly estimates a Latent Domain Prior (LDP) via probabilistic diffusion modeling to provide domain-shift compensation for existing segmentation networks, achieving state-of-the-art cross-domain generalization without requiring paired target-domain samples.

## Background & Motivation

Domain Generalized Semantic Segmentation (DGSS) faces the core challenge of distribution shift between the training domain and unseen target domains. Existing methods fall into two main categories:

**Data augmentation methods**: Increase training data diversity through synthetic variations, but rely heavily on auxiliary domains or generative models.

**Domain-invariant representation learning methods**: Extract cross-domain consistent features, but entanglement between style and content leads to loss of critical semantic information.

Recent methods (e.g., SPC, DPCL, BlindNet) achieve alignment by projecting features into restricted feature spaces, yet overlook the intrinsic properties of the latent domain prior. This paper argues that the domain prior itself should be explicitly modeled rather than simply performing feature alignment.

## Method

### Overall Architecture

PDAF formalizes DGSS as a probabilistic learning problem, introducing a latent domain prior (LDP) variable $z$ to capture unobservable domain shifts. The predictive function is: $p_{\theta,\phi}(y_t|x_t) = \int p_\theta(y_t|x_t,z)\, p_\phi(z|x_t)\, dz$. PDAF is integrated into a pretrained segmentation model and uses paired source-domain and pseudo-target-domain images to simulate domain shift for LDP modeling. Three core modules correspond respectively to the variational posterior, the predictive model, and the prior estimation terms in the ELBO.

### Key Designs

1. **Latent Prior Extractor (LPE)**: Implements the variational posterior $q_\varphi(z|x_t, x_s)$. Source-domain features $h_{\vartheta,s}$ and pseudo-target-domain features $h_{\vartheta,t'}$ extracted by a frozen encoder are concatenated and passed through residual blocks to model cross-domain feature relationships; two projection layers then yield the mean $\mu$ and variance $\sigma$, from which the optimal LDP $\tilde{z} \in \mathbb{R}^{c' \times h \times w}$ ($c'=4$) is sampled via the reparameterization trick. Constraining $\tilde{z}$ to follow a standard normal distribution serves as a regularizer.

2. **Domain Compensation Module (DCM)**: Implements the predictive model $p_\theta(y_{t'}|x_{t'}, z')$. Inspired by Spatial Feature Transform (SFT), the LDP is projected into affine transformation parameters $\tilde{\gamma}$ and $\tilde{\beta} \in \mathbb{R}^{c \times h \times w}$ that scale and shift the feature representation: $\tilde{h}_{\theta,t'} = \tilde{\gamma} \odot h_{\theta,t'} \oplus \tilde{\beta}$, after which the segmentation head $D_\theta$ produces the segmentation map. This design compensates for domain shift while preserving task-relevant information.

3. **Diffusion Prior Estimator (DPE)**: Implements the prior $p_\phi(z'|x_t)$. This is the key innovation — probabilistic diffusion modeling is used to estimate the LDP without paired samples. The forward diffusion process adds noise to the optimal LDP $\tilde{z}$ to obtain $z_T$; the reverse process, conditioned on target features $h_{\theta,t'}$, denoises $z_T$ to obtain the estimated LDP $\hat{z}_0$. An accelerated diffusion schedule requiring only $T=4$ steps is adopted, enabling joint training with the segmentation head. At inference, only the target-domain image is needed: the DPE estimates the LDP from Gaussian noise, which the DCM then uses to compensate the features.

### Loss & Training

Total loss: $\mathcal{L}_\text{total} = \lambda_\text{task} \cdot \mathcal{L}_\text{task} + \lambda_\text{sc} \cdot \mathcal{L}_\text{sc} + \lambda_\text{prior} \cdot \mathcal{L}_\text{prior}$

- **$\mathcal{L}_\text{task}$**: Task loss — weighted cross-entropy for DeepLabV3+, focal loss for Mask2Former.
- **$\mathcal{L}_\text{sc}$**: Semantic consistency loss measuring the discrepancy between source-domain and pseudo-target-domain predictions, accelerating convergence.
- **$\mathcal{L}_\text{prior}$**: Prior alignment loss $= \|\hat{z}_0 - \tilde{z}\|_2$, aligning the outputs of DPE and LPE.
- Loss weights: $(\lambda_\text{task}, \lambda_\text{sc}, \lambda_\text{prior}) = (0.5, 0.5, 1.0)$.
- Training: Adam optimizer, lr = 1e-5, batch size = 4, 100 epochs.
- Diffusion parameters: $T=4$ steps, $\beta$ linearly annealed from 0.1 to 0.99.
- Hardware: single RTX 4090.

## Key Experimental Results

### Main Results

Cityscapes training → cross-domain evaluation (DeepLabV3+ / ResNet-50):

| Method | BDD (B) | Mapillary (M) | GTAV (G) | SYNTHIA (S) | Avg |
|--------|---------|--------------|---------|------------|-----|
| DeepLabV3Plus | 44.96 | 51.68 | 42.55 | 23.29 | 40.62 |
| BlindNet | 51.84 | 60.18 | 47.97 | 28.51 | 47.13 |
| **PDAF** | **53.50** | **62.93** | **50.54** | **30.68** | **49.41** |

Results on Mask2Former (Swin-L):

| Method | B | M | G | S | Avg |
|--------|---|---|---|---|-----|
| CMFormer | 62.60 | 73.60 | 60.70 | 43.00 | 59.98 |
| **PDAF** | **63.00** | **74.10** | **63.20** | **44.00** | **61.08** |

ACDC adverse-weather scenarios (Mask2Former Swin-L):

| Method | Foggy | Night | Rain | Snow | Avg |
|--------|-------|-------|------|------|-----|
| HGFormer | 69.90 | 52.70 | 72.00 | 68.60 | 65.80 |
| **PDAF** | **80.72** | **55.12** | **73.13** | **71.43** | **70.10** |

### Ablation Study

Contribution of each module (Cityscapes → other domains, DeepLabV3+ ResNet-50):

| Setting | B | M | G | S | Avg |
|---------|---|---|---|---|-----|
| Baseline | 44.96 | 51.68 | 42.55 | 23.29 | 40.62 |
| PDAF w/o LPE | 49.59 | 57.63 | 46.34 | 27.19 | 45.19 |
| PDAF w/o DCM | 51.51 | 60.79 | 49.42 | 29.55 | 47.82 |
| PDAF w/o DPE | 51.98 | 60.33 | 49.38 | 28.86 | 47.64 |
| **PDAF (full)** | **53.50** | **62.93** | **50.54** | **30.68** | **49.41** |

### Key Findings

- LPE contributes the most (+4.57 Avg), confirming that latent domain prior modeling is central.
- All three modules are indispensable; removing any one causes a drop of 1.5–4.2 mIoU.
- PDAF is effective across both CNN (DeepLabV3+) and Transformer (Mask2Former) architectures.
- Gains are especially pronounced on ACDC adverse weather (average +4.3 mIoU vs. HGFormer), indicating that LDP effectively compensates for degraded features.
- PDAF also generalizes when trained on the GTAV synthetic source domain, demonstrating synthetic-to-real transfer capability.

## Highlights & Insights

- **Theoretical elegance**: DGSS is formalized as a variational inference problem with a complete ELBO derivation; the three modules correspond to distinct terms in the ELBO.
- **Plug-and-play**: PDAF can be integrated into arbitrary existing segmentation models (frozen encoder + fine-tuned target network) without modifying the original architecture.
- **No paired samples at inference**: DPE learns to estimate the LDP from pure noise, requiring no source-domain data at inference time.
- The diffusion process requires only $T=4$ steps, introducing minimal overhead (trainable on a single RTX 4090) with a small increase in parameters and computation.
- The substantial improvement on adverse weather conditions (ACDC +4.3 mIoU) demonstrates strong practical value.

## Limitations & Future Work

- Pseudo-target-domain images are generated solely via photometric augmentation, limiting the diversity of augmentation strategies.
- The LDP channel dimension is fixed at $c'=4$; different domain-shift complexities may require adaptive dimensionality.
- Validation is currently limited to urban driving scenes; generalization to other domains (e.g., indoor, medical) remains to be verified.
- No comprehensive comparison with the latest DGSS methods based on Visual Foundation Models (VFMs) is provided.

## Related Work & Insights

- Compared to methods that exploit diffusion priors for image restoration (e.g., DiffIR, CDFormer), PDAF applies diffusion to domain prior estimation rather than image reconstruction, yielding a more lightweight design.
- Compared to methods that rely on external diffusion models (e.g., DatasetDM, DGInStyle), PDAF incorporates a lightweight internal diffusion module, avoiding substantial computational overhead.
- The probabilistic modeling paradigm for LDP is generalizable to other domain adaptation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introduces probabilistic diffusion modeling into domain generalization with a complete theoretical framework and a lightweight implementation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-backbone, and adverse-weather evaluations with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous mathematical derivations, clear module relationships, and high-quality figures.
- **Value**: ⭐⭐⭐⭐ A plug-and-play domain generalization enhancement solution with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[ICCV 2025\] On the Generalization of Representation Uncertainty in Earth Observation](on_the_generalization_of_representation_uncertainty_in_earth_observation.md)
- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/mrm_masked_representation_modeling_domain_adaptive.md)
- [\[ICCV 2025\] PartField: Learning 3D Feature Fields for Part Segmentation and Beyond](partfield_learning_3d_feature_fields_for_part_segmentation_and_beyond.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](../../NeurIPS2025/segmentation/towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)

</div>

<!-- RELATED:END -->
