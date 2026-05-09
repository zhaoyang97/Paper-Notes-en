---
title: >-
  [Paper Note] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers
description: >-
  [ICCV 2025][Image Generation][End-to-end training] This paper proposes REPA-E, the first training framework to successfully enable end-to-end joint tuning of a VAE and a latent diffusion model. By updating the VAE via the REPA alignment loss rather than the diffusion loss, REPA-E achieves a 17–45× training speedup and sets a new state of the art on ImageNet 256 (FID 1.12).
tags:
  - ICCV 2025
  - Image Generation
  - End-to-end training
  - VAE
  - Latent diffusion models
  - Representation alignment
  - Training acceleration
date: 2026-05-08
content_hash: 320cea8b536cd72b
---

# REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers

**Conference**: ICCV 2025
**arXiv**: [2504.10483](https://arxiv.org/abs/2504.10483)
**Code**: [https://end2end-diffusion.github.io](https://end2end-diffusion.github.io)
**Area**: Image Generation / Diffusion Models
**Keywords**: End-to-end training, VAE, Latent diffusion models, Representation alignment, Training acceleration

## TL;DR
This paper proposes REPA-E, the first training framework to successfully enable end-to-end joint tuning of a VAE and a latent diffusion model. By updating the VAE via the REPA alignment loss rather than the diffusion loss, REPA-E achieves a 17–45× training speedup and sets a new state of the art on ImageNet 256 (FID 1.12).

## Background & Motivation

**Background**: Latent diffusion models (LDMs) adopt a two-stage training paradigm—first training a VAE, then freezing it while training the diffusion model. REPA accelerates diffusion model training by aligning intermediate representations to features from pretrained encoders such as DINO.

**Limitations of Prior Work**: (1) The two-stage paradigm means the VAE latent space is never optimized for the generative task—the VAE is trained for reconstruction, which does not necessarily yield an optimal input space for the diffusion model. (2) Different VAEs exhibit distinct failure modes: SD-VAE latent spaces contain high-frequency noise, while self-trained IN-VAE latent spaces are over-smoothed. (3) Directly back-propagating the diffusion loss through the VAE leads to latent space collapse.

**Key Challenge**: While end-to-end training is generally preferable in deep learning, directly applying it in LDMs causes the diffusion loss to "hack" the latent space into an overly simple representation that is easy to denoise but degrades generation quality.

**Goal**: To identify an effective end-to-end training scheme that jointly optimizes the VAE and the diffusion model for maximal generative performance.

**Key Insight**: Analysis reveals that the REPA representation alignment score (CKNNA) strongly correlates with generation quality, and its upper bound is constrained by the VAE feature bottleneck. Improving VAE features through end-to-end training can therefore break this bottleneck.

**Core Idea**: Update the VAE using the REPA alignment loss rather than the diffusion loss. The REPA loss encourages the VAE latent space and diffusion model features to jointly align with pretrained visual representations, thereby avoiding latent space collapse while adaptively improving the structural quality of the VAE latent space.

## Method

### Overall Architecture
The total loss is $\mathcal{L} = \mathcal{L}_{\text{DIFF}}(\theta) + \lambda \mathcal{L}_{\text{REPA}}(\theta, \phi, \omega) + \eta \mathcal{L}_{\text{REG}}(\phi)$. The diffusion loss $\mathcal{L}_{\text{DIFF}}$ updates only the diffusion model parameters $\theta$, with a stop-gradient blocking gradients from reaching the VAE. The REPA loss jointly updates both the diffusion model $\theta$ and the VAE $\phi$. The VAE regularization loss preserves reconstruction capability.

### Key Designs

1. **End-to-End Back-Propagation Through the REPA Loss**:

    - Function: Jointly optimizes the VAE and diffusion model via the representation alignment loss.
    - Mechanism: $\mathcal{L}_{\text{REPA}}(\theta, \phi, \omega) = -\mathbb{E}[\frac{1}{N}\sum_n \text{sim}(y^{[n]}, h_\omega(h_t^{[n]}))]$, where $y$ denotes DINO-v2 features and $h_t$ denotes intermediate hidden states of the diffusion Transformer. Gradients of the REPA loss are back-propagated through the diffusion model to the VAE, encouraging it to produce latent representations more conducive to alignment.
    - Design Motivation: Updating the VAE with the diffusion loss causes collapse by incentivizing an overly simple latent space. The REPA loss instead encourages alignment with pretrained visual features, which improves rather than degrades latent space structure.

2. **Batch Normalization for Latent Space Normalization**:

    - Function: Provides differentiable, dynamic normalization between the VAE and the diffusion model.
    - Mechanism: Conventional LDMs normalize VAE outputs using precomputed global statistics. During end-to-end training, continuous VAE updates invalidate these statistics. A Batch Normalization layer with exponential moving averages replaces the fixed global statistics, eliminating the need for recomputation at each step.
    - Design Motivation: As VAE parameters evolve, the latent space distribution shifts, rendering fixed normalization constants ineffective. The BN layer provides lightweight adaptive normalization.

3. **Stop-Gradient on the Diffusion Loss**:

    - Function: Prevents the diffusion loss from corrupting the VAE latent space.
    - Mechanism: The diffusion loss $\mathcal{L}_{\text{DIFF}}$ is used exclusively to update diffusion model parameters $\theta$; a stop-gradient blocks its gradients from reaching VAE parameters $\phi$.
    - Design Motivation: Empirical analysis demonstrates that the diffusion loss incentivizes low-variance, simplistic latent spaces that are easier to denoise but yield poor generation quality, necessitating this gradient blocking.

### Loss & Training
The framework consists of three loss components: (1) the diffusion loss, which updates only the LDM; (2) the REPA loss, which jointly updates both the LDM and the VAE; and (3) VAE regularization (reconstruction + KL + GAN + LPIPS), which preserves the VAE's reconstruction capability.

## Key Experimental Results

### Main Results

| Method | Training Steps | gFID↓ | Speedup |
|--------|---------------|-------|---------|
| Vanilla SiT | 1.4M | 8.61 | 1× |
| REPA | 4M | 5.90 | ~1× |
| **REPA-E (400K)** | 400K | **4.07** | **17× vs. REPA, 45× vs. vanilla** |
| REPA-E (final) | — | **1.12** (w/ CFG) | SOTA |

### Ablation Study

| Configuration | gFID | Note |
|---------------|------|------|
| REPA-E (full) | Best | REPA loss end-to-end |
| End-to-end with diffusion loss | Degraded | Latent space collapse |
| Without BN layer | Unstable | Normalization failure |
| Without VAE regularization | Reconstruction degraded | Regularization required |
| Different VAEs (SD-VAE / IN-VAE) | Consistent gains | Strong generalization |

### Key Findings
- End-to-end training adaptively improves VAE latent spaces: high-frequency noise in SD-VAE is smoothed, while the over-smoothness of IN-VAE gains added detail—the same method automatically addresses distinct failure modes.
- The CKNNA alignment score strongly correlates with gFID (correlation > 0.9), validating its use as a proxy for generation quality.
- VAEs fine-tuned via end-to-end training serve as drop-in replacements for the original VAE and improve generation performance across different training configurations and model architectures.

## Highlights & Insights
- **Counterintuitive finding**: The diffusion loss cannot be used for end-to-end VAE training, whereas the REPA loss can. This reveals the fundamentally opposing effects of the two losses on latent space structure, constituting a theoretically significant insight.
- The **17–45× training speedup** is a highly practical contribution that substantially reduces the cost of large-scale diffusion model training.
- End-to-end training also improves the VAE itself, enabling its use as an enhanced tokenizer independently of the diffusion model.

## Limitations & Future Work
- Validation is limited to ImageNet 256×256; performance at higher resolutions and on larger datasets remains to be confirmed.
- VAE regularization terms (including GAN discriminator training) introduce additional implementation complexity.
- Compatibility has been verified only with the SiT architecture; applicability to DiT and other architectures warrants further investigation.
- REPA relies on pretrained features from DINO-v2, and alignment quality is inherently bounded by that model.

## Related Work & Insights
- **vs. REPA (Yu et al.)**: REPA aligns diffusion model features without updating the VAE; REPA-E jointly optimizes both via end-to-end training.
- **vs. LSGM (Vahdat et al.)**: LSGM prevents collapse using a variational lower bound and entropy term but converges slowly; REPA-E achieves the same goal more efficiently via the REPA loss.
- **vs. VA-VAE / DC-AE**: These works improve VAE architectures but remain within the two-stage paradigm; REPA-E is the first to realize genuine end-to-end training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First successful end-to-end training of VAE + LDM; reveals the opposing effects of diffusion loss vs. REPA loss on latent space structure.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Validated across multiple VAEs, model scales, and three dimensions: training speed, final performance, and VAE quality.
- Writing Quality: ⭐⭐⭐⭐⭐ — Three key insights are presented in a well-structured progression; PCA visualizations are highly intuitive.
- Value: ⭐⭐⭐⭐⭐ — FID 1.12 SOTA combined with 45× speedup represents a paradigm-level contribution to diffusion model training.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](end-to-end_multi-modal_diffusion_mamba.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[ICCV 2025\] Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity](semantic_watermarking_reinvented_enhancing_robustness_and_generation_quality_wit.md)
- [\[ICCV 2025\] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis](infgen_a_resolution-agnostic_paradigm_for_scalable_image_synthesis.md)

<!-- RELATED:END -->
