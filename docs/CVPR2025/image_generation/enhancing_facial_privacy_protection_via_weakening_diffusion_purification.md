---
title: >-
  [Paper Note] Enhancing Facial Privacy Protection via Weakening Diffusion Purification
description: >-
  [CVPR 2025][Image Generation][Facial Privacy Protection] This paper weakens the purification effect in the reverse diffusion process of LDMs by learning timestep-wise unconditional embeddings, and leverages self-attention map guidance to maintain structural consistency, achieving an average PSR of 79.17% on CelebA-HQ and LADN, while outperforming all competing methods in FID.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Facial Privacy Protection"
  - "Diffusion Models"
  - "Adversarial Examples"
  - "Diffusion Purification"
  - "Self-Attention Guidance"
date: 2026-05-08
content_hash: eaec2684d378a940
---

# Enhancing Facial Privacy Protection via Weakening Diffusion Purification

**Conference**: CVPR 2025  
**arXiv**: [2503.10350](https://arxiv.org/abs/2503.10350)  
**Code**: [https://github.com/parham1998/Facial-Privacy-Protection](https://github.com/parham1998/Facial-Privacy-Protection)  
**Area**: Diffusion Models / Facial Privacy Protection  
**Keywords**: Facial Privacy Protection, Diffusion Models, Adversarial Examples, Diffusion Purification, Self-Attention Guidance

## TL;DR
This paper weakens the purification effect in the reverse diffusion process of LDMs by learning timestep-wise unconditional embeddings, and leverages self-attention map guidance to maintain structural consistency, achieving an average PSR of 79.17% on CelebA-HQ and LADN, while outperforming all competing methods in FID.

## Background & Motivation

**Background**: As facial recognition (FR) technology becomes widely deployed, facial privacy protection is increasingly important. Targeted de-identification generates privacy-protecting images to impersonate a target identity, thereby hiding the original identity from FR systems. Mainstream methods include noise-based (PGD, MI-FGSM), makeup-based (AMT-GAN, DiffAM), and diffusion-based approaches.

**Limitations of Prior Work**: Noise-based methods generate perceptible perturbations that degrade visual quality. Makeup-based methods require reference images and need retrained models for each target identity. DiffProtect, as the first diffusion-based method, modifies the semantic code of Diff-AE, but alters the facial structure toward the target identity and is limited by the **diffusion purification effect**—the reverse denoising process gradually removes adversarial modifications as high-frequency noise, leading to a low protection success rate.

**Key Challenge**: The reverse process of diffusion models possesses an inherent purification capability—the denoising model trained entirely on clean images will "correct" any perturbation deviating from the clean data manifold. This directly contradicts the goal of adversarial protection, which requires preserving adversarial perturbations that denoising processes tend to remove.

**Goal**: (1) Weaken the diffusion purification effect to preserve adversarial modifications; (2) achieve a better balance between protection capability and visual quality; (3) avoid structural distortion caused by modifying semantic codes.

**Key Insight**: Inspired by Null-Text Inversion, the authors discover that learning timestep-wise unconditional embeddings (null-text embeddings) not only improves image reconstruction quality but, more importantly, **weakens the model's over-purification of inputs**. The learned embeddings guide the model to retain fine-grained details of the input (including adversarial perturbations) instead of simply denoising them.

**Core Idea**: A two-stage learning framework: the first stage learns unconditional embeddings to weaken the purification effect and maintain image quality, while the second stage directly optimizes the latent code of the LDM under the condition of frozen embeddings to generate protected images, utilizing self-attention guidance to maintain structural consistency.

## Method

### Overall Architecture
Input original facial image $x$, first obtain the noisy latent code $z_t$ through DDIM Inversion. First stage: learn the unconditional embeddings $\{\emptyset_i\}_{i=1}^t$ for each timestep during the reverse sampling process, minimizing the reconstruction loss to ensure high-quality reconstruction. Second stage: freeze the learned embeddings, optimize the adversarial latent code starting from $z_{adv} = z_t$, minimizing the adversarial loss (making the protected image close to the target identity in the FR model's feature space) and the structure preservation loss (aligning self-attention maps).

### Key Designs

1. **Null-Text Guidance**:

    - **Function**: Weaken the diffusion purification effect, improve reconstruction quality, and preserve adversarial perturbations.
    - **Mechanism**: In the DDIM sampling step $z_{t-1} = f(z_t, t, \emptyset_t)$, learn an independent unconditional embedding $\emptyset_t$ for each timestep $t$ by minimizing $\mathcal{L}_{rec} = \|z_{t-1} - \bar{z}_{t-1}\|_2^2$. The learned embeddings provide additional learning capacity for the U-Net, allowing it to remember fine-grained texture and structural details of the input rather than removing all deviations as noise. AdamW optimizer, learning rate 0.1, 20 iterations.
    - **Design Motivation**: Standard DDIM Inversion using a fixed null-text embedding leads to reconstruction degradation (blurring of skin and hair details). The learned embeddings implicitly provide a "preserve input" guidance signal to the reverse process, preventing adversarial modifications from being fully purified.

2. **Adversarial Latent Code Optimization**:

    - **Function**: Generate protected images capable of fooling FR models.
    - **Mechanism**: Under the black-box setting, optimize the adversarial loss $\mathcal{L}_{adv} = \frac{1}{K}\sum_{k=1}^K [1 - \cos(\mathcal{F}_k(x_p), \mathcal{F}_k(x_t))]$ using $K$ white-box surrogate FR models, where $x_p$ is the decoded protected image and $x_t$ is the target identity image. The optimization is conducted directly in the LDM latent space to modify $z_{adv}$ unconditionally (without applying $L_\infty$ norm constraints), since structural consistency is guaranteed by self-attention guidance. AdamW optimizer, learning rate 0.01, 35 iterations.
    - **Design Motivation**: DiffProtect modifies semantic codes and constrains modification magnitudes to maintain structure, which limits protection performance. The proposed method decouples structure preservation from the latent code to self-attention guidance, allowing the latent code to be optimized more freely to maximize protection effectiveness.

3. **Structure Preservation via Self-Attention Guidance**:

    - **Function**: Maintain structural consistency with the original image during unconstrained optimization of the latent code.
    - **Mechanism**: Compare the U-Net self-attention maps before and after modification during the DDIM sampling process: $\mathcal{L}_{str} = \|S(z_{adv}) - S(\bar{z}_t)\|_2^2$. Self-attention maps encode the geometric and shape information of the image (such as facial contours and organ positions). Constraining their consistency ensures that the protected image shares the same structure as the original, while identity-related modifications are concentrated on the level of texture. The total loss is $\mathcal{L} = \lambda_{adv}\mathcal{L}_{adv} + \mathcal{L}_{str}$, where $\lambda_{adv}=0.003$.
    - **Design Motivation**: Research shows that self-attention maps control the spatial layout and structure of images, while cross-attention maps control text-image alignment. Since this method does not use text conditions, self-attention is the optimal choice for structural preservation. Compared to $L_\infty$ constraints, self-attention guidance allows a larger room for latent code modification, improving protection performance.

### Loss & Training
- Stage 1: $\mathcal{L}_{rec} = \|z_{t-1} - \bar{z}_{t-1}\|_2^2$, 20-step DDIM Inversion, starting the reverse process from the 3rd timestep, learning with 20 iterations.
- Stage 2: $\mathcal{L} = \lambda_{adv}\mathcal{L}_{adv} + \mathcal{L}_{str}$, 35 iterations to optimize $z_{adv}$.
- Total generation time is about 15 seconds per image (on a single RTX 4090).

## Key Experimental Results

### Main Results: Black-box Protection Success Rate (PSR%)

| Method | Category | IRSE50 | IR152 | FaceNet | MobileFace | Average |
|------|------|--------|-------|---------|------------|------|
| TIP-IM | Noise | 54.40 | 37.23 | 40.74 | 48.72 | 50.06 |
| AMT-GAN | Makeup | 76.96 | 35.13 | 16.62 | 50.71 | 52.84 |
| CLIP2Protect | Makeup | 81.10 | 48.42 | 41.72 | 75.26 | 64.90 |
| DiffAM | Makeup | 92.00 | 63.13 | 64.67 | 83.35 | 77.88 |
| DiffProtect | Diffusion | 67.75 | 60.14 | 35.19 | 64.33 | 51.05 |
| **Ours** | **Diffusion**| **88.87** | **67.25** | **59.53** | **91.57** | **79.17** |

### Image Quality Comparison

| Method | PSR | FID↓ | PSNR↑ | SSIM↑ |
|------|-----|------|-------|-------|
| DiffAM | 77.88 | 26.10 | 20.53 | 0.886 |
| DiffProtect | 51.05 | 28.29 | 24.21 | 0.879 |
| **Ours** | **79.17** | **15.32** | **27.72** | **0.839** |

### Key Findings
- The proposed method improves the average PSR by approximately 28% compared to DiffProtect and by about 1.3% compared to DiffAM, while reducing FID from 26.10 to 15.32 (more natural images).
- The effect of unconditional embeddings is most significant at deeper timesteps ($t=5,7$)—without embeddings, both PSR and FID degrade dramatically.
- Removing self-attention guidance yields a higher PSR but significantly increases FID, proving that self-attention guidance acts as a balancer between protection capability and visual quality.
- Under adaptive attack tests against Gaussian and mean filtering, PSR only drops from 88.87% to 86.66% ($5\times5$ mean filter), showing robust performance.
- Confidence scores on Face++ and Tencent commercial APIs are higher than all competing methods.

## Highlights & Insights
- **Fighting Fire with Fire**: The approach cleverly exploits the diffusion model's purification attribute as an entry point, using unconditional embedding learning for "anti-purification". This shows that the purification ability of diffusion models is not uncontrollable and can be selectively guided to retain or remove information.
- **Decoupled Structure Preservation and Protection**: Moving structural constraints from the latent code to the self-attention space allows greater freedom for latent code modifications while the self-attention maps naturally encode facial structure. This decoupling strategy can be transferred to other tasks requiring semantic modification while maintaining structure.
- **Multi-faceted Role of Unconditional Embeddings**: Combining both the improvement of reconstruction quality and the weakening of purification to preserve adversarial perturbations. An elegant design solving two problems with one component.

## Limitations & Future Work
- **Generation Speed**: 15 seconds per image is still relatively slow and unsuitable for real-time applications. The authors mention that acceleration could be achieved by attacking directly in the UNet semantic space.
- **Suboptimal SSIM**: The SSIM metric is lower than DiffAM (0.839 vs 0.886), indicating some sacrifice in pixel-level consistency.
- **Surrogate Model Dependency**: Black-box attacks rely on the transferability of 3 white-box surrogate models; performance may drop if the target model has a significantly different architecture.
- **Ethical Considerations**: Impersonating physical identities poses ethical risks. In the discussion, the authors propose impersonating synthesized target identities as an alternative, achieving 85-90% PSR on four FR models.

## Related Work & Insights
- **vs DiffProtect**: DiffProtect modifies the semantic code of Diff-AE and constrains the magnitude of modification, leading to limited protection capability and structural distortion. This method optimizes unconditionally in the LDM latent space and replaces magnitude constraints with self-attention guidance, improving average PSR from 51.05% to 79.17%.
- **vs DiffAM**: DiffAM requires two diffusion models (makeup removal + makeup transfer) and needs retrained models for each target identity. The proposed method is more streamlined and outperforms DiffAM in FID (15.32 vs 26.10).
- **vs Null-Text Inversion (Mokady et al.)**: The original method is used for precise reconstruction in real-image editing. This work innovatively repurposes it as a tool to weaken the purification effect.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using null-text embedding learning for anti-purification is novel, and using self-attention guided structure preservation to replace $L_\infty$ constraints is also creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation across two datasets, four FR models, and nine competing methods, with extensive ablation analyses. Tests on commercial APIs and adaptive attacks are also included.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-organized, with convincing motivational derivations and rich illustrations.
- Value: ⭐⭐⭐⭐ Clear practical value for the field of facial privacy protection; the method's design principle can be generalized to other adversary-preserving tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)
- [\[CVPR 2025\] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification](divide_and_conquer_heterogeneous_noise_integration_for_diffusion-based_adversari.md)
- [\[CVPR 2025\] Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)
- [\[NeurIPS 2025\] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](../../NeurIPS2025/image_generation/perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [\[CVPR 2025\] Instant Adversarial Purification with Adversarial Consistency Distillation](instant_adversarial_purification_with_adversarial_consistency_distillation.md)

</div>

<!-- RELATED:END -->
