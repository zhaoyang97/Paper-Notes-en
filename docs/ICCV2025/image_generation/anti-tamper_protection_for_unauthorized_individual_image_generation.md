---
title: >-
  [Paper Note] Anti-Tamper Protection for Unauthorized Individual Image Generation
description: >-
  [ICCV 2025][Image Generation][anti-tamper protection] This paper proposes Anti-Tamper Perturbation (ATP), which decouples protection perturbations (preventing forged generation) and authorization perturbations (detecting purified tampering) into separate frequency-domain regions. When an attacker attempts to purify the protective signal, the anti-tamper mechanism is triggered to deny service, achieving a 100% protection success rate against various purification attacks.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "anti-tamper protection"
  - "personalized image generation"
  - "adversarial perturbation"
  - "frequency-domain watermarking"
  - "image copyright"
date: 2026-05-08
content_hash: 47f277266eaf57ee
---

# Anti-Tamper Protection for Unauthorized Individual Image Generation

**Conference**: ICCV 2025
**arXiv**: [2508.06325](https://arxiv.org/abs/2508.06325)  
**Code**: [https://github.com/Seeyn/Anti-Tamper-Perturbation](https://github.com/Seeyn/Anti-Tamper-Perturbation)  
**Area**: AI Security / Image Generation Protection
**Keywords**: anti-tamper protection, personalized image generation, adversarial perturbation, frequency-domain watermarking, image copyright

## TL;DR
This paper proposes Anti-Tamper Perturbation (ATP), which decouples protection perturbations (preventing forged generation) and authorization perturbations (detecting purified tampering) into separate frequency-domain regions. When an attacker attempts to purify the protective signal, the anti-tamper mechanism is triggered to deny service, achieving a 100% protection success rate against various purification attacks.

## Background & Motivation
With the rapid development of personalized image generation techniques (e.g., DreamBooth, Textual Inversion), online service providers now offer convenient customized portrait generation. However, malicious actors can exploit these services to misuse others' photos for generating fabricated portraits, severely infringing on portrait rights and privacy.

Existing defenses inject **protection perturbations** to degrade the quality of generated images. However, this approach has a critical vulnerability: attackers can easily remove the protection perturbations via **purification techniques** (e.g., JPEG compression, rescaling, Gaussian blur), restoring the attack capability. Even ostensibly robust methods such as MetaCloak exhibit significant drops in protection success rate under purification attacks.

The core insight of ATP is: rather than attempting to make protection perturbations resist purification (an arms race), the problem should be reframed — **detect when purification occurs**. This is analogous to a physical tamper-evident seal: once broken (purified), the trace is detected by the service provider, who then refuses the request. The key challenge is that protection perturbations themselves alter image content, which is conceptually also a form of "tampering" — how can authorization perturbations be made immune to protection perturbations yet sensitive to purification operations?

## Method

### Overall Architecture
ATP consists of two components: protection perturbation $P_{Prot}$ and authorization perturbation $P_{Auth}$. The pipeline proceeds as follows: original image → BDCT transform to frequency domain → apply both perturbations in separate frequency-domain regions guided by a binary mask → BIDCT transform back to pixel domain → produce the protected authorized image. Downstream service providers verify the integrity of the authorization message before processing generation requests, refusing any tampered request.

### Key Designs

1. **Mask-Guided Perturbation Blending**:

    - Function: Ensures protection and authorization perturbations do not interfere with each other.
    - Mechanism: A binary mask $M$ in the frequency domain separates the operating regions of the two perturbations:
    $P_{AP}(I) = F^{-1}[M \odot P_{Auth}(F(I)) + (1-M) \odot P_{Prot}(F(I))]$
    - The mask $M$ is sampled from a Bernoulli distribution ($p=0.5$); regions with value 1 receive the authorization perturbation, and regions with value 0 receive the protection perturbation.
    - Advantage of frequency-domain operation: Since each pixel value is a linear combination of all frequency coefficients, both perturbations are uniformly distributed in the pixel domain and become indistinguishable, preventing selective purification by attackers.
    - Design Motivation: Directly separating perturbations in the pixel domain renders them distinguishable, allowing attackers to selectively purify only the protection perturbation.

2. **Block Discrete Cosine Transform (BDCT)**:

    - Function: Efficiently transforms the image into the frequency domain.
    - Core formula: The image is partitioned into non-overlapping $N\times N$ blocks ($N=16$), and DCT is applied to each block:
      $C_{u,v} = \alpha(u)\alpha(v)\sum_{i=0}^{N-1}\sum_{j=0}^{N-1}I_{i,j}\phi(u,i,N)\phi(v,j,N)$
    - The inverse transform BIDCT converts frequency-domain coefficients back to the pixel domain.
    - Design Motivation: Applying DCT to the entire image is computationally prohibitive; BDCT processes image blocks independently for greater efficiency.

3. **Authorization Perturbation**:

    - Function: Embeds an authorization message in the frequency domain as the basis for anti-tamper verification.
    - Mechanism: A convolutional autoencoder $f_\theta$ embeds a binary authorization message $m$ (of length $L$) into the mask-designated frequency-domain regions, while a message decoder $D_m$ is jointly trained to extract the message.
    - Encoding process: $g_\theta(C) = (1-M) \odot C + M \odot f_\theta(M \odot C, m)$, $I_{enc} = F^{-1}[g_\theta(F(I))]$
    - Loss function: $\mathcal{L} = \mathcal{L}_{con} + \lambda_{adv}\mathcal{L}_{adv,G} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{reg}\mathcal{L}_{reg}$
    - Includes a message consistency loss $\mathcal{L}_{con} = \|D_m(M \odot f_\theta(C)) - m\|_2^2$ and a frequency-domain regularization term $\mathcal{L}_{reg} = \|f_\theta(C) - C\|_2^2$.
    - Distinction from traditional watermarking: Conventional watermarks pursue robustness (resistance to purification), whereas the authorization perturbation takes the opposite approach — it must be **sensitive to purification**.
    - Design Motivation: Embeds verifiable information such that purification destroys message integrity, thereby triggering an alert.

4. **Improved Frequency-Domain PGD (Protection Perturbation)**:

    - Function: Accurately generates protection perturbations in the frequency domain without interfering with the authorization region.
    - Original problem: The projection operator $\Pi(\cdot)$ and sign function $\text{sgn}(\cdot)$ in standard PGD operate in the pixel domain, inevitably affecting the frequency coefficients protected by the mask.
    - Improved approach (Algorithm 1): Moves the sign function and projection constraint into the frequency domain:
        - Compute pixel-domain gradient $\nabla$
        - Transform to the frequency domain and apply the mask: $\nabla_{freq} = M_p \odot F(\nabla)$
        - Apply the sign function and step size in the frequency domain
        - Enforce the projection constraint within the $\epsilon$-ball in the frequency domain
    - Design Motivation: Ensures that protection perturbations only modify the frequency coefficients designated by the mask, leaving the authorization perturbation region intact.

### Loss & Training
- The authorization perturbation network is trained on the FFHQ dataset (70,000 face images).
- Protection perturbations are evaluated on subsets of CelebA-HQ and VGGFace2 (50 identities per dataset, 8 images per identity).
- Base generative model: Stable Diffusion v2-1; personalization algorithm: DreamBooth.
- ATP can be integrated with any PGD-based protection perturbation algorithm (Anti-DB, AdvDM, CAAT, MetaCloak).

## Key Experimental Results

### Main Results (Protection Success Rate under Purification Attacks)

| Method | Clean | JPEG 50 | Resize 4x | GridPure |
|------|-------|---------|-----------|----------|
| Anti-DB | High | Drops | Drops | Drops |
| Anti-DB + **ATP** | High | **100%** | **100%** | **100%** |
| MetaCloak | High | Partial drop | Partial drop | Drops |
| MetaCloak + **ATP** | High | **100%** | **100%** | **100%** |

Protection performance without purification (CelebA-HQ):

| Method | CLIP-IQAC↓ | ISM↓ | FDFR↑ |
|------|-----------|------|-------|
| Anti-DB | -0.287 | 0.462 | 0.458 |
| Anti-DB+ATP | -0.314 | 0.465 | **0.521** |
| AdvDM | -0.336 | 0.417 | 0.664 |
| AdvDM+ATP | -0.362 | 0.412 | **0.668** |

### Ablation Study

| Fusion Scheme | Bit-error (×$10^{-3}$) | Notes |
|---------|----------------------|------|
| No BDCT + No Mask + No improved PGD | 349.84 | Direct pixel-domain blending; authorization message severely corrupted |
| BDCT + No Mask | 42.03 | Frequency domain helps but interference remains without mask |
| No BDCT + Mask | 360.31 | Pixel-domain masking offers poor protection |
| BDCT + Mask + Standard PGD | 81.72 | Standard PGD violates mask constraint |
| **Full ATP** | **0.47** | Combination of all three components drastically reduces Bit-error |

### Key Findings
- ATP enables all baseline methods to achieve 100% protection success rate under purification attacks.
- ATP does not degrade — and in fact marginally improves — protection performance in the absence of purification.
- Frequency-domain authorization perturbations embed information more effectively and exhibit higher sensitivity to purification than pixel-domain alternatives.
- The improved frequency-domain PGD is critical for enforcing the mask constraint (Bit-error reduced from 81.72 to 0.47).
- Adaptive attacks require an attacker to simultaneously know both the BDCT parameters and mask values to bypass the system (search space of approximately $2^{786414}$).
- ATP introduces less aesthetic degradation than the original protection perturbation methods (higher CLIP-IQAC, closer to original image quality).

## Highlights & Insights
- Paradigm shift: From "resisting purification" to "detecting purification," escaping the arms race between protection perturbations and purification techniques.
- Elegant frequency-domain separation design: Leverages the linear properties of DCT to achieve uniform distribution of both perturbations in the pixel domain while keeping them non-interfering in the frequency domain.
- Plug-and-play: ATP integrates with any PGD-based protection algorithm by modifying only the gradient descent procedure.
- The anti-tamper concept is inspired by the intuition behind physical tamper-evident designs (e.g., seals and anti-disassembly mechanisms).

## Limitations & Future Work
- Relies on the service provider actively performing verification — entirely ineffective if the attacker runs the generative model locally.
- Under adaptive attacks, if an attacker simultaneously obtains both the BDCT parameters and mask values (highly unlikely in practice), protection fails.
- Introduces additional computational overhead at deployment (requires executing the authorization verification pipeline).
- Future direction: Design authorization perturbations that automatically degrade generation quality upon tampering, eliminating the need for an explicit verification step.

## Related Work & Insights
- **vs MetaCloak**: MetaCloak attempts to make protection perturbations inherently resistant to purification (robustness route), yet its PSR still degrades; ATP achieves 100% through a detection mechanism.
- **vs FaceSigns**: Traditional watermarking pursues robustness (resistance to purification), whereas ATP requires precisely the opposite — high sensitivity to purification.
- **vs GridPure**: As the strongest purification attack, GridPure significantly reduces the effectiveness of various protection methods, yet cannot circumvent ATP's anti-tamper detection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to propose an anti-tamper mechanism for protection perturbations; paradigm innovation is significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation covering 4 protection algorithms, 3 purification methods, adaptive attacks, and aesthetic impact.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, figures are intuitive, and the technical narrative is well-structured.
- Value: ⭐⭐⭐⭐ Introduces a new defensive dimension for copyright protection of AI-generated content with practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] StableGuard: Towards Unified Copyright Protection and Tamper Localization in Latent Diffusion Models](../../NeurIPS2025/image_generation/stableguard_towards_unified_copyright_protection_and_tamper_localization_in_late.md)
- [\[NeurIPS 2025\] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](../../NeurIPS2025/image_generation/perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [\[NeurIPS 2025\] BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Edit](../../NeurIPS2025/image_generation/blurguard_a_simple_approach_for_robustifying_image_protection_against_ai-powered.md)
- [\[CVPR 2026\] Adapter Shield: A Unified Framework with Built-in Authentication for Preventing Unauthorized Zero-Shot Image-to-Image Generation](../../CVPR2026/image_generation/adapter_shield_a_unified_framework_with_built-in_authentication_for_preventing_u.md)
- [\[CVPR 2025\] Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](../../CVPR2025/image_generation/nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)

</div>

<!-- RELATED:END -->
