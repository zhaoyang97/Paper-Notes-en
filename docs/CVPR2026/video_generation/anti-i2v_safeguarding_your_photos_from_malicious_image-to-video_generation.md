---
title: >-
  [Paper Note] Anti-I2V: Safeguarding your photos from malicious image-to-video generation
description: >-
  [CVPR 2026][Video Generation][Adversarial Attack] Anti-I2V proposes a defense method against malicious image-to-video generation. By optimizing perturbations in both L\*a\*b\* color space and the frequency domain…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Adversarial Attack"
  - "Video Diffusion Models"
  - "Image Protection"
  - "Dual-Space Perturbation"
  - "Deep Feature Collapse"
date: 2026-05-08
content_hash: 64181637317e10c6
---

# Anti-I2V: Safeguarding your photos from malicious image-to-video generation

**Conference**: CVPR 2026
**arXiv**: [2603.24570](https://arxiv.org/abs/2603.24570)  
**Code**: None  
**Area**: Video Generation
**Keywords**: Adversarial Attack, Video Diffusion Models, Image Protection, Dual-Space Perturbation, Deep Feature Collapse

## TL;DR
Anti-I2V proposes a defense method against malicious image-to-video generation. By optimizing perturbations in both L\*a\*b\* color space and the frequency domain, and designing Internal Representation Collapse (IRC) and Anchor (IRA) losses to disrupt semantic feature propagation within the denoising network, the method achieves state-of-the-art protection across three architecturally distinct models: CogVideoX, DynamiCrafter, and Open-Sora.

## Background & Motivation
**Background**: Video diffusion models (VDMs) are advancing rapidly. Models such as CogVideoX and Open-Sora can generate realistic videos from a single image and a text prompt, introducing severe risks of deepfake misuse.

**Limitations of Prior Work**:
   - Existing defenses primarily target text-to-image generation or specific architectures (e.g., SVD), with limited validation on large-scale DiT/MMDiT models;
   - RGB-space perturbations are easily eliminated during multi-step denoising, resulting in insufficient robustness;
   - Most methods only attack the final output (VAE encoding or the tail of the denoising network), neglecting intermediate feature propagation.

**Key Challenge**: VDMs possess larger capacity and stronger temporal modeling; conventional perturbation strategies struggle to effectively interfere with them—necessitating deeper disruption strategies.

**Key Insight**: A two-pronged approach—optimizing perturbations in a more robust non-RGB space, while identifying semantically rich layers within the network and selectively disrupting their feature propagation.

**Core Idea**: L\*a\*b\* + frequency-domain dual-space perturbation + deep-to-shallow feature collapse + cross-layer semantic anchoring = effective attack against large-scale VDMs.

## Method

### Overall Architecture
Input image $x$ → caption generation via LVLM → reference video → dual-space perturbation optimization (L\*a\*b\* + DCT) → IRC and IRA losses + diffusion loss + auxiliary loss → output protected image $x_\xi$, causing severe degradation of VDM-generated video quality.

### Key Designs
1. **Dual-Space Perturbation (DSP)**:

    - **Function**: Optimizes adversarial noise in two non-RGB spaces: the L\*a\*b\* color space and the DCT frequency domain.
    - **Mechanism**:
        - L\*a\*b\* stage: Only the $a^*$ and $b^*$ channels (chrominance) are perturbed, leaving the $L^*$ luminance channel intact, making perturbations less perceptible to human observers.
        - DCT stage: Noise is injected into low-frequency DCT coefficients (which encode structural and textural information), disrupting deeper representations via frequency-domain perturbation.
        - The two stages alternate updates, with the final perturbation projected within the $\Delta_{RGB}$ constraint in RGB space.
    - **Design Motivation**: Pixel-level RGB perturbations are prone to being "washed away" during multi-step denoising. L\*a\*b\* is more perceptually uniform; low-frequency DCT coefficients correspond to core image structure, yielding more persistent perturbation effects.

2. **Internal Representation Collapse (IRC)**:

    - **Function**: Forces deep (semantically rich) layer features to degenerate toward shallow (low-semantic) layer features.
    - **Mechanism**:
        - PCA visualization reveals that high-level semantic features emerge after layer 19 in Open-Sora and after layer 27 in CogVideoX, while layer 3 exhibits almost no semantics.
        - Loss: $\mathcal{L}_{IRC}^{i,j} = \mathbb{E}\|\epsilon_\theta^j(z_t, z_\xi, t, y) - \epsilon_\theta^i(z_t, z_\xi, t, y)\|_2^2$
        - Features from the last 3 layers are aligned to those of layer 3.
    - **Design Motivation**: By collapsing deep semantic features, the denoising process loses the ability to reconstruct meaningful structure; the effect cascades to all frames via the attention mechanism.

3. **Internal Representation Anchor (IRA)**:

    - **Function**: At each layer of both the denoising module and the VAE, anchors the protected image's features to those of an unrelated target image.
    - **Mechanism**:
    $\mathcal{L}_{IRA} = \mathcal{L}_{IRA,\epsilon_\theta} + \mathcal{L}_{IRA,E}$
        - Denoising module level: $\|\epsilon_\theta^m(z_t, z_\xi, t, y) - \epsilon_\theta^m(z_t, z_\psi, t, y)\|_2^2$
        - VAE level: $\|E^n(z_\xi) - E^n(z_\psi)\|_2^2$
    - **Design Motivation**: Rather than merely collapsing semantics (IRC), IRA actively steers features toward an incorrect direction, providing a complementary and more effective dual disruption.

### Final Objective
$$\mathcal{L}_{Anti-I2V} = \mathcal{L}_{IRC} + \mathcal{L}_{IRA} + \mathcal{L}_{auxiliary} - \mathcal{L}_{DM}$$
- Auxiliary loss: CLIP feature distance maximization + LPIPS perceptual distance maximization

## Key Experimental Results

### Main Results (CelebV-Text Dataset)

| Model | Method | ISM↓ | C-FIQA↓ | Q-A(F)↓ | Q-A(V)↓ | DINO↓ |
|------|------|------|---------|---------|---------|-------|
| CogVideoX | Clean | 0.721 | 0.522 | 0.746 | 0.802 | 0.828 |
| CogVideoX | MIST | 0.561 | 0.463 | 0.476 | 0.577 | 0.750 |
| CogVideoX | **Anti-I2V** | **0.448** | **0.433** | **0.447** | **0.532** | **0.722** |
| DynamiCrafter | Clean | 0.528 | 0.467 | 0.724 | 0.794 | 0.622 |
| DynamiCrafter | AdvDM | 0.269 | 0.370 | 0.167 | 0.207 | 0.397 |
| DynamiCrafter | **Anti-I2V** | **0.151** | **0.303** | **0.032** | **0.047** | **0.167** |

### Ablation Study

| Configuration | ISM↓ | Q-A(V)↓ | Note |
|------|------|---------|------|
| RGB perturbation only | 0.583 | 0.543 | Baseline (similar to AdvDM) |
| + L\*a\*b\* | 0.521 | 0.511 | Color-space perturbation more effective |
| + DCT | 0.498 | 0.496 | Frequency domain further improves results |
| + IRC | 0.472 | 0.558 | Semantic collapse is effective |
| + IRA | 0.460 | 0.540 | Anchor loss provides complementary gains |
| Full Anti-I2V | **0.448** | **0.532** | All components synergize optimally |

### Key Findings
- The method achieves the most significant results on DynamiCrafter (UNet architecture), reducing Q-A(V) from 0.794 to 0.047.
- It also proves effective on CogVideoX (DiT architecture), validating cross-architecture generalization.
- A simple layer selection strategy (last 3 layers → layer 3) generalizes across different architectures.

## Highlights & Insights
- **First systematic study of adversarial perturbation optimization in non-RGB spaces**; the L\*a\*b\* + frequency-domain combination represents a promising new direction.
- The IRC loss is theoretically grounded in PCA analysis of layer-wise features in the denoising network.
- Applicable to three mainstream architectures (UNet, DiT, MMDiT), demonstrating strong practical utility.

## Limitations & Future Work
- Perturbation optimization still requires white-box access to the target model; black-box transferability has not been sufficiently validated.
- Robustness against image preprocessing (JPEG compression, blurring) warrants further analysis.
- Computational efficiency: PGD-based iterative perturbation optimization incurs significant computational overhead.

## Related Work & Insights
- The text-level loss shares conceptual similarities with MIST but extends the disruption to the layer level.
- The DSP approach can be generalized to other adversarial attack and defense scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dual-space perturbation and layer-wise feature collapse is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three VDM architectures × two datasets with comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Technical details are thorough; PCA analysis is intuitive.
- Value: ⭐⭐⭐⭐ Significant practical implications for AI safety and privacy protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)
- [\[ICCV 2025\] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](../../ICCV2025/video_generation/tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [\[CVPR 2026\] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)
- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](../../ICCV2025/video_generation/realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[CVPR 2026\] From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning](from_static_to_dynamic_exploring_self-supervised_image-to-video_representation_t.md)

</div>

<!-- RELATED:END -->
