---
title: >-
  [Paper Note] NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness
description: >-
  [CVPR 2026][Image Restoration][Extreme-dark imaging] This paper proposes NEC-Diff, a diffusion-based event–RAW hybrid imaging framework that uses the illumination prior from RAW images to guide event denoising, and leverages the high-dynamic-range edges from denoised events to assist image denoising. Combined with dual-modality SNR-guided reliable information extraction and cross-modal attention diffusion, the method achieves high-quality dynamic scene reconstruction in extreme darkness (0.001–0.8 lux), reaching 24.51 dB PSNR on the REAL dataset.
tags:
  - CVPR 2026
  - Image Restoration
  - Extreme-dark imaging
  - event camera
  - RAW image
  - collaborative denoising
  - diffusion model
date: 2026-05-08
content_hash: 8d71e71edffc258f
---

# NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness

**Conference**: CVPR 2026
**arXiv**: [2603.20005](https://arxiv.org/abs/2603.20005)
**Code**: [https://github.com/jinghan-xu/NEC-Diff](https://github.com/jinghan-xu/NEC-Diff)
**Area**: Image Restoration / Low-Light Enhancement
**Keywords**: Extreme-dark imaging, event camera, RAW image, collaborative denoising, diffusion model

## TL;DR

This paper proposes NEC-Diff, a diffusion-based event–RAW hybrid imaging framework that uses the illumination prior from RAW images to guide event denoising, and leverages the high-dynamic-range edges from denoised events to assist image denoising. Combined with dual-modality SNR-guided reliable information extraction and cross-modal attention diffusion, the method achieves high-quality dynamic scene reconstruction in extreme darkness (0.001–0.8 lux), reaching 24.51 dB PSNR on the REAL dataset.

## Background & Motivation

1. **Background**: Low-light image enhancement methods are categorized into sRGB-based, RAW-based, event-based, and hybrid approaches. RAW-based methods model noise more accurately but cannot recover information lost under short exposure; event cameras offer high dynamic range but cannot restore smooth-region intensities.
2. **Limitations of Prior Work**: In extreme darkness (<1 lux), both modalities suffer from severe noise—RAW images face extreme photon-shot noise, while event cameras are dominated by shot noise that becomes the primary background activity at low light levels (with a density more than 50× higher than other noise types). Existing hybrid methods either ignore noise (EvRAW) or consider only single-modality SNR (EvLight), failing to effectively suppress noise.
3. **Key Challenge**: Under extremely low illumination, signal and noise become indistinguishable, and simple filtering or single-network denoising cannot simultaneously preserve weak signals and suppress noise.
4. **Goal**: How to effectively denoise two severely degraded modal signals and recover fine scene details?
5. **Key Insight**: Exploit the physical complementarity between RAW and event modalities—the linear illumination response of RAW can guide event denoising, while the denoised events provide high-dynamic-range edges that in turn assist image denoising.
6. **Core Idea**: Physics-constrained cross-modal collaborative denoising + SNR-guided adaptive fusion + high-fidelity reconstruction via diffusion models.

## Method

### Overall Architecture

Three modules in series: (1) **ECNS** (Event–RAW Collaborative Noise Suppression): RAW illumination prior guides event denoising → denoised events assist image denoising, with an intensity consistency constraint; (2) **SRIE** (SNR-guided Reliable Information Extraction): adaptively selects reliable features based on dual-modality SNR maps; (3) **CAD** (Cross-modal Attention Diffusion): bidirectional cross-attention fusion + conditional generation via diffusion model.

### Key Designs

1. **Event–RAW Collaborative Noise Suppression (ECNS)**:

    - Function: Achieves bidirectional denoising by exploiting cross-modal physical complementarity.
    - Mechanism: **Illumination-guided event denoising**—event shot noise density is positively correlated with illumination under low light (empirically verified); a coarse illumination prior is obtained from RAW images via Gaussian blurring and fed into an event denoising network (EDformer architecture) to guide denoising. **Event-assisted image denoising**—denoised events provide high-dynamic-range edge information, helping distinguish signal from noise in weakly textured regions and avoiding over-smoothing. **Intensity consistency loss**: derived from the physical model $\tilde{E}(t) = \frac{1}{C}\log\frac{\tilde{R}(t)}{\tilde{R}(t-\Delta t)}$, constraining the denoised RAW and events to satisfy the logarithmic relationship: $\mathcal{L}_{\text{cons}} = \|\hat{E}(t)\cdot C - \log\frac{\hat{R}(t)+\epsilon}{\hat{R}(t-\Delta t)+\epsilon}\|_1$
    - Design Motivation: Unlike prior work that directly fuses modalities or applies single-modality denoising, ECNS leverages the physical relationship between the two modalities for mutual denoising before fusion.

2. **SNR-guided Reliable Information Extraction (SRIE)**:

    - Function: Adaptively selects the most reliable modality at each spatial location based on signal reliability.
    - Mechanism: SNR maps are computed from the difference between inputs and denoised outputs: $M_{\text{SNR}} = 10\cdot\log\frac{M_{\text{in}}^2}{(M_{\text{in}}-M_{\text{den}})^2+\epsilon}$. Events yield high SNR in textured/motion regions but near-zero SNR in smooth areas; images yield high SNR in bright regions but poor SNR in dark areas. Dual-modality SNR maps are jointly processed and channel-wise softmax is applied to generate fusion weights $W_{\text{img}}, W_{\text{evt}}$.
    - Design Motivation: More comprehensive than EvLight (image-SNR-only guidance)—when event SNR approaches zero in dark smooth regions, the method avoids over-reliance on events and preserves weak image signals.

3. **Cross-modal Attention Diffusion (CAD)**:

    - Function: Deeply fuses dual-modality features and reconstructs high-fidelity outputs via a diffusion model.
    - Mechanism: Weighted image and event features are processed with bidirectional cross-attention (image query + event key/value and vice versa), concatenated to form a unified multimodal representation $F_{\text{fused}}$, which is fed as a condition into the diffusion model: $\hat{\epsilon}_\theta = \epsilon_\theta(x_t, F_{\text{fused}}, t)$, with 50-step DDIM deterministic sampling for reconstruction.
    - Design Motivation: The progressive denoising of diffusion models outperforms single-step regression in low-SNR regions; the conditioned multimodal features provide a strong prior.

### Loss & Training

- Two-stage training: Stage 1 trains image and event denoising modules independently; Stage 2 introduces cross-modal consistency constraints for joint training.
- $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + 10\cdot\mathcal{L}_{\text{grad}} + 0.5\cdot\mathcal{L}_{\text{cons}}$
- Adam optimizer, learning rate $1\times10^{-4}$, 50 epochs, input crops of 256×256.
- Forward diffusion with 1000 steps, trained on a single RTX 4090.

## Key Experimental Results

### Main Results

| Input | Method | LLRVD-simu PSNR/SSIM/LPIPS | REAL PSNR/SSIM/LPIPS |
|------|------|---------------------------|---------------------|
| sRGB | LightenDiffusion | 21.64/0.818/0.265 | 22.19/0.714/0.282 |
| RAW | BRVE | 27.58/0.817/0.137 | 21.87/0.717/0.334 |
| RAW | RID(NoiseModelling) | 26.76/0.825/0.127 | 22.72/0.729/0.258 |
| Event+sRGB | EvLight | 17.06/0.677/0.291 | 21.20/0.626/0.277 |
| **Event+RAW** | **NEC-Diff** | **27.74/0.828/0.125** | **24.51/0.742/0.201** |

### Ablation Study

| Configuration | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| w/o ECNS (SRIE+CAD only) | 21.06 | 0.653 | 0.278 |
| w/o SRIE (ECNS+CAD) | 23.24 | 0.698 | 0.243 |
| w/o CAD (ECNS+SRIE) | 22.53 | 0.671 | 0.265 |
| Full model | **24.51** | **0.742** | **0.201** |

### Key Findings

- ECNS contributes most significantly (removing it causes a 3.45 dB PSNR drop), confirming that collaborative denoising is the foundation of the entire pipeline.
- Dual SNR guidance outperforms image-only SNR guidance by 0.43 dB and direct fusion by 0.76 dB.
- The advantage is more pronounced on the REAL dataset (+1.79 dB over the best RAW method), attributed to more complex real-world noise.
- The method excels particularly in extremely dark scenes at 0.001–0.3 lux (comprising 70% of the dataset).
- Using both cross-modal input and consistency loss for event denoising yields the best results; using either alone provides limited improvement.

## Highlights & Insights

- **Physics-driven cross-modal denoising** is the core contribution: by leveraging the linear illumination response of RAW and the positive correlation between event shot noise and illumination, the paper establishes a physically grounded mutual denoising framework—considerably more principled than direct fusion or post-processing filters.
- **The REAL dataset** construction is valuable—a coaxial imaging system with optical attenuation simulates 0.001 lux extreme darkness, providing 47,800 pixel-aligned triplets (RAW/event/GT) that fill a critical gap in event–RAW low-light data.
- **SNR maps as fusion weights** is a simple yet effective design that generalizes to arbitrary multimodal fusion scenarios.

## Limitations & Future Work

- The event contrast threshold $C$ in the intensity consistency loss is learned from data; varying thresholds across different event cameras in real deployment may reduce generalizability.
- Diffusion model inference is relatively slow (50-step DDIM), limiting real-time applicability.
- Training and evaluation are conducted only at 256×256 resolution; high-resolution scenarios remain unexplored.
- Future work could investigate test-time adaptation to accommodate different event camera parameters.

## Related Work & Insights

- **vs. EvLight**: Uses only image SNR for fusion guidance, overlooking the near-zero event SNR in smooth dark regions. NEC-Diff's dual-SNR strategy is more comprehensive.
- **vs. ELEDNet/RETINEV**: Applies low-pass filtering or CNNs for event noise suppression, but simple filtering cannot balance noise suppression and detail preservation.
- **vs. EvRAW**: Focuses on detail and color recovery from event–RAW pairs but ignores sensor noise, limiting performance in extreme darkness.

## Rating

- Novelty: ⭐⭐⭐⭐ The physics-driven cross-modal collaborative denoising approach is novel, though the overall diffusion framework and conditional generation are relatively established.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons on both synthetic and real datasets with clear ablations, but broader real-world generalization evaluation is lacking.
- Writing Quality: ⭐⭐⭐⭐ Physical modeling derivations are clear and figures are well-crafted, though the method description is somewhat verbose.
- Value: ⭐⭐⭐⭐ The dataset contribution is significant, and the method addresses a well-defined application in extreme-dark imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to Translate Noise for Robust Image Denoising](learning_to_translate_noise_for_robust_image_denoising.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [\[CVPR 2026\] MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](motionaware_animatable_gaussian_avatars_deblurring.md)
- [\[ICLR 2026\] Are Deep Speech Denoising Models Robust to Adversarial Noise?](../../ICLR2026/image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)

</div>

<!-- RELATED:END -->
