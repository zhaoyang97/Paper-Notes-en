---
title: >-
  [Paper Note] HVI: A New Color Space for Low-light Image Enhancement
description: >-
  [CVPR 2025][Image Restoration][Low-light Image Enhancement] This paper proposes a new color space HVI (Horizontal/Vertical-Intensity), which eliminates red artifacts through polarized HS mapping, compresses black artifacts in dark regions using a learnable intensity component, and outperforms existing low-light enhancement state-of-the-art methods across 10 datasets in combination with the decoupling network CIDNet.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Low-light Image Enhancement"
  - "Color Space"
  - "HVI Color Space"
  - "Color Decoupling"
  - "CIDNet"
date: 2026-05-08
content_hash: d2825e2d34ef67d1
---

# HVI: A New Color Space for Low-light Image Enhancement

**Conference**: CVPR 2025  
**arXiv**: [2502.20272](https://arxiv.org/abs/2502.20272)  
**Code**: [https://github.com/Fediory/HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet)  
**Area**: Image Restoration / Low-Light Image Enhancement  
**Keywords**: Low-light Image Enhancement, Color Space, HVI Color Space, Color Decoupling, CIDNet

## TL;DR
This paper proposes a new color space HVI (Horizontal/Vertical-Intensity), which eliminates red artifacts through polarized HS mapping, compresses black artifacts in dark regions using a learnable intensity component, and outperforms existing low-light enhancement state-of-the-art methods across 10 datasets in combination with the decoupling network CIDNet.

## Background & Motivation

**Background**: Low-Light Image Enhancement (LLIE) is a fundamental task in computer vision, aiming to restore clear visual information from degraded images under low-light conditions. Mainstream methods mostly process images directly in the sRGB color space, leveraging deep networks to learn the mapping function from low-light to normal light.

**Limitations of Prior Work**: The sRGB space suffers from inherent high color sensitivity, where the RGB channels are highly coupled, making low-light enhancement operations prone to introducing color bias and luminance artifacts. A natural improvement is to process in the HSV space, as HSV separates luminance (V) from color (H, S). However, the Hue channel in the HSV space represents hue using angles, which exhibits numerical discontinuity near red (the transition region between 0° and 360°), leading to severe red noise artifacts in the enhanced results. Meanwhile, the V channel is highly compressed in extremely dark regions, creating black noise artifacts when amplified.

**Key Challenge**: The channel coupling problem in sRGB and the numerical discontinuity problem in HSV create a dilemma: processing directly in RGB causes color bias, while shifting to HSV causes red/black artifacts. Existing color spaces cannot simultaneously satisfy both channel decoupling and numerical stability/continuity.

**Goal**: To design a new color space optimized specifically for low-light enhancement, simultaneously overcoming the coupling issue of sRGB and the artifact issues of HSV.

**Key Insight**: The authors observe that the two core issues of HSV—red artifacts from the Hue angle discontinuity and black artifacts from the extreme compression of Value in dark areas—can be resolved simultaneously by eliminating angle discontinuity via coordinate transformation and improving the distribution in dark areas through learnable mapping.

**Core Idea**: Remap the polar Hue-Saturation in HSV into Cartesian coordinates H/V (Horizontal/Vertical) to eliminate the large distance issue in the red transition region. Meanwhile, replace the fixed Value channel with a learnable intensity component I to adaptively compress the dark area distribution and eliminate black artifacts.

## Method

### Overall Architecture
The system consists of two core components: HVI color space transformation and CIDNet (Color and Intensity Decoupling Network). An input low-light sRGB image is first converted to the HVI color space, where the H (Horizontal) and V (Vertical) components encode color information, and the I (Intensity) component encodes luminance information. CIDNet processes the color branch and the intensity branch separately within the HVI space, learning the photometric mapping function under low-light conditions, and finally converts the results back to the sRGB space to output the enhanced image.

### Key Designs

1. **HVI Color Space (Horizontal/Vertical-Intensity Color Space)**:

    - **Function**: Provides a numerically continuous, channel-decoupled color representation to eliminate the coupling issue in sRGB and the artifact issues in HSV.
    - **Mechanism**: Starting from HSV, the polar form of the Hue-Saturation mapping is converted into Cartesian coordinates. Specifically, it defines $H_{cart} = S \cdot \cos(2\pi \cdot Hue)$ and $V_{cart} = S \cdot \sin(2\pi \cdot Hue)$. This is known as "polarized HS maps". In polar coordinates, the angular distance between two points near red (Hue $\approx 0^{\circ}$ and $360^{\circ}$) is large, but their Euclidean distance in Cartesian coordinates is very small. This fundamentally eliminates numerical jumps in the red region. Meanwhile, the Intensity component uses learnable parameters $\alpha_s$, $\alpha_i$, and $\gamma$ to perform non-linear transformations on the original Value, adaptively compressing the extreme value distribution in dark regions.
    - **Design Motivation**: The discontinuity of the Hue channel in traditional HSV at 0/360° causing red artifacts is an inherent flaw of the space. Cartesian coordinate transformation is a classic mathematical method to resolve angular discontinuity, but its direct application in low-light enhancement was previously unexplored.

2. **CIDNet (Color and Intensity Decoupling Network)**:

    - **Function**: Decouples and processes color and intensity information in the HVI space to learn an accurate photometric mapping.
    - **Mechanism**: The network contains two parallel branches—the color branch processes the H/V channels, while the intensity branch processes the I channel. Each branch adopts a U-Net-like encoder-decoder structure containing multiple Lighten Cross-Attention (LCA) modules. LCA enables information interaction between the color and intensity branches: features of the color branch serve as Queries, while features of the intensity branch serve as Keys/Values (and vice versa), achieving complementary fusion of color and luminance information through cross-attention.
    - **Design Motivation**: Color and luminance degrade under different patterns—color shift and insufficient luminance under low-light conditions are two distinct degradations that require separate modeling. However, they are not completely independent (e.g., luminance variations affect color perception), meaning moderate information interaction must be maintained using cross-attention.

3. **Learnable Intensity Transformation**:

    - **Function**: Adaptively adjusts luminance mapping under varying lighting conditions.
    - **Mechanism**: Introduces three adjustable parameters: $\alpha_s$ (saturation scaling), $\alpha_i$ (intensity scaling), and $\gamma$ (gamma correction), to parameterize the transformation of the Intensity component in the HVI space. These parameters can be manually adjusted during inference to fit different lighting scenes, or automatically learned during training to obtain optimal values.
    - **Design Motivation**: Low-light degradation varies drastically across different datasets/scenes. Fixed spatial transformations cannot adapt to all scenarios; learnable parameters provide flexibility. Specifically, using random gamma augmentation can improve generalization across datasets.

### Loss & Training
A combination of L1 reconstruction loss, perceptual loss, and SSIM loss is utilized. L1 guarantees pixel-level accuracy, perceptual loss ensures high-level semantic consistency, and SSIM guarantees structural similarity. A warmup strategy is optional during training. It supports training on multiple datasets such as LOLv1, LOLv2-real, LOLv2-syn, LOL-Blur, SID, SICE, and FiveK.

## Key Experimental Results

### Main Results

| Dataset | Metric | CIDNet | Prev. SOTA | Gain |
|--------|------|--------|----------|------|
| LOLv1 | PSNR | 27.08 | 25.74 (Retinexformer) | +1.34 |
| LOLv1 | SSIM | 0.870 | 0.845 | +0.025 |
| LOLv2-real | PSNR | 26.03 | 24.81 | +1.22 |
| LOLv2-syn | PSNR | 27.51 | 26.32 | +1.19 |
| LOL-Blur | PSNR | 27.84 | 26.47 | +1.37 |
| DICM/LIME/MEF/NPE/VV (No-Reference) | NIQE↓ | 3.28 | 3.65 | -0.37 |

### Ablation Study

| Configuration | PSNR (LOLv1) | SSIM | Description |
|------|-------------|------|------|
| Full CIDNet (HVI) | 27.08 | 0.870 | Full Model |
| sRGB Space | 24.92 | 0.831 | Processing directly in RGB space |
| HSV Space | 25.41 | 0.842 | Processing in traditional HSV space |
| HVI w/o Learnable I | 26.23 | 0.856 | Using fixed Value instead of learnable Intensity |
| w/o LCA Module | 26.15 | 0.851 | Removing cross-attention |
| w/o Perceptual Loss | 26.67 | 0.862 | Removing perceptual loss |

### Key Findings
- The HVI color space yields a PSNR gain of 2.16 dB and 1.67 dB compared to sRGB and HSV respectively, fully validating the effectiveness of the new color space.
- The learnable Intensity transformation contributes 0.85 dB, and the LCA cross-attention contributes 0.93 dB, demonstrating the significant impact of both core designs.
- The model also performs best on 5 no-reference datasets, proving its strong generalization capability.
- An ensemble scheme based on HVI-CIDNet won first place in the NTIRE 2025 Low-Light Image Enhancement Challenge.

## Highlights & Insights
- The idea of **solving problems from the color space level** is highly fundamental—while most low-light enhancement works focus on network architecture design, this paper traces back to the more fundamental data representation level. This perspective inspires us that altering the representation space can sometimes be more effective than stacking network layers.
- **Eliminating angular discontinuity using Cartesian coordinates** is a simple and elegant mathematical operation. This trick can be transferred to any task involving angular/periodic data (e.g., optical flow direction, rotation estimation).
- The design of learnable parameters $\alpha_s$, $\alpha_i$, and $\gamma$ makes the color space transformation itself differentiable and adaptable, presenting a valuable design paradigm.

## Limitations & Future Work
- The HVI space is a variant of HSV, and its design heavily relies on the specific polar-to-Cartesian transformation of HS; whether there exists an even better color space design remains to be explored.
- In extreme low light (such as the extremely dark scenes in the SID dataset), the performance improvement is relatively small, suggesting that fundamental color space transformation has limited impact when the signal-to-noise ratio is extremely low.
- Learnable parameters require manual adjustment during inference to fit different scenarios; the degree of automation could be further enhanced.
- It is worth exploring the generalization of the HVI color space to other image restoration tasks (e.g., dehazing, deraining).

## Related Work & Insights
- **vs Retinexformer**: Retinexformer relies on the Retinex theory to decompose in the sRGB space, while CIDNet decouples and processes in the specially designed HVI space. The color space advantages of HVI allow it to achieve better performance under similar network complexity.
- **vs SNR-Aware**: SNR-Aware adaptively processes images by estimating noise levels, whereas CIDNet mitigates degradation from the source through color space transformation. These two methodologies are complementary and could theoretically be combined.
- This study demonstrates that "data representation outperforms network design" holds true in certain scenarios, advising validation of this standpoint across other low-level vision tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes a completely new solution from the color space level, with a simple and effective mapping concept from polar to Cartesian coordinates.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensively validated across 10 datasets, exhaustive ablation studies, backed by a first-place finish in the NTIRE competition.
- Writing Quality: ⭐⭐⭐⭐ Clear motivational derivation, complete mathematical description of the color space design.
- Value: ⭐⭐⭐⭐⭐ Introduces a generalizable new color space; 797 stars and the competition championship demonstrate its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](../../AAAI2026/image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)
- [\[ICCV 2025\] CWNet: Causal Wavelet Network for Low-Light Image Enhancement](../../ICCV2025/image_restoration/cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)
- [\[CVPR 2025\] URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration](urwkv_unified_rwkv_model_with_multi-state_perspective_for_low-light_image_restor.md)
- [\[CVPR 2025\] DarkIR: Robust Low-Light Image Restoration](darkir_robust_low-light_image_restoration.md)
- [\[CVPR 2025\] Efficient Diffusion as Low Light Enhancer (ReDDiT)](efficient_diffusion_as_low_light_enhancer.md)

</div>

<!-- RELATED:END -->
