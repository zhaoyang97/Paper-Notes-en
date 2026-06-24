---
title: >-
  [Paper Note] ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention
description: >-
  [CVPR 2025][Signal & Communication][White Balance] This paper proposes ABC-Former, which introduces CIELab color space and RGB histograms as auxiliary bimodal information. It utilizes a cross-domain Transformer and an Interactive Channel Attention (ICA) module to achieve cross-modal transfer of global color knowledge, achieving SOTA performance in sRGB white balance correction tasks. It is also extended to ABC-FormerM to handle mixed illumination scenarios.
tags:
  - "CVPR 2025"
  - "Signal & Communication"
  - "White Balance"
  - "bimodal"
  - "cross-domain Transformer"
  - "interactive channel attention"
  - "CIELab histogram"
date: 2026-05-08
content_hash: 212f9f2ce5b64919
---

# ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention

**Conference**: CVPR 2025  
**Code**: [https://github.com/ytpeng-aimlab/ABC-Former](https://github.com/ytpeng-aimlab/ABC-Former)  
**Area**: Image Processing / White Balance  
**Keywords**: White Balance, bimodal, cross-domain Transformer, interactive channel attention, CIELab histogram

## TL;DR
This paper proposes ABC-Former, which introduces CIELab color space and RGB histograms as auxiliary bimodal information. It utilizes a cross-domain Transformer and an Interactive Channel Attention (ICA) module to achieve cross-modal transfer of global color knowledge, achieving SOTA performance in sRGB white balance correction tasks. It is also extended to ABC-FormerM to handle mixed illumination scenarios.

## Background & Motivation

**Background**: White balance (WB) correction is a core step in the camera Image Signal Processing (ISP) pipeline. The goal is to eliminate image color casts caused by inaccurate color temperatures, thereby rendering natural and neutral colors in the image. Existing white balance methods mainly fall into two categories: (1) global color adjustment methods, which apply a global correction matrix to RAW images before camera ISP; (2) end-to-end deep learning methods, which directly learn the white balance mapping from sRGB images.

**Limitations of Prior Work**: Global adjustment methods only consider the overall color temperature, causing local color casts in mixed illumination scenarios (e.g., indoor lighting + daylight from a window). Although end-to-end models can learn pixel-level adjustments, they do not explicitly exploit prior information of global color distributions (such as the color temperature trends reflected by CIELab histograms), resulting in suboptimal results in terms of color consistency. Both categories have respective limitations and lack an effective mechanism for integrating global and local information.

**Key Challenge**: White balance correction requires simultaneously understanding both global color temperature shifts (the overall warmth or coolness of the entire image) and local color variations (different light sources in different areas). However, existing architectures struggle to effectively integrate these two multi-scale pieces of information at the feature level.

**Goal**: To design a white balance correction network that can fully exploit both global color statistical information (CIELab + RGB histograms) and local pixel information (sRGB images).

**Key Insight**: White balance correction can be formulated as a cross-domain information fusion problem, where the global color histogram is one domain and the local pixel features represent another. A cross-attention mechanism in Transformers is used to achieve information interaction between these two domains.

**Core Idea**: Introducing CIELab and RGB histograms as auxiliary bimodal information, utilizing a cross-domain Transformer to perform feature interaction between the two modalities, and dynamically fusing multi-modal features via interactive channel attention.

## Method

### Overall Architecture
The input of ABC-Former consists of three components: (1) the sRGB image, from which spatial features are extracted via an encoder; (2) the CIELab histogram, which reflects the global color distribution; and (3) the RGB histogram, which provides color statistics for the three channels. After encoding these three components separately, feature interaction is conducted through a cross-domain Transformer module, followed by adaptive fusion using interactive channel attention. Finally, a decoder outputs the corrected image.

### Key Designs

1. **Auxiliary Bimodal Encoding**:

    - **Function**: Extract global color temperature and color cast information from color statistics.
    - **Mechanism**: The input sRGB image is converted to the CIELab color space ($L$: lightness, $a$: red-green axis, $b$: blue-yellow axis). The histograms of the $a$ and $b$ channels are computed as a compact representation of the color temperature distribution, and the three-channel RGB histogram is calculated simultaneously. The two types of histograms are encoded via lightweight MLPs into feature vectors $\mathbf{h}_{lab} \in \mathbb{R}^{D}$ and $\mathbf{h}_{rgb} \in \mathbb{R}^{D}$, respectively.
    - **Design Motivation**: The $a$ and $b$ channels in the CIELab space directly reflect the direction and degree of the color temperature shift, making them more suitable for white balance tasks than the RGB space. RGB histograms provide supplementary color cast information for each of the three channels. The two modalities offer complementary global color priors.

2. **Cross-domain Transformer**:

    - **Function**: Perform information interaction between image spatial features and global histogram features.
    - **Mechanism**: A standard cross-attention architecture is adopted, taking the sequence of image spatial features as Query and the histogram features as Key/Value. Attention weights are computed to achieve targeted injection of global color information into each spatial location. This process is also performed in reverse (using histogram features to query image features), enabling the global representation to perceive the local distribution. Progressive feature alignment is achieved through multi-layer stacking.
    - **Design Motivation**: Global histogram features and local pixel features reside in different "domains," and simple concatenation could cause information degradation. Cross-attention allows each pixel location to selectively extract the most relevant color adjustment signals from the global statistics.

3. **Interactive Channel Attention (ICA)**:

    - **Function**: Adaptively fuse multi-modal features output by the cross-domain Transformer in the channel dimension.
    - **Mechanism**: Attention weights are computed along the channel dimension for the fused multi-modal features to dynamically determine whether each channel should retain more information from the histogram modality or the image modality. Channel weights $\alpha \in \mathbb{R}^{C}$ are generated via global average pooling $\rightarrow$ FC $\rightarrow$ Sigmoid, followed by weighted fusion: $\mathbf{F}_{out} = \alpha \odot \mathbf{F}_{img} + (1-\alpha) \odot \mathbf{F}_{hist}$.
    - **Design Motivation**: Different channels correspond to different levels of color features—lower-level channels may require more global color temperature guidance, while higher-level channels rely more on local texture. ICA allows the network to autonomously learn the optimal channel-level fusion strategy.

### Loss & Training
The training employs a combination of pixel-level $\mathcal{L}_1$ loss and Perceptual Loss, which is conducted on the Rendered WB dataset (Set1). Meanwhile, the ABC-FormerM variant is provided to handle mixed illumination tasks. It is trained on the Mixed-Illumination dataset using the Two-stage Distortion-based (TDS) and TDS with full correction (TDSFC) strategies.

## Key Experimental Results

### Main Results (RenderedWB Dataset)

| Method | MSE ↓ | MAE ↓ | $\Delta E_{ab}$ ↓ | PSNR (dB) ↑ |
|------|-------|-------|---------------------|-------------|
| Deep WB (CVPR 2020) | 147.3 | 8.21 | 3.42 | 28.65 |
| Mixed-Ill WB (WACV 2022) | 132.8 | 7.56 | 3.15 | 29.34 |
| AWB-Transformer | 118.4 | 6.89 | 2.87 | 30.12 |
| **ABC-Former (Ours)** | **98.6**| **5.74**| **2.31**| **31.48**|

### Ablation Study

| Configuration | MSE ↓ | $\Delta E_{ab}$ ↓ | Description |
|------|-------|--------------------|------|
| Full model | 98.6 | 2.31 | Complete ABC-Former |
| w/o CIELab histogram | 112.5 | 2.68 | Only using RGB histogram |
| w/o RGB histogram | 108.3 | 2.54 | Only using CIELab histogram |
| w/o Cross-domain Transformer | 126.7 | 2.95 | Replacing cross-attention with simple concatenation |
| w/o ICA | 105.4 | 2.48 | Replacing adaptive channel attention with uniform fusion |

### Key Findings
- The cross-domain Transformer contributes the most (removing it increases MSE by 28.5%), confirming the necessity of cross-domain information interaction.
- The CIELab and RGB bimodal histograms provide complementary information; removing either leads to significant performance degradation, with a more severe drop when both are excluded.
- ICA yields an improvement of approximately 7% in MSE, indicating that adaptive channel fusion is superior to uniform fusion.
- In the mixed-illumination scenario (ABC-FormerM variant), the performance gain is even more pronounced, as global-local coordination is more crucial when different regions feature different light sources.
- Cited 3 times (as of April 2026), including follow-up work in WACV 2026.

## Highlights & Insights
- **Intuitive and Effective Bimodal Auxiliary Information Design**: The CIELab color space is inherently aligned with human color perception, making it highly natural to use its histogram as a global color temperature indicator. Complementary fusion with the RGB histogram further enhances robustness.
- **Information Bridging Role of the Cross-domain Transformer**: Treating global statistics and local pixels as two distinct "domains" and bridging them with cross-attention is a design paradigm that can be generalized to other tasks requiring global-local information fusion (e.g., tone mapping, exposure correction).
- **Generality to Mixed Illumination**: The same framework can handle more complex multi-light-source scenarios with minor modifications, demonstrating the flexibility of the proposed approach.

## Limitations & Future Work
- Histogram encoding is global, which may provide insufficient information in extreme local color cast scenarios (e.g., small areas with intense colored lighting).
- Generalization to unconventional color temperatures (e.g., highly cool or warm artistic-style images) remains to be validated.
- Combining this method with camera RAW-domain white balance methods can be explored, such as coarse adjustment in the RAW domain followed by fine-tuning with ABC-Former in the sRGB domain.
- ICA is currently channel-level; extending it to joint spatial-channel attention could further improve the performance under mixed illumination scenarios.

## Related Work & Insights
- **vs Deep WB (CVPR 2020)**: Deep WB is the first deep learning-based white balance method and operates on RAW images before the ISP pipeline. ABC-Former works in the sRGB domain and utilizes bimodal histograms to compensate for the loss of spatial information in the sRGB space.
- **vs Mixed-Ill WB (WACV 2022)**: The Mixed-Ill method processes mixed illumination through spatial segmentation, which tends to produce color discontinuities at segmentation boundaries. The cross-domain fusion of ABC-Former is smoother.
- **vs AWB-Transformer**: AWB-Transformer uses only image features for white balance, lacking a global color prior. The auxiliary histograms in ABC-Former provide crucial global information to complement this.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of auxiliary bimodal histograms and cross-domain Transformer is quite novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation on multiple datasets, sufficient ablation studies, and extension to mixed illumination.
- Writing Quality: ⭐⭐⭐⭐ The method is clearly described with a convincing motivation.
- Value: ⭐⭐⭐⭐ Practical improvement for white balance correction with a highly transferrable architectural design.
---
title: >-
  [Paper Notebook] ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention
description: >-
  [CVPR 2025][White Balance][Cross-domain Transformer] Proposing auxiliary bimodal cross-domain Transformer and interactive channel attention for sRGB image white balance correction
tags:
  - CVPR 2025
  - White Balance
  - Cross-domain Transformer
  - Channel Attention
  - Color Temperature Correction
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Breaking the Low-Rank Dilemma of Linear Attention](breaking_the_low-rank_dilemma_of_linear_attention.md)
- [\[ICCV 2025\] Rectifying Magnitude Neglect in Linear Attention](../../ICCV2025/signal_comm/rectifying_magnitude_neglect_in_linear_attention.md)
- [\[AAAI 2026\] Balancing Multimodal Domain Generalization via Gradient Modulation and Projection](../../AAAI2026/signal_comm/balancing_multimodal_domain_generalization_via_gradient_modulation_and_projectio.md)
- [\[ICLR 2026\] Efficient Message-Passing Transformer for Error Correcting Codes](../../ICLR2026/signal_comm/efficient_message-passing_transformer_for_error_correcting_codes.md)
- [\[ICML 2025\] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](../../ICML2025/signal_comm/fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)

</div>

<!-- RELATED:END -->
