---
title: >-
  [Paper Note] CWNet: Causal Wavelet Network for Low-Light Image Enhancement
description: >-
  [ICCV 2025][Image Restoration][Low-light image enhancement] This paper proposes CWNet, a Causal Wavelet Network that models low-light image enhancement through a structural causal model (SCM), treating semantic information as causal factors and brightness/color degradation as non-causal factors, and employs a wavelet-based backbone for fine-grained frequency-domain feature restoration.
tags:
  - ICCV 2025
  - Image Restoration
  - Low-light image enhancement
  - causal inference
  - wavelet transform
  - state space model
  - CLIP semantic consistency
date: 2026-05-08
content_hash: 956d17cc8d072dec
---

# CWNet: Causal Wavelet Network for Low-Light Image Enhancement

**Conference**: ICCV 2025
**arXiv**: [2507.10689](https://arxiv.org/abs/2507.10689)
**Code**: CWNet (mentioned in the paper; specific link not provided)
**Area**: Image Restoration / Low-Light Enhancement
**Keywords**: Low-light image enhancement, causal inference, wavelet transform, state space model, CLIP semantic consistency

## TL;DR

This paper proposes CWNet, a Causal Wavelet Network that models low-light image enhancement through a structural causal model (SCM), treating semantic information as causal factors and brightness/color degradation as non-causal factors, and employs a wavelet-based backbone for fine-grained frequency-domain feature restoration.

## Background & Motivation

Conventional low-light image enhancement (LLIE) methods focus primarily on uniform brightness adjustment, neglecting instance-level semantic information and the intrinsic characteristics of different frequency components. Existing frequency-domain methods process high- and low-frequency features uniformly, limiting enhancement quality. Moreover, many approaches struggle to preserve color and semantic consistency while improving illumination, resulting in visually unnatural or semantically inaccurate outputs.

This paper addresses two key questions: (1) How can color and semantic consistency be maintained while improving illumination? (2) How can a robust model be designed to fully exploit frequency-domain features? Existing CLIP-based methods only enforce global semantic consistency and lack instance-level guarantees; wavelet-based methods do not sufficiently exploit the distinct properties of frequency subbands.

## Method

### Overall Architecture

CWNet is built upon an SCM, framing low-light enhancement as a causal inference problem. The overall architecture follows a U-Net design, comprising up/downsampling layers and Hierarchical Feature Restoration Blocks (HFRBs). Each HFRB consists of three core components: a Feature Extraction module (FE), a High-Frequency Enhancement Block (HFEB), and a Low-Frequency Enhancement Block (LFEB).

### Key Designs

1. **Causal Inference and Metric Learning**: The core idea is to define semantic information $\mathcal{S}$ in low-light scenes as causal factors and color/brightness anomalies $\mathcal{U}$ as non-causal factors. Non-causal factors are obtained via two "meaningful yet benign" interventions: an illumination degradation intervention $I_l = \frac{I}{L}L^{\gamma} + \varepsilon$ (based on a physical illumination model) and a color anomaly intervention (hue/saturation/RGB channel shifts). At the global level, a causally guided metric learning strategy is adopted: the processed low-light image serves as the anchor, the corresponding normal-light image as the positive sample, and counterfactually perturbed samples from different scenes as negative samples. The loss function is $\mathcal{L}_{ca} = \frac{\mathcal{L}_1(F_p, \hat{F})}{\xi(\sum_l \mathcal{L}_1(F_l, \hat{F}) + \sum_c \mathcal{L}_1(F_c, \hat{F}))}$. The design motivation is to compel the model to learn illumination-invariant semantic features by disentangling genuine semantic content from degradation factors.

2. **Instance-Level CLIP Semantic Loss**: A pretrained HRNet is used to extract semantic instance segmentation maps. The enhanced image is decomposed into multiple sub-instance images $I_{seg}^k$, and a CLIP encoder computes the semantic consistency probability of each instance against text prompts ("low light" / "normal light"): $\hat{y} = \frac{1}{K}\sum_{k=1}^{K}\frac{e^{\cos(\Phi_{image}(I_{seg}^k), \Phi_{text}(T_{low}))}}{...}$. Cross-entropy loss is used to optimize semantic consistency. The motivation stems from ATE analysis showing that different semantic regions exhibit significantly different sensitivity to degradation, making global consistency insufficient to guarantee local semantic integrity.

3. **Wavelet-based Backbone**: The FE module decomposes the input into four frequency subbands $\{L, H, V, D\}$ via wavelet transform. Low-frequency components are extracted using WTConv (large receptive field without additional parameter overhead), while high-frequency components are extracted using depthwise separable convolutions combined with direction-aligned convolutions (H-Conv/V-Conv/D-Conv), with information compensation from low to high frequencies. The HFEB introduces an HF-Mamba module based on Mamba, comprising three direction-aligned 2D-SSMs: H-2D-SSM for horizontal high frequencies, V-2D-SSM for vertical high frequencies, and D-2D-SSM for diagonal high frequencies, instead of a unified scanning strategy. The LFEB employs Fast Fourier Convolution (FFC) to process low-frequency components, using two residual blocks to provide globally context-aware, large-receptive-field feature enhancement.

### Loss & Training

The total loss is a weighted combination of five terms:
$$\mathcal{L}_{total} = \lambda_1\mathcal{L}_2 + \lambda_2\mathcal{L}_{ssim} + \lambda_3\mathcal{L}_{per} + \lambda_4\mathcal{L}_{ca} + \lambda_5\mathcal{L}_{sem}$$
with weights set to $[1.0, 0.3, 0.2, 0.01, 0.01]$. The Adam optimizer is used ($\beta_1=0.9, \beta_2=0.99$) with an initial learning rate of $4\times10^{-4}$, trained for $3\times10^5$ iterations, batch size 8, and randomly cropped $256\times256$ input patches.

## Key Experimental Results

### Main Results

| Dataset | Metric | CWNet | Prev. SOTA | Gain |
|--------|------|-------|----------|------|
| LOL-v1 | PSNR/SSIM/LPIPS | 23.60/0.8496/0.0648 | Wave-Mamba: 22.76/0.8419/0.0791 | +0.84/+0.0077/−0.0143 |
| LOL-v2-Real | PSNR/SSIM/LPIPS | 27.39/0.9005/0.0383 | Wave-Mamba: 27.87/0.8935/0.0451 | SSIM +0.007/LPIPS −0.007 |
| LOL-v2-Syn | PSNR/SSIM/LPIPS | 25.50/0.9362/0.0195 | RetinexMamba: 25.89/0.9346/0.0389 | Substantial LPIPS reduction |
| LSRW-Huawei | PSNR/SSIM/LPIPS | 21.50/0.6397/0.1562 | DMFourLLIE: 21.09/0.6328/0.1804 | +0.41/+0.007/−0.024 |

The model has only 1.23M parameters and 11.3G FLOPs, far fewer than MIRNet (31.79M) and SNR-Aware (39.12M).

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS | Note |
|------|------|------|-------|------|
| CWNet (full) | 21.53 | 0.6423 | 0.1631 | Baseline |
| w/o Causal Inference | 20.87 | 0.6375 | 0.1781 | PSNR drop of 0.66 |
| w/o FE | 20.98 | 0.6387 | 0.1804 | Missing frequency-domain extraction |
| w/o HFEB | 20.58 | 0.6317 | 0.1903 | Missing high-frequency enhancement |
| w/o LFEB | 20.41 | 0.6302 | 0.1985 | Low-frequency enhancement most critical |
| WTConv→Conv | 21.42 | 0.6415 | 0.1690 | Frequency-domain conv. superior to standard conv. |
| HF-Mamba→VMamba | 21.20 | 0.6394 | 0.1735 | Direction-aligned SSM superior to generic SSM |
| Semantic map→Global feature | 21.48 | 0.6417 | 0.1652 | Instance-level guidance is effective |

### Key Findings

- Removing LFEB causes the largest performance drop (PSNR falls to 20.41), indicating that low-frequency processing is the most critical component in the dual-branch architecture.
- The causal inference mechanism contributes substantially (PSNR drop of 0.66), validating the effectiveness of the causal perspective for disentangling semantics from degradation.
- In the cross-dataset setting (trained on LOL-v1, tested on LOL-v2-Real), SSIM reaches 0.9005, demonstrating strong generalization.
- The model achieves state-of-the-art performance with only 1.23M parameters, offering a clear efficiency advantage.

## Highlights & Insights

- Introducing causal inference into low-light enhancement is a novel and compelling perspective; the SCM formulation clearly defines causal and non-causal factors.
- Instance-level CLIP semantic loss provides finer-grained supervision than global CLIP loss, and the ATE analysis offers convincing theoretical support.
- The direction-aligned scanning strategy in HF-Mamba naturally aligns with the physical interpretation of wavelet high-frequency subbands (horizontal/vertical/diagonal), yielding an elegant design.
- The model is extremely lightweight (1.23M parameters), achieving an excellent balance between performance and efficiency.

## Limitations & Future Work

- Restoration quality degrades under compound degradations (e.g., low-light combined with blur or haze).
- The causal intervention strategy relies on specific degradation model assumptions that may not fully cover real-world degradation types.
- The incorporation of pretrained models such as CLIP and HRNet introduces additional computational overhead during inference.

## Related Work & Insights

- The application of causal inference to low-level vision tasks remains limited; the SCM modeling paradigm proposed here is generalizable to other image restoration tasks.
- The direction-aligned SSM scanning strategy can be extended to other vision tasks requiring directional awareness.
- The combination of frequency-domain representations (wavelet/Fourier) with causal inference introduces a new design paradigm for image enhancement.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of causal inference and wavelet transform is novel in the LLIE literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, comprehensive ablations, and cross-dataset generalization.
- Writing Quality: ⭐⭐⭐⭐ Logically clear; the causal framework is presented in a well-structured, progressive manner.
- Value: ⭐⭐⭐⭐ Lightweight, efficient, and state-of-the-art; strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[AAAI 2026\] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](../../AAAI2026/image_restoration/iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICCV 2025\] MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices](mobileie_an_extremely_lightweight_and_effective_convnet_for_real-time_image_enha.md)
- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)

</div>

<!-- RELATED:END -->
