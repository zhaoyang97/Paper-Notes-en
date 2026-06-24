---
title: >-
  [Paper Note] DarkIR: Robust Low-Light Image Restoration
description: >-
  [CVPR 2025][Image Restoration][Low-light restoration] DarkIR proposes an efficient CNN-based multi-task low-light image restoration method. The encoder uses SpAM+FreMLP (frequency magnitude enhancement) to handle illumination, while the decoder utilizes Di-SpAM (dilated spatial attention) to handle blur. With an asymmetric design, it achieves 27.30dB PSNR on LOLBlur with only 3.31M parameters.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Low-light restoration"
  - "multi-task"
  - "frequency MLP"
  - "dilated attention"
  - "efficient CNN"
date: 2026-05-08
content_hash: 1910226c06c81055
---

# DarkIR: Robust Low-Light Image Restoration

**Conference**: CVPR 2025  
**arXiv**: [2412.13443](https://arxiv.org/abs/2412.13443)  
**Code**: [https://github.com/cidautai/DarkIR](https://github.com/cidautai/DarkIR)  
**Area**: Image Restoration  
**Keywords**: Low-light restoration, multi-task, frequency MLP, dilated attention, efficient CNN

## TL;DR

DarkIR proposes an efficient CNN-based multi-task low-light image restoration method. The encoder uses SpAM+FreMLP (frequency magnitude enhancement) to handle illumination, while the decoder utilizes Di-SpAM (dilated spatial attention) to handle blur. With an asymmetric design, it achieves 27.30dB PSNR on LOLBlur with only 3.31M parameters.

## Background & Motivation

1. **Background**: Low-light image restoration faces three coupled degradations: noise, blur, and insufficient illumination. Existing methods either address a single degradation (such as only enhancing brightness) or utilize Transformers (such as RetinexFormer, Restormer) which have large parameter sizes.
2. **Limitations of Prior Work**: (1) Transformer-based methods have a large number of parameters (Restormer has 26M+), making them unsuitable for edge deployment; (2) Existing CNN methods lack specialized designs for joint low-light and blur degradation; (3) Frequency-domain information is underutilized in low-light restoration.
3. **Key Challenge**: Brightness enhancement and deblurring require different features—the former requires global illumination estimation (frequency domain), while the latter requires local structure recovery with a large receptive field (spatial domain). However, processing both with a single unified module is inefficient.
4. **Goal**: To design an asymmetric encoder-decoder where the encoder focuses on illumination enhancement (frequency domain) and the decoder focuses on deblurring (large spatial receptive field).
5. **Key Insight**: Insufficient illumination in low-light conditions primarily manifests as amplitude attenuation in the frequency domain, whereas blur requires a large spatial receptive field. These two aspects are suited for different attention mechanisms.
6. **Core Idea**: The encoder employs FreMLP (which operates solely on FFT amplitude without altering phase) for illumination enhancement, while the decoder utilizes Di-SpAM (three groups of depthwise convolutions with different dilation rates) for deblurring.

## Method

### Overall Architecture

Low-light blurry image → Encoder (SpAM + FreMLP, progressive 8x downsampling) → Bottleneck → Decoder (Di-SpAM + Gated FFN, progressive upsampling) → Residual Connection → Restored Image. Asymmetric design: the encoder processes frequency-domain enhancement, and the decoder processes spatial-domain deblurring.

### Key Designs

1. **Frequency Domain MLP (FreMLP)**

    - **Function**: To enhance the amplitude of low-light images in the frequency domain.
    - **Mechanism**: FFT → apply MLP transformation only to amplitude (phase remains unchanged) → IFFT. Inverted residual structure + Simplified Channel Attention (SCA).
    - **Design Motivation**: Low-light degradation primarily manifests as frequency amplitude attenuation (especially in low-frequency components). Performing enhancement directly on the amplitude is more efficient and targeted than global spatial operations.

2. **Dilated Spatial Attention (Di-SpAM)**

    - **Function**: To acquire a large receptive field for deblurring with low computational cost.
    - **Mechanism**: Three groups of depthwise convolutions use different dilation rates (1, 4, 9) → pooling fusion → generate a spatial attention map with a large receptive field.
    - **Design Motivation**: Deblurring requires a large receptive field to model motion range, but large convolutional kernels are computationally expensive. Dilated convolutions obtain an equivalent receptive field of $1+4+9=14$ times at the cost of standard convolutions.

3. **Multi-Task Joint Training**

    - **Function**: To simultaneously handle denoising, deblurring, and brightness enhancement.
    - **Mechanism**: $\mathcal{L} = \lambda_p L_1 + \lambda_{pe} L_{percep} + \lambda_{ed} L_{edge} + L_{lol}$, where $L_{lol} = ||x_{\downarrow 8} - \hat{x}_{\downarrow 8}||_1$ is the scale-guide loss at 8x downsampling.
    - **Design Motivation**: The three degradations co-exist in low-light scenarios, and handling them separately accumulates errors. $L_{lol}$ ensures low-resolution structural consistency.

### Loss & Training

L1 ($\lambda_p=1$) + LPIPS ($\lambda_{pe}=0.01$) + Edge Loss ($\lambda_{ed}=50$) + Low-resolution guided loss.

## Key Experimental Results

### Main Results

| Method | LOLBlur PSNR↑ | LOLBlur SSIM↑ | LOLBlur LPIPS↓ | Parameters |
|------|---------------|---------------|----------------|--------|
| LEDNet | 26.30 | - | 0.224 | 7.4M |
| RetinexFormer | 26.02 | - | 0.181 | 1.61M |
| Restormer | - | - | - | 26.13M |
| DarkIR-m | 26.62 | 0.891 | 0.148 | **3.31M** |
| **DarkIR-l** | **27.30** | **0.898** | **0.137** | 12.96M |

### Ablation Study

| Configuration | LOLv2-Real PSNR | Description |
|------|-----------------|------|
| DarkIR-mt (Multi-task) | 23.87 | Multi-task version |
| DarkIR (Single-task) | - | LOLBlur +0.68 vs Multi-task |
| w/o FreMLP | Decrease | Key to frequency-domain enhancement |
| w/o Di-SpAM | Decrease | Key to deblurring |

### Key Findings

- DarkIR-m with only 3.31M parameters is 55% smaller than LEDNet (7.4M) and 88% smaller than Restormer (26.13M), yet achieves better performance.
- The multi-task version (DarkIR-mt) loses only 0.4dB PSNR compared to the single-task version, indicating a very low cost for multi-task training.
- The contribution of frequency-domain FreMLP for low-light enhancement is greater than that of Di-SpAM for deblurring, indicating that brightness restoration is the more critical sub-task.

## Highlights & Insights

- **Asymmetric Encoder-Decoder Design**: The encoder and decoder focus on different types of degradation. This "division of labor" concept is highly generalizable for multi-task restoration.
- **FreMLP Operates Only on Amplitude, Leaving Phase Unchanged**: It preserves phase information (i.e., structural information) and only enhances energy, which physically corresponds to illumination restoration.
- **Extreme Parameter Efficiency**: Standard 3.31M parameters outperform models several times larger on the joint low-light and deblurring task.

## Limitations & Future Work

- The synthetic LOLBlur dataset uses frame averaging to simulate blur and EZ-DarkCE to reduce brightness, which still deviates from real-world nighttime degradations.
- Real-LOLBlur lacks ground-truth images, preventing quantitative perceptual evaluation.
- The frequency-domain method only processes amplitude; utilizing phase information might further improve restoration quality.
- It assumes blur arises from long exposures and does not explicitly address other blur sources (such as object motion).

## Related Work & Insights

- **vs RetinexFormer**: A Retinex-theory-based Transformer method, achieving 26.02 dB PSNR vs DarkIR's 27.30 dB. DarkIR transfer outperforms it with a simpler CNN structure.
- **vs Restormer**: A general image restoration Transformer with 26M+ parameters. DarkIR is specifically designed for low light, offering an order of magnitude higher parameter efficiency.
- **vs LEDNet**: A specialized low-light deblurring method. DarkIR achieves +1.0dB PSNR with 55% fewer parameters.

## Rating

- Novelty: ⭐⭐⭐⭐ The asymmetric design and FreMLP offer technical novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on LOLBlur, LOLv2, and Real data with multiple ablation variants.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ An edge-deployment-friendly low-light restoration solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration](urwkv_unified_rwkv_model_with_multi-state_perspective_for_low-light_image_restor.md)
- [\[CVPR 2025\] HVI: A New Color Space for Low-light Image Enhancement](hvi_a_new_color_space_for_low-light_image_enhancement.md)
- [\[CVPR 2025\] Efficient Diffusion as Low Light Enhancer (ReDDiT)](efficient_diffusion_as_low_light_enhancer.md)
- [\[ICCV 2025\] CWNet: Causal Wavelet Network for Low-Light Image Enhancement](../../ICCV2025/image_restoration/cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)

</div>

<!-- RELATED:END -->
