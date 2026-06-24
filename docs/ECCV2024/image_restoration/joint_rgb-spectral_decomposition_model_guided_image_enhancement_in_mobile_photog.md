---
title: >-
  [Paper Note] Joint RGB-Spectral Decomposition Model Guided Image Enhancement in Mobile Photography
description: >-
  [ECCV 2024][Image Restoration][Image Enhancement] This paper proposes JDM-HDRNet, which extracts shading, reflectance, and material semantic priors from low-resolution multispectral images (Lr-MSI) using a joint RGB-spectral decomposition model. These priors are integrated into HDRNet to enhance dynamic range, color mapping, and semantic bilateral grid expert learning, respectively. Additionally, the first paired RGB-hyperspectral Mobile-Spec dataset is constructed.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Image Enhancement"
  - "RGB-Spectral Fusion"
  - "Spectral Intrinsic Decomposition"
  - "HDRNet"
  - "Mobile Photography"
date: 2026-05-08
content_hash: 8c29e719d4f6fcd4
---

# Joint RGB-Spectral Decomposition Model Guided Image Enhancement in Mobile Photography

**Conference**: ECCV 2024  
**arXiv**: [2407.17996](https://arxiv.org/abs/2407.17996)  
**Code**: [Yes](https://github.com/CalayZhou/JDM-HDRNet)  
**Area**: Image Restoration / Image Enhancement  
**Keywords**: Image Enhancement, RGB-Spectral Fusion, Spectral Intrinsic Decomposition, HDRNet, Mobile Photography

## TL;DR

This paper proposes JDM-HDRNet, which extracts shading, reflectance, and material semantic priors from low-resolution multispectral images (Lr-MSI) using a joint RGB-spectral decomposition model. These priors are integrated into HDRNet to enhance dynamic range, color mapping, and semantic bilateral grid expert learning, respectively. Additionally, the first paired RGB-hyperspectral Mobile-Spec dataset is constructed.

## Background & Motivation

### Problem Introduction

Miniature spectrometers have been integrated into mobile devices, but the application of spectral sensors in mobile photography is primarily limited to illumination estimation in **automatic white balance (AWB)**. The potential of shading and reflectance components embedded in spectral information remains under-explored for image enhancement.

### Key Challenge

Integrating additional low-resolution multispectral images (Lr-MSI) into existing RGB workflows faces two main challenges:

**Intrinsic complexity of spectral images**: Factors such as scene geometry, inter-reflections, and complex artificial lighting make it difficult to directly integrate spectral information into mobile ISP pipelines.

**Limited spectral imaging capability**: Although commercial mobile spectral sensors have over 10 spectral channels, their spatial resolution is extremely low (often single-pixel), which limits the application of Lr-MSI in tone enhancement.

### Motivation: From the Perspective of Spectral Intrinsic Decomposition

Based on the theory of spectral intrinsic decomposition, a spectral image can be decomposed into three components:

$$I_{k,x} = \int_{400\text{nm}}^{1000\text{nm}} C_k(\lambda) L(\lambda) S(x) R(\lambda,x) d\lambda$$

- $L(\lambda)$ (illumination curve): Used for illumination estimation in white balance.
- $S(x)$ (shading): Reflects the interaction between object geometry and lighting, which can be utilized for local brightness adjustment.
- $R(\lambda,x)$ (reflectance): Contains the intrinsic color and texture of materials, where fine-grained spectral channels assist in material segmentation and color mapping.

Core insight of this work: **The near-infrared (NIR) band can approximate the shading prior** (as spectral curves of different colors tend to flatten in the NIR region), which enables shading estimation to be trained end-to-end rather than relying on traditional optimization methods.

## Method

### Overall Architecture

JDM-HDRNet consists of two stages:

1. **Joint Decomposition Stage**: Predicts three priors—shading ($S$), reflectance ($R$), and material semantics ($M$)—by leveraging the complementarity between RGB and Lr-MSI.
2. **Prior-Guided Enhancement Stage**: Integrates the three priors into different modules of HDRNet respectively—$S$ for dynamic range enhancement, $R$ for color mapping, and $M$ for semantic bilateral grid expert learning.

### Key Designs

#### 1. **Joint RGB-Spectral Decomposition Model (Joint Decomposition Model)**

**Function**: Predicts $S$, $R$, and $M$ by leveraging the complementary nature of RGB (high spatial resolution, low spectral resolution) and Lr-MSI (low spatial resolution of $16 \times 16$, high spectral resolution of 10 channels).

**Mechanism**: A dual independent encoder-decoder architecture (based on FCN) is adopted. After resizing Lr-MSI to the same spatial resolution as RGB, features $\mathcal{F}_{rgb}$ and $\mathcal{F}_{spec}$ are extracted separately. After concatenation, independent decoders are used to predict material segmentation $M$ and shading $S$:

$$M, S = D_{m,s}(\text{concat}(\mathcal{F}_{rgb}, \mathcal{F}_{spec}))$$

The shading prediction is transformed from a regression problem into a classification problem (divided into 8 brightness levels), which is decomposed collaboratively with material segmentation. Reflectance is derived from the shading and input image based on Retinex theory.

**Design Motivation**:
- The near-infrared band (mean of 850-1000nm) approximates the shading ground truth, avoiding the failure of traditional optimization methods in complex outdoor scenes.
- The fine-grained spectral channels of Lr-MSI improve material segmentation—t-SNE visualization shows that clustering of different classes is tighter in hyperspectral images than in RGB. By incorporating Lr-MSI, the mIoU increases from 71.86% to 78.93%.

#### 2. **Shading Prior: Localized Brightness Adaptation**

**Function**: Separates the shading component from the RGB space and transforms it into the reflectance space to simplify color mapping learning.

**Mechanism**: Converts shading $S$ into a brightness representation $\hat{S}$ through a lightweight module (2 layers of convolution + deconvolution), and then divides the 16-bit input image and Lr-MSI by $\hat{S}$ to obtain the reflectance images:

$$R_{rgb} = I_{rgb} / \hat{S}, \quad R_{msi} = I_{msi} / \hat{S}$$

**Design Motivation**: Statistical analysis reveals that the Pearson correlation coefficient between the pixel histograms of the 16-bit input and the 8-bit target is $\rho = 0.66$ in the original RGB space, but improves to $\rho = 0.91$ in the reflectance space. This suggests that after separating the shading, the color distribution of the input and output becomes more similar, reducing the difficulty of color mapping learning.

#### 3. **Reflectance Prior: Spectral Perception Self-Attention (SPSA)**

**Function**: Leverages the fine-grained spectral channels of Lr-MSI's reflectance $R_{msi}$ to enhance the color perception capability of bilateral grid coefficient prediction.

**Mechanism**: Concatenates features of $R_{msi}$ and $R_{rgb}$ to generate $Q, K, V$, and creates a **spectral attention map** $A \in \mathbb{R}^{C \times C}$ through channel-wise self-attention to model the mutual information between different spectral channels:

$$A = \text{softmax}(\sigma \hat{K} \cdot \hat{Q}), \quad \hat{\mathcal{F}}_{R_{msi}} = \hat{W}_3^{msi} \hat{V} \cdot A + W_3^{msi} \mathcal{F}_{R_{msi}}$$

SPSA, acting as a residual learning module, fuses cross-spectral features via adaptive weighted fusion.

**Design Motivation**: Bilateral grid coefficient prediction is performed at low resolution, so the low spatial resolution of Lr-MSI does not cause significant degradation. Channel attention can capture color relationships between different spectral bands, compensating for HDRNet's imbalanced learning across different color channels.

#### 4. **Material Semantic Prior: Mixture of Semantic Grid Experts**

**Function**: Learns specialized bilateral grid coefficients for different material categories (sky, building, plant, trunk, road).

**Mechanism**: Each category $M_i$ is transformed into a probability map $\Psi_i$ via a mapping function, and affine transformation parameters $(\alpha, \beta)$ are learned to modulate $Q, K$:

$$\hat{Q} = (1+\alpha_Q) \cdot Q + \beta_Q, \quad \hat{K} = (1+\alpha_K) \cdot K + \beta_K$$

The bilateral grids of different experts are dynamically fused using category weights:

$$\Phi(x,y) = \sum_{i=1}^{N} w_i \Phi_i(x,y)$$

**Design Motivation**: The spectral characteristics of different materials differ significantly (plants have a reflection peak at 550nm, while the sky is dominant in blue bands), and a single grid coefficient cannot accommodate the color preferences of all materials. A mixture-of-experts design allows for customized tone enhancement strategies for each material.

### Loss & Training

- Joint Decomposition Model: Cross-entropy loss (shading classification + material segmentation).
- JDM-HDRNet: MSE loss, batch size=4, lr=0.0001.
- Inputs are cropped to $512 \times 512$, and the low-resolution stream is downsampled to $256 \times 256$.
- Lr-MSI is downsampled from hyperspectral images ($1057 \times 960 \times 176$) to $16 \times 16 \times 10$ to simulate commercial mobile spectral sensors.

## Key Experimental Results

### Main Results

**Quantitative comparison with existing enhancement methods (Mobile-Spec dataset)**:

| Method | PSNR↑ | SSIM↑ | $\triangle E^*$↓ | Description |
|------|-------|-------|---------|------|
| DPE | 22.81 | 0.806 | 11.06 | Unpaired GAN |
| CSRNet (MLP) | 26.34 | 0.923 | 6.44 | MLP color transformation |
| 3D LUT | 27.52 | 0.926 | 5.39 | Lookup table |
| SepLUT-L | 28.08 | 0.944 | 4.26 | 1D+3D LUT |
| HDRNet (baseline) | 27.75 | 0.939 | 5.12 | Bilateral grid |
| UPE | 28.19 | 0.946 | 4.79 | Illumination mapping |
| **JDM-HDRNet (Ours)** | **29.83** | **0.967** | **3.60** | +2.08 dB vs HDRNet |

### Ablation Study

**Step-by-step accumulation of the three priors (ideal priors S\*, R\*, M\*)**:

| Configuration | PSNR↑ | SSIM↑ | $\triangle E^*$↓ | Description |
|------|-------|-------|---------|------|
| HDRNet baseline | 27.75 | 0.939 | 5.12 | No priors |
| + S\* (shading) | 28.68 | 0.957 | 4.29 | +0.93 dB |
| + S\* + R\* (reflectance) | 29.68 | 0.968 | 3.55 | +1.93 dB |
| + S\* + R\* + M\* (material) | **30.14** | **0.972** | **3.44** | +2.39 dB |

**Joint decomposition model's predicted priors vs. ideal priors**:

| Configuration | PSNR↑ | SSIM↑ | $\triangle E^*$↓ | Description |
|------|-------|-------|---------|------|
| JDM-HDRNet\* (Ideal priors) | 30.14 | 0.972 | 3.44 | Upper bound |
| JDM-HDRNet (Predicted priors) | 29.83 | 0.967 | 3.60 | Only 0.31 dB difference |

**Reflectance spectral channel ablation study**:

| Spectral Range | PSNR↑ | Description |
|---------|-------|------|
| Baseline (No R) | 28.68 | - |
| 400-520nm | 28.74 | Blue band |
| 520-640nm | 29.09 | Green band |
| 640-760nm | 29.19 | Red band, largest improvement |
| 400-760nm (Visible light) | 29.24 | Six channels |
| **400-1000nm (All bands)** | **29.68** | Ten channels, best |

**Material segmentation ablation study**: Integrating Lr-MSI improves mIoU from 71.86% to 78.93% (the trunk class improves from 10.96% to 31.14%).

### Key Findings

1. **Shading separation is the most effective single improvement**: Transforming to the reflectance space alone yields a 0.93 dB gain without changing the HDRNet architecture.
2. **Spectral information compensates for imbalanced color learning in RGB**: Mobile-Spec is dominated by blue-green tones (sky/plants), leading to insufficient learning of red tones. Incorporating all spectral bands compensates for this bias.
3. **A spatial resolution of 16×16 for Lr-MSI is close to optimal**: Performance continuously improves from 1×1 to 16×16, but shows diminishing marginal returns beyond 16×16, which aligns precisely with the spatial resolution of the bilateral grid.
4. **Despite imperfect predictions from the joint decomposition model, the prior-guided design is robust**—the difference between predicted and ideal priors is only 0.31 dB.

## Highlights & Insights

- **Designing deep learning schemes from spectral physical models**: Translating spectral intrinsic decomposition theory into an end-to-end trainable joint decomposition model; the assumption of using NIR to approximate shading is simple yet effective.
- **Counter-intuitive finding**: Extremely low-resolution (16×16) spectral information can significantly improve image enhancement performance (+2.08 dB).
- **Construction of the first paired RGB-hyperspectral mobile photography dataset, Mobile-Spec**, consisting of 200 scene groups, 176-channel hyperspectral data, and fine-grained material segmentation annotations.
- Each designed module (localized brightness adaptation, SPSA, semantic grid experts) corresponds directly to one of the three priors, featuring clear mechanisms and physical interpretability.

## Limitations & Future Work

- **16×16 spatial resolution Lr-MSI is not yet feasible on commercial smartphones** (most current mobile spectral sensors are single-pixel), requiring further hardware evolution.
- Mobile-Spec contains only 200 scenes, which is limited in scale and covers only outdoor scenarios.
- The assumption of near-infrared approximating shading does not hold under indoor LED lighting (where the NIR response of LEDs decays sharply), which limits indoor applications.
- The spectral decomposition priors could be extended to other ISP tasks, such as denoising, HDR imaging, and super-resolution.

## Related Work & Insights

- **HDRNet**: The baseline of this work, whose flexible bilateral grid design facilitates the integration of additional priors.
- **Retinex Theory**: The theoretical basis for shading-reflectance decomposition.
- **NIR Shading Approximation (Cheng et al.)**: The source of the key assumption in this work; the color independence of the NIR band makes it a reliable approximation for shading.
- Finding new application scenarios for mobile spectral sensors beyond automatic white balance, which is expected to advance the development of spectral hardware.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The approach of guiding image enhancement via spectral decomposition is novel, organically combining physical models with deep learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely detailed ablation studies (priors, number of channels, spatial resolutions, material categories) along with a rich selection of comparison methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured, with well-explained correspondences between physical models and method designs.
- **Value**: ⭐⭐⭐⭐ — Provides a new direction for the photography application of mobile spectral sensors, and the dataset holds long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [\[ECCV 2024\] DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising](denoisplit_a_method_for_joint_microscopy_image_splitting_and_unsupervised_denois.md)
- [\[CVPR 2026\] HDW-SR: High-Frequency Guided Diffusion Model based on Wavelet Decomposition for Image Super-Resolution](../../CVPR2026/image_restoration/hdw-sr_high-frequency_guided_diffusion_model_based_on_wavelet_decomposition_for_.md)
- [\[ECCV 2024\] Learning Exhaustive Correlation for Spectral Super-Resolution: Where Spatial-Spectral Attention Meets Linear Dependence](learning_exhaustive_correlation_for_spectral_super-resolution_where_spatial-spec.md)
- [\[CVPR 2026\] ReasonX: MLLM-Guided Intrinsic Image Decomposition](../../CVPR2026/image_restoration/reasonx_mllm-guided_intrinsic_image_decomposition.md)

</div>

<!-- RELATED:END -->
