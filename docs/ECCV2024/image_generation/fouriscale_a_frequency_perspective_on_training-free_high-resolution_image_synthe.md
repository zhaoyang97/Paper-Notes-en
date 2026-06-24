---
title: >-
  [Paper Note] FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis
description: >-
  [ECCV 2024][Image Generation][Diffusion Models] FouriScale is proposed. From the perspective of frequency domain analysis, it replaces convolutional layers in pre-trained diffusion models with dilated convolutions and low-pass filtering, achieving training-free high-resolution image generation of arbitrary sizes, while theoretically proving the effectiveness of dilated convolutions in maintaining structural consistency.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Diffusion Models"
  - "High-Resolution Generation"
  - "Training-Free"
  - "Frequency Domain Analysis"
  - "Dilated Convolution"
date: 2026-05-08
content_hash: 67fc8e1b9f90a0f5
---

# FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2403.12963](https://arxiv.org/abs/2403.12963)  
**Code**: [GitHub](https://github.com/LeonHLJ/FouriScale)  
**Area**: Image Generation  
**Keywords**: Diffusion Models, High-Resolution Generation, Training-Free, Frequency Domain Analysis, Dilated Convolution

## TL;DR

FouriScale is proposed. From the perspective of frequency domain analysis, it replaces convolutional layers in pre-trained diffusion models with dilated convolutions and low-pass filtering, achieving training-free high-resolution image generation of arbitrary sizes, while theoretically proving the effectiveness of dilated convolutions in maintaining structural consistency.

## Background & Motivation

Pre-trained diffusion models (e.g., SD, SDXL) exhibit severe issues when generating images beyond their training resolution:

**Pattern Repetition**: Repetitive objects or structures appear in the image, such as multiple eyes on a single face.

**Structural Distortion**: Abnormality and deformation in overall composition and local details.

Limitations of existing solutions:

| Method | Limitations |
|------|------|
| MultiDiffusion / SyncDiffusion | Merges overlapping patches, lacks global guidance, and cannot generate images centered on specific objects. |
| Attn-Entro | Approaches the problem from the perspective of attention entropy, but performs poorly under high magnification. |
| ScaleCrafter | Discovers that convolutional layers are key, using re-dilated and convolution dispersion operations, but (1) conclusions are derived from empirical observations lacking theoretical support, and (2) requires offline pre-computation of linear transformations, lacking generalizability. |

The core innovation of FouriScale is that it **provides a theoretical explanation from the perspective of frequency domain analysis**, and proposes a simpler, more generalized, and pre-computation-free solution.

## Method

### Overall Architecture

FouriScale replaces the original convolutional layers in the UNet of pre-trained diffusion models with:

$$Conv_k(F) \rightarrow Conv_{k'}(iDFT(H \odot DFT(F)))$$

That is, low-pass filtering is first applied to the feature map, followed by processing with dilated convolutions. Where:
- $k'$ is the dilated version of the original convolutional kernel $k$
- $H$ is a low-pass filter
- DFT/iDFT denote the Discrete Fourier Transform and its inverse.

In addition, a padding-then-crop strategy is employed to handle arbitrary aspect ratios, along with FouriScale Guidance to enhance detail quality.

### Key Designs

#### Structural Consistency: Frequency Domain Proof of Dilated Convolution

Core theoretical problem: How to find a new convolutional kernel $k'$ such that convolving it on high-resolution features is equivalent to convolving the original kernel on low-resolution (downsampled) features?

**Structural Consistency Equation**:
$$Down_s(F) \circledast k = Down_s(F \circledast k')$$

**Lemma 1 (Key Lemma)**: Spatial downsampling by a factor of $s$ is equivalent to dividing the Fourier spectrum into $s \times s$ equal-sized blocks and averaging their summation (scaled by $1/s^2$):

$$DFT(Down_s(F(x,y))) = \frac{1}{s^2} \sum_{i=0}^{s-1} \sum_{j=0}^{s-1} F_{(i,j)}(u,v)$$

Transforming the structural consistency equation to the frequency domain reveals that: **the Fourier spectrum of the ideal $k'$ should be an $s \times s$ periodic repetition of the Fourier spectrum of the original kernel $k$**.

**Theorem**: A dilated convolution with a dilation factor of $(H/h, W/w)$ precisely satisfies this periodicity requirement! This is because the exponential terms in the DFT of the dilated kernel revert to the values of the original kernel at integer multiples:

$$e^{-j2\pi(\frac{p'·m}{d_h·M} + \frac{q'·n}{d_w·N})} = e^{-j2\pi(\frac{pm}{M} + \frac{qn}{N})}$$

This is the most crucial theoretical contribution of this work—providing a rigorous frequency domain proof for the effectiveness of the dilation operation observed empirically in ScaleCrafter.

#### Scale Consistency: Low-Pass Filtering

Merely using dilated convolutions is insufficient to completely resolve the pattern repetition issue. This is due to the **aliasing effect**:

- Downsampling causes high frequencies to fold into low frequencies and overlap (according to Theorem 3.1).
- This alters the fundamental frequency components of the original signal, disrupting the consistency across scales.

Solution: Introduce **low-pass filtering** before dilated convolution to remove high-frequency components that can cause aliasing.

The optimal low-pass filter mask size is $M/s_h \times N/s_w$ (when the spectrum is centered), which precisely preserves all valid frequencies within the downsampled resolution.

Experimental validation (Figure 4): Adding low-pass filtering significantly narrows the gap in frequency distribution between high- and low-resolution features.

#### Arbitrary-Sized Generation: Padding-then-Crop

When the aspect ratios of the target and training resolutions differ, the dilation rates for height and width will be different, leading to structural deformation.

**Padding-then-Crop Strategy**:

1. Calculate $r = \max(\lceil H_f/h_f \rceil, \lceil W_f/w_f \rceil)$
2. Zero-pad the feature map to $r·h_f \times r·w_f$ (to enforce a uniform dilation rate)
3. Apply dilated convolution + low-pass filtering
4. Crop back to the target size $H_f \times W_f$

#### FouriScale Guidance

Directly using the UNet modified by FouriScale may introduce artifacts in the background (stemming from the ringing effect of low-pass filtering and loss of detail).

Solution: Generate three noise estimates:
1. Unconditional estimate (UNet modified by FouriScale)
2. Conditional estimate (UNet modified by FouriScale)
3. **Additional conditional estimate** (using the same dilated convolution but with a **milder low-pass filter**)

Substitute the attention map from the 2nd estimate into the 3rd estimate (similar to the concept of MasaCtrl), and use the 3rd estimate as the final conditional estimate. This maintains correct structural information while avoiding quality degradation caused by low-pass filtering.

#### Annealing Strategy

- Use ideal dilated convolution and low-pass filtering in the first $S_{init}$ steps (to establish the structure).
- From $S_{init}$ to $S_{stop}$, gradually decrease the dilation factors and $r$ to 1.
- After $S_{stop}$, use the original UNet to refine details.

### Loss & Training

**Completely training-free**! No fine-tuning or offline pre-computation is required. All operations are implemented at inference time by replacing convolutional layers.

**Special Settings for SDXL**: A milder low-pass filter is used (coefficient $\sigma=0.6$), which attenuates rather than completely removes high frequencies. Since SDXL inherently supports multi-aspect ratios during training, it exhibits certain robustness to scale changes.

## Key Experimental Results

### Main Results

**FID Comparison across Different Magnification Scales on SD 2.1**

| Resolution (Scale) | Method | FIDr↓ | KIDr↓ | FIDb↓ | KIDb↓ |
|-------------|------|-------|-------|-------|-------|
| 4× (1:1) | Vanilla | 29.90 | 1.11 | 19.21 | 0.54 |
| 4× (1:1) | ScaleCrafter | 25.19 | 0.98 | 13.88 | 0.40 |
| 4× (1:1) | **FouriScale** | **25.17** | **0.98** | **13.57** | **0.40** |
| 16× (1:1) | Vanilla | 84.01 | 3.28 | 82.25 | 3.05 |
| 16× (1:1) | ScaleCrafter | 40.91 | 1.32 | 33.23 | 0.90 |
| 16× (1:1) | **FouriScale** | **39.49** | **1.27** | **28.14** | **0.73** |

**FID Comparison across Different Magnification Scales on SDXL**

| Resolution (Scale)| Method | FIDr↓ | KIDr↓ |
|-------------|------|-------|-------|
| 4× (1:1) | Vanilla | 49.81 | 1.84 |
| 4× (1:1) | ScaleCrafter | 49.46 | 1.73 |
| 4× (1:1) | **FouriScale** | **33.89** | **1.21** |
| 16× (1:1) | Vanilla | 116.40 | 5.45 |
| 16× (1:1) | ScaleCrafter | 84.58 | 3.53 |
| 16× (1:1) | **FouriScale** | **56.66** | **2.18** |

**FouriScale's advantage on SDXL is highly pronounced**, whereas ScaleCrafter often fails to generate acceptable images on SDXL.

### Ablation Study

**Contributions of Individual Components on SD 2.1, 16× (2048×2048)**

| Method| FIDr↓ |
|------|-------|
| FouriScale (Full) | 39.49 |
| w/o Guidance | 43.75 (+4.26) |
| w/o Guidance + Low-pass Filtering | 46.74 (+7.25) |

Each component demonstrates a clear contribution: low-pass filtering addresses the scale consistency issue (-2.99), and guidance enhances the quality of details (-4.26).

### Key Findings

1. **Theoretically Elegant and Practically Effective**: The frequency-domain proof for dilated convolution is not only a theoretical contribution but also yields significantly better practical performance than ScaleCrafter.
2. **Outstanding SDXL Compatibility**: ScaleCrafter performs poorly on SDXL (FIDr 84.58 vs 56.66), whereas the proposed method shows much stronger generalizability.
3. **No Pre-computation Required**: Compared to ScaleCrafter which requires offline pre-computation of linear transformations, FouriScale is ready-to-use.
4. **Faster Inference Speed**: Under 16× SDXL, ScaleCrafter takes 577s vs FouriScale's 540s.
5. **Theoretical Explanation for Low-Pass Filtering**: The optimal filter size is theoretically selected via the Nyquist theorem.

## Highlights & Insights

1. **Profound Frequency-Domain Theoretical Analysis**: Instead of merely offering an empirically effective solution, this work rigorously proves why dilated convolution resolves the pattern repetition issue through the lens of Fourier transform.
2. **Simplicity**: The entire framework requires no training, fine-tuning, or offline computation, functioning solely by replacing convolutional layers during inference.
3. **Generality**: It is compatible with three different pre-trained models—SD 1.5, SD 2.1, and SDXL—and supports arbitrary aspect ratios.
4. **Intuitive Annealing Strategy**: Employing FouriScale to establish structures in the early phase and refining details with the original UNet in the later stage offers highly inspiring phased-processing insights.
5. **Design of FouriScale Guidance**: It cleverly employs attention map replacement to strike a balance between structural correctness and detail richness.

## Limitations & Future Work

1. **Challenges at Ultra-High Resolutions**: Artifacts may still emerge under extreme resolutions such as 4096×4096.
2. **Applicable Only to UNet Architectures**: The method focuses on convolutional layer operations and is inapplicable to diffusion models using purely Transformer-based (e.g., DiT) architectures.
3. **Special Handling Required for SDXL**: It necessitates a milder low-pass filter ($\sigma=0.6$), implying that the approach is not fully automatic.
4. **Annealing Parameters Need Tuning**: The selection of $S_{init}$ and $S_{stop}$ can affect the outcomes.
5. **Untapped Combined Potential with Patch-Merging Methods**: Combining FouriScale with methods like MultiDiffusion might yield even higher resolutions.

## Related Work & Insights

- **ScaleCrafter**: An empirical work identifying convolutional layers as critical. FouriScale provides a solid theoretical foundation and introduces a superior solution.
- **Spectral Pooling**: The inspiration source for low-pass filtering (frequency pooling).
- **MasaCtrl**: The concept of attention map replacement is leveraged in FouriScale Guidance.
- **FreeU**: Also optimizes diffusion models from a frequency perspective; FouriScale utilizes FreeU by default in all experiments.
- **Insights**: Frequency domain analysis is a powerful tool to understand the behavior of diffusion models. Future research can explore the frequency-domain behavior of attention layers and how to adapt similar paradigms to Transformer architectures.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4.5 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 4.5 |
| Value | 4 |
| Writing Quality | 4.5 |
| Overall Rating | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models](freediff_progressive_frequency_truncation_for_image_editing_with_diffusion_model.md)
- [\[ECCV 2024\] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [\[ECCV 2024\] Editable Image Elements for Controllable Synthesis](editable_image_elements_for_controllable_synthesis.md)
- [\[ECCV 2024\] Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models](enhancing_perceptual_quality_in_video_super-resolution_through_temporally-consis.md)
- [\[CVPR 2026\] Training-free, Perceptually Consistent Low-Resolution Previews with High-Resolution Image for Efficient Workflows of Diffusion Models](../../CVPR2026/image_generation/training-free_perceptually_consistent_low-resolution_previews.md)

</div>

<!-- RELATED:END -->
