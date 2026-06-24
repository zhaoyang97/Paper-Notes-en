---
title: >-
  [Paper Note] Domain-Adaptive Video Deblurring via Test-Time Blurring
description: >-
  [ECCV 2024][Image Restoration][Video Deblurring] A test-time domain adaptation method based on a diffusion blur model is proposed. By detecting relatively sharp regions from blurry videos as pseudo-sharp images and generating domain-adaptive blur conditions to synthesize training pairs, the method enables fine-tuning of deblurring models on unseen domains, achieving a maximum gain of 7.54dB across 5 real-world datasets.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Video Deblurring"
  - "Domain Adaptation"
  - "Test-Time Fine-Tuning"
  - "Diffusion Model"
  - "Blur-Conditional Generation"
date: 2026-05-08
content_hash: 2a9c53c3a61d8829
---

# Domain-Adaptive Video Deblurring via Test-Time Blurring

**Conference**: ECCV 2024  
**arXiv**: [2407.09059](https://arxiv.org/abs/2407.09059)  
**Code**: [Yes (GitHub)](https://github.com/Jin-Ting-He/DADeblur)  
**Area**: Image Restoration / Video Deblurring  
**Keywords**: Video Deblurring, Domain Adaptation, Test-Time Fine-Tuning, Diffusion Model, Blur-Conditional Generation

## TL;DR

A test-time domain adaptation method based on a diffusion blur model is proposed. By detecting relatively sharp regions from blurry videos as pseudo-sharp images and generating domain-adaptive blur conditions to synthesize training pairs, the method enables fine-tuning of deblurring models on unseen domains, achieving a maximum gain of 7.54dB across 5 real-world datasets.

## Background & Motivation

Video deblurring aims to restore blurry videos caused by camera shake or object motion. However, existing methods face a severe **domain gap** issue:

**Domain Distribution Mismatch**: Most existing deblurring models are trained on synthetic datasets (e.g., GoPro). However, in real-world scenarios, different camera settings (shutter speed, aperture, light source) produce blur patterns with varying directions and intensities. When the blur pattern of the test video is inconsistent with the training set, performance drops significantly.

**No Unsupervised Labels at Test Time**: During the inference stage, only blurry inputs are available without corresponding sharp images as supervision signals, making direct model fine-tuning impossible.

**Limitations of Prior Work**:
   - Self-supervised methods (Chi et al., Nah et al.) use reconstruction/re-blurring losses but ignore domain-specific blur information.
   - Liu et al. employ GAN supervision on a blur model to generate training data for meta-learning, but **fail to exploit temporal motion information contained in continuous video frames**. (Continuous frames reveal motion trajectories, and the degree of blur implicitly contains the blur intensity during exposure).

**Key Insight**: The blurry video itself implicitly contains domain-specific blur cues: the motion trajectory between continuous frames reflects the blur direction, and the intensity of the blurry region reflects the blur magnitude. These cues can be exploited to generate domain-adaptive blur conditions for the diffusion blur model ID-Blau, thereby synthesizing domain-consistent training pairs at test time.

## Method

### Overall Architecture

The proposed domain adaptation scheme consists of three core steps: (1) **RSDM** (Relative Sharpness Detection Module) extracts relatively sharp patches from blurry videos as pseudo-sharp images; (2) **DBCGM** (Domain-adaptive Blur Condition Generation Module) extracts temporal motion cues from the video to generate domain-specific blur conditions; (3) The ID-Blau diffusion model is used to blur the pseudo-sharp images based on these domain-specific conditions, and the synthesized pseudo-training pairs are utilized to fine-tune the deblurring model.

### Key Designs

1. **Relative Sharpness Detection Module (RSDM)**:

    - **Function**: To find relatively sharp patches from blurry videos to serve as pseudo-sharp images.
    - **Mechanism**: A Blur Magnitude Estimator (BME) is designed, which is a five-stage encoder-decoder network combined with Multi-Scale Feature Fusion (MSFF). BME is trained on the GoPro dataset, where pixel-level ground-truth blur magnitude maps are calculated via motion trajectories accumulated from optical flows:
    $G = \frac{1}{\tau}\sqrt{u^2 + v^2}$
    where $u, v$ represent horizontal and vertical motion trajectories, and $\tau$ is a normalization term. During testing, BME is used to predict the blur magnitude map $M_t^{(i)} = BME(V_t^{(i)})$ for each frame. After binarization with an adaptive threshold $\eta^{(i)}$, $256 \times 256$ sharp patches are cropped. The threshold is designed to ensure the extraction of the top $r\%=20\%$ sharpest patches.
    - **Design Motivation**: Even in "blurry" videos, the degree of blur is uneven across different frames and regions. Relatively sharp areas can always be found to serve as pseudo-ground truth.

2. **Domain-adaptive Blur Condition Generation Module (DBCGM)**:

    - **Function**: To estimate domain-specific blur directions and magnitudes from the temporal motion cues of blurry videos, and generate the blur conditions required by ID-Blau.
    - **Mechanism**: It contains a Blur Orientation Estimator (BOE) and a BME. For the pseudo-sharp patch $\tilde{S}_t^{(i)}$ and its co-located patches in adjacent frames (2 preceding and 2 succeeding frames), the motion trajectory is estimated via optical flows:
    $\tilde{\mathcal{F}}_t^{(i)} = \sum_{n=-2}^{1} f(\tilde{S}_{t+n}^{(i)}, \tilde{S}_{t+n+1}^{(i)})$
    After normalization, the domain-specific blur direction is obtained as $\tilde{O}_t^{(i)} = \frac{\tilde{\mathcal{F}}_t^{(i)}}{\sqrt{\tilde{u}^2 + \tilde{v}^2}}$. The blur magnitude is modulated through the Magnitude Adaptation Process by scaling the normalized magnitude of the current frame using the average blur magnitude of adjacent frames:
    $\tilde{M}_t^{(i)} = \text{Norm}(M_t^{(i)}) \cdot \text{Avg}(M_{t-2}^{(i)}, M_{t-1}^{(i)}, M_{t+1}^{(i)}, M_{t+2}^{(i)})$
    - **Design Motivation**: Randomly generated blur conditions do not match the blur distribution of the target domain. Domain-specific blur directions and intensity cues must be extracted from the test video itself to generate training data consistent with the target domain.

3. **Domain-Adaptive Fine-Tuning based on ID-Blau**:

    - **Function**: To blur pseudo-sharp images driven by domain-specific blur conditions using ID-Blau, generating pseudo-training pairs for fine-tuning.
    - **Mechanism**: ID-Blau is a conditional diffusion blur model that takes a sharp image $S$ and a pixel-level blur condition map $C = (x, y, z) \in \mathbb{R}^{H \times W \times 3}$ (horizontal/vertical blur direction and magnitude) to generate a blurry image $B = \text{ID-Blau}(S, C)$. The domain-specific direction and magnitude generated by DBCGM are combined to form the condition $\tilde{C}_t^{(i)}$ to blur the pseudo-sharp patches: $\tilde{B}_t^{(i)} = \text{ID-Blau}(\tilde{S}_t^{(i)}, \tilde{C}_t^{(i)})$.
    - **Design Motivation**: ID-Blau provides controllable blur generation capabilities. Combining it with domain-specific conditions allows the synthesis of training data complying with the target distribution.

### Loss & Training

- **BME Training**: Supervised using the L1 loss, $\mathcal{L} = \mathcal{L}_1(M, G)$, where $M$ is the predicted blur magnitude and $G$ is the ground truth derived from optical flows.
- **Domain-Adaptive Fine-Tuning**: Fine-tuned for 10 epochs on the pseudo-training pairs using the original loss functions of each respective deblurring model.
- **BME Optimizer**: Adam, with an initial learning rate of $1e^{-3}$, cosine annealed to $1e^{-4}$. Images are resized to $320 \times 320$, batch size is 16, and training is conducted for 50 epochs.

## Key Experimental Results

### Main Results

**Performance improvement of four deblurring models across five real-world datasets (Table 1)**:

| Model | BSD-1ms8ms | BSD-2ms16ms | BSD-3ms24ms | RealBlur | RBVD |
|------|-----------|------------|------------|---------|------|
| **ESTRNN** Baseline | 25.57 | 24.64 | 26.01 | 25.87 | 24.47 |
| **ESTRNN** +Ours | **29.44 (+3.87)** | **28.36 (+3.72)** | **28.32 (+2.31)** | **27.64 (+1.77)** | **26.83 (+2.36)** |
| **MMP-RNN** Baseline | 21.63 | 21.26 | 22.74 | 24.65 | 22.81 |
| **MMP-RNN** +Ours | **29.17 (+7.54)** | **26.95 (+5.69)** | **26.77 (+4.03)** | **27.69 (+3.04)** | **25.81 (+3.00)** |
| **DSTNet** Baseline | 25.42 | 23.50 | 24.68 | 26.57 | 23.15 |
| **DSTNet** +Ours | **28.69 (+3.27)** | **27.11 (+3.61)** | **26.69 (+2.01)** | **27.74 (+1.17)** | **25.66 (+2.51)** |
| **Shift-Net** Baseline | 25.00 | 23.75 | 24.98 | 26.01 | 23.98 |
| **Shift-Net** +Ours | **28.75 (+3.75)** | **26.31 (+2.56)** | **26.92 (+1.94)** | **27.71 (+1.70)** | **25.35 (+1.37)** |

Average improvement: BSD-1ms8ms **+4.61dB**, BSD-2ms16ms **+3.90dB**, BSD-3ms24ms **+2.57dB**, RealBlur **+1.92dB**, RBVD **+2.31dB**. MMP-RNN achieves the highest boost of **+7.54dB**.

### Ablation Study

**Ablation study on the effectiveness of RSDM and DBCGM (Table 2, ESTRNN on BSD-1ms8ms)**:

| Configuration | Pseudo-Sharp | Blur Condition | PSNR | Gain |
|------|-------------|---------------|------|------|
| (a) Baseline | — | — | 25.57 | +0.00 |
| (b) Random patch + Random blur | Random | Random | 23.88 | -1.69 |
| (c) Random patch + Optical-Flow | Random | Optical-Flow | 25.51 | -0.06 |
| (d) Random patch + **DBCGM** | Random | **DBCGM** | 29.01 | **+3.44** |
| (e) **RSDM** + Random blur | **RSDM** | Random | 24.32 | -1.25 |
| (f) **RSDM** + Optical-Flow | **RSDM** | Optical-Flow | 26.19 | +0.62 |
| (g) **RSDM + DBCGM** | **RSDM** | **DBCGM** | **29.44** | **+3.87** |

**Comparison with existing domain adaptation methods (Table 3, ESTRNN)**:

| Method | BSD-1ms8ms | BSD-2ms16ms | BSD-3ms24ms | RealBlur | RBVD |
|------|-----------|------------|------------|---------|------|
| Baseline | 25.57 | 24.64 | 26.01 | 25.87 | 24.47 |
| Liu et al. (meta-learning) | 25.58 | 24.53 | 25.15 | 26.12 | 24.83 |
| **Ours** | **29.44** | **28.36** | **28.32** | **27.64** | **26.83** |

### Key Findings

- **DBCGM is the core contribution**: Even using random patches (without RSDM), DBCGM still brings a +3.44dB improvement (configuration d), indicating that domain-specific blur condition is the key.
- **Random blur conditions are not only ineffective but also harmful**: Random patch + Random blur decreases performance by 1.69dB (configuration b), indicating that training data that does not conform to the target domain distribution can mislead the model.
- **RSDM provides additional gains**: Under the same blur conditions, using RSDM consistently outperforms Random patch (compare b/e, c/f, d/g).
- **Significant advantage over existing domain adaptation methods**: The meta-learning method of Liu et al. shows almost no improvement or even degrades on most datasets, while the proposed method achieves an advantage of up to +3.86dB on BSD-1ms8ms.
- **Adaptive threshold $r=20\%$ is optimal**: A larger ratio introduces more blurry patches, degrading the quality of pseudo-training pairs.

## Highlights & Insights

- **Reverse Thinking**: Instead of directly modifying the deblurring model architecture, a "blur-before-deblur" strategy is adopted to adapt to the target domain at test time, which is an ingenious domain adaptation idea.
- **Thorough Exploitation of Domain Cues**: Extracting motion trajectories from sequential frames of blurry videos as domain-specific blur conditions fully utilizes temporal video information.
- **Strong Generality**: The method is independent of the deblurring model, serving as a plug-and-play domain adaptation scheme applicable to any deblurring model (validated on 4 different architectures in experiments).
- **Astonishing Improvement**: A peak improvement of +7.54dB on MMP-RNN is extremely rare in the image restoration field.

## Limitations & Future Work

- It requires fine-tuning on each video for a specified number of epochs (10 epochs) at test time, which increases inference time overhead.
- The pseudo-sharp images themselves still contain residual blur, introducing noise when serving as "sharp" supervision signals.
- The training of the ID-Blau blur model is also based on GoPro, potentially leading to secondary domain bias.
- Extension to image deblurring (non-video) has not been explored.
- The adaptive threshold needs to be calculated separately for each video, increasing computational complexity.

## Related Work & Insights

- The ID-Blau diffusion blur model provides controllable blur generation capabilities, based on which this paper designs a domain-specific conditional generation strategy.
- Compared to the GAN-based blur + meta-learning method of Liu et al., this paper makes full use of temporal video information, yielding significantly superior results.
- The concept of test-time adaptation (TTA) has broad implications and can be extended to other image restoration tasks (such as denoising and super-resolution).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of extracting domain-specific blur conditions from blurry videos is highly novel, and the reverse thinking of "deblurring by blurring" is impressive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive validation across 4 models × 5 datasets, detailed ablation studies (7 configurations), and a complete threshold sensitivity analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The methodology is clearly described and rich in diagrams and tables, though there are many equations, and some symbols could be simplified.
- **Value**: ⭐⭐⭐⭐⭐ — Highly significant practical implications; the plug-and-play domain adaptation scheme showing a maximum improvement of +7.54dB is extremely persuasive in the image restoration field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Domain Generalization for Image Super-Resolution](../../ICLR2026/image_restoration/test-time_domain_generalization_for_image_super-resolution.md)
- [\[ECCV 2024\] TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts](ttt-mim_test-time_training_with_masked_image_modeling_for_denoising_distribution.md)
- [\[ECCV 2024\] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [\[CVPR 2026\] Gyro-based Deep Video Deblurring](../../CVPR2026/image_restoration/gyro-based_deep_video_deblurring.md)
- [\[ECCV 2024\] Efficient Cascaded Multiscale Adaptive Network for Image Restoration](efficient_cascaded_multiscale_adaptive_network_for_image_restoration.md)

</div>

<!-- RELATED:END -->
