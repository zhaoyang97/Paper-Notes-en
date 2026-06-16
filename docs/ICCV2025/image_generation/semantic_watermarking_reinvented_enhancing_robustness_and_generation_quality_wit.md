---
title: >-
  [Paper Note] Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity
description: >-
  [ICCV 2025][Image Generation][semantic watermarking] This paper addresses the frequency integrity loss caused by discarding the imaginary part in existing semantic watermarking methods for latent diffusion models (LDMs).…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "semantic watermarking"
  - "latent-space Fourier watermarking"
  - "Hermitian symmetry"
  - "center-aware embedding"
  - "latent diffusion models"
date: 2026-05-08
content_hash: f0f983e35e141157
---

# Semantic Watermarking Reinvented: Enhancing Robustness and Generation Quality with Fourier Integrity

**Conference**: ICCV 2025
**arXiv**: [2509.07647](https://arxiv.org/abs/2509.07647)  
**Code**: [https://github.com/thomas11809/SFWMark](https://github.com/thomas11809/SFWMark)  
**Area**: Diffusion Models / Digital Watermarking
**Keywords**: semantic watermarking, latent-space Fourier watermarking, Hermitian symmetry, center-aware embedding, latent diffusion models

## TL;DR

This paper addresses the frequency integrity loss caused by discarding the imaginary part in existing semantic watermarking methods for latent diffusion models (LDMs). It proposes Hermitian Symmetric Fourier Watermarking (SFW) and a center-aware embedding strategy to preserve frequency-domain integrity while enhancing detection robustness and generation quality.

## Background & Motivation

With the open-sourcing of large-scale vision-language models such as Stable Diffusion, copyright tracking and provenance verification of AI-generated content have become increasingly urgent. Embedding invisible watermarks during the generation process is one of the primary solutions.

**State of the Field and Limitations in Semantic Watermarking**:

**Tree-Ring / RingID and similar methods**: These methods embed geometric patterns (ring watermarks) in the Fourier domain of the latent vector using a merged-in-generation scheme, which provides inherent robustness against regeneration attacks.

**Limitations of Prior Work — Frequency Integrity Loss**: Existing methods directly discard the imaginary part after performing inverse FFT following Fourier-domain modification, leading to:
- **Distorted real-part information**: The original watermark pattern is corrupted.
- **Complete loss of imaginary part**: Critical regions in the frequency domain become empty.
- **Degraded detection accuracy**: Detection can only exploit incomplete frequency information.
- **Reduced generation quality**: The spatial-domain signal deviates from a real-valued Gaussian distribution.

**Root Cause — Vulnerability to Cropping Attacks**: Applying FFT-based watermark embedding over the full spatial matrix results in significant watermark loss after cropping.

**Core Idea**: If **Hermitian symmetry** is maintained during frequency-domain modification, the inverse FFT naturally yields a real-valued signal, eliminating the need to discard the imaginary part and thus preserving complete frequency information.

## Method

### Overall Architecture

Within the merged-in-generation pipeline of latent diffusion models: latent noise → FFT → watermark embedding in key regions → IFFT → text-guided generation of watermarked images. During detection, the latent query is obtained via DDIM inversion and the key frequency-domain regions are analyzed.

Two improvements are introduced into this pipeline: (1) Hermitian Symmetric Fourier Watermarking (SFW); and (2) a center-aware embedding strategy.

### Key Designs

1. **Hermitian Symmetric Fourier Watermarking (SFW)**:

    - **Core Principle**: The DFT of a real-valued signal satisfies the Hermitian symmetry condition $F[M-k, N-l] = \overline{F[k,l]}$, i.e., conjugate symmetry about the DC center.
    - **Design Constraint**: The free region in the frequency domain is restricted to half the domain (the other half is determined by symmetry); the imaginary parts of the DC center and Nyquist frequency points must be zero.
    - **Effect**: The spatial-domain signal after IFFT is purely real, eliminating the need to discard the imaginary part; both the real and imaginary components of the watermark are fully preserved, allowing detection to exploit complete frequency information.
    - **Gaussian Preservation**: Real-valued Gaussian noise is transformed by FFT into complex Gaussian noise: $f[m,n] \sim \mathcal{N}(0, \sigma^2) \Rightarrow F[k,l] \sim \mathcal{CN}(0, MN\sigma^2)$. Maintaining Hermitian symmetry keeps the spatial-domain signal closer to a real Gaussian distribution, stabilizing diffusion model initialization.

2. **Center-Aware Embedding Strategy**:

    - Rather than applying FFT over the full spatial matrix (64×64), FFT is applied only to the central region (44×44) prior to watermark embedding.
    - **Design Motivation**: Cropping attacks typically remove peripheral regions; the central region retains the highest proportion of watermark information.
    - This significantly improves robustness against cropping attacks at various scales.

3. **HSTR (Improved Tree-Ring)**: Hermitian symmetry constraints are applied to the Tree-Ring watermark pattern, combined with center-aware embedding.

4. **HSQR (QR Code Watermarking)**:

    - A QR code is split into two halves, which are embedded into the real and imaginary parts of the free half-region in the frequency domain, respectively.
    - Embedding formula: $\text{HSQR}(\tilde{x}, c) = \begin{cases} +|F(\tilde{x},c)|, & \text{if QR}(x)=1 \\ -|F(\tilde{x},c)|, & \text{if QR}(x)=0 \end{cases}$
    - The embedding region is offset by one pixel from the DC axis to avoid numerical instability.

### Loss & Training

The proposed method is a training-free embedding scheme (merged-in-generation) and involves no additional loss function training. Watermark embedding is completed within the generation pipeline without introducing extra processing time.

## Key Experimental Results

### Main Results — Verification Task (TPR@1%FPR, MS-COCO)

| Method | No Attack | Brightness | JPEG | Blur | Noise | BM3D | VAE-B | Diff | Center Crop | Random Crop | Avg |
|--------|-----------|------------|------|------|-------|------|-------|------|-------------|-------------|-----|
| Tree-Ring | 0.957 | 0.463 | 0.548 | 0.934 | 0.412 | 0.815 | 0.509 | 0.543 | 0.509 | 0.734 | 0.655 |
| Zodiac | 0.998 | 0.843 | 0.973 | 0.998 | 0.880 | 0.997 | 0.944 | 0.972 | 0.989 | 0.995 | 0.962 |
| **HSTR (ours)** | **1.000** | 0.899 | **0.994** | **1.000** | 0.806 | **0.999** | **0.973** | **0.997** | **1.000** | **1.000** | **0.971** |
| RingID | 1.000 | 0.988 | 1.000 | 1.000 | 0.987 | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 | 0.997 |
| **HSQR (ours)** | **1.000** | 0.991 | **1.000** | **1.000** | 0.983 | **1.000** | **0.992** | **1.000** | **1.000** | **1.000** | **0.997** |

HSTR improves over Tree-Ring by an average of 31.6 percentage points; HSQR achieves accuracy on par with RingID while delivering superior generation quality.

### Generation Quality Comparison

| Method | FID↓ | CLIP Score↑ | Note |
|--------|------|-------------|------|
| No Watermark | Baseline | Baseline | — |
| Tree-Ring | +slight | slight drop | Frequency distortion degrades quality |
| RingID | +notable | drop | High-energy patterns produce visible ring artifacts |
| **HSTR** | +negligible | nearly unchanged | Frequency integrity preserves quality |
| **HSQR** | +negligible | nearly unchanged | Same as above |

### Ablation Study

| Configuration | Verification Performance | Generation Quality | Note |
|---------------|-------------------------|--------------------|------|
| w/o SFW | Baseline (Tree-Ring) | Frequency distortion | Imaginary part loss degrades detection |
| + SFW (Hermitian symmetry) | Large improvement | Significant improvement | Frequency integrity restored |
| + Center-aware embedding | Large gain in crop robustness | No degradation | Central 44×44 region |
| Information capacity analysis | QR code capacity vs. accuracy trade-off | — | Larger QR code → lower matching rate |

### Key Findings

- **Frequency integrity is central**: Maintaining Hermitian symmetry alone — without any training or additional computation — substantially improves both detection accuracy and generation quality.
- The high-energy patterns in RingID produce visible ring-shaped artifacts (as shown in Fig. 4 of the paper), whereas HSTR/HSQR exhibit no such artifacts.
- Center-aware embedding yields significant improvements under both center-cropping and random-cropping scenarios (Tree-Ring improves from 0.509/0.734 to HSTR's 1.000/1.000).
- Under diffusion regeneration attacks (Diff), HSTR and HSQR achieve TPR close to 1.0, validating the inherent regeneration robustness of semantic watermarking.

## Highlights & Insights

- **Precise problem formulation**: The paper accurately identifies the overlooked yet critical issue of "discarding the imaginary part" in existing semantic watermarking methods.
- **Elegant solution**: The method leverages a well-known mathematical property of the Fourier transform (Hermitian symmetry) to resolve the problem without any training — a "bug-fix"-level intervention that yields substantial gains.
- **No additional computational overhead**: All improvements are completed at the embedding stage and introduce no additional inference time.
- The QR code watermarking scheme supports both high capacity (enabling identification) and strong robustness.

## Limitations & Future Work

- The current center-aware embedding uses a fixed 44×44 region; adaptive region selection could be explored.
- The information capacity of HSQR is limited by the available frequency-domain area; larger payloads require more sophisticated encoding schemes.
- Although the method is training-free, it relies on the quality of DDIM inversion — inversion errors can degrade detection performance.
- Applicability to other generative architectures (e.g., DiT, FLUX) has not yet been explored.

## Related Work & Insights

- The proposed method directly improves upon seminal works such as Tree-Ring and RingID; Zodiac's frequency-domain optimization approach is avoided due to its high iterative cost.
- Insight: A "minor correction" grounded in fundamental signal processing theory (Hermitian symmetry) can sometimes be more effective than complex learning-based approaches.

## Rating

- **Novelty**: ⭐⭐⭐ — The core contribution is the correct application of a known mathematical property; the innovation is incremental in nature.
- **Technical Depth**: ⭐⭐⭐⭐ — Thorough frequency-domain analysis with a complete proof of Gaussian preservation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 12 attack types, 4 datasets, and comparisons against multiple baselines.
- **Value**: ⭐⭐⭐⭐⭐ — Training-free, zero overhead, plug-and-play; extremely high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](../../NeurIPS2025/image_generation/ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models](revelio_interpreting_and_leveraging_semantic_information_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
