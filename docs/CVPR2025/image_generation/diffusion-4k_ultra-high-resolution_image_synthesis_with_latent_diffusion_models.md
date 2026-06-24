---
title: >-
  [Paper Note] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models
description: >-
  [CVPR 2025][Image Generation][4K Image Generation] This paper proposes the Diffusion-4K framework, which consists of the Aesthetic-4K benchmark dataset, GLCM Score/Compression Ratio evaluation metrics, and a wavelet-based fine-tuning method. It enables large-scale latent diffusion models (LDMs) such as SD3-2B and Flux-12B to directly generate high-quality 4096×4096 images with rich texture details.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "4K Image Generation"
  - "Wavelet Fine-tuning"
  - "Partitioned VAE"
  - "Diffusion Models"
  - "Aesthetic-4K Benchmark"
date: 2026-05-08
content_hash: d9a0226a81c8e8f9
---

# Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2503.18352](https://arxiv.org/abs/2503.18352)  
**Code**: [https://github.com/zhang0jhon/diffusion-4k](https://github.com/zhang0jhon/diffusion-4k)  
**Area**: Image Generation / Ultra-High Resolution  
**Keywords**: 4K Image Generation, Wavelet Fine-tuning, Partitioned VAE, Diffusion Models, Aesthetic-4K Benchmark

## TL;DR
This paper proposes the Diffusion-4K framework, which consists of the Aesthetic-4K benchmark dataset, GLCM Score/Compression Ratio evaluation metrics, and a wavelet-based fine-tuning method. It enables large-scale latent diffusion models (LDMs) such as SD3-2B and Flux-12B to directly generate high-quality 4096×4096 images with rich texture details.

## Background & Motivation

**Background**: Mainstream latent diffusion models (SD3, Flux, etc.) primarily focus on training and generation at 1024×1024 resolution, leaving direct 4K image synthesis largely unexplored. Although PixArt-Σ and Sana achieve 4K-level generation, they mainly focus on efficiency, ignoring the inherent high-frequency details and rich textures characteristic of 4K images.

**Limitations of Prior Work**: (1) There is a lack of publicly available 4K image synthesis datasets and benchmarks; (2) Conventional evaluation metrics (FID, CLIP Score) are computed at low resolutions, failing to evaluate local detail quality in 4K images; (3) Direct deployment of standard $F=8$ VAE models at 4096×4096 leads to Out-of-Memory (OOM) issues.

**Key Challenge**: Scaling resolution up to 4K introduces quadratic computational overhead, while standard training objectives (noise/velocity prediction) lack explicit focus on high-frequency details, causing 4K images to often be "large yet blurry".

**Goal**: (1) Establish a complete benchmark for 4K image synthesis; (2) Propose a generalized fine-tuning method to enable various LDMs to generate highly detailed images at 4K resolution.

**Key Insight**: Wavelet transform can decompose a signal into low-frequency approximations and high-frequency details. Applying wavelet transform to velocity/noise prediction targets allows high-frequency and low-frequency components to simultaneously participate in loss computation, thereby explicitly enhancing focus on details.

**Core Idea**: Propose the Wavelet-based Latent Enhancement (WLF) fine-tuning method + Partitioned VAE ($F=16$) to resolve OOM issues + Aesthetic-4K dataset and dedicated evaluation metrics.

## Method

### Overall Architecture
The framework collects high-quality 4K images to construct the Aesthetic-4K dataset and uses GPT-4o to generate precise captions. A Partitioned VAE is used to compress images into an $F=16$ latent space. In this latent space, pre-trained diffusion models (SD3/Flux) are fine-tuned using a wavelet transform loss while freezing the VAE and text encoders.

### Key Designs

1. **Partitioned VAE**:

    - **Function**: Resolve the OOM issue of the $F=8$ VAE at 4096×4096 resolution.
    - **Mechanism**: Apply a dilation rate of 2 in the first convolutional layer of the VAE encoder to achieve an additional 2x downsampling ($F=8 \rightarrow F=16$). In the last layer of the decoder, the feature map is partitioned, each partition is upsampled by 2x, the same convolution is applied to each partition, and then reconstructed. Pre-trained VAE parameters are fully reused without retraining.
    - **Design Motivation**: Maintain latent space consistency with pre-trained LDMs to prevent distribution shifts, while reducing GPU memory consumption to a feasible range.

2. **Wavelet-based Latent Enhancement (WLF)**:

    - **Function**: Explicitly enhance focus on high-frequency details in the fine-tuning objective.
    - **Mechanism**: Apply Discrete Wavelet Transform (DWT) with Haar wavelets to both the predicted target (velocity/noise) and ground-truth target of the diffusion model, decomposing them into four sub-bands: LL (low-frequency approximation) and LH/HL/HH (high-frequency details). The loss function is formulated as $\mathcal{L}_{WLF} = \mathbb{E}[w_t \|f(v_\Theta(z_t,t)) - f(\epsilon - x_0)\|^2]$, where $f(\cdot)$ represents DWT.
    - **Design Motivation**: Standard MSE loss treats all frequencies equally, which dilutes the relative contribution of high-frequency components in 4K scenarios. DWT forces the model to simultaneously optimize low-frequency structure and high-frequency textures.

3. **Aesthetic-4K Benchmark**:

    - **Function**: Provide a training, evaluation, and metric system for 4K image synthesis.
    - **Mechanism**: The training set consists of 12,015 high-quality 4K images with GPT-4o captions. The evaluation set consists of 2,781 images from LAION-Aesthetics. New metrics include the GLCM Score (Gray-Level Co-occurrence Matrix entropy, measuring texture richness) and Compression Ratio (JPEG compression ratio, measuring detail preservation), achieving human perception alignment (SRCC=0.75/0.53) that far exceeds MUSIQ (0.36) and MANIQA (0.20).
    - **Design Motivation**: Fill the gap in 4K image synthesis evaluation and provide detail quality metrics centered on human perception.

### Loss & Training
The WLF loss is based on MSE in the wavelet transform domain. Training utilizes the AdamW optimizer (learning rate = 1e-6) with mixed-precision training. SD3-2B is trained on 2 A800-80G GPUs, and Flux-12B is trained on 8 A100-80G GPUs.

## Key Experimental Results

### Main Results
Aesthetic-Eval@2048 Evaluation:

| Model | FID↓ | CLIPScore↑ | Aesthetics↑ |
|------|------|------------|-------------|
| SD3-F16 (baseline) | 43.82 | 31.50 | 5.91 |
| SD3-F16-WLF (Ours) | 40.18 | 34.04 | 5.96 |
| Flux-F16 (baseline) | 50.57 | - | - |

### Detail Quality Metrics

| Model | GLCM Score↑ | Compression Ratio↓ |
|------|-------------|-------------------|
| SD3-F16 | Baseline | Baseline |
| SD3-F16-WLF | Higher | Lower (more details) |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Standard Fine-tuning (w/o WLF) | FID/CLIP baseline | 4K images tend to be blurry |
| + WLF Wavelet Loss | Decreased FID, increased CLIP | Details significantly enhanced |
| Partitioned VAE ($F=16$) | rFID=1.40, PSNR=28.82 | Close to original VAE reconstruction quality |

### Key Findings
- WLF comprehensively outperforms standard fine-tuning across FID, CLIPScore, Aesthetics, and detail metrics.
- Partitioned VAE maintains latent space consistency without retraining, yielding acceptable 4K reconstruction quality.
- GLCM Score and Compression Ratio align better with human perception of details compared to existing NR-IQA metrics.
- Large models (Flux-12B) show a more pronounced advantage in 4K generation, validating the scalability of DiTs at ultra-high resolutions.

## Highlights & Insights
- **Generality of Wavelet Fine-tuning**: WLF only modifies the loss function and can seamlessly adapt to any LDM (SD3, Flux, or even UNet architectures), representing a lightweight yet highly effective approach.
- **Partitioned VAE Cleverly Solves OOM**: The strategy of only modifying the first and last convolutional layers avoids retraining the VAE, making it highly practical.
- **Human Perception-Centric Evaluation**: GLCM Score and Compression Ratio fill the gap in 4K evaluation.

## Limitations & Future Work
- The training set consists of only 12K images, indicating a limited data scale.
- The $F=16$ compression ratio of the Partitioned VAE is higher, which may result in the loss of some fine-grained information.
- The framework has only been validated on text-to-image tasks; downstream tasks like image editing and video generation remain unexplored.
- Although evaluation metrics align better with human perception, their correlation coefficients still have room for improvement.

## Related Work & Insights
- **vs PixArt-Σ / Sana**: These models prioritize 4K efficiency but ignore detail quality, whereas this work focuses on detail enhancement.
- **vs Stable Cascade**: Uses multi-stage diffusion to progressively scale resolution, which may accumulate errors; conversely, this work directly performs end-to-end 4K generation.

## Rating
- Novelty: ⭐⭐⭐⭐ A comprehensive system featuring wavelet fine-tuning, partitioned VAE, and a 4K benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model validation, with thorough analysis of the new metrics' alignment with human perception.
- Writing Quality: ⭐⭐⭐⭐ Systematic and complete, with clear motivations for each component.
- Value: ⭐⭐⭐⭐ The 4K benchmark and WLF method bring practical value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion](ultrafusion_ultra_high_dynamic_imaging_using_exposure_fusion.md)
- [\[ICLR 2026\] Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis](../../ICLR2026/image_generation/latent_wavelet_diffusion_for_ultra_high-resolution_image_synthesis.md)
- [\[CVPR 2025\] Probability Density Geodesics in Image Diffusion Latent Space](probability_density_geodesics_in_image_diffusion_latent_space.md)
- [\[CVPR 2025\] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [\[CVPR 2025\] VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis](vlogger_multimodal_diffusion_for_embodied_avatar_synthesis.md)

</div>

<!-- RELATED:END -->
