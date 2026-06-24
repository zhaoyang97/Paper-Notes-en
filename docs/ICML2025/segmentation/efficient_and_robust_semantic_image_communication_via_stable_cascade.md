---
title: >-
  [Paper Note] Efficient and Robust Semantic Image Communication via Stable Cascade
description: >-
  [ICML 2025][Segmentation][Semantic Communication] A semantic image communication framework built upon the Stable Cascade architecture. It uses EfficientNet-V2 to extract highly compact image embeddings (occupying just 0.29% of the original size) as LDM conditioning. Through noise-robust fine-tuning, the system reconstructs images faithfully even under low SNR channels, while achieving 3-16x inference acceleration.
tags:
  - "ICML 2025"
  - "Segmentation"
  - "Semantic Communication"
  - "Latent Diffusion Models"
  - "Stable Cascade"
  - "Image Compression"
  - "Channel Robustness"
date: 2026-05-08
content_hash: a78b379be69bb334
---

# Efficient and Robust Semantic Image Communication via Stable Cascade

**Conference**: ICML 2025  
**arXiv**: [2507.17416](https://arxiv.org/abs/2507.17416)  
**Code**: [GitHub](https://github.com/abilalk02/SC-SIC)  
**Area**: Semantic Communication / Generative AI  
**Keywords**: Semantic Communication, Latent Diffusion Models, Stable Cascade, Image Compression, Channel Robustness

## TL;DR

A semantic image communication framework built upon the Stable Cascade architecture. It uses EfficientNet-V2 to extract highly compact image embeddings (occupying just 0.29% of the original size) as LDM conditioning. Through noise-robust fine-tuning, the system reconstructs images faithfully even under low SNR channels, while achieving 3-16x inference acceleration.

## Background & Motivation

**Background**: Semantic communication (SemCom) aims to transmit the "meaning" of information rather than raw bits, realizing extreme bandwidth compression through deep learning and generative models. Diffusion models (DMs) have become a mainstream tool for semantic image communication (SIC) due to their outstanding image synthesis capabilities. Existing DM-based SIC systems include solutions like GESCO (segmentation map-conditioned) and Img2Img-SC (SD text + image-conditioned).

**Limitations of Prior Work**:
1. **Slow Inference**: GESCO requires 1000 denoising steps, taking 5 minutes and 24 seconds for a single 512×512 image.
2. **Generative Randomness**: Text-conditioned methods produce different results each time, making the reconstruction uncontrollable.
3. **Insufficient Compression Ratio**: The SD latent space of [4,64,64] offers only around ~48x compression.

**Key Challenge**: Existing schemes cannot simultaneously achieve high speed, extreme compression, and high reconstruction fidelity. GESCO is faithful but extremely slow, Img2Img-SC is faster but suffers from high generative randomness, and JPEG2000+LDPC collapses completely under low SNR.

**Goal**: To design a semantic communication system that simultaneously achieves **extreme compression** (0.29%), **fast inference** (<1s), and **high-fidelity reconstruction**.

**Key Insight**: Leveraging the extremely small latent space of Stable Cascade (which is significantly smaller than SD) naturally suits extreme compression, while noise-aware fine-tuning enhances channel robustness.

**Core Idea**: The combination of Stable Cascade's hyper-compressed latent space and noise-aware conditional fine-tuning yields a triple advantage of speed × compression × fidelity.

## Method

### Overall Architecture

The system consists of three stages:
- **Transmitter**: The EfficientNet-V2 encoder extracts extremely compact embeddings $Z \in \mathbb{R}^{16 \times 24 \times 24}$.
- **Channel Transmission**: $Z$ is transmitted through an AWGN channel, and the receiver obtains $\hat{Z} = Z + \epsilon$.
- **Receiver**: $\hat{Z}$ acts as the LDM conditioning → generates the VQGAN latent representation → VQGAN decodes back to the pixel space.

### Key Designs

1. **Extremely Compressed Image Embeddings (EfficientNet-V2 Encoder)**:
    - **Function**: Compresses the original image to 0.29% of its original size.
    - **Mechanism**: Utilizes the pre-trained EfficientNet-V2 encoder from Stable Cascade to encode $X \in \mathbb{R}^{3 \times 1024 \times 1024}$ into $Z \in \mathbb{R}^{16 \times 24 \times 24}$. The compression ratio is $\frac{3 \times 1024 \times 1024}{16 \times 24 \times 24} = 341$. This embedding preserves high-level semantic features, performing far better than text embeddings (which are too abstract and lead to semantic drift) and segmentation maps (which lose texture and color information).
    - **Design Motivation**: To find the optimal balance between information fidelity and compression ratio.

2. **Noise-Aware LDM Fine-Tuning (Stage B)**:
    - **Function**: Enables the LDM to learn to recover high-quality images from noisy conditional embeddings.
    - **Mechanism**: Stage B of Stable Cascade originally assumes a noise-free conditional input. This work injects channel noise into the conditional embeddings during training: $\hat{Z} = Z + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$, and the SNR is randomly sampled between 1-20 dB. The training target is the standard MSE denoising loss:
    $$L = \mathbb{E}_{(X_\text{VG,t}, t, \hat{Z}, \epsilon)}[\|\epsilon - \bar{\epsilon}(X_\text{VG,t}, t, \hat{Z})\|_2^2]$$
    Fine-tuning is conducted for 15,000 steps with batch=4 and lr=1e-4.
    - **Design Motivation**: The original SC model directly collapses under channel noise (validated by ablation studies). Noise-aware training allows the generative model itself to learn channel denoising.

3. **Reusing VQGAN Autoencoder (Stage A)**:
    - **Function**: Handles the transition between pixel space and latent space.
    - **Mechanism**: Reuses the pre-trained VQGAN from Stable Cascade (providing 4x spatial compression), where $\hat{X} = f_\Theta^{-1}(\hat{X}_\text{VG})$. Stage C (text-to-embedding, which is unnecessary in this scenario) is bypassed.
    - **Design Motivation**: Stage A is already fully pre-trained and does not require further fine-tuning.

### Loss & Training

- Only Stage B (LDM) is fine-tuned, while Stage A and the encoder are frozen.
- Standard diffusion MSE denoising loss is used.
- During training, the SNR is randomly sampled between 1 and 20 dB to guarantee robustness across the entire range.
- Text conditioning is not used (as the SC paper notes it has no significant impact on Stage B).
- Training is conducted on a single NVIDIA RTX A6000 (48GB) GPU.

## Key Experimental Results

### Main Results: Compression Efficiency Comparison

| Method | Transmitted Data Dimension | Compression Ratio | % of Original |
|------|-------------|--------|---------|
| Original Image | [3,512,512] | - | 100% |
| **Ours (SC-SIC)** | **[16,12,12]** | **341** | **0.29%** |
| Img2Img-SC | [4,64,64] | 48 | 2.08% |
| DIFFSC | [8,32,32] | 96 | 1.04% |
| CASC | [8,32,32] | 96 | 1.04% |

### Inference Speed Comparison

| Method | 512×512 Time | 1024×1024 Time | Denoising Steps |
|------|-------------|---------------|----------|
| GESCO | 5m 24s | - | 1000 |
| Img2Img-SC | 2.34s | >12s | 30 |
| **Ours** | **0.78s** | **<1s** | **10** |

Acceleration ratio: **3x** for 512×512, and **>16x** for 1024×1024.

### Reconstruction Quality (Cityscapes, Average Improvements vs. Img2Img-SC)

| Metric | Improvement | Interpretation |
|------|---------|------|
| FID ↓ | -43% | Better distribution-level generation quality |
| LPIPS ↓ | -55% | Higher perceptual similarity |
| SSIM ↑ | +56% | Better structural preservation |
| PSNR ↑ | +23% | Higher pixel-level accuracy |

### Reconstruction Predictability (LPIPS μ±σ, 25 Transmissions)

| SNR (dB) | Ours-1024 | Ours-512 | GESCO | Img2Img-SC |
|----------|----------|---------|-------|------------|
| 20 | 0.173±0.003 | 0.205±0.005 | 0.401±0.014 | 0.520±0.011 |
| 10 | 0.229±0.003 | 0.264±0.008 | 0.424±0.017 | 0.522±0.012 |
| 1 | 0.351±0.006 | 0.371±0.013 | 0.613±0.017 | 0.578±0.019 |

### Ablation Study

| Ablation Item | Effect |
|--------|------|
| Without fine-tuning (Original SC) | The image is heavily corrupted and unusable when SNR < 10dB |
| Embedding [16,24,24]→[16,32,32] | LPIPS/FID/SSIM improve by >10%, but the compression ratio falls to 192 |
| JPEG2000+LDPC at SNR < 5dB | Fails completely (cliff effect), unable to recover the image |

### Key Findings

- Even under an extreme channel of SNR=1dB, the reconstructed image remains perceptually close to the original.
- Under extreme compression of 0.29%, the quality surpasses that of Img2Img-SC, which transmits 7x more data.
- Generative consistency is exceptionally high (LPIPS σ=0.003), whereas text-conditioned methods show σ=0.011-0.019.
- On the unseen DIV2K dataset, the model still reconstructs semantically accurate images, though the colors lean toward the Cityscapes color palette.
- Traditional JPEG2000+LDPC exhibits a cliff effect, completely collapsing under low SNR.

## Highlights & Insights

1. **Record Compression Rate**: 0.29% is the highest known compression ratio for DM-based SIC.
2. **Elegant Noise-Robustness Training**: Bypassing complex channel coding, incorporating noise during training teaches the generative model to handle channel denoising natively. This reformulates communication robustness as a generative model training problem.
3. **Practical Inference Speed**: Completing 512×512 reconstruction in 0.78 seconds makes semantic communication viable for real-time scenarios for the first time.
4. **Low-Variance Reconstruction**: An LPIPS σ=0.003 implies that the results of multiple transmissions are almost completely identical.
5. **Inherent Advantages of SC Architecture**: The multi-stage design (Stage A spatial compression + Stage B semantic generation) naturally establishes a hierarchical semantic transmission scheme.

## Limitations & Future Work

- Since fine-tuning was performed only on Cityscapes, there is a color bias during cross-domain generalization (revealed by the DIV2K experiment).
- The compression rate is fixed, lacking an adaptive adjustment mechanism based on channel conditions.
- A systematic comparison with video coding standards (such as H.265/H.266) is missing.
- It only supports images; video semantic communication (inter-frame consistency) is not addressed.
- Training relies on the AWGN channel model; its performance under real-world wireless channels (fading, multipath) has not been verified.

## Related Work & Insights

- **DeepJSCC (Bourtsoulatze et al., 2019)**: End-to-end joint source-channel coding; this paper replaces it with a pre-trained generative model.
- **GESCO (Grassucci et al., 2023)**: Segmentation map-conditioned diffusion SIC; high-fidelity but extremely slow.
- **Img2Img-SC (Cicchetti et al., 2024)**: SD image + text-conditioned model; inferior to this work in both compression and speed.
- **Stable Cascade (Pernias et al., 2023)**: Provides the core foundation for the hyper-compressed latent space architecture.
- Insights: Multi-stage generative models are inherently suitable for hierarchical semantic transmission and can be extended to progressive transmission.

## Rating

- Novelty: ⭐⭐⭐⭐ Clever application of Stable Cascade to semantic communication. Although noise-aware fine-tuning is straightforward, it is highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated against multiple baselines and across various SNRs, with complete ablation studies and cross-dataset generalization tests.
- Writing Quality: ⭐⭐⭐⭐ Clear system architecture and complete formula derivations.
- Value: ⭐⭐⭐⭐⭐ Breakthroughs in compression ratio and inference speed make the real-world application of semantic communication possible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](../../ECCV2024/segmentation/efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)
- [\[CVPR 2025\] MatAnyone: Stable Video Matting with Consistent Memory Propagation](../../CVPR2025/segmentation/matanyone_stable_video_matting_with_consistent_memory_propagation.md)
- [\[ICCV 2025\] Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [\[CVPR 2025\] Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild](../../CVPR2025/segmentation/robust_3d_shape_reconstruction_in_zero-shot_from_a_single_image_in_the_wild.md)
- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](../../NeurIPS2025/segmentation/towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)

</div>

<!-- RELATED:END -->
