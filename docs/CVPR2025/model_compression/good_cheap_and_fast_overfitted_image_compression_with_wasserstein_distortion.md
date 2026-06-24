---
title: >-
  [Paper Note] Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion
description: >-
  [CVPR 2025][Model Compression][Image Compression] This paper applies Wasserstein Distortion (WD) as an optimization objective to the overfitted image codec C3. Combined with common randomness to achieve texture resampling, it achieves a visual quality-rate trade-off comparable to generative compression methods while maintaining extremely low decoding complexity (<1% MACs of HiFiC).
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Image Compression"
  - "Wasserstein Distortion"
  - "Perceptual Quality"
  - "Overfitted Codec"
  - "Texture Reconstruction"
date: 2026-05-08
content_hash: cc3ca81773621967
---

# Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion

**Conference**: CVPR 2025  
**arXiv**: [2412.00505](https://arxiv.org/abs/2412.00505)  
**Code**: None (reconstructed results data download provided)  
**Area**: Model Compression  
**Keywords**: Image Compression, Wasserstein Distortion, Perceptual Quality, Overfitted Codec, Texture Reconstruction

## TL;DR
This paper applies Wasserstein Distortion (WD) as an optimization objective to the overfitted image codec C3. Combined with common randomness to achieve texture resampling, it achieves a visual quality-rate trade-off comparable to generative compression methods while maintaining extremely low decoding complexity (<1% MACs of HiFiC).

## Background & Motivation

**Background**: In recent years, learned image compression has been inspired by generative models (GANs, diffusion models) to achieve excellent image quality through better modeling of natural image distributions. Representative methods such as HiFiC utilize adversarial loss, and CDC is based on diffusion models, both generating visually convincing reconstructed images.

**Limitations of Prior Work**: The computational complexity of these generative compression methods is several orders of magnitude higher than that of commercial codecs, making them completely impractical for real-world application scenarios like mobile devices. Among "good" (quality), "cheap" (rate), and "fast" (speed), it seems one can only choose two.

**Key Challenge**: Generative methods pursue high quality by accurately modeling natural image distributions, but accurately modeling the distribution itself requires extremely high complexity. The key question is: does high visual quality truly require precise distribution modeling?

**Goal**: To demonstrate that by modeling human visual perception (rather than data distribution), visual quality comparable to generative compression can be achieved at low decoding complexity.

**Key Insight**: The peripheral vision of the human visual system only perceives texture statistical features rather than precise pixel values. Wasserstein Distortion (WD) precisely models this characteristic—allowing texture resampling in low-saliency regions, thereby significantly saving bits.

**Core Idea**: To replace MSE with WD as the optimization objective of the C3 overfitted codec, and introduce Common Randomness (CR) to assist texture reconstruction, achieving "good, cheap, and fast" image compression.

## Method

### Overall Architecture
The method is based on the C3 overfitted codec architecture, where each image independently optimizes a set of multi-resolution latent variables, a synthesis network, and an entropy model. The input is the original image, and the latent variables are bilinearly upsampled, concatenated, and fed through a small CNN synthesis network to output RGB pixels, with the entropy network modeling conditional probabilities for each latent element. Only two modifications are made: (1) introducing common randomness noise as an additional input at the decoder side; (2) replacing the MSE loss with Wasserstein Distortion.

### Key Designs

1. **Wasserstein Distortion (WD) as Optimization Objective**:

    - **Function**: Measure the distance between original and reconstructed images in the perceptual space, while accounting for differences between foveal and peripheral vision.
    - **Mechanism**: Extract multi-level feature maps from VGG networks, determine pooling region sizes based on the $\sigma$ values at different spatial locations, and calculate the 2-Wasserstein distance of local means and standard deviations. When $\sigma$ is small, it approaches point-wise comparison (fovea); when $\sigma$ is large, it allows texture statistical matching (peripheral vision). The authors propose an efficient approximation: discretize $\sigma$ to powers of 2, construct a Gaussian-pyramid-like multi-scale cascade to precompute local statistics, and then linearly interpolate to obtain the WD value for any $\sigma$.
    - **Design Motivation**: Compared to point-wise feature distances like LPIPS, WD explicitly models the foveal-peripheral duality of the visual system, allowing texture resampling in peripheral regions and effectively reducing compression costs.

2. **Adaptive $\sigma$ Mapping Based on Saliency**:

    - **Function**: Adaptively adjust the tolerance of WD according to the visual importance of image regions.
    - **Mechanism**: Predict the saliency map $s$ using EML-net, convert it to density $p = p_{min} + (1-p_{min}) \cdot s/\bar{s}$, and then map it to $\sigma = \sigma_{max} \cdot p_{min}/p$. Saliency regions have small $\sigma$ (requiring accurate reconstruction), while non-saliency regions have large $\sigma$ (allowing texture resampling).
    - **Design Motivation**: A uniform constant $\sigma$ would cause semantically important content (e.g., text) to be treated as replaceable textures. For instance, the text "ZENITAR-M" on a camera becomes unreadable under a constant $\sigma$, but saliency-based $\sigma$ preserves these areas.

3. **Common Randomness (CR)**:

    - **Function**: Provide the same pseudo-random noise to both the encoder and decoder to enable collaborative reconstruction of stochastic textures.
    - **Mechanism**: Generate i.i.d. standard Gaussian noise across multiple resolutions (using a pseudo-random number generator with a fixed seed), upsample, and concatenate with latent variables as additional inputs to the synthesis network. Since the seed is fixed, no extra bits are transmitted.
    - **Design Motivation**: Without CR, the codec must approximate stochastic textures using deterministic structures (such as straight lines), which has limited performance. With CR, the codec can perform "noise shaping" on the noise to efficiently reconstruct stochastic textures (e.g., grass), thereby allocating more bits to structural content.

### Loss & Training
The optimization objective is the rate-distortion loss, where the distortion term is WD (replacing the MSE of the original C3), and the rate term is the cross-entropy of the latent variables under the entropy model. Two WD variants are designed: WD8 uses a constant $\sigma=8$, and WDs uses a saliency-driven $\sigma$ mapping ($\sigma_{max}=16$). The encoding complexity increases by approximately 6 times due to the WD computation.

## Key Experimental Results

### Main Results
On the CLIC2020 professional validation set (41 images), 10 compression methods were evaluated through a large-scale user study (16,659 pairwise comparisons). Target bitrates are 0.075, 0.15, and 0.3 bits/pixel.

| Method | Decoding MACs/pixel | Elo Rating Trend | Note |
|------|---------------|------------|------|
| C3/WDs (Ours) | ~10³ | Comparable to HiFiC | Common randomness + saliency-based WD |
| C3/WD8 (Ours) | ~10³ | Slightly lower than WDs | Constant σ WD |
| HiFiC | ~10⁵ | Optimal | GAN-based generative compression |
| VVC | ~10² | Moderately low | Commercial codec |
| MLIC+ | ~10⁵ | Moderately low | MSE-optimal learned method |
| CDC | ~10⁶ | Moderately low | Diffusion-based compression |
| C3/MSE | ~10³ | Lowest | Original C3 |

### Perceptual Metrics Predicting Human Ratings

| Metric | Binary Rating Prediction Accuracy | Elo PCC | Elo SRCC |
|------|------------------|---------|----------|
| WDs (Ours) | **72.87%** | **0.942** | **0.913** |
| LPIPS | ~67% | ~0.7 | ~0.7 |
| DISTS | ~67% | ~0.8 | ~0.8 |
| MS-SSIM | ~66% | ~0.7 | ~0.7 |
| PSNR | ~64% | ~0.5 | ~0.5 |

### Key Findings
- After WD optimization, the bit allocation of the highest-resolution latent variables (representing detail encoding) is significantly reduced, indicating that WD allows texture resampling and thereby frees up bit budgets for other layers.
- CR further enhances this effect: with CR provided, the codec performs less "hard-coding" of texture details.
- As a perceptual metric, WD reaches a 94% Pearson correlation with human Elo ratings, far outperforming existing metrics like LPIPS.
- Saliency-based $\sigma$ mapping effectively protects semantically important regions (such as text), resulting in significant visual improvements.

## Highlights & Insights
- The concept of **replacing distribution modeling with perceptual modeling** is ingenious: it achieves comparable perceptual quality without the complexity of generative models, primarily by leveraging the "peripheral blind spot" of human vision.
- **Common Randomness** is an elegant design: it enables collaborative reconstruction of stochastic textures between the encoder and decoder with zero extra bit cost, essentially shifting the deterministic coding problem into a stochastic coding one.
- **WD as an IQA metric** performs exceptionally well (94% PCC) unexpectedly. This side product might actually be more valuable than the compression method itself, as it can be generalized to any scenario requiring perceptual quality evaluation.

## Limitations & Future Work
- Encoding complexity increases by approximately 6-fold (since WD computation is more expensive than MSE), representing a bottleneck for real-time encoding scenarios.
- The choices of feature space and $\sigma$ mapping are purely ad-hoc and lack systematic optimization.
- The scale of the user study is limited (41 images × 3 bitrates), requiring larger-scale validation.
- The quality of the saliency model directly affects the performance of the $\sigma$ mapping; a specialized saliency model developed specifically for compression tasks might yield better results.

## Related Work & Insights
- **vs HiFiC**: HiFiC utilizes GANs for generative compression, achieving slightly better visual quality than ours, but its decoding complexity is more than two orders of magnitude higher. This work achieves comparable quality using perceptual loss with a low-complexity architecture.
- **vs CDC**: CDC is a diffusion-based compression method with the highest decoding complexity (~10⁶ MACs), yet its visual quality is merely comparable to MSE-optimized methods, highlighting the efficiency issues of generative methods in compression.
- **vs COOL-CHIC/C3 series**: This work is a direct improvement on C3. Extremely large gains in quality are achieved by simply changing the loss function and adding CR, demonstrating that the choice of loss function is more critical than the architecture.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Although the application of WD in compression is novel, core components (C3 architecture, WD metric) are pre-existing works.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The large-scale user study serves as the gold standard, offering comprehensive comparisons and insightful analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is logically clear with beautiful figures, and the motivation is excellently articulated.
- **Value**: ⭐⭐⭐⭐ This work holds high practical significance (low complexity, high-quality compression), and the side discovery of WD as an IQA metric possesses independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CoA: Towards Real Image Dehazing via Compression-and-Adaptation](coa_towards_real_image_dehazing_via_compression-and-adaptation.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[CVPR 2025\] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)
- [\[CVPR 2025\] What Makes a Good Dataset for Knowledge Distillation?](what_makes_a_good_dataset_for_knowledge_distillation.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)

</div>

<!-- RELATED:END -->
