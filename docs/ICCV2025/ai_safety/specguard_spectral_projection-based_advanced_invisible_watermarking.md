---
title: >-
  [Paper Note] SpecGuard: Spectral Projection-based Advanced Invisible Watermarking
description: >-
  [ICCV2025][AI Safety][Invisible Watermarking] SpecGuard embeds watermark information into the spectral domain of high-frequency subbands obtained via wavelet decomposition (approximated through FFT-based spectral project…
tags:
  - "ICCV2025"
  - "AI Safety"
  - "Invisible Watermarking"
  - "Spectral Projection"
  - "Wavelet Transform"
  - "FFT"
  - "Parseval's Theorem"
date: 2026-05-08
content_hash: 849a688252215c65
---

# SpecGuard: Spectral Projection-based Advanced Invisible Watermarking

**Conference**: ICCV2025
**arXiv**: [2510.07302](https://arxiv.org/abs/2510.07302)  
**Code**: [https://github.com/SpecGuard](https://github.com/SpecGuard) (available)  
**Area**: AI Security / Digital Watermarking
**Keywords**: Invisible Watermarking, Spectral Projection, Wavelet Transform, FFT, Parseval's Theorem

## TL;DR
SpecGuard embeds watermark information into the spectral domain of high-frequency subbands obtained via wavelet decomposition (approximated through FFT-based spectral projection). The encoder employs a strength factor to enhance robustness, while the decoder applies a learnable threshold derived from Parseval's theorem for bit recovery. The method achieves high image quality (PSNR > 42 dB) alongside comprehensive robustness against distortion, regeneration, and adversarial attacks, surpassing existing SOTA methods.

## Background & Motivation

**Background**: With the proliferation of AI image generation and editing tools, copyright protection and authenticity verification of digital content have become increasingly urgent. Invisible watermarking is a mainstream authentication mechanism that embeds imperceptible information into images to verify authenticity.

**Limitations of Prior Work**:
   - Traditional transform-domain watermarking methods (DCT, DWT) are vulnerable to common image operations such as scaling, cropping, compression, and noise addition.
   - Deep learning-based methods (HiDDeN, StegaStamp, Stable Signature) have advanced end-to-end embedding, but remain fragile against adversarial attacks and image regeneration (diffusion model reconstruction).
   - Generative watermarking methods (e.g., those integrated with diffusion models) incur high computational complexity and are susceptible to targeted attacks.

**Key Challenge**: There exists a fundamental trade-off between imperceptibility and robustness — stronger embedding yields greater robustness but higher perceptibility, while weaker embedding improves invisibility at the cost of fragility.

**Goal**: To design a watermarking method that simultaneously surpasses SOTA in both imperceptibility and robustness, specifically against three attack categories: distortion (rotation, cropping, noise, etc.), image regeneration (diffusion model reconstruction), and adversarial attacks.

**Key Insight**: Watermark information is embedded into the spectral domain of hidden convolutional feature maps (rather than directly in the spatial domain or a simple frequency domain) via a cascaded transform pipeline: wavelet projection → high-frequency subbands → FFT spectral projection, allowing the watermark to be deeply concealed within high-frequency spectral components.

**Core Idea**: FFT spectral projection is applied to the high-frequency subbands of the wavelet decomposition to embed the watermark in the high-frequency spectral region. A learnable threshold based on Parseval's theorem enables high-accuracy bit extraction.

## Method

### Overall Architecture
SpecGuard consists of two modules — an encoder and a decoder:
- **Encoder**: Applies wavelet decomposition to the original image → extracts the high-frequency subband $S_{HH}$ → performs FFT spectral projection on $S_{HH}$ → embeds the binary watermark in designated high-frequency spectral regions → reconstructs the watermarked image via inverse transforms.
- **Decoder**: Applies the same wavelet and spectral projection pipeline to the watermarked image → extracts signals from the corresponding regions → decodes bits using a learnable threshold $\theta$.

### Key Designs

1. **Wavelet Projection**:

    - *Function*: Decomposes the image into multi-scale, multi-directional subbands prior to embedding.
    - *Mechanism*: A 2D discrete wavelet transform decomposes the image into $S_{LL}$ (low-frequency approximation), $S_{LH}$, $S_{HL}$, and $S_{HH}$ (horizontal/vertical/diagonal high-frequency details). The number of decomposition levels is $\kappa = \lfloor\sqrt{\log(1+N)}\rfloor$, where $N$ denotes the total number of pixels.
    - *Design Motivation*: High-frequency subbands encode edge and texture details, making them suitable for embedding watermarks without affecting visual quality while facilitating concealment.

2. **FFT Spectral Projection Approximation**:

    - *Function*: Transforms the high-frequency subband $S_{HH}$ from the spatial domain to the spectral domain.
    - *Mechanism*: $S_{HH}$ is symmetrically extended ($N \times N \to 2N \times 2N$), followed by 2D FFT. The real part is taken as an approximation of the spectral projection coefficients: $\zeta(u,v) \approx \text{Re}(F(u,v))$. Symmetric extension ensures that the FFT output is purely real-valued, simplifying subsequent processing.
    - *Design Motivation*: Embedding directly in the spectral domain is more stable than in the spatial domain, and the FFT approximation is computationally more efficient than exact spectral projection.

3. **Watermark Embedding**:

    - *Function*: Embeds the binary message into designated high-frequency regions of the spectral domain.
    - *Mechanism*: Features are first extracted from $S_{HH}$ via $k$ convolutional layers with LeakyReLU activations. A radial mask centered at $(h/2, w/2)$ with radius $r$ is then constructed, and the message is embedded exclusively within the masked high-frequency spectral region: $S_{HH}^{(n+1)}[:,W_c,x_i,y_i] += M_{\text{expanded}}[:,W_c,i] \cdot s$, where $s$ is a strength factor controlling embedding intensity.
    - *Design Motivation*: The radial mask confines embedding to high-frequency regions, reducing perceptual impact. The strength factor $s$ balances imperceptibility and robustness. Without knowledge of $r$, $s$, and $W_c$, it is difficult to localize the watermark, enhancing security (black-box property).

4. **Parseval's Theorem-Guided Learnable Threshold Decoding**:

    - *Function*: Extracts watermark bits from the spectral domain using an adaptive threshold.
    - *Mechanism*: The decoder applies the same wavelet and spectral projection pipeline to the watermarked image, extracts coefficients from the masked region, and determines each bit via a learnable threshold $\theta$: $D_M[i] = 1 \text{ if Extracted}[i] > \theta$. The threshold is optimized via gradient descent: $\theta \leftarrow \theta - \eta \cdot \frac{\partial L_{\text{dec}}}{\partial \theta}$.
    - *Design Motivation*: Parseval's theorem guarantees that total energy is conserved between the spatial and spectral domains; however, watermark embedding (via the strength factor $s$) alters the local spectral energy distribution — positions embedded with "1" exhibit higher energy. The learnable threshold adapts to this energy distribution shift, dynamically adjusting the decision boundary under various attacks.

### Loss & Training
- Encoder loss $L_{\text{enc}} = \|E_\theta(I, M) - I\|^2$ (enforces imperceptibility)
- Decoder loss $L_{\text{dec}} = \|D_\theta(I_{\text{embedded}}) - M\|^2$ (enforces extraction accuracy)
- Total loss $L = \lambda_{\text{enc}} L_{\text{enc}} + \lambda_{\text{dec}} L_{\text{dec}}$, with initial values $\lambda_{\text{enc}}=0.7$ and $\lambda_{\text{dec}}=1.0$
- Adam optimizer; encoder lr=$10^{-2}$, decoder lr=$10^{-3}$ (halved every 100 steps); trained for 300 epochs

## Key Experimental Results

### Main Results: Quality and Accuracy Comparison under No Attack

| Method | Conference | BL=64 PSNR/BRA | BL=128 PSNR/BRA | BL=256 PSNR/BRA |
|--------|-----------|----------------|-----------------|-----------------|
| HiDDeN | ECCV'18 | 32.01/0.98 | 31.80/0.85 | 31.50/0.82 |
| StegaStamp | CVPR'20 | 28.50/0.99 | 28.20/0.98 | 28.00/0.94 |
| EditGuard | CVPR'24 | 41.56/0.98 | 41.30/0.97 | 40.90/0.97 |
| MuST | AAAI'24 | 41.20/0.98 | 40.90/0.93 | 40.50/0.90 |
| **SpecGuard** | **ICCV'25** | **42.59/0.99** | **42.89/0.99** | **40.86/0.98** |

At 128-bit embedding, SpecGuard achieves PSNR = 42.89 dB, SSIM = 0.99, and BRA = 0.99, outperforming all baselines comprehensively.

### Robustness Comparison (Evaluated via the Waves Framework)

| Attack Type | Metric | Tree-Ring | Stable Sig | StegaStamp | SpecGuard |
|------------|--------|-----------|------------|------------|-----------|
| Rotation | Avg P | 0.375 | 0.594 | 0.357 | **0.687** |
| Crop | Avg P | 0.332 | 0.995 | 0.540 | **0.998** |
| Regen-Diff | Avg P | 0.612 | 0.001 | 0.943 | **0.982** |
| Regen-VAE | Avg P | 0.832 | 0.516 | 1.000 | **0.995** |
| Adversarial | Avg P | 0.448 | - | - | **High** |

SpecGuard demonstrates strong performance across distortion, regeneration, and adversarial attack categories.

### Multi-Resolution Quality Evaluation

| Resolution | Dataset | PSNR | SSIM | FID | MSE |
|-----------|---------|------|------|-----|-----|
| 256×256 | CelebA-HQ | 40.361 | 0.9889 | 16.451 | 0.0002 |
| 512×512 | MS-COCO | 44.680 | 0.9927 | 17.020 | 0.0001 |
| 1024×1024 | MS-COCO | 48.081 | 0.9936 | 16.955 | 0.0001 |

Image quality improves with resolution (PSNR reaching up to 48 dB), as the watermark energy is distributed across more pixels.

### Key Findings
- **High bit-capacity with sustained accuracy**: BRA remains 0.98 at 256 bits, whereas most methods degrade substantially at this capacity.
- **Strong performance under regeneration attacks**: Against diffusion model reconstruction (Regen-Diff), SpecGuard achieves Avg P = 0.982, while Stable Signature nearly completely fails (Avg P = 0.001).
- **Excellent imperceptibility**: PSNR exceeds 40 dB across all tested resolutions.
- **Robustness on social media platforms**: Watermarks remain recoverable after uploading to various platforms.

## Highlights & Insights
- **Elegant cascaded transform strategy**: Wavelet decomposition → high-frequency subbands → FFT spectral projection progressively relocates the watermark into domains that are increasingly difficult to perceive and disrupt. This "transform-within-transform" paradigm makes it challenging for adversaries to localize the watermark without knowledge of the transform parameters.
- **Practical application of Parseval's theorem**: A classical mathematical result (conservation of total energy between spatial and frequency domains) is operationalized as the design rationale for a learnable threshold — positions embedded with "1" exhibit higher energy, and $\theta$ learns to exploit this energy differential. This theory-driven design approach is worthy of broader adoption.
- **Black-box security**: The watermark parameters $(r, s, W_c)$ constitute a key space; without these values, it is infeasible to localize the embedding region.

## Limitations & Future Work
- **Computational complexity**: The latency of combining wavelet transforms, FFT, and multi-layer convolutions on high-resolution images is not thoroughly analyzed.
- **Fixed embedding strategy**: The strength factor $s$ and radius $r$ are fixed hyperparameters and are not adaptively adjusted based on image content.
- **Attack coverage**: Although three major attack categories are evaluated, white-box attacks leveraging model knowledge are not considered.
- **Promising directions**: (a) Adaptive strength factor — dynamically adjusting $s$ based on local texture complexity; (b) Multi-scale embedding — simultaneously embedding across multiple wavelet decomposition levels to improve fault tolerance; (c) Integration with generative models — directly embedding watermarks during the sampling process of diffusion models.

## Related Work & Insights
- **vs. StegaStamp**: StegaStamp learns end-to-end embedding but achieves only ~28–29 dB PSNR, far below SpecGuard's 42+ dB. SpecGuard holds a decisive advantage in imperceptibility through spectral-domain embedding.
- **vs. Stable Signature**: Stable Signature embeds watermarks into the decoder of a generative model as a pre-processing approach. It nearly completely fails under diffusion model regeneration attacks (Avg P = 0.001), whereas SpecGuard, as a post-processing method, demonstrates substantially greater robustness.
- **vs. EditGuard**: EditGuard achieves comparable PSNR (41.56 vs. 42.59), but SpecGuard maintains higher BRA at larger bit capacities (256-bit: 0.98 vs. 0.97).
- **vs. Tree-Ring**: Tree-Ring embeds watermarks into the spectral domain of the initial noise, sharing a frequency-domain philosophy with SpecGuard; however, Tree-Ring is applicable only to images generated by diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ The cascaded wavelet and spectral projection pipeline, together with the Parseval's theorem-guided learnable threshold, represents a genuine contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three major attack categories (distortion/regeneration/adversarial), with multi-resolution, multi-dataset, and social media platform evaluations.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are thorough, though notation is dense; the overall structure is clear.
- Value: ⭐⭐⭐⭐ Achieves comprehensive improvements over SOTA in the watermarking domain with high practical utility.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy](../../NeurIPS2025/ai_safety/spectral_perturbation_bounds_for_low-rank_approximation_with_applications_to_pri.md)
- [\[NeurIPS 2025\] Provable Watermarking for Data Poisoning Attacks](../../NeurIPS2025/ai_safety/provable_watermarking_for_data_poisoning_attacks.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](../../CVPR2026/ai_safety/recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)
- [\[CVPR 2026\] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](../../CVPR2026/ai_safety/decoupling_defense_strategies_for_robust_image_watermarking.md)

</div>

<!-- RELATED:END -->
