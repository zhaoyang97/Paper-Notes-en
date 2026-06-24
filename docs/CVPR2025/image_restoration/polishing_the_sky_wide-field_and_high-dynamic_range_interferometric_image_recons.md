---
title: >-
  [Paper Note] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction with Application to Strong Lens Discovery
description: >-
  [CVPR2025][Image Restoration][radio interferometry] Based on the POLISH framework, POLISH+/++ is proposed with two key improvements—**patch-wise training + stitched inference** and **arcsinh non-linear transformation**—enabling deep learning methods to handle wide-field ($12,960 \times 12,960$ pixels) and high-dynamic-range ($\sim 10^6$) radio interferometric imaging for the first time, while demonstrating a $10\times$ potential increase in strong gravitational lensing discov…
tags:
  - "CVPR2025"
  - "Image Restoration"
  - "radio interferometry"
  - "image reconstruction"
  - "super-resolution"
  - "deep learning"
  - "strong gravitational lensing"
  - "high dynamic range"
date: 2026-05-08
content_hash: c1357373d7338acf
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction with Application to Strong Lens Discovery

**Conference**: CVPR2025  
**arXiv**: [2603.09162](https://arxiv.org/abs/2603.09162)  
**Code**: To be confirmed  
**Area**: Image Restoration  
**Keywords**: radio interferometry, image reconstruction, super-resolution, deep learning, strong gravitational lensing, high dynamic range

## TL;DR
Based on the POLISH framework, POLISH+/++ is proposed with two key improvements—**patch-wise training + stitched inference** and **arcsinh non-linear transformation**—enabling deep learning methods to handle wide-field ($12,960 \times 12,960$ pixels) and high-dynamic-range ($\sim 10^6$) radio interferometric imaging for the first time, while demonstrating a $10\times$ potential increase in strong gravitational lensing discovery through super-resolution.

## Background & Motivation
**Computational Challenges of Radio Interferometric Imaging**: The upcoming Deep Synoptic Array (DSA) will feature 1650 antennas with a raw data throughput exceeding 80 Tb/s, which traditional methods cannot process in real time.

**Limitations of Prior DL Methods**: As summarized in Table 1, existing DL methods have only been tested on small images ($<1000$ pixels) and low dynamic ranges ($<10^3$), which is far from practical deployment requirements.

**Dimensional Bottleneck of Wide-Field**: The field of view of DSA is approximately 10 square degrees, with images exceeding $10,000 \times 10,000$ pixels. A single float32 input requires about 400MB of GPU memory, and feature map storage is even larger.

**High Dynamic Range Issue**: The ratio between bright and faint sources in a wide field can reach $10^4$–$10^6$. Direct training causes the network to bias toward bright sources while ignoring faint ones.

**Model Mismatch**: In actual observations, ionospheric effects, antenna pointing errors, and other factors cause the PSF to deviate from assumptions, a problem that is rarely considered in existing works.

**Potential for Strong Gravitational Lensing Discovery**: DSA is expected to discover $10^4$–$10^5$ strong lenses, but traditional CLEAN methods are limited by the PSF resolution. Super-resolution has the potential to increase the discovery rate by an order of magnitude.

## Method

### Overall Architecture
Based on POLISH (an end-to-end CNN mapping dirty images to clean images), two key improvements targeting wide-field and high-dynamic-range imaging are proposed:
- **POLISH+**: Patch-wise training + stitched inference
- **POLISH++**: Patch-wise training + arcsinh transformation

### Key Designs

**1. Patch-Wise Processing**
- Slice the $12,960 \times 12,960$ image into $J$ non-overlapping patches ($324 \times 324$ pixels).
- Each dirty-clean training pair is divided into $J$ patch training pairs; 18 training images yield 28,800 training patches.
- Key Insight: The dirty image in a patch contains PSF sidelobe artifacts originating from bright sources in adjacent patches (cross-patch contamination). This is significantly different from simply convolving the patch with the PSF, implying that the network needs to implicitly handle non-local effects.
- During inference, the patch predictions are stitched back into the full-field-of-view image.

**2. Arcsinh Non-linear Transformation**
- Transformation function: $\text{AsinhStretch}(x; a) = \frac{\text{arcsinh}(x/a)}{\text{arcsinh}(1/a)}$
- Compresses pixel values spanning multiple orders of magnitude into the same order, reducing the dynamic range by more than an order of magnitude.
- Superior to gamma encoding: Can handle both positive and negative values (dirty images can contain negative values).
- Both training and inference are conducted in the transformed space, and the inverse transformation $\text{AsinhStretch}^{-1}$ is applied after inference to restore the original scale.
- Hyperparameters $a_{\text{dirty}}$ and $a_{\text{true}}$ control the compression intensity of the input and target, respectively.

### Loss & Training
- The $\ell_1$ loss is computed in the arcsinh-transformed space:
$$\theta^* = \arg\min_\theta \frac{1}{NJ} \sum_{i,j} \|G_\theta(\text{AsinhStretch}(\mathbf{I}^{[j]}_{\text{dirty}}; a_{\text{dirty}})) - \text{AsinhStretch}(\mathbf{I}^{[j]}_{\text{true}}; a_{\text{true}})\|_1$$

## Key Experimental Results

### Main Results: Source Detection Accuracy

| Method | Precision↑ | Recall↑ | F₁↑ |
|------|-----------|---------|-----|
| CLEAN | 0.3612 | 0.2220 | 0.2750 |
| POLISH | 0.5560 | 0.4612 | 0.5042 |
| POLISH+ | **0.8744** | 0.5751 | 0.6938 |
| POLISH++ | 0.8433 | **0.6142** | **0.7107** |

POLISH++ improves $F_1$ score by **158%** compared to CLEAN ($0.2750 \rightarrow 0.7107$), and by 41% compared to POLISH.

### Shape and Flux Estimation Accuracy (RMSE)

| Method | Major Axis FWHM (″)↓ | Minor Axis FWHM (″)↓ | Flux (Jy/pix)↓ |
|------|---------------|---------------|---------------|
| CLEAN | 1.0046 | 0.7862 | **$3.26 \times 10^{-4}$** |
| POLISH++ | **0.4654** | **0.2056** | $3.17 \times 10^{-3}$ |

Shape estimation accuracy is significantly improved, whereas CLEAN remains superior in flux estimation (as CLEAN preserves absolute flux calibration).

### Strong Gravitational Lensing Discovery
- The super-resolution of POLISH/++ enables the CNN lens finder to recover lenses with Einstein radii close to the PSF scale.
- Compared to traditional CLEAN (requiring separation $>3\times$ PSF), POLISH reduces the lower limit of detection to $\sim1\times$ PSF.
- The number of discoverable lenses in the DSA survey is **increased by approximately 10 times**.

### Model Robustness and Adaptability
- Trained on ideal PSFs and tested on PSF distortions $\gamma \in [0,30]$: Visual reconstructions remain stable (with a predictable decrease in PSNR).
- Fine-tuning to a new PSF distribution requires only **11 epochs** (compared to 57 epochs for training from scratch), achieving a **$5\times$ speedup**.

### Key Findings
- Patch-wise training not only overcomes memory limits but also implicitly learns to handle cross-patch PSF sidelobe contamination.
- The arcsinh transformation increases the recall of POLISH++ by 4% on low-SNR sources and improves the $F_1$ score.
- Super-resolution breaks the PSF diffraction limit: POLISH++ can accurately estimate sources with angular sizes much smaller than the PSF width ($\approx 3.3''$).

## Highlights & Insights
1. **Application-Driven Engineering Innovation**: Two seemingly simple improvements (patching and non-linear transformation) render deep learning methods applicable to real-world radio astronomy scales for the first time.
2. **Interesting Phenomenon of Cross-Patch Contamination**: Patched dirty images contain non-local artifacts from neighboring patches, yet deep learning methods can implicitly learn to handle this effect. While this theoretically should not hold (as a local forward model does not exist), experiments prove its effectiveness.
3. **Impact of Super-Resolution on Scientific Discovery**: Rather than just improving image quality, this directly translates to an order-of-magnitude increase in the number of strong gravitational lensing discoveries.
4. **Efficient Adaptation via Fine-Tuning**: The pre-training + fine-tuning strategy enables the model to rapidly adapt to different observational conditions, which is crucial for practical deployment.

## Limitations & Future Work
1. **Inaccurate Flux Estimation**: The non-linear nature of learning-based methods results in a higher RMSE for flux estimation compared to CLEAN, lacking an explicit flux calibration mechanism.
2. **Simple Astronomical Model Assumptions**: The training utilizes T-RECS simulations (Gaussian/Sérsic profiles), and generalization to more complex morphologies (e.g., radio jets, extended structures) remains unverified.
3. **Pixel Scale Constraints**: The $\sim 1''$ pixel scale poses a fundamental limitation at extremely small separation scales.
4. **Small Training Dataset**: Training with only 18 images (though generating 28,800 patches through slice processing) may limit generalization capabilities.

## Related Work & Insights
- **vs CLEAN**: CLEAN is limited by the PSF resolution and cannot perform super-resolution, but it preserves flux calibration. POLISH++ exhibits strong super-resolution capability but yields less accurate flux estimation.
- **vs R2D2**: R2D2 was tested on $512^2$ images, whereas POLISH++ scales to $12,960^2$, representing a $600\times+$ increase in scale.
- **vs RML**: Optimization-based methods offer high quality but are computationally expensive, making them unsuitable for DSA real-time processing demands.
- **Insight**: Simple data transformations (such as arcsinh) can be more important than complex network designs in high-dynamic-range scenarios. The pre-training + fine-tuning paradigm is highly suitable for astronomical applications requiring adaptation to varying observational conditions.

## Rating
- Novelty: ⭐⭐⭐ — The patch-wise training and non-linear transformation methods themselves are relatively simple, but their first large-scale application in radio astronomy scenarios is significant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extremely thorough, with comprehensive evaluations spanning source detection, shape estimation, lens discovery, robustness, and adaptability.
- Writing Quality: ⭐⭐⭐⭐ — The background introduction is clear, the experimental design is rigorous, and the analysis of astronomical domain knowledge is in-depth.
- Value: ⭐⭐⭐⭐ — Enables DL methods to be deployed at the practical scale of next-generation radio telescopes for the first time, offering significant practical value to the radio astronomy community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[CVPR 2026\] DreamSR: Towards Ultra-High-Resolution Image Super-Resolution via a Receptive-Field Enhanced Diffusion Transformer](../../CVPR2026/image_restoration/dreamsr_towards_ultra-high-resolution_image_super-resolution_via_a_receptive-fie.md)
- [\[ECCV 2024\] Image Demoiréing in RAW and sRGB Domains](../../ECCV2024/image_restoration/image_demoiréing_in_raw_and_srgb_domains.md)
- [\[ICLR 2026\] Divergence-Free Neural Networks with Application to Image Denoising](../../ICLR2026/image_restoration/divergence-free_neural_networks_with_application_to_image_denoising.md)
- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](../../ICCV2025/image_restoration/idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)

</div>

<!-- RELATED:END -->
